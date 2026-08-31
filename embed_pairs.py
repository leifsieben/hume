"""Embed the Figure A pairs with every available representation.

Per-arm caching: each arm writes its own npz, so the figure can be drawn from whatever is
ready and a slow arm (full Mordred is ~80 ms/mol) never blocks a fast one.

The background set is 10,000 molecules that appear in NO pair. sigma_j must be a property of
the representation crossed with chemical space; estimating it on the edited molecules would let
each edit inflate its own denominator and pull every model toward the same score.

    python embed_pairs.py ecfp r3cfp chemberta molformer
    python embed_pairs.py desc          # slow, run it in the background
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FIGA = ROOT / "results" / "figures" / "figA"
EMB = FIGA / "embeddings"
N_BACKGROUND = 10_000


def _all_smiles():
    """-> (pairs, unique pair molecules in first-seen order, the reserved background).

    The background is READ from background.json, which make_pairs.py reserves BEFORE it builds
    any pair. It is deliberately not re-derived here as "benchmark molecules not in a pair":
    that made the sigma denominator a function of the pair set, so adding the C=C saturation
    panel moved the descriptor arm's protonation cell by 18% without touching the pairs it was
    measured on.
    """
    d = json.load(open(FIGA / "pairs.json"))
    pairs = d["pairs"]
    seen, order = set(), []
    for p in pairs:
        for s in (p["a"], p["b"]):
            if s not in seen:
                seen.add(s)
                order.append(s)
    bg = json.load(open(FIGA / "background.json"))
    assert len(bg) == N_BACKGROUND, f"background.json has {len(bg)}, expected {N_BACKGROUND}"
    assert not (set(bg) & seen), "background overlaps the pairs; re-run make_pairs.py"
    return pairs, order, bg


# --- arms ---------------------------------------------------------------------------------

def arm_ecfp(smiles, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator as fg
    RDLogger.DisableLog("rdApp.*")
    g = fg.GetMorganGenerator(radius=radius, fpSize=2048, includeChirality=True)
    out = np.zeros((len(smiles), 2048), np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = g.GetCountFingerprintAsNumPy(m)
    return out


def arm_r3cfp(smiles):
    return arm_ecfp(smiles, radius=3)


def arm_r4cfp(smiles):
    """Morgan r=4 (ECFP8). Added 2026-08-28 to extend the radius series past r=3.

    PREDICTION, RECORDED BEFORE THE NUMBER EXISTS so it reads as a result rather than a story
    fitted afterwards: r=4 should RESOLVE better than r=3 on the substitution edits and score no
    better -- probably worse -- downstream. r=3 already beats r=2 on stereo (0.559 vs 0.494),
    added methyl (0.607 vs 0.518) and ring fusion (0.863 vs 0.693) while losing to it on 20 of 28
    DEV datasets. If r=4 continues both trends, the dissociation between resolution and
    downstream value is a trend across three radii rather than a two-point coincidence.
    """
    return arm_ecfp(smiles, radius=4)


def arm_hume(smiles):
    """HUME's own 864 descriptors + ECFP, which is the arm this paper is about.

    IT WAS MISSING FROM FIGURE A ENTIRELY (noticed 2026-08-28). The plate carried
    RDKit+Mordred as `desc` and every learned encoder, but not the method the paper proposes --
    so the one figure that asks "does the representation resolve chemical change?" had no row
    for our representation.

    The stereo block is the reason this matters most. `desc` scores 0.505 on a stereocentre
    inversion, i.e. chance: mordred and RDKit's descriptor sets carry no signed stereo term at
    all. HUME's 182-block does -- `stereo()` in hume_blocks.h sums SIGNED CIP parity (R = +1,
    S = -1) over the molecule, which is directional and therefore something a tree can learn to
    read the same way on a scaffold it has never seen.

    `optional` names both expensive columns: this is a resolution measurement, not a throughput
    one, so the full 864 is what belongs here.
    """
    import hume
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mols, keep = [], []
    for i, sm in enumerate(smiles):
        m = Chem.MolFromSmiles(sm)
        if m is not None:
            mols.append(m); keep.append(i)
    # `qed` IS NO LONGER REQUESTED. The column was dropped in the cost triage because it
    # shipped 100% NaN, so asking for it now buys 69.3 us/mol of structural-alert matching
    # for a column that is not emitted. AvgIpc is still asked for -- it IS emitted.
    fp, X, _ = hume.featurize_all_from_mols(mols, optional=("AvgIpc",))
    out = np.full((len(smiles), X.shape[1] + fp.shape[1]), np.nan, np.float32)
    out[keep] = np.hstack([X, fp]).astype(np.float32)
    return out


def arm_notation(smiles):
    """A DIFFICULTY FLOOR, not a representation. Character 1- and 2-gram counts of the SMILES.

    This arm models no chemistry whatsoever -- it cannot tell an aromatic ring from a chain
    except by which characters were typed -- so whatever AUC it reaches on an edit is available
    to ANY model FOR FREE, from notation and gross composition alone. A panel where this scores
    ~1.0 does not discriminate between representations no matter what the bars look like.

    It exists because the CLIMB figure session found their regioisomer panel reading 1.000 for
    every arm INCLUDING an untrained random encoder: they had paired ortho against meta, so every
    A lacked a branch and every B had one, and the classifier was reading "is there a branch",
    not "where is the substituent". Leif's rule from that: an edit should MOVE a feature, never
    CREATE one. Our own regioisomer generator moves an existing substituent (49% of pairs have
    identical character multisets) and does not have the defect -- but asserting that is worth
    less than measuring it, and the same measurement audits every other panel at the same time.

    Counted on the SMILES AS WRITTEN, deliberately not canonicalised: the point is what the
    string hands over, and re-canonicalising would measure a different string than the one the
    CLM arms are fed. It is also why `null_enumerate` reads 0.962 here rather than 1.000 -- the
    two members genuinely are different strings.

    *** CASE IS PRESERVED, AND THAT IS NOT AN ACCIDENT. *** SMILES case is chemistry: aromatic
    `c` against aliphatic `C`. Lowercasing would hand this baseline strictly less than a real
    tokenizer sees and understate the floor, which is the one direction of error that matters --
    an understated floor lets a panel look harder than it is. Verified rather than assumed:
    `c1ccccc1` and `C1CCCCC1` produce different vectors under this function, on case alone. Do
    not add `.lower()`.
    """
    from collections import Counter
    grams, rows = {}, []
    for s in smiles:
        c = Counter(s)
        c.update(s[i:i + 2] for i in range(len(s) - 1))
        rows.append(c)
        for g in c:
            grams.setdefault(g, len(grams))
    out = np.zeros((len(rows), len(grams)), np.float32)
    for i, c in enumerate(rows):
        for g, v in c.items():
            out[i, grams[g]] = v
    return out


def arm_desc(smiles):
    from mordred import Calculator, descriptors as mdesc
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    calc = Calculator(mdesc, ignore_3D=True)
    rn = [n for n, _ in Descriptors._descList]
    lut = dict(Descriptors._descList)
    out = np.full((len(smiles), len(list(calc.descriptors)) + len(rn)), np.nan, np.float32)
    t0 = time.time()
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            md = [v if isinstance(v, (int, float)) else np.nan for v in calc(m)]
        except Exception:
            md = [np.nan] * (out.shape[1] - len(rn))
        rd = []
        for n in rn:
            try:
                rd.append(lut[n](m))
            except Exception:
                rd.append(np.nan)
        out[i] = np.array(md + rd, np.float32)
        if (i + 1) % 2000 == 0:
            print(f"    desc {i+1}/{len(smiles)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def _hf(smiles, path, batch=64, pool="mean"):
    """Mean-pooled last hidden state. `trust_remote_code` is required by MolFormer."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # `deterministic_eval=True` IS PASSED UNCONDITIONALLY, and the models that do not take it
    # ignore it. It used to be gated on `"MoLFormer" in str(path)` -- which NEVER MATCHED, because
    # the weights live in models_hf/MolFormer with a lowercase 'l'. So the one model that needs
    # the flag was the one model that never got it.
    #
    # MoLFormer-XL uses linear attention with random feature maps, and RESAMPLES THEM ON EVERY
    # FORWARD PASS unless this is set. Measured on two SMILES: max component difference between
    # two identical runs was 0.46161187 as previously called, and exactly 0.0 with the flag. The
    # embedding was therefore not a function of its input -- two calls produced two incompatible
    # spaces, and Figure A's sigma normaliser was estimated on a different draw than the pairs it
    # normalised, so it did not cancel.
    #
    # A string test against a directory name is the wrong mechanism regardless: it fails silently,
    # and it fails in the direction that produces plausible numbers.
    try:
        mod = AutoModel.from_pretrained(path, trust_remote_code=True, deterministic_eval=True)
    except TypeError:
        mod = AutoModel.from_pretrained(path, trust_remote_code=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mod = mod.to(dev).eval()
    outs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            b = tok(smiles[i:i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(dev)
            h = mod(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().float().numpy())
            if (i // batch) % 50 == 0:
                print(f"    {i}/{len(smiles)} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(outs).astype(np.float32)


# TWO ChemBERTa-2 CHECKPOINTS, not one. They are the same architecture (3 layers, 384 hidden)
# on the same 77M-molecule corpus, differing ONLY in the pretraining target: MLM is masked
# language modelling, MTR is multi-task regression onto 200 RDKit descriptors. That makes the
# pair a controlled ablation of "does supervising on descriptors help?", which Figures B and C
# rest on -- so they are separate arms and the directory name says which is which.
#
# Until 2026-08-27 there was one arm called `chemberta` reading models_hf/ChemBERTa-2, which
# holds the MLM weights, while figures/arms.py declared the MTR repo. Every published ChemBERTa
# number was MLM under a label naming MTR.
def arm_chemberta_mlm(smiles):
    return _hf(smiles, ROOT / "models_hf" / "ChemBERTa-2-MLM")


def arm_chemberta_mtr(smiles):
    return _hf(smiles, ROOT / "models_hf" / "ChemBERTa-2-MTR")


def arm_molformer(smiles):
    return _hf(smiles, ROOT / "models_hf" / "MolFormer")


def arm_selfies_ted(smiles, batch=64):
    """IBM SELFIES-TED: a BART encoder-decoder over SELFIES, not SMILES.

    Only the ENCODER is used, mean-pooled, matching every other transformer arm here.

    A molecule that fails SMILES->SELFIES conversion yields a zero row rather than being dropped,
    so the row order stays aligned with `smiles` -- silently shortening the array would misalign
    every pair after the failure. Conversion failures are counted and printed.
    """
    import selfies as sf
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = ROOT / "models_hf" / "SELFIES-TED"
    tok = AutoTokenizer.from_pretrained(path)
    mod = AutoModel.from_pretrained(path)
    enc = mod.get_encoder() if hasattr(mod, "get_encoder") else mod
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = enc.to(dev).eval()

    sel, bad = [], 0
    for s in smiles:
        try:
            sel.append(" ".join(sf.split_selfies(sf.encoder(s))))
        except Exception:
            sel.append("")
            bad += 1
    if bad:
        print(f"    selfies_ted: {bad} of {len(smiles)} failed SMILES->SELFIES (zero rows)",
              flush=True)

    outs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            b = tok(sel[i:i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(dev)
            h = enc(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().float().numpy())
            if (i // batch) % 50 == 0:
                print(f"    selfies_ted {i}/{len(sel)} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(outs).astype(np.float32)


def arm_minimol(smiles, batch=256):
    """MiniMol: a pretrained GNN fingerprint, 512-d."""
    from minimol import Minimol
    m = Minimol(batch_size=batch)
    out = []
    for i in range(0, len(smiles), batch * 4):
        chunk = list(smiles[i:i + batch * 4])
        out.append(np.stack([np.asarray(v, np.float32) for v in m(chunk)]))
        print(f"    minimol {min(i+batch*4, len(smiles))}/{len(smiles)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def _chemprop_mpnn(pretrained: bool):
    """A chemprop D-MPNN encoder. `pretrained` loads the CheMeleon foundation weights.

    The two arms differ ONLY in this flag, so `chemprop` (random init) is a clean architectural
    control for `chemeleon` (the same architecture, pretrained): whatever the untrained model
    resolves is what the message-passing structure gives you for free, before any learning.
    """
    from pathlib import Path as _P

    import torch
    from chemprop import nn as cnn
    from chemprop.models import MPNN
    if pretrained:
        ck = torch.load(_P.home() / ".chemprop" / "chemeleon_mp.pt", weights_only=True)
        mp = cnn.BondMessagePassing(**ck["hyper_parameters"])
        mp.load_state_dict(ck["state_dict"])
    else:
        mp = cnn.BondMessagePassing()
    return MPNN(mp, cnn.MeanAggregation(), cnn.RegressionFFN(input_dim=mp.output_dim))


def _chemprop_embed(smiles, pretrained, batch=256):
    import torch
    from chemprop import featurizers
    from chemprop.data import BatchMolGraph
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    model = _chemprop_mpnn(pretrained).eval()
    tag = "chemeleon" if pretrained else "chemprop"
    outs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            mols = [Chem.MolFromSmiles(s) for s in smiles[i:i + batch]]
            gs = [feat(m if m is not None else Chem.MolFromSmiles("C")) for m in mols]
            outs.append(model.fingerprint(BatchMolGraph(gs), None, None).numpy())
            if (i // batch) % 20 == 0:
                print(f"    {tag} {i}/{len(smiles)} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(outs).astype(np.float32)


def arm_chemeleon(smiles):
    return _chemprop_embed(smiles, pretrained=True)


def arm_chemprop(smiles):
    return _chemprop_embed(smiles, pretrained=False)


#: Arms whose width is a function of a version we control, checked against the cache. An arm
#: absent here is cached on the pair-set hash alone, which is right for a frozen third-party
#: model and wrong for anything in this repository.
def _hume_width():
    try:
        import hume
        return len(hume.ALL_COLUMNS) + 2048        # descriptors + the ECFP block arm_hume adds
    except Exception:
        return None


_ARM_WIDTH = {"hume": _hume_width()}

ARMS = {"ecfp": arm_ecfp, "r3cfp": arm_r3cfp, "r4cfp": arm_r4cfp, "desc": arm_desc, "hume": arm_hume,
        "notation": arm_notation,
        "chemberta_mlm": arm_chemberta_mlm, "chemberta_mtr": arm_chemberta_mtr,
        "molformer": arm_molformer,
        "selfies_ted": arm_selfies_ted, "minimol": arm_minimol,
        "chemeleon": arm_chemeleon, "chemprop": arm_chemprop}


def main() -> None:
    want = sys.argv[1:] or list(ARMS)
    EMB.mkdir(parents=True, exist_ok=True)
    pairs, order, bg = _all_smiles()
    allsmi = order + bg
    print(f"{len(pairs):,} pairs | {len(order):,} unique pair molecules | "
          f"{len(bg):,} background")
    json.dump({"order": order, "background": bg}, open(FIGA / "smiles_index.json", "w"))
    # CACHE ON THE CONTENT OF THE PAIR SET, NOT ON THE FILE EXISTING.
    #
    # This was `if f.exists(): cached`, which is wrong the moment an edit is added or a seed
    # changes: `smiles_index.json` is rewritten just above, so the index would describe 19,757
    # molecules while a stale .npz held 18,757 rows, and every consumer indexes the array BY
    # POSITION. That does not raise -- it silently pairs new molecules against old vectors, and
    # the figure it produces looks entirely normal. Found when `ch2_homolog` was added and all
    # four fast arms reported "cached" against a pair set that had grown by 1,000 pairs.
    idx_sha = hashlib.sha256(json.dumps({"order": order, "background": bg},
                                        sort_keys=True).encode()).hexdigest()
    for name in want:
        f = EMB / f"{name}.npz"
        if f.exists():
            try:
                prev = str(np.load(f)["index_sha"])
            except Exception:
                prev = ""
            # THE INDEX SHA IS NOT ENOUGH FOR `hume`, AND THIS SILENTLY SHIPPED A STALE ARM.
            # It hashes the PAIR SET, so it answers "were these the same molecules?" and not
            # "was this the same featuriser?". HUME's column set moved from 1,266 to 1,536
            # during this work and the cache happily reported "index matches", leaving figure A
            # scoring an embedding built by a different version of the thing the figure is
            # about. The width is a cheap, honest part of the key: any column added, dropped or
            # reordered changes it.
            width = ""
            try:
                width = str(np.load(f)["X"].shape[1])
            except Exception:
                pass
            want_width = _ARM_WIDTH.get(name)
            if prev == idx_sha and (want_width is None or width == str(want_width)):
                print(f"  {name}: cached (index matches)")
                continue
            if prev == idx_sha:
                print(f"  {name}: STALE -- {width} columns cached, featuriser now emits "
                      f"{want_width}; recomputing")
            else:
                print(f"  {name}: STALE -- built for a different pair set, recomputing")
        t0 = time.time()
        X = ARMS[name](allsmi)
        np.savez_compressed(f, X=X, n_pair=len(order), index_sha=idx_sha)
        print(f"  {name}: {X.shape} in {time.time()-t0:.0f}s -> {f.name}", flush=True)


if __name__ == "__main__":
    main()
