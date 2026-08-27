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


ARMS = {"ecfp": arm_ecfp, "r3cfp": arm_r3cfp, "desc": arm_desc,
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
    for name in want:
        f = EMB / f"{name}.npz"
        if f.exists():
            print(f"  {name}: cached")
            continue
        t0 = time.time()
        X = ARMS[name](allsmi)
        np.savez_compressed(f, X=X, n_pair=len(order))
        print(f"  {name}: {X.shape} in {time.time()-t0:.0f}s -> {f.name}", flush=True)


if __name__ == "__main__":
    main()
