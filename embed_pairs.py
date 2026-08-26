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
    d = json.load(open(FIGA / "pairs.json"))
    pairs = d["pairs"]
    seen, order = set(), []
    for p in pairs:
        for s in (p["a"], p["b"]):
            if s not in seen:
                seen.add(s)
                order.append(s)
    # Background: benchmark molecules that appear in no pair.
    bench = list(np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)["smiles"])
    rng = np.random.default_rng(0)
    cand = [s for s in bench if s not in seen]
    bg = [cand[i] for i in rng.choice(len(cand), min(N_BACKGROUND, len(cand)), replace=False)]
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
    mod = AutoModel.from_pretrained(path, trust_remote_code=True, deterministic_eval=True) \
        if "MoLFormer" in str(path) else AutoModel.from_pretrained(path, trust_remote_code=True)
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


def arm_chemberta(smiles):
    return _hf(smiles, ROOT / "models_hf" / "ChemBERTa-2")


def arm_molformer(smiles):
    return _hf(smiles, ROOT / "models_hf" / "MolFormer")


ARMS = {"ecfp": arm_ecfp, "r3cfp": arm_r3cfp, "desc": arm_desc,
        "chemberta": arm_chemberta, "molformer": arm_molformer}


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
