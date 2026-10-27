"""Re-embed figA's transformer and MiniMol arms over the FULL index, in this interpreter.

    .venv-figa/bin/python    research/embed_figa_local.py chemberta_mlm chemberta_mtr molformer
    .venv-minimol/bin/python research/embed_figa_local.py minimol

EVERY ROW IS RECOMPUTED, not just the 972 new ones, and that is the whole point.

`protonate_neutral` pairs a NEUTRAL BASE -- an unmodified pool molecule already in the index --
with its conjugate acid, which is new. So each pair spans an old row and a new one. Appending
rows computed by a different function from the ones already stored would put the difference
between two embedding conventions inside the pair difference, and the panel would measure the
convention.

That is not hypothetical for MoLFormer. research/embed_bench.py records that the cached matrices
were produced WITHOUT `deterministic_eval=True`, and MoLFormer-XL resamples its linear-attention
random features on every forward pass without that flag: the same molecule embeds differently run
to run, measured max component difference 0.46 between two passes. An embedding matrix that is
not a function of its input cannot be appended to. Recomputing the whole matrix with the flag on
makes it one.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIGA = ROOT / "results" / "figures" / "figA"
HF = {"chemberta_mlm": "ChemBERTa-2-MLM", "chemberta_mtr": "ChemBERTa-2-MTR",
      "molformer": "MolFormer"}


def _hf(name, smiles, batch=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = ROOT / "models_hf" / HF[name]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    kw = {"trust_remote_code": True}
    if name == "molformer":
        kw["deterministic_eval"] = True          # see the module docstring
    mod = AutoModel.from_pretrained(path, **kw)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mod = mod.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            b = tok(list(smiles[i:i + batch]), return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(dev)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().float().numpy())
            if (i // batch) % 100 == 0:
                print(f"    {name} {i}/{len(smiles)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def _minimol(smiles, batch=256):
    from minimol import Minimol
    mm = Minimol()
    out, n = [], len(smiles)
    for i in range(0, n, batch):
        chunk = list(smiles[i:i + batch])
        try:
            v = mm(chunk)
            out.append(np.stack([np.asarray(x, dtype=np.float32) for x in v]))
        except Exception:
            # Graphium's featuriser refuses some molecules; NaN marks them rather than
            # shortening the matrix, which would desynchronise every row index after it.
            rows = []
            for s in chunk:
                try:
                    rows.append(np.asarray(mm([s])[0], dtype=np.float32))
                except Exception:
                    rows.append(None)
            w = next(r.shape[0] for r in rows if r is not None)
            out.append(np.stack([r if r is not None else np.full(w, np.nan, np.float32)
                                 for r in rows]))
        print(f"    minimol {min(i + batch, n)}/{n}", flush=True)
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    idx = json.loads((FIGA / "smiles_index.json").read_text())
    rows = list(idx["order"]) + list(idx["background"]) + list(idx.get("extra", []))
    sha = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    print(f"  {len(rows):,} molecules\n")
    for arm in sys.argv[1:]:
        fn = _minimol if arm == "minimol" else (lambda s, a=arm: _hf(a, s))
        X = fn(rows)
        assert X.shape[0] == len(rows), (X.shape, len(rows))
        np.savez_compressed(FIGA / "embeddings" / f"{arm}.npz", X=X,
                            n_pair=np.int64(len(idx["order"])), index_sha=np.array(sha))
        print(f"  {arm:<16s} {X.shape}  written\n", flush=True)


if __name__ == "__main__":
    main()
