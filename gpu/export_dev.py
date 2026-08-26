"""Export the scoreable DEV datasets to plain npz, so the remote box needs no ChemPFN install.

The lake is 46 GB, almost all of it sets we are not scoring (litpcba, fsmol_train, acnet).
Exporting the 28 scoreable DEV sets as (smiles, y, task) triples ships ~200 MB instead, and the
grid runner then depends on nothing but numpy, rdkit, mordred and xgboost.

Large sets are subsampled with a fixed seed and the cap is RECORDED in the manifest, because
"we scored qm9" and "we scored 50k of qm9" are different claims and the difference must travel
with the number rather than live in someone's memory.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.home() / "VSCode" / "ChemPFN"))
OUT = Path(__file__).resolve().parent / "devsets"
CAP = 50_000


def main():
    from chempfn.data.lake import SPECS, load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    dev = [s for s in SPECS.values()
           if s.role == "DEV" and s.task in ("regression", "binary", "multiclass")
           and not s.group_col and not s.label_cols]
    man = {}
    for s in sorted(dev, key=lambda x: x.name):
        try:
            d = load_dataset(s.name)
        except Exception as e:
            print(f"  SKIP {s.name}: {type(e).__name__}: {e}")
            continue
        smi, y = list(d.smiles), np.asarray(d.y, np.float64)
        n_full = len(smi)
        capped = False
        if n_full > CAP:
            idx = np.sort(np.random.default_rng(0).choice(n_full, CAP, replace=False))
            smi, y, capped = [smi[i] for i in idx], y[idx], True
        np.savez_compressed(OUT / f"{s.name}.npz", smiles=np.array(smi, dtype=object),
                            y=y, task=d.task)
        man[s.name] = {"task": d.task, "n": len(smi), "n_full": n_full, "capped": capped}
        print(f"  {s.name:18s} {d.task:11s} {len(smi):7,}"
              f"{f'  (capped from {n_full:,})' if capped else ''}")
    json.dump(man, open(OUT / "manifest.json", "w"), indent=2)
    tot = sum(v["n"] for v in man.values())
    print(f"\n{len(man)} datasets, {tot:,} molecules, "
          f"{sum(f.stat().st_size for f in OUT.glob('*.npz'))/1e6:.0f} MB")


if __name__ == "__main__":
    main()
