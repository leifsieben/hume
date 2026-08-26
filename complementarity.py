"""Does Mordred add anything to ECFP? ChemPFN's c0 ablations never included it.

c0_xgb_v2.json covers ecfp x apfp x erg x desc x uma but has no Mordred channel, and my own
Gate 1 / eval arms were all built on `desc` (RDKit-96) rather than ECFP. So the production
question — "given that ECFP is already in the scheme, does Mordred earn its 67 ms/mol?" —
has not actually been measured.

Arms, on the 30 MoleculeACE regression sets, scaffold 5-fold, untuned XGBoost:

    ecfp                    2048            29 us/mol
    ecfp+desc               2144         1,005 us/mol
    ecfp+mordred            3661        67,284 us/mol
    ecfp+desc+mordred       3757        68,260 us/mol

True Mordred and RDKit-96 for the benchmark molecules come from the Gate 1 cache, so no new
featurisation is needed beyond ECFP.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
import sys as _s
OUT = Path(__file__).resolve().parent / "data" / f"complementarity_{_s.argv[1] if len(_s.argv)>1 else 'moleculeace'}.json"


def _cv_rmse(X, y, smiles) -> float:
    from chemtfm.bench import metrics as M
    from chemtfm.bench.datasets import REGRESSION
    from chemtfm.bench.splits import scaffold_folds, train_test
    from chemtfm.models.xgb import XGBModel

    folds = scaffold_folds(smiles, k=5, seed=0)
    out = []
    for i in range(len(folds)):
        tr, te = train_test(folds, i)
        if float(np.std(y[tr])) == 0.0:
            continue
        out.append(M.rmse(y[te], XGBModel(task=REGRESSION).fit(X[tr], y[tr]).predict(X[te])))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    c = dict(np.load(CACHE, allow_pickle=True))
    smiles, y, offsets = c["smiles"], c["y"], c["offsets"]
    suite_of, name_of = c["suite_of"], c["name_of"]
    desc, mordred = c["d_bench"], c["md_bench"]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    ecfp = np.stack([
        gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        if (m := Chem.MolFromSmiles(s)) is not None else np.zeros(2048, np.float32)
        for s in smiles])
    print(f"ECFP {ecfp.shape} | desc {desc.shape} | mordred {mordred.shape}")

    arms = {
        "ecfp": lambda s: ecfp[s],
        "ecfp+desc": lambda s: np.hstack([ecfp[s], desc[s]]),
        "ecfp+mordred": lambda s: np.hstack([ecfp[s], mordred[s]]),
        "ecfp+desc+mordred": lambda s: np.hstack([ecfp[s], desc[s], mordred[s]]),
    }

    import sys
    suite = sys.argv[1] if len(sys.argv) > 1 else "moleculeace"
    per_ds, t0 = {}, time.time()
    for j, name in enumerate(name_of):
        if suite_of[j] != suite:
            continue
        s = slice(offsets[j], offsets[j + 1])
        smi_j, y_j = list(smiles[s]), y[s]
        per_ds[name] = {a: _cv_rmse(b(s), y_j, smi_j) for a, b in arms.items()}
        print(f"  {name}: " + "  ".join(f"{a}={per_ds[name][a]:.3f}" for a in arms)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    summary = {a: float(np.nanmean([r[a] for r in per_ds.values()])) for a in arms}
    OUT.write_text(json.dumps({"summary": summary, "per_dataset": per_ds}, indent=2))

    base = summary["ecfp"]
    print(f"\n=== MoleculeACE (n={len(per_ds)}), does anything beat ECFP alone? ===")
    for a in arms:
        wins = sum(1 for r in per_ds.values() if r[a] < r["ecfp"])
        print(f"  {a:20s} {summary[a]:.4f}  vs ecfp {summary[a] - base:+.4f}  "
              f"helps on {wins}/{len(per_ds)}")


if __name__ == "__main__":
    main()
