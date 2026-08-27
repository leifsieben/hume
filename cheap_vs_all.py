"""Do the 75 budget-fitting descriptors match the full 217?

Budget rule: the descriptor suite should cost no more than the fingerprint (33 us).
Measured cumulatively, the cheapest 75 RDKit descriptors cost 25.8 us together; all 217
cost 3,911 us -- a 150x difference driven by ~19 descriptors at >50 us each.

So the question is whether those expensive ones earn their keep. If ecfp+cheap75 matches
ecfp+all217, nothing needs predicting: just drop them. If a gap remains, that gap is
exactly the prediction target -- and it is ~19 chemically meaningful quantities
(MolLogP, qed, BertzCT, BCUT2D, EState aggregates, Chi3/4, BalabanJ, Ipc), not 1,613.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
PROF = Path("data/budget_profile.json")
OUT = Path("data/cheap_vs_all.json")

def _cv_rmse(X, y, smiles):
    import _vendor  # noqa: F401  - puts vendor/chemtfm on sys.path
    from chemtfm.bench import metrics as M
    from chemtfm.bench.datasets import REGRESSION
    from chemtfm.bench.splits import scaffold_folds, train_test
    from chemtfm.models.xgb import XGBModel
    folds = scaffold_folds(smiles, k=5, seed=0); out=[]
    for i in range(len(folds)):
        tr, te = train_test(folds, i)
        if float(np.std(y[tr])) == 0.0: continue
        out.append(M.rmse(y[te], XGBModel(task=REGRESSION).fit(X[tr], y[tr]).predict(X[te])))
    return float(np.mean(out)) if out else np.nan

def main():
    prof = json.load(open(PROF))
    order = [d["name"] for d in prof["descriptors"]]
    lut = dict(Descriptors._descList)
    cheap, alld = order[:75], order
    c = dict(np.load(CACHE, allow_pickle=True))
    smiles, y, offsets = c["smiles"], c["y"], c["offsets"]
    suite_of, name_of = c["suite_of"], c["name_of"]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    ecfp, D = [], []
    t0 = time.time()
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            ecfp.append(np.zeros(2048, np.float32)); D.append(np.full(len(alld), np.nan, np.float32)); continue
        ecfp.append(gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
        row = np.empty(len(alld), np.float32)
        for j, n in enumerate(alld):
            try: row[j] = lut[n](m)
            except Exception: row[j] = np.nan
        D.append(row)
        if (i+1) % 10000 == 0: print(f"  featurised {i+1:,} ({time.time()-t0:.0f}s)", flush=True)
    ecfp = np.stack(ecfp); D = np.stack(D)
    ci = np.arange(75)
    print(f"ECFP {ecfp.shape} | descriptors {D.shape}")

    arms = {"ecfp": lambda s: ecfp[s],
            "ecfp+cheap75": lambda s: np.hstack([ecfp[s], D[s][:, ci]]),
            "ecfp+all217":  lambda s: np.hstack([ecfp[s], D[s]])}
    for suite in ("moleculeace", "moleculenet"):
        per = {}
        for j, name in enumerate(name_of):
            if suite_of[j] != suite: continue
            s = slice(offsets[j], offsets[j+1])
            per[name] = {a: _cv_rmse(b(s), y[s], list(smiles[s])) for a, b in arms.items()}
        summ = {a: float(np.nanmean([r[a] for r in per.values()])) for a in arms}
        print(f"\n=== {suite} (n={len(per)}) ===")
        for a in arms:
            print(f"  {a:16s} {summ[a]:.4f}  vs ecfp {summ[a]-summ['ecfp']:+.4f}")
        OUT.write_text(json.dumps({suite: {"summary": summ, "per_dataset": per}}, indent=2)
                       if not OUT.exists() else
                       json.dumps({**json.loads(OUT.read_text()), suite: {"summary": summ, "per_dataset": per}}, indent=2))

if __name__ == "__main__":
    main()
