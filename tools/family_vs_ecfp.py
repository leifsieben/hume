"""Which other families are already carried by the ECFP6 we emit alongside them?

Extends the fr_* consumer-inversion test to every family whose columns are, in mechanism, counts
of local atomic environments -- which is exactly what a circular fingerprint hashes.

Reported per column: detection AUROC (is it non-zero) which is prevalence-free, and R^2 on the
VALUE, which is the question for a column that is a count rather than a flag. A family is only a
drop candidate if both are high: knowing a pattern is present is not the same as knowing how
many times.
"""
import json
import re
import warnings

import numpy as np
import xgboost as xgb
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
names = list(molhume.feature_names(fingerprint=False))

FAM = {
    "E-state atom type": lambda c: re.match(r"^(N|S)[a-z]{1,3}[A-Z]", c) or re.match(r"^(MAX|MIN)", c),
    "ring perception":   lambda c: re.match(r"^n[0-9GA]", c) or "Ring" in c,
    "rdkit core counts": lambda c: c in ("NumHDonors","NumHAcceptors","NumRotatableBonds",
                                          "RingCount","NumAromaticRings","NumSaturatedRings",
                                          "NumAliphaticRings","HeavyAtomCount","NOCount","NHOHCount"),
    "constitutional":    lambda c: re.match(r"^(nAtom|nHeavyAtom|nBonds|nAromAtom|nAromBond|nC|nN|nO|nS|nP|nF|nCl|nBr|nI|nX|nRot|nHetero)", c),
}
smis = json.load(open("data/exactness_corpus.json"))["smiles"][:15000]
mols = [m for m in (__import__("rdkit").Chem.MolFromSmiles(s) for s in smis) if m is not None]
fp, X, _ = molhume.featurize_all_from_mols(mols, optional=("AvgIpc",))
fp = fp.astype(np.float32)
rng = np.random.default_rng(0); perm = rng.permutation(len(mols))
tr, te = perm[:12000], perm[12000:]
idx = {c: i for i, c in enumerate(names)}
out = {}
for fam, pred in FAM.items():
    cols = [c for c in names if pred(c)]
    rows = []
    for c in cols:
        y = np.nan_to_num(X[:, idx[c]].astype(np.float64))
        if y.std() == 0:
            continue
        occ = (y != 0).astype(int)
        auc = None
        if 10 <= occ[tr].sum() and occ[te].sum() >= 5 and occ[te].mean() < 1.0:
            clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, random_state=0, n_jobs=-1,
                                    verbosity=0, tree_method="hist", eval_metric="logloss")
            clf.fit(fp[tr], occ[tr])
            auc = float(roc_auc_score(occ[te], clf.predict_proba(fp[te])[:, 1]))
        reg = xgb.XGBRegressor(n_estimators=150, max_depth=6, random_state=0, n_jobs=-1,
                               verbosity=0, tree_method="hist")
        reg.fit(fp[tr], y[tr])
        p = reg.predict(fp[te])
        sst = ((y[te] - y[te].mean()) ** 2).sum()
        r2 = float(1 - ((y[te] - p) ** 2).sum() / sst) if sst > 0 else None
        rows.append(dict(column=c, fires=float(occ.mean()), auroc=auc, r2=r2))
    out[fam] = rows
    v = np.array([r["r2"] for r in rows if r["r2"] is not None])
    a = np.array([r["auroc"] for r in rows if r["auroc"] is not None])
    print(f"\n  {fam}  ({len(rows)} columns)")
    print(f"    value R2   : median {np.median(v):.3f}   {(v>=0.95).sum()} of {len(v)} at >=0.95"
          f"   {(v<0.8).sum()} below 0.80")
    if len(a):
        print(f"    detection  : median AUROC {np.median(a):.3f}   {(a>=0.99).sum()} of {len(a)} at >=0.99")
    for r in sorted([r for r in rows if r["r2"] is not None], key=lambda r: r["r2"])[:5]:
        print(f"      hardest: {r['column']:24s} R2 {r['r2']:6.3f}  fires {r['fires']:6.2%}")
json.dump(out, open("results/reanalysis/family_vs_ecfp.json", "w"), indent=1)
