"""Rank every remaining family by how much of it the rest of the library already carries.

For each family: remove the WHOLE family from the design, then ask the actual consumer to
rebuild each of its columns from what is left. A family whose columns all come back at high R^2
is a drop candidate; one with a low floor is carrying something nothing else has.

This is the same test that condemned ETA (median 0.9954 with the family gone) and cleared the
information-content block (median 0.9710, floor 0.910). Run over everything so the next cut is
chosen by evidence rather than by which family we happened to look at.

Large families are SAMPLED (every k-th column) for runtime; the floor is the number that matters
and sampling can only make it look better, so a low floor on a sample is conclusive while a high
one is provisional.
"""
import json
import re
import warnings

import numpy as np
import xgboost as xgb
from rdkit import RDLogger

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
names = list(molhume.feature_names(fingerprint=False, columns="full"))
idx = {c: i for i, c in enumerate(names)}

FAM = {
 "chi subgraph":        lambda c: re.match(r"^(A?X(p|c|pc|ch)-)", c) or re.match(r"^chi\d", c),
 "matrix spectrum":     lambda c: re.match(r"^(SpAbs|SpMax|SpDiam|SpMAD|VE\d|VR\d|SM1)", c),
 "Burden eigenvalue":   lambda c: c.startswith("BCUT"),
 "VSA binning":         lambda c: re.match(r"^(SlogP_VSA|SMR_VSA|PEOE_VSA|EState_VSA|VSA_EState)", c),
 "E-state atom type":   lambda c: re.match(r"^(N|S)[a-z]{1,3}[A-Z]", c) or re.match(r"^(MAX|MIN)", c),
 "path/walk counts":    lambda c: re.match(r"^(MPC|piPC|TPC|SRW|MWC|TWC)", c),
 "topological charge":  lambda c: re.match(r"^(GGI|JGI|JGT)", c),
 "ring perception":     lambda c: re.match(r"^n[0-9GA]", c) or "Ring" in c,
 "information content": lambda c: re.match(r"^(IC|BIC|MIC|ZMIC|AvgIpc)", c),
 "autocorrelation":     lambda c: re.match(r"^(AATSC|AATS|ATSC|ATS|MATS|GATS)\d", c),
}
smis = json.load(open("data/exactness_corpus.json"))["smiles"][:15000]
X = molhume.featurize(smis, standardize="none", fingerprint=False)
X = np.where(np.isfinite(X), X, np.nan)
rng = np.random.default_rng(0); p = rng.permutation(len(X)); tr, te = p[:12000], p[12000:]

out = []
for fam, pred in FAM.items():
    cols = [c for c in names if pred(c)]
    if not cols:
        continue
    others = [c for c in names if c not in set(cols)]
    F = X[:, [idx[c] for c in others]]
    sample = cols[:: max(1, len(cols) // 20)]
    r2s = []
    for c in sample:
        y = X[:, idx[c]]; ok = np.isfinite(y)
        if ok[te].sum() < 300 or np.nanstd(y[te]) == 0:
            continue
        m = xgb.XGBRegressor(n_estimators=150, max_depth=6, random_state=0, n_jobs=-1,
                             verbosity=0, tree_method="hist")
        m.fit(F[tr][ok[tr]], y[tr][ok[tr]])
        pr = m.predict(F[te][ok[te]]); yy = y[te][ok[te]]
        r2s.append(float(1 - ((yy - pr) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()))
    if r2s:
        out.append((fam, len(cols), len(r2s), float(np.median(r2s)), float(np.min(r2s)),
                    int(sum(1 for v in r2s if v < 0.9))))
out.sort(key=lambda r: -r[3])
print(f"\n  {'family':22s} {'cols':>5s} {'sampled':>8s} {'median R2':>10s} {'floor':>8s} {'<0.9':>5s}")
for fam, n, s, med, mn, lo in out:
    print(f"  {fam:22s} {n:5d} {s:8d} {med:10.4f} {mn:8.4f} {lo:5d}")
json.dump(out, open("results/reanalysis/family_screen.json", "w"), indent=1)
