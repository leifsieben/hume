"""Is the autocorrelation block important, or merely irreducible?

It is the least reproducible family in the library and has full exact rank -- nothing else
rebuilds it and no column is an identity of the others. That establishes DISTINCTNESS, not
usefulness. This removes the whole block (227 kept columns, 29% of the planned library) and
measures the downstream cost.

WHY THIS ONE CAN RESOLVE WHAT THE ETA ABLATION COULD NOT. Removing 29 ETA columns produced an
effect smaller than the fold-to-fold spread on all six datasets, and the internal check failed
(dropping both ETA and IC looked cheaper than either alone). Removing 227 columns is an eight
times larger intervention. If that also lands inside the noise, the block is genuinely not
carrying much for these endpoints -- which is a result, not a failed measurement.

photoswitch is EXCLUDED: n=392 with a fold SD of 45% of its RMSE, which is not a measurement.
Keeping it in the earlier ablation is what made the mean disagree with the median.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

warnings.simplefilter("ignore")
FEAT = Path("results/reanalysis/features")
DATASETS = ["esol", "lipophilicity", "aqsoldb", "pb_logd", "pb_water_sol"]
KEEPW = {"c", "d", "dv", "s", "Z", "pe", "v"}
OPS = ["AATSC", "AATS", "ATSC", "ATS", "MATS", "GATS"]

z0 = np.load(FEAT / "esol.npz", allow_pickle=False)
names = [str(n) for n in z0["column_names"]]
def ac(c):
    for op in OPS:
        m = re.fullmatch(rf"{op}(\d)([A-Za-z]+)", c)
        if m: return op, int(m.group(1)), m.group(2)
KEPT_AC = {c for c in names if (p := ac(c)) and p[2] in KEEPW and p[0] != "MATS"
           and not (p[0] in ("ATS", "AATS") and p[1] not in {0, 2, 4, 6, 8})}
ALL_AC = {c for c in names if ac(c)}
ARMS = {"full": set(), "minus_kept_autocorr": KEPT_AC, "minus_all_autocorr": ALL_AC}
print(f"  kept autocorrelation {len(KEPT_AC)} columns, whole block {len(ALL_AC)}\n")
rows = []
for ds in DATASETS:
    z = np.load(FEAT / f"{ds}.npz", allow_pickle=False)
    X, fp, y, folds = z["X"], z["fp"], z["y"], z["folds"]
    X = np.where(np.isfinite(X), X, np.nan)
    for arm, drop in ARMS.items():
        keep = [i for i, c in enumerate(names) if c not in drop]
        F = np.hstack([X[:, keep], fp]).astype(np.float32)
        sc = []
        for k in range(5):
            te = folds == k; tr = ~te
            m = xgb.XGBRegressor(n_estimators=300, max_depth=6, random_state=0, n_jobs=-1,
                                 verbosity=0, colsample_bynode=0.3, tree_method="hist")
            m.fit(F[tr], y[tr])
            sc.append(float(np.sqrt(np.mean((y[te] - m.predict(F[te])) ** 2))))
        rows.append(dict(dataset=ds, arm=arm, rmse=float(np.mean(sc)),
                         sd=float(np.std(sc, ddof=1)), folds=sc))
        print(f"  {ds:15s} {arm:20s} {np.mean(sc):8.4f} (+/- {np.std(sc, ddof=1):.4f})")
json.dump(rows, open("results/reanalysis/autocorr_ablation.json", "w"), indent=1)
by = {(r["dataset"], r["arm"]): r for r in rows}
print(f"\n  {'arm':22s} {'mean cost':>10s} {'median':>9s} {'worst':>9s}  vs fold noise")
for arm in ARMS:
    if arm == "full": continue
    d = [(by[(ds, arm)]["rmse"] - by[(ds, "full")]["rmse"]) / by[(ds, "full")]["rmse"]
         for ds in DATASETS]
    noise = [by[(ds, "full")]["sd"] / by[(ds, "full")]["rmse"] for ds in DATASETS]
    above = sum(1 for x, n in zip(d, noise) if x > n)
    print(f"  {arm:22s} {np.mean(d)*100:+9.2f}% {np.median(d)*100:+8.2f}% {np.max(d)*100:+8.2f}%"
          f"   {above} of {len(d)} datasets above their own fold SD")
