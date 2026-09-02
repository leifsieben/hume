"""Does dropping ETA or the information-content block cost anything downstream?

High inversion R^2 says a column is REBUILDABLE from the others given 1,200 columns and 12,000
molecules with nothing else to do. It does not say a model fitting a real target with limited
data can rebuild it AND predict at the same time. That gap is precisely what cost physchem
3.83% under minimal-v1, so the drop candidates get checked against the real task before being
dropped, on the panel where the previous cut did the damage.

This is experiment F, and F is a CHECK, never the criterion -- a family is not kept because it
helps here, nor dropped because it does not. It exists to catch a large loss before it ships.
Same cached features, same stored scaffold folds, same untuned head as the grid.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

warnings.simplefilter("ignore")
FEAT = Path("results/reanalysis/features")
DATASETS = ["esol", "lipophilicity", "aqsoldb", "pb_logd", "pb_water_sol", "photoswitch"]

z0 = np.load(FEAT / "esol.npz", allow_pickle=False)
names = [str(n) for n in z0["column_names"]]
ETA = {c for c in names if c.startswith("ETA_")}
IC = {c for c in names if re.match(r"^(IC|BIC|MIC|ZMIC|AvgIpc)", c)}
ARMS = {"full": set(), "minus_ETA": ETA, "minus_IC": IC, "minus_both": ETA | IC}
print(f"  ETA {len(ETA)} columns, information content {len(IC)} columns\n")

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
                         sd=float(np.std(sc, ddof=1)), n_cols=F.shape[1]))
        print(f"  {ds:15s} {arm:11s} {np.mean(sc):8.4f}  (+/- {np.std(sc, ddof=1):.4f})  "
              f"{F.shape[1]} cols")

json.dump(rows, open("results/reanalysis/family_ablation.json", "w"), indent=1)
print(f"\n  {'arm':12s} {'mean rel. cost vs full':>24s} {'worst':>9s}")
base = {r["dataset"]: r["rmse"] for r in rows if r["arm"] == "full"}
for arm in ARMS:
    if arm == "full":
        continue
    d = [(r["rmse"] - base[r["dataset"]]) / base[r["dataset"]] for r in rows if r["arm"] == arm]
    print(f"  {arm:12s} {np.mean(d)*100:+23.2f}% {np.max(d)*100:+8.2f}%")
