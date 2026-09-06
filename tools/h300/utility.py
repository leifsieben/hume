"""Which of the 622 minimal descriptors earn their place on 33 MPP tasks.

    .venv/bin/python tools/h300/utility.py OUT.json

Reads the cached features (results/reanalysis/features/*.npz), fits the SAME untuned XGBoost head
and the SAME stored scaffold folds the downstream grid uses, and records per-column gain.

WHY GAIN AND NOT A CORRELATION SCREEN. The question is which columns a TREE uses, and a tree does
not use linear combinations -- that is the exact error minimal-v1 made, where columns linearly
recoverable at R^2 0.997 were not recoverable by a depth-6 tree. Gain is measured on the consumer
that actually reads these features.

Aggregated as MEAN RANK ACROSS DATASETS, not summed gain: gain is not comparable between tasks
with different targets and scales, and summing it would let one large dataset choose the spec.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
import xgboost as xgb                                        # noqa: E402
import molhume                                               # noqa: E402


def main() -> None:
    out_path = sys.argv[1]
    mn = list(molhume.column_set("minimal"))
    per_ds = {}
    for f in sorted(glob.glob("results/reanalysis/features/*.npz")):
        d = np.load(f, allow_pickle=True)
        names = [str(s) for s in d["column_names"]]
        pos = {c: i for i, c in enumerate(names)}
        cols = [c for c in mn if c in pos]
        X, y, folds = d["X"][:, [pos[c] for c in cols]], d["y"], d["folds"]
        # inf -> NaN, WHICH XGBOOST TREATS AS MISSING AND inf IT REFUSES OUTRIGHT.
        # These caches were built with molhume 0.2.0, before 0.9.2 fixed BalabanJ's division by
        # zero on disconnected acyclic molecules, and 8 of the 33 datasets carry the result --
        # hiv, aqsoldb, cyp2d6_inh, herg and the four wong_* sets, i.e. exactly the ones with
        # salts in them. Recomputing the cache is the real fix and is a separate job; masking
        # here keeps this analysis from being decided by a bug we have already shipped a fix for.
        n_inf = int(np.isinf(X).sum())
        if n_inf:
            X = np.where(np.isinf(X), np.nan, X)
        # TWO CACHE GENERATIONS. The older files carry `task`/`metric`; the newer ones fold the
        # same information into a `meta` dict. Reading whichever is present beats regenerating
        # 477,115 molecules of features to make the keys agree.
        if "task" in d.files:
            task = str(d["task"])
        else:
            raw = d["meta"]
            raw = raw.item() if hasattr(raw, "item") else raw
            meta = json.loads(raw) if isinstance(raw, str) else raw   # it is a JSON string
            task = str(meta.get("task", ""))
        task = "classif" if task in ("binary", "classif", "classification") else "regress"
        ds = f.split("/")[-1][:-4]
        gain = np.zeros(len(cols))
        for k in sorted(set(folds.tolist())):
            tr = folds != k
            M = (xgb.XGBClassifier if task == "classif" else xgb.XGBRegressor)(
                n_estimators=300, max_depth=6, random_state=0, n_jobs=-1, verbosity=0,
                colsample_bynode=0.3, tree_method="hist",
                **({"eval_metric": "logloss"} if task == "classif" else {}))
            M.fit(X[tr], y[tr])
            b = M.get_booster().get_score(importance_type="gain")
            for kk, v in b.items():
                gain[int(kk[1:])] += v
        # rank within the dataset: 1 = most used. Unused columns share the worst rank.
        order = np.argsort(-gain)
        rank = np.empty(len(cols), int); rank[order] = np.arange(1, len(cols) + 1)
        rank[gain == 0] = len(cols)
        per_ds[ds] = {"task": task, "cols": cols, "rank": rank.tolist(),
                      "used": int((gain > 0).sum()), "n_inf_masked": n_inf}
        print(f"  {ds:16s} {task:8s} {int((gain>0).sum()):3d}/{len(cols)} columns used"
              + (f"   ({n_inf} inf masked)" if n_inf else ""), flush=True)
    json.dump(per_ds, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
