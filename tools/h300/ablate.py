"""Grouped ablation: does removing a block of columns cost anything measurable?

    .venv/bin/python tools/h300/ablate.py OUT.json

Stage 2 of HUME_300_METHOD.md. Stage 1 ranks columns; this decides. Groups rather than columns,
because gain splits between correlated columns and a correlated team has to be dropped together
or its members hide behind each other.

THE fr_* SPLIT IS DEFINED LEAVE-ONE-DATASET-OUT. The statistic that separates a useful sparse
flag from a dead one -- how many tasks it reaches the top 50 on -- is recomputed on the OTHER 32
datasets for every dataset it is scored on. Defining the split on all 33 and then reporting on
all 33 would let the split see its own test set, which is the selection bias the method document
commits to controlling.

`fr_tail` is a CONTROL, not a candidate. If the split means anything, keeping the 40 and dropping
the 32 should cost less than the reverse. If the two arms are level, the split is noise and the
right move is to treat fr_* as one block.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.simplefilter("ignore")


def metric(task, y_true, pred):
    """Lower is better in both, and comparable across datasets: 1-auroc, or rmse/sd(y)."""
    if task == "classif":
        from sklearn.metrics import roc_auc_score
        return 1.0 - roc_auc_score(y_true, pred)
    sd = float(np.std(y_true))
    return float(np.sqrt(np.mean((y_true - pred) ** 2)) / (sd if sd > 0 else 1.0))


def load(f):
    d = np.load(f, allow_pickle=True)
    names = [str(s) for s in d["column_names"]]
    if "task" in d.files:
        task = str(d["task"])
    else:
        raw = d["meta"]; raw = raw.item() if hasattr(raw, "item") else raw
        task = str((json.loads(raw) if isinstance(raw, str) else raw).get("task", ""))
    task = "classif" if task in ("binary", "classif", "classification") else "regress"
    X = np.where(np.isinf(d["X"]), np.nan, d["X"])     # pre-0.9.2 BalabanJ; see utility.py
    return names, task, X, d["y"], d["folds"]


def score(args):
    f, arms = args
    import xgboost as xgb
    names, task, X, y, folds = load(f)
    pos = {c: i for i, c in enumerate(names)}
    ds = f.split("/")[-1][:-4]
    out = {}
    for arm, cols in arms.items():
        idx = [pos[c] for c in cols if c in pos]
        per_fold = []
        for k in sorted(set(folds.tolist())):
            tr, te = folds != k, folds == k
            M = (xgb.XGBClassifier if task == "classif" else xgb.XGBRegressor)(
                n_estimators=300, max_depth=6, random_state=0, n_jobs=2, verbosity=0,
                colsample_bynode=0.3, tree_method="hist",
                **({"eval_metric": "logloss"} if task == "classif" else {}))
            M.fit(X[np.ix_(tr, idx)], y[tr])
            p = (M.predict_proba(X[np.ix_(te, idx)])[:, 1] if task == "classif"
                 else M.predict(X[np.ix_(te, idx)]))
            per_fold.append(metric(task, y[te], p))
        out[arm] = per_fold
    return ds, task, out


def main() -> None:
    out_path = sys.argv[1]
    import molhume
    mn = list(molhume.column_set("minimal"))
    util = json.load(open("/tmp/h300_utility.json"))
    ucols = util[list(util)[0]]["cols"]
    uds = sorted(util)
    RANK = {k: np.array(util[k]["rank"], float) for k in uds}
    fr_all = [c for c in mn if c.startswith("fr_")]
    fpos = {c: ucols.index(c) for c in fr_all if c in ucols}

    files = sorted(glob.glob("results/reanalysis/features/*.npz"))
    jobs = []
    for f in files:
        ds = f.split("/")[-1][:-4]
        # LEAVE THIS DATASET OUT of the statistic that defines the split.
        others = [k for k in uds if k != ds]
        t50 = np.array([sum(RANK[k][fpos[c]] <= 50 for k in others) for c in fr_all])
        top = {c for c, t in zip(fr_all, t50) if t >= 3}
        tail = [c for c in fr_all if c not in top]
        arms = {
            "full":    mn,
            "no_fr":   [c for c in mn if not c.startswith("fr_")],
            "fr_top":  [c for c in mn if not c.startswith("fr_") or c in top],
            "fr_tail": [c for c in mn if not c.startswith("fr_") or c in set(tail)],
        }
        jobs.append((f, arms))

    res = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, out in ex.map(score, jobs):
            res[ds] = {"task": task, **{a: v for a, v in out.items()}}
            base = float(np.mean(out["full"]))
            deltas = {a: 100 * (np.mean(v) - base) / base for a, v in out.items() if a != "full"}
            print(f"  {ds:16s} {task:8s} full={base:.4f}  "
                  + "  ".join(f"{a}={d:+.2f}%" for a, d in deltas.items()), flush=True)
            json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
