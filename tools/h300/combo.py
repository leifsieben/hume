"""Do the individually-free reductions compose, and does it matter WHICH columns go?

    .venv/bin/python tools/h300/combo.py OUT.json

Three questions, one run:

  1. `combo418` -- all eight sweep reductions at once. Each was free with the other seven still
     present to absorb the loss. Whether they are free TOGETHER cannot be summed and has to be
     measured.

  2. `random418` x3 -- THE CONTROL. 418 columns drawn at random from the 590. If the principled
     combination is no better than a random subset of the same size, then which columns were
     chosen does not matter and the whole result is about column COUNT. Three seeds, because one
     draw can be lucky and a single control is not a control.

  3. `random300` x3 -- is 300 even reachable? If a RANDOM 300 is indistinguishable from 590, the
     selection problem is far easier than assumed and the spec should say so rather than claim a
     precision it does not have.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, "tools/h300")
from ablate import load, metric                                    # noqa: E402
from sweeps import build_arms                                      # noqa: E402

_ARMS: dict = {}


def _init(arms):
    _ARMS.clear(); _ARMS.update(arms)


def job(f):
    import xgboost as xgb
    names, task, X, y, folds = load(f)
    pos = {c: i for i, c in enumerate(names)}
    out = {}
    for arm, cols in _ARMS.items():
        idx = [pos[c] for c in cols if c in pos]
        per = []
        for k in sorted(set(folds.tolist())):
            tr, te = folds != k, folds == k
            M = (xgb.XGBClassifier if task == "classif" else xgb.XGBRegressor)(
                n_estimators=300, max_depth=6, random_state=0, n_jobs=2, verbosity=0,
                colsample_bynode=0.3, tree_method="hist",
                **({"eval_metric": "logloss"} if task == "classif" else {}))
            M.fit(X[np.ix_(tr, idx)], y[tr])
            p = (M.predict_proba(X[np.ix_(te, idx)])[:, 1] if task == "classif"
                 else M.predict(X[np.ix_(te, idx)]))
            per.append(metric(task, y[te], p))
        out[arm] = per
    return f.split("/")[-1][:-4], task, out


def main() -> None:
    out_path = sys.argv[1]
    import molhume
    mn = list(molhume.column_set("minimal"))
    util = json.load(open("/tmp/h300_utility.json"))
    ucols = util[list(util)[0]]["cols"]
    R = {k: np.array(util[k]["rank"], float) for k in util}
    fr_all = [c for c in mn if c.startswith("fr_")]
    fpos = {c: ucols.index(c) for c in fr_all}
    t50 = np.array([sum(R[k][fpos[c]] <= 50 for k in R) for c in fr_all])
    fr_top = {c for c, t in zip(fr_all, t50) if t >= 3}
    base590 = [c for c in mn if not c.startswith("fr_") or c in fr_top]

    # every column any _lo arm dropped, i.e. the union of the eight high/long halves
    sw = build_arms(base590)
    dropped = set()
    for a, cols in sw.items():
        if a.endswith("_lo"):
            dropped |= set(base590) - set(cols)
    combo = [c for c in base590 if c not in dropped]

    arms = {"base590": base590, f"combo{len(combo)}": combo}
    for n in (len(combo), 300):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            arms[f"random{n}_s{seed}"] = [base590[i] for i in
                                          sorted(rng.choice(len(base590), n, replace=False))]
    print(f"  baseline {len(base590)}   combo {len(combo)} (drops {len(dropped)})")
    print(f"  {len(arms)} arms x 33 datasets x 5 folds\n", flush=True)

    res = {}
    files = sorted(glob.glob("results/reanalysis/features/*.npz"))
    with ProcessPoolExecutor(max_workers=5, initializer=_init, initargs=(arms,)) as ex:
        for ds, task, out in ex.map(job, files):
            res[ds] = {"task": task, **out}
            b = float(np.mean(out["base590"]))
            line = "  ".join(f"{a}={100*(np.mean(v)-b)/b:+.2f}%"
                             for a, v in out.items()
                             if a in (f"combo{len(combo)}", "random418_s0", "random300_s0"))
            print(f"  {ds:16s} base={b:.4f}  {line}", flush=True)
            json.dump({"n_combo": len(combo), "res": res}, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
