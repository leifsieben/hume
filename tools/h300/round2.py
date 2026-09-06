"""HUME_N round 2: everything from scratch on the SELECTION panels only.

    .venv/bin/python tools/h300/round2.py STAGE OUT.json     STAGE in {utility, ladder}

Follows HUME_N_PREREGISTRATION.md, which was committed before this ran. Nothing here reads the
33 held-out Figure C panels -- not their labels, and not their molecules for the unsupervised
statistics either.
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, "tools/h300")
from ablate import load, metric                                    # noqa: E402

SPLIT = json.load(open("results/h300_split.json"))
SEL = [f"{d}/{n}.npz" for d, n in SPLIT["selection"]]
LADDER = (128, 200, 256, 300, 350, 418)


def fit_gain(f):
    import xgboost as xgb
    names, task, X, y, folds = load(f)
    import molhume
    mn = [c for c in molhume.column_set("minimal") if c in set(names)]
    pos = {c: i for i, c in enumerate(names)}
    idx = [pos[c] for c in mn]
    gain = np.zeros(len(mn))
    for k in sorted(set(folds.tolist())):
        tr = folds != k
        M = (xgb.XGBClassifier if task == "classif" else xgb.XGBRegressor)(
            n_estimators=300, max_depth=6, random_state=0, n_jobs=2, verbosity=0,
            colsample_bynode=0.3, tree_method="hist",
            **({"eval_metric": "logloss"} if task == "classif" else {}))
        M.fit(X[np.ix_(tr, idx)], y[tr])
        for kk, v in M.get_booster().get_score(importance_type="gain").items():
            gain[int(kk[1:])] += v
    order = np.argsort(-gain)
    rank = np.empty(len(mn), int); rank[order] = np.arange(1, len(mn) + 1)
    rank[gain == 0] = len(mn)
    return Path(f).stem, task, mn, rank.tolist()


def stage_utility(out_path):
    res = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, cols, rank in ex.map(fit_gain, SEL):
            res[ds] = {"task": task, "cols": cols, "rank": rank}
            if len(res) % 15 == 0:
                print(f"  {len(res)}/{len(SEL)}", flush=True)
            json.dump(res, open(out_path, "w"))
    print(f"  {len(res)} panels -> {out_path}")


def build_pool_and_eff(util):
    """The 622 -> fr_* trim -> sweep halving -> pool, plus per-family effective rank.

    The correlation matrix comes from SELECTION panels only, per the pre-registration.
    """
    import molhume
    from sweeps import build_arms
    mn = list(molhume.column_set("minimal"))
    ucols = util[list(util)[0]]["cols"]
    upos = {c: i for i, c in enumerate(ucols)}
    RK = {k: np.array(util[k]["rank"], float) for k in util}
    fr = [c for c in mn if c.startswith("fr_")]
    t50 = np.array([sum(RK[k][upos[c]] <= 50 for k in RK if c in upos) for c in fr])
    keep_fr = {c for c, t in zip(fr, t50) if t >= 3}
    base = [c for c in mn if not c.startswith("fr_") or c in keep_fr]
    sw = build_arms(base)
    dropped = set()
    for a, cols in sw.items():
        if a.endswith("_lo"):
            dropped |= set(base) - set(cols)
    pool = [c for c in base if c not in dropped]

    # rank-space correlation on SELECTION molecules only
    from scipy.stats import rankdata
    Xs, names = [], None
    for f in SEL:
        d = np.load(f, allow_pickle=True)
        if names is None:
            names = [str(s) for s in d["column_names"]]
        Xs.append(d["X"][:, [names.index(c) for c in pool]])
    X = np.vstack(Xs).astype(np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    if len(X) > 60000:
        X = X[np.random.default_rng(0).choice(len(X), 60000, replace=False)]
    R = np.apply_along_axis(rankdata, 0, X)
    sd = R.std(0); R = (R - R.mean(0)) / np.where(sd > 0, sd, 1)
    allc = molhume.ALL_COLUMNS
    fam_of = {c: f for f, (a, b) in molhume.FAMILY_OFFSETS.items() for c in allc[a:b]}
    fams = {}
    for i, c in enumerate(pool):
        fams.setdefault(fam_of.get(c, "other"), []).append(i)
    eff = {}
    for f, idx in fams.items():
        if len(idx) < 2:
            eff[f] = 1; continue
        ev = np.clip(np.linalg.eigvalsh(np.cov(R[:, idx], rowvar=False))[::-1], 0, None)
        eff[f] = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95)) + 1
    mr = {c: float(np.mean([RK[k][upos[c]] for k in RK if c in upos])) for c in pool}
    return mn, pool, fam_of, eff, mr


def stage_ladder(out_path):
    from rankalloc import alloc
    from strat import job
    util = json.load(open("/tmp/r2_utility.json"))
    mn, pool, fam_of, eff, mr = build_pool_and_eff(util)
    print(f"  622 -> pool {len(pool)}   sum of per-family effective ranks = {sum(eff.values())}")
    print(f"  (the pre-registered label-free cross-check on N)")
    arms = {"full622": mn, f"pool{len(pool)}": pool}
    for n in LADDER:
        if n >= len(pool):
            continue
        arms[f"rank{n}"] = alloc(pool, fam_of, mr, eff, n)
        for s in (0, 1, 2):
            rng = np.random.default_rng(9000 + 10 * n + s)
            arms[f"random{n}_s{s}"] = [mn[i] for i in
                                       sorted(rng.choice(len(mn), n, replace=False))]
    print("  arms: " + "  ".join(f"{k}={len(v)}" for k, v in arms.items() if "_s1" not in k
                                 and "_s2" not in k), flush=True)
    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, out in ex.map(job, [(f, arms) for f in SEL]):
            res[ds] = {"task": task, **out}
            done += 1
            if done % 15 == 0:
                print(f"  {done}/{len(SEL)}", flush=True)
            json.dump({"eff_sum": sum(eff.values()), "pool": len(pool), "res": res},
                      open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    {"utility": stage_utility, "ladder": stage_ladder}[sys.argv[1]](sys.argv[2])
