"""418 -> 250 / 300 / 350, stratified across families, against random controls of each size.

    .venv/bin/python tools/h300/strat.py OUT.json

THE RULE, and it is the one the controls in section 8 identified rather than the one section 7
proposed: preserve COVERAGE. For each of molhume's 19 families keep a proportional share, never
fewer than one. Within a family, members are ordered by utility rank -- which is safe here in a
way it was not in Stage 1, because the comparison is between columns of the SAME family, so the
gain-splitting bias that made cross-family ranking untrustworthy applies equally to both sides.

molhume.FAMILY_OFFSETS is used for the grouping rather than the semantic groups invented for the
overview: it is the library's own, documented, and was not chosen with these results in view.

SELECTION IS LEAVE-ONE-DATASET-OUT. The within-family ranking for a dataset is computed from the
other 32, so no arm sees the data it is scored on. The arms differ slightly per dataset by
construction; their SIZES do not.
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
SIZES = (350, 300, 250)


def _init(arms):
    _ARMS.clear(); _ARMS.update(arms)


def job(payload):
    f, arms = payload
    import xgboost as xgb
    names, task, X, y, folds = load(f)
    pos = {c: i for i, c in enumerate(names)}
    out = {}
    for arm, cols in arms.items():
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


def stratify(pool, fam_of, rank, n_keep):
    """Proportional share per family, never fewer than one, best within-family rank first."""
    fams = {}
    for c in pool:
        fams.setdefault(fam_of.get(c, "other"), []).append(c)
    out, quota = [], {}
    for fam, cs in fams.items():
        quota[fam] = max(1, round(len(cs) * n_keep / len(pool)))
    # trim or pad to hit n_keep exactly, always from the largest families first
    while sum(quota.values()) > n_keep:
        big = max(quota, key=lambda f: (quota[f], len(fams[f])))
        if quota[big] <= 1: break
        quota[big] -= 1
    while sum(quota.values()) < n_keep:
        big = max(quota, key=lambda f: len(fams[f]) - quota[f])
        if quota[big] >= len(fams[big]): break
        quota[big] += 1
    for fam, cs in fams.items():
        out += sorted(cs, key=lambda c: rank.get(c, 1e9))[:quota[fam]]
    return sorted(set(out))


def main() -> None:
    out_path = sys.argv[1]
    import molhume
    mn = list(molhume.column_set("minimal"))
    util = json.load(open("/tmp/h300_utility.json"))
    ucols = util[list(util)[0]]["cols"]
    RK = {k: np.array(util[k]["rank"], float) for k in util}
    upos = {c: i for i, c in enumerate(ucols)}
    fr_all = [c for c in mn if c.startswith("fr_")]
    t50 = np.array([sum(RK[k][upos[c]] <= 50 for k in RK) for c in fr_all])
    fr_top = {c for c, t in zip(fr_all, t50) if t >= 3}
    base590 = [c for c in mn if not c.startswith("fr_") or c in fr_top]
    sw = build_arms(base590)
    dropped = set()
    for a, cols in sw.items():
        if a.endswith("_lo"):
            dropped |= set(base590) - set(cols)
    combo = [c for c in base590 if c not in dropped]

    allc = molhume.ALL_COLUMNS
    fam_of = {}
    for fam, (a, b) in molhume.FAMILY_OFFSETS.items():
        for c in allc[a:b]:
            fam_of[c] = fam
    print(f"  pool = combo418 ({len(combo)} columns), {len(set(fam_of[c] for c in combo))} families")

    files = sorted(glob.glob("results/reanalysis/features/*.npz"))
    jobs = []
    for f in files:
        ds = f.split("/")[-1][:-4]
        others = [k for k in RK if k != ds]                      # leave this dataset out
        mean_rank = {c: float(np.mean([RK[k][upos[c]] for k in others])) for c in combo}
        arms = {"base590": base590, "combo418": combo}
        for n in SIZES:
            arms[f"strat{n}"] = stratify(combo, fam_of, mean_rank, n)
            for seed in (0, 1):
                rng = np.random.default_rng(1000 * n + seed)
                arms[f"random{n}_s{seed}"] = [combo[i] for i in
                                              sorted(rng.choice(len(combo), n, replace=False))]
        jobs.append((f, arms))
    a0 = jobs[0][1]
    print("  arm sizes: " + "  ".join(f"{k}={len(v)}" for k, v in a0.items()))
    print(f"  {len(a0)} arms x {len(files)} datasets x 5 folds\n", flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, out in ex.map(job, jobs):
            res[ds] = {"task": task, **out}
            b = float(np.mean(out["base590"]))
            line = "  ".join(f"{a}={100*(np.mean(out[a])-b)/b:+.2f}%"
                             for a in ("combo418", "strat350", "strat300", "strat250"))
            print(f"  {ds:16s} base={b:.4f}  {line}", flush=True)
            json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()


def job_seeded(payload):
    """Like job(), but any arm whose name starts with "null" is fit with a different seed.

    That arm changes NOTHING about the columns -- it is the full set scored twice -- so its
    spread across panels is the noise floor of the panel itself, which is what the decision
    thresholds are calibrated against.
    """
    import xgboost as xgb
    f, arms = payload
    names, task, X, y, folds = load(f)
    pos = {c: i for i, c in enumerate(names)}
    out = {}
    for arm, cols in arms.items():
        idx = [pos[c] for c in cols if c in pos]
        seed = 12345 if arm.startswith("null") else 0
        per = []
        for k in sorted(set(folds.tolist())):
            tr, te = folds != k, folds == k
            M = (xgb.XGBClassifier if task == "classif" else xgb.XGBRegressor)(
                n_estimators=300, max_depth=6, random_state=seed, n_jobs=2, verbosity=0,
                colsample_bynode=0.3, tree_method="hist",
                **({"eval_metric": "logloss"} if task == "classif" else {}))
            M.fit(X[np.ix_(tr, idx)], y[tr])
            p = (M.predict_proba(X[np.ix_(te, idx)])[:, 1] if task == "classif"
                 else M.predict(X[np.ix_(te, idx)]))
            per.append(metric(task, y[te], p))
        out[arm] = per
    return f.split("/")[-1][:-4], task, out
