"""Stage 2, part 2: reduce the RESOLUTION of each parametric sweep and measure the cost.

    .venv/bin/python tools/h300/sweeps.py OUT.json

Baseline is the 590 columns left after the fr_* decision, not the 622: decisions compose, and
measuring each against the original would hide interactions between them.

EVERY SWEEP GETS A CONTROL that keeps the same NUMBER of columns and the wrong ONES -- long lags
instead of short, high orders instead of low. Without it a cheap-looking reduction cannot be told
from a sweep that carries nothing either way, which is exactly what the fr_* control settled.

A reduction is only interesting if it is (a) indistinguishable from the baseline AND (b) better
than its own control. (a) alone means the columns are free; (a) and (b) together mean the
ORDERING principle -- low orders and short lags carry the information -- is real and can be
applied to the next sweep without re-measuring.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, "tools/h300")
from ablate import load, metric                                   # noqa: E402


def order_of(c):
    """The sweep parameter in a column name: chi3n -> 3, Xp-5dv -> 5, RW12_mean -> 12."""
    m = re.search(r"(\d+)", c)
    return int(m.group(1)) if m else None


def build_arms(base):
    """-> {arm: columns}. Each arm touches ONE sweep and leaves the rest of `base` alone."""
    B = set(base)
    def keep(drop):                      # everything except these
        return [c for c in base if c not in set(drop)]
    A = {"base590": list(base)}

    def sweep(name, members, low, high):
        """`low` and `high` partition `members`; drop one side, then the other as control."""
        low, high = [c for c in low if c in B], [c for c in high if c in B]
        if not low or not high:
            return
        A[f"{name}_lo"] = keep(high)      # keep the low orders / short lags
        A[f"{name}_hi"] = keep(low)       # CONTROL: keep the high ones instead
        A[f"__sizes_{name}"] = (len(low), len(high))

    chi = [c for c in base if re.match(r"^(Chi|chi|Xp-|Xc-|Xch-|Xpc-)", c)]
    sweep("chi", chi, [c for c in chi if (order_of(c) or 0) <= 3],
                      [c for c in chi if (order_of(c) or 0) > 3])

    ic = [c for c in base if re.match(r"^(IC|BIC|MIC|ZMIC)\d", c)]
    sweep("ic", ic, [c for c in ic if order_of(c) <= 2], [c for c in ic if order_of(c) > 2])

    gg = [c for c in base if re.match(r"^(GGI|JGI)\d", c)]
    sweep("charge", gg, [c for c in gg if order_of(c) <= 4], [c for c in gg if order_of(c) > 4])

    ac = [c for c in base if re.match(r"^(RATSC|RPAIR|SATS|TATS|XATS)", c)]
    sweep("autocorr", ac, [c for c in ac if (order_of(c) or 0) <= 2],
                          [c for c in ac if (order_of(c) or 0) > 2])

    rw = [c for c in base if re.match(r"^RW\d", c)]
    sweep("walk", rw, [c for c in rw if order_of(c) <= 4], [c for c in rw if order_of(c) > 4])

    rings = [c for c in base if re.match(r"^n\d+[A-Za-z]*Ring", c)]
    sweep("rings", rings, [c for c in rings if order_of(c) <= 7],
                          [c for c in rings if order_of(c) > 7])

    vsa = [c for c in base if re.match(r"^(SlogP_VSA|SMR_VSA|PEOE_VSA|EState_VSA|VSA_EState)\d", c)]
    sweep("vsa", vsa, [c for c in vsa if c.startswith(("SlogP_VSA", "PEOE_VSA"))],
                      [c for c in vsa if c.startswith(("SMR_VSA", "EState_VSA", "VSA_EState"))])

    # E-state: counts (N*) against summed state (S*) -- two views of the same typing
    es_n = [c for c in base if re.match(r"^N[a-z]", c)]
    es_s = [c for c in base if re.match(r"^S[a-z]", c)]
    sweep("estate", es_n + es_s, es_n, es_s)
    return A


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

    arms = build_arms(base590)
    sizes = {k[len("__sizes_"):]: v for k, v in arms.items() if k.startswith("__sizes_")}
    arms = {k: v for k, v in arms.items() if not k.startswith("__sizes_")}
    print(f"  baseline {len(base590)} columns (622 minus the 32 fr_* tail)")
    for name, (lo, hi) in sizes.items():
        print(f"    {name:9s} low/short {lo:3d}   high/long {hi:3d}   "
              f"-> _lo keeps {len(base590)-hi}, _hi keeps {len(base590)-lo}")
    print(f"  {len(arms)} arms x 33 datasets x 5 folds\n", flush=True)

    files = sorted(glob.glob("results/reanalysis/features/*.npz"))
    res = {}

    _ARMS.update(arms)

    with ProcessPoolExecutor(max_workers=5, initializer=_init, initargs=(arms,)) as ex:
        for ds, task, out in ex.map(job, files):
            res[ds] = {"task": task, **out}
            b = float(np.mean(out["base590"]))
            worst = sorted(((100*(np.mean(v)-b)/b, a) for a, v in out.items() if a != "base590"),
                           reverse=True)[:2]
            print(f"  {ds:16s} base={b:.4f}  worst: "
                  + ", ".join(f"{a} {d:+.2f}%" for d, a in worst), flush=True)
            json.dump({"sizes": sizes, "res": res}, open(out_path, "w"))
    print(f"  -> {out_path}")


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


if __name__ == "__main__":
    main()
