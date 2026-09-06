"""Apply the five pre-registered tests. The rule decides N, not the author.

    .venv/bin/python tools/h300/decide.py LADDER.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

LADDER = (128, 200, 256, 300, 350, 418)


def boot_ci(x, n=10000, seed=0):
    """Bootstrap 95% CI of the median. Test 1 needs the UPPER bound, not the point estimate."""
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    d = json.load(open(sys.argv[1]))
    res, eff_sum, pool_n = d["res"], d["eff_sum"], d["pool"]
    # The null arm is the full 622 scored again with a different XGBoost seed. It changes no
    # column, so its spread across the panels IS the panel's noise, and every threshold below is
    # read off it rather than chosen. Merged from a separate run so the twelve arms already
    # measured did not have to be recomputed.
    null = json.load(open("/tmp/r2_null.json"))
    for k in res:
        res[k]["null622"] = null[k]
    ds = sorted(res)
    arms = [a for a in res[ds[0]] if a != "task"]
    M = {a: np.array([np.mean(res[k][a]) for k in ds]) for a in arms}
    base = M["full622"]
    rel = {a: 100 * (M[a] - base) / base for a in arms if a != "full622"}
    cls = [i for i, k in enumerate(ds) if res[k]["task"] == "classif"]
    reg = [i for i, k in enumerate(ds) if res[k]["task"] == "regress"]
    print(f"  {len(ds)} selection panels ({len(cls)} classification, {len(reg)} regression)")
    print(f"  pool {pool_n}; label-free cross-check on N (sum of family effective ranks) = {eff_sum}")
    nv = rel["null622"]
    NULL_CI = boot_ci(nv)[1]
    NULL_TAIL = float(np.mean(nv > 3.0))
    NULL_CLS = boot_ci(nv[cls])[1]
    NULL_REG = boot_ci(nv[reg])[1]
    print(f"\n  NULL ARM (same 622 columns, different seed) -- the calibration:")
    print(f"    median {np.median(nv):+.2f}%   CI upper {NULL_CI:+.2f}%   "
          f"tail>3% {100*NULL_TAIL:.0f}%   clf UB {NULL_CLS:+.2f}%   reg UB {NULL_REG:+.2f}%\n")

    print(f"  {'N':>5s} {'median':>8s} {'95% CI of median':>20s} {'tail>3%':>8s} "
          f"{'vs random':>10s} {'sign':>7s} {'clf UB':>8s} {'reg UB':>8s}  verdict")
    qualifying = []
    for n in LADDER:
        a = f"rank{n}"
        if a not in rel:
            continue
        v = rel[a]
        lo, hi = boot_ci(v)
        t1 = hi <= NULL_CI
        t2 = float(np.mean(v > 3.0)) <= NULL_TAIL
        pooled = np.mean([rel[f"random{n}_s{s}"] for s in (0, 1, 2)], axis=0)
        p3 = stats.wilcoxon(v - pooled).pvalue
        t3 = p3 < 0.05 and np.median(v - pooled) < 0
        nb = int((v > 0).sum())
        p4 = stats.binomtest(nb, len(v), 0.5).pvalue
        t4 = p4 > 0.05
        chi = boot_ci(v[cls])[1]; rhi = boot_ci(v[reg])[1]
        t5 = chi <= NULL_CLS and rhi <= NULL_REG
        ok = all((t1, t2, t3, t4, t5))
        if ok:
            qualifying.append(n)
        flags = "".join(x for x, t in zip("12345", (t1, t2, t3, t4, t5)) if not t)
        print(f"  {n:5d} {np.median(v):+7.2f}% [{lo:+6.2f},{hi:+6.2f}] {100*np.mean(v>3):7.1f}% "
              f"{p3:10.4f} {p4:7.3f} {chi:+7.2f}% {rhi:+7.2f}%  "
              + ("QUALIFIES" if ok else f"fails {flags}"))
    print()
    if qualifying:
        n = min(qualifying)
        print(f"  ==> THE RULE SELECTS N = {n}")
        print(f"      label-free cross-check said {eff_sum}; "
              f"{'agrees within 20%' if abs(n-eff_sum)/max(n,eff_sum) < 0.2 else 'DISAGREES -- ladder wins, per the pre-registration'}")
    else:
        print("  ==> NO N QUALIFIES. Per the pre-registration: report the smallest failing only")
        print("      test 1, say so plainly, and recommend the pool size instead.")
        for n in LADDER:
            a = f"rank{n}"
            if a not in rel: continue
            v = rel[a]; lo, hi = boot_ci(v)
            pooled = np.mean([rel[f"random{n}_s{s}"] for s in (0,1,2)], axis=0)
            only1 = (hi > NULL_CI and float(np.mean(v > 3.0)) <= NULL_TAIL
                     and stats.wilcoxon(v-pooled).pvalue < 0.05
                     and stats.binomtest(int((v>0).sum()), len(v), 0.5).pvalue > 0.05
                     and boot_ci(v[cls])[1] <= NULL_CLS and boot_ci(v[reg])[1] <= NULL_REG)
            if only1:
                print(f"      smallest failing only test 1: N = {n} (median {np.median(v):+.2f}%, "
                      f"CI upper {hi:+.2f}%)")
                break
    p = f"pool{pool_n}"
    if p in rel:
        v = rel[p]; lo, hi = boot_ci(v)
        print(f"\n  the pool itself ({pool_n} columns): median {np.median(v):+.2f}% "
              f"CI [{lo:+.2f},{hi:+.2f}]  {'passes test 1' if hi <= NULL_CI else 'fails test 1'}")


if __name__ == "__main__":
    main()
