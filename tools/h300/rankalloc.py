"""Allocate the column budget by INFORMATION per family, not by column count.

    .venv/bin/python tools/h300/rankalloc.py OUT.json

Section 9 stratified proportionally: a family with twice the columns got twice the slots. That
spends the budget on the families that happen to be widest, and width is not information. `blocks`
has 121 columns and an effective rank of 38; `frag` has 43 columns and an effective rank of 29.
Proportional allocation gives blocks 87 slots at a 300 budget and frag 31 -- it over-serves the
most redundant family and starves the least redundant one.

Here the quota is proportional to each family's EFFECTIVE RANK (components for 95% of
within-family variance, in rank space, measured on 477,115 molecules). blocks 55, frag 42,
estate 38.

This is a direct test of whether 300 has to cost what section 9 measured, or whether that cost
was an artifact of how the budget was divided. `strat300` is carried as the reference arm so the
two allocations are compared on the same folds in the same run.
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
from strat import job, stratify                                    # noqa: E402
from sweeps import build_arms                                      # noqa: E402

SIZES = (350, 300, 250)


def alloc(pool, fam_of, rank, eff, n_keep):
    """Quota proportional to effective rank; within a family, best utility rank first."""
    fams = {}
    for c in pool:
        fams.setdefault(fam_of.get(c, "other"), []).append(c)
    tot = sum(eff[f] for f in fams)
    quota = {f: max(1, min(len(cs), round(eff[f] * n_keep / tot))) for f, cs in fams.items()}
    # settle to exactly n_keep, taking from / giving to whichever family is furthest from its
    # own effective rank -- so the correction follows the same principle as the allocation
    def gap(f):
        return quota[f] - eff[f]
    while sum(quota.values()) > n_keep:
        f = max((f for f in quota if quota[f] > 1), key=gap)
        quota[f] -= 1
    while sum(quota.values()) < n_keep:
        f = min((f for f in quota if quota[f] < len(fams[f])), key=gap)
        quota[f] += 1
    out = []
    for f, cs in fams.items():
        out += sorted(cs, key=lambda c: rank.get(c, 1e9))[:quota[f]]
    return sorted(set(out))


def main() -> None:
    out_path = sys.argv[1]
    import molhume
    mn = list(molhume.column_set("minimal"))
    util = json.load(open("/tmp/h300_utility.json"))
    ucols = util[list(util)[0]]["cols"]
    RK = {k: np.array(util[k]["rank"], float) for k in util}
    upos = {c: i for i, c in enumerate(ucols)}
    fr = [c for c in mn if c.startswith("fr_")]
    t50 = np.array([sum(RK[k][upos[c]] <= 50 for k in RK) for c in fr])
    top = {c for c, t in zip(fr, t50) if t >= 3}
    base590 = [c for c in mn if not c.startswith("fr_") or c in top]
    sw = build_arms(base590)
    dropped = set()
    for a, cols in sw.items():
        if a.endswith("_lo"):
            dropped |= set(base590) - set(cols)
    combo = [c for c in base590 if c not in dropped]

    allc = molhume.ALL_COLUMNS
    fam_of = {}
    for f, (a, b) in molhume.FAMILY_OFFSETS.items():
        for c in allc[a:b]:
            fam_of[c] = f
    R = np.load("/tmp/R.npy").astype(np.float64)
    kp = {c: i for i, c in enumerate(json.load(open("/tmp/keep.json")))}
    fams = {}
    for c in combo:
        fams.setdefault(fam_of.get(c, "other"), []).append(c)
    eff = {}
    for f, cs in fams.items():
        idx = [kp[c] for c in cs if c in kp]
        if len(idx) < 2:
            eff[f] = max(1, len(idx)); continue
        ev = np.clip(np.linalg.eigvalsh(np.cov(R[:, idx], rowvar=False))[::-1], 0, None)
        eff[f] = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95)) + 1

    files = sorted(glob.glob("results/reanalysis/features/*.npz"))
    jobs = []
    for f in files:
        ds = f.split("/")[-1][:-4]
        others = [k for k in RK if k != ds]
        mr = {c: float(np.mean([RK[k][upos[c]] for k in others])) for c in combo}
        arms = {"base590": base590, "combo418": combo, "strat300": stratify(combo, fam_of, mr, 300)}
        for n in SIZES:
            arms[f"rank{n}"] = alloc(combo, fam_of, mr, eff, n)
        jobs.append((f, arms))
    print("  arms: " + "  ".join(f"{k}={len(v)}" for k, v in jobs[0][1].items()), flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, out in ex.map(job, jobs):
            res[ds] = {"task": task, **out}
            b = float(np.mean(out["base590"]))
            print(f"  {ds:16s} base={b:.4f}  " + "  ".join(
                f"{a}={100*(np.mean(out[a])-b)/b:+.2f}%"
                for a in ("strat300", "rank350", "rank300", "rank250")), flush=True)
            json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
