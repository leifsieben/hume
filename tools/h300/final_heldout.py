"""ONE evaluation of the chosen spec on 43 panels that influenced no decision.

    .venv/bin/python tools/h300/final_heldout.py OUT.json

Everything in docs/selection/HUME_300_METHOD.md up to here was measured on the panels the selection was made
on. This is the only number in the study with no optimism in it, and it is run once.

Figure C's 33 datasets are SPENT for this spec. Every selection round used all of them, so a
HUME_300 arm on that plate would carry an advantage the other representations do not, and the
plate would stop being a fair comparison. Report HUME_300 from here instead, or label it on the
plate as selection-set performance.

`herg` and the four `wong_*` sets are not used here even though they are available: they were in
the 33 that made every decision, and calling them held-out now would be the same error renamed.
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
from rankalloc import alloc                                        # noqa: E402
from strat import job                                              # noqa: E402
from sweeps import build_arms                                      # noqa: E402

LOCKED_SPENT = {"herg", "wong_hepg2", "wong_hskmc", "wong_imr90", "wong_saureus"}


def main() -> None:
    out_path = sys.argv[1]
    import molhume
    mn = list(molhume.column_set("minimal"))
    util = json.load(open("/tmp/h300_utility.json"))
    ucols = util[list(util)[0]]["cols"]
    RK = {k: np.array(util[k]["rank"], float) for k in util if k not in LOCKED_SPENT}
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
    fam_of = {c: f for f, (a, b) in molhume.FAMILY_OFFSETS.items() for c in allc[a:b]}
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
    mr = {c: float(np.mean([RK[k][upos[c]] for k in RK])) for c in combo}

    arms = {"base590": base590, "combo418": combo}
    for n in (350, 300, 250):
        arms[f"rank{n}"] = alloc(combo, fam_of, mr, eff, n)
    # random controls DRAWN FROM THE 590, not from the combo -- drawing n from a pool of exactly
    # n returns the pool, which is how the random411 arm in the previous run silently became a
    # copy of combo418 and reported a meaningless p-value.
    for n in (350, 300):
        for s in (0, 1, 2):
            rng = np.random.default_rng(7000 + 10 * n + s)
            arms[f"random{n}_s{s}"] = [base590[i] for i in
                                       sorted(rng.choice(len(base590), n, replace=False))]
    files = sorted(glob.glob("results/reanalysis/heldout/*.npz"))
    print(f"  {len(files)} HELD-OUT panels, none used in any selection round")
    print("  arms: " + "  ".join(f"{k}={len(v)}" for k, v in arms.items()), flush=True)

    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, out in ex.map(job, [(f, arms) for f in files]):
            res[ds] = {"task": task, **out}
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(files)}", flush=True)
            json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
