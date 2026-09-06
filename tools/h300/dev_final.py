"""The decision run: every candidate on the expanded DEV-ONLY panel.

    .venv/bin/python tools/h300/dev_final.py OUT.json

TWO CHANGES FROM EVERY EARLIER RUN, AND BOTH MATTER.

1. LOCKED DATASETS ARE EXCLUDED. chempfn's EVAL_DATASETS.md defines LOCKED as one evaluation
   each, after freeze -- they exist so a final number cannot be tuned toward. herg and the four
   wong_* sets are LOCKED and were in the 33-dataset panel every earlier ablation was decided on,
   which means five of the thirteen classification panels behind "300 costs 4.4% on
   classification" should never have been in a selection loop. They are out here.

2. THE CLASSIFICATION PANEL IS ~46 DATASETS, NOT 8. tox21 and sider are single CSVs with 12 and
   27 label columns; split per endpoint each is an ordinary binary panel. The earlier conclusion
   about classification rested on 8 usable DEV panels, which is not enough to separate a 1% effect
   from noise.
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
from ablate import load                                            # noqa: E402
from rankalloc import alloc                                        # noqa: E402
from strat import job, stratify                                    # noqa: E402
from sweeps import build_arms                                      # noqa: E402

#: LOCKED in chempfn's lake spec. Not ours to spend on spec selection.
LOCKED = {"herg", "wong_hepg2", "wong_hskmc", "wong_imr90", "wong_saureus"}


def main() -> None:
    out_path = sys.argv[1]
    import molhume
    mn = list(molhume.column_set("minimal"))
    util = json.load(open("/tmp/h300_utility.json"))
    ucols = util[list(util)[0]]["cols"]
    RK = {k: np.array(util[k]["rank"], float) for k in util if k not in LOCKED}
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

    files = [f for f in sorted(glob.glob("results/reanalysis/features/*.npz"))
             if f.split("/")[-1][:-4] not in LOCKED]
    mr_all = {c: float(np.mean([RK[k][upos[c]] for k in RK])) for c in combo}
    jobs = []
    for f in files:
        arms = {"base590": base590, "combo418": combo}
        for n in (350, 300, 250):
            arms[f"rank{n}"] = alloc(combo, fam_of, mr_all, eff, n)
        arms["strat300"] = stratify(combo, fam_of, mr_all, 300)
        # THREE SEEDS PER SIZE, AND A random418 ARM. The first version of this run tested
        # selection at 300 and 350 with ONE draw each, and drew a null -- while the 33-panel
        # study had found selection decisive at 418 with three seeds pooled. Those are not
        # contradictory results, they are different comparisons: the size that was shown to
        # matter was never re-tested here, and one draw is not a control.
        for n in (len(combo), 350, 300):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(1000 * n + seed)
                arms[f"random{n}_s{seed}"] = [combo[i] for i in
                                              sorted(rng.choice(len(combo), n, replace=False))]
        jobs.append((f, arms))
    print(f"  {len(files)} DEV panels ({len(LOCKED)} LOCKED excluded)")
    print("  arms: " + "  ".join(f"{k}={len(v)}" for k, v in jobs[0][1].items()), flush=True)

    res = {}
    done = 0
    with ProcessPoolExecutor(max_workers=5) as ex:
        for ds, task, out in ex.map(job, jobs):
            res[ds] = {"task": task, **out}
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(files)}", flush=True)
            json.dump(res, open(out_path, "w"))
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
