"""End-to-end cost of SMILES -> ECFP + 865 descriptors, per step, with a standard deviation.

WHAT IS BEING COMPARED. Both arms must produce THE SAME 865 columns -- the deduplicated
RDKit u Mordred union from `data/dedupe.json` -- plus the same ECFP. Timing HUME against a
Mordred call that computes all 1,613 descriptors would flatter us by a factor we did not earn,
so the baseline builds a Calculator restricted to the surviving mordred columns and an RDKit
list restricted to the surviving rdkit ones.

    .venv/bin/python bench_e2e.py hume      [n_mols] [n_reps]
    <mordred env>    python bench_e2e.py baseline  [n_mols] [n_reps]
    .venv/bin/python bench_e2e.py report

TWO ARMS, TWO PROCESSES, AND THAT IS FORCED. mordred 1.2.0 requires numpy 1.x and Python 3.11
(it imports `distutils`); hume's extension is built against numpy 2.4.6 on 3.12. They cannot
coexist in one interpreter, so each arm writes its own JSON and `report` composes them. The
shared steps -- SMILES parse and ECFP -- are measured in BOTH environments precisely so the
composition can be checked rather than assumed: if the two processes disagree about how long
`Chem.MolFromSmiles` takes, they were not running under the same conditions and the comparison
is void. `report` says so instead of printing a ratio.

METHOD, and every clause here is load-bearing:

  * COLD MOLECULES, RE-PARSED EVERY REPETITION. RDKit caches Crippen contributions and Gasteiger
    charges ON THE MOLECULE. A second pass over the same objects measures a dict lookup. This
    project has already published two wrong numbers this way -- an 82x-wrong Crippen cost, and a
    187.5 us extraction figure that was really 231 cold and 130 warm. Fresh `Mol` objects per
    repetition, and the parse is timed separately rather than subtracted.
  * CPU TIME, NOT WALL. Wall-clock spreads of 26-47x were measured on this box while other jobs
    were running; that is larger than anything being compared here.
  * ORDER ROTATED ACROSS CYCLES so that machine drift is common-mode rather than charged to
    whichever step happens to run first.
  * SD IS OVER REPETITION MEANS, not over molecules. The per-molecule distribution is wildly
    skewed -- 2.9% of molecules carry 46% of BCUT2D time -- so a per-molecule SD would describe
    the corpus, not the measurement. What a reader needs to know is how much the STEP COST moves
    between runs, which is what this reports.
  * CONTENTION IS RECORDED, not assumed away. The load average and process count go in the JSON.
    A number taken on a busy box is not wrong, it is measured under stated conditions -- but it
    must say so, and `report` refuses to present a headline ratio from a contended run.
"""
from __future__ import annotations

import json
import os

# THREAD CAPS MUST BE SET BEFORE numpy IS IMPORTED, AND THIS FILE HAD THEM AT THE BOTTOM.
#
# BLAS reads these at load time. `os.environ.setdefault("OMP_NUM_THREADS", "1")` inside
# `if __name__ == "__main__"` runs long after `import numpy` on line ~50, so it did nothing at
# all -- and mordred's matrix work (TopologicalCharge's A.D2 is a dgemm) then ran on every core.
# Observed: 789% CPU on the baseline arm.
#
# THAT WOULD HAVE FLATTERED US, NOT PENALISED US, WHICH IS WHY IT MATTERS. This harness measures
# `time.process_time()`, which sums CPU across ALL THREADS of the process. A baseline spread over
# ~8 threads therefore reports ~8x the CPU time it would single-threaded, against a HUME arm that
# is single-threaded throughout. The end-to-end ratio would have come out several times too good,
# and every step of it would have looked internally consistent.
#
# Both arms are now pinned to one thread so the comparison is per-core work against per-core work.
# If a threaded baseline is ever wanted, that is a DIFFERENT and legitimate measurement -- wall
# clock on a quiet box, stated as such -- not this one with the cap removed.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "e2e"
CORPUS = ROOT / "cpp" / "hard.smi"


def _cpu() -> float:
    return time.process_time()


def _machine() -> dict:
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = float("nan")
    return {"platform": f"{platform.system().lower()} {platform.machine()}",
            "cpu_count": os.cpu_count(), "load1": round(load1, 2),
            "contended": load1 > 1.5,
            "python": platform.python_version()}


def _versions() -> dict:
    """Resolved versions FROM THE PROCESS THAT PRODUCES THE NUMBERS. A verify or bench log
    without this is not evidence in this repo -- see constraints.txt for why."""
    import rdkit
    v = {"rdkit": rdkit.__version__, "numpy": np.__version__}
    # THE NUMERIC CANARY, because a version banner is not evidence: a process can print
    # `rdkit 2025.09.2` and execute 2026.3.5's arithmetic out of unlinked-but-still-mapped dylibs,
    # which has happened in this repo. cpp/verify_hume.py's molecule and expected value, computed
    # here, by the process that produces the timings.
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        v["canary_BCUT2D_MRLOW"] = Descriptors.BCUT2D_MRLOW(Chem.MolFromSmiles(
            "O=C1CCNCCNNNCCNCCC(=O)c2ccc(o2)COCOCc2ccc1o2"))
        v["canary_ok"] = v["canary_BCUT2D_MRLOW"] == -0.07665884800196521
    except Exception as e:                                     # noqa: BLE001
        v["canary_BCUT2D_MRLOW"] = None
        v["canary_ok"] = False
        v["canary_error"] = repr(e)
    try:
        import mordred
        v["mordred"] = mordred.__version__
    except Exception:
        v["mordred"] = None
    return v


def load_smiles(n: int, seed: int = 0) -> list[str]:
    smis = CORPUS.read_text().split()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(smis), min(len(smis), n), replace=False)
    return [smis[i] for i in pick]


# --------------------------------------------------------------------------------------------
# the timing core
# --------------------------------------------------------------------------------------------

def time_steps(steps: list[tuple[str, callable]], smis: list[str], n_reps: int) -> dict:
    """-> {name: {us_mean, us_sd, us_min, n_reps}}, CPU us per molecule.

    `steps` is a list of (name, fn) where fn takes the freshly parsed molecule list. Every step
    is measured in its own pass over the SAME freshly parsed molecules, and the pass order is
    rotated by cycle.
    """
    acc: dict[str, list[float]] = {name: [] for name, _ in steps}
    from rdkit import Chem
    for cyc in range(n_reps):
        order = steps[cyc % len(steps):] + steps[:cyc % len(steps)]
        for name, fn in order:
            mols = [Chem.MolFromSmiles(s) for s in smis]      # COLD. see module docstring.
            mols = [m for m in mols if m is not None]
            t0 = _cpu()
            fn(mols)
            t1 = _cpu()
            acc[name].append((t1 - t0) / len(mols) * 1e6)
            del mols
    out = {}
    for name, xs in acc.items():
        out[name] = {"us_mean": round(statistics.mean(xs), 3),
                     "us_sd": round(statistics.stdev(xs), 3) if len(xs) > 1 else None,
                     "us_min": round(min(xs), 3), "n_reps": len(xs)}
    return out


def time_parse(smis: list[str], n_reps: int) -> dict:
    """Parse is the one step that cannot take parsed molecules as input, so it is timed here."""
    from rdkit import Chem
    xs = []
    for _ in range(n_reps):
        t0 = _cpu()
        mols = [Chem.MolFromSmiles(s) for s in smis]
        t1 = _cpu()
        xs.append((t1 - t0) / len(mols) * 1e6)
        del mols
    return {"us_mean": round(statistics.mean(xs), 3),
            "us_sd": round(statistics.stdev(xs), 3) if len(xs) > 1 else None,
            "us_min": round(min(xs), 3), "n_reps": len(xs)}


def _ecfp_step(mols):
    from rdkit.Chem import rdFingerprintGenerator as rfg
    gen = rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)
    for m in mols:
        gen.GetFingerprintAsNumPy(m)


# --------------------------------------------------------------------------------------------
# arm: HUME
# --------------------------------------------------------------------------------------------

def _survivors_covered(cols) -> int:
    """How many of the 865 deduplicated columns these names actually are.

    THE COLUMN COUNT AND THE RATIO MUST NOT BE READ APART. `hume.ALL_COLUMNS` has 1,244 entries
    but they are not 1,244 of the 865: 842 of them are, and the rest are HUME-specific (`SATS*`,
    `RW*`, `conj_*`), EState types the r>0.99 dedupe threw away, or Autocorrelation cells the
    dedupe dropped (the block emits 540 and the census keeps 419 -- every `MATS0*`/`GATS0*`
    among them). Counting the emitted columns instead of the covered ones would overstate the
    position by roughly a factor of 1.5, which is the error PORT_STATUS.md's "warning about the
    number 182" is about.

    NAMED IS NOT VALUE-PRODUCING, and the caller decides which it wants. Pass `ALL_COLUMNS` for
    the named count (842) and `set(ALL_COLUMNS) - set(PENDING_COLUMNS)` for the columns that
    actually produce a number (840; `qed` and `SPS` are NaN on every molecule today). This
    function cannot tell them apart -- it only intersects names -- so the distinction has to be
    made in what is handed to it.

    THE MATCH IS CASE-INSENSITIVE, and that is a fix for exactly one family rather than a general
    loosening. `_columns.py` names RDKit's connectivity indices `chi0n`..`chi4v` (lower case, as
    HUME's own chi.py did) while the 865 carry RDKit's `Chi0n`..`Chi4v`. Those nine are the same
    nine numbers. Checked: case-insensitivity adds those nine and NOTHING else, so it cannot be
    quietly inflating the count with a near-miss.
    """
    sys.path.insert(0, str(ROOT))
    import blocks
    fam = json.loads((ROOT / "fam.json").read_text())
    sp = blocks.split(fam)
    surv = {n for _s, n, _f in sp["core"] + sp["predict"]}
    have = {c.lower() for c in cols}
    return sum(1 for n in surv if n.lower() in have)


def arm_hume(n_mols: int, n_reps: int) -> dict:
    """SMILES -> ECFP + whatever HUME computes natively TODAY, through the PICKLE boundary.

    THE BOUNDARY IS `extract_pickles`, not `extract`. One `m.ToBinary()` per molecule, parsed in
    C++ by src/hume_core/molpickle.h; there is no per-atom Python call left in the path. The old
    boundary is still measured, as `extract_api_boundary`, because a step table that only shows
    the arm that won is not a measurement. It is reported OUTSIDE `steps` so it cannot be summed
    into the total by accident.

    The column count is reported alongside the time and is NOT assumed to be 865 -- the port is
    in progress, and a step table that silently covers fewer columns than the baseline would be
    the most flattering possible error. `report` compares column counts and says so.
    """
    import molhume as hume
    from hume._extract import extract, extract_pickles
    from hume import _core

    smis = load_smiles(n_mols)
    cols = hume.ALL_COLUMNS
    steps = [
        ("extract_pickles_boundary", lambda mols: extract_pickles(mols)),
        ("ecfp_r3_2048", _ecfp_step),
    ]
    res = time_steps(steps, smis, n_reps)
    res["smiles_parse"] = time_parse(smis, n_reps)

    # The C++ compute, timed separately: it takes the serialised molecules, not molecule objects,
    # so it does not fit time_steps' signature. The pickles are built once outside the timed
    # region -- correct here and NOT a warm-cache error, because a `bytes` caches nothing. The
    # RDKit-side caching hazard lives in extract_pickles(), which is timed cold above.
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in smis]
    mols = [m for m in mols if m is not None]

    def _timed(fn, prep):
        xs = []
        for _ in range(n_reps):
            arg = prep()
            t0 = _cpu()
            fn(arg)
            t1 = _cpu()
            xs.append((t1 - t0) / len(mols) * 1e6)
        return {"us_mean": round(statistics.mean(xs), 3),
                "us_sd": round(statistics.stdev(xs), 3) if len(xs) > 1 else None,
                "us_min": round(min(xs), 3), "n_reps": len(xs)}

    pk = extract_pickles(mols)
    rings = (pk.rings.ring_moff, pk.rings.ring_ptr, pk.rings.ring_at)

    def _all(sel=None):
        # `families` is passed BY KEYWORD: the potential-stereo pair sits between `h_blobs` and
        # it in the signature, and a positional `sel` would silently land in `stereo_a`.
        #
        # `optional` IS PASSED EXPLICITLY AND NAMES BOTH COLUMNS, which is the opposite of what a
        # user should do and exactly right here. The library default declines `qed` (69.3 us/mol
        # for one column); this file's whole claim is a LIKE-FOR-LIKE ratio against an arm that
        # computes all 864 names, so it must pay for all 864. Taking the default here would cut
        # 69 us off HUME's total and quietly compare 863 columns against 864 -- a better ratio
        # obtained by computing less, which is the one way this benchmark must not be wrong.
        # `qed` IS NOT ASKED FOR, and dropping it makes this number MORE honest rather than
        # less. It costs 69.3 us/mol and the deduplication dropped its output slot, so it is not
        # one of the 1,269 this benchmark divides by -- paying for it inflated the numerator
        # against a denominator that never contained it. See the 0.7.0 changelog.
        return _core.all_from_pickles(pk.blobs, *rings, pk.h_blobs, pk.stereo_a, pk.stereo_b,
                                      families=sel, optional=("AvgIpc",))

    res["cpp_all_columns"] = _timed(lambda _b: _all(), lambda: None)
    res["cpp_all_columns"]["columns"] = len(cols)

    # PER FAMILY, PAIRED AND DIFFERENCED PER REPETITION. `all_from_pickles(blobs, [f])` runs the
    # 182 blocks plus family f -- the blocks are not optional, they are what fills the EState
    # index -- so a family's own cost is that minus the blocks-only arm. The subtraction is done
    # WITHIN each repetition and the SD taken over the differences, not over the two arms
    # separately: on a contended box the two arms move together, and differencing the means
    # would throw away exactly the pairing that makes the number mean anything.
    #
    # TWO FAMILIES ARE NOT INDEPENDENT AND SUBTRACTING THE BLOCKS-ONLY ARM WOULD CHARGE THEM FOR
    # WORK THAT IS NOT THEIRS. `constit` consumes vsa_bins' MolLogP/MolMR/TPSA, ringcount's
    # aromatic/aliphatic counts and frag_matcher's HBD/HBA/rotatable counts, so
    # `all_from_pickles(..., ["constit"])` forces those three on (bindings.cpp's family_mask does
    # it deliberately -- the alternative is computing constit over a row of zeros). Its own arm
    # is therefore differenced against a DEPENDENCY arm rather than against blocks_only, which is
    # the same paired-within-repetition subtraction one level up.
    fam_deps = {"constit": ["vsa", "ringcount", "frag"], "alias": ["vsa"]}
    fam_names = [k for k in hume.FAMILY_OFFSETS if k not in ("blocks", "end")]
    dep_arms = {f"deps:{n}": d for n, d in fam_deps.items() if n in fam_names}
    arms = ["blocks_only"] + fam_names + list(dep_arms)
    per_rep: dict[str, list[float]] = {k: [] for k in arms}
    for cyc in range(n_reps):
        for name in arms[cyc % len(arms):] + arms[:cyc % len(arms)]:
            if name == "blocks_only":
                sel = []
            elif name in dep_arms:
                sel = dep_arms[name]
            else:
                sel = [name]
            t0 = _cpu()
            _all(sel)
            t1 = _cpu()
            per_rep[name].append((t1 - t0) / len(mols) * 1e6)
    base = per_rep["blocks_only"]
    fams = {"blocks_182": {"us_mean": round(statistics.mean(base), 3),
                           "us_sd": round(statistics.stdev(base), 3) if len(base) > 1 else None,
                           "columns": hume.N_COLS}}
    for name in fam_names:
        ref = per_rep[f"deps:{name}"] if name in fam_deps else base
        d = [a - b for a, b in zip(per_rep[name], ref)]
        lo = hume.FAMILY_OFFSETS[name]
        hi = min([v for v in hume.FAMILY_OFFSETS.values() if v > lo], default=hume.N_ALL_COLS)
        fams[name] = {"us_mean": round(statistics.mean(d), 3),
                      "us_sd": round(statistics.stdev(d), 3) if len(d) > 1 else None,
                      "columns": hi - lo}

    # Diagnostics, deliberately outside `steps`: what each family costs, what the original 182
    # cost alone, and what the boundary they replaced still costs. None of it belongs in the
    # total -- the first two are subsets of cpp_all_columns and the last is the road not taken.
    detail = {"per_family": fams,
              "cpp_blocks_182_only": _timed(_core.blocks_from_pickles, lambda: pk.blobs),
              "extract_api_boundary": _timed(extract, lambda: [Chem.MolFromSmiles(s)
                                                               for s in smis])}
    detail["cpp_blocks_182_only"]["columns"] = hume.N_COLS
    detail["offsets"] = {k: v for k, v in hume.FAMILY_OFFSETS.items()}

    covered = _survivors_covered(cols)
    return {"arm": "hume", "n_mols": len(smis), "steps": res, "detail": detail,
            "columns_descriptors": covered, "columns_emitted": len(cols),
            "columns_ecfp": 2048, "machine": _machine(), "versions": _versions()}


# --------------------------------------------------------------------------------------------
# arm: the unoptimised baseline
# --------------------------------------------------------------------------------------------

def _survivor_columns():
    """The 865, split by source. blocks.split() asserts the split is exhaustive and disjoint."""
    sys.path.insert(0, str(ROOT))
    import blocks
    from mordred import Calculator, descriptors as mdesc
    full = Calculator(mdesc, ignore_3D=True)
    fam = {str(x): type(x).__module__.split(".")[-1] for x in full.descriptors}
    sp = blocks.split(fam)
    rows = sp["core"] + sp["predict"]
    mord = {n for s, n, _ in rows if s == "mordred"}
    rdk = {n for s, n, _ in rows if s == "rdkit"}
    sub = [x for x in full.descriptors if str(x) in mord]
    return Calculator(sub, ignore_3D=True), sorted(mord), sorted(rdk)


def arm_baseline(n_mols: int, n_reps: int) -> dict:
    """SMILES -> ECFP + the same 865 columns, computed the ordinary way.

    RESTRICTED TO THE SURVIVORS ON PURPOSE. Mordred's full 1,613 would make the comparison look
    better than it is; the honest baseline is what someone would run to get these 865 columns.
    """
    from rdkit.Chem import Descriptors

    calc, mord_names, rdk_names = _survivor_columns()
    smis = load_smiles(n_mols)
    rdk_fns = [(n, f) for n, f in Descriptors._descList if n in set(rdk_names)]
    missing = set(rdk_names) - {n for n, _ in rdk_fns}
    if missing:
        raise RuntimeError(
            f"{len(missing)} of the {len(rdk_names)} surviving RDKit columns are not in "
            f"Descriptors._descList under rdkit {_versions()['rdkit']}: {sorted(missing)[:8]}. "
            f"The baseline would silently compute fewer columns than HUME and the comparison "
            f"would be meaningless. Fix the column list or the pin before re-running.")

    def step_rdkit(mols):
        for m in mols:
            for _n, f in rdk_fns:
                try:
                    f(m)
                except Exception:
                    pass

    def step_mordred(mols):
        calc.pandas(mols, nproc=1, quiet=True)

    steps = [("rdkit_descriptors", step_rdkit),
             ("mordred_descriptors", step_mordred),
             ("ecfp_r3_2048", _ecfp_step)]
    res = time_steps(steps, smis, n_reps)
    res["smiles_parse"] = time_parse(smis, n_reps)
    res["rdkit_descriptors"]["columns"] = len(rdk_fns)
    res["mordred_descriptors"]["columns"] = len(mord_names)
    return {"arm": "baseline", "n_mols": len(smis), "steps": res,
            "columns_descriptors": len(rdk_fns) + len(mord_names), "columns_ecfp": 2048,
            "machine": _machine(), "versions": _versions()}


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------

SHARED = ("smiles_parse", "ecfp_r3_2048")


def report() -> None:
    a = json.loads((OUT / "hume.json").read_text())
    b = json.loads((OUT / "baseline.json").read_text())

    print(f"corpus {CORPUS.name}, {a['n_mols']:,} / {b['n_mols']:,} molecules")
    for r in (a, b):
        m, v = r["machine"], r["versions"]
        print(f"  {r['arm']:9s} rdkit {v['rdkit']} numpy {v['numpy']} mordred {v['mordred']} | "
              f"load1 {m['load1']}{'  CONTENDED' if m['contended'] else ''}")
    print()

    for r, label in ((a, "HUME"), (b, "RDKit + Mordred")):
        print(f"{label}")
        print(f"  {'step':28s} {'us/mol':>9s} {'SD':>8s} {'cols':>6s}")
        tot = 0.0
        for name, s in sorted(r["steps"].items(), key=lambda kv: -kv[1]["us_mean"]):
            sd = f"{s['us_sd']:.2f}" if s.get("us_sd") is not None else "-"
            print(f"  {name:28s} {s['us_mean']:9.1f} {sd:>8s} {s.get('columns', ''):>6}")
            tot += s["us_mean"]
        print(f"  {'TOTAL':28s} {tot:9.1f}")
        if r.get("columns_emitted"):
            print(f"  {len(r['steps'])} steps; {r['columns_emitted']} columns emitted, "
                  f"{r['columns_descriptors']} of them members of the 865")
        for name, s in sorted(r.get("detail", {}).items()):
            if not isinstance(s, dict) or "us_mean" not in s:
                continue
            sd = f"{s['us_sd']:.2f}" if s.get("us_sd") is not None else "-"
            print(f"  ({name:26s} {s['us_mean']:9.1f} {sd:>8s}   not in the total)")
        fams = r.get("detail", {}).get("per_family")
        if fams:
            print(f"\n  {'  where the C++ time goes':28s} {'us/mol':>9s} {'SD':>8s} {'cols':>6s}")
            for name, s in sorted(fams.items(), key=lambda kv: -kv[1]["us_mean"]):
                sd = f"{s['us_sd']:.2f}" if s.get("us_sd") is not None else "-"
                print(f"    {name:26s} {s['us_mean']:9.1f} {sd:>8s} {s.get('columns', ''):>6}")
        print()

    # Cross-check the shared steps before composing anything from two processes.
    bad = [n for n in SHARED
           if n in a["steps"] and n in b["steps"]
           and abs(a["steps"][n]["us_mean"] - b["steps"][n]["us_mean"])
           > 0.25 * max(a["steps"][n]["us_mean"], b["steps"][n]["us_mean"])]
    if bad:
        print(f"REFUSING TO PRINT A RATIO. The two arms disagree by >25% on {bad}, which they "
              f"share and which should cost the same in both. They were not measured under the "
              f"same conditions; re-run them closer together on a quiet machine.")
        return
    # THE SHARED STEPS ARE THE EVIDENCE; LOAD AVERAGE IS ONLY A PROXY FOR IT.
    #
    # This used to refuse outright whenever load1 > 1.5, a threshold set when five agents were
    # saturating the box. But the question a reader actually needs answered is "were these two
    # arms measured under equivalent conditions", and `smiles_parse` and `ecfp_r3_2048` answer it
    # DIRECTLY: identical work, run in two different interpreters against two different numpy
    # versions, minutes apart. If those agree closely the arms are comparable, whatever the load
    # average was; if they do not, no amount of quiet proves anything.
    #
    # So: agreement within 10% on every shared step earns the ratio, with the load stated beside
    # it. Between 10% and 25% it is printed as indicative only. Past 25% the check above has
    # already refused. This is a change in what is tested, not a relaxation of it -- a run on an
    # idle machine whose shared steps disagreed would still be rejected, and previously would
    # have been accepted.
    worst = max((abs(a["steps"][n]["us_mean"] - b["steps"][n]["us_mean"])
                 / max(a["steps"][n]["us_mean"], b["steps"][n]["us_mean"])
                 for n in SHARED if n in a["steps"] and n in b["steps"]), default=1.0)
    ta = sum(s["us_mean"] for s in a["steps"].values())
    tb = sum(s["us_mean"] for s in b["steps"].values())
    la, lb = a["machine"]["load1"], b["machine"]["load1"]
    print(f"shared-step agreement: worst {100 * worst:.1f}%  "
          f"(load1 {la} hume / {lb} baseline)")
    if worst > 0.10:
        print("  INDICATIVE ONLY -- shared steps agree to worse than 10%.")
    print(f"end-to-end: HUME {ta:.1f} us/mol vs baseline {tb:.1f} us/mol -> {tb / ta:.1f}x")
    # 865 SURVIVOR ROWS, 864 UNIQUE NAMES. `data/dedupe.json` carries one name that is defined by
    # BOTH rdkit and mordred and survives under both sources, so the baseline computes 865 entries
    # while there are only 864 distinct columns to cover. Comparing the two counts naively made a
    # COMPLETE port print "NOT like-for-like" forever, which is a worse error than the one the
    # caveat exists to prevent -- it would have understated a finished result indefinitely.
    N_UNIQUE = 864
    if a["columns_descriptors"] < N_UNIQUE:
        print(f"  CAVEAT: HUME covers {a['columns_descriptors']} of the {N_UNIQUE} unique "
              f"descriptor names; the baseline computes all of them. The ratio is NOT "
              f"like-for-like until the port is complete.")
    else:
        print(f"  LIKE FOR LIKE: both arms cover all {N_UNIQUE} unique descriptor names "
              f"(the baseline's {b['columns_descriptors']} counts one name twice, defined by "
              f"both rdkit and mordred).")


def sweep(arm: str, sizes, n_reps: int) -> None:
    """Is N large enough for the MEAN to have stopped moving?

    THIS EXISTS BECAUSE THE PER-MOLECULE COST DISTRIBUTION IS EXTREMELY HEAVY-TAILED. Measured on
    this corpus: median ~5 ms, max ~35 s, so max/median is ~7000x and a handful of molecules
    dominate any sample's mean. The repetition SD this file already reports CANNOT detect that --
    it measures noise on a FIXED sample, so it is small and reassuring precisely when the sample
    is unrepresentative. Two different quantities:

        us_sd        how much the number moves when you re-time the SAME molecules
        this sweep   how much it moves when you draw DIFFERENT molecules

    A ratio quoted from a single N is only as good as the second, and nothing in this harness
    measured it until now. Run to convergence, or report the median alongside and say which is
    which -- do not quote a mean that is still drifting.
    """
    fn = {"hume": arm_hume, "baseline": arm_baseline}[arm]
    print(f"{arm}: sample-size sweep, {n_reps} reps each\n")
    print(f"  {'N':>7s} {'total us/mol':>13s} {'vs previous':>12s}")
    prev = None
    for n in sizes:
        res = fn(n, n_reps)
        tot = sum(st["us_mean"] for st in res["steps"].values())
        delta = "" if prev is None else f"{100 * (tot - prev) / prev:+11.1f}%"
        print(f"  {n:7,d} {tot:13.1f} {delta:>12s}", flush=True)
        prev = tot
    print("\n  The mean has converged when successive rows agree to within the repetition SD.")
    print("  If the last step is still several percent, N is too small and the ratio is not")
    print("  quotable -- draw more molecules rather than more repetitions.")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        return report()
    if cmd == "sweep":
        arm = sys.argv[2] if len(sys.argv) > 2 else "hume"
        reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        return sweep(arm, [1000, 2000, 5000, 10000, 25000], reps)
    n_mols = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    n_reps = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"hume": arm_hume, "baseline": arm_baseline}[cmd](n_mols, n_reps)
    (OUT / f"{cmd}.json").write_text(json.dumps(res, indent=1))
    m = res["machine"]
    print(f"{cmd}: {res['n_mols']:,} molecules, {res['columns_descriptors']} descriptor columns, "
          f"load1 {m['load1']}{'  CONTENDED' if m['contended'] else ''}")
    for name, s in sorted(res["steps"].items(), key=lambda kv: -kv[1]["us_mean"]):
        sd = f"+/-{s['us_sd']:.2f}" if s.get("us_sd") is not None else ""
        print(f"  {name:28s} {s['us_mean']:9.1f} {sd}")


if __name__ == "__main__":
    # Thread caps live at the TOP of this file, before `import numpy`. A setdefault here would be
    # too late to have any effect -- which is exactly the bug that used to sit on this line.
    main()
