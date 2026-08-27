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

    THE COLUMN COUNT AND THE RATIO MUST NOT BE READ APART. `hume.ALL_COLUMNS` has 529 entries but
    they are not 529 of the 865: about 160 of them are HUME-specific (`SATS*`, `RW*`, `conj_*`)
    or are EState types the r>0.99 dedupe threw away, and one family the baseline computes --
    Autocorrelation, 419 columns -- is not in there at all. Counting the emitted columns instead
    of the covered ones would overstate the position by roughly a factor of two, which is exactly
    the error PORT_STATUS.md's "warning about the number 182" is about.

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
    import hume
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
        return _core.all_from_pickles(pk.blobs, *rings, pk.h_blobs, sel)

    res["cpp_all_columns"] = _timed(lambda _b: _all(), lambda: None)
    res["cpp_all_columns"]["columns"] = len(cols)

    # PER FAMILY, PAIRED AND DIFFERENCED PER REPETITION. `all_from_pickles(blobs, [f])` runs the
    # 182 blocks plus family f -- the blocks are not optional, they are what fills the EState
    # index -- so a family's own cost is that minus the blocks-only arm. The subtraction is done
    # WITHIN each repetition and the SD taken over the differences, not over the two arms
    # separately: on a contended box the two arms move together, and differencing the means
    # would throw away exactly the pairing that makes the number mean anything.
    fam_names = [k for k in hume.FAMILY_OFFSETS if k not in ("blocks", "end")]
    per_rep: dict[str, list[float]] = {k: [] for k in ["blocks_only"] + fam_names}
    arms = ["blocks_only"] + fam_names
    for cyc in range(n_reps):
        for name in arms[cyc % len(arms):] + arms[:cyc % len(arms)]:
            sel = [] if name == "blocks_only" else [name]
            t0 = _cpu()
            _all(sel)
            t1 = _cpu()
            per_rep[name].append((t1 - t0) / len(mols) * 1e6)
    base = per_rep["blocks_only"]
    fams = {"blocks_182": {"us_mean": round(statistics.mean(base), 3),
                           "us_sd": round(statistics.stdev(base), 3) if len(base) > 1 else None,
                           "columns": hume.N_COLS}}
    for name in fam_names:
        d = [a - b for a, b in zip(per_rep[name], base)]
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
    if a["machine"]["contended"] or b["machine"]["contended"]:
        print("MEASURED UNDER CONTENTION -- no headline ratio. Re-run on a quiet machine.")
        return
    ta = sum(s["us_mean"] for s in a["steps"].values())
    tb = sum(s["us_mean"] for s in b["steps"].values())
    print(f"end-to-end: HUME {ta:.1f} us/mol vs baseline {tb:.1f} us/mol -> {tb / ta:.1f}x")
    if a["columns_descriptors"] != b["columns_descriptors"]:
        print(f"  CAVEAT: HUME covers {a['columns_descriptors']} descriptor columns, the "
              f"baseline {b['columns_descriptors']}. The ratio is NOT like-for-like until the "
              f"port is complete.")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        return report()
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
