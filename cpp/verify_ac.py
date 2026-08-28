"""Is the C++ Autocorrelation block exactly Mordred?

    cd cpp && ./ac verify mols_h.txt
    .venv-mordred/bin/python cpp/verify_ac.py [n_mols]

NEEDS THE MORDRED INTERPRETER, not `.venv`: mordred 1.2.0 imports `distutils` (python 3.11 only)
and needs numpy 1.x. `.venv-mordred` is that environment. Asking for it wrongly fails SILENTLY --
`uv` resolves mordred DOWN to 0.6.0, a different library, rather than erroring, when numpy 2 is
also requested -- so this file checks `mordred.__version__` and refuses to grade anything else,
and prints RDKit's numeric canary rather than trusting the version banner.

Compares all 540 cells (6 variants x 9 lags x 10 weights) per molecule against Mordred itself,
not against ac_reference.py -- the NumPy reference exists to pin the spec, and checking the C++
against it would only prove the two agree with each other.

NaN is a VALUE here, not a failure: Mordred returns NaN for a lag no pair reaches, and a
molecule small enough that lag 8 is empty must produce NaN on both sides. Mismatched
NaN-ness is counted as an error in its own right, because silently treating NaN == NaN as a
pass would let a block that returns nothing look perfect.

THE 486 ARE RE-GRADED ALONGSIDE THE 54, and the split is printed. `Z`, mordred's tenth weight,
was added after the 486-column values_ac.txt had already been verified and checksummed, so this
run has two jobs: show the new weight is right, and show the nine that were already right did
not move. A pass line that merged them would only prove the first. The `weight` table below is
per weight for that reason, and the summary counts `Z` separately from the rest.

ALL 54 `Z` COLUMNS ARE GRADED, BUT ONLY 52 ARE MEMBERS OF THE 865. `MATS0Z` and `GATS0Z` are
computable -- mordred answers 1.0 and 0.0 for them, as it does for `MATS0c`/`GATS0c` -- they
simply do not survive into the deduplicated census, the same way the other weights' lag-0
MATS/GATS do not. So they are checked here like any other column and counted as coverage
nowhere: 54 graded, at most 52 ever claimed as coverage.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
VARIANTS = ["ATS", "AATS", "ATSC", "AATSC", "MATS", "GATS"]
LAGS = list(range(9))
# `Z` LAST, matching cpp/ac_weights.h's append rather than mordred's own getter order: the point
# of appending is that none of the 486 pre-existing columns changes name or meaning.
WEIGHTS = ["c", "d", "dv", "i", "p", "v", "se", "pe", "are", "Z"]
NEW = "Z"      # the weight added after the 486-column artifact was verified; reported separately
# TOLERANCE IS SCALED PER COLUMN, because these are SUMS WITH CANCELLATION and neither a pure
# relative nor a pure absolute test is correct for them.
#
# ATS values span 1e-3 to 1.7e5 across columns, so a fixed absolute tolerance is meaningless.
# But ATSC sums products of CENTRED properties, which cancel: a cell whose true value is ~1e-9
# is the difference of terms of order 1, and asking for 1e-9 RELATIVE agreement there is asking
# for exact cancellation in two different summation orders, which IEEE arithmetic does not
# provide. Measured over 194,184 cells the C++ and Mordred agree to 6.9e-07 absolute at worst,
# against a maximum cell magnitude of 1.7e+05.
#
# So: relative 1e-8, plus an absolute floor set to 1e-8 of THAT COLUMN's own scale. Eight orders
# of magnitude in both directions is far below anything a porting bug produces -- the bugs this
# harness found elsewhere were off by 1e-1.
RTOL, SCALE_FRAC = 1e-8, 1e-8

NAMES = [(v, k, w) for v in VARIANTS for k in LAGS for w in WEIGHTS]
CANARY_SMI = "O=C1CCNCCNNNCCNCCC(=O)c2ccc(o2)COCOCc2ccc1o2"
CANARY = -0.07665884800196521

_CALC = None


def _init() -> None:
    """One Calculator per worker. Building it is not free and it is stateless across molecules."""
    global _CALC
    from mordred import Autocorrelation as AC, Calculator

    RDLogger.DisableLog("rdApp.*")
    _CALC = Calculator([getattr(AC, v)(k, w) for v, k, w in NAMES])


def _rows(smis: list[str]) -> np.ndarray:
    out = np.full((len(smis), len(NAMES)), np.nan)
    for i, s in enumerate(smis):
        for j, r in enumerate(_CALC(Chem.MolFromSmiles(s))):
            try:
                out[i, j] = float(r)
            except Exception:      # noqa: BLE001 -- a mordred Error object; NaN is the answer
                pass
    return out


def read_values(path: Path, n_cols: int, n_rows: int) -> np.ndarray:
    """values_ac.txt, streamed a line at a time.

    NOT np.loadtxt: the 540-column file is 708 MB of text and loadtxt builds it through Python
    objects. The row count and column count are both asserted per line, so a short or long row
    stops the run here rather than shifting every comparison after it -- the same discipline
    ac.cpp's loader applies to its input.
    """
    out = np.empty((n_rows, n_cols))
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            row = np.fromstring(line, sep=" ")
            if row.size != n_cols:
                raise ValueError(f"row {i} has {row.size} columns, expected {n_cols}")
            out[i] = row
    return out


# --------------------------------------------------------------------------------------------
# The mordred reference: ON DISK, INCREMENTALLY, RESUMABLE.
#
# THIS REPLACED `np.vstack(pool.map(...))`. A three-hour full-corpus run did produce nothing, but
# NOT because of this code -- an operator killed it deliberately, and the post-mortem written here
# first (memory exhaustion at the vstack peak) was a guess that fitted the symptoms and was wrong.
# It is recorded as a wrong guess because the symptoms are worth recognising: a parent that dies
# with NO Python traceback while its workers report `BrokenPipeError` writing into the result
# queue is what an EXTERNAL KILL looks like from the inside. It is not what MemoryError looks
# like. Do not diagnose from a broken pipe alone.
#
# The two changes stand on their own merits regardless:
#   1. Peak memory scaled with the corpus. `pool.map` returns only when every chunk is done, so
#      the parent held the `got` matrix, the list of chunk arrays and the vstack destination at
#      once -- three 427 MB copies at 98,905 x 540. `imap` streams results in order instead, so
#      the parent holds one chunk (~5 MB). On a 24 GB box shared with other jobs that headroom
#      matters even though it is not what ended the run.
#   2. NOTHING WAS PERSISTED UNTIL THE END, and this is the valuable half. It is what turns ANY
#      interruption -- an operator, a laptop lid, a scheduler -- into total loss, and an oracle
#      that costs tens of core-hours must not be all-or-nothing. The reference now lands in a
#      memmap as it is computed with a progress counter written beside it, so the next attempt
#      resumes from the last completed chunk instead of from zero.
#
# AND: KILLING THE PARENT DOES NOT KILL THE POOL. When that run was terminated, seven of its ten
# workers reparented to PID 1 and kept running at 150-180% CPU EACH for another three and a half
# hours -- about eleven cores of a twelve-core box, long after anything was reading their output.
# Load average only fell from 156 to 92 when they were hunted down by hand. Anyone stopping this
# script must kill the process GROUP, not the parent: `pkill -f verify_ac.py` leaves the workers,
# whose command line is `python -c from multiprocessing...` and matches nothing obvious.
#
# THE CACHE IS KEYED ON ITS INPUTS, not just on its shape. A stale reference silently graded
# against a regenerated values_ac.txt would be worse than no cache at all, so the header carries
# the SMILES digest and the oracle versions, and any mismatch discards the cache rather than
# resuming onto it.
# --------------------------------------------------------------------------------------------

def _cache_key(smis: list[str], n_cols: int) -> dict:
    import mordred
    return {"n_rows": len(smis), "n_cols": n_cols,
            "smi_sha": hashlib.sha256("\n".join(smis).encode()).hexdigest(),
            "mordred": mordred.__version__, "rdkit": rdkit.__version__}


def build_ref(smis: list[str], chunks: list[list[str]], nproc: int, cache: Path) -> np.ndarray:
    """-> the (n_mols, 540) mordred reference, resuming a partial run if one is on disk."""
    key = _cache_key(smis, len(NAMES))
    meta, npy = cache.with_suffix(".json"), cache.with_suffix(".npy")
    done = 0
    if meta.exists() and npy.exists():
        old = json.loads(meta.read_text())
        if {k: old.get(k) for k in key} == key:
            done = int(old.get("rows_done", 0))
            print(f"  resuming: {done:,} / {len(smis):,} reference rows already on disk")
        else:
            print("  cache discarded: inputs or oracle versions differ from the stored run")
    mode = "r+" if done and npy.exists() else "w+"
    ref = np.lib.format.open_memmap(npy, mode=mode, dtype=np.float64,
                                    shape=(len(smis), len(NAMES)))
    if done >= len(smis):
        return ref

    starts, off, todo = [], 0, []
    for c in chunks:
        starts.append(off)
        if off >= done:
            todo.append((off, c))
        off += len(c)
    if todo and todo[0][0] != done:
        # Chunk boundaries must line up with the counter or the resume would write rows into the
        # wrong place. Only whole chunks are ever counted, so this cannot happen -- assert anyway.
        raise SystemExit(f"resume misaligned: counter {done}, first pending chunk {todo[0][0]}")

    t0 = time.time()
    with mp.Pool(nproc, initializer=_init) as pool:
        for k, arr in enumerate(pool.imap(_rows, [c for _, c in todo])):
            start = todo[k][0]
            ref[start:start + len(arr)] = arr
            ref.flush()
            rows = start + len(arr)
            meta.write_text(json.dumps({**key, "rows_done": rows}))
            el = time.time() - t0
            frac = (rows - done) / max(1, len(smis) - done)
            print(f"    {rows:,} / {len(smis):,} rows  ({100*rows/len(smis):5.1f}%)  "
                  f"elapsed {el/60:.0f} min, eta {el/max(frac,1e-9)*(1-frac)/60:.0f} min",
                  flush=True)
    return ref


def main() -> None:
    import mordred

    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # THE NUMERIC CANARY, checked in the interpreter that computes the reference. PORT_STATUS.md:
    # a process can print the right RDKit version and execute another one's arithmetic out of
    # unlinked-but-still-mapped dylibs. A version banner is not evidence; this number is.
    canary = Descriptors.BCUT2D_MRLOW(Chem.MolFromSmiles(CANARY_SMI))
    print(f"mordred {mordred.__version__}   rdkit {rdkit.__version__}   numpy {np.__version__}   "
          f"python {sys.version.split()[0]}   CANARY {canary!r}")
    if canary != CANARY:
        raise SystemExit(f"CANARY MISMATCH: {canary!r}, expected {CANARY!r}. The RDKit executing "
                         f"is not the one on the label.")
    if mordred.__version__ != "1.2.0":
        raise SystemExit(f"WRONG MORDRED: {mordred.__version__}. uv resolves mordred DOWN to "
                         f"0.6.0 next to numpy 2 rather than erroring.")

    smis = HERE.joinpath("mols_h.smi").read_text().split()
    n_file = sum(1 for _ in HERE.joinpath("values_ac.txt").open())
    assert len(smis) == n_file, f"{len(smis)} smiles vs {n_file} rows in values_ac.txt"
    if n_want:
        smis = smis[:n_want]
    names = NAMES
    got = read_values(HERE / "values_ac.txt", len(names), len(smis))

    # ACROSS PROCESSES, because the corpus is 98,905 molecules and mordred's Autocorrelation on
    # it is measured in core-HOURS. Chunks are contiguous slices and `imap` preserves order, so
    # row k of the reference is molecule k by construction rather than by reassembly.
    #
    # SMALLER CHUNKS THAN THE OBVIOUS ONE, on purpose: the progress counter advances only on a
    # completed chunk, so the chunk size is also the maximum amount of work a crash can discard.
    # 400 chunks over 98,905 molecules caps that at about a quarter of one percent.
    nproc = max(1, min(mp.cpu_count() - 2, 10))
    step = max(1, (len(smis) + 399) // 400)
    chunks = [smis[i:i + step] for i in range(0, len(smis), step)]
    print(f"  mordred reference: {len(smis):,} molecules over {nproc} processes, "
          f"{len(chunks)} chunks of {step}", flush=True)
    ref = build_ref(smis, chunks, nproc, HERE / ".ac_ref_cache")
    assert ref.shape == got.shape, f"{ref.shape} vs {got.shape}"

    bad_val = np.zeros(len(names), int)
    bad_nan = np.zeros(len(names), int)
    worst = np.zeros(len(names))
    for j in range(len(names)):
        a, b = got[:, j], ref[:, j]
        na, nb = np.isnan(a), np.isnan(b)
        bad_nan[j] = int((na != nb).sum())
        ok = ~na & ~nb
        if ok.any():
            scale = float(np.median(np.abs(b[ok])))
            if not np.isfinite(scale) or scale == 0.0:
                scale = float(np.max(np.abs(b[ok]))) or 1.0
            atol = SCALE_FRAC * scale
            dev = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), atol)
            worst[j] = float(dev.max())
            bad_val[j] = int((np.abs(a[ok] - b[ok]) > atol + RTOL * np.abs(b[ok])).sum())

    print(f"{len(smis):,} molecules x {len(names)} cells = {len(smis)*len(names):,}\n")
    print(f"  {'variant':8s} {'cells':>7s} {'value err':>10s} {'NaN err':>9s} {'max rel dev':>13s}")
    ok_all = True
    for v in VARIANTS:
        sel = [j for j, nm in enumerate(names) if nm[0] == v]
        bv, bn = int(bad_val[sel].sum()), int(bad_nan[sel].sum())
        ok_all &= (bv == 0 and bn == 0)
        print(f"  {v:8s} {len(sel)*len(smis):7d} {bv:10d} {bn:9d} {worst[sel].max():13.3e}")

    # PER WEIGHT, AND THE NEW ONE ON ITS OWN LINE. The variant table above answers "is the block
    # right"; this one answers the question that made regenerating the artifact worth doing --
    # "did adding `Z` move any of the nine that were already verified". A merged pass line cannot
    # distinguish "the 486 are still exact" from "the 486 were re-graded and happened to pass
    # because the whole file was rewritten consistently".
    print(f"\n  {'weight':8s} {'cells':>7s} {'value err':>10s} {'NaN err':>9s} {'max rel dev':>13s}")
    for w in WEIGHTS:
        sel = [j for j, nm in enumerate(names) if nm[2] == w]
        bv, bn = int(bad_val[sel].sum()), int(bad_nan[sel].sum())
        mark = "   <- NEW" if w == NEW else ""
        print(f"  {w:8s} {len(sel)*len(smis):7d} {bv:10d} {bn:9d} "
              f"{worst[sel].max():13.3e}{mark}")

    old_cols = [j for j, nm in enumerate(names) if nm[2] != NEW]
    new_cols = [j for j, nm in enumerate(names) if nm[2] == NEW]
    old_ok = int(sum(bad_val[j] + bad_nan[j] == 0 for j in old_cols))
    new_ok = int(sum(bad_val[j] + bad_nan[j] == 0 for j in new_cols))
    print(f"\n  {old_ok} / {len(old_cols)} pre-existing columns exact"
          f"   |   {new_ok} / {len(new_cols)} new `{NEW}` columns exact")

    if not ok_all:
        j = int(np.argmax(bad_val + bad_nan))
        v, k, w = names[j]
        print(f"\n  worst column: {v}{k}{w}")
        a, b = got[:, j], ref[:, j]
        sc = SCALE_FRAC * float(np.nanmedian(np.abs(b)) or 1.0)
        m = np.where(~np.isclose(a, b, rtol=RTOL, atol=sc, equal_nan=True))[0]
        for i in m[:3]:
            print(f"    {smis[i][:60]}\n      c++ {a[i]!r}  mordred {b[i]!r}")
    print("\n" + ("ALL EXACT" if ok_all else "DISAGREEMENT"))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
