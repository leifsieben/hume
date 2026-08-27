"""RDKit's Crippen baseline, measured COLD -- the number cpp/crippen.cpp has to beat.

    .venv/bin/python cpp/bench_crippen.py [corpus.smi] [n]

THE TRAP THIS EXISTS TO AVOID. `_CalcCrippenContribs` MEMOISES its answer on the molecule. Call
it twice on the same Mol and the second call is a property lookup, not a typing pass. A benchmark
that loops `for m in mols: fn(m)` over several repetitions therefore times one real pass and
N-1 cache hits, and reports something like 1 us/mol for work that costs 80. This project has
already been caught by exactly that once. `force=True` re-runs the typer every time, and the two
numbers are printed side by side below so the gap is visible rather than assumed.

WHY TIMING RDKIT THROUGH PYTHON IS STILL A C++ MEASUREMENT. `_CalcCrippenContribs` is a
Boost.Python wrapper over a C++ routine; one call is one boundary crossing plus the C++ work. The
boundary is measured separately against the cheapest possible C++ accessor and subtracted, so
what is reported is the C++ work -- an upper bound, since a native caller pays no boundary. This
is the same methodology as cpp/time_rdkit_cpp.py. It is NOT a Python-implementation timing.

Molecules are built once, outside the timed region: SMILES parsing is not part of Crippen.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rd

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
REPS = 11


def bench(mols, fn) -> list[float]:
    fn(mols[0])
    out = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for m in mols:
            fn(m)
        out.append((time.perf_counter() - t0) / len(mols) * 1e6)
    return sorted(out)


def report(name: str, ts: list[float], sub: float = 0.0) -> float:
    med = statistics.median(ts) - sub
    lo, hi = ts[0] - sub, ts[-1] - sub
    print(f"  {name:44s} {med:8.2f} us/mol   [{lo:7.2f}, {hi:7.2f}]  "
          f"({100 * (lo - med) / med:+.1f}% / {100 * (hi - med) / med:+.1f}%)")
    return med


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "hard.smi"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    smis = src.read_text().split()[:n]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    na = sum(m.GetNumAtoms() for m in mols) / len(mols)
    print(f"{len(mols):,} molecules from {src.name}, mean {na:.1f} heavy atoms, "
          f"{REPS} reps, median [min, max]\n")

    over = bench(mols, lambda m: m.GetNumAtoms())
    o = report("Python->C++ boundary (subtracted below)", over)

    print()
    report("_CalcCrippenContribs, WARM (cache hit -- WRONG)", bench(mols, rd._CalcCrippenContribs), o)
    cold = report("_CalcCrippenContribs, COLD (force=True)",
                  bench(mols, lambda m: rd._CalcCrippenContribs(m, True)), o)
    print(f"\n  the memoisation is worth {cold:.1f} us/mol -- that is the whole descriptor")


if __name__ == "__main__":
    main()
