"""How far does this machine's build differ from the committed fixture, and where?

    python tools/platform_drift.py

The exactness numbers in README.md were measured on macOS arm64 with clang. This library
reproduces upstream floating-point BEHAVIOR, so the toolchain and the architecture are part of
the specification -- a different libm's log, a different FMA decision, and the last two or three
digits move. This prints how much, so the tolerance in tests/test_regression.py is a measured
number rather than a guess, and so a real regression can be told apart from ordinary
cross-platform noise.

CI runs this on every platform. Read it as a distribution, not a pass/fail: a handful of columns
at 1e-15 is arithmetic; one column at 1e-3 is a bug.
"""
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import molhume  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")

smis = (ROOT / "tests/data/fixture_smiles.txt").read_text().split()
with np.load(ROOT / "tests/data/fixture_expected.npz") as z:
    want = z["X"]
    names = [str(n) for n in z["names"]]
    ref_rdkit = str(z["rdkit_version"])
    ref_plat = str(z["platform"]) if "platform" in z else "(not recorded)"

got = molhume.featurize(smis, standardize="none")

print(f"  this machine : {platform.system()} {platform.machine()}, python "
      f"{sys.version.split()[0]}, rdkit {Chem.rdBase.rdkitVersion}")
print(f"  fixture from : {ref_plat}, rdkit {ref_rdkit}")

nan_d = np.isnan(got) != np.isnan(want)
fin = np.isfinite(got) & np.isfinite(want)
absd = np.zeros_like(want)
absd[fin] = np.abs(got[fin] - want[fin])
rel = np.zeros_like(want)
nz = fin & (np.abs(want) > 0)
rel[nz] = absd[nz] / np.abs(want[nz])

moved = sorted({int(c) for c in np.argwhere((absd > 0) | nan_d)[:, 1]})
print(f"\n  columns bit-identical : {want.shape[1] - len(moved)} / {want.shape[1]}")
print(f"  columns that moved    : {len(moved)}")
print(f"  NaN-pattern changes   : {int(nan_d.sum())} cells   <- must be 0")

# Scaled by each column's own dynamic range, which is the metric the regression test uses:
# several columns here cancel to near zero, and judging such a value against itself turns a
# last-bit wobble into a relative error of 27.
scale = np.where(fin, np.abs(want), 0.0).max(axis=0)
scale[scale == 0.0] = 1.0
scaled = absd.max(axis=0) / scale

if moved:
    print(f"\n  MAX per-cell relative difference : {rel.max():.3e}   <- misleading, see below")
    print(f"  MAX difference / column range   : {scaled.max():.3e}   <- what the test asserts")
    per_col = scaled[moved]
    print("\n  difference as a fraction of each moved column's range:")
    for lo, hi in [(0, 1e-15), (1e-15, 1e-13), (1e-13, 1e-11), (1e-11, 1e-9),
                   (1e-9, 1e-6), (1e-6, np.inf)]:
        n = int(((per_col > lo) & (per_col <= hi)).sum())
        if n:
            print(f"     {lo:8.0e} < rel <= {hi:8.0e} : {n:4d} columns")
    order = np.argsort(-scaled)
    print("  worst columns, by difference against their own range:")
    for c in order[:15]:
        if scaled[c] == 0:
            break
        print(f"     {names[c]:26s} {scaled[c]:.3e}   (abs {absd[:, c].max():.3e}, "
              f"range {scale[c]:.3e}, per-cell rel {rel[:, c].max():.3e})")
else:
    print("\n  bit-identical to the fixture.")
