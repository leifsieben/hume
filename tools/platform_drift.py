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

if moved:
    per_col = rel.max(axis=0)[moved]
    print("\n  worst relative difference per moved column:")
    for lo, hi in [(0, 1e-15), (1e-15, 1e-13), (1e-13, 1e-11), (1e-11, 1e-9),
                   (1e-9, 1e-6), (1e-6, np.inf)]:
        n = int(((per_col > lo) & (per_col <= hi)).sum())
        if n:
            print(f"     {lo:8.0e} < rel <= {hi:8.0e} : {n:4d} columns")
    print(f"\n  MAX relative difference anywhere: {rel.max():.3e}")
    order = np.argsort(-rel.max(axis=0))
    print("  worst columns:")
    for c in order[:15]:
        if rel[:, c].max() == 0:
            break
        print(f"     {names[c]:26s} rel {rel[:, c].max():.3e}  abs {absd[:, c].max():.3e}")
else:
    print("\n  bit-identical to the fixture.")
