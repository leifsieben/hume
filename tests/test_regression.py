"""Every emitted value, against the committed fixture.

A failure here means a descriptor moved. That is either a bug or an intended change; if it is
intended, regenerate with tools/gen_fixture.py and record what moved in CHANGELOG.md.

TWO TIERS, because the architecture is part of the specification. This library reproduces
upstream floating-point BEHAVIOR, not just upstream mathematics, so a different libm's log and
a different FMA decision move the last digits. Measured in CI: on the machine the fixture came
from all 1,269 columns are bit-identical, while x86-64 gcc moves 594 of them and MSVC 595.

So: EXACT on the fixture's own platform, where any movement at all is a real change, and within
RTOL elsewhere. The NaN pattern is compared exactly on every platform, because which cells are
undefined is structural and must not drift with the compiler.

THE TOLERANT TIER SCALES BY COLUMN, NOT BY CELL, and that is not a detail. Several columns here
are differences that cancel to near zero -- the centered autocorrelations (ATSC/AATSC/MATS),
Cyclicity, DeltaMean/DeltaMax. A per-cell relative comparison judges a value of 5e-14 against
itself, so an absolute wobble of 1e-12 reads as a relative error of 27, and 28 columns look
catastrophically broken when the real disagreement is in the last bits. Measured against each
column's own dynamic range instead, the worst of those same 28 is 9.5e-15.

That is the honest metric: each column must agree to RTOL of the range it actually spans.
"""
import platform

import numpy as np
import pytest
from rdkit import Chem

import molhume

#: The tolerance used everywhere else in this project to grade a column as reproduced. Observed
#: cross-platform drift is ~1e-15, so this leaves about six orders of magnitude of headroom --
#: deliberately, because it is a bug detector, not a measurement. tools/platform_drift.py prints
#: the actual distribution for the machine you are on.
RTOL = 1e-9


@pytest.fixture(scope="module")
def actual(smiles):
    # the fixture is descriptors-only; see tools/gen_fixture.py
    return molhume.featurize(smiles, standardize="none", fingerprint=False)


def test_columns_match(expected):
    assert molhume.feature_names(fingerprint=False) == expected["names"]


def _this_platform():
    return f"{platform.system()} {platform.machine()}"


def test_values_match(actual, expected):
    got, want = actual, expected["X"]
    assert got.shape == want.shape

    # Structural, and compared exactly everywhere: which cells are undefined must not depend on
    # the compiler. A NaN appearing or vanishing is a logic change, never rounding.
    bad_nan = np.argwhere(np.isnan(got) != np.isnan(want))
    if len(bad_nan):
        r, c = bad_nan[0]
        pytest.fail(f"{len(bad_nan)} cell(s) changed finite/NaN status, first at row {r} "
                    f"column {expected['names'][c]}: got {got[r, c]!r}, expected {want[r, c]!r}")

    finite = np.isfinite(want)
    absd = np.zeros_like(want)
    absd[finite] = np.abs(got[finite] - want[finite])
    rel = np.zeros_like(want)
    nz = finite & (np.abs(want) > 0)
    rel[nz] = absd[nz] / np.abs(want[nz])

    same_platform = _this_platform() == expected["platform"]
    if same_platform:
        if absd.max() > 0:
            r, c = np.unravel_index(np.argmax(absd), absd.shape)
            n_cols = len({int(j) for j in np.argwhere(absd > 0)[:, 1]})
            pytest.fail(
                f"{n_cols} column(s) moved on the fixture's own platform "
                f"({expected['platform']}), where the comparison is exact. Largest at row {r} "
                f"column {expected['names'][c]}: {got[r, c]!r} vs expected {want[r, c]!r}. "
                "If the change was intended, regenerate with tools/gen_fixture.py and record "
                "the moved columns in CHANGELOG.md.")
        return

    # Each column against its OWN dynamic range. See the module docstring: a per-cell relative
    # comparison is meaningless for a column that cancels to near zero.
    scale = np.where(finite, np.abs(want), 0.0).max(axis=0)
    scale[scale == 0.0] = 1.0                    # an all-zero column must stay all-zero
    scaled = absd.max(axis=0) / scale
    over = np.argwhere(scaled > RTOL).ravel()
    if over.size:
        c = int(over[np.argmax(scaled[over])])
        r = int(np.argmax(absd[:, c]))
        pytest.fail(
            f"{over.size} column(s) differ by more than rtol={RTOL:g} of their own range. The "
            f"fixture came from {expected['platform']}; this is {_this_platform()}. Worst is "
            f"{expected['names'][c]}: {got[r, c]!r} vs {want[r, c]!r} at row {r}, "
            f"absolute {absd[:, c].max():.3e} against a column range of {scale[c]:.3e} "
            f"= {scaled[c]:.3e}. Cross-platform arithmetic drift measures ~1e-14 by this "
            "metric, so this is too large to be that. Run tools/platform_drift.py for the "
            "full distribution.")


def test_fixture_spans_the_documented_range(expected):
    heavy = expected["n_heavy"]
    assert heavy.min() <= 3 and heavy.max() >= 50, (
        f"fixture spans only {heavy.min()}-{heavy.max()} heavy atoms; it is meant to cover the "
        "size range the exactness numbers were measured over")


def test_fixture_records_its_platform(expected):
    assert expected["platform"] != "(not recorded)", (
        "the fixture predates platform recording; regenerate it with tools/gen_fixture.py so "
        "the exact/tolerant comparison can tell which machine it came from")


def test_rdkit_version_note(expected):
    got = Chem.rdBase.rdkitVersion
    if got != expected["rdkit_version"]:
        pytest.skip(f"fixture was generated against rdkit {expected['rdkit_version']}, this "
                    f"environment has {got}; perceived atom and bond properties drift across "
                    "releases, so value differences here are expected, not regressions")
