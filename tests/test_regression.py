"""Every emitted value, against the committed fixture.

A failure here means a descriptor moved. That is either a bug or an intended change; if it is
intended, regenerate with tools/gen_fixture.py and record what moved in CHANGELOG.md.

TWO TIERS, because the architecture is part of the specification. This library reproduces
upstream floating-point BEHAVIOR, not just upstream mathematics, so a different libm's log and
a different FMA decision move the last two or three digits. Measured: on the machine the fixture
came from every one of the 1,269 columns is bit-identical, while x86-64 gcc moves 594 of them
and MSVC 595 -- all at a relative difference around 1e-15.

So: EXACT on the fixture's own platform, where any movement at all is a real change. Within
RTOL elsewhere, which still catches a logic error by orders of magnitude while not failing on
arithmetic nobody can control. The NaN pattern is compared exactly on every platform, because
which cells are undefined is structural and must not drift with the compiler.
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
    return molhume.featurize(smiles, standardize="none")


def test_columns_match(expected):
    assert molhume.feature_names() == expected["names"]


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

    worst = rel.max()
    if worst > RTOL:
        c = int(np.unravel_index(np.argmax(rel), rel.shape)[1])
        r = int(np.unravel_index(np.argmax(rel), rel.shape)[0])
        n_cols = len({int(j) for j in np.argwhere(rel > RTOL)[:, 1]})
        pytest.fail(
            f"{n_cols} column(s) differ by more than rtol={RTOL:g} from the fixture, which came "
            f"from {expected['platform']} and is being compared on {_this_platform()}. Largest "
            f"at row {r} column {expected['names'][c]}: {got[r, c]!r} vs {want[r, c]!r} "
            f"(rel {worst:.3e}). Cross-platform arithmetic drift is ~1e-15; this is too big to "
            "be that. Run tools/platform_drift.py for the full distribution.")


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
