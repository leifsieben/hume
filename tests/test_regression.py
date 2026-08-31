"""Every emitted value, against the committed fixture.

A failure here means a descriptor moved. That is either a bug or an intended change; if it is
intended, regenerate with tools/gen_fixture.py and record what moved in CHANGELOG.md.
"""
import numpy as np
import pytest
from rdkit import Chem

import molhume


@pytest.fixture(scope="module")
def actual(smiles):
    _, X, names = molhume.featurize(smiles, standardize="none")
    return X, names


def test_columns_match(actual, expected):
    assert actual[1] == expected["names"]


def test_values_match(actual, expected):
    got, want = actual[0], expected["X"]
    assert got.shape == want.shape
    same_nan = np.isnan(got) == np.isnan(want)
    bad_nan = np.argwhere(~same_nan)
    if len(bad_nan):
        r, c = bad_nan[0]
        pytest.fail(f"{len(bad_nan)} cell(s) changed finite/NaN status, first at row {r} "
                    f"column {expected['names'][c]}: got {got[r, c]!r}, expected {want[r, c]!r}")
    finite = np.isfinite(want)
    diff = np.zeros_like(want)
    diff[finite] = np.abs(got[finite] - want[finite])
    if diff.max() > 0:
        r, c = np.unravel_index(np.argmax(diff), diff.shape)
        n_cols = len({int(j) for j in np.argwhere(diff > 0)[:, 1]})
        pytest.fail(f"{n_cols} column(s) moved; largest at row {r} column "
                    f"{expected['names'][c]}: {got[r, c]!r} vs expected {want[r, c]!r}")


def test_fixture_spans_the_documented_range(expected):
    heavy = expected["n_heavy"]
    assert heavy.min() <= 3 and heavy.max() >= 50, (
        f"fixture spans only {heavy.min()}-{heavy.max()} heavy atoms; it is meant to cover the "
        "size range the exactness numbers were measured over")


def test_rdkit_version_note(expected):
    got = Chem.rdBase.rdkitVersion
    if got != expected["rdkit_version"]:
        pytest.skip(f"fixture was generated against rdkit {expected['rdkit_version']}, this "
                    f"environment has {got}; perceived atom and bond properties drift across "
                    "releases, so value differences here are expected, not regressions")
