"""The flags a user sees: what each one does, and what it says when it is given nonsense."""
import warnings

import numpy as np
import pytest

import molhume

SMIS = ["CCO", "c1ccccc1N", "CC(=O)Oc1ccccc1C(=O)O", "[Na+].CC(=O)[O-]"]
BAD = "not_a_molecule"


def _quiet(*a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return molhume.featurize(*a, **kw)


# ---------------------------------------------------------------- standardize

def test_unset_standardize_warns_once_and_behaves_as_none(monkeypatch):
    monkeypatch.setattr(molhume, "_STANDARDIZE_WARNED", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, X1, _ = molhume.featurize(SMIS)
        _, X2, _ = molhume.featurize(SMIS)
    msgs = [str(x.message) for x in w if "standardize" in str(x.message)]
    assert len(msgs) == 1, f"expected exactly one standardize warning, got {len(msgs)}"
    assert "'canonical'" in msgs[0] and "'cleanup'" in msgs[0], "warning must name the choices"
    _, X0, _ = _quiet(SMIS, standardize="none")
    assert np.array_equal(X1, X0, equal_nan=True) and np.array_equal(X2, X0, equal_nan=True)


def test_explicit_none_is_silent(monkeypatch):
    monkeypatch.setattr(molhume, "_STANDARDIZE_WARNED", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        molhume.featurize(SMIS, standardize="none")
    assert [str(x.message) for x in w if "standardize" in str(x.message)] == []


def test_cleanup_strips_the_salt_and_neutralizes():
    _, none, _ = _quiet(["[Na+].CC(=O)[O-]"], standardize="none", columns=["ExactMolWt"])
    _, clean, _ = _quiet(["[Na+].CC(=O)[O-]"], standardize="cleanup", columns=["ExactMolWt"])
    assert none[0, 0] == pytest.approx(81.99, abs=0.02), "sodium acetate as supplied"
    assert clean[0, 0] == pytest.approx(60.02, abs=0.02), "acetic acid after cleanup"


def test_canonical_is_a_round_trip_not_a_change():
    _, a, _ = _quiet(["C(C)O"], standardize="canonical")
    _, b, _ = _quiet(["CCO"], standardize="none")
    assert np.array_equal(a, b, equal_nan=True)


def test_standardize_accepts_a_callable():
    from rdkit import Chem
    seen = []

    def strip(m):
        seen.append(m)
        return Chem.MolFromSmiles(max(Chem.MolToSmiles(m).split("."), key=len))

    _, X, _ = _quiet(["[Na+].CC(=O)[O-]"], standardize=strip, columns=["ExactMolWt"])
    assert len(seen) == 1
    assert X[0, 0] == pytest.approx(59.01, abs=0.02), "acetate anion, the callable's choice"


def test_bad_standardize_names_the_choices():
    with pytest.raises(ValueError, match="canonical.*cleanup|cleanup.*canonical"):
        molhume.featurize(SMIS, standardize="normalise")


# ---------------------------------------------------------------- on_error

def test_nan_keeps_the_row_and_the_alignment():
    smis = [SMIS[0], BAD, SMIS[1]]
    with pytest.warns(UserWarning, match="did not parse"):
        _, X, _ = molhume.featurize(smis, standardize="none", on_error="nan")
    assert X.shape[0] == 3
    assert np.all(np.isnan(X[1]))
    assert np.isfinite(X[0]).any() and np.isfinite(X[2]).any()


def test_skip_drops_the_row_and_says_so():
    smis = [SMIS[0], BAD, SMIS[1]]
    with pytest.warns(UserWarning, match="no longer aligns"):
        _, X, _ = molhume.featurize(smis, standardize="none", on_error="skip")
    assert X.shape[0] == 2


def test_raise_names_the_offending_index_and_string():
    with pytest.raises(ValueError, match=r"index 1.*not_a_molecule"):
        molhume.featurize([SMIS[0], BAD], standardize="none", on_error="raise")


def test_bad_on_error_names_the_choices():
    with pytest.raises(ValueError, match="'nan'.*'raise'.*'skip'|nan.*raise.*skip"):
        molhume.featurize(SMIS, standardize="none", on_error="ignore")


def test_all_bad_input_still_returns_an_aligned_block():
    with pytest.warns(UserWarning):
        fp, X, names = molhume.featurize([BAD, BAD], standardize="none")
    assert X.shape == (2, len(molhume.ALL_COLUMNS)) and np.all(np.isnan(X))
    assert fp.shape[0] == 2 and not fp.any()


def test_empty_input():
    fp, X, names = molhume.featurize([], standardize="none")
    assert X.shape[0] == 0 and X.shape[1] == len(names) == len(molhume.ALL_COLUMNS)


# ---------------------------------------------------------------- columns

def test_columns_selects_and_orders():
    want = ["TPSA", "ExactMolWt", "SLogP"]
    _, X, names = _quiet(SMIS, standardize="none", columns=want)
    assert names == tuple(want), "the caller's order is the output order"
    _, full, full_names = _quiet(SMIS, standardize="none")
    idx = [full_names.index(c) for c in want]
    assert np.array_equal(X, full[:, idx], equal_nan=True)


def test_unknown_column_is_named_and_points_at_the_list():
    with pytest.raises(ValueError, match="NotAColumn.*ALL_COLUMNS|ALL_COLUMNS"):
        molhume.featurize(SMIS, standardize="none", columns=["TPSA", "NotAColumn"])


def test_repeated_column_is_emitted_once_with_a_warning():
    with pytest.warns(UserWarning, match="repeated"):
        _, X, names = molhume.featurize(SMIS, standardize="none", columns=["TPSA", "TPSA"])
    assert names == ("TPSA",) and X.shape[1] == 1


def test_empty_selection_is_an_error():
    with pytest.raises(ValueError, match="no columns"):
        molhume.featurize(SMIS, standardize="none", columns=[])


# ------------------------------------------------- additional_descriptors

def test_additional_off_removes_exactly_our_own_columns():
    _, _, on = _quiet(SMIS, standardize="none")
    _, _, off = _quiet(SMIS, standardize="none", additional_descriptors=False)
    from molhume._additional import ADDITIONAL_COLUMNS
    ours = set(ADDITIONAL_COLUMNS) & set(on)
    assert set(on) - set(off) == ours
    assert len(off) < len(on) and ours, "there must be some of our own columns to remove"


def test_additional_off_does_not_change_the_columns_it_keeps():
    _, a, on = _quiet(SMIS, standardize="none")
    _, b, off = _quiet(SMIS, standardize="none", additional_descriptors=False)
    idx = [on.index(c) for c in off]
    assert np.array_equal(a[:, idx], b, equal_nan=True)


def test_the_two_filters_are_combined_with_and():
    from molhume._additional import ADDITIONAL_COLUMNS
    mine = next(c for c in ADDITIONAL_COLUMNS if c in molhume.ALL_COLUMNS)
    with pytest.warns(UserWarning, match="additional_descriptors=False removed"):
        _, X, names = molhume.featurize(SMIS, standardize="none",
                                        additional_descriptors=False, columns=["TPSA", mine])
    assert names == ("TPSA",)


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_off_returns_no_columns_but_keeps_the_rows():
    fp, X, _ = _quiet(SMIS, standardize="none", fingerprint=False)
    assert fp.shape == (len(SMIS), 0) and X.shape[0] == len(SMIS)


@pytest.mark.parametrize("size", [512, 2048])
@pytest.mark.parametrize("radius", [2, 3])
def test_fp_radius_and_size_reach_the_fingerprint(size, radius):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    fp, _, _ = _quiet(SMIS, standardize="none", fp_radius=radius, fp_size=size)
    assert fp.shape == (len(SMIS), size)
    gen = rfg.GetMorganGenerator(radius=radius, fpSize=size)
    for i, s in enumerate(SMIS):
        want = np.array(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)))
        assert np.array_equal(fp[i].astype(bool), want.astype(bool)), f"row {i}"


# ---------------------------------------------------------------- package shape

def test_the_alias_package_is_the_same_module():
    import mol_hume
    assert mol_hume.featurize is molhume.featurize
    assert mol_hume.ALL_COLUMNS is molhume.ALL_COLUMNS


def test_all_columns_is_unique_and_matches_the_output_width():
    cols = molhume.ALL_COLUMNS
    assert len(set(cols)) == len(cols), "ALL_COLUMNS contains a duplicate name"
    _, X, names = _quiet(["CCO"], standardize="none")
    assert X.shape[1] == len(cols) == len(names)


def test_featurize_is_exported():
    assert "featurize" in molhume.__all__
