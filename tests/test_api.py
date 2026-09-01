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
        X1 = molhume.featurize(SMIS)
        X2 = molhume.featurize(SMIS)
    msgs = [str(x.message) for x in w if "standardize" in str(x.message)]
    assert len(msgs) == 1, f"expected exactly one standardize warning, got {len(msgs)}"
    assert "'canonical'" in msgs[0] and "'cleanup'" in msgs[0], "warning must name the choices"
    X0 = _quiet(SMIS, standardize="none")
    assert np.array_equal(X1, X0, equal_nan=True) and np.array_equal(X2, X0, equal_nan=True)


def test_explicit_none_is_silent(monkeypatch):
    monkeypatch.setattr(molhume, "_STANDARDIZE_WARNED", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        molhume.featurize(SMIS, standardize="none")
    assert [str(x.message) for x in w if "standardize" in str(x.message)] == []


def test_cleanup_strips_the_salt_and_neutralizes():
    none = _quiet(["[Na+].CC(=O)[O-]"], standardize="none", columns=["ExactMolWt"])
    clean = _quiet(["[Na+].CC(=O)[O-]"], standardize="cleanup", columns=["ExactMolWt"])
    assert none[0, 0] == pytest.approx(81.99, abs=0.02), "sodium acetate as supplied"
    assert clean[0, 0] == pytest.approx(60.02, abs=0.02), "acetic acid after cleanup"


def test_canonical_is_a_round_trip_not_a_change():
    a = _quiet(["C(C)O"], standardize="canonical")
    b = _quiet(["CCO"], standardize="none")
    assert np.array_equal(a, b, equal_nan=True)


def test_standardize_accepts_a_callable():
    from rdkit import Chem
    seen = []

    def strip(m):
        seen.append(m)
        return Chem.MolFromSmiles(max(Chem.MolToSmiles(m).split("."), key=len))

    X = _quiet(["[Na+].CC(=O)[O-]"], standardize=strip, columns=["ExactMolWt"])
    assert len(seen) == 1
    assert X[0, 0] == pytest.approx(59.01, abs=0.02), "acetate anion, the callable's choice"


def test_bad_standardize_names_the_choices():
    with pytest.raises(ValueError, match="canonical.*cleanup|cleanup.*canonical"):
        molhume.featurize(SMIS, standardize="normalise")


# ---------------------------------------------------------------- on_error

def test_nan_keeps_the_row_and_the_alignment():
    smis = [SMIS[0], BAD, SMIS[1]]
    with pytest.warns(UserWarning, match="did not parse"):
        X = molhume.featurize(smis, standardize="none", on_error="nan")
    assert X.shape[0] == 3
    assert np.all(np.isnan(X[1]))
    assert np.isfinite(X[0]).any() and np.isfinite(X[2]).any()


def test_skip_drops_the_row_and_says_so():
    smis = [SMIS[0], BAD, SMIS[1]]
    with pytest.warns(UserWarning, match="no longer aligns"):
        X = molhume.featurize(smis, standardize="none", on_error="skip")
    assert X.shape[0] == 2


def test_raise_names_the_offending_index_and_string():
    with pytest.raises(ValueError, match=r"index 1.*not_a_molecule"):
        molhume.featurize([SMIS[0], BAD], standardize="none", on_error="raise")


def test_bad_on_error_names_the_choices():
    with pytest.raises(ValueError, match="'nan'.*'raise'.*'skip'|nan.*raise.*skip"):
        molhume.featurize(SMIS, standardize="none", on_error="ignore")


def test_all_bad_input_still_returns_an_aligned_block():
    with pytest.warns(UserWarning):
        X = molhume.featurize([BAD, BAD], standardize="none", fingerprint=False)
    assert X.shape == (2, len(molhume.ALL_COLUMNS)) and np.all(np.isnan(X))


def test_empty_input():
    X = molhume.featurize([], standardize="none", fingerprint=False)
    assert X.shape == (0, len(molhume.ALL_COLUMNS))
    assert molhume.featurize([], standardize="none").shape == (0, len(molhume.ALL_COLUMNS) + 2048)


# ---------------------------------------------------------------- columns

def test_columns_selects_and_orders():
    want = ["TPSA", "ExactMolWt", "SLogP"]
    X = _quiet(SMIS, standardize="none", columns=want, fingerprint=False)
    assert molhume.feature_names(columns=want, fingerprint=False) == tuple(want), (
        "the caller's order is the output order")
    full = _quiet(SMIS, standardize="none", fingerprint=False)
    idx = [molhume.ALL_COLUMNS.index(c) for c in want]
    assert np.array_equal(X, full[:, idx], equal_nan=True)


def test_unknown_column_is_named_and_points_at_the_list():
    with pytest.raises(ValueError, match="NotAColumn.*ALL_COLUMNS|ALL_COLUMNS"):
        molhume.featurize(SMIS, standardize="none", columns=["TPSA", "NotAColumn"])


def test_repeated_column_is_emitted_once_with_a_warning():
    with pytest.warns(UserWarning, match="repeated"):
        X = molhume.featurize(SMIS, standardize="none", columns=["TPSA", "TPSA"],
                              fingerprint=False)
    with pytest.warns(UserWarning, match="repeated"):
        assert molhume.feature_names(columns=["TPSA", "TPSA"], fingerprint=False) == ("TPSA",)
    assert X.shape[1] == 1


def test_empty_selection_is_an_error():
    with pytest.raises(ValueError, match="no columns"):
        molhume.featurize(SMIS, standardize="none", columns=[])


# ------------------------------------------------- additional_descriptors

def test_additional_off_removes_exactly_our_own_columns():
    from molhume._additional import ADDITIONAL_COLUMNS
    on = molhume.feature_names()
    off = molhume.feature_names(additional_descriptors=False)
    ours = set(ADDITIONAL_COLUMNS) & set(on)
    assert set(on) - set(off) == ours
    assert len(off) < len(on) and ours, "there must be some of our own columns to remove"


def test_additional_off_does_not_change_the_columns_it_keeps():
    a = _quiet(SMIS, standardize="none")
    b = _quiet(SMIS, standardize="none", additional_descriptors=False)
    on, off = molhume.feature_names(), molhume.feature_names(additional_descriptors=False)
    idx = [on.index(c) for c in off]
    assert np.array_equal(a[:, idx], b, equal_nan=True)


def test_the_two_filters_are_combined_with_and():
    from molhume._additional import ADDITIONAL_COLUMNS
    mine = next(c for c in ADDITIONAL_COLUMNS if c in molhume.ALL_COLUMNS)
    with pytest.warns(UserWarning, match="additional_descriptors=False removed"):
        X = molhume.featurize(SMIS, standardize="none", fingerprint=False,
                              additional_descriptors=False, columns=["TPSA", mine])
    assert X.shape[1] == 1


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_is_on_by_default_and_the_bits_go_last():
    on = _quiet(SMIS, standardize="none")
    off = _quiet(SMIS, standardize="none", fingerprint=False)
    assert off.shape[1] == len(molhume.ALL_COLUMNS)
    assert on.shape[1] == off.shape[1] + 2048, "default output is descriptors + 2048 ECFP bits"
    assert np.array_equal(on[:, :off.shape[1]], off, equal_nan=True), (
        "fingerprint bits must go LAST, so descriptor column indices do not shift with the flag")
    assert set(np.unique(on[:, off.shape[1]:])) <= {0.0, 1.0}
    names = molhume.feature_names()
    assert len(names) == on.shape[1]
    assert names[off.shape[1]] == "ECFP_0" and names[-1] == "ECFP_2047"


@pytest.mark.parametrize("size", [512, 2048])
@pytest.mark.parametrize("radius", [2, 3])
def test_fp_radius_and_size_reach_the_fingerprint(size, radius):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    n_desc = len(molhume.ALL_COLUMNS)
    X = _quiet(SMIS, standardize="none", fingerprint=True, fp_radius=radius, fp_size=size)
    assert X.shape == (len(SMIS), n_desc + size)
    gen = rfg.GetMorganGenerator(radius=radius, fpSize=size)
    for i, smi in enumerate(SMIS):
        want = np.array(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi)))
        assert np.array_equal(X[i, n_desc:].astype(bool), want.astype(bool)), f"row {i}"


# ---------------------------------------------------------------- package shape

def test_the_alias_package_is_the_same_module():
    import mol_hume
    assert mol_hume.featurize is molhume.featurize
    assert mol_hume.ALL_COLUMNS is molhume.ALL_COLUMNS


def test_all_columns_is_unique_and_matches_the_output_width():
    cols = molhume.ALL_COLUMNS
    assert len(set(cols)) == len(cols), "ALL_COLUMNS contains a duplicate name"
    X = _quiet(["CCO"], standardize="none", fingerprint=False)
    assert X.shape[1] == len(cols) == len(molhume.feature_names(fingerprint=False))


def test_featurize_is_exported():
    assert "featurize" in molhume.__all__ and "feature_names" in molhume.__all__


def test_featurize_returns_a_bare_array_not_a_tuple():
    X = _quiet(SMIS, standardize="none")
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert X.shape == (len(SMIS), len(molhume.feature_names()))


@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_dtype_is_honored(dt):
    X = _quiet(SMIS, standardize="none", dtype=dt)
    assert X.dtype == dt
    ref = _quiet(SMIS, standardize="none")
    assert np.allclose(X, ref, equal_nan=True, rtol=1e-6)


def test_mol_objects_are_accepted_without_reparsing():
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in SMIS]
    assert np.array_equal(_quiet(mols, standardize="none"),
                          _quiet(SMIS, standardize="none"), equal_nan=True)


def test_a_non_molecule_type_is_rejected_by_name():
    with pytest.raises(TypeError, match="item 0 is a float"):
        molhume.featurize([1.5], standardize="none")
