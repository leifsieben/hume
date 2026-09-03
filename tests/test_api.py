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
        X = molhume.featurize([BAD, BAD], standardize="none", fingerprint=False,
                              columns="full")
    assert X.shape == (2, len(molhume.column_set("full"))) and np.all(np.isnan(X))


def test_empty_input():
    n = len(molhume.column_set("full"))
    X = molhume.featurize([], standardize="none", fingerprint=False, columns="full")
    assert X.shape == (0, n)
    assert molhume.featurize([], standardize="none", columns="full").shape == (0, n + 2048)


# ---------------------------------------------------------------- columns

def test_columns_selects_and_orders():
    want = ["TPSA", "ExactMolWt", "SLogP"]
    X = _quiet(SMIS, standardize="none", columns=want, fingerprint=False)
    assert molhume.feature_names(columns=want, fingerprint=False) == tuple(want), (
        "the caller's order is the output order")
    full = _quiet(SMIS, standardize="none", fingerprint=False, columns="full")
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
    with pytest.raises(ValueError, match="empty sequence"):
        molhume.featurize(SMIS, standardize="none", columns=[])


# ------------------------------------------------------------- column sets

def test_the_three_sets_are_nested_and_the_sizes_are_the_documented_ones():
    mn, nn, full = (molhume.column_set(k) for k in ("minimal", "full_no_new", "full"))
    assert (len(mn), len(nn), len(full)) == (622, 1109, 1269)
    assert set(mn) <= set(full) and set(nn) <= set(full)
    # ALL_COLUMNS is one WIDER than `full`: it also carries the opt-in `qed`.
    assert set(full) < set(molhume.ALL_COLUMNS)
    assert set(molhume.ALL_COLUMNS) - set(full) == set(molhume.OPTIONAL_COLUMNS)


def test_full_no_new_removes_exactly_our_own_columns():
    from molhume._additional import ADDITIONAL_COLUMNS
    on, off = molhume.column_set("full"), molhume.column_set("full_no_new")
    ours = set(ADDITIONAL_COLUMNS) & set(on)
    assert set(on) - set(off) == ours
    assert len(off) < len(on) and ours, "there must be some of our own columns to remove"


@pytest.mark.parametrize("name", ["minimal", "full_no_new"])
def test_a_set_does_not_change_the_columns_it_keeps(name):
    """The gated run must agree with the ungated one on every column it emits."""
    a = _quiet(SMIS, standardize="none", columns="full", fingerprint=False)
    b = _quiet(SMIS, standardize="none", columns=name, fingerprint=False)
    idx = [molhume.ALL_COLUMNS.index(c) for c in molhume.column_set(name)]
    assert np.array_equal(a[:, idx], b, equal_nan=True)


def test_minimal_is_the_default():
    assert np.array_equal(_quiet(SMIS, standardize="none", fingerprint=False),
                          _quiet(SMIS, standardize="none", fingerprint=False, columns="minimal"),
                          equal_nan=True)


def test_an_unknown_set_name_lists_the_three():
    with pytest.raises(ValueError, match="minimal.*full_no_new.*full"):
        molhume.column_set("small")
    with pytest.raises(ValueError, match="minimal.*full_no_new.*full"):
        molhume.featurize(SMIS, standardize="none", columns="tiny")


def test_columns_none_is_an_error_rather_than_meaning_everything():
    """It used to mean 'all of them'. Once the default changed, silence would be a wrong answer."""
    with pytest.raises(ValueError, match="columns='full'"):
        molhume.featurize(SMIS, standardize="none", columns=None)


@pytest.mark.parametrize("kw", ["additional_descriptors", "optional"])
def test_removed_keywords_say_what_replaced_them(kw):
    with pytest.raises(TypeError, match="0.7.0"):
        molhume.featurize(SMIS, standardize="none", **{kw: False})


def test_ipc_is_named_as_unreachable_rather_than_just_unknown():
    """Computable, not emitted, and worth saying so -- qed was here too until 0.8.0 appended it."""
    with pytest.raises(ValueError, match="deduplication dropped their slots"):
        molhume.featurize(SMIS, standardize="none", columns=["TPSA", "Ipc"])


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_is_on_by_default_and_the_bits_go_last():
    on = _quiet(SMIS, standardize="none")
    off = _quiet(SMIS, standardize="none", fingerprint=False)
    assert off.shape[1] == len(molhume.column_set("minimal")), (
        "the default descriptor block is the minimal set")
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
    n_desc = len(molhume.column_set("minimal"))
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
    every = list(molhume.column_set("full", extra=molhume.OPTIONAL_COLUMNS))
    X = _quiet(["CCO"], standardize="none", fingerprint=False, columns=every)
    assert X.shape[1] == len(cols) == len(every) == len(
        molhume.feature_names(fingerprint=False, columns=every))


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


# ---------------------------------------------------------------- family offsets

def test_family_offsets_index_the_emitted_layout():
    """FAMILY_OFFSETS must address ALL_COLUMNS, not the pre-dedup 1,539-column row.

    Shipped wrong in 0.1.0: the raw offsets were exported unchanged, so
    ALL_COLUMNS[FAMILY_OFFSETS["ringcount"]] was `ATS2Z` -- a column from a different family,
    with nothing to signal it. Found by a user slicing by family, not by a test. Hence this test.
    """
    fo = molhume.FAMILY_OFFSETS
    spans = sorted(fo.values())
    assert spans[0][0] == 0, "families must start at column 0"
    assert spans[-1][1] == len(molhume.ALL_COLUMNS), "families must cover the last column"
    for (a, b), (c, _) in zip(spans, spans[1:]):
        assert a <= b == c, f"family spans must tile without gaps or overlap: {(a, b)} then {c}"


def test_family_offsets_land_on_the_right_columns():
    # a spot check per family, by a name that unambiguously belongs to it
    for family, name in [("ringcount", "n5Ring"), ("estate", "NsCH3"), ("autocorr", "ATS0c"),
                         ("chi", "AXp-7dv"), ("eta", "ETA_alpha"), ("spectral", "SpAbs_A"),
                         ("frag", "fr_C_O"), ("blocks", "BalabanJ")]:
        a, b = molhume.FAMILY_OFFSETS[family]
        assert name in molhume.ALL_COLUMNS[a:b], (
            f"{name!r} should be inside family {family!r} at [{a}:{b}], which actually holds "
            f"{molhume.ALL_COLUMNS[a:a + 3]}...")


def test_raw_offsets_are_kept_but_separate():
    assert max(v for v in molhume.RAW_FAMILY_OFFSETS.values() if isinstance(v, int)) > \
        len(molhume.ALL_COLUMNS), "RAW_FAMILY_OFFSETS describes the wider pre-dedup row"


# ---------------------------------------------------------------- the minimal spec

def test_minimal_columns_is_a_subset_of_what_is_emitted():
    c = molhume.minimal_columns()
    assert len(c) == 622
    assert set(c) <= set(molhume.ALL_COLUMNS)
    assert len(set(c)) == len(c), "the spec contains a duplicate name"


def test_minimal_columns_actually_selects():
    X = _quiet(SMIS, standardize="none", columns=list(molhume.minimal_columns()),
               fingerprint=False)
    assert X.shape == (len(SMIS), 622)


def test_minimal_v1_is_withdrawn_with_a_reason():
    """v1 was an ordering on a linear-recoverability criterion no consumer satisfies."""
    with pytest.raises(ValueError, match="withdrawn"):
        molhume.minimal_columns(spec="minimal-v1")
    with pytest.raises(ValueError, match="minimal-v2"):
        molhume.minimal_columns(spec="something-else")


def test_the_composition_fr_flags_stay_out():
    """13 fr_* duplicate counts the library already emits; keeping them would be double-counting."""
    c = set(molhume.minimal_columns())
    for dup, because in (("fr_halogen", "nF/nCl/nBr/nI/nX"), ("fr_Ar_N", "the SMARTS `n`"),
                         ("fr_bicyclic", "ring perception")):
        assert dup not in c, f"{dup} duplicates {because}"
    # and exactly those three -- 0.5.0 held out ten more that are functional groups, not
    # duplicates. nS counts sulfur and cannot separate sulfide from sulfone.
    assert len([x for x in molhume.ALL_COLUMNS if x.startswith("fr_")]) - \
len([x for x in c if x.startswith('fr_')]) == 3


def test_the_retired_v1_api_is_gone():
    for gone in ("minimal_curve", "minimal_recovery", "minimal_gated"):
        assert not hasattr(molhume, gone), (
            f"{gone} belonged to minimal-v1 and its colinearity framing; leaving it exported "
            "would offer users a coverage curve for a spec that no longer exists")


def test_the_core_descriptors_survive_the_spec():
    """A reduced spec that quietly loses MolWt or TPSA is a trap, not a reduction."""
    c = set(molhume.minimal_columns())
    for expected in ("ExactMolWt", "SLogP", "TPSA", "NumHDonors", "NumHAcceptors", "nRot",
                     "RingCount", "LabuteASA", "BalabanJ", "Kappa1", "chi0n", "Lipinski",
                     # curated assertions no structural descriptor derives -- see _minimal.py
                     "fr_Ndealkylation1", "fr_quatN", "fr_nitro", "fr_pyridine", "fr_Ar_OH",
                     "fr_sulfone", "fr_SH", "fr_nitrile"):
        assert expected in c, f"{expected} is missing from minimal-v2"


# ------------------------------------------------------------------- qed, the opt-in column

def test_qed_is_emittable_by_name_and_is_last():
    assert molhume.ALL_COLUMNS[-1] == "qed", "qed is appended, so nothing else moves"
    assert molhume.OPTIONAL_COLUMNS == ("qed",)


@pytest.mark.parametrize("name", ["minimal", "full_no_new", "full"])
def test_qed_is_in_no_named_set(name):
    """`full` means every descriptor, not every possible expense -- qed is 69.3 us/mol."""
    assert "qed" not in molhume.column_set(name)


def test_extra_opts_qed_in():
    full = molhume.column_set("full")
    assert molhume.column_set("full", extra=["qed"]) == full + ("qed",)
    assert len(molhume.column_set("minimal", extra=["qed"])) == 622 + 1


def test_extra_refuses_an_ordinary_column():
    with pytest.raises(ValueError, match="not optional"):
        molhume.column_set("minimal", extra=["TPSA"])


def test_qed_matches_rdkit_exactly():
    from rdkit import Chem
    from rdkit.Chem import QED
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1N(C)C", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            "OC(=O)c1ccccc1O", "CCN(CC)CCNC(=O)c1ccc(N)cc1", "Clc1ccccc1C(=O)Nc1ccccc1"]
    got = molhume.featurize(smis, columns=["qed"], standardize="none", fingerprint=False)[:, 0]
    want = np.array([QED.qed(Chem.MolFromSmiles(s)) for s in smis])
    # 1 ULP, not bit-exact. QED is a weighted geometric mean -- eight exp/log terms -- and the
    # C++ associates them in its own order, so the last bit is not reproducible against Python's
    # and pretending otherwise would make this test a tripwire for the optimiser rather than for
    # the descriptor. Measured max |delta| over these seven: 5.6e-17.
    assert np.allclose(got, want, rtol=0, atol=1e-15), (
        f"qed differs from rdkit.Chem.QED.qed by more than 1 ULP: "
        f"max |delta| {np.max(np.abs(got - want)):.3e}")


def test_appending_qed_moved_no_other_column():
    """The whole reason it is appended. Every other column keeps the index it had in 0.7.0."""
    full = molhume.column_set("full")
    assert full == molhume.ALL_COLUMNS[:len(full)], (
        "appending qed was supposed to leave the 1,269 in place and it did not")
    assert len(full) == 1269


# ------------------------------------------------- one molecule must not kill a batch

#: An unbranched chain long enough that two fragment patterns exceed 1000 raw embeddings.
#: Until 0.9.0 this called std::abort() inside the fragment matcher -- a hard process kill,
#: uncatchable from Python, which made on_error="nan" a promise the library could not keep.
#: Reported from a 35M-molecule run where a single molecule ended the job.
POISON = "C" * 600

#: What it costs now. `fr_unbrch_alkane` and `NumRotatableBonds` have no well-defined count past
#: RDKit's truncation bound; `RotRatio`, `nRot` and `qed` are the columns that read one of them.
POISON_LOST = {"fr_unbrch_alkane", "RotRatio", "nRot", "qed"}


def _cols_lost(X_row, reference_row, names):
    """Columns NaN in `X_row` that are NOT NaN in a clean molecule of similar size.

    Comparing against a reference matters: plenty of descriptors are legitimately NaN on a long
    chain, and asserting "no NaN" would pass or fail for the wrong reason.
    """
    return {names[i] for i in np.where(np.isnan(X_row) & ~np.isnan(reference_row))[0]}


def test_the_poison_molecule_does_not_kill_the_process():
    """If this regresses the test process dies, so a failure here is not a normal assertion."""
    with pytest.warns(UserWarning, match="could not be computed"):
        X = molhume.featurize([POISON], standardize="none", fingerprint=False, on_error="nan")
    assert X.shape == (1, 622)


def test_it_costs_four_columns_and_not_the_row():
    cols = list(molhume.column_set("full", extra=["qed"]))
    names = molhume.feature_names(columns=cols, fingerprint=False)
    with pytest.warns(UserWarning, match="could not be computed"):
        X = molhume.featurize([POISON], columns=cols, standardize="none", fingerprint=False)
    ref = _quiet(["C" * 300], columns=cols, standardize="none", fingerprint=False)[0]
    assert _cols_lost(X[0], ref, names) == POISON_LOST
    assert np.isfinite(X[0]).sum() > 1200, "the rest of the row must survive"


def test_the_warning_names_the_patterns_and_the_molecule():
    with pytest.warns(UserWarning) as rec:
        molhume.featurize(["CCO", POISON], standardize="none", fingerprint=False)
    msg = "\n".join(str(w.message) for w in rec)
    assert "fr_unbrch_alkane" in msg and "index 1" in msg, msg
    assert "1000-embedding" in msg, "the message must say why the count is not well defined"


def test_a_degraded_molecule_does_not_touch_its_neighbours():
    smis = ["CCO", POISON, "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]
    with pytest.warns(UserWarning):
        X = molhume.featurize(smis, standardize="none", fingerprint=False)
    clean = _quiet([s for i, s in enumerate(smis) if i != 1], standardize="none",
                   fingerprint=False)
    assert np.array_equal(X[[0, 2, 3]], clean, equal_nan=True), (
        "the other rows must be bit-identical to the same molecules featurized without it")


@pytest.mark.parametrize("threads", [1, 0])
def test_the_blast_radius_is_one_row_not_one_shard(threads):
    """The catch used to be per WORKER, so one molecule discarded that thread's whole chunk."""
    smis = ["CCO"] * 40 + [POISON] + ["c1ccccc1O"] * 40
    with pytest.warns(UserWarning):
        X = molhume.featurize(smis, standardize="none", fingerprint=False, threads=threads,
                              batch_size=16)
    clean_lo = _quiet(["CCO"], standardize="none", fingerprint=False)[0]
    clean_hi = _quiet(["c1ccccc1O"], standardize="none", fingerprint=False)[0]
    for i in range(40):
        assert np.array_equal(X[i], clean_lo, equal_nan=True), f"row {i} was collateral damage"
    for i in range(41, 81):
        assert np.array_equal(X[i], clean_hi, equal_nan=True), f"row {i} was collateral damage"


def test_verification_callers_still_get_the_exception():
    """`errors_out=None` must keep raising: there an uncountable pattern IS the finding."""
    from molhume import _core
    from molhume._extract import extract_pickles
    from rdkit import Chem
    p = extract_pickles([Chem.MolFromSmiles(POISON)], stereo=False)
    with pytest.raises(RuntimeError, match="1000 raw embeddings"):
        _core.all_from_pickles(p.blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at,
                               p.h_blobs, p.stereo_a, p.stereo_b, errors_out=None, threads=1)


def test_a_corrupt_pickle_loses_its_row_and_only_its_row():
    """A row failure, as opposed to a column one: there is nothing to salvage from this molecule."""
    from molhume import _core
    from molhume._extract import extract_pickles
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1", "CCN")]
    p = extract_pickles(mols, stereo=False)
    blobs = list(p.blobs)
    blobs[1] = blobs[1][:12] + b"\xff" * 8 + blobs[1][20:]      # corrupt the middle molecule
    errs = []
    X = _core.all_from_pickles(blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at,
                               p.h_blobs, p.stereo_a, p.stereo_b, errors_out=errs, threads=1)
    assert errs and errs[0][0] == 1 and errs[0][1] == "row", errs
    assert np.all(np.isnan(X[1]))
    assert not np.all(np.isnan(X[0])) and not np.all(np.isnan(X[2]))


def test_on_error_branches_on_a_row_failure(monkeypatch):
    """The three on_error paths, driven by an injected row failure.

    Injected rather than found: the reproducible molecule costs COLUMNS, not its row, and the
    row-failure path is exercised for real by the corrupt-pickle test above. This checks the
    Python branching, which is the part that decides what the caller sees.
    """
    real = molhume.featurize_all_from_mols

    def fake(mols, **kw):
        errs = kw.get("errors_out")
        fp, X, names = real(mols, **{k: v for k, v in kw.items() if k != "errors_out"})
        if errs is not None and len(mols) > 1:
            X = X.copy(); X[1] = np.nan
            errs.append((1, "row", "injected failure"))
        return fp, X, names

    monkeypatch.setattr(molhume, "featurize_all_from_mols", fake)
    smis = ["CCO", "c1ccccc1", "CCN"]
    with pytest.warns(UserWarning, match="still aligns"):
        X = molhume.featurize(smis, standardize="none", fingerprint=False, on_error="nan")
    assert X.shape == (3, 622) and np.all(np.isnan(X[1]))
    with pytest.warns(UserWarning, match="no longer aligns"):
        X = molhume.featurize(smis, standardize="none", fingerprint=False, on_error="skip")
    assert X.shape == (2, 622)
    with pytest.raises(RuntimeError, match="index 1"):
        molhume.featurize(smis, standardize="none", fingerprint=False, on_error="raise")


# ------------------------------------------------- the empty molecule

def test_empty_smiles_does_not_segfault():
    """`Chem.MolFromSmiles("")` returns an empty Mol, NOT None, so it passed every parse check.

    It then reached code that indexes atom 0 unconditionally and took the process down with a
    SIGSEGV -- not an exception, not a NaN row. If this regresses the test process dies, so a
    failure here will not look like a normal assertion.
    """
    with pytest.warns(UserWarning, match="did not parse"):
        X = molhume.featurize([""], standardize="none", fingerprint=False)
    assert X.shape == (1, 622) and np.all(np.isnan(X))


def test_an_empty_smiles_costs_only_its_own_row():
    with pytest.warns(UserWarning, match="did not parse"):
        X = molhume.featurize(["", "CCO", ""], standardize="none", fingerprint=False)
    assert X.shape == (3, 622)
    assert np.all(np.isnan(X[0])) and np.all(np.isnan(X[2]))
    assert np.array_equal(X[[1]], _quiet(["CCO"], standardize="none", fingerprint=False),
                          equal_nan=True)


def test_empty_is_the_same_kind_of_failure_as_unparseable():
    """Both are bad input, so both raise ValueError -- not one ValueError and one RuntimeError."""
    with pytest.raises(ValueError, match="no atoms"):
        molhume.featurize(["", "CCO"], standardize="none", on_error="raise")
    with pytest.raises(ValueError, match="could not parse"):
        molhume.featurize(["@@@", "CCO"], standardize="none", on_error="raise")
    with pytest.warns(UserWarning):
        assert molhume.featurize(["", "CCO"], standardize="none", fingerprint=False,
                                 on_error="skip").shape == (1, 622)


def test_a_zero_atom_mol_object_is_caught_too():
    """`featurize` takes Mol objects directly, so the SMILES check alone would not cover it."""
    from rdkit import Chem
    with pytest.warns(UserWarning, match="did not parse"):
        X = molhume.featurize([Chem.Mol(), Chem.MolFromSmiles("CCO")], standardize="none",
                              fingerprint=False)
    assert np.all(np.isnan(X[0])) and not np.all(np.isnan(X[1]))


@pytest.mark.parametrize("entry", ["blocks_from_pickles", "all_from_pickles"])
def test_the_extension_raises_rather_than_crashing_on_zero_atoms(entry):
    """The guard is in the C++ too: a direct caller must get an exception, not a signal."""
    from rdkit import Chem
    from molhume import _core
    from molhume._extract import extract_pickles
    p = extract_pickles([Chem.Mol()], stereo=False)
    with pytest.raises(RuntimeError, match="no atoms"):
        if entry == "blocks_from_pickles":
            _core.blocks_from_pickles(p.blobs)
        else:
            _core.all_from_pickles(p.blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at,
                                   p.h_blobs, p.stereo_a, p.stereo_b, threads=1)


# ------------------------------------------------- BalabanJ on disconnected molecules

#: mu + 1 == 0 means bonds == atoms - 2: a disconnected, acyclic molecule. RDKit guards that case
#: and returns 0; HUME divided by zero and emitted +inf until 0.9.2. It never fired in
#: verification because the exactness corpus is standardised and desalted -- 0 of 300,000 curated
#: molecules reach it -- and it fires on any salt or solvate with an acyclic parent, which is
#: most of them. Found by fuzzing 1.22M adversarial SMILES.
DISCONNECTED = ["CC.CI", "CC.C", "CCO.[Na+]", "CC(=O)[O-].[Na+]", "CN.Cl", "CCO.O",
                "CC(=O)O.CC(=O)O", "C.C", "c1ccccc1.CC", "c1ccccc1.c1ccccc1"]


@pytest.mark.parametrize("smi", DISCONNECTED)
def test_balaban_j_matches_rdkit_on_disconnected_molecules(smi):
    from rdkit import Chem
    from rdkit.Chem import GraphDescriptors
    m = Chem.MolFromSmiles(smi)
    got = _quiet([m], columns=["BalabanJ"], standardize="none", fingerprint=False)[0, 0]
    want = GraphDescriptors.BalabanJ(m)
    assert np.isfinite(got), f"BalabanJ is {got} on {smi!r}; an inf breaks whatever consumes it"
    assert got == pytest.approx(want, rel=1e-12, abs=0), f"{got} != rdkit's {want}"


def test_no_column_is_infinite_on_a_salt():
    """A whole-row check, because the fuzz run found this by scanning columns rather than one."""
    cols = list(molhume.column_set("full", extra=["qed"]))
    names = molhume.feature_names(columns=cols, fingerprint=False)
    X = _quiet(["CC(=O)[O-].[Na+]", "CN.Cl", "CCO.O"], columns=cols, standardize="none",
               fingerprint=False)
    bad = sorted({names[j] for j in np.where(np.isinf(X))[1]})
    assert not bad, f"infinite values on an ordinary salt: {bad}"
