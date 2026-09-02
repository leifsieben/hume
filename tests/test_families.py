"""COMPUTE GATING MUST NOT CHANGE A SINGLE EMITTED VALUE.

Since 0.7.0 `featurize(columns=...)` decides what the extension COMPUTES, not just what it
returns: a descriptor family none of whose columns were asked for is skipped. That is a real
speedup and it is also the most dangerous change in the package, because a family left out
writes ZEROS into its slots rather than NaN -- so a missed dependency is a finite, plausible,
wrong descriptor with no symptom.

It was not hypothetical. `F_NEEDS_H` in bindings.cpp listed the families needing the H-added
Gasteiger charges as `F_AC | F_CONSTIT` and omitted `F_SPECTRAL`, so gated runs produced wrong
BCUTc-1h / BCUTc-1l -- 650 differing cells over 400 molecules. It had never mattered, because
`families` had exactly one caller (cpp/bench_e2e.py) and the only consequence there was an
understated timing. Deriving the mask from a column selection is what made it reachable.

So this file does not review the dependency list. It compares gated output against ungated
output, cell for cell, on molecules chosen to be hard.
"""
import numpy as np
import pytest

import molhume
from molhume import _core
from molhume._extract import extract_pickles

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _mols(n=250):
    from rdkit import Chem
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "cpp" / "hard.smi"
    if src.exists():
        smis = [l.split()[0] for l in src.read_text().splitlines() if l.strip()][:n]
    else:
        smis = [l.strip() for l in
                (pathlib.Path(__file__).parent / "data" / "fixture_smiles.txt").read_text().split()]
    out = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    assert out, "no molecules parsed for the gating test"
    return out


@pytest.fixture(scope="module")
def gated():
    mols = _mols()
    p = extract_pickles(mols, stereo=False)

    def run(families=None, optional=("AvgIpc",), select=None):
        return _core.all_from_pickles(
            p.blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at, p.h_blobs,
            p.stereo_a, p.stereo_b, families=families, optional=optional, select=select,
            threads=0)

    return run, run(None), len(mols)


def _differing(a, b):
    """Cells that differ, counting NaN == NaN as equal. Exact: gating must not perturb a bit."""
    return int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())


def test_family_spans_tile_the_emitted_row():
    """Every column belongs to exactly one family -- what makes column -> family total."""
    cov = np.zeros(len(molhume.ALL_COLUMNS), dtype=int)
    for name, (a, b) in molhume.FAMILY_OFFSETS.items():
        cov[a:b] += 1
    assert cov.min() == 1 and cov.max() == 1, (
        f"{int((cov == 0).sum())} emitted columns belong to no family and "
        f"{int((cov > 1).sum())} belong to more than one; _COLUMN_FAMILY cannot be built from "
        "FAMILY_OFFSETS if the spans do not tile ALL_COLUMNS exactly.")


@pytest.mark.parametrize("family", sorted(
    f for f, (a, b) in molhume.FAMILY_OFFSETS.items() if b > a))
def test_family_alone_matches_ungated(family, gated):
    """Each family, computed with only its own dependencies, must equal the full run."""
    run, full, n = gated
    a, b = molhume.FAMILY_OFFSETS[family]
    got = run([family])
    bad = _differing(got[:, a:b], full[:, a:b])
    if bad:
        cols = sorted({molhume.ALL_COLUMNS[a + j]
                       for j in np.where(got[:, a:b] != full[:, a:b])[1]})
        pytest.fail(
            f"family '{family}' computed alone differs from the full run in {bad} cells over {n} "
            f"molecules, in {cols[:8]}. It reads another family's output out of the row, and "
            f"family_mask() in bindings.cpp does not force that family on -- so a gated run "
            f"reduces over zeros and returns a plausible wrong number.")


@pytest.mark.parametrize("name", ["minimal", "full_no_new", "full"])
def test_column_set_gated_matches_ungated(name, gated):
    """The three predefined sets, through the real compute plan."""
    run, full, n = gated
    idx, names = molhume._resolve_columns(name)
    families, optional, select = molhume._compute_plan(idx)
    got = run(families, optional, select)
    take = np.arange(len(molhume.ALL_COLUMNS)) if idx is None else idx
    bad = _differing(got[:, take], full[:, take])
    assert bad == 0, f"columns='{name}' gated differs from ungated in {bad} cells over {n} mols"


def test_manual_selection_gated_matches_ungated(gated):
    """A hand-picked list spanning families the sets treat differently."""
    run, full, n = gated
    sel = ["TPSA", "AvgIpc", "ExactMolWt", "BCUTc-1h", "ATS0Z", "ETA_alpha", "IC3", "naRing"]
    sel = [c for c in sel if c in molhume.ALL_COLUMNS]
    idx, _ = molhume._resolve_columns(sel)
    families, optional, select = molhume._compute_plan(idx)
    bad = _differing(run(families, optional, select)[:, idx], full[:, idx])
    assert bad == 0, f"manual selection {sel} gated differs in {bad} cells over {n} molecules"


def test_avgipc_is_computed_only_when_selected():
    """The one optional column left, and the whole reason `optional=` could be retired."""
    i = molhume.ALL_COLUMNS.index("AvgIpc")
    idx, _ = molhume._resolve_columns(["AvgIpc", "TPSA"])
    assert molhume._compute_plan(idx)[1] == ("AvgIpc",)
    idx, _ = molhume._resolve_columns(["TPSA"])
    assert molhume._compute_plan(idx)[1] == ()
    assert molhume._compute_plan(None)[1] == ("AvgIpc",)
    assert "AvgIpc" in molhume.column_set("minimal"), (
        "AvgIpc left the minimal set, so the default call no longer pays its 64.6 us/mol -- "
        "which is a fine outcome, but this test and the featurize docstring both claim the "
        "opposite and need updating.")
    assert i >= 0


def test_gating_actually_skips_something():
    """A guard against the plan silently degenerating to 'compute everything'."""
    idx, _ = molhume._resolve_columns("minimal")
    fams, _, _ = molhume._compute_plan(idx)
    skipped = sorted(set(molhume.FAMILY_OFFSETS) - set(fams))
    assert skipped, (
        "columns='minimal' asks for at least one column of every family, so nothing is skipped "
        "and the compute plan buys nothing. That is possible and not wrong -- but it means the "
        "0.7.0 speedup claim no longer holds and the changelog needs correcting.")


@pytest.mark.parametrize("sel", [
    ["BCUTc-1h", "BCUTc-1l"],                       # one Burden weight, nothing else
    ["SpAbs_A", "SpMax_A", "SpDiam_A", "SpMAD_A"],  # adjacency only
    ["SpAbs_DzZ", "SM1_Dzp"],                       # two Barysz matrices
    ["SpAbs_D", "SpDiam_D", "SpMAD_D"],             # the distance matrix only
])
def test_spectral_within_family_gating_matches_ungated(sel, gated):
    """spectral is the one family gated per COLUMN, because it is four independent eigensolves.

    Each parametrisation names slots from one section, so a leak between sections -- a matrix
    built for one and read by another -- shows up as a NaN or a changed value here.
    """
    run, full, n = gated
    sel = [c for c in sel if c in molhume.ALL_COLUMNS]
    assert sel, "the spectral column names in this test no longer exist"
    idx, _ = molhume._resolve_columns(sel)
    families, optional, select = molhume._compute_plan(idx)
    got = run(families, optional, select)[:, idx]
    assert _differing(got, full[:, idx]) == 0, (
        f"spectral columns {sel} change when the other sections are skipped")
    assert not np.all(np.isnan(got)), f"{sel} came back all-NaN, so the gate skipped too much"


def test_unwanted_spectral_slots_are_nan_or_right_never_wrong(gated):
    """A skipped slot keeps its NaN. It never gets a zero, and never gets a stale value.

    Asking for one BCUT half necessarily produces the other -- one eigensolve returns both
    extremes -- so "unwanted" does not mean "absent", and the invariant that matters is not
    "all NaN" but "NaN or correct". A zero would satisfy neither, and a zero is what the FAMILY
    mask writes, which is the difference between the two mechanisms.
    """
    run, full, n = gated
    idx, _ = molhume._resolve_columns(["BCUTc-1h"])
    families, optional, select = molhume._compute_plan(idx)
    got = run(families, optional, select)
    a, b = molhume.FAMILY_OFFSETS["spectral"]
    g, f = got[:, a:b], full[:, a:b]
    wrong = np.where(~np.isnan(g) & (g != f) & ~np.isnan(f))
    assert len(wrong[0]) == 0, (
        f"{len(wrong[0])} skipped spectral cells came back finite and WRONG, in "
        f"{sorted({molhume.ALL_COLUMNS[a + j] for j in wrong[1]})[:6]}")
    assert np.isnan(g).any(), "nothing was skipped at all, so this test proves nothing"
    free = sorted({molhume.ALL_COLUMNS[a + j] for j in np.where(~np.isnan(g).all(axis=0))[0]})
    assert "BCUTc-1l" in free, (
        "BCUTc-1l used to come back for free with BCUTc-1h, because one eigensolve returns both "
        "extremes. If it no longer does, the Burden loop was restructured and the cost model in "
        "the 0.7.0 changelog needs re-measuring.")
