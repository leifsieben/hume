"""HUME -- the bridge to the verified C++ blocks.

MILESTONE 1 ONLY. This exposes `featurize_blocks`, which returns the 182 columns that have a C++
implementation *and* an oracle: they are ALL EXACT against RDKit and against HUME's own Python
modules on 98,905 molecules, and cpp/values_hume.txt is the file that says so. The other 683
columns of the full descriptor set have no C++ at all yet, and the public `featurize()` of API.md
is deliberately not here -- a bridge that carries 182 verified columns is worth more than one
that carries 865 unverified ones.

    >>> import molhume
    >>> X, cols = hume.featurize_blocks(["CCO", "c1ccccc1"])
    >>> X.shape
    (2, 182)

There is no text file anywhere in this path. RDKit molecules become flat numpy arrays
(`_extract.py`), the arrays cross the boundary once per batch, and the C++ writes float64
straight into the output array.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import warnings

import numpy as np

from . import _core
from ._columns import COLUMNS, N_COLS
from ._extract import Batch, extract, extract_pickles

__all__ = [
    # the public API
    "featurize", "feature_names", "ALL_COLUMNS", "N_ALL_COLS",
    "minimal_columns", "minimal_curve",
    # lower level: the (fp, X, names) form, and the 178-column block on its own
    "featurize_all", "featurize_all_from_mols", "featurize_blocks", "featurize_blocks_from_mols",
    "COLUMNS", "N_COLS", "FAMILY_OFFSETS", "RAW_FAMILY_OFFSETS",
]

assert _core.N_COLS == N_COLS, (
    f"extension emits {_core.N_COLS} columns, _columns.py names {N_COLS}"
)

# EVERY COLUMN THAT HAS C++ TODAY, in the order the extension emits it. The first 182 are named
# by _columns.py (generated from the same modules cpp/verify_hume.py checks); the rest are named
# by the C++ that decides their order, so the package cannot disagree with the extension about
# which number is which. FAMILY_OFFSETS says where each family starts.
# ONE FILTER, APPLIED ONCE, OVER BOTH HALVES. The C++ computes N_ROW_COLS columns and emits the
# subset `_core.EMIT_KEEP` names; the block half of that row is COLUMNS and the tail half is
# all_column_names_tail_full(). Building ALL_COLUMNS by indexing the concatenation with the same
# table is what keeps the names and the values in step -- filtering each half separately would be
# two chances to disagree, and the assertion below would only catch a length mismatch, not a
# misalignment.
_ALL_ROW_NAMES: tuple[str, ...] = COLUMNS + tuple(_core.all_column_names_tail_full())
ALL_COLUMNS: tuple[str, ...] = tuple(_ALL_ROW_NAMES[i] for i in _core.EMIT_KEEP)

# NAMED BUT NOT YET COMPUTED. A column here appears in ALL_COLUMNS and in the output array and
# is NaN on every molecule, because it is blocked on a boundary field that does not exist yet.
#
# IT IS EMPTY. Every column the package names now produces a value.
#
# `qed` WAS THE LAST ONE OUT AND IS NOT HERE ANY MORE. Its eighth property is ALERTS, the count
# of QED's 116 structural-alert SMARTS that match; the other seven were already exact in
# src/hume_core/constit.h. The alerts are compiled by cpp/gen_frag_program.py into
# cpp/qed_alert_program.h and matched by the SAME evaluator as the 74 `rdkit_core` fragment
# patterns -- src/hume_core/frag_matcher.h takes its program tables as a bound reference rather
# than reading one namespace's at global scope, so there is one subgraph-isomorphism
# implementation in this repo and not two. The alert set needed four query primitives the
# fragment set never exercised: isotope, `~`, `@` and component-level `.`. The isotope is the
# thirteenth `atom_i` column, and it is the third field molpickle.h was already decoding and
# throwing away.
#
# `SPS` CAME OUT JUST BEFORE IT. It needed RDKit's NEW stereo perception --
# FindPotentialStereo plus FindPotentialStereoBonds -- which is not in the pickle and is not
# derivable from the assigned-only `cip` / `bond_s` columns. It now crosses the boundary as two
# arrays that `_extract._potential_stereo` computes per molecule, at a measured 52 us/mol.
#
# ONE ADDITION DID NOT BUY THREE, and the note that used to stand here said it would.
# `NumAtomStereoCenters` / `NumUnspecifiedAtomStereoCenters` do NOT use FindPotentialStereo:
# Code/GraphMol/Descriptors/Lipinski.cpp counts atoms carrying `_ChiralityPossible`, which is the
# LEGACY `MolOps::assignStereochemistry(cleanIt, force, flagPossible)` perception. The two answer
# differently on 262 of 4,000 corpus molecules, so `SPS` and the two counts needed two different
# perceptions. Both are now wired, from two different boundary fields: the legacy pair out of the
# blob (`atom_i` columns 10 and 11, which molpickle.h had been skipping) and the new one out of
# the two arrays above.
#
# THE CONSTANT STAYS, EMPTY, AND IS NOT DELETED. "N of 864 have a name" and "N produce a value"
# are different claims, this project has already overstated its position twice by conflating
# them, and `covered = set(ALL_COLUMNS) - set(PENDING_COLUMNS)` is the expression every reporting
# path uses. Deleting the constant now would make the next column that lands NaN silently
# countable as covered.
#
# `AvgIpc` IS emitted now -- the Ipc bug is closed (it was RDKit's numbering-dependence, not
# ours; see the determinism evidence at the top of src/hume_core/infocontent.h) -- so it is a
# member of ALL_COLUMNS and NOT of PENDING_COLUMNS. `Ipc` and `Log2Ipc` are computed alongside it
# and deliberately not emitted: neither is one of the 865, and both are unbounded where every
# other column here is not. Three different states, deliberately kept distinct.
PENDING_COLUMNS: tuple[str, ...] = ()


# FOUR NAMES APPEAR TWICE IN ALL_COLUMNS, AND A NAIVE name->index MAP SILENTLY PICKS THE SECOND.
#
#   MaxEStateIndex  MinEStateIndex  MaxAbsEStateIndex  MinAbsEStateIndex
#
# They are emitted once by the 182-column block (from hume_blocks.h) and again by the VSA family
# (from vsa_bins.h). The two are INDEPENDENT computations of the same descriptor and differ in
# the last bit, so `{n: i for i, n in enumerate(ALL_COLUMNS)}` resolves to the second copy and a
# whole-matrix A/B against a stored dump reports thousands of molecules as moved. That cost an
# agent an hour of chasing a difference that was an indexing artifact, not a change.
#
# Deliberately NOT silently deduplicated here: dropping either copy shifts every column index
# above it, which is a schema change and belongs to the project owner rather than to an import
# hook. Until then this constant exists so the trap is discoverable, and DUPLICATE_COLUMNS is
# checked by the test suite rather than assumed empty.
#
# Note bindings.cpp already avoids exactly this for five other columns via an alias block that
# asserts a single emission -- the four below are the case that predates it.
DUPLICATE_COLUMNS: tuple[str, ...] = tuple(
    sorted({n for n in ALL_COLUMNS if ALL_COLUMNS.count(n) > 1}))
N_ALL_COLS: int = _core.N_ALL_COLS
# FAMILY OFFSETS, REMAPPED ONTO WHAT IS ACTUALLY EMITTED.
#
# `_core.ALL_OFFSETS` indexes the N_ROW_COLS (1,539) row the C++ computes, NOT the 1,269 columns
# `featurize` returns. Exporting it raw was a silent-wrong-answer bug: a user slicing
# `ALL_COLUMNS[FAMILY_OFFSETS["ringcount"]]` got `ATS2Z`, a column from a different family, with
# nothing to indicate anything was wrong. It was reported by a user, not by a test.
#
# The dedup removes columns but preserves ORDER, so every family is still a contiguous run in the
# emitted layout -- only the boundaries move. Remapping is therefore exact: for each raw offset,
# count how many kept columns lie before it.
#
# A family that the dedup emptied entirely maps to a zero-length span, which is the honest answer
# and is why the values are (start, stop) pairs rather than bare starts: a bare start cannot
# express "this family contributes nothing", and the caller would slice into the next family.
def _emitted_family_spans():
    keep = list(_core.EMIT_KEEP)
    raw = dict(_core.ALL_OFFSETS)
    # position in the emitted row for each raw index, plus the count of kept columns before it
    before = [0] * (len(_ALL_ROW_NAMES) + 1)
    kept = set(keep)
    n = 0
    for i in range(len(_ALL_ROW_NAMES)):
        before[i] = n
        if i in kept:
            n += 1
    before[len(_ALL_ROW_NAMES)] = n

    bounds = sorted((v, k) for k, v in raw.items() if isinstance(v, int) and k != "end")
    spans = {}
    for j, (start, name) in enumerate(bounds):
        stop = bounds[j + 1][0] if j + 1 < len(bounds) else raw.get("end", len(_ALL_ROW_NAMES))
        spans[name] = (before[start], before[min(stop, len(_ALL_ROW_NAMES))])
    return spans


#: ``{family: (start, stop)}`` into ALL_COLUMNS and into the descriptor block of `featurize`'s
#: output -- half-open, so ``ALL_COLUMNS[start:stop]`` is that family. Fingerprint bits are
#: appended after every family and are not covered here; use `feature_names` for those.
FAMILY_OFFSETS: dict = _emitted_family_spans()

#: The raw offsets into the 1,539-column row the extension computes internally, before the
#: deduplication filter. Almost nobody wants this; it is kept because `verify_*.py` and the
#: profiling tools reason about the pre-filter layout.
RAW_FAMILY_OFFSETS: dict = dict(_core.ALL_OFFSETS)

# `Phi` is computed in C++ as Kappa1 * Kappa2 / heavy-atom count, reading the two out of the
# block row rather than paying findAllPathsOfLengthN(mol, 2) a second time. The extension resolves
# those two by INDEX; the names live here, so this is the check that the index still points at the
# column it was written for. A reordered block tail fails at import naming what moved.
assert tuple(COLUMNS[i] for i in _core.BLOCK_KAPPA_COLS) == ("Kappa1", "Kappa2"), (
    f"the extension reads Phi's inputs from block columns {tuple(_core.BLOCK_KAPPA_COLS)}, "
    f"which _columns.py names {tuple(COLUMNS[i] for i in _core.BLOCK_KAPPA_COLS)}"
)

assert len(ALL_COLUMNS) == N_ALL_COLS, (
    f"extension emits {N_ALL_COLS} columns, {len(ALL_COLUMNS)} are named"
)


def featurize_blocks_from_mols(mols: Sequence, batch_size: int = 4096, reader: str = "pickle"):
    """The 182 verified columns for already-parsed RDKit molecules.

    Returns ``(X, COLUMNS)`` with ``X`` float64 of shape ``(len(mols), 182)``.

    `batch_size` bounds peak memory in the flat arrays, nothing else; the boundary crossing
    itself costs ~0.1 us, so the number is not performance-critical above a few hundred.

    `reader` picks how the molecule gets into C++, and the two agree BIT-FOR-BIT on all 182
    columns over both corpora -- that is the whole claim, see cpp/verify_molpickle.py:

      "pickle"  ``m.ToBinary()`` once per molecule, parsed by src/hume_core/molpickle.h. No
                per-atom Python call anywhere. The default, and roughly twice as fast.
      "api"     ``_extract.extract()``, which reads the molecule through RDKit's supported
                Python API. The reference implementation, and the one to use for molecules the
                pickle reader refuses -- query atoms, substance groups, anything unsanitised --
                or after an RDKit upgrade moves the pickle format.
    """
    mols = list(mols)
    if not mols:
        return np.zeros((0, N_COLS), dtype=np.float64), COLUMNS
    if reader not in ("pickle", "api"):
        raise ValueError(f"reader must be 'pickle' or 'api', not {reader!r}")
    out = np.empty((len(mols), N_COLS), dtype=np.float64)
    for lo in range(0, len(mols), batch_size):
        chunk = mols[lo:lo + batch_size]
        if reader == "pickle":
            # `stereo=False`: the 182 blocks read neither potential-stereo array, and running the
            # perception for them would add 52 us/mol -- a third of this path's whole boundary --
            # to fill two arrays nothing downstream looks at.
            out[lo:lo + len(chunk)] = _core.blocks_from_pickles(
                extract_pickles(chunk, stereo=False).blobs)
        else:
            out[lo:lo + len(chunk)] = compute(extract(chunk, stereo=False))
    return out, COLUMNS


def featurize_blocks(smiles: Iterable[str], batch_size: int = 4096, reader: str = "pickle"):
    """The 182 verified columns for SMILES.

    Unparseable SMILES raise rather than being dropped: every consumer of this matrix indexes by
    row, so a silently skipped molecule turns that index into a lie. Callers that want to tolerate
    bad input should parse themselves and use `featurize_blocks_from_mols`.
    """
    from rdkit import Chem

    mols = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"could not parse SMILES at index {i}: {s!r}")
        mols.append(m)
    return featurize_blocks_from_mols(mols, batch_size=batch_size, reader=reader)


# ------------------------------------------------------------------------------------------
# SMILES -> ECFP + every column that has C++, in one place
# ------------------------------------------------------------------------------------------

def featurize_all_from_mols(mols: Sequence, batch_size: int = 4096, fp_radius: int = 3,
                            threads: int = 0, fingerprint: bool = True,
                            fp_size: int = 2048, optional=None):
    """-> ``(fp, X, ALL_COLUMNS)``: the ECFP and every natively computed descriptor column.

    ``fp`` is ``(len(mols), fp_size)`` uint8 Morgan/ECFP with chirality; ``X`` is
    ``(len(mols), N_ALL_COLS)`` float64.

    ``optional`` NAMES THE EXPENSIVE COLUMNS TO COMPUTE, and it is the only knob here that
    changes what is in ``X`` rather than how it is produced. Two of the 864 cost more than most
    whole families -- ``qed`` 69.3 us/mol and ``AvgIpc`` 64.6, against 629.9 for all of them --
    so each can be declined:

        optional=None                default: `AvgIpc` on, `qed` off
        optional=()                  neither
        optional=("qed", "AvgIpc")   the full suite, and what the exactness claims are measured on

    A declined column is NaN in its usual position. THE SHAPE AND THE COLUMN LIST DO NOT MOVE:
    `ALL_COLUMNS`, `N_ALL_COLS` and every family offset are the same whichever way this is set,
    because a column set that shifts with a keyword argument is how two callers end up
    disagreeing about what a column index means.

    THE BOUNDARY IS THE PICKLE ONE: `m.ToBinary()` once per molecule plus the ring CSR, parsed
    in C++. `featurize_blocks(reader="api")` still exists as the reference implementation and the
    two agree bit-for-bit on the 182 they share; there is no `reader` here only because nothing
    has asked for one yet, not because the arrays could not be filled the other way.

    ALL TEN AUTOCORRELATION WEIGHTS ARE HERE NOW -- 540 columns, on the hydrogen-added molecule
    `extract_pickles` serialises alongside. The `Z` weight (bare atomic number) used to be the
    documented gap and closed the last 52 members of the 865 that Autocorrelation owed.
    `bench_e2e.py` counts how many of the 865 these columns actually cover and prints it next to
    the timing, so the ratio and the coverage are never read apart -- ALL_COLUMNS is the larger
    number and always has been.
    """
    from rdkit.Chem import rdFingerprintGenerator as rfg

    mols = list(mols)
    if not mols:
        return (np.zeros((0, fp_size), dtype=np.uint8),
                np.zeros((0, N_ALL_COLS), dtype=np.float64), ALL_COLUMNS)
    # THE FINGERPRINT IS AN OUTPUT, NOT A DESCRIPTOR, AND IT IS NOT FREE. GetFingerprintAsNumPy
    # is 28.1 us/mol at the corpus median -- 10% of the whole call -- and it holds the GIL, so it
    # cannot be threaded the way the descriptor block now is. A caller who wants descriptors only
    # was paying for it. `fingerprint=False` returns a (n, 0) array in its place.
    gen = (rfg.GetMorganGenerator(radius=fp_radius, fpSize=fp_size, includeChirality=True)
           if fingerprint else None)
    fp = np.empty((len(mols), fp_size if fingerprint else 0), dtype=np.uint8)
    X = np.empty((len(mols), N_ALL_COLS), dtype=np.float64)
    for lo in range(0, len(mols), batch_size):
        chunk = mols[lo:lo + batch_size]
        # `stereo=False` SINCE SPS MOVED TO C++. `_potential_stereo` existed for exactly one
        # column, and src/hume_core/sps.h now perceives potential stereo itself from boundary
        # fields the blobs already carry -- bit-identical to Chem.SpacialScore.SPS on 6,600
        # corpus molecules. Measured at the corpus median: extract_pickles 213.7 -> 156.5 us/mol.
        # The stereo_a / stereo_b arguments stay in the C++ signature and are passed empty;
        # nothing reads them any more.
        p = extract_pickles(chunk, stereo=False)
        # `threads=0` is one worker per hardware thread. The row loop is embarrassingly
        # parallel -- disjoint output slices, shared const inputs, and the scratch that lives at
        # namespace scope in hume_blocks.h is already `static thread_local`. Measured 7.71x on 8
        # performance cores, with the output bit-identical to the serial path (0 differing cells
        # of 3,682,060). Pass threads=1 if the caller is already parallel and would oversubscribe.
        X[lo:lo + len(chunk)] = _core.all_from_pickles(
            p.blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at, p.h_blobs,
            p.stereo_a, p.stereo_b, optional=optional, threads=threads)
        if fingerprint:
            for i, m in enumerate(chunk):
                fp[lo + i] = gen.GetFingerprintAsNumPy(m)
    return fp, X, ALL_COLUMNS


def featurize_all(smiles: Iterable[str], batch_size: int = 4096, fp_radius: int = 3,
                  fp_size: int = 2048, optional=None, threads: int = 0,
                  fingerprint: bool = True):
    """`featurize_all_from_mols` from SMILES. Unparseable input raises; see `featurize_blocks`."""
    from rdkit import Chem

    mols = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"could not parse SMILES at index {i}: {s!r}")
        mols.append(m)
    return featurize_all_from_mols(mols, batch_size=batch_size, fp_radius=fp_radius,
                                   fingerprint=fingerprint,
                                   fp_size=fp_size, optional=optional, threads=threads)


def compute(batch: Batch) -> np.ndarray:
    """Run the C++ blocks over an already-extracted Batch. The boundary, and only the boundary."""
    return _core.blocks(batch.atom_off, batch.bond_off, batch.chg_ok, batch.atom_i,
                        batch.atom_d, batch.bond_i, batch.bond_s, batch.bond_d)


# ==========================================================================================
# THE PUBLIC ENTRY POINT
# ==========================================================================================

#: Descriptors that are ours rather than RDKit's or Mordred's. Loaded lazily from the column
#: list so it cannot drift from what the extension actually emits.
def _additional_names():
    global _ADDITIONAL
    try:
        return _ADDITIONAL
    except NameError:
        pass
    from ._additional import ADDITIONAL_COLUMNS
    _ADDITIONAL = tuple(c for c in ADDITIONAL_COLUMNS if c in set(ALL_COLUMNS))
    return _ADDITIONAL


class _Unset:
    """Sentinel for `standardize`, so an explicit 'none' can be told apart from an unset one.

    The choice of what molecule the descriptors describe is the caller's, and getting it silently
    wrong is expensive. Leaving it unset behaves as 'none' and warns once; passing 'none' is a
    decision and is silent.
    """

    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()
_STANDARDIZE_WARNED = False


def _standardize_mols(mols, how):
    """Apply the caller's standardization choice. Returns a new list; never mutates the input."""
    global _STANDARDIZE_WARNED
    from rdkit import Chem

    if callable(how):
        return [None if m is None else how(m) for m in mols]
    if isinstance(how, _Unset):
        how = "none"
        if not _STANDARDIZE_WARNED:
            _STANDARDIZE_WARNED = True
            warnings.warn(
                "standardize='none': molecules are featurized exactly as supplied. Descriptors "
                "are computed on the graph you hand them, so a salt, a tautomer and a charge "
                "state are different molecules here. Pass standardize='canonical' for a SMILES "
                "round-trip, 'cleanup' for RDKit MolStandardize (normalize, largest fragment, "
                "uncharge), or a callable of your own. Passing standardize='none' explicitly "
                "makes that the decision and silences this.",
                UserWarning, stacklevel=3)
        return list(mols)
    if how == "none":
        return list(mols)
    if how == "canonical":
        out = []
        for m in mols:
            if m is None:
                out.append(None); continue
            r = Chem.MolFromSmiles(Chem.MolToSmiles(m))
            out.append(r if r is not None else m)
        return out
    if how == "cleanup":
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
        except ImportError as e:
            raise RuntimeError(
                "standardize='cleanup' needs rdkit.Chem.MolStandardize, which this rdkit build "
                "does not provide. Use standardize='canonical', or pass a callable."
            ) from e
        norm = rdMolStandardize.Normalizer()
        frag = rdMolStandardize.LargestFragmentChooser()
        unch = rdMolStandardize.Uncharger()
        out = []
        for m in mols:
            if m is None:
                out.append(None); continue
            try:
                out.append(unch.uncharge(frag.choose(norm.normalize(m))))
            except Exception:
                out.append(m)
        return out
    raise ValueError(
        f"standardize={how!r} is not a choice. Use 'none' (featurize as supplied), "
        "'canonical' (SMILES round-trip), 'cleanup' (RDKit MolStandardize), or a callable "
        "taking and returning an rdkit Mol.")


def _column_index(columns, additional_descriptors):
    """-> (index array into ALL_COLUMNS, names), or (None, ALL_COLUMNS) when nothing is filtered.

    `columns` fixes the output order: the caller gets the columns they asked for, in the order
    they asked for them. Without it the order is ALL_COLUMNS.
    """
    drop = set() if additional_descriptors else set(_additional_names())
    if columns is None:
        if not drop:
            return None, ALL_COLUMNS
        idx = [i for i, c in enumerate(ALL_COLUMNS) if c not in drop]
        return np.asarray(idx, dtype=np.intp), tuple(ALL_COLUMNS[i] for i in idx)

    pos = {c: i for i, c in enumerate(ALL_COLUMNS)}
    want, seen, dup = [], set(), []
    for c in columns:
        if c in seen:
            dup.append(c); continue
        seen.add(c); want.append(c)
    if dup:
        warnings.warn(
            f"columns repeated {len(dup)} name(s), e.g. {dup[:4]}; each is emitted once.",
            UserWarning, stacklevel=3)
    missing = [c for c in want if c not in pos]
    if missing:
        raise ValueError(
            f"columns names {len(missing)} descriptor(s) this build does not emit: "
            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}. "
            f"molhume.ALL_COLUMNS lists all {len(ALL_COLUMNS)} available names.")
    kept = [c for c in want if c not in drop]
    if len(kept) != len(want):
        warnings.warn(
            f"additional_descriptors=False removed {len(want) - len(kept)} column(s) that "
            f"`columns` asked for, e.g. {[c for c in want if c in drop][:4]}. The two filters "
            "are combined with AND.", UserWarning, stacklevel=3)
    if not kept:
        raise ValueError(
            "the columns / additional_descriptors filters together select no columns at all.")
    return np.asarray([pos[c] for c in kept], dtype=np.intp), tuple(kept)


def feature_names(*, fingerprint: bool = True, fp_size: int = 2048,
                  additional_descriptors: bool = True, columns=None) -> tuple:
    """The column names `featurize` produces for the same flags, in the same order.

    `featurize` returns a bare array, because the names do not change from call to call and a
    tuple whose third element is the same 1,269 strings every time is something a caller
    unpacks and discards. Ask for them here when you need them -- a DataFrame, a model's
    feature_names_in_, a plot axis::

        X = molhume.featurize(smiles, standardize="none")
        df = pandas.DataFrame(X, columns=molhume.feature_names())
    """
    _, names = _column_index(columns, additional_descriptors)
    if fingerprint:
        if int(fp_size) <= 0:
            raise ValueError(f"fp_size must be positive, got {fp_size!r}")
        names = names + tuple(f"ECFP_{i}" for i in range(int(fp_size)))
    return names


def minimal_columns(n: int = None, spec: str = "minimal-v1") -> tuple:
    """The first `n` columns of a reduced spec, as names to hand to `featurize(columns=...)`.

        X = molhume.featurize(smiles, columns=molhume.minimal_columns())     # 800 columns

    THE ORDERING IS THE PRODUCT, NOT THE NUMBER. `n` defaults to the shipped operating point,
    but the full ranking is published so a caller can pick their own: a memory-constrained user
    can take 400 knowing exactly what they gave up, against the coverage curve in
    `molhume.minimal_curve()`. A single frozen number would force everyone into one trade-off.

    The set is chosen so that every column NOT in it is linearly recoverable from the ones that
    are -- a COVERAGE criterion, not a compression one. A dropped column costs a model nothing
    if the model can rebuild it; what is lost is only a column's unique variance. It is derived
    label-free, from the descriptor matrix alone, on the training corpus stacked with an
    adversarial salts-and-mixtures set. See docs/MINIMAL_SPEC.md.

    Note this selects what is RETURNED, not what is computed: the block is monolithic, so a
    reduced spec narrows the output and speeds up whatever consumes it, not the featurization.
    """
    from . import _minimal
    if spec != "minimal-v1":
        raise ValueError(
            f"spec={spec!r} is not a spec this build carries. The only one is 'minimal-v1'; "
            "a spec is a frozen contract, so new ones are added rather than edited.")
    n = _minimal.N_DEFAULT if n is None else int(n)
    if not 1 <= n <= len(_minimal.ORDER_NAMES):
        raise ValueError(
            f"n={n} is out of range for spec {spec!r}, which ranks "
            f"{len(_minimal.ORDER_NAMES)} columns. Pass 1..{len(_minimal.ORDER_NAMES)}, or None "
            f"for the shipped default of {_minimal.N_DEFAULT}.")
    return _minimal.ORDER_NAMES[:n]


def minimal_curve(spec: str = "minimal-v1") -> tuple:
    """What each operating point of `minimal_columns` gives up. See docs/MINIMAL_SPEC.md.

    Each entry reports the WORST-CASE reconstruction R^2 over dropped columns, never the mean:
    a set where 639 columns sit at 0.999 and one at 0.40 has lost something real, and an average
    hides exactly that.
    """
    from . import _minimal
    if spec != "minimal-v1":
        raise ValueError(f"spec={spec!r} is not a spec this build carries; only 'minimal-v1'.")
    return _minimal.CURVE


def featurize(smiles: Iterable, *, standardize=_UNSET, threads: int = 0,
              fingerprint: bool = True, fp_radius: int = 3, fp_size: int = 2048,
              additional_descriptors: bool = True, columns=None, optional=None,
              on_error: str = "nan", dtype=np.float64, batch_size: int = 4096) -> np.ndarray:
    """SMILES -> one ``(n_molecules, n_features)`` array. The entry point.

    Feed it straight to a model::

        X = molhume.featurize(smiles, standardize="none")
        xgboost.XGBRegressor().fit(X, y)

    By default that is 1,269 descriptors followed by 2,048 ECFP bits. Pass
    ``fingerprint=False`` for the descriptors alone.

    Column names are not returned, because they are the same on every call for a given set of
    flags. `molhume.feature_names(...)` gives them for the same flags, in the same order.

    Accepts SMILES strings or rdkit Mol objects; a Mol is used as given, which skips a parse.

    Parameters
    ----------
    standardize : {'none', 'canonical', 'cleanup'} or callable, default 'none'
        What to do to each molecule before featurizing. Descriptors are computed on the graph
        they are given, so this decides what molecule the numbers describe: a salt, a tautomer
        and a charge state are different molecules here. Leaving it unset behaves as 'none' and
        warns once per process, because the choice is the caller's to make; passing 'none'
        explicitly is that choice, and is silent.
    threads : int, default 0
        Workers for the descriptor block; 0 is one per hardware thread. Pass 1 if the caller is
        already parallel, or 16 processes x 12 threads will spend the run in the scheduler.
    fingerprint : bool, default True
        Append `fp_size` ECFP bit columns after the descriptors, giving 1,269 + `fp_size`
        columns. They go LAST, so descriptor column indices do not shift when the flag changes.
        Turning it off saves about 30 us/molecule that cannot be threaded, and drops the output
        to descriptors alone.
    additional_descriptors : bool, default True
        Include the descriptors that are ours rather than RDKit's or Mordred's. This selects
        what is returned, not what is computed -- the block is monolithic, so turning it off
        narrows the output without making the run faster.
    columns : sequence of str, optional
        Emit only these descriptors, in this order. Combined with `additional_descriptors` by
        AND. Applies to descriptors only; fingerprint bits are appended after whatever it
        selects.
    optional : sequence of str, optional
        Expensive columns to compute. Default: 'AvgIpc' on, 'qed' off.
    on_error : {'nan', 'raise', 'skip'}, default 'nan'
        What to do with a SMILES rdkit cannot parse. 'nan' keeps the row and fills it with NaN,
        so the output aligns with the input. 'skip' drops the row, so it does not.
    dtype : default numpy.float64
        Output dtype. float32 halves the memory and is what most gradient-boosting and neural
        libraries convert to internally anyway; it costs about 7 significant digits, which
        matters if you are comparing values against a reference and not if you are fitting a
        model.
    """
    from rdkit import Chem

    if on_error not in ("nan", "raise", "skip"):
        raise ValueError(
            f"on_error={on_error!r} is not a choice. Use 'nan' (keep the row, fill with NaN), "
            "'raise', or 'skip' (drop the row, so the output no longer aligns with the input).")

    items = list(smiles)
    mols, bad = [], []
    for i, s in enumerate(items):
        if s is None:
            m = None
        elif isinstance(s, str):
            m = Chem.MolFromSmiles(s)
        elif isinstance(s, Chem.Mol):
            m = s
        else:
            raise TypeError(
                f"item {i} is a {type(s).__name__}; featurize takes SMILES strings or rdkit "
                "Mol objects.")
        if m is None:
            bad.append(i)
            if on_error == "raise":
                raise ValueError(f"could not parse SMILES at index {i}: {s!r}")
        mols.append(m)
    if bad:
        warnings.warn(
            f"{len(bad)} of {len(items)} SMILES did not parse "
            f"(first at index {bad[0]}: {items[bad[0]]!r}); "
            + ("their rows are NaN and the output still aligns with the input."
               if on_error == "nan" else
               "their rows were dropped, so the output no longer aligns with the input."),
            UserWarning, stacklevel=2)

    mols = _standardize_mols(mols, standardize)
    keep_rows = [i for i, m in enumerate(mols) if m is not None]
    good = [mols[i] for i in keep_rows]

    fp_g, X_g, _ = featurize_all_from_mols(
        good, batch_size=batch_size, fp_radius=fp_radius, fp_size=fp_size,
        optional=optional, threads=threads, fingerprint=fingerprint)

    idx, names = _column_index(columns, additional_descriptors)
    if idx is not None:
        X_g = X_g[:, idx]
    if fingerprint:
        X_g = np.hstack([X_g.astype(dtype, copy=False), fp_g.astype(dtype, copy=False)])
    else:
        X_g = X_g.astype(dtype, copy=False)

    if on_error == "skip" or not bad:
        return np.ascontiguousarray(X_g, dtype=dtype)
    X = np.full((len(items), X_g.shape[1]), np.nan, dtype=dtype)
    if keep_rows:
        X[keep_rows] = X_g
    return X
