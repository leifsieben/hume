"""HUME -- the bridge to the verified C++ blocks.

MILESTONE 1 ONLY. This exposes `featurize_blocks`, which returns the 182 columns that have a C++
implementation *and* an oracle: they are ALL EXACT against RDKit and against HUME's own Python
modules on 98,905 molecules, and cpp/values_hume.txt is the file that says so. The other 683
columns of the full descriptor set have no C++ at all yet, and the public `featurize()` of API.md
is deliberately not here -- a bridge that carries 182 verified columns is worth more than one
that carries 865 unverified ones.

    >>> import hume
    >>> X, cols = hume.featurize_blocks(["CCO", "c1ccccc1"])
    >>> X.shape
    (2, 182)

There is no text file anywhere in this path. RDKit molecules become flat numpy arrays
(`_extract.py`), the arrays cross the boundary once per batch, and the C++ writes float64
straight into the output array.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import _core
from ._columns import COLUMNS, N_COLS
from ._extract import Batch, extract, extract_pickles

__all__ = ["featurize_blocks", "featurize_blocks_from_mols", "featurize_all",
           "featurize_all_from_mols", "COLUMNS", "N_COLS", "ALL_COLUMNS", "N_ALL_COLS",
           "FAMILY_OFFSETS"]

assert _core.N_COLS == N_COLS, (
    f"extension emits {_core.N_COLS} columns, _columns.py names {N_COLS}"
)

# EVERY COLUMN THAT HAS C++ TODAY, in the order the extension emits it. The first 182 are named
# by _columns.py (generated from the same modules cpp/verify_hume.py checks); the rest are named
# by the C++ that decides their order, so the package cannot disagree with the extension about
# which number is which. FAMILY_OFFSETS says where each family starts.
ALL_COLUMNS: tuple[str, ...] = COLUMNS + tuple(_core.all_column_names_tail())
N_ALL_COLS: int = _core.N_ALL_COLS
FAMILY_OFFSETS: dict = dict(_core.ALL_OFFSETS)

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
            out[lo:lo + len(chunk)] = _core.blocks_from_pickles(extract_pickles(chunk).blobs)
        else:
            out[lo:lo + len(chunk)] = compute(extract(chunk))
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
                            fp_size: int = 2048):
    """-> ``(fp, X, ALL_COLUMNS)``: the ECFP and every natively computed descriptor column.

    ``fp`` is ``(len(mols), fp_size)`` uint8 Morgan/ECFP with chirality; ``X`` is
    ``(len(mols), N_ALL_COLS)`` float64.

    THE BOUNDARY IS THE PICKLE ONE: `m.ToBinary()` once per molecule plus the ring CSR, parsed
    in C++. `featurize_blocks(reader="api")` still exists as the reference implementation and the
    two agree bit-for-bit on the 182 they share; there is no `reader` here only because nothing
    has asked for one yet, not because the arrays could not be filled the other way.

    WHAT IS NOT IN X, stated here because a column count invites the opposite assumption: the 52
    Autocorrelation columns built on mordred's `Z` weight. The other nine weights are here (486
    columns, on the hydrogen-added molecule `extract_pickles` serialises alongside), but
    `cpp/ac_weights.h` implements nine of mordred's ten and the tenth is a bare atomic number.
    `bench_e2e.py` counts how many of the 865 these columns actually cover and prints it next to
    the timing, so the ratio and the coverage are never read apart.
    """
    from rdkit.Chem import rdFingerprintGenerator as rfg

    mols = list(mols)
    if not mols:
        return (np.zeros((0, fp_size), dtype=np.uint8),
                np.zeros((0, N_ALL_COLS), dtype=np.float64), ALL_COLUMNS)
    gen = rfg.GetMorganGenerator(radius=fp_radius, fpSize=fp_size, includeChirality=True)
    fp = np.empty((len(mols), fp_size), dtype=np.uint8)
    X = np.empty((len(mols), N_ALL_COLS), dtype=np.float64)
    for lo in range(0, len(mols), batch_size):
        chunk = mols[lo:lo + batch_size]
        p = extract_pickles(chunk)
        X[lo:lo + len(chunk)] = _core.all_from_pickles(
            p.blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at, p.h_blobs)
        for i, m in enumerate(chunk):
            fp[lo + i] = gen.GetFingerprintAsNumPy(m)
    return fp, X, ALL_COLUMNS


def featurize_all(smiles: Iterable[str], batch_size: int = 4096, fp_radius: int = 3,
                  fp_size: int = 2048):
    """`featurize_all_from_mols` from SMILES. Unparseable input raises; see `featurize_blocks`."""
    from rdkit import Chem

    mols = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"could not parse SMILES at index {i}: {s!r}")
        mols.append(m)
    return featurize_all_from_mols(mols, batch_size=batch_size, fp_radius=fp_radius,
                                   fp_size=fp_size)


def compute(batch: Batch) -> np.ndarray:
    """Run the C++ blocks over an already-extracted Batch. The boundary, and only the boundary."""
    return _core.blocks(batch.atom_off, batch.bond_off, batch.chg_ok, batch.atom_i,
                        batch.atom_d, batch.bond_i, batch.bond_s, batch.bond_d)
