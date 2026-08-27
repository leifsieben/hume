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
from ._extract import Batch, extract

__all__ = ["featurize_blocks", "featurize_blocks_from_mols", "COLUMNS", "N_COLS"]

assert _core.N_COLS == N_COLS, (
    f"extension emits {_core.N_COLS} columns, _columns.py names {N_COLS}"
)


def featurize_blocks_from_mols(mols: Sequence, batch_size: int = 4096):
    """The 182 verified columns for already-parsed RDKit molecules.

    Returns ``(X, COLUMNS)`` with ``X`` float64 of shape ``(len(mols), 182)``.

    `batch_size` bounds peak memory in the flat arrays, nothing else; the boundary crossing
    itself costs ~0.1 us, so the number is not performance-critical above a few hundred.
    """
    mols = list(mols)
    if not mols:
        return np.zeros((0, N_COLS), dtype=np.float64), COLUMNS
    out = np.empty((len(mols), N_COLS), dtype=np.float64)
    for lo in range(0, len(mols), batch_size):
        chunk = mols[lo:lo + batch_size]
        out[lo:lo + len(chunk)] = compute(extract(chunk))
    return out, COLUMNS


def featurize_blocks(smiles: Iterable[str], batch_size: int = 4096):
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
    return featurize_blocks_from_mols(mols, batch_size=batch_size)


def compute(batch: Batch) -> np.ndarray:
    """Run the C++ blocks over an already-extracted Batch. The boundary, and only the boundary."""
    return _core.blocks(batch.atom_off, batch.bond_off, batch.chg_ok, batch.atom_i,
                        batch.atom_d, batch.bond_i, batch.bond_s, batch.bond_d)
