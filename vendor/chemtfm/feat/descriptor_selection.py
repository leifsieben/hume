"""VENDORED (TRIMMED) from ChemTFM_OLD — see vendor/README.md.

Only the three names that ``chemtfm.feat.descriptors`` imports are kept:
``FEAT_DATA_DIR``, ``SHARED_COMPUTATION_GROUPS``, ``load_selected_descriptors``
(plus ``SELECTED_DESCRIPTORS_FILE``, which is the default argument of the last one).
Those four definitions are byte-identical to the originals.

WHAT WAS DROPPED, AND WHY IT IS SAFE
------------------------------------
The upstream module is 495 lines and is, by its own docstring, "archaeology": a one-shot
*generator* that measured per-descriptor wall-clock cost over a corpus and wrote the pinned
``feat/data/*.tsv`` artifacts. It ran once; its output is committed. Nothing in
universal-encoder invokes the generator — the only consumer is ``feat/descriptors.py``, which
*reads* the frozen tsv.

Dropped: ``DescriptorMeasurement``, the cost-benchmark / finiteness / dynamic-range / linear
reconstruction machinery, the tsv writer, ``_main()``/argparse, and the tuning constants that
only the generator reads (``DEFAULT_COST_BUDGET_US``, ``DEFAULT_RECONSTRUCTION_R2``,
``REQUIRED``, ``SHARED_GROUP_COST_US``, ``DEFAULT_MIN_FINITE_FRAC``,
``DEFAULT_MAX_DYNAMIC_RANGE``).

Dropping the generator is what removes the ``chemtfm.config`` dependency: ``PoolConfig`` was
referenced *only* inside ``_main()`` (upstream lines 450-468). Not vendoring it keeps
``chemtfm.config`` -> ``chemtfm.hashing`` (302 further lines, plus a feature-policy hashing
scheme this repo does not use) out of the tree.

Consequence to be aware of: the frozen descriptor list can no longer be *re-derived* here.
If that is ever needed, run the generator in ChemTFM_OLD, not in this repo.
"""

from __future__ import annotations

from pathlib import Path

# --- verbatim from ChemTFM_OLD/chemtfm/feat/descriptor_selection.py, lines 96-99 ---------
# Where the pinned artifact lives. Hashed into the feature policy version, so editing it
# invalidates every cached descriptor vector.
FEAT_DATA_DIR = Path(__file__).parent / "data"
SELECTED_DESCRIPTORS_FILE = FEAT_DATA_DIR / "descriptors_selected.tsv"

# --- verbatim from ChemTFM_OLD/chemtfm/feat/descriptor_selection.py, lines 127-149 -------
# Descriptor families where RDKit's `_descList` repeats a shared subcomputation, so the
# per-descriptor benchmark overstates their true marginal cost.
#
# Measured: calling MaxEStateIndex / MinEStateIndex / MaxAbsEStateIndex / MinAbsEStateIndex
# separately costs 273 us/mol because each one recomputes the EState vector. Computing
# `EStateIndices(mol)` once and deriving all four costs 69 us/mol — 3.9x cheaper, bit-identical
# values. The four are therefore costed as a group at their shared cost, not individually.
#
# This is a real correctness issue for cost-based selection, not an optimisation footnote:
# scored individually all four look unaffordable and get dropped, and three of them carry
# information the cheap set cannot reconstruct (R2 = 0.56 to 0.89). Naive per-descriptor
# benchmarking would have silently discarded a genuinely distinct block of the embedding.
#
# S3 IMPLEMENTATION REQUIREMENT: the descriptor provider must special-case this group and
# compute the shared vector once. Calling the `_descList` entries individually reintroduces
# the 4x cost.
SHARED_COMPUTATION_GROUPS: dict[str, tuple[str, ...]] = {
    "estate": (
        "MaxEStateIndex",
        "MinEStateIndex",
        "MaxAbsEStateIndex",
        "MinAbsEStateIndex",
    ),
}


# --- verbatim from ChemTFM_OLD/chemtfm/feat/descriptor_selection.py, lines 410-423 -------
def load_selected_descriptors(path: Path = SELECTED_DESCRIPTORS_FILE) -> tuple[str, ...]:
    """Read the pinned descriptor names, in file order.

    Order is preserved and load-bearing: it fixes the column layout of the descriptor
    matrix, and a reordering would silently permute every cached feature vector relative to
    a freshly computed one.
    """
    names: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split("\t")[0])
    return tuple(names)
