"""The descriptor block: exact physicochemical descriptors (roadmap §4.4).

This is the *exact* feature channel — not an encoder. A learned network reproducing
descriptors would be circular (its information ceiling is the descriptors, which we compute
exactly), so there is deliberately no "descriptor encoder". We just compute the pinned,
cost-selected set fast and hand it over.

THE LIST IS FROZEN. The 200 descriptor names come from ``feat/data/descriptors_selected.tsv``,
produced once by ``descriptor_selection.py`` and never re-derived per corpus (it is definitional
for the foundation model — see that module's docstring). This file *consumes* the list; it never
selects.

TWO THINGS THIS MODULE GETS RIGHT
---------------------------------
1. **Column order is the tsv order**, preserved exactly. The order fixes the layout of the
   descriptor columns; permuting it would silently misalign a freshly-computed vector against a
   cached one. We build the ordered ``(name, callable)`` plan once at import.

2. **The EState shared-computation group.** RDKit's ``_descList`` recomputes the EState index
   vector once per EState summary — calling ``MaxEStateIndex``/``MinEStateIndex``/
   ``MaxAbsEStateIndex``/``MinAbsEStateIndex`` separately costs ~4× because each rebuilds the
   same vector. ``descriptor_selection.SHARED_COMPUTATION_GROUPS`` flags this as a required
   optimisation. We compute ``EStateIndices(mol)`` once and derive all four summaries from it.
   Verified bit-identical to RDKit's individual functions.

FAILURE HANDLING
----------------
A descriptor can raise (e.g. BCUT2D on elements without Gasteiger parameters) or return a
non-finite value (inf/NaN) on a pathological molecule. The selection rule already gated the
list on ≥99% finiteness, so this is the rare tail — but at VS scale the tail is populated. We
write ``NaN`` into any cell that raises or is non-finite, per column, and keep going. NaN is a
first-class citizen downstream (XGBoost handles it natively; the ChemPFN will too), so a single
bad descriptor never poisons a whole molecule's row, let alone the run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState import EStateIndices

from chemtfm.feat.descriptor_selection import (
    FEAT_DATA_DIR,
    SHARED_COMPUTATION_GROUPS,
    load_selected_descriptors,
)

# The pinned, ordered descriptor names. Order is load-bearing (fixes the column layout).
# Frozen at the 95%-variance minimal set (96 descriptors) — see scripts/freeze_descriptor_set.py
# and ENCODERS.md §3. Morgan/fingerprints explain ~0 of these (they are complementary).
_MINIMAL_FILE = FEAT_DATA_DIR / "descriptors_min95.tsv"
DESCRIPTOR_NAMES: tuple[str, ...] = load_selected_descriptors(_MINIMAL_FILE)
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

# The four EState summaries share one underlying vector. Kept as a set for O(1) membership.
_ESTATE_MEMBERS = frozenset(SHARED_COMPUTATION_GROUPS["estate"])


def _estate_summaries(mol: Chem.Mol) -> dict[str, float]:
    """Compute all four EState summaries from a single ``EStateIndices`` call.

    Matches RDKit's per-descriptor definitions exactly (verified): the max/min of the EState
    index vector, and the max/min of its absolute values.
    """
    es = np.asarray(EStateIndices(mol), dtype=np.float64)
    if es.size == 0:  # an atom-less mol should be impossible here, but guard anyway
        return dict.fromkeys(_ESTATE_MEMBERS, float("nan"))
    abs_es = np.abs(es)
    return {
        "MaxEStateIndex": float(es.max()),
        "MinEStateIndex": float(es.min()),
        "MaxAbsEStateIndex": float(abs_es.max()),
        "MinAbsEStateIndex": float(abs_es.min()),
    }


def _build_plan() -> list[tuple[str, Callable[[Chem.Mol], float] | None]]:
    """Resolve each pinned name to a callable, once, at import.

    EState members get a ``None`` callable — a sentinel meaning "filled from the shared
    ``_estate_summaries`` computation" rather than from an individual ``_descList`` function.
    Every other name resolves to its RDKit descriptor function.
    """
    by_name = dict(Descriptors._descList)
    plan: list[tuple[str, Callable[[Chem.Mol], float] | None]] = []
    for name in DESCRIPTOR_NAMES:
        if name in _ESTATE_MEMBERS:
            plan.append((name, None))  # sentinel: shared computation
        else:
            fn = by_name.get(name)
            if fn is None:
                # A pinned name that RDKit no longer provides is a hard, load-bearing error:
                # the frozen input space cannot be computed, so nothing downstream is valid.
                raise RuntimeError(
                    f"pinned descriptor {name!r} is absent from RDKit's _descList "
                    "(RDKit version drift). The frozen descriptor set cannot be computed."
                )
            plan.append((name, fn))
    return plan


_PLAN = _build_plan()


def descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute the pinned descriptor vector for one molecule.

    Returns a ``float64`` array of length ``N_DESCRIPTORS`` in the frozen column order. Any
    descriptor that raises or returns a non-finite value becomes ``NaN`` in its cell; the rest
    of the row is unaffected.
    """
    out = np.empty(N_DESCRIPTORS, dtype=np.float64)

    # Compute the EState vector once if any EState summary is in the (frozen) set. The list is
    # fixed, so this branch is effectively constant, but computing lazily costs nothing.
    estate = _estate_summaries(mol) if _ESTATE_MEMBERS else {}

    for i, (name, fn) in enumerate(_PLAN):
        try:
            value = estate[name] if fn is None else float(fn(mol))
        except Exception:
            out[i] = np.nan
            continue
        # Reject inf/NaN explicitly — a non-finite feature column is unusable, and NaN is the
        # value we want to propagate, not inf (which would survive standardisation as a blowup).
        out[i] = value if np.isfinite(value) else np.nan
    return out
