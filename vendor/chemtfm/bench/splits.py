"""Scaffold-based cross-validation splits (roadmap §6).

**Never random splits.** Random splitting puts near-identical analogs on both sides of the
split, so the model is scored on interpolation it has effectively already seen — it flatters
every method and hides exactly the generalisation gap MPP cares about. We split on Bemis–Murcko
scaffold so that a whole congeneric series lands entirely in train or entirely in test.

The invariant this module guarantees: **no scaffold spans two folds.** If it did, a cliff pair
(two analogs on one scaffold, wildly different activity) could have one member training and the
other testing — textbook leakage across the sharpest signal we measure.

ALGORITHM
---------
1. Group molecule indices by Murcko scaffold (``chem/scaffold.py``). Unparseable molecules and
   acyclics each become their own singleton group, so they spread across folds instead of
   clumping.
2. Greedy balanced partition: sort groups largest-first, assign each to the currently-smallest
   fold. This keeps groups intact (the invariant) while balancing fold sizes as well as
   whole-group assignment allows. Bemis–Murcko on drug-like space is top-heavy — a few huge
   scaffold groups — so perfect balance is impossible; greedy-largest-first is the standard,
   near-optimal compromise.

Determinism: groups are ordered by ``(-size, scaffold_smiles)`` before assignment, and the
``seed`` only controls a stable shuffle *within* equal-size groups, so a given (smiles, k, seed)
always yields the same folds.
"""

from __future__ import annotations

import numpy as np

from chemtfm.chem.scaffold import ACYCLIC, murcko_scaffold
from chemtfm.feat.parse import parse_smiles


def scaffold_key(smi: str, index: int) -> str:
    """Scaffold grouping key for one SMILES.

    Parseable, cyclic molecules key on their Murcko scaffold SMILES. Acyclics and parse
    failures each get a unique per-index sentinel so they become singleton groups and scatter
    across folds rather than forming one giant lump that a single fold would have to absorb.
    """
    result = parse_smiles(smi, warn=False)
    if result.mol is None:
        return f"<unparsed:{index}>"
    scaf = murcko_scaffold(result.mol)
    if scaf == ACYCLIC:
        return f"<acyclic:{index}>"
    return scaf


def scaffold_groups_for(smiles: list[str]) -> dict[str, list[int]]:
    """Map scaffold key → list of molecule indices, in input order."""
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(scaffold_key(smi, i), []).append(i)
    return groups


def scaffold_folds(smiles: list[str], k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Partition indices into ``k`` scaffold-disjoint folds.

    Returns a list of ``k`` arrays of **test** indices. For fold ``i``, the training set is the
    union of all other folds — see ``train_test``. Every scaffold group is wholly inside exactly
    one fold.
    """
    if k < 2:
        raise ValueError(f"need k >= 2 folds, got {k}")
    groups = scaffold_groups_for(smiles)

    # Order groups largest-first; break ties deterministically. The seed adds a stable shuffle
    # among same-size groups so runs are reproducible but not adversarially ordered.
    rng = np.random.default_rng(seed)
    items = list(groups.items())
    # Attach a random tiebreaker, then sort by (-size, tiebreaker) — deterministic given seed.
    tiebreaks = rng.random(len(items))
    order = sorted(range(len(items)), key=lambda j: (-len(items[j][1]), tiebreaks[j]))

    fold_indices: list[list[int]] = [[] for _ in range(k)]
    fold_sizes = np.zeros(k, dtype=np.int64)
    for j in order:
        _, members = items[j]
        target = int(np.argmin(fold_sizes))  # currently-smallest fold
        fold_indices[target].extend(members)
        fold_sizes[target] += len(members)

    return [np.sort(np.array(f, dtype=np.int64)) for f in fold_indices]


def train_test(folds: list[np.ndarray], i: int) -> tuple[np.ndarray, np.ndarray]:
    """Given fold arrays and a held-out fold index, return ``(train_idx, test_idx)``."""
    test = folds[i]
    train = np.concatenate([f for j, f in enumerate(folds) if j != i])
    return np.sort(train), np.sort(test)


def check_no_leakage(smiles: list[str], folds: list[np.ndarray]) -> bool:
    """Verify the invariant: no scaffold appears in more than one fold.

    Returns True if clean. Used by the smoke test and available as a runtime assertion.
    """
    fold_of: dict[int, int] = {}
    for fi, fold in enumerate(folds):
        for idx in fold.tolist():
            fold_of[idx] = fi
    scaffold_to_fold: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        if i not in fold_of:
            continue
        key = scaffold_key(smi, i)
        # Per-index sentinels (unparsed/acyclic) are unique by construction — skip them.
        if key.startswith("<unparsed:") or key.startswith("<acyclic:"):
            continue
        if key in scaffold_to_fold and scaffold_to_fold[key] != fold_of[i]:
            return False
        scaffold_to_fold[key] = fold_of[i]
    return True
