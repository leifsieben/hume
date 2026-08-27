"""Bemis-Murcko scaffolds.

Used for two different purposes that are worth keeping distinct in your head:

- **Stratified sampling** (stage S2) — so the pool is not 40k kinase inhibitors sharing one
  hinge binder. Diversity of the pool.
- **Splitting** (stage S6) — so train and test do not share a scaffold. Leakage control.

Both need the same primitive, so it lives here rather than in either caller.

Two scaffold flavours are provided because they answer different questions:

- ``murcko_scaffold`` keeps atom types and bond orders. Two analogs differing only in a
  substituent share it; two ring systems differing by one heteroatom do not.
- ``generic_scaffold`` erases atom types and bond orders, leaving the raw ring-and-linker
  topology. Pyridine and benzene collapse together.

The generic form is deliberately *not* the default. It merges scaffolds that differ by a
single heteroatom — which is exactly the kind of edit that produces an activity cliff. Using
it for splitting would put a cliff pair's two members in the same group and hide the very
effect the benchmark exists to measure. It is offered for reporting breadth, not for splits.
"""

from __future__ import annotations

from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Sentinel for molecules with no ring system. Bemis-Murcko is undefined for acyclics — the
# scaffold is the empty string — and lumping every acyclic molecule under "" would create one
# enormous pseudo-group that a splitter would then have to assign atomically.
#
# PoolConfig.min_rings excludes acyclics from the pool for this reason, so in practice this
# should never fire on pool members. It exists so that a caller passing an acyclic gets an
# explicit marker rather than a silent empty-string group.
ACYCLIC = "<acyclic>"


def murcko_scaffold(mol: Chem.Mol, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold as canonical SMILES.

    Ring systems plus the linkers connecting them; all side chains stripped.

    ``include_chirality`` defaults to **False**, which is a deliberate asymmetry with the rest
    of the pipeline — everywhere else we preserve stereo religiously (curation spec §3.5).
    Here it is correct to drop it: the scaffold is a *grouping* key for sampling and splitting,
    and two enantiomers belong in the same group. Putting them in different groups would let
    one enantiomer train and the other test, which is textbook leakage across the sharpest
    cliff class there is.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return ACYCLIC
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ACYCLIC
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def generic_scaffold(mol: Chem.Mol) -> str:
    """Return the atom-type- and bond-order-erased scaffold ("graph framework").

    For **reporting breadth only**. See the module docstring: this merges scaffolds differing
    by a single heteroatom, which is a cliff-generating edit, so it must not be used to build
    split groups.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ACYCLIC
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    except Exception:
        return ACYCLIC
    return Chem.MolToSmiles(generic)


def scaffold_groups(
    mols: list[Chem.Mol], include_chirality: bool = False
) -> dict[str, list[int]]:
    """Group molecule indices by scaffold.

    Returns ``{scaffold_smiles: [indices]}``. Insertion order of the indices follows the input
    order, so downstream sampling is reproducible given a reproducible input order.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, mol in enumerate(mols):
        groups[murcko_scaffold(mol, include_chirality)].append(i)
    return dict(groups)


def group_size_summary(groups: dict[str, list[int]]) -> dict[str, float | int]:
    """Summary statistics for a scaffold grouping.

    This is an M0/M3 gate report, not decoration. The number that matters is the size of the
    largest group: Bemis-Murcko on drug-like space is notoriously top-heavy (benzene and
    simple biphenyls absorb a large share of any pool), and a single group holding a
    substantial fraction of the pool cannot be split 80/20 without either leaking or badly
    unbalancing the folds. If ``largest_frac`` is big, the splitter needs to know before it
    tries.
    """
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    total = sum(sizes)
    if not sizes:
        return {"n_groups": 0, "n_molecules": 0}
    return {
        "n_groups": len(sizes),
        "n_molecules": total,
        "largest": sizes[0],
        "largest_frac": sizes[0] / total,
        "singletons": sum(1 for s in sizes if s == 1),
        "singleton_frac": sum(1 for s in sizes if s == 1) / len(sizes),
        "median": sizes[len(sizes) // 2],
        "top10_frac": sum(sizes[:10]) / total,
    }
