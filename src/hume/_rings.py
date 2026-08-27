"""The ring list the boundary hands to RingCount, and the repair that makes it deterministic.

WHY THIS FILE EXISTS AT ALL. `atom_i` carries a per-atom ring COUNT, which is everything the 182
blocks and the Crippen typer ask of ring perception. mordred's RingCount asks a different
question: all 49 of its columns are predicates on a RING -- its size, whether ALL of its atoms
are aromatic, whether ANY is a heteroatom -- and its 28 fused columns additionally need
|Ri & Rj| for every pair of rings, to build the fusion graph. None of that is recoverable from
per-atom counts: benzene and cyclohexane have identical `nring` vectors and differ on 6 of the
49. So RingCount needs the ring SET. It is NOT a second perception -- the set comes from the same
single RDKit ring perception `nring` is filled from.

`RingInfo().AtomRings()` IS `Chem.GetSymmSSSR`, which is what mordred asks for by name. Verified
equal on all 100,000 molecules of cpp/hard.smi, and verified to survive RDKit's pickle round trip
byte-identically -- which is what lets the pickle path take its rings from the blob for free.

THE PROBLEM THIS FILE SOLVES. `Chem.GetSymmSSSR` is not a function of the molecular graph.
RDKit's `symmetrizeSSSR` adds symmetry-equivalent extra rings to the SSSR basis, and its own
source admits it "may miss extra rings that would need to swap two (or three...) rings to be
included". Whether it misses one depends on the order it sees the molecule in. Measured with
mordred on cpp/hard.smi: 25 of 100,000 molecules move at least one RingCount column under
renumbering, across five columns (nARing, nG12Ring, n6Ring, n7Ring, n6ARing). The SSSR BASIS is
stable; what flips is a single extra ring of a size already present.

See cpp/verify_topo3.py for the evidence and the standing checks (`canon`, `gatecheck`, `perm`).
"""
from __future__ import annotations

from rdkit import Chem

__all__ = ("canon_rings", "gate", "rings_for")


def canon_rings(mol):
    """RDKit's symmetrised SSSR, perceived in a canonical order and mapped back to the caller's
    numbering. A function of the molecular graph, which `Chem.GetSymmSSSR` is not.

    THE REPAIR IS TO THE SELECTION, NOT TO THE QUANTITY. Nothing about what is counted changes:
    on every molecule where RDKit is already stable this returns exactly RDKit's own answer, and
    on the ambiguous ones it returns whichever of RDKit's own answers the canonical order yields.
    Redefining the family on the cyclomatic number, or on the full relevant-cycle set, would
    change the value for every symmetric cage rather than only the ambiguous ones -- a different
    descriptor wearing the same name.

    CANONICAL ATOM RANKS ALONE ARE NOT ENOUGH, and this is the whole reason the function looks
    like this. `Chem.RenumberAtoms` renumbers ATOMS and leaves the BOND LIST in its original
    order with the endpoints rewritten, so two molecules can have a byte-identical canonically
    renumbered atom numbering and still present their bonds to the ring perceiver in different
    orders -- and `GetSymmSSSR` then returns different rings from what is, as a numbered graph,
    the same graph. Measured on `COc1cc2ccc1OCc1cccc(n1)COc1ccc(cc1OC)C=NCCNCCNCCN=C2`: canonical
    atom numbering identical on 15 of 15 renumberings, canonical SMILES identical on 15 of 15,
    rings DIFFERENT on 10 of 15. Ranking atoms and leaving the bonds alone made three corpus
    molecules WORSE than doing nothing.

    So perception happens on a SKELETON rebuilt from scratch: n carbons in canonical-rank order,
    bonds added in sorted (rank_u, rank_v) order. Ring perception is a property of the graph
    alone -- it reads no element, no bond order and no aromatic flag -- so the skeleton answers
    exactly the question being asked, and rebuilding it is what puts the BOND order under
    canonical control as well. Aromaticity and atomic number for the counting still come from the
    real molecule; nothing here re-perceives them, which also keeps this clear of the
    canonical-SMILES round trip that is known to move aromaticity on 19 corpus molecules.

    Rings come back sorted by (size, sorted canonical-rank vector), so the ring LIST order is a
    graph invariant too, not merely the multiset. RingCount only counts, but anything downstream
    that indexes rings gets determinism for free.

    Note `Chem.CanonicalRankAtoms(breakTies=True)` is itself not a graph invariant on symmetric
    molecules -- it varies by an AUTOMORPHISM on, for instance, a 1,4-disubstituted cyclohexane.
    That is harmless here, because an automorphic relabelling permutes the rings among themselves
    and every RingCount column is blind to it, but it means the invariance test must be on the
    COLUMN VALUES and not on ring identity. cpp/verify_topo3.py's `canon` mode tests the values.
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return []
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))    # rank[old] = canonical index
    back = [0] * n
    for old, new in enumerate(rank):
        back[new] = old
    edges = sorted((min(rank[b.GetBeginAtomIdx()], rank[b.GetEndAtomIdx()]),
                    max(rank[b.GetBeginAtomIdx()], rank[b.GetEndAtomIdx()]))
                   for b in mol.GetBonds())
    em = Chem.RWMol()
    for _ in range(n):
        em.AddAtom(Chem.Atom(6))
    for u, v in edges:
        em.AddBond(int(u), int(v), Chem.BondType.SINGLE)
    rings = [sorted(back[i] for i in r) for r in Chem.GetSymmSSSR(em.GetMol())]
    return sorted(rings, key=lambda r: (len(r), sorted(rank[i] for i in r)))


def gate(mol) -> bool:
    """Should `canon_rings` be paid for this molecule? A DELIBERATE OVER-APPROXIMATION.

    `canon_rings` costs 104 us/mol against 5.1 for reading `RingInfo`, and it changes the answer
    on 32 molecules in 100,000. Paying it unconditionally is five times all 81 RingCount /
    TopologicalCharge / PathCount columns and roughly 40% of the whole featuriser, so it is gated
    -- but the gate is tuned for SOUNDNESS, not for the smallest firing rate, because a false
    negative is a numbering-dependent column and a false positive is only microseconds.

    It fires on 21.3% of cpp/hard.smi (26.1 us/mol amortised) where the minimal gate that still
    covers every affected molecule fires on 8.5% (13.5 us/mol). The extra 12.8 points are bought
    margin, on all three axes:

      ring size >= 7        observed minimum among affected molecules: a 6-ring. Macrocyclic
                            basis choice is the large-ring failure mode, and 7 puts the threshold
                            below anything seen.
      atom in >= 3 rings    observed minimum: exactly 3, on every one of the 32. This clause
                            alone covers all of them; it is the bridgehead / cage signature.
      ring system with >= 3 independent cycles
                            pure margin on this corpus -- and NOT redundant. On HEXAPRISMANE,
                            `C12C3C4C5C6C1C1C6C5C4C3C21`, every ring has 4 atoms and no atom is
                            in more than 2 rings, so the first two clauses both miss it, while
                            its ring system carries SEVEN independent cycles. That is the shape a
                            narrower gate would let through, and it is why this clause is here.

    RESIDUAL RISK, plainly: a molecule slips past only if its ring set is ambiguous while every
    ring has at most 6 atoms, no atom lies in more than 2 rings, and no ring system has more than
    2 independent cycles -- and the consequence is that this molecule's five numbering-sensitive
    RingCount columns take a numbering-dependent value, not that any value is wrong anywhere
    else. `python cpp/verify_topo3.py gatecheck` is the standing guard: it runs the repair
    unconditionally over the whole corpus and fails loudly if the gate ever disagrees (currently
    0 disagreements over 100,000).

    The provably sound alternative -- "any ring system with 2+ independent cycles" -- is every
    fused aromatic: 50.8% of the corpus, 55.4 us/mol. Measured, and rejected as too dear.
    """
    ri = mol.GetRingInfo()
    if ri.NumRings() < 2:
        return False                      # at most one cycle: there is nothing to choose between
    for r in ri.AtomRings():
        if len(r) >= 7:
            return True
    for i in range(mol.GetNumAtoms()):
        if ri.NumAtomRings(i) >= 3:
            return True
    # cyclomatic number of each ring-bond-connected component: |E| - |V| + 1
    rb = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.IsInRing()]
    adj: dict[int, list[int]] = {}
    for u, v in rb:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen: set[int] = set()
    for s in adj:
        if s in seen:
            continue
        comp, st = [], [s]
        seen.add(s)
        while st:
            u = st.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        cs = set(comp)
        if sum(1 for u, _ in rb if u in cs) - len(comp) + 1 >= 3:
            return True
    return False


def rings_for(mol):
    """The ring list to hand the C++: repaired where it can matter, RDKit's own otherwise."""
    if gate(mol):
        return canon_rings(mol)
    return [list(r) for r in mol.GetRingInfo().AtomRings()]
