"""Figure A: matched molecular pairs for the resolution test.

Ten chemical edits that a usable representation must respond to, two notation-only controls it
must NOT respond to, and a matched-molecular-weight substitution that sets the per-model scale.

The two null controls are the load-bearing part of the design. Without them a large response
is uninterpretable: a model that reacts strongly to re-writing the same SMILES string is
reacting to formatting, not chemistry, and its response to a real edit cannot be distinguished
from that noise. Chemical edits are applied to the RDKit molecule and serialised ONCE, so the
pair is canonical by construction; notation controls are applied to the STRING and passed
through unmodified. Getting that backwards -- doing string surgery for the chemical edits --
would fold an uncontrolled notation change into every chemical one and systematically flatter
the language models.

Not every edit applies to every molecule, so each is attempted against a shuffled pool until
the target count is reached, and the achieved count is recorded. An edit that cannot reach its
target says something real about the chemistry available in the benchmark and is reported
rather than quietly padded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "figures" / "figA"
N_PER_EDIT = 1000
# Reserved for sigma estimation and excluded from every pair. Must match embed_pairs.py, which
# reads background.json rather than re-deriving it -- see the note in main().
N_BACKGROUND = 10_000


# --- chemical edits: mol -> mol|None ------------------------------------------------------

def _mol(s):
    return Chem.MolFromSmiles(s)


def stereo_flip(m):
    """Invert one specified tetrahedral centre."""
    from rdkit.Chem import ChiralType
    cs = [a for a in m.GetAtoms() if a.GetChiralTag() in
          (ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW)]
    if not cs:
        return None
    e = Chem.RWMol(m)
    a = e.GetAtomWithIdx(random.choice(cs).GetIdx())
    a.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW
                   if a.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW
                   else ChiralType.CHI_TETRAHEDRAL_CCW)
    return e.GetMol()


def ez_flip(m):
    """Invert one defined double-bond geometry, by flipping a directional bond marker.

    Done on the SMILES rather than the RWMol, deliberately. Two `RWMol` attempts produced 0 of
    1000: setting the bond's stereo tag (with or without re-setting its stereo atoms) does not
    survive canonicalisation, and every pair was discarded as a no-op.

    This is NOT the string surgery the module docstring warns against. In SMILES the `/` and
    `\\` markers *are* the double-bond stereochemistry -- flipping one changes the molecule,
    not its formatting -- and the result is re-parsed and re-canonicalised, so the pair that
    reaches the models is a clean canonical pair like any other chemical edit. Verified: it
    turns C/C=C/CCO into C/C=C\\CCO.
    """
    s = Chem.MolToSmiles(m)
    pos = [i for i, c in enumerate(s) if c in "/\\"]
    if len(pos) < 2:
        return None
    i = random.choice(pos)
    flipped = s[:i] + ("\\" if s[i] == "/" else "/") + s[i + 1:]
    n = Chem.MolFromSmiles(flipped)
    if n is None or Chem.MolToSmiles(n) == s:
        return None
    return n


def _sub(m, patt, repl):
    p = Chem.MolFromSmarts(patt)
    if p is None or not m.HasSubstructMatch(p):
        return None
    out = AllChem.ReplaceSubstructs(m, p, Chem.MolFromSmiles(repl), replaceAll=False)
    return out[0] if out else None


def halogen_swap(m):
    for a, b in (("[Cl;X1]", "F"), ("[F;X1]", "Cl"), ("[Br;X1]", "Cl")):
        r = _sub(m, a, b)
        if r is not None:
            return r
    return None


def h_to_methyl(m):
    """Add a methyl to an aromatic CH -- the classic single-atom potency change.

    Built by RWMol rather than ReplaceSubstructs: an aromatic-carbon fragment is not valid
    SMILES on its own, so `MolFromSmiles("c-[CH3]")` returns None and the first version passed
    None straight into ReplaceSubstructs (0 of 1000, ArgumentError swallowed by the try).
    """
    cand = [a.GetIdx() for a in m.GetAtoms()
            if a.GetIsAromatic() and a.GetAtomicNum() == 6 and a.GetTotalNumHs() == 1]
    if not cand:
        return None
    e = Chem.RWMol(m)
    i = random.choice(cand)
    c = e.AddAtom(Chem.Atom(6))
    e.AddBond(i, c, Chem.BondType.SINGLE)
    e.GetAtomWithIdx(i).SetNumExplicitHs(0)
    e.GetAtomWithIdx(i).SetNoImplicit(False)
    return e.GetMol()


def n_methylation(m):
    """Amide N-H -> N-methyl. The classic conformation-and-H-bonding change at constant scaffold.

    RWMol, not ReplaceSubstructs -- see the module docstring's SHATTERING note. Replacing the
    three-atom amide match detached the acyl half from the amine half in 913 of 1000 molecules.
    """
    ms = m.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H1][CX3]=O"))
    if not ms:
        return None
    e = Chem.RWMol(m)
    i = random.choice(ms)[0]
    c = e.AddAtom(Chem.Atom(6))
    e.AddBond(i, c, Chem.BondType.SINGLE)
    a = e.GetAtomWithIdx(i)
    a.SetNumExplicitHs(0)
    a.SetNoImplicit(False)
    return e.GetMol()


def isotope_13c(m):
    """[12C] against [13C] -- BOTH members carry an explicit isotope, and that is the point.

    This used to leave A as the plain molecule and give B a single [13C], so the two SMILES
    differed by a whole bracket atom appearing out of nowhere. The `notation` control scored
    1.000 on this panel: a character counter separates "contains [13C]" from "contains no
    bracket" perfectly, with no chemistry at all, and the CLM arms' scores here were therefore
    unreadable -- true that they resolve the edit, but they need only to see the token.

    Writing A as [12C] leaves the difference at ONE CHARACTER, 2 against 3, inside a string that
    is already full of digits from ring closures. The structures still differ exactly as before:
    carbon-12 against carbon-13 at the same position.

    Chemically A is unchanged -- 12C is carbon's dominant isotope -- so this is a notation change
    on the A side and nothing more.
    """
    cs = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsotope() == 0]
    if not cs:
        return None
    i = random.choice(cs)
    a, b = Chem.Mol(m), Chem.Mol(m)
    a.GetAtomWithIdx(i).SetIsotope(12)
    b.GetAtomWithIdx(i).SetIsotope(13)
    return a, b


def _rings(m, size, arom):
    """Rings of `size` made entirely of carbon, aromatic or saturated as asked."""
    out = []
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) != size:
            continue
        a = [m.GetAtomWithIdx(i) for i in ring]
        if all(x.GetAtomicNum() == 6 and x.GetIsAromatic() == arom for x in a):
            out.append(ring)
    return out


def scaffold_hop(m):
    """Benzene -> pyridine. ONE heavy atom changes element; every bond stays where it was.

    Done by mutating the atom in place, not by ReplaceSubstructs. Swapping the whole six-atom
    ring detached every substituent on it -- 825 of 1000 products came out in fragments, so the
    panel was measuring molecular demolition rather than a C->N substitution.
    """
    for ring in _rings(m, 6, arom=True):
        cand = [i for i in ring if m.GetAtomWithIdx(i).GetTotalNumHs() == 1]
        if not cand:
            continue
        e = Chem.RWMol(m)
        a = e.GetAtomWithIdx(random.choice(cand))
        a.SetAtomicNum(7)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
        return e.GetMol()
    return None


def ring_contract(m):
    """Cyclohexyl -> cyclopentyl. One CH2 deleted, its two neighbours bonded to each other.

    The atom removed must be an UNSUBSTITUTED CH2 belonging to exactly one ring, so the edit
    cannot orphan a substituent or tear a fused system apart -- which is what the substructure
    version did to 698 of 1000.
    """
    ri = m.GetRingInfo()
    for ring in _rings(m, 6, arom=False):
        cand = [i for i in ring
                if m.GetAtomWithIdx(i).GetTotalNumHs() == 2
                and m.GetAtomWithIdx(i).GetDegree() == 2
                and ri.NumAtomRings(i) == 1]
        if not cand:
            continue
        i = random.choice(cand)
        nb = [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors()]
        e = Chem.RWMol(m)
        e.RemoveAtom(i)
        j = [x - (1 if x > i else 0) for x in nb]     # indices shift down past the removed atom
        if e.GetBondBetweenAtoms(*j) is None:
            e.AddBond(j[0], j[1], Chem.BondType.SINGLE)
        return e.GetMol()
    return None


def regioisomer(m):
    """Move one substituent around a benzene ring: ortho vs meta. Identical formula and atom
    counts, different connectivity -- so it is invisible to anything that only counts atoms and
    bonds, and it is the edit CLIMB's Fig G calls `regioisomer`, which makes the two papers'
    plates directly comparable on this row.

    REPLACES `ring_fusion`, which was broken twice over. Its two SMARTS -- c1ccc2ccccc2c1 and
    c1ccc2c(c1)cccc2 -- are both naphthalene, i.e. the same molecule, so the intended edit was a
    no-op; the only reason it ever registered a response is that ReplaceSubstructs shattered 962
    of 1000 products. Linear-vs-angular fusion genuinely needs three rings (anthracene vs
    phenanthrene) and the benchmark has almost none, so the mode is retired rather than repaired.
    """
    ri = m.GetRingInfo()
    for ring in _rings(m, 6, arom=True):
        sub, free = [], []
        for i in ring:
            a = m.GetAtomWithIdx(i)
            ex = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() not in ring]
            if len(ex) == 1 and ri.NumAtomRings(i) == 1 and \
                    m.GetBondBetweenAtoms(i, ex[0]).GetBondType() == Chem.BondType.SINGLE:
                sub.append((i, ex[0]))
            elif a.GetTotalNumHs() == 1:
                free.append(i)
        if not sub or not free:
            continue
        src, ex = random.choice(sub)
        dst = random.choice(free)
        e = Chem.RWMol(m)
        e.RemoveBond(src, ex)
        e.AddBond(dst, ex, Chem.BondType.SINGLE)
        for k, h in ((src, 1), (dst, 0)):
            a = e.GetAtomWithIdx(k)
            a.SetNumExplicitHs(h)
            a.SetNoImplicit(False)
        return e.GetMol()
    return None


def protonate(m):
    # `!$(N-C=O)` and `!$(N-a)` matter: without them the pattern happily protonates an AMIDE or
    # an aniline nitrogen, neither of which is basic, so the panel would be measuring a charge
    # state no molecule adopts. Caught by eye in the regenerated pair dump -- the first
    # scaffold_hop molecule came back as C(=O)[NH+]2CCN..., a protonated piperazine amide.
    # BOTH MEMBERS ARE WRITTEN IN BRACKETS, for the same reason as isotope_13c above: the old
    # version left A as a bare `N` or `O` and gave B a `[NH+]` or `[O-]`, so the panel could be
    # solved by noticing that one string contains a bracket and the other does not. `notation`
    # scored 1.000. Forcing explicit-H notation on the A-side atom too leaves `[N]` against
    # `[NH+]` and `[OH]` against `[O-]`.
    #
    # `SetNoImplicit` + `SetNumExplicitHs` alone does NOT produce brackets -- RDKit drops them
    # again whenever the atom is in its default valence state, which was measured, not assumed.
    # What does force them is a declared isotope, so both members get the atom's DOMINANT isotope
    # pinned: 14N is 99.6% of natural nitrogen and 16O is 99.8% of natural oxygen, so this is a
    # notation change and not a chemical one, and it is identical on both sides so it carries no
    # signal of its own. Same device as isotope_13c, one panel up.
    #
    # THIS DOES NOT MAKE THE PANEL UNFREE, and it cannot: a formal charge IS a character in
    # SMILES, so `+` and `-` remain visible to a character counter however the atom is written.
    # What it removes is the much larger confound of a bracket atom appearing in one member and
    # not the other. The residual floor is reported by the `notation` arm rather than argued away.
    def _both(idx, iso, states):
        a, b = Chem.Mol(m), Chem.Mol(m)
        for mol, nh, chg in zip((a, b), *[[x[k] for x in states] for k in (0, 1)]):
            at = mol.GetAtomWithIdx(idx)
            at.SetIsotope(iso)          # pins the DOMINANT isotope: forces brackets, no chemistry
            at.SetNumExplicitHs(nh)
            at.SetFormalCharge(chg)
            at.SetNoImplicit(True)
        return a, b

    amine = Chem.MolFromSmarts("[NX3;H0;!$(N=*);!$(N-[!#6]);!$(N-C=O);!$(N-a);!+]")
    hit = m.GetSubstructMatches(amine)
    if hit:
        return _both(hit[0][0], 14, [(0, 0), (1, 1)])       # [14N]  vs  [14NH+]
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    hit = m.GetSubstructMatches(acid)
    if hit:
        return _both(hit[0][2], 16, [(1, 0), (0, -1)])      # [16OH] vs  [16O-]
    return None


def saturate(m):
    """Reduce one non-aromatic C=C to C-C. BOND ORDER changes; the connectivity does not.

    The one axis nothing else on the plate tests. Every other edit moves atoms, elements,
    charges or positions; this one leaves the skeleton exactly where it was and changes only how
    two atoms are bonded. ECFP reaches it only through the bond types folded into an environment
    hash, the descriptor block through its unsaturation counts, and a string model through a
    single character -- three quite different routes to the same fact, which is why it earns a
    panel.

    Ring alkenes are allowed (they are not rare enough to skip) but aromatic bonds are not: a
    Kekule-form aromatic bond IS a double bond to RDKit, and "reducing" one would dearomatise the
    ring -- a far larger edit than advertised, and one the null_kekulize control would then be
    entangled with. Double-bond stereo is cleared explicitly, since a saturated bond carrying a
    stale E/Z tag does not survive sanitisation.
    """
    cand = [b for b in m.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
            and not b.GetBeginAtom().GetIsAromatic() and not b.GetEndAtom().GetIsAromatic()
            and b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6]
    if not cand:
        return None
    e = Chem.RWMol(m)
    pick = random.choice(cand)
    nb = e.GetBondBetweenAtoms(pick.GetBeginAtomIdx(), pick.GetEndAtomIdx())
    nb.SetStereo(Chem.BondStereo.STEREONONE)
    nb.SetBondType(Chem.BondType.SINGLE)
    return e.GetMol()


def ch2_homolog(m):
    """Insert one CH2 into an acyclic single bond -- chain homologation, R-X -> R-CH2-X.

    Replaces `matched_mw` as a PANEL (Leif 2026-08-28). Under the AUC axis that reference is no
    longer the unit: with two unrelated molecules the A/A' label is arbitrary, so every arm reads
    ~0.5 and the panel says nothing. Homologation is a real, minimal, ubiquitous medicinal-
    chemistry change and it earns the slot.

    DIRECTIONAL BY CONSTRUCTION, which the AUC metric requires: A is always the shorter
    homologue and A' the longer, so "which side of the pair" is a fact about chemistry rather
    than about which molecule the generator happened to write first. An edit whose two members
    are interchangeable -- swap two identical substituents, say -- reads 0.5 for every
    representation and measures nothing at all.

    The bond must be acyclic (inserting into a ring changes ring size, which is a different edit
    and `ring_contract` already owns it) and both ends heavy. An aromatic ring carbon is allowed
    at one end: c-CH3 -> c-CH2-CH3 is homologation, and the BOND is single even though the atom
    is aromatic.
    """
    cand = [b for b in m.GetBonds()
            if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
            and b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1
            and 6 in (b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum())]
    if not cand:
        return None
    b = random.choice(cand)
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    e = Chem.RWMol(m)
    e.RemoveBond(i, j)
    c = e.AddAtom(Chem.Atom(6))
    e.AddBond(i, c, Chem.BondType.SINGLE)
    e.AddBond(c, j, Chem.BondType.SINGLE)
    return e.GetMol()


EDITS = {"stereo_flip": stereo_flip, "ez_flip": ez_flip, "halogen_swap": halogen_swap,
         "saturate": saturate,
         "h_to_methyl": h_to_methyl, "n_methylation": n_methylation, "isotope_13c": isotope_13c,
         "scaffold_hop": scaffold_hop, "ring_contract": ring_contract,
         "protonate": protonate, "regioisomer": regioisomer,
         "ch2_homolog": ch2_homolog}


# --- controls -----------------------------------------------------------------------------

def null_enumerate(m):
    """Same molecule, different SMILES string. MUST score zero."""
    return Chem.MolToSmiles(m, doRandom=True, canonical=False)


def null_kekulize(m):
    """Same molecule, Kekule rather than aromatic form. MUST score zero."""
    k = Chem.Mol(m)
    Chem.Kekulize(k, clearAromaticFlags=True)
    return Chem.MolToSmiles(k, kekuleSmiles=True)


def _intact(before, after):
    """Did the edit leave the molecule in one piece?

    THE GUARD THAT SHOULD HAVE BEEN HERE FROM THE START. `AllChem.ReplaceSubstructs` re-attaches
    the neighbours of a replaced fragment on a best-effort basis, and when the match carries more
    than one exocyclic bond it simply drops them: a benzene->pyridine swap turned
    CNCc1cc(C(=O)N2CCN(C3CC3)CC2)ccc1Oc1cccc(F)c1 into THREE separate fragments. Every product
    still sanitised cleanly, still canonicalised, still differed from its input -- so every check
    the generator had passed, and four modes spent their whole run measuring demolition.

    It surfaced downstream, in the descriptor arm of Fig A, as a response of ~7x "a completely
    different compound" concentrated in Radius / Diameter / WPath -- descriptors built on RDKit's
    distance matrix, which uses 1e8 for "no path between these atoms". A disconnected molecule is
    exactly what that sentinel means. The fingerprint arms absorbed it quietly, at plausible
    mid-range values, which is why it took a fourth arm to see it.

    Checked against the INPUT's fragment count rather than against 1, so a benchmark molecule that
    arrives as a salt is not rejected for a fault it had before the edit.
    """
    return len(Chem.GetMolFrags(after)) == len(Chem.GetMolFrags(before))


def main() -> None:
    random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
    pool = [s for s in d["smiles"]]
    random.shuffle(pool)
    mols = [(s, _mol(s)) for s in pool]
    mols = [(s, m) for s, m in mols if m is not None and 8 <= m.GetNumAtoms() <= 60]
    raw = len(mols)

    # DEDUPLICATE ON THE CANONICAL SMILES, and key everything downstream by it. The benchmark
    # pools 34 datasets and the same compound appears in several of them under different input
    # strings; pairs are written canonicalised, so two rows that differ as strings become the
    # same molecule the moment they are compared. Splitting before deduplicating put 700 of them
    # on both sides of the background/pair boundary -- caught by the disjointness assert below,
    # which is the whole reason it is an assert and not a comment.
    seen, uniq = set(), []
    for _, m in mols:
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen.add(c)
            uniq.append((c, m))
    mols = uniq
    print(f"pool: {len(mols):,} distinct benchmark molecules ({raw - len(mols):,} duplicates "
          f"collapsed on canonical SMILES)")

    # THE BACKGROUND IS RESERVED FIRST, AND IS NOT A FUNCTION OF THE PAIR SET.
    #
    # sigma_j -- the per-dimension spread that puts every representation on one axis -- is
    # estimated on these molecules. They used to be chosen AFTER pair generation, as "whatever
    # benchmark molecules are left over", which quietly made the denominator depend on which
    # modes existed: adding the C=C saturation panel changed the leftovers, and the descriptor
    # arm's protonation cell moved 1.199 -> 0.979 (18%) with no change to the pairs it was
    # measured on. The fingerprint arms moved <2%, because descriptor sigma is heavy-tailed and a
    # handful of dimensions carry the normalisation.
    #
    # Reserving up front makes disjointness a property of the construction rather than of a
    # filter, and freezes the reference sample: adding, removing or re-tuning a mode can no
    # longer move a number in any other panel.
    background = [s for s, _ in mols[:N_BACKGROUND]]
    mols = mols[N_BACKGROUND:]
    json.dump(background, open(OUT / "background.json", "w"))
    print(f"background: {len(background):,} reserved (excluded from every pair); "
          f"{len(mols):,} left for pairs")

    pairs, counts = [], {}
    shattered = {k: 0 for k in EDITS}       # rejected by _intact -- REPORTED, never silent
    collided = {k: 0 for k in EDITS}        # product landed on a reserved background molecule
    bgset = set(background)
    for name, fn in EDITS.items():
        got = 0
        for s, m in mols:
            if got >= N_PER_EDIT:
                break
            try:
                e = fn(m)
                if e is None:
                    continue
                # An edit may return a single product -- A is then the untouched input -- or a
                # (A, B) pair when it needs to control BOTH sides. `isotope_13c` and `protonate`
                # need that: writing A in the same bracket notation as B is what stops the panel
                # from being solvable by character counting alone.
                ma, mb = e if isinstance(e, tuple) else (m, e)
                Chem.SanitizeMol(mb)
                if ma is not m:
                    Chem.SanitizeMol(ma)
                a, b = Chem.MolToSmiles(ma), Chem.MolToSmiles(mb)
            except Exception:
                continue
            if a == b:                     # the edit was a no-op on this molecule
                continue
            if not _intact(ma, mb):
                shattered[name] += 1
                continue
            # The PRODUCT can collide with the background even though the input cannot: the
            # benchmark contains real matched pairs, so saturating a C=C in one compound
            # sometimes lands exactly on another compound already reserved for sigma. Reject the
            # pair and keep scanning -- 45k candidates for 1,000 slots, so backfilling is free,
            # and shrinking the background instead would make its size depend on the mode list
            # again.
            if b in bgset:
                collided[name] += 1
                continue
            pairs.append({"edit": name, "cls": "chemical", "a": a, "b": b})
            got += 1
        counts[name] = got
        print(f"  {name:16s} {got:5d}"
              + (f"   ({shattered[name]} rejected: fragmented)" if shattered[name] else "")
              + (f"   ({collided[name]} rejected: product is a background molecule)"
                 if collided[name] else "")
              + ("" if got >= N_PER_EDIT else "   <- SHORT"))

    for name, fn in (("null_enumerate", null_enumerate), ("null_kekulize", null_kekulize)):
        got = 0
        for s, m in mols:
            if got >= N_PER_EDIT:
                break
            try:
                b = fn(m)
                if not b or Chem.MolFromSmiles(b) is None:
                    continue
            except Exception:
                continue
            # `a` is written as-supplied too, so BOTH sides of a null pair are strings that
            # differ only in notation. Canonicalising `a` here would make the control test
            # "canonical vs non-canonical" rather than "notation A vs notation B".
            pairs.append({"edit": name, "cls": "null", "a": Chem.MolToSmiles(m), "b": b})
            got += 1
        counts[name] = got
        print(f"  {name:16s} {got:5d}   (null control -- must score 0)")

    # Matched-MW substitution: the per-model scale. Sorting by MW and pairing with a molecule
    # of near-identical mass keeps size out of the reference, so "moves as far as changing the
    # molecule" is not confounded by "moves as far as changing the size".
    byw = sorted(((Descriptors.MolWt(m), s) for s, m in mols[:8000]), key=lambda x: x[0])
    got = 0
    for i in range(0, len(byw) - 1, 2):
        if got >= N_PER_EDIT:
            break
        (w1, s1), (w2, s2) = byw[i], byw[i + 1]
        if abs(w1 - w2) < 1.0 and s1 != s2:
            pairs.append({"edit": "matched_mw", "cls": "reference", "a": s1, "b": s2})
            got += 1
    counts["matched_mw"] = got
    print(f"  {'matched_mw':16s} {got:5d}   (reference -- defines 1.00)")

    # Disjointness is asserted, not assumed. If the background ever overlapped the pairs, every
    # edit would inflate its own denominator and pull every arm toward the same score.
    inpair = {s for p in pairs for s in (p["a"], p["b"])}
    overlap = inpair & set(background)
    assert not overlap, f"{len(overlap)} background molecules appear in a pair"

    json.dump({"counts": counts, "pairs": pairs}, open(OUT / "pairs.json", "w"))
    print(f"\n{len(pairs):,} pairs across {len(counts)} conditions -> {OUT / 'pairs.json'}")


if __name__ == "__main__":
    main()
