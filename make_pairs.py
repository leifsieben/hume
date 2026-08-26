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
    cs = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsotope() == 0]
    if not cs:
        return None
    e = Chem.RWMol(m)
    e.GetAtomWithIdx(random.choice(cs)).SetIsotope(13)
    return e.GetMol()


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
    for a, b in (("[NX3;H0;!$(N=*);!$(N-[!#6]);!$(N-C=O);!$(N-a);!+]", "[NH+]"),
                 ("[CX3](=O)[OX2H1]", "C(=O)[O-]")):
        r = _sub(m, a, b)
        if r is not None:
            return r
    return None


EDITS = {"stereo_flip": stereo_flip, "ez_flip": ez_flip, "halogen_swap": halogen_swap,
         "h_to_methyl": h_to_methyl, "n_methylation": n_methylation, "isotope_13c": isotope_13c,
         "scaffold_hop": scaffold_hop, "ring_contract": ring_contract,
         "protonate": protonate, "regioisomer": regioisomer}


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
    print(f"pool: {len(mols):,} usable benchmark molecules")

    pairs, counts = [], {}
    shattered = {k: 0 for k in EDITS}       # rejected by _intact -- REPORTED, never silent
    for name, fn in EDITS.items():
        got = 0
        for s, m in mols:
            if got >= N_PER_EDIT:
                break
            try:
                e = fn(m)
                if e is None:
                    continue
                Chem.SanitizeMol(e)
                a, b = Chem.MolToSmiles(m), Chem.MolToSmiles(e)
            except Exception:
                continue
            if a == b:                     # the edit was a no-op on this molecule
                continue
            if not _intact(m, e):
                shattered[name] += 1
                continue
            pairs.append({"edit": name, "cls": "chemical", "a": a, "b": b})
            got += 1
        counts[name] = got
        print(f"  {name:16s} {got:5d}"
              + (f"   ({shattered[name]} rejected: edit fragmented the molecule)"
                 if shattered[name] else "")
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

    json.dump({"counts": counts, "pairs": pairs}, open(OUT / "pairs.json", "w"))
    print(f"\n{len(pairs):,} pairs across {len(counts)} conditions -> {OUT / 'pairs.json'}")


if __name__ == "__main__":
    main()
