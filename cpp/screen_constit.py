"""HOUSE RULE 1, run BEFORE anything is implemented: is each of the 49 small-constitutional
columns a FUNCTION OF THE MOLECULE, or of the numbering the parser happened to hand back?

Run in the pinned oracle env, and only there:

    uv run --isolated --no-project --python 3.11 --with "mordred==1.2.0" \
           --with "rdkit==2025.9.2" --with "numpy==1.26.4" python cpp/screen_constit.py 2000

THE THREE AXES, and why an atom-only screen would have been worthless here.

  * atom renumbering only          Chem.RenumberAtoms + re-sanitize.
  * atom renumbering AND bond-list shuffle    the molecule is REBUILT, bonds added in a shuffled
    order.  `Chem.RenumberAtoms` permutes atoms and LEAVES THE BOND LIST ORDER ALONE, and RDKit's
    ring perception and its KEKULIZER both read the bond list -- so the atom-only axis cannot move
    a Kekule-dependent column at all.
  * Kekule round trip              MolToSmiles(kekuleSmiles=True) re-parsed.  A second, blunter
    probe of the same axis: it hands the parser a structure in which the double bonds are already
    placed, so a column that reads GetBondType() after Chem.Kekulize() can see a DIFFERENT matching
    from the one the aromatic parse would have produced.

This block has two columns that are Kekule-dependent BY CONSTRUCTION and are the reason the third
axis is here rather than assumed away: mordred's `nBondsKD` counts DOUBLE bonds after
`Chem.Kekulize(m)`, and `CarbonTypes` sets `kekulize = True`.  Whether the COUNT moves is an
empirical question -- a perfect matching's cardinality is fixed by the atom set, but which atoms
RDKit leaves unmatched is not obviously fixed -- so it is measured, not argued.

The canonical-SMILES round trip is deliberately NOT one of the axes: it reproduces the canonical
numbering, so it is a control that should show zero, not a perturbation.  cpp/verify_ic.py already
carries that control for the corpus.
"""
from __future__ import annotations

import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Chem.SpacialScore import SPS

from verify_ic import parse_all, rebuilt, renumbered  # the SAME perturbation code the IC screen
                                                      # uses; a second copy could drift from it

RDLogger.DisableLog("rdApp.*")

MORDRED_COLS = [
    "C1SP1", "C2SP1", "C1SP2", "C2SP2", "C3SP2", "C1SP3", "C2SP3", "C3SP3", "C4SP3",
    "nH", "nB", "nC", "nN", "nO", "nS", "nCl", "nBr",
    "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM", "nBondsKD",
    "Kier1", "Kier2", "Kier3",
    "MDEC-22", "MDEC-23", "MDEC-33",
    "RNCG", "RPCG",
    "Lipinski", "GhoseFilter",
    "nAcid", "nBase",
    "Vabc", "RotRatio", "bpol", "FilterItLogS", "fMF", "fragCpx",
    "TopoPSA", "SLogP", "PEOE_VSA11", "SMR_VSA1", "EState_VSA1",
]
RDKIT_COLS = ["TPSA", "qed", "SPS"]


def versions():
    import numpy
    import mordred
    import rdkit
    print("RESOLVED: mordred %s  rdkit %s  numpy %s  python %s"
          % (mordred.__version__, rdkit.__version__, numpy.__version__,
             sys.version.split()[0]))
    # A version banner is not evidence (PORT_STATUS house rule 3): check a number only the real
    # library can produce.  CalcTPSA on phenol is 20.23 in every rdkit that has ever shipped it,
    # so this canary is about the dylib actually being loaded, not about the version string.
    from rdkit.Chem import rdMolDescriptors
    v = rdMolDescriptors.CalcTPSA(Chem.MolFromSmiles("c1ccccc1O"))
    assert abs(v - 20.23) < 1e-12, "rdkit canary failed: %r" % v


def make_calc():
    from mordred import Calculator, descriptors as mdesc
    full = Calculator(mdesc, ignore_3D=True)
    want = set(MORDRED_COLS)
    keep = [d for d in full.descriptors if str(d) in want]
    got = {str(d) for d in keep}
    missing = want - got
    assert not missing, "not in the mordred preset: %s" % sorted(missing)
    return Calculator(keep, ignore_3D=True)


def row(calc, m):
    """One molecule -> {name: value}. Errors become the string 'ERR' so they compare equal to
    each other and unequal to a number -- an error appearing or disappearing under renumbering is
    itself ill-posedness and must not be silently smoothed into a NaN."""
    out = {}
    res = calc(m)
    for d, v in zip(calc.descriptors, res):
        try:
            out[str(d)] = float(v)
        except Exception:
            out[str(d)] = "ERR"
    from rdkit.Chem import rdMolDescriptors
    out["TPSA"] = float(rdMolDescriptors.CalcTPSA(m))
    try:
        out["qed"] = float(QED.qed(m))
    except Exception:
        out["qed"] = "ERR"
    try:
        out["SPS"] = float(SPS(m))
    except Exception:
        out["SPS"] = "ERR"
    return out


# WHY THIS IS A RELATIVE DEVIATION AND NOT `a == b`, recorded because the first run of this
# screen reported 15 ill-posed columns and 13 of them were an artefact of asking the wrong
# question.  TPSA, SLogP, TopoPSA, bpol, Vabc and the three VSA columns are SUMS OVER ATOMS IN
# ATOM ORDER.  Renumbering the atoms reorders the summation, and float64 addition is not
# associative, so they move in the last bit or two on most molecules -- 88 of 150 for TPSA.  That
# is not the descriptor failing to be a function of the molecule; it is the same real number
# rounded down two different accumulation trees, and it is bounded by n * eps.
#
# So the screen measures the SIZE of the movement and reports the distribution.  A column whose
# worst movement is ~1e-16 relative is well-posed and summation-order sensitive; a column that
# moves by 1e-3 or 0.5 or changes an integer is ill-posed and needs house rule 1's treatment.
# The two are separated by ten orders of magnitude on this corpus, so the classification is not
# a judgement call -- but the threshold is reported next to the numbers rather than hidden.
ILLPOSED_RTOL = 1e-9


def deviation(a, b):
    """-> relative deviation, or float('inf') for a categorical mismatch."""
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else float("inf")
    if a != a and b != b:      # NaN == NaN, for this purpose
        return 0.0
    if a != a or b != b:
        return float("inf")
    if a == b:
        return 0.0
    d = abs(a - b)
    s = max(abs(a), abs(b))
    return d / s if s > 0 else d


def kekule_roundtrip(m):
    """Re-parse from a KEKULE smiles. Returns None if it does not round-trip to the same molecule.

    THE GUARD IS ISOMERIC AND THAT IS NOT COSMETIC.  The first version of this function wrote the
    smiles with `isomericSmiles=False` and then compared two isomeric-False smiles, so a round
    trip that DROPPED THE ISOTOPE LABELS compared equal to the original.  It reported
    FilterItLogS, qed and SPS as ill-posed on 25-26 molecules, and every discriminating example
    contained a `[13C]` or a `[125I]` -- i.e. the screen had silently replaced the molecule with
    a different one and then blamed the descriptor.  Isotopes and stereo are written and compared.
    """
    try:
        k = Chem.Mol(m)
        Chem.Kekulize(k, clearAromaticFlags=True)
        s = Chem.MolToSmiles(k, kekuleSmiles=True)
        r = Chem.MolFromSmiles(s)
    except Exception:
        return None
    if r is None:
        return None
    if Chem.MolToSmiles(r) != Chem.MolToSmiles(m):
        return None
    return r


def rebuilt_same_molecule(m, idx, bo):
    """`verify_ic.rebuilt`, plus a flag saying whether STEREO survived.

    -> (mol or None, stereo_kept)

    ONE COLUMN OF THE 49 READS STEREO and 48 do not.  rdkit's SPS multiplies an atom's score by 2
    when it is a (pseudo)stereocentre or sits on a stereo double bond, so a rebuild that dropped
    E/Z would show up as SPS "moving" when what moved was the input.  Everything else -- ring
    perception, kekulisation, element counts, charges -- is blind to stereo.

    SO THE GUARD IS PER COLUMN, NOT PER MOLECULE, AND THAT MATTERS MORE THAN IT SOUNDS.  An
    earlier version of this file rejected any rebuild that lost stereo and fell back to
    ATOM-ONLY renumbering for it -- 882 of 6,000 rebuilds on a 2,000-molecule run, concentrated
    exactly on the stereo-rich molecules.  Atom-only renumbering is the axis PORT_STATUS says is
    too weak, so those 882 molecules were being screened with the weak probe while the report
    said "atom+bond".  It showed up downstream: three molecules whose mordred `Vabc` looked
    STABLE over 400 perturbations under that guard turn out to give three different answers each
    the moment the stereo-blind guard lets the bond shuffle actually happen.  The rebuild is
    therefore kept for all 48 stereo-blind columns and only SPS is skipped when stereo moved.
    """
    r = rebuilt(m, idx, bo)
    if r is None:
        return None, False
    return r, Chem.MolToSmiles(r) == Chem.MolToSmiles(m)


def main(n):
    versions()
    calc = make_calc()
    mols, smis = parse_all(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard.smi"), n)
    print("molecules: %d" % len(mols))

    base = [row(calc, m) for m in mols]
    cols = MORDRED_COLS + RDKIT_COLS
    moved = defaultdict(set)          # column -> {(mol index, axis)} moving by > ILLPOSED_RTOL
    jitter = defaultdict(set)         # column -> {mol index} moving at all
    worst = defaultdict(float)        # column -> largest relative deviation seen
    worst_mol = {}
    rng = random.Random(20260827)
    nrebuild_fail = 0
    nstereo_lost = 0
    nkek_fail = 0

    PASSES = [("atom only", 0), ("atom only", 0), ("atom+bond", 1), ("atom+bond", 1),
              ("atom+bond", 1), ("kekule round trip", 2)]
    for pname, kind in PASSES:
        for i, m in enumerate(mols):
            stereo_kept = True
            if kind == 2:
                r = kekule_roundtrip(m)
                if r is None:
                    nkek_fail += 1
                    continue
            else:
                idx = list(range(m.GetNumAtoms()))
                rng.shuffle(idx)
                if kind == 1:
                    bo = list(range(m.GetNumBonds()))
                    rng.shuffle(bo)
                    r, kept = rebuilt_same_molecule(m, idx, bo)
                    if r is None:
                        nrebuild_fail += 1
                        r = renumbered(m, idx)
                    else:
                        stereo_kept = kept
                        if not kept:
                            nstereo_lost += 1
                else:
                    r = renumbered(m, idx)
            rv = row(calc, r)
            for c in cols:
                # SPS is the only column that reads stereo; skip it when the rebuild moved stereo.
                if c == "SPS" and not stereo_kept:
                    continue
                d = deviation(base[i][c], rv[c])
                if d > 0.0:
                    jitter[c].add(i)
                if d > worst[c]:
                    worst[c] = d
                    worst_mol[c] = i
                if d > ILLPOSED_RTOL:
                    moved[c].add((i, pname))
        print("  pass done: %s" % pname)

    print("\nrebuilds that did not reproduce the molecule (fell back to atom-only): %d"
          % nrebuild_fail)
    print("rebuilds that reproduced the graph but lost stereo (SPS skipped, 48 columns kept): %d"
          % nstereo_lost)
    print("kekule round trips that did not reproduce the molecule (skipped): %d" % nkek_fail)
    print("\n%-14s %7s %7s %11s  %s"
          % ("column", "illposd", "jitter", "max rel dev", "axes / example"))
    bad = []
    for c in cols:
        s = moved[c]
        tag = ""
        if s:
            bad.append(c)
            axes = sorted({p for _, p in s})
            ex = sorted({i for i, _ in s})[:3]
            tag = "  %s  e.g. %s" % (",".join(axes), [smis[i] for i in ex])
        print("%-14s %7d %7d %11.3g%s"
              % (c, len({i for i, _ in s}), len(jitter[c]), worst[c], tag))
    print("\nthreshold for ILLPOSED is relative deviation > %g" % ILLPOSED_RTOL)
    print("ILL-POSED COLUMNS: %d of %d  %s" % (len(bad), len(cols), bad))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
