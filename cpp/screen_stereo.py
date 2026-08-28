"""Are the three stereo-dependent columns FUNCTIONS OF THE MOLECULE, or of its numbering?

    NumAtomStereoCenters   NumUnspecifiedAtomStereoCenters   SPS

House rule 1 in PORT_STATUS.md: perturb the input ordering and recompute; any column that moves
is ill-posed and there is nothing there to be exact against. The screen must shuffle BONDS and
not only atoms, because RDKit's perception reads the bond list and an atom-only screen
under-samples the axis the answer depends on.

THE OPEN PROBLEM THIS FILE EXISTS TO CLOSE, recorded in PORT_STATUS.md. The repo's existing
atom+bond rebuild (`cpp/verify_ic.py:rebuilt`, used by `cpp/screen_constit.py`) is NOT
chirality-preserving: it copies each atom's chiral tag verbatim while permuting the bond order
that the tag is defined against. So it cannot answer this question -- it would report a
perfectly well-posed stereo column as moving, because the perturbation itself changed the
stereochemistry. `screen_constit.py` works around that by SKIPPING `SPS` whenever the rebuild
lost stereo, which means `SPS` was only ever screened on the stereo-poor molecules.

TWO PARITY-PRESERVING SHUFFLES ARE USED HERE INSTEAD, and neither is taken on trust: every
perturbed molecule is checked against the original by ISOMERIC canonical SMILES, and one that
does not reproduce it is EXCLUDED AND COUNTED rather than compared.

  atom+bond, parity-repaired   the same RWMol rebuild, plus the repair it was missing. An atom's
                               chiral tag is defined against the ORDER OF ITS BONDS, so permuting
                               the bond list permutes that reference frame; the tag must be
                               flipped exactly when the induced permutation of the atom's own
                               incident bonds is ODD. Bond stereo (E/Z) is carried across too --
                               it lives on the bond's STEREO ATOMS, which the old rebuild dropped.
  random-SMILES round trip     `MolToSmiles(canonical=False, doRandom=True)` and back. RDKit's own
                               writer and parser, so the parity bookkeeping is RDKit's rather than
                               this file's; a random root and random branch order give a different
                               atom order AND a different bond order. This is NOT the canonical
                               round trip PORT_STATUS calls a control -- that one reproduces the
                               canonical numbering and is therefore not a perturbation at all.

Plus the two axes the repo already runs, for continuity: atom-only `RenumberAtoms`, and a Kekule
round trip (which is a chemistry re-perception, not a numbering perturbation, and is reported on
its own line for that reason).

    .venv/bin/python cpp/screen_stereo.py [n_mols]
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.SpacialScore import SPS

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

COLS = ("NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters", "SPS")

_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW


def values(m):
    """The three columns, from RDKit itself, on a molecule nothing else has touched.

    A FRESH `Chem.Mol(m)` PER DESCRIPTOR, and it is not defensive. `NumAtomStereoCenters` is a
    function of what the last `AssignStereochemistry` left on the molecule, and `SPS` runs
    `FindPotentialStereoBonds` on a copy but `FindMolChiralCenters` writes `_ChiralityPossible`
    on whatever it is handed. Sharing one object between the three would let the first call decide
    the third's answer.
    """
    return (float(Descriptors.NumAtomStereoCenters(Chem.Mol(m))),
            float(Descriptors.NumUnspecifiedAtomStereoCenters(Chem.Mol(m))),
            float(SPS(Chem.Mol(m))))


def _parity_odd(old: list[int], new: list[int]) -> bool:
    """Is the permutation taking `old` to `new` odd? Both are the same set, in two orders."""
    pos = {v: i for i, v in enumerate(new)}
    perm = [pos[v] for v in old]
    seen = [False] * len(perm)
    swaps = 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, ln = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            ln += 1
        swaps += ln - 1
    return swaps % 2 == 1


FLIPS = [0, 0]     # [rebuilds that flipped at least one tag, rebuilds with a tag to flip]


def rebuilt_parity(m, aidx, bperm):
    """Renumber the atoms AND add the bonds in a shuffled order, KEEPING the stereochemistry.

    -> the molecule, or None if it did not reproduce the original (isomeric canonical SMILES).

    THE REPAIR IS THE POINT. `Atom::getChiralTag()` is CW/CCW *with respect to the order of the
    atom's incident bonds*, so a rebuild that reorders the bond list and copies the tag has
    silently inverted every centre whose bond permutation is odd. Half of them, on average. That
    is what makes the repo's existing rebuild useless for a stereo column and is why this one
    counts the parity and flips.

    IMPLICIT HYDROGENS ARE NOT IN THE BOND LIST, so they cannot be permuted and need no repair --
    an implicit H keeps its position relative to every explicit neighbour under any reordering of
    those neighbours. The isomeric-SMILES check below is what makes that an observation rather
    than an assumption: if it were wrong, the `[C@H]` centres would fail it en masse.
    """
    r = Chem.RenumberAtoms(m, aidx)
    Chem.SanitizeMol(r)
    e = Chem.RWMol()
    for a in r.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsotope(a.GetIsotope())
        na.SetNumRadicalElectrons(a.GetNumRadicalElectrons())
        na.SetNoImplicit(a.GetNoImplicit())
        na.SetNumExplicitHs(a.GetNumExplicitHs())
        na.SetChiralTag(a.GetChiralTag())
        na.SetIsAromatic(a.GetIsAromatic())
        e.AddAtom(na)
    bonds = list(r.GetBonds())
    later = []
    for k in bperm:
        b = bonds[k]
        j = e.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) - 1
        nb = e.GetBondWithIdx(j)
        nb.SetIsAromatic(b.GetIsAromatic())
        # E/Z IN A SECOND PASS, and that is a hard requirement rather than tidiness:
        # `SetStereoAtoms` asserts that the named reference atoms are already BONDED to the
        # double bond's ends, and under a shuffled insertion order those bonds may not exist yet.
        later.append((j, list(b.GetStereoAtoms()), b.GetStereo(), b.GetBondDir()))
    for j, sa, st, bd in later:
        nb = e.GetBondWithIdx(j)
        if len(sa) == 2:
            nb.SetStereoAtoms(sa[0], sa[1])
        nb.SetStereo(st)
        nb.SetBondDir(bd)
    out = e.GetMol()

    # ---- the parity repair -------------------------------------------------------------------
    # out's bond j is r's bond bperm[j]; map back so the two neighbour orders are comparable.
    r_of_new = {j: bperm[j] for j in range(len(bperm))}
    n_tagged = n_flipped = 0
    for a in r.GetAtoms():
        if a.GetChiralTag() not in (_CW, _CCW):
            continue
        n_tagged += 1
        i = a.GetIdx()
        old = [b.GetIdx() for b in a.GetBonds()]
        new = [r_of_new[b.GetIdx()] for b in out.GetAtomWithIdx(i).GetBonds()]
        if _parity_odd(old, new):
            n_flipped += 1
            na = out.GetAtomWithIdx(i)
            na.SetChiralTag(_CCW if a.GetChiralTag() == _CW else _CW)
    # THE CONTROL FOR "IS THIS AXIS DOING ANYTHING". If the bond shuffle never reordered the
    # bonds around a tagged atom, the repair would never fire and a clean screen would be
    # vacuous. `FLIPS` counts how often it did fire -- and every one of those is a centre the
    # repo's existing un-repaired rebuild would have INVERTED.
    FLIPS[1] += n_tagged > 0
    FLIPS[0] += n_flipped > 0

    try:
        Chem.SanitizeMol(out)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True,
                                   flagPossibleStereoCenters=True)
    except Exception:                                             # noqa: BLE001
        return None
    if Chem.MolToSmiles(out) != Chem.MolToSmiles(r):
        return None
    return out


def random_smiles_roundtrip(m, rng):
    """A random-order SMILES and back. RDKit's own parity bookkeeping, not this file's."""
    try:
        s = Chem.MolToSmiles(m, canonical=False, doRandom=True)
        r = Chem.MolFromSmiles(s)
    except Exception:                                             # noqa: BLE001
        return None
    if r is None or Chem.MolToSmiles(r) != Chem.MolToSmiles(m):
        return None
    return r


def kekule_roundtrip(m):
    """cpp/screen_constit.py's, verbatim: a chemistry re-perception, reported separately."""
    try:
        k = Chem.Mol(m)
        Chem.Kekulize(k, clearAromaticFlags=True)
        r = Chem.MolFromSmiles(Chem.MolToSmiles(k, kekuleSmiles=True))
    except Exception:                                             # noqa: BLE001
        return None
    if r is None or Chem.MolToSmiles(r) != Chem.MolToSmiles(m):
        return None
    return r


def renumbered(m, idx):
    r = Chem.RenumberAtoms(m, idx)
    Chem.SanitizeMol(r)
    return r


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    canary = Descriptors.BCUT2D_MRLOW(Chem.MolFromSmiles(
        "O=C1CCNCCNNNCCNCCC(=O)c2ccc(o2)COCOCc2ccc1o2"))
    print(f"rdkit {rdkit.__version__}   numpy {np.__version__}   "
          f"python {sys.version.split()[0]}   CANARY {canary!r}")
    if canary != -0.07665884800196521:
        raise SystemExit(f"CANARY MISMATCH: {canary!r}")

    smis = [s for s in (ROOT / "cpp" / "hard.smi").read_text().split("\n") if s]
    rng = np.random.default_rng(0)
    pick = [smis[i] for i in rng.choice(len(smis), min(len(smis), n), replace=False)]
    mols = [Chem.MolFromSmiles(s) for s in pick]
    if any(m is None for m in mols):
        raise ValueError("unparseable SMILES in the corpus")
    print(f"{len(mols)} molecules from cpp/hard.smi\n")

    base = [values(m) for m in mols]
    n_stereo = sum(1 for b in base if b[0] > 0)
    print(f"molecules with at least one stereocentre: {n_stereo} / {len(mols)}"
          f"   ({100.0 * n_stereo / len(mols):.1f}%)\n")

    prng = random.Random(20260828)
    AXES = [("atom only", "atom", 3),
            ("atom+bond parity-repaired", "rebuild", 3),
            ("random-SMILES round trip", "smiles", 3),
            ("kekule round trip", "kekule", 1)]

    moved = defaultdict(lambda: defaultdict(set))    # axis -> column -> {mol index}
    excluded = defaultdict(int)
    probed = defaultdict(int)
    example = {}
    for label, kind, passes in AXES:
        for _p in range(passes):
            for i, m in enumerate(mols):
                if kind == "atom":
                    idx = list(range(m.GetNumAtoms()))
                    prng.shuffle(idx)
                    r = renumbered(m, idx)
                elif kind == "rebuild":
                    idx = list(range(m.GetNumAtoms()))
                    prng.shuffle(idx)
                    bo = list(range(m.GetNumBonds()))
                    prng.shuffle(bo)
                    r = rebuilt_parity(m, idx, bo)
                elif kind == "smiles":
                    r = random_smiles_roundtrip(m, prng)
                else:
                    r = kekule_roundtrip(m)
                if r is None:
                    excluded[label] += 1
                    continue
                probed[label] += 1
                v = values(r)
                for c, a, b in zip(COLS, v, base[i]):
                    if a != b:
                        moved[label][c].add(i)
                        example.setdefault((label, c), (pick[i], b, a))
        print(f"  pass done: {label}", flush=True)

    print(f"\n{'axis':30s} {'probed':>8s} {'excluded':>9s}   "
          + "  ".join(f"{c[:22]:>22s}" for c in COLS))
    bad = 0
    for label, _kind, _p in AXES:
        cells = []
        for c in COLS:
            k = len(moved[label][c])
            cells.append(f"{k:>22d}")
            if k:
                bad += 1
        print(f"{label:30s} {probed[label]:>8d} {excluded[label]:>9d}   " + "  ".join(cells))

    print("\nEXCLUDED = the perturbation did not reproduce the molecule (isomeric canonical "
          "SMILES), so it was not compared. A high number here means the axis is weak, not that "
          "the column is stable.")
    print(f"\nTHE PARITY REPAIR FIRED on {FLIPS[0]} of the {FLIPS[1]} atom+bond rebuilds that "
          f"had a tetrahedral tag to reorder. Every one of those is a centre the repo's existing "
          f"un-repaired rebuild (cpp/verify_ic.py:rebuilt) would have INVERTED -- which is why "
          f"an unrepaired bond shuffle cannot screen a stereo column, and why "
          f"cpp/screen_constit.py had to skip SPS on exactly the stereo-rich molecules.")
    if example:
        print("\ndiscriminating examples:")
        for (label, c), (smi, was, now) in sorted(example.items()):
            print(f"  {label:30s} {c:32s} {was} -> {now}   {smi}")
    else:
        print("\nno column moved on any axis: all three are functions of the molecule, not of "
              "its numbering.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
