"""What is RDKit->array extraction actually made of, and what is left of it?

THE QUESTION THIS ANSWERED. Extraction was 43% of `featurize_blocks`, and that ratio decides an
architecture question, so it mattered what the time WAS and not merely how big it was:

  * If it is PYTHON CALL OVERHEAD -- `atom.GetAtomicNum()` and friends crossing the boundary
    ~10 times per atom, ~300 times per molecule -- it is cheap to fix by getting the same data
    out in bulk, and the "bit-identical to RDKit" claim is untouched.
  * If it is GENUINE PERCEPTION WORK inside RDKit (Gasteiger's iterative charges, Crippen's
    SMARTS typer, CIP labelling), it is irreducible from Python, and the only lever left is
    linking RDKit's C++ directly -- which costs wheels.

It was both, and the two were addressed differently. Crippen -- 93 us/mol cold on this box, the
largest single item and by far the largest RDKit-side one -- was the second case, and it left
Python entirely: src/hume_core/crippen_typer.h now answers it inside the extension for 1.4.
Everything else was the first case and is now bulk-extracted. Extraction went 231 -> 92 us/mol,
verified bit-identical on all 182 columns over the 98,905-molecule corpus.

THAT LEVER HAS NOW BEEN TAKEN, and this file profiles the path it replaced. The two `wrapper
list, no reads` rows below build the atom and bond object lists and read NOTHING off them; they
are the floor for any approach that touches an RDKit object from Python. `extract_pickles` does
not touch one: `m.ToBinary(PrivateProps | AtomProps | ComputedProps)` serialises the molecule and
src/hume_core/molpickle.h parses it, bit-identically on both corpora (cpp/verify_molpickle.py).
extract() stays as the reference implementation and this stays its profile -- and the two now
share one line that neither used to have, the ring CSR, which is a Python cost on BOTH paths.
ComputedProps is not optional there: `_GasteigerCharge` is a computed property, so the obvious
`PrivateProps | AtomProps` pair pickles CIP codes and silently no charges at all.

METHOD. Each component is timed in its own pass over the same molecules, in-process, with the
order rotated across cycles so machine drift is common-mode. CPU time, not wall: the bridge work
found wall-clock spreads of 26-47x on this shared box, larger than anything being compared.

COLD, NOT WARM. RDKit caches results on the molecule, so a second pass over the same molecules
measures a dict lookup instead of the work. That is not a hypothetical: the pre-change extract()
costs 231 us/mol on freshly parsed molecules and 130 on molecules it has already seen, and the
gap is exactly the Crippen SMARTS pass being free from the second repetition onwards. Every
component here therefore re-parses the molecules and times only what follows the parse, and
c_crippen_rdkit additionally passes force=True.

    .venv/bin/python tools/profile_extract.py [n_mols]
"""
from __future__ import annotations

import sys
import time
from itertools import repeat
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdPartialCharges

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molhume._extract import _rings_csr, extract  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

_A = Chem.Atom


def _cpu():
    return time.process_time()


def bench(fn, smis, reps=3):
    """-> us per molecule, CPU time, best of `reps`, on FRESH molecules with the parse removed."""
    best = float("inf")
    for _ in range(reps):
        mols = [Chem.MolFromSmiles(s) for s in smis]
        t0 = _cpu()
        fn(mols)
        t1 = _cpu()
        best = min(best, (t1 - t0) / len(smis) * 1e6)
        del mols
    return best


# ---- the components, each doing exactly what _extract.py does and nothing else ----------
def c_gasteiger(mols):
    for m in mols:
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
        except Exception:
            pass


def c_stereo(mols):
    for m in mols:
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)


def c_atom_list(mols):
    """Wrapper construction ALONE -- the floor for anything that touches an atom from Python."""
    for m in mols:
        list(map(m.GetAtomWithIdx, range(m.GetNumAtoms())))


def c_atom_cols(mols):
    """The eight per-atom column passes, on lists that already exist."""
    for m in mols:
        ats = list(map(m.GetAtomWithIdx, range(m.GetNumAtoms())))
        list(map(_A.GetAtomicNum, ats))
        list(map(_A.GetDegree, ats))
        list(map(_A.GetTotalNumHs, ats))
        list(map(_A.GetFormalCharge, ats))
        list(map(_A.GetHybridization, ats))
        list(map(_A.GetIsAromatic, ats))
        list(map(_A.IsInRing, ats))
        list(map(_A.GetMass, ats))


def c_charge_read(mols):
    for m in mols:
        rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
        ats = list(map(m.GetAtomWithIdx, range(m.GetNumAtoms())))
        list(map(_A.GetDoubleProp, ats, repeat("_GasteigerCharge")))


def c_cip(mols):
    for m in mols:
        ats = list(map(m.GetAtomWithIdx, range(m.GetNumAtoms())))
        list(map(_A.HasProp, ats, repeat("_CIPCode")))


def c_bond_list(mols):
    for m in mols:
        list(map(m.GetBondWithIdx, range(m.GetNumBonds())))


def c_bond_loop(mols):
    for m in mols:
        for b in map(m.GetBondWithIdx, range(m.GetNumBonds())):
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetIsConjugated(), b.IsInRing(),
             b.GetBondType(), b.GetIsAromatic(), b.GetStereo())


def c_crippen_rdkit(mols):
    """NO LONGER CALLED. Kept because it is the thing that was removed, and the number it
    prints is what the removal was worth. force=True is load-bearing: RDKit caches the
    contributions on the molecule, and this project has already reported an 82x-wrong Crippen
    number by timing the cache."""
    for m in mols:
        rdMolDescriptors._CalcCrippenContribs(m, force=True)


def c_rings(mols):
    """The ring CSR, which extract() now also builds. It is NOT a read of RDKit properties: it
    runs src/molhume/_rings.py's gate on every molecule and its canonical re-perception on the ~20%
    the gate fires on, so it belongs in the table on its own line rather than inside the residual.
    See the note on `Rings` in _extract.py for why the ring SET is carried at all."""
    _rings_csr(mols)


def c_extract(mols):
    extract(mols)


def main() -> None:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    smis = ROOT.joinpath("cpp", "mols.smi").read_text().split()
    rng = np.random.default_rng(0)
    pick = [smis[i] for i in rng.choice(len(smis), min(len(smis), n_want), replace=False)]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in pick) if m is not None]
    natoms = float(np.mean([m.GetNumAtoms() for m in mols]))
    print(f"{len(mols)} molecules, mean {natoms:.1f} heavy atoms\n")
    del mols

    # (label, fn, kind, components whose cost this one also pays and which are subtracted).
    # A pass that reads a property has to build the wrapper list first, and the charge read has
    # to compute the charges; charging those twice would make the column sum to more than
    # extract() does, which is how the old version of this file came to print 198 us of parts
    # for a 187 us whole.
    GAS = "Gasteiger charges, no read (RDKit C++)"
    STE = "AssignStereochemistry (RDKit C++)"
    ATL = "atom wrapper list, no reads"
    BOL = "bond wrapper list, no reads"
    comps = [(GAS, c_gasteiger, "RDKIT", ()),
             (STE, c_stereo, "RDKIT", ()),
             (ATL, c_atom_list, "FLOOR", ()),
             ("8 per-atom column passes", c_atom_cols, "PYTHON", (ATL,)),
             ("Gasteiger charge read pass", c_charge_read, "PYTHON", (GAS, ATL)),
             ("CIP HasProp pass", c_cip, "PYTHON", (ATL,)),
             (BOL, c_bond_list, "FLOOR", ()),
             ("per-bond loop, 7 reads", c_bond_loop, "PYTHON", (BOL,)),
             ("ring CSR (_rings.rings_for)", c_rings, "PYTHON", ())]

    tot = {}
    for cyc in range(3):
        order = comps[cyc:] + comps[:cyc]
        for name, fn, _kind, _sub in order:
            tot.setdefault(name, []).append(bench(fn, pick))

    whole = bench(c_extract, pick, reps=5)
    gone = bench(c_crippen_rdkit, pick, reps=3)

    print(f"  {'component':38s} {'us/mol':>9s} {'us/atom':>9s}  kind")
    acc = 0.0
    by_kind = {"RDKIT": 0.0, "FLOOR": 0.0, "PYTHON": 0.0}
    for name, _fn, kind, sub in comps:
        us = min(tot[name]) - sum(min(tot[s]) for s in sub)
        acc += us
        by_kind[kind] += us
        print(f"  {name:38s} {us:9.2f} {us / natoms:9.3f}  {kind}")
    rest = whole - acc
    print(f"  {'numpy assembly + loop bookkeeping':38s} {rest:9.2f} {rest / natoms:9.3f}  PYTHON")
    by_kind["PYTHON"] += rest
    print(f"  {'':38s} {'-' * 9:>9s}")
    print(f"  {'extract() end to end':38s} {whole:9.2f} {whole / natoms:9.3f}")
    print(f"\n  {'(removed) RDKit Crippen, cold':38s} {gone:9.2f} {gone / natoms:9.3f}  "
          f"NO LONGER CALLED -- this is what moved into C++")

    print(f"\n  wrapper construction alone : {by_kind['FLOOR']:8.2f} us/mol   "
          f"({100 * by_kind['FLOOR'] / whole:.0f}% of extract)")
    print(f"  RDKit's own computation    : {by_kind['RDKIT']:8.2f} us/mol   "
          f"({100 * by_kind['RDKIT'] / whole:.0f}%)")
    print(f"  reads, and moving the data : {by_kind['PYTHON']:8.2f} us/mol   "
          f"({100 * by_kind['PYTHON'] / whole:.0f}%)")
    print("\n  The first two are what a Python-side rewrite cannot remove: the first is the cost")
    print("  of an RDKit object existing in Python at all, and the second is work being done.")


if __name__ == "__main__":
    main()
