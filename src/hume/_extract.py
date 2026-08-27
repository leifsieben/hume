"""RDKit molecules -> flat numpy arrays, batched. The Python half of the boundary.

This is `cpp/export_predict.py` with the text serialisation deleted. That file is the format the
verified C++ consumes, so this one reproduces its *content* field for field; what it does not
reproduce is the round-trip through `%.10g` and `std::ifstream`, which is the whole point.

WHY ARRAYS AND NOT A FILE. The text path writes 178 MB for 100k molecules and parses it back,
which costs more than the arithmetic it feeds. It also had a failure mode that arrays cannot
have: a `nan` written into the export made C++'s `istream` fail on the token, and every
subsequent field shifted by one -- the loader then reported a 19.9-atom mean for a 30.6-atom
corpus. In a strided float64 array there are no fields to shift, only offsets, so that class of
desync is structurally gone rather than defended against.

THE NON-FINITE GASTEIGER CONTRACT is kept anyway, unchanged, because it is doing a second job.
Molecules RDKit cannot charge (no PEOE parameters -- selenium is the common case) get 0.0 in the
charge column and `chg_ok = 0`. The 0.0 is no longer protecting a parser; it is keeping BCUT2D's
Burden matrix finite, since a nan on the diagonal propagates through the eigensolver and returns
nan for all eight BCUT2D columns rather than for the two that depend on charge. The flag records
that the molecule was uncharged instead of hiding it, which is what lets a verifier compare like
with like. Callers that want to know can read `chg_ok`.

LAYOUT. One batch is a set of flat arrays plus offsets, so N molecules cross the boundary in one
call. Per-atom integer properties share one (n_atoms, 8) C-contiguous block and per-atom doubles
one (n_atoms, 2) block, because the extension reads them by row and one allocation beats twelve.

    atom_off  int32   (n_mol + 1,)   atoms of molecule k are [atom_off[k] : atom_off[k+1]]
    bond_off  int32   (n_mol + 1,)
    chg_ok    int32   (n_mol,)       0 = Gasteiger unavailable, charges are 0.0
    atom_i    int32   (n_atoms, 8)   Z, degree, nH, formal charge, hyb, aromatic, in-ring, CIP
    atom_d    float64 (n_atoms, 2)   mass, Gasteiger charge
    bond_i    int32   (n_bonds, 5)   u, v, conjugated, in-ring, SMARTS bond code
    bond_s    int32   (n_bonds,)     E/Z as +/-1, 0 for none
    bond_d    float64 (n_bonds,)     bond order

Hybridisation is passed through as RDKit's enum value rather than re-derived, for the reason
export_predict.py gives: HallKierAlpha indexes a per-element table by (hybridisation - 2), and
reimplementing RDKit's perception rules in C++ is the first place an "exact" claim would quietly
stop being true.

CRIPPEN IS NOT IN THE ARRAYS ANY MORE. It used to be two more `atom_d` columns filled by
`rdMolDescriptors._CalcCrippenContribs`, which cost 78 us/mol -- 42% of this module -- to run
110 SMARTS patterns over the whole molecule. src/hume_core/crippen_typer.h answers the same
question from the integers already in `atom_i` for 1.5 us/mol, bit-identically to RDKit on
2,869,048 atoms, so the call is gone and the extension fills the pair itself. What that costs
here is the fifth `bond_i` column: the typer needs SMARTS bond semantics, where the ORDER and
the AROMATIC FLAG are independent questions, and neither is recoverable from `bond_d` -- a
dative bond has order 1.0 without being SINGLE, and cpp/mols.smi contains TRIPLE bonds carrying
the aromatic flag. Both cases are real, and together they are one Python call per bond.

WHY THE LOOPS BELOW LOOK LIKE THAT. `a.GetAtomicNum()` is not arithmetic, it is a Boost.Python
round trip, and the old shape of this function made about eleven of them per atom and seven per
bond -- ~300 per molecule, measured at 88 us. Three things shrink that without changing a single
value, because every value is the same one RDKit would have handed back either way:

  * ONE PASS PER COLUMN, `list.extend(map(Atom.GetAtomicNum, ats))`, instead of one tuple per
    atom. The unbound method skips the per-call attribute lookup on the instance, which measured
    0.039 us/call against 0.11 for `a.GetAtomicNum()` -- and the whole pass over 29 atoms then
    costs about as much as ten individual calls did.
  * THE ATOM LIST IS BUILT ONCE, and BY INDEX. Wrapper construction alone -- building the list
    and reading nothing off it -- is a third of what is left, so the old code's three separate
    walks (charges, properties, CIP) were expensive before they read anything. Materialising
    once pays it once. And `map(m.GetAtomWithIdx, range(n))` is HALF the price of iterating the
    `m.GetAtoms()` sequence for the same objects in the same order -- 7.4 us/mol against 15.0,
    and 8.9 against 16.7 for bonds. RDKit's _ROAtomSeq iterator is simply dearer than indexing.
  * THE NON-FINITE CHARGE SCAN IS A BATCH numpy OP, not a per-atom `np.isfinite`. Calling a
    numpy ufunc on a Python float costs ~0.3 us, so the old scan cost more than the charges did.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

# E/Z as +/-1, matching stereo.py's _E exactly (TRANS is E, CIS is Z).
_EZ = {Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
       Chem.BondStereo.STEREOZ: -1, Chem.BondStereo.STEREOCIS: -1}

N_ATOM_INT, N_ATOM_DBL, N_BOND_INT = 8, 2, 5

# The bond half of the Crippen typer's input, byte-for-byte cpp/export_crippen.py's bond_code():
# a bit for the bond ORDER when it is one of the three SMARTS knows how to name, and a separate
# bit for the aromatic FLAG. `-` asks about the order, `:` about the flag, and the single bond
# joining biphenyl's two rings is the C20-vs-C19 distinction that needs them apart.
_BIT_SINGLE, _BIT_DOUBLE, _BIT_TRIPLE, _BIT_AROM = 1, 2, 4, 8
_TYPE_BIT = {Chem.BondType.SINGLE: _BIT_SINGLE,
             Chem.BondType.DOUBLE: _BIT_DOUBLE,
             Chem.BondType.TRIPLE: _BIT_TRIPLE}

# BondType -> GetBondTypeAsDouble(), memoised the first time each type is seen. The bond loop has
# to read `b.GetBondType()` anyway for the Crippen code, so caching the order against it replaces
# a second Python call per bond with a dict hit. It is filled from RDKit's own answer for a real
# bond of that type rather than transcribed, so there is no second copy of RDKit's table to go
# stale -- and the types RDKit refuses to give a number for (THREECENTER, DATIVEL, DATIVER,
# OTHER all raise) never enter the cache and raise from the call, exactly as before.
_ORDER: dict = {}

# Unbound accessors, bound once here. `map(_atomic_num, ats)` is a C-level loop over a C-level
# callable; `[a.GetAtomicNum() for a in ats]` re-resolves the attribute on every atom.
_atomic_num = Chem.Atom.GetAtomicNum
_degree = Chem.Atom.GetDegree
_total_num_hs = Chem.Atom.GetTotalNumHs
_formal_charge = Chem.Atom.GetFormalCharge
_hybridization = Chem.Atom.GetHybridization
_is_aromatic = Chem.Atom.GetIsAromatic
_atom_in_ring = Chem.Atom.IsInRing
_mass = Chem.Atom.GetMass
_has_prop = Chem.Atom.HasProp
_double_prop = Chem.Atom.GetDoubleProp

_CIP_CODE = "_CIPCode"
_GASTEIGER = "_GasteigerCharge"


@dataclass(frozen=True)
class Batch:
    """Flat arrays for N molecules. See the module docstring for the layout."""
    atom_off: np.ndarray
    bond_off: np.ndarray
    chg_ok: np.ndarray
    atom_i: np.ndarray
    atom_d: np.ndarray
    bond_i: np.ndarray
    bond_s: np.ndarray
    bond_d: np.ndarray

    @property
    def n_mol(self) -> int:
        return len(self.chg_ok)


def extract(mols) -> Batch:
    """Flatten an iterable of RDKit molecules into one Batch.

    Molecules are taken as given -- no filtering, no sanitisation beyond what the caller already
    did. `None` is rejected loudly rather than skipped, because a silently dropped molecule turns
    a row index into a lie, and every consumer of this array indexes by position.
    """
    atom_off, bond_off, chg_ok = [0], [0], []
    z: list[int] = []
    deg: list[int] = []
    nh: list[int] = []
    fchg: list[int] = []
    hyb: list[int] = []
    arom: list[int] = []
    ring: list[int] = []
    cip: list[int] = []
    mass: list[float] = []
    charge: list[float] = []
    bu: list[int] = []
    bv: list[int] = []
    bconj: list[int] = []
    bring: list[int] = []
    bcode: list[int] = []
    bs: list[int] = []
    bd: list[float] = []
    na = nb = 0

    for k, m in enumerate(mols):
        if m is None:
            raise ValueError(f"molecule {k} is None; parse before calling extract()")

        n = m.GetNumAtoms()
        ats = list(map(m.GetAtomWithIdx, range(n)))

        # Gasteiger stays RDKit's. It is 9 us of iterative C++ for the charges themselves, and
        # PEOE is a fitted parameter set rather than a graph query -- a second implementation
        # would be a second set of parameters to keep in step, which is the argument that also
        # kept hybridisation and CIP on RDKit's side of the line. Crippen was the opposite case
        # and has moved; see the module docstring.
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
            charge.extend(map(_double_prop, ats, repeat(_GASTEIGER)))
            ok = 1
        except Exception:
            del charge[na:]                      # a partial molecule must not survive the throw
            charge.extend(repeat(0.0, n))
            ok = 0
        # CIP codes for the stereo block. MolFromSmiles assigns them already; the explicit call
        # is the safety net stereo.py also carries, and is verified to change nothing here.
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)

        z.extend(map(_atomic_num, ats))
        deg.extend(map(_degree, ats))
        nh.extend(map(_total_num_hs, ats))
        fchg.extend(map(_formal_charge, ats))
        hyb.extend(map(_hybridization, ats))
        arom.extend(map(_is_aromatic, ats))
        ring.extend(map(_atom_in_ring, ats))
        mass.extend(map(_mass, ats))

        # HasProp returns 0/1, so in the overwhelmingly common case of a molecule with no
        # assigned stereocentre the flag list IS the CIP column and no second pass happens.
        flags = list(map(_has_prop, ats, repeat(_CIP_CODE)))
        if any(flags):
            cip.extend([(1 if a.GetProp(_CIP_CODE) == "R" else -1) if f else 0
                        for f, a in zip(flags, ats)])
        else:
            cip.extend(flags)

        nbonds = m.GetNumBonds()
        for b in map(m.GetBondWithIdx, range(nbonds)):
            bt = b.GetBondType()
            code = _TYPE_BIT.get(bt, 0)
            if b.GetIsAromatic():
                code |= _BIT_AROM
            bu.append(b.GetBeginAtomIdx())
            bv.append(b.GetEndAtomIdx())
            bconj.append(b.GetIsConjugated())
            bring.append(b.IsInRing())
            bcode.append(code)
            bs.append(_EZ.get(b.GetStereo(), 0))
            order = _ORDER.get(bt)
            if order is None:
                order = _ORDER[bt] = b.GetBondTypeAsDouble()
            bd.append(order)

        na += n
        nb += nbonds
        atom_off.append(na)
        bond_off.append(nb)
        chg_ok.append(ok)

    if not chg_ok:
        return _empty()

    atom_off_a = np.asarray(atom_off, dtype=np.int32)
    chg_ok_a = np.asarray(chg_ok, dtype=np.int32)
    charge_a = np.asarray(charge, dtype=np.float64)

    # The non-finite contract, once per batch instead of once per atom. RDKit hands back inf or
    # nan for elements PEOE has no parameters for; those atoms get 0.0 and their molecule gets
    # chg_ok = 0. Doing it here rather than in the loop is what makes it free when nothing is
    # wrong, which is almost always -- and `searchsorted` on the offsets is exactly the "which
    # molecule owns atom i" question the flat layout was chosen to make cheap.
    bad = ~np.isfinite(charge_a)
    if bad.any():
        charge_a[bad] = 0.0
        owners = np.searchsorted(atom_off_a, np.flatnonzero(bad), side="right") - 1
        chg_ok_a[owners] = 0

    atom_i = np.empty((na, N_ATOM_INT), dtype=np.int32)
    for col, src in enumerate((z, deg, nh, fchg, hyb, arom, ring, cip)):
        atom_i[:, col] = src
    atom_d = np.empty((na, N_ATOM_DBL), dtype=np.float64)
    atom_d[:, 0] = mass
    atom_d[:, 1] = charge_a
    bond_i = np.empty((nb, N_BOND_INT), dtype=np.int32)
    for col, src in enumerate((bu, bv, bconj, bring, bcode)):
        bond_i[:, col] = src

    return Batch(
        atom_off=atom_off_a,
        bond_off=np.asarray(bond_off, dtype=np.int32),
        chg_ok=chg_ok_a,
        atom_i=atom_i,
        atom_d=atom_d,
        bond_i=bond_i,
        bond_s=np.asarray(bs, dtype=np.int32),
        bond_d=np.asarray(bd, dtype=np.float64),
    )


def _empty() -> Batch:
    z32, z64 = np.zeros(0, np.int32), np.zeros(0, np.float64)
    return Batch(np.zeros(1, np.int32), np.zeros(1, np.int32), z32,
                 z32.reshape(0, N_ATOM_INT), z64.reshape(0, N_ATOM_DBL),
                 z32.reshape(0, N_BOND_INT), z32, z64)
