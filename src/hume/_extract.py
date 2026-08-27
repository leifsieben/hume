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
one (n_atoms, 4) block, because the extension reads them by row and one allocation beats twelve.

    atom_off  int32   (n_mol + 1,)   atoms of molecule k are [atom_off[k] : atom_off[k+1]]
    bond_off  int32   (n_mol + 1,)
    chg_ok    int32   (n_mol,)       0 = Gasteiger unavailable, charges are 0.0
    atom_i    int32   (n_atoms, 8)   Z, degree, nH, formal charge, hyb, aromatic, in-ring, CIP
    atom_d    float64 (n_atoms, 4)   mass, Gasteiger charge, Crippen logP, Crippen MR
    bond_i    int32   (n_bonds, 4)   u, v, conjugated, in-ring
    bond_s    int32   (n_bonds,)     E/Z as +/-1, 0 for none
    bond_d    float64 (n_bonds,)     bond order

Hybridisation is passed through as RDKit's enum value rather than re-derived, for the reason
export_predict.py gives: HallKierAlpha indexes a per-element table by (hybridisation - 2), and
reimplementing RDKit's perception rules in C++ is the first place an "exact" claim would quietly
stop being true.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges

# E/Z as +/-1, matching stereo.py's _E exactly (TRANS is E, CIS is Z).
_EZ = {Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
       Chem.BondStereo.STEREOZ: -1, Chem.BondStereo.STEREOCIS: -1}

N_ATOM_INT, N_ATOM_DBL, N_BOND_INT = 8, 4, 4


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
    ai: list[int] = []
    ad: list[float] = []
    bi: list[int] = []
    bs: list[int] = []
    bd: list[float] = []
    na = nb = 0

    for k, m in enumerate(mols):
        if m is None:
            raise ValueError(f"molecule {k} is None; parse before calling extract()")

        # The four BCUT2D atom properties come from RDKit rather than being reimplemented. That
        # is a deliberate split: Crippen (0.85 us) and Gasteiger (9.41 us) are already cheap C++,
        # while the EIGENVALUE step is the ~300 us. Porting the cheap half would buy nothing and
        # would put two SMARTS/PEOE implementations in the world.
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
            chg = [a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()]
            ok = True
            for i, c in enumerate(chg):
                if not np.isfinite(c):
                    chg[i], ok = 0.0, False
        except Exception:
            chg = [0.0] * m.GetNumAtoms()
            ok = False
        crip = rdMolDescriptors._CalcCrippenContribs(m)
        # CIP codes for the stereo block. MolFromSmiles assigns them already; the explicit call
        # is the safety net stereo.py also carries, and is verified to change nothing here.
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)

        n = m.GetNumAtoms()
        for i, a in enumerate(m.GetAtoms()):
            if a.HasProp("_CIPCode"):
                cip = 1 if a.GetProp("_CIPCode") == "R" else -1
            else:
                cip = 0
            ai.extend((a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs(),
                       a.GetFormalCharge(), int(a.GetHybridization()),
                       int(a.GetIsAromatic()), int(a.IsInRing()), cip))
            cl, cm = crip[i]
            ad.extend((a.GetMass(), chg[i], cl, cm))

        nbonds = m.GetNumBonds()
        for b in m.GetBonds():
            bi.extend((b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                       int(b.GetIsConjugated()), int(b.IsInRing())))
            bs.append(_EZ.get(b.GetStereo(), 0))
            bd.append(b.GetBondTypeAsDouble())

        na += n
        nb += nbonds
        atom_off.append(na)
        bond_off.append(nb)
        chg_ok.append(int(ok))

    n_mol = len(chg_ok)
    return Batch(
        atom_off=np.asarray(atom_off, dtype=np.int32),
        bond_off=np.asarray(bond_off, dtype=np.int32),
        chg_ok=np.asarray(chg_ok, dtype=np.int32),
        atom_i=np.asarray(ai, dtype=np.int32).reshape(na, N_ATOM_INT),
        atom_d=np.asarray(ad, dtype=np.float64).reshape(na, N_ATOM_DBL),
        bond_i=np.asarray(bi, dtype=np.int32).reshape(nb, N_BOND_INT),
        bond_s=np.asarray(bs, dtype=np.int32),
        bond_d=np.asarray(bd, dtype=np.float64),
    ) if n_mol else _empty()


def _empty() -> Batch:
    z32, z64 = np.zeros(0, np.int32), np.zeros(0, np.float64)
    return Batch(np.zeros(1, np.int32), np.zeros(1, np.int32), z32,
                 z32.reshape(0, N_ATOM_INT), z64.reshape(0, N_ATOM_DBL),
                 z32.reshape(0, N_BOND_INT), z32, z64)
