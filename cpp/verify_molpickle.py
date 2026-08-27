"""Is the C++ MolPickler reader byte-for-byte the same boundary as src/hume/_extract.py?

WHAT IS BEING COMPARED, and why it is a paired IN-PROCESS comparison rather than two logs. Both
paths are run over the SAME RDKit molecule objects in the SAME process, so nothing about parsing,
perception, RDKit version or machine state can differ between them. What is left when they are
held against each other is the reader, and only the reader.

  1. THE BOUNDARY, field by field. `extract()` builds eight arrays by asking each molecule ~300
     questions through RDKit's Python API; `_core.pickle_extract()` builds the same eight from
     `m.ToBinary()` with no Python call per atom at all. Every one of the 18 columns is compared
     RAW -- int32 equality, and float64 compared through a uint64 view, which is stricter than
     `==` because it also pins the bit pattern of a NaN. No tolerance: both sides are supposed
     to be the same numbers, not close ones, and a tolerance here could only hide a wrong field.
  2. THE 182 COLUMNS downstream of it, the same way. Field-level identity already implies this,
     but the implication runs through 116 KB of block code and this project's method is to check
     the thing that ships rather than the thing that should follow.
  3. THE DRIFT GUARDS, both of them, proven to FIRE rather than assumed to exist: the pinned
     MolPickler format version (by corrupting the version bytes of a real pickle) and the
     generated tables in cpp/pickle_tables.h (by re-deriving their digest from the live RDKit).

  .venv/bin/python cpp/verify_molpickle.py [n_mols] [smiles_file ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp"))

from hume import _core                                      # noqa: E402
from hume._columns import N_COLS                            # noqa: E402
from hume._extract import extract, extract_pickles          # noqa: E402
import export_pickle_tables                                 # noqa: E402

RDLogger.DisableLog("rdApp.*")
BATCH = 4096

# (array index in the Batch tuple, column, label). Column -1 means the array is 1-D.
FIELDS = [
    (0, -1, "atom_off"), (1, -1, "bond_off"), (2, -1, "chg_ok"),
    (3, 0, "atom Z"), (3, 1, "atom degree"), (3, 2, "atom nH"), (3, 3, "atom formal charge"),
    (3, 4, "atom hybridisation"), (3, 5, "atom aromatic"), (3, 6, "atom in-ring"),
    (3, 7, "atom CIP"), (3, 8, "atom ring count"),
    # SMARTS `v`. On the reference side it is `Atom.GetTotalValence()`; on the pickle side it is
    # the blob's own explicit + implicit valence, which the reader used to skip. This row is the
    # evidence that the second is the first -- the fragment columns depend on it and nothing
    # downstream of a wrong valence would look wrong.
    (3, 9, "atom total valence"),
    (4, 0, "atom mass"), (4, 1, "atom Gasteiger"),
    (5, 0, "bond u"), (5, 1, "bond v"), (5, 2, "bond conjugated"), (5, 3, "bond in-ring"),
    (5, 4, "bond SMARTS code"),
    # RDKit's Bond::BondType INTEGER. On the reference side it is `int(Bond.GetBondType())`; on
    # the pickle side it is the type byte the reader already had in hand for the SMARTS code and
    # was discarding. The Morgan fingerprint hashes this enum, and the SMARTS code cannot stand in
    # for it -- it maps DATIVE and SINGLE to the same zero, and cpp/hard.smi has 114 dative bonds.
    # Same shape of finding as `atom total valence`, and this row is the evidence for it.
    (5, 5, "bond type"),
    (6, -1, "bond E/Z"), (7, -1, "bond order"),
    # THE RING CSR. Under the dense design both boundaries fill these from the same
    # src/hume/_rings.py, so they are equal by construction and this is a standing assertion
    # that they still are -- cheap, and the thing that would break first if anyone ever swapped
    # the pickle path back to reading rings out of the blob.
    (8, -1, "ring_moff"), (9, -1, "ring_ptr"), (10, -1, "ring_at"),
]


def raw(a: np.ndarray) -> np.ndarray:
    """Bit pattern, not value. Makes float64 comparison exact and NaN-aware in one step."""
    return a.view(np.uint64) if a.dtype == np.float64 else a


def check_guards() -> int:
    """Both drift guards, made to fire. A guard nobody has seen trip is a comment."""
    bad = 0
    print("drift guards")

    weights, iso, orders = export_pickle_tables.tables()
    sha = export_pickle_tables.digest(weights, iso, orders)
    ok = sha == _core.PICKLE_TABLES_SPEC
    bad += not ok
    print(f"  cpp/pickle_tables.h digest re-derived from the live rdkit : "
          f"{'MATCH' if ok else 'DRIFT'}  {sha[:16]}...")
    print(f"    header was generated from rdkit {_core.PICKLE_TABLES_RDKIT}")

    probe = bytearray(Chem.MolFromSmiles("C").ToBinary(
        Chem.PropertyPickleOptions.PrivateProps | Chem.PropertyPickleOptions.AtomProps
        | Chem.PropertyPickleOptions.ComputedProps | Chem.PropertyPickleOptions.NoConformers))
    _core.pickle_check(bytes(probe))                       # the real one must pass
    print(f"  pinned MolPickler format version {'.'.join(map(str, _core.PICKLE_VERSION))} "
          f"accepted for the live rdkit {rdkit.__version__}")

    # Bytes 8/12/16 are majorVersion / minorVersion / patchVersion. Bump each in turn.
    for off, what in ((8, "major"), (12, "minor"), (16, "patch")):
        corrupt = bytearray(probe)
        corrupt[off] += 1
        try:
            _core.pickle_check(bytes(corrupt))
        except RuntimeError as exc:
            first = str(exc).split(".", 1)[0]
            print(f"  {what} version bumped -> RuntimeError: {first}...")
        else:
            bad += 1
            print(f"  {what} version bumped -> NO ERROR  <-- the guard does not fire")
    return bad


def verify(src: Path, n_want: int) -> int:
    smis = [s for s in src.read_text().split("\n") if s][:n_want]
    n_bad = {}
    n_col_bad = 0
    n_mol = n_at = n_bd = 0

    for lo in range(0, len(smis), BATCH):
        chunk = smis[lo:lo + BATCH]
        mols = [Chem.MolFromSmiles(s) for s in chunk]
        if any(m is None for m in mols):
            raise ValueError(f"unparseable SMILES in {src}")

        # The pickle path FIRST, on molecules neither path has touched, then extract() on the
        # very same objects. Order matters only for honesty about RDKit's caches, and either
        # order gives the same arrays; this one makes the new path the one paying cold cost.
        p = extract_pickles(mols)
        got = tuple(_core.pickle_extract(p.blobs)) + (p.rings.ring_moff, p.rings.ring_ptr,
                                                      p.rings.ring_at)
        want = extract(mols)
        want_t = (want.atom_off, want.bond_off, want.chg_ok, want.atom_i, want.atom_d,
                  want.bond_i, want.bond_s, want.bond_d, want.rings.ring_moff,
                  want.rings.ring_ptr, want.rings.ring_at)

        for ai, col, label in FIELDS:
            g, w = got[ai], want_t[ai]
            if col >= 0:
                g, w = g[:, col], w[:, col]
            if g.shape != w.shape or not np.array_equal(raw(g), raw(w)):
                n_bad[label] = n_bad.get(label, 0) + int(
                    (g.shape != w.shape) or int(np.count_nonzero(raw(g) != raw(w))))

        Xp = _core.blocks_from_pickles(p.blobs)
        Xa = _core.blocks(want.atom_off, want.bond_off, want.chg_ok, want.atom_i, want.atom_d,
                          want.bond_i, want.bond_s, want.bond_d)
        n_col_bad += int(np.count_nonzero(raw(Xp) != raw(Xa)))

        n_mol += len(mols)
        n_at += want.atom_i.shape[0]
        n_bd += want.bond_i.shape[0]

    print(f"\n{src}")
    print(f"  {n_mol} molecules, {n_at} atoms, {n_bd} bonds")
    for _ai, _col, label in FIELDS:
        k = n_bad.get(label, 0)
        print(f"    {label:22s} {'EXACT' if not k else f'MISMATCH on {k} values'}")
    total = n_mol * N_COLS
    print(f"    {'182 block columns':22s} "
          f"{'ALL EXACT' if not n_col_bad else f'MISMATCH on {n_col_bad} / {total} values'}"
          f"   ({total} compared, bitwise)")
    return sum(n_bad.values()) + n_col_bad


def main() -> int:
    print(f"rdkit {rdkit.__version__}   numpy {np.__version__}   "
          f"python {sys.version.split()[0]}")
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    srcs = [Path(a) for a in sys.argv[2:]] or [ROOT / "cpp" / "mols.smi",
                                               ROOT / "cpp" / "hard.smi"]
    bad = check_guards()
    for src in srcs:
        bad += verify(src, n_want)
    print("\nALL EXACT -- the pickle reader is the same boundary" if not bad
          else f"\n{bad} MISMATCHES")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
