"""Generate cpp/pickle_tables.h -- the three RDKit tables the pickle reader needs.

WHY A GENERATED TABLE AND NOT A TRANSCRIPTION. src/hume_core/molpickle.h reads an RDKit
MolPickler blob and fills the boundary arrays without touching a Python object. Three of the
boundary's values are NOT in the pickle and are not graph queries either -- they are lookups
into RDKit's own data:

  * atom mass, when the atom has no isotope -- PeriodicTable::getAtomicWeight(Z)
  * atom mass, when it does                 -- PeriodicTable::getMassForIsotope(Z, isotope)
  * bond order                              -- Bond::getBondTypeAsDouble() from the BondType enum

Hand-copying any of those would put a second copy of RDKit's numbers in this repository, which
is the thing cpp/crippen_tables.h and cpp/estate_tables.h both exist to avoid. So they are asked
of the live RDKit and written out, exactly as cpp/export_crippen.py does for Crippen.txt.

THE DRIFT GUARD IS A HASH OF THE NUMBERS, not of this file (house rule 6 in PORT_STATUS.md).
`--check` re-derives every value from whichever RDKit is installed and compares the digest
against the one baked into the header. cpp/verify_molpickle.py runs it as part of the same
process that produces the exactness evidence, so a moved atomic weight cannot pass unnoticed.

    .venv/bin/python cpp/export_pickle_tables.py            # write cpp/pickle_tables.h
    .venv/bin/python cpp/export_pickle_tables.py --check    # verify the header against RDKit
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import rdkit
from rdkit import Chem

HERE = Path(__file__).resolve().parent
OUT = HERE / "pickle_tables.h"

N_Z = 119          # 0 (dummy) .. 118
MAX_ISOTOPE = 320  # RDKit's heaviest tabulated isotope is 295; scan past it and stop at nothing


def tables():
    """-> (atomic weights, [(z, isotope, mass)], bond orders). Everything from the live RDKit."""
    pt = Chem.GetPeriodicTable()
    weights = [pt.GetAtomicWeight(z) for z in range(N_Z)]

    iso = []
    for z in range(N_Z):
        for a in range(1, MAX_ISOTOPE):
            try:
                m = pt.GetMassForIsotope(z, a)
            except Exception:
                m = 0.0
            if m:
                iso.append((z, a, m))

    # Bond order comes from a real bond of each type, because getBondTypeAsDouble() is a switch
    # with holes in it: UNSPECIFIED, THREECENTER, DATIVEL, DATIVER and OTHER raise rather than
    # return, and reproducing that requires knowing WHICH ones raise, not guessing.
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(6))
    rw.AddAtom(Chem.Atom(6))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    bond = rw.GetBondWithIdx(0)
    n_bt = max(Chem.BondType.values) + 1
    orders = []
    for v in range(n_bt):
        bt = Chem.BondType.values.get(v)
        if bt is None:
            orders.append(None)
            continue
        bond.SetBondType(bt)
        try:
            orders.append(bond.GetBondTypeAsDouble())
        except Exception:
            orders.append(None)
    return weights, iso, orders


def digest(weights, iso, orders) -> str:
    """sha256 of the NUMBERS, rendered exactly. %.17g round-trips a float64."""
    h = hashlib.sha256()
    h.update(b"hume-molpickle-tables-v1\n")
    for z, w in enumerate(weights):
        h.update(f"w {z} {w:.17g}\n".encode())
    for z, a, m in iso:
        h.update(f"i {z} {a} {m:.17g}\n".encode())
    for v, o in enumerate(orders):
        h.update(f"b {v} {'-' if o is None else f'{o:.17g}'}\n".encode())
    return h.hexdigest()


def render(weights, iso, orders, sha) -> str:
    def wrap(items, per_line):
        out, row = [], []
        for x in items:
            row.append(x)
            if len(row) == per_line:
                out.append("    " + " ".join(row))
                row = []
        if row:
            out.append("    " + " ".join(row))
        return "\n".join(out)

    # NaN marks a BondType RDKit refuses to give a number for. molpickle.h turns a NaN order into
    # a thrown exception at exactly the point rdkit's GetBondTypeAsDouble() would have thrown, so
    # a molecule with a THREECENTER bond fails loudly on both paths rather than on one.
    ords = ["std::numeric_limits<double>::quiet_NaN()," if o is None else f"{o!r},"
            for o in orders]
    keys = [f"{z * 1024 + a}u," for z, a, _ in iso]
    masses = [f"{m!r}," for _, _, m in iso]
    return f'''// GENERATED FILE -- DO NOT EDIT. Written by cpp/export_pickle_tables.py.
//
// RDKit's own numbers, lifted out so src/hume_core/molpickle.h can answer three questions the
// MolPickler blob does not carry: an atom's mass from (Z, isotope), and a bond's order from its
// BondType. See the generator for why these are generated rather than transcribed.
//
//   source rdkit : {rdkit.__version__}
//   spec sha256  : {sha}
//
// The digest covers the NUMBERS, not this file's bytes -- a comment edit must not cry wolf and a
// changed atomic weight must not slip through. cpp/export_pickle_tables.py --check re-derives it.
#ifndef HUME_PICKLE_TABLES_H
#define HUME_PICKLE_TABLES_H

#include <cstdint>
#include <limits>

namespace pickletab {{

inline constexpr char SPEC_SHA256[] = "{sha}";
inline constexpr char SOURCE_RDKIT[] = "{rdkit.__version__}";

// PeriodicTable::getAtomicWeight(Z), Z = 0 (dummy, 0.0) .. {N_Z - 1}.
inline constexpr int N_Z = {N_Z};
inline constexpr double ATOMIC_WEIGHT[N_Z] = {{
{wrap([f"{w!r}," for w in weights], 6)}
}};

// PeriodicTable::getMassForIsotope(Z, A), as a key-sorted array for binary search.
// key = Z * 1024 + A. Only the {len(iso)} (Z, A) pairs RDKit has a mass for are present; a miss
// means getMassForIsotope() would have returned 0.0, and Atom::getMass() then falls back to A.
inline constexpr int N_ISO = {len(iso)};
inline constexpr std::uint32_t ISO_KEY[N_ISO] = {{
{wrap(keys, 10)}
}};
inline constexpr double ISO_MASS[N_ISO] = {{
{wrap(masses, 5)}
}};

// Bond::getBondTypeAsDouble(), indexed by the BondType enum value the pickle stores.
// NaN = RDKit raises for that type.
inline constexpr int N_BONDTYPE = {len(orders)};
inline constexpr double BOND_ORDER[N_BONDTYPE] = {{
{wrap(ords, 5)}
}};

}}  // namespace pickletab

#endif
'''


def main() -> int:
    weights, iso, orders = tables()
    sha = digest(weights, iso, orders)
    if "--check" in sys.argv:
        txt = OUT.read_text()
        marker = f'inline constexpr char SPEC_SHA256[] = "{sha}";'
        ok = marker in txt
        print(f"cpp/pickle_tables.h  rdkit {rdkit.__version__}  spec {sha}")
        print("  MATCHES the header" if ok else "  DRIFT -- the header was generated from "
              "different numbers; re-run without --check")
        return 0 if ok else 1
    OUT.write_text(render(weights, iso, orders, sha))
    print(f"wrote {OUT}  ({N_Z} weights, {len(iso)} isotopes, {len(orders)} bond types)")
    print(f"  rdkit {rdkit.__version__}  spec {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
