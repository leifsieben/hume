"""Generate cpp/vsa_tables.h -- the numeric SPEC of the VSA-binning descriptor family.

    uv run --isolated --python 3.12 --with "rdkit==2025.9.2" --with "numpy==2.4.6" \
           python cpp/gen_vsa_tables.py

WHAT THE SPEC IS.  Every column in this family is `sum over atoms of contribution[i], bucketed by
which interval property[i] falls in`.  There is no SMARTS and no graph algorithm to get wrong --
the whole specification is four number tables:

  * the covalent bond radii `Rb0` that Labute's ASA uses, one per element;
  * the four bond-shortening constants indexed by RDKit's BondType enum;
  * five arrays of bin edges;
  * (implicitly) the rule that decides which side of an edge a value falls on.

WHERE EACH NUMBER COMES FROM, and it is NOT the paper.

  Rb0            PeriodicTable::getRb0(z), read out of the running RDKit.  This is the same table
                 Code/GraphMol/Descriptors/MolSurf.cpp reads in getLabuteAtomContribs().
  bondScaleFacts the literal `const double bondScaleFacts[4] = {.1, 0, .2, .3};` inside
                 getLabuteAtomContribs(), indexed by Bond::BondType -- so index 0 is
                 UNSPECIFIED, 1 SINGLE, 2 DOUBLE, 3 TRIPLE.  Aromatic bonds take index 0.
  logp/mr/chg    MolSurf.cpp hard-codes these inside calcSlogP_VSA / calcSMR_VSA / calcPEOE_VSA;
    bins         rdkit/Chem/MolSurf.py carries an identical Python copy as `logpBins`, `mrBins`,
                 `chgBins`.  This generator reads the PYTHON copy (the C++ literals are not
                 introspectable) and cpp/verify_vsa.py then proves the C++ agrees, on every atom
                 of 100,000 molecules including the ones sitting exactly on an edge.
  estate/vsa     rdkit/Chem/EState/EState_VSA.py `estateBins`, `vsaBins`.  These two families have
    bins         NO C++ implementation -- EState_VSA_ / VSA_EState_ are pure Python -- so the
                 Python lists ARE the spec, not a copy of it.

THE DRIFT GUARD HASHES THE SPEC, NOT THE FILE.  `spec` below is sha256 over the IEEE-754 bit
patterns of every number in the table, in a canonical rendering that both this script and
src/hume_core/vsa_bins.h can produce byte-identically.  vsa_bins.h recomputes it at load and
refuses to hand back numbers if it disagrees, so hand-editing a bin edge in the generated header
is caught without needing RDKit present.  A file hash would instead fire on a comment edit and
stay silent on a table edited to still-plausible numbers.

Hex bit patterns rather than %.17g on purpose: `1.29` must hash the same in Python and in C++,
and the only representation guaranteed to agree is the one that does no formatting at all.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy
import rdkit
from rdkit import Chem
from rdkit.Chem import MolSurf
from rdkit.Chem.EState import EState_VSA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "vsa_tables.h"
PINNED_RDKIT = "2025.9.2"

MAX_Z = 118

# Bond::BondType values RDKit's getLabuteAtomContribs() treats specially.  The `< 4` guard in
# MolSurf.cpp means every type from QUADRUPLE(4) upwards -- including AROMATIC(12) and
# DATIVE(17) -- shortens the bond by NOTHING when the aromatic FLAG is clear.
BOND_SCALE = [0.1, 0.0, 0.2, 0.3]


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def canonical(tables: list[tuple[str, list[float]]]) -> str:
    """The byte string the spec hash is taken over.  src/hume_core/vsa_bins.h reproduces this
    exactly; see spec_string() there."""
    out = []
    for name, vals in tables:
        for i, v in enumerate(vals):
            out.append(f"{name}[{i}]=0x{bits(v)}")
    return "\n".join(out) + "\n"


def main() -> int:
    pt = Chem.GetPeriodicTable()
    rb0 = []
    for z in range(MAX_Z + 1):
        try:
            rb0.append(float(pt.GetRb0(z)))
        except Exception:
            rb0.append(0.0)

    # PeriodicTable::getNOuterElecs(z), the `dv` term of the E-state intrinsic state.  It is part
    # of THIS spec because EState_VSA / VSA_EState bin on the E-state index and nothing else in
    # the repo pins this table -- hume_blocks.h's n_outer() is a hand-written switch.
    nouter = []
    for z in range(MAX_Z + 1):
        try:
            nouter.append(float(pt.GetNOuterElecs(z)))
        except Exception:
            nouter.append(0.0)

    tables = [
        ("RB0", rb0),
        ("NOUTER", nouter),
        ("BOND_SCALE", BOND_SCALE),
        ("LOGP_BINS", [float(x) for x in MolSurf.logpBins]),
        ("MR_BINS", [float(x) for x in MolSurf.mrBins]),
        ("CHG_BINS", [float(x) for x in MolSurf.chgBins]),
        ("ESTATE_BINS", [float(x) for x in EState_VSA.estateBins]),
        ("VSA_BINS", [float(x) for x in EState_VSA.vsaBins]),
    ]
    spec = hashlib.sha256(canonical(tables).encode()).hexdigest()

    # Informational only, and deliberately separate from `spec`: these move on a copyright edit.
    fh = {}
    for mod in (MolSurf, EState_VSA):
        p = Path(mod.__file__)
        fh[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    L = []
    a = L.append
    a("// GENERATED by cpp/gen_vsa_tables.py -- do not edit.")
    a("//")
    a("// The numeric specification of the VSA-binning descriptor family: Labute ASA's covalent")
    a("// radii and bond-shortening constants, and the five arrays of bin edges that SlogP_VSA,")
    a("// SMR_VSA, PEOE_VSA, EState_VSA and VSA_EState bucket their per-atom property with.")
    a("//")
    a("// PROVENANCE.  `spec` is sha256 over the IEEE-754 bit patterns of every number below, in")
    a("// the canonical rendering cpp/gen_vsa_tables.py:canonical() defines and")
    a("// src/hume_core/vsa_bins.h:spec_string() reproduces.  vsa_bins.h RECOMPUTES it at load")
    a("// and throws if it disagrees, so editing a bin edge here is caught in C++ with no RDKit")
    a("// in the process.  The two `file` hashes are INFORMATIONAL: they move when a comment or a")
    a("// copyright year moves, which is exactly the false alarm the spec hash exists to avoid.")
    a("//")
    a("// Regenerate under THE PIN.  An unpinned `--with rdkit` silently becomes a different")
    a("// oracle and the exactness check cannot see it, because both sides move together:")
    a('//     uv run --isolated --python 3.12 --with "rdkit==2025.9.2" --with "numpy==2.4.6" \\')
    a("//            python cpp/gen_vsa_tables.py")
    a("//")
    a(f"//   rdkit   {rdkit.__version__}   (pin: {PINNED_RDKIT})")
    a(f"//   numpy   {numpy.__version__}")
    a(f"//   python  {sys.version.split()[0]}")
    a(f"//   spec    sha256(numbers) = {spec}")
    for k, v in sorted(fh.items()):
        a(f"//   file    sha256({k}) = {v}   (informational)")
    a("")
    a("#ifndef HUME_VSA_TABLES_H")
    a("#define HUME_VSA_TABLES_H")
    a("")
    a("namespace vsa_tbl {")
    a("")
    a('constexpr char SPEC_SHA256[] = "%s";' % spec)
    a("")
    a("// PeriodicTable::getRb0(z).  Index is the atomic number; index 0 and everything past Cm")
    a("// is 0.0, which is what RDKit's own table holds -- an element with no Rb0 gets a sphere")
    a("// of radius zero and contributes nothing, rather than being an error.")
    a(f"constexpr int MAX_Z = {MAX_Z};")
    a(f"constexpr double RB0[MAX_Z + 1] = {{")
    for i in range(0, MAX_Z + 1, 8):
        chunk = ", ".join(f"{v!r}" for v in rb0[i:i + 8])
        a(f"    {chunk},")
    a("};")
    a("")
    a("// PeriodicTable::getNOuterElecs(z).  rdkit/Chem/EState/EState.py:EStateIndices computes")
    a("// dv = GetNOuterElecs(Z) - GetTotalNumHs(); this is that table, read out of RDKit rather")
    a("// than transcribed from a periodic chart.")
    a("constexpr int NOUTER[MAX_Z + 1] = {")
    for i in range(0, MAX_Z + 1, 16):
        a("    " + ", ".join(str(int(v)) for v in nouter[i:i + 16]) + ",")
    a("};")
    a("")
    a("// getLabuteAtomContribs(): `const double bondScaleFacts[4] = {.1, 0, .2, .3};`, indexed")
    a("// by Bond::BondType.  0 UNSPECIFIED, 1 SINGLE, 2 DOUBLE, 3 TRIPLE.  An aromatic-FLAGGED")
    a("// bond uses index 0 whatever its type; a type >= 4 with the flag clear shortens by 0.")
    a("constexpr double BOND_SCALE[4] = {%s};" % ", ".join(repr(v) for v in BOND_SCALE))
    a("")
    for name, vals, where in (
        ("LOGP_BINS", MolSurf.logpBins, "MolSurf.cpp calcSlogP_VSA / MolSurf.py logpBins"),
        ("MR_BINS", MolSurf.mrBins, "MolSurf.cpp calcSMR_VSA / MolSurf.py mrBins"),
        ("CHG_BINS", MolSurf.chgBins, "MolSurf.cpp calcPEOE_VSA / MolSurf.py chgBins"),
        ("ESTATE_BINS", EState_VSA.estateBins, "EState_VSA.py estateBins (no C++ equivalent)"),
        ("VSA_BINS", EState_VSA.vsaBins, "EState_VSA.py vsaBins (no C++ equivalent)"),
    ):
        a(f"// {where}")
        a(f"constexpr int N_{name} = {len(vals)};")
        a(f"constexpr double {name}[N_{name}] = {{%s}};"
          % ", ".join(repr(float(v)) for v in vals))
        a("")
    a("}  // namespace vsa_tbl")
    a("")
    a("#endif  // HUME_VSA_TABLES_H")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    print(f"  rdkit {rdkit.__version__}  numpy {numpy.__version__}  python {sys.version.split()[0]}")
    print(f"  spec sha256 = {spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
