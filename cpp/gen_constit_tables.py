"""Generate cpp/constit_tables.h -- the numeric SPEC of the small constitutional families.

    uv run --isolated --no-project --python 3.11 --with "mordred==1.2.0" \
           --with "rdkit==2025.9.2" --with "numpy==1.26.4" python cpp/gen_constit_tables.py

WHAT THE SPEC IS.  Five of the families in src/hume_core/constit.h are a graph walk plus a table
lookup, and the table is the part that can silently be wrong.  Nothing here is transcribed from a
paper or from a docstring; every number is read out of the running mordred or the running RDKit.

  POL94       mordred/data/polarizalibity94.txt, via mordred._atomic_property.polarizability94.
              `bpol` is sum over bonds of |POL94[za] - POL94[zb]| on the H-ADDED molecule.  Note
              mordred's spelling of the filename; it is not a typo here.
  BONDI       mordred/VdwVolumeABC.py `atom_contrib`, which is 4/3 pi r^3 over its own
              `bondi_radii` dict of THIRTEEN elements.  The dict is the spec in two ways: it
              gives the radii, and MEMBERSHIP of it is what decides whether Vabc is a number or
              NaN -- mordred raises `unknown atom type` on a KeyError.  BONDI_OK carries the
              membership so a legitimate contribution can never be confused with an absent one.
  MONOISO     PeriodicTable::getMostCommonIsotopeMass(z).  rdkit's CalcExactMolWt uses this for
              every atom whose isotope is unset.
  AWEIGHT     PeriodicTable::getAtomicWeight(z).  Two jobs: it is the per-added-hydrogen term of
              _CalcMolWt, and comparing it against the boundary's `mass` column is how
              constit.h detects that an atom IS isotope-labelled without an isotope column.
  ELECTRON_MASS   CalcExactMolWt subtracts one electron mass per unit of formal charge.  The
              value is NOT rdkit's documented 0.00054857990946 -- see the note below, it was
              recovered from the running library because that is what makes the column bit-exact.
  LOGS_*      mordred/LogS.py `_smarts_logs`, IN INSERTION ORDER, plus the two regression
              constants of `logS = 0.89823 - 0.10369 * sqrt(MW)`.  The order is part of the spec:
              mordred accumulates `logS += count * coefficient` by iterating the dict, and float
              addition is not associative, so a reordering of these sixteen numbers is a
              different answer in the last bits.

THE ELECTRON MASS, and why it is measured rather than quoted.  Deriving ExactMolWt from the
boundary as `sum of masses` was wrong on 238 of 4,000 molecules by up to 1.6e-12 -- every one of
them carrying a formal charge.  rdkit's calcExactMW subtracts an electron mass per unit charge,
and doing that with the constant rdkit's own headers document (0.00054857990946) still left 238
molecules out.  Solving for the constant on [NH4+], [O-], [Cl-] and C[N+](C)(C)C gives
0.00054857991 -- eleven significant figures, not fourteen -- and applying THAT per atom, inside
the accumulation loop rather than once at the end, is bit-exact on 4,000/4,000.  Both details
matter: the same constant applied once at the end is still out on 78.

THE DRIFT GUARD HASHES THE SPEC, NOT THE FILE, for the reason cpp/gen_vsa_tables.py gives at
length: sha256 of a .py moves when a copyright year moves and stays still when a number is edited
to another plausible number.  `spec` is sha256 over the IEEE-754 bit patterns of every number
below, in a canonical rendering src/hume_core/constit.h reproduces and checks at load.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from math import pi
from pathlib import Path

import numpy
import rdkit
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "constit_tables.h"
PINNED_RDKIT = "2025.9.2"
PINNED_MORDRED = "1.2.0"

MAX_Z = 118

# Recovered from the running rdkit; see the module docstring.  Do not "correct" this to the CODATA
# value -- the column stops being bit-exact if you do.
ELECTRON_MASS = 0.00054857991


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def lit(x: float) -> str:
    """A C++ literal for one table entry.

    mordred's polarizability table is NOT dense -- `polarizability94[z]` is `nan` for every element
    the Handbook row does not cover, and mordred's `bpol` therefore returns NaN for such a molecule
    rather than raising.  Emitting those as 0.0 would silently turn a missing value into a real
    one, so they are emitted as a quiet NaN and allowed to propagate exactly as they do upstream.
    The spec hash is taken over the IEEE-754 BITS, and python's float('nan') and C++'s
    quiet_NaN() are both 0x7ff8000000000000 -- if a platform ever disagrees, constit.h's
    checkSpec() fails loudly instead of computing with a different table.
    """
    v = float(x)
    if v != v:
        return "std::numeric_limits<double>::quiet_NaN()"
    if v == float("inf"):
        return "std::numeric_limits<double>::infinity()"
    if v == float("-inf"):
        return "-std::numeric_limits<double>::infinity()"
    return repr(v)


def canonical(tables):
    out = []
    for name, vals in tables:
        for i, v in enumerate(vals):
            out.append("%s[%d]=0x%s" % (name, i, bits(v)))
    return "\n".join(out) + "\n"


def main() -> int:
    import mordred
    from mordred._atomic_property import polarizability94
    from mordred.VdwVolumeABC import bondi_radii, atom_contrib
    from mordred.LogS import _smarts_logs

    pt = Chem.GetPeriodicTable()

    pol = []
    for z in range(MAX_Z + 1):
        try:
            pol.append(float(polarizability94[z]))
        except Exception:
            pol.append(0.0)

    bondi = [0.0] * (MAX_Z + 1)
    bondi_ok = [0] * (MAX_Z + 1)
    for z, v in atom_contrib.items():
        if z <= MAX_Z:
            bondi[z] = float(v)
            bondi_ok[z] = 1
    # Recompute from the radii here as well, and refuse to write a header where the two disagree:
    # `atom_contrib` is a dict comprehension over `bondi_radii`, so if a future mordred changes
    # one and not the other this generator should stop rather than pick a winner.
    for sym, r in bondi_radii.items():
        z = pt.GetAtomicNumber(sym)
        want = 4.0 / 3.0 * pi * r ** 3
        assert bondi[z] == want, "VdwVolumeABC: atom_contrib disagrees with bondi_radii for " + sym

    mono = []
    aw = []
    for z in range(MAX_Z + 1):
        try:
            mono.append(float(pt.GetMostCommonIsotopeMass(z)))
        except Exception:
            mono.append(0.0)
        try:
            aw.append(float(pt.GetAtomicWeight(z)))
        except Exception:
            aw.append(0.0)

    logs_items = list(_smarts_logs.items())          # insertion order IS the summation order
    logs_smarts = [s for s, _ in logs_items]
    logs_coef = [float(c) for _, c in logs_items]

    tables = [
        ("POL94", pol),
        ("BONDI", bondi),
        ("BONDI_OK", [float(x) for x in bondi_ok]),
        ("MONOISO", mono),
        ("AWEIGHT", aw),
        ("ELECTRON_MASS", [ELECTRON_MASS]),
        ("LOGS_COEF", logs_coef),
        ("LOGS_CONST", [0.89823, -0.10369]),
    ]
    spec = hashlib.sha256(canonical(tables).encode()).hexdigest()

    fh = {}
    for mod in ("mordred/_atomic_property.py", "mordred/VdwVolumeABC.py", "mordred/LogS.py"):
        p = Path(mordred.__file__).parent.parent / mod
        if p.exists():
            fh[mod] = hashlib.sha256(p.read_bytes()).hexdigest()

    L = []
    a = L.append
    a("// GENERATED by cpp/gen_constit_tables.py -- do not edit.")
    a("//")
    a("// The numeric specification of the small constitutional families in")
    a("// src/hume_core/constit.h: mordred's 1994 atomic polarizabilities, the thirteen Bondi")
    a("// volume contributions VdwVolumeABC is defined over, the two mass tables CalcExactMolWt")
    a("// and _CalcMolWt read, and the sixteen Filter-it LogS coefficients IN THEIR SUMMATION")
    a("// ORDER.")
    a("//")
    a("// PROVENANCE.  `spec` is sha256 over the IEEE-754 bit patterns of every number below, in")
    a("// the canonical rendering cpp/gen_constit_tables.py:canonical() defines and")
    a("// src/hume_core/constit.h:specString() reproduces.  constit.h RECOMPUTES it on first use")
    a("// and throws if it disagrees, so hand-editing a number here is caught in C++ with no")
    a("// rdkit and no mordred in the process.  The `file` hashes are INFORMATIONAL only.")
    a("//")
    a("// Regenerate UNDER THE PIN.  mordred 1.2.0 needs python 3.11 (distutils) and numpy 1.x;")
    a("// asking for mordred alongside numpy 2 does not error, it silently resolves mordred DOWN")
    a("// to 0.6.0, which is a different library:")
    a('//     uv run --isolated --no-project --python 3.11 --with "mordred==1.2.0" \\')
    a('//            --with "rdkit==2025.9.2" --with "numpy==1.26.4" \\')
    a("//            python cpp/gen_constit_tables.py")
    a("//")
    a("//   mordred %s   (pin: %s)" % (mordred.__version__, PINNED_MORDRED))
    a("//   rdkit   %s   (pin: %s)" % (rdkit.__version__, PINNED_RDKIT))
    a("//   numpy   %s" % numpy.__version__)
    a("//   python  %s" % sys.version.split()[0])
    a("//   spec    sha256(numbers) = %s" % spec)
    for k, v in sorted(fh.items()):
        a("//   file    sha256(%s) = %s   (informational)" % (k, v))
    a("")
    a("#ifndef HUME_CONSTIT_TABLES_H")
    a("#define HUME_CONSTIT_TABLES_H")
    a("")
    a("#include <limits>")
    a("")
    a("namespace constit_tbl {")
    a("")
    a('constexpr char SPEC_SHA256[] = "%s";' % spec)
    a("constexpr int MAX_Z = %d;" % MAX_Z)
    a("")
    a("// mordred._atomic_property.polarizability94, i.e. mordred/data/polarizalibity94.txt")
    a("// (Handbook of Chemistry and Physics, 94th edition).  Index is the atomic number; an")
    a("// element mordred's table does not cover is 0.0 here, which is NOT a legal input -- a")
    a("// KeyError in mordred is a hard failure and constit.h reports one rather than summing 0.")
    a("constexpr double POL94[MAX_Z + 1] = {")
    for i in range(0, MAX_Z + 1, 6):
        a("    " + ", ".join(lit(v) for v in pol[i:i + 6]) + ",")
    a("};")
    a("")
    a("// mordred/VdwVolumeABC.py atom_contrib = 4/3 pi r^3 over its thirteen-element")
    a("// `bondi_radii` dict.  BONDI_OK is the DICT MEMBERSHIP, and it is load-bearing: Vabc is")
    a("// NaN for a molecule containing any element outside the thirteen, and a 0.0 contribution")
    a("// would silently be a number instead.")
    a("constexpr double BONDI[MAX_Z + 1] = {")
    for i in range(0, MAX_Z + 1, 4):
        a("    " + ", ".join(lit(v) for v in bondi[i:i + 4]) + ",")
    a("};")
    a("constexpr int BONDI_OK[MAX_Z + 1] = {")
    for i in range(0, MAX_Z + 1, 24):
        a("    " + ", ".join(str(v) for v in bondi_ok[i:i + 24]) + ",")
    a("};")
    a("")
    a("// PeriodicTable::getMostCommonIsotopeMass(z) -- CalcExactMolWt's per-element mass for an")
    a("// atom with no explicit isotope.")
    a("constexpr double MONOISO[MAX_Z + 1] = {")
    for i in range(0, MAX_Z + 1, 4):
        a("    " + ", ".join(lit(v) for v in mono[i:i + 4]) + ",")
    a("};")
    a("")
    a("// PeriodicTable::getAtomicWeight(z) -- _CalcMolWt's per-element mass, and the reference")
    a("// constit.h compares the boundary's `mass` column against to detect an isotope label")
    a("// without an isotope column at the boundary.")
    a("constexpr double AWEIGHT[MAX_Z + 1] = {")
    for i in range(0, MAX_Z + 1, 4):
        a("    " + ", ".join(lit(v) for v in aw[i:i + 4]) + ",")
    a("};")
    a("")
    a("// One electron mass per unit of formal charge, subtracted INSIDE CalcExactMolWt's atom")
    a("// loop.  Recovered from the running rdkit, not quoted: see the generator's docstring.")
    a("constexpr double ELECTRON_MASS = %s;" % lit(ELECTRON_MASS))
    a("")
    a("// mordred/LogS.py: logS = 0.89823 - 0.10369*sqrt(MW), then one `+= count * coef` per")
    a("// pattern IN DICT INSERTION ORDER.  The order is spec, not presentation.")
    a("constexpr double LOGS_A = %s;" % lit(0.89823))
    a("constexpr double LOGS_B = %s;" % lit(-0.10369))
    a("constexpr int N_LOGS = %d;" % len(logs_coef))
    a("constexpr double LOGS_COEF[N_LOGS] = {")
    for i in range(0, len(logs_coef), 4):
        a("    " + ", ".join(lit(v) for v in logs_coef[i:i + 4]) + ",")
    a("};")
    a("// The patterns the coefficients belong to, in the same order.  constit.h implements each")
    a("// as a predicate rather than running a matcher; these strings are here so the two can be")
    a("// read side by side, and cpp/verify_constit.py checks the predicates against rdkit's own")
    a("// SMARTS matcher pattern by pattern.")
    a("constexpr const char* LOGS_SMARTS[N_LOGS] = {")
    for s in logs_smarts:
        a('    "%s",' % s)
    a("};")
    a("")
    a("}  // namespace constit_tbl")
    a("")
    a("#endif  // HUME_CONSTIT_TABLES_H")

    OUT.write_text("\n".join(L) + "\n")
    print("wrote %s" % OUT)
    print("  mordred %s  rdkit %s  numpy %s  python %s"
          % (mordred.__version__, rdkit.__version__, numpy.__version__, sys.version.split()[0]))
    print("  spec sha256 = %s" % spec)
    print("  LOGS order  = %s" % logs_smarts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
