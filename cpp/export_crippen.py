"""Generate the Crippen tables for C++, and dump molecules for the C++ typer to chew on.

Two jobs, deliberately in one re-runnable file:

  1. `tables`  -- read RDKit's own `Data/Crippen.txt` and emit `cpp/crippen_tables.h`.
                  NOTHING is hand-copied. The header carries the RDKit version and the sha256
                  of the source file, so an RDKit bump shows up as a visible diff instead of a
                  silent numeric drift. This repo has already lost four rounds each to a
                  hand-typed HallKierAlpha table and a misread BCUT2D convention.

  2. `mols`    -- dump the graph features the typer needs plus RDKit's own per-atom
                  (logP, MR) reference, so `crippen.cpp` can be checked atom-by-atom rather
                  than on molecule totals (two compensating per-atom errors cancel in a sum).

Three facts about the data file that a casual reader gets wrong, all established empirically
(see the module docstring of verify_crippen.py for the evidence):

  * There are **110** rows, not 101. Nine rows have a BLANK MR field (N10, N12, O12, three Hal
    rows, three Me2 rows). RDKit reads a blank MR as 0.0 -- it does not skip the row. A parser
    that does `float(field)` inside a try/except and `continue`s on failure silently deletes
    nine live pattern classes, including every metal and every halide anion.
  * Row order in the file IS the priority order, and the flips are deliberate (the file says so
    in its own Notes column for O12-before-O7 and S2-before-S1). Classes are contiguous, so
    RDKit's group-by-class iteration and plain row order are the same sequence.
  * **Data/Crippen.txt is STALE for two rows.** `rdMolDescriptors._CalcCrippenContribs` is C++
    and does NOT read the shipped file -- it uses a table compiled into libRDKitDescriptors, and
    in that table rows 41 and 42 use ELEMENT negation where the text file still has ALIPHATIC-
    SYMBOL negation. See CPP_OVERRIDES below. Anyone who treats the .txt as the specification
    mis-types every hydrogen on an aromatic nitrogen.

    .venv/bin/python cpp/export_crippen.py tables
    .venv/bin/python cpp/export_crippen.py mols cpp/hard.smi 100000
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent

# RDKit bond types as a 4-bit code the C++ matcher can mask against. The order bit and the
# aromatic FLAG are kept separate on purpose: SMARTS `-` asks "is the bond order single", `:`
# asks "is the bond flagged aromatic", and the default bond is the OR of the two. Collapsing
# them into one enum would quietly answer a different question for a single bond between two
# aromatic rings (biphenyl), which is exactly the C20-vs-C19 distinction.
_BIT_SINGLE, _BIT_DOUBLE, _BIT_TRIPLE, _BIT_AROM = 1, 2, 4, 8


# ---------------------------------------------------------------------------------------------
# The two rows where the shipped text file disagrees with the compiled C++ table.
#
# EVIDENCE, reproducible:
#   strings -a .venv/.../rdkit/.dylibs/libRDKitDescriptors.1.dylib
# between "[CH4]" and "[#72,#73,...]" holds exactly 110 SMARTS, in exactly the same order as
# Data/Crippen.txt, and exactly two of them differ -- these two. The behavioural consequence is
# checked by _assert_overrides() below on every run: an explicit H on an aromatic nitrogen is
# typed H3 by RDKit, which is only possible if row 42's negation is element-based
# ([!#6;!#7;!#8], which excludes n) and not aliphatic-symbol-based ([!C;!N;!O], which admits n
# because aromatic n does not satisfy the aliphatic primitive N).
#
# If a future RDKit fixes the text file, _assert_overrides() keeps passing and the override
# becomes a no-op; if it changes the C++ behaviour instead, the assertion fires.
CPP_OVERRIDES = {
    41: ("[#1]O[!C;!N;!O;!S]", "[#1]O[!#6;!#7;!#8;!#16]"),
    42: ("[#1][!C;!N;!O]",     "[#1][!#6;!#7;!#8]"),
}

# (SMILES, atom index, expected row index) -- probes that pin the override down behaviourally.
_PROBES = [
    ("[2H]n1cccc1",      0, 43),   # H on aromatic n: 42 must NOT claim it -> falls to [#1][#7]
    ("[2H]O[n+]1ccccc1", 0, 44),   # H-O-aromatic n : 41 must NOT claim it -> falls to [#1]O[#7]
    ("[2H]SC",           0, 42),   # H on S         : 42 still fires
    ("[2H]O[Si](C)(C)C", 0, 41),   # H-O-Si         : 41 still fires
    ("[2H]OC(C)=O",      0, 45),   # the one depth-3 row
    ("[2H][O-]",         0, 47),   # neither 42 nor anything else -> HS
]


def crippen_path() -> Path:
    return Path(os.path.normpath(
        os.path.join(os.path.dirname(Chem.__file__), "..", "Data", "Crippen.txt")))


def read_rows() -> list[tuple[str, str, float, float]]:
    """(class, smarts, logP, MR) in file order == RDKit's priority order, C++ overrides applied.

    Mirrors rdkit.Chem.Crippen._ReadPatts exactly, blank-MR handling included.
    """
    rows = []
    for line in crippen_path().read_text().splitlines(keepends=True):
        if not line or line[0] == "#":
            continue
        f = line.split("\t")
        if len(f) >= 4 and f[0] != "" and f[1] != "SMARTS":
            rows.append((f[0], f[1], float(f[2]), float(f[3]) if f[3] != "" else 0.0))
    for i, (was, now) in CPP_OVERRIDES.items():
        name, sma, lp, mr = rows[i]
        if sma == now:
            continue                              # RDKit fixed the text file; nothing to do
        if sma != was:
            raise SystemExit(
                f"row {i} of Crippen.txt is now {sma!r}; the C++ override expected {was!r} or "
                f"{now!r}. Re-derive the override from libRDKitDescriptors before trusting this.")
        rows[i] = (name, now, lp, mr)
    return rows


def _assert_overrides() -> None:
    """RDKit's own row index for a handful of hydrogens. Cheap, and it fails loudly."""
    for smi, idx, want in _PROBES:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise SystemExit(f"probe {smi!r} no longer parses")
        types = [0] * m.GetNumAtoms()
        rdMolDescriptors._CalcCrippenContribs(m, False, types, [""] * m.GetNumAtoms())
        if types[idx] != want:
            raise SystemExit(
                f"probe {smi!r} atom {idx}: RDKit says row {types[idx]}, expected {want}. "
                "The compiled Crippen table has changed -- re-derive CPP_OVERRIDES, do not "
                "adjust the probe.")


def gen_tables() -> None:
    _assert_overrides()
    rows = read_rows()
    src = crippen_path()
    sha = hashlib.sha256(src.read_bytes()).hexdigest()

    # (logP, MR) -> class is injective in this table; the verifier leans on that to name the
    # class RDKit picked without replaying 110 substructure searches per molecule.
    seen: dict[tuple[float, float], str] = {}
    for name, _sma, lp, mr in rows:
        prev = seen.setdefault((lp, mr), name)
        if prev != name:
            raise SystemExit(f"(logP,MR) pair {lp},{mr} is shared by {prev} and {name}; "
                             "verify_crippen.py's class attribution is no longer sound")

    classes: list[str] = []
    for name, *_ in rows:
        if name not in classes:
            classes.append(name)

    out = [
        "// GENERATED by cpp/export_crippen.py -- DO NOT EDIT BY HAND.",
        "//",
        "// Wildman-Crippen atom contributions, transcribed mechanically from RDKit's own",
        "// shipped data file so that an RDKit upgrade shows up here as a diff.",
        f"//   rdkit  : {rdkit.__version__}",
        f"//   source : {src}",
        f"//   sha256 : {sha}",
        f"//   rows   : {len(rows)}   classes: {len(classes)}",
        "//",
        "// Row order is the priority order: RDKit assigns each atom the FIRST row it matches,",
        "// and the file's own Notes column flags two order flips (O12 before O7, S2 before S1)",
        "// as intentional. Nine rows carry a BLANK MR, which RDKit reads as 0.0.",
        "//",
        "// Rows 41 and 42 are NOT what Data/Crippen.txt says. The .txt is stale; the table",
        "// compiled into libRDKitDescriptors -- which is what _CalcCrippenContribs actually",
        "// runs -- negates by ELEMENT there, not by aliphatic symbol:",
    ] + [
        f"//   row {i}: file {was!r}  ->  C++ {now!r}"
        for i, (was, now) in sorted(CPP_OVERRIDES.items())
    ] + [
        "// export_crippen.py re-checks that behaviour against RDKit on every run.",
        "#pragma once",
        "",
        "namespace crippen {",
        "",
        f"constexpr int N_ROWS = {len(rows)};",
        f"constexpr int N_CLASSES = {len(classes)};",
        "",
        "struct Row { const char* cls; const char* smarts; double logp; double mr; };",
        "",
        "inline constexpr Row ROWS[N_ROWS] = {",
    ]
    for i, (name, sma, lp, mr) in enumerate(rows):
        esc = sma.replace("\\", "\\\\").replace('"', '\\"')
        out.append(f'    /* {i:3d} */ {{"{name}", "{esc}", {lp!r}, {mr!r}}},')
    out += [
        "};",
        "",
        "inline constexpr const char* CLASS_NAMES[N_CLASSES] = {",
        "    " + ", ".join(f'"{c}"' for c in classes),
        "};",
        "",
        "// Row -> class index, for the per-class exactness breakdown.",
        "inline constexpr int ROW_CLASS[N_ROWS] = {",
        "    " + ", ".join(str(classes.index(n)) for n, *_ in rows),
        "};",
        "",
        "// Atoms matching NO row get (0.0, 0.0): RDKit initialises the contribution vector to",
        "// zero and only overwrites entries a pattern claims. Verified on [Xe], [U], [He], [*].",
        "constexpr double UNMATCHED_LOGP = 0.0;",
        "constexpr double UNMATCHED_MR = 0.0;",
        "",
        "}  // namespace crippen",
        "",
    ]
    p = HERE / "crippen_tables.h"
    p.write_text("\n".join(out))
    print(f"wrote {p}  |  {len(rows)} rows, {len(classes)} classes, rdkit {rdkit.__version__}")
    print(f"  sha256(Crippen.txt) = {sha}")


def bond_code(b: Chem.Bond) -> int:
    t = b.GetBondType()
    c = 0
    if t == Chem.BondType.SINGLE:
        c |= _BIT_SINGLE
    elif t == Chem.BondType.DOUBLE:
        c |= _BIT_DOUBLE
    elif t == Chem.BondType.TRIPLE:
        c |= _BIT_TRIPLE
    if b.GetIsAromatic():
        c |= _BIT_AROM
    return c


def dump_mols(src: str, n_want: int) -> None:
    """cpp/crip_mols.txt:  n_mols / per mol: `nat nbond` + atom lines + bond lines + ref lines.

    Atom line:  Z  degree  nH_prop  charge  aromatic
      `nH_prop` is GetTotalNumHs(False) -- implicit + the numExplicitHs property, NOT counting
      neighbouring H ATOMS. That is the quantity hume.cpp already carries. The C++ typer derives
      the two things SMARTS actually asks for from it and the graph:
          SMARTS H  ==  nH_prop + (number of neighbours with Z==1)   [ == GetTotalNumHs(True) ]
          SMARTS X  ==  degree + nH_prop                             [ == GetTotalDegree()    ]
      Both verified against RDKit on [2H]C([2H])([2H])O, where they differ.

    Bond line:  u  v  code    (code = bitmask, see _BIT_* above)
    Ref line :  row  logP  MR   per atom, from rdMolDescriptors._CalcCrippenContribs

    `row` is RDKit's OWN winning row index, obtained via the `atomTypes` out-parameter (which is
    only filled if the list handed in is pre-sized to the atom count -- pass an empty list and it
    silently stays empty). Comparing row indices is strictly stronger than comparing the (logP,
    MR) pair: two classes could in principle carry the same numbers, and the pair alone cannot
    tell "right answer for the wrong reason" from "right answer". 255 means RDKit matched nothing.
    """
    _assert_overrides()
    smis = Path(src).read_text().split()[:n_want]
    out, kept = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        na = m.GetNumAtoms()
        types, labels = [0] * na, [""] * na
        crip = rdMolDescriptors._CalcCrippenContribs(m, False, types, labels)
        # An unmatched atom leaves atomTypes at its 0 initialiser, which collides with row 0
        # ([CH4]); the LABEL is what stays empty. Use that to tell the two apart.
        types = [t if lb else 255 for t, lb in zip(types, labels)]
        atoms = [f"{a.GetAtomicNum()} {a.GetDegree()} {a.GetTotalNumHs()} "
                 f"{a.GetFormalCharge()} {int(a.GetIsAromatic())}" for a in m.GetAtoms()]
        bonds = [f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} {bond_code(b)}"
                 for b in m.GetBonds()]
        refs = [f"{t} {lp:.17g} {mr:.17g}" for t, (lp, mr) in zip(types, crip)]
        blk = [f"{m.GetNumAtoms()} {m.GetNumBonds()}"] + atoms + bonds + refs
        out.append("\n".join(blk))
        kept.append(s)

    p = HERE / "crip_mols.txt"
    p.write_text(f"{len(kept)}\n" + "\n".join(out) + "\n")
    (HERE / "crip_mols.smi").write_text("\n".join(kept) + "\n")
    nat = sum(int(o.split("\n", 1)[0].split()[0]) for o in out)
    nh = sum(1 for s in kept if "[H" in s or "H]" in s)
    print(f"wrote {p}  |  {len(kept):,} molecules, {nat:,} atoms "
          f"(mean {nat/max(len(kept),1):.1f}), {nh:,} SMILES carrying a bracketed H")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tables"
    if cmd == "tables":
        gen_tables()
    elif cmd == "mols":
        dump_mols(sys.argv[2] if len(sys.argv) > 2 else str(HERE / "hard.smi"),
                  int(sys.argv[3]) if len(sys.argv) > 3 else 100_000)
    else:
        raise SystemExit(__doc__)
