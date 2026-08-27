"""Prove cpp/crippen.cpp reproduces RDKit's Crippen typing EXACTLY, and leave nothing untested.

    .venv/bin/python cpp/verify_crippen.py          # stress corpus, then cpp/hard.smi
    .venv/bin/python cpp/verify_crippen.py stress   # just build/refresh the stress corpus

WHAT IS BEING COMPARED, AND WHY THAT AND NOT SOMETHING EASIER
The reference is `rdMolDescriptors._CalcCrippenContribs` per ATOM, and specifically the ROW INDEX
it reports through its `atomTypes` out-parameter -- not the molecule's MolLogP, and not even the
per-atom (logP, MR) pair. A molecule total lets two per-atom errors cancel. A value pair lets an
atom be right for the wrong reason. The row index is the thing the algorithm actually decides, so
that is what gets checked; the value pair is then checked separately as an assertion about the
generated table. Nothing here has a tolerance: these are table lookups, so the answer is
bit-identical or it is a bug.

THREE THINGS THIS COST A ROUND TO LEARN, ALL WORTH WRITING DOWN

1. HYDROGENS. The H* rows begin with `[#1]`, i.e. they match EXPLICIT hydrogen ATOMS.
   `_CalcCrippenContribs` does NOT add hydrogens and does NOT fold an H's contribution onto its
   heavy neighbour: on an implicit-H molecule it returns one pair per HEAVY atom and the H rows
   simply never fire. Evidence -- `CO` gives [(-0.2035, 2.753), (-0.2893, 0.8238)], two entries,
   while `AddHs(CO)` gives those two plus 3xH1 and 1xH2. So the typer must be run on whatever
   graph it is handed, and HUME hands it an implicit-H graph.
   Hydrogens are NOT irrelevant, though: `MolFromSmiles` keeps an H ATOM in the graph whenever
   removeHs cannot fold it away -- isotopes ([2H], [3H]), H2, charged H, H on a dummy. 1,198 of
   the 100,000 molecules in cpp/hard.smi carry 2,447 such atoms, so the H rows do fire in the
   corpus, and the stress corpus below exercises every one of them deliberately.
   (For completeness: `CalcCrippenDescriptors(mol, includeHs=True)` -- what MolLogP uses -- is
   exactly `_CalcCrippenContribs(AddHs(mol))` summed, verified on several molecules. It is a
   real AddHs, not nH * H1.)

2. THE SHIPPED Data/Crippen.txt IS NOT THE SPECIFICATION. `_CalcCrippenContribs` is C++ and uses
   a table compiled into libRDKitDescriptors. That table has the same 110 rows in the same order,
   but TWO of them differ from the text file:
       row 41  file [#1]O[!C;!N;!O;!S]   ->  C++ [#1]O[!#6;!#7;!#8;!#16]
       row 42  file [#1][!C;!N;!O]       ->  C++ [#1][!#6;!#7;!#8]
   The file's forms negate ALIPHATIC element symbols, so an aromatic n satisfies `!N` and row 42
   would claim any H on an aromatic nitrogen as H2. RDKit types it H3. Two molecules in
   cpp/hard.smi hit exactly this. See CPP_OVERRIDES in export_crippen.py for the derivation and
   the behavioural probes that re-check it on every run.

3. THERE ARE 110 ROWS, NOT 101. Nine rows leave the MR column blank (N10, N12, O12, three Hal
   rows, three Me2 rows) and RDKit reads blank as 0.0. Dropping rows whose MR fails to parse
   deletes every metal and every halide anion from the table.

An atom that matches no row keeps (0.0, 0.0) -- verified on [Xe], [U], [He], [Ar] and [*].
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

import export_crippen

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------
# A corpus aimed at the rows real chemistry never reaches.
#
# cpp/hard.smi is 100k adversarial molecules and still leaves 9 of the 110 rows cold: methane,
# H-O-N, the halide anions, the alkali cations, and all three transition-metal rows. A class that
# never fires is untested, not passing, so these exist to make the report honest.
# ---------------------------------------------------------------------------------------------
STRESS = [
    # -- the rows hard.smi never reaches -------------------------------------------------------
    "C", "[CH4]",                                                        # C1  [CH4]
    "[2H]ON", "[2H]ON(C)C", "[2H]ONC(=O)C", "[2H]On1ccccc1",             # H3  [#1]O[#7]
    "[F-]", "[Cl-]", "[Br-]", "[I-]", "CC[N+](C)(C)C.[Cl-]",             # Hal [#9,#17,#35,#53;-]
    "C[I+]C", "[I+3]", "[I+2]",                                          # Hal [#53;+,+2,+3]
    "[Li+]", "[Na+]", "[K+]", "[Rb+]", "[Cs+]", "CC(=O)[O-].[Na+]",      # Hal [+;#3,...]
    "[Li]", "[Na]", "[K]", "[Rb]", "[Cs]", "C[Li]", "C[Na]",             # Me1 alkali, neutral
    "[Be]", "[Mg]", "[Ca]", "[Sr]", "[Ba]", "[Mg+2]", "[Ca+2]", "[Ba+2]",  # Me1 group 2
    "[Sc]", "[Ti]", "[V]", "[Cr]", "[Mn]", "[Fe]", "[Co]", "[Ni]",       # Me2 3d
    "[Cu]", "[Zn]", "[Fe+2]", "[Fe+3]", "[Cu+2]", "[Zn+2]",
    "[Y]", "[Zr]", "[Nb]", "[Mo]", "[Tc]", "[Ru]", "[Rh]", "[Pd]",       # Me2 4d
    "[Ag]", "[Cd]", "[Ag+]", "[Cd+2]",
    "[Hf]", "[Ta]", "[W]", "[Re]", "[Os]", "[Ir]", "[Pt]", "[Au]",       # Me2 5d
    "[Hg]", "[Au+]", "[Hg+2]", "[Pt+2]",
    # -- elements no row claims: the contribution must stay (0.0, 0.0) -------------------------
    "[Xe]", "[He]", "[Ne]", "[Ar]", "[Kr]", "[Rn]", "[U]", "[Th]", "[La]", "[Ce]", "[Gd]",
    "[Ra]", "[Fr]", "[Ac]", "*", "*C", "*c1ccccc1", "[Xe]C",
    # -- Me1's remaining groups ----------------------------------------------------------------
    "B(O)O", "[BH3-]C", "[Al+3]", "[Ga]", "[In]", "[Tl]", "[Tl+]", "B1OB(O)OB(O)O1",
    "[SiH4]", "C[Si](C)(C)C", "[Ge]", "[GeH4]", "[Sn]", "C[Sn](C)(C)C", "[Pb]", "[Pb+2]",
    "[AsH3]", "C[As](C)C", "[Sb]", "[Bi]", "[Bi+3]",
    "[SeH2]", "C[Se]C", "[Te]", "C[Te]C", "[Po]", "c1ccc[se]1", "c1ccc[te]1",
    # -- every hydrogen row, in every environment it can reach ---------------------------------
    "[2H]C", "[2H][2H]", "[H][H]", "[2H]c1ccccc1", "[3H]C", "[2H]C(=O)O",   # H1
    "[2H]OC", "[2H]OC(C)(C)C", "[2H]Oc1ccccc1", "[2H]O[Si](C)(C)C",         # H2 rows 40, 41
    "[2H]O[Ge](C)(C)C", "[2H]O[Se]C", "[2H]OP(=O)(O)O", "[2H]O[As](C)C",
    "[2H]SC", "[2H][SeH]", "[2H]P(C)C", "[2H][Cl]", "[2H][Si](C)(C)C",      # H2 row 42
    "[2H][B](C)C", "[2H][Fe]", "[2H][Te]C", "[2H][AsH2]",
    "[2H]N", "[2H]N(C)C", "[2H]n1cccc1", "[2H][n+]1ccccc1", "[2H][NH3+]",   # H3 row 43
    "[2H]NC(=O)C", "[2H]Nc1ccccc1",
    "[2H]OC(C)=O", "[2H]OC=O", "[2H]OC=N", "[2H]OC=S", "[2H]OC=C",          # H4 row 45
    "[2H]OO", "[2H]OOC", "[2H]OS", "[2H]OSC",                               # H4 row 46
    "[2H][O-]", "[2H][S-]", "[2H][N-]C", "[2H][OH2+]", "[2H][SH3+]",        # HS row 47
    # -- oxygen rows that are thin in hard.smi -------------------------------------------------
    "CC(=O)[O-]", "[O-]C(=O)C(=O)[O-]", "OC(=O)[O-]",                       # O12
    "C[O-]", "[O-]c1ccccc1", "C[S-]", "[O-]P(=O)(O)O", "[O-][Si](C)(C)C",   # O7 / O6 / O5
    "C[N+](=O)[O-]", "O=[N+]([O-])c1ccccc1", "C[S+](C)[O-]", "CS(=O)(=O)[O-]",
    "O=C=O", "O=C=S", "O=C(N)N", "O=CN", "O=C(F)F", "O=C(Cl)Cl", "O=C(O)O",
    "O=Cc1ccccc1", "O=C(C)c1ccccc1", "O=C(c1ccccc1)c1ccccc1", "O=c1cccc[nH]1",
    "O=C1CCCC1", "C=O", "CC=O", "O=P(O)(O)O", "O=S(C)C", "O=[Se](C)C",
    # -- nitrogen rows that are thin ------------------------------------------------------------
    "[N-]=[N+]=NC", "C[N+]#N", "[N-]=[N+]=[N-]", "[NH-]C", "[N-3]",         # N14
    "N#[N+]C", "C[N+](C)(C)C", "C[N+](C)=C", "N#N", "[NH4+]", "[NH3+]C",
    "c1ccncc1", "c1cc[nH+]cc1", "c1cc[n-]c1", "n1ccccc1",
    # -- carbon rows that are thin --------------------------------------------------------------
    "c1ccccc1[Si](C)(C)C", "c1ccccc1[Se]C", "c1ccccc1B(O)O", "c1ccccc1P(C)C",  # C13 (P NOT excl.)
    "C[Si](C)(C)C", "C[Se]C", "CB(O)O", "CP(C)C", "C[Ge](C)(C)C",              # C27 (P IS excl.)
    "C=C=C", "CC#CC", "C#C", "C=[Si](C)C", "C=[N+]=[N-]", "CC(=C)c1ccccc1",
    "c1ccc2ccccc2c1", "c1ccc2c(c1)cccc2-c1ccccc1", "Cc1ccccc1", "Nc1ccccc1",
    "Oc1ccccc1", "Sc1ccccc1", "Fc1ccccc1", "Clc1ccccc1", "Brc1ccccc1", "Ic1ccccc1",
    "O=c1ccccc1", "C=c1ccccc1", "N=c1cccc[nH]1",
    # -- sulfur rows ----------------------------------------------------------------------------
    "CSC", "CS(C)=O", "CS(C)(=O)=O", "[S-2]", "[S-]C", "C[S+](C)C", "[SH-]",
    "c1ccsc1", "S=C(N)N", "S=P(C)(C)C", "S=S",
]


def build_stress() -> Path:
    """Write the stress corpus, keeping only SMILES RDKit actually parses."""
    good, bad = [], []
    for s in STRESS:
        (good if Chem.MolFromSmiles(s) is not None else bad).append(s)
    p = HERE / "crip_stress.smi"
    p.write_text("\n".join(good) + "\n")
    print(f"stress corpus: {len(good)} molecules parsed, {len(bad)} rejected by RDKit")
    if bad:
        print("  rejected:", " ".join(bad))
    return p


def replay(smi: str) -> list[int]:
    """An INDEPENDENT Python reading of the specification: match every row's SMARTS with RDKit's
    generic engine, in row order, and give each atom its first hit.

    This exists to separate two claims that the main check bundles together -- "crippen.cpp
    implements these 110 SMARTS" and "these 110 SMARTS in this order are what RDKit does". The
    main check uses RDKit's own reported row index, so it already settles the second; this
    settles the first through a completely different matcher, and in particular it is what makes
    the two overridden H rows falsifiable rather than a convenient assumption.
    """
    rows = export_crippen.read_rows()
    qs = [Chem.MolFromSmarts(s) for _, s, _, _ in rows]
    m = Chem.MolFromSmiles(smi)
    out = [-1] * m.GetNumAtoms()
    for ri, q in enumerate(qs):
        if q is None:
            continue
        for match in m.GetSubstructMatches(q, False, False):
            if out[match[0]] < 0:
                out[match[0]] = ri
    return out


def check_replay(smis: list[str]) -> int:
    rows = export_crippen.read_rows()
    qs = [(Chem.MolFromSmarts(s), i) for i, (_, s, _, _) in enumerate(rows)]
    bad = 0
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        n = m.GetNumAtoms()
        mine = [-1] * n
        for q, ri in qs:
            if q is None:
                continue
            for match in m.GetSubstructMatches(q, False, False):
                if mine[match[0]] < 0:
                    mine[match[0]] = ri
        types, labels = [0] * n, [""] * n
        rdMolDescriptors._CalcCrippenContribs(m, False, types, labels)
        ref = [t if lb else -1 for t, lb in zip(types, labels)]
        for i, (a, b) in enumerate(zip(mine, ref)):
            if a != b:
                bad += 1
                if bad <= 20:
                    print(f"  REPLAY MISMATCH {smi} atom {i}: "
                          f"SMARTS-in-order says row {a}, RDKit says row {b}")
    return bad


def run(corpus: Path, label: str) -> int:
    print(f"\n{'=' * 88}\n{label}  ({corpus.name})\n{'=' * 88}")
    export_crippen.dump_mols(str(corpus), 1_000_000)
    r = subprocess.run([str(HERE / "crippen"), "paranoid", str(HERE / "crip_mols.txt")])
    return r.returncode


def main() -> None:
    stress = build_stress()
    if len(sys.argv) > 1 and sys.argv[1] == "stress":
        return

    smis = stress.read_text().split()
    print("\nindependent Python replay of the 110 SMARTS in row order, vs RDKit's own row index:")
    bad = check_replay(smis)
    print(f"  {len(smis)} molecules, {bad} atom(s) where the two disagree"
          + ("" if bad else "  -- the row list and its order are confirmed"))

    rc = run(stress, "STRESS CORPUS -- the rows real chemistry does not reach")
    # The two corpora are complementary: hard.smi leaves 9 rows cold (methane, H-O-N, the halide
    # anions, the alkali cations, all three transition-metal rows) and the stress list leaves 18
    # cold for the opposite reason. Run the union so the headline report has NO untested row.
    both = HERE / "crip_all.smi"
    both.write_text(stress.read_text().rstrip("\n") + "\n"
                    + (HERE / "hard.smi").read_text().rstrip("\n") + "\n")
    rc |= run(both, "STRESS + HARD -- the union, so every one of the 110 rows is exercised")
    sys.exit(rc)


if __name__ == "__main__":
    main()
