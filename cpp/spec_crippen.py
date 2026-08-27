"""Can Crippen's ordered SMARTS be replaced by a compiled atom typer?  YES -- and here is why.

    .venv/bin/python cpp/spec_crippen.py

This is the feasibility measurement that decided the design in cpp/crippen.cpp. It has been
rewritten after the implementation landed, because THREE of its original claims were wrong in
ways that would have shipped a wrong descriptor. Those corrections are the most valuable thing
in this file, so they come first.

  WRONG: "68 patterns"  ->  then "101 patterns"  ->  actually **110**.
    Nine rows leave the MR column blank (N10, N12, O12, three Hal rows, three Me2 rows). RDKit
    reads blank as 0.0; it does not skip the row. The old parser called float() on the field
    inside a try/except and `continue`d on failure, which silently deleted every metal and every
    halide anion from the table -- and, because those classes are rare, deleted them invisibly.

  WRONG: "the typer is entirely C++, so the shipped data file IS the accessible specification."
    The first half is true and the second does not follow. `_CalcCrippenContribs` uses a table
    compiled into libRDKitDescriptors, and that table has diverged from Data/Crippen.txt for two
    rows. The file negates ALIPHATIC element symbols where the C++ negates ELEMENTS:
        row 41  file [#1]O[!C;!N;!O;!S]   ->  C++ [#1]O[!#6;!#7;!#8;!#16]
        row 42  file [#1][!C;!N;!O]       ->  C++ [#1][!#6;!#7;!#8]
    An aromatic n satisfies `!N` (aromatic n is not the ALIPHATIC primitive N) but not `!#7`, so
    the file's version claims every hydrogen on an aromatic nitrogen as H2 while RDKit types it
    H3. Two molecules in cpp/hard.smi hit exactly that. The divergence is confirmed three ways:
    `strings` on the dylib, RDKit's own reported row index via the `atomTypes` out-parameter, and
    a Python replay of all 110 SMARTS in row order. See CPP_OVERRIDES in cpp/export_crippen.py.

  MISLEADING: "96.3% of atoms match more than one pattern, so list order IS the specification."
    True but nearly vacuous, and the section below now says why. Almost all of that overlap is
    one specific row plus a catch-all (CS/NS/OS/HS), which is the trivial kind. What actually
    matters is the SPECIFIC-vs-SPECIFIC rate, measured separately here -- and it is large enough
    that order really is load-bearing, just for different reasons than the headline suggested.

WHAT THE DESIGN ACTUALLY RESTS ON: 96 of the 110 rows never look past the typed atom's first
shell, and all 14 that do reach further are "one hop to a hinge atom, then inspect that atom's
other substituents" -- hydroxyl/enol hydrogens hinging on an aliphatic O, carbonyl/carboxylate
oxygens hinging on an aliphatic C. That is a compiled first-shell decision procedure plus two
hand-written second-shell walks, not a SMARTS engine.

Result: cpp/crippen.cpp reproduces RDKit's chosen row for 2,869,048 / 2,869,048 atoms across
100,243 molecules, with all 110 rows exercised, at ~1.3 us/mol against a ~130 us/mol cold RDKit.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

import export_crippen

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent

# The rows that claim an atom purely on its element, with no structural condition at all. Their
# whole job is to be last, so an overlap with one of them says nothing about the priority order.
CATCH_ALL = {"CS", "NS", "OS", "HS"}


def depth_of(smarts: str) -> int:
    """How many bonds from the mapped atom the pattern reaches: eccentricity of query atom 0."""
    p = Chem.MolFromSmarts(smarts)
    if p is None:
        return -1
    if p.GetNumAtoms() == 1:
        return 0
    return int(max(Chem.GetDistanceMatrix(p)[0]))


def main() -> None:
    rows = export_crippen.read_rows()          # 110 rows, C++ overrides applied
    print(f"{len(rows)} Crippen rows, {len({r[0] for r in rows})} classes\n")

    depths, deep = Counter(), []
    for name, sma, lp, mr in rows:
        d = depth_of(sma)
        depths[d] += 1
        if d >= 2:
            deep.append((name, sma, d))

    print("  reach from the typed atom (bonds):")
    for d in sorted(depths):
        lab = {0: "the atom alone", 1: "first shell only"}.get(d, f"{d} bonds out")
        print(f"    {d}  {depths[d]:3d} rows   {lab}")
    print(f"\n  the {len(deep)} rows that reach 2+ bonds, and the two families they form:")
    for name, sma, d in deep:
        print(f"    {name:5s} d={d}  {sma}")

    # --- is the priority order load-bearing, and in what way? ---------------------------------
    smis = HERE.joinpath("hard.smi").read_text().split()
    random.seed(5)
    pick = random.sample(smis, 1500)
    qs = [(n, Chem.MolFromSmarts(s)) for n, s, _, _ in rows]

    n_atoms = n_multi = n_specific = 0
    pairs = Counter()
    for s in pick:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        hits = [[] for _ in range(m.GetNumAtoms())]
        for ri, (name, q) in enumerate(qs):
            if q is None:
                continue
            for match in m.GetSubstructMatches(q, False, False):
                if ri not in hits[match[0]]:
                    hits[match[0]].append(ri)
        for h in hits:
            n_atoms += 1
            if len(h) > 1:
                n_multi += 1
            # the honest question: after dropping the catch-alls, do two SPECIFIC rows still
            # compete for this atom? That is where a reimplementation silently diverges.
            spec = [ri for ri in h if rows[ri][0] not in CATCH_ALL]
            if len(spec) > 1:
                n_specific += 1
                a, b = spec[0], spec[1]
                if rows[a][0] != rows[b][0]:          # same class twice is not a real collision
                    pairs[(rows[a][0], rows[b][0])] += 1

    print(f"\n  {n_atoms:,} atoms typed over {len(pick):,} molecules")
    print(f"  {n_multi:,} ({100 * n_multi / n_atoms:.1f}%) match more than one row -- but this "
          f"number is mostly noise:")
    print(f"  {n_specific:,} ({100 * n_specific / n_atoms:.1f}%) still match more than one row "
          f"after the catch-alls (CS/NS/OS/HS) are removed.")
    print(f"  {sum(pairs.values()):,} ({100 * sum(pairs.values()) / n_atoms:.1f}%) are genuine "
          f"collisions between two DIFFERENT specific classes.")
    print("\n  the real specific-vs-specific contests (winner first) -- these are the ones a")
    print("  reimplementation has to get right, and the ones the row order actually decides:")
    for (a, b), c in pairs.most_common(12):
        print(f"    {a:5s} beats {b:5s}  {c:6,d} atoms")


if __name__ == "__main__":
    main()
