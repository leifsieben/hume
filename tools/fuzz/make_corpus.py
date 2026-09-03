"""Build an adversarial SMILES corpus for the featurization fuzz run.

    .venv/bin/python tools/fuzz/make_corpus.py OUT.txt [N_REAL]

THREE SOURCES, AND THE MIX IS THE POINT. A million curated drug-like molecules will not find the
next crash: `data/corpus1m` has been featurized many times. What finds crashes is input nobody
curated -- the empty string that segfaulted 0.9.0 was not exotic chemistry, it was a blank cell.

  1. REAL. The curated corpus, as the baseline that must never fail.
  2. MUTATED. Real SMILES with characters deleted, swapped, duplicated or truncated. Most will
     not parse, which exercises the parse path; the ones that do parse are valid molecules no
     chemist would draw, which is exactly where the boundary code is least tested.
  3. CONSTRUCTED. Shapes chosen to hit known-fragile arithmetic: sizes at and around zero, chains
     long enough to pass RDKit's 1000-embedding truncation bound, dense fused systems that make
     the O(n^3) eigensolves work, hypervalent and exotic elements, radicals, isotopes, wildcards,
     atom maps, dative bonds, and disconnected salts.

Nothing here is filtered for validity. A SMILES that does not parse is a legitimate input to
featurize() and must produce a NaN row, not a signal.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rng = random.Random(0)

#: Sizes around the boundaries that have actually broken something: 0 atoms segfaulted 0.9.0,
#: ~600 carbons passes the fragment matcher's embedding bound.
CHAIN_LENGTHS = [0, 1, 2, 3, 8, 64, 200, 400, 500, 550, 600, 650, 800, 1200, 2000]

EXOTIC = ["[He]", "[Ne]", "[Ar]", "[Kr]", "[Xe]", "[Rn]", "[U]", "[Pu]", "[Tc]", "[At]", "[Fr]",
          "[Og]", "[Ts]", "[Lv]", "[*]", "*", "[#6]", "[13CH4]", "[2H]", "[3H]", "[235U]",
          "[C-4]", "[N+5]", "[Fe+6]", "[S@@]", "[Pt](Cl)(Cl)(N)N", "[SiH4]", "[PH5]", "[SF6]",
          "F[Xe]F", "O=[Xe](=O)(=O)=O", "[B-]1234", "[Se][Se]", "[Te]", "[As]", "[Sb]", "[Bi]"]

WEIRD = ["", " ", "\t", "\n", ".", "..", "()", "[]", "[[]]", "C.", ".C", "C..C", "()C", "C()",
         "1", "%99", "C%99", "C1", "C12", "[CH]", "[C]", "[c]", "[cH]", "c", "n", "o", "s",
         "C=C=C=C=C=C", "C#C#C", "N=N=N=N", "[N-]=[N+]=N", "c1ccc1", "c1cc1", "c1c1",
         "C/C=C/C", "C/C=C\\\\C", "[C@](F)(Cl)(Br)I", "[C@@H](N)(O)S",
         "CC(=O)[O-].[Na+].[K+].[Mg+2]", "[Na+].[Cl-]", "O.O.O.O.O.O.O.O.O.O",
         "[H+]", "[H-]", "[H][H]", "[HH]", "[e]", "[Xx]", "C[Xx]C",
         "[CH3:1][CH2:2][OH:3]", "[*:1]C[*:2]", "C[N+](C)(C)C",
         "[Cu+2].[O-]S(=O)(=O)[O-]", "N#[N+][O-]", "[O-][n+]1ccccc1"]


def constructed() -> list[str]:
    out = list(WEIRD) + list(EXOTIC)
    for n in CHAIN_LENGTHS:
        out.append("C" * n)                                     # unbranched
        out.append("O" + "C" * n + "O" if n else "OO")          # capped
        if n >= 3:
            out.append("C1" + "C" * (n - 1) + "1")              # one big ring
            out.append("C(" + ")(".join("C" * 3 for _ in range(min(n, 40))) + ")C")  # branchy
    # fused aromatics of growing size -- the O(n^3) eigensolves and the ring perception
    for k in range(2, 26):
        out.append("c1ccc2cc3" + "cc4" * 0 + "ccccc3cc2c1" if k == 2 else
                   "c1ccc2cc3ccc4" + "cc5" * 0 + "ccccc4cc3cc2c1")
    for k in range(3, 30):                                       # ladder polyacenes
        out.append("c1ccc2cc3cc4" + "cc%d" % (k % 9 + 1) * 0 + "ccccc4cc3cc2c1")
    # deep nesting and long ring-closure numbering
    for d in (5, 20, 60, 200):
        out.append("C" + "(C" * d + ")" * d)
    # every printable ascii char alone and doubled: cheap, and parse errors are the point
    for c in map(chr, range(33, 127)):
        out.append(c)
        out.append(c * 2)
    return [s for s in out if s is not None]


def mutate(s: str) -> str:
    if not s:
        return s
    k = rng.randrange(5)
    i = rng.randrange(len(s))
    if k == 0:
        return s[:i] + s[i + 1:]                       # delete
    if k == 1:
        return s[:i] + rng.choice("CNOScnos()[]=#12@+-.%*") + s[i:]   # insert
    if k == 2:
        return s[:i] + rng.choice("CNOScnos()[]=#12@+-.%*") + s[i + 1:]  # substitute
    if k == 3:
        return s[:i]                                   # truncate
    return s + s[i:]                                   # duplicate a tail


def main() -> None:
    out_path = Path(sys.argv[1])
    n_real = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    src = ROOT / "data" / "corpus1m" / "selected.txt"
    real = []
    if src.exists():
        with src.open() as fh:
            for line in fh:
                s = line.split()[0] if line.split() else ""
                real.append(s)
                if len(real) >= n_real:
                    break
    hard = ROOT / "cpp" / "hard.smi"
    if hard.exists():
        real += [l.split()[0] for l in hard.read_text().splitlines() if l.strip()]

    rows = list(real)
    rows += [mutate(rng.choice(real)) for _ in range(120_000)] if real else []
    # ONCE EACH, NOT REPEATED. The first version padded these to ~30,000 rows by repeating the
    # 388 shapes about 77 times. Coverage is a property of the SHAPE, so the repeats bought
    # nothing -- and they included 2000-carbon chains and dense fused systems whose O(n^3)
    # eigensolves cost seconds apiece, so a 25,000-molecule shard that should take 15 seconds
    # took over 15 minutes. Each shape appears once, plus a second copy of the small ones, which
    # are free and exercise the parse and boundary paths.
    con = constructed()
    rows += con
    rows += [c for c in con if len(c) <= 40]
    rng.shuffle(rows)
    with out_path.open("w") as fh:
        fh.write("\n".join(rows))
    print(f"  {len(rows):,} SMILES -> {out_path}")
    print(f"    real {len(real):,}  mutated 120,000  constructed {len(con):,} (once each)")


if __name__ == "__main__":
    main()
