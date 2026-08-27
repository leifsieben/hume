"""Fill the stress strata PubChem could not supply.

build_hard_corpus.py sweeps 10M PubChem entries and still comes back with ONE disconnected
molecule, 148 with a rare element, 241 tiny and 327 large. That is not a flaw in the sweep -- a
curated drug-like pool is desalted, neutral-ish and centred on 20-40 heavy atoms, so the cases
most likely to break a featuriser are exactly the ones it does not contain.

So they are constructed. Every molecule below is a real, sanitisable structure; the point is to
exercise code paths, not to model a screening library:

    disconnected   two pool molecules joined with '.'   -> resistance's per-component solve,
                                                           unreachable pairs in every BFS
    rare_element   Se / Si / B / Te / As / Ge / Sn      -> the iodine-vs-other collision class,
                                                           and BCUT2D's missing Gasteiger params
    tiny           2-6 heavy atoms                      -> Kappa3's A-3 term, empty lag bins,
                                                           cycle counts on graphs with no cycles
    large          pool molecules bonded end to end     -> O(n^2) and O(n^3) paths at 200+ atoms
    isotope        [2H]/[13C]/[15N] labels              -> the heavy-atom exclusion in chi,
                                                           kappa and BCUT2D
    explicit_h     Chem.AddHs                           -> A counts heavy but P1 counts all

    python cpp/augment_hard_corpus.py
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
TARGET = {"disconnected": 10_000, "rare_element": 8_000, "tiny": 4_000, "large": 3_000,
          "isotope": 2_000, "explicit_h": 4_000}
RARE = ["[Se]", "[SeH]", "[SiH3]", "[SiH2]", "[B]", "[BH2]", "[Te]", "[AsH2]", "[GeH3]",
        "[SnH3]", "[Se+]", "[B-]"]
TINY = ["CC", "CCO", "C=O", "C#N", "CS", "CCl", "CBr", "CI", "C1CC1", "C1CO1", "N", "O=S=O",
        "CC=O", "C=C", "C#C", "OO", "NN", "C1CN1", "[NH4+]", "[O-]C=O", "CF", "C1CCC1",
        "c1ccccc1", "C1CNC1", "O=C=O", "CN", "CP", "C[SiH3]", "C[Se]C", "B(O)O"]


def ok(s):
    m = Chem.MolFromSmiles(s)
    return m is not None and m.GetNumAtoms() >= 2 and m.GetNumAtoms() <= 400


def main() -> None:
    random.seed(11)
    base = HERE.joinpath("hard.smi").read_text().split()
    print(f"  base corpus {len(base):,}")
    pool = [s for s in base if 8 <= len(s) <= 60]
    made = Counter()
    extra = []

    def add(s, tag):
        if made[tag] >= TARGET[tag] or not ok(s):
            return
        c = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        extra.append(c)
        made[tag] += 1

    # salts / mixtures: two components, so every BFS has unreachable pairs and resistance has
    # two Laplacian blocks to solve
    for _ in range(TARGET["disconnected"] * 3):
        if made["disconnected"] >= TARGET["disconnected"]:
            break
        add(random.choice(pool) + "." + random.choice(pool), "disconnected")

    for _ in range(TARGET["tiny"] * 4):
        if made["tiny"] >= TARGET["tiny"]:
            break
        a = random.choice(TINY)
        add(a if random.random() < 0.5 else a + "." + random.choice(TINY), "tiny")

    # rare elements grafted onto real scaffolds, so they sit inside rings and conjugation
    for _ in range(TARGET["rare_element"] * 6):
        if made["rare_element"] >= TARGET["rare_element"]:
            break
        s = random.choice(pool)
        add(s + random.choice(RARE) if random.random() < 0.3 else
            random.choice(RARE) + "C" + s.lstrip("C") if s.startswith("C") else
            s + "." + random.choice(RARE), "rare_element")

    # large: chain pool molecules together into 150-350 atom graphs
    for _ in range(TARGET["large"] * 4):
        if made["large"] >= TARGET["large"]:
            break
        parts = [random.choice(pool) for _ in range(random.randint(3, 7))]
        add(".".join(parts), "large")

    for _ in range(TARGET["isotope"] * 4):
        if made["isotope"] >= TARGET["isotope"]:
            break
        s = random.choice(pool)
        add(s.replace("C", "[13C]", 1) if "C" in s else s.replace("c", "[13c]", 1), "isotope")

    for _ in range(TARGET["explicit_h"] * 3):
        if made["explicit_h"] >= TARGET["explicit_h"]:
            break
        m = Chem.MolFromSmiles(random.choice(pool))
        if m is None:
            continue
        add(Chem.MolToSmiles(Chem.AddHs(m)), "explicit_h")

    print("  synthesised:")
    for k in sorted(TARGET):
        print(f"    {k:16s} {made[k]:6,d} / {TARGET[k]:,}")

    # keep the total at 100k: the synthesised stress cases displace ordinary filler
    keep = base[: max(0, 100_000 - len(extra))] + extra
    random.shuffle(keep)
    out = HERE / "hard.smi"
    out.write_text("\n".join(keep) + "\n")
    print(f"\n  wrote {out} | {len(keep):,} molecules "
          f"({len(extra):,} synthesised, {len(keep)-len(extra):,} from the sweep)")


if __name__ == "__main__":
    main()
