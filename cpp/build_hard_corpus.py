"""Build a deliberately ADVERSARIAL 100k corpus for the full exactness run.

Not a representative sample. A representative sample of drug-like space is what the 300- and
400-molecule development sets already were, and they are exactly the molecules the code is now
known to handle. The point of this corpus is to concentrate everything that has ALREADY broken
an implementation here, plus everything structurally likely to.

Every bug this project found in the C++ came from a chemical edge case, not from ordinary
molecules:

    iodine collided with "other"          -> rare elements
    BCUT2D raises without Gasteiger params -> Se, and anything exotic
    chi/kappa/BCUT2D walked isotopic H     -> [2H], [3H]
    Kappa's A counts heavy but P1 counts all -> explicit-H molecules
    chi's lollipop rule                    -> small rings carrying substituents
    conjugation's tie-break                -> molecules with several equal-sized pi systems
    resistance's per-component solve       -> salts and other disconnected inputs
    the nan/inf export desync              -> molecules RDKit cannot charge

So the strata below oversample each of those to saturation, and only then fill the remainder
with ordinary chemistry so the run also measures the common case.

    python cpp/build_hard_corpus.py [n]   ->  cpp/hard.smi
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/pubchem_10M.smi")
COMMON = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def classify(m, smi):
    """-> set of stress tags this molecule carries."""
    t = set()
    zs = {a.GetAtomicNum() for a in m.GetAtoms()}
    if zs - COMMON:
        t.add("rare_element")
    if any(a.GetIsotope() for a in m.GetAtoms()):
        t.add("isotope")
    if any(a.GetAtomicNum() == 1 for a in m.GetAtoms()):
        t.add("explicit_h")
    if "." in smi:
        t.add("disconnected")
    if any(a.GetFormalCharge() for a in m.GetAtoms()):
        t.add("charged")
    if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        t.add("radical")
    ri = m.GetRingInfo()
    nr = ri.NumRings()
    if nr >= 6:
        t.add("polycyclic")
    if any(len(r) <= 4 for r in ri.AtomRings()):
        t.add("small_ring")
    if any(len(r) >= 12 for r in ri.AtomRings()):
        t.add("macrocycle")
    # a fused atom shared by 3+ rings is where the chi path enumeration got it wrong
    if any(ri.NumAtomRings(i) >= 3 for i in range(m.GetNumAtoms())):
        t.add("fused_junction")
    n = m.GetNumAtoms()
    if n <= 6:
        t.add("tiny")
    if n >= 70:
        t.add("large")
    if sum(1 for a in m.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED) >= 4:
        t.add("stereo_rich")
    return t


# Target counts per stratum. Rare things get everything found; common things get a quota.
QUOTA = {
    "rare_element": 12_000, "isotope": 6_000, "explicit_h": 6_000, "disconnected": 10_000,
    "charged": 10_000, "radical": 3_000, "polycyclic": 8_000, "small_ring": 8_000,
    "macrocycle": 5_000, "fused_junction": 8_000, "tiny": 5_000, "large": 6_000,
    "stereo_rich": 6_000,
}


def main(n_want: int = 100_000) -> None:
    have = Counter()
    chosen, seen = [], set()

    def offer(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() < 2 or m.GetNumAtoms() > 300:
            return False
        c = Chem.MolToSmiles(m)
        if c in seen:
            return False
        tags = classify(m, c)
        # take it if it fills any stratum still short, or as ordinary filler at the end
        wanted = [t for t in tags if have[t] < QUOTA.get(t, 0)]
        if not wanted and len(chosen) >= n_want * 0.72:
            return False
        seen.add(c)
        chosen.append(c)
        for t in tags:
            have[t] += 1
        return True

    # the benchmark first -- it is what every downstream number is computed on
    bench = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)["smiles"]
    for s in bench:
        if len(chosen) >= n_want:
            break
        offer(str(s))
    print(f"  benchmark contributed {len(chosen):,}")

    if POOL.exists():
        with open(POOL) as fh:
            for line in fh:
                if len(chosen) >= n_want:
                    break
                offer(line.split()[0] if line.strip() else "")
    print(f"  after PubChem sweep    {len(chosen):,}")

    out = ROOT / "cpp" / "hard.smi"
    out.write_text("\n".join(chosen[:n_want]) + "\n")
    print(f"\nwrote {out} | {min(len(chosen), n_want):,} molecules\n")
    print(f"  {'stratum':16s} {'count':>8s}")
    for k in sorted(QUOTA):
        print(f"  {k:16s} {have[k]:8,d}")
    sizes = [Chem.MolFromSmiles(s).GetNumAtoms() for s in chosen[:2000]]
    print(f"\n  heavy atoms (first 2k): min {min(sizes)} median {int(np.median(sizes))} "
          f"max {max(sizes)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
