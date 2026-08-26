"""Dump molecules for the C++ PREDICT-family benchmark and exactness check.

Richer than export_graphs.py, which carries only the two Chi weights. The predict families need
element, hydrogen count, charge, hybridisation, aromaticity and bond order, so the format is:

    n_mols
    n_atoms n_bonds
    Z degree nH charge hyb aromatic inring      (n_atoms lines)
    u v order                                   (n_bonds lines)
    ...

`hyb` is RDKit's HybridizationType as an int, because HallKierAlpha indexes its per-element
table by (hybridisation - 2). Re-deriving hybridisation in C++ from the graph would be a second
implementation of RDKit's perception rules and the first place an "exact" claim would quietly
stop being true -- the point of this file is to hand C++ exactly what RDKit saw.

Writes the SMILES alongside, so the verifier can rebuild the same molecules in RDKit and compare
value for value rather than trusting index alignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def main(n_want: int = 10_000) -> None:
    d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
    smi = list(d["smiles"])
    rng = np.random.default_rng(1)
    picks = [smi[i] for i in rng.choice(len(smi), min(len(smi), n_want * 3), replace=False)]

    out, kept = [], []
    for s in picks:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() < 3:
            continue
        rows = []
        for a in m.GetAtoms():
            rows.append(f"{a.GetAtomicNum()} {a.GetDegree()} {a.GetTotalNumHs()} "
                        f"{a.GetFormalCharge()} {int(a.GetHybridization())} "
                        f"{int(a.GetIsAromatic())} {int(a.IsInRing())}")
        bonds = []
        for b in m.GetBonds():
            bonds.append(f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} "
                         f"{int(b.GetBondType())}")
        out.append(f"{m.GetNumAtoms()} {len(bonds)}\n" + "\n".join(rows) +
                   ("\n" + "\n".join(bonds) if bonds else ""))
        kept.append(Chem.MolToSmiles(m))
        if len(kept) >= n_want:
            break

    p = ROOT / "cpp" / "mols.txt"
    p.write_text(f"{len(kept)}\n" + "\n".join(out) + "\n")
    (ROOT / "cpp" / "mols.smi").write_text("\n".join(kept) + "\n")
    sizes = [int(o.split("\n")[0].split()[0]) for o in out]
    print(f"wrote {p} and mols.smi | {len(kept)} molecules | "
          f"mean {np.mean(sizes):.1f} heavy atoms, max {max(sizes)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
