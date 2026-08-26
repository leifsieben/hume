"""Dump real molecular graphs so the C++ benchmark measures chemistry, not synthetic graphs.

Text format, one molecule after another:

    n_mols
    n_atoms n_bonds
    delta_0 delta_v_0 ... (n_atoms pairs, the Chi weights)
    u v            (n_bonds lines)
    ...

Kept plain text on purpose: the parse cost is outside the timed region, and a format anyone
can eyeball is worth more here than a few milliseconds of load time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def main(n_want: int = 3000) -> None:
    d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
    smi = list(d["smiles"])
    rng = np.random.default_rng(0)
    picks = [smi[i] for i in rng.choice(len(smi), n_want * 2, replace=False)]

    out, n = [], 0
    for s in picks:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() < 3:
            continue
        na = m.GetNumAtoms()
        w = []
        for a in m.GetAtoms():
            delta = a.GetDegree()                       # sigma connectivity (heavy neighbours)
            zv = a.GetTotalValence() - a.GetTotalNumHs()  # valence connectivity, Kier-Hall
            w.append(f"{delta} {max(zv, 1)}")
        bonds = [f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()}" for b in m.GetBonds()]
        out.append(f"{na} {len(bonds)}\n" + " ".join(w) + "\n" + "\n".join(bonds))
        n += 1
        if n >= n_want:
            break

    p = ROOT / "cpp" / "graphs.txt"
    p.write_text(f"{n}\n" + "\n".join(out) + "\n")
    sizes = [int(o.split("\n")[0].split()[0]) for o in out]
    print(f"wrote {p} | {n} molecules | mean {np.mean(sizes):.1f} heavy atoms, max {max(sizes)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
