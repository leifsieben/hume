"""Export the H-ADDED graph plus the nine autocorrelation weight vectors.

A separate file from export_predict.py because autocorrelation works on a different graph.
`explicit_hydrogens = True` in Mordred, so aspirin is 21 atoms here and 13 there; sharing one
export would mean carrying both graphs for every molecule to serve one block.

THE NINE WEIGHT VECTORS COME FROM MORDRED, and that is deliberate. Six of them (i, p, v, se,
pe, are) are per-element table lookups, `d` and `dv` are two-line functions of the atom's
bonding, and `c` is the Gasteiger charge -- all O(n) and all cheap. The expensive part of the
block is the O(n^2) accumulation over 9 lags, which is what the C++ replaces. Re-deriving nine
element tables would add nine new ways to be subtly wrong in exchange for nothing measurable.

    n_mols
    n_atoms n_bonds
    w_c w_d w_dv w_i w_p w_v w_se w_pe w_are      (n_atoms lines)
    u v                                            (n_bonds lines)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges

from mordred import _atomic_property as ap

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ["c", "d", "dv", "i", "p", "v", "se", "pe", "are"]


def main(n_want: int = 2000) -> None:
    d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
    smi = list(d["smiles"])
    rng = np.random.default_rng(7)
    picks = [smi[i] for i in rng.choice(len(smi), min(len(smi), n_want * 3), replace=False)]

    out, kept = [], []
    for s in picks:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() < 3:
            continue
        mh = Chem.AddHs(m)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mh)
        except Exception:
            continue
        cols = []
        for w in WEIGHTS:
            g = ap.getters[w]
            try:
                v = [float(g(a)) for a in mh.GetAtoms()]
            except Exception:
                v = None
            if v is None or not all(np.isfinite(x) for x in v):
                cols = None
                break
            cols.append(v)
        if cols is None:
            continue                    # a non-finite weight would desync the C++ reader
        rows = [" ".join(f"{cols[w][i]:.12g}" for w in range(len(WEIGHTS)))
                for i in range(mh.GetNumAtoms())]
        bonds = [f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()}" for b in mh.GetBonds()]
        out.append(f"{mh.GetNumAtoms()} {len(bonds)}\n" + "\n".join(rows) +
                   ("\n" + "\n".join(bonds) if bonds else ""))
        kept.append(Chem.MolToSmiles(m))
        if len(kept) >= n_want:
            break

    p = ROOT / "cpp" / "mols_h.txt"
    p.write_text(f"{len(kept)}\n" + "\n".join(out) + "\n")
    (ROOT / "cpp" / "mols_h.smi").write_text("\n".join(kept) + "\n")
    sizes = [int(o.split("\n")[0].split()[0]) for o in out]
    print(f"wrote {p} and mols_h.smi | {len(kept)} molecules | "
          f"mean {np.mean(sizes):.1f} atoms WITH H, max {max(sizes)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
