"""Export the H-ADDED graph for the autocorrelation block: raw atoms and bonds, no descriptors.

A separate file from export_predict.py because autocorrelation works on a different graph.
`explicit_hydrogens = True` in Mordred, so aspirin is 21 atoms here and 13 there; sharing one
export would mean carrying both graphs for every molecule to serve one block.

THIS FILE USED TO CARRY THE NINE FINISHED WEIGHT VECTORS, and that was the single most expensive
thing in HUME. The argument for it was that six of the nine are per-element table lookups and two
are two-line functions of the bonding, so re-deriving them "would add nine new ways to be subtly
wrong in exchange for nothing measurable". The measurement came in at 473.9 us/mol -- larger than
every other block in the pipeline combined, and twenty times the 23.07 us/mol the C++ spends on
the O(n^2) accumulation it exists to do. The cost was nine Python calls per atom into
`ap.getters`, not the arithmetic.

So the weights moved to C++ (cpp/ac_weights.h) and the element tables are GENERATED from Mordred
rather than retyped (cpp/gen_ac_tables.py -> cpp/ac_tables.h), which answers the "nine new ways
to be subtly wrong" objection with provenance instead of with a Python loop.

WHAT STILL COMES FROM RDKIT: the Gasteiger charge, because ComputeGasteigerCharges is already C++
and there is no table for it. Everything else here is a field read.

    n_mols
    n_atoms n_bonds
    Z formal_charge n_implicit_H gasteiger_c      (n_atoms lines)
    u v                                           (n_bonds lines)

`n_implicit_H` is GetTotalNumHs(), the Hs *still implicit after AddHs* -- normally 0, but `dv`
adds it to the explicit-H neighbour count and Mordred reads it, so it is carried rather than
assumed. Isotope is NOT carried: all nine getters read GetAtomicNum(), so [2H] weighs what [H]
weighs. (Mordred's mass weight `m` would need it; `m` is not one of these nine.)

MOLECULES ARE NO LONGER DROPPED FOR A NON-FINITE WEIGHT. The old code dropped any molecule whose
nine vectors were not all finite, which quietly removed 5,519 of the 100k adversarial corpus --
precisely the rare-element molecules the corpus exists to test. Mordred fails one AtomicProperty
at a time, so a selenium molecule loses its 54 `se` columns and keeps the other 432; ac.cpp now
reproduces that per weight. The only guard left is on the one float this file still writes: a
non-finite Gasteiger charge is written as the SENTINEL -1e30, not as the token `nan`, because
libc++'s `istream >> double` does not parse "nan" -- it sets failbit and leaves the value zero,
which is the nan/inf export desync all over again with a silent 0.0 instead of a shifted file.
-1e30 is a number, so it costs exactly one token, and ac.cpp maps it back to NaN. The reader also
hard-fails on a short read now, so a desync stops the run instead of skewing every row after it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def gasteiger(atom) -> float:
    """Mordred's `c` getter, verbatim -- note the sum is INSIDE the conditional.

        (_GasteigerCharge + _GasteigerHCharge) if HasProp("_GasteigerHCharge") else 0.0

    An atom with no _GasteigerHCharge property contributes 0.0, not its own charge. RDKit sets
    both on every atom in practice, but the fallback is reproduced rather than assumed away.
    """
    if not atom.HasProp("_GasteigerHCharge"):
        return 0.0
    return atom.GetDoubleProp("_GasteigerCharge") + atom.GetDoubleProp("_GasteigerHCharge")


def main(n_want: int = 2000, src: str | None = None) -> None:
    if src:
        picks = Path(src).read_text().split()
    else:
        d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
        smi = list(d["smiles"])
        rng = np.random.default_rng(7)
        picks = [smi[i] for i in rng.choice(len(smi), min(len(smi), n_want * 3), replace=False)]

    C_MISSING = -1e30           # must match AC_C_MISSING in cpp/ac_weights.h
    out, kept, n_nan_c = [], [], 0
    for s in picks:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() < 3:
            continue
        mh = Chem.AddHs(m)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mh)
        except Exception:
            continue                    # RDKit refused outright; there is no charge to carry
        rows = []
        for a in mh.GetAtoms():
            c = gasteiger(a)
            if not np.isfinite(c):
                n_nan_c += 1
                c = C_MISSING           # one token either way, so the reader stays in step
            rows.append(f"{a.GetAtomicNum()} {a.GetFormalCharge()} {a.GetTotalNumHs()} {c:.12g}")
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
          f"mean {np.mean(sizes):.1f} atoms WITH H, max {max(sizes)} | "
          f"{n_nan_c} atoms with a non-finite Gasteiger charge")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000,
         sys.argv[2] if len(sys.argv) > 2 else None)
