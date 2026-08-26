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
from rdkit.Chem import rdMolDescriptors, rdPartialCharges

RDLogger.DisableLog("rdApp.*")
# E/Z as +/-1, matching stereo.py's _E exactly (TRANS is E, CIS is Z).
_EZ = {Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
       Chem.BondStereo.STEREOZ: -1, Chem.BondStereo.STEREOCIS: -1}
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
        # The four BCUT2D atom properties come from RDKit rather than being reimplemented.
        # That is a deliberate split of the work: Crippen (0.85 us) and Gasteiger (9.41 us) are
        # already cheap C++, while the EIGENVALUE step is the 300 us. Porting the cheap half
        # would buy nothing and would put two SMARTS/PEOE implementations in the world.
        # Molecules RDKit cannot charge (no Gasteiger parameters -- selenium here) are marked
        # with a flag rather than dropped, so the verifier compares like with like.
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
            chg = [float(a.GetDoubleProp("_GasteigerCharge")) for a in m.GetAtoms()]
            ok = all(np.isfinite(c) for c in chg)
            # A non-finite charge must NOT reach the file: C++ istream fails on
            # "nan"/"inf" and every subsequent field desyncs, which showed up as the
            # loader reporting a 19.9-atom mean for a 30.6-atom corpus. The chg_ok
            # flag records that the molecule was uncharged rather than hiding it.
            chg = [c if np.isfinite(c) else 0.0 for c in chg]
        except Exception:
            chg = [0.0] * m.GetNumAtoms()
            ok = False
        crip = rdMolDescriptors._CalcCrippenContribs(m)
        # CIP codes for the stereo block. MolFromSmiles assigns them already; the explicit call
        # is the safety net stereo.py also carries, and is verified to change nothing here.
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        cip = []
        for a in m.GetAtoms():
            if a.HasProp("_CIPCode"):
                cip.append(1 if a.GetProp("_CIPCode") == "R" else -1)
            else:
                cip.append(0)
        rows = []
        for i, a in enumerate(m.GetAtoms()):
            rows.append(f"{a.GetAtomicNum()} {a.GetDegree()} {a.GetTotalNumHs()} "
                        f"{a.GetFormalCharge()} {int(a.GetHybridization())} "
                        f"{int(a.GetIsAromatic())} {int(a.IsInRing())} "
                        f"{a.GetMass():.10g} {chg[i]:.10g} "
                        f"{crip[i][0]:.10g} {crip[i][1]:.10g} {cip[i]}")
        bonds = []
        for b in m.GetBonds():
            st = _EZ.get(b.GetStereo(), 0)
            bonds.append(f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} "
                         f"{b.GetBondTypeAsDouble():.10g} "
                         f"{int(b.GetIsConjugated())} {int(b.IsInRing())} {st}")
        out.append(f"{m.GetNumAtoms()} {len(bonds)} {int(ok)}\n" + "\n".join(rows) +
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
