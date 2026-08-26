"""Is the C++ predict block EXACTLY the same number RDKit computes?

    ./predict verify && python verify_predict.py

A fast descriptor that disagrees with the reference implementation is not a fast descriptor,
it is a different quantity. This compares value by value over the same 10,000 molecules, on the
SMILES the exporter recorded, so an index shift cannot pass as agreement.

Tolerance is relative and tight (1e-9). It is not zero because C++ and NumPy accumulate the
EState perturbation sum in a different order and IEEE addition is not associative; anything
above that floor is a real disagreement, not rounding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, GraphDescriptors
from rdkit.Chem.EState import EStateIndices

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
COLS = ["MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
        "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha"]
TOL = 1e-9


def main() -> None:
    smis = HERE.joinpath("mols.smi").read_text().split()
    got = np.loadtxt(HERE / "values.txt")
    assert len(smis) == len(got), f"{len(smis)} smiles vs {len(got)} value rows"

    ref = np.empty_like(got)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        e = EStateIndices(m)
        ref[i] = [max(e), min(e), max(abs(np.asarray(e))), min(abs(np.asarray(e))),
                  GraphDescriptors.Kappa1(m), GraphDescriptors.Kappa2(m),
                  GraphDescriptors.Kappa3(m), Descriptors.HallKierAlpha(m)]

    print(f"{len(smis):,} molecules, {len(COLS)} descriptors\n")
    print(f"  {'descriptor':22s} {'exact':>9s} {'max rel dev':>13s}  verdict")
    ok = True
    for j, c in enumerate(COLS):
        a, b = got[:, j], ref[:, j]
        scale = np.maximum(np.abs(b), 1e-12)
        rel = np.abs(a - b) / scale
        frac = float((rel <= TOL).mean()) * 100
        worst = float(rel.max())
        good = worst <= TOL
        ok &= good
        print(f"  {c:22s} {frac:8.3f}% {worst:13.3e}  {'MATCH' if good else 'MISMATCH'}")
        if not good:
            k = int(np.argmax(rel))
            print(f"      worst: {smis[k]}")
            print(f"      c++ {a[k]!r}  rdkit {b[k]!r}")
    print("\n" + ("ALL EXACT" if ok else "DISAGREEMENT -- the C++ value is a different quantity"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
