"""C++ cost of the PREDICT families that RDKit already implements in C++.

    python cpp/time_rdkit_cpp.py

WHY THIS IS A LEGITIMATE C++ MEASUREMENT and not the Python timing this project has banned:
Crippen contributions, Gasteiger charges, Labute ASA, TPSA, BCUT2D and Ipc are all C++ routines
inside RDKit. Calling one of them once per molecule costs exactly one Python->C++ boundary
crossing plus the C++ work. Mordred is the opposite -- it evaluates the descriptor itself in
Python, so its timing is an interpreter measurement and tells you nothing about the descriptor.

The per-call boundary cost is measured separately against a trivial C++ accessor and SUBTRACTED,
so what is reported is the C++ work. It is an upper bound on what a native implementation would
cost, because a native caller pays no boundary at all.

The VSA families are not timed separately ON PURPOSE. EState_VSA, VSA_EState, SlogP_VSA, SMR_VSA
and PEOE_VSA -- 70 of the 166 predict columns -- are binned sums over per-atom quantities these
routines already produce. Their marginal cost is a bincount over n atoms, which is noise next to
the primitive that feeds them. Timing them separately would double-count the primitive.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rd
from rdkit.Chem import rdPartialCharges

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
REPS = 5


def bench(mols, fn, label, baseline=0.0):
    """-> (us/mol net of the boundary, label, n_failed).

    Failures are COUNTED, not swallowed. BCUT2D raises on any element RDKit has no Gasteiger
    parameter for -- selenium, in this corpus -- and those molecules are a real source of NaN in
    the descriptor matrix, not a benchmarking nuisance. A family that cannot be computed for
    part of chemical space is a family the proxy has to cover anyway.
    """
    bad = 0
    try:
        fn(mols[0])
    except Exception:
        pass
    ts = []
    for r in range(REPS):
        t0 = time.perf_counter()
        for m in mols:
            try:
                fn(m)
            except Exception:
                if r == 0:
                    bad += 1
        ts.append(time.perf_counter() - t0)
    us = min(ts) / len(mols) * 1e6
    return max(us - baseline, 0.0), label, bad


def main() -> None:
    smis = HERE.joinpath("mols.smi").read_text().split()
    mols = [Chem.MolFromSmiles(s) for s in smis]
    mols = [m for m in mols if m is not None]
    na = np.mean([m.GetNumAtoms() for m in mols])
    print(f"{len(mols):,} molecules, mean {na:.1f} heavy atoms, {REPS} reps, best-of\n")

    # The Python->C++ boundary, measured against the cheapest possible C++ call.
    over, _, _ = bench(mols, lambda m: m.GetNumAtoms(), "overhead")
    print(f"  per-call Python->C++ boundary: {over:.3f} us  (subtracted below)\n")

    rows = []
    rows.append(bench(mols, rd._CalcCrippenContribs, "Crippen logP/MR contribs", over))
    rows.append(bench(mols, lambda m: rdPartialCharges.ComputeGasteigerCharges(m, nIter=12),
                      "Gasteiger charges (PEOE, 12 iters)", over))
    rows.append(bench(mols, rd._CalcLabuteASAContribs, "Labute ASA contribs", over))
    rows.append(bench(mols, rd.CalcTPSA, "TPSA", over))
    rows.append(bench(mols, rd.BCUT2D, "BCUT2D (8 eigenvalues)", over))
    rows.append(bench(mols, rd.CalcNumRotatableBonds, "rotatable bonds (reference point)", over))

    print(f"  {'family':40s} {'us/mol':>9s} {'failed':>8s}")
    for us, label, bad in rows:
        print(f"  {label:40s} {us:9.2f} {bad:8d}")
    tot = sum(u for u, _, _ in rows[:5])
    print(f"  {'-' * 50}")
    print(f"  {'subtotal (these five)':40s} {tot:9.2f}")
    print(f"\n  plus, implemented natively in cpp/predict.cpp:")
    print(f"  {'EState indices':40s} {7.69:9.2f}")
    print(f"  {'Kappa1-3 + HallKierAlpha':40s} {2.88:9.2f}")
    print(f"  {'-' * 50}")
    print(f"  {'PREDICT BLOCK, C++, approx':40s} {tot + 7.69 + 2.88:9.2f}")


if __name__ == "__main__":
    main()
