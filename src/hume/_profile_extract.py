"""What are the 187.5 us/molecule of RDKit->array extraction actually made of?

The bridge measured extraction at 187.5 us/mol against 249.8 us of C++ compute -- 43% of
`featurize_blocks`. That ratio decides an architecture question, so it is worth knowing what the
187.5 is, not merely how big it is:

  * If it is PYTHON CALL OVERHEAD -- `atom.GetAtomicNum()` and friends crossing the boundary
    ~10 times per atom, ~300 times per molecule -- it is cheap to fix by getting the same data
    out in bulk, and the "bit-identical to RDKit" claim is untouched.
  * If it is GENUINE PERCEPTION WORK inside RDKit (Gasteiger's iterative charges, Crippen's
    SMARTS typer, CIP labelling), it is irreducible from Python, and the only lever left is
    linking RDKit's C++ directly -- which costs wheels.

Those two have opposite consequences, so the split is the measurement.

METHOD. Each component is timed in its own pass over the same molecules, in-process, with the
order rotated across cycles so machine drift is common-mode. CPU time, not wall: the bridge work
found wall-clock spreads of 26-47x on this shared box, larger than anything being compared.

    .venv/bin/python src/hume/_profile_extract.py [n_mols]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdPartialCharges

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]


def _cpu():
    return time.process_time()


def bench(fn, mols, reps=3):
    """-> us per molecule, CPU time, best of `reps`."""
    best = float("inf")
    for _ in range(reps):
        t0 = _cpu()
        fn(mols)
        best = min(best, (_cpu() - t0) / len(mols) * 1e6)
    return best


# ---- the components, each doing exactly what _extract.py does and nothing else ----------
def c_gasteiger(mols):
    for m in mols:
        try:
            # same reasoning as c_crippen: charges are cached on the molecule
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12, throwOnParamFailure=False)
            [a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()]
        except Exception:
            pass


def c_crippen(mols):
    # force=True IS LOAD-BEARING. RDKit caches Crippen contributions on the molecule, so timing
    # a second pass over the same molecules measures a dict lookup rather than the 68-pattern
    # SMARTS typer. This project has already reported an 82x-wrong Crippen number that way, and
    # the first version of THIS profiler made the same mistake and inverted its own verdict.
    # In real extraction every molecule is fresh, so cold is the honest cost.
    for m in mols:
        rdMolDescriptors._CalcCrippenContribs(m, force=True)


def c_atom_loop(mols):
    """The per-atom Python attribute access -- 8 ints + mass per atom."""
    for m in mols:
        for a in m.GetAtoms():
            (a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs(), a.GetFormalCharge(),
             int(a.GetHybridization()), int(a.GetIsAromatic()), int(a.IsInRing()),
             a.GetMass())


def c_cip(mols):
    for m in mols:
        for a in m.GetAtoms():
            if a.HasProp("_CIPCode"):
                a.GetProp("_CIPCode")


def c_bond_loop(mols):
    for m in mols:
        for b in m.GetBonds():
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetIsConjugated()),
             int(b.IsInRing()), int(b.GetStereo()), b.GetBondTypeAsDouble())


def c_array_build(mols):
    """The numpy assembly at the end, on lists of the right size."""
    for m in mols:
        n = m.GetNumAtoms()
        np.asarray([0] * (n * 8), dtype=np.int32).reshape(n, 8)
        np.asarray([0.0] * (n * 4), dtype=np.float64).reshape(n, 4)


def main() -> None:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    smis = ROOT.joinpath("cpp", "mols.smi").read_text().split()
    rng = np.random.default_rng(0)
    pick = [smis[i] for i in rng.choice(len(smis), min(len(smis), n_want), replace=False)]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in pick) if m is not None]
    natoms = float(np.mean([m.GetNumAtoms() for m in mols]))
    print(f"{len(mols)} molecules, mean {natoms:.1f} heavy atoms\n")

    comps = [("Gasteiger (C++ iterative, + prop read)", c_gasteiger),
             ("Crippen contribs (C++ SMARTS typer)", c_crippen),
             ("per-ATOM python attribute loop", c_atom_loop),
             ("CIP code read", c_cip),
             ("per-BOND python attribute loop", c_bond_loop),
             ("numpy array assembly", c_array_build)]

    # rotate the order across cycles so drift is common-mode
    tot = {}
    for cyc in range(3):
        order = comps[cyc:] + comps[:cyc]
        for name, fn in order:
            tot.setdefault(name, []).append(bench(fn, mols))

    print(f"  {'component':42s} {'us/mol':>9s} {'us/atom':>9s}  kind")
    rows = []
    for name, fn in comps:
        us = min(tot[name])
        kind = "RDKit C++" if ("Gasteiger" in name or "Crippen" in name) else "PYTHON"
        rows.append((name, us, kind))
        print(f"  {name:42s} {us:9.2f} {us/natoms:9.3f}  {kind}")

    py = sum(u for _, u, k in rows if k == "PYTHON")
    cpp = sum(u for _, u, k in rows if k == "RDKit C++")
    print(f"\n  python-side total : {py:8.2f} us/mol   ({100*py/(py+cpp):.0f}%)")
    print(f"  RDKit C++ total   : {cpp:8.2f} us/mol   ({100*cpp/(py+cpp):.0f}%)")
    print(f"  measured extract  :   187.50 us/mol   (bridge report, for reference)")
    print()
    if py > cpp:
        print("  VERDICT: dominated by PYTHON BOUNDARY CROSSINGS, not by RDKit's own work.")
        print("  Recoverable in bulk without touching the exactness claim.")
    else:
        print("  VERDICT: dominated by RDKit's OWN COMPUTATION (Gasteiger/Crippen).")
        print("  Irreducible from Python; only linking RDKit's C++ would remove it.")


if __name__ == "__main__":
    main()
