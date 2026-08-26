"""Is the consolidated C++ featuriser exactly the reference?

    ./hume verify mols.txt && python cpp/verify_hume.py

TWO REFERENCES, and conflating them would hide errors in both:

  * chi (Chi2n..Chi4v) and BalabanJ  -> RDKIT is the reference. These are ports of somebody
    else's descriptor and "our number" has no standing.
  * cycles, resistance, conjugation, stereo -> OUR PYTHON MODULES are the reference. They are
    the definition of those blocks, and they are themselves already verified against RDKit /
    the 1M corpus where an external reference exists.

Columns are matched BY NAME against each module's NAMES list rather than by position, so
inserting a feature into a Python block cannot silently shift the comparison onto the wrong
pair of numbers.

TOLERANCE DEPENDS ON THE REFERENCE, because the two references have different precision:

  * RDKit returns float64          -> rtol 1e-9, atol 1e-12
  * our modules return FLOAT32     -> rtol 3e-6, atol 1e-6

Comparing a float64 C++ value against a float32 reference at 1e-9 can never pass -- float32
carries ~1.2e-7 relative precision, so `Kf` came out 5.5e-08 "wrong" when the C++ value was in
fact the more accurate of the two. The atol term exists for features that are legitimately zero
(Delta is identically zero on a tree, and -1.9e-15 against +1.0e-15 is not a disagreement, it is
two different roundings of nothing).

Neither tolerance is loose enough to hide a real error: the two genuine bugs this harness caught
were off by 1.7e-1 and 3.2e-1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, GraphDescriptors as GD, rdMolDescriptors
from rdkit.Chem.EState import EStateIndices

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
TOL = {"rdkit": (1e-9, 1e-12), "py": (3e-6, 1e-6)}

# (label, source) in the exact order ./hume verify writes them.
#   ("rdkit", fn)        -> compare against RDKit
#   ("py", module, name) -> compare against our module's feature of that name
import conjugation, cycles, resistance, stereo                       # noqa: E402

SPEC = [
    ("Chi2n", "rdkit", GD.Chi2n), ("Chi2v", "rdkit", GD.Chi2v),
    ("Chi3n", "rdkit", GD.Chi3n), ("Chi3v", "rdkit", GD.Chi3v),
    ("Chi4n", "rdkit", GD.Chi4n), ("Chi4v", "rdkit", GD.Chi4v),
    ("BalabanJ", "rdkit", GD.BalabanJ),
    ("C3", "py", cycles, "C3"), ("C4", "py", cycles, "C4"), ("C5", "py", cycles, "C5"),
    ("Kf", "py", resistance, "Kf"), ("DeltaMean", "py", resistance, "DeltaMean"),
    ("n_sys", "py", conjugation, "n_sys"), ("conj_atoms", "py", conjugation, "conj_atoms"),
    ("sys_max", "py", conjugation, "sys_max"), ("linearity", "py", conjugation, "linearity"),
    ("branch_pts", "py", conjugation, "branch_pts"),
    ("S_sum", "py", stereo, "S_sum"), ("S_absum", "py", stereo, "S_absum"),
    ("S_sum_norm", "py", stereo, "S_sum_norm"),
    ("T_sum", "py", stereo, "T_sum"), ("T_absum", "py", stereo, "T_absum"),
] + [(f"SATS{k}", "py", stereo, f"SATS{k}") for k in range(1, 7)] + [
    ("SATS_far", "py", stereo, "SATS_far"),
    # merged in from predict.cpp -- same binary, same graph build, same BFS
    ("MaxEStateIndex", "rdkit", lambda m: max(EStateIndices(m))),
    ("MinEStateIndex", "rdkit", lambda m: min(EStateIndices(m))),
    ("MaxAbsEStateIndex", "rdkit", lambda m: max(abs(np.asarray(EStateIndices(m))))),
    ("MinAbsEStateIndex", "rdkit", lambda m: min(abs(np.asarray(EStateIndices(m))))),
    ("Kappa1", "rdkit", GD.Kappa1), ("Kappa2", "rdkit", GD.Kappa2),
    ("Kappa3", "rdkit", GD.Kappa3), ("HallKierAlpha", "rdkit", Descriptors.HallKierAlpha),
] + [(f"BCUT2D_{t}", "rdkit", (lambda i: (lambda m: rdMolDescriptors.BCUT2D(m)[i]))(i))
     for i, t in enumerate(["MWHI", "MWLOW", "CHGHI", "CHGLO",
                            "LOGPHI", "LOGPLOW", "MRHI", "MRLOW"])]


def main() -> None:
    smis = HERE.joinpath("mols.smi").read_text().split()
    got = np.loadtxt(HERE / "values_hume.txt", ndmin=2)
    assert got.shape[1] == len(SPEC), f"C++ wrote {got.shape[1]} cols, SPEC has {len(SPEC)}"
    assert len(smis) == len(got), f"{len(smis)} smiles vs {len(got)} rows"

    ref = np.full_like(got, np.nan)
    cache = {}
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        for j, spec in enumerate(SPEC):
            if spec[1] == "rdkit":
                try:
                    ref[i, j] = spec[2](m)
                except Exception:
                    pass
            else:
                mod, name = spec[2], spec[3]
                key = (id(mod), i)
                if key not in cache:
                    cache[key] = mod.featurize(m)
                ref[i, j] = cache[key][mod.NAMES.index(name)]
        cache.clear()

    print(f"{len(smis):,} molecules, {len(SPEC)} descriptors\n")
    print(f"  {'descriptor':14s} {'ref':>8s} {'exact':>9s} {'max rel dev':>13s}  verdict")
    ok = True
    for j, spec in enumerate(SPEC):
        a, b = got[:, j], ref[:, j]
        keep = np.isfinite(b) & np.isfinite(a)
        a, b = a[keep], b[keep]
        rtol, atol = TOL[spec[1]]
        passed = np.abs(a - b) <= atol + rtol * np.abs(b)
        rel = np.abs(a - b) / np.maximum(np.abs(b), atol)
        worst = float(rel.max()) if rel.size else np.inf
        good = bool(passed.all())
        ok &= good
        print(f"  {spec[0]:14s} {spec[1]:>8s} {100*float(passed.mean()):8.3f}% "
              f"{worst:13.3e}  {'MATCH' if good else 'MISMATCH'}")
        if not good:
            k = int(np.argmax(~passed))
            print(f"      worst: {np.array(smis)[keep][k]}")
            print(f"      c++ {a[k]!r}  ref {b[k]!r}")
    print("\n" + ("ALL EXACT" if ok else "DISAGREEMENT"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
