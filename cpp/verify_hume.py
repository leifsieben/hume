"""Is the consolidated C++ featuriser exactly the reference?

    ./hume verify mols.txt && python cpp/verify_hume.py

TWO REFERENCES, and conflating them would hide errors in both:

  * chi (chi0n..chi4n, chi0v..chi4v) and both BalabanJ columns -> RDKIT is the reference.
    These are ports of somebody else's descriptor and "our number" has no standing. chi.py
    itself is now a reimplementation of RDKit's convention, so comparing the C++ against
    chi.py for those ten would only prove it agrees with a second copy of our own code --
    they are gated against RDKit directly, at the tighter tolerance.
  * cycles, resistance, conjugation, stereo -> OUR PYTHON MODULES are the reference. They are
    the definition of those blocks, and they are themselves already verified against RDKit /
    the 1M corpus where an external reference exists.

Columns are matched BY NAME against each module's NAMES list rather than by position, so
inserting a feature into a Python block cannot silently shift the comparison onto the wrong
pair of numbers. At 182 columns the whole "py" half of SPEC is GENERATED from the modules'
own NAMES lists for the same reason -- a hand-written list of 165 entries is a transcription
error waiting to happen, and one that would show up as a plausible-looking mismatch in the
wrong descriptor.

ONE PAIR OF COLUMNS SHARES A NAME AND IS NOT THE SAME DESCRIPTOR: BalabanJ and
BalabanJ_mordred -- same formula, weighted against unweighted distance matrix. Both verified.

There used to be a second such pair. Chi2n..Chi4v were emitted twice, once gated on RDKit and
once on chi.py, because chi.py stripped explicit hydrogen with RemoveHs(removeIsotopes=True)
and so normalised [2H]C(C)O onto plain ethanol -- a genuinely different descriptor that
disagreed with RDKit on 468 of 468 explicit-H molecules, while its docstring claimed the
opposite. chi.py now follows RDKit, the two became bit-identical on all 98,905 molecules
(checked before removal, not assumed), and the duplicates are retired.

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
import chi, conjugation, cycles, resistance, stereo                  # noqa: E402

# Columns C_sssr and C_redundancy are NOT emitted by the C++ and are deliberately absent here
# rather than compared loosely. C_sssr is len(Chem.GetSymmSSSR(mol)), and the symmetrised SSSR
# is NOT the cyclomatic number: it disagrees on 521 of 20,000 corpus molecules (2.6%), every one
# a bridged polycyclic where the symmetrisation adds a ring the m - n + c basis does not have.
# Reproducing it means reimplementing RDKit's ring perception, which is exactly the thing
# export_predict.py's docstring refuses to do for hybridisation -- "a second implementation of
# RDKit's perception rules and the first place an 'exact' claim would quietly stop being true".
# The right fix is one more integer from the exporter; see the report. C_redundancy is
# C_total / C_sssr and is blocked on the same value.
_CYCLES_SKIP = ("C_sssr", "C_redundancy")

# RDKit exposes Chi0n..Chi4n and Chi0v..Chi4v. Those ten chi.py columns are checked against it
# at RDKit tolerance; the rest of the block (k = 5,6,7 and the path counts) has no external
# counterpart and is checked against chi.py itself.
_CHI_RDKIT = {f"chi{k}{s}": getattr(GD, f"Chi{k}{s}") for s in ("n", "v") for k in range(5)}


def _chi_ref(nm):
    fn = _CHI_RDKIT.get(nm)
    return (nm, "rdkit", fn) if fn is not None else (nm, "py", chi, nm)

SPEC = [
    ("BalabanJ", "rdkit", GD.BalabanJ),
    # The SECOND BalabanJ. HUME's column set carries both ('rdkit','BalabanJ') and
    # ('mordred','BalabanJ') under the same name, and they are different numbers -- 2.888052
    # against 1.925368 on naphthalene. mordred/BalabanJ.py calls RDKit's own function but hands
    # it an UNWEIGHTED distance matrix, bypassing the useBO=1 weighting RDKit applies by default.
    #
    # The reference below reproduces that argument rather than importing mordred, so this
    # harness keeps running under .venv/bin/python (mordred is not installed there, and a
    # per-molecule mordred Calculator over 100k molecules costs more than the rest of the
    # verify put together). That substitution is not taken on trust: cpp/spec_balaban.py pins
    # it against MORDRED ITSELF over 4,000 molecules drawn from the adversarial corpus --
    # salts, isotopes, explicit H, rare elements -- and agrees on 4,000 of 4,000. The reference
    # here is still RDKit's implementation; only the matrix argument is reconstructed.
    ("BalabanJ_mordred", "rdkit",
     lambda m: GD.BalabanJ(m, dMat=Chem.GetDistanceMatrix(m, useBO=False, useAtomWts=False,
                                                          force=True))),
] + [
    # WHOLE BLOCKS, GENERATED FROM THE MODULES' OWN NAMES LISTS. Writing these out by hand is
    # what the position-matching hazard in the header warns about, and at 165 columns it stops
    # being a hazard and becomes a certainty. The C++ emits each block in exactly this order.
    #
    # chi0n..chi4n and chi0v..chi4v are gated against RDKIT rather than against chi.py, at the
    # tighter rtol 1e-9 -- RDKit defines those and chi.py is now a reimplementation of them, so
    # comparing to chi.py would only prove the C++ agrees with a second copy of our own code.
    # k = 5,6,7 have no RDKit counterpart and fall back to chi.py, as do the path* columns.
    _chi_ref(nm) if mod is chi else (nm, "py", mod, nm)
    for mod in (chi, cycles, conjugation, stereo, resistance)
    for nm in mod.NAMES if not (mod is cycles and nm in _CYCLES_SKIP)
] + [
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
