"""Pin the SECOND BalabanJ, so the C++ can emit both columns.

HUME's column set contains two descriptors that are both called BalabanJ:

    ('rdkit',   'BalabanJ', 'rdkit_core')
    ('mordred', 'BalabanJ', 'BalabanJ')

They are NOT the same number -- on naphthalene, 2.888052 vs 1.925368. The C++ implements only
the first. Shipping the second as a copy of the first would put a wrong column in the matrix,
so this pins what the second actually is before anything is written.

Reading mordred/BalabanJ.py answers most of it: mordred is a thin wrapper that calls RDKit's
own BalabanJ but passes `dMat=DistanceMatrix(explicit_hydrogens=False)` -- its own UNWEIGHTED
topological distance matrix. RDKit's default path builds a BOND-ORDER-WEIGHTED matrix
(useBO=1) instead. Same formula, different D.

Two things reading cannot settle, so they are measured here:

  H HANDLING.  mordred sets explicit_hydrogens = False and prepares its own mol. Our corpus
  has an `explicit_h` stratum (molecules round-tripped through Chem.AddHs), so whether mordred
  strips them and RDKit does not is a real divergence on ~4,000 of the 100k, not a hypothetical.

  DISCONNECTED.  RDKit writes 1e8 into unreachable cells. Our corpus is 10,000 salts and
  mixtures. Whether the 1e8 sentinel survives into mordred's matrix decides whether the C++
  can reuse the unweighted distance matrix it ALREADY builds for conjugation and resistance --
  which is the difference between this column costing ~0 and costing another BFS.

    uv run --with mordred --with rdkit python cpp/spec_balaban.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import GraphDescriptors as GD

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent

PROBES = {
    "naphthalene": "c1ccc2ccccc2c1",
    "benzene": "c1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "butane": "CCCC",
    "alkyne": "CC#CC",
    "allene": "C=C=C",
    "salt": "CCO.CCN",
    "explicit_h": "[H]C([H])([H])O[H]",
    "tiny": "CC",
    "single_atom_frag": "C.CCCC",
    "macrocycle": "C1CCCCCCCCCCCCCCC1",
}


def mordred_J(mols):
    from mordred import BalabanJ as MB, Calculator
    calc = Calculator([MB.BalabanJ()])
    return [float(next(iter(calc(m)))) for m in mols]


def candidate_J(m):
    """Our hypothesis: RDKit's formula on the UNWEIGHTED, H-STRIPPED distance matrix."""
    mh = Chem.RemoveHs(m)
    if mh.GetNumAtoms() < 2:
        return float("nan")
    D = Chem.GetDistanceMatrix(mh, useBO=False, useAtomWts=False, force=True)
    return float(GD.BalabanJ(mh, dMat=D))


def main() -> None:
    print(f"  {'probe':18s} {'rdkit useBO=1':>14s} {'mordred':>12s} {'candidate':>12s}  ok")
    for name, smi in PROBES.items():
        m = Chem.MolFromSmiles(smi)
        r = GD.BalabanJ(m)
        d = mordred_J([m])[0]
        c = candidate_J(m)
        agree = (np.isnan(d) and np.isnan(c)) or abs(c - d) <= 1e-9 + 1e-9 * abs(d)
        print(f"  {name:18s} {r:14.6f} {d:12.6f} {c:12.6f}  {'yes' if agree else 'NO'}")

    # now the real test: the adversarial corpus, where the strata that matter actually live
    smis = HERE.joinpath("hard.smi").read_text().split()
    random.seed(3)
    pick = random.sample(smis, 4000)
    mols = [Chem.MolFromSmiles(s) for s in pick]
    keep = [(s, m) for s, m in zip(pick, mols) if m is not None and m.GetNumAtoms() >= 2]
    print(f"\n  {len(keep):,} corpus molecules")

    ref = mordred_J([m for _, m in keep])
    got = [candidate_J(m) for _, m in keep]
    bad = []
    for (s, m), a, b in zip(keep, got, ref):
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) != np.isnan(b) or abs(a - b) > 1e-9 + 1e-9 * abs(b):
            bad.append((s, a, b, m))
    print(f"  candidate vs mordred: {len(keep)-len(bad):,} exact, {len(bad):,} disagree")
    for s, a, b, m in bad[:6]:
        has_h = any(at.GetAtomicNum() == 1 for at in m.GetAtoms())
        print(f"    {s[:64]}\n      cand {a!r}  mordred {b!r}  "
              f"frags {len(Chem.GetMolFrags(m))} explicitH {has_h}")

    # how far apart are the two columns in general? if they were near-identical the second
    # column would be near-redundant and worth saying so.
    rd = np.array([GD.BalabanJ(m) for _, m in keep])
    mo = np.array(ref)
    ok = np.isfinite(rd) & np.isfinite(mo)
    print(f"\n  the two columns differ: median |rdkit-mordred| = "
          f"{np.median(np.abs(rd[ok]-mo[ok])):.4f}, "
          f"correlation {np.corrcoef(rd[ok], mo[ok])[0,1]:.4f}")
    print(f"  rdkit  range [{rd[ok].min():.3f}, {rd[ok].max():.3f}]")
    print(f"  mordred range [{mo[ok].min():.3f}, {mo[ok].max():.3f}]")


if __name__ == "__main__":
    main()
