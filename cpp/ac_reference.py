"""A NumPy reference for Mordred's Autocorrelation block, checked against Mordred itself.

Written BEFORE the C++ so the specification is pinned in a language where iterating is cheap.
Every convention below was read out of mordred.Autocorrelation and then confirmed numerically;
the C++ port is a transliteration of this file, so any disagreement it shows is a porting bug
rather than a misunderstanding of the descriptor.

THE CONVENTIONS THAT MATTER, none of which are guessable:

  * explicit_hydrogens = True. Autocorrelation runs on the H-ADDED graph, unlike chi, kappa and
    BCUT2D, which are heavy-atom only. Aspirin is 13 heavy atoms and 21 with hydrogens, and the
    distance matrix is the H-added one throughout.
  * gsum_k = sum(dmat == k), HALVED for k > 0 but NOT for k = 0. Delta_0 is therefore A, not
    A/2, and AATS0 differs from AATS1.. by that factor alone.
  * ATS_0 = sum(w^2) exactly; ATS_k = 0.5 * w^T B_k w for k > 0.
  * GATS divides by 4*gsum, not 2*gsum -- the gmat sum already double-counts each pair and the
    published Geary formula assumes it does not.
  * The GATS denominator uses (A - 1), the ATSC/MATS one uses A. Two different normalisations
    inside the same family.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from mordred import _atomic_property as ap

RDLogger.DisableLog("rdApp.*")

VARIANTS = ["ATS", "AATS", "ATSC", "AATSC", "MATS", "GATS"]
LAGS = list(range(9))
WEIGHTS = ["c", "d", "dv", "i", "p", "v", "se", "pe", "are"]


def avec(mol_h, short):
    """Per-atom property vector on the H-ADDED molecule, straight from Mordred's own getter."""
    # Gasteiger charges must be COMPUTED on the H-added molecule before the getter can read
    # them off the atoms; every other getter is a pure function of the atom.
    if short == "c":
        from rdkit.Chem import rdPartialCharges
        rdPartialCharges.ComputeGasteigerCharges(mol_h)
    g = ap.getters[short]
    return np.array([float(g(a)) for a in mol_h.GetAtoms()], float)


def autocorr(mol, short, lags=LAGS):
    """-> {(variant, lag): value}. NaN where Mordred returns NaN."""
    mh = Chem.AddHs(mol)
    w = avec(mh, short)
    A = len(w)
    D = rdmolops.GetDistanceMatrix(mh)
    wc = w - w.mean()
    out = {}
    for k in lags:
        G = (D == k).astype(float)
        gsum = G.sum() if k == 0 else 0.5 * G.sum()
        ats = (w * w).sum() if k == 0 else 0.5 * float(w @ G @ w)
        atsc = (wc * wc).sum() if k == 0 else 0.5 * float(wc @ G @ wc)
        out[("ATS", k)] = ats
        out[("ATSC", k)] = atsc
        out[("AATS", k)] = ats / gsum if gsum else np.nan
        aatsc = atsc / gsum if gsum else np.nan
        out[("AATSC", k)] = aatsc
        den_m = (wc * wc).sum() / A
        out[("MATS", k)] = aatsc / den_m if den_m else np.nan
        if A <= 1 or gsum == 0:
            out[("GATS", k)] = np.nan
        else:
            num = (G * (w[:, None] - w) ** 2).sum() / (4 * gsum)
            den = (wc * wc).sum() / (A - 1)
            out[("GATS", k)] = num / den if den else np.nan
    return out


def main() -> None:
    from mordred import Autocorrelation as AC, Calculator
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1", "CC1CO1", "CCO",
            "C[C@H](N)C(=O)O", "Clc1ccc(CSc2nnc(N)s2)cc1", "O=S(=O)(N)c1ccccc1"]
    descs, names = [], []
    for v in VARIANTS:
        for k in LAGS:
            for wt in WEIGHTS:
                descs.append(getattr(AC, v)(k, wt))
                names.append((v, k, wt))
    calc = Calculator(descs)
    worst, nbad, ntot = 0.0, 0, 0
    for s in smis:
        m = Chem.MolFromSmiles(s)
        ref = list(calc(m))
        cache = {wt: autocorr(m, wt) for wt in WEIGHTS}
        for (v, k, wt), r in zip(names, ref):
            mine = cache[wt][(v, k)]
            try:
                rv = float(r)
            except Exception:
                rv = np.nan
            ntot += 1
            if np.isnan(rv) and np.isnan(mine):
                continue
            if np.isnan(rv) != np.isnan(mine):
                nbad += 1
                if nbad <= 4:
                    print(f"  NaN mismatch {v}{k}{wt} on {s}: mine={mine} mordred={rv}")
                continue
            d = abs(mine - rv) / max(abs(rv), 1e-12)
            if d > worst:
                worst = d
            if d > 1e-9:
                nbad += 1
                if nbad <= 4:
                    print(f"  {v}{k}{wt} on {s}: mine={mine!r} mordred={rv!r}")
    print(f"\n{ntot} cells over {len(smis)} molecules | mismatches {nbad} | "
          f"max rel dev {worst:.3e}")
    print("SPEC CONFIRMED" if nbad == 0 else "SPEC WRONG")
    raise SystemExit(0 if nbad == 0 else 1)


if __name__ == "__main__":
    main()
