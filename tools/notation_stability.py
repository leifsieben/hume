"""Experiment B: is each column invariant to how the molecule was written?

    .venv/bin/python tools/notation_stability.py [n_molecules] [n_variants]

A descriptor must be a function of the MOLECULE, not of the string. Anything that depends on a
graph-canonicalisation tiebreak or an atom ordering injects pure noise into every downstream
model -- noise that no amount of data averages away, because the same molecule from a different
source file gets a different value.

HUME already applies this thinking to fingerprints, where the two notation controls sit at
exactly 0.000. It has never been applied per descriptor column, which is the gap this closes.

METHOD. Take a molecule, write it as its canonical SMILES, and also as `n_variants` RANDOM
SMILES (`doRandom=True`, which permutes the atom ordering). Re-parse each and featurize. Every
column must return the same value. Report, per column, the largest deviation seen across
variants, scaled by that column's spread over the corpus so the number is comparable between a
count and a surface area.

⚠️ SOME INSTABILITY IS EXPECTED AND ALREADY DOCUMENTED. METHODS.md records descriptors whose
upstream definition depends on atom numbering or on a Kekule choice; those are reproduced
deliberately, quirk and all. This does not re-litigate that -- it MEASURES it, per column, so
the size of the problem is known rather than assumed.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = Path("results/reanalysis")
OUT.mkdir(parents=True, exist_ok=True)
names = list(molhume.feature_names(fingerprint=False))

smis = json.load(open("data/exactness_corpus.json"))["smiles"][:N]
mols = [Chem.MolFromSmiles(s) for s in smis]
mols = [m for m in mols if m is not None]
print(f"  {len(mols)} molecules x {K} random rewritings")

canon = [Chem.MolToSmiles(m) for m in mols]
Xc = molhume.featurize(canon, standardize="none", fingerprint=False)

worst = np.zeros(len(names))
nz = np.zeros(len(names), dtype=int)
nan_flip = np.zeros(len(names), dtype=int)
for k in range(K):
    rnd = [Chem.MolToSmiles(m, canonical=False, doRandom=True) for m in mols]
    Xr = molhume.featurize(rnd, standardize="none", fingerprint=False)
    both = np.isfinite(Xc) & np.isfinite(Xr)
    d = np.zeros_like(Xc)
    d[both] = np.abs(Xc[both] - Xr[both])
    worst = np.maximum(worst, d.max(axis=0))
    nz += (d > 0).sum(axis=0)
    nan_flip += (np.isfinite(Xc) != np.isfinite(Xr)).sum(axis=0)
    print(f"    variant {k+1}: {(d > 0).any(axis=0).sum()} columns moved on at least one molecule")

# scale by the column's own spread, so a count and a surface area are comparable
scale = np.nanstd(np.where(np.isfinite(Xc), Xc, np.nan), axis=0)
scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
rel = worst / scale

rows = [dict(column=names[i], worst_abs=float(worst[i]), worst_over_sd=float(rel[i]),
             cells_moved=int(nz[i]), nan_flips=int(nan_flip[i]))
        for i in range(len(names)) if nz[i] or nan_flip[i]]
rows.sort(key=lambda r: -r["worst_over_sd"])
json.dump({"n_molecules": len(mols), "n_variants": K, "unstable": rows},
          open(OUT / "notation_stability.json", "w"), indent=1)

print(f"\n  {len(rows)} of {len(names)} columns are NOT invariant to how the molecule is written")
print(f"  {sum(1 for r in rows if r['worst_over_sd'] > 0.01)} move by more than 1% of their own SD")
print(f"  {sum(1 for r in rows if r['nan_flips'])} flip between finite and NaN\n")
print(f"  {'column':24s} {'worst/SD':>10s} {'cells moved':>12s} {'NaN flips':>10s}")
for r in rows[:25]:
    print(f"  {r['column']:24s} {r['worst_over_sd']:10.4f} {r['cells_moved']:12d} "
          f"{r['nan_flips']:10d}")
print(f"\n  -> {OUT / 'notation_stability.json'}")
