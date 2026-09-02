"""Are the 75 substructure columns recoverable from the ECFP6 that ships alongside them?

    .venv/bin/python tools/frag_vs_ecfp.py [n_molecules]

A consumer-inversion test (experiment E) applied to a whole family. HUME emits an ECFP by
default, so if a fr_* column is reconstructible from those 2,048 bits BY THE ACTUAL DOWNSTREAM
MODEL, it is dead weight in every configuration anyone runs.

Unlike the linear-recoverability criterion this replaces, the consumer here is not hypothetical:
it is the same untuned XGBoost the benchmark grid uses. "Recoverable" and "recovered" came apart
badly once already; this measures the second one.

WHAT TO EXPECT, so the result can surprise us. ECFP6 hashes circular environments out to radius
3. A pattern that FITS inside radius 3 of some atom should be nearly perfectly recoverable -- a
bit essentially is that pattern. A pattern larger than that, or one defined by a global property
rather than a local environment, has no single bit to live in and should be harder. If the
result does not split roughly that way, the mental model of what ECFP encodes is wrong.

 Recoverability is necessary but not sufficient grounds to drop a column. A column that is
recoverable is redundant; a column that is NOT recoverable still has to earn its place on
mechanism, not merely on being different.
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = Path("results/reanalysis")

names = list(molhume.feature_names(fingerprint=False, columns="full"))
smis = json.load(open("data/exactness_corpus.json"))["smiles"][:N]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
print(f"  {len(mols)} molecules")
fp, X, _ = molhume.featurize_all_from_mols(mols, optional=("AvgIpc",))
fp = fp.astype(np.float32)
print(f"  ECFP {fp.shape}, descriptors {X.shape}")

frag = [i for i, c in enumerate(names) if c.startswith("fr_")]
print(f"  {len(frag)} fr_* columns\n")

import xgboost as xgb
rng = np.random.default_rng(0)
perm = rng.permutation(len(mols))
tr, te = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

rows = []
for i in frag:
    y = X[:, i].astype(np.float64)
    ok = np.isfinite(y)
    if not ok.all():
        y = np.where(ok, y, 0.0)
    fires = float((y != 0).mean())
    if y.std() == 0:
        rows.append(dict(column=names[i], fires=fires, r2=None, note="constant on this corpus"))
        continue
    m = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1,
                         verbosity=0, tree_method="hist")
    m.fit(fp[tr], y[tr])
    p = m.predict(fp[te])
    sse = float(((y[te] - p) ** 2).sum())
    sst = float(((y[te] - y[te].mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else None
    rows.append(dict(column=names[i], fires=fires, r2=r2))
    print(f"  {names[i]:26s} fires {fires:7.3%}  R2 from ECFP6 {r2 if r2 is None else round(r2,4)}")

json.dump(rows, open(OUT / "frag_vs_ecfp.json", "w"), indent=1)
good = [r for r in rows if r.get("r2") is not None]
hi = [r for r in good if r["r2"] >= 0.95]
lo = [r for r in good if r["r2"] < 0.80]
print(f"\n  {len(hi)} of {len(good)} recoverable at R2 >= 0.95 (redundant with the ECFP)")
print(f"  {len(lo)} below 0.80 (the ECFP does not carry them)")
print("\n  least recoverable:")
for r in sorted(good, key=lambda r: r["r2"])[:12]:
    print(f"    {r['column']:26s} fires {r['fires']:7.3%}  R2 {r['r2']:.4f}")
print(f"\n  -> {OUT / 'frag_vs_ecfp.json'}")
