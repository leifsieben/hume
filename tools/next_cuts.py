"""AETA / RW / chi consumer inversion, plus the autocorrelation block ablation.

The first three are the constructual-or-exact leads. The fourth answers a different question:
is the autocorrelation block IMPORTANT, as opposed to irreducible? It is the least reproducible
family in the library (nothing else rebuilds it) and has full exact rank, but that says it is
distinct, not that it helps. The earlier ETA ablation could not resolve a 29-column removal
against fold noise; removing 227 columns -- 29% of the library -- should produce an effect above
noise if there is one. If it does not, that is real evidence about importance rather than a
failed measurement.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import RDLogger

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
names = list(molhume.feature_names(fingerprint=False))
idx = {c: i for i, c in enumerate(names)}
smis = json.load(open("data/exactness_corpus.json"))["smiles"][:15000]
X = molhume.featurize(smis, standardize="none", fingerprint=False)
X = np.where(np.isfinite(X), X, np.nan)
rng = np.random.default_rng(0); p = rng.permutation(len(X)); tr, te = p[:12000], p[12000:]

def inv(target, design):
    y = X[:, idx[target]]; ok = np.isfinite(y)
    if ok[te].sum() < 300 or np.nanstd(y[te]) == 0:
        return None
    F = X[:, [idx[c] for c in design]]
    m = xgb.XGBRegressor(n_estimators=150, max_depth=6, random_state=0, n_jobs=-1,
                         verbosity=0, tree_method="hist")
    m.fit(F[tr][ok[tr]], y[tr][ok[tr]])
    pr = m.predict(F[te][ok[te]]); yy = y[te][ok[te]]
    return float(1 - ((yy - pr) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())

BLOCKS = {
  "AETA": [c for c in names if c.startswith("AETA_")],
  "RW walk stats": sorted(c for c in names if re.match(r"^RW\d+_", c)),
  "chi subgraph": [c for c in names if re.match(r"^(A?X(p|c|pc|ch)-)", c) or re.match(r"^chi\d", c)],
}
print("  CONSUMER INVERSION, whole block removed from the design\n")
for nm, cols in BLOCKS.items():
    others = [c for c in names if c not in set(cols)]
    smp = cols[:: max(1, len(cols) // 20)]
    vs = [v for c in smp if (v := inv(c, others)) is not None]
    print(f"  {nm:16s} {len(cols):4d} cols, {len(vs):3d} sampled: median R2 {np.median(vs):.4f}"
          f"  floor {min(vs):.4f}  {sum(1 for v in vs if v < 0.9)} below 0.9")

# within-family parametric structure
print("\n  WITHIN-FAMILY structure (the parametric axis):")
def r(a, b):
    x, y = X[:, idx[a]], X[:, idx[b]]
    m = np.isfinite(x) & np.isfinite(y)
    return abs(float(np.corrcoef(x[m], y[m])[0, 1])) if m.sum() > 500 else None
ks = [2, 3, 4, 6, 8, 12, 16]
for stat in ("mean", "max", "std", "q90"):
    vs = [(f"{a}~{b}", r(f"RW{a}_{stat}", f"RW{b}_{stat}"))
          for a, b in zip(ks, ks[1:]) if f"RW{a}_{stat}" in idx and f"RW{b}_{stat}" in idx]
    if vs:
        print(f"    RW *_{stat:4s} adjacent walk lengths: " +
              "  ".join(f"{k}:{v:.3f}" for k, v in vs if v is not None))
for pre in ("Xp-", "Xc-", "Xch-", "AXp-"):
    cols = sorted(c for c in names if c.startswith(pre))
    vs = []
    for a, b in zip(cols, cols[1:]):
        v = r(a, b)
        if v is not None: vs.append(v)
    if vs:
        print(f"    chi {pre:5s} adjacent orders: median |r| {np.median(vs):.3f}  ({len(vs)} pairs)")
