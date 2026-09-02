"""(1) Does the autocorrelation weight argument transfer to BCUT?  (2) Joint drop test.

PART 1 -- BCUT. The 20 mordred BCUTs are indexed by the same weight vocabulary we cut from the
autocorrelation block. But BCUT is an EXTREME EIGENVALUE of a matrix carrying the weight on its
diagonal, not a pair sum, and two affinely related diagonals do not give affinely related
eigenvalues -- the off-diagonal part does not scale with them. So the argument motivates the cut
and does not establish it. Measured here: correlation between BCUTs of same-axis weights, and
whether the consumer can rebuild the dropped ones.

PART 2 -- JOINT DROP. The family screen removes ONE family at a time. If path/walk is
reproducible from a library still containing matrix spectrum, and spectrum is reproducible from
one still containing path/walk, neither result licenses dropping both: mutual redundancy makes
each look individually free while the pair is not. Tested by removing them TOGETHER.
"""
import json
import re
import warnings

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

# ---------------- PART 1
bcut = [c for c in names if c.startswith("BCUT")]
mordred = [c for c in bcut if re.fullmatch(r"BCUT([A-Za-z]+)-1[hl]", c)]
rdkit_b = [c for c in bcut if c.startswith("BCUT2D_")]
print(f"  BCUT: {len(bcut)} columns  ({len(mordred)} mordred, {len(rdkit_b)} rdkit)\n")
def w_of(c):
    m = re.fullmatch(r"BCUT([A-Za-z]+)-1([hl])", c)
    return (m.group(1), m.group(2)) if m else None
KEEPW = {"c", "d", "dv", "s", "Z", "pe", "v"}
def r(a, b):
    x, y = X[:, idx[a]], X[:, idx[b]]
    m = np.isfinite(x) & np.isfinite(y)
    return abs(float(np.corrcoef(x[m], y[m])[0, 1])) if m.sum() > 500 else None
print("  correlation between BCUTs whose weights are the SAME element axis:")
for end in ("h", "l"):
    for a, b in (("se", "pe"), ("se", "are"), ("pe", "are"), ("v", "p"), ("i", "pe")):
        ca, cb = f"BCUT{a}-1{end}", f"BCUT{b}-1{end}"
        if ca in idx and cb in idx:
            print(f"    BCUT{a}-1{end} ~ BCUT{b}-1{end}:  {r(ca,cb):.4f}")
drop_b = [c for c in mordred if w_of(c) and w_of(c)[0] not in KEEPW]
keep_all = [c for c in names if c not in set(drop_b)]
print(f"\n  dropping BCUTs on the cut weights: {len(drop_b)} columns")
vals = [v for c in drop_b if (v := inv(c, keep_all)) is not None]
if vals:
    print(f"    rebuilt from the rest: median R2 {np.median(vals):.4f}, "
          f"floor {min(vals):.4f}, {sum(1 for v in vals if v<0.9)} below 0.9")

# ---------------- PART 2
PATH = [c for c in names if re.match(r"^(MPC|piPC|TPC|SRW|MWC|TWC)", c)]
SPEC = [c for c in names if re.match(r"^(SpAbs|SpMax|SpDiam|SpMAD|VE\d|VR\d|SM1)", c)]
print(f"\n  JOINT TEST: path/walk ({len(PATH)}) + matrix spectrum ({len(SPEC)})")
for label, removed, targets in (("path alone", PATH, PATH), ("spectrum alone", SPEC, SPEC),
                                ("BOTH -> path", PATH + SPEC, PATH),
                                ("BOTH -> spectrum", PATH + SPEC, SPEC)):
    design = [c for c in names if c not in set(removed)]
    smp = targets[:: max(1, len(targets) // 15)]
    vs = [v for c in smp if (v := inv(c, design)) is not None]
    print(f"    {label:18s} median R2 {np.median(vs):.4f}  floor {min(vs):.4f}  "
          f"{sum(1 for v in vs if v<0.9)} below 0.9")
