"""Does dropping se/are/i/p/m actually lose anything the kept columns cannot rebuild?

Consumer inversion (experiment E) applied to Decision 1, and to whether Decision 3 undermines it.

THE ARGUMENT BEING TESTED. Dropping a correlated weight was justified by: the autocorrelation is
a sum over pairs, so a second near-duplicate weight injects a TOPOLOGY term (the pair count at
lag k) on top of the first weight's signal -- and that topology term is already carried
elsewhere in the library. The second half of that was asserted without checking, and checking
shows there is NO constant-weight autocorrelation emitted, so the pair-count profile is not
directly present. It may still be reconstructible from path counts and distance functionals,
which is what this measures.

TWO SCENARIOS, because Decision 3 (subsampling ATS/AATS lags to 0,2,4,6,8) removes some of the
very columns that would carry that topology term:
    full  -- kept weights at all 9 lags
    cut   -- kept weights with ATS/AATS restricted to lags 0,2,4,6,8
If a dropped column is reconstructible under `full` but not under `cut`, the two decisions
interact and Decision 1 has to be revisited.
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
KEEP = {"c", "d", "dv", "s", "Z", "pe", "v"}
DROPW = ["se", "are", "i", "p", "m"]
OPS = ["AATSC", "AATS", "ATSC", "ATS", "MATS", "GATS"]
LAGS_KEEP = {0, 2, 4, 6, 8}

names = list(molhume.feature_names(fingerprint=False, columns="full"))
def parse(c):
    for op in OPS:
        m = re.fullmatch(rf"{op}(\d)([A-Za-z]+)", c)
        if m:
            return op, int(m.group(1)), m.group(2)

smis = json.load(open("data/exactness_corpus.json"))["smiles"][:15000]
X = molhume.featurize(smis, standardize="none", fingerprint=False)
X = np.where(np.isfinite(X), X, np.nan)
idx = {c: i for i, c in enumerate(names)}

def keep_cols(cut):
    out = []
    for c in names:
        p = parse(c)
        if p is None:
            out.append(c); continue                 # every non-autocorrelation column stays
        op, lag, w = p
        if w not in KEEP:
            continue
        if cut and op in ("ATS", "AATS") and lag not in LAGS_KEEP:
            continue
        out.append(c)
    return out

rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0])
tr, te = perm[:12000], perm[12000:]
targets = [c for c in names if (p := parse(c)) and p[2] in DROPW]
targets = targets[::4]                              # every 4th, for runtime
print(f"  {len(targets)} dropped-weight columns sampled as targets\n")

res = {}
for cut in (False, True):
    kc = [c for c in keep_cols(cut) if (p := parse(c)) is None or p[2] in KEEP]
    F = X[:, [idx[c] for c in kc]]
    r2s = []
    for t in targets:
        y = X[:, idx[t]]
        ok = np.isfinite(y)
        if ok[tr].sum() < 2000 or ok[te].sum() < 500 or np.nanstd(y[te]) == 0:
            continue
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1,
                             verbosity=0, tree_method="hist")
        m.fit(F[tr][ok[tr]], y[tr][ok[tr]])
        p_ = m.predict(F[te][ok[te]])
        yy = y[te][ok[te]]
        r2s.append(1 - ((yy - p_) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    res["cut" if cut else "full"] = np.array(r2s)
    print(f"  {'ATS/AATS lags cut to 0,2,4,6,8' if cut else 'all 9 lags kept':32s} "
          f"design {F.shape[1]:4d} cols | median R2 {np.median(r2s):.4f} | "
          f"min {np.min(r2s):.4f} | {(np.array(r2s) < 0.95).sum()} of {len(r2s)} below 0.95")

d = res["full"] - res["cut"]
print(f"\n  effect of Decision 3 on Decision 1: median R2 change {np.median(d):+.4f}, "
      f"worst {np.max(d):+.4f}")
json.dump({k: v.tolist() for k, v in res.items()},
          open("results/reanalysis/weight_inversion.json", "w"), indent=1)
