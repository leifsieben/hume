"""Does the 0.99 dedupe cost us on classification endpoints?

    python dedupe_cost.py

THE QUESTION. HUME carries 864 of the 1,830 RDKit+Mordred columns; the rest were dropped as
unusable or as |r| >= 0.99 with a column that was kept. Two columns can be 0.995-correlated on
the dedupe corpus and separate cleanly on ONE endpoint, in which case the filter was applied
correctly and the drop is still a real loss there.

THE DESIGN ISOLATES THE FILTER AND NOTHING ELSE. Both arms use the same fingerprint (Morgan
r=3, the radius HUME carries) and the same source columns computed the same way. They differ
only in whether the 966 dropped columns are present. Comparing HUME itself against
`ecfp_all_desc` would confound the filter with HUME's own extra blocks, its C++ implementations
and its fingerprint radius; this does not.

Head and tuning are protocol 2, matching the grid: 3-fold inner CV for feature_weights, and the
documented w=10 below 200 inner-validation molecules.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MORDRED_PY", ".venv-mordred/bin/python")
import numpy as np, bench_downstream as BD
from chempfn.eval.splits import scaffold_folds, train_test
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator as rfg
import xgboost as xgb

DS = ["bioavail", "hia"]   # the ONLY two with a material gap; the four 7k-13k sets are within 0.003
RD = [n for n, _ in Descriptors._descList]
MD = [n for n in open("/tmp/mordred_names.txt").read().split("\n") if n]
KEPT = json.load(open("data/dedupe.json"))
KEPT_EXACT = {n for _s, n, _c in KEPT["compute"]} | {n for _s, n, _c in KEPT["predict"]}
NAMES = RD + MD
# EXACT names, and FIRST occurrence only. 52 descriptor names appear in BOTH the RDKit and the
# Mordred lists, so selecting by name alone picked each of those twice and the "deduped" arm came
# out at 922 columns instead of 864 -- i.e. it silently carried duplicates of exactly the kind the
# filter exists to remove. Case-insensitive matching added 6 more (nAHRing / naHRing and friends).
_seen = set()
KEEP_IDX = []
for i, n in enumerate(NAMES):
    if n in KEPT_EXACT and n not in _seen:
        _seen.add(n); KEEP_IDX.append(i)
KEEP_IDX = np.array(KEEP_IDX)
assert len(KEEP_IDX) == len(KEPT_EXACT & set(NAMES)), (len(KEEP_IDX), len(KEPT_EXACT))
print(f"union {len(NAMES)} columns -> deduped arm has {len(KEEP_IDX)} (dedupe.json keeps "
      f"{len(KEPT_EXACT)})")

def ecfp3(smis):
    g = rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)
    out = np.zeros((len(smis), 2048), np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = g.GetFingerprintAsNumPy(m)
    return out

def fw(ncol, w):
    v = np.ones(ncol, np.float32); v[:2048] = w; return v

def fit_score(X, y, tr, te):
    idx = np.asarray(tr)
    if len(idx) // 3 < 200:
        best = 10.0
    else:
        best, bs = 10.0, None
        for w in (1.0, 5.0, 10.0, 100.0):
            ss = []
            for j in range(3):
                iva = idx[j::3]; itr = np.concatenate([idx[q::3] for q in range(3) if q != j])
                if len(np.unique(y[itr])) < 2: continue
                mi = xgb.XGBClassifier(tree_method="hist", colsample_bynode=0.3,
                                       feature_weights=fw(X.shape[1], w), n_jobs=8, random_state=0)
                mi.fit(X[itr], y[itr])
                ss.append(roc_auc_score(y[iva], mi.predict_proba(X[iva])[:, 1]))
            if ss and (bs is None or np.mean(ss) > bs): best, bs = w, float(np.mean(ss))
    m = xgb.XGBClassifier(tree_method="hist", colsample_bynode=0.3,
                          feature_weights=fw(X.shape[1], best), n_jobs=8, random_state=0)
    m.fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])

out = {}
for ds in DS:
    BD._BLOCK_CACHE.clear()
    d = BD.load_ds(ds); smis, y = d["smiles"], np.asarray(d["y"], float)
    if len(smis) > 50_000:
        k = np.random.default_rng(0).choice(len(smis), 50_000, replace=False)
        smis = [smis[i] for i in k]; y = y[k]
    D = np.hstack([BD.f_rdkit_desc(smis), BD.f_mordred_desc(smis)]).astype(np.float32)
    F = ecfp3(smis)
    arms = {"dedup864": np.hstack([F, D[:, KEEP_IDX]]), "full1830": np.hstack([F, D])}
    for a in arms: arms[a][~np.isfinite(arms[a])] = np.nan
    folds = scaffold_folds(smis, k=5, seed=0)
    res = {}
    for a, X in arms.items():
        v = [fit_score(X, y, *train_test(folds, i)) for i in range(5)]
        res[a] = (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(5)))
    out[ds] = res
    dd, ff = res["dedup864"], res["full1830"]
    print(f"  {ds:<13} n={len(smis):>6}  dedup864 {dd[0]:.4f}+/-{dd[1]:.4f}   "
          f"full1830 {ff[0]:.4f}+/-{ff[1]:.4f}   delta {ff[0]-dd[0]:+.4f}", flush=True)
    json.dump(out, open("results/figures/dedupe_cost.json", "w"), indent=1)

dl = [out[d]["full1830"][0] - out[d]["dedup864"][0] for d in out]
print(f"\nmean AUROC gain from the 966 dropped columns: {np.mean(dl):+.4f} "
      f"(sd {np.std(dl, ddof=1):.4f} over {len(dl)} datasets)")
