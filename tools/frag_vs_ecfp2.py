"""Substructure columns vs the ECFP6, with a metric that survives rarity -- and the mechanism.

    .venv/bin/python tools/frag_vs_ecfp2.py [n]

REPLACES the R^2 in frag_vs_ecfp.py, which was the wrong question asked of the wrong metric. For
a column that is zero on 99.95% of molecules, R^2 is set by a handful of positives: a model that
predicts ~0 everywhere and is right 99.95% of the time still scores near zero. That measures
rarity, not recoverability, and it made rare patterns look uniquely irreplaceable when the
evidence for that was an artifact.

Two questions instead, both rarity-proof:
  DETECTION  AUROC for "does this pattern occur at all", from the ECFP bits. Prevalence-free.
  COUNT      Spearman between true and predicted count, on the molecules where it DOES occur.
             ECFP here is a BINARY bit vector, so a pattern occurring three times sets the same
             bits as once -- count information has nowhere to live, and this quantifies it.

AND THE MECHANISM, which is the part that decides the argument rather than the statistic. ECFP6
hashes circular environments out to radius 3. A SMARTS pattern that fits inside some atom's
radius-3 environment essentially IS a bit; one that spans more atoms than that has no bit and
can only be inferred from a conjunction the model must learn. So recoverability should track
PATTERN SIZE. If it does, the redundancy is explained rather than merely observed, and the
prediction generalises past this corpus.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from scipy import stats

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = Path("results/reanalysis")

import re
pats = dict(re.findall(r'"(fr_[A-Za-z0-9_]+)"\s*,\s*"([^"]+)"', open("cpp/frag_tables.h").read()))
size = {}
for n, p in pats.items():
    q = Chem.MolFromSmarts(p)
    size[n] = q.GetNumAtoms() if q is not None else None

names = list(molhume.feature_names(fingerprint=False, columns="full"))
smis = json.load(open("data/exactness_corpus.json"))["smiles"][:N]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
fp, X, _ = molhume.featurize_all_from_mols(mols, optional=("AvgIpc",))
fp = fp.astype(np.float32)
print(f"  {len(mols)} molecules, ECFP {fp.shape}")

import xgboost as xgb
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
perm = rng.permutation(len(mols))
tr, te = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

rows = []
for i, c in enumerate(names):
    if not c.startswith("fr_"):
        continue
    y = np.nan_to_num(X[:, i].astype(np.float64))
    occ = (y != 0).astype(int)
    fires = float(occ.mean())
    if occ[tr].sum() < 10 or occ[te].sum() < 5:
        rows.append(dict(column=c, fires=fires, n_atoms=size.get(c), auroc=None,
                         count_rho=None, note="too few positives to score"))
        continue
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1,
                            verbosity=0, tree_method="hist", eval_metric="logloss")
    clf.fit(fp[tr], occ[tr])
    auc = float(roc_auc_score(occ[te], clf.predict_proba(fp[te])[:, 1]))
    # count recovery, only where the pattern occurs
    pos_tr = tr[occ[tr] == 1]; pos_te = te[occ[te] == 1]
    rho = None
    if len(pos_tr) >= 30 and len(pos_te) >= 15 and y[pos_te].std() > 0:
        reg = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1,
                               verbosity=0, tree_method="hist")
        reg.fit(fp[pos_tr], y[pos_tr])
        rho = float(stats.spearmanr(y[pos_te], reg.predict(fp[pos_te])).statistic)
    rows.append(dict(column=c, fires=fires, n_atoms=size.get(c), auroc=auc, count_rho=rho))

json.dump(rows, open(OUT / "frag_vs_ecfp2.json", "w"), indent=1)
scored = [r for r in rows if r["auroc"] is not None]
print(f"\n  {len(scored)} of {len([r for r in rows])} fr_ columns scored\n")
print(f"  {'column':24s} {'atoms':>5s} {'fires':>8s} {'AUROC':>7s} {'count rho':>10s}")
for r in sorted(scored, key=lambda r: r["auroc"]):
    cr = "-" if r["count_rho"] is None else f"{r['count_rho']:.3f}"
    print(f"  {r['column']:24s} {str(r['n_atoms']):>5s} {r['fires']:8.3%} {r['auroc']:7.3f} {cr:>10s}")

a = np.array([r["auroc"] for r in scored])
n = np.array([r["n_atoms"] if r["n_atoms"] else np.nan for r in scored], float)
ok = np.isfinite(n)
print(f"\n  detection AUROC: median {np.median(a):.3f}, "
      f"{(a >= 0.99).sum()} of {len(a)} at >= 0.99, {(a < 0.9).sum()} below 0.90")
print(f"  correlation between AUROC and SMARTS pattern size: "
      f"rho = {stats.spearmanr(n[ok], a[ok]).statistic:+.3f} "
      f"(p = {stats.spearmanr(n[ok], a[ok]).pvalue:.4f})")
cr = np.array([r["count_rho"] for r in scored if r["count_rho"] is not None])
print(f"  count recovery where the pattern occurs: median rho {np.median(cr):.3f}, "
      f"{(cr < 0.5).sum()} of {len(cr)} below 0.5")
