"""Is the fr_* drop conditional on the RADIUS or on the DECODER? A 2x2 to separate them.

ChemPFN measured detection of the dropped fr_* flags from an r=2 fingerprint with LOGISTIC
REGRESSION and got median AUROC 0.9929, min 0.786. We measured from r=3 with XGBOOST and got
median 1.000, none below 0.90. Two variables moved at once, so neither result attributes the
difference.

It matters which one it is:
  * if RADIUS, the drop should be gated on a declared fingerprint radius;
  * if DECODER, no fingerprint configuration saves it, and the flags must be restored outright
    for any consumer that decodes linearly -- which is what a PFN doing in-context learning is,
    by their own M18 measurement (it lands BELOW the linear ceiling).

Same protocol as theirs: 7,500 molecules, fit on half, score on the disjoint half, detection
AUROC for "is this flag non-zero".
"""
import json
import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
N = 7500
names = list(molhume.feature_names(fingerprint=False))
smis = json.load(open("data/exactness_corpus.json"))["smiles"][:N]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
X = molhume.featurize(mols, standardize="none", fingerprint=False)
frag = [(i, c) for i, c in enumerate(names) if c.startswith("fr_")]
print(f"  {len(mols)} molecules, {len(frag)} fr_* flags\n")

rng = np.random.default_rng(0)
perm = rng.permutation(len(mols))
tr, te = perm[: len(perm) // 2], perm[len(perm) // 2:]

FPS = {"r=2 / 2048": rfg.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
       "r=3 / 2048": rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)}
DEC = {"logistic": lambda: LogisticRegression(max_iter=2000, C=1.0),
       "xgboost":  lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=0,
                                             n_jobs=-1, verbosity=0, tree_method="hist",
                                             eval_metric="logloss")}
out = {}
for fpn, gen in FPS.items():
    fp = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    for dn, mk in DEC.items():
        aucs, per = [], {}
        for i, c in frag:
            y = (np.nan_to_num(X[:, i]) != 0).astype(int)
            if y[tr].sum() < 10 or y[te].sum() < 5 or y[te].mean() == 1.0:
                continue
            m = mk(); m.fit(fp[tr], y[tr])
            a = float(roc_auc_score(y[te], m.predict_proba(fp[te])[:, 1]))
            aucs.append(a); per[c] = a
        a = np.array(aucs)
        out[(fpn, dn)] = per
        print(f"  {fpn:12s} {dn:9s}  n={len(a):3d}  median {np.median(a):.4f}  "
              f"min {a.min():.4f}  <0.99 {int((a<0.99).sum()):3d}  <0.95 {int((a<0.95).sum()):3d}")
json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()},
          open("results/reanalysis/frag_decoder_matrix.json", "w"), indent=1)

THEIRS = ["fr_quatN","fr_Ndealkylation1","fr_Ar_COO","fr_urea","fr_hdrzine","fr_piperdine",
          "fr_imide","fr_NH2","fr_Ndealkylation2","fr_pyridine","fr_COO2","fr_amidine",
          "fr_allylic_oxid","fr_sulfone","fr_ketone"]
print(f"\n  the 15 they ask to restore, on OUR corpus:\n")
print(f"  {'flag':22s} {'r2/logit':>9s} {'r2/xgb':>8s} {'r3/logit':>9s} {'r3/xgb':>8s}")
for c in THEIRS:
    row=[out[k].get(c) for k in (("r=2 / 2048","logistic"),("r=2 / 2048","xgboost"),
                                  ("r=3 / 2048","logistic"),("r=3 / 2048","xgboost"))]
    print(f"  {c:22s} " + " ".join(f"{v:8.4f}" if v is not None else f"{'-':>8s}" for v in row))
