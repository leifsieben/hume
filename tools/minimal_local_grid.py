"""hume vs hume_minimal on the whole grid, locally -- no AWS, no neural weights.

The AWS boxes exist because most arms need learned embeddings. This comparison does not: it is
HUME descriptors + ECFP + y + the stored scaffold folds, all computable on a laptop at
~285 us/molecule. Caching the feature matrices makes any future column subset a slice rather
than a re-run, which is the gap that made the first hume_minimal evaluation an EC2 job.

Writes results/reanalysis/features/<dataset>.npz for anything not already cached, then scores
hume against the current molhume.minimal_columns() on identical folds.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, "/Users/lsieben/VSCode/ChemPFN")
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger  # noqa: E402
import molhume  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
OUT = Path("results/reanalysis/features"); OUT.mkdir(parents=True, exist_ok=True)
if "CHEMPFN_DATA_ROOT" not in os.environ:
    sys.exit("CHEMPFN_DATA_ROOT is not set")
import bench_downstream as BD  # noqa: E402
from chempfn.eval.splits import scaffold_folds  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

TASKS = {"physchem": ["aqsoldb","esol","lipophilicity","pb_logd","pb_water_sol","photoswitch"],
 "adme": ["pb_hum_mic_cl","pb_mou_mic_cl","pb_rat_mic_cl","pb_ppb","vdss_lombardo","ld50_zhu",
          "cycpept_pampa","pb_cyp2c9","pb_cyp2d6","pb_cyp3a4"],
 "classif": ["ames","pb_ames","cyp2d6_inh","bioavail","hia","pb_bbb","bace","hiv","herg",
             "wong_hepg2","wong_imr90","wong_hskmc","wong_saureus"],
 "quantum": ["qm8","qm9","qm9_gap","qmugs_gap"]}
panel = {d: t for t, ds in TASKS.items() for d in ds}
names = list(molhume.feature_names(fingerprint=False, columns="full"))

def cache(ds):
    dest = OUT / f"{ds}.npz"
    if dest.exists():
        return dest
    d = BD.load_ds(ds)
    smis, y = list(d["smiles"]), np.asarray(d["y"], dtype=np.float64)
    if len(smis) > 50_000:
        take = np.sort(np.random.default_rng(0).choice(len(smis), 50_000, replace=False))
        smis = [smis[i] for i in take]; y = y[take]
    folds = scaffold_folds(smis, k=5, seed=0)
    fo = np.full(len(smis), -1, np.int8)
    for k, ix in enumerate(folds): fo[list(ix)] = k
    mols = [Chem.MolFromSmiles(s) for s in smis]
    keep = [i for i, m in enumerate(mols) if m is not None]
    fp, X, _ = molhume.featurize_all_from_mols([mols[i] for i in keep], optional=("AvgIpc",))
    Xf = np.full((len(smis), X.shape[1]), np.nan, np.float32); Xf[keep] = X.astype(np.float32)
    fpf = np.zeros((len(smis), fp.shape[1]), np.uint8); fpf[keep] = fp
    np.savez_compressed(dest, X=Xf, fp=fpf, y=y, folds=fo,
                        column_names=np.array(names), task=np.array(d["task"]),
                        metric=np.array(d.get("metric", "")))
    print(f"    cached {ds:16s} n={len(smis):6d}", flush=True)
    return dest

def score(ds, cols):
    z = np.load(OUT / f"{ds}.npz", allow_pickle=False)
    X, fp, y, folds = z["X"], z["fp"], z["y"], z["folds"]
    X = np.where(np.isfinite(X), X, np.nan)
    # The first six cache files were written before `task` was a top-level key -- they carry a
    # JSON `meta` blob instead. Read either rather than re-featurizing 40,000 molecules to
    # change a field name.
    if "task" in z:
        task = str(z["task"])
    else:
        task = json.loads(str(z["meta"]))["task"]
    sel = [i for i, c in enumerate(names) if c in cols]
    F = np.hstack([X[:, sel], fp]).astype(np.float32)
    out = []
    for k in range(5):
        te = folds == k; tr = ~te
        if task == "binary":
            m = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=0, n_jobs=-1,
                                  verbosity=0, colsample_bynode=0.3, tree_method="hist",
                                  eval_metric="logloss")
            m.fit(F[tr], y[tr]); out.append(float(roc_auc_score(y[te], m.predict_proba(F[te])[:, 1])))
        else:
            m = xgb.XGBRegressor(n_estimators=300, max_depth=6, random_state=0, n_jobs=-1,
                                 verbosity=0, colsample_bynode=0.3, tree_method="hist")
            m.fit(F[tr], y[tr]); out.append(float(np.sqrt(np.mean((y[te] - m.predict(F[te])) ** 2))))
    return np.array(out), task

FULL = set(names); MIN = set(molhume.minimal_columns())
print(f"  hume {len(FULL)} columns vs hume_minimal {len(MIN)}\n  caching:")
todo = [d for t in TASKS.values() for d in t]
for ds in todo:
    try: cache(ds)
    except Exception as e: print(f"    ! {ds}: {type(e).__name__}: {e}", flush=True)
print("\n  scoring:")
rows = []
for ds in todo:
    if not (OUT / f"{ds}.npz").exists(): continue
    try:
        a, task = score(ds, FULL); b, _ = score(ds, MIN)
    except Exception as e:
        print(f"    ! {ds}: {e}", flush=True); continue
    higher = task == "binary"
    rel = (a.mean() - b.mean()) / abs(a.mean()) if higher else (b.mean() - a.mean()) / abs(a.mean())
    rows.append(dict(dataset=ds, panel=panel[ds], task=task, hume=a.mean(),
                     minimal=b.mean(), rel=rel, sd=a.std(ddof=1) / abs(a.mean()),
                     # PER-FOLD too, so these can be written back into the downstream grid as
                     # hume_minimal records. The grid merges on (dataset, arm, FOLD), so a mean
                     # alone cannot replace the AWS records it supersedes.
                     hume_folds=a.tolist(), minimal_folds=b.tolist(),
                     metric="auroc" if task == "binary" else "rmse"))
    print(f"    {ds:16s} {panel[ds]:9s} {a.mean():9.4f} {b.mean():9.4f} {rel*100:+7.2f}%", flush=True)
json.dump(rows, open("results/reanalysis/minimal_local_grid.json", "w"), indent=1)
print()
import collections
by = collections.defaultdict(list)
for r in rows: by[r["panel"]].append(r["rel"])
for p, v in by.items():
    v = np.array(v)
    print(f"  {p:10s} n={len(v):2d}  mean {v.mean()*100:+6.2f}%  worst {v.max()*100:+6.2f}%")
allv = np.array([r["rel"] for r in rows])
print(f"  OVERALL   n={len(allv):2d}  mean {allv.mean()*100:+6.2f}%  median {np.median(allv)*100:+6.2f}%")
print(f"  above own fold SD: {sum(1 for r in rows if abs(r['rel'])>r['sd'])} of {len(rows)}")
