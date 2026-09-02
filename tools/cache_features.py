"""Compute HUME features once per dataset and PERSIST them, with their scaffold folds.

    CHEMPFN_DATA_ROOT=/Users/lsieben/chempfn-data \
      .venv/bin/python tools/cache_features.py esol lipophilicity ...

WHY THIS EXISTS. Every AWS box so far recomputed features, scored them, shipped the scores and
threw the features away -- `bench_downstream.py` keeps them in `_BLOCK_CACHE`, which it clears
at the top of every dataset. That makes any reanalysis ("what if the head were an MLP?", "what
if the trees were deeper?") a full re-run on EC2 instead of a minute on a laptop, and it means
the numbers in the paper cannot be interrogated after the fact. Persisting the matrices makes
the head a free variable.

WHAT IS STORED, per dataset, in one .npz:
    X        (n, 1269) float32   the full HUME descriptor block, standardize="none"
    fp       (n, 2048) uint8     the ECFP the arms concatenate
    y        (n,)      float64
    folds    (n,)      int8      which of the 5 Murcko scaffold folds each molecule is in
    smiles   (n,)      str
plus the metric, task, the rdkit and mol-hume versions, and the minimal-spec column mask.

THE FOLDS ARE STORED, NOT RECOMPUTED, and that is the point. A reanalysis that re-derives its
own split is not comparable with the grid it is being compared against; storing the fold vector
makes "same molecules, same folds, different head" literally true rather than approximately.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/lsieben/VSCode/ChemPFN")
sys.path.insert(0, ".")

from rdkit import Chem, RDLogger  # noqa: E402

import molhume  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")
OUT = Path("results/reanalysis/features")
OUT.mkdir(parents=True, exist_ok=True)

if "CHEMPFN_DATA_ROOT" not in os.environ:
    sys.exit("CHEMPFN_DATA_ROOT is not set; point it at the lake, e.g. /Users/lsieben/chempfn-data")

import bench_downstream as BD  # noqa: E402
from chempfn.eval.splits import scaffold_folds  # noqa: E402

names = molhume.feature_names(fingerprint=False)
minimal = set(molhume.minimal_columns())
mask = np.array([c in minimal for c in names], dtype=bool)

for ds in sys.argv[1:]:
    dest = OUT / f"{ds}.npz"
    if dest.exists():
        print(f"  {ds:16s} already cached")
        continue
    d = BD.load_ds(ds)
    smis, y = list(d["smiles"]), np.asarray(d["y"], dtype=np.float64)
    if len(smis) > 50_000:                      # the cap the grid uses; keeps this comparable
        take = np.sort(np.random.default_rng(0).choice(len(smis), 50_000, replace=False))
        smis = [smis[i] for i in take]; y = y[take]
    folds = scaffold_folds(smis, k=5, seed=0)
    fold_of = np.full(len(smis), -1, dtype=np.int8)
    for k, idx in enumerate(folds):
        fold_of[list(idx)] = k
    assert (fold_of >= 0).all(), f"{ds}: scaffold_folds did not cover every molecule"

    mols = [Chem.MolFromSmiles(s) for s in smis]
    keep = [i for i, m in enumerate(mols) if m is not None]
    fp, X, _ = molhume.featurize_all_from_mols([mols[i] for i in keep], optional=("AvgIpc",))
    Xf = np.full((len(smis), X.shape[1]), np.nan, np.float32)
    fpf = np.zeros((len(smis), fp.shape[1]), np.uint8)
    Xf[keep] = X.astype(np.float32); fpf[keep] = fp

    np.savez_compressed(
        dest, X=Xf, fp=fpf, y=y, folds=fold_of, smiles=np.array(smis),
        minimal_mask=mask, column_names=np.array(names),
        meta=np.array(json.dumps({
            "dataset": ds, "task": d["task"], "metric": d.get("metric", ""),
            "n": len(smis), "n_unparsed": len(smis) - len(keep),
            "rdkit": Chem.rdBase.rdkitVersion, "molhume": "0.2.0",
            "standardize": "none", "folds": "chempfn scaffold_folds k=5 seed=0",
            "cap": "random 50,000 subsample before splitting when n > 50,000, seed 0"})))
    print(f"  {ds:16s} n={len(smis):6d}  X{Xf.shape}  -> {dest.name} "
          f"({dest.stat().st_size/1e6:.1f} MB)")
