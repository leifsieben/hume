"""Cache panels that have influenced NO decision, for a single final evaluation.

    CHEMPFN_DATA_ROOT=... PYTHONPATH=<chempfn> .venv/bin/python tools/h300/heldout.py

Every number in docs/selection/HUME_300_METHOD.md so far was measured on the panels the selection was made on.
The leave-one-dataset-out in the earlier runs protected the within-family RANKING only; the
decisions that matter -- which sweeps to cut, proportional against effective-rank allocation,
which arm to ship -- all saw every panel. A spec chosen and reported on one set of tasks has an
unknown optimism, and the only way to measure it is a set that was never consulted.

Held out here, none of which appear in any earlier run:
  * the ten chempfn LOCKED sets other than herg and wong_* (those were in the original 33 and
    are therefore already spent);
  * toxcast endpoints, which are DEV but have never been loaded by this study.

`herg` and the four `wong_*` sets are DELIBERATELY NOT re-used as held-out. They were in the
33-panel study that made the fr_* decision, the sweep decision and the combination decision.
Calling them held-out now would be the same error in a new coat.
"""
from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
from rdkit import Chem, RDLogger                                   # noqa: E402

RDLogger.DisableLog("rdApp.*")
import molhume                                                     # noqa: E402
from chempfn.eval.splits import scaffold_folds                     # noqa: E402

sys.path.insert(0, ".")
import bench_downstream as BD                                      # noqa: E402

OUT = Path("results/reanalysis/heldout")
OUT.mkdir(parents=True, exist_ok=True)
SIMPLE = ["adme_bbb", "adme_caco2", "adme_clear_hep", "adme_clear_micro", "adme_half_life",
          "adme_ppbr", "cbs", "freesolv", "qm7", "moleculeace"]
N_TOXCAST = 40
MIN_POS = 25
MAX_MOL = 25_000


def save(dest, smis, y, task):
    if dest.exists():
        return
    mols = [Chem.MolFromSmiles(s) for s in smis]
    ok = [i for i, m in enumerate(mols) if m is not None]
    if len(ok) < 100:
        print(f"    {dest.stem}: only {len(ok)} parse, skipped"); return
    folds = scaffold_folds([smis[i] for i in ok], k=5, seed=0)
    fo = np.full(len(ok), -1, np.int8)
    for k, idx in enumerate(folds):
        fo[list(idx)] = k
    names = molhume.feature_names(fingerprint=False, columns="full")
    # ISOLATE, DO NOT RAISE. A molecule that is a bare hydrogen atom throws out of the ETA family
    # (mordred's ETA_epsilon_5 indexes GetNeighbors()[0]); featurize_all_from_mols raises by
    # default, which loses the whole dataset for one molecule. With a list it becomes a NaN row.
    _errs = []
    fp, X, _ = molhume.featurize_all_from_mols(
        [mols[i] for i in ok], optional=("AvgIpc",), errors_out=_errs)
    assert X.shape[1] == len(names) + 1                       # trim the appended qed
    np.savez_compressed(dest, X=X[:, :len(names)].astype(np.float32), fp=fp,
                        y=np.asarray(y, float)[ok], folds=fo,
                        column_names=np.array(names), task=np.array(task),
                        metric=np.array("auroc" if task == "binary" else "rmse"))
    print(f"    {dest.stem:44s} n={len(ok):6,d} task={task}", flush=True)


def main() -> None:
    for name in SIMPLE:
        try:
            d = BD.load_ds(name)
        except Exception as e:
            print(f"    {name:20s} FAILED {type(e).__name__}: {str(e)[:80]}"); continue
        smis, y = list(d["smiles"]), np.asarray(d["y"], float)
        task = "binary" if d["task"] in ("binary", "classif", "classification") else "regression"
        if task == "binary" and min((y == 1).sum(), (y == 0).sum()) < MIN_POS:
            print(f"    {name}: too few of one class"); continue
        if len(smis) > MAX_MOL:
            take = np.sort(np.random.default_rng(0).choice(len(smis), MAX_MOL, replace=False))
            smis = [smis[i] for i in take]; y = y[take]
        save(OUT / f"{name}.npz", smis, y, task)

    path = Path(os.environ["CHEMPFN_DATA_ROOT"]) / "eval/dev/moleculenet_legacy/toxcast.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")))
    eps = [c for c in rows[0] if c not in ("smiles", "mol_id")]
    # Evenly spaced through the 617 rather than the first 40: the file is ordered by assay
    # family, so a prefix would be one lab's panel repeated forty times.
    eps = [eps[i] for i in np.linspace(0, len(eps) - 1, N_TOXCAST).astype(int)]
    smis_all = [r["smiles"] for r in rows]
    mols = [Chem.MolFromSmiles(s) for s in smis_all]
    ok_all = [i for i, m in enumerate(mols) if m is not None]
    names = molhume.feature_names(fingerprint=False, columns="full")
    _errs = []
    fp_a, x_a, _ = molhume.featurize_all_from_mols(
        [mols[i] for i in ok_all], optional=("AvgIpc",), errors_out=_errs)
    X = np.full((len(rows), len(names)), np.nan, np.float32)
    FP = np.zeros((len(rows), 2048), np.uint8)
    X[ok_all] = x_a[:, :len(names)].astype(np.float32); FP[ok_all] = fp_a
    print(f"  toxcast: {len(rows):,} rows, {len(eps)} endpoints sampled of {len(rows[0])-1}")
    for ep in eps:
        use = [i for i in ok_all if rows[i].get(ep) not in (None, "", "None")]
        if len(use) < 200:
            continue
        y = np.array([float(rows[i][ep]) for i in use])
        if min((y == 1).sum(), (y == 0).sum()) < MIN_POS:
            continue
        dest = OUT / f"toxcast_{ep.replace('/', '_')[:40]}.npz"
        if dest.exists():
            continue
        folds = scaffold_folds([smis_all[i] for i in use], k=5, seed=0)
        fo = np.full(len(use), -1, np.int8)
        for k, idx in enumerate(folds):
            fo[list(idx)] = k
        np.savez_compressed(dest, X=X[use], fp=FP[use], y=y, folds=fo,
                            column_names=np.array(names), task=np.array("binary"),
                            metric=np.array("auroc"))
        print(f"    {dest.stem:44s} n={len(use):6,d} pos={int((y==1).sum()):5d}", flush=True)


if __name__ == "__main__":
    main()
