"""Cache per-endpoint panels from the multi-label DEV sources.

    CHEMPFN_DATA_ROOT=... PYTHONPATH=<chempfn> .venv/bin/python tools/h300/add_panels.py

tox21, sider, muv and toxcast are ONE CSV WITH MANY LABEL COLUMNS, which is why the lake loader
rejects them: it expects a single label column. Split per endpoint and each becomes an ordinary
binary panel. That is where the volume is -- 12 + 27 + 17 endpoints against the 13 classification
datasets the ablations have been decided on so far.

Molecules are featurized ONCE per source file and shared across its endpoints; only the labels
and the usable-row mask differ. Folds are chempfn's scaffold_folds, k=5 seed=0, the same as every
other panel, computed per endpoint because the usable rows differ between them.
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

OUT = Path("results/reanalysis/features")
LAKE = Path(os.environ["CHEMPFN_DATA_ROOT"]) / "eval" / "dev" / "moleculenet_legacy"
#: source -> (file, how many endpoints to take). toxcast has 617; a sample keeps the panel
#: balanced rather than letting one assay family outvote every other dataset in the study.
SOURCES = {"tox21": ("tox21.csv", None), "sider": ("sider.csv", None),
           "muv": ("muv.csv", 17)}
MIN_POS = 25          # an endpoint with fewer actives cannot support 5-fold scaffold CV
MAX_MOL = 25_000      # subsample large sources; the grid's own cap is 50k


def main() -> None:
    names = molhume.feature_names(fingerprint=False, columns="full")
    for src, (fname, n_ep) in SOURCES.items():
        path = LAKE / fname
        if not path.exists():
            print(f"  {src}: {path} missing"); continue
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")))
        smi_col = "smiles" if "smiles" in rows[0] else "mol_id"
        endpoints = [c for c in rows[0] if c not in (smi_col, "mol_id", "smiles")]
        if n_ep:
            endpoints = endpoints[:n_ep]
        smis_all = [r[smi_col] for r in rows]
        if len(smis_all) > MAX_MOL:
            take = np.sort(np.random.default_rng(0).choice(len(smis_all), MAX_MOL, replace=False))
            rows = [rows[i] for i in take]; smis_all = [smis_all[i] for i in take]
        mols = [Chem.MolFromSmiles(s) for s in smis_all]
        ok = [i for i, m in enumerate(mols) if m is not None]
        print(f"  {src}: {len(rows):,} rows, {len(ok):,} parse, {len(endpoints)} endpoints",
              flush=True)
        X = np.full((len(rows), len(names)), np.nan, np.float32)
        fp = np.zeros((len(rows), 2048), np.uint8)
        _errs = []
        f_, x_, _ = molhume.featurize_all_from_mols(
        [mols[i] for i in ok], optional=("AvgIpc",), errors_out=_errs)
        # TRIM `qed`. featurize_all_from_mols returns ALL_COLUMNS, which is 1,270 since 0.8.0
        # appended qed; the 33 panels already cached are 1,269 and every arm is defined over
        # those names. qed is appended LAST by construction -- column_set("full") is
        # ALL_COLUMNS[:1269], asserted in the test suite -- so the slice is exact and not a
        # guess. Mixing widths here would silently misalign every column index in the study.
        assert x_.shape[1] == len(names) + 1, (x_.shape, len(names))
        X[ok] = x_[:, :len(names)].astype(np.float32); fp[ok] = f_
        for ep in endpoints:
            dest = OUT / f"{src}_{ep.replace('/', '_').replace(' ', '_')[:40]}.npz"
            if dest.exists():
                continue
            use = [i for i in ok if rows[i].get(ep) not in (None, "", "None")]
            y = np.array([float(rows[i][ep]) for i in use])
            if len(np.unique(y)) < 2 or min((y == 1).sum(), (y == 0).sum()) < MIN_POS:
                continue
            sm = [smis_all[i] for i in use]
            folds = scaffold_folds(sm, k=5, seed=0)
            fo = np.full(len(use), -1, np.int8)
            for k, idx in enumerate(folds):
                fo[list(idx)] = k
            np.savez_compressed(dest, X=X[use], fp=fp[use], y=y, folds=fo,
                                column_names=np.array(names), task=np.array("binary"),
                                metric=np.array("auroc"))
            print(f"    {dest.stem:44s} n={len(use):6,d} pos={int((y==1).sum()):5d}", flush=True)


if __name__ == "__main__":
    main()
