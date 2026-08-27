"""Why do descriptors hurt on potency but help on QM/physchem? — the column-sampling test.

HYPOTHESIS (owner's). Activity cliffs are pairs of structurally similar molecules with very
different activity. Descriptors are *smooth* functions of structure, so a cliff pair has nearly
identical descriptors while the label jumps: the descriptors cannot resolve the cliff, but they
CAN be split on for the bulk trend. XGBoost's ``colsample_by*`` samples columns UNIFORMLY, and a
dense descriptor is informative on nearly every molecule while a given ECFP bit is on in ~2.6% of
them — so descriptors win the sampling lottery on *availability*, not on merit, and crowd out the
sparse bits that could resolve the cliff.

THE KNOB. XGBoost >= 3.x supports ``DMatrix.set_info(feature_weights=w)`` — weighted column
sampling. One scalar ``w``: every ECFP column gets weight ``w``, every descriptor gets weight 1.
``w = 1`` is uniform sampling (current behaviour); ``w > 1`` up-weights fingerprints.

WHY colsample_bynode=0.3 AND NOT THE PRODUCTION colsample_bytree=0.8. ``feature_weights`` only
does anything when a ``colsample_by*`` is < 1, and its leverage depends on how *aggressive* the
sampling is. At the production 0.8, 2150 of 2687 columns are drawn at every sampling event, so
almost every descriptor is available no matter its weight: measured on CHEMBL204_Ki, sweeping
w over 0.1..100 moved the ECFP share of splits only 0.244 -> 0.392. The rate was therefore
chosen a priori, on *dynamic range of the split share alone and never on test error*: 0.3 is the
largest rate tested that gives > 5x range (0.155 -> 0.795). It is then FROZEN across every arm,
so ``w`` remains the only variable in the sweep. The production configuration is run as a
separate labelled reference arm (``baseline_bytree0.8``) so we know where it sits.

DATA. ``data/surrogate/bench.npz``. X columns 0:2048 are log1p ECFP-2048 counts (radius 2,
includeChirality), 2048:2687 are the 639 CORE descriptors — see ``assemble.py`` line 89.

CLIFF LABELS. MoleculeACE ships a per-molecule ``cliff_mol`` flag in
``/Users/lsieben/chempfn-data/eval/locked/moleculeace/*.csv``. All 48,714 benchmark rows of the
30 MoleculeACE datasets match those CSVs on exact SMILES with labels agreeing to < 1e-4, so the
flag transfers by SMILES lookup. It is a property of the molecule within its dataset (it is set
on both sides of MoleculeACE's own split), so it is valid under our scaffold folds.

SPLITS. Scaffold folds only — random splits put near-identical analogs on both sides and would
inflate every arm. ``scaffold_folds(smiles, k=5, seed=0)`` with XGBoost ``seed=0``, which is the
protocol every other experiment in this repo uses. Every arm sees the identical folds, so all
arm comparisons are paired; spread is reported as the std over the 5 folds, and the aggregate
comparisons pool 30 datasets x 5 folds = 150 paired observations per arm.

PRE-REGISTERED PREDICTION. On MoleculeACE the optimal w is > 1 and the gain is concentrated in
the cliff subset with non-cliff flat or slightly worse; on MoleculeNet physchem the optimal w is
<= 1. Same optimum on both suites, or no cliff/non-cliff difference, refutes the mechanism.

Usage:  .venv/bin/python cliff_mechanism.py [--workers N]  ->  results/cliff_mechanism.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
BENCH = ROOT / "data" / "surrogate" / "bench.npz"
ACE = pathlib.Path("/Users/lsieben/chempfn-data/eval/locked/moleculeace")
OUT = ROOT / "results" / "cliff_mechanism.json"

N_ECFP = 2048          # X[:, :2048] ECFP, X[:, 2048:] descriptors — verified in assemble.py:89
N_REPEATS = 1          # repeat r uses fold seed r and XGBoost seed r
K_FOLDS = 5            # 5 folds, seed 0 — the protocol every other experiment in this repo uses
COLSAMPLE_BYNODE = 0.3  # frozen across every arm; chosen on split-share range, not on error

W_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

# Untuned baseline hyperparameters, copied from vendor/chemtfm/models/xgb.py. Nothing here is
# tuned during the sweep; only `feature_weights` changes between w-arms.
BASE_PARAMS = {
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "min_child_weight": 1,
    "tree_method": "hist",
    "verbosity": 0,
    "objective": "reg:squarederror",
}
N_ROUNDS = 300


def load_cliff_flags(smiles, offsets, name_of, suite_of):
    """Per-row cliff flag: 1/0 for MoleculeACE rows, -1 for rows with no annotation."""
    flags = np.full(len(smiles), -1, np.int8)
    found = {}
    for j, name in enumerate(name_of):
        if suite_of[j] != "moleculeace":
            continue
        path = ACE / f"{name}.csv"
        if not path.exists():
            found[name] = "MISSING"
            continue
        lut = {r["smiles"]: int(r["cliff_mol"]) for r in csv.DictReader(path.open())}
        s = slice(offsets[j], offsets[j + 1])
        block = np.array([lut.get(x, -1) for x in smiles[s]], np.int8)
        flags[s] = block
        found[name] = {"n": int(block.size), "n_cliff": int((block == 1).sum()),
                       "n_unmatched": int((block == -1).sum())}
    return flags, found


def arms():
    """(name, kind, w) for every arm. `kind` selects the feature block and sampling scheme."""
    out = [(f"w={w:g}", "full", w) for w in W_GRID]
    out += [("desc_only", "desc", None), ("fp_only", "fp", None),
            ("baseline_bytree0.8", "baseline", None)]
    return out


def run_dataset(job):
    """All arms x all (repeat, fold) for one dataset. Returns per-fold records."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import xgboost as xgb

    import _vendor  # noqa: F401  — puts vendor/chemtfm on sys.path
    from chemtfm.bench.splits import scaffold_folds, train_test

    j, name, suite, lo, hi, nthread = job
    d = np.load(BENCH, allow_pickle=True)
    X = np.asarray(d["X"][lo:hi], np.float32)
    y = np.asarray(d["y"][lo:hi], np.float64)
    smi = list(d["smiles"][lo:hi])
    cliff = load_cliff_flags(d["smiles"], d["offsets"], d["name_of"], d["suite_of"])[0][lo:hi]

    Xfp, Xde = X[:, :N_ECFP], X[:, N_ECFP:]
    recs = []
    for rep in range(N_REPEATS):
        folds = scaffold_folds(smi, k=K_FOLDS, seed=rep)
        for fi in range(K_FOLDS):
            tr, te = train_test(folds, fi)
            if float(np.std(y[tr])) == 0.0:
                continue
            for aname, kind, w in arms():
                p = dict(BASE_PARAMS, nthread=nthread, seed=rep)
                if kind == "baseline":
                    mat, fw = X, None
                    p["colsample_bytree"] = 0.8
                else:
                    p["colsample_bytree"] = 1.0
                    p["colsample_bynode"] = COLSAMPLE_BYNODE
                    if kind == "desc":
                        mat, fw = Xde, None
                    elif kind == "fp":
                        mat, fw = Xfp, None
                    else:
                        mat = X
                        fw = np.ones(X.shape[1], np.float32)
                        fw[:N_ECFP] = w

                dtr = xgb.DMatrix(mat[tr], label=y[tr])
                if fw is not None:
                    dtr.set_info(feature_weights=fw)
                bst = xgb.train(p, dtr, num_boost_round=N_ROUNDS)
                pred = bst.predict(xgb.DMatrix(mat[te]))
                err2 = (pred - y[te]) ** 2

                # ECFP share of tree splits — the diagnostic that `w` actually moved the head.
                if kind == "full":
                    sc = bst.get_score(importance_type="weight")
                    nfp = sum(v for k, v in sc.items() if int(k[1:]) < N_ECFP)
                    ntot = sum(sc.values())
                    share = float(nfp / ntot) if ntot else float("nan")
                else:
                    share = float("nan")

                c = cliff[te]
                recs.append({
                    "dataset": name, "suite": suite, "arm": aname, "w": w,
                    "repeat": rep, "fold": fi,
                    "n": int(te.size), "sse": float(err2.sum()),
                    "n_cliff": int((c == 1).sum()),
                    "sse_cliff": float(err2[c == 1].sum()),
                    "n_noncliff": int((c == 0).sum()),
                    "sse_noncliff": float(err2[c == 0].sum()),
                    "ecfp_split_share": share,
                })
    return recs


def rmse(sse, n):
    return float(np.sqrt(sse / n)) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--nthread", type=int, default=2)
    a = ap.parse_args()

    d = np.load(BENCH, allow_pickle=True)
    smiles, offsets = d["smiles"], d["offsets"]
    name_of, suite_of = d["name_of"], d["suite_of"]
    _, cliff_report = load_cliff_flags(smiles, offsets, name_of, suite_of)

    jobs = [(j, str(name_of[j]), str(suite_of[j]), int(offsets[j]), int(offsets[j + 1]), a.nthread)
            for j in range(len(name_of))]
    # Largest datasets first so the long poles start early.
    jobs.sort(key=lambda t: t[4] - t[3], reverse=True)

    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, recs in enumerate(ex.map(run_dataset, jobs), 1):
            records.extend(recs)
            print(f"[{k}/{len(jobs)}] {recs[0]['dataset']:<18} "
                  f"{len(recs)} fits  ({time.time() - t0:.0f}s elapsed)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "config": {
            "n_ecfp": N_ECFP, "n_desc": int(d["X"].shape[1] - N_ECFP),
            "colsample_bynode": COLSAMPLE_BYNODE, "w_grid": W_GRID,
            "n_repeats": N_REPEATS, "k_folds": K_FOLDS,
            "fold_seeds": list(range(N_REPEATS)), "xgb_seeds": list(range(N_REPEATS)),
            "params": BASE_PARAMS, "num_boost_round": N_ROUNDS,
            "split": "scaffold (chemtfm.bench.splits.scaffold_folds)",
            "cliff_source": str(ACE),
        },
        "cliff_annotations": cliff_report,
        "records": records,
    }, indent=1))
    print(f"wrote {OUT}  ({len(records)} records, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
