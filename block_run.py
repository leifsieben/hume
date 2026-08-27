"""Phase 0: does each new descriptor block earn its cost?

Compute-side only. No surrogate, no LOCKED registry -- every arm computes its features
exactly, so this answers "is the block worth computing" independently of anything the
prediction model does later.

Arms, all carrying ECFP (benchmarking descriptors against a fingerprint-free baseline is the
error that produced a wrong "Mordred helps" conclusion earlier in this project):

    ecfp+core                     baseline, as stored in bench.npz
    ecfp+core+resistance          path multiplicity
    ecfp+core+cycles              exact cycle counts
    ecfp+core+conjugation         pi-system topology
    ecfp+core+stereo              CIP parity relations
    ecfp+core+chi                 connectivity indices (now computed, not predicted)
    ecfp+core+all                 everything together

Each block carries a **negative control** -- a set of datasets where it must do nothing,
because its features are identically zero or near-zero there. A block that helps uniformly,
including on its own negative control, is proxying molecular size and gets cut rather than
celebrated. That test is the point of this script; the mean gain is secondary.

Aggregation is by rank and by per-dataset win rate, never by a mean of raw RMSE across
datasets with incommensurable scales (FreeSolv spans 1.6-1.8, Lipophilicity 0.68, so a raw
mean silently weights FreeSolv ~2.5x).
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"
BLOCKS = ("resistance", "cycles", "conjugation", "stereo", "chi")


def _feat_chunk(args):
    name, smiles = args
    import importlib
    mod = importlib.import_module(name)
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = np.empty((len(smiles), mod.NDIM), np.float32)
    for i, s in enumerate(smiles):
        try:
            out[i] = mod.featurize(Chem.MolFromSmiles(s))
        except Exception:
            out[i] = np.nan
    return name, out


def featurize(smiles, workers: int = 4) -> dict:
    """Compute every block for the benchmark set, cached to disk."""
    feats = {}
    todo = []
    for b in BLOCKS:
        f = OUT / f"bench_{b}.npz"
        if f.exists():
            feats[b] = np.load(f)["R"]
            print(f"  {b}: cached {feats[b].shape}")
        else:
            todo.append(b)
    if todo:
        chunks = [(b, list(smiles)) for b in todo]
        t0 = time.time()
        with Pool(min(workers, len(chunks))) as p:
            for name, R in p.imap_unordered(_feat_chunk, chunks):
                import importlib
                np.savez_compressed(OUT / f"bench_{name}.npz", R=R,
                                    names=np.array(importlib.import_module(name).NAMES))
                feats[name] = R
                print(f"  {name}: computed {R.shape} ({time.time() - t0:.0f}s)", flush=True)
    return feats


def clean(M, ref=None):
    """Impute non-finite cells with the column median and clip to 1/99 percentiles.

    XGBoost tolerates NaN, but an unclipped outlier column silently dominates split search.
    Statistics come from the block itself: this is a compute-side arm, so there is no train
    /test leakage concern about column statistics -- the features are deterministic functions
    of structure, not of labels.
    """
    C = M.astype(np.float64)
    med = np.nanmedian(C, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    C = np.where(np.isfinite(C), C, med)
    lo, hi = np.percentile(C, 1, 0), np.percentile(C, 99, 0)
    return np.clip(C, lo, hi).astype(np.float32)


def _cv_rmse(X, y, smiles):
    import _vendor  # noqa: F401  - puts vendor/chemtfm on sys.path
    from chemtfm.bench import metrics as M
    from chemtfm.bench.datasets import REGRESSION
    from chemtfm.bench.splits import scaffold_folds, train_test
    from chemtfm.models.xgb import XGBModel
    folds = scaffold_folds(smiles, k=5, seed=0)
    out = []
    for i in range(len(folds)):
        tr, te = train_test(folds, i)
        if float(np.std(y[tr])) == 0.0:
            continue
        out.append(M.rmse(y[te], XGBModel(task=REGRESSION).fit(X[tr], y[tr]).predict(X[te])))
    return float(np.mean(out)) if out else np.nan


def covariates(smiles, feats) -> dict:
    """Per-molecule flags used to build each block's negative control."""
    import cycles as C
    import resistance as R
    import stereo as S
    cyc = feats["cycles"]
    red = cyc[:, C.NAMES.index("C_redundancy")]
    return {
        "acyclic": feats["resistance"][:, R.NAMES.index("Cyclicity")] == 0,
        "fused": np.nan_to_num(red) > 1.0,
        "has_stereo": np.abs(feats["stereo"]).sum(1) > 0,
        "conjugated": feats["conjugation"][:, 0] > 0,
    }


def main() -> None:
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    X, smiles = d["X"], list(d["smiles"])
    y, offsets, suite_of, name_of = d["y"], d["offsets"], d["suite_of"], d["name_of"]
    print(f"bench {X.shape} | {len(smiles):,} molecules | {len(name_of)} datasets\n")

    feats = featurize(smiles)
    cov = covariates(smiles, feats)
    clean_f = {b: clean(feats[b]) for b in BLOCKS}
    for b in BLOCKS:
        print(f"  {b:12s} {clean_f[b].shape[1]:3d} cols")

    arms = {"ecfp+core": lambda s: X[s]}
    for b in BLOCKS:
        arms[f"ecfp+core+{b}"] = (lambda s, B=clean_f[b]: np.hstack([X[s], B[s]]))
    allf = np.hstack([clean_f[b] for b in BLOCKS])
    arms["ecfp+core+all"] = lambda s: np.hstack([X[s], allf[s]])

    report, t0 = {}, time.time()
    for j, nm in enumerate(name_of):
        s = slice(offsets[j], offsets[j + 1])
        sm = list(np.array(smiles)[s])
        row = {a: _cv_rmse(b(s), y[s], sm) for a, b in arms.items()}
        row["_n"] = int(offsets[j + 1] - offsets[j])
        row["_suite"] = str(suite_of[j])
        for k, v in cov.items():
            row[f"_frac_{k}"] = float(v[s].mean())
        report[str(nm)] = row
        base = row["ecfp+core"]
        best = min((k for k in arms if k != "ecfp+core"), key=lambda k: row[k])
        print(f"  [{j + 1}/{len(name_of)}] {str(nm):22s} base {base:.4f}  "
              f"best {best.split('+')[-1]:11s} {row[best]:.4f} "
              f"({100 * (row[best] - base) / base:+.1f}%)  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(report, open(OUT / "block_report.json", "w"), indent=2)

    print(f"\n=== summary over {len(report)} datasets ===")
    names = list(arms)
    ranks = {a: [] for a in names}
    for row in report.values():
        order = sorted(names, key=lambda a: row[a])
        for r, a in enumerate(order):
            ranks[a].append(r + 1)
    for a in sorted(names, key=lambda a: np.mean(ranks[a])):
        wins = sum(1 for row in report.values() if row[a] < row["ecfp+core"])
        print(f"  {a:26s} mean rank {np.mean(ranks[a]):4.2f}   beats baseline "
              f"{wins:2d}/{len(report)}")

    print("\n=== negative controls (the test that can kill a block) ===")
    for b, key, want in (("resistance", "acyclic", False), ("cycles", "fused", True),
                         ("stereo", "has_stereo", True), ("conjugation", "conjugated", True)):
        arm = f"ecfp+core+{b}"
        hi = [r for r in report.values() if r[f"_frac_{key}"] > 0.5]
        lo = [r for r in report.values() if r[f"_frac_{key}"] < 0.1]
        if want is False:
            hi, lo = lo, hi
        g = lambda rs: (np.mean([100 * (r[arm] - r["ecfp+core"]) / r["ecfp+core"] for r in rs])
                        if rs else np.nan)
        print(f"  {b:12s} gain where {key} is high: {g(hi):+6.2f}% (n={len(hi):2d})   "
              f"low: {g(lo):+6.2f}% (n={len(lo):2d})")
    print(f"\nwrote {OUT / 'block_report.json'}")


if __name__ == "__main__":
    main()
