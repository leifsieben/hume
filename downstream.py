"""What does the reconstruction actually buy? Downstream RMSE for every surrogate.

Reconstruction R^2 is not the objective: Gate 1 measured a projection that retained 95% of
descriptor variance delivering 26% of the downstream gain. This is the arm that decides.

    ecfp                          reference
    ecfp + core                   the shipped `fast` configuration
    ecfp + core + predicted_M     for each model M   <- what we would ship
    ecfp + core + exact           true descriptors   <- the ceiling

The gap between `predicted_M` and `exact` is the surrogate's real cost. The gap between
`exact` and `core` is what the whole predict block is worth in the first place -- if that is
zero, no surrogate is needed at all.

Every arm carries ECFP. Benchmarking descriptors against a fingerprint-free baseline is the
error that produced a wrong "Mordred helps" conclusion earlier in this project.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"
MODELS = ["ridge", "linquad", "pinet", "mlp", "gnn"]


def apply_prep(Y, p):
    Z = np.clip(Y[:, p["keep"]].astype(np.float64), p["lo"], p["hi"])
    return np.nan_to_num((Z - p["mu"]) / p["sd"], nan=0.0).astype(np.float32)


def _cv_rmse(X, y, smiles):
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


def main() -> None:
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    X, Ytrue, smiles = d["X"], d["Y"], list(d["smiles"])
    y, offsets, suite_of, name_of = d["y"], d["offsets"], d["suite_of"], d["name_of"]
    prep = dict(np.load(OUT / "prep_blocks.npz", allow_pickle=True))
    EXACT = apply_prep(Ytrue, prep)
    ECFP, CORE = X[:, :2048], X[:, 2048:]
    print(f"bench {X.shape} | ecfp {ECFP.shape} core {CORE.shape} | predict block {EXACT.shape}")

    arms = {"ecfp": lambda s: ECFP[s],
            "ecfp+core": lambda s: X[s],
            "ecfp+core+exact": lambda s: np.hstack([X[s], EXACT[s]])}
    for m in MODELS:
        f = OUT / f"pred_bench_{m}.npz"
        if f.exists():
            P = np.load(f)["pred"]
            arms[f"ecfp+core+{m}"] = (lambda s, P=P: np.hstack([X[s], P[s]]))
        else:
            print(f"  (skipping {m}: no predictions)")

    report, t0 = {}, time.time()
    for suite in ("moleculenet", "moleculeace"):
        per = {}
        for j, nm in enumerate(name_of):
            if suite_of[j] != suite:
                continue
            s = slice(offsets[j], offsets[j + 1])
            per[nm] = {a: _cv_rmse(b(s), y[s], list(np.array(smiles)[s])) for a, b in arms.items()}
            print(f"  [{suite}] {nm} ({time.time() - t0:.0f}s)", flush=True)
        summ = {a: float(np.nanmean([r[a] for r in per.values()])) for a in arms}
        report[suite] = {"summary": summ, "per_dataset": per}
        json.dump(report, open(OUT / "downstream_report.json", "w"), indent=2)

        base, ceil = summ["ecfp+core"], summ["ecfp+core+exact"]
        span = ceil - base
        print(f"\n=== {suite} (n={len(per)}) ===")
        print(f"  {'ecfp':22s} {summ['ecfp']:.4f}")
        print(f"  {'ecfp+core':22s} {base:.4f}   <- baseline")
        print(f"  {'ecfp+core+exact':22s} {ceil:.4f}   <- ceiling (gain {span:+.4f})")
        for m in MODELS:
            k = f"ecfp+core+{m}"
            if k not in summ:
                continue
            frac = 100 * (summ[k] - base) / span if abs(span) > 1e-9 else float("nan")
            wins = sum(1 for r in per.values() if r[k] < r["ecfp+core"])
            print(f"  {k:22s} {summ[k]:.4f}   {frac:5.0f}% of ceiling   helps {wins}/{len(per)}")
        print()
    print(f"wrote {OUT / 'downstream_report.json'}")


if __name__ == "__main__":
    main()
