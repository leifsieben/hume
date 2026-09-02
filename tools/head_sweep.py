"""Does the physicochemical cost of the minimal spec survive a different prediction head?

    .venv/bin/python tools/head_sweep.py

THE HYPOTHESIS BEING TESTED, stated so it can fail. docs/MINIMAL_SPEC.md section 11 explains the
+3.83% physchem cost as follows: the spec guarantees dropped columns are LINEARLY recoverable,
but a depth-6 boosted tree splits on individual columns and cannot split on a linear combination
of thirty of them. If that explanation is right, then a head that CAN form linear combinations
should show a much smaller gap -- and a purely linear head should show almost none, since there
the guarantee is nearly a tautology.

If instead the gap persists across every head, the explanation is wrong and the dropped columns
carry something that is not merely a rotation away.

BASELINE FIRST. xgb_d6 reproduces the grid's own head, on the same molecules and the SAME STORED
FOLDS, so its gap must match the AWS number. A reanalysis that cannot reproduce the thing it is
reanalysing is measuring its own pipeline.

 Trees take NaN natively; ridge and the MLP do not. Those two get train-fold median
imputation and train-fold standardisation, both fitted on train only. That is a difference
between heads, not between arms -- both arms get identical treatment inside a head -- so the
GAP stays comparable even though absolute scores across heads are not.
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
FEAT = Path("results/reanalysis/features")
OUT = Path("results/reanalysis")
DATASETS = ["esol", "lipophilicity", "aqsoldb", "pb_logd", "pb_water_sol", "photoswitch"]


def heads():
    import xgboost as xgb
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    return {
        "xgb_d6":  ("tree", lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6,
                    random_state=0, n_jobs=-1, verbosity=0, colsample_bynode=0.3,
                    tree_method="hist")),
        "xgb_d10": ("tree", lambda: xgb.XGBRegressor(n_estimators=300, max_depth=10,
                    random_state=0, n_jobs=-1, verbosity=0, colsample_bynode=0.3,
                    tree_method="hist")),
        "xgb_d6_full_cols": ("tree", lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6,
                    random_state=0, n_jobs=-1, verbosity=0, colsample_bynode=1.0,
                    tree_method="hist")),
        "ridge":   ("dense", lambda: RidgeCV(alphas=np.logspace(-2, 5, 15))),
        "mlp":     ("dense", lambda: MLPRegressor(hidden_layer_sizes=(512, 128),
                    max_iter=400, early_stopping=True, n_iter_no_change=15,
                    random_state=0)),
    }


def prep_dense(tr, te):
    """Median-impute and standardise, fitted on TRAIN only."""
    tr = np.where(np.isfinite(tr), tr, np.nan)
    te = np.where(np.isfinite(te), te, np.nan)
    med = np.nanmedian(tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    tr = np.where(np.isnan(tr), med, tr)
    te = np.where(np.isnan(te), med, te)
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (tr - mu) / sd, (te - mu) / sd


def run():
    rows = []
    H = heads()
    for ds in DATASETS:
        z = np.load(FEAT / f"{ds}.npz", allow_pickle=False)
        X, fp, y, folds = z["X"], z["fp"], z["y"], z["folds"]
        mask = z["minimal_mask"]
        # +/-inf -> NaN, EXACTLY AS bench_downstream.py DOES. rdkit emits inf on real
        # molecules (Ipc overflows on larger graphs, the partial-charge descriptors on a few odd
        # valences) and xgboost refuses it outright: "Input data contains `inf` or a value too
        # large". Trees then read NaN as missing. Doing anything different here would make the
        # baseline fail to reproduce the grid, which is the one thing it must do.
        X = np.where(np.isfinite(X), X, np.nan)
        arms = {"hume": np.hstack([X, fp]).astype(np.float32),
                "hume_minimal": np.hstack([X[:, mask], fp]).astype(np.float32)}
        for hname, (kind, make) in H.items():
            for arm, F in arms.items():
                scores = []
                for k in range(5):
                    te = folds == k
                    tr = ~te
                    A, B = F[tr], F[te]
                    if kind == "dense":
                        A, B = prep_dense(A, B)
                    m = make()
                    m.fit(A, y[tr])
                    p = m.predict(B)
                    scores.append(float(np.sqrt(np.mean((y[te] - p) ** 2))))
                rows.append(dict(dataset=ds, head=hname, arm=arm,
                                 rmse=float(np.mean(scores)),
                                 sd=float(np.std(scores, ddof=1)), folds=scores))
                print(f"  {ds:14s} {hname:18s} {arm:13s} {np.mean(scores):9.4f}")
    json.dump(rows, open(OUT / "head_sweep.json", "w"), indent=1)
    print(f"\n  -> {OUT / 'head_sweep.json'}")
    return rows


if __name__ == "__main__":
    run()
