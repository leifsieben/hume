"""XGBoost baseline head (roadmap §4.6, M1).

XGBoost on concatenated fixed features *is* the baseline the whole project is built to beat.
It is the right baseline for a specific reason (masterplan): it attends each feature dimension
separately — no averaging or mixing across a fingerprint's bits — which is exactly why it is so
hard to beat on tabular molecular data, and exactly the property the ChemPFN must preserve.

WHY THE NATIVE API, NOT ``XGBRegressor``
----------------------------------------
We use ``xgboost.train`` + ``DMatrix``, not the ``xgboost.sklearn`` wrapper. The wrapper imports
scikit-learn, which the masterplan excludes from the dependency set. The native API is a few
lines more and pulls in nothing extra.

NaN IS A FIRST-CLASS INPUT. XGBoost learns a default split direction for missing values
natively, so the NaN rows/cells our featuriser emits for failed molecules or pathological
descriptors need no imputation — they flow straight in. This is the same native-NaN handling
the ChemPFN will inherit from TabPFN, so the baseline and the eventual model treat missingness
the same way.

Hyperparameters are fixed and conservative (``_DEFAULT_PARAMS``). M1 is a *baseline*: it should
be a competent, untuned XGBoost, not a per-dataset-tuned one. Tuning the baseline would move the
bar around and make the encoder comparison unreproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# macOS OpenMP guard — MUST precede `import xgboost`.
# torch and xgboost each bundle their own libomp. Loading xgboost's runtime first and then
# torch's aborts the process ("multiple copies of the OpenMP runtime have been linked"). Loading
# torch first makes xgboost reuse torch's runtime, and both then run multithreaded with no env
# hacks. Since xgboost is imported *only* here, importing torch first at this one site guarantees
# the safe order everywhere. Guarded so an xgboost-only install (no torch, hence no conflict)
# still works.
try:
    import torch as _torch  # noqa: F401  (imported for its side effect: load libomp first)
except ImportError:
    pass

import xgboost as xgb

from chemtfm.bench.datasets import BINARY, REGRESSION

# Fixed, conservative defaults. Deliberately untuned — see module docstring.
_DEFAULT_PARAMS: dict[str, object] = {
    "max_depth": 6,
    "eta": 0.1,  # learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "tree_method": "hist",  # fast histogram algorithm; fine on wide sparse fingerprint matrices
    "verbosity": 0,
    "nthread": 0,  # 0 = use all available cores
}
_DEFAULT_ROUNDS = 300


@dataclass
class XGBModel:
    """A thin wrapper around a trained XGBoost booster.

    One model = one property = one task (single-task, matching the ChemPFN's single-task
    design in roadmap §4.7). ``fit`` chooses the objective from the dataset's task; ``predict``
    returns a continuous score in both cases (raw value for regression, P(active) for binary),
    which is what every metric in ``bench.metrics`` consumes.
    """

    task: str
    num_boost_round: int = _DEFAULT_ROUNDS
    params: dict[str, object] = field(default_factory=lambda: dict(_DEFAULT_PARAMS))
    _booster: xgb.Booster | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBModel":
        params = dict(self.params)
        if self.task == REGRESSION:
            params["objective"] = "reg:squarederror"
        elif self.task == BINARY:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            raise ValueError(f"unsupported task {self.task!r}")
        dtrain = xgb.DMatrix(np.asarray(X, dtype=np.float32), label=np.asarray(y, np.float64))
        self._booster = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Continuous scores: regression value, or P(active) for binary."""
        if self._booster is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)))
