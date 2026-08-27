"""Evaluation metrics, written in-repo (roadmap §6).

No scikit-learn (masterplan: minimal dependencies, everything auditable). Every metric here is
a few lines of NumPy with the formula spelled out, so a reviewer can check the maths rather than
trust a black box.

The roadmap wants **three label types**, each with its own metrics:

- **regression** (primary for the cliff thesis — only it tests smooth-and-sharp simultaneity):
  ``rmse``, ``mae``, ``r2``, ``pearson``, ``spearman``.
- **binary** (deployment realism; everything becomes a step function): ``roc_auc``, ``bedroc``,
  plus early-enrichment ``nef`` on the binarised labels.
- **ranking** (monotone-invariant): ``spearman``, ``nef`` (NEF@1% — the VS metric of record),
  ``bedroc``.

Two virtual-screening subtleties that the formulas get right:

- **NEF (normalised enrichment factor)** divides the raw enrichment by its *achievable maximum*
  at that active fraction, so a value of 1.0 always means "perfect early ranking" regardless of
  how many actives exist. Raw EF@1% is uninterpretable across datasets with different hit rates.
- **BEDROC** (Truchon & Bayly 2007) is the early-recognition metric: an exponential weight
  ``alpha`` makes finding actives near the top count far more than finding them lower down,
  which is exactly the VS regime (false negatives at the top are the expensive mistake).

All functions drop non-finite pairs first (a NaN label or prediction is excluded pairwise), so
a few failed molecules never crash a metric.
"""

from __future__ import annotations

import numpy as np


def _clean(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align and drop any pair where either value is non-finite."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    keep = np.isfinite(y) & np.isfinite(p)
    return y[keep], p[keep]


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks, ties shared — the rank transform Spearman/AUC need."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Resolve ties to their average rank so tied values are exchangeable.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    starts = csum - counts
    avg = (starts + csum + 1) / 2.0  # mean of the 1-based rank block for each unique value
    return avg[inv]


# ---------------------------------------------------------------------------- regression
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _clean(y, p)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _clean(y, p)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y - p)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination, 1 − SS_res/SS_tot."""
    y, p = _clean(y, p)
    if y.size == 0:
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:  # constant target — R² undefined
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    return 1.0 - ss_res / ss_tot


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _clean(y, p)
    if y.size < 2 or y.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    """Rank correlation = Pearson on average-ranks. Monotone-invariant."""
    y, p = _clean(y, p)
    if y.size < 2:
        return float("nan")
    ry, rp = _ranks(y), _ranks(p)
    if ry.std() == 0 or rp.std() == 0:
        return float("nan")
    return float(np.corrcoef(ry, rp)[0, 1])


# -------------------------------------------------------------- binary / virtual screening
def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC via the Mann–Whitney U identity (probability a random active outranks a random
    inactive). Ties in score contribute 0.5, handled by using average ranks.
    """
    y_true, score = _clean(y_true, score)
    pos = y_true > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(score)
    auc = (r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def enrichment_factor(y_true: np.ndarray, score: np.ndarray, frac: float = 0.01) -> float:
    """Raw EF@frac: (hit rate in the top ``frac`` by score) / (overall hit rate).

    Not normalised — use ``nef`` for cross-dataset comparison. Kept because it is the quantity
    people report directly and ``nef`` is defined in terms of it.
    """
    y_true, score = _clean(y_true, score)
    n = y_true.size
    n_actives = int((y_true > 0.5).sum())
    if n == 0 or n_actives == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    top = np.argsort(-score, kind="mergesort")[:k]
    hits = int((y_true[top] > 0.5).sum())
    return (hits / k) / (n_actives / n)


def nef(y_true: np.ndarray, score: np.ndarray, frac: float = 0.01) -> float:
    """Normalised enrichment factor at ``frac`` — EF divided by its achievable maximum.

    The maximum EF at fraction ``frac`` is reached when the top-k slots are all actives, i.e.
    ``min(k, n_actives) / k`` hit rate. Dividing by that puts NEF on [0, 1] with 1 = perfect
    early ranking, which is what makes it comparable across datasets with different hit rates.
    """
    y_true, score = _clean(y_true, score)
    n = y_true.size
    n_actives = int((y_true > 0.5).sum())
    if n == 0 or n_actives == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    ef = enrichment_factor(y_true, score, frac)
    max_hits = min(k, n_actives)
    ef_max = (max_hits / k) / (n_actives / n)
    if ef_max == 0:
        return float("nan")
    return float(ef / ef_max)


def bedroc(y_true: np.ndarray, score: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly, J. Chem. Inf. Model. 2007), early-recognition metric.

    ``alpha`` sets how sharply early ranks are up-weighted; alpha=20 (the common default)
    concentrates ~80% of the weight in the top ~8%. Returns a value in [0, 1]; 0.5 is random
    for a balanced set. Formula follows the paper directly.
    """
    y_true, score = _clean(y_true, score)
    n = y_true.size
    n_actives = int((y_true > 0.5).sum())
    if n == 0 or n_actives == 0 or n_actives == n:
        return float("nan")

    ra = n_actives / n  # active ratio
    # Ranks of actives, 1-based, sorted by descending score (ties: stable order).
    order = np.argsort(-score, kind="mergesort")
    is_active = y_true[order] > 0.5
    rank_positions = np.nonzero(is_active)[0] + 1  # 1-based ranks of the actives

    # Sum of exponential rank weights over the actives (r_i is the 1-based rank, N = n).
    s = float(np.sum(np.exp(-alpha * rank_positions / n)))
    # Random expectation of that sum: n_actives placed uniformly, using the closed form
    #   Σ_{r=1}^N exp(-α r/N) = (1 - e^{-α}) / (e^{α/N} - 1).
    # So E[s] = (n_actives / N) · (1 - e^{-α}) / (e^{α/N} - 1). RIE = s / E[s].
    expected_s = (n_actives / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
    rie = s / expected_s
    # RIE → BEDROC affine rescaling to [0, 1] (paper eq. for BEDROC from RIE).
    factor = (ra * np.sinh(alpha / 2.0)) / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra))
    bedroc_val = rie * factor + 1.0 / (1.0 - np.exp(alpha * (1 - ra)))
    # BEDROC is analytically bounded to [0, 1]; clamp away floating-point excursions at the
    # extremes (the all-actives-at-bottom case lands at ~-5e-9, which is just 0 with noise).
    return float(min(1.0, max(0.0, bedroc_val)))


# ------------------------------------------------------------------------ metric bundles
def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "r2": r2(y, p),
        "pearson": pearson(y, p),
        "spearman": spearman(y, p),
    }


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auc": roc_auc(y, score),
        "bedroc": bedroc(y, score),
        "nef1pct": nef(y, score, 0.01),
    }


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Ranking treats a continuous target as the thing to order; binary VS metrics come from
    thresholding the target at its top ``frac`` (the 'true top' the screen must recover)."""
    out = {"spearman": spearman(y, score)}
    # Define actives as the top 1% of the true continuous values, then measure early recovery.
    y_clean, s_clean = _clean(y, score)
    if y_clean.size:
        k = max(1, int(round(0.01 * y_clean.size)))
        thresh = np.sort(y_clean)[-k]
        active = (y_clean >= thresh).astype(np.float64)
        out["nef1pct"] = nef(active, s_clean, 0.01)
        out["bedroc"] = bedroc(active, s_clean)
    else:
        out["nef1pct"] = float("nan")
        out["bedroc"] = float("nan")
    return out
