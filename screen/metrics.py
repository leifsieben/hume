"""Phase 3: the screen's metrics, computed locally from the predictions phase 2 wrote.

FOLLOWS Hall et al. The headline is NOT Pearson correlation, and the paper is emphatic about why:
on AmpC at 1M training molecules the best random-sampling model reached Pearson 0.86 while
recovering only 8.4% of the top 0.01% scoring molecules and 13.5% of the true binders. Their
conclusion -- "Pearson correlation reflects a model's ability to learn the overall score
distribution but underweights the top-scoring molecules crucial to enrichment" -- is the reason
this module reports recovery first and Pearson last.

Metrics, per (arm, replicate):
  * recovery@0.01%  -- fraction of the TRUE top 0.01% of the test set that appears in the
                       model's top 0.01% of predictions. The paper's headline.
  * recovery@1%     -- the same at 1%, for a less brutal read of the same quantity.
  * logAUC          -- recovery of the true top 0.01% as a function of the screened fraction,
                       integrated on a log x-axis, which is what a screening campaign cares
                       about: how deep you must go to find them.
  * pearson, rmse   -- over the whole test set, reported for comparability and NOT as the headline.
  * the same, scaffold-filtered -- test molecules sharing a Bemis-Murcko scaffold with ANY
                       training molecule are removed first (paper Figure 3 does this).

TIES ARE BROKEN PESSIMISTICALLY. With 100M predictions, a model that emits the same value for
many molecules would otherwise be flattered by argsort's arbitrary order: whichever tied
molecules happen to sit earlier in the array would be "ranked" higher, and if the true top
0.01% happened to sit early the score would be inflated for free. Within a tied block every
member is assigned the WORST rank in that block, so a model that cannot separate molecules gets
no credit for separating them.
"""
from __future__ import annotations

import json
import numpy as np


def pessimistic_top_mask(pred: np.ndarray, k: int) -> np.ndarray:
    """-> bool mask of the top `k` by `pred`, where ties get the worst rank in their block.

    A molecule is in the top k only if the number of predictions STRICTLY GREATER than its own,
    plus the size of its own tied block, is at most k. Equivalently: its whole tied block fits.
    """
    if k <= 0:
        return np.zeros(pred.shape, dtype=bool)
    order = np.argsort(-pred, kind="stable")
    s = pred[order]
    # rank of the LAST member of each tied block, 1-based
    new_block = np.empty(s.size, dtype=bool)
    new_block[0] = True
    np.not_equal(s[1:], s[:-1], out=new_block[1:])
    block_id = np.cumsum(new_block) - 1
    block_end = np.zeros(block_id[-1] + 1, dtype=np.int64)
    np.maximum.at(block_end, block_id, np.arange(1, s.size + 1))
    worst_rank = block_end[block_id]
    mask = np.zeros(pred.shape, dtype=bool)
    mask[order[worst_rank <= k]] = True
    return mask


def recovery(y_true: np.ndarray, pred: np.ndarray, frac: float) -> float:
    """-> fraction of the true top `frac` that lands in the model's top `frac`.

    "Top" means most negative: DOCK scores are energies, lower is better. Both sides use the
    same pessimistic tie rule, so a constant predictor scores ~frac rather than ~1.
    """
    n = y_true.size
    k = int(np.floor(frac * n))
    if k < 1:
        raise ValueError(f"frac={frac} over {n:,} molecules selects {k} molecules.")
    true_top = pessimistic_top_mask(-y_true, k)
    pred_top = pessimistic_top_mask(-pred, k)
    return float((true_top & pred_top).sum() / k)


def _log_area(true_top: np.ndarray, pred: np.ndarray, k: int,
              lo: float, hi: float) -> tuple[float, list]:
    """Raw area under the recovery-vs-log(screened fraction) curve, and the curve."""
    n = pred.size
    order = np.argsort(pred, kind="stable")          # ascending: best (most negative) first
    hits = np.cumsum(true_top[order])
    xs = np.unique(np.clip(np.geomspace(lo, hi, 60), 1.0 / n, 1.0))
    curve = [(float(x), float(hits[min(int(np.ceil(x * n)), n) - 1] / k)) for x in xs]
    lx = np.log10([c[0] for c in curve])
    ry = np.array([c[1] for c in curve])
    return float(np.trapezoid(ry, lx) / (lx[-1] - lx[0])), curve


def log_auc(y_true: np.ndarray, pred: np.ndarray, frac: float = 1e-4,
            lo: float = 1e-5, hi: float = 1.0,
            ideal: float | None = None) -> tuple[float, list]:
    """-> (normalised logAUC, the curve). Recovery of the true top `frac` vs screened fraction.

    NORMALISED AGAINST BOTH ENDS: (area - random) / (perfect - random), so a perfect ranker
    scores 1.0 and a random one 0.0.

    Normalising against random alone is not enough and the naive version was wrong because of it.
    A PERFECT ranker does not reach area 1 on a log axis: it has found every one of the k true
    positives by the time it has screened k/n of the library, but the curve from `lo` up to k/n
    is still climbing, so the attainable ceiling is well under 1 (0.87 for one realistic setting)
    and it MOVES with lo, hi, n and frac. An arm's score would then depend on the axis limits as
    much as on the arm. The perfect-ranker area is therefore measured, not assumed, by ranking
    with the labels themselves.

    `ideal` caches that area: it depends only on (y_true, k, lo, hi), so it is identical for every
    arm and replicate on a given test set and is computed once by `ideal_log_area`.
    """
    n = y_true.size
    k = int(np.floor(frac * n))
    true_top = pessimistic_top_mask(-y_true, k)
    area, curve = _log_area(true_top, pred, k, lo, hi)
    xs = np.array([c[0] for c in curve])
    lx = np.log10(xs)
    rand = float(np.trapezoid(xs, lx) / (lx[-1] - lx[0]))
    if ideal is None:
        ideal, _ = _log_area(true_top, y_true, k, lo, hi)
    if ideal <= rand:
        raise RuntimeError(f"perfect-ranker area {ideal:.4f} is not above random {rand:.4f}; "
                           f"the axis limits lo={lo}, hi={hi} cannot resolve the top {frac}.")
    return (area - rand) / (ideal - rand), curve


def ideal_log_area(y_true: np.ndarray, frac: float = 1e-4,
                   lo: float = 1e-5, hi: float = 1.0) -> float:
    """The perfect-ranker area for this test set. Compute once, pass to every log_auc call."""
    k = int(np.floor(frac * y_true.size))
    return _log_area(pessimistic_top_mask(-y_true, k), y_true, k, lo, hi)[0]


def evaluate(y_true: np.ndarray, pred: np.ndarray, keep: np.ndarray | None = None,
             ideal: float | None = None) -> dict:
    """Every metric for one (arm, replicate). `keep` optionally restricts the test set."""
    if keep is not None:
        y_true, pred = y_true[keep], pred[keep]
    if y_true.size != pred.size:
        raise ValueError(f"{y_true.size:,} labels against {pred.size:,} predictions")
    la, _ = log_auc(y_true, pred, ideal=ideal)
    resid = pred - y_true
    return {
        "n": int(y_true.size),
        "recovery_0.01pct": recovery(y_true, pred, 1e-4),
        "recovery_1pct": recovery(y_true, pred, 1e-2),
        "log_auc_0.01pct": la,
        "pearson": float(np.corrcoef(y_true, pred)[0, 1]),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "rmse_over_sd": float(np.sqrt((resid ** 2).mean()) / y_true.std()),
    }
