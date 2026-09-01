"""Derive the HUME_minimal column ordering by rank-revealing QR, and measure its coverage.

    .venv/bin/python tools/minimal_select.py

THE OBJECTIVE, stated so it can be falsified: find the smallest ordered column set S such that
every column NOT in S is linearly recoverable from S. This is a COVERAGE criterion, not a
compression one. A dropped column costs a downstream model nothing if the model can rebuild it;
what is lost is only a column's UNIQUE variance. So the question is never "which columns carry
the most variance" -- that keeps the loudest columns -- but "from which columns can we rebuild
all the others".

LABEL-FREE. This reads only the descriptor matrix. No target, no assay, no benchmark. Selecting
on labels leaks the benchmark, picks descriptors suited to whatever chemistry supplied the
labels, and -- decisively -- a descriptor spec is a permanent contract inherited by users whose
chemistry nobody here has seen. Redundancy is a property of the molecules; informativeness is a
property of somebody's targets. Only the first is safe to select on.

WHY NOT CORRELATION CLUSTERING. Cluster-by-|rho| cannot see multi-column dependence: a column
can be an exact linear combination of three others while correlating weakly with each. Pivoted
QR orders columns by how much NEW direction each adds given those already chosen, and it selects
ACTUAL COLUMNS rather than components -- PCA answers "how many dimensions are there" but its
components cannot be shipped as a descriptor list.

NON-FINITE HANDLING IS A REAL CHOICE AND IT IS MADE HERE EXPLICITLY. Only 4.6% of corpus rows
are finite in every one of the 1,269 columns, so dropping incomplete rows would leave ~1,100
rows for 1,269 columns -- rank-deficient, and any k derived from it would be an artifact of the
deficiency rather than a fact about chemistry. Instead: gate out unusable columns, then impute
the remainder at the column median. Median rather than mean because these distributions have
heavy tails. Imputation makes a column slightly MORE predictable than it truly is, which biases
toward dropping, so `--finite-only` re-runs the whole thing on the subset of columns that are
100% finite with no imputation at all; if the two agree, imputation is not driving the answer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq as slstsq, qr

OUT = Path("results/minimal")
KS = [256, 400, 512, 576, 640, 704, 768, 800, 832, 864, 896, 928, 960, 1024, 1100, 1200]


def _impute(X, cols):
    """Column-median imputation over `cols`. -> (dense matrix, n_cells_imputed)."""
    Xu = X[:, cols]
    fin = np.isfinite(Xu)
    med = np.array([np.median(c[f]) if f.any() else 0.0 for c, f in zip(Xu.T, fin.T)])
    return np.where(fin, Xu, med), int((~fin).sum())


def _zscore(M):
    """Centre and scale, holding a constant column at zero rather than dividing by zero.

    A column that is constant on THIS sample is trivially recoverable (its residual variance is
    zero), so it must survive into the design matrix as a zero column rather than as NaN --
    which is what broke the first version: NaNs reached lstsq and SVD failed to converge.
    """
    sd = M.std(axis=0)
    return (M - M.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def select_columns(X, finite_only):
    """The gate: which of the 1,269 columns are eligible at all, measured on repA."""
    col_finite = np.isfinite(X).mean(axis=0)
    usable = col_finite == 1.0 if finite_only else col_finite >= 0.50
    idx = np.flatnonzero(usable)
    M, _ = _impute(X, idx)
    return idx[M.std(axis=0) > 0]           # constant columns are free to drop


def prepare(X, cols):
    M, n_imp = _impute(X, cols)
    return _zscore(M), n_imp


#: Ridge penalty for the reconstruction fit, as a multiple of n. After z-scoring, diag(A'A) ~ n,
#: so this is scale-free. NOT a tuning knob for the answer -- it exists because the kept set is
#: numerically singular (cond ~1e15 by k=512) and the unregularised solve produces coefficients
#: that do not transfer AT ALL: held-out R^2 of -1e18 between two disjoint draws from the same
#: corpus. Chosen from a grid {1e-6, 1e-4, 1e-2, 1, 1e2} x n by held-out worst-case R^2 on repB;
#: 1e-2 was best at every k tested and the answer is flat between 1e-6 and 1e-2.
RIDGE = 1e-2


def coverage_heldout(Zfit, Zeval, order, k, alpha_mult=RIDGE):
    """Fit the reconstruction map on one sample, score it on another. THE decisive number.

    In-sample R^2 is not enough here. The kept set's condition number reaches 1e15 by k=512 --
    numerically singular in double precision -- so a high in-sample R^2 could be a
    least-squares fit exploiting collinear directions rather than a real, transferable
    relationship. Fitting the map on repA and scoring it on the disjoint repB answers the
    question that actually matters to a downstream user: if I drop this column, can a model
    trained on MY molecules rebuild it? If the reconstruction does not transfer between two
    draws from the same corpus, it is not recoverability.
    """
    keep, drop = order[:k], order[k:]
    if len(drop) == 0:
        return 1.0, 0
    A, B = Zfit[:, keep], Zfit[:, drop]
    G = A.T @ A + alpha_mult * len(A) * np.eye(len(keep))
    coef = np.linalg.solve(G, A.T @ B)
    resid = Zeval[:, drop] - Zeval[:, keep] @ coef
    sse = (resid ** 2).sum(axis=0)
    sst = (Zeval[:, drop] ** 2).sum(axis=0)
    r2 = 1.0 - sse / np.where(sst == 0, 1, sst)
    return float(r2.min()), int((r2 < 0.99).sum())


def coverage(Z, order, k):
    """Worst-case R^2 over dropped columns when the first k of `order` are kept.

    `gelsy` (complete orthogonal factorisation) rather than numpy's default `gelsd` (SVD).
    numpy raised "SVD did not converge in Linear Least Squares" at k=816 on repB -- the inputs
    are finite and well scaled (max |z| ~ 150, no NaN), so that is a near-rank-deficient design
    defeating the SVD driver, not bad data. gelsy handles rank deficiency by construction. It is
    also the honest reading: a kept set that is technically full rank but badly conditioned
    recovers dropped columns only in exact arithmetic, which is why the condition number is
    reported alongside.
    """
    keep, drop = order[:k], order[k:]
    if len(drop) == 0:
        return 1.0, 0, np.array([])
    A, B = Z[:, keep], Z[:, drop]
    coef, *_ = slstsq(A, B, lapack_driver="gelsy")
    resid = B - A @ coef
    sse = (resid ** 2).sum(axis=0)
    sst = (B ** 2).sum(axis=0)              # already centered and scaled
    r2 = 1.0 - sse / np.where(sst == 0, 1, sst)
    return float(r2.min()), int((r2 < 0.99).sum()), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite-only", action="store_true",
                    help="sensitivity check: 100%%-finite columns only, no imputation")
    ap.add_argument("--pool", action="store_true",
                    help="derive the ordering on repA AND adv stacked, which is the shipped "
                         "method -- a column is only safe to drop if it is redundant on BOTH")
    args = ap.parse_args()

    z = np.load(OUT / "matrices.npz", allow_pickle=False)
    names = np.array([str(n) for n in z["names"]])

    idxA = select_columns(z["repA"], args.finite_only)
    ZA, nimpA = prepare(z["repA"], idxA)
    print(f"  repA: {ZA.shape[1]} usable columns of {len(names)}, "
          f"{nimpA:,} cells imputed ({nimpA / ZA.size:.3%})")

    ZB, nimpB = prepare(z["repB"], idxA)
    ZV, nimpV = prepare(z["adv"], idxA)

    # THE ORDERING IS DERIVED ON BOTH DISTRIBUTIONS, STACKED. Deriving it on the representative
    # corpus alone produced an ordering that covers that corpus at k=704 and leaves `Phi` and
    # `Kappa2` at R^2 = 0.07 on salts and mixtures -- both descend from HallKierAlpha, whose
    # alpha table is solved over the (element, hybridisation) pairs the training corpora
    # contained and which METHODS.md 7.2 already records as failing on organometallics. On that
    # chemistry they carry unique variance nothing else reproduces, so an ordering blind to it
    # drops them.
    #
    # Each sample is z-scored with its OWN statistics before stacking, not pooled and then
    # scaled: pooling first lets whichever sample has the wider spread set the scale and
    # silently dominate the pivoting.
    if args.pool:
        Zfit = np.vstack([ZA, ZV])
        print(f"  ordering derived on repA + adv stacked: {Zfit.shape}")
    else:
        Zfit = ZA
        print(f"  ordering derived on repA only: {Zfit.shape}   "
              "(comparison; the shipped spec uses --pool)")
    _, _, piv = qr(Zfit, mode="economic", pivoting=True)
    order = np.asarray(piv)

    rows = []
    print(f"\n  {'k':>5} {'repA in':>11} {'adv in':>10} {'below':>8} "
          f"{'heldout repB':>12} {'below':>7} {'heldout adv':>11} {'cond':>10}")
    print(f"  repB: {nimpB:,} cells imputed ({nimpB / ZB.size:.3%})")
    print(f"  adv : {nimpV:,} cells imputed ({nimpV / ZV.size:.3%})   "
          "<- the adversarial set is meant to be harder")
    for k in KS:
        if k >= ZA.shape[1]:
            continue
        wa, na, _ = coverage(ZA, order, k)
        wb, nb, _ = coverage(ZB, order, k)
        wv, nv, _ = coverage(ZV, order, k)
        cond = float(np.linalg.cond(ZA[:, order[:k]]))
        ho_w, ho_n = coverage_heldout(ZA, ZB, order, k)      # fit repA -> score repB
        hv_w, hv_n = coverage_heldout(ZA, ZV, order, k)      # fit repA -> score adv
        rows.append(dict(k=k, repA_worst=wa, repA_below=na,
                         repB_worst=wb, repB_below=nb, adv_worst=wv, adv_below=nv,
                         cond=cond, heldout_repB_worst=ho_w, heldout_repB_below=ho_n,
                         heldout_adv_worst=hv_w, heldout_adv_below=hv_n))
        print(f"  {k:5d} {wa:11.4f} {wv:10.4f} {max(na, nb, nv):8d} "
              f"{ho_w:12.4f} {ho_n:7d} {hv_w:11.4f} {cond:10.1e}")

    tag = ("pooled" if args.pool else "repAonly") + ("_finite" if args.finite_only else "")
    json.dump({"variant": tag, "n_usable": int(ZA.shape[1]),
               "column_index": idxA.tolist(),
               "order": order.tolist(),
               "order_names": names[idxA][order].tolist(),
               "curve": rows},
              open(OUT / f"selection_{tag}.json", "w"), indent=1)
    print(f"\n  -> {OUT / f'selection_{tag}.json'}")


if __name__ == "__main__":
    main()
