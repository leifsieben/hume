"""Shard-aware loader for the 1M corpus.

`models.py` was written against a single `train.npz` that fits comfortably in memory at 100k.
At 1M it does not: X is 1M x ~2,900 float32 = 11.6 GB, and a torch training loop needs room
alongside it in 25 GB of RAM.

Three levers, in the order worth reaching for:

  `dtype=float16` on X      halves it to 5.8 GB. The input block is log1p(ECFP counts) plus
                            standardised descriptors, so ~3 decimal digits is ample; Y stays
                            float32 because it is the regression target.
  `max_n`                   train on a subset. Useful for a fast sanity pass before committing
                            to the full corpus, not for the real run.
  `shards=`                 hold out whole shards. Shards follow selection order, which is
                            scaffold-grouped, so a shard split is closer to a scaffold split
                            than a random row split would be -- but it is not a substitute for
                            one, and `scaffold_split` in models.py remains the real answer.

Falls back to the old `data/surrogate/train.npz` when the corpus is absent, so nothing that
currently works breaks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "corpus1m" / "packed"
LEGACY = ROOT / "data" / "surrogate" / "train.npz"


def available() -> bool:
    return CORPUS.exists() and any(CORPUS.glob("pk_*.npz"))


def load(max_n: int | None = None, dtype=np.float32, shards: slice | None = None):
    """-> (X, Y, smiles, meta). meta carries column names and per-column NaN rates."""
    import json
    if not available():
        d = np.load(LEGACY, allow_pickle=True)
        return (d["X"].astype(dtype), d["Y"], list(d["smiles"]),
                {"xnames": None, "ynames": list(d["target_names"]),
                 "families": list(d["target_families"])})

    files = sorted(CORPUS.glob("pk_*.npz"))
    if shards is not None:
        files = files[shards]
    meta_p = CORPUS.parent / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}

    Xs, Ys, S, n = [], [], [], 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        x, y, s = z["X"], z["Y"], list(z["smiles"])
        if max_n is not None and n + len(s) > max_n:
            k = max_n - n
            x, y, s = x[:k], y[:k], s[:k]
        Xs.append(x.astype(dtype, copy=False))
        Ys.append(y.astype(np.float32, copy=False))
        S.extend(s)
        n += len(s)
        if max_n is not None and n >= max_n:
            break
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    del Xs, Ys
    print(f"corpus: X{X.shape} {X.dtype} ({X.nbytes / 1e9:.1f} GB) | Y{Y.shape} | "
          f"{len(files)} shards")
    return X, Y, S, meta


BENCH = ROOT / "data" / "bench1m" / "packed"


def load_bench(dtype=np.float32):
    """Benchmark matrices built by the *same* code path as the corpus.

    Falls back to the legacy `data/surrogate/bench.npz` only if `bench1m` is absent, and that
    fallback will trip the column-count assertion in `models.py` -- deliberately, since the
    legacy file predates the CORE/PREDICT split change and its 2687 columns no longer align
    with the corpus's 2918.
    """
    import json
    if not (BENCH.exists() and any(BENCH.glob("pk_*.npz"))):
        d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
        print("WARNING: bench1m absent, falling back to legacy bench.npz (stale column layout)")
        return d["X"].astype(dtype), None, list(d["smiles"]), {}
    Xs, Ys, S = [], [], []
    for f in sorted(BENCH.glob("pk_*.npz")):
        z = np.load(f, allow_pickle=True)
        Xs.append(z["X"].astype(dtype, copy=False))
        Ys.append(z["Y"].astype(np.float32, copy=False))
        S.extend(list(z["smiles"]))
    meta_p = BENCH.parent / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    print(f"bench: X{X.shape} | Y{Y.shape} | {len(S):,} molecules")
    return X, Y, S, meta


def families_from_names(ynames):
    """Family label per target, for the per-family R^2 report.

    Derived from the column names rather than carried alongside, so it cannot drift out of
    sync with the target set the way a separately-stored array can.
    """
    import re
    import blocks
    from mordred import Calculator, descriptors as md
    fam = {str(x): type(x).__module__.split(".")[-1]
           for x in Calculator(md, ignore_3D=True).descriptors}
    out = []
    for n in ynames:
        src, _, name = n.partition(":")
        out.append(blocks.classify(src, name, fam)[1])
    return np.array(out)


def drop_dead_targets(Y, ynames, meta, max_nan: float = 0.20):
    """Remove target columns that are mostly non-finite or constant.

    A column that is 40% NaN at 1M is not a target the surrogate can learn; leaving it in
    lets it dominate a masked loss with noise and quietly drags the reported mean R^2 down.
    Reported explicitly rather than silently dropped, because "we trained on 161 targets" and
    "we trained on 140" are different claims.

    Tests `~isfinite`, not `isnan`. RDKit's `Ipc` is an information-content descriptor whose
    magnitude grows super-exponentially with molecule size and overflows float32 on large
    molecules -- the benchmark reaches 316 heavy atoms -- yielding **+inf, not NaN**. An
    isnan-only check passes those straight through, and a single inf destroys a masked MSE
    silently rather than loudly.
    """
    nan_frac = (~np.isfinite(Y)).mean(0)
    with np.errstate(invalid="ignore"):
        finite = np.where(np.isfinite(Y), Y, np.nan)
        const = np.nan_to_num(np.nanstd(finite, axis=0), nan=0.0) == 0
    keep = (nan_frac <= max_nan) & ~const
    dropped = [(ynames[i], float(nan_frac[i]), bool(const[i]))
               for i in np.flatnonzero(~keep)]
    if dropped:
        print(f"dropping {len(dropped)} of {Y.shape[1]} targets "
              f"(>{100 * max_nan:.0f}% non-finite or constant):")
        for n, f, c in dropped[:12]:
            print(f"    {n:44s} {'constant' if c else f'{100 * f:5.1f}% NaN'}")
        if len(dropped) > 12:
            print(f"    ... and {len(dropped) - 12} more")
    return keep, dropped
