"""The one implementation of the LSD AmpC split. Both the freezer and every box call this.

`freeze_split.py` records seeds and sha256 digests; this module regenerates the indices from
those seeds and REFUSES to return an array whose digest does not match what was frozen. That is
the whole point of having one module: if the generator and the consumer were two copies of the
same thirty lines, a later edit to one of them would produce a box that silently scores a
different experiment, and the digest would be checked against the spec that the edited code
itself produced. Here the spec is an input, never a by-product.

Memory: regeneration holds the sorted index arrays themselves -- 400 MB for the test set, 63 MB
for the top 1% -- plus an 800 MB draw buffer, so about 1.5 GB. Membership is tested by
searchsorted (see `isin_sorted`) rather than by full-length boolean masks, which would be 1.57 GB
each and need three live at once. The disjointness assertions are still exact, not sampled.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np

DATA = Path("data/lsd_ampc")
N_TEST = 100_000_000
N_TRAIN = 1_000_000
TOP_FRAC = 0.01          # "the top ranking 1% of molecules"
FRAC_FROM_TOP = 0.80     # "80% of the training set ... from the top ranking 1%"
VAL_MULT = 0.20          # "validation sets about 20% as large"
VAL_POOL_MULT = 3.0      # oversample the val pool; the scaffold split discards from it
N_REPLICATES = 5
SEED0 = 20261029


def unique_sample(n_total: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k distinct indices from [0, n_total), uniform, as sorted int32.

    Draw-sort-dedupe-top-up rather than `rng.choice(replace=False)`, which would materialise a
    1.57-billion-element permutation. int32 is safe here only because n_total < 2^31; asserted.
    """
    if n_total >= np.iinfo(np.int32).max:
        raise ValueError(f"{n_total:,} indices do not fit int32; widen the dtype.")
    out = np.unique(rng.integers(0, n_total, size=k, dtype=np.int64).astype(np.int32))
    while out.size < k:
        extra = rng.integers(0, n_total, size=(k - out.size) * 2 + 1024, dtype=np.int64)
        out = np.unique(np.concatenate([out, extra.astype(np.int32)]))
    return out[:k] if out.size == k else np.sort(rng.choice(out, size=k, replace=False))


def isin_sorted(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Is each element of `q` present in the sorted, unique array `ref`?

    This replaces the obvious `mask = zeros(N, bool); mask[ref] = True; mask[q]`. At
    N = 1,568,863,695 each such mask is 1.57 GB, and the draw below needs three of them live at
    once -- which killed the first run of this module on a 24 GB laptop with no traceback, just
    a dead process. searchsorted against the index arrays themselves is the same predicate in
    400 MB instead of 4.7 GB, and the frozen sha256 digests are what prove it is the same.
    """
    if ref.size == 0:
        return np.zeros(q.shape, dtype=bool)
    i = np.searchsorted(ref, q)
    np.clip(i, 0, ref.size - 1, out=i)
    return ref[i] == q


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


def geometry(data: Path = DATA) -> tuple[np.ndarray, np.ndarray, int]:
    """(shard sizes, shard start offsets, N) over shards in ascending order.

    Read from the scores manifest, which records each shard's row count as its own parquet
    footer reported it -- so the global index space here is the same one the footers define.
    """
    man = json.loads((data / "scores_manifest.json").read_text())
    if not man.get("complete"):
        raise RuntimeError("the score download is incomplete; the split was drawn over the "
                           "whole library and cannot be reproduced from a partial manifest.")
    shards = sorted(man["shards"], key=lambda s: s["shard"])
    sizes = np.array([s["n"] for s in shards], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return sizes, starts, int(sizes.sum())


def top_indices(q_top: int, data: Path = DATA) -> np.ndarray:
    """Global indices of every molecule scoring at or below the top-1% cut."""
    _, starts, _ = geometry(data)
    tops = []
    for sh, st in enumerate(starts):
        with np.load(data / "scores" / f"shard_{sh:05d}.npz") as z:
            hit = np.flatnonzero(z["q"] <= q_top)
        if hit.size:
            tops.append((hit.astype(np.int64) + st).astype(np.int32))
    return np.concatenate(tops)


def draw_test(n: int, n_test: int, seed: int) -> np.ndarray:
    return unique_sample(n, n_test, np.random.default_rng(seed))


def draw_replicate(n: int, top_idx: np.ndarray, test: np.ndarray, seed: int,
                   n_train: int = N_TRAIN) -> tuple[np.ndarray, np.ndarray]:
    """One replicate's (train, val_pool). RNG consumption order is load-bearing: it is what
    makes the frozen digests reproducible, so do not reorder the draws below.

    `test` and `top_idx` are the sorted index arrays, not full-length masks; see isin_sorted.
    """
    rng = np.random.default_rng(seed)
    n_top = int(round(n_train * FRAC_FROM_TOP))
    n_rest = n_train - n_top

    cand_top = top_idx[~isin_sorted(top_idx, test)]
    tr_top = np.sort(rng.choice(cand_top, size=n_top, replace=False))

    # 20% from everything else: rejection sampling, since the complement is 1.55e9 rows and
    # must not be materialised.
    need, seen = n_rest, np.zeros(0, np.int32)
    while need > 0:
        c = rng.integers(0, n, size=max(need * 2, 1 << 20), dtype=np.int64).astype(np.int32)
        c = c[~isin_sorted(c, test) & ~isin_sorted(c, top_idx)]
        seen = np.unique(np.concatenate([seen, c]))
        need = n_rest - seen.size
    tr_rest = np.sort(rng.choice(seen, size=n_rest, replace=False))
    train = np.sort(np.concatenate([tr_top, tr_rest]))

    n_pool = int(round(n_train * VAL_MULT * VAL_POOL_MULT))
    pool_top = top_idx[~isin_sorted(top_idx, test) & ~isin_sorted(top_idx, train)]
    vp_top = rng.choice(pool_top, size=int(round(n_pool * FRAC_FROM_TOP)), replace=False)
    need, seen2 = n_pool - vp_top.size, np.zeros(0, np.int32)
    while need > 0:
        c = rng.integers(0, n, size=max(need * 2, 1 << 20), dtype=np.int64).astype(np.int32)
        c = c[~isin_sorted(c, test) & ~isin_sorted(c, top_idx) & ~isin_sorted(c, train)]
        seen2 = np.unique(np.concatenate([seen2, c]))
        need = (n_pool - vp_top.size) - seen2.size
    val_pool = np.sort(np.concatenate(
        [vp_top, rng.choice(seen2, size=n_pool - vp_top.size, replace=False)]))

    assert not isin_sorted(train, test).any(), "training index landed in the test set"
    assert not isin_sorted(val_pool, test).any(), "validation index landed in the test set"
    assert not np.intersect1d(val_pool, train, assume_unique=True).size, "train/val overlap"
    return train, val_pool


def load_spec(data: Path = DATA) -> dict:
    return json.loads((data / "split_spec.json").read_text())


def rebuild(spec: dict, replicates: tuple[int, ...] | None = None, want_test: bool = True,
            data: Path = DATA, log=print) -> dict[str, np.ndarray]:
    """Regenerate the frozen split and verify every array against the spec's digest.

    Returns {"test": idx, "train_0": idx, "val_pool_0": idx, ...} of GLOBAL int32 indices.
    Raises on any digest mismatch -- a box must never featurize an unverified split.
    """
    _, _, n = geometry(data)
    if n != int(spec["n_molecules"]):
        raise RuntimeError(
            f"the local scores manifest covers {n:,} molecules but the split was frozen over "
            f"{int(spec['n_molecules']):,}. The global index space differs, so every index in "
            f"the spec points somewhere else. Re-download the scores or re-freeze the split.")

    out: dict[str, np.ndarray] = {}
    test = draw_test(n, int(spec["test"]["n"]), int(spec["test"]["seed"]))
    got = digest(test)
    if got != spec["test"]["sha256"]:
        raise RuntimeError(
            f"regenerated test set does not match the frozen split.\n"
            f"  expected sha256 {spec['test']['sha256']}\n  got          {got}\n"
            f"numpy here is {np.__version__}; default_rng is version-stable, so a mismatch "
            f"means the ALGORITHM changed, not the library. Do not proceed.")
    log(f"  test {test.size:,} verified {got[:16]}")

    if want_test:
        out["test"] = test

    reps = {int(r["replicate"]): r for r in spec["replicates"]}
    wanted = sorted(reps) if replicates is None else list(replicates)
    # Read once, not once per replicate: the top-1% set is a property of the library, and every
    # replicate draws its 80% from the same one.
    top_idx = top_indices(int(round(spec["top_cut_score"] * 100)), data) if wanted else None
    if top_idx is not None and top_idx.size != int(spec["top_population"]):
        raise RuntimeError(
            f"the top-1% set has {top_idx.size:,} molecules here but {int(spec['top_population']):,} "
            f"when the split was frozen. The threshold or the scores changed; the stratified draw "
            f"would differ.")
    for r in wanted:
        if r not in reps:
            raise KeyError(f"replicate {r} is not in the frozen spec (has {sorted(reps)})")
        meta = reps[r]
        train, val_pool = draw_replicate(n, top_idx, test, int(meta["seed"]))
        for name, arr in (("train", train), ("val_pool", val_pool)):
            got = digest(arr)
            if got != meta[name]["sha256"]:
                raise RuntimeError(
                    f"replicate {r} {name} does not match the frozen split.\n"
                    f"  expected sha256 {meta[name]['sha256']}\n  got          {got}")
            out[f"{name}_{r}"] = arr
            log(f"  {name}_{r} {arr.size:,} verified {got[:16]}")
    return out



def to_local(global_idx: np.ndarray, shard: int, starts: np.ndarray,
             sizes: np.ndarray) -> np.ndarray:
    """The subset of `global_idx` falling in `shard`, as sorted int32 row offsets within it.

    `global_idx` must be sorted; searchsorted is what keeps this O(log n) per shard instead of
    a 1.57-billion-element mask per shard.
    """
    lo, hi = int(starts[shard]), int(starts[shard]) + int(sizes[shard])
    a, b = np.searchsorted(global_idx, [lo, hi])
    return (global_idx[a:b].astype(np.int64) - lo).astype(np.int32)
