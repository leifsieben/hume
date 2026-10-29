"""Download the LSD AmpC `score` column -- and nothing else -- for all 246 shards.

WHY ONLY THE SCORES. The release is 52.3 GB over 246 parquet shards of {smiles, zincid, score}.
Per row the columns cost 23.37 B, 8.82 B and 0.10 B compressed, so the full score column for all
1,569,365,216 molecules is about 0.2 GB while the SMILES are 36.7 GB and the ZINC ids -- which
this benchmark never uses -- are 13.8 GB. Pulling scores first buys the exact global score
distribution for 1.5% of the bytes, and every later decision (the top-0.1% threshold, which row
groups to sample, the stratified training draw) is then made offline against local files instead
of over the network.

WHAT IS STORED, AND WHY int16. Scores are quantized to two decimals in the release, so
`round(score * 100)` is lossless -- but that is an observation about someone else's file, not a
guarantee, so every shard is checked and the run fails loudly rather than silently rounding
anyone's data. int16 holds -327.68..327.67 kcal/mol, far outside the physical range, and costs
12.8 MB per shard against 25.5 MB for float32.

COMPLETION IS PROVABLE. A shard counts as done when its stored array length equals the row count
in the shard's own parquet footer, which is read fresh on every pass -- never because the file
exists. A truncated download therefore re-runs instead of poisoning the sample.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/IrwinLab/LSD_AmpC_Liu_2025/docking_results"
N_SHARDS = 246
EXPECT_ROWS = 1_568_863_695          # docking_results ONLY. The dataset-server size
#                                     endpoint reports 1,569,365,216 for the REPO, which
#                                     adds invitro_results (1,521) and poses_top500k
#                                     (500,000) -- two configs this benchmark never reads.
OUT = Path("data/lsd_ampc/scores")
SCALE = 100                          # two decimals
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def shard_path(i: int) -> str:
    return f"{REPO}/train-{i:05d}-of-{N_SHARDS:05d}.parquet"


def already_done(dest: Path, n_rows: int) -> bool:
    """Done means the stored array is as long as the footer says. Not that the file exists."""
    if not dest.exists():
        return False
    try:
        with np.load(dest) as z:
            return int(z["q"].shape[0]) == n_rows
    except Exception:
        return False                 # unreadable or truncated -> redo


def fetch(i: int, fs: HfFileSystem, force: bool) -> dict:
    dest = OUT / f"shard_{i:05d}.npz"
    t0 = time.perf_counter()
    pf = pq.ParquetFile(fs.open(shard_path(i), "rb"))
    n_rows, n_rg = pf.metadata.num_rows, pf.metadata.num_row_groups

    if not force and already_done(dest, n_rows):
        with np.load(dest) as z:
            q = z["q"]
        return dict(shard=i, n=n_rows, n_row_groups=n_rg, cached=True,
                    min=int(q.min()), max=int(q.max()), sorted=bool(np.all(np.diff(q) >= 0)),
                    secs=time.perf_counter() - t0)

    s = pf.read(columns=["score"])["score"].to_numpy()
    if s.shape[0] != n_rows:
        raise RuntimeError(f"shard {i}: read {s.shape[0]:,} scores but the footer says "
                           f"{n_rows:,}. A short read must not be stored.")
    if not np.all(np.isfinite(s)):
        raise RuntimeError(f"shard {i}: {int((~np.isfinite(s)).sum())} non-finite scores. "
                           f"Decide what they mean before they enter a threshold.")

    q = np.rint(np.asarray(s, dtype=np.float64) * SCALE)
    err = np.max(np.abs(q / SCALE - s))
    if err > 1e-9:
        raise RuntimeError(
            f"shard {i}: scores are not exactly two decimals (max round-trip error {err:.3e}). "
            f"int16 storage would silently change the data -- store float32 instead.")
    if q.min() < np.iinfo(np.int16).min or q.max() > np.iinfo(np.int16).max:
        raise RuntimeError(f"shard {i}: score*{SCALE} range [{q.min()}, {q.max()}] "
                           f"does not fit int16.")
    q = q.astype(np.int16)

    # The temp name must itself end in .npz: np.savez APPENDS the extension to anything that
    # does not, so a ".npz.part" target is written as ".npz.part.npz" and the rename below then
    # looks for a file that was never created.
    tmp = dest.with_name(dest.stem + ".part.npz")   # atomic: no half file at the real name
    np.savez_compressed(tmp, q=q)
    tmp.replace(dest)
    return dict(shard=i, n=n_rows, n_row_groups=n_rg, cached=False,
                min=int(q.min()), max=int(q.max()), sorted=bool(np.all(np.diff(q) >= 0)),
                secs=time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent shard reads; this is latency-bound, not CPU-bound")
    ap.add_argument("--shards", type=int, default=N_SHARDS, help="first N shards (for a smoke run)")
    ap.add_argument("--force", action="store_true", help="re-download even if already complete")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    todo = list(range(a.shards))
    log(f"{len(todo)} shards, {a.workers} workers -> {OUT}")

    rows = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, i, fs, a.force): i for i in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            i = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                log(f"  shard {i:3d} FAILED: {type(e).__name__}: {e}")
                continue
            rows[i] = r
            el = time.perf_counter() - t0
            log(f"  [{k:3d}/{len(todo)}] shard {i:3d}  n={r['n']:,}  "
                f"score [{r['min']/SCALE:7.2f}, {r['max']/SCALE:7.2f}]  "
                f"sorted={str(r['sorted']):5s} {'cached' if r['cached'] else 'fetched'} "
                f"{r['secs']:5.1f}s  | {el/60:5.1f} min elapsed, "
                f"{(len(todo)-k)*el/k/60:5.1f} min left")

    missing = sorted(set(todo) - set(rows))
    total = sum(r["n"] for r in rows.values())
    log(f"\n{len(rows)}/{len(todo)} shards, {total:,} rows in "
        f"{(time.perf_counter()-t0)/60:.1f} min")
    if a.shards == N_SHARDS and not missing and total != EXPECT_ROWS:
        log(f"WARNING: {total:,} rows but the dataset server says {EXPECT_ROWS:,} "
            f"(difference {EXPECT_ROWS - total:+,})")
    if missing:
        log(f"INCOMPLETE -- missing shards: {missing}\nRe-run to retry; completed shards are kept.")

    man = OUT.parent / "scores_manifest.json"
    man.write_text(json.dumps(
        {"repo": REPO, "n_shards": a.shards, "scale": SCALE, "rows": total,
         "complete": not missing and a.shards == N_SHARDS,
         "shards": [rows[i] for i in sorted(rows)]}, indent=1))
    log(f"manifest -> {man}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
