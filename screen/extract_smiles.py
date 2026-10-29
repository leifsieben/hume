"""Phase 0: read the LSD AmpC SMILES column ONCE and keep only the 101.6M molecules in the split.

WHY THIS IS ITS OWN PHASE. The split selects 100,000,000 test + 5 x 1,600,000 train/val rows out
of 1,568,863,695, scattered uniformly. The release stores 1,000-row row groups, so a 6.5% uniform
sample touches essentially every row group (expected 65 hits per group) and selective range
reads buy nothing -- getting the SMILES means reading the whole 36.7 GB column. Doing that once,
here, and parking the result in S3 costs 36.7 GB of egress. Letting each of the six arms read
HuggingFace for itself would cost 220 GB, take six times as long, and put a third-party rate
limit on the critical path of every arm instead of just this one.

THE ROW-ORDER GATE IS THE POINT OF THIS SCRIPT. The split is a set of integer positions in a
global index space defined by concatenating the shards in order. That is only meaningful if the
file still has the rows it had when the split was drawn. So before a single SMILES is kept, each
shard's score column is compared element-by-element against the int16 copy downloaded on
2026-10-28 -- 6.4 million values per shard, not a checksum of a sample. If the upstream file were
ever re-sorted, re-uploaded or repartitioned, every index would silently point at a different
molecule and the whole screen would be scoring noise against the wrong labels. This check makes
that a loud failure, and it is cheap: the comparison data is 3.9 MB for the entire library.

OUTPUT, one .npz per shard, self-describing so phase 2 needs nothing but the file:
    rows      int32   sorted local row offsets kept from this shard (test + every replicate)
    smi_blob  uint8   the SMILES, newline-free, concatenated
    smi_off   int64   len(rows)+1 offsets into smi_blob
    q         int16   score * 100 for each kept row, from the file just read
    h         uint64  blake2b-64 of each SMILES string, for exact-duplicate and leakage counting
    test_pos  int32   positions within `rows` belonging to the test set
    train_<r>_pos, val_pool_<r>_pos  int32  likewise per replicate
`rows` is the UNION, stored once: a molecule in two replicates' training sets is one SMILES here,
not two.
"""
from __future__ import annotations

import argparse, hashlib, io, json, os, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import split as S                      # the one split implementation; see its docstring
import provenance

REPO = "datasets/IrwinLab/LSD_AmpC_Liu_2025/docking_results"
N_SHARDS = 246


def shard_path(i: int) -> str:
    return f"{REPO}/train-{i:05d}-of-{N_SHARDS:05d}.parquet"


_FS = []


def _fs():
    """One HfFileSystem for the run. Constructing it per shard re-does the HTTP session and
    token resolution 246 times for no benefit."""
    if not _FS:
        from huggingface_hub import HfFileSystem
        _FS.append(HfFileSystem())
    return _FS[0]


def hash64(blob: bytes, off: np.ndarray) -> np.ndarray:
    """blake2b-64 per SMILES. Not for security -- for counting exact duplicates across 101.6M
    strings without holding 101.6M Python objects."""
    out = np.empty(off.size - 1, dtype=np.uint64)
    mv = memoryview(blob)
    for k in range(off.size - 1):
        d = hashlib.blake2b(mv[off[k]:off[k + 1]], digest_size=8).digest()
        out[k] = int.from_bytes(d, "little")
    return out


def extract(i: int, members: dict[str, np.ndarray], starts: np.ndarray, sizes: np.ndarray,
            data: Path, out_dir: Path, force: bool) -> dict:
    """One shard: verify its row order, keep the split's rows, write the .npz."""
    dest = out_dir / f"shard_{i:05d}.npz"
    t0 = time.perf_counter()

    local = {k: S.to_local(v, i, starts, sizes) for k, v in members.items()}
    rows = np.unique(np.concatenate([v for v in local.values() if v.size])) \
        if any(v.size for v in local.values()) else np.zeros(0, np.int32)

    if dest.exists() and not force:
        try:
            with np.load(dest) as z:
                if z["rows"].shape == rows.shape and np.array_equal(z["rows"], rows):
                    return dict(shard=i, n_kept=int(rows.size), cached=True,
                                secs=time.perf_counter() - t0)
        except Exception:
            pass                        # unreadable or stale -> redo

    pf = pq.ParquetFile(_fs().open(shard_path(i), "rb"))
    n_rows = pf.metadata.num_rows
    if n_rows != int(sizes[i]):
        raise RuntimeError(
            f"shard {i}: the file now has {n_rows:,} rows but the split was drawn over a "
            f"manifest saying {int(sizes[i]):,}. Every global index past this shard is shifted. "
            f"Do not extract; re-download the scores and re-freeze the split.")

    tab = pf.read(columns=["smiles", "score"])
    sc = tab["score"].to_numpy()
    if sc.shape[0] != n_rows:
        raise RuntimeError(f"shard {i}: read {sc.shape[0]:,} rows, footer says {n_rows:,}. "
                           f"A short read must not be stored.")

    # --- the row-order gate; see the module docstring ---
    with np.load(data / "scores" / f"shard_{i:05d}.npz") as z:
        q_ref = z["q"]
    q_now = np.rint(np.asarray(sc, dtype=np.float64) * 100).astype(np.int64)
    if q_ref.shape[0] != n_rows:
        raise RuntimeError(f"shard {i}: stored score array has {q_ref.shape[0]:,} entries, "
                           f"file has {n_rows:,}.")
    bad = np.flatnonzero(q_now != q_ref.astype(np.int64))
    if bad.size:
        k = bad[0]
        raise RuntimeError(
            f"shard {i}: THE UPSTREAM FILE NO LONGER MATCHES THE SPLIT. {bad.size:,} of "
            f"{n_rows:,} scores differ; first at local row {k:,} "
            f"(stored {q_ref[k]/100:.2f}, file now {q_now[k]/100:.2f}). The split is a set of "
            f"row positions, so a re-ordered or re-uploaded shard makes every index point at a "
            f"different molecule. Refusing to extract.")

    smis = tab["smiles"].to_pylist()
    kept = [smis[int(r)] for r in rows]
    if any(("\n" in s or "\r" in s) for s in kept):
        raise RuntimeError(f"shard {i}: a SMILES contains a newline; the blob encoding assumes "
                           f"none and phase 2 would mis-split the file.")
    enc = [s.encode() for s in kept]
    off = np.concatenate([[0], np.cumsum([len(e) for e in enc])]).astype(np.int64)
    blob = b"".join(enc)

    pos = {}
    for k, v in local.items():
        pos[f"{k}_pos"] = np.searchsorted(rows, v).astype(np.int32) if v.size \
            else np.zeros(0, np.int32)
        if v.size and not np.array_equal(rows[pos[f"{k}_pos"]], v):
            raise RuntimeError(f"shard {i}: {k} membership does not round-trip through `rows`.")

    payload = dict(rows=rows, smi_blob=np.frombuffer(blob, dtype=np.uint8), smi_off=off,
                   q=q_ref[rows], h=hash64(blob, off), **pos)
    tmp = dest.with_name(dest.stem + ".part.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(dest)
    return dict(shard=i, n_rows=n_rows, n_kept=int(rows.size), bytes=int(dest.stat().st_size),
                n_test=int(local["test"].size), cached=False, secs=time.perf_counter() - t0)


def say(*a) -> None:
    """Unbuffered. On a box this log is shipped to S3 every 30 s and its MODIFICATION TIME is
    the only heartbeat the fleet watcher has; a buffered write makes a working box look stalled
    and a stalled box look fine for as long as the buffer holds."""
    print(*a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=int, default=0, help="first shard, inclusive")
    ap.add_argument("--hi", type=int, default=N_SHARDS, help="last shard, exclusive")
    ap.add_argument("--data", type=Path, default=S.DATA)
    ap.add_argument("--out", type=Path, default=Path("data/lsd_ampc/smiles"))
    ap.add_argument("--s3", default=os.environ.get("SCREEN_S3", ""),
                    help="s3://bucket/prefix to upload each shard to as it completes")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    prov = provenance.collect()
    say(json.dumps(prov, indent=1))
    a.out.mkdir(parents=True, exist_ok=True)

    spec = S.load_spec(a.data)
    say(f"\nrebuilding and verifying the frozen split "
        f"({spec['n_test']:,} test + {len(spec['replicates'])} replicates)")
    members = S.rebuild(spec, data=a.data, log=say)
    sizes, starts, n = S.geometry(a.data)
    say(f"  global index space {n:,} over {sizes.size} shards")

    rows_out, t0 = [], time.perf_counter()
    for k, i in enumerate(range(a.lo, a.hi), 1):
        r = extract(i, members, starts, sizes, a.data, a.out, a.force)
        rows_out.append(r)
        el = time.perf_counter() - t0
        size = "cached" if r.get("cached") else f"{r['bytes']/1e6:5.1f} MB"
        say(f"  [{k:3d}/{a.hi-a.lo}] shard {i:3d}  kept {r['n_kept']:>8,}  {size}"
            f"  {r['secs']:5.1f}s | {el/60:5.1f} min elapsed,"
            f" {(a.hi-a.lo-k)*el/k/60:5.1f} min left")
        if a.s3 and not r.get("cached"):
            dest = a.out / f"shard_{i:05d}.npz"
            if os.system(f"aws s3 cp {dest} {a.s3.rstrip('/')}/shard_{i:05d}.npz "
                         f"--only-show-errors") != 0:
                raise RuntimeError(f"shard {i}: extracted but the S3 upload failed. The box is "
                                   f"the only copy; failing now rather than at shutdown.")

    kept = sum(r["n_kept"] for r in rows_out)
    say(f"\n{len(rows_out)} shards, {kept:,} molecules kept in "
        f"{(time.perf_counter()-t0)/60:.1f} min")

    # COMPLETION FROM ACHIEVED WORK, not from the files existing. Every global index in this
    # shard range must have been kept exactly once: `rows` is a union, so the expected total is
    # the number of DISTINCT indices landing in [lo, hi), which is recomputed here from the
    # verified split rather than trusted from the loop above.
    lo_g, hi_g = int(starts[a.lo]), int(starts[a.hi - 1]) + int(sizes[a.hi - 1])
    want = np.unique(np.concatenate(
        [v[(v >= lo_g) & (v < hi_g)] for v in members.values()])).size
    if kept != want:
        raise RuntimeError(
            f"INCOMPLETE: shards {a.lo}..{a.hi-1} hold {want:,} distinct split indices but "
            f"{kept:,} molecules were kept. A missing or stale shard file would make phase 2 "
            f"score a test set with holes in it, so this is fatal rather than a warning.")
    say(f"completion verified: {kept:,} == {want:,} distinct split indices in shards "
        f"{a.lo}..{a.hi-1}")
    man = a.out / f"manifest_{a.lo:05d}_{a.hi:05d}.json"
    man.write_text(json.dumps({"lo": a.lo, "hi": a.hi, "n_kept": kept,
                               "provenance": prov, "shards": rows_out}, indent=1))
    if a.s3:
        os.system(f"aws s3 cp {man} {a.s3.rstrip('/')}/{man.name} --only-show-errors")
    say(f"manifest -> {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
