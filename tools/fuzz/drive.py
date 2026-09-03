"""Run the fuzz corpus in isolated subprocesses, bisect anything that kills one, aggregate.

    .venv/bin/python tools/fuzz/drive.py CORPUS.txt OUTDIR [SHARD] [JOBS] [THREADS]

WHY SUBPROCESSES. The failures worth finding are the ones an except block cannot see: a
std::abort or a SIGSEGV takes the interpreter with it. A shard that exits non-zero is therefore
the signal, and the driver bisects it -- halving until a single SMILES reproduces -- so the
report names the molecule rather than the shard.

A TIMEOUT IS ALSO A FINDING. Some constructed molecules are large enough that the O(n^3)
eigensolves run for minutes; a shard that does not finish is bisected the same way, and the
molecule is reported as slow rather than as crashing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
WORKER = str(ROOT / "tools" / "fuzz" / "worker.py")


def run_shard(smis: list[str], out: Path, threads: int, timeout: float):
    """-> (status, seconds). status: 'ok' | 'crash:<code>' | 'timeout'."""
    src = out.with_suffix(".smi")
    src.write_text("\n".join(smis), encoding="utf-8")
    t0 = time.time()
    try:
        r = subprocess.run([PY, WORKER, str(src), str(out), str(threads)],
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0
    dt = time.time() - t0
    if r.returncode != 0:
        return f"crash:{r.returncode}", dt
    return "ok", dt


def bisect(smis: list[str], tmp: Path, threads: int, timeout: float, depth: int = 0):
    """Halve until one SMILES reproduces the failure. -> (smiles, status) or None."""
    if len(smis) == 1:
        st, _ = run_shard(smis, tmp, threads, timeout)
        return (smis[0], st) if st != "ok" else None
    mid = len(smis) // 2
    for half in (smis[:mid], smis[mid:]):
        st, _ = run_shard(half, tmp, threads, timeout)
        if st != "ok":
            return bisect(half, tmp, threads, timeout, depth + 1)
    # Neither half alone reproduces: the failure needs the combination, or is not deterministic.
    return ("<not reproducible in either half>", "split")


def main() -> None:
    corpus, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    shard_n = int(sys.argv[3]) if len(sys.argv) > 3 else 25_000
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    outdir.mkdir(parents=True, exist_ok=True)

    smis = corpus.read_text(encoding="utf-8").split("\n")
    shards = [smis[i:i + shard_n] for i in range(0, len(smis), shard_n)]
    print(f"  {len(smis):,} SMILES in {len(shards)} shards of {shard_n:,}; "
          f"{jobs} jobs x {threads} threads", flush=True)

    bad: list[dict] = []
    done = 0

    def one(idx_shard):
        i, sh = idx_shard
        st, dt = run_shard(sh, outdir / f"s{i:04d}.json", threads, timeout=1800)
        return i, sh, st, dt

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, sh, st, dt in ex.map(one, list(enumerate(shards))):
            done += 1
            if st != "ok":
                print(f"  shard {i}: {st} after {dt:.0f}s -- bisecting {len(sh):,}", flush=True)
                hit = bisect(sh, outdir / f"bisect{i}.json", threads, timeout=300)
                if hit:
                    bad.append({"shard": i, "status": st, "smiles": hit[0], "how": hit[1]})
                    print(f"    >>> {hit[1]}: {hit[0][:120]!r}", flush=True)
            if done % 10 == 0 or done == len(shards):
                print(f"  {done}/{len(shards)} shards, {time.time() - t0:.0f}s elapsed",
                      flush=True)

    (outdir / "crashes.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")
    print(f"\n  shards done. {len(bad)} molecule(s) killed or hung a worker.", flush=True)
    aggregate(outdir)


def aggregate(outdir: Path) -> None:
    files = sorted(p for p in outdir.glob("s*.json"))
    if not files:
        print("  no shard results to aggregate"); return
    first = json.loads(files[0].read_text(encoding="utf-8"))
    cols = first["columns"]
    n_col = len(cols)
    tot = dict(n=0, rows_all_nan=0, warn_row=0, warn_col=0)
    nan = np.zeros(n_col, np.int64); pinf = np.zeros(n_col, np.int64)
    ninf = np.zeros(n_col, np.int64); zero = np.zeros(n_col, np.int64)
    lo = np.full(n_col, np.inf); hi = np.full(n_col, -np.inf); amax = np.zeros(n_col)
    msgs: dict[str, int] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for k in tot:
            tot[k] += d[k]
        nan += np.array(d["nan"], np.int64); pinf += np.array(d["posinf"], np.int64)
        ninf += np.array(d["neginf"], np.int64); zero += np.array(d["zero"], np.int64)
        lo = np.fmin(lo, np.array([np.inf if v is None else v for v in d["lo"]]))
        hi = np.fmax(hi, np.array([-np.inf if v is None else v for v in d["hi"]]))
        amax = np.fmax(amax, np.array(d["absmax"]))
        for k, v in d["messages"].items():
            msgs[k] = msgs.get(k, 0) + v
    n = max(tot["n"], 1)
    report = {
        "molecules": tot["n"], "shards": len(files),
        "rows_all_nan": tot["rows_all_nan"],
        "warn_row_batches": tot["warn_row"], "warn_col_batches": tot["warn_col"],
        "messages": dict(sorted(msgs.items(), key=lambda kv: -kv[1])),
        "columns": [
            {"name": cols[i], "nan_frac": round(float(nan[i]) / n, 6),
             "zero_frac": round(float(zero[i]) / n, 6),
             "inf": int(pinf[i] + ninf[i]),
             "min": None if not np.isfinite(lo[i]) else float(lo[i]),
             "max": None if not np.isfinite(hi[i]) else float(hi[i]),
             "absmax": float(amax[i]),
             "constant": bool(np.isfinite(lo[i]) and lo[i] == hi[i])}
            for i in range(n_col)],
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  aggregated {tot['n']:,} molecules -> {outdir/'report.json'}")


if __name__ == "__main__":
    main()
