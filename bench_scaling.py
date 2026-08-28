"""How HUME's end-to-end cost scales across cores. Processes, not threads.

WHY PROCESSES. `featurize_all` is SMILES -> ECFP + 864 columns with no shared state between
molecules, so the parallel form is a shard per worker and nothing else. The GIL makes threads
pointless for the Python half of the path (parse, ToBinary, the ring CSR loop) which is 33% of
the cost, so a thread pool would scale only the C++ part.

FOUR THINGS THIS HARNESS HAS TO GET RIGHT, each of which would otherwise produce a wrong number
that looks plausible:

  * WORKER STARTUP IS NOT MEASURED. Importing rdkit + hume is ~1 s per worker and would swamp
    everything at high W. Each pool runs a warm-up task on every worker BEFORE the timed map.
  * THE THREAD CAPS ARE SET BEFORE THE POOL IS CREATED, so spawned children inherit them. W
    workers each running a multithreaded BLAS would oversubscribe and the curve would bend down
    for a reason that has nothing to do with the question.
  * SHARDS ARE ROUND-ROBIN BY INDEX, not contiguous blocks. This corpus's cost distribution is
    heavy-tailed (median ~5 ms, max ~35 s), so a contiguous split can hand one worker a shard
    that costs twice the mean and the wall clock then measures load imbalance rather than
    scaling. Round-robin balances in expectation; the imbalance that remains is REPORTED
    (max worker time / mean worker time) rather than hidden.
  * RESULTS ARE NOT RETURNED. Each worker returns a checksum and its own wall time, not its
    (n, 1266) matrix. Pickling 243 MB back through a pipe would make this a benchmark of
    multiprocessing's IPC. That is a real cost for a naive caller and it is NOT what "does the
    compute scale" asks; see the note printed at the end.

Arms are ROTATED per repetition and speedup is taken against the W=1 arm OF THE SAME REPETITION,
because this box is not quiet -- a foreign process is holding ~1 core throughout -- and paired
differencing is the only way that contention cancels instead of accumulating.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RDKIT_NUM_THREADS"):
    os.environ[v] = "1"

import sys, time, statistics, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

N = 24000
REPS = 3
WS = [1, 2, 3, 4, 6, 8, 10, 12]
CORPUS = "/Users/lsieben/VSCode/universal-encoder/cpp/hard.smi"


def _init():
    import hume  # noqa: F401
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _work(shard):
    import time as _t
    import hume
    t0 = _t.perf_counter()
    fp, X, _ = hume.featurize_all(shard)
    dt = _t.perf_counter() - t0
    # a checksum, so the compute cannot be optimised or skipped, and nothing large crosses back
    return float(np.nansum(X[:, ::97])) + float(fp.sum()), dt, len(shard)


def main():
    rng = np.random.default_rng(0)
    smis = open(CORPUS).read().split()
    smis = [smis[i] for i in rng.choice(len(smis), N, replace=False)]
    print(f"{N} molecules of cpp/hard.smi | M4 Pro: 8 performance + 4 efficiency cores\n"
          f"HUME default (qed off, AvgIpc on)\n", flush=True)

    acc = {w: [] for w in WS}
    imb = {w: [] for w in WS}
    for rep in range(REPS):
        order = WS[rep % len(WS):] + WS[:rep % len(WS)]
        for w in order:
            shards = [smis[i::w] for i in range(w)]
            with ProcessPoolExecutor(max_workers=w, initializer=_init,
                                     mp_context=mp.get_context("spawn")) as ex:
                list(ex.map(_work, [smis[:8]] * w))          # warm every worker
                la = os.getloadavg()[0]
                t0 = time.perf_counter()
                out = list(ex.map(_work, shards))
                wall = time.perf_counter() - t0
            us = wall / N * 1e6
            wt = [o[1] for o in out]
            acc[w].append(us)
            imb[w].append(max(wt) / statistics.mean(wt))
            print(f"  rep {rep}  W={w:2d}  {us:7.1f} us/mol   wall {wall:6.2f}s   "
                  f"imbalance {max(wt)/statistics.mean(wt):.2f}x   load1 {la:.2f}", flush=True)

    print(f"\n  {'W':>3s}  {'us/mol':>9s}  {'sd':>6s}  {'mol/s':>8s}  {'speedup':>8s}  "
          f"{'efficiency':>10s}  {'imbalance':>9s}")
    base = acc[1]
    for w in WS:
        v = acc[w]
        sp = statistics.mean([b / x for b, x in zip(base, v)])      # paired, per repetition
        print(f"  {w:3d}  {statistics.mean(v):9.1f}  {statistics.stdev(v):6.1f}  "
              f"{1e6/statistics.mean(v):8.0f}  {sp:7.2f}x  {sp/w*100:9.0f}%  "
              f"{statistics.mean(imb[w]):8.2f}x")
    print("\n  NOT INCLUDED: returning the (n, 1266) matrices to the parent. Workers return a "
          "checksum.\n  A caller that pickles results back through a pipe pays for that "
          "separately; the fix is\n  sharded output, not a different pool.")


if __name__ == "__main__":
    main()
