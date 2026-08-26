"""How many parallel UMA workers does this machine actually want?

torch CPU inference already uses intra-op threading, so N processes do not necessarily give
N x throughput — they may just contend. But one molecule per forward pass badly underutilises
the model, so process-level parallelism may still win. Measure before committing 9 hours.

Each worker is pinned to 1 thread; parallelism comes from running several molecules at once.
"""

from __future__ import annotations

import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

CONF = sorted((Path(__file__).resolve().parent / "data" / "uma100k" / "conformers").glob("*.pkl"))
PER_WORKER = 15


def _run(args):
    wid, recs, mode = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from ase import Atoms
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate import pretrained_mlip as pm

    # Default settings compile a fast path for FIXED composition/charge/spin (MD workloads).
    # Ours is the opposite: every molecule differs, so that path is thrown away each call and
    # fairchem itself recommends inference_settings="batch" for heterogeneous evaluation.
    kw = {"inference_settings": mode} if mode else {}
    pu = pm.get_predict_unit("uma-s-1p2", device="cpu", **kw)
    calc = FAIRChemCalculator(pu, task_name="omol")
    cap: dict = {}
    pu.model.module.backbone.register_forward_hook(
        lambda _m, _i, out: cap.__setitem__("node", out["node_embedding"].detach()))

    at = Atoms(numbers=recs[0]["Z"], positions=recs[0]["R"])
    at.info = {"charge": recs[0]["charge"], "spin": 1}
    at.calc = calc
    at.get_potential_energy()  # warm-up, excluded

    t0 = time.time()
    ok, err = 0, None
    for r in recs:
        try:
            a = Atoms(numbers=r["Z"], positions=r["R"])
            a.info = {"charge": r["charge"], "spin": 1}
            a.calc = calc
            a.get_potential_energy()
            _ = cap["node"][:, 0, :].mean(dim=0).float().cpu().numpy()
            ok += 1
        except Exception as e:
            err = err or f"{type(e).__name__}: {e}"
    return ok, time.time() - t0, err


if __name__ == "__main__":
    import sys

    modes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["", "batch"]
    workers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 3, 6]
    recs = pickle.load(CONF[0].open("rb"))
    print(f"{len(recs)} conformers available in shard 0\n")
    for mode in modes:
        for w in workers:
            batches = [(i, recs[i * PER_WORKER:(i + 1) * PER_WORKER], mode) for i in range(w)]
            t0 = time.time()
            with Pool(w) as pool:
                res = pool.map(_run, batches)
            wall = time.time() - t0
            n = sum(r[0] for r in res)
            compute = max(r[1] for r in res)
            label = f"mode={mode or 'default':8s} workers={w}"
            if n == 0:
                print(f"{label}: ALL FAILED -> {res[0][2]}")
                continue
            rate = n / compute
            print(f"{label}: {n} mols | {compute / (n / w) * 1000:.0f} ms/mol/worker "
                  f"| {rate:.2f} mol/s aggregate | load+wall {wall:.0f}s "
                  f"| 100k = {100_000 / rate / 3600:.1f} h")
