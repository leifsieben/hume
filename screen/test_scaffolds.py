"""Bemis-Murcko scaffold hashes for every molecule in the frozen 100M test set.

    .venv/bin/python screen/test_scaffolds.py --workers 10

Writes one file per shard, `data/lsd_ampc/scaf/shard_{i:05d}.npz` holding `h` (uint64 scaffold
hash) and `ok` (bool, RDKit parsed and Murcko succeeded), in `test_pos` order -- the same
positional order phase 2 wrote its predictions in and phase 3 reads its labels in. That ordering
is the only thing relating the three arrays, so the shape is asserted against the shard's own
`test_pos` count on write AND on read.

WHY THIS EXISTS. `screen/aggregate.py` declined the scaffold-filtered metric on a cost estimate
of "about 14 box-hours of Murcko computation". That estimate was wrong twice over: it quoted
490 us/mol, where the measured rate on this test set is 112.6 us/mol serial, and it counted
CORE-hours as box-hours while `collect_trainval.scaffold_hashes` already runs a process pool.
The real job is 3.1 core-hours -- about 20 minutes on ten workers -- for the panel that carries
the paper's honest number. See `figures/src/fig_e.py` on why the scaffold panel is the honest one.

WHY PER-SHARD FILES RATHER THAN ONE ARRAY. 246 shards at ~406k molecules is 20 minutes of work
that must survive a laptop lid, a SIGINT, and a second thought about the worker count. A shard
already written is skipped, so a re-run costs only what is left -- and completion is decided by
re-reading the files and checking their lengths against the shards, never by a marker or by a
count of files in a directory.

THE HASH FUNCTION IS IMPORTED FROM `collect_trainval`, NOT REIMPLEMENTED. The train side's
hashes are frozen in `trainval.npz` as `train_{r}_scaf` and were produced by that function; a
second blake2b call written out by hand here would only need to differ by a byte for every test
molecule to look scaffold-novel and the metric to read 100%. There is no assertion that can
catch that after the fact, so the call site is shared.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_trainval import scaffold_hashes                           # noqa: E402


def say(*a) -> None:
    print(*a, flush=True)


def shard_smiles(path: Path) -> list[str]:
    """-> the shard's test molecules, in `test_pos` order."""
    with np.load(path) as z:
        blob = z["smi_blob"].tobytes()
        off = z["smi_off"]
        pos = z["test_pos"]
    return [blob[off[p]:off[p + 1]].decode() for p in pos]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles-dir", type=Path, default=Path("data/lsd_ampc/smiles"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/lsd_ampc/scaf"))
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=246)
    ap.add_argument("--workers", type=int, default=10,
                    help="processes for the Murcko pass; leave a couple of cores for the OS")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for i in range(a.lo, a.hi):
        src = a.smiles_dir / f"shard_{i:05d}.npz"
        if not src.exists():
            sys.exit(f"{src} is missing; phase 0 did not write shard {i}. The test set is frozen, "
                     f"so a missing shard is a lost shard, not one to skip.")
        todo.append(i)
    done = [i for i in todo if (a.out_dir / f"shard_{i:05d}.npz").exists()]
    todo = [i for i in todo if i not in set(done)]
    say(f"{len(done)} shards already hashed, {len(todo)} to go, {a.workers} workers")

    t0, n_tot = time.perf_counter(), 0
    for k, i in enumerate(todo):
        smis = shard_smiles(a.smiles_dir / f"shard_{i:05d}.npz")
        h, ok = scaffold_hashes(smis, a.workers)
        if h.size != len(smis) or ok.size != len(smis):
            sys.exit(f"shard {i}: hashed {h.size:,} of {len(smis):,} test molecules. The join to "
                     f"the labels and the predictions is positional, so a short array would "
                     f"shift every scaffold against its own molecule. Refusing to write it.")
        # Write to a temporary name and rename, so a shard file that EXISTS is a shard file that
        # is COMPLETE. A partial .npz left by an interrupt would otherwise be skipped by the
        # resume check above and silently truncate the metric.
        # (np.savez is handed an open file, not a path: given a path it appends `.npz` unless
        # the name already ends in it, which silently wrote `.part.npz` and left the rename
        # pointing at nothing.)
        tmp = a.out_dir / f".shard_{i:05d}.part.npz"
        with open(tmp, "wb") as fh:
            np.savez(fh, h=h, ok=ok)
        tmp.replace(a.out_dir / f"shard_{i:05d}.npz")
        n_tot += h.size
        el = time.perf_counter() - t0
        rate = el / n_tot * 1e6
        say(f"  shard {i:5d}  {h.size:>8,} mols  parsed {ok.mean():.5f}  "
            f"{rate:6.1f} us/mol  {el/60:5.1f} min elapsed  "
            f"eta {(len(todo)-k-1) * (el/(k+1))/60:5.1f} min")
    say(f"-> {a.out_dir}  ({n_tot:,} molecules hashed this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
