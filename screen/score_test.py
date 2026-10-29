"""Phase 2: score the 100M-molecule test set with one arm's five models, over a shard range.

One (arm, shard range) per box. Reads the SMILES phase 0 parked in S3, featurizes in chunks,
predicts with all five replicate models, and writes predictions per shard.

PREDICTIONS ARE float32, NOT float16, AND THAT IS NOT A STORAGE OVERSIGHT. The headline metric is
the overlap between the true top 0.01% and the model's top 0.01% -- a RANKING of 10,000 molecules
out of 100,000,000, decided in a narrow band of scores around -70. float16's spacing at that
magnitude is 0.0625, coarser than the 0.01 quantization of the true scores, so it would collapse
genuinely distinct predictions into ties precisely where the measurement happens and precisely
among the molecules being measured. Halving the file size is not worth quantizing away the signal.
float32 over 100M x 5 replicates x 6 arms is about 12 GB in S3.

NO FEATURE MATRIX IS EVER STORED. 100M x 3,317 float32 would be 1.3 TB per arm. Each chunk is
featurized, predicted five times, and dropped.

THE TRUE SCORES ARE NOT COPIED HERE. They are already in the phase 0 files, and writing them
again per arm would store the same 100M labels six times and invite the two copies to disagree.
Phase 3 joins predictions to labels through the shard's own `test_pos` ordering, which both sides
read from the same file.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A
import provenance

CHUNK = 50_000


def say(*a) -> None:
    print(*a, flush=True)


def load_models(arm: str, model_dir: Path, replicates: int) -> list:
    import xgboost as xgb
    out = []
    for r in range(replicates):
        f = model_dir / f"{arm}_rep{r}.ubj"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} is missing. Scoring with fewer than {replicates} models would silently "
                f"report a replicate spread over the models that happened to be present.")
        m = xgb.XGBRegressor()
        m.load_model(str(f))
        out.append(m)
    return out


def read_test(path: Path) -> tuple[list[str], int]:
    """-> (test SMILES for this shard, in `test_pos` order, n_rows_kept)."""
    with np.load(path) as z:
        blob = z["smi_blob"].tobytes()
        off = z["smi_off"]
        pos = z["test_pos"]
        smis = [blob[off[p]:off[p + 1]].decode() for p in pos]
    return smis, len(smis)


def score_shard(arm: str, models: list, src: Path, dest: Path, force: bool) -> dict:
    smis, n = read_test(src)
    if dest.exists() and not force:
        try:
            with np.load(dest) as z:
                if z["pred"].shape == (n, len(models)):
                    return dict(shard_file=src.name, n=n, cached=True, secs=0.0)
        except Exception:
            pass                        # unreadable or stale -> redo

    t0 = time.perf_counter()
    pred = np.empty((n, len(models)), dtype=np.float32)
    ok = np.empty(n, dtype=bool)
    # THE BLOCK IS SIZED BY THE POOL, not by a fixed constant: one block is one full round of
    # every worker, so no core sits idle waiting for the next block to start. CHUNK is the floor.
    block = max(CHUNK, A.n_workers(arm) * A.slab_size(arm))
    n_cols = None
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        part = smis[lo:hi]
        if n_cols is None:
            n_cols = A.featurize(arm, part[:1]).shape[1]
        X = np.empty((hi - lo, n_cols), dtype=np.float32)
        A.featurize_many(arm, part, X)
        A.parse_mask_many(arm, part, ok[lo:hi])
        for j, m in enumerate(models):
            pred[lo:hi, j] = m.predict(X)
        del X
    if not np.isfinite(pred[ok]).all():
        raise RuntimeError(
            f"{src.name}: {int((~np.isfinite(pred[ok])).sum()):,} non-finite predictions on rows "
            f"that PARSE. A NaN sorts unpredictably, so it would quietly land somewhere in the "
            f"ranking the metric is computed from. Investigate before scoring.")
    tmp = dest.with_name(dest.stem + ".part.npz")
    np.savez_compressed(tmp, pred=pred, ok=ok)
    tmp.replace(dest)
    el = time.perf_counter() - t0
    return dict(shard_file=src.name, n=n, cached=False, secs=el,
                us_per_mol=el / n * 1e6, n_unparseable=int((~ok).sum()),
                pred_min=float(pred[ok].min()), pred_max=float(pred[ok].max()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(A.ALL_ARMS))
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=246)
    ap.add_argument("--smiles-dir", type=Path, default=Path("data/lsd_ampc/smiles"))
    ap.add_argument("--models", type=Path, default=Path("out/models"))
    ap.add_argument("--out", type=Path, default=Path("out/preds"))
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--s3", default=os.environ.get("SCREEN_S3_PREDS", ""))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    A.quiet_rdkit()
    prov = provenance.collect()
    say(json.dumps(prov, indent=1))
    say(f"\narm {a.arm}  shards [{a.lo}, {a.hi})  radius {A.SPEC[a.arm]['radius']}")
    models = load_models(a.arm, a.models, a.replicates)
    say(f"loaded {len(models)} models from {a.models}")
    a.out.mkdir(parents=True, exist_ok=True)

    rows, t0, done = [], time.perf_counter(), 0
    for k, i in enumerate(range(a.lo, a.hi), 1):
        src = a.smiles_dir / f"shard_{i:05d}.npz"
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing; phase 0 output is incomplete.")
        dest = a.out / f"{a.arm}_shard_{i:05d}.npz"
        r = score_shard(a.arm, models, src, dest, a.force)
        rows.append(r); done += r["n"]
        el = time.perf_counter() - t0
        rate = "cached" if r["cached"] else f"{r['us_per_mol']:6.1f} us/mol"
        say(f"  [{k:3d}/{a.hi-a.lo}] shard {i:3d}  {r['n']:>7,} mols  {rate}"
            f"  | {el/60:6.1f} min, {(a.hi-a.lo-k)*el/k/60:6.1f} min left")
        if a.s3 and not r["cached"]:
            if os.system(f"aws s3 cp {dest} {a.s3.rstrip('/')}/{dest.name} --only-show-errors"):
                raise RuntimeError(f"shard {i}: scored but the S3 upload failed.")

    man = a.out / f"{a.arm}_preds_{a.lo:05d}_{a.hi:05d}.json"
    man.write_text(json.dumps({"arm": a.arm, "lo": a.lo, "hi": a.hi, "n_scored": done,
                               "replicates": a.replicates, "provenance": prov,
                               "shards": rows}, indent=1))
    if a.s3:
        os.system(f"aws s3 cp {man} {a.s3.rstrip('/')}/{man.name} --only-show-errors")
    say(f"\n{len(rows)} shards, {done:,} molecules scored in "
        f"{(time.perf_counter()-t0)/60:.1f} min -> {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
