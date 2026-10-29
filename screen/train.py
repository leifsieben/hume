"""Phase 1b: train one arm's five XGBoost models on the frozen train/validation sets.

One arm per box, five replicates per arm. Writes a model per replicate plus a record of what
early stopping chose, and uploads both.

MEMORY IS THE CONSTRAINT, NOT TIME. A HUME training matrix is 1,000,000 x 3,317 float32 = 13.3
GB, and XGBoost then builds its own quantized copy. Featurization therefore fills a PREALLOCATED
array in chunks rather than hstacking a list of blocks, which would transiently hold two full
copies and kill the box at the worst possible moment -- after an hour of featurizing. Train on a
64 GB instance; a 32 GB one does not fit the HUME arms.

EARLY STOPPING IS THE PAPER'S, APPLIED IDENTICALLY TO EVERY ARM. Hall et al.: "early stopping
based on validation RMSE loss, with a patience of 5 epochs". n_estimators=300 is a cap, not a
count, and the chosen iteration is recorded per replicate so a reader can see whether an arm ran
out of budget rather than converging.
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


def load_blob(z, prefix: str) -> tuple[list[str], np.ndarray]:
    blob = z[f"{prefix}_blob"].tobytes()
    off = z[f"{prefix}_off"]
    y = z[f"{prefix}_y"]
    smis = [blob[off[i]:off[i + 1]].decode() for i in range(off.size - 1)]
    if len(smis) != y.size:
        raise RuntimeError(f"{prefix}: {len(smis):,} SMILES but {y.size:,} labels.")
    return smis, y


def featurize_into(arm: str, smis: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (X, parseable mask). Fills a preallocated array in chunks; see the module docstring."""
    n = len(smis)
    probe = A.featurize(arm, smis[:8])
    n_cols = probe.shape[1]
    X = np.empty((n, n_cols), dtype=np.float32)
    ok = np.empty(n, dtype=bool)
    t0 = time.perf_counter()
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        part = smis[lo:hi]
        X[lo:hi] = A.featurize(arm, part)
        ok[lo:hi] = A.parse_mask(part)
        if lo and lo % (CHUNK * 8) == 0:
            el = time.perf_counter() - t0
            say(f"    {label} {hi:>9,}/{n:,}  {el/60:5.1f} min  "
                f"{(n-hi)*el/hi/60:5.1f} min left  {el/hi*1e6:6.1f} us/mol")
    say(f"    {label} done {n:,} x {n_cols}  {(time.perf_counter()-t0)/60:.1f} min  "
        f"unparseable {int((~ok).sum()):,}")
    return X, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(A.ALL_ARMS))
    ap.add_argument("--trainval", type=Path, default=Path("data/lsd_ampc/trainval.npz"))
    ap.add_argument("--out", type=Path, default=Path("out/models"))
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--s3", default=os.environ.get("SCREEN_S3_MODELS", ""))
    a = ap.parse_args()

    A.quiet_rdkit()
    # The radius must be ASSERTED, not assumed: the HUME arms pass fp_radius explicitly, so this
    # catches a tree whose molhume cannot honor the request rather than one whose default differs.
    expect = A.SPEC[a.arm]["radius"]
    prov = provenance.collect()
    say(json.dumps(prov, indent=1))
    say(f"\narm {a.arm}  fp_block {A.SPEC[a.arm]['fp_block']}  "
        f"radius {expect}  feature_weights w={A.W_FIXED}")

    a.out.mkdir(parents=True, exist_ok=True)
    recs = []
    with np.load(a.trainval) as z:
        for r in range(a.replicates):
            say(f"\n--- replicate {r} ---")
            tr_smis, tr_y = load_blob(z, f"train_{r}")
            va_smis, va_y = load_blob(z, f"val_{r}")
            say(f"  train {len(tr_smis):,}  val {len(va_smis):,}")

            Xtr, ok_tr = featurize_into(a.arm, tr_smis, "train")
            Xva, ok_va = featurize_into(a.arm, va_smis, "val")
            # Drop unparseable rows from TRAINING only. They are a handful, they are identical
            # across arms (parse_mask is deterministic), and a molecule no arm can featurize is
            # not a training example. The test set keeps them and phase 3 excludes them there.
            Xtr, tr_y = Xtr[ok_tr], tr_y[ok_tr]
            Xva, va_y = Xva[ok_va], va_y[ok_va]

            m = A.make_head(a.arm, Xtr.shape[1], seed=r)
            t0 = time.perf_counter()
            m.fit(Xtr, tr_y, eval_set=[(Xva, va_y)], verbose=False)
            fit_min = (time.perf_counter() - t0) / 60
            best_it = int(getattr(m, "best_iteration", m.n_estimators - 1))
            hist = m.evals_result()["validation_0"]["rmse"]
            say(f"  fit {fit_min:.1f} min  best_iteration {best_it} of {m.n_estimators}  "
                f"val rmse {hist[best_it]:.4f}")
            if best_it >= m.n_estimators - 1:
                say(f"  NOTE: early stopping never triggered -- this arm used the whole "
                    f"{m.n_estimators}-tree budget, so the cap, not the validation curve, "
                    f"decided where it stopped.")

            f = a.out / f"{a.arm}_rep{r}.ubj"
            m.save_model(str(f))
            recs.append(dict(arm=a.arm, replicate=r, n_train=int(tr_y.size),
                             n_val=int(va_y.size), n_cols=int(Xtr.shape[1]),
                             best_iteration=best_it, n_estimators=int(m.n_estimators),
                             val_rmse=float(hist[best_it]),
                             val_rmse_final=float(hist[-1]),
                             early_stopped=bool(best_it < m.n_estimators - 1),
                             w_fixed=float(A.W_FIXED), fit_minutes=round(fit_min, 2),
                             y_train_sd=float(tr_y.std()), model=f.name))
            del Xtr, Xva
            if a.s3:
                if os.system(f"aws s3 cp {f} {a.s3.rstrip('/')}/{f.name} --only-show-errors"):
                    raise RuntimeError(f"replicate {r}: model trained but the S3 upload failed. "
                                       f"The box is the only copy; failing now.")

    rep = a.out / f"{a.arm}_train.json"
    rep.write_text(json.dumps({"arm": a.arm, "provenance": prov,
                               "head": "bench_downstream.make_head + early_stopping_rounds=5",
                               "w_fixed": A.W_FIXED, "replicates": recs}, indent=1))
    if a.s3:
        os.system(f"aws s3 cp {rep} {a.s3.rstrip('/')}/{rep.name} --only-show-errors")
    say(f"\n{len(recs)} models -> {a.out}")
    say(f"val rmse by replicate: " +
        "  ".join(f"r{d['replicate']}={d['val_rmse']:.4f}" for d in recs))
    if len(recs) != a.replicates:
        raise RuntimeError(f"only {len(recs)} of {a.replicates} models trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
