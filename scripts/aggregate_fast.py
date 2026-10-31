"""Phase 3, with the label-side top-k masks hoisted out of the loop.

IDENTICAL NUMBERS TO screen/aggregate.py, BY CONSTRUCTION: it imports screen.metrics and calls
the same primitives. The only change is WHEN they are called.

`metrics.recovery(y_true, pred, frac)` computes `pessimistic_top_mask(-y_true, k)` on every
call, and `log_auc` computes it again. That mask depends only on the labels and on k -- it is
the same array for every arm and every replicate -- so over 6 arms x 5 replicates x 2 fracs it
is recomputed about ninety times. Each one is a stable argsort of 100,000,000 float32. A
`sample` of the running aggregate.py showed it sitting in atimsort_float, which is what prompted
this.

So: compute the label masks once, then do per (arm, replicate) only the work that actually
depends on the predictions. The four lines of `recovery` that remain inlined here are checked
against `metrics.recovery` on a slice before the full run, so this cannot silently drift from
the module it is replacing.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "screen"))
import arms as A
import metrics as M
from aggregate import load_labels, load_preds, SCALE


def say(*a) -> None:
    print(*a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles-dir", type=Path, default=Path("data/lsd_ampc/smiles"))
    ap.add_argument("--preds", type=Path, default=Path("data/lsd_ampc/preds"))
    ap.add_argument("--out", type=Path, default=Path("results/screen_ampc.json"))
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=246)
    ap.add_argument("--replicates", type=int, default=5)
    a = ap.parse_args()

    want = [x for x in A.ALL_ARMS if (a.preds / f"{x}_shard_{a.lo:05d}.npz").exists()]
    say(f"arms: {' '.join(want)}")

    t0 = time.perf_counter()
    y, counts = load_labels(a.smiles_dir, a.lo, a.hi)
    say(f"labels: {y.size:,} test molecules ({(time.perf_counter()-t0)/60:.1f} min)")

    keep = np.ones(y.size, dtype=bool)
    for arm in want:
        _, ok = load_preds(a.preds, arm, a.lo, a.hi, counts, a.replicates)
        keep &= ok
        say(f"  mask {arm}: {int(ok.sum()):,} parseable "
            f"({(time.perf_counter()-t0)/60:.1f} min)")
    del _
    yk = y[keep]
    del y
    say(f"\nparseable in every arm: {yk.size:,} of {keep.size:,}")

    # ---- the hoisted part: everything that depends only on the labels -----------------------
    t = time.perf_counter()
    k4 = int(np.floor(1e-4 * yk.size))
    k2 = int(np.floor(1e-2 * yk.size))
    true_top4 = M.pessimistic_top_mask(-yk, k4)
    true_top2 = M.pessimistic_top_mask(-yk, k2)
    ideal = M.ideal_log_area(yk)
    say(f"label-side masks + ideal area: {(time.perf_counter()-t)/60:.1f} min "
        f"(computed ONCE, not {len(want)*a.replicates*3} times)")
    say(f"true top 0.01%: {int(true_top4.sum()):,} molecules (k={k4:,})")

    # Equivalence check on a slice: the inlined recovery must equal metrics.recovery.
    sl = slice(0, 2_000_000)
    rng = np.random.default_rng(0)
    fake = rng.random(yk[sl].size).astype(np.float32)
    kk = int(np.floor(1e-4 * yk[sl].size))
    mine = float((M.pessimistic_top_mask(-yk[sl], kk) &
                  M.pessimistic_top_mask(-fake, kk)).sum() / kk)
    theirs = M.recovery(yk[sl], fake, 1e-4)
    assert abs(mine - theirs) < 1e-12, f"inlined recovery diverges: {mine} vs {theirs}"
    say(f"equivalence check vs metrics.recovery: {mine:.6f} == {theirs:.6f}  OK")

    out = {"dataset": "IrwinLab/LSD_AmpC_Liu_2025", "config": "docking_results",
           "n_test": int(keep.size), "n_test_scored": int(yk.size),
           "true_top_0.01pct": int(true_top4.sum()), "replicates": a.replicates,
           "w_fixed": A.W_FIXED, "arms": {}}
    for arm in want:
        P, _ok = load_preds(a.preds, arm, a.lo, a.hi, counts, a.replicates)
        P = P[keep]
        per = []
        for r in range(a.replicates):
            pr = P[:, r]
            la, _curve = M.log_auc(yk, pr, ideal=ideal)
            resid = pr - yk
            e = {"n": int(yk.size),
                 "recovery_0.01pct": float((true_top4 & M.pessimistic_top_mask(-pr, k4)).sum() / k4),
                 "recovery_1pct": float((true_top2 & M.pessimistic_top_mask(-pr, k2)).sum() / k2),
                 "log_auc_0.01pct": la,
                 "pearson": float(np.corrcoef(yk, pr)[0, 1]),
                 "rmse": float(np.sqrt((resid ** 2).mean())),
                 "rmse_over_sd": float(np.sqrt((resid ** 2).mean()) / yk.std())}
            per.append(e)
            say(f"  {arm:<17} rep{r}  recovery@0.01% {e['recovery_0.01pct']:.4f}  "
                f"logAUC {e['log_auc_0.01pct']:+.4f}  pearson {e['pearson']:.4f}  "
                f"| {(time.perf_counter()-t0)/60:.1f} min")
        del P
        agg = {}
        for key in per[0]:
            if key == "n":
                agg[key] = per[0][key]; continue
            v = np.array([p[key] for p in per], dtype=float)
            agg[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                        "min": float(v.min()), "max": float(v.max())}
        out["arms"][arm] = {"per_replicate": per, "summary": agg}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    say(f"\n-> {a.out}  ({(time.perf_counter()-t0)/60:.1f} min total)")
    say("=" * 78)
    say(f"{'arm':<18}{'recovery@0.01%':>18}{'logAUC':>12}{'pearson':>10}{'rmse/sd':>10}")
    for arm in sorted(want, key=lambda x: -out["arms"][x]["summary"]["recovery_0.01pct"]["mean"]):
        s = out["arms"][arm]["summary"]
        say(f"{arm:<18}{s['recovery_0.01pct']['mean']:>12.4f} "
            f"+-{s['recovery_0.01pct']['sd']:.4f}{s['log_auc_0.01pct']['mean']:>12.4f}"
            f"{s['pearson']['mean']:>10.4f}{s['rmse_over_sd']['mean']:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
