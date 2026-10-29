"""Phase 3: join phase 2's predictions to the true scores and report the screen's metrics.

Runs locally. Reads the per-(arm, shard) prediction files and the phase 0 shard files, which
between them hold everything needed: the predictions, the labels, and the ordering that relates
them.

THE JOIN IS BY POSITION WITHIN `test_pos`, AND NOTHING ELSE IS AVAILABLE TO CHECK IT. Phase 2
writes predictions in the order it read the SMILES, which is the order of `test_pos` in the same
shard file this reads the labels from, so index i of one is index i of the other by construction.
There is no molecule id in the prediction file to re-join on, which is deliberate -- storing one
would invite the two orderings to disagree and be silently reconciled. Instead the shapes are
asserted per shard: a prediction array whose length differs from that shard's test_pos count is
fatal, because a single off-by-one would shift every label against every prediction from that
point on and still produce a plausible-looking correlation.

THE PARSE MASK IS UNIONED ACROSS ARMS. A molecule that one arm could not featurize is dropped
from every arm, so no arm is scored on a molecule another arm never saw. The masks should be
identical -- RDKit parsing is deterministic -- so a disagreement is reported rather than absorbed.

SCAFFOLD FILTERING IS NOT DONE HERE, deliberately. Hall et al. apply the scaffold filter to the
recall curve of the 1,521 EXPERIMENTAL actives (their Figure 3), not to the top-0.01% docking
recovery that is this screen's headline. Scaffold-filtering 100,000,000 test molecules would cost
about 14 box-hours of Murcko computation for a comparison the paper does not make. The
experimental-actives analysis is a separate, tiny job against invitro_results.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A
import metrics as M

SCALE = 100.0


def say(*a) -> None:
    print(*a, flush=True)


def load_labels(smiles_dir: Path, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (true scores, per-shard test counts) for shards [lo, hi) in order."""
    ys, counts = [], []
    for i in range(lo, hi):
        with np.load(smiles_dir / f"shard_{i:05d}.npz") as z:
            pos = z["test_pos"]
            ys.append(z["q"][pos].astype(np.float32) / SCALE)
            counts.append(pos.size)
    return np.concatenate(ys), np.array(counts, dtype=np.int64)


def load_preds(pred_dir: Path, arm: str, lo: int, hi: int, counts: np.ndarray,
               n_rep: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (predictions (N, n_rep), parseable mask (N,)) for shards [lo, hi) in order."""
    total = int(counts.sum())
    P = np.empty((total, n_rep), dtype=np.float32)
    ok = np.empty(total, dtype=bool)
    at = 0
    for k, i in enumerate(range(lo, hi)):
        f = pred_dir / f"{arm}_shard_{i:05d}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing; arm {arm} did not finish shard {i}.")
        with np.load(f) as z:
            pr, o = z["pred"], z["ok"]
        if pr.shape != (counts[k], n_rep):
            raise RuntimeError(
                f"{f.name}: predictions are {pr.shape} but shard {i} holds {counts[k]:,} test "
                f"molecules and {n_rep} replicates were expected. The join is positional, so a "
                f"length mismatch would shift every label against every prediction from here on "
                f"and still look plausibly correlated. Refusing to aggregate.")
        P[at:at + counts[k]] = pr
        ok[at:at + counts[k]] = o
        at += int(counts[k])
    return P, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles-dir", type=Path, default=Path("data/lsd_ampc/smiles"))
    ap.add_argument("--preds", type=Path, default=Path("data/lsd_ampc/preds"))
    ap.add_argument("--out", type=Path, default=Path("results/screen_ampc.json"))
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=246)
    ap.add_argument("--replicates", type=int, default=5)
    a = ap.parse_args()

    want = a.arms or [x for x in A.ALL_ARMS
                      if (a.preds / f"{x}_shard_{a.lo:05d}.npz").exists()]
    if not want:
        sys.exit(f"no prediction files found in {a.preds}")
    say(f"arms: {' '.join(want)}")

    t0 = time.perf_counter()
    y, counts = load_labels(a.smiles_dir, a.lo, a.hi)
    say(f"labels: {y.size:,} test molecules over {counts.size} shards "
        f"({(time.perf_counter()-t0)/60:.1f} min)")
    say(f"  score range [{y.min():.2f}, {y.max():.2f}]  mean {y.mean():.2f}  sd {y.std():.2f}")

    # Union the parse masks first, so every arm is scored on exactly the same molecules.
    masks, keep = {}, np.ones(y.size, dtype=bool)
    for arm in want:
        _, ok = load_preds(a.preds, arm, a.lo, a.hi, counts, a.replicates)
        masks[arm] = ok
        keep &= ok
    disagree = {arm: int((masks[arm] != keep).sum()) for arm in want
                if int((masks[arm] != keep).sum())}
    say(f"\nparseable in every arm: {int(keep.sum()):,} of {keep.size:,} "
        f"(dropped {int((~keep).sum()):,})")
    if disagree:
        say(f"  NOTE: arms disagree about which molecules parse: {disagree}. RDKit parsing is "
            f"deterministic, so this should be empty; the union is used regardless.")

    yk = y[keep]
    ideal = M.ideal_log_area(yk)
    k_true = int(M.pessimistic_top_mask(-yk, int(1e-4 * yk.size)).sum())
    say(f"true top 0.01%: {k_true:,} molecules (k={int(1e-4*yk.size):,}; fewer when a tie "
        f"straddles the cut, which the labels' 0.01 quantization makes likely)")

    out = {"dataset": "IrwinLab/LSD_AmpC_Liu_2025", "config": "docking_results",
           "n_test": int(y.size), "n_test_scored": int(yk.size),
           "true_top_0.01pct": k_true, "replicates": a.replicates,
           "w_fixed": A.W_FIXED, "arms": {}}
    for arm in want:
        P, _ = load_preds(a.preds, arm, a.lo, a.hi, counts, a.replicates)
        per = []
        for r in range(a.replicates):
            e = M.evaluate(yk, P[keep, r], ideal=ideal)
            per.append(e)
            say(f"  {arm:<17} rep{r}  recovery@0.01% {e['recovery_0.01pct']:.4f}  "
                f"logAUC {e['log_auc_0.01pct']:+.4f}  pearson {e['pearson']:.4f}  "
                f"rmse/sd {e['rmse_over_sd']:.4f}")
        del P
        agg = {}
        for key in per[0]:
            if key == "n":
                agg[key] = per[0][key]; continue
            v = np.array([p[key] for p in per], dtype=float)
            agg[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                        "min": float(v.min()), "max": float(v.max())}
        out["arms"][arm] = {"per_replicate": per, "summary": agg}
        say(f"  {arm:<17} MEAN recovery@0.01% {agg['recovery_0.01pct']['mean']:.4f} "
            f"[{agg['recovery_0.01pct']['min']:.4f}, {agg['recovery_0.01pct']['max']:.4f}]\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    say(f"-> {a.out}")
    say("\n" + "=" * 78)
    say(f"{'arm':<18}{'recovery@0.01%':>18}{'logAUC':>12}{'pearson':>10}{'rmse/sd':>10}")
    say("=" * 78)
    for arm in sorted(want, key=lambda x: -out["arms"][x]["summary"]["recovery_0.01pct"]["mean"]):
        s = out["arms"][arm]["summary"]
        say(f"{arm:<18}{s['recovery_0.01pct']['mean']:>12.4f} "
            f"+-{s['recovery_0.01pct']['sd']:.4f}{s['log_auc_0.01pct']['mean']:>12.4f}"
            f"{s['pearson']['mean']:>10.4f}{s['rmse_over_sd']['mean']:>10.4f}")
    say("=" * 78)
    say("Lower is better for rmse/sd; higher for the rest. recovery@0.01% is the headline:")
    say("Hall et al. report 8.4% for their best AmpC model at 1M training molecules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
