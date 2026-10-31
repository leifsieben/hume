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

SCAFFOLD FILTERING IS DONE BY `--scaffold`, AS A SECOND PASS THAT MERGES INTO AN EXISTING
RESULTS FILE. It used to be declined here outright, on a cost estimate that was wrong twice: it
quoted 490 us/mol where the measured rate on this test set is 112.6, and it counted CORE-hours as
box-hours when `collect_trainval.scaffold_hashes` already runs a process pool. The real Murcko
pass is 3.1 core-hours, about 20 minutes on ten workers, and lives in `screen/test_scaffolds.py`.

A SECOND PASS RATHER THAN A FLAG ON THE FIRST, because the unfiltered numbers are published and
must not move when the filtered ones are added. `--scaffold` reads the existing --out, computes
only the `*_scaffold` metrics, and writes the merged object back; the unfiltered values are
copied through untouched, and the pass REFUSES to run if the file's arms, replicate count or
scored-molecule count do not match what it is about to measure.

THE MASK IS THE UNION OVER ALL REPLICATES (Leif 2026-10-31): a test molecule counts as
scaffold-split only if its Bemis-Murcko scaffold appears in NO replicate's training set. The
alternative -- each replicate judged against its own training set -- is the more defensible
per-model definition, but the five scaffold sets overlap much less than their molecule-level 13%
suggests (about 404,000 scaffolds each, 1,235,646 in the union, 59,532 common to all five), so
it would score the five replicates on five different populations and turn the min-max whisker
into a mixture of model variance and population variance. The union fixes one population for
every arm and every replicate, at the price of being pessimistic: each model faces molecules
about 3x more novel than its own training required. It is a floor on generalisation.

Note this is NOT the filter Hall et al. apply in their Figure 3, which is over the recall curve
of the 1,521 EXPERIMENTAL actives. That remains a separate, tiny job against invitro_results.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A
import metrics as M
import split as S

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


def load_ok(pred_dir: Path, arm: str, lo: int, hi: int, counts: np.ndarray) -> np.ndarray:
    """-> the parseable mask alone, without touching the predictions.

    The union-of-masks pass needs only `ok`, and `load_preds` was being called for it -- which
    read every prediction array a second time, 10 GB of float32 for 600 MB of bool. npz members
    are stored separately and decompressed on access, so asking for one is genuinely cheaper
    than asking for both. Same files, same order, same result.
    """
    ok = np.empty(int(counts.sum()), dtype=bool)
    at = 0
    for k, i in enumerate(range(lo, hi)):
        f = pred_dir / f"{arm}_shard_{i:05d}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing; arm {arm} did not finish shard {i}.")
        with np.load(f) as z:
            o = z["ok"]
        if o.shape != (counts[k],):
            raise RuntimeError(
                f"{f.name}: parse mask is {o.shape} but shard {i} holds {counts[k]:,} test "
                f"molecules. The join is positional; refusing to aggregate.")
        ok[at:at + counts[k]] = o
        at += int(counts[k])
    return ok


def load_scaffold_novel(scaf_dir: Path, trainval: Path, lo: int, hi: int, counts: np.ndarray,
                        n_rep: int) -> tuple[np.ndarray, int]:
    """-> (mask of test molecules whose scaffold is in NO replicate's training set, |union|).

    Both sides are blake2b-8 hashes of a Murcko scaffold SMILES produced by the SAME function,
    `collect_trainval._scaffold_hashes_serial`: the train side when the split was frozen, the
    test side by `screen/test_scaffolds.py`, which imports it rather than reimplementing it. If
    the two hash functions ever diverged, every test molecule would look scaffold-novel and this
    metric would read like the unfiltered one -- a failure with no symptom, which is why the
    call site is shared and not merely documented as identical.

    A molecule whose own Murcko scaffold could not be computed is NOT novel. It is excluded,
    because "we could not tell" must not be scored as "unlike anything in training".
    """
    if not scaf_dir.exists():
        raise SystemExit(
            f"no test-set scaffold hashes at {scaf_dir}.\n"
            f"  Compute them first (about 20 minutes on ten workers):\n"
            f"    .venv/bin/python screen/test_scaffolds.py --workers 10")
    if not trainval.exists():
        raise SystemExit(
            f"no frozen split at {trainval}; the train-side scaffold hashes live in it as\n"
            f"  train_{{r}}_scaf, written by screen/collect_trainval.py. Fetch it from S3:\n"
            f"    aws s3 cp s3://<bucket>/screen/trainval.npz {trainval}")
    h = np.empty(int(counts.sum()), dtype=np.uint64)
    ok = np.empty(int(counts.sum()), dtype=bool)
    at = 0
    for k, i in enumerate(range(lo, hi)):
        f = scaf_dir / f"shard_{i:05d}.npz"
        if not f.exists():
            raise SystemExit(f"{f} is missing; the scaffold pass did not finish shard {i}. "
                             f"Re-run screen/test_scaffolds.py -- it skips what it already has.")
        with np.load(f) as z:
            hh, oo = z["h"], z["ok"]
        if hh.shape != (counts[k],) or oo.shape != (counts[k],):
            raise RuntimeError(
                f"{f.name}: {hh.shape[0]:,} scaffold hashes but shard {i} holds {counts[k]:,} "
                f"test molecules. The join to the labels and the predictions is positional, so a "
                f"length mismatch would attribute every scaffold to the wrong molecule and still "
                f"produce a plausible-looking number. Refusing to aggregate.")
        h[at:at + counts[k]] = hh
        ok[at:at + counts[k]] = oo
        at += int(counts[k])
    with np.load(trainval) as z:
        missing = [f"train_{r}_scaf" for r in range(n_rep) if f"train_{r}_scaf" not in z.files]
        if missing:
            raise SystemExit(
                f"{trainval} has no {missing}; it was written before the scaffold hashes were "
                f"frozen into it, or for fewer than {n_rep} replicates. Available: "
                f"{sorted(k for k in z.files if 'scaf' in k)}")
        union = np.unique(np.concatenate([z[f"train_{r}_scaf"] for r in range(n_rep)]))
    novel = ok & ~S.isin_sorted(h, union)
    del h, ok
    return novel, int(union.size)


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
    ap.add_argument("--scaf-dir", type=Path, default=Path("data/lsd_ampc/scaf"),
                    help="per-shard test scaffold hashes from screen/test_scaffolds.py")
    ap.add_argument("--trainval", type=Path, default=Path("data/lsd_ampc/trainval.npz"),
                    help="frozen split; carries the train-side scaffold hashes")
    ap.add_argument("--scaffold", action="store_true",
                    help="second pass: compute ONLY the *_scaffold metrics and merge them into "
                         "an existing --out, leaving the unfiltered numbers untouched")
    a = ap.parse_args()

    want = a.arms or [x for x in A.ALL_ARMS
                      if (a.preds / f"{x}_shard_{a.lo:05d}.npz").exists()]
    if not want:
        sys.exit(f"no prediction files found in {a.preds}")
    say(f"arms: {' '.join(want)}")

    # THE CHEAP MERGE GUARDS RUN FIRST, before a line of data is read. The remaining one needs
    # the scored count and therefore the parse masks, so it cannot come up here -- but pointing
    # --scaffold at the wrong file, or at none, must not cost half an hour of loading to find out.
    prior = None
    if a.scaffold:
        if not a.out.exists():
            sys.exit(f"--scaffold merges into an existing {a.out}, and it does not exist. Run "
                     f"the unfiltered pass first (without --scaffold).")
        prior = json.loads(a.out.read_text())
        if int(prior.get("replicates", -1)) != a.replicates:
            sys.exit(f"{a.out} has {prior.get('replicates')} replicates, this pass has "
                     f"{a.replicates}. Refusing to merge.")
        absent = sorted(set(want) - set(prior.get("arms", {})))
        if absent:
            sys.exit(f"{a.out} has no unfiltered numbers for {absent}, so the scaffold metrics "
                     f"would be the only ones those arms carry and figure E would place them on "
                     f"a frontier the other arms are missing from. Re-run the unfiltered pass.")

    t0 = time.perf_counter()
    y, counts = load_labels(a.smiles_dir, a.lo, a.hi)
    say(f"labels: {y.size:,} test molecules over {counts.size} shards "
        f"({(time.perf_counter()-t0)/60:.1f} min)")
    say(f"  score range [{y.min():.2f}, {y.max():.2f}]  mean {y.mean():.2f}  sd {y.std():.2f}")

    # Union the parse masks first, so every arm is scored on exactly the same molecules.
    masks, keep = {}, np.ones(y.size, dtype=bool)
    for arm in want:
        masks[arm] = load_ok(a.preds, arm, a.lo, a.hi, counts)
        keep &= masks[arm]
    disagree = {arm: int((masks[arm] != keep).sum()) for arm in want
                if int((masks[arm] != keep).sum())}
    say(f"\nparseable in every arm: {int(keep.sum()):,} of {keep.size:,} "
        f"(dropped {int((~keep).sum()):,})")
    if disagree:
        say(f"  NOTE: arms disagree about which molecules parse: {disagree}. RDKit parsing is "
            f"deterministic, so this should be empty; the union is used regardless.")

    yk = y[keep]
    # Both are full-array sorts over 100,000,000 floats, and the scaffold pass replaces both
    # with the subset's own. Computing them anyway cost a couple of minutes to throw away.
    ideal = k_true = None
    if not a.scaffold:
        ideal = M.ideal_log_area(yk)
        k_true = int(M.pessimistic_top_mask(-yk, int(1e-4 * yk.size)).sum())
        say(f"true top 0.01%: {k_true:,} molecules (k={int(1e-4*yk.size):,}; fewer when a tie "
            f"straddles the cut, which the labels' 0.01 quantization makes likely)")

    # SFX IS THE WHOLE DIFFERENCE BETWEEN THE TWO PASSES. Everything below is written once and
    # runs on either the full kept set (`sub` is None) or the scaffold-split subset of it; the
    # metric keys, the ideal logAUC and the true-top-k count all follow from the population, so
    # deriving them from one `sub` is what keeps the two passes from drifting apart.
    sfx = "_scaffold" if a.scaffold else ""
    sub = None
    if a.scaffold:
        novel, n_union = load_scaffold_novel(a.scaf_dir, a.trainval, a.lo, a.hi, counts,
                                             a.replicates)
        sub = novel[keep]
        del novel
        ys = yk[sub]
        if ys.size == 0:
            sys.exit("every test molecule shares a scaffold with some training molecule; there "
                     "is nothing to score. That is not a plausible result for a 100M test set "
                     "against 1.2M training scaffolds -- suspect the hashes, not the chemistry.")
        ideal = M.ideal_log_area(ys)
        k_true_s = int(M.pessimistic_top_mask(-ys, int(1e-4 * ys.size)).sum())
        say(f"\nscaffold-split subset: {ys.size:,} of {yk.size:,} kept test molecules "
            f"({ys.size/yk.size:.2%}) have a Murcko scaffold in none of the {n_union:,} "
            f"scaffolds across all {a.replicates} training sets")
        say(f"  true top 0.01% within it: {k_true_s:,} molecules")
        # FAIL HERE, NOT INSIDE THE FIRST ARM. Pessimistic tie-breaking leaves the top 0.01%
        # EMPTY whenever the best tie block alone is larger than k, which the labels' 0.01
        # quantization makes a real possibility on a subset -- it is exactly what a 6-shard
        # slice does. The arm loop would hit it too, but only after loading 2 GB of predictions,
        # and the message would name a model for what is a property of the population.
        if k_true_s == 0:
            sys.exit(
                f"the true top 0.01% of the scaffold-split subset is EMPTY under pessimistic "
                f"tie-breaking: k={int(1e-4 * int(sub.sum())):,} over {int(sub.sum()):,} "
                f"molecules, but the best-scoring tie block alone is larger than that. The "
                f"metric is undefined for this population, not zero. Widen the shard range, or "
                f"report the subset at a coarser frac -- do not report it as 0.")
        say(f"  score range [{ys.min():.2f}, {ys.max():.2f}]  mean {ys.mean():.2f}  "
            f"sd {ys.std():.2f}  (unfiltered mean was {yk.mean():.2f})")
        del ys

    if a.scaffold:
        # MERGE, NOT OVERWRITE: the unfiltered numbers are published and must not move. This is
        # the last of the guards -- a file measured on a DIFFERENT TEST SET would otherwise merge
        # silently into one object whose two halves describe two experiments. The other guards
        # ran before any data was read.
        out = prior
        if int(out.get("n_test_scored", -1)) != int(yk.size):
            sys.exit(f"{a.out} reports {out.get('n_test_scored'):,} scored test molecules but "
                     f"this pass finds {yk.size:,}. The two halves of the merged file would "
                     f"describe different test sets. Refusing to merge.")
        out["scaffold_mask"] = ("union over all replicates: a test molecule is scaffold-split "
                                "only if its Bemis-Murcko scaffold is in NO training set")
        out["train_scaffolds_union"] = n_union
        out["n_test_scaffold_split"] = int(sub.sum())
        out["true_top_0.01pct_scaffold"] = k_true_s
    else:
        out = {"dataset": "IrwinLab/LSD_AmpC_Liu_2025", "config": "docking_results",
               "n_test": int(y.size), "n_test_scored": int(yk.size),
               "true_top_0.01pct": k_true, "replicates": a.replicates,
               "w_fixed": A.W_FIXED, "arms": {}}

    head = f"recovery_0.01pct{sfx}"
    for arm in want:
        P, _ = load_preds(a.preds, arm, a.lo, a.hi, counts, a.replicates)
        per = []
        for r in range(a.replicates):
            e = {f"{k}{sfx}": v for k, v in
                 M.evaluate(yk, P[keep, r], keep=sub, ideal=ideal).items()}
            per.append(e)
            say(f"  {arm:<17} rep{r}  recovery@0.01% {e[head]:.4f}  "
                f"logAUC {e['log_auc_0.01pct' + sfx]:+.4f}  "
                f"pearson {e['pearson' + sfx]:.4f}  rmse/sd {e['rmse_over_sd' + sfx]:.4f}")
        del P
        agg = {}
        for key in per[0]:
            if key == f"n{sfx}":
                agg[key] = per[0][key]; continue
            v = np.array([q[key] for q in per], dtype=float)
            agg[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                        "min": float(v.min()), "max": float(v.max())}
        if a.scaffold:
            for r, e in enumerate(per):
                out["arms"][arm]["per_replicate"][r].update(e)
            out["arms"][arm]["summary"].update(agg)
        else:
            out["arms"][arm] = {"per_replicate": per, "summary": agg}
        say(f"  {arm:<17} MEAN recovery@0.01%{sfx} {agg[head]['mean']:.4f} "
            f"[{agg[head]['min']:.4f}, {agg[head]['max']:.4f}]\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    say(f"-> {a.out}" + ("  (merged; unfiltered numbers untouched)" if a.scaffold else ""))
    say("\n" + "=" * 78)
    say(f"{'arm':<18}{'recovery@0.01%' + sfx:>18}{'logAUC':>12}{'pearson':>10}{'rmse/sd':>10}")
    say("=" * 78)
    for arm in sorted(want, key=lambda x: -out["arms"][x]["summary"][head]["mean"]):
        m = out["arms"][arm]["summary"]
        say(f"{arm:<18}{m[head]['mean']:>12.4f} "
            f"+-{m[head]['sd']:.4f}{m['log_auc_0.01pct' + sfx]['mean']:>12.4f}"
            f"{m['pearson' + sfx]['mean']:>10.4f}{m['rmse_over_sd' + sfx]['mean']:>10.4f}")
    say("=" * 78)
    say("Lower is better for rmse/sd; higher for the rest. recovery@0.01% is the headline:")
    say("Hall et al. report 8.4% for their best AmpC model at 1M training molecules.")
    if a.scaffold:
        say("These numbers are a FLOOR: every arm is scored on molecules about 3x more novel "
            "than its own training set required. See this module's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
