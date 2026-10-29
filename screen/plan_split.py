"""Summarize the LSD AmpC score column, and measure what a split would actually cost to fetch.

Run after `fetch_scores.py`. Produces the three numbers the split design turns on, none of which
can be guessed from the plan:

1. THE EXACT THRESHOLDS. Scores are quantized to 0.01 kcal/mol, so a histogram over
   `round(score*100)` is not an approximation -- the top-0.1% cut it yields is the cut.
2. HOW CLUSTERED A ROW GROUP IS. Row groups hold 1,000 rows and some shards are score-sorted, so
   a sampled row group can be 1,000 molecules at one score. Sampling whole row groups is still
   unbiased (equal-size groups, drawn uniformly), but the EFFECTIVE sample size for anything
   score-dependent is closer to the number of groups than the number of rows, and that has to be
   stated rather than discovered later.
3. WHETHER THE GOOD SCORES CLUSTER. The stratified training draw wants 20% of its molecules from
   the top 1%. Fetching SMILES costs a whole row group however few rows are kept, so the cost of
   that draw is the NUMBER OF ROW GROUPS the top 1% touches, not the number of molecules in it.
   If the top 1% is spread thinly over 500k groups it is dearer than the entire test set.

Memory: shards are streamed one at a time. The outputs are a 6,301-bin histogram and per-row-group
min/max (about 6 MB), so every later decision is made against those instead of 3.1 GB of scores.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

SRC = Path("data/lsd_ampc/scores")
OUT = Path("data/lsd_ampc")
SCALE = 100
RG_ROWS = 1000                       # parquet row-group size in this release
# The histogram spans the WHOLE int16 range and is indexed by `q - Q0`, so bin 0 is -327.68 and
# ascending bin order is ascending score, i.e. best-first. Two assumptions died here: that scores
# stop near -80 kcal/mol (shard 47 holds -144.3) and that they are all negative (some are
# positive, which a `-q` index cannot represent at all). Neither is worth re-deriving from a
# constant, so the range is simply the storage limit and the implausible tail is reported.
Q0 = int(np.iinfo(np.int16).min)     # -32768 -> score -327.68
NBIN = int(np.iinfo(np.int16).max) - Q0 + 1


def main() -> int:
    man_path = OUT / "scores_manifest.json"
    if not man_path.exists():
        sys.exit(f"{man_path} is missing; run screen/fetch_scores.py first.")
    man = json.loads(man_path.read_text())
    shards = [s["shard"] for s in man["shards"]]
    if not man.get("complete"):
        print(f"WARNING: the score download is marked incomplete "
              f"({len(shards)} shards present). Thresholds below are over what is on disk.\n")

    hist = np.zeros(NBIN, dtype=np.int64)          # index = q - Q0; ascending index = best first
    rg_min, rg_max, rg_shard = [], [], []
    total = 0
    for k, sh in enumerate(sorted(shards), 1):
        p = SRC / f"shard_{sh:05d}.npz"
        with np.load(p) as z:
            q = z["q"]
        hist += np.bincount(q.astype(np.int64) - Q0, minlength=NBIN)
        total += q.shape[0]

        pad = (-q.shape[0]) % RG_ROWS                # last group is short
        qq = np.concatenate([q, np.full(pad, np.iinfo(np.int16).max, np.int16)])
        g = qq.reshape(-1, RG_ROWS)
        gmin = g.min(axis=1)
        gmax = np.where(g == np.iinfo(np.int16).max, np.iinfo(np.int16).min, g).max(axis=1)
        rg_min.append(gmin); rg_max.append(gmax)
        rg_shard.append(np.full(g.shape[0], sh, np.int16))
        if k % 50 == 0 or k == len(shards):
            print(f"  {k:3d}/{len(shards)} shards, {total:,} rows", flush=True)

    rg_min = np.concatenate(rg_min); rg_max = np.concatenate(rg_max)
    rg_shard = np.concatenate(rg_shard)
    n_rg = rg_min.shape[0]

    # ---- exact quantiles, from the histogram ------------------------------------------------
    cum = np.cumsum(hist)                            # cum[i] = rows with score <= (i+Q0)/SCALE
    def threshold(frac: float) -> float:
        """The score at rank floor(frac*total) counting from the most negative. Exact: the
        release quantizes to 0.01, so the histogram is the distribution, not a summary of it."""
        want = int(np.floor(frac * total))
        return (int(np.searchsorted(cum, want, side="left")) + Q0) / SCALE
    nz = np.flatnonzero(hist)
    fracs = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    mean = (hist * (np.arange(NBIN) + Q0)).sum() / total / SCALE
    print(f"\n{total:,} rows, {n_rg:,} row groups")
    print(f"scores {(nz[0]+Q0)/SCALE:.2f} .. {(nz[-1]+Q0)/SCALE:.2f}   mean {mean:.3f}   "
          f"{int(hist[-Q0+1:].sum()):,} rows score > 0\n")
    print("exact thresholds (score at or below which a molecule is in the top fraction):")
    thr = {}
    for f in fracs:
        t = threshold(f)
        thr[f] = t
        print(f"  top {f*100:7.3f}%  score <= {t:8.2f}   "
              f"n = {int(np.floor(f*total)):>14,}")

    # ---- the implausible tail ---------------------------------------------------------------
    # DOCK3.7/3.8 totals for AmpC run to roughly -80 kcal/mol. Anything far beyond that is a
    # scoring artefact, and it matters out of all proportion to its frequency: the top-0.1% cut
    # is defined by the extreme tail, so a few thousand nonsense scores would silently BECOME
    # the positives this benchmark asks the models to find.
    print("\nthe implausible tail (DOCK totals for AmpC should stop near -80):")
    for cut in (-80, -100, -120, -150, -200):
        n = int(cum[cut * SCALE - Q0])
        print(f"  score <= {cut:5d}   {n:>12,} rows  ({n/total:9.6%})")

    # ---- how homogeneous is a row group? ----------------------------------------------------
    spread = (rg_max.astype(np.int32) - rg_min.astype(np.int32)) / SCALE
    print(f"\nwithin-row-group score spread (1,000 rows each):")
    for p in (50, 75, 90, 99):
        print(f"  p{p:<2d} {np.percentile(spread, p):7.2f} kcal/mol")
    print(f"  fraction of groups that are a SINGLE score: {np.mean(spread == 0):.3%}")

    # ---- what would the top-1% training draw cost to fetch? ---------------------------------
    print(f"\nrow groups touched by each top fraction (a SMILES fetch costs a whole group):")
    for f in (1e-4, 1e-3, 1e-2):
        t100 = int(round(thr[f] * SCALE))
        touched = int(np.sum(rg_min <= t100))        # group holds >=1 row at or below the cut
        rows = int(np.floor(f * total))
        print(f"  top {f*100:7.3f}%  {rows:>14,} rows   {touched:>10,} groups "
              f"({touched/n_rg:6.3%})   {rows/max(touched,1):7.1f} wanted rows per group")

    np.savez_compressed(OUT / "row_group_scores.npz",
                        rg_min=rg_min, rg_max=rg_max, rg_shard=rg_shard, hist=hist,
                        scale=np.int32(SCALE), rg_rows=np.int32(RG_ROWS),
                        total=np.int64(total))
    (OUT / "score_summary.json").write_text(json.dumps(
        {"rows": total, "row_groups": n_rg, "scale": SCALE, "rg_rows": RG_ROWS,
         "complete": bool(man.get("complete")),
         "thresholds": {f"{f}": thr[f] for f in fracs}}, indent=1))
    print(f"\n-> {OUT/'row_group_scores.npz'}  and  {OUT/'score_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
