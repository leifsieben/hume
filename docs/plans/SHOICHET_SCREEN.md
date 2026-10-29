# Virtual-screen benchmark on the LSD docking set — plan

**Status: proposed, not run.** Numbers below are projections from `results/scale/`.

## The question

Given a fixed compute budget, which molecular representation recovers the most top-scoring
molecules from a billion-compound docking campaign? Output is a Pareto plot: featurization cost
per molecule (x, log) against recovery of the true top 0.1% inside each model's top 1% (y).

## Source

Hall, Tummino, Tang, Mailhot, Castanon, Irwin & Shoichet, *J. Chem. Inf. Model.* 2025, 65,
4458–4467; data at lsd.docking.org. **AmpC**, 1,468,863,655 molecules with DOCK3.7/3.8 scores —
the largest single-target set and the one that paper characterises most fully.

## Arms (5)

| arm | box | µs/mol | extrapolates? |
|---|---|---|---|
| HUME_default + XGBoost | c7i.4xlarge | 122.0 | yes (1.07×) |
| ChemBERTa-2 + XGBoost | g6.xlarge | 112.5 | yes (1.07×) |
| CheMeleon + XGBoost | g6.xlarge | 1046.7 | yes (1.02×) |
| ECFP4 + RDKit-217 + XGBoost | c7i.4xlarge | 460.9 | yes (1.02×) |
| MiniMol + XGBoost | c7i.4xlarge | 1544.7 | **no (2.10×)** |

MiniMol's throughput is not flat in N, so its x-coordinate is a **lower bound**, not a
measurement. This must be stated in the caption; `is_flat()` already refuses it on Figure D.

## Why 100M and not 1.47B

Recall@1% is a property of the score distribution, not of library size. A uniform random 100M
gives an unbiased estimate of both the top-0.1% threshold and the recovery fraction. Sampling
error on recall: 0.50 pp at 10M, **0.16 pp at 100M**, 0.04 pp at the full library. The models
will differ by points, not by hundredths of a point, so the full run buys a fourth decimal for
$388. The compute axis is unaffected: it is measured per molecule and reported as µs/mol × 1.47B.

## Protocol

1. **Deduplicate on canonical SMILES before splitting.** ZINC ids recur across screens and the
   same compound appears under different input strings. The Figure A pipeline learned this the
   hard way: splitting before deduplicating put 700 molecules on both sides of a boundary.
2. **Draw the 100M test set first**, seeded, and freeze the row indices to a file with a
   sha256. Every arm reads that file. The training pool is what remains, so disjointness is a
   property of the construction rather than of a filter.
3. **Train on 1M, stratified — 80% from the top 1%, 20% from the rest.** The LSD paper
   measures this choice as dominant: on AmpC at 1M training molecules, random sampling reached
   Pearson 0.86 but logAUC **0.49** for top-0.01% recovery, where stratified reached **0.87**.
   One strategy, identical molecule indices, for every arm.

   > **This ratio was inverted in an earlier draft of this plan**, which said "80% uniform from
   > the pool, 20% from its top 1%". Hall et al. say the opposite: *"80% of the training set was
   > randomly sampled from the top ranking 1% of molecules and the remaining 20% was sampled from
   > the rest."* Weighting toward the top IS the paper's finding, so the inverted version would
   > have measured the arm the paper reports as the weak one and reported it as the headline.
   > `screen/freeze_split.py` implements the correct direction; the frozen spec records
   > `train_frac_from_top = 0.8`.
4. **One head, and nothing in it is tuned.** The `bench_downstream` XGBoost
   (`n_estimators=300`, `max_depth=6`, `colsample_bynode=0.3`, `tree_method=hist`), identical for
   every arm. `feature_weights` is **fixed a priori at w = 10**, with no inner CV and no per-arm
   or per-dataset selection, for every arm carrying a fingerprint/descriptor boundary — which
   means HUME at both radii *and* the ECFP4+RDKit baseline. It is structurally inert for the
   dense embeddings (ChemBERTa, CheMeleon, MiniMol), which have no such boundary.

   Two reasons, neither of them cost. First, Hall et al. tune nothing: *"We used the default
   parameters, featurizers, and settings."* Second, `docs/API.md` §7 measures w = 1 over 34
   datasets at this exact `colsample_bynode` and finds it **dominated on both suites** (0.8080 vs
   0.7882 activity, 0.8901 vs 0.8712 physchem): *"uniform column sampling is not a neutral
   default here; it is a bad one."* So w = 10 is the documented no-tuning default, and w = 1 would
   be a measured-bad choice that happens to be XGBoost's.

   Tuning w per arm was rejected for fairness, not expense. `feature_weights` is a no-op for an
   arm with no fingerprint/descriptor split, so an inner-CV search spends selection variance only
   on the descriptor-carrying arms — `bench_downstream.py:404` records this as *"a systematic bias
   in favour of the dense arms across the whole grid"*. Dropping the tuner removes a penalty on
   HUME rather than a privilege.

   **This differs from Figures C and D, which did tune w.** Different experiment, different
   scale; it must be stated in the methods rather than left for a referee to find.
5. **One parse-failure set**, computed once from RDKit and applied to every arm, so all arms fail
   on identical rows.
6. **Metrics, all from one scored array**: recall of true top 0.1% within model top 1%
   (headline); logAUC for top 0.01% (comparable to the source paper); overall Pearson; and the
   same three after removing test molecules sharing a Murcko scaffold with any training molecule.

## Leakage, specifically

- **Train/test overlap** — handled by construction (step 2), asserted afterwards, not assumed.
- **The threshold itself** — the top-0.1% cut is computed on the **test set only**. Computing it
  over train+test would leak the training labels into the definition of the positives.
- **Tuning** — `w` is chosen by inner CV inside the training set. Never on the test array.
- **Scaffold/analog leakage** — make-on-demand libraries are dense in analog series, so a test
  molecule can be a near-twin of a training molecule without being identical. This inflates
  recall for every arm, but not necessarily equally: representations differ in how much they
  reward near-duplicates. Reported both ways; the scaffold-filtered number is the honest one.

## Confounding

- Sampling strategy (above) — fixed across arms.
- Head — fixed across arms.
- Training molecules — identical indices across arms.
- Hardware — µs/mol is meaningless without the box. Report instance type and vCPU beside every
  point; never put a GPU wall-clock next to a CPU one without labelling both.
- **Scoring is fused with featurization and streamed.** 100M × 2,670 float32 is 1.07 TB and the
  full library would be 15.7 TB; nothing is materialised.

## Reproducibility

Pin and record: RDKit version, mol-hume version, XGBoost version, model-weight sha256s, every
seed, the test-index file's sha256, and a protocol integer stamped into each record (the pattern
`PROTO` already serves in `bench_downstream`). Persist the per-arm **predictions** (float16,
~1 GB for 5 arms × 100M) so metrics can be recomputed without re-scoring.

## Cost and schedule

| | on-demand |
|---|---|
| featurize + score, 5 arms × 100M | **$68** |
| download ~90 GB of SMILES + scores | one-off |
| re-run headroom | the rest of $500 |

Wall clock is set by CheMeleon (29 GPU-h) and MiniMol (43 CPU-h); at 8 boxes each that is
4–6 hours, so the run fits overnight. Each box ships partial results to S3 every 30 s and
self-terminates, per `scripts/.aws-job-resume`.

## Open calls

1. **AmpC only, or add 5HT2A and Sigma2?** 3× the cost ($204) and the source paper shows real
   target-to-target variation, so one target is a weaker claim than three.
2. **Top 0.1% or 0.01%?** Both come free from the same array. 0.01% matches the source paper;
   0.1% is ten times more positives and a more stable estimate. Recommend reporting both.
3. **Is scaffold-filtered recall the headline, or the raw number?**


## Status, 2026-10-29

**Phase 0 complete.** The 36.7 GB SMILES column was read **once** and reduced to the
106,978,308 molecules the split selects — 246/246 shards, 8 × c7i.2xlarge spot, ~15 min, ~$0.40,
output at `s3://hume-bench-use1-075120018132/screen/smiles/`. Reading it once rather than once
per arm is the difference between 36.7 GB and 220 GB of egress, and it keeps a third-party rate
limit off the critical path of every arm but one.

The split is frozen as **2 kB of JSON, not an index file**: `data/lsd_ampc/split_spec.json` holds
the seeds and a sha256 per array, and `screen/split.py` regenerates all 11 arrays in ~12 s and
*refuses* to return one whose digest does not match. 100M test drawn once and shared by all five
replicates (re-drawing it per replicate would multiply the only expensive step by five to measure
a variance the replicates are not about), plus 5 × (1M train + 600k validation pool).

**The row-order gate is the check that matters.** The split is a set of integer row positions, so
it is only meaningful if the upstream file still holds the rows it held when the split was drawn.
Before any SMILES is kept, each shard's score column is compared element-by-element — 6.4M values
per shard, not a sampled checksum — against the int16 copy downloaded 2026-10-28. A re-sorted or
re-uploaded shard would otherwise point every index at a different molecule, silently.

Two things learned from the data itself:

* **The release is not one score-sorted file.** All 15,705,526 top-1% molecules live in **7
  shards** (36, 47, 48, 58, 59, 68, 69); 239 of 246 contain none. It is evidently a concatenation
  of several separately sorted runs. No effect on the screen — the test set is uniform — but all
  the training signal is physically concentrated in 7 files.
* **The five replicates share ~13% of their training molecules**, by construction: 80% of each 1M
  draw comes from a pool of only 15.7M, so 1,021,692 of a nominal 108,000,000 selections are
  repeats. This is the paper's design, so we follow it, but the replicate spread is not a spread
  over independent training sets and should not be described as one.

Still to build: phase 1 (train 30 models), phase 2 (score 100M, 95.9 box-hours — 63.7 CPU plus
32.2 GPU, ~$32 spot), phase 3 (metrics). No out-of-band alerting yet; it needs to exist before
phase 2 runs unattended.

Open and **not** decided: whether to deduplicate on canonical SMILES (item 1 above is a deviation
from the paper, which does not mention it); and the 78,508 molecules scoring below −80, which the
paper does not filter and which we are leaving in on instruction, with a filtered robustness
check reported alongside.
