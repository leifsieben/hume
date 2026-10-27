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
3. **Train on 1M, stratified** — 80% uniform from the pool, 20% from its top 1%. The LSD paper
   measures this choice as dominant: on AmpC at 1M training molecules, random sampling reached
   Pearson 0.86 but logAUC **0.49** for top-0.01% recovery, where stratified reached **0.87**.
   One strategy, identical molecule indices, for every arm.
4. **One head.** The `bench_downstream` XGBoost, `feature_weights` tuned on training folds only
   for arms that mix fingerprint bits with descriptors (HUME, ECFP4+RDKit) and inert for the
   dense embeddings. Otherwise this is a head comparison wearing a representation's clothes.
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
