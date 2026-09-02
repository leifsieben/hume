# HUME_minimal: the definition, and how each column earned or lost its place

*Live working document. Records what was decided, what it was decided on, and what is still
open. Companion to `docs/DESCRIPTOR_MAP.md` (what the columns measure) and
`docs/MINIMAL_SPEC.md` (the superseded statistical derivation, kept for its negative results).*

**Status: the selection is being rebuilt.** `minimal-v1`, shipped in mol-hume 0.2.0, was derived
by rank-revealing QR on a linear-recoverability criterion. That criterion is retired — see §1 —
and this document is the replacement, built from provenance first and statistics second.

---

## 1. Why the first attempt is being replaced

`minimal-v1` selected columns so that every dropped column was **linearly recoverable** from the
kept ones. The reasoning was that a dropped column costs nothing if the model can rebuild it.

**The criterion describes a consumer that does not exist.** A depth-6 boosted tree splits on
individual columns and cannot split on a linear combination of thirty. That was stated as a
prediction — a head that *can* form linear combinations should recover the loss — and the
prediction was tested and failed:

| head | median cost of the 800-column set on 6 physicochemical datasets |
| --- | --- |
| XGBoost depth 6 (the grid's head) | +4.63% |
| XGBoost depth 10 | +4.12% |
| XGBoost, `colsample_bynode=1.0` | +3.09% |
| ridge | +1.37% |
| MLP (512, 128) | +3.20% |

Neither a deeper tree nor an MLP recovers it. Linear recoverability is a property of the matrix,
not of any model anyone runs.

**Two things from the old method survive and must be preserved.**

1. **Label-free selection.** No target, no assay, no benchmark. A descriptor spec is a permanent
   contract inherited by users whose chemistry nobody here has seen; redundancy is a property of
   molecules, informativeness is a property of somebody's targets, and only the first is safe to
   select on. The benchmark is a **test** of a spec, never an input to it.
2. **Rare features must not be ranked by variance.** Pivoted QR ranks by residual orthogonality,
   and this protected rare columns for free: those firing on <=2% of molecules had median rank
 **71** of 1,267 against **704** for those firing on >50%, and 64 of 70 were kept. Any
   successor that picks family representatives by typicality or variance would delete exactly
   the long tail on which rare-substructure activity depends. **This is a hard constraint.**

## 2. The criterion now

A column earns its place if it is **non-degenerate**, **defined on the target domain**, **stable
under notation**, **numerically sane**, and **distinct in mechanism** — not merely distinct in
value. Redundancy is judged in two separate ways, because it comes in two kinds:

- **Parametric** — within one (property, operator) cell: how many lags, how many bins. Statistics
  answer this well.
- **Constructual** — the same physical quantity in different units or scales. **Only provenance
  answers this.** No correlation cutoff separates "0.995, same construct" from "0.99, genuinely
  different" without already knowing which is which.

Where a family has an alternative already in the output (an ECFP, say), the deciding test is
**consumer inversion**: can the actual downstream model reconstruct the column from what remains?

## 3. Decisions taken

### 3.1 Autocorrelation weights: 12 -> 7, and the block 519 -> 327

The autocorrelation block is 41% of the library: 12 atom properties x 6 normalizations x 9 lags.
**Eight of the twelve weights are pure periodic-table lookups** — scalar functions of atomic
number with no molecular context (`Z`, `m`, `v`, `p`, `i`, `se`, `pe`, `are`). They are therefore
eight vectors over the ~9 elements that actually occur, and their effective rank, weighted by
real corpus element abundance, is **1.76**. Two components carry 97.4%; three carry 99.4%.

| table | axis 1 (70.5%) | axis 2 (26.9%) |
| --- | ---: | ---: |
| Allred-Rochow EN | −0.419 | 0.005 |
| Pauling EN | −0.418 | −0.045 |
| Sanderson EN | −0.412 | −0.090 |
| ionization | −0.394 | 0.048 |
| vdW volume | +0.400 | −0.196 |
| polarizability | +0.367 | −0.320 |
| atomic mass | −0.121 | **−0.652** |
| atomic number | −0.127 | **−0.650** |

**Axis 1 is electronegativity against size/polarizability — one axis, not four constructs.** They
are anti-correlated by chemistry: bigger and more polarizable means less electronegative. **Axis
2 is period/nuclear mass**, the heavy-halogen axis, and only `Z` and `m` load on it.

**KEPT (7).** Four are environment-dependent and pass the mechanism test outright, because they
are functions of the molecule rather than of the element:

| weight | why it is distinct in mechanism |
| --- | --- |
| `c` Gasteiger charge | depends on the whole connectivity, not on the element |
| `d` sigma electrons | heavy-atom degree — pure local connectivity |
| `dv` valence electrons | a formula over outer electrons, formal charge and attached H |
| `s` intrinsic state | a specific nonlinear combination of valence and sigma connectivity; not obtainable from `d` and `dv` through the aggregation |

Three represent the element axes:

| weight | why this one |
| --- | --- |
| `Z` atomic number | axis 2. Never NaN by construction |
| `pe` Pauling EN | axis 1, electronic. **Chosen for coverage: 94 elements against Sanderson's 56 and Allred-Rochow's 66** |
| `v` vdW volume | axis 1, steric. Enters the aggregation with the opposite sign to `pe` |

**DROPPED (5), with the reason:**

- **`m` atomic mass.** A lookup table indexed by `Z`, so it cannot even carry isotope
  information — the table has one value per element. r = 0.999 with `Z`, and `Z` never returns
  NaN. Provably redundant.
- **`se` Sanderson, `are` Allred-Rochow.** The same measurement as `pe` in different empirical
  scales; loadings on axis 1 are −0.412, −0.419, −0.418. Measured correlation between matched
  autocorrelation columns is 0.969–0.995.
- **`i` ionization, `p` polarizability.** Also axis 1, loadings −0.394 and +0.367.

 **Why the residual 1–6% is acceptable, and the condition attached.** These correlations are
not 0.999, and the reason is structural: the autocorrelation is a **sum over pairs**, and
`sum(g(w)) != g(sum(w))`, so monotone-relatedness of two *tables* does not survive the
aggregation. Concretely, if `w2 ~ a*w1 + b*1`, then

    ATS_k(w2) = a^2 * ATS_k(w1) + 2ab * (w1-weighted pair count) + b^2 * (pair count at lag k)

The second weight injects a **topology** term — the count of atom pairs at distance k. That is
genuinely new relative to `w1` alone, which is why a tree could use both. But it is *not* new
relative to the library, because pure-topology autocorrelations are already emitted.
**Condition: this argument dies if a future cut also removes the topology columns.** They must
be retained, or these five weights have to be reconsidered.

 **The coverage argument is latent, not measured.** NaN rate for all eight weights is 0.000%
on both the benchmark corpus and the salts/mixtures set, because neither contains a transition
metal. The 94-vs-56 element difference between Pauling and Sanderson is read off the tables, not
observed. It becomes decisive only on organometallics — and note that `minimal-v1` kept 33
Sanderson columns and 1 Pauling, which is **backwards on coverage**, precisely because QR
optimized on a corpus where the difference is invisible. This is the first thing a metal-bearing
corpus should test.

**Not yet decided: the parametric axis.** The 327 kept columns are still 7 weights x 6
normalizations x 9 lags. Whether 9 lags are needed, and whether `MATS`/`GATS` (Moran and Geary,
two classical spatial-autocorrelation coefficients) are near-duplicates, is a separate question
answerable by measurement. **The 519 -> 327 figure is the constructual cut only.**

### 3.1b Normalizations and lags (decisions 2 and 3)

**Normalizations.** The six are `ATS` (sum of w_i*w_j over pairs at lag k), `AATS` (divided by
the pair count), `ATSC` / `AATSC` (the same on centered weights), `MATS` (Moran = AATSC divided
by the variance of w) and `GATS` (Geary, which uses squared differences rather than products).
Median |r| over kept weights x lags 1-8:

| pair | median \|r\| |
| --- | ---: |
| `AATSC` ~ `MATS` | **0.921** |
| `GATS` ~ `MATS` | 0.784 |
| `ATSC` ~ `AATSC` | 0.760 |
| `ATS` ~ `AATS` | 0.269 |
| `ATS` ~ `ATSC` | 0.189 |
| `ATS` ~ `GATS` | 0.129 |

Only one pair is strongly redundant, and it is the one the definitions predict: `MATS` is
`AATSC` divided by a per-molecule scalar. **Decision: drop `MATS`, keep `AATSC`.** Note
`ATS`~`AATS` is only 0.269 -- dividing by the pair count is not cosmetic, and an earlier
assumption that those two were near-duplicates was wrong.

**Lags.** The redundancy across lags is NOT uniform, and this is the useful finding:

| normalization | adjacent lags | lag k vs k+2 |
| --- | ---: | ---: |
| `ATS` | **0.966** | **0.938** |
| `AATS` | **0.852** | |
| `ATSC` | 0.172 | |
| `AATSC` | 0.209 | |
| `MATS` | 0.231 | 0.132 |
| `GATS` | 0.207 | 0.121 |

The uncentered forms are nearly flat across lags, because `ATS_k` is approximately
`(sum w)^2 * (fraction of pairs at distance k)` and is therefore dominated by molecular size.
The centered forms are the opposite: every lag carries distinct information.

**Decision: keep lags {0, 2, 4, 6, 8}.** **Applied to `ATS` and `AATS` only** -- see §5
question 5. Applying it to the centered forms as well would discard columns whose adjacent-lag
correlation is 0.17-0.23, which is signal rather than redundancy.

### 3.1c Does the lag cut undermine the weight cut? No -- measured

The case for dropping `se`/`are`/`i`/`p`/`m` rested on the aggregation injecting a topology term
that other columns carry. **That second half was asserted without checking, and checking shows
there is no constant-weight autocorrelation emitted**, so the pair-count-at-lag-k profile is not
directly present -- only partially, through path counts (`MPC*`, `piPC*`), topological charge
(`GGI*`, `JGI*`) and distance functionals.

Tested directly by consumer inversion: can XGBoost rebuild a dropped-weight column from the kept
columns, with all 9 lags versus with `ATS`/`AATS` cut to {0,2,4,6,8}?

| design | median R^2 | min | below 0.95 |
| --- | ---: | ---: | ---: |
| all 9 lags | 0.9885 | 0.8625 | 12 of 48 |
| `ATS`/`AATS` cut | 0.9857 | 0.8603 | 12 of 48 |

**Median change +0.0001, worst +0.0417. The two decisions are independent.**

 **But note what this also shows about the weight cut on its own: 12 of 48 sampled
dropped-weight columns reconstruct below R^2 0.95, worst 0.86.** The dropped weights are not
fully recoverable from the kept ones even with every lag present. That is a real cost of the
weight decision, unchanged by the lag decision, and it is open.

### 3.2 Substructure matching (`fr_*`, 75 columns)

Tested by consumer inversion against the ECFP6 that mol-hume emits alongside them, using the
same untuned XGBoost the benchmark grid uses, on 20,000 corpus molecules.

**Detection AUROC: median 1.000. 67 of 72 scored columns at >= 0.99. None below 0.90.** The
fingerprint carries the presence of essentially every fragment pattern.

 **The first version of this test used R^2 and was wrong.** For a column that is zero on 99.95%
of molecules, R^2 is set by a handful of positives, so a model that predicts ~0 everywhere and is
right 99.95% of the time scores near zero. That measured **rarity, not recoverability**, and it
made rare patterns look uniquely irreplaceable. Detection AUROC is prevalence-free.

**A mechanistic prediction was made and failed.** ECFP6 hashes environments to radius 3, so
patterns larger than that should have no bit to live in and be harder to recover. Measured
correlation between AUROC and SMARTS pattern size: **rho = −0.16, p = 0.20.** Size does not
predict recoverability; a tree learns the conjunction of bits regardless of span. The mental
model of what a fingerprint cannot encode was wrong.

The only thing `fr_*` adds is **counts** — median Spearman 0.31 on molecules where the pattern
occurs, 35 of 63 below 0.5 — and that is an artifact of shipping *binary* bits. A count
fingerprint would likely close it.

**Decision: drop the family** (user decision, 2026-09-02). Condition in §5 question 1 stands.

### 3.3 No other family is redundant with the ECFP, and the boundary is principled

The same test applied to every other family whose columns are counts of local environments:

| family | columns | median value R^2 | at >= 0.95 | median detection AUROC |
| --- | ---: | ---: | ---: | ---: |
| `fr_*` substructure | 75 | — | — | **1.000** |
| E-state atom type | 100 | 0.896 | 13 | 1.000 |
| ring perception | 74 | 0.813 | **0** | 0.996 |
| rdkit core counts | 9 | 0.878 | **0** | 0.999 |
| constitutional | 27 | 0.881 | 2 | 0.999 |

**Nothing else is droppable, and the reason explains why `fr_*` was.** Detection is near-perfect
everywhere -- a circular fingerprint knows *whether* a local environment is present. What it does
not carry is a **magnitude**: E-state columns are a continuous electronic index rather than a
count (`SdssC` R^2 0.69 despite firing on 81% of molecules), ring counts are global topology that
no radius-3 environment sees (`n6FRing` R^2 −0.36), and `HeavyAtomCount` sits at 0.863 because a
binary bit vector cannot count.

`fr_*` was uniquely redundant because it is the one family that is a **pure presence flag for a
local pattern** -- precisely what a binary circular fingerprint encodes and nothing more. The
boundary falls exactly where the mechanism says it should, which is evidence the drop was
principled rather than a threshold artifact.

### 3.4 Ring perception, constitutional counts (decided)

**Exactly determined columns.** A grid of counts over a partition has exact additive structure --
`n7Ring == n7aRing + n7ARing` and ten others hold to machine precision on every molecule. Greedy
elimination requiring a column to raise the rank on **both** corpora finds **25 ring + 6
constitutional** columns determined by the rest.

 Checked across chemical spaces, and it mattered: rank says 23 determined on the benchmark
corpus alone, 22 on salts alone, but only **21 on the two stacked**. Dependencies that hold on
one space and break on the other are corpus artifacts, and taking the minimum is what excludes
them.

 And checked with the consumer, because linear determination is not tree determination -- a
tree cannot split on a sum of two features, which is the error that killed minimal-v1. Measured:

| rebuilt by | median R^2 | below 0.99 | worst |
| --- | ---: | ---: | ---: |
| XGBoost | **1.0000** | 3 of 27 | 0.857 |
| linear | 0.9971 | 12 of 27 | 0.626 |

The tree rebuilds them *better* than a linear model, exploiting integer structure a linear fit
cannot. The concern was unfounded; testing it was still right. One straggler: `n12FaRing`, tree
R^2 0.857, fires 0.200% -- kept.

**Ring size binning.** Decision: collapse sizes to {3,4,5,6,7,8+} **for non-fused cells only**.

 The reason for the exception is mechanical and nearly went unnoticed. For a FUSED cell,
mordred's "size" is the **atom count of the whole fused system**, not a ring size -- naphthalene's
system is 10 atoms and counts as a 10. So the fused 8+ bins are not exotic macrocycles:

| fused "size" | fires on | what it is |
| --- | ---: | --- |
| 9 | 15.5% | indole, benzofuran |
| 10 | 13.6% | naphthalene, quinoline |
| >12 | 12.0% | larger fused systems |

Collapsing those would merge indole, naphthalene and anthracene into one bin. The non-fused
cells at the same sizes are the genuine medium and macro rings and fire on 1.9%, 0.19%, 0.29% --
there the collapse is right. Implementation note: this is not a drop, it needs new summed
columns in `ringcount.h`.

### 3.5 ETA (29 columns): the whole family is reproducible

Consumer inversion with the **entire family removed** from the design:

| | median R^2 | worst | below 0.96 |
| --- | ---: | ---: | ---: |
| siblings present | 0.9970 | 0.958 | 0 |
| whole family removed | **0.9954** | 0.962 | **0 of 29** |

Those being nearly equal is the finding: ETA columns are not predicting each other, the rest of
the library predicts all of them. Mechanistically expected -- ETA is a per-atom core count
(alpha, a valence-electron-mobility count) aggregated by a distance-weighted pair sum, which is
the same operator class as the autocorrelation block applied to a weight close to `dv`, and its
beta/gamma terms are functions of degree, which is `d`.

**And it independently fails notation stability.** `ETA_epsilon_4` and `ETA_dEpsilon_C` move on
**33%** of molecules under rewriting, `ETA_dEpsilon_B` on 16%; `eta.h` records that
`ETA_epsilon_4`'s definition is **ill-posed** for saturated systems and depends on a search
order. Two independent criteria converging on one block.

### 3.6 Information content (22 columns): redundant across RADII, not across series

The intuition that these are one quantity in several weightings is **wrong**, measured:

| pair, same radius | r at 0 | at 2 | at 4 |
| --- | ---: | ---: | ---: |
| `IC` ~ `ZMIC` | 0.092 | 0.063 | **0.029** |
| `IC` ~ `MIC` | 0.880 | 0.477 | 0.389 |

`IC` and `ZMIC` are essentially uncorrelated, so keeping one series and dropping the others would
discard real information. The redundancy is **within** a series, across radii:

| series | adjacent-radius correlations |
| --- | --- |
| `IC` | 0.64, 0.72, 0.87, **0.97**, **0.99** |
| `MIC` | 0.88, 0.95, **0.98**, **1.00** |
| `ZMIC` | 0.79, 0.88, 0.88, 0.97, **0.99** |
| `BIC` | 0.80, 0.92, **0.98** |

Low radii are distinct; high radii converge, because the orbit partition saturates as the
neighborhood grows. So the defensible cut keeps all three series and drops the high radii, not
the reverse. `AvgIpc` correlates 0.13-0.40 with all of them -- it is built from the
characteristic polynomial rather than an orbit entropy, a genuinely different operator.

### 3.7 BCUT weights, path/walk, matrix spectrum, AETA (decided)

**BCUT (7 columns, ~59 us/mol).** The 20 mordred BCUTs are indexed by the same weight vocabulary
cut from the autocorrelation block. **That argument does not transfer automatically and was
measured rather than assumed**: BCUT is an extreme eigenvalue of a matrix carrying the weight on
its diagonal, and two affinely related diagonals do not give affinely related eigenvalues because
the off-diagonal part does not scale. Mostly it holds -- `BCUTpe-1l`~`BCUTare-1l` 0.981,
`BCUTv-1h`~`BCUTp-1h` 0.971 -- but **`BCUTi-1l`~`BCUTpe-1l` is 0.244**, so ionization and
electronegativity are one axis in the element tables and are NOT one axis through this operator.
The decision rests on the consumer test instead: dropping the 7 gives median R^2 0.9997, floor
0.9965, none below 0.9.

**Cost matters more than the column count here.** Each weight is its own Burden matrix and its
own eigensolve, so cost scales with WEIGHTS: 4 of 11 matrices removed is roughly **59 of BCUT's
163.3 us/mol**, about 10% of total featurization, for 7 columns. Column count and compute are not
proportional, and this is the clearest case.

**path/walk counts (30) + matrix spectrum (16), tested JOINTLY.**  The family screen removes one
family at a time, which cannot license dropping two: if each is reproducible from a library still
containing the other, mutual redundancy makes both look free while the pair is not. Measured with
both removed together:

| | median R^2 | floor |
| --- | ---: | ---: |
| path/walk alone | 0.9980 | 0.9895 |
| path/walk, spectrum also gone | **0.9980** | 0.9889 |
| spectrum alone | 0.9958 | 0.9881 |
| spectrum, path/walk also gone | **0.9957** | 0.9883 |

Unchanged to four decimals -- they are not propping each other up.

**AETA (14 columns).**  Found by auditing the residue, not by the screen: the ETA drop matched
`startswith("ETA_")`, which does not match `AETA_`. AETA is the same family normalised per atom.
Dropped on the same reasoning, with the inversion test run to confirm rather than assume.

### 3.8 Autocorrelation: why it stays large

The block is 227 of the planned library and the obvious reaction is that it is too many
dimensions. The evidence points three ways and it is worth recording all three:

| view | verdict |
| --- | --- |
| exact linear rank, both corpora | **227 of 227 -- nothing determined** |
| components for 90% / 99% of variance | **1 / 3** |
| participation ratio | **1.1** |
| family screen: can the rest of the library rebuild it? | **least reproducible family, median 0.80** |

One size-like axis carries 90% of the variance, so by variance the block looks almost
one-dimensional. But after that axis every residual is distinct: no column is an arithmetic
identity of another, and the rest of the library rebuilds it worse than any other family (13 of
21 sampled below R^2 0.9).

**This is the same trap the exercise began with** -- the source document's own observation that 325
columns reach 99.9% of variance while 640 are needed to reconstruct. A variance cut would take
autocorrelation to 3 columns and destroy the distance-resolved structure it exists to encode.
Under the constructual-or-exact standard the block is finished; what remains untested is not
redundancy but IMPORTANCE, which is a different question and needs an ablation.

### 3.9 Autocorrelation dropped (provisional)

The block is irreducible -- exact rank 227 of 227, and the least reproducible family in the
library -- so nothing else can rebuild it. That establishes DISTINCTNESS, and distinctness was
being treated as if it implied value. It does not, and the two come apart here.

Mechanistically, the uncentered forms are barely about the property at all. Writing
`w = wbar + delta`, `ATS_k = n_k*wbar^2 + ...`, and for a strictly positive weight the first term
dominates. Measured, how much of each column is explained by the pair count at lag k plus
composition alone:

| form | R^2 from distance profile + composition |
| --- | ---: |
| `ATS` | **0.95 - 0.98** |
| `AATS` | 0.62 - 0.94 |
| `ATSC` / `AATSC` / `GATS` | 0.02 - 0.24 |

So `ATS` is ~97% the distance distribution scaled by composition. The centered forms subtract the
mean first and are the ones actually measuring arrangement.

**The ablation.** Removing all 227 kept autocorrelation columns -- 29% of the library -- from five
physicochemical datasets, same folds and head as the grid:

| arm | mean | median | worst | above fold noise |
| --- | ---: | ---: | ---: | --- |
| minus 227 kept | **-0.64%** | -0.75% | +0.35% | **0 of 5 datasets** |
| minus all 519 | -0.05% | -1.23% | +6.38% | 1 of 5 |

Negative is better. Not one dataset moved by more than its own fold-to-fold spread.

 **This is the only cut in the spec that rests on "we could not measure it helping" rather than
on redundancy, and it is PROVISIONAL.** Five physicochemical datasets is not the grid;
classification, ADME and quantum are untested. It is much stronger evidence than the ETA
ablation -- a 227-column removal producing no signal is a result, where a 29-column one drowning
in noise was a failed measurement -- but it should be re-run against all 33 datasets before it is
treated as settled.

## 4. Running total

| decision | columns removed |
| --- | ---: |
| `fr_*` substructure (§3.2) | 75 |
| autocorrelation: 5 element weights (§3.1) | 192 |
| autocorrelation: `MATS` (§3.1b) | 48 |
| autocorrelation: `ATS`/`AATS` odd lags (§3.1b) | 52 |
| ring + constitutional, exactly determined (§3.4) | 29 |
| ETA (§3.5) | 29 |
| BCUT on cut weights (§3.7) | 7 |
| path/walk counts + matrix spectrum (§3.7) | 46 |
| AETA (§3.7) | 14 |
| autocorrelation, all remaining (§3.9, **provisional**) | 227 |
| **total distinct columns dropped** | **719** |

**1,269 → 612, shipped as `minimal-v2` (0.5.0; it was 550 in 0.4.0 — see §3.10).** The ring 8+ re-binning was dropped from the plan: it
removes 5 columns and adds 3, and needs new C++ to do it. Information content is kept whole (§3.6): the redundancy there is across radii
rather than across series, and the block is not worth cutting for 10 columns.

Every cut so far is **constructual or exact** — the same quantity in different units, information
already carried elsewhere in the output, or an arithmetic identity. **Nothing has been removed on
a variance ranking**, which is the property that made `minimal-v1` delete the rare tail.

 The ring 8+ re-binning turned out to cost 5 columns and add 3, a net of 2, because most
non-fused 8+ cells had already gone as exactly determined. It needs new C++ columns in
`ringcount.h` for that, which is a poor trade and should probably be dropped from the plan.

## 4b. What the spec costs, measured (29 of 33 datasets)

Same untuned XGBoost head, same 5-fold Murcko scaffold folds, `hume` against `hume_minimal` on
identical molecules.

| panel | datasets | mean | median | worst |
| --- | ---: | ---: | ---: | ---: |
| ADME & tox | 10 | **−1.55%** | −1.50% | +2.79% |
| physicochemical | 6 | **−0.94%** | −0.67% | +1.49% |
| classification | 13 | **−0.17%** | −0.18% | +1.99% |
| overall | 29 | **−0.81%** | −0.38% | — |

Negative is better. **0 of 29 datasets moved by more than their own fold SD**, and the sign test
gives 11 of 29 worse, p = 0.265. The honest reading is *no measurable difference at 43% of the
columns*, not that the reduction helps.

**The physicochemical line is the one that matters.** `minimal-v1` cost +3.83% there with 800
columns; v2 costs −0.94% with 550. The reduction is larger and the loss is gone, and the only
thing that changed is the criterion — v1 cut on linear recoverability and removed columns a tree
needed, v2 cut only on same-quantity-different-units, already-in-the-output, and exact identity.

This also largely settles §3.9. The autocorrelation drop was decided on five physicochemical
datasets and is now supported by 13 classification and 10 ADME sets that had no part in the
decision. **Quantum (4 datasets) is still outstanding** and is the panel where the cut is most
at risk, since autocorrelation is a distance-resolved property correlation and electronic
structure is the plausible consumer of one.

### 3.10 fr_* restored: detectability is corpus-conditional

0.4.0 dropped all 75 `fr_*` flags because they are detectable from the ECFP at AUROC 1.000. That
was measured on our corpus with our fingerprint, and it does not hold generally.

ChemPFN measured the same flags on a corpus with 5.4% salts using an r=2 fingerprint and a
logistic decoder: **median 0.9929, floor 0.786, 11 of 62 below 0.95.** Two variables differed
from our setup at once (radius and decoder), so we ran the 2×2 on our own corpus to separate
them:

| | median | floor | <0.95 |
| --- | ---: | ---: | ---: |
| r=2, logistic (their config) | 0.9986 | 0.9490 | 1 |
| r=2, XGBoost | 0.9984 | 0.7757 | 2 |
| r=3, logistic | 0.9982 | 0.9440 | 1 |
| r=3, XGBoost | 0.9980 | 0.7858 | 2 |

**Neither variable explains the gap** — the median moves by less than 0.001 across all four.
`fr_quatN` reads **0.9995 here and 0.73 there**, same flag, same fingerprint class. The
difference is the CORPUS: quaternary nitrogen is a salt-former, and one corpus has salts.

So detectability was never sound grounds for dropping the family, and the 62 restored are kept
on mechanism — curated assertions no structural descriptor derives. 13 stay out because they
duplicate counts already emitted, which is a mechanistic argument in the other direction.

 **The general lesson is the one that retired v1.** Both times a criterion was adopted that
described a decoding rather than a property: v1 assumed a consumer that inverts linear
combinations, and 0.4.0 assumed a consumer that decodes hashed fingerprint bits. Neither
consumer exists. A curated, named, cheap feature carries identity, count and provenance; a
fingerprint bit is an anonymous hash shared with everything that collides into it.

## 4c. The 612-column spec on the full grid (all 33 datasets)

Run locally rather than on EC2: `hume` against `hume_minimal` needs only HUME descriptors, the
ECFP, `y` and the stored scaffold folds -- no learned embeddings -- so it is 330 XGBoost fits on
a laptop. `tools/minimal_local_grid.py` caches every dataset's feature matrix on the way through,
which makes any future column subset a slice rather than an EC2 wave.

| panel | datasets | mean | worst |
| --- | ---: | ---: | ---: |
| classification | 13 | −0.65% | +0.11% |
| quantum | 4 | **−0.11%** | +1.05% |
| ADME & tox | 10 | +0.39% | +10.88% |
| physicochemical | 6 | +0.75% | +3.50% |
| **overall** | **33** | **−0.02%** | — |

Negative is better. **1 of 33 datasets exceeds its own fold-to-fold spread**, and that one is
`qmugs_gap` at −0.75%, where the reduced set wins. There is no dataset in the grid where
dropping 657 columns measurably hurts.

**§3.9 is no longer provisional.** The autocorrelation cut was decided on five physicochemical
datasets; quantum was the panel where it was most at risk, since a distance-resolved property
correlation is the plausible input to an electronic-structure endpoint. It lands at −0.11%.

 Two numbers not to over-read. `vdss_lombardo` shows +10.88% against a fold SD of **101.86%**
of its own RMSE -- that dataset cannot resolve a 10% effect. And physicochemical is the only
panel with a positive mean (+0.75%, `esol` at +3.50% inside its 6.6% fold noise); it is the panel
`minimal-v1` broke, so it is the one to watch rather than dismiss.

## 5. Open questions

1. **`fr_*` redundancy is conditional on the fingerprint being present**, and `fingerprint=False`
   is a supported flag. Three columns (`fr_azide`, `fr_SH`, `fr_benzodiazepine`) had too few
   positives in 20,000 molecules to score at all, and five more sit at 0.957–0.989. Options:
   drop and document that the reduced spec assumes an ECFP; or keep the eight that did not
   cleanly clear.
2. **No metal-bearing corpus.** Every coverage claim in §3.1 is read from lookup tables. Natural
   products, agrochemicals and materials are also absent, so "validate across chemical spaces"
   is currently aspirational.
3. **The parametric cut** — lags and normalizations — is untouched.
5. **Which normalizations does the lag cut apply to?** Recorded as `ATS`/`AATS` only, because
   the centered forms have adjacent-lag correlation 0.17-0.23. If it was meant for all six, that
   discards signal and the numbers in 3.1b are the evidence against it. **Needs confirmation.**
6. **The weight cut has an unresolved residual**: 12 of 48 sampled dropped-weight columns
   reconstruct at R^2 below 0.95 from the kept set, worst 0.86 (§3.1c). Independent of the lag
   decision, but not zero.
7. **`minimal-v1` is published** in mol-hume 0.2.0+ as a frozen contract. Whatever replaces it
   ships under a new name; the old one is not edited.
