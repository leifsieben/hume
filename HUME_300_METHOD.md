# HUME_300 — method

*How a ~300-column spec is being chosen, what each step can and cannot establish, and what
would falsify it. Live document; decisions are recorded as they are made, with the evidence.*

Target: **300 ± 20 columns**, from the 622 of `minimal-v2`.

---

## 0. Why this is a different problem from 1,269 → 622

`minimal-v2` was built on **mechanism**, and its three levers are now spent:

- the same physical quantity in different units (three electronegativity scales, mass against
  atomic number);
- already carried by the ECFP that ships alongside;
- an exact arithmetic identity of columns that remain.

None of those apply to what is left. 622 → 300 cannot be argued from construction, so it needs a
different lever and an empirical one. That change of lever is the whole reason this document
exists: a cut that cannot say *why* is a cut nobody downstream can defend.

**The lever that is available is RESOLUTION.** Most of the remaining 622 is parametric sweeps —
one construct evaluated at many orders, lags, bins or window sizes, where the resolution was
chosen by a library author and not by chemistry:

| sweep | columns | parameter |
| --- | ---: | --- |
| chi connectivity | 56 | path order 0–7, valence and simple |
| E-state atom types | 86 | atom environment type |
| VSA surface bins | 55 | 5 schemes x 10–14 bins |
| ring counts | 47 | size x aromatic/aliphatic x hetero/carbo x saturation |
| ours: walk statistics | 49 | window size x 4 statistics |
| ours: autocorrelation | 39 | lag 1–4 x 4 properties |
| charge | 26 | Galvez lag 1–10 |
| information content | 21 | order 0–5 x 4 normalisations |

Choosing a resolution is defensible in a methods section. "We pruned to 300" is not.

---

## 1. What the 622 are

Provenance, by name against RDKit's `_descList` and Mordred's 2D calculator:

| source | n | share | note |
| --- | ---: | ---: | --- |
| Mordred | 292 | 47% | 18% of Mordred's 1,613 survived deduplication |
| RDKit | 188 | 30% | nearly all of RDKit's 217 |
| ours | 142 | 23% | no RDKit or Mordred counterpart |

⚠️ The intuition that this would be ~2/3 RDKit is wrong in count and right in proportion: almost
all of RDKit is here, but RDKit is small.

## 2. How much of it is distinct information

Effective rank on 477,115 molecules (the 33 cached downstream datasets), in **rank space**,
because a tree reads order and not magnitude:

| space | 300 PCs retain | components for 99.9% |
| --- | --- | ---: |
| raw, unstandardised | 100.0000% | **2** |
| standardised | 99.38% | 394 |
| rank | 99.11% | **448** |

⚠️ **A "99.9% of variance at 300 components" claim is a preprocessing artifact.** In raw space
two components reach it, because `WPath` runs to 2.3e12 and dominates the covariance. Any PCA
statement about this matrix has to name its space.

⚠️ **AND LINEAR VARIANCE IS THE WRONG CRITERION ANYWAY.** This project has already paid for that
lesson: `minimal-v1` was an ordering derived from linear recoverability and was withdrawn,
because columns recoverable at linear R² 0.997 were **not** recoverable by a depth-6 tree (tree
median R² 1.0000 against linear 0.9971 on the same held-out columns). PCA retention is an upper
bound on what may be dropped, never a licence.

Within-group rank, columns needed for 95% of within-group variance — where the slack is:

| group | n | → 95% |
| --- | ---: | ---: |
| chi connectivity | 56 | 10 |
| ours: walks/paths | 49 | 9 |
| information content | 22 | 5 |
| charge | 28 | 12 |
| ours: rings/conjugation | 34 | 14 |
| composition counts | 40 | 18 |
| ring counts | 47 | 21 |
| ours: autocorrelation | 39 | 24 |
| VSA bins | 55 | 34 |
| E-state | 86 | 42 |
| BCUT | 19 | 11 |
| `fr_*` flags | 72 | **55** |

Summed, ~255 plus the ungrouped remainder — close to 300, which is encouraging and is not
evidence.

---

## 3. Stage 1 — per-column utility ranking (done)

`tools/h300/utility.py`. For each of the 33 cached datasets: the **same** untuned XGBoost head
and the **same** stored 5-fold Murcko scaffold splits the downstream grid uses, fitted on the 622
columns, recording per-column **gain** summed over folds. Columns are then ranked within each
dataset (1 = most used) and aggregated as **mean rank across datasets**.

**Why gain, and why on a tree.** The question is which columns a tree uses, and a tree does not
use linear combinations — the minimal-v1 error again. Gain is measured on the consumer that
actually reads these features.

**Why mean RANK and not summed gain.** Gain is not comparable between tasks with different
targets and scales; summing it would let the largest dataset choose the spec.

### What Stage 1 established

- **No column is dead.** All 622 are used by at least one dataset; 228 are used by all 33. There
  is no free lunch to collect before the real decision.
- BCUT ranks best of any family by median (186 of 622) despite being an eigenvalue summary.
- Ring counts rank worst as a family (median 428) and **no** ring-count column is used by all 33.

### ⚠️ Three things Stage 1 CANNOT establish, and one it actively gets wrong

1. **Gain splits between correlated columns.** Eight near-identical chi orders share the credit,
   so each looks mediocre while the group is load-bearing. Ranking by gain therefore
   systematically over-cuts correlated sweeps — which is most of what is left.
2. **Usage is not necessity.** A column can be used constantly and still be free to drop, because
   a correlate would have been used instead.
3. **It is a ranking, not a set.** `minimal-v1` was a ranking and was withdrawn for it.
4. **It scores sparse flags backwards.** Measured:

   | | n | median mean rank | median **best** rank | top-50 on ≥1 task |
   | --- | ---: | ---: | ---: | ---: |
   | `fr_*` | 72 | **447** (worst family) | **10** | 88% |
   | everything else | 550 | 289 | 20 | 81% |

   `fr_*` flags are **more** likely to be decisive on some task than the average column, and
   useless on most. `fr_barbitur` is unused on 27 of 33 tasks and rank 50 on `pb_logd`. Mean rank
   would delete all 72; that would be wrong, and it is why Stage 1 does not select.

   Confirmed independently by the project owner: the flags "help XGBoost quite a lot in many
   situations" despite being theoretically redundant with a good fingerprint. They are in scope
   for the cut, but they cannot be cut on mean rank.

---

## 4. Stage 2 — grouped ablation (the decision)

Stage 1 ranks; **Stage 2 decides.** For a candidate reduction, remove the whole group, refit the
same head on the same folds, and measure the change in the dataset's own metric against the
622-column baseline. Groups, not columns, because that is the only way to defeat the gain-splitting
problem in (1): a correlated team has to be dropped together or its members hide behind each other.

The units of the ablation are the sweeps in §0 at reduced resolution — chi at orders 0–3 rather
than 0–7, IC at two normalisations rather than four — plus the `fr_*` block as a whole and in
halves.

**Accept a cut when the mean change across 33 datasets is within fold noise, and no panel is
worse by more than its own fold-to-fold spread.** That is the bar `minimal-v2` was held to
(mean −0.02%, 1 of 33 above fold noise) and there is no reason to lower it.

### Selection bias, and the control for it

Choosing on the same 33 datasets the result is reported on would overfit the spec to them. The
control is **leave-datasets-out**: choose the cut on a subset of the panels, report it on the
held-out ones. A spec that only survives on the datasets that chose it is not a spec.

### What would falsify the whole exercise

A 300-column set whose loss exceeds fold noise on any panel, or whose held-out loss exceeds its
selection-set loss by more than noise. Either result means 622 is closer to the floor than the
effective-rank analysis suggests, and the honest answer becomes a number above 300.

---

## 5. Known confound in the ChemPFN experiment

The TabPFN experiment (13% of pool, 3 seeds) found no reduced variant worse than the full set,
and pruning level with projection. Its own stated confound stands: TabPFN degrades past ~500
features and `full` is 2,670-dimensional against 2,112–2,348 for the reduced arms, so `full`
coming last is partly a dimensionality handicap in the evaluator.

It is suggestive that the cut is cheap. It is not evidence about the descriptors, and the
decision above does not rest on it. The tree route in §4 needs no GPU and no checkpoint, and
answers the question the paper's downstream claims actually rest on.

**Projection is not a shippable route regardless of how it scores**: it changes the scheme hash,
destroys column interpretability, and requires fitting on a corpus — which makes the spec a
function of that corpus and versions it accordingly. Pruning keeps a spec that is a list of
names.

---

## 6. Decisions

*(recorded as they are made, each with the evidence that decided it)*

- **`fr_*` is in scope**, on the owner's call, as theoretically redundant with a good
  fingerprint. It cannot be cut on Stage 1 evidence — see §3(4).

- **DROP 32 OF THE 72 `fr_*` FLAGS. KEEP 40. 622 → 590.** Decided on the Stage 2 ablation below.

### Evidence: the `fr_*` ablation

Split at "reaches the top 50 on ≥3 tasks", giving 40 top / 32 tail. The distribution is smooth
with no natural gap, so the cut point is a choice and is recorded as one. **The statistic is
recomputed leave-one-dataset-out**, so the split never sees the data it is scored on.

Four arms, same head, same stored scaffold folds, 33 datasets. `fr_tail` is a **control**: if the
split means nothing, keeping either half should cost the same.

| arm | columns | median vs 622 | worse on | Wilcoxon vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `full` | 622 | — | — | — |
| `fr_top` — keep the 40 | 590 | **+0.29%** | 18/33 | **p = 0.634** |
| `fr_tail` — keep the 32 | 582 | +0.45% | 23/33 | p = 0.025 |
| `no_fr` — drop all 72 | 550 | +1.25% | 27/33 | p = 0.001 |

Paired across datasets, which removes dataset-level variance:

| comparison | median diff | first arm worse on | p |
| --- | ---: | ---: | ---: |
| `no_fr` vs `fr_top` | +0.94pp | 23/33 | **0.007** |
| `fr_top` vs `fr_tail` | −0.45pp | 11/33 | **0.005** |
| `no_fr` vs `fr_tail` | +0.03pp | 17/33 | 0.930 |

**Three conclusions, in order of how much they are worth.**

1. **The `fr_*` block is load-bearing.** Dropping all 72 costs a median 1.25% and is worse on 27
   of 33 datasets, p = 0.001. The owner's observation that the flags help XGBoost substantially
   is confirmed, and the theoretical argument that a good fingerprint makes them redundant does
   not survive contact with the tree.
2. **Dropping the 32-column tail is free.** `fr_top` against the full 622 is p = 0.634 — not
   distinguishable. 32 columns for nothing measurable.
3. **The split is real, and the control is what establishes it.** `fr_top` beats `fr_tail`
   paired at p = 0.005, and — the sharper statement — keeping the *wrong* 32 is no better than
   keeping none at all (`no_fr` vs `fr_tail`, p = 0.930). The tail carries essentially nothing.

⚠️ **This is also the case for the two-stage method.** Stage 1 alone ranked `fr_*` worst of any
large family and would have deleted all 72, at a measured cost of 1.25%. Stage 2 found that 40
of them are worth keeping and 32 are free. A ranking would have got this exactly backwards.

⚠️ **Two datasets dominate the arithmetic MEAN and it should not be quoted.** `hia` (−17.15%) and
`vdss_lombardo` (−19.56%) have fold noise of 29.9% and 16.5%; they drag `fr_top`'s mean to
−0.62%, which would read as "dropping columns improves the model". Medians and paired tests are
used throughout for this reason.

**Budget after this decision: 590. 290 still to find, and the sweeps in §0 are where they are.**


---

## 7. The sweep ablation — every reduction is free, and so is every control

Eight parametric sweeps, each reduced two ways against the 590 baseline: `_lo` keeps the low
orders and short lags, `_hi` is the control that keeps the same kind of thing at high orders and
long lags. 33 datasets, same head, same stored folds. Positive = worse than baseline.

| sweep | drops | `_lo` median | p | `_hi` median | p | lo vs hi | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vsa | 30 | −0.41% | 0.685 | +1.02% | 0.051 | −0.92pp | 0.060 |
| charge | 12 | −0.19% | 0.469 | +0.28% | 0.480 | −0.24pp | 0.135 |
| rings | 18 | −0.16% | 1.000 | −0.25% | 0.406 | +0.04pp | 0.846 |
| ic | 11 | −0.06% | 0.480 | +0.02% | 0.376 | −0.03pp | 0.406 |
| autocorr | 18 | −0.04% | 0.944 | +0.23% | 0.502 | −0.24pp | 0.304 |
| chi | 35 | +0.14% | 0.447 | −0.21% | 0.846 | +0.29pp | 0.242 |
| walk | 16 | +0.17% | 0.502 | −0.50% | 0.228 | +0.11pp | 0.513 |
| estate | 32 | +0.46% | 0.386 | +0.12% | 0.621 | +0.79pp | 0.367 |

**Every arm is indistinguishable from the 590 baseline.** 24 tests, and the smallest p is 0.051.

### What that does and does not license

**Each sweep is individually over-resolved.** Dropping 11–35 columns from any one of them costs
nothing measurable. That is the finding, and it is consistent across all eight.

⚠️ **THE ORDERING PRINCIPLE IS NOT ESTABLISHED, AND I EXPECTED IT TO BE.** `_lo` never beats
`_hi` significantly — the best is vsa at p = 0.060. Low orders and short lags are *not*
demonstrably more informative than high orders and long lags; **either half of a sweep suffices**.
So the reduction cannot be justified as "keep the informative end", only as "keep half, any
half". That is a weaker and more honest claim, and it means the principle cannot be extended to
an untested sweep without measuring it.

The one directional hint is `vsa`: keeping SlogP and PEOE while dropping SMR, EState and
VSA_EState is −0.41%, while the reverse is +1.02% (p = 0.051). Suggestive, not established, and
one hint out of eight tests at p ≈ 0.05 is what chance produces.

⚠️ **INDIVIDUALLY FREE DOES NOT COMPOSE.** Applying all eight reductions at once removes 172
columns, 590 → **418**. Each was measured with the other seven sweeps still present to absorb the
loss. Once they are all gone, the redundancy that made each one free is gone too. The joint
effect has to be measured; it cannot be summed.

### Next: the combination, and the control that matters most

- `combo` — all eight reductions, 418 columns.
- `random418` — **the control**: 418 columns drawn at random from the 590. If the principled
  combination is no better than a random subset of the same size, then nothing about *which*
  columns were chosen matters and the result is purely about column count. That would be worth
  knowing and would change what HUME_300 can claim.


---

## 8. The combination composes, and WHICH columns go matters decisively

33 datasets, against the 590 baseline. Positive = worse.

| arm | cols | median | worse on | p |
| --- | ---: | ---: | ---: | ---: |
| **`combo418`** | 418 | **+0.17%** | 19/33 | **0.846** |
| `random418` (3 seeds pooled) | 418 | +1.44% | 28/33 | **0.000** |
| `random300` (3 seeds pooled) | 300 | +2.54% | 31/33 | **0.000** |

| comparison | median | first worse on | p |
| --- | ---: | ---: | ---: |
| `combo418` vs `random418` | −1.17pp | 5/33 | **0.000** |

**Three findings, and the middle one is the important one.**

1. **The reductions compose. 590 → 418 is free.** p = 0.846 against the baseline, worse on 19 of
   33 — a coin flip. The non-composition risk was real and did not materialise.

2. **⚠️ WHICH COLUMNS ARE DROPPED MATTERS, DECISIVELY.** A *random* 418 costs a median 1.44% and
   is worse on 28 of 33 (p = 0.000); the principled 418 costs nothing. Head to head, the
   principled set wins on 28 of 33 at p = 0.000. **This is what justifies the whole selection
   exercise** — without it, HUME_300 would be a statement about column count and nothing else.

3. **300 is not free by accident.** A random 300 costs a median 2.54% and is worse on 31 of 33.
   It has to be earned.

### What the principle actually is — and it is not what §7 was testing

§7 found that within a sweep, either half suffices: low orders are not better than high orders.
§8 finds that the principled combination beats a random subset of the same size. Those are not in
tension, and together they identify the mechanism:

> **What matters is COVERAGE ACROSS FAMILIES, not which end of a sweep is kept.**

The principled 418 keeps some of every sweep and every family. A random 418 can, by chance,
strip a small family bare — and the small families are where the non-redundant information is.
That is why stratification wins while ordering does not.

⚠️ This was not the hypothesis. §7 was written expecting low orders and short lags to carry the
signal; that failed at p = 0.060 at best. The principle that survived is the one the control
found, not the one the design proposed, and it is recorded that way round on purpose.

### Next: 418 → 300, stratified

Applying the same rule again — proportional reduction within every family, so coverage is
preserved — against the `random300` control that is already measured at +2.54%. Arms at 250, 300
and 350 to find where it breaks rather than assuming 300 is the right stopping point.
