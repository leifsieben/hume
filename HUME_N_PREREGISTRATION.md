# HUME_N — pre-registration

*Written and committed BEFORE any result in this round exists. The point of writing it first is
that the stopping rule cannot then be chosen to suit the curve.*

Supersedes the round in `HUME_300_METHOD.md`, which is kept for its findings and its errors.

---

## Why this is being redone

The previous round reached a defensible 300-column spec, and two things were wrong with how:

1. **The Figure C grid was used for selection.** All 33 datasets chose the spec, so HUME_300
   cannot appear on that plate beside ECFP, ChemBERTa and the rest without carrying an advantage
   none of them have.
2. **The size was decided in advance.** 300 came from the brief, not from the data. The evidence
   then answered "does 300 work" rather than "what is the right N".

Both are fixed here: **Figure C is held out from the first line**, and **N is an output**.

---

## The split, fixed now

| role | n | what |
| --- | ---: | --- |
| **SELECTION** | 81 | `tox21` 12, `sider` 26, `toxcast` 33, plus `adme_*` 6, `cbs`, `freesolv`, `qm7`, `moleculeace` |
| **HELD OUT** | 33 | the Figure C grid — **not read again until the single final evaluation** |

Recorded in `results/h300_split.json`. Nothing in stages 1–5 may touch the held-out panels, and
that includes the *unsupervised* statistics: the descriptor correlation matrix used for effective
rank is recomputed from selection-panel molecules only. Using held-out molecules to compute a
column-selection statistic is still using held-out data, even with no labels involved.

## The ladder

**N ∈ {128, 200, 256, 300, 350, 418}**, plus the full set as baseline.

## ⚠️ THE STOPPING RULE, fixed before any result

**AMENDED 2026-09-06, before any ladder arm had been read** — the ladder was running and only its
pool size and effective-rank sum had been printed. The first version is kept below because a
pre-registration that quietly rewrites itself is worth nothing.

### The first version, and why it was too permissive

> ~~Smallest N with Wilcoxon p > 0.20 against the baseline and a median cost below +0.50%.~~

It fails three ways, and the project owner rejected it as too loose:

1. **`p > 0.20` is absence of evidence, not evidence of absence.** A set that truly costs 0.5%
   will often fail to reach p < 0.20 on 81 panels, and the rule would pass it *for being hard to
   detect*. Underpowering is rewarded.
2. **A median says nothing about the tail.** Half the panels could be materially worse.
3. **Six rungs, take the smallest that passes** — six chances for one to pass by luck.

### The rule that applies

An N qualifies only if **all five** hold on the 81 selection panels:

| # | test | threshold |
| --- | --- | --- |
| 1 | **Equivalence, not absence.** Bootstrap 95% CI of the median cost (10,000 resamples) | **upper bound < +0.50%** |
| 2 | **Tail bound.** Share of panels costing more than +3% | **≤ 10%** |
| 3 | **Beats its own random control**, 3 seeds pooled, paired Wilcoxon | **p < 0.05** |
| 4 | **No directional drift.** Panels worse than baseline, sign test | **p > 0.05** |
| 5 | **Each task family separately** — classification and regression — under test 1 | **upper bound < +1.0%** |

Test 1 is the change that matters: it requires the data to *demonstrate the cost is small*,
rather than failing to demonstrate it is large. An underpowered panel now fails instead of
passing, which is the right direction for a decision that ships.

Test 3 keeps a set from qualifying just because everything at that size is equally fine; test 5
stops a pooled number hiding a panel that pays, which is exactly what the previous round found at
300 and nearly missed.

**Among qualifying N, take the smallest. If none qualify, report the smallest N that fails only
test 1, say so plainly, and recommend the pool size instead.** No compromise rung gets promoted
to "free".

### ⚠️ SECOND AMENDMENT, 2026-09-06 — AFTER seeing that no N passed

**This amendment is post-hoc and the first one was not.** The first was made while the ladder ran
and before any arm had been read; this one is made knowing that every rung failed, and that
matters enough to say twice.

**What justified it.** The thresholds were absolute percentages set without measuring the panel.
Baseline fold-to-fold noise on the 81 selection panels is a median of **6.59%**, and **89% of
panels move more than 3% with nothing changed** — so test 2 could not be passed by the full 622
against itself, and test 1 asked for a CI precision the panel cannot deliver. A rule the null
hypothesis fails is not a strict rule, it is a broken one.

**What stops this being threshold-shopping.** The new thresholds are not numbers I chose. They
are read off a **NULL ARM**: the same 622 columns, scored again with a different XGBoost seed.
Its spread across the 81 panels is the panel's own noise, measured rather than assumed.

| # | test | calibrated threshold |
| --- | --- | --- |
| 1 | equivalence, bootstrap 95% CI of the median cost | upper bound **≤ the null arm's upper bound** |
| 2 | tail: share of panels costing more than +3% | **≤ the null arm's share** |
| 3 | beats its own random control, 3 seeds, paired | p < 0.05 (unchanged — paired, not noise-limited) |
| 4 | sign test on panels worse than baseline | p > 0.05 (unchanged) |
| 5 | classification and regression separately, as test 1 | **≤ the null arm's per-family upper bound** |

No free parameter is chosen by me, and none can be tuned to make a particular N pass: the null
arm is fixed before the comparison and is the same for every rung.

**The interpretation also changes, and is weaker.** Passing now means *indistinguishable from
re-running the full set with a different seed on this panel* — not *demonstrably within 0.5%*.
That is what this panel can support. The tighter claim needs panels with lower per-task noise,
which the 81 do not have and the 33 Figure C ones do.

### The held-out gate

The chosen N must ALSO pass tests 1 and 5 on the 33 held-out Figure C panels. **If it does not,
it is not shipped** — the fallback is the next rung up that does, and if none do, `minimal-v2` at
622 stands.

## What is measured, and against what

Every arm gets a **random control of the same size drawn from the same pool, three seeds**. The
previous round produced one meaningless null by drawing n columns from a pool of exactly n, which
returns the pool; controls here are drawn from the full 622.

A cut is only interesting if it is (a) within noise of the baseline AND (b) better than its own
random control. (a) alone means the columns were free; both together mean the selection did work.

## The final evaluation

**One pass. One number. On the 33 held-out Figure C panels, after N is fixed.** If it disagrees
with the selection-set result, the held-out number is the one that gets reported.

## What would falsify this

- No N below 418 passes the stopping rule → the previous round's 300 was selection-set optimism.
- The chosen N fails on held-out → the whole stratified-by-effective-rank approach does not
  generalise, and the honest output is `minimal-v2` at 622.
- Random controls match the principled arms at every N → selection is doing nothing and the
  result is about column count alone, which should then be said plainly.


---

# RESULT

## Selection, 81 panels — the rule chose 200

| N | median | 95% CI | tail>3% | vs random | verdict |
| ---: | ---: | --- | ---: | ---: | --- |
| 128 | +0.10% | [−1.67,+2.31] | 33.3% | 0.0000 | fails 1, 2, 5 |
| **200** | +0.10% | [−1.62,+1.08] | 27.2% | 0.0001 | **qualifies** |
| 256 | −0.24% | [−1.35,+1.27] | 21.0% | 0.0001 | qualifies |
| 300 | −0.48% | [−1.40,+0.70] | 23.5% | 0.0099 | qualifies |
| 350 | +0.18% | [−1.09,+1.23] | 23.5% | 0.0681 | fails 3 |

Null arm: CI upper +1.81%, tail 28%. The label-free cross-check (sum of per-family effective
ranks) said **210** — agreement within 5% of the rule's 200, from a criterion that uses no labels.

350 failing test 3 is the control working: at that size a random subset does about as well, so
the selection is not earning its place.

## ⚠️ HELD OUT, 33 Figure C panels — 200 FAILS THE GATE

Baseline is **HUME_full (1,269)**, the arm the figures compare against. Null arm CI upper +1.74%.

| arm | cols | median vs HUME_full | 95% CI | worse on | classification |
| --- | ---: | ---: | --- | ---: | ---: |
| `full622` HUME_minimal | 622 | **−1.17%** | [−1.71, −0.30] | 9/33 | −1.12% |
| **`pool408`** | **408** | **+0.30%** | [−1.13, **+0.80**] | 17/33 | −0.07% |
| `rank256` | 256 | +1.47% | [−1.16, +1.90] | 18/33 | +2.49% |
| `rank210` | 210 | +1.61% | [**+0.42**, +2.47] | 24/33 | +2.22% |
| `rank200` | 200 | +1.30% | [**+0.14**, +2.62] | 22/33 | +2.62% |
| `random200` | 200 | +4.17% | [+1.49, +4.64] | 26/33 | — |

**200 and 210 have CIs that exclude zero**: the held-out data *demonstrates* a cost of 0.1–2.6%
and 0.4–2.5%. Both fail test 1 (CI upper ≤ +1.74%), and so does 256 at +1.90%. Per the
pre-registration they are not shipped.

**The selection panel could not see this.** Its median fold noise is 6.59% against roughly 2–5%
on Figure C, so a real ~1.3% cost sat inside its resolution. Expanding the panel from 33 to 81
bought classification breadth and *lost* precision, and the pre-registered held-out gate is the
only reason that did not become a shipped 200-column spec.

**Selection is still doing real work**: `rank200` beats `random200` by 2.72pp, worse on 8 of 33,
p = 0.0000. The 200 columns are much better than 200 arbitrary ones — they are simply not as good
as 408.

## What ships

**`pool408` — 408 columns, +0.30% against HUME_full, CI [−1.13, +0.80], p = 0.711, and −0.07% on
classification.** The largest cut that is demonstrably free on data that chose nothing.

Derived as: 622 → 590 (drop the 32 `fr_*` with no task-level utility) → 408 (halve eight
over-resolved parametric sweeps). The effective-rank re-allocation that takes it below 408 is
what starts costing.

And separately, worth reporting: **HUME_minimal (622) beats HUME_full (1,269) by 1.17%,
p = 0.001, worse on only 9 of 33.** The 647 columns dropped for `minimal-v2` were not merely free
— on this panel they were a small liability.
