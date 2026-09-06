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
