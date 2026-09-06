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

> **Choose the smallest N on the ladder for which the selection-set cost is within noise of the
> full set — defined as Wilcoxon p > 0.20 against the baseline across the 81 panels, AND a median
> cost below +0.50%.**
>
> If several N qualify, take the smallest. If none qualify, report the smallest with p > 0.05 and
> state that it is a compromise rather than a free cut.

Two thresholds rather than one because they fail differently: p alone can pass on an
under-powered panel, and a median alone can pass while half the panels are badly hurt.

**Cross-check, agreed in advance:** the sum of per-family effective ranks at 95% within-family
variance is an independent estimate of N that uses no labels at all. If the ladder's answer and
that sum agree to within about 20%, that is corroboration. If they disagree, the disagreement is
reported and the ladder wins, because it measures utility and the other measures only variance.

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
