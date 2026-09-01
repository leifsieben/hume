# HUME_minimal: deriving a reduced column spec

*How `molhume.minimal_columns()` was chosen, what it costs, and what would falsify it.*

The method follows ChemPFN's `DESCRIPTOR_SELECTION_METHOD.md`, whose framing — coverage rather
than compression, label-free, pivoted QR rather than correlation clustering — is right and is
adopted here unchanged. Three things are done differently, each because measuring them changed
the answer. They are in §4, §5 and §6, and the resulting operating point is **800**, not 640.

---

## 1. The objective

**Find the smallest ordered column set `S` such that every column not in `S` is recoverable
from `S`.**

This is a **coverage** criterion, not a compression one, and the difference decides the answer.
A dropped column costs a downstream model nothing if it can be reconstructed from what remains —
the model rebuilds it internally. What is lost is only a column's *unique* variance. So the
question is never "which columns explain the most variance", which keeps the loudest columns,
but "from which columns can we rebuild all the others".

## 2. Label-free, and not negotiable

The procedure reads only the descriptor matrix. No target, no assay, no benchmark. Selecting on
labels leaks whatever benchmark supplied them and picks descriptors suited to the chemistry that
happened to be in the label set. Decisively: **a descriptor spec is a permanent contract**, and
every downstream user inherits it, including users whose chemistry nobody in this loop has seen.
Redundancy is a property of the molecules; informativeness is a property of somebody's targets.
Only the first is safe to select on.

The downstream benchmark in Figure C is therefore a **test** of this spec, never an input to it.

## 3. Pivoted QR, not correlation clustering

Cluster-by-|ρ| cannot see multi-column dependence: a column can be an exact linear combination
of three others while correlating weakly with each, so it survives the filter and adds nothing.
Column-pivoted QR orders columns by how much *new* direction each adds given those already
chosen, and it selects **actual columns** rather than components — PCA answers "how many
dimensions are there" but its components cannot be shipped as a descriptor list.

## 4. What the missing values force

**Only 4.6% of corpus molecules are finite in all 1,269 columns.** Dropping incomplete rows
would leave ~1,100 rows against 1,269 columns — rank-deficient, and any `k` derived from it would
be an artifact of the deficiency rather than a fact about chemistry. So: gate out columns below
50% finite (the same usability gate HUME's own deduplication uses; it removes 2), then impute the
remainder at the **column median**, median rather than mean because these distributions have
heavy tails. 0.49% of cells on the representative sample, 2.9% on the adversarial one.

Imputation makes a column slightly *more* predictable than it truly is, which biases toward
dropping. `tools/minimal_select.py --finite-only` re-runs the whole derivation on the 1,024
columns that are 100% finite, with no imputation at all, as a sensitivity check.

## 5. Both distributions, or the spec is only for one

Recoverability holds on the distribution you measure it on. Two samples, and the ordering is
derived on **both stacked**, not derived on one and checked against the other:

- **repA / repB** — 24,000 molecules each, disjoint, from `data/corpus1m/selected.txt`, the 1M
  *training* corpus. Not a benchmark set.
- **adv** — 24,000 from `cpp/hard.smi`: salts, mixtures, unusual elements, size extremes.

**This is not a formality, and it moved the answer.** Deriving the ordering on the representative
corpus alone closes coverage at k=704 — and leaves `Phi` and `Kappa2` at **R² = 0.07** on salts
and mixtures. Both descend from `HallKierAlpha`, whose alpha table is solved over the
(element, hybridisation) pairs the training corpora contained, and which METHODS.md §7.2 already
records as failing on organometallics and metal salts. On that chemistry they carry unique
variance nothing else reproduces. A spec derived on drug-like molecules alone would have silently
dropped them and been wrong for anyone working with salts.

Each sample is z-scored with its **own** statistics before stacking. Pooling first and scaling
after lets whichever sample has the wider spread set the scale and dominate the pivoting.

⚠️ The spec is tied to `standardize="none"`, recorded in `_minimal.py`. That setting changes
descriptor values for every multi-fragment input, so a spec derived under one does not transfer.

## 6. In-sample R² is not enough, and here it is actively misleading

The acceptance criterion is the **worst-case R²** over dropped columns — never the mean, since a
set where 799 columns sit at 0.999 and one at 0.40 has lost something real and an average hides
exactly that — plus the count below 0.99.

But **the kept set is numerically singular**: its condition number reaches 1e15 by k=512, against
a double-precision epsilon of 2.2e-16. Unregularised least squares in that regime produces
coefficients that fit collinear directions, and they do not transfer at all. Measured, fitting
the reconstruction on repA and scoring it on the disjoint repB:

| k | in-sample worst R² | held-out worst R², **unregularised** |
| --- | --- | --- |
| 400 | 0.852 | 0.803 |
| 512 | 0.914 | **−3.2 × 10¹⁸** |
| 800 | 0.991 | **−1.1 × 10²⁰** |

An in-sample number of 0.99 alongside a held-out number of −10²⁰ is not coverage; it is a solve
exploiting a singularity. **This is the single most important correction to the method**, and any
`k` chosen from in-sample R² alone — including the 640 in the source document — is reporting a
quantity that does not survive contact with a second sample.

The fix is a ridge penalty of `0.01·n` on the reconstruction fit, chosen from a grid
{1e-6, 1e-4, 1e-2, 1, 1e2}·n by held-out worst-case R²; it was best at every k tested and the
answer is flat from 1e-6 to 1e-2. It is not a knob on the answer — it exists so the question can
be asked at all.

⚠️ Linear recoverability errs in the safe direction. A column that is a *non-linear* function of
the kept set looks unrecoverable and is kept unnecessarily, so the method keeps slightly too
much — which is the right way to be wrong.

## 7. The coverage curve

Fitted on repA and the adversarial set stacked; scored in-sample on each of the three, and
held-out by fitting on repA and scoring on the disjoint repB.

| k | worst in-sample (min of repA/repB/adv) | cols < 0.99 | worst held-out | held-out < 0.99 | cond(kept) |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.585 | 991 | 0.316 | 989 | 2.5e1 |
| 400 | 0.817 | 719 | 0.803 | 738 | 5.9e1 |
| 512 | 0.913 | 516 | 0.885 | 495 | 1.1e15 |
| 640 | 0.957 | 259 | 0.952 | 240 | 1.3e15 |
| 704 | 0.977 | 105 | 0.957 | 111 | 1.4e15 |
| 768 | 0.981 | 17 | 0.973 | 30 | 2.2e15 |
| **800** | **0.990** | **1** | **0.986** | **16** | 3.1e15 |
| 832 | 0.992 | 0 | 0.989 | 7 | 3.3e15 |
| 896 | 0.994 | 0 | 0.989 | 2 | 6.0e15 |
| 1024 | 0.999 | 0 | 0.990 | 0 | 7.3e15 |

## 8. Validation before freezing

**Stability across disjoint samples (§7.1).** Re-deriving the ordering on repB + adv and
comparing the top-k set with the shipped one:

| k | overlap | Jaccard |
| --- | --- | --- |
| 640 | 600 / 640 | 0.882 |
| **800** | **779 / 800** | **0.949** |
| 896 | 878 / 896 | 0.961 |

At 800 the two independent derivations agree on 97.4% of the set. At 640 they disagree on 40
columns, which is the criterion being under-determined there — and the honest response to that
is a larger k, not a coin flip.

**Per-family survival (§7.2).** No chemically meaningful family is near-eliminated at k=800:
`autocorr` 62%, `chi` 62%, `spectral` 69%, `estate` 66%, `eta` 45%, `pathcount` 27%,
`topomisc` 31%. The low two are families whose members are near-duplicates by construction
(path counts of increasing length). The only family at 0% is `alias`, whose single column is a
documented duplicate of another — correct, not a flag. *This differs sharply from the source
document's run, which saw `chi` fall to 7% and flagged it; deriving on the pooled sample does
not gut it.*

**Conditioning (§7.3).** Reported in the curve above, and the reason §6 exists.

**Constant columns (§7.4).** 1 on repA, 2 on repB, 1 on adv — dropped free by the gate.

## 9. Why 800

Every criterion lands in the same place:

- in-sample coverage effectively closes — 1 column below 0.99 on any of the three samples;
- held-out worst-case is 0.986, and only 16 of 467 dropped columns fall below 0.99;
- the selection is stable, 779/800 across independent derivations;
- no family is gutted.

640 is defensible on the representative corpus alone and on in-sample numbers. It is not
defensible once salts are included (`Phi`, `Kappa2` at R² 0.07) or once the reconstruction has to
transfer to a second sample. **The ordering is the product, not the number** — `minimal_curve()`
publishes the whole table so a memory-constrained user can take 400 knowing exactly what they
gave up.

## 10. Frozen like an interface

`src/molhume/_minimal.py` records the ordering, the sample, the seed, the `standardize` setting,
the library and RDKit versions, and the curve. Treat a change as a **breaking** change: anyone
who cached features under `minimal-v1` must be able to tell that they did. New specs are added
under a new name rather than edited.

## 11. What would falsify this

The reconstruction criterion is a *proxy* for the thing anyone actually cares about, which is
whether a model does as well on 800 columns as on 1,269. That is measured separately and
independently, as an arm in Figure C — using the benchmark as a **test** of the spec, never as
an input to it (§2). If `hume_minimal` loses materially to `hume` there, the proxy is wrong and
this document is what should be revised.
