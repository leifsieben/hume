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

⚠️ **Why the residual 1–6% is acceptable, and the condition attached.** These correlations are
not 0.999, and the reason is structural: the autocorrelation is a **sum over pairs**, and
`sum(g(w)) != g(sum(w))`, so monotone-relatedness of two *tables* does not survive the
aggregation. Concretely, if `w2 ~ a*w1 + b*1`, then

    ATS_k(w2) = a^2 * ATS_k(w1) + 2ab * (w1-weighted pair count) + b^2 * (pair count at lag k)

The second weight injects a **topology** term — the count of atom pairs at distance k. That is
genuinely new relative to `w1` alone, which is why a tree could use both. But it is *not* new
relative to the library, because pure-topology autocorrelations are already emitted.
**Condition: this argument dies if a future cut also removes the topology columns.** They must
be retained, or these five weights have to be reconsidered.

⚠️ **The coverage argument is latent, not measured.** NaN rate for all eight weights is 0.000%
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

### 3.2 Substructure matching (`fr_*`, 75 columns)

Tested by consumer inversion against the ECFP6 that mol-hume emits alongside them, using the
same untuned XGBoost the benchmark grid uses, on 20,000 corpus molecules.

**Detection AUROC: median 1.000. 67 of 72 scored columns at >= 0.99. None below 0.90.** The
fingerprint carries the presence of essentially every fragment pattern.

⚠️ **The first version of this test used R^2 and was wrong.** For a column that is zero on 99.95%
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

**Decision: drop the family, with one stated condition.** See §5, open question 1.

## 4. Running total

| stage | columns |
| --- | ---: |
| emitted today | 1,269 |
| after the autocorrelation weight cut (§3.1) | 1,077 |
| after dropping `fr_*` (§3.2) | **1,002** |

Both cuts are **constructual** — same quantity, different scale; or same information, already
present elsewhere in the output. No parametric cut has been made yet, and that is where the
remaining reduction lives.

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
4. **`minimal-v1` is published** in mol-hume 0.2.0+ as a frozen contract. Whatever replaces it
   ships under a new name; the old one is not edited.
