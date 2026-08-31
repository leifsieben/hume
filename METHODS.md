# Methods

## 1. Selecting the descriptor set

### 1.1 The candidate space

We start from the union of every 2D descriptor in RDKit (217) and Mordred (1,613), plus the 193
columns HUME computes that have no upstream name — **2,023 candidates**. 3D Mordred descriptors
are excluded throughout: they require conformers (~140 ms/molecule), two orders of magnitude
outside any budget considered here.

Our own columns are in the same pool as everything else. They had previously been exempt from the
filter that defined the rest of the set, and an audit found that exemption was not harmless: of
193, twenty-four were unusable and sixty-four were ≥0.99 redundant, including `C4` ≡ `n4Ring` and
`C5_arom` ≡ `n5aRing` at ρ = 1.0000, `path4` ≡ `MPC4` at 0.9997, and `T_absum` ≡ `n_EZ_any` —
two of our own columns identical to each other.

### 1.2 The corpus

Descriptor redundancy is a property of the chemistry it is measured on, so the corpus is
**stratified rather than sampled from one source**. We draw 40,000 molecules at random from
ChEMBL (145,070 available) and pool them with the molecules of sixteen benchmark datasets, then
take **4,000 molecules from each of five heavy-atom strata** — 0–15, 15–25, 25–35, 35–55, 55+ —
for a corpus of 20,000.

Two earlier failures motivate this. The original corpus was the *first* 20,000 lines of a ChEMBL
file, which is not a sample: that prefix averages MW 416.1 against 438.9 for a random draw from
the same file, a 6% bias toward smaller molecules arising from nothing but where the read
stopped. It was also ChEMBL-only, while the evaluation spans QM9 (MW ≈ 120) and CycPeptMPDB
cyclic peptides (MW ≈ 1,310).

Using benchmark molecules here is not leakage: the selection is unsupervised and never sees a
label. The descriptors should be deduplicated on the chemistry they will be applied to.

### 1.3 Usability gates

A column is a candidate only if it is (i) finite on ≥50% of molecules, (ii) non-constant, and
(iii) differs from its own most-common value on more than 0.1% of rows. **1,693 of 2,023 pass.**

The finite threshold is 50% rather than the more usual 95% because the evaluation harness maps
non-finite values to NaN and XGBoost treats NaN natively as missing; a column defined on 60% of
molecules and informative there is usable. The variance gate exists *because* of that loosening:
without it, near-constant columns enter the correlation stage and behave pathologically (§1.5).

### 1.4 Redundancy criterion

Redundancy is measured as **|Spearman ρ| computed within each stratum, and a pair is scored by
its minimum across the five**. Two descriptors are redundant only if they are redundant
*everywhere*; a pair that collapses on drug-like space and separates on peptides is kept.

This is a direct response to a measured failure of the pooled-corpus version. Columns dropped at
|ρ| ≥ 0.99 on a ChEMBL prefix turned out to be only 0.88–0.96 correlated on the `bioavail`
dataset, and restoring them was worth **+0.016 AUROC** there (and +0.006 on `hia`), while being
worth nothing measurable on any classification set above 7,000 molecules.

**Correlations use pairwise-complete observations and no imputation.** The previous
implementation median-imputed NaN before ranking. Since 4,099 column pairs share a missingness
pattern, both members received their median on exactly the same rows, and that manufactured
agreement was counted as evidence of redundancy. The pairwise-complete Pearson-on-ranks is
computed in closed form from four matrix products, so no pair is ever compared on a row where
either member is undefined.

### 1.5 Sparse columns are judged on support overlap, not correlation

Any two columns that are zero on 99.9% of molecules correlate at ≈0.99 for that reason alone. In
a first pass this let one of our autocorrelation columns "absorb" nine mutually unrelated rare
features — nine-membered rings, isothiocyanates and a boron E-state among them.

A column whose modal value covers ≥90% of its finite rows is therefore treated as
**sparse-binary-like**, and any pair involving one is scored by the **Jaccard overlap of the
non-modal supports** — do they fire on the same molecules? — rather than by correlation. A sparse
column paired with a dense one scores near zero, which is the correct answer. Thirty-three pairs
passed the correlation test and failed this one.

### 1.6 Cover, and which member survives

Survivors are chosen by a **greedy cover in ascending compute cost**: a candidate is kept unless
it is redundant with something already kept, so the survivor of each group is its cheapest member.
Our own columns are computed inside a single C++ pass, so their marginal cost (~0.1 µs) is far
below Mordred's tens to hundreds, and they win every true duplicate. The upstream name is recorded
as an **alias** on the survivor, so a column can be reported as "Mordred's `n4Ring`, our
implementation" rather than introducing a new name for a quantity that already has one. 257 of the
survivors carry such an alias.

**The cover is deterministic but not unique.** Given the cost table and a stable tie-break the
result is reproducible, but a different ordering yields a different, equally valid cover. We do
not claim minimality.

### 1.7 The threshold raises candidates; mechanism decides

A numerical |ρ| ≥ 0.99 is a reason to *examine* a pair, not to delete a column. Every absorbed
pair is recorded with its partner, its per-stratum correlations and whether the two columns come
from the same family. All **142 cross-family pairs were reviewed by hand**, since that is where a
numerical coincidence is most likely to be standing in for a mechanism that is not there.

130 were unambiguous. Roughly sixty are the same descriptor under two package names (Mordred's
`MoeType` module wraps RDKit, so `PEOE_VSA1`, every `SlogP_VSA*`, `EState_VSA*`, `VSA_EState*`,
`SMR_VSA*` and `LabuteASA` appear in both). Others are established identities: RDKit's `Chi*v` is
bit-identical to Mordred's `Xp-*dv`; an E-state atom-type count is a count of the corresponding
functional group (`NddsN` ≡ `fr_nitro`).

Twelve were flagged and eleven dissolved under measurement:

* **`Spe`, `Sare`, `Sse`, `Si`, `ATS0se`** are all ≥0.998 correlated with plain `nAtom`. Each is
  an *extensive* sum over atoms, so all are molecular size under a different weighting. That the
  weightings are physically different properties is beside the point: none of these descriptors
  is measuring the property.
* **`Mse`, `Mpe`, `Mi`, `Mare`** are *intensive* (0.06–0.16 against `nAtom`) and pair as mean(p)
  against mean(p²) of the same property. Not redundant by coincidence — redundant by definition.
* **`ETA_eta_RL` ≡ `Xp-1d`** at ρ = 1.0000 is an identity of construction. Mordred's
  `EtaCompositeIndex(reference=True, local=True)` is a sum over *edges* of √(γᵢγⱼ) evaluated on
  the all-carbon reference alkane, where γ depends only on vertex degree — the same functional
  form as the Randić index.
* **The Barysz pairs** (`VR1_D`/`VR1_DzZ`, `VR2_D`/`VR2_DzZ`, `VR3_D`/`VR3_Dzv`) required a test
  the size-stratified run could not provide, since the Barysz matrix exists to inject *heteroatom*
  information. Re-partitioning the same matrix by heteroatom fraction, ρ decays from 0.9999 to
  0.9982 as heteroatom content rises — the Z-weighting does something — but never enough to clear
  the threshold, even in the most heteroatom-rich decile.

### 1.8 Result

**2,023 candidates → 1,693 usable → 1,327 descriptors**, of which 974 are Mordred's, 193 RDKit's
and 160 ours. 366 columns were absorbed. Exactly one member of each redundant group survives.

---

## 2. Implementation status

**1,035 of the 1,327 are implemented in C++; 292 are not.** This is a real gap and is stated here
rather than in a footnote: HUME's current 1,266 columns were built against an earlier and stricter
selection that kept 864, and the corrected pipeline (§1.4, no imputation) keeps materially more.

Outstanding, by family:

| columns | family | note |
|---:|---|---|
| 85 | assorted RDKit / Mordred singletons | `MinPartialCharge`, `ExactMolWt`, `fr_lactam`, … |
| **52** | autocorrelations weighted by intrinsic state (`s`) | the machinery exists; the property vector does not |
| 38 | Barysz and adjacency spectral (`SpAbs_A`, `VR1_A`, …) | needs an eigensolver on weighted matrices |
| 31 | ring and atom counts | mostly cheap |
| 29 | ETA (Extended Topochemical Atom) | `ETA_alpha`, `ETA_shape_*`, `ETA_beta`, … |
| 25 | E-state per atom-type extremes and spectral sums | |
| 20 | BCUT eigenvalue pairs on further weightings | four eigensolves each |
| **12** | autocorrelations weighted by mass (`m`) | as for `s` |

The two autocorrelation weightings are the cheapest to close — 64 columns that reuse existing
machinery and need only the per-atom mass and intrinsic-state vectors, both of which already cross
the boundary. The spectral families (Barysz, BCUT, adjacency) are the substantial work.

Until these land, the shipped block is the 1,035-column intersection, and any claim about "the
1,327" is a claim about the *selection*, not about what the package computes today.

---

## 3. How the descriptors are computed

### 3.1 The thesis

RDKit and Mordred compute each descriptor by walking the molecule from scratch. Ask for two
hundred descriptors and the molecule is walked two hundred times, in Python, re-deriving the same
atom degrees and the same shortest-path matrix each time.

Almost every 2D descriptor in either catalogue is a function of a *small fixed set* of per-atom
and per-bond quantities plus the graph. HUME computes those once, in C++, after which every
descriptor is a reduction over arrays already in cache.

### 3.2 What crosses the boundary

The complete input is 21 values per atom or bond. `src/hume_core/bindings.cpp` defines the layout
and asserts it against `src/hume/_extract.py`, so a mismatch raises rather than silently
transposing columns.

**Per atom (13 integers):** atomic number, degree, total hydrogens, formal charge, hybridisation,
aromatic flag, in-ring boolean, CIP code, ring-membership count, total valence,
`_ChiralityPossible`, chiral tag, isotope.
**Per atom (2 doubles):** mass, Gasteiger partial charge.
**Per bond (6 integers):** the two atom indices, conjugated, in-ring, a SMARTS-order code, and
RDKit's `BondType` enum.

Four are carried rather than re-derived, each for a measured reason:

* **Ring count alongside the ring boolean**, because SMARTS asks both questions independently:
  `[R]` is membership, `[R1]`/`[R2]` are counts, and the count cannot be recovered from the
  boolean.
* **Total valence**, because recomputing it as `round(Σ bond orders) + nH` disagrees with RDKit on
  **11,238 of 575,571** corpus atoms — aromatic bonds and hydrogens pass through RDKit's own
  rounding rule.
* **The ring flags**, from RDKit's single ring perception rather than a second perception C++-side.
  Perception is numbering-dependent (§4.2), so a second one is a second chance to disagree.
* **Isotope**, not derived from mass: `getMass()` reports that an atom *is* labelled, not which
  isotope it carries.

From these, five derived primitives are computed once per molecule: the all-pairs distance matrix,
the ring set, Crippen (logP, MR) contributions, E-state indices, and Labute ASA contributions.

### 3.3 Families, grouped by the input they need

The grouping is by *what a family consumes*, not by what its descriptors mean, because that is
the axis the implementation is organised on: families sharing an input share the expensive part.

**Needs only the atom and bond arrays.** `constit` (43) — hybridisation census, atom and bond
counts, molecular weight, `FractionCSP3`, Lipinski/Ghose/Veber filters, `SLogP`, QED, SPS.
`rdkcore` (21) — RDKit ring counts, `HeavyAtomMolWt`, `Phi`, `FpDensityMorgan1-3`, stereocentre
counts. A single pass over the atom table.

**Needs per-atom contribution vectors.** `vsa_bins` (76) — `SlogP_VSA*`, `SMR_VSA*`, `PEOE_VSA*`,
`EState_VSA*`, `VSA_EState*`, plus `MolLogP`, `MolMR`, `TPSA`, `LabuteASA`. `estate` (158) —
Kier–Hall electrotopological state as a per-atom-type count and sum over 79 types.

A *VSA* descriptor bins a per-atom property against per-atom surface area: `SlogP_VSA3` is the
surface area carried by atoms whose Crippen logP contribution falls in bin 3. The bin edges are
module-level constants in RDKit, not per-molecule work. This is the thesis at its sharpest —
verified on 6,000 adversarial molecules, **62 columns** (the 57 `*_VSA`, `LabuteASA` and the four
E-state extremes) reconstruct **bit-exactly** (rtol 1e-9) from four per-atom vectors. Once the
vector is paid for, each column is a single `bisect_right`.

Three details break the reconstruction silently, all found by measurement: `LabuteASA` returns its
hydrogen term separately from the heavy-atom vector; `PEOE_VSA` does not clamp NaN charges
(elements without Gasteiger parameters), which fall through into the final bin; and `MolLogP`/
`MolMR` sum over the H-added molecule while every `*_VSA` column does not.

**Needs the distance matrix.** `autocorr` (540) — Moreau-Broto, Moran and Geary autocorrelations
over 6 variants × 9 lags × 10 atomic weightings. `topocharge` (21) — Galvez indices.
`topomisc` (15) — walk counts, Wiener index, diameter, `ABCGG`. `infocontent` (33) — Shannon
information content of atom-equivalence classes at orders 0–5.

An autocorrelation at lag *k* sums a product of an atomic property over every pair of atoms
exactly *k* bonds apart. The *weighting* is that property — mass, van der Waals volume, Sanderson
electronegativity, polarisability, intrinsic state, Gasteiger charge. The variants differ only in
normalisation. One distance matrix and ten property vectors yield all 540 columns.

**Needs subgraph enumeration.** `chi` (40) — Kier–Hall connectivity over paths, chains, clusters
and path-clusters. `pathcount` (11). `ringcount` (49). A chi index of order *n* sums, over every
subgraph of that order, the product of 1/√δ across its atoms. The four *shapes* are why this
requires true subgraph enumeration rather than path walking, and why it is the costliest family.

*A naming collision worth stating:* `chi.h` implements Mordred's subgraph chi (`Xp-*`);
`hume_blocks.h` implements RDKit's Kier–Hall chi (`chi0n`…`chi4v`). They share no code, and
RDKit's are registered in lowercase.

**Needs SMARTS matching.** `frag` (76) — 74 RDKit `fr_*` counts plus `NHOHCount` and
`HeavyAtomCount`, against a compiled SMARTS program, with a second program for QED's 116
structural alerts.

**Mixed.** `blocks` (182) — RDKit's `chi0n`…`chi7v`, `Kappa1-3`, `HallKierAlpha`, `BalabanJ`,
`BertzCT`, `Ipc`, the four `BCUT2D_*` eigenvalue pairs.

### 3.4 Cost

Measured end-to-end on `c7i.4xlarge` (16 vCPU), flat across 10⁴–10⁶ molecules:

| arm | µs/molecule | hours per 10⁹ |
|---|---:|---:|
| ECFP4 alone | 16.6 | 4.6 |
| **HUME** (1,266 descriptors + ECFP6) | **124** | **34** |
| ECFP4 + RDKit-180 | 444 | 123 |
| ECFP4 + Mordred-685 | 4,683 | 1,301 |
| ECFP4 + all descriptors (Python) | 6,190 | 1,722 |

Roughly **50× faster than the equivalent Python descriptor block**, for the same information.

---

## 4. Divergences from the reference implementations

For the overwhelming majority of columns HUME computes exactly what RDKit and Mordred compute, in
C++ rather than Python, and is verified bit-exact against them. The governing rule is a single
test:

> Reproduce a **quirk**; diverge from an **ill-posed definition**. Is the upstream descriptor a
> function of the molecule?

### 4.1 Upstream bugs we reproduce deliberately

A quirk returns the same wrong answer every time. It is part of the descriptor's de-facto
definition, every published value computed with that package carries it, and matching it is what
makes our numbers comparable to the literature. Two are reproduced and commented at the site:

* **Five E-state SMARTS rows whose missing semicolon voids an aromaticity constraint.** The
  pattern was evidently intended to require aromaticity and does not; the atoms it types are
  therefore a superset of the intended ones. We type the same superset.
* **`[SeD2H0]` decoding as an element-number query** with no aromaticity constraint at all.

Both are deterministic. Diverging would make our values incomparable with every Mordred result in
the literature, for a definition nobody has agreed to change.

### 4.2 Ill-posed definitions we diverge from

An ill-posed descriptor returns *different values for the same molecule* depending on atom
numbering or which Kekulé structure the perceiver chose. There is nothing to be exact against.

**Detection is mechanical**: permute the input ordering and recompute; any column that moves is
ill-posed. The screen must shuffle **bonds, not only atoms** — `Chem.RenumberAtoms` permutes atoms
and leaves the bond list alone, while RDKit's ring perception reads the bond list. We shipped the
weaker atom-only screen for a period. `O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2` is
stable across **201** atom renumberings and yields two different ring sets the moment bond order
is shuffled as well.

**(i) `InformationContent`.** Two independent defects. `InformationContentBase` sets
`kekulize = True`, so atom-equivalence codes are built on a Kekulé structure and an aromatic bond
enters as SINGLE or DOUBLE depending on which structure was chosen. And `BFSTree._expand` mutates
a visited set while iterating over it: two adjacent siblings at the same depth are both in the
tree and neither is visited when the loop starts, so whichever the dict yields first claims the
other as its child — and dict order is insertion order, which is neighbour order, which is atom
numbering. **32.3% of the first 2,000 adversarial molecules change at least one IC column under a
single input permutation.** We keep the aromatic bond's own bond-type symbol rather than
kekulizing, and layer the tree by graph distance. Orders 1–5 differ from Mordred by design;
order 0 is unchanged.

**(ii) Ring perception.** `Chem.GetSymmSSSR` is not a function of the graph. The SSSR *basis* is
stable; what flips is whether `symmetrizeSSSR` finds a symmetry-equivalent extra ring of a size
already present. `C1=CC2C3C(C=C1)C23` gives ring sizes (3,3,7) on 33 of 60 numberings and
(3,3,7,7) on the other 27; brute force over every simple cycle confirms the larger answer is the
**relevant-cycle set**, the object `symmetrizeSSSR` is reaching for and reaches only sometimes. We
perceive rings on a skeleton rebuilt from scratch — *n* carbons in canonical-rank order, bonds
added in sorted rank order — so that ring perception, which reads only the graph, is asked exactly
the right question with bond order under canonical control. Over 100,000 molecules × 49 columns ×
5 numberings, **22 molecules move before and 0 after**. It changes RDKit's answer on 32 molecules,
all 32 independently confirmed unstable. RDKit's own 13 ring columns move on those 32 too; the
alternative is two different ring sets inside one feature vector.

*An earlier repair was wrong and the record says so.* Canonical atom ranks with rings compared by
(size, sorted rank vector) left 3 of 100,000 still moving and made those 3 worse, because
canonical ranks fix atom numbering and not bond order, which is the axis that decides.

**(iii) Aromaticity**, where HUME does its own ring reasoning rather than inheriting the
boundary's flag: a ring sulfur carrying an exocyclic double bond is a sulfoxide — pyramidal,
therefore not aromatic; and "a bond in an all-aromatic ring is aromatic" must run *after*
perception, not during it, since a fused system can contain ring bonds belonging to no tested
subset.

**How to state this:** HUME reproduces RDKit and Mordred bit-exactly for all well-posed columns.
Three families are ill-posed upstream, and for these HUME implements the well-posed definition the
upstream code was evidently reaching for and is deterministic under atom and bond permutation.
That is a *stronger* guarantee than bit-exactness, not a weaker one.

### 4.3 Our own descriptors

160 of the 1,327 have no upstream equivalent. They fall into three groups.

**Extended families.** Autocorrelation lags and chi orders that Mordred defines but leaves empty,
computed with Mordred's own formulas. These are not novel definitions; they are the same
descriptor evaluated where the reference implementation stops.

**Ring and path summaries** — per-ring-size counts and path-length distributions with their own
maxima and means. Several of these were duplicates of Mordred columns and were removed by the
audit in §1.1; what survives is what the same filter that governs everything else allows.

**Stereochemistry.** The one place we add something the catalogues genuinely lack. Both RDKit's
and Mordred's 2D descriptor sets are almost entirely blind to stereochemistry: the resolution
analysis finds Morgan fingerprints and every published embedding scoring at or near chance on an
inverted stereocentre. Our block sums *signed* CIP parity (R = +1, S = −1) rather than counting
stereocentres, which generalises across scaffolds in a way a bit-pattern does not, and it is the
single clearest reason our descriptor block resolves stereochemical change where a fingerprint
does not.

Being ours is not itself a justification, and the audit in §1.1 is the evidence: our columns were
put through the same filter as everything else, and a third of them did not survive it.

---

## 5. Cost triage: columns dropped after implementation

Section 1 selected on **redundancy** and never on **cost**. Nothing in the greedy cover asked
whether a surviving column earned its compute. Once all 228 remaining descriptors were
implemented in C++ and could be timed, that question was asked separately, and 20 columns were
dropped. This section records why, because a drop is a claim that has to be defensible later.

### 5.1 The test: reconstruction from a cheap basis

A column earns its place if a model cannot already get its information for free. So each
candidate was regressed on a **cheap basis** of 21 descriptors that HUME already computes, whose
marginal cost is therefore zero:

    nHeavyAtom nAromAtom nRing nBondsD MolLogP TPSA NumHDonors NumHAcceptors
    NumRotatableBonds Chi0 Chi1 Chi0v Chi1v Kappa1 Kappa2 Kappa3 HallKierAlpha
    FractionCSP3 nN nO nS

Two scores, both on a **30% held-out split** of an 8,000-molecule subsample of the stratified
20,000-molecule corpus of section 1.2:

  * **linear R2** -- ordinary least squares.
  * **GBM R2** -- `HistGradientBoostingRegressor`, 120 iterations.

The GBM score is the one that decides. Linear R2 alone is not sufficient evidence for a drop,
because the downstream model is gradient-boosted trees: a column that is only 93% linearly
predictable can be 99.9% predictable non-linearly, and `LogEE_A` is exactly that case
(0.929 linear, 0.999 GBM). Judging it on the linear number would have understated the case for
dropping it. Raw values are in `results/dedupe2/nonlinear_reconstruction.json`.

Timings are per molecule over the same corpus, measured with the standalone family drivers in
`build_misc/`, best-of-5 per molecule, and independently reproduced by a second party within 4%.

### 5.2 Dropped

| column(s) | n | us/mol | GBM R2 | reason |
|---|---:|---:|---:|---|
| `MID*`, `AMID*` | 12 | 55.6 | 0.986 | identifier, not a property descriptor |
| `BertzCT` | 1 | 55.5 | 0.994 | complexity index that is size and branching |
| `LogEE_A` | 1 | 26.5 | 0.999 | Estrada index, dominated by lambda_1 |
| `VE1/2/3_A`, `VR1/2/3_A` | 6 | -- | 0.876 | ill-posed: numerically degenerate Perron vector |

**MolecularId (`MID*`, `AMID*`) -- dropped on mechanism first, cost second.** Mordred's
`MolecularId.py` builds a graph weighted by `deg(a)*deg(b)` and enumerates paths to a cutoff
`1/eps^2`. This is Randic's molecular *identification* number: it is designed so that distinct
structures receive distinct values, i.e. to tell isomers apart. That objective is the opposite
of what a property descriptor needs. A feature used for regression should be **smooth** --
similar structures give similar values, so the model can interpolate between them -- whereas an
identifier is engineered to be maximally **discriminative**, separating structures that are
nearly identical. Using one as a feature is closer to handing the model a hash of the molecule
than a physical quantity. The path enumeration is also the source of the group's p99 tail:
16 ms on a single 76-atom bridged cage.

**BertzCT -- the worst value in the package.** An entropy over bond and atom equivalence
classes, presented as "molecular complexity". Complexity so defined is size plus branching in
practice, and the measurement agrees: GBM R2 0.994 from descriptors already computed. At
55.5 us for one column it was the highest per-column cost in HUME -- more than the entire
31-column ring/atom count group and the entire 29-column ETA group combined -- for 0.6% of new
information.

**LogEE_A -- a size statistic in spectral clothing.** Confirmed against mordred's own source as
the "Estrada-like index", the log-sum-exp of the adjacency eigenvalues. The sum is dominated by
the largest eigenvalue, which for a molecular graph is a branching and size measure. GBM R2
0.999 -- effectively free from what we already have -- at 26.5 us for one column.

**The six adjacency eigenvector columns -- dropped on correctness, not on cost.** These derive
from the Perron eigenvector of the adjacency matrix. For molecules built of two near-identical
halves on a linker the leading eigenvalue gap is exponentially small: 4e-15 on corpus molecule
19279 (97 heavy atoms), and roughly 64 of 20,000 corpus molecules have a gap below 1e-9. Mordred
returns an arbitrary member of the numerically degenerate pair, and its value there is
demonstrably not a function of the molecule -- it moves with atom numbering, with
`OMP_NUM_THREADS`, and between runs on the same input (re-running mordred today returns NaN
where the stored reference holds 1.3e12; mordred disagrees with its own earlier run on 31
cells). HUME's inverse-iteration implementation is *better* -- closer to a 60-digit `mpmath`
reference on nine of ten sampled cells -- but a column whose reference definition disagrees with
itself has no place in a package whose central claim is bit-exactness. Their GBM R2 of 0.876
means they were not carrying much unique information either, but that is the secondary reason.

**The ten Barysz eigenvector columns (see 5.3) -- dropped on the same criterion.** Every one
is reconstructible from the cheap basis at GBM R2 >= 0.997, four of them at 0.9998:

    VE1_DzZ 0.9998   VE1_Dzv 0.9998   VR3_Dzv 0.9998   VE1_Dzp 0.9997
    VR1_DzZ 0.9992   VE2_DzZ 0.9990   VE2_Dzv 0.9990   VE2_Dzp 0.9988
    VR2_DzZ 0.9977   VR2_Dzv 0.9973

and the two distance-matrix members, found by applying the same bar to the rest of the
spectral block:

    VE1_D   0.9999   VE2_D   0.9991

They are all `VE*`/`VR*` -- the eigenvector-derived Barysz summaries -- which is the same
failure mode as the six adjacency `VE/VR_A` columns above, and for the same reason: an
eigenvector summary of a molecular graph is a branching statistic, and the heteroatom weighting
that is supposed to distinguish the Barysz matrix from plain topology does not survive it. The
`SpAbs_*`, `SpDiam_*`, `SpMAD_*` and `SM1_*` members are kept: they sit at 0.95-0.99 and one,
`SM1_DzZ` at 0.687, carries substantial independent information.

Dropping `VE1_D`/`VE2_D` has a structural payoff beyond the two columns: they were the last
consumers of the leading **eigenvector**, so the inverse-iteration solve and its `lu_small.h`
`dgetf2`/`dgetrs` factorisation leave the spectral kernel altogether. Every retained spectral
column is an eigen*value* aggregate, which the existing `sytd2`/`sterf` path already produces.

Total: **32 columns**, taking the set from 1,327 survivors to **1,295**. The 20 timed columns
account for ~138 us/molecule; the ten Barysz columns are eigenvector reductions computed inside
the shared spectral kernel, so their saving is the inverse-iteration work they no longer force,
not a separable per-column figure.

Every one of these 30 was dropped **before wiring**, not removed afterwards: the 228 newly
implemented descriptors live in standalone headers, and a dropped column is simply one that is
never registered in `bindings.cpp`. The implementations remain in the tree, verified, so a
reversal is a wiring change rather than a reimplementation.

### 5.3 Kept, and why the expensive ones survive

**BCUT (20 columns, ~137 us) -- kept, and it is the most defensible expensive family.**
Median GBM R2 **0.656, and not one of the 32 BCUT columns exceeds 0.97** -- the only family
tested where nothing is reconstructible. It is also the least size-correlated family measured
(median |r| with heavy-atom count 0.222, against 0.953 for BertzCT). The mechanism explains it:
a BCUT is an extreme eigenvalue of a Burden matrix whose diagonal carries an atomic property
(charge, polarizability, H-bond capacity) and whose off-diagonal carries connectivity, so it
encodes the *joint* distribution of a property with topology -- "how is charge arranged across
this scaffold", not "how much charge is there". Counts, logP and TPSA cannot express that.
This is where the cost buys something the cheap basis provably does not contain.

**ETA (29 columns, 20.5 us) -- kept.** Median GBM R2 0.882 with 12 of 31 above 0.97, so it is
partly reconstructible, but at 0.7 us per column the cost does not justify the analysis needed
to split it.

**Barysz (30 columns) -- kept, but flagged.** Median GBM R2 **0.977, with 19 of 30 above 0.97**,
and ten columns at 0.997 or higher: `VE1_DzZ`, `VE1_Dzv`, `VE1_Dzp` and `VR3_Dzv` all reach
GBM R2 **1.000**. The dedup of section 1 pruned Barysz from 104 columns to 30 on redundancy
grounds, but never asked whether the survivors were cheaply reconstructible, and most of them
are. One column stands out in the other direction: `SM1_DzZ` at GBM R2 **0.687**, the only
Barysz survivor carrying substantial independent information.

The mechanistic reading is consistent: the Barysz matrix is the topological distance matrix
weighted by atomic number, so its spectral summaries are heteroatom-weighted restatements of
topology -- and Chi and Kappa, which are in the cheap basis, are topology. The heteroatom
weighting is what ought to add information, and for the `VE*`/`VR*` eigenvector-derived members
it evidently does not. Acted on: the ten `VE*`/`VR*` members at GBM R2 >= 0.997 are dropped (listed in 5.2). The 20
retained Barysz columns are the `SpAbs`/`SpDiam`/`SpMAD`/`SM1` summaries, which are spectral
*aggregates* rather than eigenvector reductions and are not reconstructible to the same degree.

---

## 6. Cost work after the triage

Section 5 removed columns. This section removes *work*, and it is separated because the two
have different consequences: a dropped column changes what the package answers, an optimisation
should not. Where an optimisation does change an answer, the size of the change is stated.

Baseline and result, both measured at the corpus median stratum (25-35 heavy atoms, 1,200
molecules, min of 5, quiet machine):

| stage | before | after |
|---|---:|---:|
| SMILES parsing (RDKit) | 84.0 | 78.3 |
| boundary extraction (python) | 215.9 | **70.2** |
| C++ descriptor compute | 464.4 | 434.6 |
| ECFP + overhead | 50.1 | 40.2 |
| **end-to-end** | **814.4** | **623.3** |

**-23%, with 1,374 columns still emitted.** None of the 228 newly implemented descriptors is
wired yet, so this is the shipping set getting cheaper, not a smaller set.

### 6.1 The hydrogen-added pickle, deleted

The largest single win, and it was hiding in the boundary rather than in the arithmetic.

`extract_pickles` built a *second* molecule per input: `Chem.AddHs`, a second
`ComputeGasteigerCharges`, a second `ToBinary`. Measured at 16.5 + 15.0 + 22.3 = **53.8 us/mol,
34% of the whole python boundary** -- and it existed for the Autocorrelation family alone.

All of it is derivable from the heavy molecule. Verified BEFORE implementing, over 3,677
molecules including `cpp/hard.smi`:

* `AddHs` appends hydrogens *after* the heavy atoms, in heavy-atom order, nH each, and leaves
  every atom's `GetTotalNumHs()` at 0 -- **0 mismatches** in atomic number, formal charge, nH or
  the bond list.
* a heavy atom's mordred `c` on the AddHs molecule is its own `_GasteigerCharge` (its
  `_GasteigerHCharge` is 0 once hydrogens are explicit), and each hydrogen's is the parent's
  `_GasteigerHCharge` split over its hydrogen count -- worst difference **1.67e-16**.

`molpickle.h` now splits the two halves of mordred's getter apart as `ac_own` / `ac_h`;
`bindings.cpp` builds the graph and feeds both Autocorrelation and constit's RNCG/RPCG.
Removing the second pickle also removes its C++ parse, so the boundary fell 215.9 -> 70.2, more
than the 53.8 the python steps alone accounted for.

**Cost in exactness: 54 of 1,374 columns move by at most 2.665e-15** -- the 52 charge-weighted
autocorrelation columns plus RNCG and RPCG. The NaN pattern is identical and the other 1,320
columns are bit-unchanged. The difference is the division `qh/nH` against RDKit's own
per-hydrogen charge: floating point, not chemistry.

A side effect worth recording: Autocorrelation's measured cost fell from 23.1 to 6.8 us/mol,
because most of what leave-one-out attributed to it was the second pickle's parse. Its true
cost was 76.9 us/mol for 648 columns, not 23.1.

### 6.2 SPS moved to C++, and the stereo boundary with it

`_potential_stereo` -- RDKit's new perception, driven from python -- existed for one column.
`src/hume_core/sps.h` ports the perception itself (`FindStereo.cpp`'s `runCleanup`, `new_canon`'s
`rankFragmentAtoms`, `hanoiSort`, and the legacy CIP ranking) and is bit-identical to
`Chem.SpacialScore.SPS` on 6,600 corpus molecules through the shipping path. `extract_pickles`
now runs with `stereo=False`: 213.7 -> 156.5 us/mol, before 6.1 took it to 70.2.

`constit.h` keeps the column's slot and `Inputs::stereoAtom` / `stereoBond` are always null.
The `stereo_a` / `stereo_b` arguments remain in the C++ signature, passed empty.

One wiring bug is worth recording because of how it presented. The boundary's `bond_s` is
`_extract.py`'s `_EZ` encoding (E/TRANS -> +1, Z/CIS -> -1, none -> 0); `sps.h` wants RDKit's raw
`BondStereo` ordinal (NONE=0, ANY=1, Z=2, E=3). Passing it through unconverted read +1 as
STEREOANY, and showed up as **one molecule in 4,000** with SPS off by 1.256 -- almost always
right, because most bonds carry no stereo.

### 6.3 The E-state accumulation order

`rdkit/Chem/EState/EState.py` accumulates the pair deltas into a **zero** vector and adds the
intrinsic state at the end (`res = accum + Is`); `hume_blocks.h`'s `estate_from` seeded
`S[i] = I[i]` and accumulated into it. Algebraically identical, different rounding, and 40
columns read that vector -- `MaxEStateIndex` and friends plus the 79 `S<t>` typer sums.

With RDKit's order, over 1,800 corpus molecules:

    MaxEStateIndex      252/800 -> 1800/1800 bit-identical
    MinEStateIndex      142/800 -> 1800/1800
    MaxAbsEStateIndex   252/800 -> 1800/1800
    MinAbsEStateIndex    61/800 -> 1800/1800

This is a pure accuracy gain, not a cost change. It is recorded here because it was found,
withdrawn on a bad measurement, and reinstated: the withdrawal was based on a probe run against
a **stale installed binary**. `cmake --build` writes into `build/`; only `uv pip install -e .`
copies the module into the venv. Any timing or exactness claim in this repo must be made after
the install step, not after the build step.

### 6.4 Spectral: one Barysz spectrum, no eigenvector

Covered in 5.2/5.3 as column drops; the work removed is the point here. 264.2 -> 193.5 us/mol
at the median molecule, for 29 of 65 columns. Five Barysz eigensolves collapse to one because
`SpAbs`/`SpDiam`/`SpMAD` are 0.982-0.999 correlated across the six property weightings, and the
inverse-iteration solve leaves entirely because nothing consumes the leading eigenvector.

### 6.5 BCUT: the approximation was authorised, implemented, measured, and rejected

BCUT is the single most expensive item anywhere in the package -- **163.3 us/mol for 20
columns**, 78% of what remains in the spectral family. It is also the one expensive family that
provably earns its place: median GBM R2 0.656 against the cheap basis with **nothing above
0.97**, and pairwise |r| between its weightings of 0.14-0.18 (min 0.003), against 0.98-0.99 for
Barysz. Dropping it was never on the table; making it cheaper was.

The proposal was to exploit the Burden matrix's structure. It is dense only because its
background is a uniform 0.001, and a uniform background is rank one:

    B = 0.001 * J  +  S  +  D        J all-ones, S the bond corrections, D the property diagonal

so a matvec needs no matrix: `(Jx)_i = sum(x)` is one reduction, `S` has `2*nb` non-zeros, `D`
is diagonal. **9.8x fewer operations per product at n=32, measured.** On that basis the earlier
negative Lanczos result recorded in this repository was judged to have used dense matvecs and to
have been measuring the wrong thing, and the trade -- a well-posed definition approximated for
speed, unlike every other divergence in this document -- was put to the project owner and
authorised at the 1e-7 level.

**It was then implemented and it does not work.** Measured over the 20,000-molecule corpus,
median us/mol:

| stratum | dense sytd2/sterf | structured Lanczos | speedup |
|---|---:|---:|---:|
| 0-15 | 20.8 | 29.3 | 0.71x |
| 15-25 | 93.6 | 157.2 | 0.60x |
| 25-35 | 172.0 | 323.5 | 0.53x |
| 35-55 | 287.2 | 628.2 | 0.46x |
| 55+ | 856.8 | 2331.0 | 0.37x |

Slower at every size, and **getting worse with size** -- the opposite of the prediction. The
accuracy does not redeem it either: 99.44% of BCUT cells land within 1e-9 relative, but the
worst is **6.06e-01**, a 60% error on some molecules, far outside the authorised 1e-7.

The reason is that **the matvec was never the cost.** Full reorthogonalisation is, at
`O(n*k^2)`, and it cannot be dropped -- without it the basis loses orthogonality and Lanczos
returns spurious copies of the extremal eigenvalue, which is a wrong BCUT with no symptom. For
extremal eigenvalues `k` must approach `n` at molecular sizes, so the reorthogonalisation costs
as much as the tridiagonalisation it was meant to replace, with worse constants than a tuned
`sytd2`/`sterf`. The 9.8x figure was real and irrelevant.

The implementation is kept behind `-DBCUT_LANCZOS` with this result written above it, so the
next person to have the idea can re-run it in one command rather than re-deriving it. **BCUT
keeps the dense path and its 163 us/mol stands.**

This is the fifth time in this work that a conclusion already recorded in the codebase was
re-derived and found to have been right the first time -- after the `BCUT_SOLVER` switch, the
`-ffp-contract` question, `vsa_bins.h`'s second E-state copy, and `eigen_small.h` itself. The
pattern is worth naming: grep for the decision before measuring it.

### 6.6 What is slowest now

C++ compute is 434.6 us of the 623.3 (70%). Leave-one-out attribution within it:

| family | us/mol | share of C++ |
|---|---:|---:|
| blocks (mandatory) | 115.0 | 26.4% |
| chi | 46.7 | 10.7% |
| infocontent | 45.1 | 10.4% |
| constit | 19.3 | 4.4% |
| pathcount | 8.8 | 2.0% |
| autocorr | 6.8 | 1.6% |

Leave-one-out sums to well under the total because the families share work; the residual is that
shared part, principally the distance matrix and the ring perception.

Two observations for whatever is done next. `blocks` is **O(n^2)-dominated**, not O(n^3):
across the five strata us/n^2 is flat at 143-259 while us/n and us/n^3 each vary about 9x, so
the distance matrix and the per-pair passes over it dominate, and the eigensolves do not -- a
full dense eigensolve at n=29 is ~18 us against blocks' 128. And `chi` and `infocontent` are each
about **half reconstructible** from the eleven cheap families (20 of 40 and 21 of 40 sampled
columns above GBM R2 0.97, against 13 of 40 for `blocks`), so they are the next candidates for
the section 5 treatment -- with the caveat that neither cuts proportionally, because both
enumerate once and read many columns off the result.

---

## 7. Exactness, measured on 42,000 diverse molecules

Section 4 says where the implementations differ and why. This section says how much, on a corpus
large and varied enough for the number to mean something.

### 7.1 The corpus

`data/exactness_corpus.json`: **42,000 molecules**, 41,781 of them the first 6,000 unique SMILES
from each of seven benchmark sets (litpcba, muv, pb_ames, pb_bbb, qm8, qm9, tox21) and 219
adversarial structures from `cpp/hard.smi`. 41,992 parse. It is real screening and quantum
chemistry rather than a drug-like sample: metals, salts, radicals and cages are in it, which is
the point -- the divergences below are concentrated in exactly those.

References were computed live in float64, not read from a stored matrix: RDKit's
`Descriptors._descList` (217 columns) and mordred's full `Calculator` (1,613 columns).

### 7.2 Against RDKit

| | |
|---|---:|
| columns compared | 191 |
| cells compared | 8,018,587 |
| **bit-identical** | **95.83%** |
| columns 100% bit-identical | **163 / 191** |
| columns >=99.9% within 1e-9 | 183 / 191 |

The residue is one failure mode, not many. **114 molecules -- 0.271% of the corpus -- account for
`Kappa2`, `Kappa3`, `Phi` and `HallKierAlpha`**, and they are organometallics and metal salts:
Hg, Na, Zn, Fe, Cu, Cr and Au, 110 of the 114 from tox21. `HallKierAlpha`'s table is solved over
the 31 (element, hybridisation) pairs the training corpora contained (see the note above `ALPHA`
in `hume_blocks.h`); these elements are outside it. On the 99.7% of the corpus that is organic,
those four columns are exact.

### 7.3 Against mordred

| | |
|---|---:|
| columns compared | 1,191 |
| cells compared | 47,997,388 |
| bit-identical | 60.10% |
| **within 1e-9 relative** | **99.32%** |
| columns 100% bit-identical | 520 / 1,191 |
| columns >=99.9% within 1e-9 | **1,104 / 1,191** |
| NaN disagreements | 41 |

Bit-identity is the wrong headline against mordred and the 1e-9 figure is the right one: mordred
sums in numpy, we sum in C++, and a centred autocorrelation subtracts a mean before summing. The
87 columns below the 1e-9 bar are four groups, and only one of them is a divergence we chose:

**65 centred and normalised autocorrelations** (`ATSC*` 21, `AATSC*` 22, `MATS*` 22). Floating
point, not arithmetic: they agree to 1e-9 on 98-99.97% of molecules and the disagreements are
last-bit. The mean subtraction is what costs the bit-identity the uncentred `ATS*` columns keep.

**20 information-content columns, orders 1-5** (`IC1-5`, `BIC1-5`, `MIC1-5`, `ZMIC1-5`), and this
one is deliberate and was declared before it was measured. `infocontent.h:112` states "ORDERS 1-5
DIFFER FROM MORDRED BY DESIGN", because mordred's atom-equivalence refinement is numbering
dependent there; what this package claims instead is DETERMINISM, demonstrated under atom
renumbering, bond-list shuffling and a canonical-SMILES round trip. The same note names order 0
as the control -- "if IC0/TIC0/SIC0/CIC0/MIC0/ZMIC0 do not match mordred, this file has a bug".
Measured here: **IC0, BIC0 and ZMIC0 are 100% bit-identical over all 41,992 molecules and MIC0 is
99.99%**, and every divergent member of the family is order 1 or above. The prediction the file
made is exactly the result. The size of the divergence, for the record: `IC3` differs on 37% of
molecules with a median relative difference of 3.65e-02.

**`BalabanJ`, and it is not our divergence.** HUME agrees with RDKit on **100.00%** of molecules
within 1e-9. RDKit and mordred agree with EACH OTHER on **8.24%**, median relative difference
1.40e-01. The two reference packages compute different quantities under one name; we match
RDKit's. Reporting this as a HUME error would be reporting someone else's disagreement as ours.

**`PEOE_VSA8`**, 119 molecules of 41,992. The differences are tiny in absolute terms against a
near-zero bin total, which is what produces relative figures of 1e14; the bin is empty or nearly
so on those molecules.

### 7.4 What this supports saying

Against RDKit, on a diverse 42,000-molecule corpus, 163 of 191 shared columns are bit-identical
on every molecule and 95.83% of all cells are. Against mordred, 99.32% of 48 million cells agree
to 1e-9 and 1,104 of 1,191 columns agree to 1e-9 on at least 99.9% of molecules. Every column
that falls below that is accounted for above, and one of the four groups is a disagreement
between the two reference packages rather than a divergence of ours.
