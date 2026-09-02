# How HUME computes descriptors

A walkthrough of the descriptor block, in the order a methods section should present it: which
descriptors we decided to carry, how they are computed, what the computation is built out of,
how the families share that work, where the port deliberately does not match RDKit or Mordred,
and what we added that neither library has.

---

## 1. Which descriptors we carry

The candidate set is the **union of RDKit and Mordred**: RDKit's 217 `Descriptors._descList`
entries plus Mordred's 1,613 two-dimensional descriptors, **1,830 columns**. Mordred's 3D block
is excluded throughout — it needs conformers at ~140 ms/molecule, two orders of magnitude
outside any budget contemplated here.

That union is heavily redundant, so it is deduplicated (`dedupe.py` → `data/dedupe.json`).

### The corpus the correlations are estimated on

This matters more than the threshold does, and the first version of it was wrong.

**40,368 molecules, from two sources, seeded at 0:**

* **20,000 from ChEMBL** — `random.sample` over all 145,070 lines. This was previously
  `lines[:20000]`, a prefix of an unshuffled file, which reads 6% light: mean MW 416 against
  439 for a random 20k of the same file, median 387 against 413.
* **20,368 from the benchmark lake** — up to 1,500 each from 16 datasets, capped per dataset so
  no single one dominates. This is where QM9's MW ≈ 120 and CycPeptMPDB's MW ≈ 1,310 enter;
  ChEMBL alone contains neither extreme.

Using benchmark molecules here is legitimate rather than leakage: the selection is
**unsupervised** — it never sees a label — and descriptors should be deduplicated on the
chemistry they will actually be applied to.

### The keep decision

1. Compute all 1,830 columns on the corpus.
2. **Drop the unusable** — constant, or >5% undefined. They cannot carry signal.
3. **Rank-transform** every column, so Pearson on ranks is Spearman. The criterion is monotone
   redundancy, not merely linear: two columns related by a monotone curve are the same
   information to a tree.
4. **Greedy cover in ascending compute cost.** Walk columns cheapest-first; keep one unless it
   correlates |ρ| ≥ 0.99 with something already kept. Cost-ordering is what makes the survivor
   of each redundant group the cheap member — otherwise you pay for an expensive descriptor
   that duplicates a cheap one.

Costs come from `data/budget_profile.json` (RDKit, measured per descriptor) and
`data/mordred_families.json` (Mordred, measured per family and divided by family size).

**Every drop is recorded with its partner.** `data/dedupe.json`'s `drops` list carries
(dropped, kept-partner, |ρ|) for every absorbed column. `|ρ| ≥ 0.99` is a numerical claim;
whether the pair is *mechanistically* the same quantity is a chemical one, and only the second
justifies calling a column redundant. Without the partner written down nobody can check it.

### The numbers, and their current status

The last completed run gives:

```
1,830 union
  ->  1,275 usable   (555 dropped: constant, or >5% undefined)
  ->    865 kept     (410 dropped: |Spearman rho| >= 0.99 with a column already kept)
```

> **These three numbers are from the previous, ChEMBL-prefix corpus.** The corpus change above
> is implemented and the re-run is in flight; `data/dedupe.json` has not been regenerated yet
> (the prefix result is preserved as `data/dedupe.PREFIX20K.json`). **Do not put 1,275 / 865
> into the paper next to the 40,368-molecule corpus description** — one of the two is stale
> until the run lands. Everything downstream that says "865" inherits this.

**A known limitation, quantified.** The threshold is estimated on a corpus, and two columns
that are redundant *there* can separate on a specific endpoint. Measured (`dedupe_cost.py`)
with the filter as the only difference between two arms: the dropped columns are worth
**+0.016 AUROC on `bioavail` (n=640)** and **+0.006 on `hia` (n=578)**, and nothing measurable
on any classification set above 7,000 molecules. On `bioavail` the top dropped columns are only
0.88–0.96 correlated with their nearest surviving partner *on that dataset*, against the ≥0.99
that caused the drop. `dedupe_loo.py` adds one dropped column back at a time to see whether the
deficit is one pair or a diffuse tail.

**A provenance caveat.** `ABCGG` is one of the kept columns, and under the pinned verification
environment (mordred 1.2.0 / numpy 1.26.4) `mordred/ABCIndex.py` raises on every molecule —
it ends `return np.float(...)`, and numpy removed `np.float` in 1.24. A column that raises
cannot have a correlation computed for it, so the previous selection was produced under some
numpy < 1.24 and its provenance is not the environment every exactness claim is pinned to
(`PORT_STATUS.md:499`). The re-run should close this; until it does, cite the kept set as
inherited rather than as reproducible.

---

## 2. How they are computed: co-generation

RDKit and Mordred each compute a descriptor by walking the molecule object from scratch. Ask
for 200 descriptors and you walk the molecule 200 times, in Python, re-deriving the same atom
degrees and the same shortest-path matrix each time. Mordred's own caching helps within a
family and not across them.

HUME's claim is that this is unnecessary, because **almost every 2D descriptor in the two
catalogues is a function of a small fixed set of per-atom and per-bond quantities plus the
graph.** We are already walking the molecule to build the fingerprint; extract those quantities
in the same pass, hand them to C++ once, and every descriptor becomes a cheap reduction over
arrays that are already in cache.

```
RDKit Mol  ──►  15 per-atom values + 8 per-bond values + the bond list + the ring set
                          │
                          ├─►  derived primitives  (distance matrix, Crippen and E-state
                          │     per-atom vectors, subgraph enumeration, …)
                          │
                          └─►  1,269 descriptor columns, 19 families
```

Everything after the boundary is C++. No Python runs in the compute path.

Roughly 900 descriptors are too many to describe individually and there is no need to: the
implementation is organised by *shared input*, not by descriptor, and §4 is that organisation.

### What the 1,269 emitted columns are

The emitted set is not the same object as the kept set, and the difference is worth stating
once:

| | columns | |
|---|---:|---|
| the deduplicated selection | **865** | every one is emitted; 0 missing, verified case-insensitively |
| free riders | **206** | ported columns the filter dropped, which a family produces anyway |
| HUME-original | **193** | no RDKit or Mordred counterpart at all (§6) |
| **total emitted** | **1,269** | `molhume.column_set("full")` |
| opt-in, in no set | **1** | `qed` -- `molhume.OPTIONAL_COLUMNS`; `molhume.ALL_COLUMNS` is 1,270 |

 **THE THREE COUNTS ABOVE ARE PRE-1,269 AND DO NOT ADD UP -- 865 + 206 + 193 is 1,264.** They
were written against an earlier emitted set and have not been re-derived; the total is correct
and the split is not. What IS current, and is read straight off the package: 1,269 emitted, of
which **160** have no RDKit or Mordred counterpart (`molhume.column_set("full_no_new")` is the
other 1,109).

A **free rider** costs a memory write and nothing else. E-state is the clearest case: typing
every atom yields all 79 types, so emitting the full 158-column count-and-sum grid is free
where the filter kept only 50 of them. Autocorrelation contributes 83 and information content
9. Dropping them would save no time, so they are emitted and the filter is applied downstream
by whoever wants it.

### What co-generation buys

| arm | µs/molecule | hours per 10⁹ | USD per 10⁹ |
|---|---:|---:|---:|
| ECFP4 alone | 17 | 5 | $3 |
| **HUME_minimal** (622 descriptors + ECFP6) | **138** | **38** | $27 |
| **HUME_full** (1,269 descriptors + ECFP6) | **155** | **43** | $31 |
| **HUME_no_new** (1,109 descriptors + ECFP6) | **162** | **45** | $32 |
| ECFP4 + RDKit-180 | 444 | 123 | $88 |
| ECFP4 + Mordred-685 | 4,683 | 1,301 | $929 |
| ECFP4 + all descriptors (Python) | 6,200 | 1,722 | $1,230 |

Measured on `c7i.4xlarge` (16 vCPU) at N=10⁶, one run, all three HUME arms together --
`results/scale/`, and the same numbers Figure D draws.

 **This table used to read "HUME (1,266 descriptors) 124 µs".** That measurement was of a
build with a different column set, on a different box, and it is superseded rather than
corrected: the emitted set has since gone 1,266 -> 1,536 -> 1,269, and the run-to-run spread on
this instance type is 2-3% on top of that.

**HUME_no_new is not really dearer than HUME_full.** It cannot be cheaper -- its 1,109 columns
span all nineteen descriptor families, so there is nothing for the compute plan to skip -- and
the 4% is inside the noise floor, which is measured: `hume` itself read 159.0 µs on an earlier
run of the same instance type and 155.4 here. The three arms agree to within 1% at N=10⁴ and are
identical at N=10⁵.

**50× faster than the equivalent Python descriptor block**, for the same information.

---

## 3. What crosses the boundary

This is the whole input. `src/hume_core/bindings.cpp` defines it and asserts the layout against
`src/hume/_extract.py`, so a mismatch is an exception rather than silently transposed columns.
One batch is a set of flat arrays plus offsets, so N molecules cross in one call.

**Per atom — 13 integers and 2 doubles.** `Z`, degree, total hydrogens, formal charge,
hybridisation, aromatic flag, in-ring flag, CIP code, ring-membership *count*, total valence,
`_ChiralityPossible`, chiral tag, isotope; then mass and Gasteiger partial charge.

**Per bond — 6 integers, 1 sign, 1 double.** The two atom indices, conjugated, in-ring, a
SMARTS-order code and RDKit's `BondType` enum; then E/Z as ±1; then the bond order.

**Per molecule — the ring set** as a two-level CSR, and a `chg_ok` flag recording whether
Gasteiger charges were available at all.

**Optional — two stereo arrays** from RDKit's *new* perception (`FindPotentialStereo`), when a
caller asks for the columns that need it.

Six of these are carried rather than re-derived, each for a measured reason:

* **`nring` alongside the in-ring boolean.** SMARTS asks both questions independently: `[R]` is
  membership, `[R1]`/`[R2]` are counts, and the count cannot be recovered from the boolean.
* **Total valence.** Recomputing it as `round(Σ bond orders) + nH` disagrees with RDKit on
**11,238 of 575,571** corpus atoms, because aromatic bonds and hydrogens go through RDKit's
  own rounding rule — pyrrole's `[nH]` has two aromatic bonds summing to 3.0 and one hydrogen,
  and RDKit's total valence is 3, not 4. Three fragment patterns ask SMARTS `v`.
* **`BondType` alongside the SMARTS bond code.** They answer different questions. The SMARTS
  code collapses everything but "is the order nameable" and "is the aromatic flag set", which
  is exact for every query in `cpp/frag_program.h`; Morgan hashes the raw enum, and
  `cpp/hard.smi` contains 114 DATIVE bonds that the collapse cannot distinguish.
* **Isotope**, carried and not derived from mass: `getMass()` says an atom *is* labelled, not
  which isotope it carries.
* **The ring set**, because it is not recoverable from per-atom ring counts — benzene and
  cyclohexane have identical `nring` vectors and differ on 6 of the 49 RingCount columns, and
  the 28 fused columns need |Rᵢ ∩ Rⱼ| for every ring pair. It is **not a second perception**:
  it comes from the same single RDKit perception the per-atom counts do, in the repaired
  canonical order of §5(ii).
* **Both stereo perceptions.** The two atom `_ChiralityPossible` columns are the *legacy*
  perception; the optional arrays are the *new* one. They differ on 262 of 4,000 corpus
  molecules (6.6%), and RDKit's own descriptors read them in exactly that split —
  `NumAtomStereoCenters` counts `_ChiralityPossible`, while `SPS` asks `FindPotentialStereo`.

Three of these six cost nothing on the pickle path: `molpickle.h` was already decoding total
valence, the bond-type byte and the isotope out of the blob and throwing them away.

Crippen contributions used to cross the boundary too and no longer do. They were two more
per-atom doubles filled by `_CalcCrippenContribs` at 78 µs/mol — 42% of the extraction module —
to run 110 SMARTS over the whole molecule. `crippen_typer.h` answers the same question from
integers already in hand for 1.5 µs/mol, bit-identically to RDKit on 2,869,048 atoms.

---

## 4. The families

Nineteen families, 1,269 columns. Grouped first by what they *mean*, then — which is the axis
the implementation is actually built on — by what they *need*.

### By meaning

| | families | cols |
|---|---|---:|
| **composition and size** | `constit`, `rdkcore`, `alias` | 65 |
| **graph shape indices** | `topomisc`, and `blocks`' `Kappa`/`HallKierAlpha`/`BCUT2D`/`BalabanJ`/E-state extremes | 33 |
| **surface, lipophilicity, polarity** | `vsa` | 66 |
| **electronic state** | `estate`, `topocharge` | 179 |
| **connectivity and paths** | `chi`, `pathcount`, and `blocks`' RDKit χ and path counts | 77 |
| **rings** | `ringcount` | 49 |
| **spatial distribution of atomic properties** | `autocorr` | 540 |
| **graph information content** | `infocontent` | 43 |
| **substructure** | `frag` | 76 |
| **beyond the union** (§6) | the four new blocks inside `blocks` | 138 |

The point of the second grouping is that this one is not how the work divides. `estate` (an
electronic property) and `vsa` (a surface property) mean entirely different things and fall out
of the same four per-atom vectors; `autocorr` (540 columns) and `infocontent` (43) mean
different things and both wait on the same distance matrix.

### By shared input

**(a) Atom and bond arrays only — no graph traversal.**

| family | cols | what it measures |
|---|---:|---|
| `constit` | 43 | Constitutional counts: the `C1SP1`…`C4SP3` carbon-hybridisation census, atom and bond counts, Kier shape indices, `MDEC-*` distance edges, Lipinski/Ghose filters, `qed`, `SPS`, `Vabc`, `fMF` |
| `rdkcore` | 21 | RDKit's ring-count family, `HeavyAtomMolWt`, `FractionCSP3`, `Phi`, `FpDensityMorgan1-3`, the two stereocentre counts |
| `alias` | 1 | `SLogP` — Mordred's name for RDKit's `MolLogP`, emitted separately because both names are in the union |

A single pass over the atom table. Cheapest block in the set.

**(b) Per-atom contribution vectors — Crippen, Gasteiger, Labute, E-state.**

| family | cols | what it measures |
|---|---:|---|
| `vsa` | 66 | `SlogP_VSA1-12`, `SMR_VSA1-10`, `PEOE_VSA1-14`, `EState_VSA1-11`, `VSA_EState1-10`, plus `MolLogP`, `MolMR`, `TPSA`, `TopoPSA`, `LabuteASA` and the four E-state extremes |
| `estate` | 158 | Kier–Hall electrotopological state: per atom-type **count** (`N…`) and **sum** (`S…`) over 79 atom types |

A *VSA descriptor* bins a per-atom property against per-atom van der Waals surface area —
`SlogP_VSA3` is "how much surface area sits on atoms whose Crippen logP contribution falls in
bin 3". The bin edges are module-level constants in RDKit, not per-molecule work.

This is the sharpest illustration of the thesis. **Verified on 6,000 adversarial molecules
(`cpp/hard.smi`): 62 columns — the 57 RDKit `*_VSA` columns, `LabuteASA` and the four E-state
extremes — reconstruct bit-exactly (rtol 1e-9) from four per-atom vectors**: Crippen (logP, MR)
contributions, Gasteiger charges, E-state indices and Labute ASA contributions. Once the vector
is paid for, each column is one `bisect_right` pass. Sixty-two descriptors for the price of
four vectors.

Three details silently break the reconstruction, all found by measurement:

* `LabuteASA` is `sum(contribs) + hContrib`, and RDKit returns the hydrogen term *separately*
  from the heavy-atom vector. Dropping it is a small uniform shortfall.
* `PEOE_VSA` does **not** clamp NaN charges (elements with no Gasteiger parameters, e.g. Sn).
  NaN falls through `bisect_right` into the final bin; clamping to 0.0 misroutes it.
* `MolLogP` and `MolMR` — unlike every `*_VSA` column — sum over the **H-added** molecule, so
  they need the Crippen hydrogen rows.

**(c) The distance matrix (all-pairs shortest paths).**

| family | cols | what it measures |
|---|---:|---|
| `autocorr` | 540 | Moreau–Broto / Moran / Geary autocorrelations: 6 variants × 9 lags × 10 atomic weightings |
| `topocharge` | 21 | Galvez topological charge indices `GGI1-10`, `JGI1-10`, `JGT10` |
| `topomisc` | 15 | Walk counts (`MWC`, `SRW`, `TSRW10`), Wiener index, `Diameter`, `TopoShapeIndex`, `ABCGG` |
| `infocontent` | 43 | Shannon information content of atom-equivalence classes at orders 0–5 (`IC`, `TIC`, `SIC`, `BIC`, `CIC`, `MIC`, `ZMIC`), plus `AvgIpc` |

Autocorrelation is the largest family and the best illustration of sharing. An autocorrelation
at lag *k* sums a product of an atomic property over every pair of atoms exactly *k* bonds
apart: ATS(k,w) = Σ wᵢwⱼ over pairs at distance k. The *weighting* w is the atomic property —
Gasteiger charge (`c`), sigma electrons (`d`), valence electrons (`dv`), ionisation potential
(`i`), polarisability (`p`), van der Waals volume (`v`), Sanderson electronegativity (`se`),
Pauling electronegativity (`pe`), Allred–Rochow (`are`), atomic number (`Z`). The variants
differ in normalisation: `ATSC`/`AATSC` are centred, `MATS`/`GATS` are centred and scaled,
`AATS` is averaged over the pair count. One distance matrix plus ten per-atom property vectors
yields all 540 columns.

**(d) Subgraph enumeration.**

| family | cols | what it measures |
|---|---:|---|
| `chi` | 40 | Kier–Hall connectivity indices over *subgraphs*: `Xp-*` paths, `Xch-*` chains, `Xc-*` clusters, `Xpc-*` path-clusters, `AXp-*` averaged |
| `pathcount` | 11 | Simple path counts `MPC*`, `piPC1`–`piPC10`, `TpiPC10` |
| `ringcount` | 49 | Mordred's ring census by size, aromaticity, heteroatom content, fusion |

A **chi index** of order *n* sums, over every subgraph of that order, the product of 1/√δ
across its atoms, where δ is the atom's degree (`d`) or valence-corrected degree (`dv`). The
four *shapes* — path, chain, cluster, path-cluster — are why this needs true subgraph
enumeration rather than path walking, and why it is the most expensive family here.

Note the naming collision, which has caused confusion twice: **`chi.h` implements Mordred's
subgraph chi (`Xp-*`) while `hume_blocks.h` implements RDKit's Kier–Hall chi (`chi0n`…`chi7v`)**.
They share no code, and RDKit's are registered in *lowercase*.

**(e) SMARTS matching.**

| family | cols | what it measures |
|---|---:|---|
| `frag` | 76 | 68 RDKit `fr_*` functional-group counts, plus `NumHDonors`, `NumHAcceptors`, `NOCount`, `NHOHCount`, `NumHeteroatoms`, `NumAmideBonds`, `NumRotatableBonds`, `HeavyAtomCount` |

Substructure counting against a compiled SMARTS program (`cpp/frag_program.h`), with a second
program for QED's 116 structural alerts. Both matchers share one molecule representation and
one subgraph-isomorphism implementation, and keep their recursive-query caches warm across a
batch.

**(f) Mixed.**

| family | cols | what it measures |
|---|---:|---|
| `blocks` | 182 | RDKit's Kier–Hall `chi0n`…`chi7v`, `Kappa1-3`, `HallKierAlpha`, `BalabanJ` (both variants), the four `BCUT2D_*` eigenvalue pairs — and the four new blocks of §6 |

---

## 5. Fidelity: what we match, what we fix, and what we deliberately keep broken

**For the overwhelming majority of columns HUME does exactly what RDKit and Mordred do, in C++
rather than Python, and is verified bit-exact against them** — per decision rather than per
column, since a count such as `NsCH3` lets two opposite mistypings cancel: 2,868,290 atoms for
E-state type tuples, 2,869,048 for Crippen row indices, 98,905 molecules for the block columns.
Those need no justification and are not discussed here. What follows is the short list of
places where the number differs from what you would get by calling the library.

The house rule (`PORT_STATUS.md` §1):

> Reproduce a **quirk**; diverge from an **ill-posed definition**. The test is a single
> question: *is the upstream descriptor a function of the molecule?*

A quirk gives the same wrong answer every time — so it is reproduced bit-for-bit and commented.
An **ill-posed** descriptor gives *different answers for the same molecule* depending on atom
numbering or on which Kekulé structure the perceiver happened to pick. There is nothing there
to be exact against, and "we reproduce Mordred" would be a claim about a coin flip.

**How ill-posedness is detected, mechanically:** permute the input ordering and recompute; any
column that moves is ill-posed. Critically, **the screen must shuffle bonds, not only atoms** —
`Chem.RenumberAtoms` permutes atoms and leaves the bond list alone, while RDKit's ring
perception reads the bond list. This repo shipped the weaker atom-only screen for a while. The
molecule `O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2` is stable across **201** atom
renumberings and yields two different ring sets the moment bond order is shuffled too. The
canonical-SMILES round trip is a *control that should show zero*, not a second probe.

### 5a. Bugs we reproduce on purpose

Each is deterministic — the same molecule always gets the same wrong answer — so matching it is
matching the library everyone else uses. Each is commented at the site.

| # | the defect | where | columns |
|---|---|---|---:|
| 1 | Five E-state SMARTS rows are written `[N,nD2H0]` with no semicolon, so the aliphatic alternative carries no degree or hydrogen constraint. The neighbouring carbon rows use `[C,c;D2H0]` and do not. | `estate_typer.h:28` | E-state + `EState_VSA*` |
| 2 | `[SeD2H0]` and its Li/Be/Si/Ge/As/Sn/Pb equivalents parse to element-*number* queries with no aromaticity constraint, so `aaSe` fires on selenophene's aromatic selenium. The same quirk recurs in the SMARTS compiler, where only the organic subset gets `AtomType`. | `estate_typer.h:24`, `PORT_STATUS.md:785` | `ssSe`, `aaSe`, QED alert 26 |
| 3 | RDKit writes **1e8** for unreachable atom pairs and Mordred feeds it into arithmetic unguarded. `Diameter` is literally 100000000 for a salt. 17,050 of `cpp/hard.smi`'s 100,000 molecules are disconnected. | `topomisc.h:58`, `topocharge.h:18`, `hume_blocks.h:451`, `constit.h:230` | `Diameter`, `WPath`, `BalabanJ`, the 21 `GGI`/`JGI`, `MDEC-*` |
| 4 | Labute ASA subtracts **one** hydrogen-sized cap per heavy atom into a single scalar — a methyl and a quaternary carbon each remove exactly one. And the `fabs(hContrib) > 1e-4` guard is not an else-zero. | `vsa_bins.h:280` | `LabuteASA` |
| 5 | RDKit's `Kappa` counts heavy atoms for A and *all* bonds for P1. Verified on a tritiated molecule, where only `GetNumBonds()` reproduces its value. | `hume_blocks.h:1414` | `Kappa1-3` |
| 6 | RDKit's χ is internally inconsistent about explicit hydrogen: `Chi0n`/`Chi1n` use all atoms and bonds, `Chi0v`/`Chi1v` are heavy-only, and `Chi2`+ are heavy in both. | `hume_blocks.h:259` | the 9 RDKit `chi*` |
| 7 | Mordred's `AXp-0dv` is NaN for every molecule carrying an explicit hydrogen — order 0 skips the subgraph enumeration and a `[2H]` has `dv == 0`. | `chi.h:38` | `AXp-0dv` |
| 8 | `SIC`/`BIC` are *missing*, not infinite: Mordred wraps the divisions in `np.errstate(divide="raise")`, so IEEE infinity becomes a raise and a missing value. `Cl[Se]` is the one molecule in `cpp/hard.smi` that discriminates them. | `infocontent.h:1269` | `SIC*`, `BIC*` |
| 9 | `ABCGG` accumulates via the builtin `sum` in bond-index order with no pairwise reassociation — the opposite of what the surrounding numpy code suggests. | `topomisc.h:44` | `ABCGG` |
| 10 | The K4 cycle enumerator yields exactly twice the true count; verified by brute force on cubane, prismane, adamantane and naphthalene. Reproduced rather than "fixed". | `hume_blocks.h:478` | cycle counts |
| 11 | `AATS<k>` is 0/0 → NaN at a lag longer than the molecule's diameter. Confirmed against Mordred, which returns an error object for exactly those. | `cpp/ac_weights.h` | `AATS*` |
| 12 | `MolecularDistanceEdge`'s product overflows float64 to +inf, after which `n / dx**2` is 0.0. Computing it in log space would give a better answer and a *different column*. | `constit.h:230` | `MDEC-*` |

> **`PORT_STATUS.md:133` says "two have already been found and kept". That sentence is a summary
> of the E-state work only and undercounts the inventory by an order of magnitude.** The methods
> section should say "a dozen", not "three".

Two further categories must not be confused with this list. **Floating-point tolerance is not
divergence:** autocorrelation is checked at rel 1e-8 because `ATSC`/`AATSC` centre by
subtracting a mean; `TopologicalCharge` is bit-exact on 12 of 21 columns and within 6.66e-16 on
the rest, where **Mordred disagrees with itself** on 21–70% of the corpus purely by summation
order; `qed` is bitwise on 79,645 of 100,000 with max relative deviation 1.9e-15 because libm's
`exp`/`log` are not correctly rounded. And **FP contraction is deliberately suppressed**
(`-ffp-contract=off`): fusing a multiply and an adjacent add is *more* accurate and therefore a
different number from the one Python computed. The rule is to match the reference, not to
maximise accuracy.

### 5b. Where we deliberately differ

**(i) `InformationContent` — Mordred's is not a function of the molecule.** Two independent
defects, both established by reading the source and confirmed by measurement:

1. `InformationContentBase` sets `kekulize = True`, so the atom-equivalence codes are built on
   a Kekulé structure. An aromatic bond enters the code as SINGLE or DOUBLE depending on which
   structure the perceiver picked.
2. `BFSTree._expand` mutates a visited set *while iterating over it*. Two adjacent siblings at
   the same depth are both in the tree and neither is visited when the loop starts; whichever
   the dict yields first claims the other as its child. Dict order is insertion order, which is
   `GetNeighbors()` order, which is atom numbering.

**Measured: on the first 2,000 molecules of `cpp/hard.smi`, 32.3% change at least one
InformationContent column under a single perturbation of input order** — 15.6% under atom
renumbering alone, 28.0% once bonds are shuffled too, with 16.8% visible *only* to the
bond-order screen. The smallest molecule that flips is `ON=Cc1ccccn1`: Mordred gives
`IC1 = 2.682589` as parsed and `2.815922` after one swapped label, 5% apart.

*Resolution:* an aromatic bond keeps its own bond-type symbol rather than being kekulized away,
and the tree is layered by graph distance. **Orders 1–5 therefore differ from Mordred by
design; order 0 is unchanged.** The `rethrow_zerodiv` quirk (5a #8) is reproduced *inside* this
family — the divergence is scoped to the equivalence codes, not to the arithmetic.

**(ii) Ring perception — one repaired ring set, used everywhere.** `Chem.GetSymmSSSR` is not a
function of the graph. The SSSR *basis* is stable; what flips is whether `symmetrizeSSSR` finds
a symmetry-equivalent extra ring of a size already present — RDKit's own source admits it "may
miss extra rings that would need to swap two (or three...) rings to be included".
`C1=CC2C3C(C=C1)C23` gives ring sizes (3,3,7) on 33 of 60 numberings and (3,3,7,7) on the other
27. Brute force over every simple cycle confirms the larger answer is the **relevant-cycle
set** — the object `symmetrizeSSSR` is reaching for and reaches only sometimes.

*Resolution:* perceive rings on a **skeleton rebuilt from scratch** — *n* carbons in
canonical-rank order, bonds added in sorted `(rank_u, rank_v)` order. Ring perception reads only
the graph, so the skeleton asks exactly the right question and puts bond order under canonical
control too. The repair is to the *selection*, not the quantity: on every molecule where RDKit
is already stable it returns RDKit's own answer.

**100,000 molecules × 49 columns × 5 numberings: 22 molecules move before, 0 after.** It changes
RDKit's answer on 32 molecules, all 32 independently confirmed unstable. Five columns are
affected upstream: `nARing` (25), `nG12Ring` (11), `n6Ring` (6), `n7Ring` (6), `n6ARing` (6).

An earlier prescribed repair — canonical atom ranks, rings compared by (size, sorted rank
vector) — was **wrong** and is recorded as such: it left 3 of 100,000 still moving and made
those 3 worse, because canonical ranks fix atom numbering and not bond order, which is the axis
that decides. `Chem.CanonicalRankAtoms(breakTies=True)` is also not a graph invariant on
symmetric molecules.

The consequence is deliberate and propagates. RDKit's own 13 ring columns move on those 32
molecules too; taking RDKit's raw rings for those would put **two different ring sets inside one
feature vector**. `Vabc` inherits it and is the one column of `constit`'s 43 that is not
bit-exact against Mordred: 99,970 of 100,000 exact, 30 differ, and on every one of those 30
Mordred itself gives two or three different answers under perturbation. The claim for that
column is "exact on 99,970, and deterministic where Mordred is not", which is strictly stronger
than matching a coin flip.

**(iii) Aromaticity, two repairs**, relevant only where HUME does its own ring reasoning rather
than inheriting the boundary's aromatic flag:

* A ring sulfur carrying an exocyclic double bond is a sulfoxide — pyramidal, therefore not
  aromatic.
* "A bond in an all-aromatic ring is aromatic" must run **after** perception, not during it: a
  fused system can contain ring bonds belonging to no tested subset.

**(iv) `ExtendedTopochemicalAtom`, recorded for the principle only.** Mordred gates the π
contribution on the *kekulized* bond order, so a pyrrole nitrogen scores 0 while a pyridine
nitrogen scores 2 — and which you get can flip with atom numbering. Mordred's own source has an
`if bond.GetIsAromatic(): y = 2.0` branch that its `kekulize = True` setting largely defeats;
the intent was there, the wiring was not. **This family contributes no surviving columns under
the dedupe**, so it is recorded for the principle and changes no emitted value.

### 5c. One place where upstream is simply wrong

**`Ipc` / `AvgIpc` / `Log2Ipc`.** RDKit runs Le Verrier–Faddeev–Frame in floating point, and the
characteristic-polynomial coefficients come out of a trace after catastrophic cancellation.
HUME uses exact integer arithmetic with an overflow-checked width-doubling fast path. Of the
96,244 molecules whose largest coefficient needs ≤40 bits, 96,221 match RDKit to the last bit;
in the 41–53 bit band only 638 of 1,271 match, and **there RDKit is the inaccurate one** — ours
within ~1e-15 of exact, RDKit out by up to 1e-2. Above ~70 heavy atoms RDKit's version is not a
function of the molecule at all: six renumberings of one 199-atom molecule give `AvgIpc` from
0.6905 to 1.5129, a factor of 2.2, on 2.9% of `cpp/hard.smi`. `Ipc` saturates at `DBL_MAX` with
an overflow flag rather than returning a silent infinity.

> Cite `src/hume_core/infocontent.h:222` for the determinism evidence, **not**
> `PORT_STATUS.md:379` — that block is stale and still says the columns are unwired.

### How to state all of this in the paper

> HUME reproduces RDKit and Mordred bit-exactly for all well-posed columns. Two families are
> ill-posed upstream — their value depends on atom numbering or Kekulé choice rather than on
> the molecule — and for these HUME implements the well-posed definition the upstream code was
> evidently reaching for, and is deterministic under atom and bond permutation.

That is a **stronger** claim than bit-exactness, not a weaker one, and it should be presented
that way rather than buried.

---

## 6. What we added

193 emitted columns have no RDKit or Mordred name. They are not all equally novel, and the
distinction should be made before any of it reaches a reviewer:

| | cols | what it is |
|---|---:|---|
| **four new blocks** | **138** | genuinely new descriptors — the claim |
| autocorrelation grid completion | 38 | standard Moreau-Broto/Moran/Geary formulas at cells Mordred leaves empty (`MATS0*`, `GATS0*`, charge-weighted `ATS*c`/`AATS*c`) |
| χ / path extension | 16 | `chi5n`–`chi7v` past RDKit's order-4 stop, `path1`–`path7`, and two ratios |
| `BalabanJ_mordred` | 1 | a port; it appears here only because the name collides with RDKit's |

**The defensible number is 138.** Presenting all 193 as "new descriptors" would not survive
review: 38 are Mordred's own formulas at unregistered lags — and lag-0 Moran/Geary are
near-degenerate by construction — while `path1`–`path7` are the same quantity as Mordred's
`MPC1`–`MPC10`, of which the dedupe happened to keep only three. Genuinely original in that
group are the two ratios, `path_ratio` and `chi_nv_ratio`.

### Why add anything at all

Not "what information is ECFP missing?" — by the data-processing inequality, and because ECFP
is near-injective on real molecules, the answer is essentially nothing. The generating question
is instead:

> **What is present in the fingerprint but not in a form a gradient-boosted tree can
> construct?**

That follows from the algorithm. ECFP is a bag of *unary* local features, as *hashed presence
counts*, built by WL colour refinement. Each property blocks a class of quantity:

* **Pairwise sums** Σᵢⱼ f(i,j) need quadratic interaction across 2,048 hashed bits — which is
  why the 419 surviving autocorrelations pay, and the shape every new block should take.
* **Ratios** — trees cannot divide.
* **Cyclic patterns** — WL-1 distinguishes exactly the graphs that *tree* homomorphism counts
  distinguish (Dell/Grohe/Rattan), so any pattern containing a cycle is outside colour
  refinement.

Only the third is an expressivity limit in the theorem sense; the other two are conditioning,
and the blocks are shaped accordingly. The illustration: ECFP separates anthracene from
phenanthrene, but `Kf` = 163.85 against 160.78 is one *ordered scalar on a physical axis*,
while `NumAromaticRings`, `RingCount`, `NumBridgeheadAtoms`, `Kappa2` and `LabuteASA` are all
bit-identical between them.

**Orthogonality by construction, not by hope.** Each block is identically zero when its axis is
absent — an acyclic molecule gets 33 zeros from cycles, a molecule with no conjugated bond gets
24, a molecule with no stereocentre and no E/Z bond gets 23. So a block cannot help or hurt a
molecule it does not describe, and a block that helps *uniformly* is proxying size and should
be cut rather than celebrated.

### Resistance — 60 columns

Treat every bond as a unit resistor; Ωᵢⱼ is the effective resistance between atoms *i* and *j*,
from the pseudo-inverse of the graph Laplacian, per connected component. The block is built
around

    Δ_ij = d_ij − Ω_ij        d = shortest path, Ω = resistance distance

On a tree there is exactly one route between any two atoms, so Ω = d and **Δ ≡ 0**. The quantity
measures path multiplicity — ring fusion — and cannot proxy for size, weight or atom count.
Emitted: centred autocorrelations of four atomic properties binned by Δ (`RATSC*`, `RPAIR*`),
the Kirchhoff index and its normalisations (`Kf`, `Kf_n`, `Kf_norm`, `Cyclicity`), and pooled
random-walk return probabilities diag(Sᵏ) for k ∈ {2,3,4,6,8,12,16} with S = D^−½ A D^−½.

Why not the obvious thing: **all 110 of Mordred's spectral scalars** — `SpMax`/`SpDiam`/`SpAD`/
`SpMAD`/`VE1-3`/`VR1-3` over the adjacency, Barysz, distance and detour matrices — **were killed
by the dedupe, zero survivors.** The cover runs in ascending cost order, so every one had a
*cheaper* non-spectral correlate. Collapsing n×n to one number discards whatever was new about
the matrix.

*Precision about the orthogonality claim:* only the 30 Δ-derived columns are identically zero on
acyclic molecules. `Kf` and the random-walk columns are not — on a tree `Kf` equals the Wiener
index.

*The bin-edge problem, and why it is the most fragile thing in the block.* Ω is a rational
function of the graph, so Δ lands **exactly** on bin edges (0.1, 0.5, 1.0, 2.0) for symmetric
molecules — on tetra-*tert*-butyl tetrahedrane every atom pair has Ω = 0.5. The binned columns
were then decided by floating-point noise: **Accelerate's own two LAPACKs disagreed with each
other on 9.00%** of a 98,905-molecule corpus, more than reference LU disagreed with either.
Values within 1e-9 of an edge are now snapped before binning, which changed 13.17% of molecules
and makes the columns implementation-independent. **Any `RATSC*`/`RPAIR*` number produced before
that snap is superseded.**

### Cycles — 31 columns emitted (33 in the reference module)

Exact counts of simple cycles of length 3–8, per-atom participation, and hetero/aromatic
typing. This is the block that targets the WL-1 gap directly.

It is **not `RingCount`**: the SSSR is a *basis* for the cycle space, not a list of cycles, and
the two diverge sharply — cubane's cycle space has dimension 5, yet the molecule contains 16
distinct six-membered cycles and 28 cycles of length ≤ 8. Naphthalene has two 6-rings in its
SSSR and three cycles in total, the third being the 10-membered perimeter.

Lengths 3–5 in closed form from traces of A³, A⁴, A⁵ (already built for `WalkCount`, so
essentially free); 6–8 by bounded DFS from each cycle's lowest-indexed vertex, each cycle
enumerated in two directions and halved. Cut at 8 on chemical grounds: rings of 3–8 cover
essentially all drug-like chemistry, longer cycles in a fused system are perimeters rather than
chemically ring-like, and macrocycles are better served by `RingCount`.

*The implementation detail that changed values:* **a cycle is not its vertex set.** In K₄ three
distinct 4-cycles share a single vertex set, so de-duplicating enumerated cycles on their sorted
vertex tuple silently merges two of the three — on a tetrahedrane derivative that removed
exactly 0.125 from both χ₄n and χ₄v. Cycles are canonicalised on `path[1] < path[-1]` instead.
`C_sssr` and `C_redundancy` are computed by the reference module but not emitted.

### Conjugation — 24 columns

Union-find over RDKit's conjugated-bond flags recovers the π-system connected components; then
their count, sizes, six-bucket size histogram, topological diameter (from the distance matrix
the core already builds, so free), branch points (atoms with ≥3 conjugated bonds —
cross-conjugation), heteroatom content and non-aromatic extent.

Coverage in both libraries is **zero**: no descriptor name in either matches `/conjug/`. The
nearest objects are `NumAromaticRings` and friends, which count *aromatic rings* — a strictly
narrower thing. Three cases a ring count cannot express:

1. **Non-aromatic conjugation** — enones, dienes, acrylamides, vinyl sulfones. An
   α,β-unsaturated ketone is the reactive warhead of a large fraction of covalent inhibitors
   and contributes nothing to any aromatic ring count.
2. **Merging across boundaries** — a conjugated system runs through biphenyl's ring–ring bond
   and out of a ring into a pendant carbonyl. Ring counts see two rings and a substituent; the
   π system is one object of 12–14 atoms.
3. **Shape** — anthracene and a C₁₄ linear polyene are both one conjugated system of 14 atoms,
   separated only by linearity (diameter over size): 0.54 against 1.00.

Why ECFP cannot assemble it: connected-component extent is a global question and ECFP is a bag
of radius-2 environments. Every interior atom of a decapentaene and of a hexatriene has the
identical radius-2 environment; only the two ends differ. The bag can tell you a polyene is
present, not how long it is.

### Stereo — 23 columns

Signed CIP parities sᵢ ∈ {−1, 0, +1} and E/Z bond parities t_b ∈ {−1, 0, +1}, combined at two
orders, which behave differently under reflection — mirroring flips every sᵢ and leaves every
t_b alone:

* **odd in s** — Σ sᵢ flips sign, so it separates **enantiomers** (absolute configuration)
* **even in s** — Σ_{d(i,j)=k} sᵢsⱼ is mirror-invariant, so it separates **diastereomers**
* **any in t** — achiral, safe to mix with either

The motivation is a measurement: the descriptor union is *completely* stereo-blind. Across four
enantiomer pairs (butane-2,3-diol, alanine, ibuprofen, thalidomide), **zero** of RDKit's 217 and
**zero** of Mordred's 1,613 descriptors change. ECFP with `includeChirality=True` does see it —
4 to 10 bits move — so this block is **conditioning rather than new information**, and should be
reported as such.

*Stated weakness.* CIP R/S is a priority convention, not a physical quantity: L-cysteine is R
and L-serine is S despite identical spatial arrangement, because sulfur outranks oxygen. So
Σ sᵢ is a categorical separator with a convention-dependent sign, not a smooth chemical axis.
The even-order terms are on firmer ground. Coverage in the benchmark: 30.9% of 56,197 molecules
carry ≥1 specified stereocentre (0.0% to 91.8% by dataset); defined E/Z is 4.5% overall and
never above 11.6%, so the t-features are along for the ride.

### The evidence for these blocks is currently weak, and the doc should say so

`block_run.py` ran the intended test — each block added to an `ecfp+core` baseline over 34
scaffold-split regression datasets, aggregated by mean rank and win rate rather than by
averaging RMSE across incommensurable scales. The result (`data/surrogate/block_report.json`):

| arm | mean rank | beats baseline | median ΔRMSE |
|---|---:|---:|---:|
| ecfp+core+conjugation | 3.50 | 19/34 | −0.25% |
| ecfp+core+cycles | 3.76 | 18/34 | −0.03% |
| **ecfp+core** (baseline) | **3.82** | — | — |
| ecfp+core+stereo | 3.91 | 15/34 | +0.18% |
| ecfp+core+all | 3.94 | 18/34 | −0.12% |
| ecfp+core+chi | 4.26 | 15/34 | +0.08% |
| ecfp+core+resistance | 4.79 | 11/34 | +0.59% |

Every arm sits within ±0.6% of the baseline and the ranks straddle it. **No block has a
demonstrated downstream effect on this suite**, and resistance — the largest and most expensive
of them — ranks last.

Worse for the argument, **three of the four negative controls could not run.** The suite has no
acyclic datasets (resistance's control is n=0), no non-conjugated datasets (conjugation's
control is n=0), and exactly one dataset above 50% fused rings. Only stereo's control was
defined, and it came out backwards: the block helps *more* where stereo is absent (−1.04% on 6
low-stereo datasets against +0.06% on 7 high-stereo ones), which is the direction that would
condemn it if the sample were bigger.

Two caveats before treating this as a verdict:

* The run predates the resistance bin-edge snap by two days, so the resistance arm was measured
  on columns that were partly decided by BLAS rounding on 13.17% of molecules — the block's own
  documentation says those values are superseded.
* All 34 datasets are regression; the block was never tested on the classification suite, and
  it was never tested inside the current HUME pipeline, only against a Python `ecfp+core`.

**The honest claim today is a design claim, not an empirical one:** the blocks target axes the
union provably does not carry, they are orthogonal by construction, and they cost little. Any
sentence asserting that they *help* needs a re-run against the shipped C++ columns with the
negative controls actually populated.

---

## Known documentation drift

* `FINDINGS.md:189` lists cyclic homomorphism counts as "held in reserve — deliberately not
  built". They were built; `cycles.py` and `hume_blocks.h:471` are shipping.
* `METHODS.tex:4` still describes selection as "a run of 20,000 ChEMBL molecules" and the
  reproduced defects as "three".
* `PORT_STATUS.md:379` still says `Ipc`/`AvgIpc`/`Log2Ipc` "stay unwired"; they ship.
