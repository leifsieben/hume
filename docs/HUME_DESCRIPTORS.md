# How HUME computes descriptors

A walkthrough of the descriptor block: what it contains, how it is organised, and the small
number of places where it deliberately does not do what RDKit or Mordred does.

---

## 1. The thesis

RDKit and Mordred each compute a descriptor by walking the molecule object from scratch. Ask for
200 descriptors and you walk the molecule 200 times, in Python, re-deriving the same atom degrees
and the same shortest-path matrix each time. Mordred's own caching helps within a family and not
across them.

HUME's claim is that this is unnecessary, because **almost every 2D descriptor in the RDKit and
Mordred catalogues is a function of a small fixed set of per-atom and per-bond quantities plus
the graph.** Compute those once, in C++, and every descriptor becomes a cheap reduction over
arrays that are already in cache.

Concretely:

```
RDKit Mol  ──►  15 per-atom values + 6 per-bond values + the bond list
                          │
                          ├─►  derived primitives  (distance matrix, ring set,
                          │     Crippen contributions, E-state indices, …)
                          │
                          └─►  1,266 descriptor columns, 14 families
```

Everything after the boundary is C++. No Python runs in the compute path.

---

## 2. What crosses the boundary

This is the whole input. `src/hume_core/bindings.cpp` defines it and asserts the layout against
`src/hume/_extract.py`, so a mismatch is an exception rather than silently transposed columns.

**Per atom, 13 integers** — `Z` (atomic number), `deg` (degree), `nH` (total hydrogens),
`fchg` (formal charge), `hyb` (hybridisation), `arom` (aromatic flag), `ring` (in-ring boolean),
`cip` (CIP code), `nring` (ring-membership *count*), `tval` (total valence), `_ChiralityPossible`,
chiral tag, and isotope.

**Per atom, 2 doubles** — mass and Gasteiger partial charge.

**Per bond, 6 integers** — the two atom indices, conjugated, in-ring, a SMARTS-order code, and
RDKit's `BondType` enum.

Four of these are carried rather than re-derived, and each for a measured reason:

* **`nring` alongside `ring`.** SMARTS asks both questions independently: `[R]` is membership,
  `[R1]`/`[R2]` are counts, and the count cannot be recovered from the boolean.
* **`tval`.** Recomputing total valence as `round(Σ bond orders) + nH` disagrees with RDKit on
  **11,238 of 575,571** corpus atoms, because aromatic bonds and hydrogens go through RDKit's own
  rounding rule.
* **The ring flags come from RDKit's single ring perception**, not a second one C++-side. Ring
  perception is numbering-dependent (§5), so a second perception is a second chance to disagree.
* **Isotope**, carried and not derived from mass: `getMass()` says an atom *is* labelled, not
  which isotope it carries.

---

## 3. The families, grouped by what they need

The organising idea is that families sharing an input share the expensive part. The table is
ordered by **input**, not by what the descriptors mean, because that is the axis the
implementation is built on.

### 3a. Needs only the atom/bond arrays — no graph traversal

| family | cols | what it measures |
|---|---:|---|
| **constit** | 43 | Constitutional counts: `C1SP1`…`C4SP3` carbon-hybridisation census, atom and bond counts, molecular weight, `FractionCSP3`, Lipinski/Ghose/Veber filters, `SLogP`, `qed`, `SPS` |
| **rdkcore** | 21 | RDKit's ring-count family, `HeavyAtomMolWt`, `Phi`, `FpDensityMorgan1-3`, stereocentre counts |

These are a single pass over the atom table. Cheapest block in the set.

### 3b. Needs per-atom contribution vectors (Crippen, Gasteiger, Labute, E-state)

| family | cols | what it measures |
|---|---:|---|
| **vsa_bins** | 76 | `SlogP_VSA*`, `SMR_VSA*`, `PEOE_VSA*`, `EState_VSA*`, `VSA_EState*`, plus `MolLogP`, `MolMR`, `TPSA`, `LabuteASA` |
| **estate** | 158 | Kier–Hall electrotopological state: per atom-type **count** (`N…`) and **sum** (`S…`) over 79 atom types |

**Why these belong together.** A *VSA descriptor* (van der Waals surface area) bins a per-atom
property against per-atom surface area — `SlogP_VSA3` is "how much surface area sits on atoms
whose Crippen logP contribution falls in bin 3". The bin edges are module-level constants in
RDKit (`logpBins`, `mrBins`, `chgBins`, `estateBins`), not per-molecule work.

This is the sharpest illustration of the thesis. **Verified on 6,000 adversarial molecules
(`cpp/hard.smi`): 62 columns — the 57 RDKit `*_VSA` columns, `LabuteASA`, and the four E-state
extremes — reconstruct bit-exactly (rtol 1e-9) from four per-atom vectors**: Crippen (logP, MR)
contributions, Gasteiger charges, E-state indices, and Labute ASA contributions. Once the vector
is paid for, each column is one `bisect_right` pass. Sixty-two descriptors for the price of four
vectors.

Three details that silently break the reconstruction, all found by measurement:

* `LabuteASA` is `sum(contribs) + hContrib`; RDKit returns the hydrogen term *separately* from the
  heavy-atom vector. Dropping it is a small uniform shortfall.
* `PEOE_VSA` does **not** clamp NaN charges (elements with no Gasteiger parameters, e.g. Sn). NaN
  falls through `bisect_right` into the final bin; clamping to 0.0 misroutes it.
* `MolLogP` and `MolMR` — unlike every `*_VSA` column — sum over the **H-added** molecule, so they
  need the Crippen hydrogen rows.

### 3c. Needs the distance matrix (all-pairs shortest paths)

| family | cols | what it measures |
|---|---:|---|
| **autocorr** | 540 | Moreau–Broto / Moran / Geary autocorrelations: 6 variants × 9 lags × 10 atomic weightings |
| **topocharge** | 21 | Galvez topological charge indices `GGI1-10`, `JGI1-10`, `JGT10` |
| **topomisc** | 15 | Walk counts (`MWC`, `SRW`, `TSRW10`), Wiener index, `Diameter`, `TopoShapeIndex`, `ABCGG` |
| **infocontent** | 33 | Shannon information content of atom-equivalence classes at orders 0–5 (`IC`, `TIC`, `SIC`, `BIC`, `CIC`, `MIC`, `ZMIC`) |

**Autocorrelation** is the largest family and worth understanding. An autocorrelation at lag *k*
sums a product of an atomic property over every pair of atoms exactly *k* bonds apart:
ATS(k,w) = Σ w_i w_j over pairs at distance k. The *weighting* w is the atomic property — mass
(`m`), van der Waals volume (`v`), Sanderson electronegativity (`se`), polarisability (`pe`),
intrinsic state (`s`), Gasteiger charge (`c`), and so on. The variants differ in normalisation:
Moran (`MATS`) and Geary (`GATS`) are centred and scaled, `ATSC`/`AATSC` are centred, `AATS` is
averaged over the number of pairs. So a single distance matrix plus ten per-atom property vectors
yields all 540 columns.

### 3d. Needs subgraph enumeration

| family | cols | what it measures |
|---|---:|---|
| **chi** | 40 | Kier–Hall connectivity indices over *subgraphs*: `Xp-*` paths, `Xch-*` chains, `Xc-*` clusters, `Xpc-*` path-clusters, `AXp-*` averaged |
| **pathcount** | 11 | Simple path counts `piPC1`–`piPC10`, `TpiPC10` |
| **ringcount** | 49 | Mordred's ring census by size, aromaticity, heteroatom content, fusion |

A **chi index** of order *n* sums, over every subgraph of that order, the product of
1/√δ across its atoms, where δ is the atom's degree (`d`) or valence-corrected degree (`dv`).
The four *shapes* — path, chain (cycle), cluster (star), path-cluster — are why this needs true
subgraph enumeration rather than path walking, and why it is the most expensive family here.

Note the naming collision, which has caused confusion twice: **`chi.h` implements Mordred's
subgraph chi (`Xp-*`) while `hume_blocks.h` implements RDKit's Kier–Hall chi (`chi0n`…`chi4v`)**.
They share no code. RDKit's are registered in *lowercase*.

### 3e. Needs SMARTS matching

| family | cols | what it measures |
|---|---:|---|
| **frag** | 76 | 74 RDKit `fr_*` functional-group counts, plus `NHOHCount` and `HeavyAtomCount` |

Substructure counting against a compiled SMARTS program (`cpp/frag_program.h`), with a second
program for QED's 116 structural alerts. Both matchers share one molecule representation and keep
their recursive-query caches warm across a batch.

### 3f. The block family — mixed, and mostly RDKit-shaped

| family | cols | what it measures |
|---|---:|---|
| **blocks** | 182 | RDKit's Kier–Hall `chi0n`…`chi7v`, `Kappa1-3`, `HallKierAlpha`, `BalabanJ`, `BertzCT`, `Ipc`, the four `BCUT2D_*` eigenvalue pairs, and assorted graph invariants |

---

## 4. What this buys

Measured end-to-end on `c7i.4xlarge` (16 vCPU), per billion molecules:

| | µs/molecule | hours per 1e9 |
|---|---:|---:|
| ECFP4 alone | 16.6 | 4.6 |
| **HUME** (1,266 descriptors + ECFP6) | **124** | **34** |
| ECFP4 + RDKit-180 | 444 | 123 |
| ECFP4 + Mordred-685 | 4,683 | 1,301 |
| ECFP4 + all descriptors (Python) | 6,190 | 1,722 |

**~50× faster than the equivalent Python descriptor block**, for the same information.

---

## 5. Where HUME deliberately differs

This is the short list, and it is short on purpose. **For the overwhelming majority of columns
HUME does exactly what RDKit and Mordred do, in C++ rather than Python, and is verified bit-exact
against them.** The house rule (`PORT_STATUS.md` §1) is:

> Reproduce a **quirk**; diverge from an **ill-posed definition**. The test is a single question:
> *is the upstream descriptor a function of the molecule?*

A quirk gives the same wrong answer every time — so it is reproduced bit-for-bit and commented.
Two are kept deliberately: five E-state SMARTS rows whose missing semicolon voids an aromaticity
constraint, and `[SeD2H0]` decoding as an element-number query with no aromaticity constraint.

An **ill-posed** descriptor gives *different answers for the same molecule* depending on atom
numbering or on which Kekulé structure the perceiver happened to pick. There is nothing there to
be exact against, and "we reproduce Mordred" would be a claim about a coin flip.

**How ill-posedness is detected, mechanically:** permute the input ordering and recompute; any
column that moves is ill-posed. Critically, **the screen must shuffle bonds, not only atoms** —
`Chem.RenumberAtoms` permutes atoms and leaves the bond list alone, while RDKit's ring perception
reads the bond list. This repo shipped the weaker atom-only screen for a while. The molecule
`O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2` is stable across **201** atom renumberings
and yields two different ring sets the moment bond order is shuffled too.

### The three divergences

**(i) `InformationContent` — Mordred's is not a function of the molecule.** Two independent
defects, both established by reading the source and confirmed by measurement:

1. `InformationContentBase` sets `kekulize = True`, so the atom-equivalence codes are built on a
   Kekulé structure. An aromatic bond enters the code as SINGLE or DOUBLE depending on which
   structure the perceiver picked.
2. `BFSTree._expand` mutates a visited set *while iterating over it*. Two adjacent siblings at the
   same depth are both in the tree and neither is visited when the loop starts; whichever the dict
   yields first claims the other as its child. Dict order is insertion order, which is
   `GetNeighbors()` order, which is atom numbering.

**Measured: on the first 2,000 molecules of `cpp/hard.smi`, 32.3% change at least one
InformationContent column under a single perturbation of input order** — 15.6% under atom
renumbering alone, 28.0% once bonds are shuffled too.

*Resolution:* an aromatic bond keeps its own bond-type symbol rather than being kekulized away,
and the tree is layered by graph distance. **Orders 1–5 therefore differ from Mordred by design;
order 0 is unchanged.**

**(ii) Ring perception — one repaired ring set, used everywhere.** `Chem.GetSymmSSSR` is not a
function of the graph. The SSSR *basis* is stable; what flips is whether `symmetrizeSSSR` finds a
symmetry-equivalent extra ring of a size already present. `C1=CC2C3C(C=C1)C23` gives ring sizes
(3,3,7) on 33 of 60 numberings and (3,3,7,7) on the other 27. Brute force over every simple cycle
confirms the larger answer is the **relevant-cycle set** — the object `symmetrizeSSSR` is reaching
for and reaches only sometimes.

*Resolution:* perceive rings on a **skeleton rebuilt from scratch** — *n* carbons in canonical-rank
order, bonds added in sorted `(rank_u, rank_v)` order. Ring perception reads only the graph, so
the skeleton asks exactly the right question and puts bond order under canonical control too.
**100,000 molecules × 49 columns × 5 numberings: 22 molecules move before, 0 after.** It changes
RDKit's answer on 32 molecules, all 32 independently confirmed unstable.

An earlier prescribed repair — canonical atom ranks, rings compared by (size, sorted rank vector)
— was **wrong** and is recorded as such: it left 3 of 100,000 still moving and made those 3 worse,
because canonical ranks fix atom numbering and not bond order, which is the axis that decides.

The consequence is deliberate: RDKit's own 13 ring columns move on those 32 molecules too. Taking
RDKit's raw rings for those would give **two different ring sets inside one feature vector**.

**(iii) Aromaticity, two repairs**, relevant only where HUME does its own ring reasoning rather
than inheriting the boundary's `arom` flag:

* A ring sulfur carrying an exocyclic double bond is a sulfoxide — pyramidal, therefore not
  aromatic.
* "A bond in an all-aromatic ring is aromatic" must run **after** perception, not during it: a
  fused system can contain ring bonds belonging to no tested subset.

### How to state this in the paper

> HUME reproduces RDKit and Mordred bit-exactly for all well-posed columns. Three families are
> ill-posed upstream — their value depends on atom numbering or Kekulé choice rather than on the
> molecule — and for these HUME implements the well-posed definition the upstream code was
> evidently reaching for, and is deterministic under atom and bond permutation.

That is a **stronger** claim than bit-exactness, not a weaker one, and it should be presented that
way rather than buried.

---

## 6. Which 865 of the 1,830, and why

HUME carries 1,266 columns. The RDKit ∪ Mordred union is 1,830 (217 + 1,613). The selection
(`dedupe.py` → `data/dedupe.json`) is:

```
1,830 union
  ->  1,275 usable   (555 dropped: constant, or >5% undefined)
  ->    864 kept     (411 dropped: |Spearman rho| >= 0.99 with a column already kept)
```

The cover is **greedy in ascending compute cost**, so the survivor of each redundant group is the
cheap member. Ranks rather than raw values, so the criterion is monotone redundancy (Spearman),
not merely linear.

**A known limitation, quantified.** The threshold is estimated on a corpus, and two columns that
are redundant *there* can separate on a specific endpoint. Measured (`dedupe_cost.py`) with the
filter as the only difference between two arms: the dropped columns are worth **+0.016 AUROC on
`bioavail` (n=640)** and **+0.006 on `hia` (n=578)**, and nothing measurable on any classification
set above 7,000 molecules. On `bioavail` the top dropped columns are only 0.88–0.96 correlated
with their nearest surviving partner *on that dataset*, against the ≥0.99 that caused the drop.
