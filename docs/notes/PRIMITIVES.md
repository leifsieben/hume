# Input primitives for the descriptor suite

Reference for the C++ implementation. Every descriptor in the 865-column deduplicated set
(|ρ| ≥ 0.99) is a function of a small number of shared primitives. Compute the primitive
once, and every descriptor built on it becomes nearly free.

All timings are **marginal on an already-parsed `Mol`**, measured on 400 ChEMBL molecules
(26 heavy atoms mean, M4 Pro, single core). The parse floor is 59.0 µs and ECFP adds 30.4 µs;
neither is avoidable and both are already C++.

## The primitives

| primitive | µs/mol | what it is |
|---|---|---|
| ring info (SSSR) | **0.6** | smallest set of smallest rings; already computed during sanitisation, so effectively free |
| adjacency matrix `A` | **0.9** | n×n binary bond-connectivity |
| Labute ASA contribs | **6.0** | per-atom approximate surface area |
| TPSA contribs | **8.9** | per-atom polar surface contributions |
| Gasteiger charges | **17.1** | per-atom partial charges, iterative PEOE |
| atom property vectors | **24.3** | per-atom Z, mass, vdW volume, valence electrons, electronegativity, polarizability. Pure table lookups — this is a Python loop over atoms and should be ~0 µs in C++ |
| distance matrix `D` | **26.8** | n×n topological (bond-count) distances |
| adjacency eigenvalues | **40.0** | spectrum of `A` (or the Burden matrix) |
| adjacency powers `A¹..A⁸` | **51.7** | walk counts of each length |
| Crippen contribs | **89.5** | per-atom logP and molar-refractivity contributions |
| **EState indices** | **242.4** | per-atom electrotopological state; by far the most expensive |
| **sum if all computed** | **508.2** | |
| **measured together** | **466.8** | only ~8% is shared, so these are close to additive |

Two primitives are deliberately absent because they are the reason their descriptors get
predicted rather than computed: the **detour matrix** (longest path — NP-hard) and
**path/walk enumeration** for Chi, PathCount and MolecularId.

## Which descriptors each primitive unlocks

Counts are from the family inventory; the mapping is derived from family definitions, not
measured per-descriptor.

| primitive | µs | descriptors enabled | desc/µs |
|---|---|---|---|
| **distance matrix `D`** | 26.8 | Autocorrelation (606), MolecularDistanceEdge (19), DistanceMatrix (12), TopologicalCharge (21), ExtendedTopochemicalAtom (45), WienerIndex (2), BalabanJ (1), TopologicalIndex (4) — **~710** | **26.5** |
| Gasteiger | 17.1 | PEOE_VSA (14), charge-weighted BCUT variants — ~18 | 1.1 |
| **EState indices** | 242.4 | Mordred EState (316), EState_VSA (11), VSA_EState (10), Max/Min aggregates (4) — **~341** | **1.4** |
| ring info | 0.6 | RingCount (138), Aromatic (2), ring constitutional counts — ~145 | 240 |
| adjacency `A` | 0.9 | AdjacencyMatrix (12), ZagrebIndex (4), degree-based indices — ~20 | 22 |
| `A¹..A⁸` | 51.7 | WalkCount (21), VertexAdjacencyInformation, connectivity — ~25 | 0.5 |
| eigenvalues | 40.0 | BCUT (24), BCUT2D (8), spectral — ~32 | 0.8 |
| Crippen | 89.5 | SlogP_VSA (12), SMR_VSA (10), MolLogP, MolMR — ~24 | 0.27 |
| Labute ASA | 6.0 | LabuteASA, prerequisite for every `*_VSA` binning | — |
| TPSA | 8.9 | TPSA (2) | 0.22 |
| atom properties | 24.3 | prerequisite for Autocorrelation, Barysz, most weighted descriptors | — |

**The distance matrix is the single best purchase in the entire suite**: 26.8 µs unlocks
~710 descriptors, including the whole 606-member Autocorrelation family. Nothing else is
close.

**EState is the expensive one to think hard about**: 242 µs — half the entire primitive
budget — for ~341 descriptors. Worth it if those descriptors carry signal, but it is the
one primitive where "predict instead" is a serious option.

**Crippen is the worst ratio**: 89.5 µs for ~24 descriptors. Worth keeping only because
MolLogP is chemically load-bearing.

## Recommended primitive set

| tier | primitives | µs | descriptors reachable |
|---|---|---|---|
| **core** | ring info, `A`, atom properties, `D`, Labute ASA | 58.6 (≈24 in C++) | ~880 |
| **+cheap** | Gasteiger, TPSA | 84.6 | ~900 |
| **+moderate** | Crippen, eigenvalues, `A^k` | 265.8 | ~980 |
| **+EState** | EState indices | 508.2 | ~1,320 |

The core tier is the striking one: **~58 µs of primitives reaches ~880 descriptors**, and
roughly 24 µs of that is a Python atom loop that disappears in C++.

## Compute vs predict

**Compute** — everything reachable from the core + cheap tiers, i.e. anything expressible as
a masked quadratic form on `D`, a count over ring info or `A`, a table lookup, or a `*_VSA`
binning. That is the large majority of the 865.

**Predict** — descriptors whose primitive is itself the expensive part and does not amortise:

| family | n | why it stays expensive |
|---|---|---|
| Chi | 56 | path enumeration, not a matrix operation |
| PathCount | 21 | same |
| MolecularId | 12 | same |
| DetourMatrix | 14 | NP-hard longest path |
| InformationContent | 42 | entropy over atom-type orbits |
| BCUT / BCUT2D | 32 | eigendecomposition per weighting scheme |
| qed / SPS / BertzCT | 3 | composite scores, 405 µs each |
| Framework | 1 | scaffold perception |
| *(optionally EState-derived)* | *341* | *only if the 242 µs primitive is dropped* |

## Notes for the C++ implementation

1. **Compute each primitive exactly once per molecule and pass it down.** RDKit's Python
   descriptor functions each recompute their own intermediates — the four EState aggregates
   recompute a 242 µs primitive four times. Deriving them from one shared vector was measured
   **2.5× faster and bit-exact** (`MaxEStateIndex`, `MinEStateIndex`, `MaxAbsEStateIndex`,
   `MinAbsEStateIndex`, `MolLogP`, `MolMR`, `LabuteASA`: 1,006 µs → 402 µs).
2. **Use `CalcCrippenDescriptors`, not a sum of `_CalcCrippenContribs`.** Summing contribs on
   a heavy-atom-only molecule omits implicit-hydrogen contributions and is wrong by ~1.8 logP
   units on average — a semantic error, not rounding.
3. **Atom property vectors should be compile-time tables** indexed by atomic number. The
   24.3 µs measured here is Python iteration, not lookup cost.
4. **Sort molecules by atom count before batching.** Padding to the batch maximum wastes
   quadratically; naive batching measured *slower* than per-molecule (309 vs 210 µs), while
   size-bucketed batches of 256 reached 115 µs with 1.10× padding waste.
5. **Verify bit-exactness against RDKit/Mordred on ≥10⁴ molecules per descriptor.** Partial
   sanitisation looked fine on wall-clock and was silently wrong for 12% of molecules.
