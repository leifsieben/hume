# What is in the embedding, and why

The single map of the representation: what ECFP already covers, what we compute from scratch,
and what is left over to predict. Costs are per molecule on this machine; see `PRIMITIVES.md`
for the primitive-by-primitive breakdown and `FINDINGS.md` for the experiments behind each
decision.

```
  SMILES
    |
    +-- parse (RDKit, C++)                                    ~48 us   shared by everything
    |
    +-- ECFP-2048 counts, r=2, chirality on                   ~29 us   2048 cols
    |
    +-- CORE descriptors      (from cheap shared primitives)  ~59 us    639 cols
    +-- resistance / stereo / conjugation / cycles (new)      ~100 us*  157 cols
    |
    +-- PREDICT block         (surrogate forward pass)        ~0 us     226 cols
                                                                       ---------
                                                              total    3070 cols
```

\* **Projection, not a measurement.** The four new blocks measure **367 us** as reference
Python (resistance 173, cycles 119, stereo 38, conjugation 37). The ~100 us figure is the C++
target after (a) sharing one distance matrix instead of recomputing it per block, (b) removing
small-array numpy overhead, which is the majority of the current cost, and (c) restricting
`L+` to biconnected components. Python total for ECFP + CORE + new blocks is **455 us**, which
breaches the 10x-ECFP budget (290 us); the C++ target of ~159 us does not. Every other cost in
this document is measured.

---

## 1. What ECFP already covers

ECFP is Weisfeiler-Lehman colour refinement with hashing. Three steps, and each one determines
a class of things the fingerprint does and does not express.

1. **Initial invariant.** Each atom is hashed from the Daylight tuple: atomic number, degree,
   attached hydrogens, formal charge, isotope, ring-membership *flag*, plus the chiral tag
   when `includeChirality=True`.
2. **Refinement.** For r = 1..2, `id_r(a) = hash(id_{r-1}(a), sorted{(bond order, id_{r-1}(nbr))})`.
3. **Collection and folding.** The multiset of all identifiers over all atoms and radii,
   modulo 2048.

### Covered directly and well

| property | how |
|---|---|
| atom identity, valence, charge, isotope | initial invariant |
| hydrogen counts, degree, branching (local) | initial invariant |
| bond orders, aromaticity | refinement step |
| **all substructure to radius 2** (5-atom diameter) | the identifiers themselves |
| functional groups, substitution patterns | ditto — this is why the 99 surviving `rdkit_core` columns are almost entirely `fr_*` SMARTS counts, largely redundant with ECFP |
| **atom stereochemistry (R/S)** | chiral tag — verified, 4-10 bits move between enantiomers |
| **bond stereochemistry (E/Z)** | also carried — verified, Tanimoto 0.50 for a butenoic acid E/Z pair |
| fragment *counts*, not just presence | count fingerprint rather than binary |

Stereo is worth stating twice because we got it wrong once: ECFP is **not** stereo-blind. The
descriptor union is — zero of RDKit's 217 and zero of Mordred's 1,613 descriptors move across
any enantiomer pair.

### Present but not expressible

ECFP is near-injective on real molecules: anthracene and phenanthrene, decalin isomers, and
fused-ring isomers all give different fingerprints. By the data-processing inequality, nothing
derived from the 2D graph is *missing*. The problem is form, not information.

A gradient-boosted tree over 2048 hashed presence counts cannot cheaply construct:

- **Pairwise sums** `sum_ij f(i,j)` — needs quadratic interaction across bits. This is why the
  419 autocorrelation columns pay, and it is the shape every new block below takes.
- **Ratios** — trees cannot divide. `Kf/n`, `FractionCSP3`, densities, fractions.
- **Ordered global axes** — anthracene vs phenanthrene differ in *which bits are set*, but
  `Kf` = 163.85 vs 160.78 is a single ordered scalar on a physical axis. Same information,
  radically different usability.

This is the confirmed mechanism of the whole project — **degree reduction, not information
gain**. It is why Chi and PathCount pay (+0.126, +0.061) despite being fully determined by the
graph ECFP already sees.

### Genuinely outside its expressive power

WL-1 distinguishes exactly the graphs that homomorphism counts *from trees* distinguish
(Dell/Grohe/Rattan). Every pattern containing a **cycle** is therefore outside colour
refinement. This is the one place where the limitation is a theorem rather than a conditioning
argument, and it is what `cycles.py` targets.

---

## 2. The inputs we compute

Everything in section 3 is a function of a small set of **primitives** — intermediate objects
computed once per molecule and reused across hundreds of descriptors. That sharing is the
whole reason CORE costs 59 us rather than the 325 us a naive per-descriptor sum predicts.

Below, each primitive is defined, then costed, then attributed to what it unlocks.

### 2.1 Cheap primitives — the CORE basis

**Molecule parse and sanitise — ~48 us**
Turning a SMILES string into a graph object. RDKit reads the string into atoms and bonds,
kekulises, perceives aromaticity, computes implicit hydrogen counts, assigns stereochemistry
from the directional bonds and `@`/`@@` markers, and validates valences. This is unavoidable
and shared: ECFP needs it too, so it is not attributable to descriptors. It is also the single
largest fixed cost in the pipeline and is already C++.

**Adjacency matrix `A` — ~0 us**
An n x n symmetric binary matrix over *heavy atoms* (hydrogens are implicit), where
`A[i][j] = 1` exactly when atoms i and j share a bond. It is **unweighted**: bond order does
not appear. `A` is therefore pure connectivity, stripped of chemistry. Mordred's Barysz matrix
is the weighted counterpart, substituting bond-order and electronegativity terms for the 1s.
*Unlocks:* everything topological, and every primitive below that is built from it.

**Distance matrix `D` — ~2 us**
n x n, where `D[i][j]` is the number of bonds on the **shortest path** between i and j —
topological distance, not spatial distance in angstroms. Diagonal is 0, the maximum entry is
the graph diameter, and disconnected pairs come back as 1e8 (a sentinel that must be masked,
not binned). *Unlocks:* all **419 autocorrelation** columns, Wiener, Balaban, Kappa, and the
distance-binning in every new block.

**Ring perception (SSSR) — ~3 us**
The Smallest Set of Smallest Rings: a *basis* for the molecule's cycle space, of size
`bonds - atoms + components` (the cyclomatic number). Benzene gives one ring, naphthalene two,
cubane five. Note this is a basis, not a list of all cycles — the distinction is exactly what
`cycles.py` exploits. RDKit's `GetSymmSSSR` returns a symmetrised version, because the basis
is not unique for symmetric molecules. *Unlocks:* RingCount's 49 columns, ring-membership
flags, fusion and bridgehead detection.

**Atom property vectors — ~1 us**
Per-atom scalars obtained by **table lookup on atomic number** — no computation, just
indexing: mass, van der Waals volume (4/3 pi r^3), Pauling electronegativity, polarizability,
ionisation potential, and intrinsic state. These are the six weights Mordred's autocorrelations
use, plus unity. Because they are lookups, adding another property is free.
*Unlocks:* the property axis of every autocorrelation — `D` supplies "which pairs", these
supply "weighted by what".

**Walk matrices `A^1..A^8` — ~14 us**
`(A^k)[i][j]` is the **number of walks of length exactly k** from i to j. A walk may revisit
atoms and bonds; this is what makes it cheap, because walk counts compose by matrix
multiplication. Eight dense matmuls on an n<=60 matrix. The diagonal `(A^k)[i][i]` counts
closed walks — walks that return to their starting atom.
*Unlocks:* WalkCount (traces), the RWSE features in `resistance.py` (diagonals), and exact
`C3`/`C4`/`C5` in `cycles.py` (inclusion-exclusion on traces).

**Labute ASA — ~8 us**
An approximate solvent-accessible surface area estimated from atomic contributions plus
connectivity corrections — a 2D stand-in for a 3D quantity, requiring no conformer.
*Unlocks:* the surface-area normalisation used by the VSA-family descriptors.

> **The walk/path distinction is the load-bearing one in this whole document.** A *walk* may
> revisit atoms, so walk counts are matrix powers and cost microseconds. A *path* may not, so
> path counts have no matrix shortcut and must be enumerated. That single fact is why
> WalkCount is in CORE at 14 us while Chi and PathCount are in PREDICT.

### 2.2 New primitives added by the four new blocks

**Laplacian `L = Deg - A` and its pseudo-inverse `L+` — ~31 us**
`Deg` is the diagonal matrix of atom degrees, so `L` has degrees down the diagonal and -1 at
each bond. Every row sums to zero, which makes `L` singular — it always has eigenvalue 0, and
the number of zero eigenvalues equals the number of disconnected fragments. Because it is
singular it cannot be inverted, so we take the Moore-Penrose pseudo-inverse, which for a
connected component is exactly `(L + J/n)^-1 - J/n` with `J` the all-ones matrix — one dense
solve rather than an eigendecomposition.

The physical reading: treat every bond as a 1-ohm resistor. Then
`Omega[i][j] = L+[i][i] + L+[j][j] - 2*L+[i][j]` is the **effective resistance** between atoms
i and j. Two atoms joined by many parallel routes are electrically closer than shortest path
suggests; on a tree, where there is exactly one route, `Omega == d` exactly.
*Unlocks:* the entire `resistance.py` block. Restricting the solve to biconnected components
(resistance is additive over them, and `Omega == d` across every bridge) is the ~5 us C++ path.

**Normalised Laplacian spectrum — ~9 us**
Eigenvalues of `I - D_deg^-1/2 A D_deg^-1/2`. The normalisation matters: the plain Laplacian's
spectrum scales with degree, so it is not comparable across molecules, whereas the normalised
version always lies in [0, 2] and can be histogrammed in fixed bins. The smallest nonzero
eigenvalue is the **Fiedler value** (algebraic connectivity) — how hard the molecule is to cut
into two pieces, which is a global branching measure.
*Unlocks:* the spectral-density and Fiedler features in `resistance.py`.

**Conjugated-bond union-find — ~2 us**
RDKit flags each bond as conjugated or not (part of a delocalised pi system). Union-find then
merges the flagged bonds into **connected components** — the actual pi systems, which routinely
span several rings and their substituents. *Unlocks:* the whole `conjugation.py` block.

**CIP codes — ~2 us**
Cahn-Ingold-Prelog R/S labels, assigned by ranking each stereocentre's four substituents by
priority and reading the handedness. Already computed during parsing, so the explicit call is
a safety net for molecules built some other way, not a cost. *Unlocks:* `stereo.py`.

**Bounded cycle enumeration (k <= 8) — ~119 us**
Depth-first search from each atom, restricted to higher-indexed atoms so each cycle is found
once from its lowest member, halted at length 8. Unlike walks this cannot be done with matrix
powers, because a cycle may not revisit atoms — but the return-to-start constraint prunes the
search hard, which is why cycles are affordable to enumerate when general paths are not.
*Unlocks:* exact `C6`/`C7`/`C8` and all per-atom participation counts in `cycles.py`.

### 2.3 Expensive primitives — why the PREDICT block exists

**EState (electrotopological state) indices — 242 us, the single worst offender**
Kier-Hall. Each atom gets an intrinsic state `I = ((2/N)^2 * dv + 1) / d`, combining its
principal quantum number `N`, valence connectivity `dv` (electrons available for bonding) and
simple connectivity `d` (count of heavy neighbours). Every atom is then **perturbed by every
other atom**, damped by squared distance:
`S[i] = I[i] + sum_j (I[i] - I[j]) / (d(i,j) + 1)^2`. The all-pairs perturbation is what costs;
it fuses electronic and topological information into one per-atom number.
*Blocks:* 73 descriptors (50 Mordred EState + 23 RDKit `EState_VSA`/`VSA_EState`).

**Crippen contributions — 90 us**
Wildman-Crippen atom typing: each atom is matched against roughly 110 SMARTS patterns to
assign it a logP contribution and a molar-refractivity contribution. The **SMARTS matching** is
the cost, not the arithmetic. Summing gives `MolLogP` and `MolMR`; binning by surface area
gives the `SlogP_VSA` and `SMR_VSA` families.
*Blocks:* 20 descriptors.

**Burden matrix eigenvalues (BCUT2D) — 40 us**
A Burden matrix puts an atomic property on the diagonal (charge, logP, mass, or polarizability)
and bond-order-derived values off-diagonal for bonded pairs, with a small constant elsewhere.
Its highest and lowest eigenvalues are the BCUT descriptors — property-weighted spectral
extremes. Four properties means four eigendecompositions.
*Blocks:* 8 descriptors.

**Gasteiger partial charges — 17 us**
Partial Equalisation of Orbital Electronegativity. Charge is transferred iteratively between
bonded atoms according to their electronegativity difference, damped by a factor of one half
each round, until it converges after about six iterations. The result is a per-atom partial
charge without any quantum calculation.
*Blocks:* 13 descriptors (`PEOE_VSA` and charge extremes).

**TPSA contributions — 9 us**
Topological polar surface area: a lookup of tabulated surface contributions for N, O, S and P
atoms according to their bonding environment. Cheap in absolute terms, but it buys exactly one
descriptor, so it does not earn a place in the shared basis.
*Blocks:* 1 descriptor.

**Path enumeration — no cheap primitive exists**
Chi indices need counts of **paths** of length k — subgraphs that never revisit an atom —
along with cluster and chain fragments. Because paths cannot revisit, there is no
matrix-power shortcut, and unlike cycles there is no return-to-start constraint to prune the
search. Enumeration is the only route.
*Blocks:* Chi (54) and PathCount (11) — and this is the **highest-value predicted family we
have measured**, at +0.126 downstream for Chi.

> **Open question raised by `cycles.py`.** Having now written a bounded enumerator, it is worth
> measuring whether bounded *path* enumeration is also affordable at k <= 7, which would move
> Chi and PathCount from PREDICT into CORE and take the most valuable predicted family off the
> surrogate entirely. The pruning argument says paths are much worse than cycles, but the
> difference has not been measured. Not yet tested.

The split is drawn on **cost**, not on downstream benefit. A descriptor is in CORE if it is
reachable from the cheap basis, and in PREDICT otherwise.

---

## 3. Descriptor groups we compute

**639 CORE columns**, all reachable from the cheap primitives above.

| group | cols | primitive | what it measures |
|---|---|---|---|
| Autocorrelation | **419** | `D` + atom properties | `sum_{d(i,j)=k} p_i p_j` for six atom properties at lags 1-8. The long-range channel ECFP's radius-2 ball cannot reach. |
| `rdkit_core` | 99 | SSSR + SMARTS | `fr_*` functional-group counts, ring counts, H-bond donors/acceptors. Mostly redundant with ECFP; kept because they are free. |
| RingCount | 49 | SSSR | rings by size, aromatic/aliphatic/saturated, carbo/heterocyclic, spiro, bridgehead |
| TopologicalCharge | 21 | `A`, `D` | charge-transfer indices from the Galvez matrix |
| CarbonTypes | 9 | atom properties | primary/secondary/tertiary/quaternary, sp/sp2/sp3 |
| AtomCount, BondCount | 14 | trivial | composition |
| WalkCount | 6 | `A^k` | molecular and self-returning walk counts (traces) |
| Constitutional | 4 | trivial | size and composition ratios |
| MolecularDistanceEdge | 3 | `D` | distance-edge indices by carbon type |
| KappaShapeIndex | 3 | `A`, `D` | shape anisotropy — linear vs spherical |
| WienerIndex, TopologicalIndex | 4 | `D` | global compactness |
| Lipinski, RotatableBond, Polarizability, ABCIndex, BalabanJ, VdwVolumeABC, FragmentComplexity | 7 | mixed | drug-likeness and bulk |

**157 new columns** in four blocks, each built so that it is **identically zero when its
axis is absent** — an acyclic molecule cannot be helped or hurt by the resistance block. That
is orthogonality by construction, not by hope.

| block | cols | quantity | what ECFP cannot express about it |
|---|---|---|---|
| `resistance.py` | 77 | `Delta_ij = d_ij - Omega_ij`, resistance-binned autocorrelation, Kirchhoff index, RWSE pooling, Laplacian spectrum | **Path multiplicity.** `Omega == d` on any tree, so this is zero for acyclic molecules and measures ring fusion alone. |
| `cycles.py` | 33 | exact `C3..C8`, per-atom participation, cycle redundancy, hetero/aromatic ring typing | **The WL-1 theorem case.** Cycle counts are provably outside colour refinement. Also not `RingCount`: cubane has 5 SSSR rings but 16 six-cycles, redundancy 4.67. |
| `conjugation.py` | 24 | pi-system components, sizes, diameter, linearity, cross-conjugation, non-aromatic extent | **Global connected-component extent.** Anthracene and a C14 polyene are both one system of 14 atoms; linearity separates them 0.54 vs 1.00. |
| `stereo.py` | 23 | CIP parity `sum s_i` (odd, flips on mirroring), `sum_{d=k} s_i s_j` (even, invariant), E/Z bond parity, cross terms | **Conditioning only** — ECFP has this information in 4-10 bits. Demoted accordingly. |

---

## 4. What is left over

**226 PREDICT columns**, reached by a surrogate model from ECFP + CORE rather than computed.
Naive computation cost is **26,946 us** against **325 us** for everything above — an 83x
economic argument, which is the entire reason this project exists.

| group | cols | blocked by | what it measures |
|---|---|---|---|
| EState (Mordred + RDKit) | **73** | EState indices, 242 us | electrotopological state — the single most expensive primitive |
| Chi + Kappa (Mordred + RDKit) | **54** | path enumeration | connectivity indices over paths, clusters, chains. **Highest measured downstream value of any predicted family (+0.126).** |
| InformationContent | 33 | neighbourhood-symmetry entropy | Shannon entropy over atom equivalence classes |
| Crippen | 20 | Crippen contributions, 90 us | `MolLogP`, `MolMR`, `SlogP_VSA`, `SMR_VSA` |
| Gasteiger | 13 | partial charges, 17 us | `PEOE_VSA`, charge extremes |
| PathCount | 11 | path enumeration | self-returning and total path counts (+0.061) |
| BCUT2D | 8 | Burden matrix eigenvalues, 40 us | charge- and mass-weighted spectral extremes |
| MoeType, AcidBase, CPSA, TopoPSA, SLogP, LogS, Framework, composite | 14 | mixed | surface-area and physicochemical composites |

**By default these are predicted.** The surrogate model has not been selected — five candidates
(ridge, linear+quadratic, Pi-net, MLP, GNN) are trained but ranking them on four MoleculeNet
datasets is not a decision, and the full LOCKED-suite run is Phase 1 step 0.

**The menu exposes exact computation as an opt-in.** The packaged API offers four independent
blocks so a caller can trade speed for exactness:

```
  ecfp              2048    ~29 us     always computed
  fast              796     ~159 us    CORE + the four new blocks, always computed
  predicted         226     ~0 us      surrogate forward pass          <- default
  exact             226     ~27 ms     deduplicated RDKit + Mordred    <- opt-in
```

A billion molecules at the default is roughly 2 core-days for featurisation; the same billion
with `exact` is about 310 core-days. That ratio is the product.

### Known holes

- **3D and QM.** All of the above is graph-only. Conformer-dependent shape, and anything
  electronic beyond a 2D proxy, is Phase 4 (UMA distillation) and deliberately deferred.
- **Enantiomers in the predicted block.** Every predicted descriptor is a 2D graph invariant,
  so the surrogate cannot and need not carry stereo — ECFP carries it directly. This makes a
  stereo failure in the surrogate impossible rather than merely unlikely.
- **Tautomers and protonation state.** Not addressed anywhere; the representation describes
  the molecule as written.

---

## The block flags (decided 2026-08-26)

**Every block is independently switchable, and any combination is legal.** The caller decides
what to pay for; nothing is welded to anything else. There are four:

| flag | columns | what it is |
|---|---|---|
| `ecfp` | 2048 | Morgan r=2 counts, chirality on. (`r3cfp` swaps radius 3 in its place.) |
| `core` | 699 | descriptor families cheap enough to compute exactly |
| `blocks` | ~166 | our five: resistance, cycles, conjugation, stereo, chi |
| `predict` | 166 | the expensive families, predicted by a proxy — or `exact` to compute them |

So `ecfp` alone, `ecfp+core`, `ecfp+blocks`, `ecfp+core+blocks+predict`, and every other subset
are all valid requests. `blocks.split()` already returns `core` and `predict` as disjoint column
lists, so this is concatenation, not coupling — the DEV grid has been building arms this way all
along.

### The default is the hard part, and it is benchmark-dependent

The two benchmarks disagree about `core`, and not marginally:

* **34-dataset potency suite (MoleculeACE-style):** bare `ecfp` has the BEST mean rank (2.85) and
  beats `ecfp+core` on 28 of 34 datasets. `core` HURTS.
* **28-dataset DEV grid (QM, physchem, ADMET, tox):** descriptors help — +25.8% on QM, +17.0% on
  physchem.

This is the same MoleculeNet/MoleculeACE split seen throughout the project, and it means there is
no single defensible default across all chemistry. The decision rule adopted:

1. Set the default from the **DEV grid**, because that is the benchmark where the descriptor
   question has a positive answer and where a wrong default costs the most.
2. If `core` proves to hurt *consistently* across both suites, ship it **off** by default.
3. Whatever is chosen, the docs must state the regime it was chosen for. A default that is right
   for QM and wrong for potency cliffs is not a bug, but silently presenting it as universal is.

### Cost note that shapes the flags

Measured on this machine, per molecule, in the CURRENT Python pipeline:

    core block, exact       52,091 us     <- three times the predict block
    predict block, exact    17,241 us
    gnn proxy                  211 us     (139 graph build + 72 forward)
    ridge proxy                  0.75 us

The CORE/PREDICT split was drawn on **C++** cost estimates, not Python ones. In Python today the
"cheap" half is the expensive half, by 3x. The split only earns its name once `core` is in the
C++ core -- which is exactly what `cpp/` is for, and it is now on the critical path rather than
being an optimisation.
