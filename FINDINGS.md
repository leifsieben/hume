# Is it worth predicting expensive molecular descriptors?

**The question dissolved.** We set out to predict Mordred cheaply, established that we
could not, and then discovered Mordred was not worth having in the first place.

Run 2026-08-23/24. RMSE (lower better), scaffold 5-fold CV, untuned XGBoost, on the
`ChemTFM_OLD` harness — which reproduces its recorded MoleculeNet references exactly
(1.0196 vs 1.020; 0.9634 vs 0.963).

## 1. The headline: Mordred does not help

| arm | MoleculeACE (30, cliffs) | MoleculeNet (4, smooth) |
|---|---|---|
| ecfp | **0.7732** | 1.0855 |
| ecfp + desc (RDKit-96) | 0.7700 (−0.003, 17/30) | **0.9363 (−0.149, 3/4)** |
| ecfp + mordred | 0.7903 (**+0.017**, 7/30) | 0.9850 (−0.101, 3/4) |
| ecfp + desc + mordred | 0.7883 (**+0.015**, 5/30) | 0.9719 (−0.114, 3/4) |

On cliffs Mordred is **harmful**. On smooth properties descriptors help a great deal, but
**96 RDKit descriptors (976 µs) capture all of it** — adding Mordred's 1,613 (72,700 µs)
makes it *worse* (0.9363 → 0.9719).

### The error that hid this

Every prior Mordred evaluation used **`desc` (RDKit-96) alone** as the baseline, never ECFP —
including `benchmark_results.md`'s "Mordred downstream control" (0.868 → 0.840) and every
arm in sections 2–3 below. Against a fingerprint-free baseline Mordred looks valuable. But
plain ECFP scores 0.7732, well below `desc+Mordred`'s 0.8054. **Once fingerprints are
present — as they always are in production — the effect disappears.**

Corollary: the "full Mordred available as a rescoring tier" note in `benchmark_results.md`
is not a known limitation worth reopening. The locked v1 scheme (ECFP + RDKit-96 + ErG)
already captures what is there.

## 2. Superseded: the surrogate result

Before the baseline error was found, we trained ECFP → descriptor surrogates. Recorded for
completeness; the conclusion stands but the motivation does not.

An MLP (2048→2048→1024→1753, 30 epochs, 100k scaffold-diverse PubChem molecules) reached
median **R² 0.949** on the descriptor union — and delivered **2%** of Mordred's downstream
gain over `desc`. `pred_union` was *worse* than baseline.

The structural reason still holds and generalises: **a surrogate can only transmit
information present in its input.** Its output is a function of ECFP, so a downstream model
already holding ECFP gains nothing. This is the data-processing inequality, and it also
rules out "leapfrogging" chains (RDKit-96 → Morgan → UMA): composition never creates
information.

Three attempts to obtain Mordred's value without Mordred:

| approach | % of gain over `desc` |
|---|---|
| top-30 cherry-pick (supervised selection) | 26% |
| PCA-64 of the union (unsupervised projection) | 26% |
| ECFP surrogate at R² 0.949 | 2% |

Also superseded: `true_union_prep` (preprocessed full union) scored 0.7960, beating raw
Mordred's 0.8054 — but still losing to plain ECFP's 0.7732.

## 3. UMA

The 128-d ℓ=0 mean-pooled UMA embedding is **rank-deficient**: 5 components carry 90% of the
variance over 99,932 molecules, 20 carry 99%, and one dimension correlates 0.649 with atom
count. Richer pooling barely helps — all nine spherical-harmonic blocks at 1152 dims still
give rank 8 at 90%, 40 at 99%. This is intrinsic to averaging per-atom environments into a
molecule vector, not an artefact of the extraction hook.

Consequently UMA distils *easily* (R² 0.912 from ECFP alone, versus RDKit-96's 0.913), which
means it holds little ECFP does not. ChemPFN's own `c0_xgb_v2.json` agrees: UMA is
complementary to ECFP alone (aqsoldb 1.8297 → 1.5122) but **redundant with descriptors** —
`ecfp+desc` (1.3687) beats `ecfp+desc+uma` (1.3895).

That is a better explanation for ROADMAP's "UMA harmful on cliffs" than the charge bug: 128
dense tokens carrying ~5 dimensions of size information dilute the smooth channel for
almost no return.

## 4. Cost structure

Measured on an M4 Pro, single core.

| channel | µs/mol |
|---|---|
| `MolFromSmiles` (= graph construction) | 54 |
| ECFP-2048 counts | 29 |
| ErG-315 | 102 |
| RDKit-96 | 976 |
| Mordred 2D (1613) | 72,700 |
| UMA-small + conformer | ~93,000 (6 workers) |

Mordred by family — cost is diffuse, and the highest-signal family is nearly the cheapest:

| family | n | µs/mol | % | µs/descriptor |
|---|---|---|---|---|
| Chi | 56 | 13,742 | 18.9% | 245 |
| PathCount | 21 | 9,984 | 13.7% | 475 |
| InformationContent | 42 | 6,008 | 8.3% | 143 |
| BCUT | 24 | 5,457 | 7.5% | 227 |
| **Autocorrelation** | **606** | **4,823** | **6.6%** | **8** |
| EState | 316 | 2,683 | 3.7% | 8 |

Top 5 families are only 56% of runtime; top 11 are 85%. "Drop the slow ones" fails for the
same reason "keep the important ones" failed — both are diffuse.

## 5. Corrections to the record

* **`benchmark_results.md` cost error.** RDKit-96 is recorded as "~µs/mol". Measured: **976
  µs/mol** — 34× ECFP, 11.3 core-days per billion. The cherry-pick's "~10⁴× the RDKit-96
  channel" is really **9.2×**; its other two reasons stand.
* **Two records disagree.** `benchmark_results.md` lists MoleculeACE `desc` = 0.868;
  `mordred_cherry_validate.json` says 0.8399. Independent rerun: 0.8399.
* **`uma_embed.py` charge bug.** Hardcodes `charge=0, spin=1` ("toy molecules"). UMA takes
  charge and spin as global conditioning inputs, so a carboxylate embedded as neutral gets
  the energy surface of a species that does not exist. 4.71% of this PubChem selection is
  charged; on ChEMBL at recorded protonation states, more. Fixed in `uma_100k.py`.
* **Storage.** Gzipped SMILES is 7.3 bytes/molecule; packed ECFP-2048 is 256. Storing
  fingerprints instead of SMILES costs **35× more**, not less.
* **`.venv-uma` has RDKit 2026.3.5** against the pinned 2025.9.2. Harmless for the embed path
  (it reads pickled coordinates and never calls RDKit) but a hazard for anything keying a
  cache on canonical SMILES.

## 6. What this leaves

ECFP alone is the strongest single channel on cliffs. Descriptors earn their place only on
smooth physicochemical endpoints, and RDKit-96 is enough there. Mordred and UMA are both
dispensable. The locked v1 scheme already reflects this.

Open, if anyone wants it: whether a **large-radius unfolded** ECFP beats radius-2/2048 (our
input could not see past 2 bonds while Mordred's autocorrelations run to lag 8), and whether
CheMeleon's frozen embedding beats plain ECFP on these folds — it was absent from the Praski
benchmark, so nobody has checked.

## 7. ECFP's actual blind spots (2026-08-24)

Prompted by MiniMol's positional encodings. MiniMol v1 conditions on the Laplacian
(`laplacian_eigvec` + `laplacian_eigval`, 8 each) and the random-walk matrix
(`rw_return_probs`, k=16) — two matrices, three feature blocks, no shortest-path matrix.

Tested rather than assumed, and most of the folklore did not survive:

| claim | verdict |
|---|---|
| E/Z stereo invisible to ECFP | **false** — Tanimoto 0.50; `includeChirality` picks up bond stereo |
| enantiomers invisible to ECFP | **false** — 4–10 bits differ on alanine, ibuprofen, thalidomide, butane-2,3-diol |
| anthracene vs phenanthrene indistinguishable | **false** — Tanimoto 0.56; angular fusion makes a distinct bay-region environment |
| ring fusion topology invisible | **false**, in the information sense |

The enantiomer test was initially run on `C[C@H](O)[C@H](O)C` vs `C[C@@H](O)[C@@H](O)C`,
which are (1S,3R) and (1R,3S) — both **meso**, the same compound, identical canonical SMILES.
Identical fingerprints were correct. Check that test molecules are what you think they are
before concluding a representation is blind.

**What did survive, and it is the more useful half:** the descriptor union is *completely*
stereo-blind. Zero of 217 RDKit and zero of 1,613 Mordred descriptors move across any
enantiomer pair. Two consequences for the architecture:

* The predicted block **cannot** carry stereo information and **need not** — ECFP carries it
  directly. The two menu channels are non-overlapping on this axis by construction, and a
  stereo failure in the surrogate is impossible rather than merely unlikely.
* Any stereo descriptor we add would be pure conditioning, not new information.

**The framing was wrong.** By the data-processing inequality nothing derived from the graph
is unavailable to ECFP, which is near-injective in practice. The confirmed mechanism in this
project was never information — it was degree reduction (Chi +0.126 from quantities ECFP
fully determines). The generator is therefore not "what is missing" but:

> **what is present in ECFP but not in a form XGBoost can construct?**

That follows from the algorithm. ECFP is (1) a bag of *unary* local features, (2) *hashed
presence counts*, (3) built by WL colour refinement. Each blocks a class of quantity:

* **Pairwise sums** `Σ_ij f(i,j)` need quadratic interaction across 2048 bits — why the 419
  autocorrelations pay, and the shape every new block should take.
* **Ratios** — trees cannot divide. `Kf/n`, `FractionCSP3`, densities.
* **Cyclic patterns** — WL-1 distinguishes exactly what *tree* homomorphism counts do
  (Dell/Grohe/Rattan), so any pattern containing a cycle is outside colour refinement.
  Available free from the `A^1..A^8` already built for `WalkCount`.

Anthracene vs phenanthrene is the illustration: ECFP separates them, but `Kf` = 163.85 vs
160.78 is one ordered scalar on a physical axis, while `NumAromaticRings`, `RingCount`,
`NumBridgeheadAtoms`, `Kappa2` and `LabuteASA` are all bit-identical between them.

### Candidate blocks

| candidate | coverage today | cost | case |
|---|---|---|---|
| path multiplicity (resistance) | zero in both libraries | ~15 µs (C++) | **built**, `resistance.py` |
| conjugated-system topology | **zero**, verified | 37 µs Python → ~5 µs C++ | **built**, `conjugation.py` |
| cyclic homomorphism counts | ~80% already in `resistance.py`'s RWSE features | free from `A^k` | **held in reserve** — see below |
| signed-parity stereo | ECFP has it; descriptors do not | 38 µs Python → ~10 µs C++ | **built**, `stereo.py` — conditioning only |
| symmetry / automorphism classes | proxied by `FpDensityMorgan1/2/3` | one canonical rank | weakest |

### `resistance.py`

Built around `Δ_ij = d_ij − Ω_ij` (shortest path minus resistance distance), which is
**identically zero for every acyclic molecule** — so it cannot proxy for size, weight or atom
counts, and measures path multiplicity alone. 77 features. Self-test: dodecane 0.00,
naphthalene 42.57 vs two separate benzenes 19.00, chrysene 110.22.

Not the spectral scalars: all 110 SpMax/SpDiam/SpAD/SpMAD/VE*/VR* descriptors over the
adjacency, Barysz, distance and detour matrices were killed by the |ρ|≥0.99 dedupe, **zero
survivors**. The cover runs in ascending cost order, so each had a *cheaper* non-spectral
correlate. Collapsing n×n to one number discards whatever was new.

Cost: 173 µs/mol reference Python, of which only **69 µs is arithmetic** — the rest was
`np.quantile` on 20-element arrays and five masked passes over the pair list. C++ ≈ 70 µs,
≈ 15 µs once L⁺ is restricted to biconnected components (Ω = d across every bridge, so only
ring systems need inverting).

**Falsifiable prediction:** the block must help on fused/polycyclic datasets and do nothing on
acyclic or monocyclic ones. If it helps uniformly it is leaking size information and should be
rejected. Test is `ecfp+core` vs `ecfp+core+resistance` over the 34 datasets already cached in
`bench.npz` — not blocked on the LOCKED registry, since this is a compute-side question that
needs no surrogate model.

### Sequencing on cyclic homomorphism counts

Deliberately not built. The cheap 80% is already present: `RingCount` contributes 49 CORE
columns (by size, aromatic/aliphatic/saturated, carbo/heterocycle, spiro, bridgehead),
Mordred's surviving `WalkCount` columns give trace(A^k) for k=5,7 plus TSRW10, and
`resistance.py` already pools per-atom diag(S^k) for k in {2,3,4,6,8,12,16} -- which *is* the
homomorphism-count idea, in normalised per-atom form rather than as a trace.

What would be strictly additional is exact *cycle* counts rather than closed-walk counts, via
inclusion-exclusion on the walk counts. Those are genuinely beyond WL-1 in a way walks are
not, but the formulas get unpleasant past k=5 and carry real implementation risk. Correct
order: let the 34-dataset run report feature importance on the RW block first. If those
features rank, exact cycle counts are the obvious follow-up; if they are dead, the whole
family is answered and no formula needs writing.

### Combined budget

| | reference Python | arithmetic floor / C++ with shared D |
|---|---|---|
| ECFP + CORE | 88 µs | 88 µs |
| resistance | 173 | ~70 (≈15 with L⁺ on biconnected components only) |
| stereo | 38 | ~10 |
| conjugation | 37 | ~5 |
| **total** | **336 µs** | **~175 µs** |

The Python total breaches the 10×-ECFP rule (290 µs); the C++ total does not, with room left.
Most of the Python overhead is small-array numpy glue and a distance matrix each block
recomputes independently -- both disappear when the primitives are shared, which is the whole
argument for the C++ core.
