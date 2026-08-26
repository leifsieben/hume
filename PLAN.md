# Universal molecular encoder — status and roadmap

One parse, shared primitives, four selectable blocks. C++ core, Python API, packaged.

**Where we are:** the design is validated end-to-end on 100k molecules. The descriptor
surrogate works, the winning model is the *simplest* one, and it ships as BLAS with no
runtime dependency. What remains is engineering, scale, and honest OOD evaluation.

---

## 1. The headline result

Downstream on MoleculeNet (4 regression sets, scaffold 5-fold, untuned XGBoost). Baseline
`ecfp+core` = 1.0393; ceiling `ecfp+core+exact` = 0.9644, i.e. the predict block is worth
**−0.0749 RMSE (7.2%)**.

| model | reconstruction R² | downstream | % of ceiling | inference |
|---|---|---|---|---|
| **linquad** | 0.9755 | **0.9612** | **104%** | **2 GEMMs, BLAS-only** |
| ridge | 0.9754 | 0.9709 | 91% | 1 GEMM, BLAS-only |
| MLP | **0.9843** | 0.9828 | 75% | 3 GEMMs + runtime dep |
| Π-net (deg 3) | 0.9841 | 0.9877 | 69% | 3 GEMMs + runtime dep |
| GNN (broken, see §5) | 0.2379 | 1.0472 | −11% | message passing |

### NO MODEL HAS BEEN SELECTED. n=4 CANNOT RANK THESE MODELS.

Per-dataset deltas versus baseline behind those means:

| | ESOL | FreeSolv | Lipophilicity | BACE | mean | **sd** |
|---|---|---|---|---|---|---|
| exact | −0.093 | −0.127 | −0.079 | −0.001 | −0.075 | 0.054 |
| ridge | −0.036 | −0.182 | −0.069 | +0.013 | −0.069 | 0.083 |
| linquad | −0.052 | **−0.221** | −0.060 | +0.019 | −0.078 | **0.101** |
| Π-net | −0.056 | −0.083 | −0.072 | +0.004 | −0.052 | 0.039 |
| MLP | −0.058 | −0.107 | −0.072 | +0.011 | −0.057 | 0.050 |

**Every standard deviation exceeds every between-model difference.** At n=4 the standard
error is sd/2, so linquad's is ±0.05 against an apparent 0.022 edge over the MLP.

**Drop FreeSolv (642 molecules, smallest and noisiest) and the ranking inverts**: exact
−0.057, Π-net −0.041, MLP −0.040, linquad −0.031, ridge −0.031. linquad goes first to last and
the "beats the true descriptors" claim disappears. The entire result was one dataset.

**Mean rank reverses the mean-RMSE ordering.** Per dataset (1 = best of 5):

| model | ESOL | FreeSolv | Lipophilicity | BACE | mean rank |
|---|---|---|---|---|---|
| Π-net | 2 | 4 | **1** | **1** | **2.00** |
| MLP | **1** | 3 | 2 | 2 | **2.00** |
| ridge | 4 | 2 | 3 | 4 | 3.25 |
| linquad | 3 | **1** | 4 | 5 | 3.25 |
| GNN | 5 | 5 | 5 | 3 | 4.50 |

By rank the flexible models tie for first and the linear ones tie for third — the opposite of
the RMSE means, and back in line with the reconstruction ranking.

**Second methodological error, independent of small n: averaging raw RMSE across datasets
with incommensurable scales.** FreeSolv's RMSEs run 1.6–1.8, Lipophilicity's 0.68, so a mean
over raw RMSE weights FreeSolv ~2.5× more. **Aggregate by rank, or normalise per dataset
before averaging.** Applies to every multi-dataset comparison in this project.

Also visible per dataset: **BACE behaves like a cliff set** — every model is worse than
baseline and `exact` barely helps (0.7784 → 0.7776), so the descriptor block has nothing to
offer there; this is not surrogate failure. And **FreeSolv is the only place anything beats
`exact`**, on 642 molecules, so the "predicted descriptors are better-conditioned features"
claim is withdrawn until it reappears on a larger set.

What the n=4 run *does* support, weakly: the predict block is worth roughly −0.05 to −0.075
RMSE, and every surrogate recovers a substantial fraction of it. Which surrogate is
unresolved.

**Standing lesson, now three times over.** Reconstruction R² has anti-correlated with
downstream value in every test: a PCA projection retaining 95% of variance gave 26% of the
gain; an ECFP surrogate at R² 0.949 gave 2%; here the two best-reconstructing models placed
3rd and 4th. **Reconstruction loss is a training objective, never a selection criterion.**

---

## 2. Settled by measurement — do not revisit

| | |
|---|---|
| descriptor union | 1,830 (217 RDKit + 1,613 Mordred 2D) → 1,275 usable → **865 after dedup** at \|ρ\|≥0.99 |
| **compute** | **639** from CORE primitives (ring, `A`, atom props, `D`, Labute) — 59 µs, 10.8 desc/µs |
| **predict** | **226** needing EState (242 µs), Crippen (90), eigenvalues (40), Gasteiger (17), TPSA (9), or path enumeration |
| EState | **predict** — 242 µs (4× the whole core tier) buys −0.0065 RMSE. Rejected on ratio |
| surrogate input | ECFP + the 639 core descriptors (free at inference) |
| compression | **dead** — PCA to 64/128/256 retained 21–26% of downstream gain; the projection, not the preprocessing, was responsible |
| task dependence | core **hurts** cliffs (+0.024 MoleculeACE), **helps** smooth (−0.077 MoleculeNet). Blocks must be selectable |

Cost basis (M4 Pro, single core, 26 heavy atoms): parse 48.5 µs · ECFP 28.6 · core primitives
58.6 (24 of which is a Python atom loop) · surrogate 5.7 · full Mordred 67,000.

---

## 3. Roadmap

### Phase 1 — validate at scale (next)

0. **Model selection on the full ChemPFN LOCKED suite — this gates everything else.** Train
   all five (ridge, linquad, Π-net, MLP, Chemprop) and rank on **downstream gain across the
   whole locked suite**, not a 4-dataset subset. Report per-dataset deltas with spread, not
   just means, and a win-rate; means over small n are what produced the retracted
   recommendation above.
   *Blocker:* the LOCKED registry is not in the ChemPFN repo — `data/` is empty locally and
   the datasets appear to live in the S3 lake (`chempfn-data-use1-…`). Need the concrete
   dataset list and loader before this can run. Named in `docs/REBUILD_CHECKLIST.md:133`:
   QMugs, FartDB, CycPeptMPDB, RAscore, `ld50_zhu`, `vdss_lombardo`, plus FS-Mol test assays;
   largest task is 39,121 molecules.
1. **Chemprop GNN, fairly.** Install Chemprop, pass the **639 core descriptors as
   `--features-path`** so it gets the same information the MLP gets. Current GNN result is
   invalid (§5). AWS GPU, 1–2 h — the only component here that genuinely wants a GPU.
2. **1M-molecule training run.** MaxMin-diverse selection over PubChem/ChEMBL. Compute inputs
   (~90 µs/mol) and *only* the 226 targets (~30 ms/mol). Parallelise `assemble.py` first —
   it is single-threaded today and would take an hour instead of minutes.
3. **OOD evaluation.** Report R² and downstream RMSE **binned by Tanimoto distance to the
   nearest training molecule**, plus held-out scaffolds and non-druglike chemistry. This is
   the generalisation question and it is currently unanswered.
4. **Finish the downstream matrix.** MoleculeACE (30 cliff sets) was cut off; only MoleculeNet
   completed. Needed to confirm the 104% result is not specific to 4 datasets.
5. **Degree-4 Π-net** — the ladder was still climbing (degree 3 added more than degree 2). One
   extra layer. Cheap to test, though given §1 it may well lose downstream anyway.

### Phase 2 — the C++ core

Realistic headroom is **~1.7× single-threaded, ~3× with proper threading** — capped because
parse + ECFP (77 µs) are already C++ and are 70% of the optimised total.

| target | now | after | how |
|---|---|---|---|
| atom property vectors | 24.3 µs | ~1 | compile-time table indexed by atomic number |
| descriptor derivation | ~50 µs | ~15 | primitives computed once and passed down; measured 2.5× on a 7-descriptor group |
| cheap Mordred families | 8.6 µs/desc | ~2 | Autocorrelation/EState/RingCount/Barysz are masked quadratic forms on one `D` |
| **parallel efficiency** | **4.4× on 12 cores** | **~8×** | C++ thread pool, no GIL, no process overhead — **the single biggest lever** |
| parse | 48.5 µs | 48.5 | irreducible; partial sanitisation is slower *and* 12% inexact |

Implementation rules (all measured, not guessed):
* `CalcCrippenDescriptors`, never a sum of `_CalcCrippenContribs` — the latter omits implicit
  hydrogens and is wrong by ~1.8 logP units.
* **Size-bucket before batching.** Naive padding measured *slower* than per-molecule
  (309 vs 210 µs); sorting by atom count gave 115 µs at 1.10× waste.
* Exploit ECFP sparsity in the surrogate's first layer (~43 of 2048 non-zero → ~2.5× on the
  model). Low priority: the model is already 5% of the pipeline.
* Verify bit-exactness against RDKit/Mordred on ≥10⁴ molecules per descriptor.

Optional: pre-parsed binary molecules cut parse 48.5 → 7.5 µs at 400 bytes/mol (55× gzipped
SMILES). Worth it only for repeat screening of a fixed library.

### Phase 3 — package

```python
emb = molenc.encode(smiles, blocks=["ecfp", "fast", "predicted"], n_jobs=32)
```

* pybind11, zero-copy numpy out, streaming interface, thread pool with `n_jobs`
* `blocks`: `ecfp` (2048) · `fast` (639 exact) · `predicted` (226 surrogate) · `exact` (226
  computed — a genuinely better RDKit+Mordred wrapper: one parse, shared primitives, 865
  deduplicated columns instead of 1,830 raw, ~2× faster)
* surrogate weights embedded in the wheel; BLAS only, no PyTorch
* **document the task dependence**: cliff/bioactivity work wants `["ecfp"]`; physicochemical
  and ADMET work wants `["ecfp","fast","predicted"]`. This is a measured effect, not a
  preference.
* wheels for macOS/Linux × arm64/x86-64, pinned RDKit (cached features are keyed on its
  version — canonicalisation changes silently invalidate caches)

### Phase 4 — 3D and UMA

Conformers cost ~140 ms/mol and do not scale, so the question is whether 3D Mordred and MLIP
embeddings can be **predicted from 2D** — a conformer-ensemble expectation, so expect
irreducible error and a lower ceiling than the 2D surrogate.

Carry forward: UMA's ℓ=0 mean-pooled embedding is rank-deficient (5 components carry 90% of
variance over 99,932 molecules; all nine spherical-harmonic blocks at 1,152 dims still give
rank 8), and is redundant with descriptors in ChemPFN's c0 ablations (`ecfp+desc` 1.3687 beats
`ecfp+desc+uma` 1.3895). Any 3D tier must use **sum pooling, not mean** — mean-pooling is what
destroyed the rank. 99,932 embeddings and matched conformers exist under `data/uma100k/`.

---

## 4. Scale target

| | 32 CPUs (now) | after Phase 2 |
|---|---|---|
| 1B molecules, `["ecfp","fast","predicted"]` | ~7 h | **~2–3 h** |

No GPU needed — RDKit parsing and descriptor computation are CPU-only and ~90% of the work.

---

## 5. Known gaps and invalid results

1. **The GNN result (R² 0.238, −11% downstream) is invalid.** It trained 12 epochs with
   unstable loss (0.19 → 0.48 between epochs 8 and 9) *and* was given only the raw graph while
   every other model received ECFP + 639 core descriptors. Not a comparison of
   representations. Redo with Chemprop and matched features.
2. **MoleculeACE downstream incomplete** — only MoleculeNet (4 datasets) finished. The
   headline 104% rests on 4 datasets.
3. **100k, not 1M.** Model *ranking* should be stable, absolute numbers will move.
4. **OOD untested.** No Tanimoto-distance binning yet.
5. **`assemble.py` is single-threaded** — fine at 100k, an hour at 1M.
6. **A methodological trap that already bit once:** inputs must be cleaned as carefully as
   targets. 159 core columns contain NaN and the block spans ~1e5; unhandled, ridge returns
   NaN (visibly) and the neural models converge badly (silently). There is now an assertion.

---

# Roadmap update — 2026-08-24

Three decisions taken, all on **measured C++ numbers** rather than projections from Python
timings. Build: `clang++ -O3 -march=native -std=c++17 cpp/bench.cpp -framework Accelerate`,
run over 3,000 benchmark molecules (mean 30.4 heavy atoms).

| primitive | C++ measured | Python reference | decision |
|---|---|---|---|
| Chi paths k<=7 (14 desc) | **6.82 us** | RDKit 408 / Mordred 10,328 | **-> CORE** |
| cycle enumeration k<=8 | **2.18 us** | 119 | keep |
| resistance `L+` (dposv) | **5.80 us** | 31 | keep |
| normalised Laplacian spectrum (dsyevd) | **20.79 us** | 9 | **CUT** |

## 1. The Laplacian eigendecomposition is cut

`resistance.py` 77 -> 65 features. `dsyevd` cost 3.6x the resistance solve and bought the
least defensible features in the block (Fiedler value, spectral density) in a project where
all 110 of Mordred's spectral scalars had already been killed by the |rho|>=0.99 dedupe.

Also dropped: the planned biconnected-component optimisation for `L+`. At 5.80 us on the whole
molecule it is unnecessary, so that work comes off the C++ roadmap entirely.

## 2. Chi and PathCount move PREDICT -> CORE

65 columns, so PREDICT drops **226 -> 161**. `blocks.py` updated; `chi.py` written.

The original classification was wrong for a reason worth remembering: Chi was assigned to
PREDICT because Mordred's Chi family costs 10,328 us. That is **Mordred being Python**, not
paths being expensive. RDKit proves it from the inside -- `Chi0n`/`Chi1n`/`Chi0v` are C++ and
cost 0.17-0.23 us, while `Chi2n`/`Chi3n`/`Chi4n` fall back to `GraphDescriptors.py` and cost
35/56/87 us.

Why paths are affordable at all: molecular graphs are sparse and nearly tree-like, so path
counts barely grow with length -- 60.6 paths of length 3, 102.8 of length 5, 144.5 of length 7
per molecule. The combinatorial explosion that makes path enumeration frightening in general
graph theory does not occur in drug-like chemistry.

**This is the highest-value change made so far.** Chi is the strongest predicted family we have
measured (+0.126 downstream), so this removes the largest single source of surrogate error from
the default path.

**Kappa is a likely follow-up**, not yet taken: it needs only `HallKierAlpha` (0.22 us in
RDKit's C++) plus path counts we now compute anyway. Unmeasured, so it stays in PREDICT.

## 3. Exactness verification — `verify.py`

Once a descriptor moves into CORE, *we* compute it and correctness stops being RDKit's problem.
The failure mode is silent, so it needs a gate rather than a spot check.

`verify.py` checks every self-computed descriptor against its reference implementation, reports
per-descriptor exact-match fractions and worst absolute deviation, prints the SMILES of the
first divergence, and **exits nonzero** below 99.99% so it can gate a build.

    python verify.py                 # benchmark set, quick
    python verify.py --corpus        # full 1M training corpus  <- the real gate
    python verify.py --corpus -n 50000

**Run on the full 1M corpus, not a sample.** A 2,000-molecule sample cannot see a failure mode
affecting one chemotype in ten thousand, and the corpus is what we actually featurise.

It earned its place immediately, catching four bugs on the day it was written -- three in the
new code and one in itself:

1. valence delta computed as `TotalValence - numH` instead of Kier-Hall `nOuterElecs - nH`.
   Correct on every hydrocarbon, wrong on every N and O.
2. the row correction `/(Z - nV - 1)` applied to first-row atoms too. Correct on C/N/O *by
   accident* (the denominator is 1 there), wrong on S, Cl, Br.
3. ring closures assigned to order L-1 instead of L. RDKit emits an L-atom cycle as an
   (L+1)-entry walk -- `(0,1,2,0)` for cyclopropane -- with the product over the L *distinct*
   atoms. Verified against cyclopropane Chi3n = 0.7071^3, cyclobutane Chi4n = 0.7071^4.
4. the harness itself asserted `len(GetSymmSSSR) == cyclomatic number` and failed on 0.5% of
   the benchmark. The *check* was wrong: GetSymmSSSR is symmetrised and deliberately returns
   more rings than the cyclomatic number for symmetric fused systems. Relaxed to `>=`.

Every one of these matches perfectly on the easy cases and diverges only on a subset, which is
exactly why sampling and eyeballing would not have found them.

### Current exactness status

| descriptor | exact | note |
|---|---|---|
| `Chi0n`..`Chi4n`, `Chi0v`..`Chi4v` | **100.00%** | verified against RDKit, 3,000 molecules |
| `C3`, `C4`, `C5` | 100.00% | closed form vs independent enumeration |
| SSSR >= cyclomatic | 100.00% | invariant holds |

**RESOLVED — `Chi4` (2026-08-25).** The failure was a wrong product convention for lollipops
(a simple tail feeding into a cycle). RDKit's rule, derived from its values rather than assumed:

    bare cycle of L atoms          -> order L,   product over its L distinct atoms
    cycle L + tail of t atoms      -> order L+t, product over those atoms
                                                 AND the attachment atom counted TWICE

The doubling is the part that is not guessable. `C1CCC1` Chi4n = 0.25 is the plain 4-atom ring
product, but `CC1CC1` Chi4n = 0.16667 requires the attachment doubled. Cross-checked on
`CCC1CC1`, where ring x inv[tail] x inv[attach] plus the two open 5-atom paths gives
0.11785 + 0.40825 = 0.52610, matching RDKit exactly. Fix is one term; all ten Chi checks are
now 100.00% on 3,000 molecules and `verify.py` exits clean.

**Cost note.** The 6.82 us C++ figure timed path enumeration *without* ring closures. Closures
need the cycle enumerator too, which is 2.18 us -- but `cycles.py` pays that already, so Chi's
*marginal* cost stays 6.82 us provided the two share one enumeration pass in the C++ core.
`cpp/bench.cpp` does not yet implement the closure term; it timed the dominant loop only.

## 4. Revised cost and dimension

| | µs | note |
|---|---|---|
| parse + ECFP | 77 | measured, already C++ |
| CORE descriptors | 59 | measured |
| Chi paths k<=7 | 6.8 | **C++ measured** |
| resistance `L+` (no spectrum) | ~8 | 5.80 measured + binning |
| cycles k<=8 | 2.2 | **C++ measured** |
| conjugation + stereo | ~4 | estimated, not yet in C++ |
| **total** | **~157** | against the 290 µs ceiling (10x ECFP) |

    ecfp 2048 + core 639 + new blocks 145 + chi 65 = 2897 computed
                                       + predicted 161 = 3058 total

(new blocks: resistance 65 after the cut, cycles 33, conjugation 24, stereo 23)

## 5. Next actions, in order

1. ~~Finish or cap `Chi4`~~ — **done 2026-08-25**, all ten checks exact.
2. **Re-run `assemble.py`** — the CORE/PREDICT split changed, so both X and Y matrices are
   stale. Parallelise it first; it is single-threaded and would take an hour at 1M.
3. **`python verify.py --corpus`** on the full 1M once assemble has run. This is the gate.
4. **The 34-dataset block run.** Arms: `ecfp+core` baseline, plus each of resistance / cycles /
   conjugation / stereo / chi, plus all together. Not blocked on the LOCKED registry — it is a
   compute-side question needing no surrogate. Each block carries a falsifiable prediction:
   resistance must help on fused/polycyclic sets and not on acyclic ones; stereo must track the
   per-dataset stereo fraction (0.0% on CHEMBL4203_Ki and ESOL, 91.8% on CHEMBL4616_EC50);
   cycles must help where cycle redundancy > 1 (12.2% of molecules).
5. **Port `conjugation` and `stereo` to C++** so their ~4 µs is measured rather than assumed.
6. **Model selection on the full LOCKED suite** — still Phase 1 step 0, still blocked on the
   registry location.

---

# 1M corpus + evaluation plan — 2026-08-25

Everything below is sized from measurements taken today on this machine, not estimates.

## Measured basis

| fact | value | source |
|---|---|---|
| machine | 12 cores, 25.7 GB RAM, 187 GB free | `sysctl` |
| pool | 10M PubChem SMILES, 467 MB | `pubchem_10M.smi` |
| scaffolds in a 3M scan | 415,126 distinct; **122,789 with >=3 members** | cached pickle |
| Mordred, all 1,613 | 80,088 us/mol -> **22.25 core-h** per 1M | timed, 120 mols |
| Mordred, **only the 685 we need** | 35,811 us/mol -> **9.95 core-h** per 1M | timed, 100 mols |
| RDKit 217 | 4,168 us/mol -> **1.16 core-h** per 1M | timed |
| our five blocks (Python) | ~517 us/mol -> **0.14 core-h** per 1M | timed |
| **pool/benchmark overlap** | **0.93%** — ~93,000 of the 10M | 200k-line scan |

Two consequences that change the job before it starts.

**Restrict Mordred to the 685 columns still needed.** Chi moved to CORE and we compute it
ourselves; a further 928 Mordred columns were killed by the dedupe or are simply unused.
Computing all 1,613 wastes 12.3 core-hours per million molecules for nothing. This alone takes
the corpus job from ~2.3 h to ~1.2 h wall on 10 workers.

**Exclude benchmark molecules from the corpus.** 0.93% of the pool canonicalises into the
benchmark's 42,390 unique structures, so a naive 1M draw would contain ~9,300 of them. The
surrogate would then be trained to reproduce descriptors for molecules it is later scored on,
and the downstream comparison would flatter it. Exclusion is a hash-set lookup at selection
time and costs nothing. **This is the single most important line in this plan** -- it is the
kind of leak that produces a good result and a wrong conclusion.

## Phase 0 — block run (unblocked, start now, independent of everything else)

Does each new block earn its cost? Compute-side only, so it needs no surrogate and no LOCKED
registry. Runs against the 34 datasets already cached in `bench.npz`.

Arms: `ecfp+core` baseline, then `+resistance`, `+cycles`, `+conjugation`, `+stereo`, `+chi`,
then all five together. Each block carries a prediction that can kill it:

| block | must help on | must **not** help on |
|---|---|---|
| resistance | fused/polycyclic sets | acyclic and monocyclic sets |
| cycles | the 12.2% with cycle redundancy > 1 | ring-basis-only molecules |
| stereo | high-stereo sets (CHEMBL4616_EC50, 91.8%) | CHEMBL4203_Ki and ESOL, both 0.0% |
| conjugation | extended-pi sets | saturated aliphatics |

Flat gain across a block's negative control means it is proxying size, and it gets cut.

## Phase 1 — the 1M corpus (the long pole, ~3 h wall total)

1. **Rescan the pool for scaffolds.** The cached scan covers 3M and yields 122,789 scaffolds
   with >=3 members -- enough for 368k molecules at depth 3, not 1M. The full 10M should give
   roughly 400k+ such scaffolds. ~10 min on 10 workers.
2. **Select 1M**, scaffold-stratified at **depth 3**, excluding the 42,390 benchmark
   structures. Depth 3 rather than depth 1 for the reason established earlier: depth 1
   maximises breadth but yields zero within-scaffold pairs, and the benchmarks are congeneric
   series, so the model would never see how descriptors move under a single-atom substitution.
3. **Parallelise `assemble.py`.** Currently single-threaded; at 1M that is 11 core-hours in one
   process. Shard by 50k, `multiprocessing` over 10 workers, resumable.
4. **Compute.** Restricted Mordred (685) + RDKit 217 + the five blocks. **~1.2 h wall.**
5. **Storage.** Store ECFP as `uint8` counts, not `float32`: 2.0 GB instead of 8.2 GB. Total
   ~6 GB sharded, comfortable against 187 GB free and loadable in 25 GB RAM.
6. **`python verify.py --corpus`** — the exactness gate, on all 1M. Nonzero exit blocks Phase 2.

Result: `X` = 2,897 columns (2048 ECFP + 639 CORE + 145 new blocks + 65 Chi),
`Y` = 161 PREDICT columns.

## Phase 2 — surrogate training (local, ~overnight)

Train all five candidates on the 1M: ridge, linear+quadratic, Pi-net, MLP, GNN.
Report reconstruction R^2 **per family**, not pooled -- pooled R^2 has now failed as a proxy
three separate times in this project.

Do not train before Phase 1 finishes. The CORE/PREDICT split changed today (Y went 226 -> 161,
X gained 65), so both current matrices are stale and any model trained now would be ranked on
a target set we have already abandoned.

## Phase 3 — evaluation suite

Four arms, so the surrogate is measured against both floor and ceiling:

    ecfp                      reference
    ecfp + core               the shipped `fast` path
    ecfp + core + predicted   the product
    ecfp + core + exact       the ceiling (true descriptors)

`predicted` vs `exact` is the surrogate's real cost. `exact` vs `core` is what the whole
predict block is worth -- if that gap is zero, no surrogate is needed at all and the project
simplifies enormously. Report per-dataset and by rank, never as a mean of raw RMSE across
incommensurable scales.

Add **OOD binning by Tanimoto distance** from each benchmark molecule to its nearest corpus
neighbour. With benchmark structures excluded from the corpus this is now a clean measurement
rather than a contaminated one.

## Phase 4 — model selection (still blocked)

Needs the ChemPFN LOCKED registry, which is not in the repo. Phases 0-3 all run without it;
only the final commitment to one model requires it.

## Risks

- **Scaffold supply.** If the 10M scan yields fewer than ~333k scaffolds with >=3 members, the
  corpus falls short of 1M at depth 3. Fallback is to mix depth 3 for rich scaffolds with
  depth 1 for the tail, which the existing selector already supports.
- **`.venv-uma` RDKit is 2026.3.5 against the pinned 2025.9.2.** Harmless for the embed path
  but a hazard for anything keying a cache on canonical SMILES. Use the pinned environment for
  corpus work.
- **Mordred NaN rate at 1M.** 159 CORE columns already carry NaN at 100k. Worth reporting the
  per-column NaN rate from Phase 1 before training rather than discovering it in the loss.

## Corpus build — findings during implementation (2026-08-25)

**Acyclic molecules would have been reduced to three.** Every acyclic molecule has the *empty*
Murcko scaffold, so plain depth-3 stratification takes three of them out of the whole 10M pool.
Measured: the pool is **15.9% acyclic** while the benchmark is **0.2% acyclic**, so heavy
down-weighting is correct -- but three molecules is not down-weighting, it is deletion. The
selector now takes an explicit `--acyclic-frac` (default 2%): an order of magnitude above
benchmark prevalence so the surrogate has plenty of acyclic examples, far below pool prevalence
so the corpus is not spent on the easy case. Several blocks are identically zero on acyclic
input, so there is genuinely less to learn there.

This is the class of bug that produces a corpus which looks fine and a model that fails on a
whole chemotype, and it came from a default in someone else's scaffold definition rather than
from any decision we made.

**Scaffold supply is the open risk.** A 100k sample gives 3,322 scaffolds with >=3 members; the
cached 3M scan gives 122,789. Discovery has diminishing returns, so the 10M scan will not simply
scale to 400k+ -- if it lands short, the corpus falls back to the singleton tail, which the
selector already handles but which weakens the within-scaffold-pair rationale for depth 3.
Report the actual number before running `compute`.

## Pipeline as built

    corpus.py scaffolds --workers 6    scan 10M, cache scaffold + canonical SMILES   ~8 min
    corpus.py select --n 1000000       stratify, exclude benchmark, acyclic quota    ~3 min
    corpus.py compute --workers 6      restricted Mordred + RDKit + 5 blocks         ~1.5 h
    corpus.py pack                     slice into X / Y, report per-column NaN rate  ~10 min
    verify.py --corpus                 exactness gate over all 1M                    ~30 min

Sharded at 50k and resumable throughout: `compute` skips shards that already exist, so an
interrupted run costs at most one shard. Writes only under `data/corpus1m/`; the pool and the
`ChemTFM_OLD` venv are read-only.

`pack` keeps the output sharded rather than building one array -- X at 1M x 2,897 float32 is
11.6 GB, which does not belong in one allocation alongside a training process in 25 GB of RAM.
ECFP is stored as `uint8` counts (2.0 GB rather than 8.2 GB); radius-2 environment counts never
approach 255 in molecules of <=60 atoms.

## Corpus build — actual results (2026-08-25)

    scan     9,999,994 usable molecules, 1,227,938 distinct scaffolds        380 s, 6 workers
    select   380,847 scaffolds with >=3 members -> 1,000,000 at 3.06/scaffold
             20,000 acyclic (2.0%), 10,103 benchmark structures excluded
    compute  20 shards x 50k, 5 workers, ~1.4 h projected from a 300-molecule smoke test

**Scaffold supply risk closed.** 380,847 scaffolds with >=3 members against the ~327k needed,
so this is a true depth-3 corpus with no singleton fallback. The 3M cache had suggested this
might fall short; scaffold discovery scaled better from 3M to 10M than diminishing returns
implied (415,126 -> 1,227,938 distinct).

**Correction to the leakage estimate.** The 0.93% overlap figure came from the first 200,000
pool lines. Over the full 10M the true rate is **0.10%** -- 10,103 molecules. The pool is
ordered rather than shuffled, so its head is not a random sample and the estimate was ~9x too
high. The exclusion still earned its place: 10,103 molecules would otherwise have been trained
on and then scored against.

**Smoke test before committing 1.4 h.** One 300-molecule shard end to end: all five blocks
100% finite, Mordred 98.6% finite, RDKit 100%, 25.3 ms/mol.

### Two correctness measures taken during the build

**One code path for train and bench.** `bench.npz` was built under the old CORE/PREDICT split
and its columns no longer line up. Rather than patch it, `corpus.py compute --bench` routes the
benchmark through the identical code. Two separately-maintained builders is how column
misalignment happens -- the split changed today, and any independent benchmark builder would
already be one convention behind. `models.py` now asserts the widths match rather than trusting
them.

**`verify.py --corpus` re-derives from SMILES.** It reads `selected.txt`, not the packed
shards, so a bug in the shard writer is *caught* rather than reproduced. Verifying the output
of the thing under test against itself is not verification.

---

# Phase 0 result — no descriptors are dropped (2026-08-25)

34 datasets, XGBoost, 5-fold scaffold CV, every arm carrying ECFP.

| arm | mean Δ | median Δ | wins | Wilcoxon p |
|---|---|---|---|---|
| resistance | +0.42% | +0.59% | 11/34 | 0.033 |
| cycles | +0.09% | −0.03% | 18/34 | 0.906 |
| conjugation | −0.24% | −0.25% | 19/34 | 0.317 |
| stereo | −0.00% | +0.18% | 15/34 | 0.761 |
| chi | +0.24% | +0.08% | 15/34 | 0.278 |
| all five | −0.02% | −0.12% | 18/34 | 0.853 |

(positive = higher RMSE = worse)

## Decision: keep everything. Nothing is cut.

I recommended cutting all four new blocks. That recommendation was **wrong on two counts** and
is retracted.

**1. The "dilution" mechanism I proposed does not exist.** I explained resistance's poor
showing as 65 noisy columns diluting the feature space. Regressing mean effect on columns
added: **+0.0000% per column, r = 0.00, p = 0.993**. The `all` arm adds 171 columns and lands
at −0.02%. Width is measurably free. This is the regime one would expect: median dataset is
1,228 molecules against 2,687 features, and **27 of 34 have fewer molecules than features**, so
another 171 columns changes nothing.

**2. Nothing was significant.** Six arms were tested. Resistance's p = 0.033 becomes **p =
0.198** under Bonferroni. There is no significant result in this experiment in either
direction, including the one I called harmful.

## The power bound is the honest headline

Per-dataset SD is 1.0–1.4%, so at n = 34 this design could only detect effects **larger than
~0.5–0.65%** (80% power, two-sided). Every observed effect is below that threshold.

| block | per-dataset SD | min detectable effect | observed |
|---|---|---|---|
| resistance | 1.20% | 0.58% | +0.42% |
| cycles | 0.99% | 0.48% | +0.09% |
| conjugation | 1.37% | 0.66% | −0.24% |
| stereo | 1.33% | 0.64% | −0.00% |
| chi | 1.03% | 0.50% | +0.24% |

The defensible claim is therefore **"these blocks change downstream RMSE by less than ~0.6%,
if at all"** -- not "they do not help". A null result at this power justifies "not worth paying
much for"; it never justifies "useless". A reviewer would be right to reject the stronger
claim, and erring toward inclusion is the better-supported position.

## What survives as a real finding

The **stereo negative control failed in the informative direction.** Predicted: gain tracks
each dataset's stereo fraction. Measured correlation **+0.260** -- wrong sign. The ten datasets
with almost no stereo (8%) gained −0.89%; the ten richest in stereo (62%) gained +0.16%. The
block "helps" where its features are near-zero, which is the signature of noise. That is a
failed falsifiable prediction, which is stronger evidence than a null -- but it bounds the
stereo block specifically, not the others.

**Two of four controls were degenerate, and that is a design error.** All 34 datasets are >50%
conjugated and none is >50% acyclic, so those controls had zero datasets on one side and tested
nothing. The thresholds (0.5 / 0.1) were chosen without checking the covariate distributions
first, despite the stereo distribution having been measured the same day. The one control built
with real variance is the one that produced an answer.

## Where the tradeoff actually lives

Compute, not accuracy. In C++ the four blocks cost roughly resistance 8 + cycles 2.2 +
conjugation 5 + stereo 10 = **~25 µs**, about 16% of the ~157 µs budget, for no measurable
accuracy change. That is the honest case for a menu entry -- throughput, not correctness:
*inclusion is measurably harmless; a tight set exists for users who need speed.*

## HUME menu, updated

    ecfp        2048    ~29 µs    always
    fast         796   ~159 µs    CORE + all four blocks + chi   <- default, nothing dropped
    predicted    161     ~0 µs    surrogate forward pass         <- default
    exact        161    ~27 ms    deduplicated RDKit + Mordred   <- opt-in
    tight          ?         ?    empirically-selected subset    <- determined on LOCKED

**Note on the dedupe.** The 1,830 -> 865 reduction stays. That was removing columns with
|rho| >= 0.99 against a cheaper column -- near-identical information by construction -- which is
a different justification from "it did not help downstream". If the full 1,830 is wanted, that
is a separate call and should be made explicitly.

## Determining the tight set later — fix the power first

Repeating this design on LOCKED hits the same wall. Two changes, both cheap:

* **Pair on folds, not datasets.** 34 datasets x 5 folds = 170 paired observations rather than
  34. Cuts the standard error ~2.2x, taking MDE from ~0.6% to ~0.27%.
* **Repeat CV across seeds.** Most of the 1.0–1.4% SD is split noise, not real between-dataset
  variation.

## LOCKED suite — entry point found

    PYTHONPATH=. CHEMPFN_DATA_ROOT=~/chempfn-data python scripts/suite_inventory.py --check-s3

Run from `~/VSCode/ChemPFN`. The script exists and `~/chempfn-data` is present, so the registry
is reachable. This unblocks Phase 4 model selection, which has been blocked since the split
change.

---

# TODO / roadmap as of 2026-08-25

## Now (corpus in flight)

1. `corpus.py compute` — 1M descriptors, 15/20 shards
2. `corpus.py compute --bench` — same code path for the 56k benchmark (~5 min)
3. `corpus.py pack` and `pack --bench` — slice to X/Y, report per-column NaN (~15 min)
4. `verify.py --corpus` — exactness gate on all 1M (~30 min). Blocks everything downstream.

## Then

5. **Train all five surrogates on 1M**: ridge, linear+quadratic, Pi-net, MLP, GNN. Report
   reconstruction R^2 **per family**, never pooled — pooled R^2 has failed as a proxy three
   times in this project.
6. **Phase 3 eval, four arms**: `ecfp` / `ecfp+core` / `ecfp+core+predicted` / `ecfp+core+exact`.
   Report **split by endpoint type**, not pooled — see the scoping note below.
7. **OOD binning** by Tanimoto distance to nearest corpus neighbour. Clean now that benchmark
   structures are excluded from the corpus.

## Tight set (added 2026-08-25)

8. **Determine the `tight` menu entry on the full LOCKED suite.** A speed option, not an
   accuracy one: inclusion is measurably harmless, so `tight` exists for users who need
   throughput, and that is the framing that makes it defensible.
   * Pair on **folds, not datasets** — 34 x 5 = 170 paired observations rather than 34, cutting
     SE ~2.2x and MDE from ~0.6% to ~0.27%. CLIMB's redundancy figure already does exactly
     this and states why: both arms see the same folds, so fold difficulty cancels in the
     difference, and the marginal SD of either arm alone runs 2-8x the lift.
   * Repeat CV across seeds — most of the 1.0-1.4% per-dataset SD is split noise.
   * Include a **positive control** (descriptors over ECFP) so a flat bar is readable.

## LOCKED suite — unblocks model selection

9. From `~/VSCode/ChemPFN`:

       PYTHONPATH=. CHEMPFN_DATA_ROOT=~/chempfn-data python scripts/suite_inventory.py --check-s3

   Script and `~/chempfn-data` both confirmed present.
10. **Model selection** on the full suite. Rank by downstream gain with per-dataset spread and
    win rates. Never a mean of raw RMSE across incommensurable scales.

## Engineering

11. C++ core: port the blocks that survive, share one distance matrix and one enumeration pass
    between `chi` and `cycles`. Measured so far: Chi 6.82 us, cycles 2.18, resistance L+ 5.80.
12. Package as HUME — pybind11, pip, docs, the five-entry menu.
13. **SI: bit-identical verification on 1M** (`verify.py --corpus`). Already built; this is the
    claim in the draft that "the descriptors we compute are bit-identical with RDKit/Mordred".

## Paper figures — design notes

**Figure A (resolution).** Keep CLIMB's skeleton: 10 chemical edits x 100 pairs, response
normalised by *that model's own* shift under a matched-MW substitution, positive control at
1.00 by construction, notation-only edits (SMILES enumeration, Kekulisation) that must score 0.
HUME's addition is an arm for the **predicted** block: does a neural surrogate blur the
resolution ECFP provides? Architecturally it cannot — ECFP passes through uncompressed, so
resolution is preserved by construction, which is a real advantage over any bottleneck model
where the loss is irreversible. Show it anyway.

**Figure B (redundancy).** Reframe from CLIMB's. CLIMB asked "is our DL model redundant?"; HUME
asks "does the predicted block recover the non-redundant part?" Three bars over an ECFP
baseline: **true descriptors / HUME-predicted / best DL embedding**. If predicted ~ true >> DL,
that is the paper in one panel. Keep CLIMB's positive control — without it a flat bar cannot be
distinguished from an assay that detects nothing.

**Scoping caveat, to state before a reviewer does.** CLIMB measures descriptors gaining 12.8%
on QM7 and 6.2% on Tox21 but **flat on BACE and HIV**. Our own data agrees: on MoleculeACE,
ECFP alone (0.7732) beats ECFP+desc+Mordred (0.8054). The descriptor lift is concentrated on
physicochemical and QM endpoints and is roughly absent on bioactivity and cliffs. HUME is
therefore not "a better embedding for everything" but "descriptor-quality performance at
fingerprint cost, on the task classes where descriptors matter". Pooling across endpoint types
would hide this.

## Paper figure work — model roster settled (2026-08-25)

Full methodology in `FIGURES.md`. Roster decisions recorded there; summary of what changed:

* **GROVER dropped, Uni-Mol swapped in.** No local GROVER copy and legacy torch/DGL pins.
  Uni-Mol is the model a referee names, it is 3D so it probes a channel ECFP and descriptors
  both lack, and `unimol_tools` is pip-installable and maintained.
* **MiniMol does not fill the graph-transformer slot** — its shipped config is
  `layer_type: 'pyg:gine'`, message passing rather than attention, verified in
  `minimol/ckpts/minimol_v1/config.yaml`. Graphormer is the optional fix if the architecture
  class must be represented explicitly.
* **Kulik code exists**: `github.com/hjkgrp/SpectralScore`. No reimplementation needed. It
  needs conformers, so it sits in UMA's cost bracket, not ECFP's — that must appear on the
  Figure D cost axis rather than in a footnote.
* **UMA added as an optional third 3D channel** — the conformer pipeline and the charge/spin
  fix already exist from earlier work, so marginal cost is near zero.

### Outstanding work for the figures

| item | effort |
|---|---|
| SMI-TED download (IBM `materials.smi-ted`, HuggingFace) | small |
| Uni-Mol install + embed | small |
| Kulik SpectralScore clone + embed (needs conformers) | medium |
| Graphormer (optional) | medium |
| Build the 13,000 edit pairs for Figure A | medium |
| Figure B fold assignment, shared across all arms, seeded and recorded first | small |
| `results/figures/` provenance tree + `backup_figures.sh` (written, untested — no data yet) | done |

## Corpus complete — two data-quality findings (2026-08-25)

`corpus.py compute` finished: 20 shards, 1,000,000 molecules, 2.6 GB, **133 min** on 5 workers.

### 1. Size-range mismatch between corpus and benchmark

The corpus is filtered to 5-60 heavy atoms. The benchmark is not:

    benchmark heavy atoms: median 30, max 316
      <5                      136  (0.24%)
      5-60 (corpus range)  55,401  (98.58%)
      >60                     660  (1.17%)

So the surrogate is **extrapolating on 1.17% of benchmark molecules** and has never seen a
molecule above 60 heavy atoms. Two consequences worth reporting rather than absorbing:

* OOD analysis must bin by size as well as by Tanimoto distance. A large-molecule failure would
  otherwise be averaged into the Tanimoto bins.
* If the >60 tail turns out to matter, the fix is to raise `MAX_ATOMS` and recompute, not to
  quietly drop those benchmark rows.

### 2. `Ipc` overflows float32 — and it produces +inf, not NaN

RDKit's `Ipc` is an information-content descriptor whose magnitude grows super-exponentially
with molecule size. On the benchmark's large tail it exceeds float32 range and casts to **+inf**.
It does not appear in the corpus at all (0 inf in 217 RDKit and 685 Mordred columns) precisely
because of the 60-atom cap, so this surfaced only when the benchmark was built through the same
code path -- which is the argument for having built them through one code path.

**This exposed a bug in `corpus_data.drop_dead_targets`**, now fixed: it tested `isnan`, which
is False for inf, so infinite target columns would have passed the filter untouched and
destroyed a masked MSE silently. It now tests `~isfinite`. `pack`'s NaN report already used
`~isfinite` and was correct.

`Ipc` is a PREDICT target (`rdkit_Kappa_Ipc`). Options if it survives the filter: use RDKit's
`AvgIpc`, which exists specifically to avoid this, or log-transform. Decide when `pack` reports
its non-finite rate.

### NaN structure at 1M (shard 0, 50k molecules)

    RDKit  : 12 of 217 columns carry NaN
    Mordred: 202 of 685 columns carry NaN, worst 7.21%

Worst offenders are the lag-8 autocorrelations (`GATS8c`, `AATSC8c`, `MATS8c`, `AATS8v`,
`GATS8v`), which are undefined for molecules whose topological diameter is below 8 -- expected,
structural, and not a defect. Consistent with the 159 NaN-carrying columns seen at 100k.

## Fingerprint radius — r=3 measured (2026-08-25)

Cost on this machine (400 benchmark molecules, 2048 bins, counts, chirality on; verify was
using 6 cores concurrently so absolute numbers are slightly inflated, ratios are not):

    r=2   38.8 us/mol   52.3 on-bits
    r=3   47.9 us/mol   72.3 on-bits      +9.1 us  (+23% on the fingerprint, +4.5% of pipeline)
    r=4   57.5 us/mol   85.5 on-bits
    MolFromSmiles alone 76.2 us/mol       <- the fingerprint is not the expensive part

Ring-size resolution, Tanimoto (lower = better resolved):

    pair                            r=2      r=3      r=4
    bare cyclopentane/cyclohexane  0.833    0.842    0.842     <- r=3 marginally WORSE
    + phenyl                       0.917    0.867    0.870
    + amide arm                    0.893    0.824    0.824
    drug-like acetanilide          0.935    0.893    0.902
    cyclohexyl vs cycloheptyl      0.939    0.871    0.843
    pyrrolidine vs piperidine      0.880    0.793    0.793

**r=3 wins on substituted rings, loses on bare ones.** With only 5-6 atoms the radius-3 ball
wraps the entire ring and the extra environments are redundant; on a substituted ring it does
not. The mechanism: r=2 spans a 5-atom ball, which does not close a 5- or 6-ring for most
atoms, so ring size is invisible to the environment hash. r=3 spans 7 atoms, which closes both.
r=4 adds nothing further except on 7-rings.

**Collision caveat.** r=3 gives 72 on-bits against 52. Matching r=2's collision density needs
4096 bins, which is +19.6 us and doubles fingerprint storage rather than +9.1 us.

**This is not purely a featurisation choice.** The surrogate takes ECFP+core as *input*, so
changing the radius changes its input space and requires retraining. Two routes:

* **Retrain on r=3** -- +9.1 us total. The 1M corpus needs only its ECFP block rebuilt and
  re-packed; descriptors are radius-independent, so this is a cheap re-pack, not a 133-minute
  re-compute.
* **Keep the r=2 surrogate, append r=3 as an extra output block** -- no retraining, but pays
  both fingerprints (+48 us) and carries 4096 extra columns.

**Recommendation:** expose `radius` as a HUME menu parameter, default 2 for comparability with
the literature, and carry r=3 as an arm in Figure A. The existing resolution data already has
`r3fp` beating ECFP4 on stereo (normalised count 0.59 vs 0.29), so it strengthens that figure
rather than complicating it.

### Decision: two parallel surrogates, r=2 first (2026-08-25)

HUME ships **two models**, selected by the `radius` menu parameter:

    HUME-r2   ECFP4 (r=2, 2048)  + core + blocks -> predicted    <- default; all decisions made here
    HUME-r3   r3fp  (r=3, 4096)  + core + blocks -> predicted    <- add-on, trained after

**Sequence is r=2 first.** Every methodological decision -- model class among the five
candidates, tight-set membership, the four-arm evaluation, LOCKED model selection -- is settled
on r=2, which is also the literature-standard baseline and therefore the comparable one. r=3 is
then trained as an add-on against the already-settled choices, not co-developed with them.
Deciding twice in parallel would double the multiple-comparison burden on every question we
have already shown ourselves to be underpowered for.

**What the second model actually costs.** Only the ECFP block of X differs; the 699 core
columns, the 171 block columns and all 166 targets are radius-independent and are reused
verbatim. So HUME-r3 needs a **re-pack, not a re-compute**: regenerate the fingerprint block
from `selected.txt` and re-slice, rather than another 133-minute Mordred run. Budget roughly
15-20 minutes of featurisation plus one training run.

**Open sub-decision, to settle when r=3 is trained:** 2048 bins or 4096. r=3 produces 72 on-bits
against r=2's 52, so 2048 raises the collision rate ~40%; 4096 restores parity at +19.6 us and
double the fingerprint storage. Measure both rather than assuming -- the collision cost may be
invisible downstream given 27 of 34 benchmark datasets already have fewer molecules than
features.

### Counts vs binary for r3fp (2026-08-25)

**Use counts**, for convention parity with r=2 and because generation cost is identical -- but
the count channel carries *less* at r=3 than at r=2, which is the opposite of the naive
expectation.

    representation   on-bits   % of on-bits with count>1   mean count
    r=2, 2048           52.6                        26.8%         1.64
    r=3, 2048           72.9                        21.0%         1.48
    r=3, 4096           73.8                        20.1%         1.46

(1,500 benchmark molecules.)

Mechanism: at r=2 many atoms share the same 5-atom environment -- every CH2 in a chain -- so
multiplicity is what distinguishes chain lengths. At r=3 the 7-atom environments are already
close to unique per atom, so multiplicity is largely encoded in *having distinct bits* and the
count channel does less work. **r=3's advantage comes from ring closure, not from counts.**

Binarisation cost, four hand-picked pairs (Tanimoto, positive = binary resolves worse):

    pair                       r=2       r=3
    drug-like C5/C6         +0.0295   +0.0019
    pyrrolidine/piperidine  +0.0575   +0.0069
    homologation CH2        +0.0015   -0.0378
    sym vs asym dimethyl    +0.0096   -0.0217

Consistently harmful at r=2; a wash at r=3. **Four pairs is anecdote, not measurement** -- it
illustrates the mechanism and should not be quoted as evidence. The on-bit statistics above are
the solid part.

### FIXED: two fingerprint conventions, each with its own proxy model (2026-08-25)

Settled. Not to be revisited without a measurement.

    ECFP    Morgan r=2, 2048 bins, COUNTS, chirality on   -> proxy model P2   (default)
    R3cFP   Morgan r=3,    ? bins, COUNTS, chirality on   -> proxy model P3   (alternative)

Both are **count** fingerprints, so the two conventions differ in exactly one variable --
radius -- and any downstream difference is attributable. Both get their own proxy (surrogate)
model, since the fingerprint is the proxy's input and changing it changes the input space.

Everything downstream of the fingerprint is shared and radius-independent: the 699 core
columns, the 171 block columns and all 166 prediction targets are identical between the two.
Standing up R3cFP is therefore a **re-pack plus one training run** (~15-20 min featurisation),
not a second 133-minute descriptor build.

Decision order stays: **all methodology settled on ECFP first** -- proxy model class,
tight set, four-arm evaluation, LOCKED selection -- then R3cFP trained against those fixed
choices. Co-developing them would double the multiple-comparison burden on exactly the
questions Phase 0 showed us to be underpowered for.

Open when R3cFP is built: 2048 vs 4096 bins (72 on-bits vs 52 raises collisions ~40% at 2048;
4096 restores parity at +19.6 us and doubles fingerprint storage). Measure, do not assume.

## verify.py --corpus on 1,000,000 molecules — first run FAILED, cause found (2026-08-25)

    8 of 14 checks below 99.99%: Chi0v, Chi1v, Chi2n, Chi2v, Chi3n, Chi3v, Chi4n, Chi4v
    612 of 1,000,000 molecules (0.061%). Every one carried explicit deuterium.
    cycles:C3/C4/C5 100.000%. sssr_ge_cyclomatic 99.999% (6 molecules, unexamined).

**This is exactly what the 1M gate was built for.** The same checks passed 100.000% on 3,000
molecules. A 0.061% failure rate is invisible at that scale and would have shipped.

### Cause: RDKit's Chi variants disagree with each other about explicit hydrogens

Not deuterium as such. RDKit reaches the atoms by three different routes:

* `Chi0n`, `Chi1n` -> `_nVal` over `mol.GetAtoms()`         -> **include** explicit H
* `Chi0v`          -> `_hkDeltas`, which has `skipHs=1`     -> **exclude** them
* `Chi2n` and above -> `FindAllPathsOfLengthN(useHs=False)` -> **exclude** them

On a C14-perdeuterated chain the Chi0n/Chi0v gap is exactly 29 x 1/sqrt(1), one term per
deuterium, and the Chi1n/Chi1v gap exactly 29 x 0.5, one per C-D bond. Our implementation
included explicit H uniformly, so it matched the first route and diverged from the other two.

### Fix: normalise the input rather than reproduce the inconsistency

`chi.strip_explicit_h()` removes explicit H and D before featurising. Verified: with no explicit
H in the graph, all three of RDKit's routes compute the same thing, and ours agrees with every
variant simultaneously -- all ten Chi checks match on the previously-failing molecules.

`verify.py` applies the **same** normalisation to the reference side. Comparing our normalised
value against RDKit's un-normalised one would report a difference of convention as a difference
of correctness.

**Isotope information is not lost from HUME.** ECFP carries isotope in its atom invariant; only
Chi, a pure connectivity index with no isotope term, is normalised.

### Follow-ups

* Re-run in progress (`logs/verify_corpus2.log`).
* `cycles:sssr_ge_cyclomatic` fails on 6 of 1,000,000 -- unexamined, below the 99.99% threshold
  but worth a look. Explicit H are terminal so they cannot sit in a cycle; cycle *counts* are
  unaffected, but per-atom statistics and `frac_in_cycle` are diluted by H atoms.
* `resistance.py`, `conjugation.py` and `stereo.py` walk the same un-stripped adjacency and
  should take the same normalisation for consistency, even though none of them currently fails.

## verify.py --corpus PASSED on 1,000,000 molecules (2026-08-25)

    1,000,000 molecules in 1,475 s (6 workers)

    chi:Chi0n .. chi:Chi4v   exact 1,000,000/1,000,000 (100.000%)  worst 0.00e+00
    cycles:C3, C4, C5        exact 1,000,000/1,000,000 (100.000%)  worst 0.00e+00
    cycles:sssr_ge_cyclomatic      999,994/1,000,000 ( 99.999%)
    EXIT 0 -- all descriptors exact

**Worst deviation 0.00e+00 on every Chi variant: bit-identical, not merely within tolerance.**
This is the SI claim in the draft ("the descriptors we compute are bit-identical with RDKit or
Mordred implementation on 1M SMILES"), now demonstrated rather than asserted. Artifact:
`data/surrogate/verify_corpus.json`.

### Isotope handling — checked, not assumed

`chi.strip_explicit_h()` normalises explicit H and D away *inside* Chi only. Verified that
nothing is lost from HUME overall:

* **ECFP carries isotope**: 5-7 bits differ between protiated and deuterated forms across three
  test pairs (ethane/-d6, acetanilide/-d3, trimethylamine/-d3).
* **`corpus.py` computes ECFP on the original molecule** -- the block modules receive the same
  unmodified `m`, and only `chi.featurize` strips internally.

The design argument for stripping, which is stronger than the compatibility one: `CC` and
`[H]C([H])([H])C([H])([H])[H]` give **identical** Chi (2.0000) because RDKit folds plain
explicit H into implicit counts, while ethane-d6 gives 7.0000. So "including H" would not have
encoded *isotope* -- it would have encoded *notation style*. That is exactly the failure mode
Figure A criticises learned embeddings for, and it has no place in a connectivity index that
carries no isotope term.

### The one remaining non-exact check, explained

`cycles:sssr_ge_cyclomatic` fails on 6 of 1,000,000 (0.0006%), all **highly bridged polycyclic
cages** -- adamantane-like frameworks and cyclophanes:

    O[C@@]12CC3C4C5CC6C4C3C1C(NCc1ccccc1)C6C52        SSSR=6  cyclomatic=7
    CC(=O)c1c2c3cc4c1CCCc1cc(c(cc1CCC4)CCC3)CCC2      SSSR=4  cyclomatic=5

`GetSymmSSSR` under-reports a ring basis for bridged cages. This is an RDKit limitation, not a
defect in our code, and the check flagged it correctly. Only effect on us: `C_redundancy =
total/sssr` is marginally inflated for those six molecules. Passed the 99.99% threshold.

---

# Four-arm downstream — the scoping claim, measured (2026-08-26)

34 datasets, 8 arms, XGBoost 5-fold scaffold CV. `data/surrogate/pick_model.json`.

## Pooled, this looks like a failure. It is not.

    ecfp                  rank 2.88   vs core  -1.71%   beats core 28/34
    ecfp+core+exact       rank 3.91   vs core  -1.43%   beats core 26/34
    ecfp+core             rank 5.82   vs core  +0.00%   beats core  0/34

Plain ECFP ranks best and `ecfp+core` ranks worst. Taken at face value that says the whole
descriptor programme is counterproductive. **Split by endpoint type it inverts.**

## Split by suite — change vs ECFP alone, negative = better

    MoleculeNet (n=4, physicochemical)     MoleculeACE (n=30, activity cliffs)
      core          -3.85%                   core          +2.73%   p<0.001
      exact        -10.74%                   exact         +2.10%   p<0.001
      mlp          -11.36%                   every proxy   +2.2..+2.5%  p<0.001
      gnn          -10.62%
      pinet         -9.91%

**Descriptors help ~11% on physicochemical endpoints and hurt ~2% on cliffs.** The pooled
result was 30 cliff datasets swamping 4 physicochemical ones. This is exactly why the plan
requires reporting split by endpoint type and never pooling: pooling here produces the
conclusion "descriptors do not work", which is false in both directions at once.

This is the paper's scoping claim with numbers behind it: **descriptor-quality performance at
fingerprint cost, on the task classes where descriptors matter** — and honest reporting that on
activity cliffs ECFP alone is already optimal and descriptors are a net cost.

## Reconstruction anti-correlates with downstream value. Again.

    model     reconstruction R2      % of core->exact gap recovered (median)
    pinet          0.9837  (best)          40.3%  (worst)
    mlp            0.9815                  62.6%
    gnn            0.9712  (worst)         86.5%  (best)
    ridge          0.9505                  59.4%
    linquad        0.9593                  45.5%

Near-perfect inversion, and the **fourth** time reconstruction has failed as a proxy for
downstream value in this project (after Gate 1's 95%-variance/26%-gain, the PCA arms, and the
Mordred cherry-pick). Selecting on reconstruction R2 would have shipped the worst model.

Note the GNN here is the **undertrained** h=128/depth=4/12-epoch run that was nearly written
off. A converged h=512/depth=8/60-epoch run is training.

## What cannot be concluded yet

MoleculeNet is **n=4**; no p-value is computable and MLP edging `exact` (-11.36 vs -10.74) is
not a result at that sample size. MoleculeACE is registered as **contaminated** in the ChemPFN
lake (it was ChemTFM's Phase-1 selection metric). **Model selection waits for the 28-dataset DEV
grid**, which spans physicochemical, QM, ADMET, toxicity and synthesis and is neither
contaminated nor cliff-dominated.

---

# DEV grid — Figure B panel 1 (2026-08-26)

28 DEV datasets, 6 arms, XGBoost 5-fold scaffold CV, 840 rows.
`results/figures/figB/`. Ran on a c7i.8xlarge, 14-way parallel, ~1 h.

## Overall — rank and median improvement over ECFP alone

    arm           rank   beats ecfp   median % vs ecfp
    ecfp_desc     2.11        24/28            +2.65
    hume_exact    2.54        23/28            +1.97
    hume          3.29        22/28            +1.64
    desc          3.54        21/28            +1.85
    ecfp          4.50           --             0.00
    r3cfp         5.04         8/28            -0.97

## By endpoint group — median % improvement over ECFP alone

    arm            QM    physchem   toxicity   ADMET   other
    ecfp_desc   +25.8      +17.0       +2.0     +1.1    +2.0
    hume_exact  +24.7      +16.2       +2.0     +1.4    +1.9
    hume        +22.1      +16.4       +0.6     +1.1    +1.3
    r3cfp        -1.3       -2.0       -0.3     -0.7    -0.5

    group sizes: ADMET 13, QM 5, physchem 4, toxicity 3, other 3

## Three findings

**1. The descriptor lift is concentrated, not uniform.** ~25% on QM, ~16% on physicochemical,
~1-2% on ADMET and toxicity. Combined with the MoleculeACE result (-2%, p<0.001), the picture
is a single spectrum: descriptors are worth a great deal on physicochemical and quantum
endpoints, nothing on ADMET/toxicity, and are a net cost on activity cliffs. **HUME's claim is
scoped to the left half of that spectrum and must be reported that way.**

**2. HUME recovers 86-97% of the descriptor benefit.** physchem 16.36 vs 16.96 (96%),
QM 22.13 vs 25.81 (86%), ADMET 1.05 vs 1.08 (97%). This is with the **pinet** proxy, which was
the *worst* of five at downstream recovery in the four-arm test (40.3% of ceiling). These are
a floor; the mlp or a converged gnn proxy should do better.

**3. r=3 is WORSE than r=2**, on 20 of 28 datasets, median -0.97%, and negative in every
endpoint group. This contradicts the plan's R3cFP decision, which was taken on a four-pair
Tanimoto argument -- exactly the kind of anecdotal evidence this project keeps having to
retract. **R3cFP should not get a second proxy model until this is understood.** The ring-size
resolution argument may still be right in principle while being irrelevant downstream, which
would be the fifth time a good mechanism failed to predict downstream value here.

## Caveat on metric mixing

The 28 sets mix rmse (regression), auroc (binary) and accuracy (multiclass). Percentages are
direction-corrected but are not strictly commensurable across metric types; the per-group
medians mitigate this, and rank is metric-free. Report rank alongside any percentage.
