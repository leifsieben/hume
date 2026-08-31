# HUME — API design record

*Why each flag is what it is. The reference documentation is `README.md` and the docstrings;
this file is the argument behind them.*

**Status: superseded in part.** This began as a specification written before the package
existed, and much of it has now been built, measured, and in three places decided differently.
Read it for the reasoning, not for the signature. Where the two disagree, `molhume.featurize`
is right and this file is history.

What changed between the specification and the implementation, and why:

| this file said | what shipped | why |
| --- | --- | --- |
| `featurize` returns `(fp, X, names)` | returns one `ndarray` | the names are identical on every call for a given set of flags, so returning them is something every caller unpacks and discards. `feature_names(**flags)` gives them on demand, the way sklearn's `get_feature_names_out` does. |
| the fingerprint is part of the output | `fingerprint=False` by default, bits appended last | this is a descriptor library and the bits are not descriptors. Folding 2,048 of them into the same float64 matrix makes the default output mostly fingerprint by width and nearly triples its memory. Appending rather than prepending keeps descriptor column indices stable when the flag changes. |
| `descriptors` selects a mode | `additional_descriptors` (bool) plus `columns` (names) | two orthogonal questions — *whose* descriptors, and *which* ones — were one parameter. They are combined with AND. |
| `standardize` has no default and raises | defaults to a sentinel that behaves as `"none"` and warns once | section 3 below is still the right argument; only the enforcement changed. Raising makes the library unusable in a one-liner; a silent default makes the wrong molecule the quiet case. The sentinel keeps the decision explicit — passing `"none"` yourself is silent, omitting it is not — without making the first call fail. |

Two things section 5 promised and the implementation kept: the column set is versioned, and the
emitted set is exactly `ALL_COLUMNS`. Section 8's open items are settled — see `CHANGELOG.md`.

Every default here is justified by a measurement in this repo, or it is marked OPEN. A default
chosen by argument alone is a bug waiting to be discovered by a user.

---

## 1. The call

```python
import hume

X, columns = hume.featurize(
    smiles,                        # iterable of SMILES strings
    standardize   = ...,           # REQUIRED -- no default, see section 3
    descriptors   = "exact",       # "exact" | "fast" | "none"
    hume_blocks   = True,          # our five blocks
    radius        = 3,
    bits          = 2048,
    counts        = False,         # binary fingerprint
    spec          = "v1",          # frozen column set, see section 5
    dtype         = "float32",
    on_error      = "nan",         # "nan" | "skip" | "raise"
    n_jobs        = -1,
)
```

Returns the feature matrix and the column list. **The column list is part of the return value,
not an attribute to be looked up later** — a matrix whose column meanings live somewhere else is
how a representation silently becomes two different objects.

---

## 2. Flags

| flag | default | justification |
|---|---|---|
| `standardize` | **none — REQUIRED** | section 3 |
| `descriptors` | `"exact"` | section 4 |
| `hume_blocks` | `True` | the five blocks are ~44 µs/mol of the ~195 µs C++ total; cheap enough that off-by-default would be a false economy |
| `radius` | `3` | section 2.1 |
| `bits` | `2048` | at r=3, folding to 2048 loses **2.3%** of distinct features to collisions (66.3 distinct/molecule, 64.7 on-bits). r=2 loses 1.8%. Widening buys 2% of features for 2× the memory |
| `counts` | `False` (binary) | **OPEN** — section 2.2 |
| `dtype` | `"float32"` | what every downstream consumer already uses (`bench.npz` is float32; our own modules return float32). float64 doubles memory for precision nothing reads |
| `on_error` | `"nan"` | a billion-molecule run must not die on one bad SMILES, and must not silently shorten its output either — `"skip"` desynchronises the caller's row indexing unless they are careful |
| `n_jobs` | `-1` | |
| `spec` | `"v1"` | section 5 |

### 2.1 Why radius 3

Measured, not inherited:

* **Sensitivity** (`figures/build/fig_a.csv`, matched molecular pairs): r=3 responds more strongly
  than r=2 on **6 of 10** chemistry-edit modes, median ratio **1.129**. Largest gains are
  topological — ring contraction 1.53×, halogen swap 1.23×, regioisomer 1.20×, scaffold hop 1.16×
  — which is what a wider radius should see. It loses slightly on protonation (0.93) and E/Z
  flip (0.95).
* **Both notation controls stay at exactly 0.000.** The extra radius buys chemistry without
  buying notation artefacts, which is the failure that would have disqualified it.
* **Cost is 1.28×**: 26.25 µs/mol against 20.57, i.e. **+5.68 µs** — about 2% of the pipeline.
  Measured back-to-back and alternating on a contended machine, so the ratio is trustworthy and
  the absolutes are not.

**Why r=2 became the standard, and why that reasoning does not apply here.** ECFP4 means
*diameter* 4, i.e. radius 2 (Rogers & Hahn, 2010). It became the default on **similarity-search**
benchmarks, where a larger radius makes features more specific, so two similar molecules share
fewer bits and retrieval recall drops. That is an argument about lookup in large tables. HUME is a
supervised representation fed to a gradient-boosted head, where the specificity is the point. The
convention is sound and simply answers a different question.

### 2.2 OPEN — binary or counts

Counts strictly dominate binary in information, so if binary wins it is a *generalisation* effect
at small n, not an information one. Cost is not the deciding factor: counts are 1.08× binary
(28.37 vs 26.25 µs/mol at r=3).

This is an arm in Figure C (`hume_counts`) rather than a default chosen by argument. **Do not
freeze `counts` until that measurement lands.**

---

## 3. `standardize` is required, and deliberately has no default

```python
standardize="none"              # compute on the molecule exactly as supplied
standardize="largest_fragment"  # keep only the largest connected component
```

**There is no safe default, so the library does not pick one.** The two options give different
numbers for any multi-fragment input, and which is correct depends on what the caller is doing:

* `"none"` is what **RDKit and Mordred do**. It keeps descriptor values bit-identical to those
  references, which is the claim the entire C++ verification effort exists to support. But for a
  salt, graph descriptors are dominated by the "no path between fragments" sentinel (RDKit writes
  1e8) and carry little chemical meaning.
* `"largest_fragment"` is conventional cheminformatics preprocessing and usually what you want for
  property prediction. Its values **will not match** RDKit or Mordred computed on the original
  SMILES.

The C++ handles disconnected input properly either way — per-component resistance solve,
unreachable-pair sentinels — and `cpp/hard.smi` carries 10,000 salts and mixtures specifically to
exercise it.

**Implementation note that makes the error message possible.** `standardize` must NOT be a plain
required keyword argument. Python would raise `TypeError` at the call boundary, before anything is
parsed, and the message could then not say how much the choice affects *this* input. Use a
sentinel default, parse first, then raise. See section 6.

### Explicit hydrogens need no flag

`Chem.RemoveHs` is a **no-op on molecules parsed from SMILES** — verified 20,000/20,000 on the
hard corpus. Sanitisation has already folded away everything foldable; the 219 survivors are
isotopic, charged or H2, which `RemoveHs` keeps too. So "no explicit H" is the default for free.

Going further would be wrong: stripping `[2H]` changes the molecule, and it changes the answer.
Deuterated ethanol has `Chi2n` 0.258199 against ethanol's 0.316228 under RDKit's convention.

---

## 4. `descriptors` — three modes, possibly two

```python
descriptors="exact"   # compute everything, in C++
descriptors="fast"    # predict the slow families with a surrogate    <- OPEN, may not ship
descriptors="none"    # fingerprint only
```

`"exact"` reuses this project's own vocabulary — `ALL EXACT` is the verification verdict — so it
needs no explanation. The internal `core` / `predict` / `blocks` terms stay inside `blocks.py`;
they are implementation structure and were never a good user-facing vocabulary.

**OPEN: whether `"fast"` should exist at all.** Two measurements decide it, both in flight:

1. **The core/predict boundary is stale.** It was drawn on cost, and the costs have since changed
   by 3–90× — BCUT2D 306 → 110 µs, Crippen 138 → 1.53 µs, EState and Kappa now native, the
   autocorrelation weights 541 µs of Python → 0. Roughly **72 of the 166 predicted columns** are
   RDKit families we may now compute for almost nothing. A surrogate trained to approximate those
   is strictly worse than computing them: it costs accuracy for no speed.
2. **Ridge inference is not free.** Predicting `d_out` columns from `d_in` features is
   `d_in × d_out` multiply-adds; from ECFP-2048 + 639 core to 166 outputs that is ~446k MACs. That
   may well exceed the cost of computing the columns now that they are C++. ECFP sparsity
   (~66 on-bits of 2048) helps that half by ~30×; the 639 dense descriptor inputs do not shrink.

If the ridge is slower than computing, **`"fast"` should not ship** — a library with two modes,
no surrogate to train, validate, version or explain, and a genuinely interesting negative result:
once the descriptors are properly implemented, prediction stops being a speedup.

### Hard requirement: no Python computes any descriptor, in any mode

Stated by the owner and treated as a constraint, not a preference. Every column in every mode must
be computed by C++ — ours or RDKit's. An audit of what currently violates this is in flight; the
suspect surface is Mordred's `RingCount` (49), `Chi` (40), `TopologicalCharge` (21), `PathCount`
(11), `CarbonTypes` (9), `AtomCount` (8), `WalkCount` (6), `BondCount` (6) and `Constitutional`
(4) inside CORE.

---

## 5. The column set is versioned, and that is a guarantee

`spec="v1"` pins a **frozen, hashed column list**. Two runs with the same `spec` produce the same
columns in the same order, or the call fails.

This is not bureaucracy. The column count changed **twice in one day** during development
(28 → 193 → 182), for good reasons each time. Without a pin, "HUME embeddings" in two papers are
not the same object, and nobody finds out.

The following are part of `v1`'s definition, because each changes column values or membership:

* **chi follows RDKit's convention throughout, including where RDKit is internally inconsistent.**
  `Chi0n`/`Chi1n` count all atoms and bonds — an explicit H is a δ=1 vertex — while `Chi0v`/`Chi1v`
  are heavy-atom only, differing by exactly 1.0 per explicit hydrogen. At k ≥ 2 both agree. The
  trigger is an explicit H *atom*, not an isotope label: `CC[13CH3]` matches on both variants. For
  k = 5,6,7 there is no RDKit counterpart and the k ≥ 2 convention is the consistent extension.
* **Two different descriptors are both named BalabanJ**, and both are emitted. RDKit's uses a
  bond-order-weighted distance matrix; Mordred's passes RDKit's own formula an unweighted one.
  Naphthalene: 2.888052 against 1.925368.
* **The five `RATSC*_c` columns are removed.** They used a unity weight in a *centred*
  autocorrelation, which is identically zero for every molecule that can exist. `RPAIR{b}` already
  carries the uncentred version. Note Mordred's `ATSC0c` means Gasteiger *charge*; ours meant
  unity. Same letter, unrelated quantity.

---

## 6. Error contract

Every error carries four things: **what went wrong, why it matters, exactly what to do**, and —
where it can be computed — **how much it affects this input**. The last is what separates an error
message from friction. "Choose one of two options" teaches people to type the first one without
reading; "this changes 12.5% of your molecules, here is one" does not.

```
ValueError: featurize() requires an explicit `standardize=`. There is no safe default,
so HUME will not choose for you.

  standardize="none"
      Compute on the molecule exactly as supplied; multi-fragment inputs keep every
      fragment. This is what RDKit and Mordred do, so values stay bit-identical to
      those references. But for a salt, graph descriptors are dominated by the
      "no path between fragments" sentinel and carry little chemical meaning.

  standardize="largest_fragment"
      Keep only the largest connected component, discarding counterions. Conventional
      preprocessing, and usually what you want for property prediction. Values will
      NOT match RDKit/Mordred computed on the original SMILES.

  12,481 of your 100,000 inputs have more than one fragment, so this affects 12.5%
  of them. Example: CC(=O)Oc1ccccc1C(=O)O.[Na+] (index 47)
```

Applied to the rest:

| situation | the message must carry |
|---|---|
| unparseable SMILES | the string, its index, RDKit's own message, what each `on_error` value would do, and how many others failed |
| `descriptors="fast"` without weights | which file is missing, where to get it, and that `"exact"` works today with no download |
| exotic element, no Gasteiger parameters | the element, which columns are affected, that the rest still compute, and that this is Mordred's per-*weight* behaviour rather than our bug |
| `spec` mismatch | both hashes, both column counts, and which release each came from |

---

## 7. Using HUME with a tree ensemble: `feature_weights`

**This is a hyperparameter of the HEAD, not a flag of the representation.** HUME emits the same
columns either way. Nothing about it has to be decided before the task — it is tuned per task
like `max_depth`, on training folds only.

XGBoost's `colsample_by*` samples features **uniformly**. That is the wrong prior for a matrix
that is half sparse fingerprint bits and half dense descriptors: a descriptor is informative on
nearly every molecule, while a given ECFP bit is on in ~2% of them, so descriptors win the
sampling lottery on *availability* rather than on merit for the task. `feature_weights` makes the
sampling weighted, and one scalar `w` (fingerprint bits weight `w`, descriptors weight 1) turns
"descriptors on or off" into a continuous knob.

Measured, 34 datasets, scaffold folds, `colsample_bynode=0.3`:

| arm | ACE all | ACE cliff | ACE non-cliff | MoleculeNet | ECFP split share |
|---|---|---|---|---|---|
| `fp_only` | 0.7854 | 0.8610 | 0.7338 | 0.9487 | — |
| `desc_only` | 0.8460 | 0.9065 | 0.8055 | 0.9358 | — |
| w = 1 (XGBoost default) | 0.8080 | 0.8752 | 0.7626 | 0.8901 | 0.222 |
| w = 5 | 0.7953 | 0.8663 | 0.7471 | 0.8738 | 0.321 |
| **w = 10** | **0.7882** | 0.8605 | 0.7389 | **0.8712** | 0.395 |
| w = 100 | **0.7721** | 0.8464 | 0.7214 | 0.8931 | 0.756 |

**Recommendations:**

* **Tune `w` in the inner CV loop** alongside the other head hyperparameters. Tuning it on the
  test fold leaks, and the effect is large enough to leak meaningfully.
* **If you do not tune, use `w = 10`, not 1.** w=1 is dominated on BOTH suites — 0.8080 against
  0.7882 on activity, 0.8901 against 0.8712 on physchem. Uniform column sampling is not a neutral
  default here; it is a bad one.
* `feature_weights` requires `colsample_by* < 1` to do anything. Use `colsample_bynode` rather
  than `bytree`: weights apply at each sampling event, so per-node gives the reweighting far more
  chances to bite.

HUME should expose the block boundaries so this vector can be built without the caller
hard-coding column indices — `hume.feature_weights(columns, fp_weight=10.0)`. **OPEN**: not yet
implemented.

### What this measurement refuted

The sweep was run to test a specific mechanism: that descriptors *cannot resolve activity cliffs*
because they are smooth functions of structure, so up-weighting fingerprints should help
**specifically on cliff compounds**. **That prediction failed.** Going from w=1 to w=100 improves
non-cliff RMSE by 5.4% and cliff RMSE by only 3.3% — the gain is *larger* on the molecules that
are not cliffs.

The surviving explanation is a task-family one, not a cliff one. MoleculeACE datasets are
**congeneric series**: structurally similar molecules against a single target. Descriptors vary
little across the *whole series*, not merely across cliff pairs, so they are uninformative for the
entire dataset. That is consistent with Figure A's matched-pair result — the descriptor block moves
only **0.37×** as far as ECFP4 under a single graph edit, and a congeneric series is graph edits
throughout.

Note the shape of the two curves, which is the real finding: **ACE improves monotonically to the
boundary** and w=100 *beats* pure fingerprints, so the answer is not "drop descriptors" but "keep
them and starve them". **MoleculeNet has an interior optimum** at w≈10 and degrades by w=100, with
`fp_only` clearly worst. Descriptors genuinely help there and genuinely do not on ACE.

⚠️ **Provisional.** These numbers are `n_repeats: 1`, `fold_seeds: [0]`, `xgb_seeds: [0]` — a
single seed with no error bars, and several gaps are 0.01–0.04. The cliff-versus-non-cliff
comparison in particular (3.3% against 5.4%) is exactly the size that needs replicates before it
is quoted. A multi-seed re-run is in progress.

## 8. Open items

| item | decided by |
|---|---|
| `counts` binary vs count fingerprint | Figure C arm `hume_counts` |
| whether `"fast"` mode exists | boundary redraw + ridge inference cost |
| which columns are in `v1` | boundary redraw, then frozen |
| `descriptors` default if `core` hurts on potency | the cliff-mechanism experiment; see below |

**On the last one.** Descriptors *hurt* on the 34-dataset potency suite (bare `ecfp` beats
`ecfp+core` on 28 of 34) and *help* on QM and physchem (+25.8%, +17.0%). The mechanism appears to
be that descriptors are smooth functions of structure: under a single matched-pair edit the
descriptor block moves only **0.37×** as far as ECFP4 does, and exactly **0.000** on stereo and
E/Z flips. Activity cliffs are defined by exactly those edits, so descriptors cannot resolve a
cliff but can still be split on for the bulk trend, diluting the sparse bits that could.

If that holds, the fix is a head parameter rather than a representation default —
XGBoost's `feature_weights` makes column sampling weighted rather than uniform, turning "descriptors
on or off" from a binary choice into a continuous knob. A default that is right for QM and wrong
for potency is not a bug; presenting it as universal would be.
