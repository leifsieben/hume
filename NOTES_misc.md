# F_misc — 81 columns, eleven sub-families

`src/hume_core/misc_ext.h` · `verify_misc.py` · build dir `build_misc/`

> **Where it landed** (numbers from §5, all 20,000 corpus molecules, float64 references):
>
> * **76 of the 81 columns are bit-identical to the reference on every molecule**, NaNs included.
> * `LogEE_A` is exact modulo floating point — 11,663/19,896 bit-identical, all within
>   **6.8e-16** relative. It cannot be made bit-identical without a bit-compatible LAPACK
>   `dsyevd`; that is stated, not rounded away (§2.5).
> * The four partial-charge columns are bit-identical on 19,999/20,000 and disagree on **one**
>   molecule, because the boundary destroys *which* atoms RDKit failed to charge. This is a
>   boundary gap with a one-line fix I am not permitted to make; §2.1 has the measurement and
>   the patch.
> * `AETA_eta_BR` differs on **14 of 20,000** and that is a **deliberate divergence**: it is the
>   only column here that reads a ring *count*, `Chem.GetSymmSSSR` is not a function of the graph
>   on exactly those 14 cages (shown), and this header uses the same repaired ring set the other
>   62 ring-derived columns in the feature vector already use (§2.4).
>
> Cost: **256 µs/mol** as a standalone call (~178 µs/mol if the wiring hoists the three families
> this header reuses), against HUME's ~830 µs/mol whole-pipeline budget. §5.3.

---

## 1. The sub-classification (done first, before any code)

The 81 columns are not a family. They are eleven unrelated groups that happen to have survived
the same dedupe, and they were worked in this order — cheapest and most reusable first:

| # | cluster | n | columns | where it comes from |
|---|---------|---|---------|---------------------|
| S1 | plain scalars | 11 | `Chi0` `ExactMolWt` `NumValenceElectrons` `VAdjMat` `VMcGowan` `mZagreb1` `mZagreb2` `ECIndex` `Radius` `HybRatio` `Sv` | one loop each over atoms/bonds/the distance matrix |
| S2 | Chi subgraphs | 13 | `Xch-3d` `Xch-3dv` `Xch-4dv` `Xc-6dv` `Xpc-6d` `Xp-1d` `Xp-3d` `Xp-7d` `AXp-0d` `Xp-2dv` `Xp-3dv` `Xp-4dv` `Xp-7dv` | **reuses `src/hume_core/chi.h`** |
| S3 | PathCount | 8 | `MPC5` `MPC7` `MPC8` `MPC10` `TMPC10` `piPC7` `piPC9` `TpiPC10` | **reuses `src/hume_core/pathcount.h`** |
| S4 | WalkCount | 5 | `MWC02` `MWC04` `MWC07` `MWC09` `TMWC10` | **reuses `src/hume_core/topomisc.h`** |
| S5 | MolecularDistanceEdge | 5 | `MDEC-11` `MDEC-12` `MDEC-13` `MDEO-11` `MDEN-22` | generalises `constit.h`'s `molecularDistanceEdge` |
| S6 | partial charge | 4 | `MinPartialCharge` `MaxPartialCharge` `MinAbsPartialCharge` `MaxAbsPartialCharge` | a fold over the boundary's Gasteiger column |
| S7 | `fr_*` SMARTS | 7 | `fr_lactam` `fr_benzodiazepine` `fr_barbitur` `fr_azo` `fr_nitro_arom` `fr_phenol_noOrthoHbond` `fr_phos_ester` | **reuses `src/hume_core/frag_matcher.h`** + a new compiled program |
| S8 | ETA, averaged | 14 | `AETA_alpha` `AETA_beta{,_s,_ns,_ns_d}` `AETA_eta{,_L,_R,_RL,_F,_FL,_B,_BR}` `AETA_dBeta` | new: ETA machinery + the reference alkane |
| S9 | MolecularId | 12 | `MID` `AMID` `MID_h` `AMID_h` `MID_C` `AMID_C` `MID_N` `AMID_N` `MID_O` `AMID_O` `MID_X` `AMID_X` | new: the weighted DFS |
| S10 | BertzCT | 1 | `BertzCT` | new: Balaban matrix + symmetry classes + two entropies |
| S11 | LogEE_A | 1 | `LogEE_A` | new: the adjacency spectrum |

Sources: 14 columns are RDKit's, 66 are mordred's, and `BertzCT` is **both** (mordred's `BertzCT`
is a wrapper around `rdkit.Chem.GraphDescriptors.BertzCT`; it is graded against both and is exact
against both).

**Four of the eleven clusters are not new machinery.** `chi.h` accumulates every
`(shape, order, property)` bucket and emits 40 of them; `pathcount.h` accumulates every path
order and emits 11; `topomisc.h` walks every adjacency power and emits 6. These 26 columns are
other buckets of the same, already-verified accumulations, read straight out of those headers'
`Scratch`. No second enumeration exists to drift.

---

## 2. What was hard, and what the answer turned out to be

### 2.1 The partial charges — the boundary, not the arithmetic

The brief warned that Gasteiger is iterative and convergence-sensitive. **It is not computed
here at all**: `src/hume/_extract.py` already calls `rdPartialCharges.ComputeGasteigerCharges(m,
nIter=12)` (RDKit's own default iteration count and damping) and ships the per-atom charge across
the boundary, exactly as `Autocorrelation` consumes it. So there is no second PEOE parameter set
and no convergence question — these four columns are a *fold*, and the whole difficulty is in the
fold and in the NaN.

Two things had to be read rather than assumed, both in `rdkit/Chem/Descriptors.py`:

* `_ChargeDescriptors` folds with `min(chg, minChg)` — Python's **two-argument builtin**, which
  seeds the accumulator with its *first* argument. A NaN charge therefore *replaces* the running
  minimum and is then itself replaced by the next atom, rather than being skipped or being
  absorbing. The empty molecule keeps upstream's sentinels: `MinPartialCharge` 500,
  `MaxPartialCharge` −500. Both are reproduced.
* **The failure is per connected component, not per molecule.** RDKit propagates the NaN along
  bonds, so an unparameterised atom NaNs its own fragment and leaves the others alone. Measured:
  `[Se]C` → two NaNs; `O=[Sn]([O-])[O-].[Ca+2]` → four NaNs **and a 2.0**, and RDKit's own
  `MinPartialCharge` for that molecule is `2.0`, not NaN.

**The one thing I could not make exact, and why.** `_extract.py` replaces every non-finite charge
with `0.0` and records `chg_ok = 0` for the molecule. That flag cannot distinguish "this atom was
NaN" from "this atom really is 0.0", so the header takes `chg_ok == 0` to mean *all four columns
NaN*. Measured over the 20,000-molecule corpus:

| | molecules |
|---|---|
| `chg_ok == 0` | 4 |
| of those, entirely non-finite (header's NaN is right) | 2 |
| partially non-finite | 2 |
| **columns actually wrong** | **1 molecule** (`O=[Sn]([O-])[O-].[Ca+2]`), on all four columns |

So each of the four charge columns is **19,999 / 20,000 bit-identical, 1 NaN-vs-2.0
disagreement.** This is a boundary gap, not an arithmetic one: the fold in `partialCharges()`
already reads the charge array itself and would produce `2.0` if the NaN survived.

**Requested boundary change (I may not make it — `_extract.py` is outside this agent's files).**
In `src/hume/_extract.py`, the non-finite scan currently does

```python
bad = ~np.isfinite(charge_a)
if bad.any():
    charge_a[bad] = 0.0          # <- destroys which atoms failed
    chg_ok_a[owners] = 0
```

Either (a) add a per-atom `chg_finite` int32 array to `Batch` alongside `chg_ok`, or (b) add a
third `atom_d` column carrying the raw charge. (a) is cheaper and does not disturb the
BCUT2D-keeps-finite contract the zeroing exists for. With it, `miscext::Mol::chg` can be
NaN-filled for the flagged atoms and all four columns become 20,000/20,000.

### 2.2 Two FMA contractions inside RDKit, and one this header must NOT do

`BertzCT` is a function of `rdkit.ML.InfoTheory.entropy.InfoEntropy`, which is the **C++**
implementation (`Code/ML/InfoTheory/InfoGainFuncs.h`) — the Python `PyInfoEntropy` in the same
file is shadowed at import. Its inner line is `accum += -d * log(d)` in a single statement, and
clang contracts it to an FMA. Measured against the running rdkit over 3,000 random count vectors:

| form | mismatches |
|---|---|
| `accum += -d * std::log(d)` (no contraction) | 799 / 3000 (27%) |
| `accum = std::fma(-d, std::log(d), accum)` | **0 / 3000** |

The same applies to `getExactMolWt`'s closing `res += nHsToCount * mass_H`: without the FMA,
`ExactMolWt` is wrong on 20 of 4,000 corpus molecules. And RDKit's electron mass is
**0.00054857991** (`Code/GraphMol/atomic_data.cpp`), *not* the CODATA 0.000548579909065 — the
CODATA value moves the answer by 5.7e-13 per unit of formal charge, which is ~20 ulps at 148 Da.

The mirror-image trap is that clang will *also* contract multiplies that Python never fuses. Two
columns were one ulp out on 2 of the first 200 molecules purely from this: `VMcGowan`
(`a - nbonds * 6.56`) and `AETA_eta_BR` (`... + 0.086 * NR`). Every such site now goes through
`mulNoFma()`, a `volatile`-guarded multiply, so the header's answer does not depend on the
build's `-ffp-contract` setting. `TMWC10` and `ETA_eta_B`'s `sqrt(2) + 0.5*(N-3)` were guarded
the same way pre-emptively.

### 2.3 ETA is kekulized and the boundary is not

`EtaBase.kekulize = True`, so mordred hands the ETA descriptors `Chem.Kekulize(mol)` — former
aromatic bonds become SINGLE or DOUBLE while `GetIsAromatic()` stays true. In
`get_eta_nonsigma_contribute` that is the difference between a 0.0 and a 2.0 contribution, and
the Kekulé structure is not at the boundary.

It is **reconstructed, not re-kekulized**, using the per-atom identity `constit.h`'s `nBondsKD`
already relies on and has verified against `Chem.Kekulize` on 4,000 molecules (0 mismatches, the
flag never outside {0,1}):

```
takesDouble(i) = tval(i) - nH(i) - round(non-aromatic valence contributions) - nAromaticBonds(i)
```

**Which** aromatic bond becomes the double one is a Kekulé choice and is *not needed*: every
former-aromatic bond contributes 2.0 to `beta_ns` if double and 0.0 if single, so an atom's total
is `2.0 * takesDouble(i)` under any matching. That is why the header reconstructs a per-ATOM flag
and never a per-BOND assignment — the per-bond one would be ill-posed and the per-atom one is
not. The header throws, naming the molecule, if the identity ever fails.

The reference alkane (`AlterMolecule`) needed three details that are only in the source: the
degree > 4 check is on the **original** molecule's degrees and fails the whole descriptor; the
hydrogens are **dropped**, so the reference graph can have fewer atoms than the original and
`AETA_eta_R`/`AETA_eta_RL` divide by *its* atom count while `AETA_eta_F`/`_FL`/`_B`/`_BR` divide
by the original's; and `ETA_eta_B` has `ring=False`, so its ring term is `0.086 * 0`, not the
ring count (only `_BR` uses `RingCount()`).

### 2.4 `AETA_eta_BR` — a deliberate divergence, measured

`ETA_eta_BR` is the one column in this group that reads a ring **count**, and `mordred` asks for
`len(Chem.GetSymmSSSR(mol))`. `GetSymmSSSR` is not a function of the molecular graph — the
repository already establishes this (`src/hume/_rings.py`, `src/hume_core/rdkcore.h`), and the
boundary ships the *repaired*, canonical-order ring set that the 49 `RingCount` columns and the
13 RDKit ring predicates already use. This header uses the same repaired count.

Measured on this corpus: the two counts differ on **14 of 20,000 molecules** (0.07%), all small
cages, and the repaired count is one higher (5 vs 4). That the definition really is ill-posed
there was checked directly — rebuilding the same graph with atoms and bonds presented in a
shuffled order gives `len(GetSymmSSSR)` ∈ {4, 5} on every one of them:

```
CC12CC3C1OCCN23    skeleton GetSymmSSSR under atom+bond shuffle: [4, 5]   repaired: 5   as-given: 4
OC12CC3C1OCOC23    skeleton GetSymmSSSR under atom+bond shuffle: [4, 5]   repaired: 5   as-given: 4
CN1C2C=CC3OC2C13   skeleton GetSymmSSSR under atom+bond shuffle: [4, 5]   repaired: 5   as-given: 4
```

So `AETA_eta_BR` is 19,881/19,895 bit-identical and differs on 14 by `0.086 / A`. Taking
`GetSymmSSSR` here instead would make this column agree with the matrix and be **inconsistent
with the 62 other ring-derived columns in the same feature vector**. House rule 4: diverge from
the ill-posed definition, and say so with a number.

### 2.5 `LogEE_A` — the one column that is not bit-exact

`LogEE` is `a + log(sum(exp(λ - a)) + exp(-a))` over the **whole** adjacency spectrum, and
`mordred` gets its eigenvalues from `numpy.linalg.eigh`, i.e. LAPACK's **dsyevd** (divide and
conquer). This header uses `cpp/eigen_small.h` — LAPACK's `dsytd2` + `dsterf` written out, the
solver this repository ships to keep BCUT2D off the host BLAS. Those are different algorithms and
they differ in the last bits.

Measured over the 19,896 connected molecules: **11,663 bit-identical, 19,896/19,896 within 1e-9
relative, worst relative deviation 6.77e-16** (about 3 ulps). It is stated here rather than
rounded away: this column is *not* exact, and making it exact means shipping a bit-compatible
`dsyevd`, which is out of scope for one column and is the same wall `B_spectral` will hit for its
65. The log-sum-exp damps the eigenvalue noise, which is why 6.8e-16 is the whole error.

### 2.6 The trap I actually fell into

The first version of this header carried a **hand-typed `MCGOWAN_VOL` table**. It was correct
through Z=55 and silently wrong from Z=56 onwards (lanthanides and beyond, shifted). It survived
the first 200-molecule check and was caught by the full corpus on exactly two molecules —
`[O-2].[O-2].[O-2].[Sm+3].[Sm+3]` and `[Ce+3].[F-].[F-].[F-]`. Both element tables in this file
are now **spliced from the pinned mordred process's own `_atomic_property` tables** and checked
entry by entry against them (0 mismatches over Z = 0…118). This is house rule 2 in miniature: a
table typed from memory is a defect that does not announce itself, and only a corpus with rare
elements finds it.

### 2.7 Smaller upstream facts that are load-bearing

* `Chi0` (RDKit) is `sum(numpy.sqrt(1./deltas))` with **zero degrees removed** and the *builtin*
  `sum` — left to right in atom order, not numpy's pairwise sum. `Sv` and `mZagreb1` are the
  opposite: they are `np.sum` over a float64 array and need `topomisc::npPairwiseSum`.
* `Sv` and `VMcGowan` are computed on the **hydrogen-added** molecule (their descriptor classes
  leave `explicit_hydrogens` at the base class's `True`), while everything else in this group is
  on the heavy graph. `VMcGowan` also subtracts `6.56` per bond **of the H-added molecule**.
* `TMWC10` is `A + 0.5*A.sum() + sum_{k=2..10} log(sum(A^k)+1)` — note `MWC01` is not a log, and
  the recursion accumulates upwards, so the order is part of the definition. Same for `TMPC10`
  (integer, `acc_0 = int(A)`) and `TpiPC10` (`acc_0 = float(A)`, and only the final term is
  logged).
* Unreachable pairs are **1e8**, not infinity, everywhere the distance matrix appears, so
  `Radius` for a salt is literally `100000000` and `ECIndex` is enormous. `topomisc.h` note 4
  makes the same point for `Diameter`; both are quirks (deterministic, a function of the
  molecule) and are reproduced.
* `_LookUpBondOrder` inside BertzCT returns `float(BondType)` for anything not aromatic, so a
  **DATIVE bond has "order" 17.0**. Reproduced.
* `mZagreb1` is `np.sum(V ** -2)` under `errstate(divide="raise")` → NaN if any atom has degree 0;
  `mZagreb2` is a *builtin* sum over bonds and is never at risk, because a bond endpoint has
  degree ≥ 1.
* `require_connected` covers 26 of these 81 columns (all of S8, all of S9, `LogEE_A`) and is a
  NaN, not an error. `n_frags` is computed here by BFS rather than carried.
* `AXp-0d`'s order-0 bucket is a *different atom set* from every other Chi column (chi.h note 1);
  it is NaN for the 90 corpus molecules carrying an explicit hydrogen, and the reference agrees.

---

## 3. The `fr_*` program

The seven SMARTS are rows of `$RDDATA/FragmentDescriptors.csv`, counted by `Fragments.py` as
`len(GetSubstructMatches(patt, uniquify=True))`. They are **not typed into this header**: they are
compiled by `cpp/gen_frag_program.py`'s own compiler — the one that produced `cpp/frag_program.h`
and `cpp/qed_alert_program.h` — and evaluated by the one matcher in `src/hume_core/frag_matcher.h`.
The compiler's `validate` re-renders every node in RDKit's own `DescribeQuery()` format:

```
7 top-level patterns + 13 recursive sub-queries validated against RDKit's own DescribeQuery()
MISMATCHES: 0
PARSE PROVEN IDENTICAL TO RDKIT'S, per atom and per bond, structure/value/negation.
patterns 20 (7 named + 13 recursive), nodes 260, aroots 84, qbonds 70
SPEC_SHA256 334cb9027881feec3312ddff56cd9f02f2b1164327692b6d04052c49b0d6b75c
```

**Requested follow-up for the wiring (I may not add files under `cpp/`).** The compiled program
currently lives inside `misc_ext.h` in `namespace miscext::fr_prog`. It should be moved to
`cpp/frag_misc_program.h` and registered as a third entry in `gen_frag_program.py`'s `SPEC_SETS`
(`"misc": (misc_specs, ".../frag_misc_program.h", "frag_misc_prog", ...)`), so that
`gen_frag_program.py check` covers these seven patterns with the same drift guard as the other
190. The arrays are byte-identical to what that generator emits; only the file they sit in
changes.

---

## 4. Grading

`.venv/bin/python verify_misc.py` — a test harness only; nothing in it computes a descriptor.
It drives the real boundary (`hume._extract.extract`), writes the arrays as a binary blob,
generates and compiles `build_misc/driver_misc.cpp` against `misc_ext.h`, and grades.

Ground truth is **float64**, not the shipped matrix: `data/dedupe2/matrix.npz` stores its values
as **float32**, which cannot support a bit-identical claim (it agrees with the float64 reference
to 5.9e-8 worst case, consistent with float32's 1.2e-7 eps). So

* the 14 RDKit columns are recomputed live from the pinned rdkit, and
* the 67 mordred columns come from a mordred 1.2.0 run under `.venv-mordred`, float64, sharded
  eight ways and cached in `build_misc/md_ref_*.npz`.

Regenerate those shards with the harness itself — the generator is embedded in `verify_misc.py`
and written out to `build_misc/md_ref.py`, so there is no fourth file:

```
.venv/bin/python verify_misc.py --gen-md-ref
```

### Per-column result, all 20,000 molecules

See §5. Reported per column: n bit-identical, n within 1e-9 relative, n within 1e-6, n
mismatched, NaN agreement as its own count, and the SMILES / reference / ours of every mismatch.

### Timing

See §5. Note for the wiring: **26 of the 81 columns are re-reads of `chi.h`, `pathcount.h` and
`topomisc.h`, all three of which `bindings.cpp` already calls for other families.** If the
wiring calls `miscext::compute` in addition to those, that work is paid twice. The per-part
timing in §5 quotes each of those separately so the parent session can decide whether to hoist
them; the honest "what F_misc adds" number is the total minus the three reused parts.

---

## 5. Measurements

All numbers below are from `.venv/bin/python verify_misc.py --reps 3` over the whole
20,000-molecule corpus, against float64 references (`build_misc/full7.log`).

### 5.1 Verdict

```
SUMMARY  82 columns graded: 76 EXACT, 1 exact-modulo-fp(<=1e-9), 5 MISMATCH
  MISMATCH MinPartialCharge         bad 1 / 20000   worst rel 0     <- boundary gap, S2.1
  MISMATCH MinAbsPartialCharge      bad 1 / 20000   worst rel 0     <- same molecule
  MISMATCH MaxAbsPartialCharge      bad 1 / 20000   worst rel 0     <- same molecule
  MISMATCH MaxPartialCharge         bad 1 / 20000   worst rel 0     <- same molecule
  MISMATCH AETA_eta_BR              bad 14 / 20000  worst rel 0.252 <- deliberate divergence, S2.4
```

(82 rather than 81 because `BertzCT` is graded twice — once against RDKit's own
`GraphDescriptors.BertzCT` and once against mordred's wrapper of it. Both are 20,000/20,000.)

So: **76 of the 81 columns are bit-identical on every one of the 20,000 molecules**, one
(`LogEE_A`) is exact-modulo-floating-point at 6.8e-16, and the remaining four are the single
charge molecule described in §2.1. `AETA_eta_BR` is not a defect but a divergence, §2.4.

### 5.2 Per column

"finite pairs" is the number of molecules where reference and ours are both finite; "NaN agreed"
counts reference-NaN ↔ ours-NaN, and no column has a NaN disagreement except `MinPartialCharge`
and its three siblings (1 each).

| column | grade | bit-identical / finite pairs | NaN agreed | wrong | worst rel |
|---|---|---|---|---|---|
| `MinPartialCharge` | see below | 19996 / 19996 | 3 | 1 | 0 |
| `MinAbsPartialCharge` | see below | 19996 / 19996 | 3 | 1 | 0 |
| `MaxAbsPartialCharge` | see below | 19996 / 19996 | 3 | 1 | 0 |
| `ExactMolWt` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_lactam` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_benzodiazepine` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_barbitur` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_azo` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_nitro_arom` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_phenol_noOrthoHbond` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `fr_phos_ester` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Chi0` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `LogEE_A` | 1e-9 | 11663 / 19896 | 104 | 0 | 6.77e-16 |
| `BertzCT` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `HybRatio` | **exact** | 19992 / 19992 | 8 | 0 | 0 |
| `Xch-3d` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xch-3dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xch-4dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xc-6dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xpc-6d` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xp-1d` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xp-3d` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xp-7d` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `AXp-0d` | **exact** | 19910 / 19910 | 90 | 0 | 0 |
| `Xp-2dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xp-3dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xp-4dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Xp-7dv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Sv` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `ECIndex` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `AETA_alpha` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_beta` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_beta_s` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_beta_ns` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_beta_ns_d` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta_L` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta_R` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta_RL` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta_F` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta_FL` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AETA_eta_B` | **exact** | 19895 / 19895 | 105 | 0 | 0 |
| `AETA_eta_BR` | see below | 19881 / 19895 | 105 | 14 | 0.252 |
| `AETA_dBeta` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `VMcGowan` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MDEC-11` | **exact** | 10050 / 10050 | 9950 | 0 | 0 |
| `MDEC-12` | **exact** | 15026 / 15026 | 4974 | 0 | 0 |
| `MDEC-13` | **exact** | 14990 / 14990 | 5010 | 0 | 0 |
| `MDEO-11` | **exact** | 11835 / 11835 | 8165 | 0 | 0 |
| `MDEN-22` | **exact** | 11459 / 11459 | 8541 | 0 | 0 |
| `MID` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AMID` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `MID_h` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AMID_h` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `MID_C` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AMID_C` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `MID_N` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AMID_N` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `MID_O` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AMID_O` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `MID_X` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `AMID_X` | **exact** | 19896 / 19896 | 104 | 0 | 0 |
| `MPC5` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MPC7` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MPC8` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MPC10` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `TMPC10` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `piPC7` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `piPC9` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `TpiPC10` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `Radius` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `VAdjMat` | **exact** | 19995 / 19995 | 5 | 0 | 0 |
| `MWC02` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MWC04` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MWC07` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MWC09` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `TMWC10` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `mZagreb1` | **exact** | 19910 / 19910 | 90 | 0 | 0 |
| `mZagreb2` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `NumValenceElectrons` | **exact** | 20000 / 20000 | 0 | 0 | 0 |
| `MaxPartialCharge` | see below | 19996 / 19996 | 3 | 1 | 0 |

The NaN columns are not failures: `MDEC-*` is NaN when the molecule has no atom pair of the right
element and valences (upstream's `1.0/(2.0*0)` → `ZeroDivisionError` → NaN), the ETA and MID
families are NaN for the 104 disconnected molecules (`require_connected`), `AXp-0d` is NaN for the
90 molecules carrying an explicit hydrogen (chi.h note 1), `mZagreb1` is NaN for the 90 with an
isolated atom, `VAdjMat` for the 5 with no heavy–heavy bond, and `HybRatio` for the 8 with no
sp2/sp3 carbon. Every one of those counts is matched by the reference exactly.

### 5.3 Timing

Statistic: **best of 3 per molecule**, from `verify_misc.py --reps 3`. The corpus was measured
three times — once with four other agents hammering the same 8-core box (`load average` ~95) and
twice after they finished (`load average` ~8) — and the three runs agree within **0.4%**
(255.81 / 255.62 / 254.77 µs/mol). The per-molecule minimum is what makes it robust to
contention; whole-pass totals on the loaded box were 3-5x higher and are not quoted. The table
below is the middle run (`build_misc/final.log` now holds the third; they are interchangeable).

```
per-molecule cost of the whole 81-column group (best of 3, quiet machine):
  stratum 0-15         n= 4000      30.52 +-    16.31 us/mol   median   25.17   max    327.67
  stratum 15-25        n= 4000     105.77 +-    37.08 us/mol   median  101.79   max    613.92
  stratum 25-35        n= 4000     185.06 +-    53.26 us/mol   median  178.29   max    769.04
  stratum 35-55        n= 4000     316.20 +-   149.04 us/mol   median  287.17   max   4888.33
  stratum 55+          n= 4000     640.56 +-   504.52 us/mol   median  561.75   max  16186.04
  ALL                  n=20000     255.62 +-   319.75 us/mol   median  178.6    p99   1121
```

**Against HUME's ~830 µs/mol whole-pipeline budget this group is 256 µs/mol standalone — about
31% — and it is strongly size-dependent: 25 µs at the median of the smallest stratum, 562 µs at
the median of the largest.** That is a real cost and it is stated plainly rather than averaged
into invisibility. Where it goes:

| part | µs/mol | share | note |
|---|---|---|---|
| `chi.h` enumeration (S2) | 58.6 | 23% | **already paid** — `bindings.cpp` calls it for its own 40 columns |
| MolecularId DFS (S9) | ~57.9 | 23% | 12 columns; measured by difference (255.6 with, 197.8 without) |
| BertzCT (S10) | 55.5 | 22% | **one column** |
| LogEE_A (S11) | 26.7 | 10% | **one column**; a full symmetric eigendecomposition |
| `pathcount.h` (S3) | 13.1 | 5% | **already paid** |
| `fr_*` matcher (S7) | 11.6 | 5% | |
| `topomisc` walkTraces (S4) | 5.2 | 2% | **already paid** |
| S1 + S5 + S6 + S8 (ETA) + graph build | ~27 | 11% | by subtraction |

**What F_misc actually adds depends on the wiring.** If `bindings.cpp` calls `miscext::compute`
*in addition to* `chisub::compute`, `pathcount::compute` and `topomisc::compute`, those three
families are paid twice and the group costs 256 µs/mol. If the parent hoists them — computing
each once and reading both column sets out of the same `Scratch` — the group adds about
**178 µs/mol**. That needs `miscext::compute` to accept pre-filled scratch objects; it is a
small API change and I have not made it, because it is a decision about `bindings.cpp`'s shape.

**The tail is MolecularId, and it is upstream's own exponential.** The three slowest molecules in
the corpus are 14-16 ms each and all are heavily bridged polycyclic cages; disabling S9 takes the
worst (`n=76`, 95 bonds) from 16,134 µs to 1,603 µs. `AtomicId._search` walks every simple path
whose product of `deg(u)*deg(v)` edge weights stays under `int(1/eps**2)`, and in a cage the
number of such paths explodes. Nothing can be trimmed without changing the value — the sum is
over exactly those paths, in that order. mordred pays the same cost in Python. The p99 for the
whole group is 1,121 µs; the mean is 256.

**Two optimisations were taken inside BertzCT, both proved inert.** It started at ~2,240 µs/mol
on its own (same corpus, same contention) and is now ~55:

1. The Floyd-Warshall relaxation runs **in place** instead of into a second buffer. That is
   exact, not approximate: at step *k*, row *k* is unchanged (because `d[k][k] == 0`), so an
   in-place sweep reads exactly the numbers the buffered sweep would. Two skips ride along — a
   row with `d[i][k] == LOCAL_INF` cannot change (every entry is <= `LOCAL_INF`), and `i == k`
   cannot change — both exact for the same reason.
2. `'%.4f' % x` is evaluated **once per distinct double** instead of once per matrix entry, and
   each distinct format string is interned to a small integer id. The symmetry-class key is a
   tuple compared only for equality, so interning through the *string* (never the bit pattern)
   gives an identical equivalence relation — including the case where two different doubles
   format alike. For n > 100 the per-row sort became a `partial_sort` of the first 100, which is
   the same tuple because `cutoff` is 100.

Both were checked the only way that counts: `BertzCT` is 20,000/20,000 bit-identical against
rdkit before and after.

### 5.4 Remaining headroom, flagged and not taken

* BertzCT's Floyd–Warshall is now ~30% of that column's cost and is O(n³) in a 100+-atom
  molecule. Nothing cheaper is exact, because the value of a path is assembled from sub-path sums
  in *k*-order and a Dijkstra would associate the additions differently.
* `LogEE_A` computes the full spectrum with `cpp/eigen_small.h`'s reduction + QL/QR sweep. The
  sweep resolves all *n* eigenvalues because LogEE needs all of them, so `eigen_small.h`'s own
  "extremal-only" headroom note does not apply here.
* The ETA and MolecularId blocks allocate a handful of `std::vector`s per molecule that could
  live in `Scratch`. Small (a few µs), untaken.
* `MolecularId` cannot be made cheaper without changing the number (see §5.3). If the tail ever
  matters operationally, the lever is not the code but the descriptor set: the 12 MID columns
  cost as much as the 40-column `chi.h` family.

### 5.5 Reproducing

```
.venv/bin/python verify_misc.py                 # all 20,000, grades everything
.venv/bin/python verify_misc.py --n 2000        # the smallest stratum, ~30 s
.venv/bin/python verify_misc.py --reps 3        # with timing
.venv/bin/python verify_misc.py --only BertzCT  # one column
```

The script compiles `build_misc/driver_misc.cpp` (which it generates) against `misc_ext.h` with
plain `c++ -std=c++17 -O2`; it does **not** go through CMake, because the header is not wired
into `bindings.cpp` yet and this agent may not wire it. `build_misc/` also caches the RDKit
reference (`rd_ref_20000.npy`) and holds the mordred shards.
