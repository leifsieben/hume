# NOTES_spectral -- the 65 B_spectral columns

`src/hume_core/spectral.h`, verified by `verify_spectral.py`.

    cmake -S . -B build_spectral -DCMAKE_BUILD_TYPE=Release && cmake --build build_spectral -j4
    .venv/bin/python verify_spectral.py             # grade against both references + time
    .venv/bin/python verify_spectral.py --shuffle   # + the atom-renumbering probe
    .venv/bin/python verify_spectral.py --no-ref    # skip the float64 regeneration

The cmake line builds the extension as it stands; `spectral.h` is **not wired into
`bindings.cpp`** (contract: five agents, one enum). `verify_spectral.py` compiles its own
driver into `build_spectral/` from a string in the script, so it needs neither the extension
nor a fourth committed file.

---

## 1. What was built, and what was reused

65 columns, four matrices, one eigensolver:

| matrix | columns | per molecule |
|---|---|---|
| adjacency `_A` | SpAbs SpMax SpDiam SpMAD VE1 VE2 VE3 VR1 VR2 VR3 | 1 spectrum + 1 leading eigenvector |
| Burden (mordred's) | `BCUT{c,dv,d,s,Z,v,se,pe,are,p,i}-1h/-1l` | 11 extremal solves |
| Barysz `_Dz{Z,v,se,are,p,i}` | SpAbs SpDiam SpMAD SM1 VE1 VE2 VR1 VR2 VR3 | 5 Floyd-Warshall + 5 spectra + 3 eigenvectors |
| topological distance `_D` | SpAbs SpDiam SpMAD VE1 VE2 | 1 spectrum + 1 leading eigenvector |

**No new eigensolver was written.** `spectral.h` calls the two stages BCUT2D already runs --
`hume_eig::sytd2_upper` (dsytd2, UPLO='U') and `hume_eig::sterf_min_max` (dsterf) -- and keeps
the whole `d` array instead of only its two extremes, because SpAbs/SpMAD are functions of the
entire spectrum. The 20 BCUT columns call `hume_eig::extremal()` completely unchanged.

Three further reuses, each replacing something it would have been easy to rewrite:

* **`cpp/lu_small.h`** for the leading eigenvector. `eigen_small.h` returns eigenvalues only, and
  VE1/VR1 need a vector. Two steps of inverse iteration on the dense `M - lambda*I` using the
  existing reference `dgetf2`/`dgetrs` costs `n^3/3` against the reduction's `4n^3/3` and adds
  no kernel. The alternative -- dstein-style inverse iteration on the tridiagonal plus a dormtr
  back-transform through the Householder vectors -- is asymptotically cheaper and would have
  been a fourth linear-algebra kernel in the repository, on a step that is a fifth of this
  family's cost. It was not written.
* **`cpp/ac_weights.h`** for all twelve mordred atomic properties. Every descriptor in this
  family sets `explicit_hydrogens = False`, so the same verified function is called on the
  heavy-atom graph: `d` is then the heavy degree and `dv`'s hydrogen count is
  `GetTotalNumHs()` alone, which is exactly what mordred computes on `RemoveHs(m)`.
* **`topomisc.h::npPairwiseSum`** for the columns that are numpy reductions. Every `SpAbs`,
  `SpMAD`, `SM1` and `VE1` here is an `ndarray.sum()` or an `np.trace()` -- 25 of the 65
  directly, plus the 6 derived from a `VE1` -- and none of them is bit-exact unless numpy's
  8-accumulator pairwise association is reproduced. That the six `SM1_Dz*` columns come out
  bit-identical (section 2c) is the evidence the transliteration is right. There is one copy of
  it in the repository and this includes it rather than making a second.

**Lanczos was not reconsidered** -- the measurement above `bcut2d()` in `hume_blocks.h` (0.10x
to 0.63x at every size) still stands, and seven of these matrices want the *full* spectrum,
which Lanczos does not cheaply give.

---

## 2. The exactness result

### 2a. The ground-truth matrix is float32, and that had to be worked around

`data/dedupe2/matrix.npz` stores `MD` as **float32**. Seven significant digits is not enough to
evaluate the contract's 1e-9 tier, and "bit-identical" against it is a statement about the first
24 bits of a double. So the harness grades against **two** references:

* the shipped float32 matrix, all 20,000 molecules, both sides rounded to float32 -- full
  coverage at ~7 digits;
* a **float64 regeneration** from mordred 1.2.0 on the same 20,000 SMILES for exactly these 65
  columns (`verify_spectral.py --ref`, cached in the scratchpad) -- this one carries the grade.

The regeneration is cross-checked against the shipped matrix cell by cell after rounding back
to float32. It reproduces it on 1,299,969 of 1,300,000 cells -- and the 31 that differ are
themselves a finding, not a bug in the regeneration; see 2c.

### 2b. Against the float32 matrix, 20,000 molecules x 65 columns

    float32-identical cells: 1,299,962 of 1,300,000  (99.9971%)
    cells outside 1e-6 relative or disagreeing on NaN: 25
    NaN pattern: identical on all 65 columns. 104 disconnected molecules everywhere;
                 BCUTc-1h/-1l have 2 more (Se and Cu, where RDKit's Gasteiger throws);
                 BCUTs-1h/-1l, VR3_A and VR3_Dzv have 1 more, all the same molecule `[Cs]`
                 -- one atom, so the intrinsic state divides by a zero sigma-electron count,
                 and no bonds, so VR1 is 0.0 and log(0) is a missing value (fact 6).

Every one of the 25 is in `VE1_A / VE2_A / VE3_A / VR1_A / VR2_A / VR3_A`, on five molecules,
and section 4 shows that on all five **our value is the correct one and mordred's is not**.
The other 59 columns are float32-identical on all 19,896 defined cells. (Against the *float64*
reference the same disagreement counts 21 rather than 25, because three of the float32 cells
were NaN there -- see 2c.)

### 2c. Against the float64 regeneration -- the actual grade

    cells outside 1e-6 relative or disagreeing on NaN:  21 of 1,300,000
    columns within 1e-9 relative on EVERY cell mordred defines:  59 of 65
    columns bit-identical on every cell:                          6 of 65  (SM1_Dz*)

| group | worst relative deviation, any cell | within 1e-9 |
|---|---|---|
| `SM1_DzZ/Dzv/Dzse/Dzare/Dzp/Dzi` (6 cols) | **0** -- bit-identical, 19,896/19,896 | all |
| the 20 `BCUT*` | 2.6e-12 (`BCUTdv-1l`), typical 1e-14 | all |
| `SpAbs/SpMax/SpDiam/SpMAD` on all four matrices | 4.8e-15 | all |
| `VE1/VE2/VR1/VR2/VR3` on `_DzZ _Dzv _Dzp _D` | 3.2e-15 | all |
| `VE1_A VE2_A VE3_A VR1_A VR2_A VR3_A` | 4.1e-01 | 19,871-19,890 of 19,896 |

The `SM1_Dz*` columns are `np.trace` and never touch an eigensolver, so reproducing numpy's
pairwise association was enough to be exact. Everything else reaches mordred through dsyevd or
dgeev and reaches us through dsterf; it cannot be bit-identical, and 1e-15 is what "the same
eigenproblem, a different LAPACK driver" costs. The last row is section 4 and is not a defect.

**mordred disagrees with itself by more than we disagree with mordred.** The regenerated float64
reference, rounded back to float32, differs from `data/dedupe2/matrix.npz` -- the same package,
the same version, the same 20,000 SMILES, computed twice -- in **31 cells on 8 molecules**,
including three NaN/number flips. Every one of those 31 is in `VE1_A VE2_A VE3_A VR1_A VR2_A
VR3_A`; the other 59 columns reproduce exactly between the two runs. Our own 21 disagreements
are in the same six columns and on a subset of the same molecules.

---

## 3. Quirks reproduced, and the two unreproducible roundings

**Reproduced, deliberately:**

1. `require_connected` -- 104 corpus molecules are salts, and mordred short-circuits *before*
   `calculate()`, so all 65 cells are missing. Reproduced by one BFS on the heavy-atom graph.
   The heavy-graph component count equals `len(Chem.GetMolFrags(mol))` on 20,000 of 20,000
   corpus molecules (checked, not assumed).
2. mordred's **Burden matrix is dense**: `0.001 * np.ones((N, N))`, every non-bonded pair, not
   zero. It is not RDKit's BCUT2D Burden matrix -- different sparsity, different off-diagonal
   -- and `hume_blocks.h::bcut2d` is untouched by this file.
3. The **Barysz diagonal is written after Floyd-Warshall**, so the shortest paths are computed
   with a zero diagonal. Getting this backwards changes every off-diagonal entry of any molecule
   with a heteroatom.
4. **networkx's Floyd-Warshall k-loop is part of the answer.** Every Barysz off-diagonal is a
   sum of edge weights accumulated in that specific order; a Dijkstra that finds the same path
   rounds differently. `detail::floyd_warshall` is that loop.
5. `rethrow_zerodiv` is `np.errstate(divide="raise", invalid="raise")`, and it decides three
   sets of NaNs: a property that is zero on a bonded atom makes the *whole* Barysz family
   missing rather than infinite, and `VE3`/`VR3` are missing (not `-inf`) when `VE1`/`VR1` is
   zero. That last one is exactly why `VR3_A` has one more NaN in the corpus than `VR1_A` does
   -- `[Cs]`, one atom, no bonds, `VR1 = 0.0`.
6. `VR1` is a **Python for-loop over `GetBonds()`**, not a numpy reduction: strictly bond-index
   order, left to right, from the float `0.0`, and `** -0.5` is libm's `pow`, not `1/sqrt`.
7. `np.linalg.eigh` returns eigenvalues **ascending**, and `SpAbs`/`SpMAD` are pairwise sums
   over arrays in that order, so the spectrum is sorted before it is summed even though `SpAbs`
   is mathematically order-free.

**Not reproducible, and quantified instead:**

8. **mordred solves the symmetric Burden matrix with `np.linalg.eig`** -- the *unsymmetric*
   driver (dgeev: Hessenberg reduction + Francis QR), then takes `.real`. That is a genuine
   quirk, and unlike the eigenVECTOR path of section 4 it really is deterministic: BCUT wants
   only the two extreme eigenVALUES, which are well conditioned whether or not they are
   degenerate. But dgeev's rounding is not a function anyone
   can transliterate from the outside, and writing a Hessenberg QR to chase it would be adding
   the second eigensolver this task explicitly forbids. `spectral.h` runs the symmetric path
   instead. Measured cost: all 20 BCUT columns are float32-identical to mordred on every defined
   cell of the corpus, and in float64 agree to better than 1e-9 relative everywhere -- worst
   single cell 2.6e-12 (`BCUTdv-1l`), typical 1e-14.
9. `numpy.linalg.eigh` is dsyevd (divide-and-conquer, UPLO='L'); `sterf_min_max` is dsterf
   (Pal-Walker-Kahan, UPLO='U'). Different algorithm, different triangle, therefore different
   last bits on every eigenvalue. Same story: float32-identical everywhere, ~1e-15 in float64.

---

## 4. THE DIVERGENCE: six adjacency columns where mordred's value is not a function of the molecule

`VE1_A`, `VE2_A`, `VE3_A`, `VR1_A`, `VR2_A`, `VR3_A` are built from the eigenvector of the
adjacency matrix's largest eigenvalue. By Perron-Frobenius that eigenvalue is *simple* on a
connected graph, so the definition is well posed in exact arithmetic. In double precision it is
not, for a specific and chemically common shape: **a molecule made of two near-identical halves
joined by a long flexible linker**. The Perron root's separation from the next eigenvalue is
exponentially small in the linker length, and on corpus molecule 19279 (a symmetric bis-sulfonamide
PEG dimer, n = 97) it is **8.9e-16 -- below one ulp of the eigenvalue itself**.

How common: of the 19,896 connected corpus molecules, the relative Perron gap is below 1e-12 on
5, below 1e-9 on 27, and below 1e-6 on 60. All 27 are in the 55+ heavy-atom stratum.

**mordred's value there is a coin flip**, and there are three independent demonstrations of it,
all running mordred's own code on molecule 19279:

1. **Atom numbering.** Under nine random relabellings: `VE1_A` lands anywhere in 4.244 .. 5.011
   (an 18% spread) and `VR1_A` anywhere from 1.3e12 to NaN. NaN because the arbitrary vector
   picked out of the degenerate pair has mixed signs, so some bond's `v_i * v_j` is negative and
   `(-x) ** -0.5` is NaN.
2. **Thread count.** `VR1_A`'s last digits move between `OMP_NUM_THREADS` 1, 4 and 8 on the
   neighbouring molecules 18115 / 18694 / 19817.
3. **Run to run.** `data/dedupe2/matrix.npz` holds `VR1_A = 1.302269e12` for this cell. Re-running
   mordred 1.2.0 today on the same SMILES returns **NaN** -- single process and `nproc=10`, the
   four-module registry and the full `mordred.descriptors` one, all four combinations. Same code,
   same molecule, two different answers. (Row alignment is not the explanation: the six `SM1_Dz*`
   columns are bit-identical to the shipped matrix on all 19,896 defined rows.)

**We diverge, and the divergence is measurably an improvement.** Inverse iteration from the
all-ones start vector is not arbitrary in the same way: the true Perron vector is strictly
positive, so `ones` has an O(1) overlap with it and essentially none with the antisymmetric
partner that shares its eigenvalue. Against a 60-digit `mpmath` power iteration (the true
Perron vector), on the five molecules where the two implementations differ at all:

| molecule | n | lambda1-lambda2 | ours VE1_A | mordred VE1_A | ours VR1_A | mordred VR1_A |
|---|---|---|---|---|---|---|
| 19279 | 97 | 8.9e-16 | **3.8e-04** | 2.9e-01 | **7.7e-04** | NaN |
| 18115 | 90 | 1.2e-13 | **3.8e-10** | 2.7e-05 | **7.7e-10** | 5.5e-05 |
| 18694 | 94 | 1.9e-13 | **5.1e-09** | 2.2e-05 | **1.0e-08** | 4.6e-05 |
| 19817 | 90 | 1.9e-13 | **1.2e-07** | 3.5e-07 | **2.4e-07** | 7.0e-07 |
| 18726 | 57 | 2.5e-01 | 0.0e+00 | 5.5e-16 | **3.0e-15** | 6.9e-05 |

(relative error against the 60-digit value; smaller is better, the winner in bold)

Ours is better on nine of the ten cells and exact on the tenth; mordred is better on none.

18726 is the second, independent failure mode: its Perron root is well separated (gap 0.25),
but its eigenvector has entries down to 4.4e-13, and `VR1 = sum (v_i v_j)^-1/2` raises their
PRODUCT to the -1/2 -- so a 1e-16 absolute error in a 1e-13 entry is a 1e-3 relative error in
that bond's term. `VR1_A` for that molecule is 3.79e12, and it reaches 1.1e16 elsewhere in the
corpus (molecule 17223, n = 95, where we and mordred agree to float32). That is a property of
the descriptor, not of any solver.

**No NaN gate was added.** Replacing a value that is right to four digits (19279, the worst
case) with a missing value would be worse than keeping it, and any threshold would be arbitrary.
The divergence is 5 molecules x 6 columns out of 1,300,000 cells and it is recorded here rather
than hidden.

### 4a. How stable OUR values are under renumbering

`verify_spectral.py --shuffle` recomputes the whole corpus under four random atom relabellings
and reports the spread. Every one of the 65 is a permutation invariant in exact arithmetic, so
any spread is floating-point noise and its size is the column's honest resolution.

    columns with any cell moving more than 1e-9:   VE1_A VE2_A VE3_A VR1_A VR2_A VR3_A  (6 of 65)
    cells moving more than 1e-9:                   4 or 5 of 20,000, per affected column
    cells moving more than 1e-6:                   1 of 20,000 (molecule 19279), per column
    largest spread anywhere in the 65 x 20,000:    1.7e-03   (VR1_A / VR2_A, molecule 19279)
    largest spread in the OTHER 59 columns:        6.1e-12 (BCUTdv-1l); the other 58 <= 6.9e-14
    cells whose NaN-ness changes under renumbering: 0

Set that against mordred's 18% spread and NaN flip on the same molecule and the same probe.
The instability is a property of `eigh`'s treatment of a degenerate pair, not of the descriptor.

**Why two inverse-iteration steps, measured rather than defaulted** (the sweep over iteration
counts 1..4 was run in numpy against the same 60-digit truth). One is not enough: 18726 needs
the second -- VR1 relative error 1.5e-3 after one step, 1.4e-15 after two. Three is worse
than two at the degenerate end, because each extra solve injects a little more of the
antisymmetric partner -- 19279's VE1 error runs 2.4e-4 -> 6.7e-4 -> 1.1e-3 -> 1.5e-3 over four
iterations. Two is the minimum of that trade-off.

---

## 5. Timing

Best of 3 per molecule, single thread, `c++ -O3 -std=c++17`, all 20,000 corpus molecules.
Reported as the **minimum** over repetitions, not the mean: five agents were building and
testing in this checkout at once and the load average ran between 6 and 205, so a mean is a
measurement of the other agents. The minimum turned out to be robust to that -- the same run at
load 73 and at load 7 differs by under 1% in every cell of this table, which is why it is the
statistic used.

| stratum (heavy atoms) | n mols | mean us/mol | SD | median | max |
|---|---|---|---|---|---|
| 0-15 | 4000 | 34.6 | 10.7 | 33.7 | 84.8 |
| 15-25 | 4000 | 152.5 | 36.1 | 155.8 | 230.5 |
| 25-35 | 4000 | 297.2 | 60.2 | 295.4 | 452.5 |
| 35-55 | 4000 | 547.0 | 143.3 | 503.0 | 1106.5 |
| 55+ | 4000 | 1799.4 | 863.8 | 1532.5 | 9115.0 |
| ALL (stratified, NOT a natural distribution) | 20000 | 566.1 | 750.9 | 292.8 | 9115.0 |

**What this adds to HUME.** At the corpus median of 29 heavy atoms the family costs about
**297 us**, against HUME's stated ~830 us whole-molecule budget: **+36%**. The corpus is
deliberately stratified 4,000 per bin, so its 566 us mean is not a drug-like average -- a
drug-like distribution sits in the first three rows, i.e. 35-300 us. This is the largest single
descriptor family in the package and it is also the most expensive; the parent session should
know that before wiring it, and section 5a says exactly where the time goes.

### 5a. Where the time goes, and it is not where it looks

`build_spectral/parts.cpp` (generated, same build directory) times the three kernels in
isolation and weights them by the counts `compute()` actually makes: **11** `extremal` solves
(BCUT), **7** full `spectrum` calls (adjacency, distance, and five Barysz), **5** inverse
iterations, **5** Floyd-Warshall passes. Microseconds, quiet machine:

| n | `extremal` | `spectrum` | inverse iter. | Floyd-Warshall | modelled total | measured median |
|---|---|---|---|---|---|---|
| 10 | 2.2 | 2.2 | 0.4 | 0.5 | 44.6 | 33.7 (0-15) |
| 29 | 12.8 | 12.8 | 2.1 | 12.8 | **305.3** | **295.4** (25-35) |
| 45 | 33.9 | 34.2 | 5.6 | 50.8 | 893.7 | 503.0 (35-55) |
| 80 | 107.5 | 107.4 | 26.2 | 355.5 | 3842.0 | 1532.5 (55+) |
| 153 | 513.8 | 520.0 | 179.0 | 2462.8 | 22500.9 | 9115.0 (max) |

At n = 29 the model lands within 3% of the measured median, which is the check that the counts
above are the real counts. (It over-predicts above n = 45 because the kernels are timed on dense
random matrices, whose tridiagonals deflate more slowly and whose shortest-path graphs are
complete; real molecular matrices are sparser and cheaper.)

The split at drug-like size is **BCUT 46%, the other seven spectra 29%, Floyd-Warshall 21%,
inverse iteration 3%**. Above n ~ 80 that inverts and Floyd-Warshall becomes the majority
(46% at n = 80, 55% at n = 153), because it is the one O(n^3) term with no deflation to save it.

**The one identified lever, and why it is not taken.** The 11 BCUT solves want only the two
extreme eigenvalues, and `cpp/eigen_small.h`'s own header note already flags Sturm-sequence
bisection as the outstanding opportunity there -- O(n) per bisection step against the sweep's
O(n^2) per deflation, on the 46% of the family that is BCUT at drug-like sizes. It is not taken
here for the reason that note gives: bisection stops being "the same algorithm LAPACK runs",
which is the property this family's whole agreement argument rests on, and it would have to
defend a new agreement number rather than inherit this one. It is also the wrong lever for large
molecules, where Floyd-Warshall dominates and is not optional -- fact 4 makes its k-loop part of
the answer, so it cannot be replaced with a Dijkstra.

---

## 6. What `bindings.cpp` needs -- ONE new boundary field

Not made here, per the contract. `spectral.h` is complete and 64 of its 65 columns need nothing
new; `BCUTc-1h` / `BCUTc-1l` need one per-atom double that the boundary does not currently carry.

mordred's `c` atomic property, on the **heavy-atom** graph, is

    atom.GetDoubleProp("_GasteigerCharge") + atom.GetDoubleProp("_GasteigerHCharge")
        if atom.HasProp("_GasteigerHCharge") else 0.0

-- the atom's charge **plus the charge RDKit assigned to its implicit hydrogens**. `Mol.gast` in
`hume_blocks.h`, and `atom_d` column 1 in `bindings.cpp`, carry `_GasteigerCharge` alone. The two
differ by up to 0.30 per atom (aspirin's carboxyl oxygen: -0.4775 with the H charge, -0.1809
without). Nothing in HUME has needed the sum before: Autocorrelation's `c` weight runs on
`AddHs(m)`, where every hydrogen is its own atom and `_GasteigerHCharge` is deliberately cleared.

The data is already in the blob -- `_PICKLE_FLAGS` includes `ComputedProps` precisely because
`_GasteigerCharge` is a computed property, and `_GasteigerHCharge` rides along in the same
section (the note above `_PICKLE_FLAGS` in `_extract.py` says so). `molpickle.h`'s reader skips
it. The change is:

1. `molpickle.h` -- read `_GasteigerHCharge` alongside `_GasteigerCharge` into a new `atom_d`
   column (or add it to the existing charge, but a separate column keeps the existing one
   meaning what its name says).
2. `_extract.py` -- the array path needs the same field: `N_ATOM_DBL` 2 -> 3, and
   `charge_h.extend(map(_double_prop, ats, repeat("_GasteigerHCharge")))` guarded by
   `HasProp`, with the same `chg_ok` treatment.
3. `bindings.cpp` -- fill `spectral::Mol::at[i].c` with the sum, or `AC_C_MISSING` when
   `chg_ok == 0`. `ac_weights.h` turns that sentinel into the NaN that makes both BCUTc columns
   missing, which is exactly what mordred does when `AtomicProperty.calculate()` fails.

Until then, pass `AC_C_MISSING` and the two columns are NaN; the other 63 are unaffected.
`verify_spectral.py` supplies the real value from RDKit, so both columns **are** graded above --
they are float32-identical to mordred on all 19,894 cells where mordred defines them (the two
extra NaNs relative to the other columns are the Se and Cu molecules where RDKit's Gasteiger
throws). The implementation is verified; only the wiring is missing.

Everything else `spectral::Mol` needs is already at the boundary: `A_Z`, `A_FCHG`, `A_NH` from
`atom_i`, `B_U`/`B_V` from `bond_i`, and `bond_d`'s single column (`GetBondTypeAsDouble()`).
`spectral::compute` derives the fragment count itself.

---

## 7. Anything not exact

* **21 cells** (5 molecules x 6 columns, of 1,300,000) are outside 1e-6 relative or flip NaN
  against mordred. All are section 4's divergence, all are in our favour against a 60-digit
  reference, and mordred disagrees with its own earlier run on 31 cells of the same six columns.
* **Nothing is bit-identical to mordred except the six `SM1_Dz*` columns**, and it cannot be:
  mordred reaches the other 60 through dsyevd (`eigh`) or dgeev (`eig`), and we reach them
  through dsterf. Every one of the other 59 agrees to better than 1e-9 relative on every cell
  mordred defines -- worst 2.6e-12 -- and is float32-identical to the shipped matrix.
* **`BCUTc-1h` / `BCUTc-1l` are verified but not yet wireable** -- section 6. The harness feeds
  the correct `c` property; `bindings.cpp` cannot yet.
* **The guard in `leading_vector` fires**, 62 times over the corpus on 59 molecules, and this
  was found by counting rather than by assuming it would not. It is an exactly-singular
  `M - lambda*I` on small graphs with representable spectra (an isolated bond has
  `lambda_max = 1.0` exactly); the nudged shift resolves all 59 and none of them is among the
  five divergent molecules. `verify_spectral.py` prints the count on every run.
* **Timing is +36% on HUME's budget at the corpus median.** Section 5. Not hidden, not fixable
  without changing the algorithm the agreement argument rests on.
