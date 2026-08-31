# C_eta — mordred's ExtendedTopochemicalAtom block, 29 columns

Deliverables: `src/hume_core/eta.h`, `verify_eta.py`, this file. `bindings.cpp` untouched; the
wiring it needs is spelled out at the bottom.

Specification read line for line: `.venv-mordred/lib/python3.11/site-packages/mordred/`
`ExtendedTopochemicalAtom.py`, `_atomic_property.py`, `_base/context.py`, `_base/calculator.py`,
`RingCount.py`, `_util.py`. Nothing here is from memory or from the ETA papers.

---

## 1. Grade

`.venv/bin/python verify_eta.py` — all 20,000 molecules of `data/dedupe2/corpus.json`, graded
against **float64 mordred recomputed live**, not against `matrix.npz`. The stored matrix is
`float32`, so "bit-identical" against it would mean agreeing to seven digits; the script shells
out to `.venv-mordred/bin/python` once (126 s), caches the float64 answers in
`build_eta/mordred_eta.npz`, and still cross-checks the float32 matrix afterwards as a NaN-set
audit (0 disagreements).

| | |
|---|---|
| columns bit-identical on every finite cell | **25 of 29** |
| cells graded | 19,896 × 29 (+ 104 all-NaN molecules) |
| cells outside 1e-6 relative | **17 of 576,984** (0.0029%) |
| NaN-set disagreements vs mordred | **0** |
| NaN-set disagreements vs `matrix.npz` | **0** |

The 17 are three separate stories, all listed in §3. There is no column where the answer is
merely "close": every cell is either bit-identical or one of the 17 named divergences.

NaN agreement: 104 molecules are salts (`n_frags != 1`) and mordred returns
`Missing(MultipleFragments())` for all 29 before `calculate` is ever called; `ETA_eta_B` and
`ETA_eta_BR` are NaN on one more, the single-atom `[Cs]`, where `EtaBranchingIndex` calls
`self.fail(ValueError("single atom"))`. Both are reproduced exactly.

## 2. Timing

`eta::build_from_rows` + `eta::compute`, CPU time (`CLOCK_THREAD_CPUTIME_ID`), best of three
passes with all input already in memory:

| stratum | n | mean µs | sd |
|---|---|---|---|
| 0–15 | 4,000 | 2.20 | 0.76 |
| 15–25 | 4,000 | 8.54 | 1.81 |
| 25–35 | 4,000 | 14.89 | 2.80 |
| 35–55 | 4,000 | 23.95 | 5.62 |
| 55+ | 4,000 | 52.07 | 16.48 |
| **all** | **20,000** | **20.33** | **19.14** |

**This group adds 20.3 µs to HUME's ~830 µs per-molecule budget — 2.5%.** (Run to run on a busy
machine the mean moves by a few percent; the strata ordering and the 2.5% do not.)

Measure it with CPU time, not wall clock. Up to five agents build and run in this checkout at
once, and on a loaded machine the same molecule measured 161,386 µs of wall clock and 120 µs of
work; the first version of the driver reported a 151 µs mean and a 2,478 µs standard deviation
that were entirely scheduler.

Where it goes: 70% is `composite_pair`, which is one divide and one square root per atom PAIR,
done twice (the molecule's graph and the reference alkane's). That is what `EtaCompositeIndex`
is, and `sqrt(g_i g_j / r^2)` has no algebraic shortcut that rounds the same way. Materialising
mordred's full distance matrix was already dropped (one BFS row at a time, and the plain and
`local` sums share their square root) — worth 55.7 → 52.1 µs on the 55+ stratum, mostly a memory
win. **The one lever left**, not taken: on the reference alkane `gamma_i` is `1/deg_i` with
`deg_i` in 1..4, so `sqrt(g_i g_j / r^2)` takes at most `16 × diameter` distinct values and could
be memoised on `(deg_i, deg_j, r)` — bit-exact, since identical doubles give identical results.
That would roughly halve the block, to ~14 µs. It was left out because it needs a generation-
stamped cache to avoid a per-molecule `memset` that would eat the gain, and 0.8% of the budget
is not worth a cache that can be subtly wrong.

## 3. Divergences and quirks

### 3a. Ill-posed, diverged: `ETA_epsilon_4`, `ETA_dEpsilon_B`, `ETA_dEpsilon_C` (Kekulé axis)

`EtaEpsilon(4)` averages `eta_epsilon` over `AlterMolecule(saturated=True)`: same elements and
formal charges, **C–C bonds reduced to single**, every bond touching a heteroatom left at its
*kekulized* order, then sanitize and `AddHs`. Its hydrogen count therefore depends on **which**
aromatic bonds took the double, not just how many per atom — and for a molecule with an aromatic
hetero–hetero bond there can be more than one Kekulé structure with different numbers of C–C
doubles. Pyridazine is the smallest case: `N1=N2, C3=C4, C5=C6` and `N2=C3, C4=C5, C6=N1` are
both valid and give the saturated ring 8 hydrogens or 6.

Measured by enumerating **every** perfect matching of the "needs a ring double bond" atoms over
the aromatic bonds, for all 20,000 corpus molecules:

* 15,943 molecules have aromatic bonds; 3,048 have an aromatic hetero–hetero bond.
* **79 molecules (0.40%)** admit perfect matchings with *different* C–C double counts. Those
  three columns are not functions of the molecule there.
* RDKit's own choice follows no rule this header could copy: over those 79 it takes the **fewest**
  C–C doubles 43 times, the **most** 35 times, and something in between once.

`eta.h` therefore makes its own deterministic choice (`KekuleMatcher`: always extend the
lowest-index unmatched atom, trying its aromatic bonds in bond-index order) and is allowed to
diverge. **It happens to agree with RDKit on all 79** — two similar depth-first searches — so all
three columns come out bit-identical on 19,895 of 19,896 finite cells. That is luck, not a
guarantee, and `verify_eta.py --invariance` shows the real exposure: renumbering the atoms of
4,000 molecules three times each moves **14 cells of 12,000** in each of these three columns
(max |Δ| 4.8e-2) and **nothing** in the other 26. A Kekulé round trip
(`Kekulize(clearAromaticFlags=True)` → SMILES → re-parse) moves **nothing at all**, in any
column, because this header rebuilds the saturated skeleton from the graph and never reads the
input's bond types for it — the axis that moves mordred's value cannot reach ours.

### 3b. Ill-posed upstream, already diverged from by this repo: `ETA_eta_BR` (14 molecules)

`EtaBranchingIndex(ring=True)` adds `0.086 * RingCount()`, which is
`len(Chem.GetSymmSSSR(mol))`. `GetSymmSSSR` is not a function of the graph — RDKit's own source
says its symmetrisation "may miss extra rings" depending on presentation order — and
`src/hume/_rings.py` already perceives rings on a canonically rebuilt skeleton and hands that set
to `RingCount`. `eta.h` takes `n_rings` from **the same place**, because two ring counts inside
one package that disagree with each other is worse than one that disagrees with an unstable
upstream.

Price: **14 molecules of 20,000** where `ETA_eta_BR` differs from mordred by exactly 0.086 — the
same 14 where HUME's `nRing` already differs from mordred's, all small bridged cages
(`CC12CC3C1OCCN23`, `OC12CC3C1OCOC23`, `OC12CN3C(CC13)CC2`, `CN1C2C=CC3OC2C13`, …). This is not a
new divergence, it is an existing one propagating into a new column, and `ETA_eta_BR` is
invariant under renumbering where mordred's is not.

### 3c. Cannot make exact: `[Cs]` (1 molecule, 3 cells)

`AlterMolecule(saturated=True)` builds fresh `Chem.Atom(Z)` objects carrying only the formal
charge, so an atom whose bracket form set `noImplicit` (`[Cs]`, `[Sr]`, `[Rh]`) loses that flag
and RDKit's sanitisation gives it implicit hydrogens it did not have. `eta.h` reconstructs the
saturated hydrogen count in closed form rather than re-sanitising —

    H_S = sum over non-H atoms of ( nH_i + reduction_i + hbonds_i )

where `reduction_i` is the valence given up by that atom's C–C bonds and `hbonds_i` the valence
given up by its bonds to explicit hydrogen ATOMS — and that is exact on **19,997 of 20,000**
molecules, checked against a verbatim port of `AlterMolecule`. The three exceptions are free
metal atoms; two are salts and NaN anyway; the third is the single-atom molecule `[Cs]`, where
mordred's saturated skeleton is `[CsH]` and ours is `[Cs]`:

| column | mordred | ours |
|---|---|---|
| `ETA_epsilon_4` | −5.1 | −10.5 |
| `ETA_dEpsilon_B` | −5.4 | 0.0 |
| `ETA_dEpsilon_C` | 5.48 | 10.88 |

Fixing this properly means reproducing `Atom::calcImplicitValence` — RDKit's charge-adjusted
valence-list walk. That code is C++ and is **not** in `.venv/lib/python3.12/site-packages/rdkit/`,
so writing it would be reimplementing from memory, which house rule 2 forbids. It was left as a
measured miss: 1 molecule in 20,000, and that molecule is a caesium atom.

### 3d. Quirks reproduced (deterministic upstream oddities, same answer every time)

1. **Hydrogen ATOMS live in the "heavy" graph.** `Chem.RemoveHs` keeps an explicit `[H]` that
   defines double-bond stereochemistry, so `[H]/N=C(...)` reaches mordred with a real hydrogen
   among its atoms — **558 corpus molecules (2.8%)**. `mol.GetNumAtoms()` counts it, so
   `ETA_dAlpha_*`, `ETA_psi_1`, `ETA_epsilon_2` and `ETA_eta_B` all divide by 15 and not 14 for
   the example in the header. It also gets a `beta` of its own (0.75 towards the nitrogen, since
   `get_eta_beta_sigma` filters the *neighbour* on `Z != 1`, not the atom) and hence a `gamma` of
   0 rather than NaN. Reproduced; it is the difference between a number and a missing value for
   the whole molecule.
2. **Two atom counts in one formula.** `ETA_dAlpha_A = max((alpha − alpha_R)/A, 0)` where `alpha_R`
   is summed over the reference skeleton (hydrogen atoms dropped, `A_R`) and `A` counts them.
   `ETA_eta_B` mixes `A` with the skeleton's `eta_RL` the same way. Reproduced.
3. **`max(nan, 0.0)` is `nan` in Python** (every comparison against nan is false, so the first
   argument stands) but `std::fmax(nan, 0.0)` is `0.0`. `ETA_dAlpha_*` and `ETA_dPsi_B` restore
   mordred's answer explicitly.
4. **Python's `/` raises rather than returning an infinity.** `ETA_shape_*` divide by `alpha` and
   `ETA_psi_1` by `A·epsilon_2`; a zero denominator is a `ZeroDivisionError` and therefore a
   missing value, not `inf`. Guarded.
5. **Helium would raise, not return NaN.** `get_core_count` divides by `PN − 1`, which is 0 for
   helium. `eta.h` returns NaN there instead, which propagates through *exactly* the same set of
   columns the exception would have failed (every operation downstream is arithmetic) and leaves
   the reference-alkane columns finite, as mordred does. No corpus molecule contains helium; the
   branch is there so the failure mode is the reproduced one if one ever arrives.
6. **`GetBondTypeAsDouble() == Chem.BondType.TRIPLE`** compares a float against an int-valued
   Boost enum, i.e. asks "is this bond's order 3.0". A DATIVE bond is not `SINGLE`, so it takes
   the `y·f` branch with `f = 1`. Both reproduced.
7. **`0.3·4 − 0.5` is 0.7000000000000002, not 0.7**, and mordred adds `0.3` once per hydrogen
   rather than multiplying. `ETA_epsilon_3` for a 14-atom skeleton is `0.4400000000000004`.
   Reproduced with loops, not multiplies.

### 3e. Not a divergence: FP contraction

Written as one expression, `0.3 * Z_v - core` is contracted by clang under plain `-O3` into an
FMA, and the fused result — *more* accurate than the one Python computed — is a different number:
59 of the first 300 corpus molecules got a last-bit different `ETA_epsilon_2`, and eight other
columns moved with it. Every `a ± b*c` in `eta.h` is split into separate statements, which is the
same fix `cpp/verify_wiring.py` uses. Nothing in this file needs `-ffp-contract=off` on the
command line; it is bit-exact under the `-O3` the extension is built with.

---

## 4. `ETA_eta_RL` — the flagged column, and what it actually is

**It is not near-constant, and the correlations are not an artefact of shared zeros. It is
literally the Randić connectivity index, and it is bit-identical to a column in another family.**

Derivation, from the source rather than from the shape of the numbers.
`EtaCompositeIndex(reference=True, local=True)` evaluates

    eta_RL = sum over pairs at distance 1 of sqrt( gamma_i * gamma_j )

on the **reference alkane** — `AlterMolecule` with every heavy atom replaced by a carbon and every
bond by a single bond. On that molecule every `get_eta_*` collapses:

| quantity | on the reference alkane |
|---|---|
| `get_core_count` | `(6−4)/(4·(2−1))` = **0.5** for every atom |
| `get_eta_beta_sigma` | every neighbour is a carbon, so every `|Δε|` is 0 ≤ 0.3 → **0.5 · deg_i** |
| `get_eta_beta_non_sigma` | every bond is SINGLE → **0** |
| `get_eta_beta_delta` | carbon has `NOuterElecs − valence = 0`, never `> 0` → **0** |
| `get_eta_gamma` | `0.5 / (0.5·deg_i)` = **1 / deg_i** |

so

    eta_RL = sum over edges of sqrt( (1/deg_i)(1/deg_j) ) = sum over edges 1/sqrt(deg_i deg_j)

which is the 1975 Randić/connectivity index of the hydrogen-suppressed skeleton. It is a
**genuine rescaling of an existing column**, not a degenerate one:

* range on the corpus 0 … 63.4, mean 16.0, standard deviation 10.0 — nothing like constant;
* **bit-identical to mordred's own `Xp-1d`** (Chi, path order 1, weighted by `d` = sigma electron
  count = heavy degree) on all **19,896** molecules where both are finite: `max |diff| = 0.0`,
  19,896 of 19,896 cells equal. The only difference between the two columns is which molecules
  they refuse: `ETA_eta_RL` sets `require_connected = True` and is NaN on the 104 salts, `Xp-1d`
  is not;
* r = 0.99989 with RDKit's `Chi1`, 0.99959 with `HeavyAtomCount`, 0.99949 with `nHeavyAtom` —
  because for a drug-like skeleton the Randić index is close to `n/2`.

So the ≥0.99 correlations with columns that look mechanistically unrelated are exactly right:
`ETA_eta_RL` *is* one of those columns, arrived at through a different family's vocabulary. The
dedupe decision to drop it is correct and the column set is not changed here.

It survives inside `eta.h` as an **intermediate**, because `ETA_eta_B` and `ETA_eta_BR` are affine
in it (`eta_B = sqrt(2) + 0.5(A−3) − eta_RL`). Which also explains their own correlation
structure: they are `A/2 − Randić` plus a ring term, i.e. a branching residual.

---

## 5. What `bindings.cpp` needs (NOT made — house rule)

```cpp
#include "eta.h"
// ... in the offset enum, after the last existing block:
OFF_ETA = <previous offset> + <previous N_COLS>,
// ... in the per-molecule body:
if (fams & F_ETA) {
  eta::build_from_rows(W.etam, n, nb, ai, N_ATOM_INT, bi, N_BOND_INT, bd, n_rings);
  eta::compute(W.etam, out + OFF_ETA, W.etaw);
}
// ... and in the column-name list:
for (int c = 0; c < eta::N_COLS; c++) out.append(py::str(eta::col_name(c)));
```

`AllWork` gains `eta::Mol etam; eta::Work etaw;`.

* `build_from_rows` reads atom columns `A_Z, A_DEG, A_NH, A_AROM, A_RING, A_TVAL` (it hard-codes
  their indices 0, 1, 2, 5, 6, 9, as `topomisc::build_from_rows` hard-codes its own) and bond
  columns `B_U, B_V, B_CODE, B_BTYPE` (0, 1, 4, 5), plus `bd`, the bond-order doubles.
* `n_rings` is the molecule's ring count — `ring_ptr[q+1] - ring_ptr[q]` in the same CSR span
  `ringcount::compute` is given, i.e. the number RingCount reports as `nRing`. See §3b.
* Column order is the order of `col_name(0..28)` and of `results/dedupe2/agent_groups.json`'s
  `C_eta`. `eta::C_*` enumerators are exported for anything that needs to index one by name.
* The 29 columns need no new boundary field. Nothing is added to `_extract.py`.

## 6. Reproducing the grade

```
.venv/bin/python verify_eta.py                # 20,000 molecules, ~3 min the first time
.venv/bin/python verify_eta.py --invariance   # plus the renumbering / Kekule screen
.venv/bin/python verify_eta.py --limit 300    # quick
```

`verify_eta.py` computes no descriptor value. It parses the SMILES, marshals
`src/hume/_extract.py`'s own boundary arrays into a flat file, emits a small C++ driver into
`build_eta/` (this agent may not add files under `cpp/`, where `cpp/ac.cpp` is the equivalent
standalone for `autocorr.h`), compiles it with `c++ -O3 -std=c++17`, and compares. Ground truth
comes from `.venv-mordred/bin/python` and is cached in `build_eta/mordred_eta.npz`; delete that
file to recompute it.
