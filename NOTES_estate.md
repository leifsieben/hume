# NOTES — group D_estate (22 columns)

`src/hume_core/estate_ext.h`, verified by `verify_estate.py` over all 20,000 molecules of
`data/dedupe2/corpus.json`.

    SZ                                                     1   Constitutional sum of atomic number
    MAX/MIN {sCH3 ssCH2 aaCH sssCH dssC aasC ssNH dO}     16   per-atom-type E-state extrema
    SRW04 SRW06 SRW08 SRW09 SRW10                          5   log(trace(A^k) + 1)

Specifications read, not remembered: `mordred/EState.py`, `mordred/Constitutional.py`,
`mordred/WalkCount.py`, `mordred/_graph_matrix.py`, `mordred/_base/descriptor.py` (for
`rethrow_na`) at 1.2.0, plus `rdkit/Chem/EState/EState.py` for the index the extrema reduce over.

---

## Headline

**6 of 22 columns are exact in float64 with nothing to qualify. The other 16 are exact only
once `hume_blocks.h`'s per-atom E-state index is accumulated in RDKit's order — and they are then
exact on every cell. The defect is in `estate_from()`, not in this header, and it already affects
the E-state columns HUME ships today.** Details and the measurement are two sections down.

### Against `data/dedupe2/matrix.npz` (all 20,000 molecules)

    column      finite   exact  <=1e-9  <=1e-6  NaN both  NaN dis mismatch
    SZ           20000   20000   20000   20000         0        0        0
    MAXsCH3      14850   14850   14850   14850      5150        0        0
    MAXssCH2     17343   17343   17343   17343      2657        0        0
    MAXaaCH      15699   15699   15699   15699      4301        0        0
    MAXsssCH     13028   13028   13028   13028      6972        0        0
    MAXdssC      14163   14162   14162   14162      5837        0        1
    MAXaasC      15822   15822   15822   15822      4178        0        0
    MAXssNH      10991   10991   10991   10991      9009        0        0
    MAXdO        15111   15111   15111   15111      4889        0        0
    MINsCH3      14850   14850   14850   14850      5150        0        0
    MINssCH2     17343   17343   17343   17343      2657        0        0
    MINaaCH      15699   15699   15699   15699      4301        0        0
    MINsssCH     13028   13027   13027   13027      6972        0        1
    MINdssC      14163   14162   14162   14162      5837        0        1
    MINaasC      15822   15822   15822   15822      4178        0        0
    MINssNH      10991   10991   10991   10991      9009        0        0
    MINdO        15111   15111   15111   15111      4889        0        0
    SRW04        20000   20000   20000   20000         0        0        0
    SRW06        20000   20000   20000   20000         0        0        0
    SRW08        20000   20000   20000   20000         0        0        0
    SRW09        20000   20000   20000   20000         0        0        0
    SRW10        20000   20000   20000   20000         0        0        0

354,011 of 354,014 finite cells bit-identical, 85,986 cells NaN on both sides, **zero NaN
disagreements**, 3 mismatched cells on 2 molecules.

### THIS TABLE FLATTERS THE RESULT AND THE HARNESS SAYS SO

`matrix.npz` stores mordred's answers as **float32**, which throws away about 29 bits of every
value. Graded only against it, all 22 look essentially exact — and 16 of them are not.
`verify_estate.py` therefore runs **live mordred** (`.venv-mordred/bin/python`, only the 22
descriptor objects of this group registered, so it takes 4.5 s) on every 20th molecule and asks
for bit-identity in float64:

    column      finite    exact as-is exact rdkit-ord    max rel
    SZ            1000           1000           1000   0.00e+00
    MAXsCH3        746        185  <-            746   1.15e-15
    MAXssCH2       868        182  <-            868   2.96e-13
    MAXaaCH        772        218  <-            772   9.88e-16
    MAXsssCH       630         64  <-            630   7.34e-13
    MAXdssC        732         45  <-            732   5.03e-13
    MAXaasC        780        133  <-            780   4.53e-14
    MAXssNH        563        184  <-            563   1.19e-15
    MAXdO          783        237  <-            783   9.16e-16
    MINsCH3        746        190  <-            746   1.39e-15
    MINssCH2       868        167  <-            868   2.96e-13
    MINaaCH        772        167  <-            772   2.10e-14
    MINsssCH       630         70  <-            630   6.64e-13
    MINdssC        732         57  <-            732   5.03e-13
    MINaasC        780         81  <-            780   4.17e-13
    MINssNH        563        182  <-            563   1.15e-15
    MINdO          783        240  <-            783   1.23e-15
    SRW04         1000           1000           1000   0.00e+00
    SRW06         1000           1000           1000   0.00e+00
    SRW08         1000           1000           1000   0.00e+00
    SRW09         1000           1000           1000   0.00e+00
    SRW10         1000           1000           1000   0.00e+00

`SZ` and the five `SRW` columns are **bit-identical to mordred in float64 on every molecule**.
The sixteen MAX/MIN columns are not — as shipped they agree on 6% to 30% of cells, with a worst
relative error of 7.3e-13. In the `exact rdkit-ord` column they are **bit-identical on every
finite cell of all sixteen**. That column is the same 22 reductions in this header, run over a
per-atom index accumulated the way RDKit accumulates it. The reduction is exact; the vector it
reduces is not.

`verify_estate.py` exits non-zero on this and prints `NOT FULLY EXACT`.

---

## The divergence, and it is not in this header

`rdkit/Chem/EState/EState.py` accumulates the pair deltas into an array that **starts at zero**,
and adds the intrinsic state **once, at the end**:

    accum = numpy.zeros(nAtoms, numpy.float64)
    ... accum[i] += tmp; accum[j] -= tmp ...
    res = accum + Is

`estate_from()` in `src/hume_core/hume_blocks.h` computes the same terms over the same distance
matrix, but seeds `S[i] = I[i]` and accumulates into it. Same arithmetic, different association.
`I[i]` is O(1) and each delta is O(1/p^2), so folding the large term in first costs low-order
bits on every subsequent add — and an extremum picks a single atom out of the molecule, so it
exposes what the `S<t>` sum columns average away.

**Measured over the same 20,000 molecules, both orders compared against
`rdkit.Chem.EState.EStateIndices` in float64, atom by atom:**

| per-atom index | atoms differing from RDKit bitwise | worst absolute |
|---|---|---|
| `hume_blocks.h` `estate_from()` as shipped | **541,049 of 670,280** (80.7%), on 19,969 of 20,000 molecules | 2.13e-14 |
| the same terms in RDKit's order | **0 of 670,280** | 0.0 |

So this is not a property of the extrema at all: **HUME's per-atom E-state index disagrees with
RDKit's on four atoms in five today**, and restoring RDKit's association makes it bit-identical
on every atom of the corpus. The 79 shipping `S<t>` columns are built on the same vector.

The three cells that survive even the float32 grade are the sharpest form of it: atoms whose
E-state index is *mathematically exactly zero*, where the noise is the entire value.

    MAXdssC    CCN1CC=CNC1=O      ref= 2.220446049e-16   ours= 6.661338148e-16
    MINdssC    CCN1CC=CNC1=O      ref= 2.220446049e-16   ours= 6.661338148e-16
    MINsssCH   CCC(C#C)C(C)C#N    ref=-2.220446049e-16   ours=-6.661338148e-16

### What I did about it, and why

**Nothing, on purpose.** `hume_blocks.h` is outside this group's file boundary, the fix changes
the 79 `S<t>` columns and every other consumer of `BlockWork::ES`, and four other agents are in
this tree. Making that edit mid-flight is the parent session's call, not this agent's.

The change itself is three lines inside `estate_from()`: accumulate the deltas into a separate
zero-initialised array, then add `I[i]` in a final pass. It costs one extra `n`-length vector.
`verify_estate.py` carries the corrected order as a **diagnostic** (`drv <in> <out> rdkitorder`,
reported as the `exact rdkit-ord` column) so the claim can be re-checked in one command; it is
not a second implementation and `estate_ext.h` never computes an index.

I deliberately did **not** work around it by computing a private, RDKit-ordered index inside
`estate_ext.h`. That would make `MAXsCH3` and `SsCH3` disagree about what the E-state of an atom
is, inside one module, to make one table look better. The honest fix is upstream of both.

---

## The quirk: MAX/MIN over an absent atom type is NaN. All sixteen. Never 0.

`AtomTypeEState.calculate` builds `indices` as a lazy filter over the typed atoms and calls the
**builtin** `max`/`min` on it inside `with self.rethrow_na(ValueError)`. An empty filter makes the
builtin raise `ValueError: max() arg is an empty sequence`; `rethrow_na` calls `self.fail(e)`,
which is a `MissingValueException`, and the cell is NaN. The docstring says it outright:
*"returns NaN when type in [min, max] and N_X = 0"*. This is a different code path from `count`
and `sum`, which start from a real zero and legitimately return 0 for an absent type.

All 22 columns of this group therefore split into exactly two conventions:

| columns | absent-type / empty-graph convention |
|---|---|
| `MAX*`, `MIN*` (16) | **NaN** when `N<t> == 0`. Never 0 — this group emits zero `0.0` cells across all 16 columns and all 20,000 molecules. |
| `SZ`, `SRW04/06/08/09/10` (6) | **no NaN path at all.** Both are reductions over an empty set that are genuinely 0: `SZ` is `np.sum` of an array, `SRW` is `log(0 + 1) = 0`. Neither produces a NaN anywhere in the corpus. |

Not a corner case, which is why the harness checks it explicitly. 9,009 of 20,000 molecules (45%)
contain no `ssNH` and 6,972 contain no `sssCH`; a "0 when absent" convention would have produced
a column that grades ~55% exact with the whole failure sitting in one explicable-sounding bucket.
`verify_estate.py` prints, per column, the molecules with `N<t> == 0` from the reference matrix
beside the molecules our implementation marks NaN, and requires them to be the same set:

    MAXssNH    NssNH==0 on  9009 molecules; ours NaN on  9009; agree on 20000/20000
    ...
    -> NaN <-> absent holds for all 16 columns
    (zeros produced by these 16 columns: 0 -- the convention is NaN, never 0)

**One further upstream detail reproduced rather than tidied:** `max(it)`/`min(it)` are CPython's,
so an element replaces the incumbent only on a **strict** `>` / `<`. The loops in `estate_ext.h`
are written that way rather than as `std::max_element`, so ties, signed zeros and any future NaN
in the index resolve the way Python's do. No summation happens in these sixteen columns, so
there is no associativity question *in the reduction* — the association that matters is
upstream, in the index.

---

## Reuse: this header adds almost no arithmetic, and that was the assignment

Every one of the 22 columns is a reduction over a vector this repository already builds and has
already verified. Nothing was re-derived:

| needed | taken from |
|---|---|
| the E-state **type tuple** per atom | `esttyper::typeAtom()` — `src/hume_core/estate_typer.h` |
| the E-state **index** per atom | `estate_from()` via `BlockWork::ES` — `src/hume_core/hume_blocks.h` |
| `trace(A^k)`, k = 1..10 | `topomisc::detail::walkTraces()` — **called, not copied** |
| numpy's pairwise summation | `topomisc::npPairwiseSum()` |

`MAX<t>` is literally `S<t>` with `+=` replaced by a running maximum, over the same atoms the
same typer selects. `SRW04` is a different subscript of the same `tr[]` array that already
produces the shipping `SRW05` and `SRW07`: `walkTraces` computes all ten traces because `TSRW10`
needs them, via `trace(A^2p) = ||A^p||_F^2` and `trace(A^(2p+1)) = <A^p, A^(p+1)>_F`, so these
five columns needed no new matrix work at all. `SZ` is `np.sum` of the identical property array
`topomisc.h` already builds for `MZ` (`MZ` is that sum divided by its length), through the same
numpy-pairwise transliteration.

**The input is `esttyper::Mol` and nothing else.** That struct already carries `z`, `nH` and the
CSR adjacency of the hydrogen-suppressed graph — every input all three families need — so wiring
costs no new marshalling and no second graph build. The hydrogen-added atom count `SZ` needs is
derived from `nH` exactly as `topomisc.h` derives it for `MZ`/`Mv`/`Mp`: `Constitutional` is the
one mordred family that never sets `explicit_hydrogens`, so it inherits the base class's `True`
and sees `Chem.AddHs`; `AddHs` appends and `Constitutional` needs no bonds, so the array is the
heavy atoms in their own order followed by `sum(GetTotalNumHs())` hydrogens.

---

## Timing

`estate_ext::compute()` alone, best of 5 per molecule, Release `-O3`, one thread. The per-atom
index and the distance matrix are **excluded** — both are already paid for by the columns HUME
ships today, so this is the marginal cost of the group.

    stratum            n   mean us        sd       p95
    0-15            4000      0.99      0.25      1.62
    15-25           4000      2.75      0.45      3.33
    25-35           4000      4.33      0.61      5.25
    35-55           4000      6.75      1.24      9.42
    55-1000000      4000     16.27      6.01     29.88
    ALL            20000      6.22      6.04     17.75

**Against HUME's ~830 us per-molecule budget this group adds 0.75%.**

That can be cut to **0.23%** for free, and the header is already built for it. Splitting the
measurement in half (same 20,000 molecules, `walkTraces` timed separately):

    stratum         total      SRW     rest
    0-15             0.98     0.41     0.57
    15-25            2.77     1.55     1.22
    25-35            4.38     2.70     1.68
    35-55            6.78     4.60     2.18
    55-1000000      16.33    12.56     3.77
    ALL              6.25     4.36     1.88

**70% of this group's cost is the `walkTraces` call, and it is work `topomisc.h` already does, on
the same graph, in the same batch, for `TSRW10`.** `compute()` therefore takes an optional
`const int64_t *tr_in`: pass `topomisc`'s own `tr[]` and the walk half of the group costs
literally nothing. It is optional rather than mandatory only because `topomisc::compute()` keeps
`tr[]` local today and this group may not edit that file.

---

## Wiring (described, not done — `bindings.cpp` is not edited, per the contract)

Two lines in the `F_ESTATE` block, immediately after the `esttyper::aggregate()` already there.
`W.em` is filled and `W.bw.ES` is the per-atom index, so nothing new has to be marshalled:

    // AllWork:
    estate_ext::Scratch ex;

    // in the F_ESTATE block, after esttyper::aggregate(...):
    estate_ext::compute(W.em, W.bw.ES.data(), out + OFF_ESTATE_EXT, W.ex);

Four things the parent needs to know:

1. **It must sit inside the `F_ESTATE` guard.** Sixteen of the 22 read `BlockWork::ES`, exactly
   like the `S<t>` columns, and would otherwise reduce over a stale molecule's index. The header
   throws an informative error rather than substituting a value if the index pointer is null.

2. **Column order is the `agent_groups.json` order**, unchanged: `SZ`, the eight `MAX*` in the
   listed type order, the eight `MIN*` in the same type order, then
   `SRW04 SRW06 SRW08 SRW09 SRW10`. `estate_ext::COLS[]` and `col_name()` are the single source
   of that ordering.

3. **`estate_from()`'s accumulation order is a live defect** with a three-line fix and a
   measurement attached (section 2 above). It is worth fixing before these 16 columns ship, and
   fixing it will also move the 79 existing `S<t>` columns — re-grade them in the same change.

4. **The optional performance change**, worth 4.4 us/molecule (0.53% of budget) and nothing else:
   have `topomisc::compute()` write its `tr[]` out through a caller-supplied pointer (it is a
   local `int64_t tr[11]` today, filled and then read for `SRW05`/`SRW07`/`TSRW10`) and pass it
   as `estate_ext::compute(..., W.ex, tr)`. That makes `F_ESTATE`'s output depend on
   `F_TOPOMISC` having run, which is a real coupling — so it is offered, not assumed, and the
   default path stands alone.

---

## Things worth knowing that are not defects

* **`SRW09` is 0 on 6,460 of 20,000 molecules and that is correct.** An odd-length closed walk
  needs an odd cycle, so `trace(A^9) = 0` for every bipartite molecular graph. `SRW04` is 0 on
  exactly 5 molecules, all bond-free salts (`[Cs]`, `S.[Mo]`, `C.[Nb]`,
  `[Ce+3].[F-].[F-].[F-]`, `[O-2].[O-2].[O-2].[Sm+3].[Sm+3]`): no edges, no closed walk of any
  length. Both are graph facts, not missing values, and mordred emits 0 rather than NaN.
* **`log` of an exact integer is the only floating-point operation in the five `SRW` columns.**
  The matrices are integer throughout (`Chem.GetAdjacencyMatrix` is int32 and `An.dot(A1)` stays
  int32); `int64` reproduces that with room to spare. Both sides agree on the integer bit for
  bit, which is why those five are exact in float64 rather than merely close.
* **The eight type names are resolved against `cpp/estate_tables.h` by NAME, once**, not
  hard-coded as row indices. That table is generated from RDKit's own `_rawD` and carries
  upstream's row order; a literal index would silently point at a different pattern if a row were
  ever inserted. A missing name is a hard error naming the affected columns and the regeneration
  command, not a column of NaN.

## Nothing in this group is ill-posed

All 22 are functions of the molecule. The typer is a table lookup plus a graph matching (already
verified per atom against RDKit's `TypeAtoms` by `cpp/verify_estate.py`); the extrema are
order-independent reductions; `trace(A^k)` is invariant under atom numbering; and `SZ`'s only
order-sensitive step is a summation whose order is fixed by RDKit's atom indices, which
`npPairwiseSum` reproduces. There is no Kekulé choice and no numbering dependence to diverge
from, so this group makes **no deliberate divergence at all** — the one disagreement is the
`estate_from()` association above, and that is a defect with a known fix, not a choice.

## Reproducing

    .venv/bin/python verify_estate.py          # dump, build the driver, grade x3, check, time
    .venv/bin/python verify_estate.py --keep   # keep build_estate/*.txt and the mordred JSON

Exits 0 only if all 22 columns are bit-identical to mordred in float64; today it exits 1 for the
reason above. No descriptor value is computed in Python anywhere in it: Python marshals RDKit's
graph to text, shells out to `.venv-mordred/bin/python` for the float64 oracle, reads
`matrix.npz`, and compares. The harness writes and compiles its C++ driver into `build_estate/`
on every run rather than keeping a source file in the repository, because this group's contract
limits which files it may create.

The extension still builds clean with this header present and unwired:

    .venv/bin/cmake -S . -B build_estate -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR="$(.venv/bin/python -c 'import pybind11;print(pybind11.get_cmake_dir())')" \
        -DPython_EXECUTABLE=.venv/bin/python
    .venv/bin/cmake --build build_estate -j4
