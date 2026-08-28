# HUME — packaging architecture

*How the verified C++ becomes an importable, installable Python package.*

**The problem this solves.** As of 2026-08-27, **842 of 865 descriptor columns are computed in
Python**, and our C++ computes **zero** of them in the actual pipeline. There is no bridge: the
only call to a compiled binary anywhere in the repo is a test harness
(`cpp/verify_crippen.py:194`), and `assemble.py` builds every matrix from
`rdkit.Chem.Descriptors._descList` and `mordred.Calculator`. Everything in `bench.npz`, the dev
grid and Figure A's `desc` arm came out of Mordred.

So the C++ this project has verified to `ALL EXACT` on 98,905 molecules is correct, fast and
**orphaned**. `REPRESENTATION.md`'s "CORE ~59 µs" is a projection of what C++ *would* cost;
measured as actually computed, CORE is **47,710 µs/mol**. Closing that 800× gap is what makes the
paper's central claim true rather than aspirational.

---

## Decisions

### 1. Binding: pybind11, not ctypes, not subprocess, not Cython

`pybind11`, with `scikit-build-core` + CMake as the build backend.

* **Not subprocess/text files.** The current `./hume verify mols.txt` interface writes a 178 MB
  text file for 100k molecules and parses it back. That was right for a verification harness and
  is wrong for a library — it is slower than the arithmetic it feeds.
* **Not ctypes.** Works, but hand-marshalling every array is exactly the surface where a silent
  desync lives, and this project has already lost a day to one (a `nan` written into a text
  export shifted every subsequent field).
* **Not Cython.** Fine tool, but the C++ already exists and pybind11 binds existing C++ with less
  ceremony.
* **nanobind** is smaller and faster, and is the thing to revisit if binding overhead ever
  measures. It is newer, and this is not the place to spend novelty budget.

### 2. Python orchestrates RDKit; C++ computes descriptors

The C++ needs per-atom and per-bond properties — atomic number, degree, implicit H, formal
charge, hybridisation, aromaticity, ring membership, Gasteiger charge, Crippen contributions, bond
orders. Two ways to get them:

| | |
|---|---|
| **(a) link RDKit's C++** | fastest, no boundary — but requires RDKit C++ headers, libs and Boost at build time, pinned to a version. Wheels become very hard, and users would compile from source |
| **(b) Python extracts, C++ computes** ← **chosen** | uses the `rdkit` wheel already required as a dependency. Costs an array-extraction pass per molecule |

(b) is chosen because a pip package that cannot ship wheels is not a pip package. The extraction
cost is real and must be measured, not assumed — it is the same work `cpp/export_predict.py` does
today, minus the text serialisation.

**Batch the boundary.** Extract arrays for N molecules in one pass into flat numpy arrays with an
offsets vector, then make ONE call into the extension. The per-call boundary was measured at
0.106 µs; per-molecule crossing would be affordable, but per-molecule *Python attribute access on
RDKit objects* is not, and batching is what amortises it.

### 3. Layout

```
pyproject.toml
src/hume/__init__.py         featurize(), the public API in API.md
src/hume/_extract.py         RDKit -> flat numpy arrays
src/hume/_columns.py         the frozen column set, per `spec`
src/hume/_core.*.so          the compiled extension
src/hume_core/               C++ sources
  hume_blocks.h/.cpp           computation, no I/O          <- refactored out of cpp/hume.cpp
  bindings.cpp                 pybind11 surface
cpp/                         UNCHANGED -- the verification harness stays a standalone binary
```

**`cpp/` stays as it is.** It is the thing that proved the C++ correct against RDKit and Mordred
on 98,905 molecules, it has its own text interface for that purpose, and breaking it to build a
package would destroy the evidence. The refactor moves *computation* into a header both can call;
`cpp/hume.cpp` keeps its `main()`, its text loader and its verify/bench modes.

### 4. Portability

* **No `-march=native`, and it costs nothing.** It compiles for whatever core is present, so the
  binary cannot ship. Measured on arm64, `-O3` and `-O3 -march=native` produce a **byte-identical
  object file** for `cpp/hume.cpp` — re-checked after the BLAS removal, same sha256 both ways. So
  plain `-O3` is what both the harness and `CMakeLists.txt` use and nothing is being left on the
  table.
* **The LAPACK dependency is GONE.** This was the hard part and it is done. `hume_blocks.h` used
  to bind Accelerate's `dgesv$NEWLAPACK` / `dgemm$NEWLAPACK` by asm label and `CMakeLists.txt`
  refused to configure off Apple, which made HUME a macOS-only package. Two header-only
  replacements removed both call sites:
  * `cpp/eigen_small.h` — BCUT2D's extremal eigenvalues (Householder + Pal-Walker-Kahan QL/QR,
    the same two stages `dsytd2`+`dsterf` perform). Max deviation 4.26e-13 across all 395,620
    Burden matrices, and **25% faster** than Accelerate at molecular sizes.
  * `cpp/lu_small.h` — resistance's per-component solve (reference `dgetf2`+`dgetrs`) and
    `rw_returns`' matrix powers (reference `dgemm`). The solve is ~2.15× **slower** than
    Accelerate's blocked kernel, but BCUT2D's win is larger: the whole pipeline came out
    **faster**, 202 µs/mol against 206 (paired, on a contended machine).

  Nothing links a BLAS any more — proven, not assumed:

  ```
  cmake -S . -B build/nolapack -DCMAKE_FIND_FRAMEWORK=NEVER \
        -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON -DCMAKE_DISABLE_FIND_PACKAGE_LAPACK=ON
  cmake --build build/nolapack           # then: otool -L / ldd on the .so
  c++ -O3 -std=c++17 -o hume cpp/hume.cpp   # no -framework Accelerate
  ```

  Both produce a binary whose only dylibs are `libc++` and `libSystem`, with no BLAS/LAPACK
  symbols undefined. CMake even reports `CMAKE_DISABLE_FIND_PACKAGE_BLAS` as an *unused*
  variable, which is the point: the project no longer looks.
* **What it cost numerically, stated plainly.** BCUT2D moved in the last bits on 795 of 98,905
  molecules (max 1.9e-10 relative, inside its rtol 1e-9). The resistance **bin** columns
  (`RATSC*`/`RPAIR*`) moved on up to 8.1% of molecules — because those columns were never well
  defined in floating point. In exact rational arithmetic the atom pairs that flip sit *exactly*
  on a bin edge, and Accelerate's own two LAPACKs disagree with each other on 9.00% of the corpus,
  more than reference LU disagrees with either. See the note above `resistance()` in
  `src/hume_core/hume_blocks.h`.
* **`cpp/eigen_small.h` and `cpp/lu_small.h` must ship in the sdist.** `hume_blocks.h` includes
  them by relative path from `src/hume_core/`. scikit-build-core decides sdist contents from git,
  so both files have to stay tracked.
* Wheels: `cibuildwheel`, macOS arm64 + x86_64, manylinux x86_64 + aarch64.

---

## Milestone 1 — the bridge, minimal and verifiable

**`hume.featurize_blocks(smiles) -> (X, columns)` returning the 182 verified columns, bit-identical
to `cpp/values_hume.txt`, with no text file anywhere in the path.**

Chosen because it is the smallest change that proves the whole architecture, and because it has an
oracle: those 182 columns are `ALL EXACT` against RDKit and our own modules on 98,905 molecules,
and `values_hume.txt` is a file we already trust. If the packaged path reproduces it bit-for-bit,
the bridge is correct. If it does not, the difference localises to the boundary rather than to the
arithmetic.

**Status: done, with one caveat that has to be stated precisely, because the obvious phrasing of
it is false.** Fed the *same input* — `cpp/mols.txt`, the molecules as the harness sees them —
`hume.featurize_blocks` reproduces `values_hume.txt` **bit-for-bit on all 98,905 rows**. That is
the claim the milestone was after, and the bridge meets it.

Fed the SMILES in `cpp/mols.smi` instead, 63,583 rows differ at `%.12g` in 30 columns. Neither
cause is a bridge defect and both are *input* differences:

* **63,564 are the reference file's own precision.** `cpp/export_predict.py` writes Gasteiger
  charges as `%.10g`, so the text path feeds the C++ a rounded charge. Applying that same rounding
  to the array path collapses the discrepancy to 19 rows.
* **19 are RDKit aromaticity perception on a canonical-SMILES round trip.** Row 19723 has 20 of
  its 24 atoms aromatic in `mols.txt` and **zero** after re-parsing its own canonical SMILES.

The discrepancy profile is byte-identical between the pre-change extraction path and the current
one, so this predates the bulk-extraction work and is not caused by it. It is worth knowing for a
sharper reason than bookkeeping: it is a direct measurement of how much RDKit perception — not our
arithmetic — moves HUME's output, which is why `spec` has to name an RDKit version (see
`constraints.txt` and `API.md`).

Deliberately NOT in milestone 1: the other 683 columns (most have no C++ at all), the surrogate,
wheels, and the public `featurize()` signature from `API.md`. Those come after the bridge exists.

## Milestone 2 — the columns that have no C++

~591 CORE columns (`RingCount`, `PathCount`, `TopologicalCharge`, the chain/cluster Chi variants,
`WalkCount`, `Constitutional`, …) have **no C++ implementation at all**. Until they do, a packaged
HUME either omits them or calls Mordred, and the second option makes the speed claim false.

This is a larger body of work than milestone 1 and should be scoped only once the bridge exists
and its overhead is measured.

---

## What running on Linux taught us — 2026-08-28

Five boot attempts on EC2 (`c7i.4xlarge`, Ubuntu 24.04 x86_64) to run the Figure D benchmark.
Four failed. Every failure is a packaging fact, so they are recorded here rather than in a
benchmark log.

### The portable build works, and this is the first evidence rather than a claim

`hume` compiled and ran on **x86_64 Linux** for the first time:

    envA rdkit 2025.09.2 numpy 2.4.6 torch 2.13.0+cpu
    hume columns 1266
    hume smoke ok, finite cells 1265

That is the whole point of the Accelerate removal recorded above under **Portability** — the
header-only `cpp/eigen_small.h` and `cpp/lu_small.h` replaced the two `$NEWLAPACK` call sites and
the build no longer refuses to configure off macOS. It had never actually been compiled anywhere
else. Now it has, on a different compiler, a different libc and a different ISA.

### 1. The source build needs Python development headers, and says so unhelpfully

On a stock Ubuntu 24.04 image the build dies in pybind11's CMake with:

    Could NOT find Python (missing: Development.Module) (found suitable version "3.12.3")
    Reason given by package: Development: Cannot find the directory "/usr/include/python3.12"

Nothing in that message mentions `python3-dev`, which is what is missing. **Anyone who
`pip install`s from an sdist on a clean Linux box will hit this.**

**Consequence for packaging: ship wheels.** This is the single highest-value action left, and it
is worth more than any further optimisation — a user who gets a wheel never has a compiler, a
CMake, or this error message. Until then, README must say `apt install python3-dev` (or
`python3-devel`) in the same breath as `pip install`.

### 2. `[tool.uv] constraint-dependencies` leaks into every install run from the repo

    [tool.uv]
    constraint-dependencies = ["rdkit==2025.9.2", "numpy==2.4.6"]

This is the pin that stopped the development venv being clobbered three times, and it is correct
for this project. But uv applies it to **any** install invoked from this directory. Building a
second environment for mordred 1.2.0 (which requires numpy 1.x) inside a checkout is therefore
unsatisfiable, and the error blames numpy rather than the config:

    Because you require numpy==1.26.4 and numpy==2.4.6, we can conclude that
    your requirements are unsatisfiable.

`uv pip install --no-config` is the escape hatch. This affects any *user* who clones the repo and
builds an unrelated environment while inside it, so it belongs in the README, not only here.

### 3. HUME has no runtime system libraries, and that is worth saying out loud

The same fleet lost a boot to `ImportError: libXrender.so.1` — chemprop imports
`cuik_molmaker`, whose native extension links X11, absent from every headless server image.
That is not our dependency. But it is the failure mode a compiled Python package usually has, and
**HUME does not have it**: no BLAS, no LAPACK, no X11, no graphics stack, nothing to `apt install`
at run time. Only rdkit and numpy. State that in the README — for anyone deploying into a slim
container it is a real feature, and it is only true because of the Accelerate removal.

### Blockers before `pip install` can work at all

1. **There is no LICENSE file.** Without one the code is all-rights-reserved by default and
   nobody may legally use it — the same objection this project raised against mordred-x in
   `COMPARISON_mordred_x.md`, and it applies here unchanged. Owner's decision; BSD-3-Clause
   matches both RDKit and mordred, which HUME is verified against and distributes tables derived
   from.
2. **`hume` is taken on PyPI** (Hume AI's SDK, which also imports as `hume`). A distribution
   rename is unavoidable, and renaming the *import* module too is the honest choice rather than
   colliding in anyone's site-packages. Free as of 2026-08-28: `hume-descriptors`, `hume-mol`,
   `hume-chem`, `humedesc`, `molhume`.
3. **No CI and no wheel build.** `cibuildwheel` over manylinux x86_64/aarch64 and macOS
   arm64/x86_64, with a smoke test per wheel that imports and featurises one molecule — the same
   assertion the EC2 preflight now makes, for the same reason: importing proves linkage, not
   that a single descriptor is right.

Roughly a day's work, most of it waiting on CI. The two blockers above are decisions, not
engineering.

### Releases are not immutable in the way that matters

PyPI will not let a version be overwritten, but that does not freeze the project: publish
`0.1.1`, `0.2.0`, and users get them with `pip install -U`. A bad release can be **yanked** —
hidden from new resolves while staying available to anything that pinned it. So shipping early
costs little. What is genuinely expensive to change later is the **distribution name** and the
**public API surface**, which is exactly why both are on the blocker list rather than deferred.
