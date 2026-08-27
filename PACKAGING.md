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

* **No `-march=native`.** It compiles for whatever core is present, so the binary cannot ship. A
  baseline build plus runtime dispatch is the shape; whether `-march=native` is buying anything
  is being measured separately.
* **The LAPACK dependency is the hard part.** `bcut2d` calls Accelerate on macOS and would call
  OpenBLAS or MKL elsewhere; measured, the same code runs 138 µs against 219 µs across those two —
  a 1.6× swing the user cannot see or control, on the block that is ~57% of the total. Whether a
  self-contained eigensolver can remove that dependency entirely is under measurement. **If it
  can, wheels get dramatically simpler** — no BLAS to find, vendor or version-match.
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

Deliberately NOT in milestone 1: the other 683 columns (most have no C++ at all), the surrogate,
wheels, and the public `featurize()` signature from `API.md`. Those come after the bridge exists.

## Milestone 2 — the columns that have no C++

~591 CORE columns (`RingCount`, `PathCount`, `TopologicalCharge`, the chain/cluster Chi variants,
`WalkCount`, `Constitutional`, …) have **no C++ implementation at all**. Until they do, a packaged
HUME either omits them or calls Mordred, and the second option makes the speed claim false.

This is a larger body of work than milestone 1 and should be scoped only once the bridge exists
and its overhead is measured.
