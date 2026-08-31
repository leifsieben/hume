# Contract for descriptor-implementation agents

You are implementing a group of missing descriptor columns in C++ for HUME.
Read this file fully before writing code.

## Non-negotiable house rules

1. **Never use Python to compute a descriptor value.** Python may only be used to
   drive tests and to call the reference packages (RDKit/Mordred) for ground truth.
   The shipped implementation is C++ only.
2. **Never reimplement another repo's code from memory.** Read the actual source of
   the reference implementation (Mordred source: `.venv-mordred/lib/python3.12/site-packages/mordred/`;
   RDKit source: `.venv/lib/python3.12/site-packages/rdkit/`) and follow it line for line.
   Those installed packages are READ-ONLY. Do not modify them.
3. **Error messages must be very informative** — state which column, which molecule
   (SMILES), and what was expected vs produced.
4. **Reproduce upstream quirks; diverge only from ill-posed definitions.**
   A *quirk* is a deterministic upstream oddity (same wrong answer every time) —
   reproduce it and note it. An *ill-posed* definition is one whose value depends on
   atom numbering or Kekule choice — diverge, and document the divergence with a
   measurement. Never diverge silently.

## What you deliver

- ONE new self-contained header `src/hume_core/<group>.h` following the exact
  pattern of `src/hume_core/autocorr.h`: an `N_COLS`, a `col_name(int)`, and a
  `compute(...)` taking the already-parsed molecule structs.
- ONE verification script `verify_<group>.py` (test harness only — see rule 1).
- A short `NOTES_<group>.md`: divergences found, quirks reproduced, per-molecule
  timing, and anything you could not make exact.

**Do NOT edit `src/hume_core/bindings.cpp`.** Five agents are working in parallel and
the column-offset enum in that file is the one place a silent transposition could hide.
The parent session wires each header in serially. If you believe you need a change in
bindings.cpp, describe it in your NOTES file instead of making it.

Do not edit any file outside `src/hume_core/<your header>`, your verify script and your
NOTES file. Do not touch other agents' groups.

## Ground truth and the exactness bar

`data/dedupe2/matrix.npz` holds Mordred+RDKit reference values for all 2,023 columns
over the stratified 20,000-molecule corpus (`data/dedupe2/corpus.json`, 5 heavy-atom
strata: 0-15, 15-25, 25-35, 35-55, 55+). Keys: `rd_names`, `md_names`, `X`.
Use it as ground truth rather than recomputing the reference.

Your verify script must report, per column, over all 20,000 molecules:
  - n exact (bit-identical), n within 1e-9 relative, n within 1e-6, n mismatched
  - NaN agreement (reference NaN <-> ours NaN) as a separate count
  - for every mismatch: the SMILES, the reference value, ours
**The bar is exact agreement modulo floating-point associativity.** Report the true
numbers. A column you cannot make exact is a finding to report, not a rounding to hide.

## Build and test

    # use YOUR OWN build dir -- other agents build concurrently in this same checkout
    cmake -S . -B build_<group> -DCMAKE_BUILD_TYPE=Release && cmake --build build_<group> -j4
    .venv/bin/python verify_<group>.py

Report timing with `python -X importtime`-free direct timing: microseconds per
molecule with SD, per stratum. HUME's whole per-molecule budget is ~830 us, so state
what your group adds.

## Interpreters

- `.venv/bin/python`  -- the project venv (rdkit, numpy, the built `hume`). Use this to run
  your verify script and to import `hume`.
- `.venv-mordred/bin/python` -- a SEPARATE venv that has `mordred` installed. Mordred and the
  project venv cannot coexist. If you need to call Mordred live, shell out to this interpreter.
  You should rarely need to: `data/dedupe2/matrix.npz` already holds its values for the corpus.
- There is no bare `python` on PATH. Always use an explicit interpreter path.
