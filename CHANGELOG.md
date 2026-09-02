# Changelog

## 0.2.1 — 2026-09-02

- **`molhume.minimal_recovery(columns=None)`** — held-out reconstruction R² per dropped column.
  `minimal_curve()` reports a worst case over 467 columns, which cannot answer "is the
  descriptor I care about safe"; this can. Kept columns are absent rather than reported at 1.0,
  since a kept column is present, not reconstructed. At n=800: median 0.995, 44 below 0.99,
  none below 0.95.
- **`molhume.minimal_gated()`** — the 2 columns excluded from the ranking before it was derived,
  with the reason. **Neither is a dead column and neither is an error in the emitted 1,269**:
  `n5FHRing` is nonzero on 0.78% of benchmark molecules but 0.0008% of the training corpus the
  spec was derived from, so a 24k draw contained none; `MDEC-11` is 52% finite on the benchmark
  corpus (passing the emit gate) and 40% on the training corpus (failing this one). They are
  chemistry the derivation sample under-represents. With these,
  `minimal_columns() | minimal_recovery() | minimal_gated()` partitions `ALL_COLUMNS` exactly,
  and there is a test asserting it.

## 0.2.0 — 2026-09-01

Adds a reduced column specification. No change to any descriptor value, and no change to the
default behaviour of `featurize`.

- **`molhume.minimal_columns(n=800)`** — the `minimal-v1` reduced spec, an ordered ranking of
  all 1,267 eligible columns from which any prefix may be taken. Chosen so every dropped column
  is linearly recoverable from those kept, derived **label-free** by rank-revealing QR on the
  training corpus stacked with an adversarial salts-and-mixtures set. `molhume.minimal_curve()`
  publishes what each `n` costs.
- Measured downstream cost against the full 1,269, as an independent test: **free on ADME
  (+0.29% mean) and classification (−0.12%), about +3.83% mean and +7.33% worst on
  physicochemical endpoints.** Take more than 800 columns for solubility or logP work.
- `minimal-v1` is a frozen contract: `src/molhume/_minimal.py` records the sample, seed,
  `standardize` setting and library versions, and a change to it is a breaking change.

## 0.1.1 — 2026-09-01

Fixes a wrong-answer bug in `FAMILY_OFFSETS` and makes the version constraints much less
aggressive. No descriptor value changes: the fixture regenerates to the same hash.

- **`FAMILY_OFFSETS` was wrong and silently so.** It exported the offsets of the internal
  1,539-column row, while `featurize` returns 1,269 — so `ALL_COLUMNS[FAMILY_OFFSETS["ringcount"]]`
  was `ATS2Z`, a column from a different family, with nothing to indicate a problem. It is now
  `{family: (start, stop)}` half-open pairs into the emitted layout, the 19 families tile
  `[0, 1269)` exactly, and there are tests. Reported by a user slicing by family. The old values
  are still available as `RAW_FAMILY_OFFSETS` for the tools that reason about the pre-dedup row.
- **Python 3.10 is supported.** The `>=3.11` floor was never a property of the code — it came
  from this repo's pinned dev NumPy leaking into published metadata. 3.10 is the lowest
  interpreter RDKit ships across the whole supported RDKit range.
- **The RDKit cap is much looser**: `<2028.01` rather than `<2026.09`, which was the exact
  measured boundary and would have made the package uninstallable the day RDKit 2026.09 shipped.
- **An unsupported RDKit no longer breaks `import molhume`.** The pickle-format guard used to
  raise `ImportError` at import, which also took down `featurize_blocks(reader="api")` — a path
  that reads through RDKit's Python API and is unaffected by the pickle layout. The check is now
  lazy: it fires when a pickle is actually read, names your RDKit, and says what to do.
- Documented that the ~285 us/molecule figure is the **threaded** number. `threads=1` is about
  861 us/molecule on a 12-thread laptop; the table is in the README.

## 0.1.0 — 2026-09-01

First release.

- `molhume.featurize(smiles, ...)` — SMILES (or RDKit `Mol` objects) to one
  `(n_molecules, n_features)` array. 1,269 descriptors per molecule, about 285 us/molecule
  including the fingerprint.
  Column names are not returned, since they are identical for a given set of flags;
  `molhume.feature_names(**flags)` gives them in the same order.
- The default output is the 1,269 descriptors followed by 2,048 ECFP bits. The bits are
  appended *after* the descriptors, so descriptor column indices do not shift when
  `fingerprint` is turned off.
- `standardize` has no silent default: leaving it unset behaves as `"none"` and warns once,
  because what molecule the descriptors describe is the caller's decision.
- `columns`, `additional_descriptors`, `on_error`, `threads`, `fingerprint`, `fp_radius`,
  `fp_size`, `optional`, `batch_size`.
- `mol_hume` is an alias for `molhume`.
- BSD 3-Clause, matching rdkit and mordred.
- Requires `rdkit>=2024.09.1,<2026.09`. The MolPickler blob is read directly and that format is
  not a stable API, so the range is measured rather than assumed. Pickle formats 16.2.0
  (rdkit 2024.09 - 2025.09) and 16.3.0 (rdkit 2026.03) are both supported: the two differ only
  in `AtomMonomerInfo`, which a SMILES molecule never carries, and all 1,269 columns over 8,000
  molecules are bit-identical between rdkit 2025.9.2 and 2026.3.5.
- Released against RDKit 2025.9.2 as the reference; the published wheel reproduces the
  committed fixture bit-for-bit on macOS arm64, and installs and runs against RDKit 2026.3.5.
- Verified against RDKit 2025.9.2 and Mordred 1.2.0 over a 42,000-molecule corpus: 167/186
  RDKit and 412/968 Mordred columns bit-identical, 99.99% and 99.23% of values within 1e-9.
  Divergences are enumerated in `METHODS.md`.
- Wheels for CPython 3.10-3.14 on Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows.
- Values are bit-identical across RDKit releases in the supported range, but **not across
  architectures**: the exactness numbers are from macOS arm64/clang, and x86-64 moves 594
  columns (gcc) or 595 (MSVC) by at most 1.1e-14 of each column's range. NaN patterns are
  identical everywhere. See "Values are not bit-identical across architectures" in the README.

### Regenerating the test fixture

`tests/data/fixture_expected.npz` records what this build produces. When a descriptor value
changes on purpose, regenerate it with `tools/gen_fixture.py` and say here which columns moved
and why. A value that moves without an entry here is a bug.
