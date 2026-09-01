# Changelog

## 0.1.0 — unreleased

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
- Verified against RDKit 2025.9.2 and Mordred 1.2.0 over a 42,000-molecule corpus: 167/186
  RDKit and 412/968 Mordred columns bit-identical, 99.99% and 99.23% of values within 1e-9.
  Divergences are enumerated in `METHODS.md`.
- Wheels for CPython 3.11-3.14 on Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows.
- Values are bit-identical across RDKit releases in the supported range, but **not across
  architectures**: the exactness numbers are from macOS arm64/clang, and x86-64 moves 594
  columns (gcc) or 595 (MSVC) by at most 1.1e-14 of each column's range. NaN patterns are
  identical everywhere. See "Values are not bit-identical across architectures" in the README.

### Regenerating the test fixture

`tests/data/fixture_expected.npz` records what this build produces. When a descriptor value
changes on purpose, regenerate it with `tools/gen_fixture.py` and say here which columns moved
and why. A value that moves without an entry here is a bug.
