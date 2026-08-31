# Changelog

## 0.1.0 — unreleased

First release.

- `molhume.featurize(smiles, ...)` — SMILES to `(fp, X, names)`, 1,269 descriptors per molecule.
- `standardize` has no silent default: leaving it unset behaves as `"none"` and warns once,
  because what molecule the descriptors describe is the caller's decision.
- `columns`, `additional_descriptors`, `on_error`, `threads`, `fingerprint`, `fp_radius`,
  `fp_size`, `optional`, `batch_size`.
- `mol_hume` is an alias for `molhume`.
- BSD 3-Clause, matching rdkit and mordred.
- Verified against RDKit 2025.9.2 and Mordred 1.2.0 over a 42,000-molecule corpus: 167/186
  RDKit and 412/968 Mordred columns bit-identical, 99.99% and 99.23% of values within 1e-9.
  Divergences are enumerated in `METHODS.md`.

### Regenerating the test fixture

`tests/data/fixture_expected.npz` records what this build produces. When a descriptor value
changes on purpose, regenerate it with `tools/gen_fixture.py` and say here which columns moved
and why. A value that moves without an entry here is a bug.
