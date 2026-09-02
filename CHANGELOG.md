# Changelog

## 0.6.0 — 2026-09-02

**`minimal-v2` now has 622 columns, up from 612 in 0.5.0.** ⚠️ That is the **third** change to the
contents of a spec whose name never changed — 550 in 0.4.0, 612 in 0.5.0, 622 here. **The package
version is the only thing distinguishing them.** Editing a published spec in place stops now:
future changes get a new spec name.

**10 more `fr_*` flags restored, correcting a grouping error rather than a new finding.** 0.5.0
held out 13 flags as "duplicates of counts we already emit". Only three are: `fr_halogen` is
`[F,Cl,Br,I]` against `nF`/`nCl`/`nBr`/`nI`/`nX`, `fr_Ar_N` is the SMARTS `n`, and `fr_bicyclic`
is `[R2][R2]`. The other ten are **functional groups**, the same category as the 62 restored in
0.5.0 — `nS` counts sulfur atoms and cannot separate a sulfide from a sulfoxide from a sulfone,
and sulfones are oxidative metabolites of sulfides.

Restored: `fr_sulfone`, `fr_sulfide`, `fr_SH`, `fr_nitrile`, `fr_C_S`, `fr_alkyl_halide`,
`fr_ArN`, `fr_phos_ester`, `fr_term_acetylene`, `fr_unbrch_alkane`.

72 of the 75 `fr_*` flags are now in the spec. All 15 flags requested by the ChemPFN project are
present — `fr_sulfone` was the one missing in 0.5.0.

## 0.5.0 — 2026-09-02

**`minimal-v2` now has 612 columns, up from 550 in 0.4.0.** ⚠️ The spec *name* did not change,
so **the package version is the only thing distinguishing the two sets** — if you cached
features under 0.4.0, check the installed version. Editing a published spec in place is not the
normal practice here (specs are added, not edited); it was a deliberate call taken while 0.4.0
was about an hour old with no pinned consumers.

**62 of the 75 `fr_*` substructure flags are restored.** They were dropped in 0.4.0 because they
come back from the ECFP at detection AUROC 1.000. That figure turned out to be conditional on
the **corpus**: on a 5.4%-salt corpus the same measurement gives median 0.9929, floor 0.786, and
`fr_quatN` reads 0.9995 on one corpus against 0.73 on the other. A 2×2 over radius (2 vs 3) and
decoder (logistic vs XGBoost) moves our median by under 0.001 — neither explains the gap, so
detectability was not sound grounds for the drop.

They return on **mechanism** instead: curated assertions that no structural descriptor derives —
metabolic liability (`fr_Ndealkylation1/2`, `fr_para_hydroxylation`), permanent ionization
(`fr_quatN`), toxicophores (`fr_nitro`, `fr_hdrzine`), heterocycle identity (our ring counts say
"6-membered aromatic hetero ring" but never which heteroatom), and hydroxyl type (`NumHDonors`
counts aliphatic and aromatic OH identically, though they differ by ~4 pKa units).

**13 stay out**, on the same mechanistic reasoning applied the other way: they duplicate counts
already emitted. `fr_halogen` is `[F,Cl,Br,I]` against `nF`/`nCl`/`nBr`/`nI`/`nX`; `fr_Ar_N` is
the SMARTS `n`; `fr_bicyclic` is `[R2][R2]`. Also out: `fr_ArN`, `fr_C_S`, `fr_SH`,
`fr_alkyl_halide`, `fr_nitrile`, `fr_phos_ester`, `fr_sulfide`, `fr_sulfone`,
`fr_term_acetylene`, `fr_unbrch_alkane`.

Reported by the ChemPFN project, whose fingerprint and corpus differ from ours — the kind of
conditionality a single-corpus measurement cannot reveal.

## 0.4.0 — 2026-09-02

**Breaking: `minimal-v1` is withdrawn and its API is removed.**

- `molhume.minimal_curve()`, `minimal_recovery()` and `minimal_gated()` are **gone**. They
  described a coverage curve for a specification that no longer exists, and keeping them would
  offer users a colinearity analysis we have retired.
- `minimal_columns()` now returns the frozen 550-column `minimal-v2` set and takes no `n`.
  Asking for `spec="minimal-v1"` raises an error explaining why it was withdrawn.

**Why v1 went.** It was an ordering derived by rank-revealing QR on a *linear-recoverability*
criterion: a column could be dropped if the kept ones could rebuild it linearly. That describes
a consumer that does not exist — a depth-6 boosted tree cannot split on a linear combination of
thirty columns — and when that was tested against a deeper tree and an MLP, neither recovered
the loss.

**What v2 is.** 550 columns, chosen only for being the same quantity in different units, already
present in the output, or an exact arithmetic identity. Nothing removed on a variance ranking,
which is what made v1 delete the rare tail. Full reasoning per decision in
`HUME_Minimal_definition.md`; family provenance in `docs/DESCRIPTOR_MAP.md`.

**What it costs**, benchmarked against the full 1,269 with the same untuned XGBoost head and the
same 5-fold scaffold folds, on 29 of the 33 grid datasets:

| panel | datasets | mean | worst |
| --- | ---: | ---: | ---: |
| ADME & tox | 10 | −1.55% | +2.79% |
| physicochemical | 6 | −0.94% | +1.49% |
| classification | 13 | −0.17% | +1.99% |
| **overall** | **29** | **−0.81%** | — |

Negative means the reduced set scored better. **No dataset moved by more than its own
fold-to-fold spread**, and a sign test puts the reduced set behind on 11 of 29 (p = 0.27) — so
this is *no measurable difference at 43% of the columns*, not evidence that fewer columns help.
For contrast, v1 cost +3.83% on the physicochemical panel with 800 columns.

⚠️ **The quantum panel is not yet included.** `qm8`, `qm9`, `qm9_gap` and `qmugs_gap` are still
running. It is the panel to watch: the 227-column autocorrelation block was dropped on a
physicochemical ablation, and autocorrelation is a distance-resolved property correlation, which
is the kind of signal an electronic-structure endpoint might depend on. If it turns out to matter
there, that will be a follow-up release rather than a change to this one.

Also in this release: six conjugation descriptors are now notation-invariant (see 0.3.0 notes
below, which were never published separately).

## 0.3.0 — 2026-09-02

**Six descriptor values change.** `linearity`, `diam_max`, `het_in_max`, `het_frac_max`,
`extra_arom_max` and `sys_max_rings` were not invariant to how a molecule is written: rewriting
the same molecule with a different atom ordering changed them, on roughly 1.2% of molecules.
That is label noise no amount of training data removes, since the same compound from a different
source file got a different number.

- The six descriptors pick a "largest" conjugated pi system. When two systems tie on size, the
  winner was decided by RDKit's atom numbering. A previous fix pinned the sort to `kind="stable"`,
  which made the value **reproducible** (same SMILES, same answer) but not **canonical** (same
  molecule, same answer) — stable sort keeps the last maximal system *in atom order*. The two
  properties were conflated.
- Ties now break on chemical invariants: size, then diameter, then heteroatom count, then
  aromatic-atom count, then the sorted multiset of atomic numbers. The first four pin all six
  outputs; the fifth makes the order total on chemically distinct systems.
- Verified with `tools/notation_stability.py`: all six now move on **zero** cells across 3,000
  molecules x 4 random rewritings, down from 126-179 cells and up to 4.2x their own SD.
- The test fixture is regenerated. Exactly those six columns moved, on 2 of 200 rows.

Also recorded, not fixed: `XATS2`, `XATS4` and `T_sum` remain notation-unstable on ~0.03% of
molecules. The cause is upstream — RDKit's SMILES round-trip is not idempotent for a Z double
bond in a medium ring when the randomized form carries the direction marker on a ring-closure
bond, so the molecule is read back as the E isomer. Reproduced with no mol-hume code involved.
Fixing it would mean overriding RDKit's stereo perception, which is not worth the 0.03%.

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
