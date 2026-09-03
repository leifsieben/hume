# mol-hume

Molecular descriptors computed in C++, verified column by column against RDKit and Mordred.

`mol-hume` emits up to 1,269 descriptors per molecule in about 285 microseconds, plus a
2,048-bit ECFP alongside them. 1,109 of the descriptors reproduce definitions RDKit or Mordred
already provide; 160 are new. Nothing is computed in Python.

```bash
pip install mol-hume
```

```python
import molhume

X = molhume.featurize(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], standardize="none")
# (2, 2670) float64: 622 minimal descriptors, then 2,048 ECFP bits

xgboost.XGBRegressor().fit(X, y)
```

`featurize` returns one array. Column names are identical for identical arguments, so they are
available on request rather than returned every call:

```python
df = pandas.DataFrame(X, columns=molhume.feature_names())
```

Pass both functions the same arguments and the names line up with the columns.

## Selecting columns

One parameter, four ways to answer it:

```python
molhume.featurize(smiles, columns="minimal")      # 622, the default
molhume.featurize(smiles, columns="full_no_new")  # 1,109 -- RDKit and Mordred definitions only
molhume.featurize(smiles, columns="full")         # all 1,269
molhume.featurize(smiles, columns=["TPSA", "AvgIpc", "BCUTc-1h"])   # these, in this order
```

`column_set(name)` returns the names in any of the three sets. `ALL_COLUMNS` lists every name a
manual selection can use.

**The selection decides what is computed, not only what is returned.** A descriptor family none
of whose columns are selected is not calculated, and neither are the individual eigensolves of
the `spectral` family. Output is identical either way — `tests/test_families.py` compares every
family and every set against an ungated run, cell for cell — but a narrow selection is cheaper
as well as smaller. On 1,200 molecules of `cpp/hard.smi`, one thread, against 918 us/mol ungated:

| selection | us/mol | |
| --- | ---: | --- |
| `"minimal"` (622) | 762 | 17% faster |
| `"full_no_new"` (1,109) | 908 | within noise |
| `"full"` (1,269) | 900 | within noise |
| `["TPSA", "ExactMolWt", "SLogP"]` | 288 | 69% faster |

The two full sets gain nothing: they request every family, so there is nothing to skip.

### Sets and families

**Column sets** are the three choices above. In the paper's figures, where they appear beside
other methods, they are written `HUME_minimal`, `HUME_no_new` and `HUME_full`; inside the package
the prefix is redundant.

**Families** are the nineteen internal groupings the descriptors are computed in — `autocorr`,
`spectral`, `chi`, `estate`, `constit` and so on. `FAMILY_OFFSETS` maps each to a half-open
`(start, stop)` span:

```python
lo, hi = molhume.FAMILY_OFFSETS["ringcount"]
ring_counts = X[:, lo:hi]                     # 47 columns, n5Ring .. nG12FAHRing
```

A set is what you request; a family is what gets computed. That is why `minimal` is cheaper —
it needs none of `autocorr`, `eta` or `pathcount` — and `full_no_new` is not, since its 1,109
columns touch all nineteen.

### qed

One column is in `ALL_COLUMNS` and in none of the three sets. `qed` costs 69.3 us/mol on its own
(116 structural-alert subgraph searches, the most expensive column here) and is a drug-likeness
score: a weighted geometric mean of eight properties already emitted as columns in their own
right. `full` means every descriptor, not every expense, so it is opt-in:

```python
molhume.featurize(smiles, columns=molhume.column_set("full", extra=["qed"]))
molhume.featurize(smiles, columns=["TPSA", "qed"])
```

`OPTIONAL_COLUMNS` names them. `qed` is appended after every other column, so opting in shifts
nothing: `column_set("full")` is `ALL_COLUMNS[:1269]`.

## Standardization

`standardize` has no safe default. Descriptors are computed on the graph they are given, so a
salt, a tautomer and a charge state are three different molecules and no library can infer which
was meant. Leaving it unset warns once and lists the options.

| value | effect |
| --- | --- |
| `"none"` | featurize exactly what was supplied |
| `"canonical"` | SMILES round-trip, nothing else |
| `"cleanup"` | RDKit `MolStandardize`: normalize, largest fragment, uncharge |
| a callable | your own `Mol -> Mol` |

Passing `"none"` explicitly is silent; omitting it warns.

## Arguments

| argument | default | effect |
| --- | --- | --- |
| `columns` | `"minimal"` | `"minimal"` (622), `"full_no_new"` (1,109), `"full"` (1,269), or a list of names in the order wanted. Decides what is computed as well as what is returned |
| `standardize` | `"none"`, warns if unset | what molecule the numbers describe |
| `threads` | `0` | descriptor-block workers; `0` is one per hardware thread. Pass `1` when the caller is already parallel; it costs about 3x |
| `fingerprint` | `True` | append `fp_size` ECFP bit columns after the descriptors, so descriptor indices do not shift when the flag changes. Off saves about 30 us/molecule that cannot be threaded |
| `fp_radius` | `3` | ECFP radius |
| `fp_size` | `2048` | ECFP bits |
| `on_error` | `"nan"` | a molecule that cannot be parsed **or cannot be featurized**: `"nan"` keeps the row and fills it, preserving alignment with the input; `"raise"` names the molecule and the reason; `"skip"` drops the row. Every molecule is isolated, so one failure never costs another its row |
| `dtype` | `float64` | `float32` halves memory and is what the boosting libraries convert to internally |
| `batch_size` | `4096` | rows per batch. Affects memory, not values |

`featurize` also accepts RDKit `Mol` objects, which skips a parse.

`import mol_hume` returns the same module object. The distribution is `mol-hume`; `import
mol-hume` is a syntax error, which no package can fix.

## Verification

Every column was compared against its upstream definition over a 42,000-molecule corpus spanning
1 to 64 heavy atoms:

- 167 of 186 RDKit columns and 412 of 968 Mordred columns are bit-identical.
- 99.99% (RDKit) and 99.23% (Mordred) of values agree to within 1e-9.

The remainder are documented divergences rather than unexplained differences: cases where the
upstream definition depends on atom numbering or on a Kekule choice and therefore has no single
correct answer. Each is listed with a measurement in `METHODS.md`.

### Timing

The 285 us figure is threaded, with `threads=0` (one worker per hardware thread), the default.
The descriptor block is the parallel part. Measured on a 12-thread M-series laptop over 4,000
corpus molecules:

| | us/molecule |
| --- | ---: |
| `threads=0` (default, 12 threads) | 282 |
| `threads=0`, `fingerprint=False` | 247 |
| `threads=1` | 861 |
| `threads=1`, `fingerprint=False` | 846 |

`threads=1` costs roughly 3x rather than 12x: the per-molecule boundary work does not
parallelize. Use it when the caller is already parallel across processes.

### RDKit version range

`mol-hume` requires `rdkit>=2024.09.1,<2026.09`, and this is a hard requirement. The library
reads RDKit's `MolPickler` blob directly — a large part of where the speed comes from — and that
format is explicitly not a stable API. Outside the measured range `mol-hume` refuses to import
rather than misparse a molecule into wrong numbers with no symptom.

Within the range the format is checked rather than assumed. RDKit 2026.03 writes a different
format version from 2025.09 and is accepted because 4,000 corpus molecules pickle to bytes
differing only in the version triple, and all 1,269 columns over 8,000 molecules come out
bit-identical. Widening the range for a future release is one command,
`tools/check_rdkit_release.py`, plus work on the reader if the blobs did change. See
`MAINTENANCE.md`.

The upper bound is loose deliberately: it is a courtesy to resolvers, stopping a fresh install
from picking an RDKit years newer than anything measured, not a claim about 2027. If the pickle
format does change, `featurize` raises an error naming the installed RDKit and what to do, the
package still imports, and `featurize_blocks(reader="api")` still works on any RDKit, since it
goes through the supported Python API.

Values are quoted against RDKit 2025.9.2 specifically. Perceived atom and bond properties drift
across releases, so a different RDKit inside the range can still move the last digits.

## Why 1,269 and not 1,539

The implemented set was 1,539 columns. Pairs carrying the same information were removed by a
greedy cover in ascending compute cost: a column is dropped when a cheaper surviving column
predicts it at |Spearman| >= 0.99 on ranks, and that must hold in every one of five heavy-atom
strata rather than only on the pooled corpus — so a correlation that exists only because small
and large molecules sit at opposite ends of both scales does not count. Columns that are NaN
more than half the time, or that take one value for 99.9% of molecules, are dropped as unusable.
1,269 survive.

## The minimal set

`minimal-v2` is a 622-column subset, and the default since 0.7.0:

```python
X = molhume.featurize(smiles)                     # the same call
X = molhume.featurize(smiles, columns="minimal")
```

It is a set, not a ranking. Every column was removed for one of three reasons, none of them a
variance threshold:

- **the same physical quantity in different units** — three electronegativity scales, atomic mass
  against atomic number, polarizability against volume. Read from the definitions, since no
  correlation cutoff separates "0.995, same construct" from "0.99, genuinely different";
- **already carried by the ECFP** that ships alongside, or a duplicate of a count already
  emitted. Three `fr_*` flags go for the second reason: `fr_halogen` is `[F,Cl,Br,I]` against
  `nF`/`nCl`/`nBr`/`nI`/`nX`, `fr_Ar_N` is the SMARTS `n`, `fr_bicyclic` is `[R2][R2]`;
- **an exact arithmetic identity** of columns that remain — ring and constitutional counts that
  are sums of others, verified on two chemical spaces.

The expected descriptors are all present: molecular weight, Crippen logP, TPSA, H-bond donors
and acceptors, rotatable bonds, ring counts, Kappa shape, chi connectivity, Labute ASA, Balaban
J, Lipinski, and 72 of the 75 `fr_*` substructure flags.

**The `fr_*` flags were dropped in 0.4.0 and restored in 0.5.0 and 0.6.0.** They were dropped
because they are detectable from the ECFP at AUROC 1.000, but that figure is conditional on the
corpus rather than on the fingerprint. On a corpus with 5.4% salts the same measurement gives a
median of 0.9929 and a floor of 0.786; `fr_quatN` reads 0.9995 on one corpus and 0.73 on the
other. A 2x2 over radius (2 vs 3) and decoder (logistic vs XGBoost) moves the median by less than
0.001, so neither explains the gap. Detectability was never sound grounds for the drop. They are
kept on mechanism: they encode curated assertions no structural descriptor derives — that a CYP
enzyme attacks at a position, that a nitrogen is permanently charged, that a fragment is a
toxicophore, which heteroatom sits in a ring, whether a hydroxyl is aliphatic or aromatic.

### Measured cost

Benchmarked against the full 1,269 with the same untuned XGBoost head and the same 5-fold
scaffold splits, on 29 of the 33 grid datasets:

| panel | datasets | mean cost | worst |
| --- | ---: | ---: | ---: |
| ADME and tox | 10 | -1.55% | +2.79% |
| physicochemical | 6 | -0.94% | +1.49% |
| classification | 13 | -0.17% | +1.99% |
| **overall** | **29** | **-0.81%** | |

Negative means the reduced set scored better. On none of the 29 datasets did the difference
exceed that dataset's own fold-to-fold spread, and a sign test puts the reduced set behind on 11
of 29 (p = 0.27). The claim is no measurable difference at 49% of the columns, not that fewer
columns help.

For contrast, the retired `minimal-v1` cost +3.83% on the physicochemical panel at 800 columns.
v2 is smaller and that loss is gone; the difference is what the two cut on. Full reasoning,
decision by decision, is in `HUME_Minimal_definition.md`.

## Platforms

Wheels are built for the platforms RDKit itself ships, since a `mol-hume` wheel for a platform
with no RDKit wheel could not be imported:

| | CPython 3.10 - 3.14 |
| --- | --- |
| Linux x86_64, aarch64 | manylinux_2_28 |
| macOS arm64 | 11.0+ |
| macOS x86_64 | 10.15+, needs `rdkit<=2025.9.2` (RDKit dropped Intel Mac after that) |
| Windows x86_64 | MSVC |

No musl, no 32-bit, no PyPy: RDKit publishes none of those. The extension links only the C++
runtime — no BLAS, no RDKit library, no NumPy ABI — so one wheel works across NumPy 1.x and 2.x.

### Values are not bit-identical across architectures

This matters when comparing outputs between machines, and not at all when fitting a model. The
exactness numbers above were measured on macOS arm64 with clang. The same source on x86-64 moves
the last bits: 594 of the 1,269 columns under gcc, 595 under MSVC, with a maximum disagreement of
1.1e-14 of each column's range. Nothing structural changes; the NaN pattern is identical on all
three.

No build flag removes this. The library reproduces upstream floating-point behavior, so a
different libm's `log` and a different FMA decision are part of the result. CI measures it on
every platform (`tools/platform_drift.py`) and the test suite asserts a bound, exactly rather
than approximately on the reference platform.

Compare against each column's range rather than per value: several columns are differences that
cancel to near zero (the centered autocorrelations, `Cyclicity`, `DeltaMean`), where a last-bit
wobble reads as a relative error of 27.

## Development

```bash
uv pip install -e . --python .venv/bin/python -c constraints.txt
.venv/bin/python -m pytest tests/
```

The pinned RDKit in `constraints.txt` is the oracle every exactness claim is measured against;
install with `-c constraints.txt` or a bare editable install will silently upgrade it. `tests/`
runs in seconds against a committed fixture. The full exactness verifications against RDKit and
Mordred are the root-level `verify_*.py`, which need the corpus and a second environment. See
`tests/README.md`.

## Acknowledgments

`mol-hume` reproduces descriptors first defined and published by two projects, and would not
exist without either:

- **[RDKit](https://www.rdkit.org/)** — Greg Landrum and contributors. RDKit parses the molecule
  and supplies every perceived atom and bond property this library computes from, and 186 of the
  emitted columns reproduce RDKit descriptor definitions. Several parameter tables here derive
  from published RDKit values, including the Crippen logP/MR atom-type contributions and the
  Hall-Kier alpha table. BSD 3-Clause.
- **[Mordred](https://github.com/mordred-descriptor/mordred)** — Hirotomo Moriwaki et al.,
  *J. Cheminform.* **10**, 4 (2018). 968 of the emitted columns reproduce Mordred definitions.
  BSD 3-Clause.

Where this library's values differ from either, the difference is deliberate and documented:
cases where the upstream definition depends on atom numbering or on a Kekule choice and so has no
single correct answer. Each is listed with a measurement in `METHODS.md`.

## License

BSD 3-Clause. See `LICENSE`.
