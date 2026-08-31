# mol-hume

Molecular descriptors computed in C++, verified column by column against RDKit and Mordred.

`mol-hume` emits **1,269 descriptors** per molecule in about **285 microseconds**, from a single
call. Of those, 1,109 reproduce descriptors that RDKit or Mordred already define, and 160 are new.
Nothing is computed in Python.

```bash
pip install mol-hume
```

```python
import molhume

X = molhume.featurize(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], standardize="none")
# X -> (2, 1269) float64, ready for a model

xgboost.XGBRegressor().fit(X, y)
```

One array, not a tuple. The column names do not change from call to call, so returning them
every time is something you would unpack and discard; ask for them when you need them, and they
come back in the same order for the same flags:

```python
df = pandas.DataFrame(X, columns=molhume.feature_names())
```

## The one decision you have to make

`standardize` has no safe default, so leaving it unset warns once and tells you the options.
Descriptors are computed on the graph you hand them: a salt, a tautomer and a charge state are
three different molecules, and no library can guess which one you meant.

| value | what it does |
| --- | --- |
| `"none"` | featurize exactly what you supplied |
| `"canonical"` | SMILES round-trip, nothing else |
| `"cleanup"` | RDKit `MolStandardize`: normalize, largest fragment, uncharge |
| a callable | your own `Mol -> Mol` |

Passing `"none"` explicitly is a decision and is silent; omitting it is not, and warns.

## Flags

| flag | default | what it controls |
| --- | --- | --- |
| `standardize` | `"none"` (warns if unset) | what molecule the numbers describe |
| `threads` | `0` | descriptor-block workers; `0` is one per hardware thread. Pass `1` if your own code is already parallel |
| `fingerprint` | `False` | append `fp_size` ECFP bit columns *after* the descriptors, so descriptor column indices never shift. Off by default: this is a descriptor library, and folding 2,048 bits into the same float64 matrix would make the output mostly fingerprint by width |
| `fp_radius` | `3` | ECFP radius |
| `fp_size` | `2048` | ECFP bits |
| `additional_descriptors` | `True` | include the 160 descriptors that are ours rather than RDKit's or Mordred's. Selects what is returned, not what is computed |
| `columns` | `None` | emit only these names, in this order. Combined with `additional_descriptors` by AND |
| `optional` | `None` | expensive columns to compute. `AvgIpc` is on by default, `qed` off |
| `on_error` | `"nan"` | unparseable SMILES: `"nan"` keeps the row and fills it, so the output stays aligned with the input; `"raise"`; `"skip"` drops the row, so it does not |
| `dtype` | `float64` | `float32` halves the memory and is what the boosting libraries convert to internally anyway |
| `batch_size` | `4096` | rows per batch. Affects memory, not values |

`featurize` also takes RDKit `Mol` objects instead of SMILES, which skips a parse.

`molhume.feature_names(**flags)` gives the names for any set of flags; `molhume.ALL_COLUMNS` is
the full list, and `molhume._additional.ADDITIONAL_COLUMNS` the ones that are ours.

`import mol_hume` works too, and is the same module object — the distribution is `mol-hume`, and
`import mol-hume` is a Python syntax error, not something a package can fix.

## What "verified" means

Every column was compared against its upstream definition over a 42,000-molecule corpus spanning
1 to 64 heavy atoms:

- **167 of 186** RDKit columns and **412 of 968** Mordred columns are **bit-identical**.
- **99.99%** (RDKit) and **99.23%** (Mordred) of values agree to within 1e-9.

The remainder are deliberate, documented divergences, not unexplained differences: they are cases
where the upstream definition depends on atom numbering or on a Kekule choice, and therefore has
no single correct answer. Every one of them is listed with a measurement in `METHODS.md`.

### The RDKit range

`mol-hume` requires **`rdkit>=2024.09.1,<2026.03`**, and this is a hard requirement rather than a
preference. The library reads RDKit's `MolPickler` blob directly — a large part of where the
speed comes from — and that format is explicitly not a stable API. Those are the releases that
write pickle format 16.2.0; outside them `mol-hume` refuses to import rather than misparse a
molecule into wrong numbers with no symptom. Widening it is a maintenance task with a
verification step, not a metadata edit; see `MAINTENANCE.md`.

Within that range, values are quoted against **RDKit 2025.9.2** specifically. RDKit's perceived
atom and bond properties drift across releases, so a different RDKit inside the range can still
move values in the last digits.

## Why 1,269 and not 1,539

The implemented set was 1,539 columns. Pairs that carry the same information were removed by a
greedy cover in ascending compute cost: a column is dropped when some cheaper surviving column
predicts it at |Spearman| >= 0.99 on ranks, and that has to hold in **every one of five
heavy-atom strata**, not just on the pooled corpus, so a correlation that only exists because
small and large molecules sit at opposite ends of both scales does not count. Columns that are
NaN more than half the time, or that take one value for 99.9% of molecules, are dropped as
unusable. What survives is 1,269.

## Platforms

Wheels are built for the platforms RDKit itself ships, since a `mol-hume` wheel for a platform
with no RDKit wheel could not be imported:

| | CPython 3.11 - 3.14 |
| --- | --- |
| Linux x86_64, aarch64 | manylinux_2_28 |
| macOS arm64 | 11.0+ |
| macOS x86_64 | 10.15+, needs `rdkit<=2025.9.2` (RDKit dropped Intel Mac after that) |
| Windows x86_64 | MSVC |

No musl, no 32-bit, no PyPy — RDKit publishes none of those. The extension links only the C++
runtime: no BLAS, no RDKit library, and no NumPy ABI, so one wheel works across NumPy 1.x
and 2.x.

## Development

```bash
uv pip install -e . --python .venv/bin/python -c constraints.txt
.venv/bin/python -m pytest tests/
```

The pinned RDKit in `constraints.txt` is the oracle every exactness claim is measured against —
install with `-c constraints.txt` or a bare editable install will silently upgrade it. `tests/`
runs in seconds against a committed fixture; the full exactness verifications against RDKit and
Mordred are the root-level `verify_*.py`, which need the corpus and a second environment. See
`tests/README.md`.

## Acknowledgments

`mol-hume` reproduces descriptors first defined and published by two projects, and would not
exist without either:

- **[RDKit](https://www.rdkit.org/)** — Greg Landrum and contributors. RDKit parses the molecule
  and supplies every perceived atom and bond property this library computes from, and 186 of the
  emitted columns reproduce RDKit descriptor definitions. Several parameter tables here are
  derived from published RDKit values, including the Crippen logP/MR atom-type contributions and
  the Hall-Kier alpha table. BSD 3-Clause.
- **[Mordred](https://github.com/mordred-descriptor/mordred)** — Hirotomo Moriwaki et al.,
  *J. Cheminform.* **10**, 4 (2018). 968 of the emitted columns reproduce Mordred definitions.
  BSD 3-Clause.

Where this library's values differ from either, the difference is deliberate and documented:
those are cases where the upstream definition depends on atom numbering or on a Kekule choice
and so has no single correct answer. Every one is listed with a measurement in `METHODS.md`.

## License

BSD 3-Clause. See `LICENSE`.
