# mol-hume

Molecular descriptors computed in C++, verified column by column against RDKit and Mordred.

`mol-hume` emits **1,269 descriptors** per molecule in about **285 microseconds**, from a single
call, plus a 2,048-bit ECFP alongside them. Of the descriptors, 1,109 reproduce ones that RDKit
or Mordred already define, and 160 are new. Nothing is computed in Python.

```bash
pip install mol-hume
```

```python
import molhume

X = molhume.featurize(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], standardize="none")
# X -> (2, 3317) float64: 1,269 descriptors then 2,048 ECFP bits, ready for a model

xgboost.XGBRegressor().fit(X, y)
```

One array, not a tuple. The column names do not change from call to call, so returning them
every time is something you would unpack and discard; ask for them when you need them, and they
come back in the same order for the same flags:

```python
df = pandas.DataFrame(X, columns=molhume.feature_names())
```

Pass the same flags to both and the names line up: `feature_names(fingerprint=False)` for the
1,269 descriptors alone, `feature_names(columns=[...])` for a subset.

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
| `threads` | `0` | descriptor-block workers; `0` is one per hardware thread. Pass `1` if your own code is already parallel — but see the timing note below, because it costs about 3x |
| `fingerprint` | `True` | append `fp_size` ECFP bit columns *after* the descriptors, so descriptor column indices never shift when the flag changes. Turning it off saves about 30 us/molecule that cannot be threaded |
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

To take one descriptor family, `molhume.FAMILY_OFFSETS` maps a family name to a half-open
`(start, stop)` into `ALL_COLUMNS` and into the descriptor block of the output:

```python
lo, hi = molhume.FAMILY_OFFSETS["ringcount"]
ring_counts = X[:, lo:hi]                     # 47 columns, n5Ring .. nG12FAHRing
```

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

### About that 285 us

**That is the threaded number**, with `threads=0` (one worker per hardware thread), which is the
default. The descriptor block is the parallel part, so the single-threaded figure is very
different. Measured on a 12-thread M-series laptop, 4,000 corpus molecules:

| | us/molecule |
| --- | --- |
| `threads=0` (default, 12 threads) | 282 |
| `threads=0`, `fingerprint=False` | 247 |
| `threads=1` | 861 |
| `threads=1`, `fingerprint=False` | 846 |

So `threads=1` costs roughly 3x, not 12x — the per-molecule boundary work does not parallelize.
Pass `threads=1` when your own code is already parallel across processes; leave it at `0`
otherwise. Quoting a per-molecule cost without saying which of these it is makes the number
meaningless, so always say.

### The RDKit range

`mol-hume` requires **`rdkit>=2024.09.1,<2026.09`**, and this is a hard requirement rather than a
preference. The library reads RDKit's `MolPickler` blob directly — a large part of where the
speed comes from — and that format is explicitly not a stable API. Outside the range that has
been measured, `mol-hume` refuses to import rather than misparse a molecule into wrong numbers
with no symptom.

Within the range, the pickle format is checked rather than assumed: RDKit 2026.03 writes a
different format version from 2025.09, and it is accepted because 4,000 corpus molecules pickle
to bytes that differ only in the version triple, and all 1,269 columns over 8,000 molecules come
out bit-identical. Widening it for a future release is one command —
`tools/check_rdkit_release.py` — plus, if the blobs really changed, work on the reader. See
`MAINTENANCE.md`.

The upper bound is loose on purpose. It is a courtesy to resolvers — it stops a fresh install
picking an RDKit years newer than anything measured — not a claim that 2027 will work. If the
pickle format does change, `featurize` raises an error naming your RDKit and what to do, the
package still imports, and `featurize_blocks(reader="api")` still works on any RDKit at all,
because it goes through RDKit's supported Python API.

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

## A reduced column set

`minimal-v2` is a 612-column subset of the 1,269:

```python
X = molhume.featurize(smiles, columns=list(molhume.minimal_columns()))
```

It is a **set, not a ranking**. Every column was removed for one of three reasons, and none of
them is a variance threshold:

- **the same physical quantity in different units** — three electronegativity scales, atomic mass
  against atomic number, polarizability against volume. Read from the definitions, because no
  correlation cutoff separates "0.995, same construct" from "0.99, genuinely different";
- **already carried by the ECFP** that ships alongside, *or* a duplicate of a count we already
  emit — 13 `fr_*` flags go for the second reason (`fr_halogen` is `[F,Cl,Br,I]` against
  `nF`/`nCl`/`nBr`/`nI`/`nX`; `fr_Ar_N` is the SMARTS `n`; `fr_bicyclic` is `[R2][R2]`);
- **an exact arithmetic identity** of columns that remain — ring and constitutional counts that
  are sums of others, verified on two chemical spaces.

All the descriptors you would expect are in it: molecular weight, Crippen logP, TPSA, H-bond
donors and acceptors, rotatable bonds, ring counts, Kappa shape, chi connectivity, Labute ASA,
Balaban J, Lipinski, and 62 of the 75 `fr_*` substructure flags.

⚠️ **The `fr_*` flags were dropped in 0.4.0 and restored in 0.5.0**, and the reason is worth
knowing. They were dropped because they are detectable from the ECFP at AUROC 1.000 — but that
figure is conditional on the **corpus**, not just the fingerprint. On a corpus with 5.4% salts
the same measurement gives a median of 0.9929 and a floor of 0.786; `fr_quatN` reads 0.9995 on
one corpus and 0.73 on the other. A 2×2 over radius (2 vs 3) and decoder (logistic vs XGBoost)
moves our median by less than 0.001, so neither explains the gap. Detectability was never sound
grounds for the drop. The 62 are kept on **mechanism**: they encode curated assertions no
structural descriptor derives — that a CYP enzyme attacks here, that a nitrogen is permanently
charged, that a fragment is a toxicophore, which heteroatom sits in a ring, whether a hydroxyl
is aliphatic or aromatic.

### What it costs, measured

Benchmarked against the full 1,269 with the same untuned XGBoost head and the same 5-fold
scaffold splits, on 29 of the 33 grid datasets:

| panel | datasets | mean cost | worst |
| --- | ---: | ---: | ---: |
| ADME & tox | 10 | −1.55% | +2.79% |
| physicochemical | 6 | −0.94% | +1.49% |
| classification | 13 | −0.17% | +1.99% |
| **overall** | **29** | **−0.81%** | — |

Negative means the reduced set scored *better*. **On none of the 29 datasets did the difference
exceed that dataset's own fold-to-fold spread**, and a sign test puts the reduced set behind on
11 of 29 (p = 0.27). So the claim is *no measurable difference at 43% of the columns* — not that
fewer columns help.

For contrast, the retired `minimal-v1` cost **+3.83%** on the physicochemical panel with 800
columns. v2 is smaller and that loss is gone; the difference is what the two cut on.

⚠️ **The quantum panel (`qm8`, `qm9`, `qm9_gap`, `qmugs_gap`) is not yet included**, and it is
the one to watch: the 227-column autocorrelation block was dropped on a physicochemical ablation,
and autocorrelation is a distance-resolved property correlation, which is the kind of thing an
electronic-structure endpoint might lean on. See `HUME_Minimal_definition.md`.

## Platforms

Wheels are built for the platforms RDKit itself ships, since a `mol-hume` wheel for a platform
with no RDKit wheel could not be imported:

| | CPython 3.10 - 3.14 |
| --- | --- |
| Linux x86_64, aarch64 | manylinux_2_28 |
| macOS arm64 | 11.0+ |
| macOS x86_64 | 10.15+, needs `rdkit<=2025.9.2` (RDKit dropped Intel Mac after that) |
| Windows x86_64 | MSVC |

No musl, no 32-bit, no PyPy — RDKit publishes none of those. The extension links only the C++
runtime: no BLAS, no RDKit library, and no NumPy ABI, so one wheel works across NumPy 1.x
and 2.x.

### Values are not bit-identical across architectures

This matters if you are comparing outputs between machines, and not at all if you are fitting a
model. The exactness numbers above were measured on **macOS arm64 with clang**. The same source
on x86-64 moves the last bits: 594 of the 1,269 columns under gcc, 595 under MSVC, with a
maximum disagreement of **1.1e-14 of each column's range**. Nothing structural changes — the
NaN pattern is identical on all three.

That is not a bug that a build flag removes. The library reproduces upstream floating-point
*behavior*, so a different libm's `log` and a different FMA decision are part of the result. CI
measures this on every platform (`tools/platform_drift.py`) and the test suite asserts a bound
on it, exactly rather than approximately on the reference platform.

Beware per-value relative error when you compare: several columns are differences that cancel
to near zero (the centered autocorrelations, `Cyclicity`, `DeltaMean`), where a last-bit wobble
reads as a relative error of 27. Compare against each column's range.

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
