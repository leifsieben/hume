# `vendor/` — third-party code copied into this repo

Nothing in here is original work of `universal-encoder`. Treat it as read-only: fix bugs
upstream, then re-vendor.

## `vendor/chemtfm`

| | |
|---|---|
| **Source** | `/Users/lsieben/VSCode/ChemTFM_OLD/chemtfm` |
| **Upstream commit** | `422872615549d671fae401ef3e29bbe2188b2ee4` |
| **Vendored on** | 2026-08-27 |
| **Vendored by** | mechanical copy; only `feat/descriptor_selection.py` was trimmed (below) |

### Why

Ten scripts in this repo import `chemtfm`, which is not installed in `.venv`. They only ran
under ChemTFM_OLD's own interpreter (see the `PYTHONPATH=` line still visible in
`run_night.sh`). That put a hard runtime dependency on another live checkout on the
**evaluation** half of the pipeline — and per `PLAN.md` downstream RMSE is the scoreboard, so
the scoreboard could not be computed in this repo's own environment at all.

Call sites: `block_run.py`, `build_targets.py`, `cheap_vs_all.py`, `complementarity.py`,
`downstream.py`, `estate_ablation.py`, `eval_surrogate.py`, `family_ablation.py`,
`noise_threshold.py`, `pick_model.py`.

### What the ten scripts actually use

```
chemtfm.bench.metrics                 (as M — only M.rmse is called)
chemtfm.bench.datasets.REGRESSION
chemtfm.bench.splits.scaffold_folds, .train_test
chemtfm.models.xgb.XGBModel, ._DEFAULT_PARAMS   (_DEFAULT_PARAMS: estate_ablation.py only)
chemtfm.feat.descriptors.descriptors  (as rdkit96 — build_targets.py only)
```

### Transitive closure that was vendored

| Module | Pulled in by | Copied |
|---|---|---|
| `bench/metrics.py` | direct | verbatim |
| `bench/datasets.py` | direct, and `models/xgb.py` (`BINARY`, `REGRESSION`) | verbatim |
| `bench/splits.py` | direct | verbatim |
| `models/xgb.py` | direct | verbatim |
| `feat/descriptors.py` | direct | verbatim |
| `feat/parse.py` | `bench/splits.py`, `bench/datasets.py` | verbatim |
| `chem/scaffold.py` | `bench/splits.py` | verbatim |
| `feat/descriptor_selection.py` | `feat/descriptors.py` | **trimmed — see below** |
| `feat/data/descriptors_min95.tsv` | `feat/descriptors.py` (the frozen 96-descriptor list) | verbatim |
| `feat/data/descriptors_selected.tsv` | default arg of `load_selected_descriptors` | verbatim |
| the four `__init__.py` files | package structure | verbatim (root one gained a provenance banner) |

Eight Python modules plus two frozen data files. The closure stopped there because every
vendored module either has no `chemtfm` imports or imports another module already in this
table.

### The one deliberate cut: `feat/descriptor_selection.py`

Upstream is 495 lines and is, by its own docstring, "archaeology": a one-shot *generator*
that timed each RDKit descriptor over a corpus and wrote the frozen `feat/data/*.tsv`
artifacts. Its output is committed; the generator is never invoked here.

`feat/descriptors.py` imports exactly three names from it — `FEAT_DATA_DIR`,
`SHARED_COMPUTATION_GROUPS`, `load_selected_descriptors`. Those three (plus
`SELECTED_DESCRIPTORS_FILE`, the default argument of the third) were copied **byte-identically**,
comments included, and everything else was dropped: `DescriptorMeasurement`, the
cost-benchmark / finiteness / dynamic-range / linear-reconstruction machinery, the tsv writer,
`_main()`/argparse, and the tuning constants only the generator reads.

That cut is what keeps `chemtfm.config` out of the tree: `PoolConfig` was referenced *only*
inside `_main()` (upstream lines 450–468), and vendoring it would have dragged in
`chemtfm/config.py` + `chemtfm/hashing.py` (302 further lines and a feature-policy hashing
scheme this repo does not use) to support a code path that cannot run here.

**Consequence:** the frozen descriptor list can no longer be *re-derived* in this repo, only
read. If a re-derivation is ever needed, run the generator in ChemTFM_OLD.

No behaviour was reimplemented or "simplified" anywhere. Every line that survives is the
upstream line.

### How the imports resolve

`vendor/` is put on `sys.path` (not made a package), so the vendored code keeps its original
top-level name `chemtfm` and **no import statement inside it was rewritten** — it is a
byte-comparable copy, which is what makes re-vendoring a plain `cp`.

Each of the ten call sites gained exactly one line next to its existing `chemtfm` imports:

```python
import _vendor  # noqa: F401  — puts vendor/chemtfm on sys.path
```

`_vendor.py` lives at the repo root and does the `sys.path` insert once per interpreter.
Rejected alternatives: repeating an inline `sys.path.insert` at each site (the chemtfm imports
sit inside per-fold helpers, so an unguarded insert would grow `sys.path` on every call), and a
`.pth` file in `.venv/site-packages` (invisible, and lost whenever the venv is rebuilt).

### Third-party requirements

`models/xgb.py` needs **`xgboost`**, which is **not installed in `.venv`** — see the report /
`vendor/chemtfm/models/xgb.py` docstring. It also does an optional `import torch` first, on
purpose, as a macOS OpenMP guard; `torch` is present.
`feat/descriptors.py`, `feat/parse.py` and `chem/scaffold.py` need `rdkit` (present).
Everything else is `numpy` + stdlib. Upstream deliberately avoids scikit-learn and pandas.
