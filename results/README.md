# results/ — what is in here and what produced it

Every number in the paper traces to a file in this directory, and every file names the script
that made it. Nothing here is edited by hand; regenerate rather than patch.

---

## The figure contracts

| path | produced by | consumed by |
| --- | --- | --- |
| `figures/downstream_raw.json` | `collect_downstream.py` | the two below |
| `figures/figB/results.json` | `collect_downstream.py` | `figures/src/fig_b.py` |
| `figures/figC/results.json` | `collect_downstream.py` | `figures/src/fig_c.py` |
| `scale/*.json` | `collect_scale.py` | figure C's cost axis, figure D |

`./refresh_figures.sh` runs the whole chain: pull from S3, rebuild the contracts, re-render, and
print what is still missing. It is safe to run at any time and as often as you like.

**Provenance of the raw grid.** `downstream_raw.json` is merged from every
`s3://hume-bench-.../downstream/<instance-id>.json`, keyed per instance so two boxes can never
collide, and deduplicated on `(dataset, arm, fold)` with newest-wins and protocol-beats-timestamp.
A dataset re-run on a second box after a spot reclaim is therefore safe and is how several
datasets in the current grid were actually produced.

## The minimal-spec derivation

| path | produced by |
| --- | --- |
| `minimal/matrices.npz` | `tools/minimal_matrices.py` — repA / repB / adv descriptor matrices |
| `minimal/matrices_meta.json` | sample sizes, seed, `standardize`, rdkit and mol-hume versions |
| `minimal/selection_pooled.json` | `tools/minimal_select.py --pool` — the shipped ordering and coverage curve |
| `minimal/selection_repAonly.json` | the same, derived on the training corpus alone (the comparison that motivated pooling) |

The frozen result lives in the package as `src/molhume/_minimal.py`, not here. Method and
validation: `docs/MINIMAL_SPEC.md`.

## Reanalysis

| path | produced by |
| --- | --- |
| `reanalysis/features/<dataset>.npz` | `tools/cache_features.py` — X, ECFP, y, **stored scaffold folds**, minimal mask |
| `reanalysis/head_sweep.json` | `tools/head_sweep.py` — per (dataset, head, arm) RMSE and per-fold scores |
| `reanalysis/head_sweep_summary.json` | `tools/head_sweep_report.py` |

**The cached features are the point.** Until they existed, changing the prediction head meant a
full EC2 re-run, because `bench_downstream.py` holds features in memory and ships only scores.
With them, "same molecules, same folds, different head" is a minute on a laptop — and the folds
are *stored* rather than recomputed, so a reanalysis is comparable with the grid it is being
compared against rather than approximately so.

## Older / superseded

`dedupe2/`, `dev_grid_v2/`, `scale_rerun/`, `e2e/`, `embeddings/` are earlier stages kept for
traceability. `cost_table.json` and `exactness_vs_mordred.json` back specific claims in
`METHODS.md`.

---

## Regenerating from nothing

```bash
export CHEMPFN_DATA_ROOT=/path/to/chempfn-data      # the lake, outside this repo
./refresh_figures.sh                                # figures from whatever is in S3
.venv/bin/python tools/minimal_matrices.py          # then tools/minimal_select.py --pool
.venv/bin/python tools/cache_features.py esol ...   # then tools/head_sweep.py
```

`.aws-job-resume` carries the live AWS wave: instance ids, what each is running, and how to find
them again from AWS alone.
