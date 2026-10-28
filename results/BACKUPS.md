# Off-machine backups

Large artefacts that are expensive to recompute and gitignored (embedding matrices, cached
feature blocks) live in S3 rather than in git. This file is the index; without it a later
session has no way to know a backup exists, which is the only failure mode that matters here.

Bucket: `s3://hume-bench-use1-075120018132/backups/`

## 20261028T072438Z

Figure A's embedding matrices after the 2026-10-27 rebuild, plus every JSON contract the
figures read.

| what | detail |
| --- | --- |
| `figA_embeddings/` | 13 arms x 32,676 rows. HUME is the shipped 622-column `default`; ChemBERTa, MoLFormer, SELFIES-TED, MiniMol, CheMeleon, Chemprop and `desc` were all recomputed in full rather than appended |
| `consistency.json` | the model-free direction metric, 11 arms x 16 edits |
| `resolution.json` | the XGBoost AUC metric, untouched and still the reference |
| `pairs.json`, `smiles_index.json`, `background.json` | the pair set, including `protonate_neutral` |
| `downstream_raw.json` | the merged downstream grid at r=4 |
| `figC_results.json`, `SI_figA_results.json`, `figB_results.json` | the figure contracts |

`desc` alone takes 37 minutes to recompute (Mordred at ~14 ms/mol), and the eleven arms
together are several hours plus four separate interpreters, so this is the artefact worth
keeping. The cached downstream feature blocks under `results/reanalysis/features/` are NOT
backed up: they are regenerable from the lake in minutes by `tools/recache_features.py`.

Three `results.json` files collide on basename, so they are stored prefixed by plate. The first
upload did not do that and silently kept only the last one written.

## Still on disk, superseded

`results/reanalysis/features_molhume020_20260908T191229Z/` is the 1.8 GB feature cache built by
mol-hume 0.2.0, kept when it was replaced so the old numbers stayed auditable. It has served that
purpose -- the 714 inf cells it carried are documented -- and can be deleted whenever the disk is
wanted.
