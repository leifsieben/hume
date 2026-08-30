# Why HUME loses to ECFP4 + full descriptors on classification

Asked 2026-08-30.

## CORRECTION to the first version of this note

The first pass compared HUME against the FULL 1,830-column RDKit + Mordred union and reported
717 "missing" columns as a gap. That was the wrong baseline and the alarming conclusion it
produced was wrong.

HUME's descriptor set is the output of a deliberate selection (`data/dedupe.json`):

    1,830 union  ->  1,275 usable   (555 dropped: constant or unusable on the corpus)
                 ->    865 kept     (410 dropped: |r| >= 0.99 with a column that WAS kept)

That is exactly the intended design -- every RDKit and Mordred descriptor, minus the
unusable and minus anything 99% correlated with something already carried.

**Of the 865 descriptors the filter KEPT, only NINE are absent from HUME**, and they are one
family:

    Chi0n Chi0v Chi1n Chi1v Chi2n Chi2v Chi3n Chi3v Chi4v

That is the real gap. It is nine columns, not seven hundred.

## Why the 30%-of-gain figure does not mean what it looks like

The anchor arm (`ecfp_all_desc`) uses all 1,830 columns, deduped or not. So XGBoost is free to
cut on a column that is 0.995-correlated with one HUME carries, and the gain lands on the name
HUME does not have. The measured share of anchor gain in HUME-absent columns:

| | gain in HUME-absent columns |
|---|---:|
| classification (bioavail, ames, pb_bbb, cyp2d6_inh) | 28.6 / 31.1 / 31.5 / 31.8 % |
| regression (pb_ppb, esol, lipophilicity) | 17.6 / 20.3 / 22.3 % |

Both numbers are correct and neither measures lost INFORMATION -- they measure lost column
NAMES, most of which have a >=0.99 twin inside HUME. The direct evidence that the information
is mostly still there: HUME wins fold 0 of `bioavail` outright (0.663 vs 0.655) despite 28.6%
of the anchor's gain sitting in columns it lacks.

What the classification-vs-regression split (30.8% vs 20.1%) does say is that the redundant
families concentrate more of the model's gain on classification endpoints. It is consistent
with, but not evidence for, a real information gap.

## What is actually worth doing

1. **Add the nine Chi columns.** They survived a 0.99 correlation filter, so they carry
   information nothing else in the set does, and `Chi4n` appears in the anchor's top-25 on
   `bioavail`. This is the only defensible "we lost something" item.
2. **Do not add the other 708.** They were dropped on purpose, by the criterion the project
   chose. Re-adding them would undo the deduplication.
3. **`qed` is in ALL_COLUMNS but OFF by default** (NaN unless `optional=("qed",...)`), which is
   the intended treatment for an expensive descriptor that is a function of other descriptors.

## Superseded detail from the first pass

The families absent from HUME relative to the FULL union were: 158 per-atom-type EState
extremes, ~136 autocorrelations weighted by mass and intrinsic state, 22 Mordred BCUT variants,
12 RDKit connectivity indices. All but the nine Chi columns above were dropped by the filter as
unusable or as >=0.99 correlated. Recorded here so the earlier numbers are traceable rather
than silently deleted.
