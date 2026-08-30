# Is HUME missing descriptors? No.

Asked 2026-08-30, after HUME trailed ECFP4 + full descriptors on classification. The answer
went through two wrong versions before landing; both are recorded below so the numbers are
traceable.

## Conclusion

**Every one of the 864 descriptors the selection keeps is present in HUME.** There is no
implementation gap and nothing to close.

    1,830 RDKit + Mordred union
      ->  1,275 usable       (555 dropped: constant or unusable on the corpus)
      ->    864 kept         (411 dropped: |r| >= 0.99 with a column that WAS kept)
      ->    864 in HUME      (0 missing)

That is exactly the intended design: every RDKit and Mordred descriptor, minus the unusable,
minus anything 99% correlated with something already carried. `data/dedupe.json` is the record.

## The two wrong versions, and why

**v1 claimed 717 missing columns.** Wrong baseline: it compared HUME against the FULL 1,830-column
union rather than against the 864 the filter keeps. Most of the "missing" columns were dropped
on purpose -- 158 per-atom-type EState extremes, ~136 autocorrelations weighted by mass and
intrinsic state, 22 Mordred BCUT variants. Removing them is the deduplication working.

**v2 claimed 9 missing columns** -- `Chi0n Chi0v Chi1n Chi1v Chi2n Chi2v Chi3n Chi3v Chi4v` --
and called it "the real gap". Also wrong, and more embarrassingly: HUME registers RDKit's chi
family in LOWERCASE (`chi0n` ... `chi4v`), and the comparison was case-sensitive. Verified
against RDKit on six molecules: every one matches to a maximum absolute difference of 3e-15.
HUME additionally carries `chi5n/6n/7n`, `chi5v/6v/7v` and `chi_nv_ratio`, which RDKit does not
have at all.

Case-insensitively, **0 of 864** kept descriptors are absent.

## What the 30%-of-gain figure actually measures

The anchor arm (`ecfp_all_desc`) uses all 1,830 columns, deduplicated or not. XGBoost therefore
cuts freely on a column 0.995-correlated with one HUME carries, and the gain lands on a name
HUME does not have. Measured (case-insensitive, five folds):

| | gain in HUME-absent columns |
|---|---:|
| classification (bioavail, ames, pb_bbb, cyp2d6_inh) | ~27-32% |
| regression (pb_ppb, esol, lipophilicity) | ~18-22% |

This measures lost column NAMES, not lost information -- every absent column has a >=0.99 twin
inside HUME by construction. The direct evidence: HUME wins fold 0 of `bioavail` outright
(0.663 vs 0.655) despite 27.5% of the anchor's gain sitting in columns it lacks.

## So why does HUME still trail on classification?

Not because of missing descriptors. Remaining candidates, in order of how much they are worth
checking:

1. **The 0.99 threshold is corpus-dependent.** Two columns can be 0.995-correlated on the
   dedupe corpus and separate cleanly on a specific downstream endpoint. Dropping one is then a
   real loss on that endpoint even though the filter was applied correctly. This is the most
   likely explanation and it is testable: rerun the classification arms with the FULL union and
   see whether the deficit disappears.
2. **Fingerprint radius**, r=3 in HUME against r=2 in the anchor. Weak candidate: bits carry
   only 2-5% of total gain on these datasets.
3. **Dimensionality at fixed n.** HUME is 3,314 columns against the anchor's 2,913, and the
   deficit is concentrated on the smallest datasets.

## Lesson for the harness

Nothing checks that every column `data/dedupe.json` keeps is actually emitted by HUME. That
check would have answered this question in one line instead of three rounds, and would catch a
genuine porting gap if one ever appeared. Worth adding regardless of the answer here.
