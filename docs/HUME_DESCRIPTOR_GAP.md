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

## So why does HUME trail on classification? The 0.99 threshold, on small datasets only.

**First: the gap is two datasets.** With the protocol-2 tuning fix in, HUME against the full
descriptor block on classification:

| dataset | n | delta(1 - auroc), + = HUME worse |
|---|---:|---:|
| pb_ames | 9,139 | **-0.0004** (HUME better) |
| ames | 7,278 | +0.0008 |
| pb_bbb | 8,301 | +0.0022 |
| cyp2d6_inh | 13,130 | +0.0030 |
| hia | 578 | +0.0056 |
| bioavail | 640 | +0.0200 |

On every dataset with 7k-13k molecules HUME is within +/-0.003 of a block costing 50x more.
The lag is `bioavail` and `hia`, both under 700 molecules.

**Second: on those two, the dedupe threshold accounts for it.** `dedupe_cost.py` isolates the
filter and nothing else -- same fingerprint (Morgan r=3), same source columns, same computation,
same protocol-2 head; the arms differ only in whether the 966 dropped columns are present:

| dataset | deduped 864 | full 1,830 | cost of the filter | HUME's gap | explained |
|---|---:|---:|---:|---:|---:|
| bioavail | 0.6806 +/- 0.0217 | 0.6966 +/- 0.0309 | **+0.0160** | +0.0200 | 80% |
| hia | 0.9739 +/- 0.0040 | 0.9798 +/- 0.0029 | **+0.0059** | +0.0056 | 104% |

So the mechanism is: **|r| >= 0.99 is measured on the dedupe corpus, and two columns that are
redundant there can separate on a specific small endpoint.** The filter was applied correctly;
the drop is still a real loss on those two datasets. Nothing about the port, the implementation,
or the fingerprint is involved.

**Read the size dependence, not just the sign.** The filter costs nothing measurable at 7k+
molecules and ~0.016 AUROC at 640. That is what you would expect if the dropped columns carry a
small amount of genuinely independent signal: with enough data the surviving correlated twin
recovers it, and with 128 test molecules per fold it does not.

**Caveat on the evidence.** n = 578 and 640 with five folds, so the SEMs are +/-0.02-0.03 and each
delta is roughly one standard error. Individually weak; the direction agrees across both
datasets and the magnitudes match HUME's independently-measured gap, which is what makes it
persuasive rather than either number alone.

## Options, if this is worth acting on

1. **Leave it.** The cost is confined to datasets under ~1,000 molecules, and it is the price of
   a deduplication that is the whole point of the descriptor set.
2. **Raise the threshold** from 0.99 to, say, 0.995. Keeps more columns, costs inference time,
   and would need re-measuring.
3. **Report it.** A one-line statement that the filter costs ~0.016 AUROC on sub-1,000-molecule
   classification sets and nothing measurable above that is more useful than either change.

## Lesson for the harness

Nothing checks that every column `data/dedupe.json` keeps is actually emitted by HUME. That
check would have answered the original question in one line instead of three wrong rounds, and
would catch a genuine porting gap if one ever appeared. Worth adding regardless.
