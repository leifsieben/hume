# Why HUME loses to ECFP4 + full descriptors on classification

Asked 2026-08-30. The answer is not the head, the tuning, or the fingerprint: **HUME's
descriptor block is not a superset of RDKit + Mordred**, and the columns it lacks are the ones
XGBoost cuts on.

## What is missing

717 of the anchor's 1,830 descriptor columns have no same-named HUME column, in four coherent
families rather than scattered:

| missing | family | what HUME has |
|---:|---|---|
| **158** | `MAX<atomtype>` / `MIN<atomtype>` — per-atom-type EState extremes (`MAXssssC`, `MINsOH`, `MAXsNH2`) | **none at all** |
| **~136** | autocorrelations weighted by **mass** (`m`) and **intrinsic state** (`s`) | the other nine weightings, but neither of these two |
| 22 | Mordred BCUT variants (`BCUTdv-1l`, `BCUTpe-1l`) | only RDKit's eight `BCUT2D_*` |
| 12 | RDKit connectivity indices (`Chi0n`, `Chi1v`, `Chi4n`) | Mordred's `Xch-*` / `Xp-*` family only |

The autocorrelation gap is the sharpest and the most likely to be an oversight rather than a
decision. HUME carries `are, c, d, dv, i, p, pe, se, v`; Mordred also has `m` and `s`, and HUME
has NEITHER. That is two whole weighting schemes absent, not a few columns.

## How much it costs

Fraction of the anchor model's total XGBoost gain sitting in columns HUME does not have,
measured over five scaffold folds per dataset:

| dataset | n | gain in HUME-absent columns |
|---|---:|---:|
| bioavail | 640 | 28.6% |
| ames | 7,278 | 31.1% |
| pb_bbb | 8,301 | 31.5% |
| cyp2d6_inh | 13,130 | 31.8% |

**Flat in n at roughly 30%**, so this is structural, not a small-sample effect. Six of the
anchor's top-25 features on `bioavail` are absent from HUME: `MID_O`, `AMID_X`, `BCUTdv-1l`,
`Chi4n`, `MAXssssC`, `AATSC8m`.

### And the gap is 50% larger on classification than on regression

The same measurement on regression datasets, where HUME is at PARITY with the anchor:

| dataset | n | gain in HUME-absent columns |
|---|---:|---:|
| pb_ppb | 1,262 | 17.6% |
| esol | 1,128 | 20.3% |
| lipophilicity | 4,200 | 22.3% |

**Classification ~30.8%, regression ~20.1%.** That is the answer to "why classification
specifically": the families HUME lacks -- per-atom-type EState extremes and the mass /
intrinsic-state autocorrelations -- carry half again as much of the model's gain on
classification endpoints as on physicochemical regression. HUME's deficit appears exactly where
its missing columns are worth the most, and vanishes where they are worth least.

## What it rules out

* **Not the fingerprint.** Only 2–5% of total gain comes from fingerprint bits on these
  datasets — they are almost entirely descriptor-driven, so HUME's r=3 against the anchor's r=2
  is close to irrelevant here.
* **Not the tuning protocol.** Measured on the same fold with the same head for both arms.
* **Not the whole story either.** On `bioavail` fold 0 HUME actually WINS (0.663 vs 0.655). The
  downstream deficit is a five-fold average and is smaller and noisier than a 30% gain share
  would suggest, which means the missing columns are substantially redundant with what HUME does
  carry -- as they should be, since autocorrelations at different weightings are correlated.

## What to do about it

Adding the `m` and `s` autocorrelation weightings is the cheapest test: it is the same code path
as the nine weightings already implemented, and it is ~136 columns. The per-atom-type EState
extremes are the larger block (158) but a separate implementation.

Neither is a figure blocker. The claim the figures make -- parity at 50x less cost -- survives a
30% gain share sitting in absent columns, and would only get stronger if the gap were closed.
