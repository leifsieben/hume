# HUME_minimal v2 — handover

*Written for the ChemPFN side, whose `DESCRIPTOR_SELECTION_METHOD.md` this work implements and
in three places revises. Delivered as a file because the peer session had exited.*

---

## What it is

```bash
pip install --upgrade mol-hume        # 0.4.0
```
```python
X = molhume.featurize(smiles, columns=list(molhume.minimal_columns()))   # 550 of 1,269
```

## Breaking, if you used minimal-v1

- `minimal_curve()`, `minimal_recovery()`, `minimal_gated()` are **removed**, not deprecated.
- `minimal_columns()` takes no `n` — v2 is a **set**, not a ranking, so there is no prefix.
- `minimal_columns(spec="minimal-v1")` raises with an explanation.
- **Cached features from v1 must be recomputed.** The sets differ in size (800 → 550) *and*
  membership, and there is no migration path because the criteria are incompatible.

## Why v1 went — the part relevant to your method document

Your framing was right and is kept: coverage rather than compression, label-free, pivoted QR
rather than correlation clustering. What did not survive is **linear recoverability as the
acceptance criterion**. It describes a consumer that does not exist. The prediction that a head
which *can* form linear combinations should recover the loss was stated, then tested:

| head | cost of the 800-column set, 6 physchem datasets |
| --- | ---: |
| XGBoost depth 6 (the grid's head) | +4.63% |
| XGBoost depth 10 | +4.12% |
| XGBoost `colsample_bynode=1.0` | +3.09% |
| ridge | +1.37% |
| MLP (512, 128) | +3.20% |

Neither a deeper tree nor an MLP recovers it. A depth-6 tree cannot split on a linear combination
of thirty columns, so *recoverable* and *recovered* come apart.

## Three findings from your method worth having

**1. In-sample R² is not merely optimistic here, it is meaningless.** The kept set is numerically
singular (condition number ~1e15 by k=512). Fitting the reconstruction on one 24k draw and
scoring it on a **disjoint draw from the same corpus**: in-sample 0.991 at k=800, held-out
**−1.1e20**. Any k chosen from in-sample R² alone — including 640 — reports a quantity that does
not survive a second sample. A ridge penalty of 0.01·n makes the question askable.

**2. Deriving on drug-like molecules alone breaks silently on salts.** An ordering from the
training corpus alone closes coverage at k=704 and leaves `Phi` and `Kappa2` at **R² = 0.07** on
`cpp/hard.smi`. Both descend from `HallKierAlpha`, whose table is solved over the
(element, hybridisation) pairs the training corpora happened to contain. Derive on the pooled
sample; do not derive on one and check against the other.

**3. Desideratum 7 (rare-but-decisive) does not bite for QR, and the reason matters.** QR ranks by
residual **orthogonality**, not variance. Columns firing on ≤2% of molecules got median rank
**71** of 1,267, against **704** for columns firing on >50%; 64 of 70 were kept. A rare binary
flag is nearly orthogonal to everything, so QR grabs it early. **That protection is free with QR
and would be lost by any family-representative scheme that picks representatives by typicality or
variance.** It is a constraint on any successor.

## What v2 was built on — three criteria, none a variance threshold

**Same quantity, different units** — read from the definitions, not inferred from correlation.
Eight of the twelve autocorrelation weights are pure periodic-table lookups; their effective rank
over the elements that actually occur is **1.76** (axis 1 electronegativity-against-size, axis 2
nuclear mass). Kept `Z`, Pauling EN and vdW volume, plus the four environment-dependent weights
(`c`, `d`, `dv`, `s`). **Pauling over Sanderson on coverage — 94 elements against 56** — which is
the reverse of what QR chose, because QR optimised on a corpus containing no transition metals.

**Already in the output** — the 75 `fr_*` substructure flags come back from the ECFP that ships
alongside them at detection AUROC **1.000**. ⚠️ Our first version of that test used R² and was
wrong: for a column that is zero on 99.95% of molecules, R² measures rarity, not recoverability.
No other family is droppable this way — E-state, ring counts and constitutional counts all have
near-perfect *detection*, but a fingerprint carries no **magnitude** (`HeavyAtomCount` comes back
at R² 0.863 because a binary bit vector cannot count).

**An exact arithmetic identity** — ring and constitutional counts that are exact sums of others,
verified on two chemical spaces (rank says 23 determined on the benchmark corpus alone, 22 on
salts alone, but only **21 stacked**; the extras are corpus artifacts) and confirmed with the real
consumer, which rebuilds them at median R² 1.0000 against a linear model's 0.9971.

⚠️ **One cut is weaker and is flagged as such.** The 227-column autocorrelation block was dropped
because removing it cost nothing measurable — *not* because anything can rebuild it. Nothing can:
it is the least reproducible family in the library and has full exact rank (227 of 227).

## What it costs

29 of 33 grid datasets, same untuned XGBoost head, same 5-fold Murcko scaffold folds:

| panel | datasets | mean |
| --- | ---: | ---: |
| ADME & tox | 10 | −1.55% |
| physicochemical | 6 | −0.94% |
| classification | 13 | −0.17% |
| **overall** | **29** | **−0.81%** |

Negative means the reduced set scored better. **No dataset moved by more than its own
fold-to-fold spread**; sign test 11 of 29 worse, p = 0.27. So: *no measurable difference at 43% of
the columns*. For contrast, v1 cost **+3.83%** on physchem with 800 columns.

⚠️ **The quantum panel (`qm8`, `qm9`, `qm9_gap`, `qmugs_gap`) is still running and is not in those
numbers.** It is the one to watch: autocorrelation is a distance-resolved property correlation,
which is the kind of signal an electronic-structure endpoint may lean on. If it fails there, that
is a follow-up release.

## Where the detail lives

`HUME_Minimal_definition.md` — every decision with its evidence.
`docs/DESCRIPTOR_MAP.md` — what the 1,269 actually measure, by reading rather than correlating.
