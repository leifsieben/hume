# Figures A–D: methods, inputs, state, and where things live

Written 2026-08-28 so this work can be picked up without re-deriving it. Every number quoted
here is measured; anything not yet measured is in the TODO and marked as such.

---

## Where everything is

| what | path |
|---|---|
| Figure scripts | `figures/src/fig_{a,b,c,d}.py`, shared style in `figures/src/style.py`, palette in `figures/src/arms.py` |
| Rendered PDFs | `figures/fig_{a,b,c,d}.pdf` (PNG + CSV under `figures/build/`) |
| Fig A pair set | `results/figures/figA/pairs.json`, `smiles_index.json` |
| Fig A embeddings | `results/figures/figA/embeddings/*.npz` (gitignored — large) |
| Fig A resolution scores | `results/figures/figA/resolution.json` |
| Fig B/C downstream grid | `results/figures/figB/grid_all.json` (legacy), new runs land in `s3://hume-bench-use1-075120018132/downstream/` |
| Fig D throughput | `results/scale/*.json`, raw in `s3://hume-bench-use1-075120018132/results/` |
| EC2 logs / status | `s3://hume-bench-use1-075120018132/{logs,status}/` |
| Fleet coordinates | `.aws-job-resume` |

Harnesses: `make_pairs.py` (Fig A pairs) · `embed_pairs.py` (Fig A embeddings) ·
`figa_resolution.py` (Fig A metric) · `bench_downstream.py` (Fig B/C grid) ·
`bench_aws.py` + `bench_scale_e2e.py` (Fig D throughput) · `collect_scale.py` (Fig D collection).

---

## Figure A — does the representation resolve chemical change?

**Metric (changed 2026-08-28).** Held-out ROC-AUC of a tree trained to tell A from A'. For one
edit, label every A molecule 0 and its edited partner 1, fit XGBoost, score on held-out pairs.
0.5 = chance, 1.0 = separates every held-out pair.

**Why not a magnitude.** A tree splits on single dimensions, so resolution means "is there a
dimension that separates them", not "how far did the vector move" — and a magnitude axis
understates a representation where one coordinate is decisive. Counting dimensions fails too:
for a continuous embedding the count moves three orders of magnitude with the threshold
(CheMeleon stereo: 27% of dims at median-of-reference, 0.05% at p95 of background pairs, 0% at
p99.9), and the calibration is degenerate because for a continuous embedding "a different
molecule" IS a random background pair. AUC has no free parameter and is scale-free.

**What 0.5 means.** Not "identical vectors". Two cases give 0.5: identical vectors (a fingerprint
cannot see a re-written SMILES) and vectors that differ in an *inconsistent direction*
(ChemBERTa on stereo — they move, but differently per molecule, so nothing transfers). The
second is invisible to a magnitude axis and is the one that matters downstream.

**Splitting: whole CONNECTED COMPONENTS, not pairs.** Splitting by pair is sufficient only if no
molecule is in more than one pair. 32 of ours are (null_kekulize 20, ez_flip 4, null_enumerate 4,
n_methylation 2, regioisomer 2). With reuse, a molecule lands on both sides and the model
memorises it instead of learning the edit — and **the leak raises AUC**, so it looks like a
better result. Pairs are grouped by union-find under "shares a molecule" and whole components are
assigned. Asserted in `figa_resolution.py`, not assumed. *Found by the CLIMB figures session
running the same metric and asserting what our pilot merely intended.*

**Degenerate pairs are counted, never scored.** Where A and A' have the same vector, both classes
are one point and 0.5 is forced — a fact about the pair set, not about the arm. Measured: ECFP is
degenerate on 969/1000 isotope pairs, 1000/1000 on both nulls, and **121/1000 on stereo**; HUME
on 56/1000 stereo.

**Five split seeds, mean ± SD.** The representations are deterministic, so the split is the only
stochastic element and its spread is the uncertainty of the estimate. This matters: HUME's stereo
lead over ECFP is 0.042 against SDs of 0.014/0.023 — it survives replication, and under the
single-split pilot it did not.

**Edits.** Eleven chemical edits plus `ch2_homolog` (new, replaces `matched_mw` as a panel), plus
two notation controls where a HIGH score is a FAILURE. `matched_mw` is no longer the unit: under
AUC the A/A' label between two unrelated molecules is arbitrary, so it reads ~0.5 and is a sanity
check. Edits must be DIRECTIONAL — an edit whose members are interchangeable reads 0.5 for
everything and measures nothing.

**Inputs.** Benchmark chemotypes → `make_pairs.py` → 15,000 pairs across 15 conditions;
10,000 background molecules reserved and in no pair.

**Measured so far** (5 seeds, component split):

| arm | stereo | E/Z | CH2 homolog | isotope | scaffold hop | nulls |
|---|---|---|---|---|---|---|
| ECFP4 | 0.752 ± 0.023 | 0.957 | 0.799 | 0.502 (969 degenerate) | 0.948 | 0.500 |
| HUME | **0.794 ± 0.014** | 0.965 | 0.766 | **0.624** | **0.969** | 0.500 |

---

**Panel admission, and the free-separability floor.** A panel earns its cell only if it can
distinguish representations. Two ways it fails, and both look like a finding:

* **Nobody clears chance.** Then "ill-posed question" cannot be told from "universally
  unresolved". This is why `matched_mw` is no longer a panel — two unrelated molecules have no
  consistent A/B signature, so every arm sits at 0.50 by construction. It survives as a HARNESS
  CHECK (all arms 0.500–0.513, so the split is not leaking) and `fig_a.py` asserts on it.
* **Everybody hits the ceiling.** A panel at 1.000 for every arm measures the pair generator,
  not the representations.

The `notation` arm is the instrument for the second case: character 1- and 2-gram counts of the
SMILES, no chemistry at all. Whatever it scores on an edit is available to any model for free.
It is an AUDIT ROW in `figures/build/fig_a.csv`, not a bar on the plate.

It exists because the CLIMB figure session found their regioisomer panel reading 1.000 for all
seven arms including an untrained random encoder: they had paired ortho against meta, so every A
lacked a branch and every B had one, and the classifier read "is there a branch". Leif's rule
from it — **an edit must MOVE a feature, never CREATE one**. Audited across our 15 modes: our
`regioisomer` generator moves an existing substituent (49% of pairs have identical character
multisets, and it scores 0.73–0.78, not 1.000), so we do not have their defect. `n_methylation`
does add a branch in 94% of pairs and reads 0.997–0.998 for every arm — but that is inherent to
the chemistry rather than an artefact of the template, which is exactly why `ch2_homolog` earns
its place beside it: homologation inserts CH2 into a chain and creates nothing.

---

## Figure B — does each block earn its place downstream?

Two panels with a changing reference block: (1) do descriptors carry information ECFP lacks;
(2) does any pretrained model add anything on top of ECFP+descriptors.

**Contract** (`figures/src/fig_b.py`): `bases` × `adds`, where a record's features are
`base ⊕ add` and `add: null` is the block alone. **Panel 2 has never been run** — no pretrained
arm appears anywhere in `grid_all.json`.

**Protocol.** 28 DEV datasets from the ChemPFN lake, 5 scaffold folds (seed 0), capped at 50,000
molecules, metrics rmse / auroc / acc — all matching `grid_all.json` so new arms are comparable.

**The 12 concatenations are generated, not typed** (`FIGB_BASES` × `FIGB_ADDS` in
`bench_downstream.py`): 3 bases (`ecfp`, `desc`, `ecfp_all_desc`) × 4 embeddings
(`chemeleon`, `minimol`, `chemberta_mtr`, `molformer`), named `base__add`. A hand-written list
is where `ecfp_chemeleon` quietly becomes `desc_chemeleon`.

**`grid_all.json` cannot be reused.** Its 840 records have no `w` field — they predate
`feature_weights` — so they do not satisfy the methods below and the whole grid is being
re-measured.

**Fleet hardening (2026-08-28), each item a failure that had already happened or was one boot
away:**

* `shutdown -h +330` against a ~14 h grid would have killed dsA 40% through. The cap is now an
  argument to `launch_ds.sh`, sized to the work (20 h for the full 28-dataset grid).
* `grid.json` was uploaded only after the LAST dataset, so a killed box took every completed
  dataset with it. It now ships to `downstream/<iid>.partial.json` every 30 s alongside the log.
* The DONE marker read `"complete": len(r) > 0` — a box killed after one dataset wrote a
  completion marker. It now names the missing arms and datasets explicitly.
* `minimol` failed to build (`torch-sparse` imports torch in its own setup.py without declaring
  it) and the line ended in `|| echo "(minimol unavailable)"`, so a 5-arm box would have produced
  a 4-arm grid and looked complete. Now `--no-build-isolation`, and a hard failure when minimol
  is a requested arm.
* The bundle carried the ChemBERTa weights at its ROOT while `embed_pairs.py` looks under
  `models_hf/`, and had no MolFormer at all. Both fixed; every learned arm on dsB would have
  died after the classical arms were paid for.
* The preflight now calls `BD.ARMS[arm](['CCO','c1ccccc1'])` for EVERY requested arm. Two boots
  have already been lost to gates that tested a path the run does not take.

`launch_ds.sh` + `queue_runner.sh` (in the session scratchpad) hold the queue: the account cap is
32 Standard vCPUs and each box is 16, so at most two run at once and the rest launch as headroom
appears.

**Head: untuned XGBoost, with one documented exception.** Everything is left at a fixed default
so a difference between arms is a difference between representations. The exception is
`feature_weights` (API.md §7): fingerprint bits get weight `w`, descriptors 1, and `w` is tuned
in an INNER CV loop on training folds only — tuning it on the test fold leaks and the effect is
large enough to matter. `colsample_bynode=0.3` is part of the mechanism, not a tuning choice:
`feature_weights` is inert unless some `colsample_by* < 1`. Arms with no fingerprint/descriptor
boundary skip it (it is a no-op there) and cost one fit instead of five.

---

## Figure C — performance against inference cost

Same grid supplies the performance axis; the cost axis comes from Figure D.
Arms: ChemBERTa, CheMeleon, MiniMol, ECFP, ECFP+RDKit, ECFP+Mordred, ECFP+both, HUME.

---

## Figure D — what it costs to featurise a billion molecules

Three panels: (A) is the extrapolation legitimate — µs/mol vs N, flat means multiplying to 1e9 is
licensed; (B) wall clock for 1e9 by hardware budget; (C) USD at public on-demand prices.
`is_flat()` refuses to draw any arm in B or C that A did not show flat within 25%.

**Measured on EC2**, N as a STRIDE over 1M PubChem SMILES (the corpus is size-ordered, so a
prefix manufactures a scaling bend), µs/mol at 1e4/1e5/1e6:

    c7i.4xlarge, 16 vCPU        g6.xlarge, 1x L4, 4 vCPU
      ecfp       20.7 19.0 18.9   chemberta  104.8 106.9 112.5
      hume      125.6 123.7 124.1  chemprop   572.9 606.7 604.2
      chemprop  149.2 141.4 140.1  chemeleon 1030.1 1042.8 1046.7
      chemberta 364.6 344.5 340.4
      chemeleon 2892.1 2892.7 2892.5
      mordred   6172.1 6189.5 (1e6 pending)

Every arm flat. HUME 1.5% across 100x; CheMeleon 0.02%.
**chemprop on the GPU box is 4.3x SLOWER than on the CPU box** — 581.4 µs/mol of RDKit
featurisation against 22.7 of forward pass, 96% prep, GPU idle. The GPU pays in proportion to
arithmetic per molecule.

---

## TODO

**Figure A**
- [ ] Re-embed 7 learned arms against the new pair set (running); then `desc` via the Mordred venv
- [ ] Run `figa_resolution.py` over all arms
- [ ] Rewrite `fig_a.py` to the AUC axis: y from 0.5 to 1.0, no shaded null band, drop the
      matched-MW panel, add `ch2_homolog`, error bars from the 5 seeds, report degenerate counts
- [ ] Decide whether sub-0.5 cells render as zero-height with an explicit label

**Figure B**
- [ ] dsA (classical arms) and dsB (learned arms) to finish
- [ ] **Third box: the 12 base×add CONCATENATIONS** — 3 bases × 4 adds. Not currently scheduled;
      without them Panel 2 stays empty. Featurisation is cached per arm, so this is concatenation
      plus 12 × 28 × 5 fits, not a re-featurisation
- [ ] Write the aggregator: per-fold records → the `results.json` contract (mean/sem/n_folds)

**Figure C**
- [ ] **Three missing cost measurements**: `minimol`, `ecfp_rdkit_desc`, `ecfp_mordred_desc`.
      `ecfp_all_desc` can take the Mordred arm's figure (same work)
- [ ] Assemble `results/figures/figC/results.json`

**Figure D — DONE.** All six arms measured at 1e4/1e5/1e6; `figures/fig_d.pdf` regenerated
2026-08-28 from `results/scale/*.json` with prices pulled the same day.

**Cross-cutting**
- [ ] `fig_b.pdf` and `fig_c.pdf` on disk are NOT reproducible — their `results.json` was never
      committed. Both are regenerated from the new grid rather than trusted
- [ ] r=4 downstream: is it a better HUME default? **The old answer was an artefact of the old
      metric.** The magnitude-ratio axis said r=4 > r=3 > r=2 on every substitution edit, and we
      wrote up the resulting "resolution improves, downstream worsens" dissociation as a result in
      its own right. On the AUC axis the ordering REVERSES — r=2 leads on ch2_homolog
      (0.799/0.786/0.782), regioisomer (0.776/0.768/0.748) and stereo (0.752/0.751/0.744), and the
      three are within noise on ring_contract. The first downstream cell agrees (ames auroc 0.853
      / 0.848 / 0.838). So resolution and downstream now point the SAME way, the dissociation
      claim is withdrawn, and the default stays r=2 unless the full grid overturns it
