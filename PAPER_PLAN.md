# Paper plan — what lands today, what does not

Written 2026-08-26 09:45. Honest confidence, not optimistic scheduling.

---

## State right now

| thing | status |
|---|---|
| 1M corpus + 56k benchmark | **done**, packed, verified bit-identical on all 1M |
| CORE/PREDICT split, fingerprint conventions | **frozen** — ECFP r=2 counts, R3cFP r=3 counts, each with its own proxy |
| descriptor blocks (resistance/cycles/conjugation/stereo/chi) | **built and measured**; nothing dropped, effect bounded < 0.6% |
| proxy model ladder | **training now** — ridge 0.9505, linquad 0.9593, pinet running; GNN on a T4 at **27 s/epoch**, done in ~6 min |
| checkpoints | ridge only so far; the rest land with the current run |
| LOCKED suite | **reachable** — 51 datasets, 16 LOCKED / 35 DEV, local at `~/chempfn-data` |
| Figure A/B methodology | **written** (`FIGURES.md`), rebuilt around the ratio metric after review |

**The architecture is done.** What is not done is *which model wins*, which is what today's runs decide.

---

## Can you write the paper today? Partly. Here is the split.

### Achievable today — high confidence

**Figure D (cost).** Almost entirely measurement, and most of it is already done: parse 76 µs,
ECFP 39 µs, r3fp 48 µs, CORE 59 µs, Chi 6.8 µs C++, cycles 2.2, resistance 5.8. What is missing
is the *learned* models' throughput on GPU and CPU, which is one script per model. The
extrapolation to 1B/10B is arithmetic on measured points. **Half a day, and it is the figure
least likely to surprise us.**

**Proxy model selection.** Five arms finish today. Reconstruction R² per family, plus the
four-arm downstream comparison on the 34 cached benchmark datasets.

**OOD evaluation.** Tanimoto distance to nearest corpus neighbour, binned. Cheap — the corpus
excludes benchmark structures, so this is clean rather than contaminated. Add a **size bin**
too: the corpus is capped at 5-60 heavy atoms and 1.17% of benchmark molecules exceed it.

### Achievable today — medium confidence

**Figure A (resolution), partial.** The arms we already hold — ECFP, r3fp, descriptors,
CheMeleon, MiniMol, Chemprop, CLIMB — can be embedded and scored today. The pair construction
(13,000 pairs across 10 edits + 2 notation controls + the matched-MW reference) is a few hours.
**The three we do not hold are the schedule risk**: SMI-TED needs a HuggingFace download,
Uni-Mol needs `unimol_tools` plus conformers, Kulik needs `hjkgrp/SpectralScore` plus conformers.
Any of those can eat a day on environment problems alone.

**Figure B (redundancy), on a subset.** The statistics are cheap; the *compute* is not. Two
panels x ~10 arms x 16 LOCKED datasets x 5 folds, with `wong_*` at 39k rows each and
`moleculeace` at 48k. That is hundreds of XGBoost fits. On this Mac, competing with training,
it does not finish today. **On a dedicated CPU box it does.**

### Not today

* **Figure C** at full scope. It is Figure B's machinery plus HUME plus the best CLM and GNN,
  and it inherits every dependency above.
* Packaging, pip, docs — explicitly deferred, and correctly so.
* R3cFP proxy — the second model is trained *after* r=2 decisions are settled, by design.

---

## What another box buys, and which one

The binding constraint on B and C is XGBoost CV throughput, which is CPU-bound and embarrassingly
parallel over (dataset, fold, arm). This is exactly the shape that a large CPU box collapses.

    c7i.24xlarge   96 vCPU   ~$4.28/h on-demand   ~$1.50/h spot
    -> ~200 XGBoost fits/hour against maybe 20 on the contended Mac

**Note the quota.** The G/VT on-demand quota is fully consumed by `chempfn-gpu-job` and
`climb-figA-repair` and the P quota is 0, but `c7i` is a *standard* instance and draws on a
different quota entirely, with the Standard Spot quota at 32 vCPUs. A c7i.8xlarge (32 vCPU) fits
spot exactly; anything larger needs on-demand or a quota bump.

**Proposed fleet for today:**

| box | job |
|---|---|
| this Mac | finish the 5 proxies, OOD binning, Figure D CPU timings |
| current T4 spot | GNN now; then CLM/GNN embeddings for Figures A and D |
| **new c7i (32-96 vCPU)** | **Figure B + C: the whole LOCKED x arms x folds grid** |
| optional 2nd GPU spot | SMI-TED + Uni-Mol setup and embedding, in parallel with the above |

---

## Ordered plan

### Track 1 — finish the proxy (this Mac, ~2 h)
1. Five arms complete, checkpoints for all.
2. Per-family reconstruction R² — never pooled.
3. Four-arm downstream on the 34 cached datasets: `ecfp` / `+core` / `+predicted` / `+exact`.
4. **Pick the architecture.** Then optionally scale it — wider/deeper on the same 1M.

### Track 2 — OOD (this Mac, ~2 h, after Track 1)
5. Bin benchmark molecules by Tanimoto to nearest corpus neighbour; report the four arms per bin.
6. Bin by heavy-atom count as well, flagging the 1.17% above the corpus cap.

### Track 3 — Figure D (parallel, ~3 h)
7. Featurisation throughput per representation, CPU and GPU, measured not projected.
8. Extrapolate 10k → 10B; mirror plot in GPU-hours for the learned models.

### Track 4 — Figures A and B (new CPU box + T4)
9. Build the 13,000 edit pairs with their two notation controls and the matched-MW reference.
10. Embed every available arm; score `n_res` and the ratio metric.
11. Figure B grid on LOCKED, **paired on folds** (170+ paired observations, MDE ~0.27% rather
    than ~0.6%), with the descriptors-over-ECFP positive control.
12. Report split by endpoint type — the descriptor lift is +12.8% on QM7 and flat on BACE, and
    pooling would hide the scoping of HUME's claim.

### GNN cost correction (2026-08-26 09:50)

The GNN was assumed to be the expensive arm and is not: **27 s/epoch on a T4 for 900k
molecules**, so twelve epochs is ~6 minutes once graphs are built. Graph construction, not
training, is the cost. Two consequences:

* the GNN arm of Figures C and D is cheap, not a schedule risk
* a **much larger** GNN is affordable — more depth, wider hidden, more epochs — which matters
  because if the GNN wins on reconstruction it is the arm most likely to keep improving

### Track 5 — the three missing models (highest schedule risk)
13. SMI-TED, Uni-Mol, Kulik SpectralScore. Start these **first** in parallel, because they fail
    on environment problems rather than on compute, and that failure mode does not respect a
    schedule.

---

## What I would tell you to write today

A defensible draft of **Results §Figure D** and **§proxy model selection**, plus a partial
Figure A covering the seven representations we hold. Figures B and C get their scaffolding and
their control arms, with the full grid landing tomorrow once the CPU box has chewed through
LOCKED.

The honest risk is Track 5. If SMI-TED, Uni-Mol and Kulik all cooperate, A and B are complete
tomorrow. If any one of them fights, it takes a day on its own and should be cut to the SI
rather than allowed to hold the figure hostage.

---

## LOCKED gate discovered — plan corrected (2026-08-26 10:20)

ChemPFN gates LOCKED labels behind `LockedGate` + a **frozen model hash**, one evaluation per
(dataset, model_hash), appended to a permanent ledger. The proxy architecture is not chosen yet,
so LOCKED cannot be touched today without spending the one-shot evaluation on a non-final model.

**All development, model selection and figure iteration moves to the 28 scoreable DEV sets.**
LOCKED runs exactly once, at the end, on a frozen HUME — and that is Figure C's headline.

This is a better substrate for Figure B regardless. The 28 DEV sets span physicochemical, QM,
ADMET, toxicity, synthesis and macrocycles, which is precisely the endpoint-type breakdown the
scoping claim needs — "descriptor-quality performance where descriptors matter" is a statement
about *which* endpoints, and it cannot be made from a pooled number.

Contamination note: MoleculeACE and FreeSolv were ChemTFM's Phase-1 selection metric and are
registered as contaminated. The 34-dataset benchmark used all day includes MoleculeACE.
