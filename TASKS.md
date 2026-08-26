# TASKS — live worklist

Single source of truth for what is running, done, and next. **Update this file when anything
changes state.** It exists so the thread survives a context loss, a restart, or a day off.

Legend: `[ ]` todo · `[~]` running · `[x]` done · `[!]` blocked/failed

Last updated: 2026-08-26 10:25

---

## Fleet

| box | id | what it is doing | stop condition |
|---|---|---|---|
| this Mac | — | idle — proxy ladder complete | — |
| GPU spot | terminated | GNN done, results fetched | done |
| CPU spot | `i-06ab6fd22e04ce74a` | DEV grid, 28 datasets 14-way parallel | `fetch_cpu.sh` then terminate |

**Never touch** `chempfn-gpu-job` or `climb-figA-repair`. Not ours; the user monitors them.

---

## HARD-WON LESSONS (do not repeat)

1. **Never auto-shutdown a box before its results are durable.** The first GNN run completed,
   shut itself down on success, and took its results with it — `DeleteOnTermination=true` meant
   the EBS volume went too. Results must reach S3 or the local disk *before* anything shuts down.
2. **Verify the remote interpreter before arming a shutdown.** An earlier instance died in two
   minutes because `python: command not found` chained into `; sudo shutdown -h now`.
3. **Check memory before launching a big fit.** `train_ridge` built a 21 GB float64 copy and
   swapped the Mac to a standstill. Chunk the Gram accumulation.
4. **Aggregate by rank or per-fold pairing, never a mean of raw RMSE** across datasets with
   incommensurable scales.
5. **State the MDE.** A flat bar means "smaller than X%" and the reader is owed X.
6. **Match the training budget across arms.** The first GNN got 12 epochs against 25 for every
   other model and was still improving 4%/epoch when stopped — an unfair comparison that
   nearly became a paper claim.
7. **1M is the gate, not 3k.** The deuterium bug was 0.061% of molecules and invisible at 3,000.

---

## Track 1 — proxy model selection

- [x] 1M corpus built, packed, verified bit-identical
- [x] **all five trained on the full 1M (900k/100k contiguous scaffold split, 165 targets)**

      pinet    0.9837 / 0.9634   799s  CPU   <- best on reconstruction
      mlp      0.9815 / 0.9623   925s  CPU
      gnn      0.9712 / 0.9438   790s  T4
      linquad  0.9593 / 0.9325   258s  CPU
      ridge    0.9505 / 0.9109    18s  CPU

      **CLAIM WITHDRAWN.** I wrote that the GNN "loses despite seeing the graph". It does not
      support that. The run was **not converged and undertrained relative to every other arm**:

        * 12 epochs against 25 for pinet/mlp/linquad
        * loss still falling **4.0% per epoch, monotonically**, at the point it was stopped
          (0.2970 -> 0.1208 -> ... -> 0.0426 -> 0.0409, never flattening)
        * h=128 depth=4, against Chemprop's ~300 hidden and MiniMol's 336 x depth 16

      A model still improving 4%/epoch when you stop it has been interrupted, not beaten.
      Re-running at **h=512 depth=8 for 60 epochs** before any claim is made.

      Reconstruction is NOT the decision regardless — it has anti-correlated with downstream
      value three times here. The four-arm downstream comparison decides.
- [x] checkpoints for all five
- [ ] per-family reconstruction R² (never pooled)
- [ ] four-arm downstream on the 34 cached datasets
- [ ] **pick the architecture**, then scale it (GNN is only 27 s/epoch — a much bigger one is cheap)

## Track 2 — OOD

- [ ] bin benchmark by Tanimoto to nearest corpus neighbour, report four arms per bin
- [ ] bin by heavy-atom count; flag the 1.17% above the corpus 60-atom cap

## Track 3 — Figure D (cost)

- [x] parse 76 µs · ECFP 39 · r3fp 48 · CORE 59 · Chi 6.8 (C++) · cycles 2.2 · resistance 5.8
- [ ] learned-model throughput, CPU and GPU, measured
- [ ] extrapolate 10k → 10B; mirror plot in GPU-hours

## Track 4 — Figures A and B

- [ ] build 13,000 edit pairs: 10 chemical edits × 1,000, + 2 notation controls, + matched-MW ref
- [ ] embed each arm; score the ratio metric (headline) and normalised `n_res` (SI)
- [ ] Figure B grid on LOCKED, **paired on folds** (MDE ~0.27%, not ~0.6%)
- [ ] positive control: descriptors over ECFP
- [ ] report split by endpoint type, never pooled

## Track 5 — baseline models (highest schedule risk; start FIRST)

Roster changed 2026-08-26: **CLIMB dropped at the user's direction** ("pretty bad").

- [x] ECFP4, r3fp, RDKit+Mordred — local
- [x] CheMeleon — `CLIMB/chemeleon_fingerprint.py`, cached features
- [x] MiniMol — `.venv-minimol`
- [x] Chemprop — `.venv-web`
- [~] **ChemBERTa-2** — downloading, `DeepChem/ChemBERTa-77M-MLM`
- [~] **MolFormer** — downloading, `ibm/MoLFormer-XL-both-10pct`
- [~] **SMI-TED** — downloading, `ibm/materials.smi-ted`
- [ ] Kulik SpectralScore — `github.com/hjkgrp/SpectralScore`, needs conformers
- [ ] Uni-Mol — `unimol_tools`, needs conformers
- [~] ~~CLIMB~~ — dropped
- [x] ~~GROVER~~ — dropped (legacy DGL pins)

## Track 6 — benchmark grid (Figures B and C)

**LOCKED IS OFF LIMITS UNTIL HUME IS FROZEN.** `load_dataset` on a LOCKED set requires a
`LockedGate` with a frozen model hash, and `open_locked` raises if the model is unfrozen *or
already evaluated*. Every evaluation is appended to `manifests/locked_evals.jsonl`, which is
the paper's audit trail. Running it today would burn the one-shot evaluation on a proxy we are
still selecting, and it would be permanently on the record as having happened.

Also: **MoleculeACE and FreeSolv are registered as contaminated** — they were ChemTFM's Phase-1
selection metric. Our 34-dataset benchmark includes MoleculeACE, so Phase 0's numbers inherit
that. Fine for a compute-side question; not usable as a headline.

- [x] suite reachable: 51 datasets, 16 LOCKED / 35 DEV, `~/chempfn-data`
- [x] **28 scoreable DEV datasets identified** — the development substrate, and a better Figure B
      substrate than LOCKED anyway: physicochemical (esol, aqsoldb, lipophilicity, pb_water_sol),
      QM (qm8, qm9, qm9_gap, qmugs_gap, photoswitch), ADMET (the pb_* block, vdss_lombardo,
      hia, bioavail, cyp2d6_inh), toxicity (ames, pb_ames, ld50_zhu), plus rascore, fartdb,
      cycpept_pampa. That spread is exactly what the endpoint-type split needs.
- [~] **RUNNING** on the CPU box: 28 DEV sets x 6 arms x 5 folds, 14-way parallel
      arms: `ecfp` / `r3cfp` / `desc` (fingerprint-free control) / `ecfp_desc` (POSITIVE
      CONTROL) / `hume_exact` (ceiling) / `hume` (Pi-net predicted)
- [ ] add learned-baseline arms (CheMeleon, MiniMol, Chemprop, ChemBERTa-2, MolFormer, SMI-TED)
- [ ] paired on folds, positive control, MDE in the caption
- [ ] **LAST STEP OF THE PROJECT**: freeze HUME, register the hash, run LOCKED once

## Deferred (explicitly not today)

- packaging, pip, docs
- R3cFP proxy (trained after r=2 decisions settle, by design)
- C++ core port of the blocks
