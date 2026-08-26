# Figures A and B — methodology

Publication-grade specification for the two field-level figures in the HUME paper. Both are
statements about the state of molecular representation, not about HUME; HUME appears in each
only as a demonstration that the two observations compose.

The logical spine of the paper is:

    A.  ECFP resolves chemical change; learned embeddings largely do not.
    B.  Descriptors add real information over ECFP; no pretrained model adds anything over both.
    ->  Therefore: compute ECFP, compute the cheap descriptors, predict the slow ones.

HUME is the conclusion, not a premise. Both figures must stand if HUME is deleted from them.

---

## Model roster

The same eight representations appear in both figures. Availability checked 2026-08-25.

| model | class | 2D/3D | dim | status |
|---|---|---|---|---|
| ECFP4 (counts, chirality on) | fingerprint | 2D | 2048 | ready — RDKit |
| RDKit + Mordred descriptors | descriptors | 2D | 865 (dedup) | ready |
| CheMeleon | D-MPNN | 2D | 2048 | ready — `CLIMB/chemeleon_fingerprint.py`, cached features, `chemeleon_mp.pt` |
| MiniMol | GNN (GINE) | 2D | 512 | ready — `ChemTFM_OLD/.venv-minimol`, minimol 1.3.5 |
| Chemprop | D-MPNN | 2D | 300 | ready — `ChemTFM_OLD/.venv-web` |
| CLIMB | CLM | 2D | — | ready — `CLIMB/climb`, checkpoints present |
| SMI-TED | CLM | 2D | 768 | **needs download** — IBM `materials.smi-ted` on HuggingFace |
| **Uni-Mol** | **transformer (3D)** | **3D** | 512 | **to add** — `unimol_tools`, pip-installable, maintained |
| Kulik SpectralScore | physics/spectral | 3D | — | **code exists** — [github.com/hjkgrp/SpectralScore](https://github.com/hjkgrp/SpectralScore) |
| UMA (optional) | MLIP | 3D | 128 | ready — `.venv-uma`, conformer pipeline and charge/spin fix already built |
| ~~GROVER~~ | ~~graph transformer~~ | | | **dropped** — see below |
| Graphormer (optional) | graph transformer | 2D | 768 | not started — canonical architecture representative |

### Roster decisions (2026-08-25)

**GROVER dropped, Uni-Mol added in its place.** GROVER has no local copy and carries
known-painful legacy torch/DGL pins. Uni-Mol is the better use of the slot for three reasons:
it is the model a referee will actually name, so omitting it is the obvious objection to a
claim that no pretrained model adds anything; it is **3D**, so it probes a genuinely different
information channel rather than being a fourth 2D-graph model; and `unimol_tools` is
pip-installable and maintained.

**MiniMol does not cover the graph-transformer slot.** Its shipped config is
`layer_type: 'pyg:gine'` — GINE message passing, not attention. Verified directly in
`minimol/ckpts/minimol_v1/config.yaml`. Dropping GROVER therefore leaves the roster with no
graph transformer at all; Graphormer is the optional fix if the architecture class must be
covered explicitly, but Uni-Mol is what closes the referee objection.

**The 3D arm matters for the argument.** ECFP and descriptors are both 2D. A fair test of
"is anything orthogonal left" needs at least one representation that sees geometry, or the
conclusion is circular. Uni-Mol, Kulik and UMA give three, at little marginal cost since the
conformer pipeline (with the charge/spin fix) already exists from the UMA work.

**Kulik: code exists, no implementation needed.** `hjkgrp/SpectralScore`, titled exactly
"Physics-Based Molecular Fingerprints from Spectral Graph Theory". The method is a complete
graph in 3D with physics-heuristic edge weights, eigendecomposed via the graph Laplacian;
permutation- and E(3)-invariant. **It requires conformers**, which puts it in UMA's cost
bracket rather than ECFP's — this belongs on the Figure D cost axis, not buried in a footnote.

We hold a prior that it will be **flat in Figure B**: it is spectral-graph-theory based, the
same family as our own `resistance.py`, which measured no downstream benefit across 34 datasets
(mean +0.42%, p=0.033 uncorrected, p=0.198 Bonferroni). Reporting it either way is informative;
expecting a null is not a licence to omit it.

---

## Figure A — Does the representation resolve chemical change?

### Claim

ECFP resolves essentially every chemical edit, exactly and permanently. Learned embeddings
move by amounts too small for a downstream learner to exploit. This matters because activity
cliffs routinely form on a single stereocentre: if the representation cannot separate the pair,
no model built on it can.

### Design

Ten chemical edits, N = 1,000 matched pairs each, drawn from the benchmark chemotypes so the
molecules are drug-like rather than toy.

| # | edit | should resolve? |
|---|---|---|
| 1 | stereocentre inversion (R -> S) | yes |
| 2 | double-bond geometry (E -> Z) | yes |
| 3 | ring size (6 -> 5) | yes |
| 4 | halogen swap (Cl -> F, Cl -> Br) | yes |
| 5 | scaffold hop (benzene -> pyridine) | yes |
| 6 | H -> methyl | yes |
| 7 | isotope (12C -> 13C) | yes |
| 8 | amide N-methylation | yes |
| 9 | ring fusion, linear -> angular | yes |
| 10 | protonation state change | yes |
| **11** | **SMILES re-enumeration (non-canonical)** | **NO — null control** |
| **12** | **Kekulised vs aromatic form** | **NO — null control** |
| **13** | **matched-MW substitution (different compound)** | **reference, = 1.00 by construction** |

Edits 11-12 are the negative controls: they change notation, not chemistry, and any response
above zero is the model reacting to how the string was written. Edit 13 sets the per-model
scale.

**How each class reaches the model, precisely.** Chemical edits (1-10) are applied to the RDKit
molecule object and serialised **once**, so the pair is canonical by construction rather than by
re-canonicalisation. Notation controls (11-12) are applied at the string level and passed
through **unmodified**. The distinction matters and must be phrased this way in the methods:
saying chemical edits arrive "exactly as supplied" would imply string surgery, which would mix
an uncontrolled notation change into every chemical edit and systematically flatter the CLMs.
Conversely, re-canonicalising a notation control turns it into a no-op and destroys the only
negative control the figure has.

### Methods paragraph (as it should appear in the paper)

> **Resolution.** For each of ten chemical edits we construct 1,000 matched pairs (A, A') drawn
> from benchmark chemotypes, together with two notation-only controls -- re-enumeration of the
> SMILES string and interconversion between Kekule and aromatic forms, which alter the input
> string but not the molecule -- and a matched-molecular-weight substitution replacing A with an
> unrelated compound of similar mass, which serves as the per-model reference. Chemical edits
> are applied to the molecule object and serialised once; notation controls are applied to the
> string and passed to each model unmodified, since re-canonicalising them would reduce them to
> no-ops. Displacement is measured as the root-mean-square change across dimensions in units of
> each dimension's standard deviation over a fixed background library of 10,000 unedited
> drug-like molecules, d(A, B) = rms_j[ (x_j(A) - x_j(B)) / sigma_j ], with sigma_j estimated
> once per representation on the background set. We report **response = d(A, A') / d(A, A_MW)**,
> the displacement an edit produces relative to that same model's displacement under complete
> compound substitution, so that 1.00 means "moves the representation as far as changing the
> molecule". The per-model denominator is what allows a 2048-bit sparse fingerprint and a
> 512-dimensional dense embedding to share an axis: representations differ by more than an
> order of magnitude in how many coordinates they spend per edit -- ECFP4 moves 1.4% of its bits
> under complete compound substitution where a CLM moves 82% of its coordinates -- and any
> statistic not normalised by the model's own reference reports that density difference rather
> than chemical sensitivity. We report a ratio rather than a norm or a threshold count because
> gradient-boosted trees subsample features at each split, so what matters downstream is how
> large an edit's displacement is against the scale of displacements that model produces at all,
> not whether the representation is a mathematically injective function of structure -- a
> criterion every deterministic model satisfies trivially and which consequently cannot
> discriminate between them. Cells report median and interquartile range over the 1,000 pairs;
> contrasts between arms are paired on molecule, so pair difficulty cancels, and are reported
> with the paired standard error.

### Why the headline is a ratio and not a threshold count

An earlier draft of this methodology proposed counting dimensions displaced by more than half a
standard deviation. It was tested against the existing pair data rather than reasoned about, and
it fails on three counts. Recorded here because the failure mode is instructive and the
proposal is superficially attractive.

**1. It measures sparsity, not resolution.** The 0.5 sigma scaling makes individual dimensions
commensurable; it does nothing about how many dimensions a representation spends per edit, and
fingerprints are sparse by construction.

    median n_res at 0.5 sigma   ECFP4   r3fp   CLIMB sup   CLIMB uns-canon
    add_methyl                     15     32          56               152
    isotope_13c                     0      0          46               182
    matched_mw (reference)         28     34         364               418

ECFP4 spends 28 of 2048 bits on a *completely different compound* -- 1.4% of its dimensions --
where CLIMB spends 418 of 512, or 82%. A raw count therefore reports that CLIMB resolves an
unrelated molecule fifteen times better than ECFP4 and that a methyl group moves CLIMB ten times
further. Both are artefacts of density. The headline claim survives; three other panels invert.

**2. The threshold becomes a free parameter exactly where the argument lives.** Fingerprints sit
nowhere near 0.5 sigma so the count is flat for them; the CLMs sit on the knife edge.

    add_methyl, median n_res      1/3 sigma   1/2 sigma   1 sigma
    ECFP4                                15          15        15
    Morgan r3-counts                     33          32        30
    CLIMB sup                           140          56         1
    CLIMB unsup (canon)                 250         152        18

A 3x move in an arbitrary constant moves the CLM by 140x. That is the reviewer question that
ends the figure. It is also the failure the ratio metric was originally written to escape --
`resolution_effect_size.py` records that a previous threshold-calibrated version "drove every
class-A cell to 0%".

**3. The log axis cannot be drawn.** 36 of 91 cells have median n_res = 0, including every
stereo, E/Z and ring-size cell for all three CLM arms -- which is the paper's headline. The
ratio places those at 0.009-0.030 against 0.679-0.920, a measurable 20-100x. The count turns it
into "0 versus 8": an undefined ratio on an axis that renders neither endpoint.

### Normalised count (SI only)

If a threshold-based corroboration is wanted, the version that survives is the count normalised
by the model's own reference, which cancels the sparsity term:

    n_res(A, A') / n_res(A, A_MW)

On stereo inversion this reads ECFP4 0.29, r3fp 0.59, CLIMB 0.00 -- consistent with the ratio
metric. It belongs in the SI as corroboration, with its threshold sensitivity stated as a
limitation rather than serving as the main axis.

### Sample sizes and dispersion

Fixed from the review of the existing implementation, which printed one number per cell with no
dispersion at all -- making claims such as "augmentation does not buy chemical sensitivity"
(0.140 vs 0.234) unfalsifiable as drawn.

| quantity | was | now | why |
|---|---|---|---|
| pairs per edit | 100 | **1,000** | feasible on every mode: the MoleculeACE pool alone has 35,633 unique molecules, 11,908 with a stereocentre, 1,663 with an E/Z bond |
| background for sigma | 989 | **10,000** | sigma is estimated once and divides everything; cheap insurance |
| per-cell statistic | single value | **median + IQR** | IQRs are tight and non-overlapping: stereo ECFP4 0.679 [0.383, 0.852] vs CLIMB sup 0.009 [0.007, 0.010], two orders of magnitude apart |
| arm-vs-arm contrast | none | **paired SE** | both CLM arms see the same molecule pairs, so pair difficulty cancels in the difference |

Computed properly, the augmentation claim holds and strongly:

    add_methyl        aug - canon = -0.0860 +- 0.0062   (paired SE, n=100)
    add_fluorine                    -0.0749 +- 0.0035
    matched_desc                    -0.1526 +- 0.0080
    stereo_flip                     -0.0058 +- 0.0017

All five modes distinguishable.

---

## Figure B — What is orthogonal to what?

### Claim, in two steps

The reference block **changes between panels**. This is the whole design.

**Panel 1 — reference: ECFP4.** *Do descriptors carry information ECFP lacks?*
Bars: `+RDKit` · `+Mordred` · `+both` · `+HUME full` · `+HUME tight` · `+best DL`

**Panel 2 — reference: ECFP4 + descriptors.** *Does any pretrained model add anything on top?*
Bars: `+CheMeleon` · `+MiniMol` · `+Chemprop` · `+CLIMB` · `+SMI-TED` · `+GROVER` · `+Kulik`

Panel 1 establishes the premise and simultaneously places HUME as a **substitute** for the
descriptors rather than an addition to them, which is what it actually is. Panel 2 is the field
claim. CLIMB's existing measurement already sits there: both CLIMB embeddings negative on all
six panels, -0.9% on QM7 to -5.4% on MoleculeACE.

### Statistics

* Same XGBoost, same scaffold 5-fold CV, one recorded environment throughout.
* **Error bars are +-1 SD of the PER-FOLD lift, not of either arm alone.** Both arms see the
  same folds, so fold difficulty cancels in the difference. The marginal SD of a single arm
  runs 2-8x the lift and describes a different quantity. This is CLIMB's convention and it is
  correct.
* Signed so positive is always better.
* Repeat across seeds; most per-dataset variance is split noise.

### Power — state it, do not discover it

Our own Phase 0 ran 34 datasets paired **at the dataset level** and could only detect effects
above ~0.5-0.65% (80% power, two-sided; per-dataset SD 1.0-1.4%). Every observed effect fell
below that, so the only defensible conclusion was a bound, not a null.

Fold-level pairing gives 34 x 5 = 170 paired observations, cutting the standard error ~2.2x and
the minimum detectable effect to ~0.27%. **Report the MDE in the caption.** A flat bar means
"smaller than X%", and the reader is entitled to know X.

### The positive control is mandatory

Without it a flat bar cannot be distinguished from an assay that detects nothing. CLIMB's
control: descriptors added to ECFP4 gain 12.8% on QM7 and 6.2% on Tox21, and are flat on BACE
and HIV. That same control is Panel 1's `+both` bar, so it is load-bearing twice.

### Scoping — state before a reviewer does

The descriptor lift is **not uniform across endpoint types**. CLIMB measures +12.8% on QM7 and
+6.2% on Tox21 but flat on BACE and HIV. Our own data agrees: on MoleculeACE, ECFP alone
(0.7732) beats ECFP+descriptors+Mordred (0.8054). The lift concentrates on physicochemical and
QM endpoints and is roughly absent on bioactivity and cliffs.

HUME is therefore **"descriptor-quality performance at fingerprint cost, on the task classes
where descriptors matter"** -- not a universal embedding. Panels must be reported split by
endpoint type. Pooling would hide this and a referee would find it.

---

## Provenance, storage, backup

Every cell in both figures is regenerated by script from stored inputs; nothing is hand-copied.

    results/figures/
      figA/
        pairs.json           the 13,000 matched pairs, with edit type and provenance
        embeddings/          one npz per model: (n_pairs, 2, dim), float32
        metrics.parquet      per-pair n_resolved and response, long format
        figA.py              plotting, reads only the above
      figB/
        folds.json           dataset -> fold assignment, seeded, shared by every arm
        scores.parquet       (dataset, fold, arm, metric, value), long format
        figB.py
      MANIFEST.json          sha256 + row count + git commit for every file above
      ENVIRONMENT.json       python, rdkit, xgboost, torch versions; hostname; date

Rules, adopted from CLIMB's `figure_data_manifest.json` convention:

1. **Long format, never wide.** `(dataset, fold, arm, metric, value)` survives adding an arm;
   a wide table does not.
2. **Every arm sees identical folds**, recorded in `folds.json` before any arm runs. Fold
   difficulty cancels only if the folds are literally the same.
3. **One environment per figure**, captured in `ENVIRONMENT.json`. Two RDKit versions in one
   figure is a silent correctness bug -- note `.venv-uma` carries RDKit 2026.3.5 against the
   pinned 2025.9.2.
4. **Checksums in `MANIFEST.json`**, so a figure can be proven to match the data it claims.
5. **Backups** via `scripts/backup_figures.sh` to a dated archive, mirroring CLIMB's
   `backup_paper_artifacts.sh`. Embeddings are the expensive artefact -- GROVER and SMI-TED
   especially -- and must not need recomputing.
6. **Negative and null results are stored and plotted.** The Kulik fingerprint is expected to
   be flat given our resistance result; that expectation is not a licence to omit it.
