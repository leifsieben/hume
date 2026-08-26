# `figures/` — the HUME paper figure pipeline

Every figure in the paper is produced by a script in this directory. No notebook, no hidden
state: edit a script, run it, the PNG/PDF changes.

```bash
python3 -m figures.fig_a          # renders figures/fig_a.png, .pdf, .csv
```

| File | Role |
|---|---|
| `arms.py` | **single source of truth** — arm keys, labels, colours. Never hard-code either in a figure script |
| `style.py` | matplotlib rcParams, page widths, panel tints, `save()` (incl. the page-width check), `check_no_empty_panels()` |
| `fig_<id>.py` | one script per paper figure; the artefact carries the SAME name as its script |

## The design system is CLIMB's, on purpose

Leif, 2026-08-26: *"Study their design guidelines, I want visual continuity with them for all
figures going forward."* `style.py` and `arms.py` are ports of `CLIMB/figures/style.py` and
`CLIMB/figures/arms.py`. The two papers share a reader and share arms — ECFP4, Morgan r=3,
ChemBERTa-2 and MoLFormer appear in both — so the same model keeps the same colour across them:

| arm | hex | same in CLIMB? |
|---|---|---|
| ECFP4 | `#8A5F1B` | yes |
| ChemBERTa-2 | `#5C4A85` | yes |
| MoLFormer | `#8B7BB5` | yes |
| Morgan r=3 | `#E8B86A` | **no — deliberate**, see below |

**The one break.** CLIMB gives R3FP `#4E340B`, one shade darker than its ECFP4. That is safe
there because `fig_G` drops R3FP and the two never sit side by side. HUME's Figure A puts them
adjacent in all thirteen panels and *r=2 vs r=3* is a headline claim, so Morgan r=3 takes the
light end of the same amber family instead. Same family, maximal lightness separation.

## Colour semantics — slots, not tastes

amber = classical featurisations (fingerprints, descriptor block) · crimson = **HUME** ·
violet = external chemical language models · teal = external graph foundation models ·
blue = the descriptor **proxy** ladder (ridge → GNN) · grey = controls

Shades within a family run dark → light and each family spans a distinct lightness band, so the
figures also survive greyscale printing. The hues are CLIMB's CVD-nudged set: plain
orange/red/green is the one triple deuteranopes cannot separate, so "red" is a magenta-leaning
crimson and "green" a bluish teal, both anchored on Okabe-Ito.

**Exact vs predicted is a HATCH, not a hue.** `hume_core_predict` carries `hume_core`'s colour
with `///` over it. The paper's central comparison is a descriptor block computed exactly against
the same block predicted by a proxy; two unrelated colours would make that pair read as two
unrelated arms, and it would burn hues the crimson family does not have to spare.

## Page width

Every figure is authored at `STYLE["col2"]` = **6.69 in**, the 170 mm text block of A4 with 20 mm
margins, so figures go into LaTeX at `width=\textwidth` with NO downscaling and the point sizes in
`FS` are the sizes that print. `save()` measures the width actually written and WARNS on >5%
deviation: `savefig(bbox_inches="tight")` trims slack margins and GROWS past the canvas when a
legend or title is anchored outside the axes, so two figures authored at the same width can land
an inch apart and LaTeX then prints their fonts at different sizes. Fix a deviation by making the
axes fill the canvas or moving anchored content inside — **not** by rescaling `figsize`, which
tight-bbox simply re-trims.

This is why Figure A parks its legend in a spare gridspec cell rather than under the axes.

## Rules

- **One font, one size scale.** Never a local tweak like `fs_annot - 0.5`; panels get assembled
  into multi-panel plates later and a one-off size makes one panel's text differ from its
  neighbour's.
- **All text is black** (`#000000`). No grey text anywhere.
- **Titles are sentence case.**
- **Never draw the caption into the image.** Captions belong in the LaTeX `\caption{}`; write the
  caption source in the script's module docstring instead. There is deliberately no `caption()`
  helper in `style.py`.
- **A tint marks a different reading RULE**, never decoration. `TINT_CONTROL` (warm) = a high bar
  is a *failure*; `TINT_REF` (very faint, cool) = this panel defines the unit. The reference tint
  is much fainter than the control tint on purpose — a reference tinted as strongly as a control
  invites the reader to apply the inverted rule to it.
- **Label exact zeros.** An unlabelled flat baseline is indistinguishable from a bar that was
  never drawn, and half of Figure A's cells are 0.000 by construction.
- **`check_no_empty_panels()` runs inside `save()`** and raises. A panel that silently draws
  nothing is worse than one that fails: the reader concludes the arm scored zero rather than that
  the arm is missing. Declare a deliberately blank cell with `mark_empty(ax, why)`.
- **One script → one figure.** No `v1/v2/...` suffixes in the committed state.

## Roster

| ID | Figure | Script | Status |
|---|---|---|---|
| A | does the representation respond when the chemistry changes? 13 panels | `fig_a.py` | rendering; 3 of 8 arms embedded (`ecfp`, `r3cfp`, `chemberta`). `desc` in flight; MolFormer / SMI-TED / MiniMol / Chemprop pending |
| B | downstream value of the descriptor block across the DEV grid | — | pending the 9-arm grid |
| C | proxy ladder: reconstruction vs downstream recovery | — | pending |
| D | throughput and the cost argument, extrapolated to 10B | — | pending |
