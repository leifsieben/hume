"""Fig A — can a model TELL TWO MOLECULES APART after one defined chemical change?

ONE script, ONE figure: figures/fig_a.pdf (panels a–n), plus figures/build/fig_a.csv as the
durable numeric record.

    python3 figa_resolution.py          # measures; writes results/figures/figA/resolution.json
    python3 -m figures.fig_a            # draws

THE AXIS IS HELD-OUT AUC, NOT A DISTANCE (Leif 2026-08-28). Every bar is a supervised question
asked of the representation itself: label the pre-edit molecule 0 and the post-edit molecule 1,
fit XGBoost on 80% of the pairs, and score ROC-AUC on the 20% it never saw. 1.0 means the edit
leaves a signature the model can find and generalise; 0.5 means it does not.

WHY THIS REPLACED THE DISTANCE RATIO. The old axis was |dx|/sigma relative to swapping in a
different compound of matched mass, and it had two defects that no amount of tuning removes.
(i) It rewards MAGNITUDE, and magnitude is not resolution: a model can move a long way in a
direction that is different for every pair, which is unusable downstream and scored well. (ii)
For a continuous embedding the "different compound" denominator IS a random background pair, so
any background-calibrated threshold drives the reference cell to ~0 by construction. AUC has no
free parameter, is scale-free, and asks the question a tree ensemble actually asks: is there a
coordinate, or a combination, that separates A from A' CONSISTENTLY.

THE AXIS STARTS AT 0.5 BECAUSE 0.5 IS THE NULL. Below-chance is noise around it, not a
direction, so cells at or under 0.5 are drawn at the floor and labelled. The null is NOT shaded
(Leif 2026-08-28) -- a band across every panel reads as a confidence interval on the bars, which
it is not.

*** THREE DIFFERENT THINGS LOOK LIKE "NO BAR", AND THE CAPTION MUST SEPARATE THEM. ***
(raised by the CLIMB figure session, 2026-08-28, and it is the most likely misreading here)
  1. NO CONSISTENT SIGNATURE. A class-A panel at 0.5: the edit does move the embedding, but not
     in a direction that generalises to a held-out pair. This is the failure the figure is for.
  2. CANNOT POSSIBLY DIFFER. The two vectors are bitwise identical, so there is nothing to
     learn -- Morgan invariants do not carry isotope, so ECFP4 is identical on 969 of 1,000
     13-C pairs. Marked with a triple bar on the plate; it is a fact about the representation,
     not a score it earned.
  3. THE QUESTION IS ILL-POSED. In the two notation controls the "edit" is not a chemical change
     at all, so "is this the A or the B of its pair" has no answer that transfers. 0.5 is the
     CORRECT result there and a HIGH bar is the failure. Tinted, for that reason.
*** TWO PANELS ARE FREE TO A MODEL THAT READS THE STRING, AND THE CAPTION MUST SAY WHICH. ***
`isotope_13c` scores 1.000 for the `notation` control -- a character n-gram counter with no
chemistry at all -- because an isotope IS a token in SMILES and no rewriting removes it. [12C]
against [13C] was measured, and so was moving the 13C between an aromatic and an sp3 carbon;
both still read 1.000. That panel is therefore a test OF THE GRAPH AND DESCRIPTOR ARMS, which
cannot see a character: Morgan is degenerate on every pair, HUME reads 0.62 through exact mass,
and a CLM's high score there is evidence of tokenisation, not of chemistry.

`protonate` had the same defect and was FIXED rather than caveated, by applying the same
move-don't-create rule: both members are now cations of the same formula differing only in which
basic nitrogen carries the proton. 969 of 1,000 pairs have an identical character multiset and
the notation floor falls from 1.000 to 0.918. Every arm used to score exactly 1.000 there, which
is the mirror of the "nobody clears chance" failure -- a panel where everybody saturates ranks
nothing.

Measured on our own set, the same holds for a matched-mass substitution: the molecules are as
different as two molecules get (mean Tanimoto ~0.1) and every arm still scores ~0.50, because
which of two unrelated compounds is called "A" is arbitrary. That is reported in the printed
summary and the CSV as the empirical null, and it is deliberately NOT a panel -- it measures the
metric, not a representation (Leif 2026-08-28: "drop it and replace it with another change we
could measure"), and CH2 homologation took its cell.

ERROR BARS ARE OVER FIVE SPLIT SEEDS, and they earn their place: CLIMB measured 0.833 +/- 0.020
across seeds on one cell, so a single split was worth about +/-0.04 -- wider than several gaps
this figure is asked to support.

SPLITS ARE BY CONNECTED COMPONENT, NOT BY PAIR. 32 molecules in our set appear in more than one
pair; a plain 80/20 over pairs can then put a molecule in train and in test, and the leak RAISES
AUC, so the failure mode looks like a better result. figa_resolution.py groups pairs under
"shares a molecule" and assigns whole components, with an assertion that the two sides are
disjoint. Largest component is 2 pairs (0.2%) in every mode, so the split stays clean.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                       # allow `python figures/fig_a.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import Patch                                        # noqa: E402

import arms as A                                                            # noqa: E402
from style import (FS, STYLE, TINT_CONTROL, check_font,                      # noqa: E402
                   mark_empty, row_ncol, save)

check_font()
ROOT = Path(__file__).resolve().parents[2]
FIGA = ROOT / "results" / "figures" / "figA"
RES = FIGA / "resolution.json"
INK = STYLE["ink"]

#: Scored but NOT DRAWN (Leif 2026-08-29). Every one is still measured, still in
#: resolution.json and still in figures/build/fig_a.csv -- they are dropped from the PLATE only.
#:   notation       a difficulty floor, not a representation; it belongs in the audit table
#:   chemberta_mlm  the MLM/MTR pair is a pretraining ablation for the text, not for this plate
NOT_DRAWN = {"notation", "chemberta_mlm"}

NULL = 0.5                  # chance. The floor of the axis and the null of the test.
YMAX = 1.0
DEGEN_MARK = 0.50           # >=50% identical vectors -> the cell is degenerate, not weak
FLOOR_TOL = 0.005           # within this of chance -> label it rather than draw a sliver

# (mode, class, two-line panel title). CLASS decides the reading rule and the tint:
#   "A"  a real chemical change -- a HIGH bar is good
#   "B"  notation only, same molecule -- a HIGH bar is a FAILURE -- warm tint
#
# ROW ORDER IS SEMANTIC. Row 1 is edits that leave the plain heavy-atom graph untouched or
# nearly so -- stereochemistry, isotope, charge, element, bond order -- i.e. exactly the
# properties a graph invariant has to carry EXPLICITLY, and where the arms disagree most. Row 2
# is edits to the skeleton itself. Each row ENDS on a tinted control, so the eye meets the
# reading inversion at a row boundary rather than mid-scan.
MODES = [
    ("stereo_flip",    "A", "Inverted\nstereocentre"),
    ("ez_flip",        "A", "Flipped E/Z\ndouble bond"),
    ("isotope_13c",    "A", "$^{12}$C→$^{13}$C,\ngraph unchanged"),
    ("protonate",      "A", "Proton moved to\nanother amine"),
    ("halogen_swap",   "A", "Halogen\nswapped"),
    ("saturate",       "A", "C=C reduced\nto C–C"),
    ("null_enumerate", "B", "Re-written\nSMILES"),
    ("h_to_methyl",    "A", "One methyl\nadded"),
    ("n_methylation",  "A", "Amide N–H →\nN–methyl"),
    ("ch2_homolog",    "A", "One CH$_2$ inserted\n(homologation)"),
    ("scaffold_hop",   "A", "Aromatic C→N\n(benzene→pyridine)"),
    ("ring_contract",  "A", "Cyclohexyl →\ncyclopentyl"),
    ("regioisomer",    "A", "Substituent moved\n(ortho ↔ meta)"),
    ("null_kekulize",  "B", "Kekulé\nform"),
]
NROW, NCOL = 2, 7
TINT = {"A": None, "B": TINT_CONTROL}

# NOT A PANEL. The matched-mass substitution measures the METRIC's null, not a representation,
# so it is reported in the summary and the CSV and never drawn. See the module head.
NULL_MODE = "matched_mw"


def load():
    """-> (armlist, {arm: {mode: cell}}) from resolution.json.

    Nothing is recomputed here. The AUCs come from figa_resolution.py, which owns the split, the
    leak assertion and the seeds; a figure that re-derived them could disagree with the record.
    """
    if not RES.exists():
        raise SystemExit(f"fig_a: {RES} does not exist. Run `python3 figa_resolution.py` first --\n"
                         f"       it embeds nothing, it only scores whatever .npz files are in\n"
                         f"       {FIGA / 'embeddings'}.")
    raw = json.load(open(RES))
    known = {m for m, _, _ in MODES} | {NULL_MODE}
    for arm, cells in raw.items():
        missing = [m for m, _, _ in MODES if m not in cells]
        if missing:
            print(f"  ! {arm}: no cell for {missing} -- those panels will be blank for this arm")
        for m, c in cells.items():
            if m not in known:
                continue
            if c["n_seeds"] < 2:
                raise SystemExit(
                    f"fig_a: {arm}/{m} was scored on {c['n_seeds']} split seed(s). The error bars "
                    f"on this plate ARE the seed spread, and one seed cannot produce one. "
                    f"Re-run figa_resolution.py (N_SEEDS is 5).")
    return A.order([a for a in raw if a not in NOT_DRAWN]), raw


def _panel(ax, armlist, cells, mode, klass, title):
    """One edit. Bar height is held-out AUC above chance; the baseline of the axis IS chance."""
    if TINT[klass]:
        ax.set_facecolor(TINT[klass])
    x = np.arange(len(armlist))
    mu = np.array([cells[a].get(mode, {}).get("mean", np.nan) for a in armlist])
    sd = np.array([cells[a].get(mode, {}).get("sd", np.nan) for a in armlist])
    dg = np.array([cells[a].get(mode, {}).get("degenerate_frac", np.nan) for a in armlist])

    # Drawn from the null upward. A bar whose base is 0 would put nine tenths of its ink below
    # the lowest value any cell can meaningfully take, and compress the whole range that matters.
    shown = np.clip(mu, NULL, YMAX)
    ax.bar(x, shown - NULL, bottom=NULL, width=0.80, color=[A.color(a) for a in armlist],
           edgecolor=INK, linewidth=0.45, zorder=3)
    for xi, a in zip(x, armlist):                 # hatch marks PREDICTED, never a second hue
        if A.hatch(a):
            ax.patches[xi].set_hatch(A.hatch(a))
    lo = np.clip(shown - sd, NULL, YMAX)
    hi = np.clip(shown + sd, NULL, YMAX)
    ax.errorbar(x, shown, yerr=np.vstack([shown - lo, hi - shown]), fmt="none", ecolor=INK,
                elinewidth=0.55, capsize=1.3, capthick=0.55, zorder=4)

    # A CELL AT THE FLOOR IS LABELLED, and the two reasons a cell can sit there are labelled
    # DIFFERENTLY. An unlabelled flat baseline is indistinguishable from an arm that was never
    # drawn, and "cannot possibly differ" is not the same finding as "no consistent signature".
    for xi, m_, d_ in zip(x, mu, dg):
        if not np.isfinite(m_):
            continue
        if np.isfinite(d_) and d_ >= DEGEN_MARK:
            ax.text(xi, NULL + (YMAX - NULL) * 0.02, "≡", ha="center", va="bottom",
                    fontsize=FS["annot"] - 1.0, color=INK, zorder=5)

    ax.set_ylim(NULL, YMAX)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "", "0.7", "", "0.9", "1"])
    ax.set_xticks([])
    ax.set_xlim(-0.70, len(armlist) - 0.30)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
    ax.set_title(title, fontsize=FS["annot"], fontweight="bold", color=INK, pad=3, loc="center")


def main():
    armlist, cells = load()
    print(f"arms: {armlist}")

    # A4 WIDTH IS THE BUDGET. Fourteen panels on 2x7 across the text block, and the height is
    # whatever the panels plus a two-row legend actually need -- the first version reserved a
    # fifth of the plate for a legend band that used a third of it.
    fig = plt.figure(figsize=(STYLE["col2"], 3.15))
    gs = fig.add_gridspec(NROW, NCOL, left=0.062, right=0.995, top=0.885, bottom=0.175,
                          wspace=0.42, hspace=0.72)
    assert len(MODES) == NROW * NCOL, (
        f"fig_a: {len(MODES)} panels on a {NROW}x{NCOL} grid. The plate is designed to fill "
        f"exactly -- adding or removing a mode means re-choosing the grid, not leaving a hole.")
    tags = "abcdefghijklmnopqrstuvwxyz"[:len(MODES)]
    for i, (mode, klass, title) in enumerate(MODES):
        ax = fig.add_subplot(gs[i // NCOL, i % NCOL])
        assert any(mode in cells[a] for a in armlist), f"fig_a: no data for {mode}"
        _panel(ax, armlist, cells, mode, klass, title)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL == 0:
            ax.set_ylabel("held-out AUC:\nA vs A′", fontsize=FS["annot"])

    lax = fig.add_axes([0.062, 0.005, 0.933, 0.135])
    lax.axis("off")
    mark_empty(lax, "holds the legend")
    handles = [Patch(facecolor=A.color(a), edgecolor=INK, lw=0.6, hatch=A.hatch(a),
                     label=A.short_label(a)) for a in armlist]
    # frameon=False, matching fig_b, fig_c, fig_d and the CLIMB house style. A boxed legend
    # below the axes is a second rectangle competing with fourteen panel frames.
    lax.legend(handles=handles, loc="center",
               ncol=row_ncol(handles, rows=1 if len(handles) <= 6 else 2),
               fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.1,
               labelspacing=0.35, borderpad=0.45, frameon=False)

    save(fig, "fig_a")
    plt.close(fig)

    with open(ROOT / "figures" / "build" / "fig_a.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "label", "mode", "klass", "auc_mean", "auc_sd", "auc_min", "auc_max",
                    "n_pairs", "n_seeds", "degenerate_pairs", "degenerate_frac"])
        for a in armlist:
            for mode, klass in [(m, k) for m, k, _ in MODES] + [(NULL_MODE, "NULL")]:
                c = cells[a].get(mode)
                if not c:
                    continue
                w.writerow([a, A.label(a), mode, klass, f"{c['mean']:.4f}", f"{c['sd']:.4f}",
                            f"{c['min']:.4f}", f"{c['max']:.4f}", c["n_pairs"], c["n_seeds"],
                            c["degenerate_pairs"], f"{c['degenerate_frac']:.3f}"])
    print("  wrote  figures/build/fig_a.csv")

    print("\n   " + "mode".ljust(17) + "".join(f"{A.label(a)[:15]:>17s}" for a in armlist))
    for mode, klass, _ in MODES + [(NULL_MODE, "NULL", "")]:
        row = f"   {mode:<17}"
        for a in armlist:
            c = cells[a].get(mode)
            row += (f"{c['mean']:>11.3f}±{c['sd']:.3f}" if c else f"{'—':>17}")
        print(row + ("   <- the METRIC's null, not a panel" if klass == "NULL" else ""))
    print("\n   Held-out ROC-AUC over 5 component splits, mean ± sd. 0.500 = chance.")
    print("   ≡ on the plate: the two vectors are IDENTICAL in >=50% of pairs, so the cell is")
    print("   degenerate by construction and cannot be resolved by anything.")
    print("   null_enumerate and null_kekulize are the SAME molecule written two ways: there a")
    print("   HIGH number is a failure, and 0.500 is the correct answer.")

    # THE NULL IS AN ASSERTION, NOT A FOOTNOTE. If a matched-mass substitution -- two unrelated
    # compounds -- ever scores well above chance, the split leaked and every bar above is inflated.
    bad = {a: cells[a][NULL_MODE]["mean"] for a in armlist
           if NULL_MODE in cells[a] and cells[a][NULL_MODE]["mean"] > 0.60}
    assert not bad, (
        f"fig_a: the matched-mass null is above 0.60 for {bad}. Two unrelated molecules carry no "
        f"consistent A/B signature, so this can only be a leak between the train and test halves "
        f"of the split. Do not publish the plate; re-check split_components() in "
        f"figa_resolution.py.")


if __name__ == "__main__":
    main()
