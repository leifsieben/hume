"""Figure A — does one defined chemical change move the representation the SAME WAY every time?

    .venv/bin/python research/figa_consistency.py   # measures -> results/figures/figA/consistency.json
    .venv/bin/python figures/src/fig_a.py           # draws

THIS PLATE BECAME FIGURE A ON 2026-10-31 (Leif), taking the name from the XGBoost-AUC plate,
which is now fig_a_old.py. It was written as the companion to that one -- identical layout,
identical arms, identical ordering, one axis changed -- so the two can still be laid side by
side and read as a single comparison. Nothing about either measurement changed in the rename,
only the output stems.

WHY THE DIRECTION AXIS LEADS. The metric is the fraction of pairs whose difference vector agrees
in DIRECTION with the mean of all the others: 0.5 for random directions in any dimension, and it
needs no background corpus, no threshold and no estimator. The AUC plate answers "can a
supervised head find the edit", which is a question about the head as much as the
representation; this one answers "does the edit mean the same thing twice", which is the
property a downstream model actually consumes. See research/figa_consistency.py for the metric
and for what it gives up against the AUC.

TWO DIFFERENCES FROM fig_a_old.py, both deliberate.

1. `protonate` (the proton moved between two basic nitrogens) is replaced by `protonate_neutral`
   (the neutral base against its conjugate acid). That pair was avoided on the AUC plate because
   every arm AND the character-level control scored 1.000 there -- `+` is a literal character in
   SMILES, so the panel saturated and could not rank. Under a direction axis saturation is no
   longer the failure mode, and the question becomes the one that matters for binding.

2. The whiskers are a percentile bootstrap over PAIRS, not a spread over split seeds: there is no
   split and no seed, so there is nothing to average over. `n_seeds` is therefore not checked.

READ THE NULL PANELS FIRST. The language models score ~1.0 on `null_enumerate` and
`null_kekulize` -- a consistent direction for a change that is chemically nothing -- while every
graph-based arm is pinned at 0.5 because its difference vector is exactly zero. Any average taken
across all fourteen panels hands the string readers a free quarter of a point. The chemical
panels are the comparison; the null panels are the reason to distrust a summary statistic.
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
RES = FIGA / "consistency.json"
INK = STYLE["ink"]

#: Scored but NOT DRAWN (Leif 2026-08-29). Every one is still measured, still in
#: consistency.json and still in figures/build/fig_a.csv -- dropped from the PLATE only.
#:   notation       a difficulty floor, not a representation; it belongs in the audit table
#:   chemberta_mlm  the MLM/MTR pair is a pretraining ablation for the text, not for this plate
NOT_DRAWN = {"notation", "chemberta_mlm"}

NULL = 0.5                  # chance. The floor of the axis and the null of the test.
YMAX = 1.0                  # the largest value an AUC can take; bars are clipped here
YTOP = 1.03
#: THE AXIS RUNS SLIGHTLY PAST 1.0 SO THE TOP WHISKERS SURVIVE (Leif 2026-08-29). An AUC cannot
#: exceed 1, but mean + sd can, and clipping the upper whisker AT 1.0 made it invisible on
#: exactly the cells that sit near the ceiling -- which is most of row 2. The extra headroom is
#: not a tick and carries no label, so the axis still reads 0.5 to 1.
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
    ("protonate_neutral", "A", "Neutral base vs\nconjugate acid"),
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
    """-> (armlist, {arm: {mode: cell}}) from consistency.json.

    Nothing is recomputed here. The consistencies come from research/figa_consistency.py, which
    owns the standardisation, the leave-one-out and the bootstrap; a figure that re-derived them
    could disagree with the record. (These two sentences named figa_resolution.py and "the AUCs"
    until 2026-10-31 -- left over from this file's life as the companion plate, and pointing a
    reader at the scorer that writes the OTHER json.)
    """
    if not RES.exists():
        raise SystemExit(f"fig_a: {RES} does not exist. Run `python3 research/figa_consistency.py` first --\n"
                         f"       it embeds nothing, it only scores whatever .npz files are in\n"
                         f"       {FIGA / 'embeddings'}.")
    raw = json.load(open(RES))["arms"]
    known = {m for m, _, _ in MODES} | {NULL_MODE}
    for arm, cells in raw.items():
        missing = [m for m, _, _ in MODES if m not in cells]
        if missing:
            print(f"  ! {arm}: no cell for {missing} -- those panels will be blank for this arm")
        for m, c in cells.items():
            if m not in known:
                continue
            if "consistency" not in c:
                raise SystemExit(f"fig_a: {arm}/{m} has no `consistency` key; it was "
                                 f"written by an older scorer. Re-run research/figa_consistency.py.")
    return A.order([a for a in raw if a not in NOT_DRAWN]), raw


def _panel(ax, armlist, cells, mode, klass, title):
    """One edit. Bar height is held-out AUC above chance; the baseline of the axis IS chance."""
    if TINT[klass]:
        ax.set_facecolor(TINT[klass])
    x = np.arange(len(armlist))
    mu = np.array([cells[a].get(mode, {}).get("consistency", np.nan) for a in armlist])
    lo_b = np.array([cells[a].get(mode, {}).get("ci_lo", np.nan) for a in armlist])
    hi_b = np.array([cells[a].get(mode, {}).get("ci_hi", np.nan) for a in armlist])
    dg = np.array([cells[a].get(mode, {}).get("degenerate_frac", np.nan) for a in armlist])

    # Drawn from the null upward. A bar whose base is 0 would put nine tenths of its ink below
    # the lowest value any cell can meaningfully take, and compress the whole range that matters.
    shown = np.clip(mu, NULL, YMAX)
    ax.bar(x, shown - NULL, bottom=NULL, width=0.80, color=[A.color(a) for a in armlist],
           edgecolor=INK, linewidth=0.45, zorder=3)
    for xi, a in zip(x, armlist):                 # hatch marks PREDICTED, never a second hue
        if A.hatch(a):
            ax.patches[xi].set_hatch(A.hatch(a))
    lo = np.clip(np.where(np.isfinite(lo_b), lo_b, shown), NULL, YTOP)
    hi = np.clip(np.where(np.isfinite(hi_b), hi_b, shown), NULL, YTOP)
    # CLIPPING THE BAR CAN INVERT THE WHISKER. `shown` is clipped into [NULL, YMAX] while the
    # bootstrap bounds are clipped into [NULL, YTOP], so a cell below the null floor (hume on
    # null_kekulize is 0.333) ends up with an interval that does not bracket its own drawn value.
    # Clamping at zero keeps the bar honest -- the cell is at the floor and its whisker has
    # nowhere to go -- rather than asking matplotlib to draw a negative length.
    err = np.vstack([np.maximum(shown - lo, 0.0), np.maximum(hi - shown, 0.0)])
    ax.errorbar(x, shown, yerr=err, fmt="none", ecolor=INK,
                elinewidth=0.55, capsize=1.3, capthick=0.55, zorder=4)

    # A CELL AT THE FLOOR IS LABELLED, and the two reasons a cell can sit there are labeled
    # DIFFERENTLY. An unlabeled flat baseline is indistinguishable from an arm that was never
    # drawn, and "cannot possibly differ" is not the same finding as "no consistent signature".
    for xi, m_, d_ in zip(x, mu, dg):
        if not np.isfinite(m_):
            continue

    ax.set_ylim(NULL, YTOP)
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
            ax.set_ylabel("direction agreement:\nA vs A′", fontsize=FS["annot"])

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
        w.writerow(["arm", "label", "mode", "klass", "consistency", "ci_lo", "ci_hi",
                    "median_cos_z", "median_cos_raw", "n_pairs", "degenerate_pairs",
                    "degenerate_frac"])
        for a in armlist:
            for mode, klass in [(m, k) for m, k, _ in MODES] + [(NULL_MODE, "NULL")]:
                c = cells[a].get(mode)
                if not c:
                    continue
                w.writerow([a, A.label(a), mode, klass, f"{c['consistency']:.4f}",
                            f"{c['ci_lo']:.4f}", f"{c['ci_hi']:.4f}",
                            f"{c['median_cos_z']:.4f}", f"{c['median_cos_raw']:.4f}",
                            c["n_pairs"], c["degenerate_pairs"],
                            f"{c['degenerate_frac']:.3f}"])
    print("  wrote  figures/build/fig_a.csv")

    print("\n   " + "mode".ljust(17) + "".join(f"{A.label(a)[:15]:>17s}" for a in armlist))
    for mode, klass, _ in MODES + [(NULL_MODE, "NULL", "")]:
        row = f"   {mode:<17}"
        for a in armlist:
            c = cells[a].get(mode)
            row += (f"{c['consistency']:>17.3f}" if c else f"{'—':>17}")
        print(row + ("   <- the METRIC's null, not a panel" if klass == "NULL" else ""))
    print("\n   Leave-one-out direction agreement, percentile bootstrap over pairs.")
    print("   0.500 = random directions in any dimension: an analytic null, not measured.")
    print("   A bar at the floor means the arm did not resolve the edit. Whether that is because")
    print("   the two vectors are BITWISE IDENTICAL or because the change has no consistent")
    print("   direction is in fig_a.csv as degenerate_pairs, not on the plate.")
    print("   null_enumerate and null_kekulize are the SAME molecule written two ways: there a")
    print("   HIGH number is a failure, and 0.500 is the correct answer.")

    # THE NULL IS AN ASSERTION, NOT A FOOTNOTE. If a matched-mass substitution -- two unrelated
    # compounds -- ever scores well above chance, the split leaked and every bar above is inflated.
    # "mean" IS NOT A KEY OF THESE CELLS -- it was copied from the AUC plate, whose cells have
    # one. The condition was right and the message was right, but building `bad` would have
    # raised KeyError instead of reporting, so the one assertion guarding against a silently
    # inflated plate could not actually fire its own error. Caught on the 2026-10-31 rename.
    bad = {a: round(cells[a][NULL_MODE]["consistency"], 4) for a in armlist
           if NULL_MODE in cells[a] and cells[a][NULL_MODE]["consistency"] > 0.60}
    assert not bad, (
        f"fig_a: the matched-mass null is above 0.60 for {bad}. Two unrelated molecules have no "
        f"shared direction between them, so a high value here means the standardisation or the "
        f"leave-one-out is leaking the answer, and every bar above it is inflated. Do not "
        f"publish the plate; re-check research/figa_consistency.py.")


if __name__ == "__main__":
    main()
