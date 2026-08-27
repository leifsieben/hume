"""Fig A — does the representation respond when the CHEMISTRY changes?

ONE script, ONE figure: figures/fig_a.pdf (panels a–m), plus figures/build/fig_a.csv as the
durable numeric record.

    python3 -m figures.fig_a

Ten chemical edits, 1,000 molecule pairs each. In every one the two molecules are GENUINELY
DIFFERENT, so a usable representation has to respond. "Respond" is measured in a unit the reader
can check on the same axis: the shift the change produces, divided by THAT MODEL'S OWN shift when
a completely different compound of matched molecular weight is substituted. 1.00 therefore means
"this change moves the embedding as far as changing the molecule". No threshold anywhere, and the
denominator is measured per model, so a 768-d transformer and a 2048-bit fingerprint sit on one
axis honestly. Panel (k) IS that reference and is 1.000 by construction — it stays on the plate so
the unit is visible rather than asserted.

THE TWO CONTROLS (l, m) ARE THE LOAD-BEARING PART OF THE DESIGN and they invert the reading: the
same molecule written two ways, so a HIGH bar is a FAILURE. They carry a warm tint for that
reason, and the caption MUST state the inversion — a reader who carries the class-A reading
across gets both panels backwards. Without them a large response is uninterpretable, because a
model that reacts to re-writing a SMILES string is reacting to formatting, and its response to a
real edit cannot be separated from that noise.

WHY THE UNIT IS A RATIO AND NOT A THRESHOLD COUNT. Counting dimensions displaced past 0.5 sigma
was proposed and measured on this data. It tracks how DENSE a representation is rather than how
well it resolves an edit — ECFP4 moves ~1.4% of its bits for a completely different compound
where a CLM moves ~82% of its coordinates — and it is threshold-critical exactly where the CLM
arms sit. Kept in the SI as a normalised ratio; not the main axis. See FIGURES.md.

WHISKERS ARE THE INTERQUARTILE RANGE OVER THE 1,000 PAIRS, NOT AN ERROR BAR. Every representation
here is deterministic: re-embedding reproduces the vectors exactly, so there is no sampling noise
for an error bar to describe. The spread shown is chemistry — an inverted stereocentre on a rigid
ring is not the same edit as one on a flexible chain. Reading whisker overlap as "not different"
is therefore wrong, and the caption has to say so.

sigma_j is estimated once per representation on 10,000 background molecules that appear in NO
pair, so no edit can inflate its own denominator.

WHAT THE FIGURE SAYS (three arms as of 2026-08-26; desc and the CLM/GNN roster still landing)
--------------------------------------------------------------------------------------------
1. THE CONTROLS ARE THE RESULT. ChemBERTa-2 moves 1.108 for a Kekulé re-write of the SAME
   molecule — FURTHER than substituting a completely different compound of matched mass — and
   0.772 for a re-written SMILES. Both fingerprints are exactly 0.000 on both, as they must be:
   they are notation-invariant by construction. Without panels (l) and (m) the CLM's mid-range
   class-A numbers would read as respectable sensitivity; with them, an unknown fraction of that
   response is notation noise.
2. STEREOCHEMISTRY AND PROTONATION ARE THE CLM'S BLIND SPOTS. An inverted stereocentre moves
   ECFP4 0.494 and ChemBERTa-2 0.000; a protonation change moves ECFP4 0.835 and ChemBERTa-2
   0.000; E/Z is 0.689 against 0.027.
3. ISOTOPES ARE THE FINGERPRINTS' BLIND SPOT (c). 12C→13C is exactly 0.000 for both Morgan
   variants — the atom invariants do not carry isotope for a same-element substitution — against
   0.402 for ChemBERTa-2. The descriptor block should close this (it carries exact mass), which
   is the single most informative thing the pending `desc` arm will tell us.
4. RADIUS 3 RESOLVES BETTER AND SCORES WORSE DOWNSTREAM. Morgan r=3 beats r=2 on stereo
   (0.559 vs 0.494), added methyl (0.607 vs 0.518) and ring fusion (0.863 vs 0.693), while losing
   to it on 20 of 28 DEV datasets. That dissociation between resolution and downstream value is a
   result in its own right and belongs in the text, not only here.
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

import arms as A                                               # noqa: E402
from style import (FS, STYLE, TINT_CONTROL, TINT_REF, check_font,   # noqa: E402
                           LEGEND_BOX, mark_empty, row_ncol, save)

check_font()
ROOT = Path(__file__).resolve().parents[2]
FIGA = ROOT / "results" / "figures" / "figA"
EMB = FIGA / "embeddings"
INK = STYLE["ink"]

# (mode, class, two-line panel title). CLASS decides the reading rule and the tint:
#   "A"    a real chemical change -- a HIGH bar is good
#   "REF"  the matched-MW substitution that DEFINES 1.00 -- cool tint
#   "B"    notation only, same molecule -- a HIGH bar is a FAILURE -- warm tint
#
# ROW ORDER IS SEMANTIC, not alphabetical. Row 1 collects the edits that leave the plain
# heavy-atom graph untouched or nearly so -- stereochemistry, isotope, charge, one halogen for
# another -- i.e. exactly the properties a graph invariant has to carry explicitly. Row 2 is
# edits to the graph itself. Row 3 is the calibration row: the unit, then the two controls.
# FOURTEEN PANELS ON A 2x7 GRID, NO SPARE CELL, and each row ENDS on a tinted panel
# (Leif 2026-08-26). The rows are semantic, not arbitrary:
#
#   Row 1  the skeleton is untouched and something else changed -- stereochemistry, isotope,
#          charge, element, bond order. These are the properties a graph invariant has to carry
#          explicitly, and they are where the arms disagree most.
#   Row 2  the skeleton itself changed, then the reference that defines the unit.
#
# Each row closes on a panel whose reading rule is different from its neighbours', so the eye
# meets the tint at a row boundary rather than mid-scan.
MODES = [
    ("stereo_flip",    "A",   "Inverted\nstereocentre"),
    ("ez_flip",        "A",   "Flipped E/Z\ndouble bond"),
    ("isotope_13c",    "A",   "$^{12}$C→$^{13}$C,\ngraph unchanged"),
    ("protonate",      "A",   "Protonation\nstate changed"),
    ("halogen_swap",   "A",   "Halogen\nswapped"),
    ("saturate",       "A",   "C=C reduced\nto C–C"),
    ("null_enumerate", "B",   "Re-written\nSMILES"),
    ("h_to_methyl",    "A",   "One methyl\nadded"),
    ("n_methylation",  "A",   "Amide N–H →\nN–methyl"),
    ("scaffold_hop",   "A",   "Aromatic C→N\n(benzene→pyridine)"),
    ("ring_contract",  "A",   "Cyclohexyl →\ncyclopentyl"),
    ("regioisomer",    "A",   "Substituent moved\n(ortho ↔ meta)"),
    ("matched_mw",     "REF", "Different molecules,\nsame MW"),
    ("null_kekulize",  "B",   "Kekulé\nform"),
]
NROW, NCOL = 2, 7
TINT = {"A": None, "REF": TINT_REF, "B": TINT_CONTROL}

# What counts as "0" on the plate. NOT `== 0.0`, which was the first rule and it drew an
# inconsistent figure: the fingerprints and CheMeleon are bitwise invariant to a re-written
# SMILES and label as zero, while MiniMol and untrained Chemprop land at 1e-6 -- float32
# residue from a message-passing forward pass over the same graph reached in a different atom
# order -- and silently got no label at all, which reads as "no bar was drawn" rather than "this
# model does not respond".
#
# 1e-3 is three orders above that residue and 700x below the smallest genuine response anywhere
# on the plate (Chemprop's 0.061 on regioisomer), so nothing real can be swallowed by it. The
# label therefore means "zero to within floating-point reproducibility", not "bitwise identical";
# the CSV carries the unrounded value for anyone who needs the distinction.
ZERO_TOL = 1e-3


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------

def compute():
    """-> (armlist, {arm: {mode: (median, q1, q3, n)}}, {arm: raw matched-MW sigma-RMS}).

    Effect size is the RMS per-dimension change in units of that dimension's spread over the
    10,000 background molecules; relative response divides it by the arm's OWN median matched-MW
    effect size. Arms are whatever embeddings exist -- the roster is still landing, and a figure
    that renders from what is ready beats one that waits for the slowest arm.
    """
    pairs = json.load(open(FIGA / "pairs.json"))["pairs"]
    idx = json.load(open(FIGA / "smiles_index.json"))
    pos = {s: i for i, s in enumerate(idx["order"])}
    n_pair = len(idx["order"])

    out, refs = {}, {}
    for f in sorted(EMB.glob("*.npz")):
        z = np.load(f)
        X = z["X"].astype(np.float64)
        # sigma from the BACKGROUND block only -- rows n_pair: onward appear in no pair.
        sd = np.nanstd(X[n_pair:], axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = np.inf       # dead dims contribute nothing
        Xp = np.nan_to_num(X[:n_pair])

        raw = {}
        for p in pairs:
            ia, ib = pos.get(p["a"]), pos.get(p["b"])
            if ia is None or ib is None:
                continue
            d = (Xp[ia] - Xp[ib]) / sd
            raw.setdefault(p["edit"], []).append(float(np.sqrt(np.mean(d * d))))

        ref = float(np.median(raw.get("matched_mw", [np.nan])))
        assert np.isfinite(ref) and ref > 0, (
            f"{f.stem}: matched-MW reference is {ref}. That is the DENOMINATOR of every cell in "
            f"this arm's column -- without it the arm cannot share the axis and must not be drawn "
            f"on an unnormalised one.")
        refs[f.stem] = ref
        out[f.stem] = {m: (float(np.median(v)) / ref, float(np.percentile(v, 25)) / ref,
                           float(np.percentile(v, 75)) / ref, len(v))
                       for m, v in raw.items() if len(v)}

    assert out, "no embeddings yet -- run embed_pairs.py first"
    n = sorted({c[3] for a in out.values() for c in a.values()})
    assert min(n) >= 1000, f"fig_a: expected >=1000 pairs per cell, found n in {n[:5]}"
    return A.order(out), out, refs


# --------------------------------------------------------------------------------------------
# draw
# --------------------------------------------------------------------------------------------

def _panel(ax, armlist, cells, mode, klass, title, ymax):
    """One mode. Bars are the response RELATIVE to swapping in a different molecule of matched MW.

    The reference line at 1.0 is the whole point of the unit: without it a reader has no way to
    know whether 0.49 is large. With it, panel (a) reads "an inverted stereocentre moves ECFP4
    half as far as a completely different compound, and ChemBERTa-2 not at all".

    EXACT ZEROS ARE LABELLED. Many cells here are 0.000 by construction -- a fingerprint cannot
    see a re-written string, and Morgan invariants cannot see an isotope -- and an unlabelled flat
    baseline is indistinguishable from a bar that was never drawn. That ambiguity is the single
    most likely misreading on this plate, so it is closed at the draw site rather than left to the
    caption.
    """
    if TINT[klass]:
        ax.set_facecolor(TINT[klass])
    x = np.arange(len(armlist))
    med = np.array([cells[a][mode][0] if mode in cells[a] else np.nan for a in armlist])
    q1 = np.array([cells[a][mode][1] if mode in cells[a] else np.nan for a in armlist])
    q3 = np.array([cells[a][mode][2] if mode in cells[a] else np.nan for a in armlist])

    ax.bar(x, med, width=0.80, color=[A.color(a) for a in armlist],
           hatch=None, edgecolor=INK, linewidth=0.45, zorder=3)
    for xi, a in zip(x, armlist):                 # hatch marks PREDICTED, never a second hue
        if A.hatch(a):
            ax.patches[xi].set_hatch(A.hatch(a))
    err = np.vstack([np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)])
    ax.errorbar(x, med, yerr=err, fmt="none", ecolor=INK, elinewidth=0.55,
                capsize=1.3, capthick=0.55, zorder=4)
    ax.axhline(1.0, color=INK, ls=(0, (3, 2)), lw=0.7, zorder=4)
    for xi, v in zip(x, med):
        if np.isfinite(v) and v < ZERO_TOL:
            ax.text(xi, ymax * 0.022, "0", ha="center", va="bottom",
                    fontsize=FS["annot"] - 2.5, color=INK, zorder=5)

    ax.set_ylim(0, ymax)
    ticks = np.arange(0, ymax + 1e-9, 0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xticks([])
    ax.set_xlim(-0.70, len(armlist) - 0.30)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
    ax.set_title(title, fontsize=FS["annot"], fontweight="bold", color=INK, pad=3, loc="center")


def main():
    armlist, cells, refs = compute()
    print(f"arms: {armlist}")
    for a in armlist:
        print(f"  {A.label(a):<24s} matched-MW reference (raw sigma-RMS) = {refs[a]:.4f}")

    # THE CEILING IS COMPUTED, NOT HARDCODED. CLIMB pins YMAX because its arm roster is closed;
    # ours is not -- arms are still being embedded, and a ceiling set on today's three would
    # silently clip tomorrow's. Rounded up to the next half so the tick set stays clean.
    q3max = max(c[2] for a in armlist for c in cells[a].values())
    ymax = max(1.5, np.ceil(q3max / 0.5) * 0.5)
    print(f"  tallest drawn q3 = {q3max:.3f}  ->  y-limit {ymax:g}")

    # The legend row is NOT a gridspec row. hspace is uniform, so a short third row still gets a
    # full row's gap above it and the plate rendered with a dead band taller than the legend
    # itself. Panels get the gridspec; the legend gets its own axes underneath them.
    fig = plt.figure(figsize=(STYLE["col2"], 3.30))
    gs = fig.add_gridspec(NROW, NCOL, left=0.062, right=0.995, top=0.885, bottom=0.215,
                          wspace=0.42, hspace=0.72)
    tags = "abcdefghijklmnopqrstuvwxyz"[:len(MODES)]
    for i, (mode, klass, title) in enumerate(MODES):
        ax = fig.add_subplot(gs[i // NCOL, i % NCOL])
        assert any(mode in cells[a] for a in armlist), f"fig_a: no data for {mode}"
        _panel(ax, armlist, cells, mode, klass, title, ymax)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL == 0:
            ax.set_ylabel("response relative to a\ndifferent molecule", fontsize=FS["annot"])

    assert len(MODES) == NROW * NCOL, (
        f"fig_a: {len(MODES)} panels on a {NROW}x{NCOL} grid. The plate is designed to fill "
        f"exactly -- adding or removing a mode means re-choosing the grid, not leaving a hole.")

    # THE LEGEND LIVES IN THE GRID, in its own short row, not floating below the axes. A legend
    # anchored outside the canvas grows the plate under savefig("tight") and LaTeX then scales the
    # whole figure -- and every font on it -- down to fit the text block. Inside the gridspec it
    # cannot do that, and the rendered width stays pinned to the 6.69in text block.
    lax = fig.add_axes([0.062, 0.008, 0.933, 0.172])
    lax.axis("off")
    mark_empty(lax, "holds the legend")
    handles = [Patch(facecolor=A.color(a), edgecolor=INK, lw=0.6, hatch=A.hatch(a),
                     label=A.label(a)) for a in armlist]
    # ONE ROW up to six arms, TWO beyond. Measured, not guessed: eight arms in one row rendered
    # the plate at 7.15in against a 6.69in text block (+7%), because savefig("tight") grows the
    # canvas to whatever hangs off the axes and the legend -- not the panels -- was setting the
    # width. LaTeX would then scale the plate down and every font on it with it.
    lax.legend(handles=handles, loc="center",
               ncol=row_ncol(handles, rows=1 if len(handles) <= 6 else 2),
               fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.1,
               labelspacing=0.35, borderpad=0.45, **LEGEND_BOX)

    save(fig, "fig_a")
    plt.close(fig)

    # The durable record. Same resolution path as the bars, so the CSV cannot disagree with them.
    with open(ROOT / "figures" / "build" / "fig_a.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "label", "mode", "klass", "relative_response", "q1", "q3", "n",
                    "matched_mw_reference_raw"])
        for a in armlist:
            for mode, klass, _ in MODES:
                if mode in cells[a]:
                    m, lo, hi, n = cells[a][mode]
                    w.writerow([a, A.label(a), mode, klass, f"{m:.6f}", f"{lo:.6f}",
                                f"{hi:.6f}", n, f"{refs[a]:.6f}"])
    print(f"  wrote  figures/fig_a.csv")

    print("\n   " + "mode".ljust(18) + "".join(f"{A.label(a)[:19]:>21s}" for a in armlist))
    for mode, klass, _ in MODES:
        row = f"   {mode:<18}"
        for a in armlist:
            c = cells[a].get(mode)
            row += (f"{c[0]:>7.3f}[{c[1]:.2f},{c[2]:.2f}]" if c else f"{'—':>21}")
        print(row)
    print("\n   median [IQR] over 1,000 pairs. 1.000 = moves the embedding as far as a completely")
    print("   different compound of the same molecular weight; matched_mw IS that reference, so")
    print("   its median is 1.000 by construction and its IQR is the reference's own spread.")
    print("   null_enumerate and null_kekulize are the SAME molecule written two ways: for those")
    print("   two rows a HIGH number is a failure, not a response.")


if __name__ == "__main__":
    main()
