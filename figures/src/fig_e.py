"""Figure E -- the LSD AmpC virtual screen: top-0.01% recovery against screening cost.

    .venv/bin/python figures/src/fig_e.py [screen_ampc.json] [cost.json]

THE ARGUMENT. Figures C and D measure representations on small supervised benchmarks, where a
cost difference of 40x costs a reader nothing because every arm finishes in seconds. A docking
campaign is where that difference becomes the experiment: AmpC is 1,468,863,655 molecules, so
the question is not "which representation is most accurate" but "given a compute budget, which
one recovers the most top-scoring molecules". This plate is that trade, with both axes measured.

READING THE PLATE. Up is better and left is cheaper in both panels. The dashed staircase is the
Pareto frontier -- an arm below and to the right of it is dominated, meaning some other arm is
both better and cheaper. There is no legend by design: each point carries its own name, so the
eye never has to travel to a key and back.

WHY NO AXIS IS INVERTED HERE, unlike figure C. Both plotted metrics are already
higher-is-better (a recovery fraction and a normalized logAUC), so "up is better" needs no
negation and no flipped axis. Figure C inverts because its metric is an error ratio; saying so
here is worth a line, because a reader who has internalized figure C's inversion would
otherwise wonder whether this plate does it too.

THE X-AXIS IS WALL-CLOCK PER MOLECULE ON ONE BOX, and it is the whole point of the figure, so
it has to be a measurement rather than a projection. `measured_on` is required per arm and the
loader REFUSES an axis that mixes hardware: µs/molecule from a 16-vCPU c7i and from a GPU box
are not the same quantity and placing them on one axis would invent a comparison. The cost
figure is featurization plus prediction, i.e. everything phase 2 actually spends, because a
screener cannot buy featurization without scoring.

MINIMOL'S X IS A LOWER BOUND, NOT A MEASUREMENT. Its throughput is not flat in N
(`docs/plans/SHOICHET_SCREEN.md`: 2.10x over the measured range, where every other arm is within
1.07x), so the per-molecule cost it will actually incur over 100M molecules is at least the
plotted value and probably more. An arm whose cost entry carries `"lower_bound": true` is drawn
with a right-pointing arrow and must be described that way in the caption. Plotting it as a
plain point would put a number on the axis that the measurement does not support.

THE WHISKER IS THE OBSERVED RANGE OVER FIVE REPLICATES, AND IT IS NOT A CONFIDENCE INTERVAL.
Figure C draws a percentile-bootstrap CI over datasets; there is one dataset here and five
models, so the honest mark is min-to-max over what was actually run. It is deliberately not an
SEM or an SD whisker: five replicates whose training sets share about 13% of their molecules by
construction (80% of each 1M draw comes from a pool of only 15.7M -- see the plan's "Status,
2026-10-29") are not five independent samples, so any interval implying a sampling distribution
would invite a test that cannot be run on this design.

THE REFERENCE LINE IN PANEL (a) IS THE SOURCE PAPER'S OWN NUMBER. Hall et al. report 8.4%
recovery of the true top 0.01% for their best AmpC model at 1M training molecules -- the same
quantity, the same target, the same training budget. It is the only external anchor on the
plate, and an arm below it has not matched the published baseline.

WHAT WOULD FALSIFY THE PAPER'S CLAIM. Any learned embedding landing above the frontier that the
classical and HUME arms define -- better recovery at equal or lower cost. CheMeleon is the
candidate to watch: it is pretrained to predict Mordred descriptors, so if descriptor
information is what matters here it has it, and it is also the most expensive arm on the axis by
an order of magnitude. The plate shows whichever way that falls.

DATA CONTRACT -- two files, because they are produced by two different machines at two
different times and fusing them would let one invent the other.

  1. `results/screen_ampc.json`, written verbatim by `screen/aggregate.py` (phase 3):

        {"n_test": int, "n_test_scored": int, "true_top_0.01pct": int, "replicates": int,
         "arms": {<arm>: {"per_replicate": [...],
                          "summary": {<metric>: {"mean","sd","min","max"}, ...}}}}

  2. `results/figures/figE/cost.json`, the measured cost axis, in the same shape figure C uses:

        {<arm>: {"us_per_mol": float, "measured_on": str,
                 "lower_bound": bool (optional, default false),
                 "breakdown": {...} (optional)}}

Arm keys must exist in `figures/src/arms.py`; an unregistered key draws in default gray on
purpose, so a new arm is visibly unstyled rather than silently missing. This script invents no
numbers -- if either file is absent it says which command produces it and exits.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A                                                      # noqa: E402
# THE FRONTIER RULE LIVES IN fig_c AND IS IMPORTED, NOT COPIED. "Dominated" is a paper-level
# definition and two hand-written copies of it is how the two plates come to disagree about
# which arms are on the frontier. Note that fig_c computes the frontier and does not currently
# draw it, so this figure is the only place the rule reaches a reader -- whether fig_c should
# draw it too is a separate question from this plate.
from fig_c import pareto                                              # noqa: E402
from style import FS, STYLE, check_font, save, title                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "results" / "screen_ampc.json"
COST = ROOT / "results" / "figures" / "figE" / "cost.json"
STEM = "fig_e"

#: AmpC's full library size, for the box-hours column in the CSV dump. From Hall et al. 2025
#: (lsd.docking.org); the screen itself measures a uniform 100M sample of it, and the cost axis
#: is per molecule, so the full-library cost is a multiplication rather than another measurement.
N_AMPC = 1_468_863_655
#: Hall et al. 2025, J. Chem. Inf. Model. 65, 4458-4467: best AmpC model at 1M training
#: molecules recovers 8.4% of the true top 0.01%. The same quantity this figure plots.
HALL_RECOVERY = 0.084

#: ONE PANEL PER METRIC, and the x positions are identical in both, because cost is a property
#: of the featurization and not of the metric. That is what makes the pair readable as one
#: plate: the reader sees whether the frontier's SHAPE survives a change of metric, which is the
#: real question behind "is recovery@0.01% a fragile way to rank these arms".
PANELS = (
    dict(key="recovery_0.01pct", scale=100.0,
         title="recovery of the true top 0.01%",
         # `subtitle` is filled from the results file, so the panel states how many molecules
         # the percentage is OF rather than leaving the reader to multiply it out.
         subtitle="% of the {k:,} true top scorers, ↑ better",
         ylabel="recovery (%)",
         ref=HALL_RECOVERY * 100.0, ref_text="Hall et al. 2025"),
    dict(key="log_auc_0.01pct", scale=1.0,
         title="enrichment over the whole curve",
         subtitle="logAUC, 0 random, 1 perfect, ↑ better",
         ylabel="logAUC, top 0.01%",
         ref=None, ref_text=None),
)

#: Minimum vertical separation between two point labels, as a fraction of the panel's y-range.
#: A 7 pt label in a panel about 150 pt tall needs ~0.06 to clear its neighbor; 0.065 leaves a
#: hairline of white between them. Labels that still cannot be separated are reported, not
#: silently overlapped.
LABEL_GAP = 0.065


def load() -> tuple[dict, dict]:
    perf, cost = PERF, COST
    if len(sys.argv) > 1:
        perf = Path(sys.argv[1])
        print(f"  reading {perf}  (override; not the canonical results file)")
    if len(sys.argv) > 2:
        cost = Path(sys.argv[2])
        print(f"  reading {cost}  (override; not the canonical cost file)")
    if not perf.exists():
        raise SystemExit(
            f"figure E: no screen results at {perf}.\n"
            f"  Phase 2 scores the 100M test set and phase 3 aggregates it:\n"
            f"    ARMS=\"all\" ./scripts/launch_screen2.sh      # phase 2, on AWS\n"
            f"    .venv/bin/python screen/aggregate.py        # phase 3, locally -> {perf.name}\n"
            f"  This script deliberately does NOT invent placeholder numbers.")
    if not cost.exists():
        raise SystemExit(
            f"figure E: no measured cost axis at {cost}.\n"
            f"  Cost is the x-axis, so an arm without one cannot be placed. The file is\n"
            f"  {{<arm>: {{\"us_per_mol\": float, \"measured_on\": str, \"lower_bound\": bool}}}};\n"
            f"  see this file's docstring. Every value must come from one machine, measured,\n"
            f"  not quoted from a model's own paper.")
    d = json.loads(perf.read_text())
    c = json.loads(cost.read_text())
    for k in ("arms", "n_test_scored", "replicates", "true_top_0.01pct"):
        if k not in d:
            raise SystemExit(f"figure E: {perf.name} is missing '{k}'; it was not written by "
                             f"screen/aggregate.py, or was written by an older version of it.")
    if not d["arms"]:
        raise SystemExit(f"figure E: {perf.name} holds no arms.")
    missing = sorted(set(d["arms"]) - set(c))
    if missing:
        raise SystemExit(
            f"figure E: no measured cost for {missing}. Cost is the x-axis; an arm without one "
            f"cannot be placed. Measure it on the same box as the others, or drop the arm.")
    boxes = sorted({c[a].get("measured_on") for a in d["arms"]})
    if len(boxes) != 1 or boxes[0] in (None, ""):
        raise SystemExit(
            f"figure E: the cost axis mixes hardware ({boxes}). µs/molecule on one box type is "
            f"not comparable to µs/molecule on another, so two machines on one axis would "
            f"invent a comparison. Record `measured_on` per arm and re-measure the odd one out.")
    for a in d["arms"]:
        for p in PANELS:
            s = d["arms"][a].get("summary", {}).get(p["key"])
            if s is None:
                raise SystemExit(
                    f"figure E: arm {a} has no summary for {p['key']!r}. Available: "
                    f"{sorted(d['arms'][a].get('summary', {}))}. Re-run screen/aggregate.py.")
            for f in ("mean", "min", "max"):
                if f not in s:
                    raise SystemExit(f"figure E: {a}/{p['key']} has no {f!r}; the whisker is the "
                                     f"observed replicate range and cannot be drawn without it.")
    print(f"  {len(d['arms'])} arms, {d['n_test_scored']:,} test molecules, "
          f"{d['replicates']} replicates, cost measured on {boxes[0]}")
    return d, c


def _place_labels(ax, xs, ys, keys, right_of):
    """Name every point beside itself, nudged apart where two would overlap.

    THE PLATE HAS NO LEGEND (Leif), so this IS the key, and it has to survive data the script
    has never seen: six arms whose y-values are not known until phase 3 runs can land anywhere,
    including on top of each other. The rule is deterministic -- offset to the chosen side,
    then, going up one side at a time, push any label that sits within LABEL_GAP of the one
    below it up to exactly LABEL_GAP. A label moved far enough that it no longer reads as
    belonging to its point is REPORTED rather than quietly left there, because a misattributed
    name on a plate with no legend is worse than a missing one.
    """
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    # PLAIN bools, not numpy's. `right_of` is built from a comparison on a float64 array, whose
    # elements are np.bool_ -- and `np.True_ is True` is False, so an identity test against the
    # side silently selected nothing and the plate rendered with no labels at all on a figure
    # that has no legend to fall back on. Compared by value, and narrowed at the boundary.
    side_of = [bool(v) for v in right_of]
    moved = []
    for side in (True, False):
        idx = sorted((i for i in range(len(xs)) if side_of[i] == side), key=lambda i: ys[i])
        at = None
        for i in idx:
            y = ys[i]
            if at is not None and y - at < LABEL_GAP * span:
                y = at + LABEL_GAP * span
            at = y
            shifted = abs(y - ys[i]) > 0.3 * LABEL_GAP * span
            if shifted:
                moved.append((keys[i], (y - ys[i]) / span))
            ax.annotate(A.short_label(keys[i]), (xs[i], y),
                        xytext=(8 if side else -8, 0), textcoords="offset points",
                        ha="left" if side else "right", va="center",
                        fontsize=FS["annot"], color=STYLE["ink"],
                        # A WHITE HALO, NOT AN OPAQUE BOX. The frontier staircase and the
                        # gridlines pass behind these names and a box would punch rectangles
                        # out of both; a stroke clears only the glyphs, so the dashes still
                        # read as a continuous line between the letters.
                        path_effects=[patheffects.withStroke(linewidth=2.2,
                                                             foreground="white")],
                        # A LEADER LINE ONLY WHEN THE LABEL HAD TO MOVE. An unmoved label sits
                        # beside its own marker and needs nothing; a moved one is the only case
                        # where a reader could attach a name to the wrong point, which on a
                        # plate with no legend is the one unrecoverable error.
                        arrowprops=(dict(arrowstyle="-", lw=0.5, color=STYLE["mute"],
                                         shrinkA=0, shrinkB=3) if shifted else None),
                        annotation_clip=False, zorder=8)
    if moved:
        print("  labels nudged to clear a neighbor (fraction of the y-range, leader drawn): "
              + ", ".join(f"{k} {dy:+.2f}" for k, dy in moved))


def main() -> None:
    check_font()
    d, cost = load()
    keys = A.order(list(d["arms"]))
    xs = np.array([float(cost[a]["us_per_mol"]) for a in keys])
    lower = [bool(cost[a].get("lower_bound", False)) for a in keys]
    box = cost[keys[0]]["measured_on"]

    fig = plt.figure(figsize=(STYLE["col2"], 3.05))
    gs = fig.add_gridspec(1, len(PANELS), hspace=0.0, wspace=0.22,
                          left=0.065, right=0.99, top=0.82, bottom=0.175)
    tags = "abcdefgh"

    rows = {}
    for p_i, p in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, p_i])
        sc = p["scale"]
        ys = np.array([d["arms"][a]["summary"][p["key"]]["mean"] * sc for a in keys])
        los = np.array([d["arms"][a]["summary"][p["key"]]["min"] * sc for a in keys])
        his = np.array([d["arms"][a]["summary"][p["key"]]["max"] * sc for a in keys])

        # Y-RANGE SET BEFORE ANYTHING IS LABELLED, because _place_labels measures its minimum
        # separation as a fraction of the range and matplotlib would otherwise autoscale to the
        # annotations themselves -- the labels would then set the range they were spaced against.
        bot, top = float(min(los)), float(max(his))
        if p["ref"] is not None:
            bot, top = min(bot, p["ref"]), max(top, p["ref"])
        pad = 0.14 * (top - bot) if top > bot else max(abs(top), 1.0) * 0.1
        ax.set_ylim(bot - pad, top + pad)
        ax.set_xscale("log")
        # Horizontal headroom for the labels, which hang off the points in both directions. A
        # log axis autoscaled to the points alone puts the cheapest and most expensive arms
        # against the frame and their names outside it.
        # ASYMMETRIC HEADROOM, because the two sides carry different things. The cheapest
        # arms' names hang to the right of their markers and the most expensive arms' names
        # hang to the LEFT, back over the panel, so the right margin only has to clear a
        # marker -- or a lower-bound arrow, which reaches 2x its own point.
        right = float(xs.max()) * 1.6
        for i, lb in enumerate(lower):
            if lb:
                right = max(right, float(xs[i]) * 2.4)
        ax.set_xlim(float(xs.min()) / 2.4, right)

        if p["ref"] is not None:
            # The ONLY external number on the plate. Thin, dashed and annotated in place --
            # a key for one line would cost more space than the line does.
            ax.axhline(p["ref"], ls=(0, (3, 2)), lw=STYLE["lw_thin"], color=STYLE["mute"],
                       zorder=1)
            ax.annotate(f"{p['ref_text']}  {p['ref']:.1f}%", (0.015, p["ref"]),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=FS["annot"], color=STYLE["mute"], style="italic", zorder=2)

        # THE FRONTIER, drawn UNDER the points so no marker is cut by it. `where="post"` gives
        # the staircase its honest shape: between two frontier arms you can have the cheaper
        # one's performance at any cost in between, and nothing better, so the segment is flat
        # and then steps up.
        front = pareto(list(xs), list(ys))
        if len(front) > 1:
            ax.step(xs[front], ys[front], where="post", ls=(0, (4, 2)),
                    lw=STYLE["lw_thin"], color=STYLE["ink"], zorder=2)
        on_front = {keys[i] for i in front}

        for i, a in enumerate(keys):
            kw = A.bar_kw(a)
            face = kw.pop("color")
            kw.pop("hatch", None)                  # hatch reads as noise at marker size
            # WHISKER OVER MARKER, as in figure C: a range shorter than the 5 pt marker
            # vanishes behind it at the default draw order, and "the replicates agree" then
            # looks identical to "there is only one replicate".
            yerr = np.array([[ys[i] - los[i]], [his[i] - ys[i]]])
            ax.errorbar(xs[i], ys[i], yerr=yerr, fmt="o", ms=STYLE["marker_size"], color=face,
                        ecolor=STYLE["ink"], elinewidth=STYLE["lw_thin"],
                        capsize=STYLE["cap_size"], zorder=5,
                        markeredgecolor=STYLE["ink"], markeredgewidth=0.7)
            if lower[i]:
                # "AT LEAST THIS EXPENSIVE". The arrow is the whole statement: this arm's
                # throughput is not flat in N, so its measured per-molecule cost is a floor and
                # the true point lies somewhere to the right of the marker.
                ax.annotate("", xy=(xs[i] * 2.0, ys[i]), xytext=(xs[i] * 1.12, ys[i]),
                            arrowprops=dict(arrowstyle="-|>", lw=STYLE["lw_thin"],
                                            color=STYLE["ink"], shrinkA=0, shrinkB=0),
                            zorder=4)

        # Labels to the right by default, to the left for the two most expensive arms, whose
        # right-hand side is the frame.
        cut = np.sort(xs)[-2] if len(xs) > 2 else xs.max()
        _place_labels(ax, xs, ys, keys, [x < cut for x in xs])

        ax.grid(axis="both")
        ax.set_ylabel(p["ylabel"], fontsize=FS["label"])
        title(ax, f"{p['title']}\n({p['subtitle'].format(k=d['true_top_0.01pct'])})",
              pad=4)
        ax.text(-0.13, 1.03, tags[p_i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="right")
        rows[p["key"]] = (ys, los, his, on_front)

    # ONE shared x-label for the row. Two copies of a long label overprint each other in the
    # gutter; the quantity is identical in both panels, which is the point of the figure.
    fig.supxlabel(f"featurization + scoring cost ({box}) — µs / molecule, log",
                  fontsize=FS["label"], y=0.055)
    save(fig, STEM)
    plt.close(fig)

    # THE NUMBERS THE PLATE CANNOT CARRY. A whisker narrower than its marker is unreadable
    # whatever the draw order, the per-replicate values are not on the plate at all, and the
    # full-library cost is the quantity a reader actually budgets with -- so none of them may
    # exist only in the figure. Same reason figure C dumps its CIs.
    build = ROOT / "figures" / "build"
    build.mkdir(parents=True, exist_ok=True)
    with open(build / f"{STEM}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "us_per_mol", "cost_measured_on", "cost_is_lower_bound",
                    "box_hours_full_ampc_library"]
                   + [f"{p['key']}_{f}" for p in PANELS
                      for f in ("mean", "sd", "min", "max", "on_frontier")]
                   + ["n_test_scored", "replicates", "true_top_0.01pct"])
        for i, a in enumerate(keys):
            s = d["arms"][a]["summary"]
            out = [a, f"{xs[i]:.2f}", box, int(lower[i]),
                   f"{xs[i] * N_AMPC / 1e6 / 3600:.1f}"]
            for p in PANELS:
                m = s[p["key"]]
                out += [f"{m['mean']:.6f}", f"{m.get('sd', float('nan')):.6f}",
                        f"{m['min']:.6f}", f"{m['max']:.6f}",
                        int(a in rows[p["key"]][3])]
            out += [d["n_test_scored"], d["replicates"], d["true_top_0.01pct"]]
            w.writerow(out)
    print(f"  wrote  figures/build/{STEM}.csv")


if __name__ == "__main__":
    main()
