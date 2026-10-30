"""Figure E -- the LSD AmpC virtual screen: top-0.01% recovery against screening cost.

    .venv/bin/python figures/src/fig_e.py [screen_ampc.json] [cost.json]

THE ARGUMENT. Figures C and D measure representations on small supervised benchmarks, where a
40x cost difference costs the reader nothing because every arm finishes in seconds. A docking
campaign is where that difference becomes the experiment: AmpC is 1,468,863,655 molecules, so
the question is not "which representation is most accurate" but "given a compute budget, which
one recovers the most top-scoring molecules". This plate is that trade, with both axes measured.

READING THE PLATE -- TOP RIGHT IS BEST, which is why the cost axis runs backwards.

A Pareto plate is read from a corner, and a reader arrives already knowing which corner: best
sits top-right (or bottom-left) and the frontier bows toward it. Plotted the obvious way --
cost ascending to the right, recovery up -- the best corner is top-LEFT, and the plate was
genuinely confusing to read even for the people who made it.

So the x-axis is INVERTED rather than the quantity replaced. That is this paper's standing rule,
set in figure C for its error metrics: flip the axis, keep the field's own units, so the numbers
a reader lifts off the plate are still microseconds per molecule and still multiply by a library
size. Cheaper is to the right, better is up, the frontier bows toward the top-right corner, and
anything below-and-left of it is dominated -- something else is both better and cheaper.

(The alternative was an x-axis of molecules per second, which ascends left-to-right with no
inversion. It was rejected because it changes the unit: throughput does not multiply by 1.47
billion into a bill, and the cost axis in figures C and D is in microseconds per molecule.)

LEFT PANEL AND RIGHT PANEL ARE THE SAME MEASUREMENT ON TWO TEST POPULATIONS, which is why they
share a y-axis. Make-on-demand libraries are dense in analog series, so a test molecule can be a
near-twin of a training molecule without being identical to it. Every arm is flattered by that,
but not necessarily equally -- representations differ in how much they reward a near-duplicate --
so the honest number is the scaffold-split one and the pair is what shows the size of the effect.
A shared y-axis makes the drop from left to right a vertical distance the eye can measure
directly, rather than two panels a reader has to subtract by reading tick labels.

HUME r=2 IS NOT DRAWN (see EXCLUDE). It is the radius ablation, not an arm of the comparison,
and six points on a scatter with no legend is where the labels start fighting. Its numbers are
in the CSV beside the figure, so nothing is lost -- only the plate is quieter.

EVERY ARM'S X IS ONE MEASURED NUMBER, AND NONE OF THEM IS A BOUND. An earlier version of this
plate drew MiniMol with an arrow meaning "at least this expensive", on the plan's statement that
its throughput is not flat in N. The number behind that is real -- MiniMol's cost per molecule
moves 2.10x across the measured decades where every other arm moves under 1.08x -- but the
direction was read backwards. From results/scale/c7i.4xlarge_cpu.json:

    n = 10,000      3,243.9 us/mol
    n = 100,000     1,730.8 us/mol
    n = 1,000,000   1,544.7 us/mol

It gets CHEAPER per molecule as N grows, because a fixed startup cost -- model load into a
separate interpreter -- amortizes. The last two points fit 20.7 s of overhead plus a marginal
1,524 us/mol, so the 1,544.7 measured at a million molecules is 1.4% above the asymptote and is
an UPPER bound on the cost at a hundred million, not a lower one. An arrow pointing toward more
expensive was the wrong way round, and a plate with no legend cannot carry a symbol that needs a
paragraph. One measured number per arm, taken at the largest N measured.

Note this also means figure D's `is_flat()` gate drops MiniMol for a reason that is not the one
the gate is checking for. The gate asks whether us/mol moves across decades, in order to license
extrapolating to 1e9; an arm whose cost is still FALLING and converging extrapolates more
safely than a flat one, not less. That is figure D's business, not this plate's, but it should
not be left unexamined.

THE WHISKER IS THE OBSERVED RANGE OVER FIVE REPLICATES, AND IT IS NOT A CONFIDENCE INTERVAL.
Figure C draws a percentile-bootstrap CI over datasets; there is one dataset here and five
models, so the honest mark is min-to-max over what was actually run. Deliberately not an SEM or
SD whisker: five replicates whose training sets share about 13% of their molecules by
construction (80% of each 1M draw comes from a pool of only 15.7M -- see the plan's "Status,
2026-10-29") are not five independent samples, and an interval implying a sampling distribution
would invite a test this design cannot support.

THERE IS NO LEGEND, BY INSTRUCTION. Every point is named beside itself.

WHAT WOULD FALSIFY THE PAPER'S CLAIM. Any learned embedding landing above the frontier the
classical and HUME arms define -- better recovery at equal or lower cost -- or a scaffold split
that removes HUME's margin while leaving a learned embedding's intact.

DATA CONTRACT -- two files, because they are produced by different machines at different times
and fusing them would let one invent the other.

  1. `results/screen_ampc.json`, written by `screen/aggregate.py` (phase 3):

        {"n_test_scored": int, "true_top_0.01pct": int, "replicates": int,
         "arms": {<arm>: {"summary": {<metric>: {"mean","sd","min","max"}, ...}}}}

     Panel (a) reads `recovery_0.01pct`; panel (b) reads `recovery_0.01pct_scaffold`, the same
     metric over test molecules whose Bemis-Murcko scaffold appears in NO training set. As of
     this writing aggregate.py computes only the first and declines the second in its docstring;
     panel (b) therefore marks itself not-yet-measured rather than failing the whole render, in
     the same way figure C handles a task that has not been run.

  2. `results/figures/figE/cost.json`, the measured cost axis, in figure C's shape:

        {<arm>: {"us_per_mol": float, "measured_on": str}}

     Once phase 2 has run, build this from its own manifests ({arm}_preds_*.json, which carry
     us_per_mol per shard) rather than from the projected RATES table in
     scripts/launch_screen2.sh. Phase 2 measures every arm over the same 100,000,000 molecules
     on the same box type, which is a better cost axis than any benchmark: it is the cost of
     the very run the y-axis reports.

     `measured_on` is required and the loader REFUSES an axis that mixes hardware:
     microseconds per molecule on a 16-vCPU CPU box and on a GPU box are not the same quantity,
     and putting both on one axis would invent a comparison.

Arm keys must exist in `figures/src/arms.py`; an unregistered key draws in default gray on
purpose. This script invents no numbers -- absent a file it names the command that writes it.
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
# definition and two hand-written copies is how two plates come to disagree about which arms are
# on the frontier. fig_c computes it and does not currently draw it, so this is the only place
# it reaches a reader; whether fig_c should draw it too is a separate question.
from fig_c import pareto                                              # noqa: E402
from style import FS, STYLE, check_font, mark_empty, save, title      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "results" / "screen_ampc.json"
COST = ROOT / "results" / "figures" / "figE" / "cost.json"
STEM = "fig_e"

#: AmpC's full library size, for the box-hours column in the CSV. The screen measures a uniform
#: 100M sample of it and the cost axis is per molecule, so this is a multiplication rather than
#: another measurement.
N_AMPC = 1_468_863_655

#: ARMS THAT ARE MEASURED BUT NOT DRAWN, each with the reason, because a silently dropped arm
#: and a missing one look identical. Everything here still reaches the CSV.
#:
#: EMPTY ON PURPOSE. hume_r2 was in here as "the radius ablation", and that was a judgement made
#: against invented numbers. The radius is a live question -- 1.1.0 moves the default to r=4 and
#: `paper/radius_choice.tex` still argues r=3 -- so how r=2 behaves on a billion-molecule screen
#: is a result to look at, not an ablation to tuck into a CSV. Put an arm back in here only once
#: its real numbers say it adds nothing.
EXCLUDE: dict[str, str] = {}

#: THE TWO PANELS: one metric, two test populations. Sharing the y-axis is the point -- see the
#: module docstring.
PANELS = (
    dict(key="recovery_0.01pct", title="All test molecules"),
    dict(key="recovery_0.01pct_scaffold", title="Scaffold-split from training"),
)
YLABEL = "recovery of the true top 0.01%  (%, ↑ better)"

#: Minimum vertical separation between two point labels, as a fraction of the panel's y-range.
#: A 7 pt label in a panel about 150 pt tall needs ~0.06 to clear its neighbor.
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
            f"  {{<arm>: {{\"us_per_mol\": float, \"measured_on\": str}}}};\n"
            f"  see this file's docstring. Every value must be measured on one machine, not\n"
            f"  quoted from a model's own paper.")
    d = json.loads(perf.read_text())
    c = json.loads(cost.read_text())
    for k in ("arms", "n_test_scored", "replicates", "true_top_0.01pct"):
        if k not in d:
            raise SystemExit(f"figure E: {perf.name} is missing {k!r}; it was not written by "
                             f"screen/aggregate.py, or by an older version of it.")
    if not d["arms"]:
        raise SystemExit(f"figure E: {perf.name} holds no arms.")
    missing = sorted(set(d["arms"]) - set(EXCLUDE) - set(c))
    if missing:
        raise SystemExit(
            f"figure E: no measured cost for {missing}. Cost is the x-axis; an arm without one "
            f"cannot be placed. Measure it on the same box as the others, or drop the arm.")
    drawn = [a for a in d["arms"] if a not in EXCLUDE]
    boxes = sorted({c[a].get("measured_on") for a in drawn})
    if len(boxes) != 1 or boxes[0] in (None, ""):
        raise SystemExit(
            f"figure E: the cost axis mixes hardware ({boxes}). Microseconds per molecule on one "
            f"box type is not comparable to the same number on another, so two machines on one "
            f"axis would invent a comparison. Record `measured_on` per arm and re-measure.")
    for a in drawn:
        s = d["arms"][a].get("summary", {}).get(PANELS[0]["key"])
        if s is None or not {"mean", "min", "max"} <= set(s):
            raise SystemExit(
                f"figure E: arm {a} has no mean/min/max for {PANELS[0]['key']!r}. Available: "
                f"{sorted(d['arms'][a].get('summary', {}))}. Re-run screen/aggregate.py.")
    print(f"  {len(drawn)} arms drawn, {d['n_test_scored']:,} test molecules, "
          f"{d['replicates']} replicates, cost measured on {boxes[0]}")
    for a in sorted(set(d["arms"]) & set(EXCLUDE)):
        print(f"  not drawn: {a} ({EXCLUDE[a]}); its numbers are in the CSV")
    return d, c


def _label_points(ax, xs, ys, keys, on_left):
    """Name every point beside itself, nudged apart where two would overlap.

    The plate has no legend, so this IS the key, and it has to survive data the script has never
    seen. The rule is deterministic: offset to the chosen side, then, walking up one side at a
    time, push any label within LABEL_GAP of the one below it up to exactly LABEL_GAP. A label
    that had to move gets a leader line back to its own point, because a name attached to the
    wrong marker is the one error this plate cannot recover from.
    """
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    # PLAIN bools, not numpy's: `np.True_ is True` is False, and an identity test against a
    # numpy bool once selected nothing at all, rendering the whole plate with no labels.
    left = [bool(v) for v in on_left]
    moved = []
    for side in (True, False):
        idx = sorted((i for i in range(len(xs)) if left[i] == side), key=lambda i: ys[i])
        at = None
        for i in idx:
            y = ys[i] if at is None else max(ys[i], at + LABEL_GAP * span)
            at = y
            shifted = abs(y - ys[i]) > 0.3 * LABEL_GAP * span
            if shifted:
                moved.append((keys[i], (y - ys[i]) / span))
            ax.annotate(A.short_label(keys[i]), (xs[i], y),
                        # Offsets are in DISPLAY points, so they mean left and right on the page
                        # whatever the data axis is doing -- which matters here, because the
                        # cost axis is inverted.
                        xytext=(-8 if side else 8, 0), textcoords="offset points",
                        ha="right" if side else "left", va="center",
                        fontsize=FS["annot"], color=STYLE["ink"],
                        # A WHITE HALO, NOT AN OPAQUE BOX: the frontier and the gridlines pass
                        # behind these names, and a box would punch rectangles out of both.
                        path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
                        arrowprops=(dict(arrowstyle="-", lw=0.5, color=STYLE["mute"],
                                         shrinkA=0, shrinkB=3) if shifted else None),
                        annotation_clip=False, zorder=8)
    if moved:
        print("  labels nudged to clear a neighbor (fraction of the y-range, leader drawn): "
              + ", ".join(f"{k} {dy:+.2f}" for k, dy in moved))


def main() -> None:
    check_font()
    d, cost = load()
    keys = A.order([a for a in d["arms"] if a not in EXCLUDE])
    xs = np.array([float(cost[a]["us_per_mol"]) for a in keys])
    box = cost[keys[0]]["measured_on"]

    # X-LIMITS ONCE, SHARED, AND DESCENDING. Descending limits are the inversion: cheap on the
    # right, expensive on the left, so best is the top-right corner. Both panels get the same
    # limits because cost is a property of the featurization, not of the test population.
    #
    # LINEAR, AND ANCHORED AT ZERO (Leif). A log axis spreads the cheap arms out, but it also
    # flatters them: it makes a 20x cost difference look like a short step, which is the
    # opposite of what this plate is for. On a linear axis anchored at zero, the distance from
    # the right edge IS the cost, and an arm that costs twenty times another sits twenty times
    # further out. The crowding that produces at the cheap end is not a drawing fault -- it is
    # the finding, and it is why the labels are placed rather than left to fall where they may.
    x_cheap = 0.0
    x_dear = float(xs.max()) * 1.12
    # ONE Y-RANGE FOR BOTH PANELS, computed before either is drawn. The axes are shared, so the
    # last set_ylim would otherwise win for both and crop the other panel's points off the plate
    # -- and a shared range is what makes the left-to-right drop readable as a distance.
    vals = [d["arms"][a]["summary"][p["key"]][f] * 100.0
            for p in PANELS for a in keys for f in ("min", "max")
            if p["key"] in d["arms"][a].get("summary", {})]
    y_bot, y_top = min(vals), max(vals)
    y_pad = 0.10 * (y_top - y_bot) if y_top > y_bot else max(abs(y_top), 1.0) * 0.1

    # WHICH SIDE EACH NAME GOES ON, decided by where the marker sits in the panel rather than
    # by how cheap the arm is. A name hangs roughly half an inch off its point, so a marker in
    # the right fifth of the panel has no room on its right -- the frame is there. By rank, the
    # third-cheapest arm kept its name on the right and pushed it straight through the frame.
    span = x_dear - x_cheap
    on_left = [(x_dear - float(x)) / span > 0.78 for x in xs]

    fig = plt.figure(figsize=(STYLE["col2"], 3.05))
    gs = fig.add_gridspec(1, len(PANELS), wspace=0.26, left=0.105, right=0.995,
                          top=0.87, bottom=0.175)
    tags = "abcdefgh"
    ax0, frontiers = None, {}
    for p_i, p in enumerate(PANELS):
        # sharey, because the left-to-right drop IS the measurement the pair reports.
        ax = fig.add_subplot(gs[0, p_i], sharey=ax0)
        ax0 = ax0 or ax
        have = [a for a in keys if p["key"] in d["arms"][a].get("summary", {})]
        if not have:
            # Loud skip, with the reason written into the cell -- the same handling figure C
            # gives a task that has not been run. A blank framed panel and a panel whose arms
            # all scored zero look identical on paper.
            print(f"  SKIPPING panel {p['key']!r}: no arm reports it")
            mark_empty(ax, f"{p['title']}: not measured yet")
            ax.text(0.5, 0.5, f"{p['title']}\n(not measured yet)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=STYLE["mute"],
                    style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            title(ax, p["title"], pad=4)
            continue
        if len(have) != len(keys):
            raise SystemExit(
                f"figure E: {p['key']!r} exists for {have} but not for "
                f"{sorted(set(keys) - set(have))}. A frontier drawn over a subset would call an "
                f"arm undominated because its competitor is missing, not because it won.")

        ys = np.array([d["arms"][a]["summary"][p["key"]]["mean"] * 100.0 for a in keys])
        los = np.array([d["arms"][a]["summary"][p["key"]]["min"] * 100.0 for a in keys])
        his = np.array([d["arms"][a]["summary"][p["key"]]["max"] * 100.0 for a in keys])

        # THE FRONTIER, under the points so no marker is cut by it. `where="post"` is the honest
        # shape: at any budget you can have the best arm at or below that cost and nothing
        # better, so the curve is flat and then steps. Computed on costs and recoveries, not on
        # screen positions, so the axis inversion cannot change which arms are on it.
        front = pareto(list(xs), list(ys))
        if len(front) > 1:
            # EXTENDED LEFTWARD ONLY, to the expensive edge of the panel. That segment states
            # something the measurements support -- above the best arm's cost, nothing measured
            # here does better -- and it turns a two-point frontier from a stub into the
            # boundary it is. It is NOT extended rightward: that would claim a recovery at a
            # cost cheaper than anything on the plate was run at.
            fx = list(xs[front]) + [x_dear]
            fy = list(ys[front]) + [ys[front[-1]]]
            ax.step(fx, fy, where="post", ls=(0, (4, 2)),
                    lw=STYLE["lw_thin"], color=STYLE["ink"], zorder=2)
        frontiers[p["key"]] = {keys[i] for i in front}

        for i, a in enumerate(keys):
            face = A.bar_kw(a)["color"]
            # WHISKER OVER MARKER: a range shorter than the 5 pt marker vanishes behind it at
            # the default draw order, and "the replicates agree" then looks identical to "there
            # is only one replicate".
            ax.errorbar(xs[i], ys[i], yerr=np.array([[ys[i] - los[i]], [his[i] - ys[i]]]),
                        fmt="o", ms=STYLE["marker_size"], color=face, ecolor=STYLE["ink"],
                        elinewidth=STYLE["lw_thin"], capsize=STYLE["cap_size"], zorder=5,
                        markeredgecolor=STYLE["ink"], markeredgewidth=0.7)

        ax.set_xlim(x_dear, x_cheap)                 # descending = inverted = cheap on the right
        ax.set_ylim(y_bot - y_pad, y_top + y_pad)
        _label_points(ax, xs, ys, keys, on_left)
        ax.grid(axis="both")
        # LABELLED AND TICKED ON BOTH PANELS (Leif), even though the scale is shared. The two
        # panels are read as a pair and compared across the gutter, so a reader's eye arrives at
        # the right-hand panel with no axis beside it and has to travel back across the plate to
        # find out what the height means. Repeating the axis costs a little width and removes
        # that trip. The scale itself stays shared -- that is what makes the drop measurable.
        ax.set_ylabel(YLABEL, fontsize=FS["label"])
        ax.tick_params(labelleft=True)
        title(ax, p["title"], pad=4)
        ax.text(-0.17, 1.04, tags[p_i], transform=ax.transAxes,
                fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="right")

    # ONE x-label for the row, carrying the direction. The arrow is not decoration: an axis whose
    # numbers decrease to the right is read wrong without it.
    # ONE x-label for the row: the unit, and the direction. Nothing else (Leif).
    #
    # The box type came off the axis for the reason figure C took it off its own -- the instance
    # is an implementation detail of how the number was produced and belongs in the caption
    # text, not on the plate. It is not lost: the loader REFUSES an axis that mixes hardware,
    # which enforces what the label merely advertised, and the box is printed on every render
    # and written into every row of the CSV.
    #
    # The arrow is not decoration. An axis whose numbers decrease to the right is read backwards
    # without it.
    fig.supxlabel("µs / molecule     cheaper →", fontsize=FS["label"], y=0.045)
    save(fig, STEM)
    plt.close(fig)

    # THE NUMBERS THE PLATE CANNOT CARRY: the arms it does not draw, the per-replicate spread
    # where the whisker is narrower than its marker, and the full-library cost a reader actually
    # budgets with. None of them may exist only in the figure.
    build = ROOT / "figures" / "build"
    build.mkdir(parents=True, exist_ok=True)
    with open(build / f"{STEM}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "drawn", "us_per_mol", "cost_measured_on",
                    "box_hours_full_ampc_library"]
                   + [f"{p['key']}_{f}" for p in PANELS
                      for f in ("mean", "sd", "min", "max", "on_frontier")]
                   + ["n_test_scored", "replicates", "true_top_0.01pct"])
        for a in A.order(list(d["arms"])):
            c = cost.get(a, {})
            us = float(c["us_per_mol"]) if "us_per_mol" in c else float("nan")
            row = [a, int(a not in EXCLUDE), f"{us:.2f}", c.get("measured_on", ""),
                   f"{us * N_AMPC / 1e6 / 3600:.1f}"]
            for p in PANELS:
                m = d["arms"][a].get("summary", {}).get(p["key"])
                row += ([f"{m['mean']:.6f}", f"{m.get('sd', float('nan')):.6f}",
                         f"{m['min']:.6f}", f"{m['max']:.6f}",
                         int(a in frontiers.get(p["key"], ()))] if m else ["", "", "", "", ""])
            row += [d["n_test_scored"], d["replicates"], d["true_top_0.01pct"]]
            w.writerow(row)
    print(f"  wrote  figures/build/{STEM}.csv")


if __name__ == "__main__":
    main()
