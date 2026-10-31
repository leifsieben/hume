"""Figure E -- the LSD AmpC virtual screen: top-0.01% recovery against screening cost.

    .venv/bin/python figures/src/fig_e.py [screen_ampc.json] [cost.json]

THE ARGUMENT. Figures C and D measure representations on small supervised benchmarks, where a
40x cost difference costs the reader nothing because every arm finishes in seconds. A docking
campaign is where that difference becomes the experiment: AmpC is 1,568,863,695 molecules, so
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

THE Y-LABEL IS THE SAME SENTENCE ON BOTH PANELS AND DOES NOT MEAN THE SAME SET OF MOLECULES.
"The true top 0.01%" is the top 0.01% OF THAT PANEL'S OWN POPULATION: 9,964 molecules out of
100,000,000 on the left, and the top 0.01% of the scaffold-split subset on the right -- a
smaller set, and not a subset of the first, because re-ranking a different population moves the
cut. Both panels therefore report the same QUESTION ("of the molecules worth finding, how many
did you find") against different answers to "which molecules were worth finding", which is the
only way the two can be compared at all: recovering the full test set's top 0.01% from a subset
that contains only some of them would be bounded above by that fraction and would measure the
filter, not the model. The paper's caption must say this; the plate cannot.

HUME r=2 IS NOT DRAWN, AND THE REASON IS NOW A MEASUREMENT RATHER THAN A PREFERENCE. An earlier
version excluded it as "the radius ablation", which was a judgement made before the screen ran;
it was then put back in, on the grounds that how r=2 behaves on a billion molecules is a result
and not an ablation. The real numbers settle it. Over all 100,000,000 test molecules:

    HUME r=4    19.07% recovery [18.69, 19.78 over 5 replicates]    152.55 us/mol
    HUME r=2    18.65% recovery [18.09, 18.99 over 5 replicates]    152.70 us/mol

The two cost the SAME to 0.1%, and the 0.42-point gap in recovery is smaller than either arm's
own replicate range (1.09 and 0.90 points). The plate's axes therefore cannot separate them: they
land as two markers one marker-width apart whose names have to be nudged off their own points,
and a reader would be invited to read a difference the replicates do not support. Drawing only
r=4 is not hiding the ablation -- both arms' means, sds and ranges are in the CSV beside the
figure, which is where a difference this size belongs. `paper/radius_choice.tex`, which argues
r=3, needs to be reconciled with this; that is the paper's business, not the plate's.

THE ARM IS LABELLED PLAIN "HUME", WITH NO RADIUS (Leif). r=4 is what 1.1.0 ships, so on this
plate the radius is not a qualifier a reader needs. See `figures/src/arms.py` on the name
collision that creates with the `hume` arm of figures B/C/D, and why they must never share a
panel.

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
     metric over test molecules whose Bemis-Murcko scaffold appears in NO training set of ANY
     replicate -- the union, not each replicate's own, which is Leif's call of 2026-10-31. The
     five replicates' scaffold sets turn out to overlap far less than the molecule-level 13%
     suggests (about 404,000 scaffolds each, 1,235,646 in the union, only 59,532 common to all
     five), so the two readings differ by roughly 3x in how much of the test set they remove.
     The union buys ONE FIXED TEST POPULATION: every arm and every replicate is scored on the
     same molecules, so the whisker in panel (b) is model variance exactly as it is in panel (a)
     and the left-to-right drop is a comparison rather than two differently-shaped samples. The
     cost of that choice is stated plainly: each model is scored on molecules ~3x more novel
     than its own training set required, so panel (b) is pessimistic in a way no single model's
     training justifies. It is a floor on generalisation, not an estimate of it.

     The scaffold hashes come from `screen/test_scaffolds.py`, and the train-side hashes are the
     ones frozen in `trainval.npz` by `screen/collect_trainval.py` -- the same blake2b call, not
     a second one written out again. A panel whose metric no arm reports is DROPPED from the
     layout rather than drawn empty, which is how this plate rendered as a single panel for as
     long as the metric did not exist.

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
from style import FS, STYLE, check_font, save, title                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "results" / "screen_ampc.json"
COST = ROOT / "results" / "figures" / "figE" / "cost.json"
STEM = "fig_e"

#: AmpC's full library size, for the box-hours column in the CSV. The screen measures a uniform
#: 100M sample of it and the cost axis is per molecule, so this is a multiplication rather than
#: another measurement.
#:
#: THREE DIFFERENT NUMBERS ARE IN CIRCULATION FOR THIS LIBRARY AND ONLY ONE OF THEM IS THE SET
#: WE SAMPLED (sorted out 2026-10-31, while citing the dataset):
#:
#:     1,568,863,695   the `docking_results` config's own row count -- HuggingFace's
#:                     num_examples, and screen/fetch_scores.py's EXPECT_ROWS, which the fetch
#:                     checks the realised count against. THIS ONE.
#:     1,568,323,216   Hall et al., Table 1, "compounds with docking scores" for AmpC.
#:                     540,479 fewer, unexplained.
#:     1,468,863,655   Hall et al., in the Chemprop section of the same paper. Differs from the
#:                     row count by exactly 100,000,040 and looks like a transcription slip.
#:
#: This constant held the third until today, taken from the paper rather than from the data,
#: which understated every box-hours figure in the CSV by 6.4%. The right number here is the
#: row count of the file the 100M test set was actually drawn from, because that is what the
#: extrapolation claims: our measured us/molecule, times the library we sampled. Quoting a
#: paper's figure for the size of a set we did not take from the paper would be a worse kind of
#: wrong than being 540,479 adrift from Table 1.
#:
#: It is the `docking_results` config ONLY: the dataset-server size endpoint reports
#: 1,569,365,216 for the whole repo, which adds invitro_results (1,521) and poses_top500k
#: (500,000), two configs this benchmark never reads.
N_AMPC = 1_568_863_695

#: ARMS THAT ARE MEASURED BUT NOT DRAWN, each with the reason, because a silently dropped arm
#: and a missing one look identical. Everything here still reaches the CSV.
#:
#: hume_r2 IS HERE ON THE STRENGTH OF ITS OWN MEASURED NUMBERS, not on a prior judgement -- it
#: was excluded, then drawn, and is now excluded again because the real screen separates the two
#: radii by less than the replicates separate each arm from itself. See the module docstring for
#: the four numbers. Both arms reach the CSV.
EXCLUDE: dict[str, str] = {
    "hume_r2": "costs the same as r=4 (152.70 vs 152.55 us/mol) and differs by 0.42 recovery "
               "points against replicate ranges of 1.09 and 0.90 -- the plate cannot separate "
               "them, so only the shipped default is drawn",
}

#: THE TWO PANELS: one metric, two test populations. Sharing the y-axis is the point -- see the
#: module docstring.
#: PANELS ARE WHAT IS DRAWN. CSV_METRICS IS WHAT IS RECORDED, and it is deliberately larger:
#: the house rule is that no number may exist only inside a figure, so a metric dropped from the
#: plate still reaches figures/build/fig_e.csv. Dropping a panel must cost a reader nothing but
#: the picture.
PANELS = (
    # "DOCKING HITS", NOT "INHIBITORS" (Leif 2026-10-31, correcting a first draft of the title).
    # The screen ranks DOCK scores from the dataset's `docking_results` config, so the top 0.01%
    # is the 9,964 best-SCORING molecules of 100,000,000 -- computational predictions. The
    # dataset's 1,521 experimentally tested compounds live in a separate `invitro_results`
    # config this screen never reads, and Liu et al. found 247 true inhibitors among the 1,293
    # top-ranked molecules they assayed, so even at the very top of a docking list most
    # molecules are not inhibitors. "Inhibitors" would have claimed HUME recovers 19.07% of
    # AmpC's actual binders; it recovers 19.07% of the top docking hits.
    #
    # SENTENCE CASE, matching every other title in the set (fig_b/fig_c: "Physicochemical",
    # "ADME & tox", "Classification", "Quantum energy"). AmpC keeps its capitals as a proper
    # noun; "docking hits" does not.
    dict(key="recovery_0.01pct", title="Recovery of top 0.01% AmpC docking hits",
         ylabel="recovery of the true top 0.01%  (%, ↑ better)"),
    # TOP 1% IS MEASURED AND NOT DRAWN (Leif 2026-10-31, "just give me the left half"). It is a
    # real result and a tempting panel -- the ranking FLIPS with depth, ECFP4+RDKit taking the
    # lead at 29.14% against HUME's 28.11%, and the frontier becomes a two-point staircase
    # rather than one dominating point. It is in the CSV in full. Re-enable by moving this entry
    # back into PANELS; the layout, the shared y-range and the per-panel axes all still handle
    # two, and style.save() will want `left` re-measured for the two-panel width.
    # dict(key="recovery_1pct", title="Top 1% of the library",
    #      ylabel="recovery of the true top 1%  (%, ↑ better)"),
)

#: Every metric that reaches the CSV, drawn or not. Panels first, then the rest.
CSV_METRICS = ("recovery_0.01pct", "recovery_1pct", "recovery_0.01pct_scaffold",
               "recovery_1pct_scaffold", "log_auc_0.01pct", "pearson", "rmse_over_sd")

#: Minimum vertical separation between two point labels, as a fraction of the panel's y-range.
#: A 7 pt label in a panel about 150 pt tall needs ~0.06 to clear its neighbor.
LABEL_GAP = 0.065

#: How close below the frontier line a marker may sit, in multiples of the label's own text
#: height, before its name is hung underneath the marker instead of beside it. In text heights
#: rather than a fraction of the y-range because what is being avoided is the name touching the
#: line, which is a typographic distance and does not scale with the data.
FRONTIER_CLEAR = 2.5


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


def _label_points(ax, xs, ys, keys, front_xy=()):
    """Name every point beside itself, clear of its neighbours AND of the frontier.

    The plate has no legend, so this IS the key, and it has to survive data the script has never
    seen. Three things can collide, and earlier versions handled each one wrongly in turn:

    1. ANOTHER LABEL. Tested on BOTH axes. The first version compared only recoveries, so
       CheMeleon (18.04%) and ECFP4+RDKit (17.86%) counted as colliding although they sit
       1,961 us/mol apart -- more than half the panel -- and CheMeleon was shoved onto the
       frontier line with a leader so short it read as a stray hyphen. Each label now has a real
       interval in data units and two collide only if their intervals OVERLAP and their heights
       are within LABEL_GAP. Intervals, not centre distances: both labels on a side run the same
       way from their markers, so two names 580 units apart may not overlap at all.

    2. THE FRONTIER'S HORIZONTAL TREADS. Sampled at the x where the text actually sits, not at
       the marker, and the label is pushed AWAY -- down when the tread is above, up when it is
       below. Both directions are needed: dropping every crowded label is right for a runner-up
       under the line and exactly wrong for ECFP4+RDKit in panel (b), whose own step passes just
       below its name. Frontier points are NOT exempt, which is the case first missed: every
       point on the staircase has a tread running from it toward the expensive side, and the
       expensive side is page-LEFT here, so HUME -- the cheapest arm, the one the plate is
       about -- had the line through its own label in both panels.

    3. THE VERTICAL RAY. A horizontal label cannot lie ALONG a vertical line, which is why the
       first version returned "no frontier here" for that region -- but it can lie ACROSS one,
       and ECFP4+RDKit's name ran straight through the ray in panel (a). Nudging up or down
       cannot help, because the ray spans the panel; the label has to change sides.

    WHICH SIDE A NAME GOES ON IS MEASURED, NOT GUESSED. It used to be "the marker is in the right
    fifth of the panel", a proxy for "there is no room on the right" that knows nothing about how
    long the name is. A name now goes on the right only if it FITS between its marker and the
    cheap edge, and flips to the left if it would cross the vertical ray.
    """
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    x_lo, x_hi = ax.get_xlim()                       # ascending: faster to the right
    fig = ax.figure
    fwd = ax.transData.transform
    # Data units per typographic point, from the transform as it currently stands. Taken after
    # set_xlim/set_ylim -- before them it would describe the default 0-1 axes and every width
    # here would be wrong by the axis scale.
    inv = ax.transData.inverted().transform
    px = fig.dpi / 72.0                              # display pixels per typographic point
    # HORIZONTAL GEOMETRY IS IN DISPLAY PIXELS, NOT DATA UNITS, so this function does not care
    # whether the x-axis is linear or logarithmic. It used to convert one typographic point into
    # a fixed number of data units and reuse that everywhere, which is true only on a linear
    # axis: on a log axis a point is worth a few units near the slow edge and thousands near the
    # fast one, so every width, gap and overlap test would have been evaluated at whatever scale
    # happened to hold at the origin. Names at the slow end would have claimed far too much room
    # and names at the fast end almost none. Y stays in data units -- that axis is linear, and
    # LABEL_GAP and FRONTIER_CLEAR are expressed against the data range and the text height.
    xd = [float(fwd((float(xs[i]), float(ys[i])))[0]) for i in range(len(xs))]
    hi_d = float(fwd((x_hi, y0))[0])
    dy_pt = abs(inv((0.0, px))[1] - inv((0.0, 0.0))[1])
    gap = 8.0 * px                                   # marker-to-text gap, in display pixels
    # 0.52 em per character is a serviceable average for this font at label sizes; it only has
    # to be right to within a character or two to tell "same place" from "half a panel away".
    tw = [0.52 * FS["annot"] * len(A.short_label(k)) * px for k in keys]
    text_h = FS["annot"] * dy_pt

    ray_x = float(fwd((front_xy[-1][0], y0))[0]) if front_xy else None    # the ray descends at the FASTEST point
    ray_top = front_xy[-1][1] if front_xy else None

    def interval(i, side):
        """-> (lo, hi) in DISPLAY PIXELS spanned by label i's text. side True = page-LEFT.

        Page-left is DECREASING throughput, the ordinary orientation. It was the opposite while
        this plate drew an inverted cost axis, and the sign is exactly the kind of thing that
        places every name correctly on one axis and nowhere near its point on the other.
        """
        return ((xd[i] - gap - tw[i], xd[i] - gap) if side
                else (xd[i] + gap, xd[i] + gap + tw[i]))

    def crosses_ray(i, side):
        """Does label i, on `side`, lie ACROSS the frontier's vertical ray?

        The ray descends at the FASTEST frontier point, which is the right-hand end of the
        staircase now rather than the left-hand one. A label below its top and spanning its x
        runs straight through it, and no vertical nudge can help, so the label changes sides.
        """
        if ray_x is None or ys[i] > ray_top:
            return False
        lo, hi = interval(i, side)
        return lo <= ray_x <= hi

    # SIDE: right if the name fits before the fast edge and clears the ray, else left.
    left = []
    for i in range(len(xs)):
        fits_right = (xd[i] + gap + tw[i]) <= hi_d and not crosses_ray(i, False)
        left.append(not fits_right)

    def frontier_at(x):
        """-> the staircase's tread height at throughput `x`, or None past its fast end.

        The staircase is the boundary of the dominated region, f(x) = max{y_i : x_i >= x} over
        the frontier: the best recovery still available at or above this speed. It therefore
        DESCENDS to the right, and is undefined beyond the fastest frontier point, where the
        only thing drawn is the vertical ray.
        """
        if not front_xy or x > front_xy[-1][0]:
            return None
        return next(fy for fx, fy in front_xy if fx >= x)

    moved, dodged = [], []
    for side in (True, False):
        idx = sorted((i for i in range(len(xs)) if left[i] == side), key=lambda i: ys[i])
        placed = []                                  # (lo, hi, y) already on this side
        for i in idx:
            lo, hi = interval(i, side)
            y = ys[i]
            for plo, phi, py in placed:
                if lo <= phi and plo <= hi and y - py < LABEL_GAP * span:
                    y = py + LABEL_GAP * span
            placed.append((lo, hi, y))
            shifted = abs(y - ys[i]) > 0.3 * LABEL_GAP * span
            if shifted:
                moved.append((keys[i], (y - ys[i]) / span))
            # back to data units: the staircase is defined on throughput, not on pixels
            fy = frontier_at(float(inv((0.5 * (lo + hi), 0.0))[0]))
            dodge = 0                                # -1 push down, +1 push up
            # THE DODGE APPLIES TO A NUDGED LABEL TOO. It used to be skipped whenever the label
            # had already moved to clear a neighbour, on the theory that two corrections would
            # fight. In practice the nudge is what PUSHES a label onto a tread -- CheMeleon,
            # lifted clear of ECFP4+RDKit, landed exactly on the frontier line -- so the one
            # case the guard excluded was the one that needed it most. A nudged label already
            # carries a leader back to its marker, so a further 4 pt cannot orphan it.
            if fy is not None and abs(fy - y) < FRONTIER_CLEAR * text_h:
                # A tread at or above the label pushes it down; only one clearly below lifts it.
                # Equality goes DOWN: that is a label sitting ON its own tread, and the open
                # space is underneath it.
                dodge = -1 if (fy - y) >= -0.25 * text_h else +1
                # UNLESS THE LABEL WAS ALREADY NUDGED, in which case the dodge must not undo the
                # nudge. A nudged label was lifted to clear the neighbour directly below it, so
                # sending it back down puts it onto that neighbour's marker -- CheMeleon, raised
                # off ECFP4+RDKit and then dropped onto it by a tread 0.03 points overhead. The
                # two corrections were solving different problems in opposite directions; the
                # nudge is the one with a reason to prefer a side, so it wins the tie.
                if shifted:
                    dodge = +1
                dodged.append(f"{keys[i]} {'down' if dodge < 0 else 'up'}")
            ax.annotate(A.short_label(keys[i]), (xs[i], y),
                        # Offsets are in DISPLAY points, so they mean left and right on the page
                        # whatever the data axis is doing -- which matters here, because the
                        # cost axis is inverted.
                        xytext=(-8 if side else 8, -4 if dodge < 0 else (4 if dodge else 0)),
                        textcoords="offset points",
                        ha="right" if side else "left",
                        va={-1: "top", 0: "center", 1: "bottom"}[dodge],
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
    if dodged:
        print(f"  labels moved off the frontier: {', '.join(dodged)}")


def main() -> None:
    check_font()
    d, cost = load()
    keys = A.order([a for a in d["arms"] if a not in EXCLUDE])
    # THE MEASUREMENT IS MICROSECONDS PER MOLECULE; THE AXIS IS ITS RECIPROCAL (Leif
    # 2026-10-31). `us` stays around because the Pareto frontier is defined on cost and must not
    # be re-derived from screen positions, and because the CSV reports the measured quantity.
    us = np.array([float(cost[a]["us_per_mol"]) for a in keys])
    xs = 1e6 / us                                   # molecules per second
    box = cost[keys[0]]["measured_on"]

    # X-LIMITS ONCE, SHARED, AND ASCENDING. THROUGHPUT NEEDS NO INVERSION, which is the whole
    # reason for the change: more is better on both axes, so the best corner is top-right
    # without the axis having to run backwards. Both panels get the same limits because
    # throughput is a property of the featurization, not of the test population.
    #
    # THIS REVERSES A DECISION THIS FILE ARGUED FOR AT LENGTH, so the old reasoning is recorded
    # rather than deleted. The plate used to show microseconds per molecule on an INVERTED axis,
    # on the grounds that a reader lifts a number off the axis and multiplies it by a library
    # size into a bill, and that throughput does not multiply that way. Both halves of that are
    # still true. What outweighed them (Leif) is that an axis whose numbers decrease to the
    # right is read wrong by default, and no arrow fully repairs it; molecules per second
    # ascends left to right like every other axis in the paper, and "faster" needs no gloss.
    # The cost in microseconds is still in figures/build/fig_e.csv, which is where a reader who
    # wants to compute a bill should be looking anyway.
    #
    # WHAT THE RECIPROCAL COSTS, stated because it is not small and it is not a drawing fault.
    # 1/x compresses the slow end: on a linear throughput axis anchored at zero, CheMeleon,
    # MiniMol and ECFP4+RDKit land between 2.0% and 7.3% of the width and four of six arms sit
    # in the left 15%. The cost axis crowded the CHEAP end instead. Neither ordering is wrong --
    # they are the same six numbers -- but the eye reads spacing as importance, so this axis
    # makes the fast arms look well separated and the slow ones interchangeable, where the cost
    # axis did the opposite. The labels are placed rather than left to fall where they may,
    # which is what keeps the crowded end legible.
    #
    # LOG, NOT LINEAR (Leif 2026-10-31). On a linear throughput axis anchored at zero the
    # reciprocal piles the slow arms up: CheMeleon, MiniMol and ECFP4+RDKit landed between 2.0%
    # and 7.3% of the width, four of six arms sat in the left 15%, and a third of the plate was
    # empty between HUME and ECFP4. The eye reads spacing as importance, so that version made
    # the fast arms look well separated and the slow ones interchangeable.
    #
    # The objection that retired a log COST axis does not transfer. There the complaint was
    # that log makes a 20x cost difference look like a short step, and cost is the quantity a
    # reader multiplies by a library size into a bill, so foreshortening it understates the
    # thing the plate exists to show. On a throughput axis a reader is comparing ratios --
    # "twice as fast", "fifty times as fast" -- and a log axis renders equal ratios as equal
    # distances, which is the honest geometry for that question. The bill is still recoverable
    # and is in figures/build/fig_e.csv in microseconds per molecule and in box-hours.
    #
    # Limits are padded by a constant FACTOR rather than a constant amount, because on a log
    # axis that is what "a margin" means.
    x_slow = float(xs.min()) / 1.45
    x_fast = float(xs.max()) * 1.45
    # ONE Y-RANGE FOR BOTH PANELS, computed before either is drawn. The axes are shared, so the
    # last set_ylim would otherwise win for both and crop the other panel's points off the plate
    # -- and a shared range is what makes the left-to-right drop readable as a distance.
    vals = [d["arms"][a]["summary"][p["key"]][f] * 100.0
            for p in PANELS for a in keys for f in ("min", "max")
            if p["key"] in d["arms"][a].get("summary", {})]
    y_bot, y_top = min(vals), max(vals)
    y_pad = 0.10 * (y_top - y_bot) if y_top > y_bot else max(abs(y_top), 1.0) * 0.1

    # WHICH PANELS HAVE DATA, decided before the layout exists. A panel whose metric no arm
    # reports is dropped from the grid rather than drawn as an empty cell: with only two panels
    # an empty one is half the plate, and a reader's eye spends as long on the hole as on the
    # data. Figure C keeps and marks its empty cells because it has four and the grid shape
    # itself carries meaning; this plate has no such grid to preserve. What was dropped is said
    # on the console, because a silently missing panel and a metric nobody computed look alike.
    panels = [pp for pp in PANELS
              if any(pp["key"] in d["arms"][a].get("summary", {}) for a in keys)]
    for pp in PANELS:
        if pp not in panels:
            print(f"  panel {pp['key']!r} NOT DRAWN: no arm reports it")
    if not panels:
        raise SystemExit("figure E: no arm reports any plotted metric.")

    # THE TWO-PANEL PLATE LANDS AT EXACTLY THE PAGE TEXT BLOCK (Leif). `figsize` is col2 = the
    # 6.69in text block, but savefig(bbox_inches="tight") crops the canvas to its content, so the
    # width actually written is whatever is left after the slack margins go -- 6.647in at the
    # previous left=0.105, and the set as a whole spans 6.48 to 6.89. `left` is therefore tuned,
    # not guessed: 0.090 is the value at which the cropped PDF measures 6.692in, within 0.02% of
    # the text block, so \includegraphics[width=\textwidth] scales it by 1.000 and the point
    # sizes set in style.FS are the point sizes that print. Re-measure it if the y-label, the
    # tick format or the panel count changes -- style.save() prints the width on every render and
    # warns past 5%, which is the check this number has to keep satisfying.
    # WIDTH FOLLOWS THE PANEL COUNT. Two panels fill the text block; ONE panel is authored at
    # col1, the single-column width, because that is how a lone Pareto plate is set -- in a
    # wrapfig beside the text (Leif 2026-10-31) rather than stretched across the page. A
    # one-panel plate at col2 would be a very wide, very short box with five points adrift in
    # it, and would print its 7 pt labels at 7 pt only if nothing scaled it on the way in.
    solo = len(panels) == 1
    width = STYLE["col1"] if solo else STYLE["col2"]
    fig = plt.figure(figsize=(width, 2.95 if solo else 3.05))
    # `left` is MEASURED, not guessed, for each layout: savefig(bbox_inches="tight") crops to
    # content, so the width written is whatever survives the slack margins. These two values are
    # the ones at which the cropped PDF lands within 0.1% of its declared width. Re-measure if
    # the y-label, the tick format or the panel count changes -- style.save() prints the width
    # on every render and warns past 5%, which is the check these numbers have to keep passing.
    gs = fig.add_gridspec(1, len(panels), wspace=0.26,
                          left=0.115 if solo else 0.090,
                          right=0.975 if solo else 0.995,
                          top=0.88 if solo else 0.87, bottom=0.165)
    tags = "abcdefgh"
    ax0, frontiers = None, {}
    for p_i, p in enumerate(panels):
        # sharey, because the left-to-right drop IS the measurement the pair reports.
        ax = fig.add_subplot(gs[0, p_i], sharey=ax0)
        ax0 = ax0 or ax
        have = [a for a in keys if p["key"] in d["arms"][a].get("summary", {})]
        if len(have) != len(keys):
            raise SystemExit(
                f"figure E: {p['key']!r} exists for {have} but not for "
                f"{sorted(set(keys) - set(have))}. A frontier drawn over a subset would call an "
                f"arm undominated because its competitor is missing, not because it won.")

        ys = np.array([d["arms"][a]["summary"][p["key"]]["mean"] * 100.0 for a in keys])
        los = np.array([d["arms"][a]["summary"][p["key"]]["min"] * 100.0 for a in keys])
        his = np.array([d["arms"][a]["summary"][p["key"]]["max"] * 100.0 for a in keys])

        # THE FRONTIER, under the points so no marker is cut by it. COMPUTED ON COST, NOT ON
        # THE PLOTTED THROUGHPUT. "Dominated" means another arm is both cheaper and better, and
        # that is a statement about the measurement; deriving it from whatever the axis happens
        # to be showing is how a plate comes to disagree with its own CSV. Membership is of
        # course identical under a monotone reciprocal -- which is exactly why it costs nothing
        # to compute it on the honest quantity. `pareto` returns ascending cost, so reversing
        # gives ascending THROUGHPUT, in which recovery necessarily descends: no point on the
        # frontier can be both faster and better than another.
        front = list(reversed(pareto(list(us), list(ys))))
        # DRAWN AS THE BOUNDARY OF THE DOMINATED REGION, f(x) = max{y_i : x_i >= x} over the
        # frontier -- the best recovery still available at or above this speed. It descends to
        # the right, and every arm that is both slower and worse than some frontier point sits
        # underneath it. Each segment is a claim the measurements support and only that:
        #
        #   the treads say   at this throughput or less, nothing measured here beats this
        #   the risers say   buying that extra recovery costs you this much speed
        #   the final drop   at the fastest frontier point, and NOT extended further right,
        #                    which would claim a recovery at a speed nothing was run at
        #
        # A SINGLE FRONTIER POINT IS A REAL OUTCOME, NOT AN EMPTY ONE: one arm both fastest and
        # best dominates everything and the staircase is one tread plus the closing drop. The
        # guard was `len > 1`, which drew nothing in exactly the case the plate most needs to
        # show, and the AmpC screen landed in it before the ECFP4 control was added.
        if front:
            fxs = [float(xs[i]) for i in front]
            fys = [float(ys[i]) for i in front]
            # where="post" holds each y until the next x, so the point list is (x of this step,
            # y of the NEXT one): the horizontal runs at the better recovery and the drop
            # happens AT the faster arm, which is what the boundary actually does.
            px = [x_slow] + fxs[:-1] + [fxs[-1]]
            py = [fys[0]] + fys[1:] + [y_bot - y_pad]
            ax.step(px, py, where="post", ls=(0, (1, 2)),
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

        ax.set_xscale("log")
        ax.set_xlim(x_slow, x_fast)                  # ascending: faster to the right, no inversion
        ax.set_ylim(y_bot - y_pad, y_top + y_pad)
        _label_points(ax, xs, ys, keys,
                      front_xy=[(float(xs[i]), float(ys[i])) for i in front])
        ax.grid(axis="both")
        # PLAIN GRIDLINES AT ROUND THROUGHPUTS, NOT A LOG DECADE GRID (Leif: "normal lines not
        # the log lines, it's too crowded"). A log axis defaults to labelling 10^3 and 10^4 and
        # nothing between, which over the 1.7 decades here leaves a reader with two lines to
        # place six points against; adding the 2..9 minor grid fixed that and made the panel
        # busy. Explicit ticks at 500/1k/2k/5k/10k/20k give the same readability with six lines
        # instead of eighteen, and read as ordinary round numbers rather than as powers.
        # set_xticks on a log axis also disables the minor ticks it would otherwise draw.
        ticks = [t for t in (200, 500, 1000, 2000, 5000, 10000, 20000)
                 if x_slow <= t <= x_fast]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t//1000}k" if t >= 1000 else str(t) for t in ticks])
        ax.tick_params(axis="x", which="minor", bottom=False)
        # LABELLED AND TICKED ON BOTH PANELS (Leif), even though the scale is shared. The two
        # panels are read as a pair and compared across the gutter, so a reader's eye arrives at
        # the right-hand panel with no axis beside it and has to travel back across the plate to
        # find out what the height means. Repeating the axis costs a little width and removes
        # that trip. The scale itself stays shared -- that is what makes the drop measurable.
        ax.set_ylabel(p["ylabel"], fontsize=FS["label"])
        ax.tick_params(labelleft=True)
        # THE X-AXIS IS LABELLED ON BOTH PANELS (Leif), for the reason the y-axis is: the pair
        # is read across the gutter, and an axis a reader has to look elsewhere to identify is
        # one they will read wrong -- especially this one, whose numbers DECREASE to the right.
        # The arrow is not decoration; it is what stops the inversion being misread.
        ax.set_xlabel("molecules / second     faster →", fontsize=FS["label"], labelpad=2)
        title(ax, p["title"], pad=4)
        # NO PANEL TAG ON A ONE-PANEL PLATE. The letter exists so a caption can say "(b)";
        # with a single panel there is nothing to disambiguate and it reads as a stray glyph.
        if len(panels) > 1:
            ax.text(-0.17, 1.04, tags[p_i], transform=ax.transAxes,
                    fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="right")

    # NO figure-level x-label: each panel now carries its own (see the draw loop).
    #
    # The box type is deliberately NOT on the axis, for the reason figure C took it off its own:
    # the instance is an implementation detail of how the number was produced and belongs in the
    # caption text, not on the plate. It is not lost -- the loader REFUSES an axis that mixes
    # hardware, which enforces what a label would merely have advertised, and the box is printed
    # on every render and written into every row of the CSV.
    save(fig, STEM, target_in=width)
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
                   + [f"{k}_{f}" for k in CSV_METRICS
                      for f in ("mean", "sd", "min", "max", "on_frontier")]
                   + ["n_test_scored", "replicates", "true_top_0.01pct"])
        for a in A.order(list(d["arms"])):
            c = cost.get(a, {})
            us = float(c["us_per_mol"]) if "us_per_mol" in c else float("nan")
            row = [a, int(a not in EXCLUDE), f"{us:.2f}", c.get("measured_on", ""),
                   f"{us * N_AMPC / 1e6 / 3600:.1f}"]
            for k in CSV_METRICS:
                m = d["arms"][a].get("summary", {}).get(k)
                # `on_frontier` is blank, not 0, for a metric that was never drawn: a frontier is
                # a property of a panel, and writing 0 would assert "measured, and not on it".
                row += ([f"{m['mean']:.6f}", f"{m.get('sd', float('nan')):.6f}",
                         f"{m['min']:.6f}", f"{m['max']:.6f}",
                         (int(a in frontiers[k]) if k in frontiers else "")]
                        if m else ["", "", "", "", ""])
            row += [d["n_test_scored"], d["replicates"], d["true_top_0.01pct"]]
            w.writerow(row)
    print(f"  wrote  figures/build/{STEM}.csv")


if __name__ == "__main__":
    main()
