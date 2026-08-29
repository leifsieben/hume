"""Figure C -- performance against inference cost, for every representation.

    .venv/bin/python figures/fig_c.py [results.json]

THE ARGUMENT, and why this is a scatter and not a bar chart.

The paper makes two claims that a bar chart can only carry one of:

    1. a classical featurisation plus descriptors matches or beats every learned embedding;
    2. HUME reaches essentially that performance for a fraction of the cost.

Claim 2 is about the x-axis. Drawing bars of accuracy and putting the timings in a caption
makes the central contribution of the work invisible in the central figure, so cost is a
plotted, measured axis here and the frontier is drawn on it.

READING THE PLATE. One panel per task. Up is better in every panel, always -- for a
lower-is-better metric the axis is INVERTED rather than the value negated, so the numbers on
the axis stay in the units the field uses while the reading direction stays constant. Left is
cheaper. The dashed line is the Pareto frontier: an arm below-and-right of it is dominated,
i.e. something else is both better and cheaper. The claim of the paper is that HUME sits ON
that frontier at the cheap end, and that the learned embeddings sit below and to the right.

WHY THE X-POSITIONS ARE IDENTICAL ACROSS PANELS. Cost is a property of the featurisation, not
of the task, so an arm sits at the same x in all four panels. That is a feature: it lets the
reader see whether the frontier's SHAPE is stable across task families, which is the real
question behind "should this be the default representation".

WHAT WOULD FALSIFY IT. Any learned embedding that lands above the frontier defined by the
classical arms -- better performance at comparable or lower cost. Uni-Mol on the QM panel is
the most likely candidate, because a conformer carries geometry no 2D descriptor block can
reconstruct. It is drawn like everything else, and if it wins there the figure says so.

COST MUST BE MEASURED, NOT CITED. Every number on the x-axis has to come from this machine,
one molecule at a time, including the DL arms' forward passes. A throughput figure quoted from
a model's own paper is measured on different hardware at a different batch size and is not
comparable to a per-molecule C++ featuriser. `cost.measured_on` is required per arm for exactly
this reason, and the axis label states the batch convention.

THE HEAD IS A RESULT, NOT A CONSTANT. Each arm is fitted with its best head rather than with
XGBoost by assumption. `record.head` names the winner, and any arm whose winner is NOT XGBoost
is annotated on the plate -- if a learned embedding only competes under a different head, that
is worth seeing rather than hiding behind a uniform protocol.

DATA CONTRACT -- results/figures/figC/results.json:

    {"meta":  {...free-form provenance...},
     "tasks": [{"key": ..., "label": ..., "metric": ..., "lower_is_better": bool}, ...],
     "arms":  [<arm key>, ...],
     "cost":  {<arm>: {"us_per_mol": float, "measured_on": str, "breakdown": {...}}},
     "records": [{"task": ..., "arm": ..., "head": str,
                  "mean": float, "sem": float, "n_folds": int}, ...]}

Arm keys must exist in figures/arms.py; an unregistered key draws in default grey on purpose,
so a new arm is visibly unstyled rather than silently missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A                                            # noqa: E402
from style import (FS, STYLE, check_font, mark_empty, row_ncol,    # noqa: E402
                           save, title)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "figures" / "figC" / "results.json"
DEFAULT_HEAD = "xgboost"


def load():
    global RESULTS
    if len(sys.argv) > 1:
        RESULTS = Path(sys.argv[1])
        print(f"  reading {RESULTS}  (override; not the canonical results file)")
    if not RESULTS.exists():
        raise SystemExit(
            f"figure C: no results at {RESULTS.relative_to(ROOT) if RESULTS.is_relative_to(ROOT) else RESULTS}.\n"
            f"  Run the Figure C evaluation first; the contract is in this file's docstring.\n"
            f"  This script deliberately does NOT invent placeholder numbers.")
    d = json.loads(RESULTS.read_text())
    for k in ("tasks", "arms", "cost", "records"):
        if k not in d:
            raise SystemExit(f"figure C: results.json is missing '{k}'")
    # READABILITY CEILING. Past roughly nine arms the learned embeddings cluster and no label
    # placement rescues the panel -- measured, not guessed: fifteen arms collided on every panel.
    # The fix is fewer arms in the main figure and a full version in the SI, which is a decision
    # for the evaluation to make, so this warns rather than truncates.
    if len(d["arms"]) > 9:
        print(f"  WARNING  figure C: {len(d['arms'])} arms drawn; past ~9 the labels collide. "
              f"Consider a focused main figure and a full SI variant.")
    no_cost = [a for a in d["arms"] if a not in d["cost"]]
    if no_cost:
        raise SystemExit(f"figure C: no measured cost for {no_cost}. Cost is the x-axis; an arm "
                         f"without one cannot be placed. Measure it or drop the arm.")
    return d


def pareto(xs, ys):
    """Indices on the upper-left frontier: nothing is both cheaper and better.

    `ys` is already oriented so LARGER IS BETTER (see main()), so the frontier is the staircase
    of points for which no other point has both a smaller x and a larger y.
    """
    keep = []
    for i in range(len(xs)):
        if not any((xs[j] <= xs[i]) and (ys[j] >= ys[i]) and (j != i) and
                   ((xs[j] < xs[i]) or (ys[j] > ys[i])) for j in range(len(xs))):
            keep.append(i)
    return sorted(keep, key=lambda i: xs[i])


def main() -> None:
    check_font()
    d = load()
    tasks, arm_keys = d["tasks"], A.order(d["arms"])
    rec = {(r["task"], r["arm"]): r for r in d["records"]}

    # ONE ROW up to four tasks. A4 width is the budget, not A4 length (Leif 2026-08-28), and
    # 2x2 spent 5.4in of height on what fits in 3.0.
    if len(tasks) <= 4:
        nrow, ncol, h = 1, len(tasks), 3.05
        gsk = dict(hspace=0.0, wspace=0.42, left=0.075, right=0.99, top=0.87, bottom=0.30)
    else:
        nrow = 2
        ncol = int(np.ceil(len(tasks) / nrow))
        h = 5.4
        gsk = dict(hspace=0.38, wspace=0.26, left=0.085, right=0.985, top=0.93, bottom=0.175)
    fig = plt.figure(figsize=(STYLE["col2"], h))
    gs = fig.add_gridspec(nrow, ncol, **gsk)

    tags = "abcdefgh"
    odd_heads = set()
    label_sets = []
    for t_i, t in enumerate(tasks):
        ax = fig.add_subplot(gs[t_i // ncol, t_i % ncol])
        lo_better = bool(t.get("lower_is_better", True))

        xs, ys, es, ks = [], [], [], []
        for a in arm_keys:
            r = rec.get((t["key"], a))
            if r is None:
                continue
            xs.append(float(d["cost"][a]["us_per_mol"]))
            ys.append(float(r["mean"]))
            es.append(float(r.get("sem", 0.0)))
            ks.append(a)
            if r.get("head", DEFAULT_HEAD) != DEFAULT_HEAD:
                odd_heads.add((a, r["head"]))
        if not ks:
            raise SystemExit(f"figure C: task {t['key']!r} has no records for any drawn arm")

        # Orient for the frontier only. The AXIS keeps the field's own units and is inverted
        # below, so nothing plotted is a negated metric the reader has to undo mentally.

        for x, y, e, a in zip(xs, ys, es, ks):
            kw = A.bar_kw(a)
            face = kw.pop("color")
            kw.pop("hatch", None)                     # hatch reads as noise at marker size
            ax.errorbar(x, y, yerr=e, fmt="o", ms=STYLE["marker_size"], color=face,
                        ecolor=STYLE["ink"], elinewidth=STYLE["lw_thin"],
                        capsize=STYLE["cap_size"], zorder=3,
                        # AN INK CIRCLE ON EVERY MARKER (Leif 2026-08-29). It was
                        # `kw.get("linewidth", 0.0)`, so any arm whose style carried no explicit
                        # width drew with NO outline and the pale fills (MiniMol, ECFP+Mordred)
                        # dissolved into the panel background.
                        markeredgecolor=STYLE["ink"], markeredgewidth=0.7)
        label_sets.append(list(zip(xs, ys, ks)))
        ax.set_xscale("log")
        if lo_better:
            ax.invert_yaxis()                          # up is better, in every panel, always
        # ONE shared x-label for the row. Four copies of a long label overprinted each other
        # into "featurisation cost (µs / molecule, lfeg)aturisation cost ...".
        if nrow > 1 or t_i == 0:
            ax.set_xlabel("" if nrow == 1 else "featurisation cost (µs / molecule, log)",
                          fontsize=FS["label"])
        # The arrow carries "lower/higher is better" in two characters. Spelled out, panel b's
        # y-label was drawn on top of panel a's data.
        ax.set_ylabel(f"{t['metric']} ({'↓' if lo_better else '↑'} better)",
                      fontsize=FS["label"])
        ax.grid(axis="both")
        title(ax, t["label"])
        ax.text(-0.16, 1.06, tags[t_i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="right")

    # ---- ONE-ROW LEGEND ------------------------------------------------------------------
    # In-panel labels were tried first and rejected (Leif 2026-08-27: "labels are hard to
    # read"). At 7pt against gridlines, on a scatter where the learned embeddings cluster, they
    # are genuinely worse than a key -- the label has to sit off the point, so the eye travels
    # anyway, and it travels to something smaller and unaligned. A legend at least puts every
    # name in one place, at one size, in a fixed order.
    #
    # ONE ROW. That is the house rule for this paper, and it is also the readability ceiling
    # here: it caps the main figure at the ~7 arms that carry the argument and pushes the rest
    # to an SI variant, which is the right pressure to be under.
    #
    # Order follows arms.ARM_ORDER -- classical, then HUME, then graph models, then string
    # models -- so the key reads in the same left-to-right order as every other figure.
    if nrow == 1:
        fig.supxlabel("featurisation cost (µs / molecule, log)", fontsize=FS["label"], y=0.175)
    from matplotlib.lines import Line2D
    handles = []
    for a in arm_keys:
        kw = A.bar_kw(a)
        handles.append(Line2D([], [], marker="o", ls="", ms=STYLE["marker_size"],
                              color=kw["color"], label=A.short_label(a),
                              markeredgecolor=STYLE["ink"], markeredgewidth=0.7))
    if nrow == 1:
        fig.supxlabel("featurisation cost (µs / molecule, log)", fontsize=FS["label"], y=0.175)
    lax = fig.add_axes([0.005, 0.005, 0.99, 0.135 if nrow == 1 else 0.055])
    lax.axis("off")
    mark_empty(lax, "legend strip -- holds no data by design")
    lax.legend(handles=handles, loc="center", ncol=row_ncol(handles, rows=2), frameon=False,
               fontsize=FS["legend"], handlelength=0.8, columnspacing=0.8,
               handletextpad=0.4)
    save(fig, "fig_c")
    plt.close(fig)


if __name__ == "__main__":
    main()
