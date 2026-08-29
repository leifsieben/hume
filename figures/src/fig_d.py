"""Figure D -- what it costs to featurise a billion molecules, measured rather than extrapolated.

    .venv/bin/python figures/src/fig_d.py [results/scale/*.json ...]

THE ARGUMENT, and why these three panels rather than the obvious two.

The obvious figure is "wall clock vs N, one line per method, one panel per hardware budget".
On log-log every method is then a straight line of slope 1 and the panels differ only in their
intercepts -- a great deal of ink to say "a GPU box is faster than a CPU box". Worse, it invites
the reader to trust an extrapolation from 10^6 to 10^9 that the figure never justifies.

So the panels are split by the QUESTION each answers, not by the hardware:

  A  IS THE EXTRAPOLATION LEGITIMATE?  us/mol against N, log x, LINEAR y. A method whose cost
     per molecule is flat across three decades may be multiplied out to 10^9; one that bends may
     not. This panel is load-bearing for the other two and is drawn first for that reason. It is
     also where the failure modes live: the non-streaming caller bends upward at 10^6 because it
     allocates its whole output matrix (12.2 GB at N=1M over 12 workers) and the box swaps.

  B  HOW LONG DOES IT TAKE?  Wall-clock hours for 10^9, grouped by hardware budget, log y. The
     neural arms are STACKED into input-preparation and forward pass, because the finding that
     matters is not their total -- it is that the RDKit half dominates. A GNN on a GPU is a CPU
     method with an accelerator attached, and a stacked bar says so without a sentence.

  C  WHAT DOES IT COST?  USD for 10^9 at public on-demand prices, log y, with a spot band. This
     is the only hardware-neutral axis in the plate: a GPU-hour and a core-hour are not
     comparable quantities, and dollars are how the field actually resolves that. The instance
     types and the date the prices were pulled go in the caption, never in a footnote.

WHAT THIS SCRIPT WILL NOT DO. It will not invent a point. Every marker is a measurement from a
named instance; the 10^9 bars in B and C are the measured 10^6 rate multiplied out, and that
multiplication is licensed by panel A or it is not drawn at all -- `--require-flat` refuses to
render B and C for any arm whose us/mol moves more than `FLAT_TOL` across the measured decades.
An arm that fails is listed in the caption as unextrapolated rather than silently dropped.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import (BUILDDIR, FS, ROOT, STYLE, check_font, install,  # noqa: E402
                   mark_empty, save)
from matplotlib.patches import Patch  # noqa: E402
import arms as A  # noqa: E402

TARGET = 1e9
FLAT_TOL = 0.25          # us/mol may move this much across decades and still be extrapolated
SECONDS_PER_HOUR = 3600.0

# Display names and colours. `arms.py` owns the paper's palette; anything it does not know about
# gets a neutral grey rather than a new colour invented here, so the plate cannot drift from the
# other figures.
# THE KEYS HERE ARE THE ARM NAMES bench_aws.py WRITES, and an arm missing from this dict is
# silently invisible in every panel -- which is exactly what happened on the first render with
# real data: `ecfp` produced three good points and drew nothing. Keep this in step with
# bench_aws.py's ARMS, and note the ORDER is the plotting and bar order.
# THE DESCRIPTOR COUNT IS READ FROM THE PACKAGE, NOT TYPED. It was hard-coded as 864 -- the
# census figure from an earlier build -- and the block has since grown to 1,266, so the legend
# had been understating HUME's width by a third on a plate whose whole argument is what that
# width costs. Falls back to the last measured value only if hume is not importable, and says so.
try:
    import hume as _hume
    _HUME_NDESC = len(_hume.ALL_COLUMNS)
except Exception:
    _HUME_NDESC = "1266?"

LABEL = {
    "ecfp_r2":   "ECFP4 r2-2048 (the floor)",
    "hume":      f"HUME ({_HUME_NDESC} desc + ECFP)",
    "chemprop":  "chemprop D-MPNN (300x3)",
    "chemberta": "ChemBERTa-2 (3M)",
    "chemeleon": "CheMeleon D-MPNN (2048x6)",
    "rdkit_desc":   "ECFP4 + RDKit-180",
    "mordred_desc": "ECFP4 + Mordred-685",
    "mordred":   "ECFP4 + all descriptors",
}
# The palette key in arms.py, so this plate cannot drift from A, B and C. `mordred` here is
# ECFP + RDKit-180 + mordred-685, which is the `ecfp_all_desc` arm elsewhere in the paper -- it
# used to point at `ecfp_mordred_desc`, a DIFFERENT arm, and the same measurement was therefore
# drawn in two colours across two figures.
ARMKEY = {"ecfp_r2": "ecfp", "hume": "hume", "chemprop": "chemprop",
          "chemberta": "chemberta_mlm", "chemeleon": "chemeleon",
          "mordred": "ecfp_all_desc",
          "rdkit_desc": "ecfp_rdkit_desc", "mordred_desc": "ecfp_mordred_desc"}

# Bar order, matching arms.py's ARM_ORDER: cheapest classical first, then HUME, then graph, then
# string. Every figure in the set puts the same arms in the same order; two orders read as two
# different comparisons.
ORDER = ["ecfp_r2", "rdkit_desc", "mordred_desc", "mordred", "hume",
         "chemeleon", "chemprop", "chemberta"]

#: MEASURED BUT NOT DRAWN HERE. These exist for Figure C's cost axis, which plots more arms than
#: this plate does. check_known() refuses to render an arm it does not know -- that guard caught a
#: silent omission once already -- so arms that are deliberately absent have to be declared rather
#: than left to fall through it.
#:
#: `ecfp` is the r=3 variant. This plate now draws `ecfp_r2` instead, because ECFP4 (r=2) is the
#: baseline Figures A, B and C actually run; r=3 is the radius HUME carries INTERNALLY and the two
#: were being used interchangeably. Both stay measured, and the legend says which is drawn.
COST_ONLY = {"ecfp", "minimol"}
HATCH = "///"


SHORT = {"ecfp_r2": "ECFP4", "hume": "HUME", "chemprop": "chemprop",
         "chemberta": "ChemBERTa", "chemeleon": "CheMeleon",
         "rdkit_desc": "+RDKit", "mordred_desc": "+Mordred", "mordred": "+all desc"}
BUDGET_LABEL = {"cpu": "16 vCPU", "gpu": "1 GPU + 4 vCPU"}


def _group_header(ax, groups):
    """Name each hardware block once, above its bars, instead of on every tick."""
    seen, start = None, 0
    for i, g in enumerate(groups + [(None, None, None)]):
        bud = g[0]
        if bud != seen:
            if seen is not None:
                mid = (start + i - 1) / 2
                inst = groups[start][2]["instance"]
                ax.text(mid, 1.03, f"{inst}\n{BUDGET_LABEL.get(seen, seen)}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=FS["annot"], color=STYLE["mute"])
                if i < len(groups):
                    ax.axvline(i - 0.5, color=STYLE["faint"], lw=0.8, zorder=0)
            seen, start = bud, i


def colour(arm: str) -> str:
    try:
        return A.color(ARMKEY.get(arm, arm))
    except Exception:
        return "#9AA0A6"


# ------------------------------------------------------------------------------------------
# Data contract. The AWS harness writes one JSON per instance; this is the whole of what the
# figure needs, and it is written down here so the harness and the figure cannot drift.
#
#   {"meta": {"instance": "c7g.4xlarge", "region": "us-east-1", "vcpu": 16, "gpu": null,
#             "usd_per_hour_ondemand": 0.58, "usd_per_hour_spot": 0.2010,
#             "priced_on": "2026-08-28", "budget": "cpu"},
#    "points": [{"arm": "hume", "n": 10000, "wall_s": 1.10,
#                "prep_s": null, "fwd_s": null}, ...]}
#
# `prep_s` / `fwd_s` are optional and only the neural arms carry them; when present they must
# sum to `wall_s` so panel B's stack cannot disagree with panel B's total.
# ------------------------------------------------------------------------------------------
def load(paths):
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        m = d["meta"]
        for pt in d["points"]:
            bad = (pt.get("prep_s") is not None and pt.get("fwd_s") is not None
                   and abs(pt["prep_s"] + pt["fwd_s"] - pt["wall_s"]) > 0.02 * pt["wall_s"])
            if bad:
                raise ValueError(
                    f"{p}: arm {pt['arm']} at N={pt['n']} has prep_s + fwd_s = "
                    f"{pt['prep_s'] + pt['fwd_s']:.3f}s against wall_s {pt['wall_s']:.3f}s. "
                    "Panel B stacks those two into the total, so they must agree; fix the "
                    "harness rather than the tolerance.")
            runs.append({**pt, **m, "us_per_mol": pt["wall_s"] / pt["n"] * 1e6})
    return runs


def check_known(runs):
    """Refuse to render if the data contains an arm this figure does not know how to draw."""
    unknown = sorted({r["arm"] for r in runs} - set(LABEL) - COST_ONLY)
    if unknown:
        raise SystemExit(
            f"fig_d: the results contain arm(s) {unknown} that are not in LABEL, so they would "
            f"be silently omitted from every panel. Add them to LABEL/ARMKEY (and pick a colour "
            f"in arms.py) rather than letting the figure quietly under-report.")


def by(runs, **kw):
    return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]


def is_flat(pts) -> bool:
    """Does us/mol stay put across the measured decades?

    EVERY BAR ON THIS PLATE IS AN EXTRAPOLATION from 1e6 to 1e9, and it is only licensed if cost
    per molecule does not move with N. The check that used to be panel A is now a GATE rather
    than a panel: an arm that fails it is dropped and named on the console, because a figure that
    silently draws an unlicensed extrapolation is worse than one that draws fewer arms.
    """
    if len(pts) < 2:
        return False
    v = [p["us_per_mol"] for p in sorted(pts, key=lambda p: p["n"])]
    return (max(v) - min(v)) / min(v) <= FLAT_TOL


def _hours_for_target(pt) -> float:
    return pt["us_per_mol"] * 1e-6 * TARGET / SECONDS_PER_HOUR


def best(runs, arm, budget):
    """The arm's own best configuration on that hardware, at the largest N measured."""
    pts = sorted(by(runs, arm=arm, budget=budget), key=lambda p: p["n"])
    if not pts or not is_flat(pts):
        return None
    return pts[-1]


def cells(runs, budget):
    """-> [(arm, point, copied_from_cpu)] in the paper's arm order.

    ARMS WITH NO GPU IMPLEMENTATION CARRY THEIR CPU NUMBER INTO THE GPU PANELS (Leif: "for the
    CPU only ones just copy the results"). ECFP, the descriptor block and HUME have no forward
    pass to put on a device -- on a GPU box they run on that box's CPU, which is exactly what
    the copied number represents. They are hatched and the legend says so, because a reader who
    took them for measured GPU throughput would conclude the GPU does nothing for them, when in
    fact there is nothing there to accelerate.
    """
    out = []
    for arm in ORDER:
        pt = best(runs, arm, budget)
        copied = False
        if pt is None and budget == "gpu":
            pt = best(runs, arm, "cpu")
            copied = pt is not None
        if pt is not None:
            out.append((arm, pt, copied))
    return out


def _bars(ax, rows, values, fmt):
    x = np.arange(len(rows))
    v = np.array(values, float)
    ax.bar(x, v, width=0.72, color=[colour(a) for a, _p, _c in rows],
           edgecolor=STYLE["ink"], linewidth=0.45, zorder=3,
           hatch=[HATCH if c else None for _a, _p, c in rows])
    ax.set_yscale("log")
    # VERTICAL VALUE LABELS. Horizontally they collided with each other once the descriptor
    # blocks were split out -- "1,301h" ran into "1,722h" on adjacent bars -- and the roster is
    # still growing, so a spacing that fits today collides on the next arm. Rotated, the label
    # occupies one bar's width however long the number is. The extra headroom is for the taller
    # rotated text, not for the bars.
    ax.set_ylim(top=ax.get_ylim()[1] * 5.0)
    for xi, val in zip(x, v):
        ax.text(xi, val * 1.25, fmt(val), ha="center", va="bottom", rotation=90,
                fontsize=FS["annot"], zorder=5)
    ax.set_xticks(x)
    # 45 degrees, not 90. Six short names per frame do not collide at 45, and vertical labels
    # were reserving a full inch of an A4-width plate that the bars should be using.
    ax.set_xticklabels([SHORT[a] for a, _p, _c in rows], fontsize=FS["tick"],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _hours(v):
    return f"{v:,.0f}h" if v >= 10 else f"{v:.1f}h"


def _usd(v):
    return f"${v:,.0f}" if v >= 10 else f"${v:.1f}"


def main(paths):
    check_font()
    install()
    runs = load(paths) if paths else []
    check_known(runs)
    if not runs:
        raise SystemExit("fig_d: no measurements in results/scale/. Run bench_aws.py on the "
                         "instances and collect with collect_scale.py.")
    for bud in ("cpu", "gpu"):
        dropped = [a for a in ORDER if by(runs, arm=a, budget=bud) and best(runs, a, bud) is None]
        if dropped:
            print(f"  not flat within {FLAT_TOL:.0%} on {bud}, so not extrapolated: {dropped}")

    # FOUR FRAMES, ONE ROW, sized to the A4 text block: time then cost, CPU then GPU (Leif).
    # The plate is WIDE AND SHORT on purpose -- it is the last figure in the set and the page
    # budget is horizontal, not vertical.
    fig, axes = plt.subplots(1, 4, figsize=(STYLE["col2"], 2.75))
    # THE CPU AND GPU FRAMES OF EACH PAIR SHARE ONE Y-AXIS. The whole question the pair asks is
    # "does the hardware help?", and two independently scaled log axes answer it wrong -- 803h
    # and 291h drew at nearly the same height. Sharing also frees the width that a second set of
    # tick labels and a second axis title were taking, which is what pushed the plate 6% past the
    # A4 text block.
    for a1, a2 in ((axes[0], axes[1]), (axes[2], axes[3])):
        a2.sharey(a1)
    spec = [("cpu", "hours", "a  Time, CPU"), ("gpu", "hours", "b  Time, GPU"),
            ("cpu", "usd", "c  Cost, CPU"), ("gpu", "usd", "d  Cost, GPU")]
    for i, (ax, (bud, kind, ttl)) in enumerate(zip(axes, spec)):
        rows = cells(runs, bud)
        if not rows:
            mark_empty(ax, f"no {bud} measurements")
            continue
        hrs = [_hours_for_target(p) for _a, p, _c in rows]
        vals = (hrs if kind == "hours" else
                [h * p["usd_per_hour_ondemand"] for h, (_a, p, _c) in zip(hrs, rows)])
        _bars(ax, rows, vals, _hours if kind == "hours" else _usd)
        if i % 2 == 0:
            ax.set_ylabel(("hours" if kind == "hours" else "USD") + " per 1e9 molecules",
                          fontsize=FS["label"])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(ttl, fontsize=FS["title"], fontweight="bold", loc="left", pad=4)
    for a1, a2 in ((axes[0], axes[1]), (axes[2], axes[3])):
        top = max(a1.get_ylim()[1], a2.get_ylim()[1])
        a1.set_ylim(top=top)

    handles = [Patch(facecolor=colour(a), edgecolor=STYLE["ink"], lw=0.6, label=LABEL[a])
               for a in ORDER if any(by(runs, arm=a))]
    handles.append(Patch(facecolor="white", edgecolor=STYLE["ink"], lw=0.6, hatch=HATCH,
                         label="no GPU path: CPU measurement"))
    # ncol=3: at four columns the legend row was WIDER THAN THE PANELS and savefig("tight")
    # grew the canvas to it, so the plate rendered 6% past the A4 text block and LaTeX would
    # scale every font on it down to fit.
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=FS["legend"], bbox_to_anchor=(0.5, -0.16), handletextpad=0.6,
               columnspacing=1.4, labelspacing=0.35)
    fig.tight_layout()
    save(fig, "fig_d")
    BUILDDIR.mkdir(parents=True, exist_ok=True)
    (BUILDDIR / "fig_d_points.json").write_text(json.dumps(runs, indent=1))
    with open(BUILDDIR / "fig_d.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "budget", "instance", "n", "us_per_mol", "hours_per_1e9",
                    "usd_per_1e9_ondemand", "usd_per_hour", "copied_from_cpu", "priced_on"])
        for bud in ("cpu", "gpu"):
            for a, p, c in cells(runs, bud):
                h = _hours_for_target(p)
                w.writerow([a, bud, p["instance"], p["n"], f"{p['us_per_mol']:.2f}", f"{h:.2f}",
                            f"{h * p['usd_per_hour_ondemand']:.2f}",
                            p["usd_per_hour_ondemand"], int(c), p["priced_on"]])
    print(f"  wrote  figures/build/fig_d.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(str(p) for p in (ROOT / "results" / "scale").glob("*.json"))
    main(args)
