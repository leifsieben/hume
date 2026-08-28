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

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import (BUILDDIR, FS, STYLE, LEGEND_BOX, check_font, install,  # noqa: E402
                   mark_empty, save, title)
import arms as A  # noqa: E402

TARGET = 1e9
FLAT_TOL = 0.25          # us/mol may move this much across decades and still be extrapolated
SECONDS_PER_HOUR = 3600.0

# Display names and colours. `arms.py` owns the paper's palette; anything it does not know about
# gets a neutral grey rather than a new colour invented here, so the plate cannot drift from the
# other figures.
LABEL = {"hume": "HUME (864 desc + ECFP)", "hume_naive": "HUME, non-streaming caller",
         "gin": "GIN 5x300", "chemeleon": "CheMeleon (D-MPNN)",
         "chemberta": "ChemBERTa-2 (3M)", "molformer": "MolFormer (44M)",
         "mordred": "RDKit + Mordred"}
ARMKEY = {"hume": "hume_core", "gin": "gnn", "chemeleon": "chemeleon",
          "chemberta": "chemberta_mlm", "molformer": "molformer",
          "mordred": "ecfp_mordred_desc"}


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


def by(runs, **kw):
    return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]


def is_flat(pts) -> bool:
    """Does us/mol stay put across the measured decades? Panels B and C depend on this."""
    if len(pts) < 2:
        return False
    v = [p["us_per_mol"] for p in sorted(pts, key=lambda p: p["n"])]
    return (max(v) - min(v)) / min(v) <= FLAT_TOL


# ------------------------------------------------------------------------------------------
def panel_a(ax, runs):
    """us/mol vs N. Flat is the result; a bend is a finding and is drawn, not hidden."""
    budgets = sorted({r["budget"] for r in runs})
    any_data = False
    for arm in LABEL:
        for bud in budgets:
            pts = sorted(by(runs, arm=arm, budget=bud), key=lambda p: p["n"])
            if not pts:
                continue
            any_data = True
            ls = "-" if bud == "cpu" else "--"
            ax.plot([p["n"] for p in pts], [p["us_per_mol"] for p in pts],
                    ls, marker="o", ms=3.5, lw=1.4, color=colour(arm),
                    label=LABEL[arm] if bud == budgets[0] else None)
    if not any_data:
        return mark_empty(ax, "no scaling points yet -- run bench_scale_e2e.py on the instances")
    ax.set_xscale("log")
    ax.set_xlabel("molecules featurised (N)", fontsize=FS["ax"])
    ax.set_ylabel("cost per molecule (us)", fontsize=FS["ax"])
    title(ax, "A  Is the extrapolation legitimate?")
    ax.text(0.02, 0.96, f"flat within {FLAT_TOL:.0%} -> may be multiplied out to 1e9",
            transform=ax.transAxes, va="top", fontsize=FS["annot"], color=STYLE["ink"])


def _hours_for_target(pt) -> float:
    return pt["us_per_mol"] * 1e-6 * TARGET / SECONDS_PER_HOUR


def panel_b(ax, runs, skipped):
    """Wall-clock hours for 1e9, grouped by budget, neural arms stacked prep vs forward."""
    budgets = sorted({r["budget"] for r in runs})
    groups, labels, drawn = [], [], False
    for bud in budgets:
        for arm in LABEL:
            pts = sorted(by(runs, arm=arm, budget=bud), key=lambda p: p["n"])
            if not pts:
                continue
            if not is_flat(pts):
                skipped.append(f"{LABEL[arm]} on {pts[-1]['instance']}")
                continue
            groups.append((bud, arm, pts[-1]))
            labels.append(f"{LABEL[arm]}\n{pts[-1]['instance']}")
    if not groups:
        return mark_empty(ax, "nothing passed the flatness gate in panel A")
    for i, (_bud, arm, pt) in enumerate(groups):
        h = _hours_for_target(pt)
        c = colour(arm)
        if pt.get("prep_s") and pt.get("fwd_s"):
            frac = pt["prep_s"] / pt["wall_s"]
            ax.bar(i, h * frac, color=c, width=0.72, label=None)
            ax.bar(i, h * (1 - frac), bottom=h * frac, color=c, width=0.72,
                   alpha=0.42, hatch="//", edgecolor="white", linewidth=0.0)
        else:
            ax.bar(i, h, color=c, width=0.72)
        ax.text(i, h * 1.08, f"{h:,.0f}h", ha="center", fontsize=FS["annot"])
        drawn = True
    if drawn:
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=FS["annot"], rotation=30, ha="right")
        ax.set_ylabel("wall clock for 1e9 molecules (hours)", fontsize=FS["ax"])
        title(ax, "B  How long does it take?")
        ax.text(0.02, 0.96, "solid = input preparation (RDKit)   hatched = forward pass",
                transform=ax.transAxes, va="top", fontsize=FS["annot"])


def panel_c(ax, runs):
    """USD for 1e9 at on-demand price, with the spot price as a lower band."""
    rows = []
    for bud in sorted({r["budget"] for r in runs}):
        for arm in LABEL:
            pts = sorted(by(runs, arm=arm, budget=bud), key=lambda p: p["n"])
            if pts and is_flat(pts):
                rows.append((arm, pts[-1]))
    if not rows:
        return mark_empty(ax, "no extrapolable arms to price")
    for i, (arm, pt) in enumerate(rows):
        h = _hours_for_target(pt)
        od, sp = h * pt["usd_per_hour_ondemand"], h * (pt.get("usd_per_hour_spot") or 0.0)
        ax.bar(i, od, color=colour(arm), width=0.72)
        if sp:
            ax.plot([i - 0.36, i + 0.36], [sp, sp], color=STYLE["ink"], lw=1.1)
            ax.plot([i, i], [sp, od], color=STYLE["ink"], lw=0.7, alpha=0.5)
        ax.text(i, od * 1.08, f"${od:,.0f}", ha="center", fontsize=FS["annot"])
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{LABEL[a]}\n{p['instance']}" for a, p in rows],
                       fontsize=FS["annot"], rotation=30, ha="right")
    ax.set_ylabel("USD per 1e9 molecules", fontsize=FS["ax"])
    title(ax, "C  What does it cost?")
    ax.text(0.02, 0.96, "bar = on-demand   tick = spot", transform=ax.transAxes,
            va="top", fontsize=FS["annot"])


def main(paths):
    install()
    check_font()
    runs = load(paths) if paths else []
    # STYLE["col2"] is the page text block; every other figure in the set uses it, and
    # save() warns loudly if the rendered PDF drifts more than 5% from it.
    fig, axes = plt.subplots(1, 3, figsize=(STYLE["col2"], 3.9))
    skipped: list[str] = []
    panel_a(axes[0], runs)
    panel_b(axes[1], runs, skipped)
    panel_c(axes[2], runs)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=A.__dict__.get("_", None) or len(l),
                   fontsize=FS["annot"], bbox_to_anchor=(0.5, -0.04), **LEGEND_BOX)
    if skipped:
        fig.text(0.5, -0.10, "not extrapolated (cost per molecule not flat in N): "
                 + "; ".join(skipped), ha="center", fontsize=FS["annot"])
    if runs:
        m = runs[0]
        fig.text(0.5, -0.145,
                 f"prices: AWS EC2 on-demand, {m['region']}, pulled {m['priced_on']}",
                 ha="center", fontsize=FS["annot"])
    fig.tight_layout()
    save(fig, "fig_d")
    BUILDDIR.mkdir(parents=True, exist_ok=True)
    if runs:
        (BUILDDIR / "fig_d_points.json").write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(str(p) for p in Path("results/scale").glob("*.json"))
    main(args)
