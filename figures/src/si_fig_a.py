"""SI figure A -- the four shipped HUME widths, side by side, against the classical baseline.

    .venv/bin/python figures/src/si_fig_a.py [results.json]

WHY THIS IS NOT FIGURE C WITH A DIFFERENT ARM LIST. Figure C's x-axis is measured featurization
cost, which is the right axis when the arms span two orders of magnitude of it -- ECFP at 17
us/mol against ECFP+all-desc at 6,200. The four HUME widths span 116.7 to 144.5. On a log axis
reaching 6,200 they land on top of each other, and the question this plate exists to answer is
what the widths cost in ACCURACY, not what they cost in time.

So the x-axis here is just the arm, ordered widest to narrowest. The reader looks down a column
and sees the four distances directly.

The anchor is unchanged from Figure C -- ECFP + all descriptors, which is 0 by construction --
so the two plates are on the same y-scale and can be read together.

The marker is the MEDIAN over datasets and the whisker a percentile-bootstrap 95% CI for it,
matching the median-and-Wilcoxon statistics every claim in the text is made with. Behind each
arm sit its individual datasets, so a wide interval can be read for what it is -- usually one
hard endpoint rather than an unstable representation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A                                                    # noqa: E402
from style import FS, STYLE, check_font, save, title                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "figures" / "SI_figA" / "results.json"

#: Widest to narrowest, with the anchor first. Fixed here rather than taken from the file so the
#: panels cannot silently reorder when an arm is added.
ANCHOR = "ecfp_all_desc"
ORDER = [ANCHOR, "hume", "hume_622", "hume_408", "hume_256"]
NCOL = {"hume": 1269, "hume_622": 622, "hume_408": 408, "hume_256": 256}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    if not path.exists():
        raise SystemExit(
            f"SI figure A: no results at {path}. Run bench/collect_downstream.py first; this "
            f"script does not invent placeholder numbers.")
    d = json.loads(path.read_text())
    check_font()
    recs = {(r["task"], r["arm"]): r for r in d["records"]}
    tasks = d["tasks"]
    arms = [a for a in ORDER if any((t["key"], a) in recs for t in tasks)]
    missing = [a for a in ORDER if a not in arms]
    if missing:
        print(f"  WARNING  SI figure A: no records for {missing}; drawn without them")

    # One row of panels, as Figure C has: four tasks read left to right, and the shared y-label
    # on the first panel only. row_ncol() in style.py sizes a LEGEND, not a subplot grid.
    # One row of panels, as Figure C has. WIDTH IS THE PAGE, NOT A PREFERENCE: style.save()
    # warns when the rendered figure is wider than the text block, because LaTeX then scales it
    # down and the fonts stop matching the rest of the set. 6.69in is the block; four panels get
    # a sixth of an inch of margin each.
    ncol, nrow = len(tasks), 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.69, 2.5), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(arms))
    rng = np.random.default_rng(0)
    for ax, t in zip(axes, tasks):
        ax.axhline(0.0, color=STYLE["ink"], lw=0.8, ls=(0, (2, 2)), zorder=1)
        ys, los, his = [], [], []
        for j, a in enumerate(arms):
            r = recs.get((t["key"], a))
            if r is None:
                ys.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            ys.append(r["median"]); los.append(r["ci_lo"]); his.append(r["ci_hi"])
            # EVERY DATASET, BEHIND THE SUMMARY. A panel is 4 to 13 datasets, which is few enough
            # that showing them costs nothing and hides nothing: a bar that looks wide because one
            # endpoint is hard is then visibly one endpoint, not a representation that is unstable.
            # The anchor's points are all exactly 0 by construction, so they are not drawn -- a
            # stack of dots on the zero line would read as data.
            d = np.asarray(r.get("deltas", []), dtype=float)
            if a != ANCHOR and d.size:
                jitter = rng.uniform(-0.16, 0.16, d.size)
                ax.scatter(np.full(d.size, j) + jitter, d, s=7, c=A.color(a), alpha=0.35,
                           linewidths=0, zorder=2)
        ys, los, his = np.array(ys), np.array(los), np.array(his)
        err = np.vstack([ys - los, his - ys])
        ax.errorbar(x, ys, yerr=err, fmt="none", ecolor=STYLE["ink"], elinewidth=0.9,
                    capsize=2.5, zorder=3)
        ax.scatter(x, ys, s=46, c=[A.color(a) for a in arms], edgecolor=STYLE["ink"],
                   linewidth=0.6, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([A.short_label(a) for a in arms], rotation=45, ha="right",
                           fontsize=FS["tick"])
        ax.set_xlim(-0.6, len(arms) - 0.4)
        arrow = "\u2193" if t["lower_is_better"] else "\u2191"
        title(ax, f"{t['label']}\n({t['metric']}, {arrow} better)")
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].set_ylabel("Δ error vs ECFP + all desc\n(↓ better)", fontsize=FS["label"])
    fig.tight_layout()
    save(fig, "SI_fig_a")

    build = ROOT / "figures" / "build"
    build.mkdir(parents=True, exist_ok=True)
    import csv
    with open(build / "SI_fig_a.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "arm", "label", "n_columns", "median_delta_vs_anchor",
                    "ci_lo", "ci_hi", "n_datasets", "mean_delta", "sem", "n_folds"])
        for t in tasks:
            for a in arms:
                r = recs.get((t["key"], a))
                if r is None:
                    continue
                w.writerow([t["key"], a, A.short_label(a), NCOL.get(a, ""),
                            f"{r['median']:.6f}", f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                            r.get("n_datasets", ""), f"{r['mean']:.6f}", f"{r['sem']:.6f}",
                            r.get("n_folds", "")])
    print("  wrote  figures/build/SI_fig_a.csv")


if __name__ == "__main__":
    main()
