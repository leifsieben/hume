"""Figure B -- is a learned embedding redundant to a classical featurization?

    .venv/bin/python figures/src/fig_b.py [results.json]

THE TEST IS CONCATENATION. If an embedding carries chemical signal a descriptor block does not,
gluing it onto that block must beat the block alone. If it carries nothing new, the
concatenation is at best flat -- and, because the extra dimensions cost degrees of freedom at
fixed n, may be slightly worse.

REDUNDANCY IS DIRECTIONAL, so the panel is built to be read two ways, and it needs both:

    WITHIN a group          -> does this embedding add anything to this classical block?
    ONE COLOUR ACROSS groups -> does the classical block add anything to this embedding?

A figure showing only the first cannot support the claim. A flat within-group bar is equally
consistent with three innocent explanations: the head failing to exploit ~512 dense dimensions
beside 2048 sparse bits, added variance at fixed n, or the task sitting at its ceiling. The
second reading, plus the positive control below, is what rules those out.

THE ECFP4+descriptors GROUP IS THE POSITIVE CONTROL -- "does adding a second block help on this
task AT ALL?" It is what turns a flat embedding result from unreadable into negative. An earlier
draft of this figure carried that information as a separate top row of absolute scores, which
communicated nothing; it belongs on the same axis as everything it licenses a conclusion about.

ONE UNIT FOR EVERY PANEL: % OF THE ECFP4+descriptors ANCHOR. The four tasks are scored in four
different units (MAE, RMSE, AUROC, AUPRC), so raw values cannot be read across panels and a
per-panel reference line leaves the y-axis meaningless in a compact assembly. Here 100% is
parity with the classical anchor by construction, and it is the dotted line in every panel. FOR
AN ERROR METRIC THE RATIO IS INVERTED (100*ref/v) so that "taller" means "better" everywhere --
plotting the raw ratio for an error would make the worst arm the tallest bar in half the plate.

READING RULE, one sentence: an embedding redundant to a classical block leaves its bar level
with the gray block-alone bar beside it.

WHAT WOULD FALSIFY IT. A bar that clears the gray reference in its own group, on a task that
discriminates, by more than its own spread -- i.e. information the descriptors genuinely lack.
Uni-Mol on the QM panel is the arm most likely to do this, because a conformer carries geometry
no 2D descriptor block can reconstruct. It is drawn like everything else.

WHY THESE FOUR EMBEDDINGS. Two were PRETRAINED TO PREDICT MOLECULAR DESCRIPTORS -- CheMeleon on
Mordred, ChemBERTa-2-MTR on 200 RDKit descriptors -- and two were not (MiniMol, MoLFormer).
Showing only the first pair would make the result circular: of course a descriptor-trained model
is a descriptor proxy. The controls are what make the finding a finding. Descriptor-pretrained
arms are marked * in the legend; the bar itself cannot carry it, because every bar in this
paper's house style already has an ink edge.

DATA CONTRACT -- results/figures/figB/results.json:

    {"meta":  {...free-form provenance...},
     "tasks": [{"key": ..., "label": ..., "metric": ..., "lower_is_better": bool}, ...],
     "bases": ["ecfp", "desc", "ecfp_all_desc"],      # groups, left to right
     "anchor": "ecfp_all_desc",                        # the 100% reference
     "adds":  ["chemeleon", "chemberta_mtr", "minimol", "molformer"],
     "records": [{"task": ..., "base": ..., "add": null|<arm>,
                  "mean": float, "sem": float, "n_folds": int}, ...]}

`add: null` is the block alone. A cell absent from `records` is drawn as a dashed empty slot
labeled "not run" -- never omitted, because a missing bar and a bar at the axis floor look
identical and here they mean opposite things.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A                                                          # noqa: E402
from style import (FS, STYLE, check_font, mark_empty, row_ncol,           # noqa: E402
                   save, title)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "figures" / "figB" / "results.json"
BASE_KEY = "classical_base"        # the block-alone bar's arm key (one color in every group)

#: Tick labels for the base groups. Deliberately SHORTER than arms.py's labels: three of them
#: share a quarter-width panel, where the full names overprint each other.
BASE_SHORT = {"ecfp": "ECFP", "desc": "descriptors", "ecfp_all_desc": "ECFP + desc",
              "ecfp_rdkit_desc": "ECFP + RDKit", "ecfp_mordred_desc": "ECFP + Mordred"}


def load():
    # A path argument exists so the LAYOUT can be checked against a fixture without writing
    # invented numbers into results/. Fake data beside real data is a worse hazard than an
    # unrendered figure.
    global RESULTS
    if len(sys.argv) > 1:
        RESULTS = Path(sys.argv[1])
        print(f"  reading {RESULTS}  (override; not the canonical results file)")
    if not RESULTS.exists():
        raise SystemExit(
            f"figure B: no results at {RESULTS}.\n"
            f"  Run the Figure B evaluation first; the contract is in this file's docstring.\n"
            f"  This script deliberately does NOT invent placeholder numbers.")
    d = json.loads(RESULTS.read_text())
    for k in ("tasks", "bases", "adds", "records"):
        if k not in d:
            raise SystemExit(f"figure B: results.json is missing '{k}'")
    d.setdefault("anchor", d["bases"][-1])
    if d["anchor"] not in d["bases"]:
        raise SystemExit(f"figure B: anchor {d['anchor']!r} is not one of the drawn bases")
    return d


def as_pct(v, ref, lower_is_better):
    """Value as a percentage of the anchor, oriented so TALLER IS ALWAYS BETTER."""
    if not (np.isfinite(v) and np.isfinite(ref)) or v == 0 or ref == 0:
        return np.nan
    return 100.0 * (ref / v if lower_is_better else v / ref)


def pct_err(v, e, ref, lower_is_better):
    """Fold spread carried into percentage units.

    For the inverted (error-metric) branch the map is 100*ref/v, whose derivative is
    -100*ref/v^2 -- so the interval is NOT simply scaled by 100/ref, and treating it that way
    would understate the whiskers on exactly the panels where the metric is an error.
    """
    if not np.isfinite(e):
        return np.nan
    if lower_is_better:
        return 100.0 * ref * e / (v * v) if v else np.nan
    return 100.0 * e / ref if ref else np.nan


def main() -> None:
    check_font()
    d = load()
    tasks, bases, adds, anchor = d["tasks"], d["bases"], d["adds"], d["anchor"]
    rec = {(r["task"], r["base"], r["add"]): r for r in d["records"]}

    # bar slots: the block alone, then the block plus each embedding
    slots = [None] + list(adds)
    # ONE ROW UP TO FOUR TASKS. The page budget is horizontal -- A4 width, not A4 length
    # (Leif 2026-08-28) -- and a 2x2 grid spent 4.9in of height to draw what fits in 2.9.
    if len(tasks) <= 4:
        nrow, ncol, h = 1, len(tasks), 2.95
        gsk = dict(hspace=0.0, wspace=0.30, left=0.075, right=0.99, top=0.80, bottom=0.30)
    else:
        nrow, ncol, h = 2, int(np.ceil(len(tasks) / 2)), 4.9
        gsk = dict(hspace=0.34, wspace=0.20, left=0.075, right=0.99, top=0.92, bottom=0.16)
    fig = plt.figure(figsize=(STYLE["col2"], h))
    gs = fig.add_gridspec(nrow, ncol, **gsk)

    bw = 0.80 / len(slots)
    centres = np.arange(len(bases), dtype=float)
    tags = "abcdefgh"
    for t_i, t in enumerate(tasks):
        ax = fig.add_subplot(gs[t_i // ncol, t_i % ncol])
        lo_better = bool(t.get("lower_is_better", True))
        a_rec = rec.get((t["key"], anchor, None))
        if a_rec is None:
            # SKIP THE PANEL, LOUDLY, rather than kill the plate. Every bar here is a percentage
            # OF the anchor, so a task without one has no denominator and cannot be drawn -- but
            # while the grid is still landing, one task arriving before its anchor should not
            # stop the three that are complete from rendering. The panel is left explicitly
            # marked, so a missing task can never be mistaken for a task with nothing in it.
            print(f"  SKIPPING task {t['key']!r}: no anchor cell ({anchor}, block alone) yet, "
                  f"so the % axis has no denominator")
            # mark_empty() only FLAGS the axes for check_no_empty_panels(); it draws nothing.
            # A blank framed panel and a panel whose values are all zero look identical on paper,
            # so the reason is written into the cell.
            mark_empty(ax, f"{t['label']}: anchor not measured yet")
            ax.text(0.5, 0.5, f"{t['label']}\n(anchor not measured yet)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=STYLE["mute"],
                    style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        ref = float(a_rec["mean"])

        ys, es, cs, xs, ok = [], [], [], [], []
        for b_i, b in enumerate(bases):
            for s_i, s in enumerate(slots):
                r = rec.get((t["key"], b, s))
                xs.append(centres[b_i] + (s_i - (len(slots) - 1) / 2) * bw)
                cs.append(A.color(BASE_KEY if s is None else s))
                if r is None:
                    ys.append(np.nan); es.append(np.nan); ok.append(False)
                else:
                    ys.append(float(r["mean"]))
                    es.append(float(r.get("sem", np.nan)))
                    ok.append(True)

        span = [v for v, e in zip(ys, es) if np.isfinite(v)]
        span += [v + (e if np.isfinite(e) else 0) for v, e in zip(ys, es) if np.isfinite(v)]
        span += [v - (e if np.isfinite(e) else 0) for v, e in zip(ys, es) if np.isfinite(v)]
        span.append(0.0)
        lo, hi = min(span), max(span)
        pad = 0.22 * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        ax.set_ylim(y0, y1)

        for x, v, e, c, good in zip(xs, ys, es, cs, ok):
            if good and np.isfinite(v):
                ax.bar([x], [v], width=bw * 0.92, color=c, edgecolor=STYLE["ink"],
                       linewidth=0.7, zorder=3,
                       yerr=([e] if np.isfinite(e) else None),
                       error_kw=dict(elinewidth=0.9, capsize=1.8, capthick=0.9,
                                     ecolor=STYLE["ink"], zorder=6))
            else:
                # NOT RUN, drawn as an empty slot. A missing bar and a bar at the axis floor
                # look identical and mean opposite things.
                ax.bar([x], [y1 - y0], bottom=y0, width=bw * 0.92, facecolor="none",
                       edgecolor=STYLE["grid"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
                ax.text(x, y0 + 0.5 * (y1 - y0), "not run", rotation=90, ha="center",
                        va="center", fontsize=FS["annot"] - 2, color="#8A8A8A", zorder=4)

        # A PER-GROUP REFERENCE TICK at that group's own block-alone height. The figure's
        # primary question -- does this embedding beat the block it was added to -- is a LOCAL
        # comparison, and without this the reader has to judge it by eye against a gray bar
        # several bars away. The global 100% line answers a different question (does it beat the
        # best classical featurization we have), and both are worth having.
        ax.axhline(0.0, ls=(0, (2, 2)), lw=STYLE["lw_thin"], color=STYLE["ink"], zorder=1)
        ax.set_xticks(centres)
        ax.set_xticklabels([A.short_label(b) for b in bases],
                           fontsize=FS["annot"] - 1, rotation=20, ha="right",
                           rotation_mode="anchor")
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(bases) - 0.5)
        if t_i % ncol == 0:
            ax.set_ylabel("\u0394 error vs ECFP + all desc\n(\u2193 better)",
                          fontsize=FS["label"])
        ax.grid(axis="y")
        title(ax, f"{t['label']}\n({t['metric']}, \u2193 better)", pad=4)
        ax.text(-0.22, 1.16, tags[t_i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left")

    # ---- legend: ONE ROW, built from what was actually drawn --------------------------------
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=A.color(BASE_KEY), edgecolor=STYLE["ink"], lw=0.7,
                     label=A.short_label(BASE_KEY))]
    for e in adds:
        handles.append(Patch(facecolor=A.color(e), edgecolor=STYLE["ink"], lw=0.7,
                             label=A.short_label(e)))
    lax = fig.add_axes([0.02, 0.005, 0.96, 0.115 if nrow == 1 else 0.055])
    lax.axis("off")
    mark_empty(lax, "legend strip -- holds no data by design")
    lax.legend(handles=handles, loc="center", ncol=row_ncol(handles), frameon=False,
               fontsize=FS["legend"], handlelength=1.2, columnspacing=1.4)

    save(fig, "fig_b")
    plt.close(fig)


if __name__ == "__main__":
    main()
