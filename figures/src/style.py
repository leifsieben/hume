"""Shared matplotlib style + save helper for the HUME paper figures.

Every figure script starts with `from style import ...` (figures/src is put on sys.path by
the script itself) so the typography, sizes and
output paths are identical across the paper. Figures are written to `figures/` as both PNG
(screen/review) and PDF (vector, for LaTeX).

DERIVED FROM THE CLIMB PAPER'S figures/style.py, deliberately and near-verbatim (Leif
2026-08-26: "I want visual continuity with them for all figures going forward"). The two papers
share a reader and several arms -- ECFP4 and ChemBERTa-2 appear in both -- so a figure that looks
like it came from a different lab makes the same model read as two different objects. Where this
file departs from CLIMB's it is noted at the site; nothing is changed for taste.

The one structural difference: CLIMB renders scripts from `figures/` into `figures_v2/`. HUME
renders into `figures/` itself (Leif: "exports these figures into a folder figures/"), so the
scripts and their artefacts sit together. `save()` therefore writes beside the script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
# PDFs are the deliverable and sit alone at figures/; PNG (screen review) and CSV (per-figure
# data dumps) go to figures/build/ so the folder a human opens contains only finished figures.
OUTDIR = ROOT / "figures"
BUILDDIR = ROOT / "figures" / "build"

# A FLAT MIRROR OF EVERY FINISHED PDF, one directory up from the repo (Leif 2026-08-27: "one
# folder /figures_out that is on the same level as the repo and just has the exported figures as
# pdfs -- it makes it easier for me to look at them").
#
# Deliberately OUTSIDE the repository, so it is not version-controlled and cannot be confused
# with the tracked artefacts beside the scripts. It is a convenience view, never a source: it is
# overwritten on every render, nothing reads from it, and deleting it loses nothing. PDFs only --
# the PNGs are for screen review and would just double the file count in a folder whose whole
# purpose is being quick to scan.
EXPORT_DIR = ROOT.parent / "figures_out"

# ---------------------------------------------------------------------------------------------
# ONE font, ONE size scale. These figures get combined into multi-panel layouts later, so every
# script must use these exact point sizes -- never a local tweak like `fs_annot - 0.5`, which
# would make one panel's text a different size from its neighbour's.
# ---------------------------------------------------------------------------------------------
FONT = "Arial"                                  # pinned, no silent fallback to a different face
FS = dict(
    title=9,        # panel title / suptitle
    label=8,        # axis labels
    tick=8,         # tick labels, model names
    annot=7,        # in-plot numbers and keys
    legend=7,       # legend entries
    caption=6.5,    # figure caption (screen review only; the paper caption is LaTeX)
    panel_tag=10,   # the "a", "b", "c" panel letters
)

# WIDTHS. The paper is set on A4 (210 mm) with 20 mm margins, so the text block is 170 mm =
# 6.69 in. `col2` IS that text block: every full-width figure uses it, so figures arrive at 1:1
# scale in LaTeX (\includegraphics[width=\textwidth]) with no downscaling, and font sizes on the
# page are exactly the point sizes set in FS above. Do not hard-code a width in a figure script.
A4_TEXT = 6.69                                  # 170 mm text block on A4
STYLE = dict(
    col1=3.25, col15=4.75, col2=A4_TEXT,       # single, 1.5, and full text-block widths (inches)
    lw=1.2, lw_thin=0.7, marker_size=5.0, cap_size=2.0,
    dpi_screen=120, dpi_save=300,
    grid="#A6A6A6", ink="#000000", mute="#000000", faint="#E6E6E6",
    **{f"fs_{k}": v for k, v in FS.items()},   # STYLE["fs_title"] etc. stay available
)

# PANEL BACKGROUND TINTS. A tint marks a panel whose reading RULE differs from its neighbours',
# and nothing else -- never decoration. Two are defined because this paper's Figure A has two
# such rules, and they must not be confusable with each other:
#   TINT_CONTROL  warm  -- a HIGH bar is a FAILURE (same molecule, written two ways)
#   TINT_REF      cool  -- the panel that DEFINES the unit, 1.000 by construction
# TINT_REF is deliberately much fainter than TINT_CONTROL. They are not two members of a pair:
# the reference panel is read exactly like a class-A panel and only needs marking as "this is the
# denominator", whereas a control panel INVERTS the reading. A reference tinted as strongly as a
# control makes the reader apply the inverted rule to it.
TINT_CONTROL = "#F0EDE6"
TINT_REF = "#F4F5F7"


def install():
    mpl.rcParams.update({
        "figure.dpi": STYLE["dpi_screen"], "savefig.dpi": STYLE["dpi_save"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "font.size": FS["tick"],
        "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "mathtext.default": "regular", "mathtext.fontset": "custom",
        "mathtext.rm": FONT, "mathtext.it": FONT, "mathtext.bf": FONT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#000000", "axes.labelcolor": "#000000",
        "axes.titlepad": 5.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.color": "#000000", "ytick.color": "#000000",
        "text.color": "#000000",
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "lines.linewidth": STYLE["lw"], "lines.markersize": STYLE["marker_size"],
        "hatch.linewidth": 0.35,          # fine dots, not fat blobs
        # Alpha is 1.0 and the COLOUR carries the lightness, so what you set is what you get. A
        # 0.30 alpha on a light grey renders an effective ~#EFEFEF, i.e. gridlines invisible in
        # print -- the exact bug CLIMB fixed here.
        "axes.grid": False, "grid.color": STYLE["grid"], "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.3,
        "legend.columnspacing": 1.0, "legend.labelspacing": 0.35,
    })
    OUTDIR.mkdir(parents=True, exist_ok=True)
    BUILDDIR.mkdir(parents=True, exist_ok=True)


# ONE LEGEND FRAME FOR THE WHOLE SET. A legend that sits ON the data needs an opaque box or the
# points behind it show through and the labels stop being readable. Defined here and imported
# rather than restated per figure, because two hand-written copies of "the same" frame is how
# they drift.
LEGEND_BOX = dict(frameon=True, framealpha=1.0, edgecolor=STYLE["ink"], facecolor="white")


def row_ncol(handles, rows=1):
    """Column count that lays `handles` out in ONE ROW -- the paper's default.

    Use it as `ncol=row_ncol(handles)` instead of a literal. A literal cannot know that a legend
    grew a key, so adding an arm silently wraps the legend onto a second row; deriving the count
    from the handles themselves means adding an arm keeps the row.

    Why one row is the default and not merely a preference: savefig(bbox_inches="tight") grows
    the canvas to fit whatever hangs off the axes, so a legend that wraps to two rows pushes the
    plate taller and the panels get scaled DOWN in LaTeX to fit the text block. A one-row legend
    is the cheapest vertical space in the figure. It costs width.

    `rows` is the escape hatch for a legend whose one-row form would genuinely overrun the text
    block. Reach for it only after measuring the rendered width, never pre-emptively.
    """
    n = len(list(handles))
    if n <= 0:
        return 1
    if rows <= 1:
        return n
    return -(-n // int(rows))                      # ceil, so `rows` rows is the tallest it gets


def mark_empty(ax, why="no data for this panel"):
    """Declare an axes DELIBERATELY empty, so check_no_empty_panels() does not fail on it.

    Use only where the emptiness is the message, or where a gridspec cell is held open for
    something else (Figure A parks its legend in one). Anywhere else, an empty panel is a bug.
    """
    ax._hume_intentionally_empty = why


def _panel_has_data(ax):
    """Does this axes draw anything a reader could read a VALUE off?

    Reference lines are excluded on purpose: a panel carrying only an axhline at 1.0 and no bars
    looks like a real panel and would pass a naive artist count. axhline/axvline build a BLENDED
    transform (data in one axis, axes-fraction in the other), which is exactly the signature of a
    line that spans the panel rather than describing points in it.
    """
    from matplotlib.transforms import BlendedGenericTransform
    for ln in ax.lines:
        if isinstance(ln.get_transform(), BlendedGenericTransform):
            continue                       # axhline / axvline: a reference, not data
        if len(ln.get_xdata()):
            return True
    return bool(ax.patches or ax.collections or ax.images or ax.containers)


def check_no_empty_panels(fig, name):
    """RAISE if any axes was handed to the renderer and drew no data.

    A figure that silently draws nothing is worse than one that fails: an empty panel is
    indistinguishable from a populated one at a glance, and the reader concludes the arm scored
    zero rather than that the arm is missing.
    """
    empty = [ax for ax in fig.axes
             if not getattr(ax, "_hume_intentionally_empty", None)
             and not _panel_has_data(ax)
             and ax.get_label() != "<colorbar>"]
    if empty:
        where = ", ".join(ax.get_title() or f"axes at {ax.get_position().bounds}" for ax in empty)
        raise AssertionError(
            f"{name}: {len(empty)} panel(s) drew no data -- {where}. If that is intended, call "
            f"figures.style.mark_empty(ax, why) at the draw site.")


def _pdf_width_in(path):
    """Width of a saved PDF's media box, in inches (None if it cannot be parsed)."""
    import re
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                  path.read_bytes()[:4000])
    return (float(m.group(3)) - float(m.group(1))) / 72 if m else None


def save(fig, name, formats=("png", "pdf"), subdir=None, wide=False):
    """Save to figures/<name>.<ext>. Returns the PNG path.

    savefig uses bbox_inches="tight", so the width actually written is NOT the figsize width: it
    shrinks when a figure has slack margins and GROWS when anything (a legend anchored outside
    the axes, a suptitle above the canvas) sits beyond the canvas. Two figures authored at the
    same width can therefore land 1.5in apart, and LaTeX then scales them differently at
    \\includegraphics[width=\\textwidth] -- so their fonts print at different sizes even though
    every script sets the same points. This check makes that loud instead of silent.
    """
    check_no_empty_panels(fig, name)
    out = OUTDIR / subdir if subdir else OUTDIR
    build = BUILDDIR / subdir if subdir else BUILDDIR
    out.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig((out if ext == "pdf" else build) / f"{name}.{ext}")
    rel = f"figures/{subdir}/{name}" if subdir else f"figures/{name}"
    print(f"  saved  {rel}." + "/".join(formats))
    if "pdf" in formats:
        # Mirror to the flat browse folder. Failing to write it must never break a render --
        # it is a convenience, and the authoritative copy is the one beside the script.
        try:
            import shutil
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / f"{name}.pdf", EXPORT_DIR / f"{name}.pdf")
            print(f"  mirrored {EXPORT_DIR / (name + '.pdf')}")
        except OSError as e:
            print(f"  (could not mirror to {EXPORT_DIR}: {e})")
    if "pdf" in formats:
        w = _pdf_width_in(out / f"{name}.pdf")
        if wide:
            print(f"  (wide figure: {w:.2f}in -- set landscape/full-bleed on purpose, "
                  f"not scaled to the {A4_TEXT:.2f}in text block)")
        elif w is not None and abs(w - A4_TEXT) / A4_TEXT > 0.05:
            print(f"  WARNING  {name}: rendered {w:.2f}in vs page width {A4_TEXT:.2f}in "
                  f"({(w / A4_TEXT - 1) * 100:+.0f}%) -- fonts will not match the rest of the set")
    return out / f"{name}.png"


def title(target, text, pad=6, **kw):
    """Title a figure. Pass an Axes for a single-panel figure -- an axes title sits directly above
    the plot, whereas a suptitle floats at the top of the canvas and leaves a band of white space.
    Pass a Figure only for multi-panel layouts."""
    if hasattr(target, "set_title"):
        target.set_title(text, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"],
                         pad=pad, **kw)
    else:
        target.suptitle(text, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"], **kw)


# NOTE: there is deliberately no caption() helper. Captions are NEVER drawn into the figure --
# they belong in the LaTeX \caption{}. Document a figure in the script's docstring instead.


def check_font():
    """Fail loudly if the pinned font is missing -- a silent fallback would change every figure."""
    from matplotlib import font_manager as fm
    if FONT not in {f.name for f in fm.fontManager.ttflist}:
        raise RuntimeError(
            f"font {FONT!r} not installed; figures would silently use a different face")


install()
