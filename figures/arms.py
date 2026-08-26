"""Single source of truth for the HUME paper figures: arm nomenclature and colours.

Imported by every figure script under `figures/`. Nothing else in the repo defines an arm label
or an arm colour -- never hard-code either in a figure script.

VISUAL CONTINUITY WITH THE CLIMB PAPER IS DELIBERATE (Leif 2026-08-26). The hue families and the
exact hexes are CLIMB's, re-mapped onto HUME's semantic slots, so a model that appears in both
papers keeps its colour across them:

    ECFP4        #8A5F1B   same amber as CLIMB's ECFP4 anchor
    Morgan r=3   #4E340B   same as CLIMB's R3FP
    ChemBERTa-2  #5C4A85   same as CLIMB's ChemBERTa-2
    MoLFormer    #8B7BB5   same as CLIMB's MoLFormer

Colour scheme -- SEMANTIC SLOTS, not tastes:

    amber   classical featurisations: fingerprints and the descriptor block
    crimson HUME, i.e. this paper's contribution
    violet  external chemical language models
    teal    external graph / GNN foundation models
    blue    the descriptor PROXY ladder (ridge -> GNN) -- figures C/D only
    grey    controls

Shades within a family run dark -> light, and every family also spans a distinct LIGHTNESS band
so the figures survive greyscale printing. The hues themselves are CLIMB's CVD-nudged set: the
plain orange/red/green triple is the one pairing deuteranopes cannot separate, so "red" here is
a magenta-leaning crimson and "green" a bluish teal, both anchored on Okabe-Ito.

EXACT vs PREDICTED IS A HATCH, NOT A HUE. `hume_core_predict` carries the SAME colour as
`hume_core` with a hatch over it. The paper's central comparison is a descriptor block computed
exactly against the same block predicted by a proxy, and encoding that as two unrelated colours
would make the pair read as two unrelated arms; encoding it as a hatch says "same thing, cheaper
route" at a glance and costs no hue. It also keeps the crimson family readable when all four
HUME arms are drawn together.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------------
# colour families
# --------------------------------------------------------------------------------------------
FAMILY_COLORS = {
    "anchor":  "#8A5F1B",   # amber   -- classical featurisations
    "hume":    "#A3455E",   # crimson -- this paper
    "clm":     "#5C4A85",   # violet  -- external chemical language models
    "gnn":     "#3D8073",   # teal    -- external graph foundation models
    "proxy":   "#3F6E9C",   # blue    -- the descriptor proxy ladder
    "control": "#8A8A8A",   # grey
}

SHADES = {
    # SIX ambers, not CLIMB's five: HUME's Figure B splits the descriptor block four ways
    # (all / RDKit-only / Mordred-only / none) and each split needs to be separable from the two
    # bare fingerprints beside it. Roughly even lightness steps, so the family also prints grey.
    # THE ONE DELIBERATE BREAK WITH CLIMB. CLIMB gives R3FP #4E340B, one step darker than its
    # ECFP4 #8A5F1B -- fine there, because fig_G drops R3FP and the two never sit side by side.
    # HUME's Figure A puts them adjacent in all thirteen panels and the radius comparison is a
    # headline claim, so two dark browns a shade apart is not a colour scheme, it is a hazard.
    # Morgan r=3 takes the LIGHT end of the same family instead: same amber, maximal lightness
    # separation, and the pair still reads as one family. ECFP4 keeps CLIMB's hex exactly.
    "anchor": ["#8A5F1B",   # [0] ECFP4          (CLIMB's ECFP4 hex, kept)
               "#E8B86A",   # [1] Morgan r=3     (light end -- see note above)
               "#C8912F",   # [2] ECFP + all descriptors  (CLIMB's ECFP4+desc hex, kept)
               "#A87A22",   # [3] ECFP + RDKit only
               "#E0BC80",   # [4] ECFP + Mordred only
               "#4E340B"],  # [5] descriptors alone, no fingerprint
    "hume":   ["#A3455E",   # [0] hume_core_custom -- the headline arm
               "#CB8C9C",   # [1] hume_core
               "#6E2437",   # [2] spare (dark)
               "#DBAEB9"],  # [3] spare (light)
    "clm":    ["#5C4A85", "#8B7BB5", "#B9AED5"],
    "gnn":    ["#2A5C50", "#3D8073", "#5E9C90", "#84B7AD"],
    "proxy":  ["#2A4E73", "#3F6E9C", "#6B93B8", "#9AB6D0", "#C3D5E4"],
    "control": ["#8A8A8A", "#B4B4B4", "#2B2B2B"],
}

# --------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------
# key     -> the string used in every results file and every embedding npz stem
# label   -> the ONLY string that may appear in a figure
# family  -> colour family
# color   -> exact colour
# hatch   -> bar hatch, or None. Reserved for "predicted rather than computed" (see module head).
#
# LABEL RULES, because these strings sit next to each other on a page:
#   * "ECFP4" not "ECFP4+stereo" -- chirality is on for BOTH fingerprints, so the suffix would
#     mark a property they share.
#   * "Morgan r=3" not "R3FP". CLIMB calls it R3FP; this paper contrasts the two RADII directly
#     and in that context the radius has to be visible in the label, which "R3FP" hides.
#   * The descriptor library is named where it distinguishes arms ("+ RDKit", "+ Mordred") and
#     elided where it does not ("+ descriptors" = both).
#   * "predicted" spelled out. Never "pred", never "surrogate" -- the paper uses one word for
#     this and a reader should not have to learn a second.
ARMS = {
    # ---- classical featurisations (amber) ---------------------------------------------------
    "ecfp": dict(label="ECFP4", family="anchor", color=SHADES["anchor"][0]),
    "r3cfp": dict(label="Morgan r=3", family="anchor", color=SHADES["anchor"][1]),
    "ecfp_all_desc": dict(label="ECFP4 + descriptors", family="anchor",
                          color=SHADES["anchor"][2]),
    "ecfp_rdkit_desc": dict(label="ECFP4 + RDKit", family="anchor", color=SHADES["anchor"][3]),
    "ecfp_mordred_desc": dict(label="ECFP4 + Mordred", family="anchor",
                              color=SHADES["anchor"][4]),
    "desc": dict(label="RDKit + Mordred", family="anchor", color=SHADES["anchor"][5]),

    # ---- HUME (crimson) ---------------------------------------------------------------------
    # The `_predict` arms share their exact counterpart's colour and add a hatch. See module head.
    "hume_core": dict(label="HUME core", family="hume", color=SHADES["hume"][1]),
    "hume_core_custom": dict(label="HUME core + blocks", family="hume", color=SHADES["hume"][0]),
    "hume_core_predict": dict(label="HUME core, predicted", family="hume",
                              color=SHADES["hume"][1], hatch="///"),
    "hume_core_custom_predict": dict(label="HUME core + blocks, predicted", family="hume",
                                     color=SHADES["hume"][0], hatch="///"),

    # ---- external chemical language models (violet) -----------------------------------------
    # THE NUMBER IN A CLM'S NAME IS PRETRAINING DATA, NOT PARAMETERS, and the two orderings are
    # opposite: ChemBERTa-77M-MTR is 3.4M parameters, MoLFormer-XL 44.4M, SMI-TED 358.1M. Never
    # print a name's number as a size.
    "chemberta": dict(label="ChemBERTa-2", family="clm", color=SHADES["clm"][0],
                      hf="DeepChem/ChemBERTa-77M-MTR"),
    "molformer": dict(label="MoLFormer", family="clm", color=SHADES["clm"][1],
                      hf="ibm-research/MoLFormer-XL-both-10pct"),
    "smi_ted": dict(label="SMI-TED", family="clm", color=SHADES["clm"][2],
                    hf="ibm/materials.smi-ted"),

    # ---- external graph foundation models (teal) --------------------------------------------
    "minimol": dict(label="MiniMol", family="gnn", color=SHADES["gnn"][0]),
    "chemprop": dict(label="Chemprop", family="gnn", color=SHADES["gnn"][1]),
    "chemeleon": dict(label="CheMeleon", family="gnn", color=SHADES["gnn"][2]),

    # ---- the descriptor proxy ladder (blue) -- figures C/D ----------------------------------
    # These are not representations, they are the five candidate models for the PREDICT block.
    # Their own family so a proxy can never be mistaken for an arm in a downstream comparison.
    "ridge": dict(label="ridge", family="proxy", color=SHADES["proxy"][0]),
    "linquad": dict(label="linear + quadratic", family="proxy", color=SHADES["proxy"][1]),
    "pinet": dict(label="Π-net", family="proxy", color=SHADES["proxy"][2]),
    "mlp": dict(label="MLP", family="proxy", color=SHADES["proxy"][3]),
    "gnn": dict(label="GNN", family="proxy", color=SHADES["proxy"][4]),
}

# DISPLAY ORDER, used by every figure that draws more than one arm. Fixed here so two figures
# cannot put the same arms in two different orders -- which reads as two different comparisons.
# Grouped classical -> HUME -> external, i.e. cheapest to most expensive at inference, which is
# the axis the whole paper is about.
ARM_ORDER = ["ecfp", "r3cfp", "desc", "ecfp_all_desc", "ecfp_rdkit_desc", "ecfp_mordred_desc",
             "hume_core", "hume_core_predict", "hume_core_custom", "hume_core_custom_predict",
             "chemberta", "molformer", "smi_ted", "minimol", "chemprop", "chemeleon"]
_unknown = set(ARM_ORDER) - set(ARMS)
assert not _unknown, f"arms.py: ARM_ORDER names an arm that does not exist: {sorted(_unknown)}"

PROXY_ORDER = ["ridge", "linquad", "pinet", "mlp", "gnn"]


def label(key: str) -> str:
    return ARMS[key]["label"] if key in ARMS else key


def color(key: str) -> str:
    return ARMS[key]["color"] if key in ARMS else "#999999"


def hatch(key: str):
    return ARMS.get(key, {}).get("hatch")


def order(keys) -> list:
    """`keys` sorted into ARM_ORDER, with anything unregistered appended alphabetically.

    Unregistered arms are APPENDED rather than dropped: a figure drawn from whatever results
    exist should show a new arm in an obvious place and in the default grey, so it is visible
    that it needs registering -- not silently vanish from the plate.
    """
    known = [a for a in ARM_ORDER if a in set(keys)]
    return known + sorted(set(keys) - set(known))
