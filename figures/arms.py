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
    "anchor":  "#8A5F1B",   # amber   -- FINGERPRINTS, and fingerprint+descriptor combinations
    "desc":    "#3D8073",   # teal    -- the DESCRIPTOR block on its own
    "hume":    "#A3455E",   # crimson -- this paper
    "clm":     "#5C4A85",   # violet  -- external chemical language models
    "graph":   "#2E8BC0",   # blue    -- external graph foundation models
    "proxy":   "#6B6494",   # indigo  -- the descriptor proxy ladder (figures C/D only)
    "control": "#8A8A8A",   # grey
}

# WHY DESCRIPTORS-ALONE LEFT THE AMBER FAMILY (Leif 2026-08-26: "give descriptors a different
# colour, they should only look similar to fingerprints if it's fp+desc, on their own they should
# be more distinct"). Amber now means "a fingerprint is in this arm". `desc` contains no
# fingerprint at all, and Figure A shows it is not a paler version of one either -- it wins on
# protonation (1.199 vs ECFP4's 0.811) and halogen swap (0.810 vs 0.353) while losing 2-3x on
# every graph edit. Two arms that behave that differently must not share a hue.
#
# Teal was the graph-foundation-model family until this change; those move to blue, and the proxy
# ladder moves from blue to indigo. The proxies appear only in figures C/D, where no arm is drawn,
# so indigo there cannot collide with anything.

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
    # THE THREE HUME DRAWS IN FIGURE A -- [0] ECFP4, [1] Morgan r=3, [5] descriptors alone --
    # take the three WIDEST-SPACED lightness steps, because those three sit adjacent in all
    # thirteen panels. The +descriptor combinations at [2]-[4] only ever appear in Figure B,
    # where the fingerprint-only arms are not drawn beside them.
    "anchor": ["#8A5F1B",   # [0] ECFP4          (CLIMB's ECFP4 hex, kept)
               "#E8B86A",   # [1] Morgan r=3     (light end)
               "#4E340B",   # [2] ECFP + all descriptors
               "#A87A22",   # [3] ECFP + RDKit only
               "#E0BC80",   # [4] ECFP + Mordred only
               "#C8912F"],  # [5] descriptors alone (mid, saturated -- CLIMB's ECFP4+desc hex)
    "hume":   ["#A3455E",   # [0] hume_core_custom -- the headline arm
               "#CB8C9C",   # [1] hume_core
               "#6E2437",   # [2] spare (dark)
               "#DBAEB9"],  # [3] spare (light)
    "desc":   ["#3D8073", "#5E9C90", "#84B7AD"],
    "clm":    ["#5C4A85", "#8B7BB5", "#B9AED5"],
    # Saturated blues, not the pale end of one: the graph family sits directly beside the violet
    # CLM family in Figure A's legend and a pale blue against a pale violet is the one pair a
    # reader would have to check the legend for twice.
    "graph":  ["#0F5C8C", "#2E8BC0", "#8FC4E3"],
    "proxy":  ["#2E2A4A", "#4A4370", "#6B6494", "#8F89B2", "#B7B3CE"],
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
    # Its OWN family -- no fingerprint in this arm. See the note above FAMILY_COLORS.
    "desc": dict(label="RDKit + Mordred", family="desc", color=SHADES["desc"][0]),

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
    # SELFIES-TED, not SMI-TED (Leif 2026-08-26: "more interesting model"). Both are IBM
    # encoder-decoders; SELFIES-TED reads SELFIES rather than SMILES, which makes it the only arm
    # on the plate whose input grammar cannot express an invalid molecule. That is exactly the
    # property Figure A's two notation controls are built to test, so it is the more informative
    # of the pair here. SMI-TED's weights stay on disk but no figure draws it.
    "selfies_ted": dict(label="SELFIES-TED", family="clm", color=SHADES["clm"][2],
                        hf="ibm-research/materials.selfies-ted"),

    # ---- external graph foundation models (blue) --------------------------------------------
    # `chemprop` is a randomly-initialised D-MPNN -- the architecture with NO pretraining. It is
    # a CONTROL, not a pretrained comparator, and its label says so: an untrained encoder that
    # still responds to chemistry tells you how much of a graph model's sensitivity is
    # architectural rather than learned. CheMeleon is the same architecture pretrained, so the
    # two form a matched pair and must keep adjacent shades.
    "minimol": dict(label="MiniMol", family="graph", color=SHADES["graph"][0]),
    "chemeleon": dict(label="CheMeleon", family="graph", color=SHADES["graph"][1]),
    "chemprop": dict(label="Chemprop, untrained", family="graph", color=SHADES["graph"][2]),

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
ARM_ORDER = ["ecfp", "r3cfp", "ecfp_all_desc", "ecfp_rdkit_desc", "ecfp_mordred_desc", "desc",
             "hume_core", "hume_core_predict", "hume_core_custom", "hume_core_custom_predict",
             "chemberta", "molformer", "selfies_ted", "minimol", "chemeleon", "chemprop"]
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
