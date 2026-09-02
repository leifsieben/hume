"""Single source of truth for the HUME paper figures: arm nomenclature and colors.

Imported by every figure script under `figures/`. Nothing else in the repo defines an arm label
or an arm color -- never hard-code either in a figure script.

VISUAL CONTINUITY WITH THE CLIMB PAPER IS DELIBERATE (Leif 2026-08-26). The hue families and the
exact hexes are CLIMB's, re-mapped onto HUME's semantic slots, so a model that appears in both
papers keeps its color across them:

    ECFP4        #8A5F1B   same amber as CLIMB's ECFP4 anchor
    Morgan r=3   #4E340B   same as CLIMB's R3FP
    ChemBERTa-2  #5C4A85   same as CLIMB's ChemBERTa-2
    MoLFormer    #8B7BB5   same as CLIMB's MoLFormer

Color scheme -- SEMANTIC SLOTS, not tastes:

    amber   classical featurizations: fingerprints and the descriptor block
    crimson HUME, i.e. this paper's contribution
    violet  external chemical language models
    teal    external graph / GNN foundation models
    blue    the descriptor PROXY ladder (ridge -> GNN) -- figures C/D only
    gray    controls

Shades within a family run dark -> light, and every family also spans a distinct LIGHTNESS band
so the figures survive grayscale printing. The hues themselves are CLIMB's CVD-nudged set: the
plain orange/red/green triple is the one pairing deuteranopes cannot separate, so "red" here is
a magenta-leaning crimson and "green" a bluish teal, both anchored on Okabe-Ito.

EXACT vs PREDICTED IS A HATCH, NOT A HUE. `hume_core_predict` carries the SAME color as
`hume_core` with a hatch over it. The paper's central comparison is a descriptor block computed
exactly against the same block predicted by a proxy, and encoding that as two unrelated colors
would make the pair read as two unrelated arms; encoding it as a hatch says "same thing, cheaper
route" at a glance and costs no hue. It also keeps the crimson family readable when all four
HUME arms are drawn together.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------------
# color families
# --------------------------------------------------------------------------------------------
FAMILY_COLORS = {
    "anchor":  "#8A5F1B",   # gold    -- FINGERPRINTS, and fingerprint+descriptor combinations
    "desc":    "#2E6FAF",   # blue    -- the DESCRIPTOR block on its own
    "clm":     "#8F2D3B",   # red     -- external chemical language models
    "graph":   "#5B3D8F",   # purple  -- external graph foundation models
    "hume":    "#2A7F62",   # teal    -- this paper
    "proxy":   "#4E6273",   # slate   -- the descriptor proxy ladder (figures C/D only)
    "control": "#8A8A8A",   # gray
}

# ASSIGNMENT FIXED BY LEIF 2026-08-26: "a color scheme for CLMs (red) and then one for GNNs
# (purple), then descriptors are blue and fingerprints are gold as they are now."
#
# HUME MOVED TO TEAL AS A CONSEQUENCE, not as a preference. It held crimson, which under the new
# scheme is the CLM family's hue -- and the two do co-occur (the final benchmark figure puts HUME
# against every external baseline), so leaving it would have put this paper's own arm in the
# color of the models it is being compared against.
#
# Teal rather than a pure green: HUME sits beside the red CLM family, and red/green is precisely
# the pairing a deuteranope cannot separate. A bluish green keeps the "distinct new thing"
# reading while staying separable, and it is far enough from the descriptor blue at #2E6FAF to
# hold up in the figures where both appear.
#
# The proxy ladder went indigo -> slate for the same reason: indigo beside the new purple GNN
# family was two purples. Slate is desaturated, and the proxies are only ever drawn against each
# other in figures C/D, where no arm shares the axes.

# WHY DESCRIPTORS-ALONE LEFT THE AMBER FAMILY (Leif 2026-08-26: "give descriptors a different
# color, they should only look similar to fingerprints if it's fp+desc, on their own they should
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
    # bare fingerprints beside it. Roughly even lightness steps, so the family also prints gray.
    # THE ONE DELIBERATE BREAK WITH CLIMB. CLIMB gives R3FP #4E340B, one step darker than its
    # ECFP4 #8A5F1B -- fine there, because fig_G drops R3FP and the two never sit side by side.
    # HUME's Figure A puts them adjacent in all thirteen panels and the radius comparison is a
    # headline claim, so two dark browns a shade apart is not a color scheme, it is a hazard.
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
    "hume":   ["#2A7F62",   # [0] hume_core_custom -- the headline arm
               "#63B097",   # [1] hume_core
               "#164A38",   # [2] spare (dark)
               "#9ED3C0"],  # [3] spare (light)
    "desc":   ["#2E6FAF", "#6396CA", "#A3C3E2"],
    # FIVE clm shades, not three: Figures B and C draw ChemBERTa-2 TWICE (the MLM and MTR
    # pretraining variants, see the ARMS entries) and add CDDD, whose input is also a string.
    # The MLM/MTR pair take ADJACENT shades [0]/[1] on purpose -- they are the same architecture
    # on the same corpus and the figure's argument is that only the pretraining target differs,
    # so they must read as a matched pair rather than as two unrelated models.
    # SHADE ORDER WITHIN A FAMILY IS SET BY WHAT SITS ADJACENT IN A FIGURE, not by ARM_ORDER.
    # Figure B draws CheMeleon beside MiniMol and ChemBERTa-2-MTR beside MoLFormer inside every
    # group, five bars wide; on the first build those pairs were one shade apart and the panel
    # was unreadable. Each pair now takes the family's darkest and lightest step. Two documented
    # constraints survive: CheMeleon/Chemprop stay ADJACENT (same architecture, pretrained vs
    # untrained -- a matched pair), and ChemBERTa MLM/MTR stay ADJACENT (the controlled
    # pretraining ablation). The ramp is therefore NOT monotonic in ARM_ORDER, and separability
    # wins over that, on the same reasoning that moved Morgan r=3 to the light end of the ambers.
    "clm":    ["#8F2D3B", "#B03D49", "#C04A55", "#D9737C", "#E08A92"],
    # FOUR graph shades: Uni-Mol joins MiniMol / CheMeleon / Chemprop.
    "graph":  ["#5B3D8F", "#7A56A8", "#8464B5", "#B9A5D6"],
    "proxy":  ["#26303A", "#4E6273", "#7B8FA1", "#A8B7C4", "#D0D9E1"],
    "control": ["#8A8A8A", "#B4B4B4", "#2B2B2B"],
}

# --------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------
# key     -> the string used in every results file and every embedding npz stem
# label   -> the ONLY string that may appear in a figure
# family  -> color family
# color   -> exact color
# hatch   -> bar hatch, or None. Reserved for "predicted rather than computed" (see module head).
#
# LABEL RULES, because these strings sit next to each other on a page:
#   * "ECFP" not "ECFP4+stereo" -- chirality is on for BOTH fingerprints, so the suffix would
#     mark a property they share.
#   * "Morgan r=3" not "R3FP". CLIMB calls it R3FP; this paper contrasts the two RADII directly
#     and in that context the radius has to be visible in the label, which "R3FP" hides.
#   * The descriptor library is named where it distinguishes arms ("+ RDKit", "+ Mordred") and
#     elided where it does not ("+ descriptors" = both).
#   * "predicted" spelled out. Never "pred", never "surrogate" -- the paper uses one word for
#     this and a reader should not have to learn a second.
# SHORT LABELS (Leif). "ECFP + all desc", "ChemBERTa-2 (MTR)" and friends spent a third of
# every legend on detail no comparison in this set turns on -- the radius, the pretraining
# objective and the embedding width are in the methods, and repeating them on four plates made
# the axes narrower without making any of them clearer. The keys are unchanged, so nothing that
# reads a CSV or a results file has to know this happened.
ARMS = {
    # ---- classical featurizations (amber) ---------------------------------------------------
    "ecfp": dict(label="ECFP", family="anchor", color=SHADES["anchor"][0]),
    "r3cfp": dict(label="Morgan r=3", family="anchor", color=SHADES["anchor"][1]),
    # r=4 extends the radius series. It sits between r=2 and r=3 in the amber family rather than
    # taking a new color: the three are one variable, and Figure A puts them adjacent in every
    # panel, so they must read as a series and not as three unrelated arms.
    "r4cfp": dict(label="Morgan r=4", family="anchor", color=SHADES["anchor"][5]),
    "ecfp_all_desc": dict(label="ECFP + all desc", family="anchor",
                          color=SHADES["anchor"][2]),
    "ecfp_rdkit_desc": dict(label="ECFP + RDKit", family="anchor", color=SHADES["anchor"][3]),
    "ecfp_mordred_desc": dict(label="ECFP + Mordred", family="anchor",
                              color=SHADES["anchor"][4]),
    # Its OWN family -- no fingerprint in this arm. See the note above FAMILY_COLORS.
    "desc": dict(label="RDKit + Mordred", family="desc", color=SHADES["desc"][0]),
    # THE DESCRIPTOR BLOCK SPLIT BY LIBRARY, with no fingerprint. Figure B's x-axis walks from
    # "one library alone" to "everything", and the point of the walk is that the DL embedding's
    # contribution shrinks as the classical base gets more complete -- which needs the
    # intermediate rungs to exist as arms, not just the endpoints.
    "desc_rdkit": dict(label="RDKit only", family="desc", color=SHADES["desc"][1]),
    # THE WITHIN-GROUP REFERENCE BAR. In Figure B color means "which embedding was added" and
    # the group name means "to which classical block", so the block-alone bar must carry ONE
    # color across every group -- otherwise color would mean two things in one panel. Neutral
    # gray, because it is the thing everything beside it is measured against.
    "classical_base": dict(label="classical block alone", family="control",
                           color=SHADES["control"][0]),
    "desc_mordred": dict(label="Mordred only", family="desc", color=SHADES["desc"][2]),

    # ---- HUME (crimson) ---------------------------------------------------------------------
    # The `_predict` arms share their exact counterpart's color and add a hatch. See module head.
    # THE HUME ARM. One arm, not a family (Leif 2026-08-28: "there is just one hume ... right
    # now its ecfp + all descriptors (incl our own) we compute full stop"). The `hume_core*`
    # entries below are an earlier factorisation that no figure draws any more; they are kept
    # so an old CSV still resolves its labels. They remain in ARM_ORDER, which is harmless --
    # `order()` only positions arms that are actually passed to it -- but a figure must not draw
    # `hume` and `hume_core_custom` together, because they share a shade.
    # NAMED HUME_full, NOT "HUME" (Leif 2026-09-02). Three HUME arms are drawn together in
    # Figures C and D and one of them being the bare project name made the other two read as
    # variants of it rather than as three widths of one specification.
    "hume": dict(label="HUME_full", family="hume", color=SHADES["hume"][0]),
    # THE ABLATION PAIR. `hume_no_new` is HUME with the 185 columns wired after the
    # deduplication masked out -- counts_ext, estate_ext, eta, spectral and misc_ext, minus the
    # 43 the cost triage dropped. Same block, same model, same folds; the only difference is
    # those columns, so the gap between the two IS what they are worth. Lighter shade of the
    # same family, because it is the same method and not a competitor.
    "hume_no_new": dict(label="HUME_no_new", family="hume", color=SHADES["hume"][1]),
    # The 622-column reduced spec (minimal-v2).
    #
    #  ITS x POSITION IS NO LONGER HUME_full's, AND THAT IS A REAL CHANGE. Until mol-hume
    # 0.7.0 the column selection chose what was RETURNED and not what was COMPUTED, so all three
    # HUME arms sat on one point by construction and the panel read as "what does dropping the
    # columns cost, for free". Since 0.7.0 the selection is a compute plan -- a descriptor family
    # none of whose columns are selected is not calculated -- so HUME_minimal is genuinely
    # cheaper and moves LEFT. HUME_no_new does not move: its 1,109 columns still span every one
    # of the nineteen families, so there is nothing for the plan to skip. Any figure that still
    # draws the three at one x is reading a pre-0.7.0 cost file.
    "hume_minimal": dict(label="HUME_minimal", family="hume", color=SHADES["hume"][2]),
    # NOT A REPRESENTATION -- a difficulty floor. Character 1- and 2-gram counts of the SMILES,
    # no chemistry at all, so whatever it scores on an edit is free to any model that reads the
    # string. Gray, like every other control in the set.
    "notation": dict(label="SMILES characters", family="control", color="#9AA0A6"),
    "hume_core": dict(label="HUME core", family="hume", color=SHADES["hume"][1]),
    "hume_core_custom": dict(label="HUME core + blocks", family="hume", color=SHADES["hume"][0]),
    "hume_core_predict": dict(label="HUME core, predicted", family="hume",
                              color=SHADES["hume"][1], hatch="///"),
    "hume_core_custom_predict": dict(label="HUME + blocks, predicted", family="hume",
                                     color=SHADES["hume"][0], hatch="///"),
    # FIGURE C splits "predicted" by WHICH PROXY did the predicting, so the hatch carries the
    # proxy identity while the hue still says "this is HUME". Extending the hatch channel rather
    # than the hue keeps the module-head rule intact: hatch means "predicted rather than
    # computed", and now also which route. The proxies keep their own colors in the `proxy`
    # family for figures that compare proxies to each other rather than to arms.
    "hume_predict_ridge": dict(label="HUME, ridge-predicted", family="hume",
                               color=SHADES["hume"][0], hatch="///"),
    "hume_predict_gnn": dict(label="HUME, GNN-predicted", family="hume",
                             color=SHADES["hume"][0], hatch="xxx"),
    # THE WIDTH ABLATION (Leif 2026-08-27: "I just worry XGBoost might be overwhelmed with too
    # many features"). Same descriptor block, ECFP folded to 1024 instead of 2048. It is a
    # separate ARM and not a styling of the headline one, because if it wins it is what ships.
    # Note Gate 1 already tested the descriptor half of that worry from two directions -- a
    # supervised top-30 cherry-pick and an unsupervised PCA-64 -- and BOTH lost, so this arm
    # isolates the remaining variable, which is fingerprint width rather than descriptor count.
    "hume_1024": dict(label="HUME, ECFP-1024", family="hume", color=SHADES["hume"][3]),
    # THE FINGERPRINT-ENCODING ABLATION (Leif 2026-08-27: "let's include in C just to see if we
    # can pick up a trend there"). Same descriptor block, same radius, same width -- the
    # fingerprint is COUNTS rather than binary. Counts strictly dominate binary in information,
    # so if binary wins it is a generalisation effect at small n rather than an information one,
    # and that is worth knowing before a default is picked by argument. Costs 1.08x to compute
    # (28.37 vs 26.25 us/mol at r=3), so this is not a speed trade either way.
    "hume_counts": dict(label="HUME, count fingerprint", family="hume", color=SHADES["hume"][2]),

    # ---- external chemical language models (violet) -----------------------------------------
    # THE NUMBER IN A CLM'S NAME IS PRETRAINING DATA, NOT PARAMETERS, and the two orderings are
    # opposite: ChemBERTa-77M-MTR is 3.4M parameters, MoLFormer-XL 44.4M, SMI-TED 358.1M. Never
    # print a name's number as a size.
    #
    # CHEMBERTA IS TWO ARMS, AND THIS ENTRY USED TO BE A PROVENANCE BUG. It declared
    # hf="DeepChem/ChemBERTa-77M-MTR" while gpu/fetch_hf.py downloaded ...-77M-MLM and the
    # on-disk config.json says RobertaForMaskedLM. Every ChemBERTa number in Figure A is
    # therefore the MLM checkpoint, under a registry entry naming the MTR one -- i.e. the paper
    # would have printed the wrong model name. Found 2026-08-27.
    #
    # Splitting the arm turns that bug into the paper's cleanest experiment. The two checkpoints
    # are the SAME architecture (3 layers, 384 hidden), the SAME 77M-molecule corpus and the
    # same parameter count; the only difference is the pretraining target. MTR is multi-task
    # regression onto 200 RDKit descriptors -- so the pair is a controlled ablation of
    # "does supervising on descriptors make the embedding better?", run by the model's own
    # authors and reported as "our model is better" rather than as a statement about
    # descriptors. Figures B and C reinterpret it.
    "chemberta_mlm": dict(label="ChemBERTa (MLM)", family="clm", color=SHADES["clm"][1],
                          hf="DeepChem/ChemBERTa-77M-MLM", desc_pretrained=False),
    "chemberta_mtr": dict(label="ChemBERTa", family="clm", color=SHADES["clm"][0],
                          hf="DeepChem/ChemBERTa-77M-MTR", desc_pretrained=True),
    "molformer": dict(label="MoLFormer", family="clm", color=SHADES["clm"][4],
                      hf="ibm-research/MoLFormer-XL-both-10pct", desc_pretrained=False),
    # CDDD IS DROPPED, and this entry stays as the record of why rather than vanishing.
    # (Leif 2026-08-27: "Let's drop CDDD.") It is a 2019 translation autoencoder whose latent
    # vector was explicitly proposed as a DESCRIPTOR SUBSTITUTE -- this paper's thesis, stated as
    # a design goal seven years earlier -- so it would have been the single most on-the-nose arm
    # in Figure B. It cannot be run:
    #   * the `cddd` PyPI package requires tensorflow-gpu==1.10.0, wheels cp27-cp36, linux/win
    #     x86_64 only;
    #   * tensorflow==1.15.5 wheels are cp36m/cp37m, linux + macOS x86_64 -- no arm64 build of
    #     TF 1.x exists at all;
    #   * uv's lowest macOS-arm64 Python is 3.8, so there is no interpreter it would install
    #     against;
    #   * no PyTorch port exists on PyPI (cddd-pytorch, pytorch-cddd, cddd2, molvecgen: absent).
    # The only route is an x86_64 Python 3.7 under Rosetta with TF 1.15. The PAPER SHOULD SAY SO
    # -- "we could not run it" is a finding about the field's reproducibility, and it is more
    # honest than a silent omission a reader would read as an oversight.
    # SELFIES-TED, not SMI-TED (Leif 2026-08-26: "more interesting model"). Both are IBM
    # encoder-decoders; SELFIES-TED reads SELFIES rather than SMILES, which makes it the only arm
    # on the plate whose input grammar cannot express an invalid molecule. That is exactly the
    # property Figure A's two notation controls are built to test, so it is the more informative
    # of the pair here. SMI-TED's weights stay on disk but no figure draws it.
    "selfies_ted": dict(label="SELFIES-TED", family="clm", color=SHADES["clm"][2],
                        hf="ibm-research/materials.selfies-ted", desc_pretrained=False),

    # ---- external graph foundation models (blue) --------------------------------------------
    # `chemprop` is a randomly-initialized D-MPNN -- the architecture with NO pretraining. It is
    # a CONTROL, not a pretrained comparator, and its label says so: an untrained encoder that
    # still responds to chemistry tells you how much of a graph model's sensitivity is
    # architectural rather than learned. CheMeleon is the same architecture pretrained, so the
    # two form a matched pair and must keep adjacent shades.
    "minimol": dict(label="MiniMol", family="graph", color=SHADES["graph"][3],
                    desc_pretrained=False),
    # CheMeleon is PRETRAINED TO PREDICT MORDRED DESCRIPTORS. That is the paper's own pitch --
    # descriptor pretraining beating masked-atom pretraining -- and it is why this arm carries
    # desc_pretrained=True rather than because of anything measured here.
    "chemeleon": dict(label="CheMeleon", family="graph", color=SHADES["graph"][0],
                      desc_pretrained=True),
    # Uni-Mol needs a CONFORMER, so it is the only arm here whose input is 3D. It stays in the
    # graph family (it is a geometry-aware graph transformer, not a string model) and the label
    # carries the 3D rather than a sixth hue: a new color family for one arm costs the reader
    # more than a two-character suffix does. It is also the arm most likely to legitimately BEAT
    # descriptors on the QM panel, and the figure should show that rather than hide it.
    "unimol": dict(label="Uni-Mol (3D)", family="graph", color=SHADES["graph"][2],
                   desc_pretrained=False),
    "chemprop": dict(label="Chemprop", family="graph", color=SHADES["graph"][1],
                     desc_pretrained=False),

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
ARM_ORDER = ["ecfp", "r3cfp", "r4cfp", "ecfp_all_desc", "ecfp_rdkit_desc", "ecfp_mordred_desc",
             "desc_rdkit", "desc_mordred", "desc",
             "hume", "hume_no_new", "hume_minimal",
             "hume_core", "hume_core_predict", "hume_core_custom", "hume_core_custom_predict",
             "hume_predict_ridge", "hume_predict_gnn", "hume_1024", "hume_counts",
             # GRAPH BEFORE STRING (Leif 2026-08-27: "all ECFP on the very left, then all
             # hume, then all GNN, then all CLM ... so the ordering from left to right makes
             # comparisons easier"). Applies to every figure at once, which is the point of
             # this list existing: two figures must not put the same arms in two orders.
             "minimol", "chemeleon", "unimol", "chemprop",
             "chemberta_mlm", "chemberta_mtr", "molformer", "selfies_ted"]
_unknown = set(ARM_ORDER) - set(ARMS)
assert not _unknown, f"arms.py: ARM_ORDER names an arm that does not exist: {sorted(_unknown)}"

PROXY_ORDER = ["ridge", "linquad", "pinet", "mlp", "gnn"]


def label(key: str) -> str:
    return ARMS[key]["label"] if key in ARMS else key


#: PLAIN NAMES, NO PARENTHESES, for use as tick labels and legend entries (Leif 2026-08-29:
#: "just write ChemBERTa and no parentheses, same for all the other models"). `label()` keeps the
#: fully-qualified name for anywhere the distinction is load-bearing -- Figure A draws BOTH
#: ChemBERTa-2 checkpoints and must not call them the same thing.
SHORT_LABEL = {
    "ecfp": "ECFP", "r3cfp": "Morgan r=3", "r4cfp": "Morgan r=4",
    "desc": "descriptors", "ecfp_rdkit_desc": "ECFP + RDKit",
    "ecfp_mordred_desc": "ECFP + Mordred", "ecfp_all_desc": "ECFP + all desc",
    "hume": "HUME_full", "hume_no_new": "HUME_no_new", "hume_minimal": "HUME_minimal",
    "minimol": "MiniMol",
    "chemeleon": "CheMeleon", "chemprop": "Chemprop",
    "chemberta_mtr": "ChemBERTa", "chemberta_mlm": "ChemBERTa", "molformer": "MoLFormer",
    "selfies_ted": "SELFIES-TED", "classical_base": "classical block alone",
}


def short_label(key: str) -> str:
    """A parenthesis-free display name. Falls back to `label()` for anything unregistered."""
    return SHORT_LABEL.get(key, label(key))


def color(key: str) -> str:
    return ARMS[key]["color"] if key in ARMS else "#999999"


def hatch(key: str):
    return ARMS.get(key, {}).get("hatch")


# DESCRIPTOR PRETRAINING IS A THIRD VISUAL CHANNEL, and it has to be, because it is the whole
# argument of Figure B. Hue is already spent on the model family and hatch on exact-vs-predicted,
# so this rides on the bar EDGE: a heavy dark outline means "this model was pretrained to predict
# molecular descriptors".
#
# It is a property of the MODEL, recorded here as data, and the figure decides how to draw it --
# so a second figure cannot encode it a different way. Only external pretrained models carry the
# flag; it is meaningless for a fingerprint or for HUME itself, and `is None` distinguishes "not
# applicable" from an explicit False.
#
# The two arms flagged True are CheMeleon (pretrained on Mordred descriptors) and ChemBERTa-2-MTR
# (multi-task regression on 200 RDKit descriptors). Both facts come from those models' own
# papers, not from anything measured here -- if either turns out to be wrong on checking, the
# flag is wrong and the figure's claim with it, so verify before publishing.
EDGE_DESC_PRETRAINED = dict(edgecolor="#1A1A1A", linewidth=0.9)


def desc_pretrained(key: str):
    """True / False / None -- None meaning the question does not apply to this arm."""
    return ARMS.get(key, {}).get("desc_pretrained")


def bar_kw(key: str) -> dict:
    """Every visual property of one arm's bar, in one place.

    Figures call this instead of assembling color + hatch + edge themselves, so the three
    channels cannot drift apart between Figure B and Figure C.
    """
    kw = dict(color=color(key))
    h = hatch(key)
    if h:
        kw["hatch"] = h
    if desc_pretrained(key):
        kw.update(EDGE_DESC_PRETRAINED)
    return kw


def order(keys) -> list:
    """`keys` sorted into ARM_ORDER, with anything unregistered appended alphabetically.

    Unregistered arms are APPENDED rather than dropped: a figure drawn from whatever results
    exist should show a new arm in an obvious place and in the default gray, so it is visible
    that it needs registering -- not silently vanish from the plate.
    """
    known = [a for a in ARM_ORDER if a in set(keys)]
    return known + sorted(set(keys) - set(known))
