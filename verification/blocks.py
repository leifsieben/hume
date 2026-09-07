"""Shared definition of the CORE / PREDICT descriptor split.

Single source of truth so the training data, the models and the downstream evaluation cannot
drift apart. The split is by *primitive cost*, not by downstream benefit:

    CORE     767 descriptors reachable from primitives the C++ featuriser already computes
    PREDICT   98 descriptors that are not

REDRAWN 2026-08-27 (stage 1). The 2026-08 C++ port changed every cost the old boundary was
drawn on, and left it asking a surrogate to approximate columns CORE WAS ALREADY PAYING THE
PRIMITIVE FOR:

  * CORE's `Lipinski`/`GhoseFilter` depend on mordred `SLogP`/`SMR`, which are verbatim
    `Crippen.MolLogP` / `Crippen.MolMR` (mordred/SLogP.py is a five-line rdkit wrapper).
    CORE therefore already pays Crippen -- and PREDICT then asked a model to approximate the
    20 columns (`SlogP_VSA*`, `SMR_VSA*`, `MolMR`) that are binned sums of those same contribs.
  * 34 CORE `Autocorrelation` columns (`ATSC0-8c`, `AATSC0-8c`, `MATS1-8c`, `GATS1-8c`) use
    mordred's `c` atomic property, which is `ComputeGasteigerCharges`. CORE already pays
    Gasteiger -- and PREDICT then asked a model to approximate the 13 `PEOE_VSA*` columns
    binned from those same charges.

Verified here on 6,000 adversarial molecules (cpp/hard.smi, first 6k): 62 columns -- the 57
RDKit `*_VSA` columns, `LabuteASA`, and the four EState extremes -- reconstruct BIT-EXACTLY
(rtol 1e-9) from four per-atom vectors: Crippen (logP, MR) contribs, Gasteiger charges, EState
indices, Labute ASA contribs. The bin edges are module-level constants in rdkit/Chem/MolSurf.py
(`logpBins`, `mrBins`, `chgBins`) and rdkit/Chem/EState/EState_VSA.py (`estateBins`, `vsaBins`),
not per-molecule work -- each column is one O(n) `bisect_right` pass over a vector the
primitive already produced. So there is nothing here for a surrogate to approximate: once the
primitive is paid, the exact answer is cheaper to compute than to predict. (The MARGINAL cost
of the promotion is not measured in this file. Measure it on a shared machine only with
in-process alternating pairs, and do not quote a Python number for it.)

This redraw moves 68 columns from PREDICT to CORE and moves nothing the other way; CORE is a
strict superset of what it was. No descriptor's VALUE changes -- only which block owns it.

Three details the port must reproduce, each of which silently breaks reconstruction:
  * `LabuteASA` is `sum(contribs) + hContrib`; `_CalcLabuteASAContribs` returns the H term
    SEPARATELY from the length-N heavy-atom vector. Dropping it is a small, uniform shortfall.
  * `PEOE_VSA` does NOT clamp NaN charges (elements with no Gasteiger parameters, e.g. Sn).
    NaN falls through `bisect_right` into the FINAL bin. Clamping to 0.0 misroutes it.
  * `MolLogP` and `MolMR` -- unlike every `*_VSA` column -- sum over the H-ADDED molecule, so
    they need the Crippen H rows (H1-H4). cpp/crippen.cpp has them (rules r039+).

STAGE 2 is deliberately NOT applied: the mordred `EState` family (50), `AcidBase` (2), `LogS`
(1) and `Framework` (1) stay in PREDICT pending ports that do not exist yet. See classify().
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEDUPE = ROOT / "data" / "dedupe.json"

CORE_FAM = {
    # Chi and PathCount were moved out of PREDICT on 2026-08-24. They were classified as
    # expensive because Mordred's Chi family costs 10,328 us -- but that is Mordred being
    # Python, not paths being expensive. Chi is also the highest-value predicted family at
    # +0.126 downstream, so this removes the largest single source of surrogate error.
    #
    # WHAT THE CITED MEASUREMENT ACTUALLY COVERS (corrected 2026-08-27). The 6.82 us from
    # cpp/bench.cpp is SIMPLE PATH enumeration to k<=7. Counted against mordred's own
    # descriptor metadata, CORE's 40 surviving Chi columns are:
    #     path          21   (AXp-*, Xp-*)          <- covered by the 6.82 us
    #     cluster        7   (Xc-*,   max order 6)  \
    #     path_cluster   5   (Xpc-*,  max order 6)   > 19 columns needing
    #     chain          7   (Xch-*,  max order 7)  /  FindAllSubgraphsOfLengthN
    # and CORE's 11 PathCount columns run to k=10 (piPC10), not k=7. So 6.82 us bounds
    # neither the subgraph half of Chi nor PathCount. Both remain in CORE on the judgement
    # that they are C++-tractable, but that judgement is UNMEASURED -- do not cite 6.82 us
    # for it.
    "Chi", "PathCount",
    "RingCount", "AtomCount", "BondCount", "Constitutional", "CarbonTypes",
    "Autocorrelation", "TopologicalCharge",
    "WienerIndex", "BalabanJ", "TopologicalIndex", "MolecularDistanceEdge",
    "KappaShapeIndex",
    "VdwVolumeABC", "Polarizability", "ABCIndex", "RotatableBond",
    "Lipinski", "FragmentComplexity", "WalkCount",

    # --- PROMOTED 2026-08-27: thin wrappers over primitives CORE already computes -----------
    # Marginal cost is the O(n) aggregation only; the primitive is already on the bill.
    "MoeType",    # 3: PEOE_VSA11 / SMR_VSA1 / EState_VSA1. mordred/MoeType.py resolves these
                  #    with getattr(rdkit.Chem.MolSurf | EState_VSA, name), so they are the
                  #    SAME code path -- and the same bin edges -- as the rdkit_* VSA columns.
    "SLogP",      # 1: literally `Crippen.MolLogP(mol)`. CORE already pays Crippen via Lipinski.
    "TopoPSA",    # 1: `rdMolDescriptors.CalcTPSA` plus an O(n) P/S contribution table.
    "CPSA",       # 2: RNCG/RPCG only. Both depend on AtomicCharge alone -- no surface area,
                  #    just Qmax/sum(q) over Gasteiger charges CORE already pays for via the
                  #    34 charge-weighted Autocorrelation columns. (The CPSA members that DO
                  #    need surface area, PNSA/PPSA/..., did not survive dedupe.)

    # --- STAGE 2, gated on ports that do not exist yet. Uncomment WITH the C++. --------------
    # "EState",   # 50: needs the 79-pattern E-state atom TYPER. The INDEX is already native.
    #             #     The 29 N* columns need only the typer, not the index.
    # "AcidBase", # 2: one fused recursive SMARTS each (4 and 6 alternatives).
    # "LogS",     # 1: 16 SMARTS counts + MolWt.
    # "Framework",# 1: SSSR + shortest paths over a ring-contracted graph.

    # --- Contribute ZERO surviving columns under data/dedupe.json (verified 2026-08-27) ------
    # Every member of each of these families was dropped by the r>0.99 dedupe, so listing them
    # is currently inert. They are KEPT rather than deleted because CORE_FAM answers "is this
    # family's primitive cheap?", which is independent of which columns survive dedupe: if the
    # dedupe is ever rerun at a different threshold, deleting these would silently demote them
    # to PREDICT. Do not read their presence as evidence that CORE computes anything for them.
    "Aromatic",                  # 2 defined
    "Weight",                    # 2 defined
    "DistanceMatrix",            # 12 defined
    "AdjacencyMatrix",           # 12 defined
    "DetourMatrix",              # 14 defined
    "BaryszMatrix",              # 104 defined
    "ExtendedTopochemicalAtom",  # 45 defined
    "ZagrebIndex",               # 4 defined
    "VertexAdjacencyInformation",   # 1 defined
    "EccentricConnectivityIndex",   # 1 defined
    "McGowanVolume",             # 1 defined
    "HydrogenBond",              # 2 defined
}


def classify(src: str, name: str, fam: dict) -> tuple[str, str]:
    """-> (block, family). block is 'core' or 'predict'."""
    if src == "mordred":
        f = fam.get(name, "?")
        # The mordred EState special-case survives, but NOT as the cost claim it used to be:
        # the old "242 us" was RDKit's Python EStateIndices, which the C++ port replaces. What
        # still blocks the family is the 79-SMARTS ATOM TYPER, which is unported. That is a
        # PORTING gap, not a cost gap -- different problem, different fix. Say which.
        if f == "EState":
            return "predict", "EState"          # unported atom typer, NOT an arithmetic cost
        return ("core", f) if f in CORE_FAM else ("predict", f)
    # PROMOTED 2026-08-27. EState_VSA / VSA_EState are binned sums over the EState index and
    # the Labute ASA contribs; Max/Min/MaxAbs/MinAbs EStateIndex are reductions over the same
    # vector. Reconstruction verified bit-exact -- see module docstring.
    if re.match(r"^(EState_VSA|VSA_EState)", name) or re.match(r"^(Max|Min).*EStateIndex", name):
        return "core", "rdkit_EState"
    # PROMOTED. SlogP_VSA / SMR_VSA bin the per-atom Crippen (logP, MR) contribs. MolLogP and
    # MolMR are the plain sums but over the H-ADDED molecule, so they additionally need the
    # Crippen H rows (H1-H4).
    if re.match(r"^(SlogP_VSA|SMR_VSA)", name) or name in ("MolLogP", "MolMR"):
        return "core", "rdkit_Crippen"
    # PROMOTED. PEOE_VSA bins the Gasteiger charges CORE already computes for Autocorrelation.
    # Reproduce RDKit's NaN handling: NaN is not clamped, it lands in the final bin.
    if re.match(r"^PEOE_VSA", name) or "PartialCharge" in name:
        return "core", "rdkit_Gasteiger"
    if name == "TPSA":
        return "core", "rdkit_TPSA"             # PROMOTED: rdMolDescriptors.CalcTPSA
    if name.startswith("BCUT2D"):
        # STAYS PREDICTED, and after this redraw it is the only RDKit family left with a
        # genuine cost argument rather than a porting gap: four eigensolves, and a heavy size
        # tail. Together with mordred InformationContent (33) it is most of what remains.
        return "predict", "rdkit_BCUT2D"
    if re.match(r"^Chi", name):
        return "core", "rdkit_Chi"              # see the CORE_FAM note on what is measured
    # SPLIT 2026-08-27: these two halves never belonged in one rule. Kappa1-3 + HallKierAlpha
    # need only HallKierAlpha plus the path counts CORE now computes anyway. Ipc/AvgIpc are
    # unrelated -- the information content of the characteristic polynomial's coefficients,
    # unported, and numerically delicate for large n. Only the Kappa half is promoted.
    if re.match(r"^(Kappa|HallKier)", name):
        return "core", "rdkit_Kappa"
    if re.match(r"^(Ipc|AvgIpc)", name):
        return "predict", "rdkit_Ipc"
    if name in ("qed", "SPS", "BertzCT"):
        return "predict", "rdkit_composite"
    return "core", "rdkit_core"


def survivors():
    d = json.load(open(DEDUPE))
    return [(s, n) for s, n, _ in d["compute"]] + [(s, n) for s, n, _ in d["predict"]]


def split(fam: dict):
    """-> {'core': [(src,name,family)], 'predict': [...]} over the 865 deduplicated columns.

    The two blocks are disjoint and exhaustive over the survivors, by construction and by
    assertion: every column is assigned exactly once, and nothing is dropped or duplicated.
    """
    surv = survivors()
    out = {"core": [], "predict": []}
    for s, n in surv:
        b, f = classify(s, n, fam)
        out[b].append((s, n, f))
    core, pred = {(s, n) for s, n, _ in out["core"]}, {(s, n) for s, n, _ in out["predict"]}
    assert not core & pred, f"CORE/PREDICT overlap: {sorted(core & pred)}"
    assert core | pred == set(surv), "split is not exhaustive over the survivors"
    assert len(out["core"]) + len(out["predict"]) == len(surv), "column count changed"
    return out
