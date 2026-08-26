"""Shared definition of the CORE / PREDICT descriptor split.

Single source of truth so the training data, the models and the downstream evaluation cannot
drift apart. The split is by *primitive cost*, not by downstream benefit:

    CORE     639 descriptors reachable from ring info + adjacency + atom properties +
             distance matrix + Labute ASA          -> 59 us total, computed exactly
    PREDICT  226 descriptors needing EState (242 us), Crippen (90 us), eigenvalues (40 us),
             Gasteiger (17 us), TPSA (9 us), or path enumeration (no cheap primitive)
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
    # Python, not paths being expensive. Measured C++ (cpp/bench.cpp): 6.82 us for bounded
    # path enumeration to k<=7. Chi is also the highest-value predicted family at +0.126
    # downstream, so this removes the largest single source of surrogate error.
    "Chi", "PathCount",
    "RingCount", "Aromatic", "AtomCount", "BondCount", "Constitutional", "CarbonTypes",
    "Weight", "Autocorrelation", "DistanceMatrix", "AdjacencyMatrix", "TopologicalCharge",
    "WienerIndex", "BalabanJ", "TopologicalIndex", "MolecularDistanceEdge",
    "ExtendedTopochemicalAtom", "ZagrebIndex", "KappaShapeIndex",
    "VertexAdjacencyInformation", "EccentricConnectivityIndex", "McGowanVolume",
    "VdwVolumeABC", "Polarizability", "ABCIndex", "RotatableBond", "HydrogenBond",
    "Lipinski", "FragmentComplexity", "BaryszMatrix", "WalkCount", "DetourMatrix",
}


def classify(src: str, name: str, fam: dict) -> tuple[str, str]:
    """-> (block, family). block is 'core' or 'predict'."""
    if src == "mordred":
        f = fam.get(name, "?")
        if f == "EState":
            return "predict", "EState"          # 242 us: too expensive, surrogate handles it
        return ("core", f) if f in CORE_FAM else ("predict", f)
    if re.match(r"^(EState_VSA|VSA_EState)", name) or re.match(r"^(Max|Min).*EStateIndex", name):
        return "predict", "rdkit_EState"
    if re.match(r"^(SlogP_VSA|SMR_VSA)", name) or name in ("MolLogP", "MolMR"):
        return "predict", "rdkit_Crippen"
    if re.match(r"^PEOE_VSA", name) or "PartialCharge" in name:
        return "predict", "rdkit_Gasteiger"
    if name == "TPSA":
        return "predict", "rdkit_TPSA"
    if name.startswith("BCUT2D"):
        return "predict", "rdkit_BCUT2D"
    if re.match(r"^Chi", name):
        return "core", "rdkit_Chi"          # 6.82 us measured in C++; see CORE_FAM note
    if re.match(r"^(Kappa|HallKier|Ipc|AvgIpc)", name):
        # Kappa needs only HallKierAlpha (0.22 us in RDKit's C++) plus path counts we now
        # compute anyway, so it is a likely follow-up promotion -- but unmeasured, so it stays
        # in PREDICT until someone times it. Ipc/AvgIpc are information-content, unrelated.
        return "predict", "rdkit_Kappa_Ipc"
    if name in ("qed", "SPS", "BertzCT"):
        return "predict", "rdkit_composite"
    return "core", "rdkit_core"


def survivors():
    d = json.load(open(DEDUPE))
    return [(s, n) for s, n, _ in d["compute"]] + [(s, n) for s, n, _ in d["predict"]]


def split(fam: dict):
    """-> {'core': [(src,name,family)], 'predict': [...]} over the 865 deduplicated columns."""
    out = {"core": [], "predict": []}
    for s, n in survivors():
        b, f = classify(s, n, fam)
        out[b].append((s, n, f))
    return out
