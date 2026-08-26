"""Does the core descriptor block lower the degree, or just add columns?

Claim under test: feeding the 639 core descriptors alongside ECFP is not "extra information
smuggled in" but *degree reduction* — the core block contains already-aggregated quadratic
forms, so a low-degree model over them reaches functions that would be high-degree in the raw
atom properties.

The claim makes a specific, falsifiable prediction about WHICH targets improve:

  * Chi / PathCount / MolecularId  -> improve sharply. These count paths; the core block
    contains walk counts (from A^1..A^8), and paths relate to walks by inclusion-exclusion
    over lower-order terms. That is a short algebraic step from core, and a long one from a
    radius-2 hash.
  * MolLogP / MolMR / TPSA         -> barely move. These are sums of per-atom-type
    contributions, i.e. already degree-1 in atom counts, which ECFP encodes directly. There
    is no degree to reduce.

If instead everything improves uniformly, the core block is just adding capacity/information
and the degree-reduction story is wrong.

Ridge only: closed form, seconds, and the Gram matrix is shared across all targets.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = "/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/chembl_150k.smi"
MAT = ROOT / "data" / "dedupe_matrix.npz"
DEDUPE = ROOT / "data" / "dedupe.json"
N_MOL = 20000
LAMBDA = 10.0

CORE_FAM = {
    "RingCount", "Aromatic", "AtomCount", "BondCount", "Constitutional", "CarbonTypes",
    "Weight", "Autocorrelation", "DistanceMatrix", "AdjacencyMatrix", "TopologicalCharge",
    "WienerIndex", "BalabanJ", "TopologicalIndex", "MolecularDistanceEdge",
    "ExtendedTopochemicalAtom", "ZagrebIndex", "KappaShapeIndex",
    "VertexAdjacencyInformation", "EccentricConnectivityIndex", "McGowanVolume",
    "VdwVolumeABC", "Polarizability", "ABCIndex", "RotatableBond", "HydrogenBond",
    "Lipinski", "FragmentComplexity", "BaryszMatrix", "WalkCount", "DetourMatrix",
}


def classify(src, name, fam):
    if src == "mordred":
        f = fam.get(name, "?")
        if f == "EState":
            return "estate", f
        return ("core", f) if f in CORE_FAM else ("predict", f)
    if re.match(r"^(EState_VSA|VSA_EState)", name) or re.match(r"^(Max|Min).*EStateIndex", name):
        return "estate", "rdkit_EState"
    if re.match(r"^(SlogP_VSA|SMR_VSA)", name) or name in ("MolLogP", "MolMR"):
        return "predict", "rdkit_Crippen"
    if re.match(r"^PEOE_VSA", name) or "PartialCharge" in name:
        return "predict", "rdkit_Gasteiger"
    if name == "TPSA":
        return "predict", "rdkit_TPSA"
    if name.startswith("BCUT2D"):
        return "predict", "rdkit_BCUT2D"
    if re.match(r"^(Chi|Kappa|HallKier|Ipc|AvgIpc)", name):
        return "predict", "rdkit_Chi_Kappa"
    if name in ("qed", "SPS", "BertzCT"):
        return "predict", "rdkit_composite"
    return "core", "rdkit_core"


def ridge_r2(Xtr, Ytr, Xte, Yte, lam):
    """Closed-form multi-output ridge; one Gram solve serves every target."""
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float64)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float64)])
    G = Xtr.T @ Xtr
    G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, Xtr.T @ Ytr)
    P = Xte @ W
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def main() -> None:
    from mordred import Calculator, descriptors as mdesc
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    md_desc = list(Calculator(mdesc, ignore_3D=True).descriptors)
    md_pos = {str(d): i for i, d in enumerate(md_desc)}
    fam = {str(d): type(d).__module__.split(".")[-1] for d in md_desc}
    rd_names = [n for n, _ in Descriptors._descList]
    rd_pos = {n: i for i, n in enumerate(rd_names)}

    z = np.load(MAT, allow_pickle=True)
    RD, MD = z["RD"].astype(np.float64), z["MD"].astype(np.float64)
    smis = []
    with open(CORPUS) as fh:
        for line in fh:
            s = line.split()[0] if line.strip() else ""
            if s:
                smis.append(s)
            if len(smis) >= N_MOL:
                break
    mols = [Chem.MolFromSmiles(s) for s in smis]
    keep_mol = [i for i, m in enumerate(mols) if m is not None][:len(RD)]
    mols = [mols[i] for i in keep_mol]
    print(f"{len(mols):,} molecules; RD{RD.shape} MD{MD.shape}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    E = np.log1p(np.stack([gen.GetCountFingerprintAsNumPy(m).astype(np.float64) for m in mols]))

    d = json.load(open(DEDUPE))
    surv = [(s, n) for s, n, _ in d["compute"]] + [(s, n) for s, n, _ in d["predict"]]
    cols = defaultdict(list)   # group -> list of (global column, family)
    for s, n in surv:
        g, f = classify(s, n, fam)
        if s == "rdkit" and n in rd_pos:
            cols[g].append((RD[:, rd_pos[n]], f))
        elif s == "mordred" and n in md_pos:
            cols[g].append((MD[:, md_pos[n]], f))

    CORE = np.stack([c for c, _ in cols["core"]], axis=1)
    TGT = np.stack([c for c, _ in cols["predict"]], axis=1)
    tgt_fam = np.array([f for _, f in cols["predict"]])
    print(f"core block {CORE.shape} | predict targets {TGT.shape}")

    def clean(A):
        A = np.where(np.isfinite(A), A, np.nan)
        med = np.nanmedian(A, axis=0)
        A = np.where(np.isnan(A), med, A)
        lo, hi = np.nanpercentile(A, 1, axis=0), np.nanpercentile(A, 99, axis=0)
        A = np.clip(A, lo, hi)
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        return (A - mu) / sd

    CORE, TGT = clean(CORE), clean(TGT)

    groups = defaultdict(list)
    for i, m in enumerate(mols):
        try:
            groups[MurckoScaffold.MurckoScaffoldSmiles(mol=m)].append(i)
        except Exception:
            groups[""].append(i)
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    rng = np.random.default_rng(0)
    rng.shuffle(keys)
    tr, te = [], []
    for k in keys:
        (tr if len(tr) < int(0.85 * len(mols)) else te).extend(groups[k])
    tr, te = np.array(tr), np.array(te)
    print(f"scaffold split: {len(tr):,} train / {len(te):,} test\n")

    t0 = time.time()
    r2_e = ridge_r2(E[tr], TGT[tr], E[te], TGT[te], LAMBDA)
    r2_ec = ridge_r2(np.hstack([E, CORE])[tr], TGT[tr], np.hstack([E, CORE])[te], TGT[te], LAMBDA)
    print(f"ridge fits done ({time.time() - t0:.0f}s)")
    print(f"\noverall median R2:  ECFP {np.median(r2_e):.4f}  ->  ECFP+core {np.median(r2_ec):.4f}"
          f"   (delta {np.median(r2_ec) - np.median(r2_e):+.4f})\n")

    print(f"{'target family':22s}{'n':>5s}{'ECFP':>9s}{'ECFP+core':>11s}{'delta':>9s}")
    rows = []
    for f in sorted(set(tgt_fam)):
        m = tgt_fam == f
        a, b = float(np.median(r2_e[m])), float(np.median(r2_ec[m]))
        rows.append((b - a, f, int(m.sum()), a, b))
    for delta, f, n, a, b in sorted(rows, reverse=True):
        print(f"{f:22s}{n:5d}{a:9.3f}{b:11.3f}{delta:+9.3f}")

    json.dump({"median_ecfp": float(np.median(r2_e)), "median_ecfp_core": float(np.median(r2_ec)),
               "families": {f: {"n": n, "ecfp": a, "ecfp_core": b} for _, f, n, a, b in rows}},
              open(ROOT / "data" / "ridge_input_test.json", "w"), indent=2)


if __name__ == "__main__":
    main()
