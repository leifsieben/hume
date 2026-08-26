"""Does the EState block earn its 242 us?

EState indices are the most expensive primitive in the suite — 242 us/mol, roughly half the
entire primitive budget and 4x the whole CORE tier (59 us). They unlock 73 of the 865
deduplicated descriptors. Every other tier boundary is obvious from the cost/yield ratio;
this one is not, so measure it.

Arms (on top of ECFP, which is always present in production):

    ecfp                2048          29 us
    ecfp + core         2048 + 639    59 us of primitives
    ecfp + core + estate 2048 + 712   301 us of primitives   <- is the delta worth 242 us?

Run at two `colsample_bytree` settings. The default 0.8 matches every earlier result in this
project and keeps them comparable; 0.3 guards against the p>>n regime penalising the wider
arm by construction, which would manufacture a negative result for EState purely because it
adds columns.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
DEDUPE = Path(__file__).resolve().parent / "data" / "dedupe.json"
OUT = Path(__file__).resolve().parent / "data" / "estate_ablation.json"

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
    """Which primitive does this descriptor need? Mirrors the mapping in PRIMITIVES.md."""
    if src == "mordred":
        f = fam.get(name, "?")
        if f == "EState":
            return "estate"
        return "core" if f in CORE_FAM else "other"
    if re.match(r"^(EState_VSA|VSA_EState)", name) or re.match(r"^(Max|Min).*EStateIndex", name):
        return "estate"
    if (re.match(r"^(SlogP_VSA|SMR_VSA|PEOE_VSA)", name) or name in ("MolLogP", "MolMR", "TPSA")
            or "PartialCharge" in name or name.startswith("BCUT2D")
            or re.match(r"^(Chi|Kappa|HallKier|Ipc|AvgIpc)", name)
            or name in ("qed", "SPS", "BertzCT")):
        return "other"
    return "core"


def _cv_rmse(X, y, smiles, colsample):
    from chemtfm.bench import metrics as M
    from chemtfm.bench.datasets import REGRESSION
    from chemtfm.bench.splits import scaffold_folds, train_test
    from chemtfm.models.xgb import XGBModel, _DEFAULT_PARAMS

    p = dict(_DEFAULT_PARAMS)
    p["colsample_bytree"] = colsample
    folds = scaffold_folds(smiles, k=5, seed=0)
    out = []
    for i in range(len(folds)):
        tr, te = train_test(folds, i)
        if float(np.std(y[tr])) == 0.0:
            continue
        mdl = XGBModel(task=REGRESSION, params=p).fit(X[tr], y[tr])
        out.append(M.rmse(y[te], mdl.predict(X[te])))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    from mordred import Calculator, descriptors as mdesc
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    md_desc = list(Calculator(mdesc, ignore_3D=True).descriptors)
    md_pos = {str(d): i for i, d in enumerate(md_desc)}
    fam = {str(d): type(d).__module__.split(".")[-1] for d in md_desc}
    rd_names = [n for n, _ in Descriptors._descList]
    rd_pos = {n: i for i, n in enumerate(rd_names)}
    lut = dict(Descriptors._descList)

    d = json.load(open(DEDUPE))
    surv = [(s, n) for s, n, _ in d["compute"]] + [(s, n) for s, n, _ in d["predict"]]
    groups = {"core": {"rdkit": [], "mordred": []}, "estate": {"rdkit": [], "mordred": []}}
    for s, n in surv:
        g = classify(s, n, fam)
        if g in groups:
            groups[g][s].append(n)
    print("deduplicated survivors by primitive:")
    for g in groups:
        print(f"  {g:8s} {len(groups[g]['rdkit']):4d} RDKit + {len(groups[g]['mordred']):4d} Mordred"
              f" = {len(groups[g]['rdkit']) + len(groups[g]['mordred'])}")

    c = dict(np.load(CACHE, allow_pickle=True))
    smiles, y, offsets = c["smiles"], c["y"], c["offsets"]
    suite_of, name_of = c["suite_of"], c["name_of"]
    MD = c["md_bench"]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    t0 = time.time()
    ecfp, RD = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            ecfp.append(np.zeros(2048, np.float32))
            RD.append(np.full(len(rd_names), np.nan, np.float32))
            continue
        ecfp.append(gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
        row = np.empty(len(rd_names), np.float32)
        for j, n in enumerate(rd_names):
            try:
                row[j] = lut[n](m)
            except Exception:
                row[j] = np.nan
        RD.append(row)
        if (i + 1) % 15000 == 0:
            print(f"  featurised {i + 1:,} ({time.time() - t0:.0f}s)", flush=True)
    ecfp = np.stack(ecfp)
    RD = np.stack(RD)

    def block(g):
        cols = [RD[:, rd_pos[n]] for n in groups[g]["rdkit"] if n in rd_pos]
        cols += [MD[:, md_pos[n]] for n in groups[g]["mordred"] if n in md_pos]
        return np.stack(cols, axis=1) if cols else np.zeros((len(smiles), 0), np.float32)

    CORE, ES = block("core"), block("estate")
    print(f"\nECFP {ecfp.shape} | core {CORE.shape} | estate {ES.shape}")

    arms = {"ecfp": lambda s: ecfp[s],
            "ecfp+core": lambda s: np.hstack([ecfp[s], CORE[s]]),
            "ecfp+core+estate": lambda s: np.hstack([ecfp[s], CORE[s], ES[s]])}

    report = {}
    for colsample in (0.8, 0.3):
        for suite in ("moleculeace", "moleculenet"):
            per = {}
            for j, nm in enumerate(name_of):
                if suite_of[j] != suite:
                    continue
                s = slice(offsets[j], offsets[j + 1])
                per[nm] = {a: _cv_rmse(b(s), y[s], list(smiles[s]), colsample)
                           for a, b in arms.items()}
                print(f"  [cs={colsample}] {nm}: "
                      + " ".join(f"{a.replace('ecfp+', '')}={per[nm][a]:.3f}" for a in arms)
                      + f"  ({time.time() - t0:.0f}s)", flush=True)
            summ = {a: float(np.nanmean([r[a] for r in per.values()])) for a in arms}
            report[f"{suite}_cs{colsample}"] = {"summary": summ, "per_dataset": per}
            base = summ["ecfp+core"]
            delta = summ["ecfp+core+estate"] - base
            wins = sum(1 for r in per.values() if r["ecfp+core+estate"] < r["ecfp+core"])
            print(f"\n=== {suite} colsample={colsample} (n={len(per)}) ===")
            for a in arms:
                print(f"   {a:20s} {summ[a]:.4f}")
            print(f"   EState delta vs core: {delta:+.4f}   helps on {wins}/{len(per)}\n")
            OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
