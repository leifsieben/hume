"""Deduplicate the full RDKit + Mordred descriptor space, preferring cheap survivors.

Implements the design rule directly: among descriptors that carry the same information,
keep the one that is fastest to compute. Doing it as a single criterion avoids the trap of
spending budget computing something that duplicates a descriptor we predict anyway.

Procedure
  1. compute all 217 RDKit + 1613 Mordred (2D) descriptors on a diverse sample
  2. drop columns that are constant or >5% undefined  (they cannot carry signal)
  3. rank-transform, so Pearson on ranks == Spearman (monotone-redundancy, not just linear)
  4. greedy cover in ASCENDING COST order: keep a descriptor unless it correlates
     |rho| >= THRESH with one already kept. Cost order is what makes the survivor the cheap
     member of each redundant group.
  5. split survivors at the time budget: computed vs predicted

3D Mordred descriptors are excluded throughout — they need conformers (~140 ms/mol), which
is two orders of magnitude outside any budget contemplated here.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
# WHERE THE CORRELATIONS ARE ESTIMATED, and it matters more than the threshold does.
#
# This was the FIRST 20,000 lines of chembl_150k.smi. Two problems, both measured:
#   * A PREFIX, NOT A SAMPLE. The file is not shuffled: its first 20k average MW 416.1 against
#     438.9 for a random 20k of the same file, and median 386.8 against 413.3 -- a 6% bias
#     toward smaller molecules, from nothing but where the read stopped.
#   * ChEMBL ONLY. The benchmark this feeds spans QM9 (MW ~120) and CycPeptMPDB cyclic peptides
#     at the other extreme. A redundancy that holds across drug-like ChEMBL space need not hold
#     on either, and `dedupe_cost.py` measured exactly that: columns dropped at |rho| >= 0.99
#     here are only 0.88-0.96 correlated on `bioavail`.
#
# Now: a stride over the whole ChEMBL file, plus the benchmark lake's own molecules. Using the
# benchmark molecules is legitimate and not leakage -- this selection is UNSUPERVISED, it never
# sees a label, and the descriptors should be deduplicated on the chemistry they will actually
# be applied to.
CORPUS = "/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/chembl_150k.smi"
LAKE_DATASETS = ["aqsoldb", "esol", "lipophilicity", "pb_logd", "photoswitch", "cycpept_pampa",
                 "ld50_zhu", "vdss_lombardo", "ames", "bioavail", "hia", "pb_bbb",
                 "qm8", "qm9", "qmugs_gap", "rascore"]
OUT = ROOT / "data" / "dedupe.json"
N_MOL = 20000
THRESH = 0.99    # near-exact duplicates only, not mere correlation
BUDGET_US = 330.0  # 10x the measured 33 us ECFP cost


def costs():
    """us/mol per descriptor: RDKit measured individually, Mordred per-family / n."""
    c = {}
    prof = json.load(open(ROOT / "data" / "budget_profile.json"))
    for d in prof["descriptors"]:
        c[("rdkit", d["name"])] = d["us"]
    for f in json.load(open(ROOT / "data" / "mordred_families.json")):
        c[("mordred_family", f["family"])] = f["us"] / f["n"]
    return c


def main() -> None:
    from mordred import Calculator, descriptors as mdesc
    from multiprocessing import Pool

    import random
    allc = [l.split()[0] for l in open(CORPUS) if l.strip()]
    random.seed(0)
    smis = random.sample(allc, min(N_MOL, len(allc)))
    n_chembl = len(smis)
    # plus the benchmark chemistry, capped per dataset so one large set cannot dominate
    try:
        import bench_downstream as BD
        for ds in LAKE_DATASETS:
            try:
                v = BD.load_ds(ds)["smiles"]
            except Exception as e:
                print(f"  (lake {ds}: {type(e).__name__}, skipped)")
                continue
            smis += random.sample(v, min(1500, len(v)))
    except Exception as e:
        print(f"  (lake unavailable: {type(e).__name__}; ChEMBL only)")
    print(f"corpus: {n_chembl:,} ChEMBL (random) + {len(smis)-n_chembl:,} benchmark = {len(smis):,}")
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    print(f"{len(mols):,} molecules")
    MAT = ROOT / "data" / "dedupe_matrix.npz"

    rd_names = [n for n, _ in Descriptors._descList]
    lut = dict(Descriptors._descList)

    def rd_row(m):
        r = np.empty(len(rd_names), np.float32)
        for j, n in enumerate(rd_names):
            try:
                r[j] = lut[n](m)
            except Exception:
                r[j] = np.nan
        return r

    if MAT.exists():
        z = np.load(MAT, allow_pickle=True)
        RD, MD, md_fam = z["RD"], z["MD"], list(z["md_fam"])
        md_names = list(z["md_names"])
        print(f"reusing cached matrices RD{RD.shape} MD{MD.shape}")
        _skip = True
    else:
        _skip = False
    t0 = time.time()
    RD = RD if _skip else np.stack([rd_row(m) for m in mols])
    print(f"RDKit {RD.shape} ({time.time() - t0:.0f}s)")

    calc = Calculator(mdesc, ignore_3D=True)
    if not _skip:
        md_names = [str(d) for d in calc.descriptors]
        md_fam = [type(d).__module__.split(".")[-1] for d in calc.descriptors]
        t0 = time.time()
        MD = np.asarray([[v if isinstance(v, (int, float)) else np.nan for v in r.values()]
                         for r in calc.map(mols, nproc=10, quiet=True)], dtype=np.float32)
        print(f"Mordred {MD.shape} ({time.time() - t0:.0f}s)")
        np.savez_compressed(MAT, RD=RD, MD=MD, md_fam=np.array(md_fam, dtype=object),
                            md_names=np.array(md_names, dtype=object))

    X = np.hstack([RD, MD])
    src = [("rdkit", n) for n in rd_names] + [("mordred", n) for n in md_names]
    C = costs()
    cost = np.array([C.get(("rdkit", n), 50.0) if s == "rdkit"
                     else C.get(("mordred_family", md_fam[i - len(rd_names)]), 30.0)
                     for i, (s, n) in enumerate(src)])
    print(f"union: {X.shape[1]} descriptors ({len(rd_names)} RDKit + {len(md_names)} Mordred)")

    # --- step 2: drop unusable columns -------------------------------------------------
    finite = np.isfinite(X).mean(axis=0)
    with np.errstate(all="ignore"):
        sd = np.nanstd(X, axis=0)
    usable = (finite >= 0.95) & (sd > 0)
    print(f"usable (>=95% finite, non-constant): {usable.sum()}  "
          f"(dropped {(~usable).sum()}: {(finite < 0.95).sum()} sparse, {(sd <= 0).sum()} constant)")

    idx = np.where(usable)[0]
    Xu = X[:, idx]
    Xu = np.where(np.isfinite(Xu), Xu, np.nan)
    # median-impute so ranking is well defined, then rank-transform for Spearman
    med = np.nanmedian(Xu, axis=0)
    Xu = np.where(np.isnan(Xu), med, Xu)
    R = np.argsort(np.argsort(Xu, axis=0), axis=0).astype(np.float32)
    R -= R.mean(0)
    R /= np.linalg.norm(R, axis=0) + 1e-12
    print("rank-transformed; computing correlation matrix...")
    M = np.abs(R.T @ R)  # |Spearman|
    print(f"correlation matrix {M.shape}")

    # --- step 4: greedy cover, cheapest first -------------------------------------------
    order = idx[np.argsort(cost[idx], kind="stable")]
    pos = {g: k for k, g in enumerate(idx)}

    def cover(th, record=False):
        """-> kept, and (when asked) the (dropped, kept-partner, |rho|) triple for every drop.

        THE PARTNER IS THE WHOLE POINT OF RECORDING IT. `|rho| >= 0.99` is a numerical claim;
        whether the pair is MECHANISTICALLY the same quantity is a chemical one, and only the
        second justifies calling a column redundant. Without the partner written down, nobody can
        check the second -- which is how `MDEO-12` (a distance-edge descriptor over oxygens) and
        `nO` (a count of oxygens) end up indistinguishable from `ETA_dBeta` and whatever happened
        to correlate with it on 20k prefix molecules.
        """
        kept, drops = [], []
        for g in order:
            k = pos[g]
            hit = None
            for h in kept:
                r = M[k, pos[h]]
                if r >= th:
                    hit = (h, float(r)); break
            if hit is None:
                kept.append(g)
            elif record:
                drops.append([src[g][1], src[hit[0]][1], hit[1]])
        kept = sorted(kept, key=lambda g: cost[g])
        return (kept, drops) if record else kept

    print(f"\n{'|rho| >=':>10s}{'survivors':>11s}{'absorbed':>10s}   interpretation")
    note = {1.0: "bitwise-identical only", 0.9999: "numerically identical",
            0.999: "near-exact duplicates", 0.99: "very tight",
            0.98: "tight", 0.95: "empirically correlated (too loose)"}
    results = {}
    for th in (1.0, 0.9999, 0.999, 0.99, 0.98, 0.95):
        k = cover(th)
        results[th] = k
        print(f"{th:10.4f}{len(k):11d}{int(usable.sum())-len(k):10d}   {note[th]}")
    kept, drops = cover(THRESH, record=True)
    kept = sorted(kept, key=lambda g: cost[g])
    drops.sort(key=lambda d: d[2])   # closest to the threshold first: the ones worth reviewing

    # --- step 5: budget split ------------------------------------------------------------
    cum, compute, predict = 0.0, [], []
    for g in kept:
        if cum + cost[g] <= BUDGET_US:
            compute.append(g)
            cum += cost[g]
        else:
            predict.append(g)
    n_rd_c = sum(1 for g in compute if src[g][0] == "rdkit")
    n_rd_p = sum(1 for g in predict if src[g][0] == "rdkit")
    pred_cost = float(cost[predict].sum())

    print(f"\nCOMPUTE : {len(compute):5d} descriptors, {cum:8.1f} us/mol "
          f"({n_rd_c} RDKit + {len(compute) - n_rd_c} Mordred)")
    print(f"PREDICT : {len(predict):5d} descriptors, {pred_cost / 1000:8.1f} ms/mol avoided "
          f"({n_rd_p} RDKit + {len(predict) - n_rd_p} Mordred)")
    print(f"\nEMBEDDING = 2048 ECFP + {len(compute)} computed + {len(predict)} predicted "
          f"= {2048 + len(kept)} dims")
    print(f"  dense (non-ECFP) portion: {len(kept)} dims")

    json.dump({"n_union": int(X.shape[1]), "n_usable": int(usable.sum()),
               "n_kept": len(kept), "thresh": THRESH, "budget_us": BUDGET_US,
               "compute": [list(src[g]) + [float(cost[g])] for g in compute],
               "predict": [list(src[g]) + [float(cost[g])] for g in predict],
               "compute_us": cum, "predict_us": pred_cost,
               "n_corpus": int(X.shape[0]),
               "drops": drops}, open(OUT, "w"), indent=2)
    print(f"recorded {len(drops)} (dropped, kept-partner, |rho|) triples")
    print("\nthe 15 drops CLOSEST to the threshold -- review these for a mechanistic link:")
    for a, b, r in drops[:15]:
        print(f"   {r:.4f}  {a:<22} absorbed by  {b}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
