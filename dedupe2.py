"""Deduplicate the FULL descriptor space -- RDKit + Mordred + ours -- under one criterion.

    python dedupe2.py corpus     # build + save the stratified corpus
    python dedupe2.py compute    # compute all 2,023 columns on it   (the expensive step)
    python dedupe2.py analyse    # correlate, cover, and write the review reports

WHAT THIS FIXES relative to dedupe.py, each item measured rather than assumed:

1. OUR OWN COLUMNS ARE IN THE POOL. The 193 HUME columns with no upstream name had never been
   through the filter that defines the other 864. Audited separately (results/figures/
   own_column_audit.json): 24 are unusable and 64 are >=0.99 redundant, including `C4` == `n4Ring`
   and `C5_arom` == `n5aRing` at rho = 1.0000, and `T_absum` == `n_EZ_any` -- two of OUR columns
   identical to each other. One pool, one criterion, no exemptions.

2. NO IMPUTATION. dedupe.py median-imputed NaN before correlating. 4,099 column pairs share their
   missingness pattern at >=0.99, so both columns received their median on exactly the same rows
   and that manufactured agreement was counted as evidence of redundancy. Correlation is now
   PAIRWISE-COMPLETE, computed in closed form from four matrix products.

3. STRATIFIED CORPUS, AND THE MINIMUM ACROSS STRATA. Redundancy is not a property of a corpus
   average. Two columns are redundant only if they are redundant EVERYWHERE, so |rho| is computed
   within each size stratum and the pair is scored by its MINIMUM. A pair that collapses on
   drug-like space and separates on peptides is kept. This is the direct fix for the failure
   measured in dedupe_cost.py, where columns dropped at >=0.99 on a ChEMBL prefix turned out to be
   0.88-0.96 correlated on `bioavail` and worth +0.016 AUROC there.

4. THE THRESHOLD PRODUCES CANDIDATES, NOT DROPS (Leif 2026-08-30). A numerical |rho| >= 0.99 is a
   reason to LOOK at a pair, not to delete a column. Every candidate is written out with its
   partner, its per-stratum correlations, and whether the two come from the same family; the
   cross-family ones are the review queue, because that is where a numerical coincidence is most
   likely to be standing in for a mechanistic link that does not exist.

5. THE UNUSABLE GATE IS 50%, NOT 5%. The old rule predates the harness mapping non-finite to NaN,
   which XGBoost treats natively as missing. A column defined on 60% of molecules and informative
   there is usable now.

SURVIVOR CHOICE: the CHEAPEST member of each redundant group (Leif). Our own columns are computed
inside one C++ pass, so their marginal cost is ~0.1 us against Mordred's tens to hundreds, and
ours therefore win every true duplicate. The upstream name is recorded as an ALIAS on the
survivor so the column can still be reported as "Mordred's n4Ring, our implementation" rather
than introducing a new name for a quantity that already has one.

THE COVER IS NOT UNIQUE. Greedy in ascending cost is deterministic given the cost table and a
stable tie-break, but a different order yields a different, equally valid cover. Say so.
"""
from __future__ import annotations
import json, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "dedupe2"
OUT.mkdir(parents=True, exist_ok=True)
THRESH = 0.99
MIN_FINITE = 0.50
PER_STRATUM = 4000
STRATA = [(0, 15), (15, 25), (25, 35), (35, 55), (55, 10**6)]
CHEMBL = "/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/chembl_150k.smi"
LAKE = ["aqsoldb", "esol", "lipophilicity", "pb_logd", "photoswitch", "cycpept_pampa",
        "ld50_zhu", "vdss_lombardo", "ames", "bioavail", "hia", "pb_bbb", "qm8", "qm9",
        "qmugs_gap", "rascore"]


def build_corpus():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    random.seed(0)
    pool = []
    if os.path.exists(CHEMBL):
        pool += random.sample([l.split()[0] for l in open(CHEMBL) if l.strip()], 40000)
    try:
        sys.path.insert(0, str(ROOT)); import bench_downstream as BD
        for ds in LAKE:
            try: pool += BD.load_ds(ds)["smiles"]
            except Exception: pass
    except Exception as e:
        print(f"  lake unavailable ({type(e).__name__}); ChEMBL only")
    seen, uniq = set(), []
    for s in pool:
        if s not in seen: seen.add(s); uniq.append(s)
    buckets = {b: [] for b in STRATA}
    for s in uniq:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        a = m.GetNumHeavyAtoms()
        for lo, hi in STRATA:
            if lo <= a < hi: buckets[(lo, hi)].append(s); break
    corpus, strat = [], []
    for b in STRATA:
        take = random.sample(buckets[b], min(PER_STRATUM, len(buckets[b])))
        corpus += take; strat += [f"{b[0]}-{b[1]}"] * len(take)
        print(f"  stratum {b[0]:>3}-{b[1] if b[1] < 10**6 else 'inf':<4} heavy atoms: "
              f"{len(take):>5} of {len(buckets[b]):>7} available")
    json.dump({"smiles": corpus, "stratum": strat}, open(OUT / "corpus.json", "w"))
    print(f"  -> {OUT/'corpus.json'}  {len(corpus):,} molecules, {len(set(strat))} strata")


def compute():
    """Three blocks: RDKit 217, Mordred 1613 (own interpreter), ours 193 (hume)."""
    d = json.load(open(OUT / "corpus.json"))
    smis = d["smiles"]
    print(f"{len(smis):,} molecules")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in smis]

    t0 = time.time()
    rd_names = [n for n, _ in Descriptors._descList]
    lut = dict(Descriptors._descList)
    RD = np.full((len(mols), len(rd_names)), np.nan, np.float32)
    for i, m in enumerate(mols):
        if m is None: continue
        for j, n in enumerate(rd_names):
            try: RD[i, j] = lut[n](m)
            except Exception: pass
    print(f"  RDKit {RD.shape} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    from mordred import Calculator, descriptors as mdesc
    calc = Calculator(mdesc, ignore_3D=True)
    md_names = [str(x) for x in calc.descriptors]
    md_fam = [type(x).__module__.split(".")[-1] for x in calc.descriptors]
    ok = [m for m in mols if m is not None]
    MDv = np.asarray([[v if isinstance(v, (int, float)) else np.nan for v in r.values()]
                      for r in calc.map(ok, nproc=int(os.environ.get("NPROC", "10")), quiet=True)],
                     dtype=np.float32)
    MD = np.full((len(mols), len(md_names)), np.nan, np.float32)
    MD[[i for i, m in enumerate(mols) if m is not None]] = MDv
    print(f"  Mordred {MD.shape} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    import hume
    hcols = list(hume.ALL_COLUMNS)
    parts = []
    for lo in range(0, len(smis), 2000):
        _f, xb, _ = hume.featurize_all(smis[lo:lo+2000], optional=("qed", "AvgIpc"))
        parts.append(np.asarray(xb, np.float32)); del _f, xb
    H = np.vstack(parts); del parts
    union_lc = {n.lower() for n in rd_names} | {n.lower() for n in md_names}
    own_i = [i for i, c in enumerate(hcols) if c.lower() not in union_lc]
    OW = H[:, own_i]; own_names = [hcols[i] for i in own_i]
    print(f"  ours {OW.shape} ({time.time()-t0:.0f}s)")

    np.savez_compressed(OUT / "matrix.npz", RD=RD, MD=MD, OW=OW,
                        rd_names=np.array(rd_names, dtype=object),
                        md_names=np.array(md_names, dtype=object),
                        md_fam=np.array(md_fam, dtype=object),
                        own_names=np.array(own_names, dtype=object),
                        stratum=np.array(d["stratum"], dtype=object))
    print(f"  -> {OUT/'matrix.npz'}   total {RD.shape[1]+MD.shape[1]+OW.shape[1]} columns")


def pairwise_complete_corr(A, M):
    """Exact pairwise-complete Pearson, in closed form. A has NaN replaced by 0; M is the mask.

    No imputation anywhere: a pair's correlation uses exactly the rows on which BOTH columns are
    defined. dedupe.py median-imputed first, and 4,099 pairs share a missingness pattern, so the
    imputed rows agreed by construction and inflated |rho|.
    """
    n = M.T @ M
    Sx = A.T @ M
    Sy = Sx.T
    Sxx = (A * A).T @ M
    Syy = Sxx.T
    Sxy = A.T @ A
    with np.errstate(all="ignore"):
        num = n * Sxy - Sx * Sy
        den = np.sqrt(np.clip(n * Sxx - Sx * Sx, 0, None) *
                      np.clip(n * Syy - Sy * Sy, 0, None))
        r = np.where((den > 0) & (n >= 30), num / den, 0.0)
    return np.abs(r)


def analyse():
    z = np.load(OUT / "matrix.npz", allow_pickle=True)
    X = np.hstack([z["RD"], z["MD"], z["OW"]]).astype(np.float64)
    names = list(z["rd_names"]) + list(z["md_names"]) + list(z["own_names"])
    src = ["rdkit"] * len(z["rd_names"]) + ["mordred"] * len(z["md_names"]) + \
          ["hume"] * len(z["own_names"])
    fam = ["rdkit_" + n for n in z["rd_names"]] + ["mordred_" + f for f in z["md_fam"]] + \
          ["hume_own"] * len(z["own_names"])
    strat = np.array(z["stratum"]); levels = sorted(set(strat.tolist()))
    print(f"{X.shape[0]:,} molecules x {X.shape[1]:,} columns, {len(levels)} strata")

    fin = np.isfinite(X)
    with np.errstate(all="ignore"): sd = np.nanstd(X, 0)
    usable = (fin.mean(0) >= MIN_FINITE) & (sd > 0)
    idx = np.where(usable)[0]
    print(f"usable (>= {MIN_FINITE:.0%} finite, non-constant): {len(idx)} of {X.shape[1]} "
          f"(dropped {X.shape[1]-len(idx)})")

    # rank per column over its OWN finite rows, per stratum; NaN preserved
    Rmin = np.ones((len(idx), len(idx)))
    for lv in levels:
        rows = np.where(strat == lv)[0]
        Xs = X[np.ix_(rows, idx)]
        Ms = np.isfinite(Xs).astype(np.float64)
        Rk = np.zeros_like(Xs)
        for j in range(Xs.shape[1]):
            col = Xs[:, j]; m = np.isfinite(col)
            if m.sum() > 1:
                o = np.argsort(np.argsort(col[m]))
                Rk[m, j] = o.astype(np.float64)
        Rk = np.where(np.isfinite(Xs), Rk, 0.0)
        C = pairwise_complete_corr(Rk, Ms)
        np.fill_diagonal(C, 1.0)
        Rmin = np.minimum(Rmin, C)
        print(f"  stratum {lv:<10} n={len(rows):>5}  pairs >= {THRESH}: "
              f"{int((C >= THRESH).sum() - len(idx)) // 2:,}")
    np.fill_diagonal(Rmin, 0.0)
    print(f"pairs with MIN-across-strata |rho| >= {THRESH}: {int((Rmin >= THRESH).sum())//2:,}")

    cost = column_costs(names, src, z)
    order = sorted(range(len(idx)), key=lambda k: (cost[idx[k]], k))
    kept, cands = [], []
    for k in order:
        hit = None
        for h in kept:
            if Rmin[k, h] >= THRESH: hit = h; break
        if hit is None:
            kept.append(k)
        else:
            gi, hi = idx[k], idx[hit]
            cands.append({"dropped": names[gi], "kept": names[hi],
                          "min_rho": float(Rmin[k, hit]),
                          "dropped_src": src[gi], "kept_src": src[hi],
                          "dropped_fam": fam[gi], "kept_fam": fam[hi],
                          "same_family": fam[gi] == fam[hi],
                          "dropped_us": float(cost[gi]), "kept_us": float(cost[hi])})
    keep_names = [names[idx[k]] for k in kept]
    print(f"\nSURVIVORS: {len(kept)}   candidates absorbed: {len(cands)}")

    # alias: for each survivor, the upstream names it absorbed
    alias = {}
    for c in cands:
        if c["dropped_src"] in ("rdkit", "mordred"):
            alias.setdefault(c["kept"], []).append(c["dropped"])
    json.dump({"n_union": int(X.shape[1]), "n_usable": int(len(idx)), "n_kept": len(kept),
               "thresh": THRESH, "min_finite": MIN_FINITE, "strata": levels,
               "n_corpus": int(X.shape[0]),
               "kept": [{"name": names[idx[k]], "src": src[idx[k]], "fam": fam[idx[k]],
                         "us": float(cost[idx[k]]), "aliases": alias.get(names[idx[k]], [])}
                        for k in kept],
               "candidates": cands}, open(OUT / "dedupe2.json", "w"), indent=1)
    cross = sorted([c for c in cands if not c["same_family"]], key=lambda c: c["min_rho"])
    with open(OUT / "cross_family_review.md", "w") as fh:
        fh.write("# Cross-family dedup candidates, for mechanistic review\n\n")
        fh.write(f"{len(cands)} pairs cleared |rho| >= {THRESH} in EVERY size stratum. "
                 f"{len(cross)} of them pair columns from DIFFERENT families -- those are where a "
                 f"numerical coincidence is most likely to be standing in for a mechanism that is "
                 f"not there. Closest to the threshold first.\n\n")
        fh.write("| min rho | dropped | family | absorbed by | family |\n|---|---|---|---|---|\n")
        for c in cross:
            fh.write(f"| {c['min_rho']:.4f} | `{c['dropped']}` | {c['dropped_fam']} | "
                     f"`{c['kept']}` | {c['kept_fam']} |\n")
    print(f"  -> {OUT/'dedupe2.json'}")
    print(f"  -> {OUT/'cross_family_review.md'}   {len(cross)} cross-family pairs to review")


def column_costs(names, src, z):
    """us/mol per column. Ours are one C++ pass, so their MARGINAL cost is HUME's total spread
    over its columns -- ~0.1 us, which is why ours win every true duplicate under 'keep the
    fastest'. That is a real property of the implementation, not a thumb on the scale."""
    prof = json.load(open(ROOT / "data" / "budget_profile.json"))
    c = {}
    for d in prof["descriptors"]:
        c[(d.get("source", "rdkit"), d["name"])] = float(d.get("us", 50.0))
    fam_us = {k: float(v) for k, v in prof.get("mordred_family_us", {}).items()}
    md_fam = list(z["md_fam"]); md_names = list(z["md_names"])
    fm = dict(zip(md_names, md_fam))
    out = np.zeros(len(names))
    for i, (n, s) in enumerate(zip(names, src)):
        if s == "hume": out[i] = 124.0 / 1266.0
        elif s == "rdkit": out[i] = c.get(("rdkit", n), 50.0)
        else: out[i] = fam_us.get(fm.get(n, ""), 30.0)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    {"corpus": build_corpus, "compute": compute, "analyse": analyse}[cmd]()
