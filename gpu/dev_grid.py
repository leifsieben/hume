"""Figure B / C grid over the 28 scoreable DEV datasets.

Arms, all carrying ECFP except where the point is to isolate it:

    ecfp            ECFP4 r=2 counts                          the reference
    r3cfp           ECFP r=3 counts                           does radius help?
    desc            RDKit + Mordred alone, no fingerprint     the fingerprint-free control
    ecfp_desc       ECFP + full descriptors                   the POSITIVE CONTROL
    hume_exact      ECFP + core + blocks + true predict block the ceiling
    hume            ECFP + core + blocks + PREDICTED block    the product

`ecfp_desc` is load-bearing twice: it is Figure B panel 1's positive control, and it is the
number that says whether descriptors carry anything orthogonal to ECFP at all. If it is flat,
HUME has nothing to be a fast version of.

Output is LONG format -- (dataset, fold, arm, metric, value) -- which survives adding an arm
where a wide table does not. Folds are computed ONCE per dataset and shared by every arm, so
fold difficulty cancels in the paired differences and the error bars describe the lift rather
than the marginal spread of either side.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEV = HERE / "devsets"
OUT = HERE / "grid_out"


def scaffold_folds(smiles, k=5, seed=0):
    from collections import defaultdict
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    g = defaultdict(list)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        try:
            g[MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""].append(i)
        except Exception:
            g[""].append(i)
    groups = sorted(g.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = [[] for _ in range(k)]
    for grp in groups:                       # greedy: keep folds balanced, scaffolds intact
        folds[int(np.argmin([len(f) for f in folds]))].extend(grp)
    return [np.array(sorted(f)) for f in folds]


def featurize(smiles, workers=1):
    """-> dict arm -> (n, d) float32. One pass over the molecules for all arms."""
    import blocks as B
    import chi, conjugation, cycles, resistance, stereo
    from mordred import Calculator, descriptors as mdesc
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    g2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    g3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)
    full = Calculator(mdesc, ignore_3D=True)
    fam = {str(x): type(x).__module__.split(".")[-1] for x in full.descriptors}
    sp = B.split(fam)
    mnames = [str(x) for x in full.descriptors]
    rnames = [n for n, _ in Descriptors._descList]
    rlut = dict(Descriptors._descList)
    mpos = {n: i for i, n in enumerate(mnames)}
    rpos = {n: i for i, n in enumerate(rnames)}
    mods = {"resistance": resistance, "cycles": cycles, "conjugation": conjugation,
            "stereo": stereo, "chi": chi}

    n = len(smiles)
    E2 = np.zeros((n, 2048), np.float32)
    E3 = np.zeros((n, 2048), np.float32)
    MD = np.full((n, len(mnames)), np.nan, np.float32)
    RD = np.full((n, len(rnames)), np.nan, np.float32)
    BK = {k: np.full((n, m.NDIM), np.nan, np.float32) for k, m in mods.items()}
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        E2[i] = np.log1p(g2.GetCountFingerprintAsNumPy(m))
        E3[i] = np.log1p(g3.GetCountFingerprintAsNumPy(m))
        for j, nm in enumerate(rnames):
            try:
                RD[i, j] = rlut[nm](m)
            except Exception:
                pass
        try:
            MD[i] = np.array([v if isinstance(v, (int, float)) else np.nan for v in full(m)],
                             np.float32)
        except Exception:
            pass
        for k, mod in mods.items():
            try:
                BK[k][i] = mod.featurize(m)
            except Exception:
                pass

    def take(items):
        cols = []
        for src, nm, _ in items:
            if src == "mordred" and nm in mpos:
                cols.append(MD[:, mpos[nm]])
            elif src == "rdkit" and nm in rpos:
                cols.append(RD[:, rpos[nm]])
        return np.stack(cols, 1).astype(np.float32) if cols else np.zeros((n, 0), np.float32)

    CORE, PRED = take(sp["core"]), take(sp["predict"])
    BLK = np.hstack([BK[k] for k in ("resistance", "cycles", "conjugation", "stereo", "chi")])

    # Fixed arm definitions (2026-08-26). Each isolates ONE decision:
    #   ecfp_all_desc  vs ecfp        -> do descriptors help at all?
    #   ecfp_rdkit_desc / _mordred_   -> which library carries it?
    #   hume_core      vs ecfp_all_   -> what does the |rho|>=0.99 dedupe cost?
    #   hume_core_custom vs hume_core -> do our five new blocks add anything?
    #   *_predict      vs *_exact     -> what does the proxy cost?
    # `hume_core*` are the DEDUPLICATED 865-column set; `ecfp_all_desc` is the raw 1830.
    out = {
        "ecfp": E2,                    # Morgan r=2 counts, chirality on
        "r3cfp": E3,                   # Morgan r=3 counts -- direct radius comparison
        "ecfp_all_desc": np.hstack([E2, MD, RD]),
        "ecfp_rdkit_desc": np.hstack([E2, RD]),
        "ecfp_mordred_desc": np.hstack([E2, MD]),
        "hume_core": np.hstack([E2, CORE]),
        "hume_core_custom": np.hstack([E2, CORE, BLK]),
        # exact-descriptor counterparts, so the proxy cost is measurable against a ceiling
        "hume_core_exact": np.hstack([E2, CORE, PRED]),
        "hume_core_custom_exact": np.hstack([E2, CORE, BLK, PRED]),
        "_in_core": np.hstack([E2, CORE]),
        "_in_core_custom": np.hstack([E2, CORE, BLK]),
    }
    return out


def add_hume_predicted(F, ckpt_dir: Path, proxy: str = "pinet"):
    """Append the predicted arms. GATED: only runs when a proxy checkpoint is present.

    Deliberately separate from featurize() so the six proxy-free arms can be measured before
    the proxy is chosen. Those six answer questions that do not depend on it -- whether
    descriptors help, which library carries the signal, what the dedupe costs, whether our
    blocks add anything -- and none of them should wait on a model decision.
    """
    try:
        import torch
        import models
        xp = dict(np.load(ckpt_dir / "prep_x.npz", allow_pickle=True))
        prep = dict(np.load(ckpt_dir / "prep_blocks.npz", allow_pickle=True))
        c = torch.load(ckpt_dir / f"ckpt_{proxy}.pt", map_location="cpu", weights_only=False)
        net = models.PiNet.build(c["d"], c["t"])
        net.load_state_dict(c["state_dict"])
        net.eval()
        X = models.apply_xprep(F["_in_core_custom"].copy(), xp)
        with torch.no_grad():
            P = np.concatenate([net(torch.from_numpy(X[i:i + 4096])).numpy()
                                for i in range(0, len(X), 4096)])
        F[f"hume_core_predict_{proxy}"] = np.hstack([F["_in_core"], P]).astype(np.float32)
        F[f"hume_core_custom_predict_{proxy}"] = np.hstack([F["_in_core_custom"], P]).astype(np.float32)
    except Exception as e:
        print(f"    (no hume arm: {type(e).__name__}: {e})", flush=True)
    return F


def score(Xtr, ytr, Xte, yte, task):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, roc_auc_score
    common = dict(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=2,
                  tree_method="hist", verbosity=0)
    if task == "regression":
        m = xgb.XGBRegressor(**common).fit(Xtr, ytr)
        return "rmse", float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    m = xgb.XGBClassifier(**common, eval_metric="logloss").fit(Xtr, ytr.astype(int))
    p = m.predict_proba(Xte)
    if task == "binary":
        return "auroc", float(roc_auc_score(yte.astype(int), p[:, 1]))
    return "acc", float((p.argmax(1) == yte.astype(int)).mean())


def clean(M):
    C = M.astype(np.float64)
    med = np.nanmedian(C, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    C = np.where(np.isfinite(C), C, med)
    lo, hi = np.nanpercentile(C, 1, 0), np.nanpercentile(C, 99, 0)
    return np.clip(C, lo, hi).astype(np.float32)


def main():
    only = sys.argv[1:] or None
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.load(open(DEV / "manifest.json"))
    names = [n for n in sorted(man) if not only or n in only]
    print(f"{len(names)} datasets", flush=True)
    rows = []
    for nm in names:
        f = OUT / f"{nm}.json"
        if f.exists():
            rows.extend(json.load(open(f)))
            print(f"  {nm}: cached", flush=True)
            continue
        z = np.load(DEV / f"{nm}.npz", allow_pickle=True)
        smi, y, task = list(z["smiles"]), z["y"], str(z["task"])
        t0 = time.time()
        F = featurize(smi)
        if (HERE / "PROXY").exists():
            F = add_hume_predicted(F, HERE, (HERE / "PROXY").read_text().strip())
        F = {k: clean(v) for k, v in F.items() if not k.startswith("_")}
        folds = scaffold_folds(smi)
        out = []
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(smi)), te)
            if task != "regression" and len(np.unique(y[tr])) < 2:
                continue
            for arm, M in F.items():
                try:
                    met, val = score(M[tr], y[tr], M[te], y[te], task)
                    out.append({"dataset": nm, "fold": fi, "arm": arm,
                                "metric": met, "value": val, "task": task, "n": len(smi)})
                except Exception as e:
                    print(f"    {nm} f{fi} {arm}: {type(e).__name__}", flush=True)
        json.dump(out, open(f, "w"))
        rows.extend(out)
        print(f"  {nm:18s} {task:11s} n={len(smi):6,} arms={len(F)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(rows, open(OUT / "grid_all.json", "w"))
    print(f"\n{len(rows)} rows -> {OUT / 'grid_all.json'}")


if __name__ == "__main__":
    main()
