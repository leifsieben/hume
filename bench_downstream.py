"""Figures B and C: does each representation help downstream, and at what cost?

    CHEMPFN_DATA_ROOT=~/chempfn-data python bench_downstream.py --arms ecfp hume --out out.json

WHAT THIS PRODUCES. One record per (dataset, fold, arm) in the same schema as the existing
results/figures/figB/grid_all.json, so new arms are DIRECTLY COMPARABLE with the base arms
already measured rather than forming a fresh, incomparable grid. Same 28 DEV datasets, same
5-fold scaffold split, same seed.

THE DATA COMES FROM THE CHEMPFN LAKE, read-only. `chemtfm` is a discontinued repo and its
vendored copy here resolves only 8 of the 28 datasets; the chempfn lake resolves all 28, every
one role DEV, so nothing LOCKED is touched and nothing needs downloading. The lake is 46 GB but
these 28 sets are 209 MB.

SCAFFOLD FOLDS, NOT RANDOM. A random split puts near-duplicates of a test molecule in train and
every representation looks good; the whole question here is whether a representation carries
information that transfers to an unseen scaffold. chempfn.eval.splits.scaffold_folds is used
rather than a local reimplementation so these numbers sit beside chempfn's own.

METRICS ARE SCALE-FREE AND ARE NOT POOLED ACROSS DATASETS IN NATIVE UNITS -- Spearman for
regression, ROC-AUC for binary, per chempfn.eval.protocol's rule. A mean of RMSE over datasets
whose units differ by two orders of magnitude is arithmetic on incommensurables.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

CHEMPFN = os.environ.get("CHEMPFN_SRC", "/Users/lsieben/VSCode/ChemPFN")
sys.path.insert(0, CHEMPFN)


def _quiet():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


# ------------------------------------------------------------------ featurisers
def f_ecfp(smis, radius=2, size=2048):
    """ECFP4 -- Morgan RADIUS 2, which is what `ecfp` means everywhere else in this project.

    This defaulted to radius=3 for one commit and the reproduction check caught it: `ecfp` on
    esol came out 1.089 against grid_all's 1.207, and on bioavail 0.696 against 0.675. Every
    new arm would have been silently incomparable with the base arms already in the grid, and
    the only symptom would have been a figure where the new arms all looked slightly different
    for no stated reason. arms.py labels this arm "ECFP4"; radius 2 is what ECFP4 is.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as fg
    g = fg.GetMorganGenerator(radius=radius, fpSize=size, includeChirality=True)
    out = np.zeros((len(smis), size), np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = g.GetCountFingerprintAsNumPy(m)
    return out


def f_r3cfp(smis):
    return f_ecfp(smis, radius=3)


def f_r4cfp(smis):
    return f_ecfp(smis, radius=4)


def f_rdkit_desc(smis):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fns = Descriptors._descList
    out = np.full((len(smis), len(fns)), np.nan, np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for j, (_n, f) in enumerate(fns):
            try:
                out[i, j] = f(m)
            except Exception:
                pass
    return out


MORDRED_PY = os.environ.get("MORDRED_PY", "")


def f_mordred_desc(smis):
    """Mordred, computed in a SEPARATE INTERPRETER, because it cannot share this one.

    mordred 1.2.0 requires numpy 1.x (it calls `np.float`, removed in numpy 1.24) while hume and
    torch here run on numpy 2.4.6. The two cannot coexist in one environment, so `MORDRED_PY`
    points at a second venv and the features cross as a .npy file. Without this the three
    Mordred-carrying arms would have died on ImportError -- after the box had already paid for
    every other arm.

    Set MORDRED_PY to run it; unset, this raises rather than silently returning a column of NaN
    that would look like a legitimately missing descriptor.
    """
    if not MORDRED_PY:
        raise RuntimeError(
            "MORDRED_PY is not set. Mordred needs its own interpreter (numpy 1.x) and this "
            "process is on numpy 2.x; point MORDRED_PY at a python that has mordred installed. "
            "Refusing to return NaN, which would be indistinguishable from a real missing value.")
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        smi_f, out_f = f"{td}/in.txt", f"{td}/out.npy"
        open(smi_f, "w").write("\n".join(smis))
        code = (
            "import sys, numpy as np\n"
            "from mordred import Calculator, descriptors as md\n"
            "from rdkit import Chem, RDLogger\n"
            "RDLogger.DisableLog('rdApp.*')\n"
            "smis = open(sys.argv[1]).read().split('\\n')\n"
            "calc = Calculator(md, ignore_3D=True)\n"
            "mols = [Chem.MolFromSmiles(s) for s in smis]\n"
            "ok = [m for m in mols if m is not None]\n"
            "rows = [[float(v) if isinstance(v,(int,float)) else np.nan for v in r]\n"
            "        for r in calc.map(ok, nproc=1, quiet=True)]\n"
            "vals = np.array(rows, np.float32)\n"
            "out = np.full((len(smis), vals.shape[1]), np.nan, np.float32)\n"
            "out[[i for i,m in enumerate(mols) if m is not None]] = vals\n"
            "np.save(sys.argv[2], out)\n")
        r = subprocess.run([MORDRED_PY, "-c", code, smi_f, out_f], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"mordred subprocess failed: {r.stderr.decode()[-400:]}")
        return np.load(out_f)


def f_hume(smis):
    """ECFP + every descriptor HUME computes, including our own. One arm, no variants."""
    import hume
    from rdkit import Chem
    mols, keep = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m); keep.append(i)
    fp, X, _ = hume.featurize_all_from_mols(mols, optional=("qed", "AvgIpc"))
    out = np.full((len(smis), X.shape[1] + fp.shape[1]), np.nan, np.float32)
    out[keep] = np.hstack([X, fp]).astype(np.float32)
    return out


def _cat(*fs):
    return lambda smis: np.hstack([f(smis) for f in fs])


def f_learned(kind):
    """ChemBERTa-2 / MiniMol / CheMeleon, whichever is asked for. Imported lazily: none of the
    three shares a dependency set with the others and a missing one must not kill the run."""
    def go(smis):
        import embed_pairs as EP
        return EP.ARMS[kind](smis)
    return go


#: Which columns of each arm are FINGERPRINT BITS, as (offset, width) or None.
#:
#: `feature_weights` needs this and nothing else does. It only has an effect on an arm that MIXES
#: sparse bits with dense descriptors -- for a pure fingerprint or a pure descriptor block every
#: weight is equal and the parameter is a no-op, so those arms skip the tuning entirely and cost
#: one fit instead of five.
#:
#: `hume` is the odd one: featurize_all returns (fp, X) and f_hume stacks them DESCRIPTORS FIRST,
#: so its bits are the LAST 2048 columns, not the first. Getting that backwards would weight the
#: descriptors by w and the bits by 1 -- the exact inverse of the intended prior, and completely
#: silent.
FP_BLOCK = {
    "ecfp": "all", "r3cfp": "all", "r4cfp": "all",
    "desc": None, "chemberta_mtr": None, "chemberta_mlm": None,
    "minimol": None, "chemeleon": None, "molformer": None,
    "ecfp_rdkit_desc": ("head", 2048), "ecfp_mordred_desc": ("head", 2048),
    "ecfp_all_desc": ("head", 2048), "hume": ("tail", 2048),
}

#: The `w` grid from API.md section 7. w=1 is XGBoost's uniform default and is dominated on BOTH
#: suites there (0.8080 vs 0.7882 activity, 0.8901 vs 0.8712 physchem), so it is kept in the grid
#: as a control rather than as a plausible answer.
W_GRID = (1.0, 5.0, 10.0, 100.0)


def fp_weights(arm, n_cols, w):
    """-> the feature_weights vector, or None when the arm has no fingerprint/descriptor split."""
    blk = FP_BLOCK.get(arm)
    if blk is None or blk == "all" or w == 1.0:
        return None
    where, width = blk
    v = np.ones(n_cols, dtype=np.float32)
    if where == "head":
        v[:width] = w
    else:
        v[-width:] = w
    return v


ARMS = {
    "ecfp":            f_ecfp,
    "r3cfp":           f_r3cfp,
    "r4cfp":           f_r4cfp,
    "ecfp_rdkit_desc": _cat(f_ecfp, f_rdkit_desc),
    "ecfp_mordred_desc": _cat(f_ecfp, f_mordred_desc),
    "ecfp_all_desc":   _cat(f_ecfp, f_rdkit_desc, f_mordred_desc),
    "desc":            _cat(f_rdkit_desc, f_mordred_desc),
    "hume":            f_hume,
    "chemberta_mtr":   f_learned("chemberta_mtr"),
    "chemberta_mlm":   f_learned("chemberta_mlm"),
    "minimol":         f_learned("minimol"),
    "chemeleon":       f_learned("chemeleon"),
    "molformer":       f_learned("molformer"),
}

DATASETS = ["ames", "aqsoldb", "bioavail", "cycpept_pampa", "cyp2d6_inh", "esol", "fartdb",
            "hia", "ld50_zhu", "lipophilicity", "pb_ames", "pb_bbb", "pb_cyp2c9", "pb_cyp2d6",
            "pb_cyp3a4", "pb_hum_mic_cl", "pb_logd", "pb_mou_mic_cl", "pb_ppb",
            "pb_rat_mic_cl", "pb_water_sol", "photoswitch", "qm8", "qm9", "qm9_gap",
            "qmugs_gap", "rascore", "vdss_lombardo"]


def load_ds(name):
    """Read a lake dataset with an EXPLICIT encoding, rather than the ambient locale's.

    chempfn's own `load_dataset` opens CSVs with a bare `open(newline="")`, which decodes using
    whatever `locale.getpreferredencoding()` returns. That is UTF-8 on this laptop and was
    something else on a cloud-init EC2 box, where the first dsA run died on
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3` reading a file that is
    byte-identical and valid UTF-8 here. Rather than depend on a box's locale -- or patch a repo
    that is not ours -- this reads the same files the same way everywhere, and replaces an
    undecodable byte instead of losing a whole dataset to one of them.

    Column names, task and file layout still come from chempfn's spec, so the dataset definition
    stays theirs and only the byte-level read is ours.
    """
    import csv as _csv
    from chempfn.data.lake import spec, _csv_files
    sp = spec(name)
    smis, ys = [], []
    for f in _csv_files(sp):
        with open(f, newline="", encoding="utf-8", errors="replace") as fh:
            for row in _csv.DictReader(fh):
                s_, y_ = row.get(sp.smiles_col), row.get(sp.label_col)
                if not s_ or y_ in (None, ""):
                    continue
                try:
                    ys.append(float(y_))
                except ValueError:
                    continue
                smis.append(s_)
    if not smis:
        raise RuntimeError(f"{name}: no usable rows from {[str(x) for x in _csv_files(sp)]}")
    return {"name": name, "smiles": smis, "y": ys, "task": sp.task}


def run(arm_names, datasets, out_path, folds_k=5, seed=0):
    _quiet()
    from chempfn.data.lake import spec, lake_root, _csv_files
    from chempfn.eval.splits import scaffold_folds, train_test
    from sklearn.metrics import roc_auc_score, accuracy_score
    import xgboost as xgb

    def make(task, fw=None):
        """Untuned XGBoost, per METHODS -- with the one documented exception.

        Everything about the head is left at a fixed default so a difference between arms is a
        difference between REPRESENTATIONS. The exception is `feature_weights`, which API.md
        section 7 recommends tuning: it sets how strongly column sampling favours fingerprint
        bits over descriptors, and the best value differs between potency and physicochemical
        endpoints.

        `colsample_bynode=0.3` IS PART OF THE MECHANISM, NOT A TUNING CHOICE. feature_weights is
        inert unless some colsample_by* < 1, so at XGBoost's default of 1.0 the whole parameter
        does nothing and would have been silently ignored. Per-node rather than per-tree because
        weights apply at each sampling event, which gives the reweighting far more chances to
        act.
        """
        cls = xgb.XGBClassifier if task in ("binary", "multiclass") else xgb.XGBRegressor
        kw = dict(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1, verbosity=0,
                  colsample_bynode=0.3, tree_method="hist")
        if fw is not None:
            kw["feature_weights"] = fw
        return cls(**kw)

    def score(task, yte, p):
        if task == "binary":
            return roc_auc_score(yte, p)
        if task == "multiclass":
            return accuracy_score(yte, np.rint(p).astype(int))
        return float(np.sqrt(np.mean((yte - p) ** 2)))

    def predict(m, task, X):
        return m.predict_proba(X)[:, 1] if task == "binary" else m.predict(X)

    records, t0 = [], time.time()
    for ds in datasets:
        sp = spec(ds)
        if sp.role != "DEV":
            print(f"  SKIP {ds}: role {sp.role}, not DEV -- refusing to touch it", flush=True)
            continue
        d = load_ds(ds)
        smis, y = list(d["smiles"]), np.asarray(d["y"], dtype=np.float64)
        # CAP AT 50,000, matching grid_all.json exactly. Four datasets exceed it (qm9 133,885;
        # qm9_gap 133,885; qmugs_gap 665,911; rascore 199,348) and every existing record for
        # them carries n = 50,000. A new arm measured on the full set would not be comparable
        # with the base arms already in the grid -- and Mordred on 665,911 molecules alone is
        # ~60 core-hours, so this is a correctness requirement that happens to also be cheap.
        if len(smis) > 50_000:
            take = np.random.default_rng(seed).choice(len(smis), 50_000, replace=False)
            take.sort()
            smis = [smis[i] for i in take]; y = y[take]
        folds = scaffold_folds(smis, k=folds_k, seed=seed)
        task = d["task"]
        feats = {}
        for a in arm_names:
            try:
                feats[a] = ARMS[a](smis)
            except Exception as e:
                print(f"  {ds}/{a}: FEATURISATION FAILED {type(e).__name__}: {e}", flush=True)
        for a, X in feats.items():
            for i in range(len(folds)):
                tr, te = train_test(folds, i)
                if task in ("binary", "multiclass") and len(np.unique(y[tr])) < 2:
                    continue
                if task == "regression" and float(np.std(y[tr])) == 0.0:
                    continue
                try:
                    # TUNE w ON TRAINING FOLDS ONLY. API.md: "Tuning it on the test fold leaks,
                    # and the effect is large enough to leak meaningfully." An inner 80/20 of the
                    # outer training set picks w; the final model is refit on the whole training
                    # set with the winner. Arms with no fingerprint/descriptor split skip this.
                    lower_better = task == "regression"
                    if FP_BLOCK.get(a) in (None, "all"):
                        best_w = 1.0
                    else:
                        cut = int(0.8 * len(tr))
                        itr, iva = tr[:cut], tr[cut:]
                        best_w, best_s = 1.0, None
                        for w in W_GRID:
                            if len(iva) < 20:
                                break
                            mi = make(task, fp_weights(a, X.shape[1], w))
                            mi.fit(X[itr], y[itr])
                            si = score(task, y[iva], predict(mi, task, X[iva]))
                            if best_s is None or (si < best_s if lower_better else si > best_s):
                                best_w, best_s = w, si
                    m = make(task, fp_weights(a, X.shape[1], best_w))
                    m.fit(X[tr], y[tr])
                    v = score(task, y[te], predict(m, task, X[te]))
                except Exception as e:
                    print(f"  {ds}/{a}/fold{i}: {type(e).__name__}: {e}", flush=True)
                    continue
                records.append({"dataset": ds, "fold": i, "arm": a,
                                "metric": {"binary": "auroc", "multiclass": "acc"}.get(
                                    task, "rmse"),
                                "value": float(v), "task": task, "n": len(smis),
                                "w": float(best_w)})
            got = [r["value"] for r in records if r["dataset"] == ds and r["arm"] == a]
            print(f"  {ds:<16s} {a:<20s} {np.mean(got):6.3f} over {len(got)} folds "
                  f"({time.time()-t0:6.0f}s)", flush=True)
        with open(out_path, "w") as fh:
            json.dump(records, fh)
    print(f"  -> {out_path}  {len(records)} records in {time.time()-t0:.0f}s", flush=True)
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad = [x for x in a.arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known: {sorted(ARMS)}")
    run(a.arms, a.datasets, a.out)
