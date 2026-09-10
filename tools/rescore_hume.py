"""Re-score every HUME arm on the cached matrices, with the SAME head as the rest of the grid.

TWO THINGS WERE WRONG WITH THE RECORDS THIS REPLACES.

1. The head. The local re-score that produced the HUME arms built its own XGBoost and never
   passed `feature_weights`. Every other arm in Figure C went through `bench_downstream`, which
   tunes that multiplier on the training folds -- the anchor `ecfp_all_desc` chose w=100 on 74 of
   its 165 folds. So the HUME arms were compared, at w=1, against an anchor that had tuned. This
   script imports `bench_downstream.fit_fold` rather than reimplementing it, which is the only
   way to make "the same head" checkable instead of asserted.

2. The features. The cache was written by molhume 0.2.0, before BalabanJ stopped returning +inf
   on disconnected acyclic molecules (8 of 33 datasets carried inf cells) and before six
   ring-system columns were corrected.

Records are written in the grid's own schema, with `proto` and `w` stamped, so they merge with
the AWS records on (dataset, arm, fold) exactly as the ones they supersede did.
"""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import importlib.metadata as _md

sys.path.insert(0, "/Users/lsieben/VSCode/ChemPFN"); sys.path.insert(0, "."); sys.path.insert(0, "bench")
warnings.simplefilter("ignore")
import molhume  # noqa: E402
import bench_downstream as BD  # noqa: E402

FEAT = Path("results/reanalysis/features")
OUT = Path("results/reanalysis/hume_rescore_1_0_0.json")
NAMES = list(molhume.feature_names(fingerprint=False, columns="full"))
TASKS = {"physchem": ["aqsoldb","esol","lipophilicity","pb_logd","pb_water_sol","photoswitch"],
 "adme": ["pb_hum_mic_cl","pb_mou_mic_cl","pb_rat_mic_cl","pb_ppb","vdss_lombardo","ld50_zhu",
          "cycpept_pampa","pb_cyp2c9","pb_cyp2d6","pb_cyp3a4"],
 "classif": ["ames","pb_ames","cyp2d6_inh","bioavail","hia","pb_bbb","bace","hiv","herg",
             "wong_hepg2","wong_imr90","wong_hskmc","wong_saureus"],
 "quantum": ["qm8","qm9","qm9_gap","qmugs_gap"]}
DATASETS = [d for t in TASKS.values() for d in t]

# The arms, as COLUMN MASKS over the full 1,269.
#
# `hume_no_new` IS THE SHIPPED `full_no_new`, NOT results/dedupe2/new_columns.json. The two are
# different ablations and the figure was labelled with the wrong one. dedupe2's file lists the 185
# columns WIRED LATE in development -- mostly Mordred standards like AETA_* and BCUT* -- and
# masking it leaves 1,084. The shipped set masks the 160 columns that are OURS (the charge
# autocorrelations, the ring-system counts, Kf, Cyclicity, RATSC) and leaves 1,109, which is what
# the legend has always claimed the arm is: "everything RDKit or Mordred already defines". The two
# exclusion lists are nearly disjoint, so this is not a rounding difference in the count -- it was
# a different ablation under the right label.
ARM_COLS = {
    "hume":        set(NAMES),
    "hume_no_new": set(molhume.feature_names(fingerprint=False, columns="full_no_new")),
    "hume_622":    set(molhume.column_set("minimal-v2")),
    "hume_408":    set(molhume.column_set("small-v1")),
    "hume_256":    set(molhume.column_set("minimal-v3")),
}
for a, cs in ARM_COLS.items():
    missing = cs - set(NAMES)
    if missing:
        sys.exit(f"{a}: {len(missing)} columns are not in this build's output, e.g. "
                 f"{sorted(missing)[:5]}. The spec and the build disagree; re-derive rather "
                 f"than silently truncating.")
ARM_IDX = {a: np.array([i for i, c in enumerate(NAMES) if c in cs]) for a, cs in ARM_COLS.items()}
print("  arms: " + ", ".join(f"{a} ({len(i)})" for a, i in ARM_IDX.items()), flush=True)
print(f"  head: bench_downstream.fit_fold, proto={BD.PROTO}, "
      f"w grid {BD.W_GRID}, inner K={BD.W_INNER_K}\n", flush=True)

records = json.loads(OUT.read_text()) if OUT.exists() else []
done = {(r["dataset"], r["arm"]) for r in records}
t0 = time.time()
for ds in DATASETS:
    f = FEAT / f"{ds}.npz"
    if not f.exists():
        print(f"  ! {ds}: not cached", flush=True); continue
    z = np.load(f, allow_pickle=False)
    X0, fp, y, folds = z["X"], z["fp"], z["y"], z["folds"]
    task = str(z["task"])
    ver = str(z["molhume"]) if "molhume" in z else "?"
    # An unparsed molecule is an ALL-NaN descriptor row; its fingerprint must be NaN too, or the
    # row silently becomes "a molecule with no substructures" rather than a missing one. That is
    # what `_hume_block` does on the AWS side, and the two must agree row for row.
    dead = ~np.isfinite(X0).any(axis=1)
    fpn = fp.astype(np.float32)
    fpn[dead] = np.nan
    n_bad = int((~np.isfinite(X0)).sum() - np.isnan(X0).sum())
    if n_bad:
        print(f"    {ds}: {n_bad} non-finite descriptor cells -> NaN", flush=True)
    Xn = np.where(np.isfinite(X0), X0, np.nan)
    for arm, idx in ARM_IDX.items():
        if (ds, arm) in done:
            continue
        F = np.hstack([Xn[:, idx], fpn]).astype(np.float32)
        vals, ws = [], []
        for i in range(5):
            te = folds == i; tr = ~te
            tr_i, te_i = np.where(tr)[0], np.where(te)[0]
            if task in ("binary", "multiclass") and len(np.unique(y[tr_i])) < 2:
                continue
            if task == "regression" and float(np.std(y[tr_i])) == 0.0:
                continue
            try:
                v, w = BD.fit_fold(arm, task, F, y, tr_i, te_i, seed=0)
            except Exception as e:
                print(f"    ! {ds}/{arm}/fold{i}: {type(e).__name__}: {e}", flush=True); continue
            vals.append(v); ws.append(w)
            records.append({"proto": BD.PROTO, "dataset": ds, "fold": i, "arm": arm,
                            "metric": {"binary": "auroc", "multiclass": "acc"}.get(task, "rmse"),
                            "value": float(v), "task": task, "n": int(len(y)),
                            "w": float(w), "molhume": ver, "src": "local-rescore-1.0.0"})
        OUT.write_text(json.dumps(records))
        print(f"    {ds:<16s} {arm:<12s} {np.mean(vals):7.4f}  w={sorted(set(ws))}  "
              f"({time.time()-t0:6.0f}s)", flush=True)
print(f"\n  -> {OUT}  {len(records)} records in {time.time()-t0:.0f}s", flush=True)
