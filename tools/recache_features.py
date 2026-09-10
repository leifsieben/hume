"""Re-featurize the Figure C panel with the INSTALLED molhume, and stamp the version in.

The cache these replace was written by molhume 0.2.0. Two things changed since:
BalabanJ returned +inf on disconnected acyclic molecules (8 of 33 datasets carried inf cells,
masked to NaN downstream), and six ring-system columns were wrong. Everything else agrees to
float32 precision, so this is a targeted correction rather than a different feature set -- but a
cache with no version recorded cannot prove that, which is why the version is written here.
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import importlib.metadata as _md

sys.path.insert(0, "/Users/lsieben/VSCode/ChemPFN"); sys.path.insert(0, "."); sys.path.insert(0, "bench")
from rdkit import Chem, RDLogger  # noqa: E402
import molhume  # noqa: E402
RDLogger.DisableLog("rdApp.*"); warnings.simplefilter("ignore")
if "CHEMPFN_DATA_ROOT" not in os.environ:
    sys.exit("CHEMPFN_DATA_ROOT is not set; point it at the lake, e.g. /Users/lsieben/chempfn-data")
import bench_downstream as BD  # noqa: E402
from chempfn.eval.splits import scaffold_folds  # noqa: E402

OUT = Path("results/reanalysis/features"); OUT.mkdir(parents=True, exist_ok=True)
TASKS = {"physchem": ["aqsoldb","esol","lipophilicity","pb_logd","pb_water_sol","photoswitch"],
 "adme": ["pb_hum_mic_cl","pb_mou_mic_cl","pb_rat_mic_cl","pb_ppb","vdss_lombardo","ld50_zhu",
          "cycpept_pampa","pb_cyp2c9","pb_cyp2d6","pb_cyp3a4"],
 "classif": ["ames","pb_ames","cyp2d6_inh","bioavail","hia","pb_bbb","bace","hiv","herg",
             "wong_hepg2","wong_imr90","wong_hskmc","wong_saureus"],
 "quantum": ["qm8","qm9","qm9_gap","qmugs_gap"]}
names = list(molhume.feature_names(fingerprint=False, columns="full"))
VER = _md.version("mol-hume")
print(f"  molhume {VER}, {len(names)} columns\n", flush=True)

todo = sys.argv[1:] or [d for t in TASKS.values() for d in t]
for ds in todo:
    dest = OUT / f"{ds}.npz"
    if dest.exists():
        print(f"    {ds:16s} already cached", flush=True); continue
    try:
        d = BD.load_ds(ds)
        smis, y = list(d["smiles"]), np.asarray(d["y"], dtype=np.float64)
        # The SAME 50,000 cap and the SAME rng seed as the grid, or the rows are different molecules.
        if len(smis) > 50_000:
            take = np.sort(np.random.default_rng(0).choice(len(smis), 50_000, replace=False))
            smis = [smis[i] for i in take]; y = y[take]
        folds = scaffold_folds(smis, k=5, seed=0)
        fo = np.full(len(smis), -1, np.int8)
        for k, ix in enumerate(folds): fo[list(ix)] = k
        assert (fo >= 0).all(), f"{ds}: scaffold_folds did not cover every molecule"
        mols = [Chem.MolFromSmiles(s) for s in smis]
        keep = [i for i, m in enumerate(mols) if m is not None]
        errs = []
        fp, X, _ = molhume.featurize_all_from_mols([mols[i] for i in keep],
                                                   optional=("AvgIpc",), errors_out=errs)
        if errs:
            print(f"      {ds}: {len(errs)} row/column errors, first: {errs[0]}", flush=True)
        Xf = np.full((len(smis), X.shape[1]), np.nan, np.float32); Xf[keep] = X.astype(np.float32)
        fpf = np.zeros((len(smis), fp.shape[1]), np.uint8); fpf[keep] = fp
        ninf = int(np.isinf(X).sum())
        np.savez_compressed(dest, X=Xf, fp=fpf, y=y, folds=fo,
                            column_names=np.array(names), task=np.array(d["task"]),
                            metric=np.array(d.get("metric", "")), smiles=np.array(smis),
                            molhume=np.array(VER))
        print(f"    {ds:16s} n={len(smis):6d}  unparsed={len(smis)-len(keep):3d}  inf={ninf}",
              flush=True)
    except Exception as e:
        print(f"    ! {ds}: {type(e).__name__}: {e}", flush=True)
print("\n  done", flush=True)
