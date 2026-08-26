"""Which proxy do we ship? Downstream, not reconstruction.

Reconstruction R² ranks pinet > mlp > gnn > linquad > ridge. That ordering has been wrong
before: Gate 1 measured a projection retaining 95% of descriptor variance and delivering 26%
of the downstream gain, and reconstruction has anti-correlated with downstream value three
separate times in this project. So the ladder is re-ranked here on what the descriptors are
actually for.

Arms, per benchmark dataset:

    ecfp                    reference
    ecfp+core               the shipped `fast` path, no predicted block
    ecfp+core+M             for each proxy M   <- the candidates
    ecfp+core+exact         true descriptors   <- the ceiling

`predicted_M` vs `exact` is the surrogate's real cost. `exact` vs `core` is what the predict
block is worth at all — if that gap is zero, no proxy is needed and the project simplifies.

Aggregation is by **rank and win-rate**, never a mean of raw RMSE across datasets whose scales
differ by 2-3x. Averaging FreeSolv (1.6-1.8) against Lipophilicity (0.68) silently weights the
former ~2.5x, which is how a wrong recommendation got made here once already.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"
MODELS = ["ridge", "linquad", "pinet", "mlp", "gnn"]


def main() -> None:
    import corpus_data
    Xb, Yb, smi_b, meta = corpus_data.load_bench()

    # Labels and dataset boundaries live in the legacy bench.npz; the new packed bench carries
    # only X/Y/smiles. Align on SMILES order rather than assuming it -- a silent misalignment
    # here would score every arm against the wrong labels.
    old = np.load(OUT / "bench.npz", allow_pickle=True)
    smi_old = list(old["smiles"])
    assert smi_b == smi_old, "bench1m and bench.npz SMILES order differ; cannot align labels"
    y, offsets, suite_of, name_of = old["y"], old["offsets"], old["suite_of"], old["name_of"]

    # The keep mask must come from the CORPUS, not be recomputed on the benchmark. Training
    # dropped columns that were constant *in the corpus*; the benchmark has a different set of
    # constants, so recomputing here yields a different width and silently misaligns every
    # target. Derived once from the corpus Y and cached.
    kf = OUT / "keep_targets.npz"
    if kf.exists():
        keep_t = np.load(kf)["keep"]
    else:
        import glob
        Yc = np.concatenate([np.load(f)["Y"] for f in
                             sorted(glob.glob(str(ROOT / "data/corpus1m/packed/pk_*.npz")))])
        keep_t, _ = corpus_data.drop_dead_targets(Yc, meta["ynames"], meta)
        np.savez_compressed(kf, keep=keep_t)
        del Yc
    print(f"corpus keep mask: {int(keep_t.sum())} of {keep_t.size} targets")
    prep = dict(np.load(OUT / "prep_blocks.npz", allow_pickle=True))
    Z = np.clip(Yb[:, keep_t][:, prep["keep"]].astype(np.float64), prep["lo"], prep["hi"])
    EXACT = np.nan_to_num(((Z - prep["mu"]) / prep["sd"]), nan=0.0).astype(np.float32)

    n_ecfp = 2048
    ECFP, CORE = Xb[:, :n_ecfp], Xb[:, n_ecfp:]
    print(f"bench {Xb.shape} | ecfp {ECFP.shape[1]} | core+blocks {CORE.shape[1]} | "
          f"predict {EXACT.shape[1]} | {len(name_of)} datasets")

    arms = {"ecfp": lambda s: ECFP[s],
            "ecfp+core": lambda s: Xb[s],
            "ecfp+core+exact": lambda s: np.hstack([Xb[s], EXACT[s]])}
    for m in MODELS:
        f = OUT / f"pred_bench_{m}.npz"
        if f.exists():
            P = np.load(f)["pred"].astype(np.float32)
            if P.shape[0] == Xb.shape[0] and P.shape[1] == EXACT.shape[1]:
                arms[f"ecfp+core+{m}"] = (lambda s, P=P: np.hstack([Xb[s], P[s]]))
            else:
                print(f"  (skip {m}: pred {P.shape} vs expected "
                      f"({Xb.shape[0]}, {EXACT.shape[1]}))")
        else:
            print(f"  (skip {m}: no predictions)")
    print(f"arms: {list(arms)}\n")

    from chemtfm.bench import metrics as M
    from chemtfm.bench.datasets import REGRESSION
    from chemtfm.bench.splits import scaffold_folds, train_test
    from chemtfm.models.xgb import XGBModel

    report, t0 = {}, time.time()
    for j, nm in enumerate(name_of):
        s = slice(offsets[j], offsets[j + 1])
        sm = list(np.array(smi_b)[s])
        folds = scaffold_folds(sm, k=5, seed=0)
        row = {}
        for a, build in arms.items():
            X, yy = build(s), y[s]
            vals = []
            for i in range(len(folds)):
                tr, te = train_test(folds, i)
                if float(np.std(yy[tr])) == 0.0:
                    continue
                vals.append(M.rmse(yy[te],
                                   XGBModel(task=REGRESSION).fit(X[tr], yy[tr]).predict(X[te])))
            row[a] = float(np.mean(vals)) if vals else np.nan
        row["_suite"], row["_n"] = str(suite_of[j]), int(offsets[j + 1] - offsets[j])
        report[str(nm)] = row
        json.dump(report, open(OUT / "pick_model.json", "w"), indent=2)
        base, ceil = row["ecfp+core"], row["ecfp+core+exact"]
        print(f"  [{j+1}/{len(name_of)}] {str(nm):22s} core {base:.4f} exact {ceil:.4f} "
              f"({100*(ceil-base)/base:+.1f}%)  ({time.time()-t0:.0f}s)", flush=True)

    names = list(arms)
    print(f"\n=== {len(report)} datasets ===")
    ranks = {a: [] for a in names}
    for row in report.values():
        for r, a in enumerate(sorted(names, key=lambda a: row[a])):
            ranks[a].append(r + 1)
    base_key = "ecfp+core"
    for a in sorted(names, key=lambda a: np.mean(ranks[a])):
        wins = sum(1 for row in report.values() if row[a] < row[base_key])
        d = [100 * (row[a] - row[base_key]) / row[base_key] for row in report.values()]
        print(f"  {a:24s} rank {np.mean(ranks[a]):4.2f}  vs core {np.mean(d):+6.2f}%  "
              f"beats core {wins:2d}/{len(report)}")

    span = [row["ecfp+core+exact"] - row["ecfp+core"] for row in report.values()]
    print(f"\nceiling gap (exact - core): mean {np.mean(span):+.4f}, "
          f"{sum(1 for x in span if x < 0)}/{len(span)} datasets where exact helps")
    for m in MODELS:
        k = f"ecfp+core+{m}"
        if k not in names:
            continue
        frac = [100 * (row[k] - row[base_key]) / (row["ecfp+core+exact"] - row[base_key])
                for row in report.values()
                if abs(row["ecfp+core+exact"] - row[base_key]) > 1e-6]
        print(f"  {m:8s} recovers {np.median(frac):6.1f}% of the ceiling (median)")
    print(f"\nwrote {OUT / 'pick_model.json'}")


if __name__ == "__main__":
    main()
