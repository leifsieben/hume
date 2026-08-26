"""Score the surrogate where it counts: downstream RMSE on the locked benchmark.

Gate 1's lesson was that fidelity metrics mislead — a projection retaining 95% of the
descriptor *variance* delivered 26% of the downstream *gain*. So R^2 stays diagnostics and
this script is the verdict.

Arms (scaffold 5-fold CV, untuned XGBoost — identical protocol to validate_mordred_cherry.py,
whose numbers this reproduces):

  desc                      true RDKit-96              976 us/mol    known: 0.8399 ACE / 1.0196 net
  desc+mordred_true         + raw Mordred           68,231 us/mol    known: 0.8054 ACE / 0.9634 net
  desc+mordred_true_prep    + preprocessed Mordred  68,231 us/mol    the REAL ceiling (86-89% of gain)
  desc+mordred_pred         + predicted Mordred        976 us/mol    hybrid fallback
  pred_union                predicted everything        29 us/mol    THE PRODUCT

``desc+mordred_true_prep`` is the arm the surrogate should be compared against, not
``mordred_true``: the model is trained on preprocessed targets, so preprocessing loss is
conceded before it makes a single error. Comparing to the raw arm would understate it.

``pred_union`` uses nothing computed but ECFP — no RDKit-96, no ErG, no Mordred. That is the
artefact being proposed, and its cost is 29 us/mol against 68,333.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
SURR = ROOT / "data" / "surrogate"


def _cv_rmse(X, y, smiles) -> float:
    from chemtfm.bench import metrics as M
    from chemtfm.bench.datasets import REGRESSION
    from chemtfm.bench.splits import scaffold_folds, train_test
    from chemtfm.models.xgb import XGBModel

    folds = scaffold_folds(smiles, k=5, seed=0)
    out = []
    for i in range(len(folds)):
        tr, te = train_test(folds, i)
        if float(np.std(y[tr])) == 0.0:
            continue
        pred = XGBModel(task=REGRESSION).fit(X[tr], y[tr]).predict(X[te])
        out.append(M.rmse(y[te], pred))
    return float(np.mean(out)) if out else np.nan


def ecfp_for(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.zeros(2048, np.float32) if m is None
                    else gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
    # Same transform as training: ECFP counts are heavy-tailed.
    return np.log1p(np.stack(rows))


def main() -> None:
    import torch

    c = dict(np.load(CACHE, allow_pickle=True))
    smiles, y, offsets = c["smiles"], c["y"], c["offsets"]
    suite_of, name_of = c["suite_of"], c["name_of"]
    d_true, md_true = c["d_bench"], c["md_bench"]

    ck = torch.load(SURR / "surrogate.pt", map_location="cpu", weights_only=False)
    prep = dict(np.load(SURR / "prep.npz", allow_pickle=True))
    bounds = ck["bounds"]

    print("featurising benchmark molecules (ECFP only)...")
    X = ecfp_for(list(smiles))
    if ck["with_rdkit96"]:
        X = np.hstack([X, np.nan_to_num(d_true)])

    import torch.nn as nn
    n_out = int(prep["keep"].sum())  # target width is exactly the retained-column count
    net = nn.Sequential(
        nn.Linear(ck["in_dim"], 2048), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(2048, 1024), nn.GELU(),
        nn.Linear(1024, n_out))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(X)).numpy()
    print(f"predicted {pred.shape} descriptor columns for {len(smiles):,} molecules")

    # Slice the prediction back into blocks. `keep` indexes the concatenated raw target space,
    # so map each retained column to the block it came from.
    kept = np.where(prep["keep"])[0]

    def block(name):
        s, e = bounds[name]
        return pred[:, (kept >= s) & (kept < e)]

    # The honest ceiling: TRUE descriptors put through the training-time preprocessing. The
    # surrogate concedes preprocessing loss before it errs at all, so comparing it against the
    # raw-Mordred arm would understate it. Needs ErG for the benchmark molecules, which the
    # Gate 1 cache does not hold — it is 102 us/mol, so just compute it.
    from rdkit import Chem
    from rdkit.Chem import rdReducedGraphs
    n_erg = bounds["erg"][1] - bounds["erg"][0]
    erg_true = np.stack([
        np.asarray(rdReducedGraphs.GetErGFingerprint(m), np.float32)
        if (m := Chem.MolFromSmiles(s)) is not None else np.full(n_erg, np.nan, np.float32)
        for s in smiles])
    raw_true = np.hstack([d_true, erg_true, md_true])
    Z = np.clip(raw_true[:, prep["keep"]].astype(np.float64), prep["lo"], prep["hi"])
    true_prep = np.nan_to_num((Z - prep["mu"]) / prep["sd"], nan=0.0).astype(np.float32)

    md_pred = block("mordred")
    arms = {
        "desc": lambda s: d_true[s],
        "desc+mordred_true": lambda s: np.hstack([d_true[s], md_true[s]]),
        "true_union_prep": lambda s: true_prep[s],
        "desc+mordred_pred": lambda s: np.hstack([d_true[s], md_pred[s]]),
        "pred_union": lambda s: pred[s],
    }

    per_ds, t0 = {}, time.time()
    for j, name in enumerate(name_of):
        s = slice(offsets[j], offsets[j + 1])
        smi_j, y_j = list(smiles[s]), y[s]
        per_ds[name] = {"suite": suite_of[j]}
        for arm, build in arms.items():
            per_ds[name][arm] = _cv_rmse(build(s), y_j, smi_j)
        print(f"  [{suite_of[j]}] {name}: "
              + "  ".join(f"{a}={per_ds[name][a]:.3f}" for a in arms)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for suite in ("moleculeace", "moleculenet"):
        rows = [v for v in per_ds.values() if v["suite"] == suite]
        summary[suite] = {a: float(np.nanmean([r[a] for r in rows])) for a in arms}
        summary[suite]["n_datasets"] = len(rows)

    (SURR / "eval_report.json").write_text(
        json.dumps({"summary": summary, "per_dataset": per_ds}, indent=2))

    print("\n=== surrogate downstream (RMSE, lower better) ===")
    for suite, r in summary.items():
        base, ceil = r["desc"], r["desc+mordred_true"]
        print(f"{suite} (n={int(r['n_datasets'])}):  baseline {base:.4f} -> ceiling {ceil:.4f}")
        for arm in arms:
            gain = r[arm] - base
            frac = 100 * gain / (ceil - base) if ceil != base else float("nan")
            print(f"   {arm:20s} {r[arm]:.4f}  gain {gain:+.4f}  = {frac:4.0f}% of ceiling")
    print(f"\nwrote {SURR / 'eval_report.json'}")


if __name__ == "__main__":
    main()
