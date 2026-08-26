"""Which Mordred families pay for themselves? Signal per millisecond, no model involved.

Everything so far has tried to get Mordred's value more cheaply by learning it. The profile
suggests a blunter route: cost and signal are distributed very unevenly across families, and
not in the same way.

    Autocorrelation   606 descriptors,  6.6% of runtime,  11 of the cherry-pick's top 30
    Chi                56 descriptors, 18.9% of runtime
    PathCount          21 descriptors, 13.7% of runtime

The prior cherry-pick selected 30 *individual* descriptors and failed because the signal is
diffuse. But diffuse-within-a-family is exactly what taking whole families fixes: you keep
all 606 autocorrelations rather than the 11 that happened to rank highest.

So: for each family, measure downstream gain over the RDKit-96 baseline, and divide by its
measured cost. Families above the cost/benefit knee get computed exactly; the rest get
dropped. No surrogate, no fidelity threshold, no architecture.

Baseline desc = 0.8399, full Mordred = 0.8054 on the 30 MoleculeACE regression sets.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
OUT = Path(__file__).resolve().parent / "data" / "family_ablation.json"

# us/mol, measured single-core on this machine (see FINDINGS.md).
COST = {
    "Chi": 13742, "PathCount": 9984, "InformationContent": 6008, "MolecularId": 5495,
    "BCUT": 5457, "ExtendedTopochemicalAtom": 5395, "Autocorrelation": 4823,
    "BaryszMatrix": 3476, "MolecularDistanceEdge": 3016, "EState": 2683,
    "DetourMatrix": 1602, "MoeType": 772, "KappaShapeIndex": 706, "BertzCT": 592,
}
MIN_COLS = 8  # families too small to move a 30-dataset mean are not worth a run


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
        out.append(M.rmse(y[te], XGBModel(task=REGRESSION).fit(X[tr], y[tr]).predict(X[te])))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    from mordred import Calculator, descriptors as mdesc

    fam = np.array([type(d).__module__.split(".")[-1]
                    for d in Calculator(mdesc, ignore_3D=True).descriptors])

    c = dict(np.load(CACHE, allow_pickle=True))
    smiles, y, offsets = c["smiles"], c["y"], c["offsets"]
    suite_of, name_of = c["suite_of"], c["name_of"]
    desc, mordred = c["d_bench"], c["md_bench"]

    groups = defaultdict(list)
    for i, f in enumerate(fam):
        groups[f].append(i)
    todo = {f: np.array(ix) for f, ix in groups.items() if len(ix) >= MIN_COLS}
    print(f"{len(todo)} families with >={MIN_COLS} descriptors")

    arms = {"desc": lambda s: desc[s], "desc+all": lambda s: np.hstack([desc[s], mordred[s]])}
    for f, ix in sorted(todo.items()):
        arms[f"desc+{f}"] = (lambda s, ix=ix: np.hstack([desc[s], mordred[s][:, ix]]))

    per_ds, t0 = {}, time.time()
    for j, name in enumerate(name_of):
        if suite_of[j] != "moleculeace":
            continue
        s = slice(offsets[j], offsets[j + 1])
        smi_j, y_j = list(smiles[s]), y[s]
        per_ds[name] = {a: _cv_rmse(b(s), y_j, smi_j) for a, b in arms.items()}
        print(f"  {name} done ({time.time() - t0:.0f}s)", flush=True)

    summary = {a: float(np.nanmean([r[a] for r in per_ds.values()])) for a in arms}
    OUT.write_text(json.dumps({"summary": summary, "per_dataset": per_ds, "cost": COST},
                              indent=2))

    base, full = summary["desc"], summary["desc+all"]
    rows = []
    for f, ix in todo.items():
        g = summary[f"desc+{f}"] - base
        cost_ms = COST.get(f, np.nan) / 1000.0
        rows.append((g / cost_ms if cost_ms == cost_ms and cost_ms > 0 else np.nan,
                     f, len(ix), summary[f"desc+{f}"], g, cost_ms))
    rows.sort(key=lambda r: (r[0] if r[0] == r[0] else 0))

    print(f"\n=== signal per millisecond (n={len(per_ds)}) ===")
    print(f"baseline desc {base:.4f} | full Mordred {full:.4f} (gain {full - base:+.4f}, 72.7 ms)")
    print(f"{'family':26s}{'n':>5s}{'RMSE':>9s}{'gain':>9s}{'ms':>8s}{'gain/ms':>10s}")
    for eff, f, n, rmse, g, ms in rows:
        print(f"{f:26s}{n:5d}{rmse:9.4f}{g:+9.4f}{ms:8.1f}{eff:10.4f}")


if __name__ == "__main__":
    main()
