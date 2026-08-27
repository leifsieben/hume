"""How accurate must a descriptor surrogate be before its predictions carry the signal?

This is the assumption underneath the whole surrogate direction and it has never been
measured. Known: a surrogate at median R^2 0.949 delivers 2% of Mordred's downstream gain.
Unknown: whether 0.99 delivers 40% or 4%.

Rather than train better surrogates and see, degrade the TRUE descriptors to a target R^2 by
adding calibrated Gaussian noise, and measure downstream gain at each level. That isolates
fidelity from every other property of a surrogate — no model, no training, no architecture.

For a standardised column, adding noise of variance s^2 gives R^2 = 1/(1+s^2) against the
truth, so s = sqrt(1/R^2 - 1).

Interpretation:
  * gain collapses only below ~0.99  -> a graph model plausibly clears the bar; build the GNN
  * gain collapses even at 0.999     -> no surrogate can ever work; the direction is closed
  * gain degrades gracefully         -> our 2% is a model problem, not a structural one,
                                        and better inputs/architecture should recover it

The noise is a best case: it is isotropic and independent of structure, whereas a real
surrogate's error is systematic and correlated with exactly the molecules it finds hard. So
this curve is an UPPER bound on what a surrogate of the same R^2 could achieve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
OUT = Path(__file__).resolve().parent / "data" / "noise_threshold.json"
LEVELS = [1.0, 0.999, 0.99, 0.97, 0.95, 0.90]
SEED = 0


def _cv_rmse(X, y, smiles) -> float:
    import _vendor  # noqa: F401  - puts vendor/chemtfm on sys.path
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
    c = dict(np.load(CACHE, allow_pickle=True))
    smiles, y, offsets = c["smiles"], c["y"], c["offsets"]
    suite_of, name_of = c["suite_of"], c["name_of"]
    desc, mordred = c["d_bench"], c["md_bench"]

    # Standardise Mordred so the noise scale is interpretable per column, dropping columns
    # that are constant or mostly undefined (same filter as everywhere else).
    keep = (np.isfinite(mordred).mean(axis=0) >= 0.95) & (np.nanstd(mordred, axis=0) > 0)
    Z = mordred[:, keep].astype(np.float64)
    lo, hi = np.nanpercentile(Z, 1, axis=0), np.nanpercentile(Z, 99, axis=0)
    Z = np.clip(Z, lo, hi)
    mu, sd = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0)
    sd[sd == 0] = 1.0
    Zs = np.nan_to_num((Z - mu) / sd, nan=0.0).astype(np.float32)
    print(f"{keep.sum()} Mordred columns retained, standardised")

    rng = np.random.default_rng(SEED)
    variants = {}
    for r2 in LEVELS:
        if r2 >= 1.0:
            variants["true"] = Zs
            continue
        s = np.sqrt(1.0 / r2 - 1.0)
        noisy = Zs + rng.normal(0.0, s, Zs.shape).astype(np.float32)
        # verify the realised R^2 rather than trusting the algebra
        num = ((Zs - noisy) ** 2).sum(0)
        den = ((Zs - Zs.mean(0)) ** 2).sum(0)
        realised = float(np.median(1 - num / np.maximum(den, 1e-9)))
        variants[f"r2_{r2}"] = noisy
        print(f"  target R2 {r2}: noise sd {s:.4f}, realised median R2 {realised:.4f}")

    arms = {"desc": lambda s, V=None: desc[s]}
    for k, V in variants.items():
        arms[f"desc+{k}"] = (lambda s, V=V: np.hstack([desc[s], V[s]]))

    per_ds, t0 = {}, time.time()
    for j, name in enumerate(name_of):
        if suite_of[j] != "moleculeace":
            continue
        s = slice(offsets[j], offsets[j + 1])
        smi_j, y_j = list(smiles[s]), y[s]
        per_ds[name] = {a: _cv_rmse(b(s), y_j, smi_j) for a, b in arms.items()}
        print(f"  {name}: " + " ".join(f"{a.replace('desc+','')}={per_ds[name][a]:.3f}"
                                       for a in arms) + f"  ({time.time() - t0:.0f}s)", flush=True)

    summary = {a: float(np.nanmean([r[a] for r in per_ds.values()])) for a in arms}
    OUT.write_text(json.dumps({"summary": summary, "per_dataset": per_ds}, indent=2))

    base, ceil = summary["desc"], summary["desc+true"]
    print(f"\n=== fidelity vs downstream gain (n={len(per_ds)} MoleculeACE) ===")
    print(f"baseline desc {base:.4f} | ceiling desc+true {ceil:.4f} (gain {ceil - base:+.4f})")
    for a in arms:
        if a == "desc":
            continue
        frac = 100 * (summary[a] - base) / (ceil - base)
        print(f"  {a:16s} {summary[a]:.4f}  gain {summary[a] - base:+.4f}  = {frac:4.0f}% of ceiling")
    print("\nOur ECFP surrogate sat at median R2 0.949 and delivered 2%.")


if __name__ == "__main__":
    main()
