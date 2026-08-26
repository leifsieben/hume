"""Train the descriptor surrogate: ECFP counts -> the full expensive descriptor block.

Design decisions and where they came from:

* **Input is ECFP only.** Measured cost: ECFP 29 us/mol vs RDKit-96 976 us/mol, so feeding
  RDKit-96 in would forfeit most of the speed win to make the model's job easier. Running
  ``--with-rdkit96`` adds it as input; that is the documented fallback if the ECFP-only gap
  turns out large, not the default.
* **Output is the whole block, uncompressed.** Gate 1 established that projecting the
  descriptor union to 64/128/256 dims keeps only 21-26% of its downstream value, and the
  follow-up control showed the projection (not the preprocessing) is what destroys it. So
  there is no bottleneck here and no dimensionality to tune.
* **Ridge before MLP.** A closed-form linear fit costs seconds and gives the floor any
  nonlinear model must beat. If the MLP cannot clear it convincingly, the problem is the
  input representation, not model capacity.
* **R^2 is diagnostics, not the verdict.** Gate 1 showed 95% of the *variance* can buy 26%
  of the *gain*. The scoreboard is downstream RMSE via the benchmark harness; per-family
  R^2 only tells us where to intervene.

Reports R^2 per Mordred family because the families that carry the signal (autocorrelation,
chi-path, walk counts) are global topological invariants that a bag of local ECFP
environments is least able to reach. That is the predicted failure mode and it should be
visible per-family, not buried in an average.

Usage:
    .venv/bin/python train_surrogate.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TGT_DIR = ROOT / "data" / "uma100k" / "targets"
UMA_DIR = ROOT / "data" / "uma100k" / "embeddings"
OUT = ROOT / "data" / "surrogate"


def load_targets(with_uma: bool):
    shards = sorted(TGT_DIR.glob("tgt_*.npz"))
    if not shards:
        raise SystemExit(f"no target shards in {TGT_DIR}; run build_targets.py first")
    smi, ecfp, blocks = [], [], defaultdict(list)
    for p in shards:
        z = np.load(p, allow_pickle=True)
        smi.extend(list(z["smiles"]))
        ecfp.append(z["ecfp"])
        for k in ("rdkit96", "erg", "mordred"):
            blocks[k].append(z[k])
    data = {"smiles": smi, "ecfp": np.concatenate(ecfp),
            **{k: np.concatenate(v) for k, v in blocks.items()}}

    if with_uma:
        emb = {}
        for p in sorted(UMA_DIR.glob("emb_*.npz")):
            z = np.load(p, allow_pickle=True)
            emb.update(dict(zip(z["smiles"], z["emb"])))
        keep = [i for i, s in enumerate(data["smiles"]) if s in emb]
        print(f"UMA embeddings available for {len(keep):,}/{len(data['smiles']):,} molecules")
        for k in ("ecfp", "rdkit96", "erg", "mordred"):
            data[k] = data[k][keep]
        data["smiles"] = [data["smiles"][i] for i in keep]
        data["uma"] = np.stack([emb[s] for s in data["smiles"]])
    return data


def scaffold_split(smiles, frac=0.9, seed=0):
    """Split by Bemis-Murcko scaffold so validation molecules are structurally held out."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        try:
            groups[MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""].append(i)
        except Exception:
            groups[""].append(i)
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_train = int(len(smiles) * frac)
    tr, va = [], []
    for k in keys:
        (tr if len(tr) < n_train else va).extend(groups[k])
    return np.array(tr), np.array(va)


def fit_prep(Y):
    """Column filter + winsorise + standardise, fit on the train split only.

    Matches the pipeline the Gate 1 control validated at 86-89% of the raw-Mordred gain,
    which is therefore the ceiling for anything trained against these targets.
    """
    keep = (np.isfinite(Y).mean(axis=0) >= 0.95) & (np.nanstd(Y, axis=0) > 0)
    Z = Y[:, keep].astype(np.float64)
    lo, hi = np.nanpercentile(Z, 1, axis=0), np.nanpercentile(Z, 99, axis=0)
    Z = np.clip(Z, lo, hi)
    mu, sd = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0)
    sd[sd == 0] = 1.0
    return {"keep": keep, "lo": lo, "hi": hi, "mu": mu, "sd": sd}


def apply_prep(Y, p):
    Z = np.clip(Y[:, p["keep"]].astype(np.float64), p["lo"], p["hi"])
    return ((Z - p["mu"]) / p["sd"]).astype(np.float32)


def r2_per_col(true, pred):
    """R^2 per column, NaN-aware. Targets are standardised, so denominator ~1."""
    out = np.full(true.shape[1], np.nan)
    for j in range(true.shape[1]):
        m = np.isfinite(true[:, j])
        if m.sum() < 10:
            continue
        t, q = true[m, j], pred[m, j]
        ss = float(((t - t.mean()) ** 2).sum())
        out[j] = 1.0 - float(((t - q) ** 2).sum()) / ss if ss > 0 else np.nan
    return out


def mordred_families():
    from mordred import Calculator, descriptors as mdesc
    return np.array([type(d).__module__.split(".")[-1]
                     for d in Calculator(mdesc, ignore_3D=True).descriptors])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--with-rdkit96", action="store_true",
                    help="fallback variant: add RDKit-96 to the input (34x the ECFP cost)")
    ap.add_argument("--with-uma", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = load_targets(a.with_uma)
    n = len(d["smiles"])
    print(f"{n:,} molecules")

    X = d["ecfp"]
    if a.with_rdkit96:
        X = np.hstack([X, np.nan_to_num(d["rdkit96"])])
    X = np.log1p(np.clip(X, 0, None)).astype(np.float32)  # ECFP counts are heavy-tailed

    target_blocks = ["rdkit96", "erg", "mordred"] + (["uma"] if a.with_uma else [])
    Y_raw = np.hstack([d[k] for k in target_blocks])
    bounds, off = {}, 0
    for k in target_blocks:
        bounds[k] = (off, off + d[k].shape[1])
        off += d[k].shape[1]
    print(f"input {X.shape} -> targets {Y_raw.shape} ({', '.join(target_blocks)})")

    tr, va = scaffold_split(d["smiles"])
    print(f"scaffold split: {len(tr):,} train / {len(va):,} val")

    prep = fit_prep(Y_raw[tr])
    Ytr, Yva = apply_prep(Y_raw[tr], prep), apply_prep(Y_raw[va], prep)
    print(f"{prep['keep'].sum()} of {Y_raw.shape[1]} target columns retained")

    # ---- ridge: the linear floor -------------------------------------------------------
    t0 = time.time()
    Xtr = np.hstack([X[tr], np.ones((len(tr), 1), np.float32)]).astype(np.float64)
    G = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])
    W = np.linalg.solve(G, Xtr.T @ np.nan_to_num(Ytr))
    Xva = np.hstack([X[va], np.ones((len(va), 1), np.float32)]).astype(np.float64)
    ridge_pred = (Xva @ W).astype(np.float32)
    r2_ridge = r2_per_col(Yva, ridge_pred)
    print(f"ridge: median R2 {np.nanmedian(r2_ridge):.3f} "
          f"mean {np.nanmean(r2_ridge):.3f} ({time.time() - t0:.0f}s)")

    # ---- MLP ---------------------------------------------------------------------------
    import torch
    import torch.nn as nn

    dev = "cpu"
    net = nn.Sequential(
        nn.Linear(X.shape[1], 2048), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(2048, 1024), nn.GELU(),
        nn.Linear(1024, Ytr.shape[1]),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X[tr]).to(dev)
    Yt = torch.from_numpy(Ytr).to(dev)
    Mt = torch.from_numpy(np.isfinite(Ytr).astype(np.float32)).to(dev)
    Yt = torch.nan_to_num(Yt)
    Xv = torch.from_numpy(X[va]).to(dev)

    t0, bs = time.time(), 512
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            p = net(Xt[idx])
            # Masked MSE: undefined descriptor cells must not contribute gradient.
            loss = (((p - Yt[idx]) ** 2) * Mt[idx]).sum() / Mt[idx].sum().clamp(min=1)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = net(Xv).cpu().numpy()
            r2 = r2_per_col(Yva, pv)
            print(f"  ep{ep + 1:3d} train {tot / len(Xt):.4f} | val median R2 "
                  f"{np.nanmedian(r2):.3f} ({time.time() - t0:.0f}s)", flush=True)

    net.eval()
    with torch.no_grad():
        mlp_pred = net(Xv).cpu().numpy()
    r2_mlp = r2_per_col(Yva, mlp_pred)

    # ---- per-block and per-Mordred-family diagnostics -----------------------------------
    kept_idx = np.where(prep["keep"])[0]
    report = {"n_molecules": n, "n_targets": int(prep["keep"].sum()),
              "ridge_median_r2": float(np.nanmedian(r2_ridge)),
              "mlp_median_r2": float(np.nanmedian(r2_mlp)), "blocks": {}, "mordred_families": {}}
    print("\nper block (MLP):")
    for k, (s, e) in bounds.items():
        sel = (kept_idx >= s) & (kept_idx < e)
        if sel.sum():
            report["blocks"][k] = {"n": int(sel.sum()),
                                   "median_r2": float(np.nanmedian(r2_mlp[sel]))}
            print(f"  {k:10s} n={sel.sum():5d}  median R2 {np.nanmedian(r2_mlp[sel]):.3f}")

    fam = mordred_families()
    s, e = bounds["mordred"]
    rows = []
    for f in sorted(set(fam)):
        cols = np.where((kept_idx >= s) & (kept_idx < e))[0]
        sel = cols[fam[kept_idx[cols] - s] == f]
        if len(sel) >= 3:
            rows.append((f, len(sel), float(np.nanmedian(r2_mlp[sel]))))
    rows.sort(key=lambda r: r[2])
    print("\nMordred families, worst first (the global-topology ones are the risk):")
    for f, cnt, v in rows:
        report["mordred_families"][f] = {"n": cnt, "median_r2": v}
        print(f"  {f:24s} n={cnt:5d}  median R2 {v:.3f}")

    (OUT / "train_report.json").write_text(json.dumps(report, indent=2))
    torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1],
                "bounds": bounds, "with_rdkit96": a.with_rdkit96},
               OUT / "surrogate.pt")
    np.savez_compressed(OUT / "prep.npz", **prep)
    print(f"\nwrote {OUT}/train_report.json, surrogate.pt, prep.npz")


if __name__ == "__main__":
    main()
