"""Train the surrogate ladder and record reconstruction quality for each.

Five models, simplest first. The first four take the fixed-size input (ECFP + core
descriptors); the GNN takes the molecular graph, which is the only representation that holds
the actual variables the descriptors are polynomials in.

    ridge            1 GEMM   closed form, Gram shared across all targets
    linear+quadratic 2 GEMMs  shared rank-R quadratic basis; matches the target class
    pi-net (deg 3)   3 GEMMs  Hadamard recursion, arbitrary degree at linear param cost
    mlp              3 GEMMs  generic baseline
    gnn              msg pass sum readout (extensive - mean pooling is what collapsed UMA)

Ridge and linear+quadratic are BLAS-only at inference, so they ship inside the C++ core with
no PyTorch dependency. That is a real reason to stop early on the ladder.

Writes per-model bench predictions so downstream.py can measure what the reconstruction
actually buys.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"


# ----------------------------------------------------------------------------- data prep

def fit_prep(Y):
    keep = (np.isfinite(Y).mean(0) >= 0.95) & (np.nanstd(Y, 0) > 0)
    Z = Y[:, keep].astype(np.float64)
    lo, hi = np.nanpercentile(Z, 1, 0), np.nanpercentile(Z, 99, 0)
    Z = np.clip(Z, lo, hi)
    mu, sd = np.nanmean(Z, 0), np.nanstd(Z, 0)
    sd[sd == 0] = 1.0
    return {"keep": keep, "lo": lo, "hi": hi, "mu": mu, "sd": sd}


def apply_prep(Y, p):
    Z = np.clip(Y[:, p["keep"]].astype(np.float64), p["lo"], p["hi"])
    return ((Z - p["mu"]) / p["sd"]).astype(np.float32)


def fit_xprep(X, n_ecfp=2048):
    """Clean the INPUT matrix. Two problems, both fatal if ignored:

    * 159 core columns contain NaN (descriptors undefined for some molecules). A single NaN
      poisons the ridge Gram matrix and every weight derived from it.
    * the core block spans ~1e5 while log1p(ECFP) spans 0-3, which wrecks ridge conditioning
      and makes gradient descent crawl.

    ECFP is left as log1p counts (already well-scaled and sparse); only the core block is
    winsorised and standardised. Statistics are fit on the training split only.
    """
    # Statistics from a row sample. A float64 cast of the full 1M x 870 core block is 7 GB
    # before np.where/np.clip make their copies; 200k rows give percentiles and moments that
    # are identical to four decimals for this purpose.
    if X.shape[0] > 200_000:
        idx = np.random.default_rng(0).choice(X.shape[0], 200_000, replace=False)
        C = X[np.sort(idx), n_ecfp:].astype(np.float64)
    else:
        C = X[:, n_ecfp:].astype(np.float64)
    med = np.nanmedian(C, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    C = np.where(np.isfinite(C), C, med)
    lo, hi = np.percentile(C, 1, 0), np.percentile(C, 99, 0)
    C = np.clip(C, lo, hi)
    mu, sd = C.mean(0), C.std(0)
    sd[sd == 0] = 1.0
    return {"n_ecfp": n_ecfp, "med": med, "lo": lo, "hi": hi, "mu": mu, "sd": sd}


def apply_xprep(X, p, chunk=100_000):
    """Standardise in place, in row chunks.

    The original built a float64 copy of the whole core block and then hstacked it -- at 1M x
    2918 that peaks near 30 GB against 25 GB of RAM. Writing back into X in chunks keeps the
    peak at one chunk (~2 GB) and returns the same array, so callers must not rely on X being
    unmodified afterwards.
    """
    n = int(p["n_ecfp"])
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    med = p["med"].astype(np.float32)
    lo, hi = p["lo"].astype(np.float32), p["hi"].astype(np.float32)
    mu, sd = p["mu"].astype(np.float32), p["sd"].astype(np.float32)
    for i in range(0, X.shape[0], chunk):
        j = min(i + chunk, X.shape[0])
        E = X[i:j, :n]
        np.nan_to_num(E, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        C = X[i:j, n:]
        bad = ~np.isfinite(C)
        if bad.any():
            C[bad] = np.broadcast_to(med, C.shape)[bad]
        np.clip(C, lo, hi, out=C)
        C -= mu
        C /= sd
    return X


def scaffold_split(smiles, frac=0.9, seed=0):
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
    keys = sorted(g, key=lambda k: -len(g[k]))
    np.random.default_rng(seed).shuffle(keys)
    n = int(len(smiles) * frac)
    tr, va = [], []
    for k in keys:
        (tr if len(tr) < n else va).extend(g[k])
    return np.array(tr), np.array(va)


def r2(true, pred):
    out = np.full(true.shape[1], np.nan)
    for j in range(true.shape[1]):
        m = np.isfinite(true[:, j])
        if m.sum() < 10:
            continue
        t, q = true[m, j], pred[m, j]
        ss = float(((t - t.mean()) ** 2).sum())
        out[j] = 1.0 - float(((t - q) ** 2).sum()) / ss if ss > 0 else np.nan
    return out


# --------------------------------------------------------------------------------- models

def train_ridge(Xtr, Ytr, Xs, lam=10.0, chunk=50_000, return_w=False):
    """Closed-form ridge with the Gram matrix accumulated in chunks.

    The obvious form materialises `hstack([Xtr, 1]).astype(float64)` in one go. At 900k x 2919
    that is 21 GB, on top of the 11.7 GB float32 X -- which swapped this machine to a standstill
    (9.85 GB of swap, 1.8 GB free) before this was rewritten. G and B are the only things that
    need to be full precision, and both are tiny: (d+1)^2 and (d+1) x T. Peak is now one chunk.
    """
    d = Xtr.shape[1]
    G = np.zeros((d + 1, d + 1), np.float64)
    B = np.zeros((d + 1, Ytr.shape[1]), np.float64)
    Yc = np.nan_to_num(Ytr)
    for i in range(0, len(Xtr), chunk):
        j = min(i + chunk, len(Xtr))
        A = np.empty((j - i, d + 1), np.float64)
        A[:, :d] = Xtr[i:j]
        A[:, d] = 1.0
        G += A.T @ A
        B += A.T @ Yc[i:j]
        del A
    G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, B)
    out = []
    for X in Xs:
        preds = np.empty((len(X), W.shape[1]), np.float32)
        for i in range(0, len(X), chunk):
            j = min(i + chunk, len(X))
            A = np.empty((j - i, d + 1), np.float64)
            A[:, :d] = X[i:j]
            A[:, d] = 1.0
            preds[i:j] = (A @ W).astype(np.float32)
            del A
        out.append(preds)
    return (*out, W) if return_w else out


def _torch_train(net, Xtr, Ytr, Xs, epochs, lr, bs, label, threads=10):
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(np.nan_to_num(Ytr))
    mt = torch.from_numpy(np.isfinite(Ytr).astype(np.float32))
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            loss = (((net(xt[k]) - yt[k]) ** 2) * mt[k]).sum() / mt[k].sum().clamp(min=1)
            loss.backward()
            opt.step()
            tot += float(loss) * len(k)
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"    [{label}] ep{ep + 1:3d} train {tot / len(xt):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    outs = []
    with torch.no_grad():
        for X in Xs:
            outs.append(np.concatenate([net(torch.from_numpy(X[i:i + 4096])).numpy()
                                        for i in range(0, len(X), 4096)]))
    return outs


class LinQuad:
    """y = Wx + P[(Ux)*(Vx)] -- shared rank-R quadratic basis, then a linear readout.

    Per-target quadratic forms would be R*d*2*T parameters (~440M); computing the R basis
    features once and mapping them linearly to all targets is ~370k.
    """
    @staticmethod
    def build(d, t, R=64):
        import torch, torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(d, t)
                self.U = nn.Linear(d, R, bias=False)
                self.V = nn.Linear(d, R, bias=False)
                self.P = nn.Linear(R, t, bias=False)

            def forward(self, x):
                return self.lin(x) + self.P(self.U(x) * self.V(x))
        return M()


class PiNet:
    """Hadamard recursion: x_n = (W_n x) * x_{n-1} + x_{n-1}. N steps -> degree N."""
    @staticmethod
    def build(d, t, h=1024, deg=3):
        import torch, torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(d, h)
                self.ws = nn.ModuleList([nn.Linear(d, h) for _ in range(deg - 1)])
                self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(deg - 1)])
                self.out = nn.Linear(h, t)

            def forward(self, x):
                z = self.inp(x)
                for w, nrm in zip(self.ws, self.norms):
                    z = nrm(w(x) * z + z)   # LayerNorm: Hadamard products otherwise explode
                return self.out(z)
        return M()


def build_mlp(d, t):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(2048, 1024), nn.GELU(), nn.Linear(1024, t))


# ------------------------------------------------------------------------------------ GNN

ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
NDIM_ATOM = len(ATOMS) + 1 + 5      # element one-hot + explicit OTHER slot + 5 scalars
NDIM_BOND = 6                       # bond order one-hot(4) + conjugated + in-ring


def graph_of(smi):
    """SMILES -> (atom features, edge index, bond features).

    TWO FIXES over the first version, both found by reading the featuriser back on 2026-08-26
    while writing up what the GNN actually sees:

    1. IODINE COLLIDED WITH "OTHER". The old code did `ATOMS.index(z) if z in ATOMS else 11`
       over a 12-element list, so index 11 meant BOTH iodine and every unlisted element --
       the model could not tell an iodine from a selenium or a metal. OTHER now has its own
       slot at the end and the element block is len(ATOMS)+1 wide.
    2. THERE WERE NO BOND FEATURES AT ALL. Bonds existed only as edges, so single/double/
       triple/aromatic were invisible except indirectly through hydrogen counts. That is a
       strange blind spot for a model whose job is to predict descriptors built on bond
       orders, and Figure A's new C=C -> C-C panel is precisely an edit it could barely see.

    Formal charge is added at the same time: protonation state moves a large part of the
    descriptor block (Figure A panel d) and the old feature set had no way to represent it.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return (np.zeros((1, NDIM_ATOM), np.float32), np.zeros((2, 0), np.int64),
                np.zeros((0, NDIM_BOND), np.float32))
    n = m.GetNumAtoms()
    nel = len(ATOMS)
    F = np.zeros((n, NDIM_ATOM), np.float32)
    for i, a in enumerate(m.GetAtoms()):
        z = a.GetAtomicNum()
        F[i, ATOMS.index(z) if z in ATOMS else nel] = 1.0        # nel == the OTHER slot
        F[i, nel + 1] = a.GetDegree() / 4.0
        F[i, nel + 2] = a.GetTotalNumHs() / 4.0
        F[i, nel + 3] = float(a.GetIsAromatic())
        F[i, nel + 4] = float(a.IsInRing())
        F[i, nel + 5] = np.clip(a.GetFormalCharge(), -2, 2) / 2.0
    e, bf = [[], []], []
    BT = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
          Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        v = np.zeros(NDIM_BOND, np.float32)
        v[BT.get(b.GetBondType(), 0)] = 1.0
        v[4] = float(b.GetIsConjugated())
        v[5] = float(b.IsInRing())
        e[0] += [i, j]
        e[1] += [j, i]
        bf += [v, v]                      # the same feature vector on both directed copies
    E = np.array(e, np.int64) if e[0] else np.zeros((2, 0), np.int64)
    B = np.stack(bf) if bf else np.zeros((0, NDIM_BOND), np.float32)
    return F, E, B


def build_gnn(h, depth, t_out):
    """The D-MPNN, at module level so a CHECKPOINT CAN BE REBUILT WITHOUT TRAINING.

    Lifted out of train_gnn verbatim -- same attribute names, so the state_dict keys of every
    checkpoint written before this refactor still load (proxy_cost.py asserts that with
    strict=True). Inference-only consumers need the architecture without the 35-minute training
    function wrapped around it, and a second hand-written copy of a network is exactly how the
    copy and the original drift apart.
    """
    import torch
    import torch.nn as nn

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(NDIM_ATOM, h)
            # Edge-conditioned messages: the message from a neighbour is a function of the
            # neighbour's state AND the bond it arrives over, rather than a bare sum of
            # neighbour states. This is what lets bond order reach the model at all.
            self.emsg = nn.ModuleList([nn.Linear(h + NDIM_BOND, h) for _ in range(depth)])
            self.msg = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(depth)])
            self.nrm = nn.ModuleList([nn.LayerNorm(h) for _ in range(depth)])
            self.out = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, t_out))

        def forward(self, F, E, B, batch, nb):
            x = self.emb(F)
            for em, lin, nrm in zip(self.emsg, self.msg, self.nrm):
                agg = torch.zeros_like(x)
                if E.shape[1] > 0:
                    agg.index_add_(0, E[1], torch.relu(em(torch.cat([x[E[0]], B], -1))))
                x = nrm(x + torch.relu(lin(torch.cat([x, agg], -1))))
            # SUM readout: descriptors are extensive; mean pooling is what collapsed UMA to rank 5
            g = torch.zeros(nb, x.shape[1], device=x.device)
            g.index_add_(0, batch, x)
            return self.out(g)

    return GNN()


def collate_graphs(idx, graphs, dev):
    """Pack (atom features, edge index, bond features) into one disjoint-union batch."""
    import torch
    Fs, Es, Bs, bt, off = [], [], [], [], 0
    for k, i in enumerate(idx):
        F, E, B = graphs[i]
        Fs.append(F)
        Es.append(E + off)
        Bs.append(B)
        bt.append(np.full(len(F), k, np.int64))
        off += len(F)
    return (torch.from_numpy(np.concatenate(Fs)).to(dev),
            torch.from_numpy(np.concatenate(Es, axis=1)).to(dev),
            torch.from_numpy(np.concatenate(Bs)).to(dev),
            torch.from_numpy(np.concatenate(bt)).to(dev), len(idx))


def train_gnn(smi_tr, Ytr, smi_sets, epochs, t_out, h=128, depth=4, bs=256, threads=10,
              device="cpu", ckpt=None):
    """D-MPNN over the molecular graph. `device` is the ONLY difference between the local CPU
    run and the GPU run -- same class, same init, same seed, same optimiser, same readout --
    so the GPU arm is comparable with the four matrix models rather than a separate experiment.
    """
    import torch
    torch.set_num_threads(threads); torch.manual_seed(0)
    dev = torch.device(device)

    print("    [gnn] building graphs...", flush=True)
    G = [graph_of(s) for s in smi_tr]
    net = build_gnn(h, depth, t_out).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    # Resume. Spot reclaimed a 60-epoch run at epoch ~20 and the volume went with it; without
    # this, every interruption restarts from scratch and a long run on spot is a lottery.
    start_ep = 0
    if ckpt and Path(ckpt).exists():
        try:
            c = torch.load(ckpt, map_location=dev, weights_only=False)
            if c.get("arch", {}) == {"h": h, "depth": depth, "t_out": t_out,
                                     "na": NDIM_ATOM, "nb": NDIM_BOND}:
                net.load_state_dict(c["state_dict"])
                opt.load_state_dict(c["opt"])
                start_ep = int(c["epoch"])
                print(f"    [gnn] resumed from epoch {start_ep}", flush=True)
            else:
                print(f"    [gnn] checkpoint arch {c.get('arch')} != current; ignoring",
                      flush=True)
        except Exception as e:
            print(f"    [gnn] checkpoint unreadable ({type(e).__name__}); starting fresh",
                  flush=True)
    yt = torch.from_numpy(np.nan_to_num(Ytr)).to(dev)
    mt = torch.from_numpy(np.isfinite(Ytr).astype(np.float32)).to(dev)

    def collate(idx, graphs):
        return collate_graphs(idx, graphs, dev)

    t0 = time.time()
    for ep in range(start_ep, epochs):
        net.train(); perm = np.random.default_rng(ep).permutation(len(G)); tot = 0.0
        for i in range(0, len(G), bs):
            idx = perm[i:i + bs]
            F, E, Bf, bt, nb = collate(idx, G)
            opt.zero_grad()
            p = net(F, E, Bf, bt, nb)
            k = torch.from_numpy(idx).to(dev)
            loss = (((p - yt[k]) ** 2) * mt[k]).sum() / mt[k].sum().clamp(min=1)
            loss.backward(); opt.step(); tot += float(loss) * len(idx)
        print(f"    [gnn] ep{ep + 1:3d} train {tot / len(G):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if ckpt:
            # Every epoch, not just at the end. On a spot instance an interruption arrives with
            # ~2 minutes' notice; a run that only saves at completion loses everything.
            torch.save({"epoch": ep + 1, "state_dict": net.state_dict(),
                        "opt": opt.state_dict(), "loss": tot / len(G),
                        "arch": {"h": h, "depth": depth, "t_out": t_out,
                                 "na": NDIM_ATOM, "nb": NDIM_BOND}}, ckpt)

    net.eval(); outs = []
    with torch.no_grad():
        for smis in smi_sets:
            gs = [graph_of(s) for s in smis]
            preds = []
            for i in range(0, len(gs), 512):
                idx = list(range(i, min(i + 512, len(gs))))
                F, E, Bf, bt, nb = collate(idx, gs)
                preds.append(net(F, E, Bf, bt, nb).cpu().numpy())
            outs.append(np.concatenate(preds))
    return outs


# ------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge,linquad,pinet,mlp,gnn")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--gnn-epochs", type=int, default=12)
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap training molecules (sanity pass; omit for the real run)")
    ap.add_argument("--contiguous-split", action="store_true",
                    help="split on the shard order, which is already scaffold-grouped; keeps "
                         "X[tr] a view instead of an 10.5 GB copy at 1M")
    ap.add_argument("--fp16", action="store_true",
                    help="hold X as float16 -- 5.8 GB rather than 11.6 GB at 1M")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the GNN: cpu | mps | cuda. The ONLY difference "
                         "between the local and GPU runs -- same class, seed and optimiser.")
    ap.add_argument("--gnn-ckpt", default=str(OUT / "ckpt_gnn.pt"),
                    help="written every epoch and resumed from if the architecture matches")
    a = ap.parse_args()

    import corpus_data
    X, Y, smi, meta = corpus_data.load(max_n=a.max_n,
                                       dtype=np.float16 if a.fp16 else np.float32)
    Xb, _, smib, _ = corpus_data.load_bench(dtype=np.float32)
    ynames = meta["ynames"]
    fams = corpus_data.families_from_names(ynames)

    # Targets that are mostly NaN or constant cannot be learned and quietly poison a masked
    # loss. Drop them explicitly and say so -- "trained on 161 targets" and "trained on 140"
    # are different claims.
    keep_t, _ = corpus_data.drop_dead_targets(Y, ynames, meta)
    if not keep_t.all():
        Y, fams = Y[:, keep_t], fams[keep_t]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    print(f"train X{X.shape} Y{Y.shape} | bench X{Xb.shape}")
    assert Xb.shape[1] == X.shape[1], (
        f"bench X has {Xb.shape[1]} columns but corpus has {X.shape[1]} -- the CORE/PREDICT "
        f"split changed; re-run assemble.py for the benchmark before training")

    if a.contiguous_split:
        # `corpus.py select` emits each scaffold's DEPTH molecules adjacently, so a contiguous
        # cut is a scaffold split with at most one scaffold straddling the boundary. That
        # matters for memory, not just tidiness: X[tr] with fancy indices COPIES, which at
        # 1M x 2918 float32 means 11.7 GB plus a 10.5 GB copy against 25 GB of RAM. A
        # contiguous slice is a view and costs nothing.
        cut = int(len(smi) * 0.9)
        tr, va = np.arange(cut), np.arange(cut, len(smi))
        print(f"contiguous scaffold split at {cut:,} (view, not copy)")
    else:
        tr, va = scaffold_split(smi)
    prep = fit_prep(Y[tr])
    Ytr, Yva = apply_prep(Y[tr], prep), apply_prep(Y[va], prep)
    fams_k = fams[prep["keep"]]
    print(f"split {len(tr):,}/{len(va):,} | targets kept {prep['keep'].sum()} of {Y.shape[1]}")

    Xtr = X[:len(tr)] if a.contiguous_split else X[tr]
    xp = fit_xprep(Xtr)
    X, Xb = apply_xprep(X, xp), apply_xprep(Xb, xp)
    Xtr = X[:len(tr)] if a.contiguous_split else X[tr]
    Xva = X[len(tr):] if a.contiguous_split else X[va]
    assert np.isfinite(X).all() and np.isfinite(Xb).all(), "non-finite survived X preprocessing"
    print(f"X cleaned: NaN {int(np.isnan(X).sum())}, core block now "
          f"[{X[:, 2048:].min():.2f}, {X[:, 2048:].max():.2f}]\n")
    np.savez_compressed(OUT / "prep_blocks.npz", **prep)
    np.savez_compressed(OUT / "prep_x.npz", **xp)

    report = {}
    for name in a.models.split(","):
        print(f"--- {name} ---", flush=True)
        t0 = time.time()
        net_ref, W_ridge = None, None
        if name == "ridge":
            pv, pb, W_ridge = train_ridge(Xtr, Ytr, [Xva, Xb], return_w=True)
        elif name == "gnn":
            pv, pb = train_gnn([smi[i] for i in tr], Ytr,
                               [[smi[i] for i in va], smib], a.gnn_epochs, Ytr.shape[1],
                               device=a.device, ckpt=a.gnn_ckpt)
        else:
            d, t = X.shape[1], Ytr.shape[1]
            net = {"linquad": lambda: LinQuad.build(d, t),
                   "pinet": lambda: PiNet.build(d, t),
                   "mlp": lambda: build_mlp(d, t)}[name]()
            net_ref = net
            pv, pb = _torch_train(net, Xtr, Ytr, [Xva, Xb], a.epochs, 1e-3, 512, name)
        rr = r2(Yva, pv)
        byfam = {f: float(np.nanmedian(rr[fams_k == f])) for f in sorted(set(fams_k))
                 if (fams_k == f).sum() >= 3}
        report[name] = {"median_r2": float(np.nanmedian(rr)), "mean_r2": float(np.nanmean(rr)),
                        "seconds": time.time() - t0, "by_family": byfam}
        np.savez_compressed(OUT / f"pred_bench_{name}.npz", pred=pb.astype(np.float32))
        np.savez_compressed(OUT / f"pred_val_{name}.npz", pred=pv.astype(np.float32),
                            r2=rr.astype(np.float32))
        # The trained model IS the product. Predictions alone mean the winner has to be
        # retrained before anything can ship, and reproduced exactly to be trustworthy.
        if name == "ridge":
            np.savez_compressed(OUT / "ckpt_ridge.npz", W=W_ridge)
        elif net_ref is not None:
            import torch
            torch.save({"state_dict": net_ref.state_dict(), "model": name,
                        "d": X.shape[1], "t": Ytr.shape[1]}, OUT / f"ckpt_{name}.pt")
        print(f"  {name}: median R2 {report[name]['median_r2']:.4f}  "
              f"mean {report[name]['mean_r2']:.4f}  ({report[name]['seconds']:.0f}s)\n", flush=True)
        json.dump(report, open(OUT / "models_report.json", "w"), indent=2)

    print("=== reconstruction summary ===")
    for k, v in sorted(report.items(), key=lambda kv: -kv[1]["median_r2"]):
        print(f"  {k:10s} median R2 {v['median_r2']:.4f}  mean {v['mean_r2']:.4f}  {v['seconds']:.0f}s")


if __name__ == "__main__":
    main()
