"""Measured inference overhead of each descriptor proxy, per molecule.

    python proxy_cost.py            ->  data/surrogate/proxy_cost.json

THE ACCOUNTING QUESTION THIS ANSWERS, stated precisely because it is easy to get wrong:
HUME already builds the RDKit mol, already computes ECFP, and already computes the CORE
descriptor block and the five custom blocks -- those are the cheap half of the split and they
are paid whether or not a proxy runs. So the MARGINAL cost of adding a predict block is only
what the proxy needs BEYOND that.

For the four matrix models that is a forward pass over a vector HUME has already assembled:
ridge is one (2918 x 165) matmul, the rest are small MLPs.

For the GNN it is different in kind, and this is the fact that matters for the decision: the
GNN does NOT read the fingerprint or the descriptor block at all. It reads the MOLECULAR GRAPH.
Its marginal cost is graph construction from the existing mol object plus a message-passing
forward pass -- and it would still work if ECFP were switched off entirely, which none of the
other four would.

Timings are wall clock at a realistic screening batch size, single process, and are reported
per molecule. The RDKit mol build and the ECFP/descriptor computation are deliberately OUTSIDE
the timed region.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"
N = 4096
BATCH = 1024


def _timeit(fn, repeats=3):
    fn()                                            # warm up: first call pays lazy init
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) / N * 1e6                        # microseconds per molecule


def main() -> None:
    import torch
    import models

    torch.set_num_threads(1)                        # per-core figure; screening scales by cores
    bench = np.load(OUT / "bench.npz", allow_pickle=True)
    smi = [str(s) for s in bench["smiles"][:N]]

    import corpus_data
    Xb, _, _, _ = corpus_data.load_bench()
    X = Xb[:N].astype(np.float32)
    d, t = X.shape[1], 165
    print(f"{N} molecules | proxy input {d} dims -> {t} targets | 1 thread\n")

    res = {}

    # ridge: a single dense matmul. The bias row is folded in during training, so inference is
    # exactly X @ W -- nothing else.
    W = np.zeros((d + 1, t), np.float32)
    def _ridge():
        for i in range(0, N, BATCH):
            b = X[i:i + BATCH]
            np.concatenate([b, np.ones((len(b), 1), np.float32)], 1) @ W
    res["ridge"] = _timeit(_ridge)

    Xt = torch.from_numpy(X)
    for name, net in (("mlp", models.build_mlp(d, t)), ("pinet", models.PiNet.build(d, t))):
        net.eval()
        def _fwd(net=net):
            with torch.no_grad():
                for i in range(0, N, BATCH):
                    net(Xt[i:i + BATCH])
        res[name] = _timeit(_fwd)

    # GNN: graph construction is timed SEPARATELY from the forward pass, because they scale
    # differently -- graph building is pure Python/RDKit and does not benefit from a GPU, while
    # the forward pass does. Reporting one number would hide which half dominates.
    res["gnn_graph_build"] = _timeit(lambda: [models.graph_of(s) for s in smi], repeats=2)

    c = torch.load(OUT / "ckpt_gnn.pt", map_location="cpu", weights_only=False)
    arch = c.get("arch", {"h": 128, "depth": 4, "t_out": t})
    gnet = models.build_gnn(arch["h"], arch["depth"], arch["t_out"])
    # strict=True on purpose: this is also the check that lifting GNN out of train_gnn kept the
    # state_dict keys identical, so the 60-epoch checkpoint is still loadable by the refactor.
    gnet.load_state_dict(c["state_dict"], strict=True)
    gnet.eval()
    print(f"  gnn checkpoint: epoch {c['epoch']}, arch {arch} -- loaded strict\n")
    G = [models.graph_of(s) for s in smi]

    def _gnn():
        with torch.no_grad():
            for i in range(0, N, BATCH):
                gnet(*models.collate_graphs(range(i, min(i + BATCH, N)), G, torch.device("cpu")))
    res["gnn_forward"] = _timeit(_gnn)
    res["gnn"] = res["gnn_graph_build"] + res["gnn_forward"]

    print(f"{'proxy':18s} {'us/mol':>9s}")
    for k in ("ridge", "mlp", "pinet", "gnn_graph_build", "gnn_forward", "gnn"):
        print(f"  {k:16s} {res[k]:9.2f}")
    json.dump(res, open(OUT / "proxy_cost.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'proxy_cost.json'}")


if __name__ == "__main__":
    main()
