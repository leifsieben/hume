"""Runs on the GPU instance. Imports the SAME train_gnn as the local CPU runs.

`device="cuda"` is the only argument that differs. Nothing about the architecture, seed,
optimiser, batch size, readout or target preprocessing is re-decided here -- if any of it were,
the GNN would be a different experiment rather than a fifth arm.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    import torch
    import models
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {dev} | "
          f"{torch.cuda.get_device_name(0) if dev=='cuda' else 'NO GPU'}", flush=True)

    z = np.load(HERE / "job.npz", allow_pickle=True)
    man = json.load(open(HERE / "manifest.json"))
    smi_tr, smi_va = list(z["smi_tr"]), list(z["smi_va"])
    smi_b, Ytr, Yva = list(z["smi_bench"]), z["Ytr"], z["Yva"]
    print(f"train {len(smi_tr):,} | val {len(smi_va):,} | bench {len(smi_b):,} | "
          f"targets {Ytr.shape[1]} | cut {man['cut']:,}", flush=True)

    t0 = time.time()
    print(f"arch h={h} depth={depth} epochs={epochs}", flush=True)
    pv, pb = models.train_gnn(smi_tr, Ytr, [smi_va, smi_b], epochs, Ytr.shape[1],
                              h=h, depth=depth, device=dev, threads=8,
                              ckpt=str(HERE / "ckpt_gnn.pt"))
    rr = models.r2(Yva, pv)
    out = {"median_r2": float(np.nanmedian(rr)), "mean_r2": float(np.nanmean(rr)),
           "seconds": time.time() - t0, "device": dev, "epochs": epochs, "h": h, "depth": depth,
           "n_train": len(smi_tr), "n_targets": int(Ytr.shape[1])}
    np.savez_compressed(HERE / "pred_bench_gnn.npz", pred=pb.astype(np.float32))
    np.savez_compressed(HERE / "pred_val_gnn.npz", pred=pv.astype(np.float32),
                        r2=rr.astype(np.float32))
    json.dump(out, open(HERE / "gnn_report.json", "w"), indent=2)
    print(f"DONE median R2 {out['median_r2']:.4f}  mean {out['mean_r2']:.4f}  "
          f"({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
