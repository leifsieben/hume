"""Package the GNN job so the GPU run is comparable with the local CPU runs, not a new experiment.

The GNN consumes SMILES rather than X, so the payload is 0.7 GB instead of 11.7 GB. What must
travel is everything that defines the *comparison*:

  * the identical contiguous split point (900,000), so train/val molecules match exactly
  * `prep_blocks.npz` -- the target preprocessing FIT ON THE LOCAL TRAINING SPLIT. Re-fitting it
    remotely would standardise against slightly different statistics and make the R^2 values
    incomparable with ridge/linquad/pinet/mlp.
  * `models.py` itself, so `train_gnn` and `graph_of` are the same code, not a reimplementation.
    `device` is the only argument that differs between the two runs.
"""
from __future__ import annotations
import glob, json, sys, tarfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "gpu" / "payload"
CUT = 900_000


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    smi = (ROOT / "data" / "corpus1m" / "selected.txt").read_text().split()
    Ys = [np.load(f)["Y"] for f in sorted(glob.glob(str(ROOT / "data/corpus1m/packed/pk_*.npz")))]
    Y = np.concatenate(Ys); del Ys
    bs = []
    for f in sorted(glob.glob(str(ROOT / "data/bench1m/packed/pk_*.npz"))):
        bs.extend(list(np.load(f, allow_pickle=True)["smiles"]))
    if limit:
        smi, Y, bs = smi[:limit], Y[:limit], bs[:2000]
    cut = min(CUT, int(len(smi) * 0.9))

    # Same order of operations as models.py: drop dead targets FIRST, then apply the prep
    # fitted on the survivors. prep["keep"] is 165 wide because the constant column was already
    # removed when it was fitted; applying it to the raw 166 would silently misalign every
    # target by one from the drop point onward.
    sys.path.insert(0, str(ROOT))
    import json as _json, corpus_data
    meta = _json.load(open(ROOT / "data/corpus1m/meta.json"))
    keep_t, _ = corpus_data.drop_dead_targets(Y, meta["ynames"], meta)
    Y = Y[:, keep_t]
    prep = dict(np.load(ROOT / "data" / "surrogate" / "prep_blocks.npz", allow_pickle=True))
    keep = prep["keep"]
    assert Y.shape[1] == keep.shape[0], (
        f"target count {Y.shape[1]} != prep mask {keep.shape[0]} -- the local run and this "
        f"packager disagree about which targets exist")
    print(f"corpus {len(smi):,} | cut {cut:,} | targets kept {int(keep.sum())} of {Y.shape[1]}")

    def apply_prep(Yb):
        Z = np.clip(Yb[:, keep].astype(np.float64), prep["lo"], prep["hi"])
        return ((Z - prep["mu"]) / prep["sd"]).astype(np.float32)

    np.savez_compressed(OUT / "job.npz",
                        smi_tr=np.array(smi[:cut], dtype=object),
                        smi_va=np.array(smi[cut:], dtype=object),
                        smi_bench=np.array(bs, dtype=object),
                        Ytr=apply_prep(Y[:cut]), Yva=apply_prep(Y[cut:]))
    json.dump({"cut": cut, "n_corpus": len(smi), "n_bench": len(bs),
               "n_targets": int(keep.sum())}, open(OUT / "manifest.json", "w"), indent=2)
    for f in ("models.py",):
        (OUT / f).write_bytes((ROOT / f).read_bytes())
    (OUT / "prep_blocks.npz").write_bytes((ROOT / "data/surrogate/prep_blocks.npz").read_bytes())

    tar = ROOT / "gpu" / "gnn_job.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for p in sorted(OUT.iterdir()):
            tf.add(p, arcname=p.name)
    print(f"wrote {tar} ({tar.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
