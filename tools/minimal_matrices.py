"""Compute the descriptor matrices the minimal-spec selection runs on.

    .venv/bin/python tools/minimal_matrices.py

Three samples, and the split is the scientific claim rather than a convenience (see
docs/MINIMAL_SPEC.md section on sampling):

  repA  24,000 from the 1M TRAINING corpus -- the representative draw. Not a benchmark set:
        selecting on benchmark molecules would tune the spec to the chemistry those benchmarks
        happen to contain, and the spec is permanent.
  repB  24,000 more from the same corpus, DISJOINT from repA. Used only to check that the
        selection is stable; if the two orderings disagree at k, the criterion is
        under-determined there and the honest answer is a larger k.
  adv   24,000 from cpp/hard.smi -- salts, mixtures, unusual elements, size extremes. A column
        redundant on repA alone may be carrying something the corpus under-represents.

standardize="none" throughout, and it is recorded in the output: it changes descriptor values
for every multi-fragment input, so a spec derived under one setting does not transfer.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

import molhume

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter("ignore")

N = 24_000
SEED = 20260901
OUT = Path("results/minimal")
OUT.mkdir(parents=True, exist_ok=True)


def read_smiles(path, limit=None):
    out = []
    with open(path) as fh:
        for line in fh:
            s = line.split()[0].strip() if line.strip() else ""
            if s:
                out.append(s)
            if limit and len(out) >= limit:
                break
    return out


def featurize(smis, tag):
    ok = [s for s in smis if Chem.MolFromSmiles(s) is not None]
    X = molhume.featurize(ok, standardize="none", fingerprint=False)
    print(f"  {tag:5s} {len(ok):6d} molecules  X{X.shape}  "
          f"non-finite {(~np.isfinite(X)).mean():.4%}")
    return X, ok


pool = read_smiles("data/corpus1m/selected.txt")
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(pool))
repA_s = [pool[i] for i in idx[:N]]
repB_s = [pool[i] for i in idx[N:2 * N]]
adv_s = read_smiles("cpp/hard.smi", limit=int(N * 1.2))

print(f"  training pool {len(pool)} molecules; drawing {N} + {N} disjoint")
XA, sA = featurize(repA_s, "repA")
XB, sB = featurize(repB_s, "repB")
XV, sV = featurize(adv_s[:N], "adv")

np.savez_compressed(OUT / "matrices.npz", repA=XA, repB=XB, adv=XV,
                    names=np.array(molhume.feature_names(fingerprint=False, columns="full")))
json.dump({"n": N, "seed": SEED, "standardize": "none",
           "molhume_version": "0.1.1",
           "rdkit": Chem.rdBase.rdkitVersion,
           "repA_source": "data/corpus1m/selected.txt",
           "repB_source": "data/corpus1m/selected.txt (disjoint from repA)",
           "adv_source": "cpp/hard.smi"},
          open(OUT / "matrices_meta.json", "w"), indent=1)
print(f"  -> {OUT/'matrices.npz'}")
