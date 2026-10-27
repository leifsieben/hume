"""Re-embed figA's CheMeleon and Chemprop arms over the FULL index.

    .venv-chemprop/bin/python research/embed_figa_chemprop.py chemeleon chemprop

NO BOX NEEDED. The CheMeleon checkpoint is already on this machine at
~/.chemprop/chemeleon_mp.pt, and 32,676 molecules is about ninety seconds of CPU for the
pretrained arm and a few seconds for the random-init one. The arms were originally computed on
EC2 because they ran beside everything else; on their own they are a laptop job.

Every row is recomputed rather than appended, for the reason in research/embed_figa_local.py:
`protonate_neutral` pairs an existing molecule with a new one, so a matrix assembled from two
runs would carry the difference between two runs inside the pair difference.

The encoder definitions are imported from verification/embed_pairs.py rather than restated, so
these matrices stay the same function of their input that the cached ones were.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIGA = ROOT / "results" / "figures" / "figA"
sys.path.insert(0, str(ROOT / "verification"))


def main() -> None:
    import embed_pairs as EP
    idx = json.loads((FIGA / "smiles_index.json").read_text())
    rows = list(idx["order"]) + list(idx["background"]) + list(idx.get("extra", []))
    sha = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    print(f"  {len(rows):,} molecules", flush=True)
    for arm in sys.argv[1:]:
        X = np.asarray(EP.ARMS[arm](rows), dtype=np.float32)
        assert X.shape[0] == len(rows), (X.shape, len(rows))
        np.savez_compressed(FIGA / "embeddings" / f"{arm}.npz", X=X,
                            n_pair=np.int64(len(idx["order"])), index_sha=np.array(sha))
        print(f"  {arm:<12s} {X.shape}  written", flush=True)


if __name__ == "__main__":
    main()
