"""Extend figA's embedding matrices: new molecules appended, HUME recomputed for r=4.

    .venv/bin/python research/embed_extend.py            # arms this venv can do
    .venv-figa/bin/python research/embed_extend.py       # the transformer arms

TWO SEPARATE JOBS, deliberately in one script because they must agree on row order.

1. NEW ROWS. `protonate_neutral` introduced 972 molecules the index has never seen. They are
   APPENDED as a third block after [order, background], so every existing row index keeps its
   meaning and the matrices already on disk stay valid for every other edit.

2. HUME IS RECOMPUTED IN FULL. Its default fingerprint radius moved 3 -> 4, so every one of its
   rows changed -- not just the new ones. Appending to the old matrix would have produced a block
   whose first 31,704 rows were r=3 and whose last 972 were r=4, which no check downstream would
   have caught. Arms whose definition did not change are only extended.

`ecfp` stays at radius 2 (it is the ECFP4 baseline and is NOT the package default); `r4cfp` was
already radius 4 and is now the same fingerprint HUME carries, which is why the figure labels it
ECFP8.
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
EMB = FIGA / "embeddings"

RECOMPUTE_ALL = {"hume"}        # arms whose DEFINITION changed, not just their row count


def fp_arm(smis, radius):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    g = rfg.GetMorganGenerator(radius=radius, fpSize=2048, includeChirality=True)
    out = np.zeros((len(smis), 2048), np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = g.GetFingerprintAsNumPy(m)
    return out


#: THE FIGURES SHOW WHAT `pip install mol-hume` GIVES YOU (Leif 2026-10-27): the 622-column
#: `default`, not the 1,269-column `full`. Every plate with a single HUME arm was drawing the
#: full set, which is not what `featurize(smiles)` returns and not what a reader would reproduce.
#: Plates that deliberately contrast widths -- C and SI A -- keep their separate arms.
HUME_COLUMNS = "default"


def hume_arm(smis):
    from rdkit import Chem
    import molhume
    good = [(i, s) for i, s in enumerate(smis) if Chem.MolFromSmiles(s) is not None]
    keep = [i for i, _ in good]
    X = molhume.featurize([s for _, s in good], standardize="none",
                          columns=HUME_COLUMNS, fingerprint=True)
    out = np.full((len(smis), X.shape[1]), np.nan, np.float32)
    out[keep] = np.asarray(X, dtype=np.float32)
    return out


LOCAL = {"ecfp": lambda s: fp_arm(s, 2), "r4cfp": lambda s: fp_arm(s, 4), "hume": hume_arm}


def main() -> None:
    idx = json.loads((FIGA / "smiles_index.json").read_text())
    extra_f = FIGA / "new_molecules_protonate_neutral.json"
    fresh = json.loads(extra_f.read_text()) if extra_f.exists() else []
    # EVERY block, `extra` included. Omitting it appended the same 972 molecules a second time
    # on the next run -- the index doubled to 1,944 and every matrix was suddenly "short" against
    # an index describing rows that did not exist.
    known = set(idx["order"]) | set(idx["background"]) | set(idx.get("extra", []))
    seen = set()
    fresh = [s for s in fresh if s not in known and not (s in seen or seen.add(s))]
    rows = list(idx["order"]) + list(idx["background"]) + list(idx.get("extra", [])) + fresh
    n_old = len(idx["order"]) + len(idx["background"]) + len(idx.get("extra", []))
    print(f"  index: {n_old:,} existing rows + {len(fresh)} new = {len(rows):,}\n")

    want = [a for a in sys.argv[1:]] or sorted(LOCAL)
    for arm in want:
        if arm not in LOCAL:
            print(f"  {arm:<14s} SKIP -- not computable in this interpreter"); continue
        f = EMB / f"{arm}.npz"
        old = np.load(f)["X"] if f.exists() else None
        if arm in RECOMPUTE_ALL or old is None:
            X = LOCAL[arm](rows)
            why = "recomputed in full (definition changed)"
        else:
            if old.shape[0] == len(rows):
                print(f"  {arm:<14s} already {old.shape}, nothing to add"); continue
            X = np.vstack([old, LOCAL[arm](rows[old.shape[0]:])])
            why = f"extended by {len(rows) - old.shape[0]} rows"
        assert X.shape[0] == len(rows), (X.shape, len(rows))
        sha = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        np.savez_compressed(f, X=X.astype(np.float32), n_pair=np.int64(len(idx["order"])),
                            index_sha=np.array(sha))
        print(f"  {arm:<14s} {X.shape}  {why}")

    idx["extra"] = list(idx.get("extra", [])) + fresh
    (FIGA / "smiles_index.json").write_text(json.dumps(idx))
    print(f"\n  smiles_index.json now carries an 'extra' block of {len(idx['extra'])} molecules.")
    missing = sorted(p.stem for p in EMB.glob("*.npz")
                     if np.load(p)["X"].shape[0] != len(rows))
    if missing:
        print(f"  STILL SHORT, need embedding elsewhere: {', '.join(missing)}")


if __name__ == "__main__":
    main()
