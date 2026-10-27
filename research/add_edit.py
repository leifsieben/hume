"""Add ONE edit mode to an existing pair set, without disturbing the ones already embedded.

    .venv/bin/python research/add_edit.py protonate_neutral

WHY THIS EXISTS RATHER THAN RE-RUNNING make_pairs.py. That script writes pairs.json wholesale
from a shuffled pool, so re-running it to gain one mode redraws every other mode as well. The
embeddings in results/figures/figA/embeddings/ are keyed by row index into smiles_index.json and
cost GPU hours across eleven arms; a redraw silently invalidates all of them, and the index_sha
stored beside each matrix is the only thing that would have caught it.

This reuses the SAME pool, the SAME seed, the SAME ordering and the SAME guards as main(), takes
the first N_PER_EDIT molecules the new edit accepts, and appends. The background reservation is
read from disk rather than recomputed, so sigma does not move either.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pairs as MP  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "figA"


def main() -> None:
    name = sys.argv[1]
    fn = MP.EDITS.get(name) or getattr(MP, name, None)
    if fn is None:
        raise SystemExit(f"{name!r} is not an edit in make_pairs.EDITS and not a function there")

    pj = json.loads((OUT / "pairs.json").read_text())
    pairs, counts = pj["pairs"], pj["counts"]
    if name in counts:
        raise SystemExit(f"{name!r} already has {counts[name]} pairs in pairs.json; "
                         f"delete them first if you mean to regenerate")

    # The identical pool construction as make_pairs.main(), seed included.
    random.seed(0)
    d = np.load(ROOT / "data" / "surrogate" / "bench.npz", allow_pickle=True)
    pool = [s for s in d["smiles"]]
    random.shuffle(pool)
    mols = [(s, MP._mol(s)) for s in pool]
    mols = [(s, m) for s, m in mols if m is not None and 8 <= m.GetNumAtoms() <= 60]
    seen, uniq = set(), []
    for _, m in mols:
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen.add(c); uniq.append((c, m))
    mols = uniq

    background = json.loads((OUT / "background.json").read_text())
    bgset = set(background)
    assert len(background) == MP.N_BACKGROUND, (len(background), MP.N_BACKGROUND)
    mols = mols[MP.N_BACKGROUND:]            # the reservation is read, never recomputed

    got = shattered = collided = noop = 0
    new = []
    for s, m in mols:
        if got >= MP.N_PER_EDIT:
            break
        try:
            e = fn(m)
            if e is None:
                continue
            ma, mb = e if isinstance(e, tuple) else (m, e)
            Chem.SanitizeMol(mb)
            if ma is not m:
                Chem.SanitizeMol(ma)
            a, b = Chem.MolToSmiles(ma), Chem.MolToSmiles(mb)
        except Exception:
            continue
        if a == b:
            noop += 1; continue
        if not MP._intact(ma, mb):
            shattered += 1; continue
        if b in bgset:
            collided += 1; continue
        new.append({"edit": name, "cls": "chemical", "a": a, "b": b})
        got += 1

    print(f"  {name}: {got} pairs"
          + (f"   ({shattered} rejected: fragmented)" if shattered else "")
          + (f"   ({collided} rejected: product is a background molecule)" if collided else "")
          + (f"   ({noop} no-ops)" if noop else "")
          + ("" if got >= MP.N_PER_EDIT else "   <- SHORT"))

    idx = json.loads((OUT / "smiles_index.json").read_text())
    known = set(idx["order"]) | set(idx["background"])
    fresh = []
    for p in new:
        for s in (p["a"], p["b"]):
            if s not in known:
                known.add(s); fresh.append(s)
    print(f"  {len(fresh)} molecules are new and need embedding across every arm")

    (OUT / "pairs.json").write_text(json.dumps(
        {"counts": {**counts, name: got}, "pairs": pairs + new}))
    (OUT / f"new_molecules_{name}.json").write_text(json.dumps(fresh))
    print(f"  -> pairs.json now {len(pairs) + got} pairs; new SMILES in "
          f"new_molecules_{name}.json")
    print(f"  NOTE: smiles_index.json and the embeddings are NOT touched. Until the new rows are "
          f"embedded and appended, {name} has no vectors and must be skipped by the scorers.")


if __name__ == "__main__":
    main()
