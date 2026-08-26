"""Build a scaffold-diverse 100k UMA embedding set — the training data for the UMA surrogate.

Two stages, both resumable, because this is an overnight job:

  prepare  sample + filter PubChem -> ETKDGv3/MMFF conformers -> conformer shards (~min)
  embed    UMA-small forward per conformer -> 128-d l=0 mean-pooled embedding (~hours)

CORRECTNESS FIX vs ChemTFM_OLD/scripts/uma_embed.py
---------------------------------------------------
That script hardcodes ``at.info = {"charge": 0, "spin": 1}`` with the comment "(toy
molecules)". UMA takes total charge and spin as *global conditioning inputs* — feeding
charge=0 for a molecule that is actually a carboxylate or a quaternary amine asks the model
for the potential-energy surface of a species that does not exist, and the embedding is
correspondingly wrong. A large fraction of ChEMBL/PubChem is charged at the recorded
protonation state, so this matters at scale even though it was harmless on neutral toys.
Here the formal charge is computed per molecule and passed through; radicals (unpaired
electrons, i.e. spin != 1) are skipped rather than silently mislabelled as singlets.

DIVERSITY
---------
Scaffold-stratified round-robin over Bemis-Murcko scaffolds rather than random sampling or
MaxMin. MaxMin at k=100k over a 10M pool is O(n*k) and infeasible; scaffold round-robin is
one pass, and scaffold coverage is the axis the benchmark splits on anyway, so it is the
diversity that bears on generalisation.

ELEMENT / SIZE FILTER
---------------------
Restricted to organic elements and <=60 heavy atoms. UMA is trained across 83 elements but
its mass is inorganic crystals and catalysis; staying in organic space keeps us inside the
OMol25 part of its distribution, and cost scales with atom count.

Usage:
    .venv/bin/python uma_100k.py prepare --n 100000
    .venv/bin/python uma_100k.py embed --workers 3
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

POOL = Path("/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/pubchem_10M.smi")
OUT = Path(__file__).resolve().parent / "data" / "uma100k"
CONF_DIR = OUT / "conformers"
EMB_DIR = OUT / "embeddings"
SHARD = 2000
SEED = 0xF00D

ORGANIC = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
MAX_HEAVY = 60
POOL_SCAN = 3_000_000  # how much of the 10M pool to scan for scaffolds


# ------------------------------------------------------------------------------- prepare


def _acceptable(mol) -> bool:
    if mol.GetNumHeavyAtoms() > MAX_HEAVY or mol.GetNumHeavyAtoms() < 5:
        return False
    if any(a.GetAtomicNum() not in ORGANIC for a in mol.GetAtoms()):
        return False
    # Radicals get a spin multiplicity we are not modelling; skip rather than mislabel.
    if sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) != 0:
        return False
    return True


def _select(n: int, depth: int) -> list[str]:
    """Scaffold-stratified selection: ~n/depth scaffolds, up to `depth` molecules from each.

    ``depth=1`` maximises breadth (every molecule a distinct scaffold) but produces a training
    set with no within-scaffold pairs at all. The surrogate is evaluated on MoleculeACE, whose
    datasets are congeneric series, so a depth-1 set would never show the model how descriptors
    move under a single-atom substitution — the regime the cliff benchmark is made of. depth>1
    trades a little breadth for that local gradient information.
    """
    # The scan is ~8 minutes; cache it so re-selecting at a different depth is instant.
    scaf_cache = OUT / f"scaffolds_{POOL_SCAN}.pkl"
    if scaf_cache.exists():
        with scaf_cache.open("rb") as fh:
            by_scaffold = pickle.load(fh)
        print(f"reusing scaffold map: {len(by_scaffold):,} scaffolds")
        return _pick(by_scaffold, n, depth)

    by_scaffold: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    scanned = 0
    t0 = time.time()
    with POOL.open() as fh:
        for line in fh:
            smi = line.split()[0] if line.strip() else ""
            if not smi:
                continue
            scanned += 1
            if scanned > POOL_SCAN:
                break
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not _acceptable(mol):
                continue
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                scaf = ""
            by_scaffold[scaf].append(can)
            if scanned % 250_000 == 0:
                print(f"  scanned {scanned:,} | {len(seen):,} kept | "
                      f"{len(by_scaffold):,} scaffolds ({time.time() - t0:.0f}s)", flush=True)

    scaf_cache.parent.mkdir(parents=True, exist_ok=True)
    with scaf_cache.open("wb") as fh:
        pickle.dump(dict(by_scaffold), fh)
    print(f"cached scaffold map -> {scaf_cache.name}")
    return _pick(by_scaffold, n, depth)


def _pick(by_scaffold: dict, n: int, depth: int) -> list[str]:
    rng = np.random.default_rng(SEED)
    keys = list(by_scaffold)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(by_scaffold[k])

    # Restrict to ~n/depth scaffolds, preferring those with at least `depth` members. The
    # candidate list must be TRUNCATED, not merely reordered: with all 415k scaffolds in play,
    # a single pass reaches n before any scaffold contributes a second molecule, which
    # silently collapses back to depth 1.
    want = max(1, n // depth)
    rich = [k for k in keys if len(by_scaffold[k]) >= depth]
    poor = [k for k in keys if len(by_scaffold[k]) < depth]
    ordered = (rich + poor)[:want]
    spare = (rich + poor)[want:]

    # Round-robin by pass, so every chosen scaffold contributes its 1st molecule before any
    # contributes its 2nd — rare scaffolds are never crowded out by prolific ones.
    picked, used, passes = [], set(), 0
    while len(picked) < n and passes < depth:
        added = 0
        for k in ordered:
            if passes < len(by_scaffold[k]):
                picked.append(by_scaffold[k][passes])
                used.add(k)
                added += 1
                if len(picked) >= n:
                    break
        if added == 0:
            break
        passes += 1

    # Short because some chosen scaffolds had fewer than `depth` members: top up from spare
    # scaffolds at one molecule each rather than deepening the existing series further.
    i = 0
    while len(picked) < n and i < len(spare):
        picked.append(by_scaffold[spare[i]][0])
        used.add(spare[i])
        i += 1

    print(f"selected {len(picked):,} molecules from {len(used):,} distinct scaffolds "
          f"({len(picked) / max(len(used), 1):.1f} per scaffold, {passes} pass(es)) "
          f"out of a pool of {len(by_scaffold):,} scaffolds")
    return picked


def _conformer(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    charge = Chem.GetFormalCharge(mol)
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = SEED
    if AllChem.EmbedMolecule(mol, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int16)
    return {"smiles": smi, "Z": Z,
            "R": conf.GetPositions().astype(np.float32), "charge": int(charge)}


def cmd_prepare(n: int, workers: int, depth: int) -> None:
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    sel_file = OUT / "selected.txt"
    if sel_file.exists():
        picked = sel_file.read_text().split()
        print(f"reusing {len(picked):,} selected molecules")
    else:
        picked = _select(n, depth)
        sel_file.write_text("\n".join(picked))

    from multiprocessing import Pool

    t0 = time.time()
    for start in range(0, len(picked), SHARD):
        path = CONF_DIR / f"conf_{start:07d}.pkl"
        if path.exists():
            continue
        chunk = picked[start:start + SHARD]
        with Pool(workers) as pool:
            recs = [r for r in pool.map(_conformer, chunk) if r is not None]
        with path.open("wb") as fh:
            pickle.dump(recs, fh)
        print(f"  conformers {start + len(chunk):,}/{len(picked):,} "
              f"({len(recs)}/{len(chunk)} ok, {time.time() - t0:.0f}s)", flush=True)
    print(f"conformers done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------------- embed


def _embed_shard(path: Path) -> None:
    """Embed one conformer shard. Separate process per shard keeps torch threads bounded."""
    import torch
    from ase import Atoms
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate import pretrained_mlip as pm

    out_path = EMB_DIR / (path.stem.replace("conf_", "emb_") + ".npz")
    if out_path.exists():
        return
    with path.open("rb") as fh:
        recs = pickle.load(fh)

    pu = pm.get_predict_unit("uma-s-1p2", device="cpu")
    calc = FAIRChemCalculator(pu, task_name="omol")
    captured: dict = {}
    pu.model.module.backbone.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("node", out["node_embedding"].detach()))

    smiles, vecs = [], []
    for r in recs:
        try:
            at = Atoms(numbers=r["Z"], positions=r["R"])
            at.info = {"charge": r["charge"], "spin": 1}
            at.calc = calc
            at.get_potential_energy()
            v = captured["node"][:, 0, :].mean(dim=0).float().cpu().numpy()
            if not np.isfinite(v).all():
                continue
            smiles.append(r["smiles"])
            vecs.append(v.astype(np.float32))
        except Exception:
            continue
    np.savez_compressed(out_path, smiles=np.array(smiles, dtype=object),
                        emb=np.stack(vecs) if vecs else np.zeros((0, 128), np.float32))
    print(f"  {out_path.name}: {len(vecs)}/{len(recs)}", flush=True)


def cmd_embed(workers: int) -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(CONF_DIR.glob("conf_*.pkl"))
    todo = [p for p in shards
            if not (EMB_DIR / (p.stem.replace("conf_", "emb_") + ".npz")).exists()]
    print(f"{len(shards)} shards, {len(todo)} to embed, {workers} workers")
    # Each worker is single-threaded; parallelism comes from running several shards at once.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    t0 = time.time()
    if workers == 1:
        for p in todo:
            _embed_shard(p)
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            pool.map(_embed_shard, todo)
    done = sum(len(np.load(f, allow_pickle=True)["smiles"]) for f in EMB_DIR.glob("emb_*.npz"))
    dt = time.time() - t0
    print(f"embedded {done:,} molecules in {dt / 3600:.2f} h "
          f"({dt / max(done, 1) * 1000:.0f} ms/mol effective)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "embed"])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--depth", type=int, default=3,
                    help="molecules per scaffold; 1 = pure breadth, >1 adds within-series pairs")
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare(a.n, a.workers, a.depth)
    else:
        cmd_embed(a.workers)
