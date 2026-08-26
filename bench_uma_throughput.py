"""Measure UMA embedding throughput on this machine, to size the 100k pilot.

The 100k UMA run is either a 28-hour local job or a couple of cloud-GPU hours, and the
deciding number — batched UMA-small inference on Apple MPS for small druglike molecules —
is not published anywhere. So measure it.

Pipeline per molecule (same as ChemTFM_OLD/scripts/uma_embed.py, whose hook this reuses):
    SMILES -> AddHs -> ETKDGv3 -> MMFF -> ASE Atoms -> UMA backbone forward
           -> node_embedding (n_atoms, 9, 128) -> l=0 invariant block -> mean-pool -> 128-d

Reports conformer and UMA cost separately, on CPU and MPS, because they scale differently
and only one of them is the bottleneck.

NOTE: RDKit is pinned to 2025.9.2 to match ChemTFM_OLD's pyproject. uma_embed.py keys its
cache by canonical SMILES, so a different RDKit here would silently produce keys that do
not match the model-side venv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from ase import Atoms
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

CORPUS = "/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/chembl_150k.smi"


def conformer(smi: str, seed: int = 0xF00D):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass
    conf = m.GetConformer()
    return (np.array([a.GetAtomicNum() for a in m.GetAtoms()]), conf.GetPositions())


def make_embedder(device: str):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate import pretrained_mlip as pm

    pu = pm.get_predict_unit("uma-s-1p2", device=device)
    calc = FAIRChemCalculator(pu, task_name="omol")
    captured: dict = {}

    def hook(_mod, _inp, out):
        captured["node"] = out["node_embedding"].detach()

    pu.model.module.backbone.register_forward_hook(hook)

    def embed(Z, R):
        at = Atoms(numbers=Z, positions=R)
        at.info = {"charge": 0, "spin": 1}
        at.calc = calc
        at.get_potential_energy()
        return captured["node"][:, 0, :].mean(dim=0).float().cpu().numpy().astype(np.float32)

    return embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--devices", type=str, default="cpu,mps")
    args = ap.parse_args()

    smiles = []
    with open(CORPUS) as fh:
        for line in fh:
            s = line.split()[0] if line.strip() else ""
            if s:
                smiles.append(s)
            if len(smiles) >= args.n * 2:
                break

    # --- conformers (CPU, parallelisable across cores) ---
    t0, confs = time.time(), []
    for s in smiles:
        c = conformer(s)
        if c is not None:
            confs.append((s, c))
        if len(confs) >= args.n:
            break
    t_conf = time.time() - t0
    atoms = np.array([len(c[1][0]) for c in confs])
    print(f"conformers: {len(confs)} ok in {t_conf:.1f}s -> {t_conf / len(confs) * 1000:.0f} ms/mol "
          f"(1 core, contended); heavy+H atoms mean={atoms.mean():.0f} max={atoms.max()}")

    # --- UMA forward, per device ---
    for device in args.devices.split(","):
        try:
            t0 = time.time()
            embed = make_embedder(device)
            t_load = time.time() - t0
            embed(*confs[0][1])  # warm-up: excluded from timing
            t0 = time.time()
            vecs = [embed(*c[1]) for c in confs]
            dt = time.time() - t0
            v = np.stack(vecs)
            print(f"UMA[{device}]: load {t_load:.1f}s | {len(vecs)} mols in {dt:.1f}s "
                  f"-> {dt / len(vecs) * 1000:.0f} ms/mol | {len(vecs) / dt:.1f} mol/s "
                  f"| dim={v.shape[1]} finite={np.isfinite(v).all()}")
            for n, label in ((100_000, "100k"), (1_000_000, "1M")):
                hrs = n * (dt / len(vecs) + t_conf / len(confs)) / 3600
                print(f"    -> {label}: {hrs:.1f} h single-stream (conformers + UMA)")
        except Exception as e:
            print(f"UMA[{device}]: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
