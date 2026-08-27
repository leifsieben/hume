"""Build surrogate training data: ECFP in, the full expensive descriptor block out.

Runs on the SAME molecules as uma_100k.py (reads its selected.txt), so one corpus serves
both surrogates and the input featurisation is shared.

Cost basis for the design, measured on this machine (contended, so pessimistic; ratios hold):

    ECFP-2048 counts       29 us/mol      <- input, the only thing computed at inference
    ErG-315               102 us/mol   \\
    RDKit-96              976 us/mol    >- targets: 68,333 us/mol replaced by one forward pass
    Mordred 2D (1613)  67,255 us/mol   /

Note that benchmark_results.md records the RDKit-96 channel as "~us/mol"; it is actually
~1 ms/mol, i.e. 34x ECFP and 11.3 core-days per billion molecules. That error is why an
earlier draft of this pipeline wrongly treated RDKit-96 as free and fed it in as input.

Targets are stored raw. Standardisation and the NaN mask belong to the training script,
which must fit them on the train split only.

Usage:
    .venv/bin/python build_targets.py --workers 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdReducedGraphs

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).resolve().parent / "data" / "uma100k"
TGT_DIR = OUT / "targets"
SHARD = 5000

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def _worker(smi: str):
    """One molecule -> (ecfp, rdkit96, erg, mordred). Returns None if unparseable."""
    import _vendor  # noqa: F401  - puts vendor/chemtfm on sys.path
    from chemtfm.feat.descriptors import descriptors as rdkit96

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        ecfp = _gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        d96 = np.asarray(rdkit96(mol), dtype=np.float32)
        erg = np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
    except Exception:
        return None
    return smi, ecfp, d96, erg


def _mordred_block(mols, calc, nproc):
    rows = []
    for res in calc.map(mols, nproc=nproc, quiet=True):
        rows.append([v if isinstance(v, (int, float)) else np.nan for v in res.values()])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    from mordred import Calculator, descriptors as mdesc
    from multiprocessing import Pool

    calc = Calculator(mdesc, ignore_3D=True)
    TGT_DIR.mkdir(parents=True, exist_ok=True)
    smiles = (OUT / "selected.txt").read_text().split()
    print(f"{len(smiles):,} molecules; {len(calc.descriptors)} Mordred descriptors")

    t0 = time.time()
    for start in range(0, len(smiles), SHARD):
        path = TGT_DIR / f"tgt_{start:07d}.npz"
        if path.exists():
            continue
        chunk = smiles[start:start + SHARD]
        with Pool(a.workers) as pool:
            recs = [r for r in pool.map(_worker, chunk) if r is not None]
        if not recs:
            continue
        smi = [r[0] for r in recs]
        mols = [Chem.MolFromSmiles(s) for s in smi]
        md = _mordred_block(mols, calc, a.workers)
        np.savez_compressed(
            path,
            smiles=np.array(smi, dtype=object),
            ecfp=np.stack([r[1] for r in recs]),
            rdkit96=np.stack([r[2] for r in recs]),
            erg=np.stack([r[3] for r in recs]),
            mordred=md,
        )
        done = start + len(chunk)
        rate = (time.time() - t0) / done * 1000
        print(f"  {done:,}/{len(smiles):,} ({len(recs)} ok) {rate:.1f} ms/mol wall "
              f"| eta {(len(smiles) - done) * rate / 1000 / 60:.0f} min", flush=True)
    print(f"targets done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
