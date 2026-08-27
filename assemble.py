"""Assemble surrogate training and evaluation matrices.

Produces, for both the training corpus and the locked benchmark:

    X  = log1p(ECFP-2048 counts) + 767 CORE descriptors     (2815 cols, free at inference)
    Y  = the 98 PREDICT descriptors                         (targets)

Counts follow the 2026-08-27 boundary redraw in blocks.py; they are printed at run time from
blocks.split(), so treat the numbers above as documentation, not as a contract.

Mordred is reused from existing caches (uma100k targets for training, gate1 cache for the
benchmark); only the 217 RDKit descriptors are recomputed, which is a few minutes.

Also emits SMILES so the training split can be scaffold-based and the benchmark rows stay
aligned with their datasets.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

import blocks

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"
BENCH_CACHE = Path("/Users/lsieben/VSCode/ChemTFM_OLD/results/gate1_mordred_cache.npz")
TRAIN_TGT = ROOT / "data" / "uma100k" / "targets"

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
_rd_names = [n for n, _ in Descriptors._descList]
_lut = dict(Descriptors._descList)


def _rd_row(m):
    r = np.empty(len(_rd_names), np.float32)
    for j, n in enumerate(_rd_names):
        try:
            r[j] = _lut[n](m)
        except Exception:
            r[j] = np.nan
    return r


def _rdkit_and_ecfp(smiles, label):
    t0, E, R = time.time(), [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            E.append(np.zeros(2048, np.float32))
            R.append(np.full(len(_rd_names), np.nan, np.float32))
        else:
            E.append(_gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
            R.append(_rd_row(m))
        if (i + 1) % 20000 == 0:
            print(f"    [{label}] {i + 1:,}/{len(smiles):,} ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(E), np.stack(R)


def main() -> None:
    from mordred import Calculator, descriptors as mdesc
    md_desc = list(Calculator(mdesc, ignore_3D=True).descriptors)
    md_pos = {str(d): i for i, d in enumerate(md_desc)}
    fam = {str(d): type(d).__module__.split(".")[-1] for d in md_desc}
    rd_pos = {n: i for i, n in enumerate(_rd_names)}

    sp = blocks.split(fam)
    print(f"CORE {len(sp['core'])} | PREDICT {len(sp['predict'])}")
    OUT.mkdir(parents=True, exist_ok=True)

    def assemble(smiles, MD, label):
        E, RD = _rdkit_and_ecfp(smiles, label)
        def stack(items):
            cols, names, fams = [], [], []
            for s, n, f in items:
                if s == "rdkit" and n in rd_pos:
                    cols.append(RD[:, rd_pos[n]])
                elif s == "mordred" and n in md_pos:
                    cols.append(MD[:, md_pos[n]])
                else:
                    continue
                names.append(f"{s}:{n}"); fams.append(f)
            return np.stack(cols, axis=1).astype(np.float32), np.array(names), np.array(fams)
        CORE, core_names, _ = stack(sp["core"])
        TGT, tgt_names, tgt_fams = stack(sp["predict"])
        X = np.hstack([np.log1p(np.clip(E, 0, None)), CORE]).astype(np.float32)
        print(f"  [{label}] X {X.shape} (2048 ecfp + {CORE.shape[1]} core) | Y {TGT.shape}")
        return X, TGT, tgt_names, tgt_fams

    # --- training corpus -----------------------------------------------------------------
    shards = sorted(TRAIN_TGT.glob("tgt_*.npz"))
    smi, MD = [], []
    for p in shards:
        z = np.load(p, allow_pickle=True)
        smi.extend(list(z["smiles"])); MD.append(z["mordred"])
    MD = np.concatenate(MD)
    print(f"training corpus: {len(smi):,} molecules, Mordred {MD.shape}")
    X, Y, tn, tf = assemble(smi, MD, "train")
    np.savez_compressed(OUT / "train.npz", X=X, Y=Y, smiles=np.array(smi, dtype=object),
                        target_names=tn, target_families=tf)

    # --- benchmark -----------------------------------------------------------------------
    c = dict(np.load(BENCH_CACHE, allow_pickle=True))
    bs = list(c["smiles"])
    print(f"benchmark: {len(bs):,} molecules")
    Xb, Yb, _, _ = assemble(bs, c["md_bench"], "bench")
    np.savez_compressed(OUT / "bench.npz", X=Xb, Y=Yb, smiles=np.array(bs, dtype=object),
                        y=c["y"], offsets=c["offsets"], suite_of=c["suite_of"],
                        name_of=c["name_of"], target_names=tn, target_families=tf)
    print(f"\nwrote {OUT}/train.npz and {OUT}/bench.npz")


if __name__ == "__main__":
    main()
