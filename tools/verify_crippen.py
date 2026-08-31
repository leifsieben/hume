"""Is the extension's Crippen typer exactly RDKit's, atom by atom?

`molhume.featurize_blocks` no longer calls `rdMolDescriptors._CalcCrippenContribs`: the (logP, MR)
pair is computed inside the extension by src/hume_core/crippen_typer.h, from the integers
`_extract.py` was already sending. That is the single largest saving in the pipeline and also the
only place where the bridge stopped being a pass-through, so it needs its own oracle rather than
being inferred from four BCUT2D columns agreeing.

The typer itself was verified in cpp/crippen.cpp against a text export on 2,869,048 atoms. This
checks the thing that is actually shipped -- the header, compiled into molhume._core, fed by
_extract.py's arrays -- against RDKit called on the same live molecules. It is a strictly
stronger comparison than watching the 182 columns, because two atoms with compensating errors
cancel in a molecular sum and a per-atom comparison admits no such cancellation.

EXACT, not approximate: both sides are table lookups from the same shipped Crippen.txt, so any
tolerance could only ever hide a wrong row.

    .venv/bin/python tools/verify_crippen.py [n_mols] [smiles_file]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molhume import _core                    # noqa: E402
from molhume._extract import extract         # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
BATCH = 4096


def main() -> int:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "cpp" / "mols.smi"
    smis = [s for s in src.read_text().split("\n") if s][:n_want]

    n_at = n_bad = n_bad_mol = 0
    shown = 0
    for lo in range(0, len(smis), BATCH):
        chunk = smis[lo:lo + BATCH]
        mols = [Chem.MolFromSmiles(s) for s in chunk]
        if any(m is None for m in mols):
            raise ValueError("unparseable SMILES in the corpus")
        b = extract(mols)
        got = _core.crippen(b.atom_off, b.bond_off, b.atom_i, b.bond_i)
        want = np.concatenate([np.asarray(rdMolDescriptors._CalcCrippenContribs(m),
                                          dtype=np.float64).reshape(-1, 2)
                               for m in mols])
        n_at += len(want)
        bad = np.flatnonzero((got != want).any(axis=1))
        n_bad += len(bad)
        if len(bad):
            n_bad_mol += len(np.unique(np.searchsorted(b.atom_off, bad, side="right") - 1))
            for i in bad[:max(0, 20 - shown)]:
                k = int(np.searchsorted(b.atom_off, i, side="right") - 1)
                shown += 1
                print(f"  mol {lo + k} atom {i - b.atom_off[k]}  Z={b.atom_i[i, 0]} "
                      f"arom={b.atom_i[i, 5]} chg={b.atom_i[i, 3]} "
                      f"got {tuple(got[i])} want {tuple(want[i])}   {chunk[k]}")

    print("\nhume._core Crippen vs rdMolDescriptors._CalcCrippenContribs")
    print(f"  {len(smis)} molecules, {n_at} atoms, source {src}")
    print(f"  exact : {n_at - n_bad} / {n_at}   ({100.0 * (n_at - n_bad) / max(n_at, 1):.6f}%)")
    if n_bad:
        print(f"  MISMATCH on {n_bad} atoms in {n_bad_mol} molecules")
        return 1
    print("  ALL EXACT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
