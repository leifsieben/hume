"""Extract the exact molecules behind the three singleton mismatches.

verify_hume.py prints only the WORST offender per column and recomputes all 45 descriptors
for 98,905 molecules to get there. Chasing three columns does not need the other 42, so this
recomputes only Chi4n, Chi4v and linearity and dumps EVERY disagreeing row, not just the
worst -- "one molecule" is a claim about the count, and the count is what has to be checked.

    python cpp/find_mismatch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import GraphDescriptors as GD

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import conjugation  # noqa: E402

# column indices into values_hume.txt, from verify_hume.py's SPEC order
COLS = {"Chi4n": 4, "Chi4v": 5, "linearity": 15}
TOL = {"Chi4n": (1e-9, 1e-12), "Chi4v": (1e-9, 1e-12), "linearity": (3e-6, 1e-6)}
LIN = conjugation.NAMES.index("linearity")


def main() -> None:
    smis = HERE.joinpath("mols.smi").read_text().split()
    got = np.loadtxt(HERE / "values_hume.txt", ndmin=2, usecols=sorted(COLS.values()))
    order = {c: k for k, c in enumerate(sorted(COLS.values()))}
    print(f"{len(smis):,} molecules loaded")

    bad = {k: [] for k in COLS}
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            ref = {"Chi4n": GD.Chi4n(m), "Chi4v": GD.Chi4v(m),
                   "linearity": float(conjugation.featurize(m)[LIN])}
        except Exception:
            continue
        for k, v in ref.items():
            if not np.isfinite(v):
                continue
            a = got[i, order[COLS[k]]]
            rtol, atol = TOL[k]
            if not np.isfinite(a) or abs(a - v) > atol + rtol * abs(v):
                bad[k].append((s, a, v))
        if i % 20000 == 0:
            print(f"  {i:,} ...", flush=True)

    print()
    for k in COLS:
        print(f"{k}: {len(bad[k])} disagreements")
        for s, a, v in bad[k][:10]:
            m = Chem.MolFromSmiles(s)
            print(f"    {s}")
            print(f"      atoms {m.GetNumAtoms()} rings {m.GetRingInfo().NumRings()} "
                  f"frags {len(Chem.GetMolFrags(m))}")
            print(f"      c++ {a!r}  ref {v!r}  reldev {abs(a-v)/max(abs(v),1e-12):.3e}")
    Path(HERE / "mismatch.txt").write_text(
        "\n".join(f"{k}\t{s}\t{a!r}\t{v!r}" for k in COLS for s, a, v in bad[k]) + "\n")
    print(f"\nwrote {HERE/'mismatch.txt'}")


if __name__ == "__main__":
    main()
