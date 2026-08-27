"""What the pickle boundary costs, against the current one, paired and cold.

METHOD, and every clause of it is load-bearing on this box.

COLD. RDKit caches on the molecule -- Gasteiger charges become atom properties, Crippen
contributions a cached vector -- so a second pass over the same objects measures a dict lookup
instead of the work. This project has published two wrong numbers that way: an 82x-wrong Crippen
cost and a 187.5 us extraction figure that was really 231 cold and 130 warm. Every repetition of
every arm therefore gets its own freshly parsed molecules, and the parse is not inside the timer.

PAIRED. The arms alternate, and the rotation shifts by one each repetition, so a machine that
gets busier partway through slows every arm by roughly the same amount rather than the one that
happened to be running. The comparison is between arms in the same run; a number from one run
against a number from another run means nothing here.

CPU TIME, NOT WALL. The bridge work measured wall-clock spreads of 26-47x on this shared box,
larger than anything being compared.

MEAN +/- SD OVER REPETITION MEANS. Not best-of: the spread is the point when the machine is
contended, and a minimum hides it.

    .venv/bin/python cpp/bench_molpickle.py [n_mols] [reps]
"""
from __future__ import annotations

import sys
import time
from itertools import repeat
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hume import _core                                       # noqa: E402
from hume._extract import (_PICKLE_FLAGS, _rings_csr, extract,      # noqa: E402
                           extract_pickles)

RDLogger.DisableLog("rdApp.*")


def cpu() -> float:
    return time.process_time()


# ---- the arms -------------------------------------------------------------------------------
def a_extract(mols):
    """The current boundary: ~300 Python calls per molecule into RDKit, arrays out."""
    extract(mols)


def a_pickles_only(mols):
    """The Python half of the new boundary, alone: charges, stereo, ToBinary. No per-atom call."""
    extract_pickles(mols)


def a_pickles_no_rings(mols):
    """extract_pickles WITHOUT the ring CSR, so the difference from the arm above IS the rings.

    Not a shippable path -- RingCount needs the ring set and there is no substitute (benzene and
    cyclohexane have identical `nring` vectors and differ on 6 of the 49). It is here because the
    dense-ring decision was taken against an estimate of ~4 us/mol, which counted reading
    `RingInfo` on the 78.7% of molecules `_rings.gate()` does not fire on and counted neither the
    gate itself -- a Python loop over every bond of EVERY molecule -- nor `canon_rings()` at
    104 us on the 21.3% where it does fire. The measured number belongs next to the decision.
    """
    for m in mols:
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
        except Exception:
            m.ClearComputedProps()
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        m.ToBinary(_PICKLE_FLAGS)


def a_rings_only(mols):
    """The ring CSR alone: `_rings.rings_for` over the batch, gate included."""
    _rings_csr(mols)


def a_hgraph_only(mols):
    """The Autocorrelation molecule alone: AddHs, its OWN Gasteiger charges, its own blob.

    540 of the emitted columns describe `Chem.AddHs(m)`, and its charges are not derivable from the
    heavy-atom molecule's -- PEOE is not invariant to making hydrogens explicit (5,221 of 42,359
    heavy atoms move, and 7,395 of 38,326 hydrogen charges differ from `_GasteigerHCharge / nH`).
    So a second molecule is charged and serialised. This is what that costs.
    """
    for m in mols:
        mh = Chem.AddHs(m)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mh)
        except Exception:
            mh.ClearComputedProps()
        mh.ToBinary(_PICKLE_FLAGS)


def a_pickle_boundary(mols):
    """The new boundary end to end -- serialise, parse, and materialise the SAME eight arrays.

    This is the like-for-like against a_extract: both produce the boundary arrays. The shipped
    path does not build them (blocks_from_pickles keeps the parse in C++ vectors), so this arm
    is an upper bound on what the new boundary costs.
    """
    _core.pickle_extract(extract_pickles(mols).blobs)


def a_hybrid(mols):
    """THE ROAD NOT TAKEN, measured rather than argued about.

    `_GasteigerCharge` is a computed property, so carrying it costs the ComputedProps flag and
    the blob goes 452 -> 5034 bytes/mol. The alternative is the small blob plus ONE per-atom
    Python pass for the charges -- which is what this arm is. It is a couple of microseconds
    cheaper and it puts an atom-wrapper list and a read pass back in the path, which is the
    thing the whole exercise exists to remove. The gap is the price of the stated goal, and it
    is here so that price stays a number instead of a belief.
    """
    lean = (Chem.PropertyPickleOptions.PrivateProps | Chem.PropertyPickleOptions.AtomProps
            | Chem.PropertyPickleOptions.NoConformers)
    gd = Chem.Atom.GetDoubleProp
    for m in mols:
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
        except Exception:
            m.ClearComputedProps()
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        m.ToBinary(lean)
        ats = list(map(m.GetAtomWithIdx, range(m.GetNumAtoms())))
        list(map(gd, ats, repeat("_GasteigerCharge")))


def a_blocks_api(mols):
    """182 columns, old boundary."""
    b = extract(mols)
    _core.blocks(b.atom_off, b.bond_off, b.chg_ok, b.atom_i, b.atom_d, b.bond_i, b.bond_s,
                 b.bond_d)


def a_blocks_pickle(mols):
    """182 columns, new boundary. What hume.featurize_blocks() now runs."""
    _core.blocks_from_pickles(extract_pickles(mols).blobs)


ARMS = [
    ("extract()                    BOUNDARY, old", a_extract),
    ("extract_pickles()            python half", a_pickles_only),
    ("  of which: ToBinary, no rings", a_pickles_no_rings),
    ("  of which: the ring CSR alone", a_rings_only),
    ("  of which: the H-added pickle", a_hgraph_only),
    ("extract_pickles + parse      BOUNDARY, new", a_pickle_boundary),
    ("lean pickle + python charges  NOT TAKEN", a_hybrid),
    ("featurize_blocks reader=api  END TO END", a_blocks_api),
    ("featurize_blocks reader=pickle END TO END", a_blocks_pickle),
]


def main() -> int:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    smis = ROOT.joinpath("cpp", "mols.smi").read_text().split()
    rng = np.random.default_rng(0)
    pick = [smis[i] for i in rng.choice(len(smis), min(len(smis), n_want), replace=False)]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in pick) if m is not None]
    natoms = float(np.mean([m.GetNumAtoms() for m in mols]))
    del mols

    print(f"rdkit {rdkit.__version__}   numpy {np.__version__}   python {sys.version.split()[0]}")
    print(f"{len(pick)} molecules, mean {natoms:.1f} heavy atoms, {reps} repetitions, CPU time\n")

    times: dict[str, list[float]] = {name: [] for name, _ in ARMS}
    parse_us = []
    for rep in range(reps):
        for name, fn in ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]:
            t0 = cpu()
            mols = [Chem.MolFromSmiles(s) for s in pick]
            t1 = cpu()
            fn(mols)
            t2 = cpu()
            parse_us.append((t1 - t0) / len(pick) * 1e6)
            times[name].append((t2 - t1) / len(pick) * 1e6)
            del mols

    print(f"  {'arm':44s} {'us/mol':>9s} {'+/- SD':>8s} {'us/atom':>9s}")
    for name, _fn in ARMS:
        v = np.asarray(times[name])
        print(f"  {name:44s} {v.mean():9.2f} {v.std(ddof=1):8.2f} {v.mean() / natoms:9.3f}")
    p = np.asarray(parse_us)
    print(f"  {'(MolFromSmiles, not in any arm)':44s} {p.mean():9.2f} {p.std(ddof=1):8.2f}")

    mean = {name: float(np.asarray(times[name]).mean()) for name, _ in ARMS}
    old, new = mean[ARMS[0][0]], mean[ARMS[5][0]]
    e_old, e_new = mean[ARMS[7][0]], mean[ARMS[8][0]]
    print(f"\n  boundary   {old:7.2f} -> {new:7.2f} us/mol   ({old / new:.2f}x)")
    print(f"  end to end {e_old:7.2f} -> {e_new:7.2f} us/mol   ({e_old / e_new:.2f}x)")

    mols = [Chem.MolFromSmiles(s) for s in pick]
    blobs = extract_pickles(mols).blobs
    print(f"\n  pickle size: mean {np.mean([len(b) for b in blobs]):.0f} bytes/mol, "
          f"max {max(len(b) for b in blobs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
