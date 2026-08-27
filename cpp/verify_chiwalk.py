"""Are src/hume_core/chi.h and src/hume_core/topomisc.h exactly mordred?

    Chi 40 + WalkCount 6 + Constitutional 4 + TopologicalIndex 2 + WienerIndex 2 + ABCIndex 1
    = 55 columns, on cpp/hard.smi (100,000 molecules).

RUN IT LIKE THIS, AND ONLY LIKE THIS:

    uv run --isolated --no-project --python 3.11 --with "mordred==1.2.0" \
           --with "rdkit==2025.9.2" --with "numpy==1.26.4" python cpp/verify_chiwalk.py all

`--isolated --no-project` matters as much as the versions: without it the project's
`[tool.uv] constraint-dependencies` block pins numpy to 2.4.6, which mordred 1.2.0 cannot use,
and uv then resolves MORDRED DOWN TO 0.6.0 rather than erroring. Every mode prints the resolved
versions FROM THE PROCESS THAT PRODUCED THE NUMBERS, because a verify log without them on it is
not evidence.

TWO SHIMS. Both restore an alias numpy deleted; neither changes an arithmetic result.

  * `np.float = float`. mordred/ABCIndex.py's last line is `return np.float(np.sum(...))`, and
    numpy removed `np.float` in 1.24. Under the pin ABCGG therefore raises AttributeError for
    EVERY molecule and yields an Error object -- there is no oracle at all as shipped. `np.float`
    was only ever a deprecated alias for the builtin `float`, so restoring it changes nothing but
    whether the function returns. This is a DEAD ALIAS, not an ill-posed definition: the intended
    value is unambiguous and `perm` below shows it is a function of the molecule.
  * `np.product = np.prod`, needed only to import `mordred.descriptors`; none of these 55
    descriptors calls it. Same shim verify_topo3.py carries.

WHAT THE REFERENCE IS. mordred itself, and every one of the 55 descriptor objects is pulled OUT
OF MORDRED'S OWN `preset()` GENERATORS by name -- never constructed from a tuple retyped here --
so `check_spec()` failing means the installed mordred no longer emits a column we claim to port.

THE MOLECULE IS `Chem.RemoveHs(mol, updateExplicitCount=True)`, which is literally what
mordred/_base/context.py does for every descriptor with `explicit_hydrogens = False`, and all 55
of these are. Note `updateExplicitCount=True`: plain `Chem.RemoveHs(mol)` is NOT the same call,
though on this corpus the two agree.

MODES
    dump      cpp/hard.smi -> cpp/chiwalk_mols.txt        (input for the C++)
    ref       cpp/hard.smi -> cpp/chiwalk_ref.txt         (mordred, 55 columns)
    compare   cpp/chiwalk_cpp.txt vs cpp/chiwalk_ref.txt, per column
    perm      HOUSE RULE 1. Is each column a function of the molecule? mordred vs mordred under
              (a) ATOM + BOND shuffling -- the molecule is rebuilt atom by atom and bond by bond,
              because Chem.RenumberAtoms leaves the bond list order alone and an atom-only screen
              is therefore too weak; (b) a Kekule/aromatic round trip, the axis that made
              InformationContent and ETA ill-posed; (c) a canonical-SMILES round trip, which is a
              CONTROL that must show zero, not a probe.
    selftest  the two numeric transliterations, checked against the library they copy:
              npPairwiseSum vs np.sum, and C++ std::log vs np.log on the integers WalkCount feeds
              it. Both must be 0 mismatches or the exactness claim below is about the wrong thing.
    benchpy   mordred us/mol on a subsample, for the C++/Python comparison
    all       dump, ref, build, run, compare
"""
from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SMI = Path(os.environ.get("CHIWALK_SMI", HERE / "hard.smi"))
MOLS = HERE / "chiwalk_mols.txt"
REF = HERE / "chiwalk_ref.txt"
CPP = HERE / "chiwalk_cpp.txt"
BIN = HERE / "chiwalk"
NPROC = int(os.environ.get("CHIWALK_NPROC", "8"))


def _numpy_shim() -> None:
    import numpy as np
    if not hasattr(np, "product"):
        np.product = np.prod        # mordred/MolecularDistanceEdge.py, module scope
    if not hasattr(np, "float"):
        np.float = float            # mordred/ABCIndex.py; see the docstring
    if not hasattr(np, "float_"):
        np.float_ = np.float64


def versions(tag: str) -> None:
    import numpy as np
    import rdkit
    import mordred
    print(f"[{tag}] mordred {mordred.__version__}  rdkit {rdkit.__version__}  "
          f"numpy {np.__version__}  python {sys.version.split()[0]}", flush=True)
    if mordred.__version__ != "1.2.0":
        raise SystemExit(f"WRONG MORDRED: {mordred.__version__}. uv resolved it down; re-read "
                         f"the docstring and pass --isolated --no-project.")


# ---------------------------------------------------------------------------------------------
# The 55 columns, in the order the C++ emits them (chi::COLS then topomisc::COLS).
# ---------------------------------------------------------------------------------------------
CHI_NAMES = (
    "Xch-4d Xch-5d Xch-6d Xch-7d Xch-5dv Xch-6dv Xch-7dv "
    "Xc-3d Xc-4d Xc-5d Xc-6d Xc-3dv Xc-4dv Xc-5dv "
    "Xpc-4d Xpc-5d Xpc-4dv Xpc-5dv Xpc-6dv "
    "Xp-2d Xp-4d Xp-5d Xp-6d AXp-1d AXp-2d AXp-3d AXp-4d AXp-5d AXp-6d AXp-7d "
    "Xp-5dv Xp-6dv AXp-0dv AXp-1dv AXp-2dv AXp-3dv AXp-4dv AXp-5dv AXp-6dv AXp-7dv").split()
MISC_NAMES = ("MWC03 MWC05 MWC08 SRW05 SRW07 TSRW10 Sp MZ Mv Mp Diameter TopoShapeIndex "
              "WPath WPol ABCGG").split()
NAMES = CHI_NAMES + MISC_NAMES
BLOCK = [("Chi", 0, 40), ("WalkCount", 40, 46), ("Constitutional", 46, 50),
         ("TopologicalIndex", 50, 52), ("WienerIndex", 52, 54), ("ABCIndex", 54, 55)]


def descriptors():
    """The 55 mordred objects in the C++'s emit order, taken from mordred's own preset()."""
    from mordred import ABCIndex, Chi, Constitutional, TopologicalIndex, WalkCount, WienerIndex
    pre = {}
    for gen in (Chi.Chi.preset(None),
                WalkCount.WalkCount.preset(None),
                Constitutional.ConstitutionalSum.preset(None),
                Constitutional.ConstitutionalMean.preset(None),
                TopologicalIndex.Radius.preset(None),
                TopologicalIndex.Diameter.preset(None),
                TopologicalIndex.TopologicalShapeIndex.preset(None),
                TopologicalIndex.PetitjeanIndex.preset(None),
                WienerIndex.WienerIndex.preset(None),
                ABCIndex.ABCIndex.preset(None),
                ABCIndex.ABCGGIndex.preset(None)):
        for d in gen:
            pre[str(d)] = d
    missing = [n for n in NAMES if n not in pre]
    if missing:
        raise SystemExit(f"SPEC DRIFT: {missing} are not produced by mordred's preset()s")
    return [pre[n] for n in NAMES]


def check_spec() -> None:
    """DRIFT GUARD. Every name the binary emits must be str() of a live mordred object that
    mordred's OWN preset() generator produced, in this order, and the six families must
    contribute exactly 40 / 6 / 4 / 2 / 2 / 1."""
    ds = descriptors()
    assert [str(d) for d in ds] == NAMES, "emit order does not match mordred's objects"
    got = {}
    for d in ds:
        got[type(d).__module__.split(".")[-1]] = got.get(type(d).__module__.split(".")[-1], 0) + 1
    want = {"Chi": 40, "WalkCount": 6, "Constitutional": 4,
            "TopologicalIndex": 2, "WienerIndex": 2, "ABCIndex": 1}
    assert got == want, f"family counts moved: {got} != {want}"
    print(f"[check_spec] 55 columns, {want}", flush=True)


# ---------------------------------------------------------------------------------------------
def read_smiles():
    return [l.split()[0] for l in SMI.read_text().splitlines() if l.strip()]


def prepare(smi):
    """The molecule a mordred USER hands the Calculator -- deliberately NOT pre-processed.

    mordred's own Context applies `Chem.AddHs(mol)` or `Chem.RemoveHs(mol,
    updateExplicitCount=True)` per descriptor, from `explicit_hydrogens`, and among these 55 it
    goes BOTH WAYS: the four Constitutional columns get the hydrogen-added molecule and the other
    51 get the hydrogen-suppressed one (Descriptor.explicit_hydrogens defaults to True and
    Constitutional.py never overrides it). Pre-applying RemoveHs here would quietly make the
    reference agree with a C++ port that had the same misconception."""
    from rdkit import Chem
    return Chem.MolFromSmiles(smi)


def dump_mols() -> None:
    versions("dump")
    smis = read_smiles()
    out = [str(len(smis))]
    changed = 0
    from rdkit import Chem
    for s in smis:
        m0 = Chem.MolFromSmiles(s)
        if m0 is None:
            raise SystemExit(f"unparseable SMILES in the corpus: {s!r}")
        m = Chem.RemoveHs(m0, updateExplicitCount=True)
        if m.GetNumAtoms() != m0.GetNumAtoms():
            changed += 1
        out.append(f"{m.GetNumAtoms()} {m.GetNumBonds()}")
        for a in m.GetAtoms():
            out.append(f"{a.GetAtomicNum()} {a.GetFormalCharge()} {a.GetTotalNumHs()}")
        for b in m.GetBonds():
            out.append(f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()}")
    MOLS.write_text("\n".join(out) + "\n")
    nH = sum(1 for s in smis if "[2H]" in s or "[3H]" in s)
    print(f"[dump] {len(smis)} molecules -> {MOLS}; RemoveHs changed the atom count on "
          f"{changed}; {nH} SMILES carry an isotopic hydrogen (AXp-0dv sees those, nothing "
          f"else does)", flush=True)


def _ref_chunk(arg):
    lo, smis = arg
    _numpy_shim()
    import numpy as np
    from mordred import Calculator
    calc = Calculator(descriptors(), ignore_3D=True)
    rows = []
    with np.errstate(all="ignore"):
        for s in smis:
            m = prepare(s)
            vals = calc(m)
            rows.append(" ".join(
                "%.17g" % (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                           else float("nan")) for v in vals))
    return lo, rows


def build_ref() -> None:
    versions("ref")
    check_spec()
    smis = read_smiles()
    t0 = time.time()
    chunk = 250
    jobs = [(i, smis[i:i + chunk]) for i in range(0, len(smis), chunk)]
    import multiprocessing as mp
    res = {}
    with mp.Pool(NPROC) as pool:
        for k, (lo, rows) in enumerate(pool.imap_unordered(_ref_chunk, jobs)):
            res[lo] = rows
            if (k + 1) % 40 == 0:
                done = sum(len(v) for v in res.values())
                print(f"  {done}/{len(smis)}  {time.time()-t0:.0f}s", flush=True)
    with REF.open("w") as f:
        for lo in sorted(res):
            for r in res[lo]:
                f.write(r + "\n")
    print(f"[ref] {len(smis)} molecules x 55 columns -> {REF} in {time.time()-t0:.0f}s "
          f"({NPROC} processes)", flush=True)


# ---------------------------------------------------------------------------------------------
def compare() -> None:
    versions("compare")
    import numpy as np
    A = np.loadtxt(CPP)
    B = np.loadtxt(REF)
    if A.shape != B.shape:
        raise SystemExit(f"shape mismatch: c++ {A.shape} vs mordred {B.shape}")
    n = A.shape[0]
    print(f"\n{'column':16s} {'exact':>9s} {'nan-agree':>9s} {'maxrel':>12s}  status")
    print("-" * 62)
    worst = 0.0
    allexact = True
    for c, name in enumerate(NAMES):
        a, b = A[:, c], B[:, c]
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            print(f"{name:16s} {'-':>9s} {int((na==nb).sum()):>9d} {'-':>12s}  NaN PATTERN DIFFERS")
            allexact = False
            continue
        ok = ~na
        eq = int((a[ok] == b[ok]).sum()) + int(na.sum())
        if eq == n:
            print(f"{name:16s} {eq:>9d} {int(na.sum()):>9d} {0.0:>12.3e}  EXACT")
            continue
        allexact = False
        den = np.where(b[ok] == 0, 1.0, np.abs(b[ok]))
        rel = np.abs(a[ok] - b[ok]) / den
        mx = float(rel.max())
        worst = max(worst, mx)
        bad = int((a[ok] != b[ok]).sum())
        print(f"{name:16s} {eq:>9d} {int(na.sum()):>9d} {mx:>12.3e}  {bad} differ")
    print("-" * 62)
    print(f"{n} molecules.  {'ALL 55 COLUMNS BIT-EXACT' if allexact else f'max rel dev {worst:.3e}'}")


# ---------------------------------------------------------------------------------------------
# HOUSE RULE 1
# ---------------------------------------------------------------------------------------------
def atom_bond_shuffled(m, rng):
    """A fresh molecule with the atoms permuted AND the bond list order randomised.

    Chem.RenumberAtoms permutes atoms and LEAVES THE BOND LIST ORDER ALONE, so it cannot move
    anything that reads bonds in order. Rebuilding through an RWMol moves both axes. Everything
    that survives a SMILES round trip is copied explicitly; the sanitise then re-perceives."""
    from rdkit import Chem
    n = m.GetNumAtoms()
    perm = list(range(n))
    rng.shuffle(perm)                       # perm[new] = old
    inv = [0] * n
    for new, old in enumerate(perm):
        inv[old] = new
    rw = Chem.RWMol()
    for new in range(n):
        a = m.GetAtomWithIdx(perm[new])
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetNoImplicit(a.GetNoImplicit())
        na.SetNumExplicitHs(a.GetNumExplicitHs())
        na.SetIsAromatic(a.GetIsAromatic())
        na.SetIsotope(a.GetIsotope())
        na.SetChiralTag(a.GetChiralTag())
        rw.AddAtom(na)
    bonds = list(m.GetBonds())
    rng.shuffle(bonds)
    for b in bonds:
        rw.AddBond(inv[b.GetBeginAtomIdx()], inv[b.GetEndAtomIdx()], b.GetBondType())
    q = rw.GetMol()
    Chem.SanitizeMol(q)
    Chem.AssignStereochemistry(q, cleanIt=True, force=True)
    return q


def kekulized(m):
    """The same molecule presented as a Kekule structure and re-perceived. This is the axis that
    made InformationContent and ExtendedTopochemicalAtom ill-posed."""
    from rdkit import Chem
    q = Chem.Mol(m)
    Chem.Kekulize(q, clearAromaticFlags=True)
    s = Chem.MolToSmiles(q, kekuleSmiles=True)
    r = Chem.MolFromSmiles(s)
    return r


def canon_roundtrip(m):
    """CONTROL, not a probe: this reproduces the canonical numbering, so it should show zero."""
    from rdkit import Chem
    return Chem.MolFromSmiles(Chem.MolToSmiles(m))


def _perm_one(arg):
    idx, smi, ntrial = arg
    _numpy_shim()
    import numpy as np
    from mordred import Calculator
    calc = Calculator(descriptors(), ignore_3D=True)

    def row(mm):
        with np.errstate(all="ignore"):
            return np.array([float(v) if isinstance(v, (int, float, np.floating, np.integer))
                             else float("nan") for v in calc(mm)])

    m = prepare(smi)
    if m is None or m.GetNumAtoms() == 0:
        return None
    base = row(m)
    rng = random.Random(1000 + idx)
    moved = {"shuffle": np.zeros(len(NAMES), bool),
             "kekule": np.zeros(len(NAMES), bool),
             "canon": np.zeros(len(NAMES), bool)}
    mag = {k: np.zeros(len(NAMES)) for k in moved}
    variants = []
    for _ in range(ntrial):
        try:
            variants.append(("shuffle", atom_bond_shuffled(m, rng)))
        except Exception:
            pass
    for tag, fn in (("kekule", kekulized), ("canon", canon_roundtrip)):
        try:
            q = fn(m)
            if q is not None:
                variants.append((tag, q))
        except Exception:
            pass
    for tag, q in variants:
        r = row(q)
        d = ~((base == r) | (np.isnan(base) & np.isnan(r)))
        moved[tag] |= d
        # THE SIZE OF THE MOVE IS THE WHOLE QUESTION. A relative move at 1e-16 is a summation
        # ORDER -- the multiset of terms is a graph invariant and only the association changed --
        # and is the TopologicalCharge situation. A move at 1e-3 would be a second ANSWER, i.e.
        # the InformationContent situation, and would mean the definition is ill-posed. Reporting
        # only a count cannot tell those apart, and a count is what an earlier screen reported.
        with np.errstate(all="ignore"):
            den = np.where((base == 0) | ~np.isfinite(base), 1.0, np.abs(base))
            rel = np.where(d & np.isfinite(base) & np.isfinite(r), np.abs(base - r) / den, 0.0)
        mag[tag] = np.maximum(mag[tag], rel)
    return idx, smi, moved, mag


def perm(nmol=400, ntrial=5) -> None:
    versions("perm")
    check_spec()
    import multiprocessing as mp
    import numpy as np
    smis = read_smiles()
    rng = random.Random(7)
    sample = rng.sample(range(len(smis)), min(nmol, len(smis)))
    jobs = [(i, smis[i], ntrial) for i in sample]
    counts = {k: np.zeros(len(NAMES), int) for k in ("shuffle", "kekule", "canon")}
    mags = {k: np.zeros(len(NAMES)) for k in counts}
    examples = {}
    done = 0
    t0 = time.time()
    with mp.Pool(NPROC) as pool:
        for out in pool.imap_unordered(_perm_one, jobs):
            done += 1
            if out is None:
                continue
            idx, smi, moved, mag = out
            for k, v in moved.items():
                counts[k] += v.astype(int)
                mags[k] = np.maximum(mags[k], mag[k])
                for c in np.nonzero(v)[0]:
                    examples.setdefault((k, int(c)), smi)
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"\nHOUSE RULE 1 SCREEN -- mordred vs mordred, {len(jobs)} molecules, "
          f"{ntrial} atom+bond shuffles each, plus one Kekule and one canonical round trip.")
    print(f"{'column':16s} {'shuf n':>7s} {'shuf rel':>10s} {'kek n':>6s} {'kek rel':>10s} "
          f"{'can n':>6s} {'can rel':>10s}  verdict")
    print("-" * 108)
    worst = 0.0
    for c, name in enumerate(NAMES):
        s, k, cn = counts["shuffle"][c], counts["kekule"][c], counts["canon"][c]
        ms, mk, mc = mags["shuffle"][c], mags["kekule"][c], mags["canon"][c]
        mx = max(ms, mk, mc)
        worst = max(worst, mx)
        if not (s or k or cn):
            v = "INVARIANT"
        elif mx < 1e-12:
            v = "last-bit only"
        else:
            v = "*** ILL-POSED: A SECOND ANSWER ***"
        print(f"{name:16s} {s:>7d} {ms:>10.2e} {k:>6d} {mk:>10.2e} {cn:>6d} {mc:>10.2e}  {v}")
    print("-" * 108)
    print(f"worst relative movement anywhere: {worst:.3e}")
    print("READ THE MAGNITUDE, NOT THE COUNT. Everything here is a sum of irrational terms, so a "
          "count of 'moved' says only that the ASSOCIATION changed. 1e-16 is a summation order "
          "(the TopologicalCharge situation, matched exactly on a given numbering); 1e-3 would "
          "be a second answer (the InformationContent situation).")
    print("The canon columns are a CONTROL: they reproduce the canonical numbering, so a "
          "non-zero there is RDKit re-perceiving, not the descriptor.")


# ---------------------------------------------------------------------------------------------
def selftest() -> None:
    """The two numeric transliterations in topomisc.h, checked against the library they copy."""
    versions("selftest")
    import numpy as np
    rng = np.random.default_rng(20260827)
    lens = list(range(0, 40)) + [63, 64, 120, 127, 128, 129, 136, 200, 255, 256, 257, 512, 1000]
    cases = []
    for L in lens:
        for _ in range(3):
            cases.append(rng.standard_normal(L) * rng.choice([1e-3, 1.0, 1e3]))
    pw = HERE / "chiwalk_pw.txt"
    with pw.open("w") as f:
        f.write(f"{len(cases)}\n")
        for a in cases:
            f.write(f"{len(a)}\n")
            f.write(" ".join("%.17g" % v for v in a) + ("\n" if len(a) else "\n"))
    got = subprocess.run([str(BIN), "pwtest", str(pw)], capture_output=True, text=True, check=True)
    cpp = [float.fromhex(x) if x.startswith(("0x", "-0x")) else float(x)
           for x in got.stdout.split()]
    bad = sum(1 for a, c in zip(cases, cpp) if np.sum(a) != c and not
              (np.isnan(np.sum(a)) and np.isnan(c)))
    print(f"[selftest] npPairwiseSum vs np.sum: {len(cases)-bad}/{len(cases)} bit-identical")

    ints = sorted({0, 1, 2, 3} | {int(x) for x in rng.integers(1, 40_000_000, 40000)})
    lg = HERE / "chiwalk_log.txt"
    lg.write_text(f"{len(ints)}\n" + "\n".join(str(i) for i in ints) + "\n")
    got = subprocess.run([str(BIN), "logtest", str(lg)], capture_output=True, text=True, check=True)
    cpp = [float(x) for x in got.stdout.split()]
    ref = np.log(np.array(ints, dtype=np.int64))
    badl = int((ref != np.array(cpp)).sum())
    print(f"[selftest] std::log vs np.log on {len(ints)} integers: "
          f"{len(ints)-badl}/{len(ints)} bit-identical")
    if bad or badl:
        raise SystemExit("a transliteration does not reproduce its library -- fix before "
                         "quoting any exactness number")


def benchpy(n=200) -> None:
    versions("benchpy")
    _numpy_shim()
    import numpy as np
    from mordred import Calculator
    calc = Calculator(descriptors(), ignore_3D=True)
    smis = read_smiles()
    rng = random.Random(3)
    sample = [smis[i] for i in rng.sample(range(len(smis)), n)]
    mols = [prepare(s) for s in sample]
    reps = []
    with np.errstate(all="ignore"):
        for _ in range(3):
            t0 = time.process_time()
            for m in mols:
                calc(m)
            reps.append((time.process_time() - t0) / n * 1e6)
    reps.sort()
    print(f"[benchpy] mordred, 55 columns, {n} molecules: median {reps[1]:.0f} us/mol "
          f"(min {reps[0]:.0f}, max {reps[2]:.0f})  CONTENDED")


def build_cpp() -> None:
    cmd = ["c++", "-std=c++17", "-O3", "-o", str(BIN), str(HERE / "chiwalk.cpp")]
    print("[build] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "dump":
        dump_mols()
    elif mode == "ref":
        build_ref()
    elif mode == "compare":
        compare()
    elif mode == "perm":
        perm(*(int(x) for x in sys.argv[2:4]))
    elif mode == "selftest":
        selftest()
    elif mode == "benchpy":
        benchpy(*(int(x) for x in sys.argv[2:3]))
    elif mode == "build":
        build_cpp()
    elif mode == "all":
        dump_mols()
        build_cpp()
        subprocess.run([str(BIN), "dump", str(MOLS), str(CPP)], check=True)
        selftest()
        build_ref()
        compare()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
