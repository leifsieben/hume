"""Is `_core.all_from_pickles` feeding each family the graph its own harness verified it on?

WHAT THIS DOES AND DOES NOT CHECK. The six headers behind the new entry point are each already
verified by their owner's harness -- vsa_bins.h against RDKit on 100,000 molecules, estate_typer.h
per ATOM on 2.87M, ringcount/pathcount/topocharge against mordred, infocontent.h under
renumbering. None of that says anything about the WIRING: whether bindings.cpp's all_row() hands
each of them the right column of `atom_i`, the right bond code, the right E-state index, the
right ring lists. A transposed column there is a wrong descriptor with no symptom, and it would
pass every one of those harnesses, because none of them runs through this path.

So this checks the WIRING, against an oracle outside it, family by family:

  VSA (66)          RDKit's own Descriptors, in this process, on the same molecules. 65 of the
                    66 have an RDKit answer; TopoPSA is mordred's and has none here.
  EState (158)      RDKit's EState.AtomTypes.TypeAtoms + EState.EStateIndices, recombined into
                    N<t> and S<t> exactly as mordred's AtomTypeEState does. This is the check
                    that the `S*` columns really are weighted by BlockWork::ES.
  Autocorr (486)    `./cpp/ac`, the binary that wrote the evidence, fed a SECOND independent
                    construction of the hydrogen-added graph (charges, mordred's getter and all)
                    built here from RDKit. Graded at %.12g because that is the oracle's own
                    output format; see check_autocorr for why that is the right limit.
  RingCount (49)    the owner's own binary, cpp/ringcount, fed the same molecules through
  PathCount (11)    cpp/topo_io.h's text format, written here from the molecule as given and the
  TopoCharge (21)   ring set `_rings.rings_for` supplies -- the shipped inputs, not tidier ones.
                    If the wiring hands the C++ a different graph or different rings, these
                    disagree. Bit-exact for RingCount and PathCount; TopologicalCharge is graded
                    at 1e-12 because its last bits are a dgemm accumulation order and the two
                    paths sum in different orders (see cpp/verify_topo3.py).
  InfoContent (42)  invariance under `Chem.RenumberAtoms`, which is what infocontent.h claims and
                    the only well-posed thing to claim -- mordred's own IC is numbering-dependent.
  Fragments (76)    RDKit's own `Descriptors`, in this process, on the same molecules -- the
                    oracle OUTSIDE the code path, not cpp/frag's standalone harness, which shares
                    src/hume_core/frag_matcher.h with the wiring and so could only confirm that
                    the matcher agrees with itself. These are integer counts, so "exact" is
                    bitwise with nowhere to hide. This is also the only check that the tenth
                    `atom_i` column (`tval`, SMARTS `v`) survives the pickle path: `fr_Imine`,
                    `NumHDonors` and `NumHAcceptors` are the three columns that read it.

    .venv/bin/python cpp/verify_wiring.py [n_mols]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import EState as _EState
from rdkit.Chem.EState import AtomTypes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hume import _core                        # noqa: E402
from hume._extract import extract_pickles      # noqa: E402
from hume._columns import COLUMNS              # noqa: E402
from hume._rings import rings_for              # noqa: E402

RDLogger.DisableLog("rdApp.*")
OFF = _core.ALL_OFFSETS
TAIL = list(_core.all_column_names_tail())
NAMES = list(COLUMNS) + TAIL


def all_cols(mols) -> np.ndarray:
    p = extract_pickles(mols)
    return _core.all_from_pickles(p.blobs, p.rings.ring_moff, p.rings.ring_ptr,
                                  p.rings.ring_at, p.h_blobs)


def col(name: str) -> int:
    return NAMES.index(name)


def report(label: str, got: np.ndarray, want: np.ndarray, names, tol: float = 0.0,
           note: str = "") -> int:
    """-> number of failing columns.

    THE MAX DEVIATION IS ALWAYS PRINTED, tolerance or not. House rule 5 in PORT_STATUS.md: a
    tolerance is allowed only alongside the max observed deviation and a floating-point reason,
    and a column reported as EXACT under a tolerance should have to show how much slack it used.

    THE MEASURE IS |g - w| / max(|w|, 1) -- relative above 1, absolute below. A pure relative
    error is the wrong ruler for a column like MinAbsEStateIndex, which is a minimum of absolute
    values and so lives arbitrarily close to zero: two answers a single ULP apart there score
    1e-12 relative and 1e-16 absolute, and only the second says anything about the arithmetic.
    """
    bad = 0
    worst = 0.0
    for j, nm in enumerate(names):
        g, w = got[:, j], want[:, j]
        fin = np.isfinite(g) & np.isfinite(w)
        dev = float(np.max(np.abs(g[fin] - w[fin]) /
                           np.maximum(np.abs(w[fin]), 1.0))) if fin.any() else 0.0
        worst = max(worst, dev)
        if tol == 0.0:
            ok = np.array_equal(g.view(np.uint64), w.view(np.uint64))
        else:
            ok = bool((np.isfinite(g) == np.isfinite(w)).all()) and dev <= tol
        if not ok:
            bad += 1
            n_off = int(np.count_nonzero(g.view(np.uint64) != w.view(np.uint64)))
            print(f"    {nm:22s} DIFFERS on {n_off} molecules, max dev {dev:.3e}")
    kind = "bitwise" if tol == 0.0 else f"dev <= {tol:g}"
    print(f"  {label:26s} {'EXACT' if not bad else f'{bad} / {len(names)} COLUMNS DIFFER'}"
          f"   ({len(names)} cols x {got.shape[0]} mols, {kind}, max dev {worst:.3e}){note}")
    return bad


# --------------------------------------------------------------------------------------------
# the three pure-topology families, through their owner's binary
# --------------------------------------------------------------------------------------------
def write_topo_io(mols, path: Path) -> None:
    """cpp/topo_io.h's format, filled from exactly what the shipped path supplies.

    THE RINGS COME FROM `_rings.rings_for`, NOT FROM `Chem.GetSymmSSSR`. That is the whole point
    of the comparison: the wiring hands the C++ the REPAIRED ring set, and feeding this file the
    raw one would either flag a false mismatch on a gated molecule or -- worse -- agree for the
    wrong reason on an ungated one. The molecule is taken as given, without `Chem.RemoveHs`, for
    the same reason: that is _extract.py's contract with its caller (see its docstring), and this
    file must exercise the contract rather than a tidier version of it.
    """
    out = [str(len(mols))]
    for m in mols:
        rings = [list(r) for r in rings_for(m)]
        out.append(f"{m.GetNumAtoms()} {m.GetNumBonds()} {len(rings)}")
        for a in m.GetAtoms():
            out.append(f"{a.GetAtomicNum()} {int(a.GetIsAromatic())}")
        for b in m.GetBonds():
            out.append(f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} "
                       f"{b.GetBondTypeAsDouble():.17g}")
        for r in rings:
            out.append(str(len(r)) + " " + " ".join(str(a) for a in r))
    path.write_text("\n".join(out) + "\n")


def write_ac_io(mols, path: Path) -> None:
    """cpp/export_ac.py's format, rebuilt here so `./cpp/ac` sees the same molecules we do.

    Every line of this is export_ac.py's, including mordred's charge getter with the sum inside
    the conditional and the -1e30 sentinel for a non-finite charge (libc++'s `istream >> double`
    will not parse the token "nan", which is the old export desync wearing a quieter disguise).
    Nothing here is shared with the wiring: this is a second, independent construction of the
    hydrogen-added graph, which is the point.
    """
    from rdkit.Chem import rdPartialCharges

    C_MISSING = -1e30
    out = [str(len(mols))]
    for m in mols:
        mh = Chem.AddHs(m)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mh)
        except Exception:
            mh.ClearComputedProps()
        rows = []
        for a in mh.GetAtoms():
            if not a.HasProp("_GasteigerHCharge"):
                c = 0.0
            else:
                c = a.GetDoubleProp("_GasteigerCharge") + a.GetDoubleProp("_GasteigerHCharge")
            if not np.isfinite(c):
                c = C_MISSING
            # %.17g, NOT export_ac.py's %.12g. The exporter's precision is a property of its
            # text file, not of the descriptor: feeding `./ac` a charge rounded to 12 digits
            # makes the 54 `c` columns differ from the wiring in the 12th digit and says nothing
            # about whether the graph is right. %.17g round-trips a float64 exactly, so the
            # comparison below is against the same numbers the wiring used. (This is the same
            # effect PACKAGING.md records for the old %.10g Gasteiger export, one digit worse.)
            rows.append(f"{a.GetAtomicNum()} {a.GetFormalCharge()} {a.GetTotalNumHs()} {c:.17g}")
        bonds = [f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()}" for b in mh.GetBonds()]
        out.append(f"{mh.GetNumAtoms()} {len(bonds)}\n" + "\n".join(rows) +
                   ("\n" + "\n".join(bonds) if bonds else ""))
    path.write_text("\n".join(out) + "\n")


def check_autocorr(mols, X, tmp: Path) -> int:
    """The 486 Autocorrelation columns against `./cpp/ac`, the binary that produced the evidence.

    GRADED AT %.12g, NOT BITWISE, and the reason is the oracle's format rather than the
    arithmetic: `./ac verify` writes cpp/values_ac.txt with `%.12g`, and that file is the
    verified artifact, so widening it would change what the evidence is. That the ARITHMETIC is
    unchanged by the header refactor is established separately and exactly -- cpp/ac.cpp now
    includes src/hume_core/autocorr.h instead of carrying its own copy, and values_ac.txt over
    all 98,905 molecules is byte-identical (same md5) before and after. What is left for this to
    catch is the wiring: a wrong graph, a wrong charge, a wrong `nh`, and every one of those
    moves a value far above the twelfth digit.
    """
    exe = ROOT / "cpp" / "ac"
    if not exe.exists():
        print(f"  {'autocorr vs cpp/ac':26s} SKIPPED -- {exe} is not built "
              f"(c++ -O3 -std=c++17 -o cpp/ac cpp/ac.cpp)")
        return 0
    inp = tmp / "ac_in.txt"
    write_ac_io(mols, inp)
    # cwd=tmp because `ac verify` hard-codes values_ac.txt as its OUTPUT name, and the real one
    # in cpp/ is 645 MB of evidence that must not be clobbered by a verification run.
    r = subprocess.run([str(exe), "verify", str(inp)], capture_output=True, text=True, cwd=tmp)
    if r.returncode != 0:
        print(f"  {'autocorr vs cpp/ac':26s} SKIPPED -- binary failed: {r.stderr.strip()[:120]}")
        return 0
    want = [ln.split() for ln in (tmp / "values_ac.txt").read_text().strip().split("\n")]
    lo = OFF["autocorr"]
    # The family's own width -- the next offset above it, not "everything to the end". This used
    # to read OFF["end"], which was the same number only while Autocorrelation happened to be the
    # last family in the layout; adding the fragment columns after it broke that silently.
    n_cols = min(v for v in OFF.values() if v > lo) - lo
    bad_cells = 0
    bad_cols: dict[int, float] = {}
    for i, row in enumerate(want):
        if len(row) != n_cols:
            raise ValueError(f"cpp/ac emitted {len(row)} columns, the wiring has {n_cols}")
        for j, tok in enumerate(row):
            g = X[i, lo + j]
            if f"{g:.12g}" != tok:
                bad_cells += 1
                w = float(tok)
                dev = abs(g - w) / max(abs(w), 1.0) if np.isfinite(g) and np.isfinite(w) else 1.0
                bad_cols[j] = max(bad_cols.get(j, 0.0), dev)
    for j in sorted(bad_cols)[:10]:
        print(f"    {NAMES[lo + j]:22s} DIFFERS, max dev {bad_cols[j]:.3e}")
    print(f"  {'autocorr vs cpp/ac':26s} "
          f"{'EXACT' if not bad_cells else f'{len(bad_cols)} / {n_cols} COLUMNS DIFFER'}"
          f"   ({n_cols} cols x {len(want)} mols, %.12g -- the oracle's format)")
    return len(bad_cols)


def run_binary(name: str, inp: Path, tmp: Path, n_cols: int, n_mols: int) -> np.ndarray | None:
    exe = ROOT / "cpp" / name
    if not exe.exists():
        print(f"  {name:22s} SKIPPED -- {exe} is not built "
              f"(c++ -O3 -std=c++17 -o cpp/{name} cpp/{name}.cpp)")
        return None
    out = tmp / f"{name}.txt"
    r = subprocess.run([str(exe), "dump", str(inp), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {name:22s} SKIPPED -- binary failed: {r.stderr.strip()[:120]}")
        return None
    v = np.array([float(x) for x in out.read_text().split()], dtype=np.float64)
    return v.reshape(n_mols, n_cols)


# --------------------------------------------------------------------------------------------
def main() -> int:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"rdkit {rdkit.__version__}   numpy {np.__version__}   "
          f"python {sys.version.split()[0]}")
    print(f"{_core.N_ALL_COLS} columns, offsets {dict(OFF)}\n")

    smis = [s for s in (ROOT / "cpp" / "hard.smi").read_text().split("\n") if s]
    rng = np.random.default_rng(0)
    pick = [smis[i] for i in rng.choice(len(smis), min(len(smis), n_want), replace=False)]
    mols = [Chem.MolFromSmiles(s) for s in pick]
    if any(m is None for m in mols):
        raise ValueError("unparseable SMILES in the corpus")
    X = all_cols(mols)
    print(f"{len(mols)} molecules from cpp/hard.smi\n")

    bad = 0

    # THE TWO REASONS A COLUMN HERE IS NOT BITWISE, both established before this wiring existed
    # and neither of them a wiring defect. They are separated out rather than absorbed into one
    # loose tolerance, because a single global rtol would also hide a real transposition.
    #
    #   1. THE NON-FINITE GASTEIGER CONTRACT. src/hume/_extract.py's docstring: elements PEOE has
    #      no parameters for (Se, Ge, As here) get 0.0 in the charge column and the molecule gets
    #      chg_ok = 0, so that a nan cannot propagate through BCUT2D's eigensolver. RDKit bins the
    #      nan instead. The charge-derived VSA columns therefore cannot agree on those molecules
    #      and are compared on the rest, with the exclusion counted.
    #   2. THE E-STATE INDEX IS SUMMED IN A DIFFERENT ORDER. hume_blocks.h's estate_from() is a
    #      C++ double loop; RDKit's EStateIndices is numpy. Same formula, different accumulation
    #      order, last bit only -- the harness prints the max deviation rather than asserting the
    #      claim.
    ok_mol = np.asarray(_core.pickle_extract(extract_pickles(mols).blobs)[2]) == 1
    n_uncharged = int((~ok_mol).sum())
    CHARGED = tuple(f"PEOE_VSA{i}" for i in range(1, 15))
    ES_TOL = 1e-13   # see report()'s note on the measure; observed max is ~2.5e-13/1e-15

    # ---- VSA, against RDKit itself -----------------------------------------------------------
    vsa_names = [n for n in TAIL[:66] if n != "TopoPSA"]
    want = np.empty((len(mols), len(vsa_names)))
    fns = dict(Descriptors._descList)
    for j, nm in enumerate(vsa_names):
        f = fns[nm]
        for i, m in enumerate(mols):
            want[i, j] = f(Chem.Mol(m))      # a fresh copy: RDKit caches Crippen on the molecule
    got = X[:, [col(n) for n in vsa_names]]
    plain = [j for j, n in enumerate(vsa_names) if n not in CHARGED and "EState" not in n]
    charged = [j for j, n in enumerate(vsa_names) if n in CHARGED]
    esdep = [j for j, n in enumerate(vsa_names) if "EState" in n]
    bad += report("VSA vs RDKit", got[:, plain], want[:, plain], [vsa_names[j] for j in plain])
    bad += report("VSA charge cols vs RDKit", got[np.ix_(ok_mol, charged)],
                  want[np.ix_(ok_mol, charged)], [vsa_names[j] for j in charged],
                  note=f"  [{n_uncharged} chg_ok=0 molecules excluded]")
    bad += report("VSA EState cols vs RDKit", got[:, esdep], want[:, esdep],
                  [vsa_names[j] for j in esdep], tol=ES_TOL)

    # ---- EState N* / S*, against RDKit's own typer and index ----------------------------------
    est_names = TAIL[66:66 + 158]
    types = [t[1:] for t in est_names[:79]]
    want = np.zeros((len(mols), len(est_names)))
    for i, m in enumerate(mols):
        ta = AtomTypes.TypeAtoms(m)
        idx = _EState.EStateIndices(m)
        pos = {t: j for j, t in enumerate(types)}
        for a, ts in enumerate(ta):
            for t in ts:
                j = pos[t]
                want[i, j] += 1.0
                want[i, 79 + j] += idx[a]
    got = X[:, [col(n) for n in est_names]]
    # The 79 COUNT columns need the typer only and never touch the index, so they are bitwise.
    bad += report("EState N* vs RDKit", got[:, :79], want[:, :79], est_names[:79])
    bad += report("EState S* vs RDKit", got[:, 79:], want[:, 79:], est_names[79:], tol=ES_TOL)

    # ---- RingCount / PathCount / TopologicalCharge, against their own binaries ----------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        inp = tmp / "topo_in.txt"
        write_topo_io(mols, inp)
        for name, off_key, n_cols, tol in (("ringcount", "ringcount", 49, 0.0),
                                           ("pathcount", "pathcount", 11, 0.0),
                                           ("topocharge", "topocharge", 21, 1e-12)):
            want = run_binary(name, inp, tmp, n_cols, len(mols))
            if want is None:
                continue
            lo = OFF[off_key]
            names = NAMES[lo:lo + n_cols]
            bad += report(f"{name} vs cpp/{name}", X[:, lo:lo + n_cols], want, names, tol)
        bad += check_autocorr(mols, X, tmp)

    # ---- InformationContent: numbering invariance, which is the only well-posed claim ---------
    # The family's own width, not "everything after it" -- Autocorrelation now sits below
    # InformationContent in the layout, and its accumulation order is numbering-dependent at the
    # 1e-14 level (a pairwise sum over atoms), so folding it in here would test a claim nobody
    # makes and drown the one that is being made.
    lo = OFF["infocontent"]
    n_ic = min(v for v in OFF.values() if v > lo) - lo
    perm = []
    for i, m in enumerate(mols):
        p = [int(x) for x in rng.permutation(m.GetNumAtoms())]
        perm.append(Chem.RenumberAtoms(m, p))
    Y = all_cols(perm)
    bad += report("InfoContent renumbered", Y[:, lo:lo + n_ic], X[:, lo:lo + n_ic],
                  NAMES[lo:lo + n_ic])

    # ---- rdkit_core fragments, against RDKit's own Descriptors ------------------------------
    # `fn(m)` on the molecule as given, exactly as cpp/verify_frag.py's `verify` grades the
    # standalone harness -- none of these caches anything on the molecule the way Crippen does.
    lo = OFF["frag"]
    frag_names = NAMES[lo:OFF["end"]]
    want = np.empty((len(mols), len(frag_names)))
    for j, nm in enumerate(frag_names):
        f = fns[nm]
        for i, m in enumerate(mols):
            want[i, j] = float(f(m))
    got = X[:, lo:OFF["end"]]
    bad += report("fragments vs RDKit", got, want, frag_names)
    # PER COLUMN, printed whether or not it passed. These are integer counts: "exact" means every
    # molecule, and a family-level EXACT line hides which of the 76 were ever exercised. The
    # `nonzero` column is that -- a pattern no corpus molecule matches is reported as exact on a
    # column of zeros, which is worth seeing rather than inferring.
    print(f"\n  per-column, {len(mols)} molecules from cpp/hard.smi:")
    for j, nm in enumerate(frag_names):
        n_ok = int(np.count_nonzero(got[:, j] == want[:, j]))
        nz = int(np.count_nonzero(want[:, j]))
        print(f"    {nm:24s} {n_ok:6d} / {len(mols):6d}"
              f"   {'EXACT' if n_ok == len(mols) else 'MISMATCH'}"
              f"   nonzero on {nz}")

    print("\nWIRING EXACT -- every family sees the graph its own harness verified it on"
          if not bad else f"\n{bad} COLUMNS DISAGREE")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
