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
  Autocorr (540)    TWO oracles, because one of them cannot decide the question. `./cpp/ac`, the
                    binary that wrote the evidence, fed a SECOND independent construction of the
                    hydrogen-added graph (charges, mordred's getter and all) built here from
                    RDKit -- that grades the GRAPH, and only the graph, since `cpp/ac` includes
                    the same src/hume_core/autocorr.h the wiring does and could otherwise only
                    agree with itself. Then MORDRED ITSELF, in this process, on the same
                    molecules: its own AddHs, its own Gasteiger charges, its own accumulation.
                    Reported per weight, with mordred's tenth weight `Z` -- the 52 columns that
                    were the family's last gap in the 865 -- on its own line, so "the nine did
                    not move" and "the tenth is right" are two readable facts rather than one
                    merged pass line.
  RingCount (49)    the owner's own binary, cpp/ringcount, fed the same molecules through
  PathCount (11)    cpp/topo_io.h's text format, written here from the molecule as given and the
  TopoCharge (21)   ring set `_rings.rings_for` supplies -- the shipped inputs, not tidier ones.
                    If the wiring hands the C++ a different graph or different rings, these
                    disagree. Bit-exact for RingCount and PathCount; TopologicalCharge is graded
                    at 1e-12 because its last bits are a dgemm accumulation order and the two
                    paths sum in different orders (see cpp/verify_topo3.py).
  InfoContent (42)  invariance under `Chem.RenumberAtoms`, which is what infocontent.h claims and
                    the only well-posed thing to claim -- mordred's own IC is numbering-dependent.
  Chi (40)          mordred's own `Chi` objects, pulled out of mordred's `preset()` generators by
  TopoMisc (15)     name and evaluated IN THIS PROCESS on the same molecules. Needs the mordred
  Constit (41)      environment -- see MORDRED below -- and is SKIPPED with a loud line when
  SLogP (1)         mordred is not importable, so the RDKit-graded families still run in `.venv`.
                    `qed` and `SPS` are excluded from the 43: they are RDKit's, not mordred's,
                    and are graded against RDKit by `check_qed` / `check_sps` in EITHER
                    environment. `check_qed` also recovers the alert count the C++ used from the
                    emitted float and grades it as an integer, because a wrong count would show
                    up in the composite as a small float difference.
  rdkcore (19)      RDKit's own `Descriptors`, in this process: the thirteen ring predicates,
                    HeavyAtomMolWt, FractionCSP3, Phi and the three Morgan fingerprint densities.
                    Graded bitwise, with the 32-in-100,000 repaired-ring-set population split out
                    and reported the way `constit`'s is.
  Fragments (76)    RDKit's own `Descriptors`, in this process, on the same molecules -- the
                    oracle OUTSIDE the code path, not cpp/frag's standalone harness, which shares
                    src/hume_core/frag_matcher.h with the wiring and so could only confirm that
                    the matcher agrees with itself. These are integer counts, so "exact" is
                    bitwise with nowhere to hide. This is also the only check that the tenth
                    `atom_i` column (`tval`, SMARTS `v`) survives the pickle path: `fr_Imine`,
                    `NumHDonors` and `NumHAcceptors` are the three columns that read it.

    .venv/bin/python cpp/verify_wiring.py [n_mols]

MORDRED. Three of the families above have no RDKit oracle at all -- they are mordred's, and
mordred 1.2.0 needs python 3.11 (it imports `distutils`) and numpy 1.x, neither of which the
pinned `.venv` has. So this file runs in EITHER environment and grades what that environment can
answer, printing which. To grade the mordred families, install this package into a python-3.11
environment that has mordred and run the same command there:

    uv venv --python 3.11 .venv-mordred
    uv pip install --python .venv-mordred/bin/python "mordred==1.2.0" "rdkit==2025.9.2" \
                   "numpy==1.26.4" "networkx"
    uv pip install -e . --python .venv-mordred/bin/python --no-deps --no-build-isolation-package hume
    .venv-mordred/bin/python cpp/verify_wiring.py 5000

THE ORACLE IS NEVER cpp/chiwalk OR cpp/constit. Those binaries include the same headers the
wiring does and can therefore only confirm that the arithmetic agrees with itself; what is under
test here is whether all_row() hands each header the right graph. mordred is asked directly, in
process, on the same RDKit molecule objects.

TWO NUMPY SHIMS, both dead aliases rather than behaviour changes, both already carried by
cpp/verify_chiwalk.py and cpp/verify_topo3.py: `np.float = float` (mordred/ABCIndex.py's last
line, and without it ABCGG raises AttributeError on EVERY molecule under the pin, so there is no
oracle at all) and `np.product = np.prod` (module scope in mordred/MolecularDistanceEdge.py,
needed only to import `mordred.descriptors`).
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
                                  p.rings.ring_at, p.h_blobs, p.stereo_a, p.stereo_b)


def col(name: str) -> int:
    return NAMES.index(name)


def report(label: str, got: np.ndarray, want: np.ndarray, names, tol: float = 0.0,
           note: str = "", nan_eq: bool = False) -> int:
    """-> number of failing columns.

    THE MAX DEVIATION IS ALWAYS PRINTED, tolerance or not. House rule 5 in PORT_STATUS.md: a
    tolerance is allowed only alongside the max observed deviation and a floating-point reason,
    and a column reported as EXACT under a tolerance should have to show how much slack it used.

    THE MEASURE IS |g - w| / max(|w|, 1) -- relative above 1, absolute below. A pure relative
    error is the wrong ruler for a column like MinAbsEStateIndex, which is a minimum of absolute
    values and so lives arbitrarily close to zero: two answers a single ULP apart there score
    1e-12 relative and 1e-16 absolute, and only the second says anything about the arithmetic.

    `nan_eq` KEEPS THE COMPARISON BITWISE AND STOPS IT FAILING ON A SIGN BIT NOBODY DEFINED. The
    mordred families below produce NaN legitimately and often -- chi.h's `c <= 0` abort, mordred's
    ZeroDivisionError on an empty subgraph list, ABCGG on a bondless graph -- and there is no
    specification anywhere for whether that NaN comes out 0x7ff8.. or 0xfff8... `float("nan")` on
    the python side and `0.0/0.0` on the C++ side can differ in the SIGN BIT alone, which the
    uint64 comparison reports as every column differing on thousands of molecules. Under `nan_eq`
    two NaNs match and everything else must still be bit-for-bit identical -- and WHICH cells are
    NaN is still compared exactly, so a column that went NaN when it should not have still fails.
    """
    bad = 0
    worst = 0.0
    for j, nm in enumerate(names):
        g, w = got[:, j], want[:, j]
        fin = np.isfinite(g) & np.isfinite(w)
        dev = float(np.max(np.abs(g[fin] - w[fin]) /
                           np.maximum(np.abs(w[fin]), 1.0))) if fin.any() else 0.0
        worst = max(worst, dev)
        same = g.view(np.uint64) == w.view(np.uint64)
        if nan_eq:
            same = same | (np.isnan(g) & np.isnan(w))
        if tol == 0.0:
            ok = bool(same.all())
        else:
            ok = bool((np.isfinite(g) == np.isfinite(w)).all()) and dev <= tol
        if not ok:
            bad += 1
            n_off = int(np.count_nonzero(~same))
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


def check_autocorr_mordred(mols, X) -> int:
    """The 540 Autocorrelation columns against MORDRED ITSELF, in this process.

    WHY THIS EXISTS ALONGSIDE check_autocorr. That one feeds `./cpp/ac` a second, independent
    H-added graph -- which tests the GRAPH the wiring builds, and nothing else, because `cpp/ac`
    and the extension both `#include src/hume_core/autocorr.h` and so can only ever agree with
    each other on the ARITHMETIC. This one asks mordred, so the wiring is graded end to end
    against the thing it claims to reproduce: mordred does its own `Chem.AddHs`, its own
    `ComputeGasteigerCharges` on that graph, and its own accumulation, and none of it is ours.

    A DIFFERENT COMPARISON FROM cpp/verify_ac.py, on purpose. That file grades cpp/values_ac.txt,
    the 98,905-molecule artifact, at the oracle's own `%.12g`. This grades the MATRIX THE PACKAGE
    RETURNS on a few thousand molecules at full float64 -- so the two answer different questions
    ("is the block mordred" vs "does featurize_all hand the block the right molecule") and neither
    stands in for the other.

    GRADED AT A SCALED TOLERANCE, not bitwise, for the reason cpp/verify_ac.py's own comment
    gives: ATSC and AATSC are sums of CENTRED products, which cancel, so a cell whose true value
    is near zero is the difference of terms of order one and the two summation orders cannot agree
    relatively. Same constants as that harness (rel 1e-8, absolute floor 1e-8 of the column's own
    median scale); the max observed deviation is printed per weight whether or not it passed.

    NaN-NESS IS GRADED SEPARATELY AND STRICTLY. mordred returns an Error object for a lag no pair
    reaches, and this file maps that to NaN; a cell that is NaN on one side and finite on the
    other is a failure in its own right, because letting NaN == NaN pass would let a family that
    returned nothing at all look perfect.
    """
    try:
        _numpy_shim()
        import mordred
        from mordred import Autocorrelation as AC, Calculator
    except Exception as e:                                     # noqa: BLE001
        print(f"  {'autocorr vs mordred':26s} SKIPPED -- mordred not importable here ({e}); the "
              f"540 Autocorrelation columns are graded against cpp/ac only in this run")
        return 0
    if mordred.__version__ != "1.2.0":
        raise SystemExit(f"WRONG MORDRED: {mordred.__version__}")

    lo = OFF["autocorr"]
    hi = min(v for v in OFF.values() if v > lo)
    ac_names = NAMES[lo:hi]
    # PULLED APART BY NAME, not rebuilt from a retyped (variant, lag, weight) triple: the names
    # come from autocorr.h's col_name() through _core, so if the C++ ever emits a name mordred
    # cannot construct, this raises instead of quietly grading 539 columns.
    objs = []
    for nm in ac_names:
        for v in ("AATSC", "AATS", "ATSC", "ATS", "MATS", "GATS"):   # longest prefix first
            if nm.startswith(v):
                k, w = int(nm[len(v)]), nm[len(v) + 1:]
                objs.append(getattr(AC, v)(k, w))
                break
        else:
            raise SystemExit(f"SPEC DRIFT: {nm!r} is not an Autocorrelation column name")
    calc = Calculator(objs)
    assert [str(d) for d in calc.descriptors] == ac_names, "mordred reordered the descriptors"

    W = np.empty((len(mols), len(ac_names)))
    for i, m in enumerate(mols):
        for j, v in enumerate(calc(m)):
            try:
                W[i, j] = float(v)
            except Exception:                                  # noqa: BLE001 -- a mordred Error
                W[i, j] = np.nan
    got = X[:, lo:hi]

    RTOL, SCALE_FRAC = 1e-8, 1e-8
    bad_val = np.zeros(len(ac_names), int)
    bad_nan = np.zeros(len(ac_names), int)
    worst = np.zeros(len(ac_names))
    for j in range(len(ac_names)):
        a, b = got[:, j], W[:, j]
        na, nb = np.isnan(a), np.isnan(b)
        bad_nan[j] = int((na != nb).sum())
        ok = ~na & ~nb
        if ok.any():
            scale = float(np.median(np.abs(b[ok])))
            if not np.isfinite(scale) or scale == 0.0:
                scale = float(np.max(np.abs(b[ok]))) or 1.0
            atol = SCALE_FRAC * scale
            worst[j] = float((np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), atol)).max())
            bad_val[j] = int((np.abs(a[ok] - b[ok]) > atol + RTOL * np.abs(b[ok])).sum())

    # PER WEIGHT, WITH THE NEW ONE MARKED. `Z` was added after the other nine had been verified
    # and the artifact checksummed, so the useful shape of this table is "did the nine move" next
    # to "is the tenth right" -- a single family-level EXACT line answers neither.
    weights = ["c", "d", "dv", "i", "p", "v", "se", "pe", "are", "Z"]
    print(f"  {'autocorr vs mordred':26s} per weight, {len(mols)} molecules, "
          f"{len(ac_names)} columns (rel {RTOL:g}, abs floor {SCALE_FRAC:g} x column scale):")
    bad = 0
    for w in weights:
        sel = [j for j, nm in enumerate(ac_names) if _ac_weight(nm) == w]
        bv, bn = int(bad_val[sel].sum()), int(bad_nan[sel].sum())
        bad += sum(1 for j in sel if bad_val[j] + bad_nan[j])
        print(f"    {w:6s} {len(sel):4d} cols  {'EXACT' if not (bv or bn) else 'MISMATCH'}"
              f"   value err {bv:6d}   NaN err {bn:6d}   max dev {worst[sel].max():.3e}"
              f"{'   <- NEW' if w == 'Z' else ''}")
    n_old = sum(1 for nm in ac_names if _ac_weight(nm) != "Z")
    ok_old = sum(1 for j, nm in enumerate(ac_names)
                 if _ac_weight(nm) != "Z" and not (bad_val[j] + bad_nan[j]))
    ok_new = sum(1 for j, nm in enumerate(ac_names)
                 if _ac_weight(nm) == "Z" and not (bad_val[j] + bad_nan[j]))
    print(f"    {ok_old} / {n_old} pre-existing columns exact through the wiring"
          f"   |   {ok_new} / {len(ac_names) - n_old} new `Z` columns exact")
    return bad


def _ac_weight(name: str) -> str:
    """The weight suffix of an Autocorrelation column name: everything after the lag digit."""
    for v in ("AATSC", "AATS", "ATSC", "ATS", "MATS", "GATS"):
        if name.startswith(v):
            return name[len(v) + 1:]
    raise ValueError(name)


def check_autocorr(mols, X, tmp: Path) -> int:
    """The 540 Autocorrelation columns against `./cpp/ac`, the binary that produced the evidence.

    THIS CHECKS THE GRAPH, NOT THE ARITHMETIC, and that limit is structural rather than a choice:
    `cpp/ac` includes src/hume_core/autocorr.h, the same header the extension does, so it can only
    confirm that the accumulation agrees with itself. What it CAN catch is the wiring -- a wrong
    H-added graph, a wrong charge, a wrong `nh` -- because the input it is fed here is a second,
    independent construction (see write_ac_io). check_autocorr_mordred above is the one that grades
    the arithmetic, against mordred itself, and it is where the 540 are really decided.

    GRADED AT %.12g, NOT BITWISE, and the reason is the oracle's format rather than the
    arithmetic: `./ac verify` writes cpp/values_ac.txt with `%.12g`, and that file is the
    verified artifact, so widening it would change what the evidence is. Every wiring fault this
    can catch moves a value far above the twelfth digit.
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
# the three mordred families, against mordred itself, in this process
# --------------------------------------------------------------------------------------------

# constit.h's 43 columns minus the two that are RDKit's rather than mordred's. Neither belongs in
# the mordred comparison, and NEITHER IS A PLACEHOLDER ANY MORE:
#   `qed` is graded against rdkit's own `Chem.QED.qed` by `check_qed` below, which also recovers
#         the alert count the C++ used and grades THAT as an integer;
#   `SPS` is graded against rdkit's own `Chem.SpacialScore.SPS` by `check_sps`.
# Both oracles are RDKit's and both run in the pinned `.venv`, so neither needs the mordred
# interpreter.
#
# CONSTIT_NAN IS EMPTY AND IS KEPT. `nan_audit` asserts that the always-NaN set is EXACTLY this
# one, so the empty tuple is now an assertion that no constit column is NaN on every molecule --
# which is a stronger statement than the list it replaced, and it fails loudly if a boundary
# field ever stops arriving.
CONSTIT_NAN = ()
CONSTIT_RDKIT = ("SPS", "qed")


def _numpy_shim() -> None:
    """The two dead aliases mordred 1.2.0 needs under numpy 1.26. See the module docstring."""
    if not hasattr(np, "product"):
        np.product = np.prod          # mordred/MolecularDistanceEdge.py, module scope
    if not hasattr(np, "float"):
        np.float = float              # mordred/ABCIndex.py's last line -- ABCGG has no oracle
        np.float_ = np.float64        # without it


def check_mordred(mols, X) -> int:
    """Chi 40 + TopoMisc 15 + Constit 41 + SLogP 1, against mordred's own descriptor objects.

    THE OBJECTS ARE PULLED OUT OF MORDRED'S PRESET BY NAME, never constructed from a tuple
    retyped here, so a name we emit that mordred no longer produces is a hard failure rather than
    a silently skipped column. mordred's own Context then hands each descriptor `Chem.AddHs` or
    `Chem.RemoveHs` as that descriptor declares -- which is the whole point of asking mordred
    instead of a text file: three different graphs are in play across these 55 columns
    (`Constitutional` alone is the H-ADDED one) and the wiring's claim is that it fed each header
    the graph its own harness verified it on.
    """
    try:
        _numpy_shim()
        import mordred
        from mordred import Calculator, descriptors as mdesc
    except Exception as e:                                     # noqa: BLE001
        print(f"  {'mordred families':26s} SKIPPED -- mordred not importable here ({e}). "
              f"See the docstring: it needs python 3.11 + numpy 1.x, which the pinned .venv is "
              f"not. Chi (40), TopoMisc (15), Constit (41) and SLogP are UNGRADED in this run.")
        return 0
    if mordred.__version__ != "1.2.0":
        raise SystemExit(f"WRONG MORDRED: {mordred.__version__}. uv resolves mordred DOWN to "
                         f"0.6.0 next to numpy 2 rather than erroring; see the docstring.")
    import numpy
    print(f"  mordred {mordred.__version__}  numpy {numpy.__version__}   "
          f"-- the mordred families are graded IN THIS PROCESS")

    lo_chi, lo_tm, lo_ct = OFF["chi"], OFF["topomisc"], OFF["constit"]
    chi_names = NAMES[lo_chi:lo_tm]
    tm_names = NAMES[lo_tm:lo_ct]
    ct_names = [n for n in NAMES[lo_ct:OFF["alias"]]
                if n not in CONSTIT_NAN and n not in CONSTIT_RDKIT]
    want_names = chi_names + tm_names + ct_names + ["SLogP"]

    full = Calculator(mdesc, ignore_3D=True)
    by_name = {}
    for d in full.descriptors:
        by_name.setdefault(str(d), d)
    missing = [n for n in want_names if n not in by_name]
    if missing:
        raise SystemExit(f"SPEC DRIFT: mordred's preset no longer produces {missing}")
    calc = Calculator([by_name[n] for n in want_names], ignore_3D=True)
    order = [str(d) for d in calc.descriptors]
    assert order == want_names, "mordred reordered the descriptors"

    W = np.empty((len(mols), len(want_names)))
    for i, m in enumerate(mols):
        for j, v in enumerate(calc(m)):
            try:
                W[i, j] = float(v)
            except Exception:                                  # noqa: BLE001
                W[i, j] = np.nan                               # a mordred Error object

    bad = 0
    got = X[:, [col(n) for n in want_names]]
    n_chi, n_tm, n_ct = len(chi_names), len(tm_names), len(ct_names)
    a, b, c = n_chi, n_chi + n_tm, n_chi + n_tm + n_ct
    bad += report("chi vs mordred", got[:, :a], W[:, :a], chi_names, nan_eq=True)
    bad += report("topomisc vs mordred", got[:, a:b], W[:, a:b], tm_names, nan_eq=True)
    # ONE POPULATION IS DELIBERATELY DIFFERENT AND MUST NOT HIDE INSIDE THE OTHER, which is the
    # split cpp/verify_constit.py already makes for the standalone binary. The wiring feeds constit
    # the REPAIRED ring set (`_rings.rings_for`, the same one RingCount gets), while mordred's
    # `Rings()` is raw `Chem.GetSymmSSSR` -- and the two differ on roughly 1 molecule in 3,000,
    # where `symmetrizeSSSR` finds a symmetry-equivalent extra ring it does not always find
    # (PORT_STATUS.md: 22 of 100,000 move under atom+bond shuffling before the repair, 0 after).
    # Two constit columns read rings: `fMF` directly and `Vabc` through naRing/nARing.
    ringdiff = np.array([sorted(sorted(int(i) for i in r) for r in rings_for(m)) !=
                         sorted(sorted(int(i) for i in r) for r in Chem.GetSymmSSSR(m))
                         for m in mols])
    keep = ~ringdiff
    bad += report("constit vs mordred", got[np.ix_(keep, range(b, c))],
                  W[np.ix_(keep, range(b, c))], ct_names, nan_eq=True,
                  note=f"  [{int(ringdiff.sum())} molecules excluded: repaired ring set != "
                       f"GetSymmSSSR]")
    if ringdiff.any():
        # Reported, never graded: on this population the two answers are supposed to differ, and
        # the useful fact is WHICH columns moved and by how much -- anything outside {Vabc, fMF}
        # would mean the ring set is reaching a column nobody thought read rings.
        moved = []
        for j, nm in enumerate(ct_names):
            g, w = got[ringdiff, b + j], W[ringdiff, b + j]
            same = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
            if not same.all():
                fin = np.isfinite(g) & np.isfinite(w)
                dev = float(np.max(np.abs(g[fin] - w[fin]) /
                                   np.maximum(np.abs(w[fin]), 1.0))) if fin.any() else 0.0
                moved.append((nm, int((~same).sum()), dev))
        print(f"  {'  ring-set population':26s} {int(ringdiff.sum())} molecules, columns that "
              f"move: {[(n, k) for n, k, _ in moved] or 'none'}")
        for nm, k, dev in moved:
            print(f"    {nm:22s} differs on {k}, max dev {dev:.3e}   "
                  f"{'EXPECTED (reads the ring set)' if nm in ('Vabc', 'fMF') else 'UNEXPECTED'}")
        if any(nm not in ("Vabc", "fMF") for nm, _, _ in moved):
            bad += 1
    # The alias, graded as its own line. `SLogP` is copied from vsa_bins' MolLogP inside the C++,
    # so this is the check that mordred/SLogP.py really is `Crippen.MolLogP(mol)` and that the
    # copy landed in the right column -- not a check of any arithmetic.
    bad += report("SLogP alias vs mordred", got[:, c:], W[:, c:], ["SLogP"], nan_eq=True)

    # The five columns that were already emitted under their mordred names before this wiring, and
    # are therefore NOT re-emitted by the alias block. Claimed in constit.h's wiring note as
    # "already computed"; measured here, against mordred, so the claim is not left as a comment.
    already = ["TopoPSA", "TPSA", "PEOE_VSA11", "SMR_VSA1", "EState_VSA1"]
    have = [n for n in already if n in by_name]
    skip = [n for n in already if n not in by_name]
    if skip:
        # `TPSA` is RDKit's name, not mordred's -- mordred calls the same quantity `TopoPSA`. It is
        # graded above, against RDKit, in the "VSA vs RDKit" line.
        print(f"  {'':26s} (not mordred names, graded against RDKit above: {skip})")
    if have:
        calc2 = Calculator([by_name[n] for n in have], ignore_3D=True)
        W2 = np.empty((len(mols), len(have)))
        for i, m in enumerate(mols):
            for j, v in enumerate(calc2(m)):
                try:
                    W2[i, j] = float(v)
                except Exception:                              # noqa: BLE001
                    W2[i, j] = np.nan
        bad += report("already-emitted vs mordred", X[:, [col(n) for n in have]], W2, have,
                      nan_eq=True, note="  [not re-emitted; already in ALL_COLUMNS]")

    # PER COLUMN, PRINTED WHETHER OR NOT IT PASSED -- the same discipline as the fragment table
    # above. A family-level EXACT line hides which of the 97 were ever EXERCISED: a column that is
    # NaN on every molecule, or constant, is reported exact by any comparison and says nothing.
    # `nonzero` and `nan` are that, and they are worth seeing rather than inferring.
    print(f"\n  per-column, {len(mols)} molecules from cpp/hard.smi "
          f"(mordred 1.2.0, in-process; `excl` = molecules on the deliberately-different ring "
          f"set):")
    for j, nm in enumerate(want_names):
        g, w = got[:, j], W[:, j]
        same = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
        n_ok = int(same[keep].sum())
        n_ex = int(len(mols) - keep.sum())
        nz = int(np.count_nonzero(np.nan_to_num(w)))
        nn = int(np.count_nonzero(~np.isfinite(w)))
        print(f"    {nm:22s} {n_ok:6d} / {int(keep.sum()):6d}"
              f"   {'EXACT' if n_ok == int(keep.sum()) else 'MISMATCH'}"
              f"   nonzero on {nz:5d}   mordred non-finite on {nn:5d}   excl {n_ex}")
    for nm in CONSTIT_NAN:
        n_nan = int(np.count_nonzero(np.isnan(X[:, col(nm)])))
        print(f"    {nm:22s} {'':6s}   {'':6s}   NOT COMPUTED -- NaN on {n_nan} / {len(mols)}")
    return bad


# src/hume_core/rdkcore.h's 19 columns, in the order it emits them. The first thirteen are the
# ones that read the ring set and therefore have a deliberately-different population; the last six
# do not.
RDKCORE_RINGCOLS = 13


def check_sps(mols, X, smis=None) -> int:
    """`SPS` against rdkit's own `Chem.SpacialScore.SPS`, in this process.

    THE ORACLE IS OUTSIDE THE CODE PATH: rdkit/Chem/SpacialScore.py, called here, not
    src/hume_core/constit.h's `sps()` under another name. What the wiring claims is that the two
    potential-stereo arrays `src/hume/_extract.py` computes reach `constit::Inputs` unshifted and
    on the right molecule -- a transposed atom index there would be a wrong-but-plausible number,
    which is the failure this whole file exists to catch.

    GRADED BITWISE. SPS is an integer sum divided by the heavy-atom count; both sides accumulate
    in atom-index order and neither has a reordering to hide behind.

    TWO ORACLES, AND THE SECOND IS THE STRICTER ONE. `all_cols` has already run
    `extract_pickles`, which calls `AssignStereochemistry(cleanIt=True, force=True,
    flagPossibleStereoCenters=True)` on the caller's molecules -- so a `Chem.Mol(m)` copy is a
    copy of a MUTATED molecule. When the SMILES are available the same columns are graded a second
    time against a molecule parsed FRESH from text, which is the only oracle that owes nothing to
    the code path. If those two ever disagree, the descriptor depends on the pipeline's own
    mutation and the divergence is the finding, not a rounding difference.
    """
    from rdkit.Chem.SpacialScore import SPS

    j = col("SPS")
    got = X[:, j:j + 1]
    want = np.array([[float(SPS(Chem.Mol(m)))] for m in mols])
    bad = report("SPS vs RDKit", got, want, ["SPS"],
                 note="  [the potential-stereo boundary pair; bitwise]")
    if smis is not None:
        fresh = np.array([[float(SPS(Chem.MolFromSmiles(s)))] for s in smis])
        bad += report("  vs a FRESH parse", got, fresh, ["SPS"],
                      note="  [oracle owes nothing to the code path]")
    return bad


def check_qed(mols, X, smis=None) -> int:
    """`qed` against rdkit's own `Chem.QED.qed`, AND the alert count it was built from.

    THE ORACLE IS OUTSIDE THE CODE PATH: rdkit/Chem/QED.py, called here. Seven of QED's eight
    properties were already exact in src/hume_core/constit.h and are re-checked by
    cpp/verify_constit.py; what is new is ALERTS, and what the wiring claims is that the 116
    compiled alert patterns in cpp/qed_alert_program.h reach `constit::Inputs::qedAlerts` on the
    right molecule, through the same `fragmatch::Mol` the 76 fragment columns read.

    IT IS NOT BITWISE, AND THE REASON IS STATED RATHER THAN TOLERATED. `qed` is
    `exp(sum(w_i * log(ads_i)) / sum(w_i))` over eight desirability functions, each of which is
    two `exp`s, a division and a subtraction. libm's `exp` and `log` are not correctly rounded and
    are not required to agree between CPython's and clang's calls into them, and constit.h already
    splits every `a + b*c` in the expression to stop clang contracting it into an FMA that python
    did not use. So the claim here is a MEASURED max relative deviation, printed, not a tolerance
    chosen to make a line say EXACT. The bitwise count is printed beside it.

    AND THAT IS WHY `qedAlerts` IS RECOVERED AND GRADED AS AN INTEGER. A wrong alert count would
    show up in `qed` as a small float difference -- exactly the shape a rounding difference has --
    so the composite alone cannot tell the two apart. The count the C++ used is recovered by
    asking RDKit for `qed` under every candidate ALERTS value with the molecule's OWN other seven
    properties, and taking the candidate closest to the emitted value. That is well posed only if
    the runner-up is far away, so the MARGIN is measured too and printed: if the nearest and the
    next-nearest candidate were ever within the observed float noise, this recovery would be
    meaningless and the number below says so.
    """
    from rdkit.Chem import QED

    j = col("qed")
    got = X[:, j:j + 1]
    props = [QED.properties(Chem.Mol(m)) for m in mols]
    want = np.array([[float(QED.qed(Chem.Mol(m), qedProperties=p))]
                     for m, p in zip(mols, props)])
    bitwise = int(np.count_nonzero(got.view(np.uint64) == want.view(np.uint64)))
    fin = np.isfinite(got[:, 0]) & np.isfinite(want[:, 0])
    dev = float(np.max(np.abs(got[fin, 0] - want[fin, 0]) /
                       np.maximum(np.abs(want[fin, 0]), 1e-300))) if fin.any() else 0.0
    n_nan = int(np.count_nonzero(~np.isfinite(got[:, 0])))
    print(f"  {'qed vs RDKit':26s} bitwise on {bitwise:6d} / {len(mols):6d}   "
          f"max rel dev {dev:.3e}   non-finite {n_nan}"
          f"   [float; see the docstring for why not bitwise]")

    # -- the eighth property, recovered from the emitted value and graded as an integer --------
    cand = sorted({int(p.ALERTS) for p in props} |
                  {max(0, int(p.ALERTS) - 1) for p in props} |
                  {int(p.ALERTS) + 1 for p in props})
    bad_alert = 0
    worst_margin = float("inf")
    first = None
    for i, (m, p) in enumerate(zip(mols, props)):
        vals = [(abs(got[i, 0] - float(QED.qed(Chem.Mol(m), qedProperties=p._replace(ALERTS=k)))),
                 k) for k in cand]
        vals.sort()
        used, runner = vals[0][1], vals[1][0]
        worst_margin = min(worst_margin, runner)
        if used != int(p.ALERTS):
            bad_alert += 1
            if first is None:
                first = (Chem.MolToSmiles(m), int(p.ALERTS), used)
    print(f"  {'qedAlerts (recovered)':26s} EXACT on {len(mols) - bad_alert:6d} / {len(mols):6d}"
          f"   integer count vs sum(HasSubstructMatch)"
          f"   smallest recovery margin {worst_margin:.3e}")
    if first is not None:
        print(f"    first mismatch: {first[0]}  rdkit {first[1]}, recovered {first[2]}")

    bad = 0
    if bad_alert:
        bad += 1
    # The float claim: report it as a tolerance so a REAL divergence (a wrong property, a shifted
    # column) still fails. 1e-12 is ~4 orders above the observed deviation and ~4 below the
    # smallest difference one alert makes.
    bad += report("qed vs RDKit (tolerance)", got, want, ["qed"], tol=1e-12,
                  note="  [8 ADS functions of exp/log; not bitwise, see check_qed]")
    if smis is not None:
        fresh = np.array([[float(QED.qed(Chem.MolFromSmiles(s)))] for s in smis])
        bad += report("  vs a FRESH parse", got, fresh, ["qed"], tol=1e-12,
                      note="  [oracle owes nothing to the code path]")
    return bad


def check_rdkcore(mols, X, smis=None) -> int:
    """The last 19 `rdkit_core` columns against RDKit's own `Descriptors`, in this process.

    THE ORACLE IS OUTSIDE THE CODE PATH, which is the whole discipline of this file: RDKit's
    Lipinski.cpp / MolProps.cpp / ConnectivityDescriptors.cpp / MorganGenerator.cpp, called here,
    not a standalone binary that includes src/hume_core/rdkcore.h and could only agree with itself.
    Sixteen of the nineteen are integer counts or ratios of them, so "exact" is bitwise with
    nowhere to hide; `HeavyAtomMolWt`, `Phi` and the three `FpDensityMorgan*` are float64 and are
    ALSO graded bitwise -- they are a sum in atom-index order, a product of two block columns, and
    an integer over an integer, and none of them has a reordering to hide behind.

    A FRESH `Chem.Mol(m)` PER CALL, AND IT IS NOT DEFENSIVE. `_FingerprintDensity` MEMOISES the
    fingerprint on the molecule object it is handed, and `all_cols` above has already run
    `extract_pickles`, which calls `Chem.AssignStereochemistry(cleanIt=True, force=True)` on the
    caller's molecule. Grading against a mutated, cached oracle is how a wiring harness passes
    while measuring nothing.

    ONE POPULATION IS DELIBERATELY DIFFERENT, exactly as it is for `constit`. The wiring feeds
    these columns the REPAIRED ring set (`_rings.rings_for`, the same one RingCount gets) while
    RDKit reads its own raw `GetSymmSSSR`, and the two differ on 32 of the 100,000 corpus
    molecules. Those are excluded from the graded population and reported separately, and anything
    that moves there OUTSIDE the thirteen ring predicates is a failure: the other six do not read
    the ring SET at all. (`FpDensityMorgan*` reads ring MEMBERSHIP, per atom, through `atom_i`'s
    `nring` column -- RDKit's own raw count. The repair only ever adds or drops a
    symmetry-equivalent ring of a size already present, so no atom changes ring membership under
    it, and those three columns are graded on the whole population.)
    """
    # A DEFENSIVE SKIP RATHER THAN A KeyError, because this file is run from two environments
    # (.venv and .venv-mordred) whose extensions are installed separately: an env that still has
    # a pre-rdkcore `hume._core` should grade the families it does have and say so, not die.
    if "rdkcore" not in OFF:
        print(f"  {'rdkcore vs RDKit':26s} SKIPPED -- this environment's hume._core predates the "
              f"rdkcore family (no 'rdkcore' offset); reinstall it to grade the 19 columns")
        return 0
    lo = OFF["rdkcore"]
    # THE COUNT IS READ OFF THE LAYOUT, not written down. rdkcore is the LAST family, so its
    # columns run to the end of the row; hard-coding 19 here is what would have to be edited every
    # time the family grows, and forgetting to would silently stop grading the new columns.
    names = NAMES[lo:OFF["end"]]
    fns = dict(Descriptors._descList)
    missing = [n for n in names if n not in fns]
    if missing:
        raise SystemExit(f"SPEC DRIFT: RDKit's Descriptors no longer produces {missing}")
    want = np.empty((len(mols), len(names)))
    for j, nm in enumerate(names):
        f = fns[nm]
        for i, m in enumerate(mols):
            want[i, j] = float(f(Chem.Mol(m)))
    got = X[:, lo:lo + len(names)]
    # THE TWO STEREO COUNTS, GRADED A SECOND TIME AGAINST A FRESH PARSE. `Chem.Mol(m)` above is a
    # copy of a molecule `extract_pickles` has already run `AssignStereochemistry(cleanIt=True,
    # force=True, flagPossibleStereoCenters=True)` over, and these two columns are a function of
    # exactly what that call leaves behind -- so the copy is the weaker oracle. A molecule parsed
    # fresh from SMILES owes nothing to the code path, and if the two disagree that IS the finding.
    bad_fresh = 0
    if smis is not None:
        stereo_cols = [j for j, nm in enumerate(names)
                       if nm in ("NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters")]
        if stereo_cols:
            fresh = np.array([[float(fns[names[j]](Chem.MolFromSmiles(s))) for j in stereo_cols]
                              for s in smis])
            bad_fresh = report("  stereo counts vs a FRESH parse", got[:, stereo_cols], fresh,
                               [names[j] for j in stereo_cols],
                               note="  [oracle owes nothing to the code path]")

    ringdiff = np.array([sorted(sorted(int(i) for i in r) for r in rings_for(m)) !=
                         sorted(sorted(int(i) for i in r) for r in Chem.GetSymmSSSR(m))
                         for m in mols])
    keep = ~ringdiff
    # `Phi` IS NOT BITWISE AGAINST RDKit AND WAS NEVER GOING TO BE, and the reason is inherited
    # rather than introduced here. calcPhi is `kappa1 * kappa2 / heavy` over the SAME kappas
    # calcKappa1 / calcKappa2 return, and this repo's Kappa1 / Kappa2 have been graded at rtol
    # 1e-9 against RDKit since they were ported (cpp/verify_hume.py's TOL table) -- HallKierAlpha
    # falls back to `rB0(Z)/rB0(C) - 1` for the elements RDKit has no tabulated alpha for (As, Sn,
    # Te and friends are in this corpus) and the last bits of that division differ. So Phi is
    # graded TWICE, and the two answer different questions:
    #
    #   * bitwise against `Kappa1 * Kappa2 / GetNumHeavyAtoms()` computed HERE from the row's own
    #     Kappa columns -- which is the wiring claim, and the thing this file exists to test;
    #   * at a tolerance against RDKit's own Phi, printed NEXT TO the same measure on Kappa1 and
    #     Kappa2 so the deviation can be seen to be theirs and not the multiplication's.
    j_phi = names.index("Phi")
    bit = [j for j in range(len(names)) if j != j_phi]
    bad = bad_fresh
    bad += report("rdkcore vs RDKit", got[np.ix_(keep, bit)], want[np.ix_(keep, bit)],
                  [names[j] for j in bit],
                  note=f"  [{int(ringdiff.sum())} molecules excluded: repaired ring set != "
                       f"GetSymmSSSR; Phi graded separately below]")
    k1, k2 = X[:, col("Kappa1")], X[:, col("Kappa2")]
    A = np.array([m.GetNumHeavyAtoms() for m in mols], dtype=np.float64)
    bad += report("Phi vs its own Kappa cols", got[:, j_phi:j_phi + 1],
                  (k1 * k2 / np.where(A > 0, A, 1.0))[:, None], ["Phi"],
                  note="  [the wiring claim: bitwise, no tolerance]")
    wk = np.array([[Descriptors.Kappa1(Chem.Mol(m)), Descriptors.Kappa2(Chem.Mol(m))]
                   for m in mols])
    bad += report("Phi vs RDKit", got[:, j_phi:j_phi + 1], want[:, j_phi:j_phi + 1], ["Phi"],
                  tol=1e-9)
    bad += report("  its inputs vs RDKit", np.column_stack([k1, k2]), wk, ["Kappa1", "Kappa2"],
                  tol=1e-9, note="  [pre-existing; cpp/verify_hume.py grades these at the same "
                                 "rtol]")
    if ringdiff.any():
        moved = []
        for j, nm in enumerate(names):
            if j == j_phi:
                continue                      # not bitwise anywhere; see above
            g, w = got[ringdiff, j], want[ringdiff, j]
            same = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
            if not same.all():
                moved.append((nm, int((~same).sum())))
        print(f"  {'  ring-set population':26s} {int(ringdiff.sum())} molecules, columns that "
              f"move: {[m[0] for m in moved] or 'none'}")
        for nm, k in moved:
            ok = names.index(nm) < RDKCORE_RINGCOLS
            print(f"    {nm:22s} differs on {k}   "
                  f"{'EXPECTED (reads the ring set)' if ok else 'UNEXPECTED'}")
        if any(names.index(nm) >= RDKCORE_RINGCOLS for nm, _ in moved):
            bad += 1

    print(f"\n  per-column, {len(mols)} molecules from cpp/hard.smi "
          f"(`excl` = the deliberately-different ring-set population):")
    n_ex = int(ringdiff.sum())
    for j, nm in enumerate(names):
        g, w = got[:, j], want[:, j]
        if j == j_phi:
            dev = float(np.max(np.abs(g - w) / np.maximum(np.abs(w), 1.0)))
            print(f"    {nm:26s} {'':6s}   {'':6s}   max rel dev vs RDKit {dev:.3e}"
                  f"   (rtol 1e-9, inherited from Kappa1/Kappa2)   excl {n_ex}")
            continue
        same = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
        n_ok = int(same[keep].sum())
        nz = int(np.count_nonzero(np.nan_to_num(w)))
        print(f"    {nm:26s} {n_ok:6d} / {int(keep.sum()):6d}"
              f"   {'EXACT' if n_ok == int(keep.sum()) else 'MISMATCH'}"
              f"   nonzero on {nz:6d}   excl {n_ex}")
    return bad


def nan_audit(mols, X) -> int:
    """WHICH cells of the three new families are non-finite, and whether that is the two known
    placeholders or something nobody decided.

    PORT_STATUS.md records that non-finite values are CORRECT and expected in this matrix (144 of
    1015 columns for ethanol are `AATS<k>*` beyond the molecule's diameter). That makes a bare NaN
    count useless as a guard, so this prints the per-column non-finite rate for the new families
    and asserts only the thing that IS decided: the set of columns that are NaN on EVERY molecule
    is EXACTLY `CONSTIT_NAN`.

    THAT SET IS NOW EMPTY, AND THE ASSERTION IS STRONGER FOR IT. `SPS` left it when the
    potential-stereo arrays landed and `qed` left it when the QED alert program did. Asserting
    the set exactly rather than as a lower bound is what makes this useful in both directions:
    if either boundary field ever stops arriving, that column goes back to always-NaN and this
    audit fails instead of shrugging.
    """
    lo, hi = OFF["chi"], OFF["end"]
    print(f"\n  non-finite rate, {X.shape[0]} molecules, the four new families:")
    allnan = []
    for j in range(lo, hi):
        n = int(np.count_nonzero(~np.isfinite(X[:, j])))
        if n:
            print(f"    {NAMES[j]:22s} {n:6d} / {X.shape[0]:6d} non-finite"
                  f"{'   <- ALL' if n == X.shape[0] else ''}")
        if n == X.shape[0]:
            allnan.append(NAMES[j])
    expect = list(CONSTIT_NAN)
    if sorted(allnan) != sorted(expect):
        print(f"    UNEXPECTED: always-NaN columns are {sorted(allnan)}, expected {sorted(expect)}")
        return 1
    print(f"    always-NaN: {sorted(allnan) or 'none -- every column produces a value on '
          f'some molecule'}")
    return 0


# --------------------------------------------------------------------------------------------
def main_rdkcore(n_want: int) -> int:
    """`cpp/verify_wiring.py N rdkcore` -- the rdkcore family plus `SPS`, over the whole corpus.

    WHY A SEPARATE ENTRY POINT RATHER THAN A BIGGER `N`. House rule 5 wants exactness reported on
    all 100,000 molecules of cpp/hard.smi; the full run cannot be asked for that, because the VSA
    arm alone calls 65 RDKit descriptors per molecule and the mordred arm is minutes per thousand.
    This runs the same `check_rdkcore` -- same oracle, same code path, same in-process comparison
    -- in batches over the whole file, and accumulates.

    `SPS` RIDES ALONG HERE BECAUSE IT IS THE SAME KIND OF CLAIM AND THE SAME KIND OF ORACLE:
    rdkit's own descriptor, in this process, over the whole corpus. It is not an rdkcore column
    -- it lives in constit.h -- but it has no mordred oracle and would otherwise be graded only on
    the 3,000-molecule sample the main run uses.

    THE ORACLE IS A FRESH PARSE FOR THE THREE STEREO COLUMNS. `all_cols` runs `extract_pickles`,
    which mutates the caller's molecules with `AssignStereochemistry(cleanIt=True, force=True,
    flagPossibleStereoCenters=True)`, and `NumAtomStereoCenters` /
    `NumUnspecifiedAtomStereoCenters` are a function of exactly what that leaves behind. So those
    three are graded BOTH ways -- against a `Chem.Mol(m)` copy like every other column here, and
    against a molecule parsed fresh from the same SMILES text, which owes nothing to the code
    path. Both numbers are printed.
    """
    smis = [s for s in (ROOT / "cpp" / "hard.smi").read_text().split("\n") if s][:n_want]
    print(f"{len(smis)} molecules from cpp/hard.smi, rdkcore + SPS only\n")
    if "rdkcore" not in OFF:
        raise SystemExit("this environment's hume._core has no rdkcore family; reinstall it")
    lo = OFF["rdkcore"]
    # rdkcore is the LAST family, so its columns run to the end of the row. `SPS` is appended to
    # the graded list by name, out of constit's block; `col()` resolves it.
    names = NAMES[lo:OFF["end"]] + ["SPS"]
    j_sps = len(names) - 1
    idx = list(range(lo, OFF["end"])) + [col("SPS")]
    fns = dict(Descriptors._descList)
    missing = [n for n in names if n not in fns]
    if missing:
        raise SystemExit(f"SPEC DRIFT: RDKit's Descriptors no longer produces {missing}")
    # The three columns whose oracle must not be a mutated molecule; see the docstring.
    FRESH = ("NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters", "SPS")
    j_fresh = [j for j, nm in enumerate(names) if nm in FRESH]
    n_fresh_ok = [0] * len(names)
    j_phi = names.index("Phi")
    n_ok = [0] * len(names)
    n_graded = 0
    n_excl = 0
    nz = [0] * len(names)
    phi_dev = 0.0
    kap_dev = 0.0
    moved_excl: dict[str, int] = {}
    B = 2000
    for lo_i in range(0, len(smis), B):
        texts = smis[lo_i:lo_i + B]
        chunk = [Chem.MolFromSmiles(t) for t in texts]
        if any(m is None for m in chunk):
            raise ValueError("unparseable SMILES in the corpus")
        X = all_cols(chunk)
        got = X[:, idx]
        want = np.empty((len(chunk), len(names)))
        for j, nm in enumerate(names):
            f = fns[nm]
            for i, m in enumerate(chunk):
                want[i, j] = float(f(Chem.Mol(m)))
        for j in j_fresh:
            f = fns[names[j]]
            fresh = np.array([float(f(Chem.MolFromSmiles(t))) for t in texts])
            g = got[:, j]
            n_fresh_ok[j] += int(np.count_nonzero(g.view(np.uint64) == fresh.view(np.uint64)))
        ringdiff = np.array([sorted(sorted(int(i) for i in r) for r in rings_for(m)) !=
                             sorted(sorted(int(i) for i in r) for r in Chem.GetSymmSSSR(m))
                             for m in chunk])
        keep = ~ringdiff
        n_excl += int(ringdiff.sum())
        n_graded += int(keep.sum())
        k1 = X[:, col("Kappa1")]
        k2 = X[:, col("Kappa2")]
        wk = np.array([[Descriptors.Kappa1(Chem.Mol(m)), Descriptors.Kappa2(Chem.Mol(m))]
                       for m in chunk])
        kap_dev = max(kap_dev, float(np.max(np.abs(np.column_stack([k1, k2]) - wk) /
                                            np.maximum(np.abs(wk), 1.0))))
        for j in range(len(names)):
            g, w = got[:, j], want[:, j]
            nz[j] += int(np.count_nonzero(np.nan_to_num(w)))
            if j == j_phi:
                phi_dev = max(phi_dev, float(np.max(np.abs(g - w) /
                                                    np.maximum(np.abs(w), 1.0))))
                # the wiring claim is still bitwise, on every molecule
                A = np.array([m.GetNumHeavyAtoms() for m in chunk], dtype=np.float64)
                own = k1 * k2 / np.where(A > 0, A, 1.0)
                n_ok[j] += int(np.count_nonzero(g.view(np.uint64) == own.view(np.uint64)))
                continue
            same = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
            n_ok[j] += int(same[keep].sum())
            if ringdiff.any() and not same[ringdiff].all():
                moved_excl[names[j]] = moved_excl.get(names[j], 0) + int((~same[ringdiff]).sum())
        print(f"  {lo_i + len(chunk):7d} / {len(smis)} ...", flush=True)

    bad = 0
    print(f"\n  per-column, {len(smis)} molecules, {n_graded} graded and {n_excl} on the "
          f"deliberately-different ring set:")
    for j, nm in enumerate(names):
        if j == j_phi:
            ok = n_ok[j] == len(smis)
            print(f"    {nm:26s} {n_ok[j]:7d} / {len(smis):7d} bitwise vs its own Kappa columns"
                  f"   {'EXACT' if ok else 'MISMATCH'}   max rel dev vs RDKit {phi_dev:.3e}")
            bad += not ok
            bad += phi_dev > 1e-9
            continue
        ok = n_ok[j] == n_graded
        extra = ""
        if j in j_fresh:
            fok = n_fresh_ok[j] == len(smis)
            extra = (f"   vs a FRESH parse {n_fresh_ok[j]:7d} / {len(smis):7d} "
                     f"{'EXACT' if fok else 'MISMATCH'}")
            bad += not fok
        print(f"    {nm:26s} {n_ok[j]:7d} / {n_graded:7d}"
              f"   {'EXACT' if ok else 'MISMATCH'}   nonzero on {nz[j]:7d}{extra}")
        bad += not ok
    print(f"    {'Kappa1/Kappa2 vs RDKit':26s} max rel dev {kap_dev:.3e}  (rtol 1e-9, "
          f"pre-existing -- this is what Phi inherits)")
    print(f"\n  on the {n_excl} excluded molecules the columns that move are: "
          f"{moved_excl or 'none'}")
    for nm in moved_excl:
        if names.index(nm) >= RDKCORE_RINGCOLS:
            print(f"    UNEXPECTED: {nm} does not read the ring set")
            bad += 1
    print("\nRDKCORE EXACT" if not bad else f"\n{bad} COLUMNS DISAGREE")
    return 0 if not bad else 1


def main() -> int:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    only = sys.argv[2] if len(sys.argv) > 2 else None
    # THE NUMERIC CANARY, not the version banner. cpp/verify_hume.py's note in full: a process can
    # print `rdkit 2025.09.2` and execute another version's arithmetic out of unlinked-but-still-
    # mapped dylibs, which has happened in this repo. The number is computed here, by this
    # process, in the interpreter that produces every comparison below.
    canary = Descriptors.BCUT2D_MRLOW(Chem.MolFromSmiles(
        "O=C1CCNCCNNNCCNCCC(=O)c2ccc(o2)COCOCc2ccc1o2"))
    print(f"rdkit {rdkit.__version__}   numpy {np.__version__}   "
          f"python {sys.version.split()[0]}   CANARY {canary!r}")
    if canary != -0.07665884800196521:
        raise SystemExit(f"CANARY MISMATCH: {canary!r}, expected -0.07665884800196521 (rdkit "
                         f"2025.09.2). The RDKit executing is not the one on the label.")
    print(f"{_core.N_ALL_COLS} columns, offsets {dict(OFF)}\n")
    if only == "rdkcore":
        return main_rdkcore(n_want)

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
    # TWO ORACLES FOR ONE FAMILY, and only the second grades the arithmetic. `cpp/ac` above shares
    # src/hume_core/autocorr.h with the wiring and can only confirm the accumulation agrees with
    # itself; mordred is outside the code path entirely, builds its own H-added graph and its own
    # charges, and is what the 540 columns claim to reproduce. Needs the mordred interpreter, and
    # says so rather than passing silently when it is not there.
    bad += check_autocorr_mordred(mols, X)

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
    # The family's OWN width -- the next offset above it, not "everything to the end". This read
    # OFF["end"] and was correct only while `frag` happened to be the last family in the layout;
    # adding chi / topomisc / constit after it broke that, exactly as adding `frag` after
    # Autocorrelation broke the same idiom in check_autocorr.
    hi = min(v for v in OFF.values() if v > lo)
    frag_names = NAMES[lo:hi]
    want = np.empty((len(mols), len(frag_names)))
    for j, nm in enumerate(frag_names):
        f = fns[nm]
        for i, m in enumerate(mols):
            want[i, j] = float(f(m))
    got = X[:, lo:hi]
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

    # ---- the last rdkit_core columns, against RDKit's own Descriptors ------------------------
    print()
    bad += check_rdkcore(mols, X, pick)

    # ---- `SPS`, against rdkit's own Chem.SpacialScore -----------------------------------------
    print()
    bad += check_sps(mols, X, pick)

    # ---- `qed`, against rdkit's own Chem.QED -- and the alert count it is built from ----------
    print()
    bad += check_qed(mols, X, pick)

    # ---- the three mordred families and the alias --------------------------------------------
    print()
    bad += check_mordred(mols, X)
    bad += nan_audit(mols, X)

    print("\nWIRING EXACT -- every family sees the graph its own harness verified it on"
          if not bad else f"\n{bad} COLUMNS DISAGREE")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
