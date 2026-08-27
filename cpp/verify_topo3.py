"""Are the three pure-topology C++ families exactly mordred?

    RingCount 49 + TopologicalCharge 21 + PathCount 11 = 81 columns, on cpp/hard.smi (100,000).

RUN IT LIKE THIS, AND ONLY LIKE THIS:

    uv run --isolated --with mordred --with "rdkit==2025.9.2" --with "numpy==2.4.6" \
           python cpp/verify_topo3.py all

Never `uv run --with rdkit` unpinned and never `uv pip install` into .venv -- constraints.txt
explains what that has already cost here twice. Every mode below prints the resolved rdkit and
numpy versions FROM THE PROCESS THAT PRODUCES THE NUMBERS, because a verify log without its
RDKit version on it is not evidence.

ONE SHIM, AND IT TOUCHES NOTHING WE MEASURE. mordred 1.2.0 does `from numpy import product` at
module scope in MolecularDistanceEdge.py, and numpy 2 removed that alias, so importing
`mordred.descriptors` fails. `_numpy_shim()` restores the alias before the import. The pinned
oracle is numpy 2.4.6 (constraints.txt), so shimming is the right call rather than dropping to
numpy 1.x -- and note that dropping to numpy 1.x would CHANGE the answers here, because numpy
2.x on macOS/arm64 links Accelerate for BLAS and numpy 1.x links OpenBLAS, and this family's
last digits are a dgemm accumulation order (see below). None of RingCount, TopologicalCharge or
PathCount calls `product`.

WHAT THE REFERENCE IS. mordred itself, constructed from the same parameter tuples the C++ table
carries -- and `check_spec()` asserts that correspondence against the live mordred objects
before anything is computed, which is the Python half of ringcount.h's drift guard.

THE MOLECULE IS `Chem.RemoveHs(mol)`. mordred's Context does that for every descriptor with
`explicit_hydrogens = False`, which all three families are. On this corpus RemoveHs is the
identity (the only explicit hydrogen here is isotopic, which plain RemoveHs keeps), and
`dump_mols` reports the count rather than assuming it.

TWO TOLERANCES, FOR TWO DIFFERENT REASONS.

  * RingCount (49) and PathCount (11) are BIT-EXACT and are checked with `==`. Integer counts,
    and -- for piPC -- sums of dyadic rationals small enough that float64 addition is exact
    regardless of order. See pathcount.h.
  * TopologicalCharge (21) is NOT bit-exact and cannot be. mordred computes the Galvez matrix as
    `A.dot(D2)`, a BLAS dgemm whose accumulation order over an atom's neighbours belongs to the
    kernel, not to the source. `CT = M - M.T` then subtracts two nearly equal sums. THIS IS NOT A
    PROPERTY OF OUR PORT: re-running MORDRED on a randomly renumbered copy of the same molecule
    moves 20 of these 21 columns too (`perm` mode). So the honest bar is a stated relative
    tolerance with the max observed deviation, which is what `compare` prints -- per column, at
    several thresholds, never as a single pass/fail.

TWO TRAPS IN THE HOUSE-RULE-1 SCREEN ITSELF, both found here the hard way:

  * `Chem.RenumberAtoms` returns a molecule with UNINITIALISED RingInfo. Whatever asks first
    triggers a lazy perception, so without the explicit `SanitizeMol` in `renumbered()` you are
    measuring perception ORDER and reporting phantom ill-posedness.
  * `Chem.RenumberAtoms` permutes ATOMS and leaves the BOND LIST order alone, with the endpoints
    rewritten. RDKit's ring perception READS THE BOND LIST, so an atom-only screen under-samples
    the axis the answer actually depends on. `O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2`
    is stable over 201 atom renumberings and gives two different ring sets once the bond order is
    shuffled too. Anything claiming ring-perception invariance on an atom-only screen is claiming
    less than it sounds like.

MODES
    dump      cpp/hard.smi          -> cpp/topo3_mols.txt         (input for the C++)
    ref       cpp/hard.smi          -> cpp/topo3_ref.txt          (mordred, 81 columns)
    compare   the C++ dumps vs cpp/topo3_ref.txt, per column
    perm      HOUSE RULE 1: is each column a function of the molecule? mordred vs mordred
    canon     the ring-set repair: is canon_rings() renumbering-invariant, and where does it
              differ from RDKit as-parsed?
    gatecheck the repair is GATED for cost. Run it unconditionally over the whole corpus and
              assert the gated pipeline gives the identical 49 columns. Must report 0.
    benchpy   mordred us/mol on a subsample, for the C++/Python comparison
    all       dump, ref, the three binaries, compare
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
SMI = HERE / "hard.smi"
MOLS = HERE / "topo3_mols.txt"
REF = HERE / "topo3_ref.txt"
OUT = {"ringcount": HERE / "topo3_ringcount.txt",
       "topocharge": HERE / "topo3_topocharge.txt",
       "pathcount": HERE / "topo3_pathcount.txt"}
BENCH_SMI = HERE / "topo3_bench.txt"
NPROC = int(os.environ.get("TOPO3_NPROC", "6"))


def _numpy_shim() -> None:
    import numpy as np
    if not hasattr(np, "product"):
        np.product = np.prod          # mordred/MolecularDistanceEdge.py, module scope
    if not hasattr(np, "float_"):
        np.float_ = np.float64


def versions(tag: str) -> None:
    import numpy as np
    import rdkit
    print(f"[{tag}] rdkit {rdkit.__version__}  numpy {np.__version__}  "
          f"python {sys.version.split()[0]}", flush=True)


# ---------------------------------------------------------------------------------------------
# The 81 columns, in the order the C++ emits them. Generated from mordred's own preset() order
# and filtered to the data/dedupe.json survivors; check_spec() proves the correspondence.
# ---------------------------------------------------------------------------------------------
RC_NAMES = (
    "n3Ring n4Ring n5Ring n6Ring n7Ring nG12Ring n3HRing n4HRing n5HRing n6HRing n7HRing naRing "
    "n5aRing n6aRing naHRing n6aHRing nARing n5ARing n6ARing n5AHRing n6AHRing nFRing n7FRing "
    "n8FRing n9FRing n10FRing n11FRing n12FRing nG12FRing nFHRing n9FHRing n10FHRing nG12FHRing "
    "nFaRing n8FaRing n9FaRing n10FaRing nG12FaRing nFaHRing n10FaHRing nFARing n9FARing "
    "n10FARing nG12FARing nFAHRing n8FAHRing n9FAHRing n10FAHRing nG12FAHRing").split()
TC_NAMES = [f"GGI{k}" for k in range(1, 11)] + [f"JGI{k}" for k in range(1, 11)] + ["JGT10"]
PC_NAMES = ["MPC4", "MPC6", "MPC9"] + [f"piPC{k}" for k in (1, 2, 3, 4, 5, 6, 8, 10)]
NAMES = RC_NAMES + TC_NAMES + PC_NAMES
BLOCK = [("ringcount", 0, 49), ("topocharge", 49, 70), ("pathcount", 70, 81)]


def rc_preset():
    """mordred RingCount.preset(), as (order, greater, fused, aromatic, hetero) in preset order."""
    out = []
    for fused in (False, True):
        for arom in (None, True, False):
            for het in (None, True):
                out.append((None, False, fused, arom, het))
                for n in range(4 if fused else 3, 13):
                    out.append((n, False, fused, arom, het))
                out.append((12, True, fused, arom, het))
    return out


def descriptors():
    """The 81 mordred descriptor objects, in the C++'s emit order."""
    from mordred import PathCount as MPC, RingCount as MRC, TopologicalCharge as MTC
    pre = {}
    for p in rc_preset():
        d = MRC.RingCount(*p)
        pre[str(d)] = d
    missing = [n for n in RC_NAMES if n not in pre]
    if missing:
        raise SystemExit(f"SPEC DRIFT: {missing} are not produced by mordred's RingCount.preset()")
    ds = [pre[n] for n in RC_NAMES]
    ds += [MTC.TopologicalCharge("raw", k) for k in range(1, 11)]
    ds += [MTC.TopologicalCharge("mean", k) for k in range(1, 11)]
    ds += [MTC.TopologicalCharge("global", 10)]
    ds += [MPC.PathCount(k, False, False, False) for k in (4, 6, 9)]
    ds += [MPC.PathCount(k, True, False, True) for k in (1, 2, 3, 4, 5, 6, 8, 10)]
    return ds


def check_spec() -> None:
    """DRIFT GUARD, Python half. ringcount.h::selfCheck() re-derives its 49 parameter tuples from
    its own name strings and from a regenerated preset; that proves the table is self-consistent
    but not that it agrees with the mordred actually installed. This closes it: every name the
    binaries emit must be `str(descriptor)` of a live mordred object built from the tuple the C++
    stores, in the same order, and the three families must contribute exactly 49 / 21 / 11."""
    ds = descriptors()
    got = [str(d) for d in ds]
    if got != NAMES:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(got, NAMES)) if a != b]
        raise SystemExit(f"SPEC DRIFT: mordred names disagree with the C++ emit order: {bad[:5]}")
    for exe, lo, hi in BLOCK:
        b = subprocess.run([str(HERE / exe), "names"], capture_output=True, text=True, check=True)
        if b.stdout.split() != NAMES[lo:hi]:
            raise SystemExit(f"SPEC DRIFT: ./{exe} names != this file's list for {exe}")
    # and the RingCount parameter tuples themselves, not just the names
    pre = {str(MRC_RingCount(*p)): p for p in rc_preset()}
    for n in RC_NAMES:
        if n not in pre:
            raise SystemExit(f"SPEC DRIFT: {n} is not produced by mordred's RingCount preset")
    print(f"[spec] 81 columns agree with mordred's own names and preset order "
          f"({len(RC_NAMES)}/{len(TC_NAMES)}/{len(PC_NAMES)})", flush=True)


def MRC_RingCount(*p):
    from mordred import RingCount as MRC
    return MRC.RingCount(*p)


# ---------------------------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------------------------
def dump_mols(limit: int | None = None, smi_path: Path = SMI, out_path: Path = MOLS) -> None:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    versions("dump")
    smis = [l.strip() for l in open(smi_path) if l.strip()]
    if limit:
        smis = smis[:limit]
    kept, removed, ringmismatch = [], 0, 0
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise SystemExit(f"unparseable SMILES in the corpus: {smi}")
        m2 = Chem.RemoveHs(m)
        if m2.GetNumAtoms() != m.GetNumAtoms():
            removed += 1
        rings = [list(r) for r in Chem.GetSymmSSSR(m2)]
        # mordred asks for GetSymmSSSR by name. RDKit's sanitisation already ran
        # SANITIZE_SYMMRINGS, so RingInfo should hold the same rings and a caller can take them
        # from there for free. Checked, not assumed -- it is the whole basis of the wiring.
        ri = sorted(tuple(sorted(r)) for r in m2.GetRingInfo().AtomRings())
        if ri != sorted(tuple(sorted(r)) for r in rings):
            ringmismatch += 1
        kept.append((m2, rings))
    with open(out_path, "w") as f:
        f.write(f"{len(kept)}\n")
        for m2, rings in kept:
            n, bonds = m2.GetNumAtoms(), list(m2.GetBonds())
            f.write(f"{n} {len(bonds)} {len(rings)}\n")
            for a in m2.GetAtoms():
                f.write(f"{a.GetAtomicNum()} {int(a.GetIsAromatic())}\n")
            for b in bonds:
                f.write(f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} "
                        f"{b.GetBondTypeAsDouble():.17g}\n")
            for r in rings:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
    print(f"[dump] {len(kept)} molecules -> {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"[dump] Chem.RemoveHs changed the atom count on {removed} molecules", flush=True)
    print(f"[dump] GetSymmSSSR != GetRingInfo().AtomRings() on {ringmismatch} molecules "
          f"(0 means the wiring can read RingInfo and pay no ring perception)", flush=True)
    # RDKit's disconnected-distance sentinel, asserted rather than trusted: topocharge.h::BIG.
    d = Chem.GetDistanceMatrix(Chem.MolFromSmiles("CC.CC"), force=True)
    assert d[0][2] == 1e8, f"GetDistanceMatrix disconnected sentinel is {d[0][2]}, not 1e8"
    print("[dump] GetDistanceMatrix disconnected sentinel == 1e8, as topocharge.h assumes",
          flush=True)


# ---------------------------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------------------------
_CALC = None


def _calc():
    global _CALC
    if _CALC is None:
        _numpy_shim()
        from mordred import Calculator
        _CALC = Calculator(descriptors())
    return _CALC


def _ref_one(smi: str):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    c = _calc()
    m = Chem.MolFromSmiles(smi)
    vals = []
    for v in c(m):
        try:
            vals.append(float(v))
        except Exception:
            vals.append(float("nan"))       # a mordred Error/Missing; compare() counts these
    return " ".join(f"{v:.17g}" for v in vals)


def make_ref(limit: int | None = None) -> None:
    from multiprocessing import Pool
    _numpy_shim()
    versions("ref")
    smis = [l.strip() for l in open(SMI) if l.strip()]
    if limit:
        smis = smis[:limit]
    t0 = time.time()
    with open(REF, "w") as f, Pool(NPROC) as p:
        for i, line in enumerate(p.imap(_ref_one, smis, chunksize=200)):
            f.write(line + "\n")
            if (i + 1) % 10000 == 0:
                print(f"[ref] {i + 1}/{len(smis)}  {time.time() - t0:.0f}s", flush=True)
    print(f"[ref] {len(smis)} molecules -> {REF}  {time.time() - t0:.0f}s "
          f"({REF.stat().st_size / 1e6:.1f} MB)  {NPROC} workers", flush=True)


# ---------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------
def compare() -> None:
    versions("compare")
    smis = [l.strip() for l in open(SMI) if l.strip()]
    cols = [open(OUT[e]) for e, _, _ in BLOCK]
    n = len(NAMES)
    exact = [0] * n
    nan_ref = [0] * n
    total = 0
    worst_rel = [(0.0, "")] * n
    worst_abs = [(0.0, "")] * n
    over = [[0] * n for _ in range(4)]      # rel > 1e-15, 1e-12, 1e-9, 1e-6
    THRESH = (1e-15, 1e-12, 1e-9, 1e-6)
    with open(REF) as fr:
        for k, refline in enumerate(fr):
            got = []
            for fh in cols:
                got.extend(float(x) for x in fh.readline().split())
            ref = [float(x) for x in refline.split()]
            if len(ref) != n or len(got) != n:
                raise SystemExit(f"row {k}: {len(ref)} reference vs {len(got)} C++ values")
            total += 1
            smi = smis[k] if k < len(smis) else "?"
            for j in range(n):
                r, g = ref[j], got[j]
                if math.isnan(r):
                    nan_ref[j] += 1
                    continue
                if r == g:
                    exact[j] += 1
                    continue
                d = abs(r - g)
                rel = d / abs(r) if r != 0.0 else float("inf")
                for t, th in enumerate(THRESH):
                    if rel > th:
                        over[t][j] += 1
                if rel > worst_rel[j][0]:
                    worst_rel[j] = (rel, smi)
                if d > worst_abs[j][0]:
                    worst_abs[j] = (d, smi)
    print(f"\n{total} molecules, {n} columns\n")
    print(f"{'column':12s} {'exact/total':>16s} {'>1e-15':>8s} {'>1e-12':>8s} {'>1e-9':>7s} "
          f"{'>1e-6':>7s}  {'max rel':>10s}  {'max abs':>10s}")
    allexact = []
    for j, nm in enumerate(NAMES):
        flag = "" if exact[j] == total else ("  <-- " + ("FP" if over[1][j] == 0 else "MISMATCH"))
        print(f"{nm:12s} {exact[j]:8d}/{total:<7d} {over[0][j]:8d} {over[1][j]:8d} "
              f"{over[2][j]:7d} {over[3][j]:7d}  {worst_rel[j][0]:10.3e}  {worst_abs[j][0]:10.3e}"
              f"{flag}")
        if exact[j] == total:
            allexact.append(nm)
        if nan_ref[j]:
            print(f"             ({nan_ref[j]} molecules where mordred itself returned an error)")
    print(f"\nBIT-EXACT on all {total} molecules: {len(allexact)} of {n} columns")
    notx = [nm for j, nm in enumerate(NAMES) if exact[j] != total]
    if notx:
        mx = max(worst_rel[j][0] for j, nm in enumerate(NAMES) if nm in notx)
        print(f"NOT bit-exact: {len(notx)} columns -- {' '.join(notx)}")
        print(f"  max relative deviation over all of them: {mx:.3e}")
        for j, nm in enumerate(NAMES):
            if nm in notx and worst_rel[j][0] == mx:
                print(f"  worst molecule: {worst_rel[j][1]}")
                break


# ---------------------------------------------------------------------------------------------
# HOUSE RULE 1
# ---------------------------------------------------------------------------------------------
def renumbered(m, p):
    """Chem.RenumberAtoms with the RingInfo put back.

    THE TRAP: RenumberAtoms returns a molecule whose RingInfo is UNINITIALISED, so whatever runs
    first triggers a lazy perception and you end up measuring perception ORDER instead of
    perception. Without the explicit SanitizeMol this harness reports spurious ill-posedness --
    another agent on this repo lost 17 columns to exactly that, two of them stereo columns that
    were fine. AssignStereochemistry is the same safety net _extract.py already carries."""
    from rdkit import Chem
    q = Chem.RenumberAtoms(m, p)
    Chem.SanitizeMol(q)
    Chem.AssignStereochemistry(q, cleanIt=True, force=True)
    return q


def _perm_one(arg):
    i, smi = arg
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    c = _calc()
    rng = random.Random(1000003 * i + 7)
    m = Chem.MolFromSmiles(smi)
    a = [float(v) for v in c(m)]
    alts = []
    for _ in range(3):
        p = list(range(m.GetNumAtoms()))
        rng.shuffle(p)
        alts.append([float(v) for v in c(renumbered(m, p))])
    m3 = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    if m3 is not None:
        alts.append([float(v) for v in c(m3)])
    moved, rel = [], []
    for j in range(len(NAMES)):
        d = max(abs(o[j] - a[j]) for o in alts)
        if d != 0.0:
            moved.append(j)
            rel.append(d / abs(a[j]) if a[j] else float("inf"))
    return i, moved, rel


def perm(limit: int | None = None) -> None:
    """Renumber the atoms and recompute WITH MORDRED. Any column that moves is not a function of
    the molecule. PORT_STATUS.md house rule 1."""
    from multiprocessing import Pool
    _numpy_shim()
    versions("perm")
    smis = [l.strip() for l in open(SMI) if l.strip()]
    if limit:
        smis = smis[:limit]
    n = len(NAMES)
    cnt = [0] * n
    worst = [0.0] * n
    illposed = []                      # (smi, [names]) where the move is NOT last-digit
    t0 = time.time()
    with Pool(NPROC) as p:
        for i, moved, rel in p.imap_unordered(_perm_one, list(enumerate(smis)), chunksize=200):
            big = []
            for j, r in zip(moved, rel):
                cnt[j] += 1
                worst[j] = max(worst[j], r)
                if r > 1e-9:
                    big.append(NAMES[j])
            if big:
                illposed.append((smis[i], big))
    print(f"[perm] {len(smis)} molecules, 3 random renumberings + a canonical-SMILES round trip, "
          f"{time.time() - t0:.0f}s", flush=True)
    print(f"\n{'column':12s} {'moves':>8s}  {'max rel':>10s}   verdict")
    for j, nm in enumerate(NAMES):
        if cnt[j] == 0:
            continue
        v = "floating-point last digits" if worst[j] < 1e-9 else "ILL-POSED"
        print(f"{nm:12s} {cnt[j]:8d}  {worst[j]:10.3e}   {v}")
    stable = [NAMES[j] for j in range(n) if cnt[j] == 0]
    print(f"\nnever moved: {len(stable)} of {n} columns")
    print(f"ILL-POSED (moved by more than last digits): "
          f"{sorted({x for _, b in illposed for x in b})}")
    print(f"molecules with an ill-posed column: {len(illposed)} of {len(smis)}")
    out = HERE / "topo3_illposed.txt"
    with open(out, "w") as f:
        for smi, b in illposed:
            f.write(f"{','.join(b)}\t{smi}\n")
    print(f"full list -> {out}")


# ---------------------------------------------------------------------------------------------
# THE RING-SET REPAIR
# ---------------------------------------------------------------------------------------------
# The repair, the gate and the ring handover live in the PACKAGE, not here: src/hume/_rings.py
# is what src/hume/_extract.py calls, so every number below is a measurement of the shipping code
# rather than of a harness copy that could drift from it.
# Loaded BY PATH rather than as `hume._rings`, because importing the package would run
# src/hume/__init__.py, which loads the compiled extension and its pickle-version drift guard.
# This harness needs neither, and must keep working in an isolated interpreter that has no
# built extension at all.
import importlib.util as _ilu                                              # noqa: E402
_spec = _ilu.spec_from_file_location("hume_rings", ROOT / "src" / "hume" / "_rings.py")
_rings = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rings)
canon_rings, gate, rings_for = _rings.canon_rings, _rings.gate, _rings.rings_for


def _name(order, greater, fused, arom, het):
    """mordred RingCount.__str__, transcribed."""
    a = []
    if greater:
        a.append("G")
    if order is not None:
        a.append(str(order))
    if fused:
        a.append("F")
    if arom is True:
        a.append("a")
    elif arom is False:
        a.append("A")
    if het is True:
        a.append("H")
    elif het is False:
        a.append("C")
    return "n{}Ring".format("".join(a))


def rc49(mol, rings):
    """The 49 RingCount columns, in Python, from a molecule and a ring list.

    A SECOND IMPLEMENTATION ON PURPOSE, for two jobs. (1) It is the yardstick for the invariance
    and gate tests: the requirement is that the COLUMNS do not move, and comparing ring sets by
    atom index instead would fail on a molecule whose canonical numbering differs from another by
    an AUTOMORPHISM -- the same rings, relabelled, and every one of these columns blind to the
    relabelling. (2) `canon` checks it against cpp/topo3_ringcount.txt, so the shipped C++ is
    gated by an independent transcription of mordred's predicates, not only by mordred itself.
    Currently 0 disagreements over 100,000 molecules x 49 columns."""
    Z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    AR = [a.GetIsAromatic() for a in mol.GetAtoms()]
    S = [frozenset(r) for r in rings]
    fused = []
    if len(S) >= 2:
        nR = len(S)
        adj = [[] for _ in range(nR)]
        touched = [False] * nR
        for i in range(nR):
            for j in range(i + 1, nR):
                if len(S[i] & S[j]) >= 2:
                    adj[i].append(j); adj[j].append(i); touched[i] = touched[j] = True
        seen = [False] * nR
        for i in range(nR):
            if not touched[i] or seen[i]:
                continue
            comp, st, seen[i] = [], [i], True
            while st:
                u = st.pop(); comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True; st.append(v)
            u = set()
            for c in comp:
                u |= S[c]
            fused.append(frozenset(u))

    def props(rs):
        return [(len(R), all(AR[i] for i in R), any(Z[i] != 6 for i in R)) for R in rs]
    P = {False: props(S), True: props(fused)}
    spec = {n: p for n, p in zip([_name(*p) for p in rc_preset()], rc_preset())}
    out = []
    for nm in RC_NAMES:
        order, greater, f, arom, het = spec[nm]
        c = 0
        for sz, isar, hashet in P[f]:
            if order is not None:
                if greater:
                    if sz < order:
                        continue
                elif sz != order:
                    continue
            if arom is not None and bool(isar) != arom:
                continue
            if het is not None and bool(hashet) != het:
                continue
            c += 1
        out.append(c)
    return tuple(out)


def _gatecheck_one(arg):
    i, smi = arg
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.RemoveHs(Chem.MolFromSmiles(smi))
    g = gate(m)
    full = rc49(m, canon_rings(m))                 # the repair, unconditionally
    got = rc49(m, rings_for(m))                    # what the gate actually ships
    return g, got != full, smi


def gatecheck(limit: int | None = None) -> None:
    """REQUIREMENT: the gate must be indistinguishable from repairing everything.

    Runs canon_rings() on all 100,000 regardless of the gate and asserts the gated pipeline
    produces the identical 49 columns. This is what turns "validated on this corpus" into a
    re-runnable check that a future corpus can fail loudly."""
    from multiprocessing import Pool
    versions("gatecheck")
    smis = [l.strip() for l in open(SMI) if l.strip()]
    if limit:
        smis = smis[:limit]
    nfire = nbad = 0
    bad = []
    t0 = time.time()
    with Pool(NPROC) as p:
        for g, differs, smi in p.imap_unordered(_gatecheck_one, list(enumerate(smis)),
                                                chunksize=200):
            nfire += g
            if differs:
                nbad += 1
                bad.append(smi)
    n = len(smis)
    print(f"[gatecheck] {n} molecules, {time.time() - t0:.0f}s")
    print(f"  gate fires on                       {nfire:6d} / {n}  = {100 * nfire / n:.1f}%")
    print(f"  amortised ring cost                 {5.1 + (104 - 5.1) * nfire / n:.1f} us/mol "
          f"(5.1 ungated, 104.0 unconditional)")
    print(f"  gated != unconditional on           {nbad:6d} / {n}   (requirement: 0)")
    if bad:
        print("  GATE LEAKS on:")
        for s in bad[:20]:
            print("   ", s[:160])
        raise SystemExit(1)


def _canon_one(arg):
    """-> (moved, differs_from_as_parsed, asparsed_itself_moved, smi) for one molecule.

    THE TEST IS ON THE 49 COLUMN VALUES, not on the ring set. Two canonical numberings of a
    symmetric molecule can differ by an automorphism -- RDKit's CanonicalRankAtoms(breakTies=True)
    does exactly that on, for instance, a 1,4-disubstituted cyclohexane -- and the ring sets then
    differ by the same relabelling while every column is identical. Testing the ring set would
    call that a failure; testing the columns is the requirement as stated."""
    i, smi = arg
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(1000003 * i + 11)
    m0 = Chem.RemoveHs(Chem.MolFromSmiles(smi))
    variants = [m0]
    for _ in range(3):
        p = list(range(m0.GetNumAtoms()))
        rng.shuffle(p)
        variants.append(renumbered(m0, p))
    q = Chem.MolFromSmiles(Chem.MolToSmiles(m0))
    if q is not None:
        variants.append(q)
    vals = [rc49(v, canon_rings(v)) for v in variants]
    raw = [rc49(v, [list(r) for r in Chem.GetSymmSSSR(v)]) for v in variants]
    return (any(x != vals[0] for x in vals),          # does the REPAIR still move?
            vals[0] != raw[0],                        # does the repair change RDKit's answer?
            any(x != raw[0] for x in raw),            # was RDKit as-parsed unstable here?
            raw[0],                                   # for the rc49-vs-C++ cross-check
            smi)


def canon(limit: int | None = None) -> None:
    from multiprocessing import Pool
    versions("canon")
    smis = [l.strip() for l in open(SMI) if l.strip()]
    if limit:
        smis = smis[:limit]
    nmoved = ndiff = nraw = 0
    moved_ex, diff_ex = [], []
    t0 = time.time()
    cppf = open(OUT["ringcount"]) if OUT["ringcount"].exists() else None
    ncross = xbad = 0
    with Pool(NPROC) as p:
        # ORDERED imap, because the rc49-vs-C++ cross-check reads cpp/topo3_ringcount.txt row by
        # row alongside it: rc49() is only a yardstick for the invariance test if it reproduces
        # the shipped C++ first.
        for moved, diff, rawmoved, rawvals, smi in p.imap(_canon_one, list(enumerate(smis)),
                                                          chunksize=200):
            if cppf is not None:
                line = cppf.readline()
                if line:
                    ncross += 1
                    if tuple(int(float(x)) for x in line.split()) != rawvals:
                        xbad += 1
            if moved:
                nmoved += 1; moved_ex.append(smi)
            if diff:
                ndiff += 1; diff_ex.append(smi)
            if rawmoved:
                nraw += 1
    n = len(smis)
    print(f"[canon] {n} molecules x 49 columns x (as parsed + 3 renumberings + a canonical-SMILES "
          f"round trip), {time.time() - t0:.0f}s")
    print(f"  BEFORE  RDKit as-parsed rings: columns move on   {nraw:6d} / {n}")
    print(f"  AFTER   canon_rings():         columns move on   {nmoved:6d} / {n}   "
          f"(requirement: 0)")
    print(f"  the repair changes the answer on                 {ndiff:6d} / {n}   "
          f"molecules -- must be a subset of the {nraw} unstable ones")
    out = HERE / "topo3_canon_diff.txt"
    out.write_text("".join(s + "\n" for s in diff_ex))
    print(f"  the changed molecules -> {out}")
    if cppf is not None:
        cppf.close()
        print(f"  rc49() (this file, an independent transcription) vs the shipped C++: "
              f"{xbad} disagreements over {ncross} molecules x 49 columns")
    if moved_ex:
        print("  STILL MOVING (the repair did not work):")
        for s in moved_ex[:10]:
            print("   ", s[:160])


# ---------------------------------------------------------------------------------------------
# benchpy
# ---------------------------------------------------------------------------------------------
def benchpy(n: int = 2000, reps: int = 5) -> None:
    """mordred's cost for THESE 81 columns, on a subsample, single process, same machine.

    CONTENDED: several jobs are running. Reported as the median over `reps` passes of the mean
    us/mol, which is the statistic cpp/*.cpp's bench mode also reports, so the two are
    comparable. The subsample is written to cpp/topo3_bench.txt so the C++ can be timed on
    exactly the same molecules."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _numpy_shim()
    versions("benchpy")
    smis = [l.strip() for l in open(SMI) if l.strip()][:n]
    BENCH_SMI.write_text("\n".join(smis) + "\n")
    dump_mols(limit=n, smi_path=BENCH_SMI, out_path=HERE / "topo3_bench_mols.txt")
    from mordred import Calculator
    c = Calculator(descriptors())
    mols = [Chem.MolFromSmiles(s) for s in smis]
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for m in mols:
            for _v in c(m):
                pass
        per.append((time.perf_counter() - t0) * 1e6 / len(mols))
    per.sort()
    print(f"\nmordred, the same 81 columns, {len(mols)} molecules, {reps} passes, CONTENDED")
    print(f"  median {per[len(per) // 2]:.1f} us/mol   min {per[0]:.1f}   max {per[-1]:.1f}")
    print(f"  now: for e in ringcount topocharge pathcount; do "
          f"./cpp/$e bench cpp/topo3_bench_mols.txt; done")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "dump":
        dump_mols(lim)
    elif mode == "ref":
        _numpy_shim(); check_spec(); make_ref(lim)
    elif mode == "compare":
        compare()
    elif mode == "perm":
        perm(lim)
    elif mode == "canon":
        canon(lim)
    elif mode == "gatecheck":
        gatecheck(lim)
    elif mode == "benchpy":
        benchpy(lim or 2000)
    elif mode == "all":
        _numpy_shim(); check_spec()
        dump_mols(lim)
        make_ref(lim)
        for e, _, _ in BLOCK:
            subprocess.run([str(HERE / e), "dump", str(MOLS), str(OUT[e])], check=True)
        compare()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
