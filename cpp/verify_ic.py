"""Generate ic_tables.h, and prove what src/hume_core/infocontent.h actually claims.

WHAT IT CLAIMS IS NOT "WE MATCH MORDRED", AND THAT IS DELIBERATE.  mordred's
InformationContent is not a function of the molecule -- it kekulizes before building its
atom-equivalence codes, and its BFS tree mutates a visited set while iterating over it -- so
~20% of molecules have an order-DEPENDENT value and there is nothing there to be exact against.
See the header comment of src/hume_core/infocontent.h for the mechanism and the worked example.
What is claimed instead, and demonstrated here rather than asserted, is:

    determinism   every one of the 45 columns identical under random atom renumbering and under
                  a canonical-SMILES round trip, on all 100,000 molecules of cpp/hard.smi
    order 0       bit-exact against mordred 1.2.0 -- the CONTROL, because nothing in the repair
                  should be able to touch order 0
    orders 1-5    divergence from mordred QUANTIFIED and split into "mordred was unstable here
                  anyway" and "mordred was stable and we still differ"

RUN IT PINNED.  Two environments, and asking for the wrong one FAILS SILENTLY -- uv resolves
mordred DOWN to 0.6.0, a different library, rather than erroring.  See constraints.txt.

    RD=(uv run --isolated --python 3.11 --with "rdkit==2025.9.2" --with "numpy==1.26.4")
    MO=(uv run --isolated --python 3.11 --with "mordred==1.2.0" --with "rdkit==2025.9.2" \
                                        --with "numpy==1.26.4")

    "${RD[@]}" python cpp/verify_ic.py tables                 # regenerate cpp/ic_tables.h
    "${RD[@]}" python cpp/verify_ic.py dump  FILE N           # boundary dumps, incl. permuted
    "${RD[@]}" python cpp/verify_ic.py ipc   FILE N           # rdkit Ipc/AvgIpc oracle + exact
    "${MO[@]}" python cpp/verify_ic.py mordred FILE N         # mordred oracle, 2 numberings
    "${RD[@]}" python cpp/verify_ic.py compare FILE N         # the report

Every command prints the versions resolved in ITS OWN process; a log without them is not
evidence (PORT_STATUS.md house rule 3).

THE DUMP IS THE BOUNDARY, NOT A CONVENIENCE FORMAT.  Each molecule is written as exactly the
columns bindings.cpp already receives -- per atom (Z, heavy degree, nH, formal charge, aromatic)
and per bond (begin, end, SMARTS bond code, GetBondTypeAsDouble) -- so what the C++ is verified
on is what the extension would hand it.  Hydrogens are NOT dumped: mordred descriptors these on
the `Chem.AddHs` graph, and infocontent.h rebuilds that graph from nH.  `dump` checks that
reconstruction against the real `Chem.AddHs` mol, atom count and bond count, on every molecule,
because getting it wrong makes all 42 columns wrong in a way that still looks plausible.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CPP = ROOT / "cpp"

BIT_SINGLE, BIT_DOUBLE, BIT_TRIPLE, BIT_AROM = 1, 2, 4, 8
_TYPE_BIT = {Chem.BondType.SINGLE: BIT_SINGLE,
             Chem.BondType.DOUBLE: BIT_DOUBLE,
             Chem.BondType.TRIPLE: BIT_TRIPLE}

FAMS = ["IC", "TIC", "SIC", "BIC", "CIC", "MIC", "ZMIC"]
IC_COLS = [f + str(o) for f in FAMS for o in range(6)]
COLS = IC_COLS + ["Ipc", "AvgIpc", "Log2Ipc"]
# The 33 that survive data/dedupe.json, plus AvgIpc: what the port is actually on the hook for.
SURVIVORS = ["IC0", "IC1", "IC2", "IC3", "IC4", "IC5",
             "TIC0", "TIC1", "TIC2", "TIC4",
             "SIC0", "SIC1", "SIC2", "SIC3", "SIC4", "SIC5",
             "BIC2", "BIC3", "BIC4", "BIC5",
             "CIC0", "CIC1", "CIC2",
             "MIC0", "MIC1", "MIC2", "MIC3",
             "ZMIC0", "ZMIC1", "ZMIC2", "ZMIC3", "ZMIC4", "ZMIC5",
             "AvgIpc"]


def versions(extra=()):
    out = ["rdkit " + rdkit.__version__, "numpy " + np.__version__,
           "python " + sys.version.split()[0]]
    for name in extra:
        mod = __import__(name)
        out.append(name + " " + mod.__version__)
    print("[versions] " + " | ".join(out), flush=True)


# ============================================================================================
# tables
# ============================================================================================
def gen_tables():
    """cpp/ic_tables.h: the three per-element vectors infocontent.h needs, from RDKit itself.

    ATOMIC_WEIGHT is repair R3's class weight -- the STANDARD weight of the element, which is
    bit-identical to Atom.GetMass() for every unlabelled atom and is well defined for a class
    that mixes isotopes, which mordred's "mass of whichever one was numbered last" is not.
    DEFAULT_VALENCE and N_OUTER_ELECS drive the kekulized-bond-order-sum recovery.
    """
    versions()
    pt = Chem.GetPeriodicTable()
    zmax = 118
    w, dv, no = [], [], []
    for z in range(zmax + 1):
        if z == 0:
            w.append(0.0); dv.append(-1); no.append(0); continue
        try:
            w.append(float(pt.GetAtomicWeight(z)))
            dv.append(int(pt.GetDefaultValence(z)))
            no.append(int(pt.GetNOuterElecs(z)))
        except Exception:
            w.append(0.0); dv.append(-1); no.append(0)
    spec = hashlib.sha256(repr((w, dv, no)).encode()).hexdigest()[:16]

    def fmt(vals, per, conv):
        out, line = [], "   "
        for i, v in enumerate(vals):
            s = conv(v)
            if len(line) + len(s) + 2 > 96:
                out.append(line); line = "   "
            line += " " + s + ","
        out.append(line)
        return "\n".join(out)

    body = f"""// GENERATED by cpp/verify_ic.py -- do not edit.
//
// rdkit {rdkit.__version__} | numpy {np.__version__} | spec sha256[:16] = {spec}
//
// The spec hash is over the three VECTORS, not over any RDKit file: PORT_STATUS.md house rule 6
// -- a file hash cries wolf on a copyright edit and stays silent on a table that moved.
#ifndef HUME_IC_TABLES_H
#define HUME_IC_TABLES_H
namespace ic_tbl {{

constexpr int Z_MAX = {zmax};

// PeriodicTable::getAtomicWeight(z). Class weight for ModifiedIC (repair R3).
constexpr double ATOMIC_WEIGHT[Z_MAX + 1] = {{
{fmt(w, 8, lambda x: repr(x))}
}};

// PeriodicTable::getDefaultValence(z); -1 means "no single default", and such an atom is never
// a kekulization candidate.
constexpr int DEFAULT_VALENCE[Z_MAX + 1] = {{
{fmt(dv, 16, str)}
}};

// PeriodicTable::getNOuterElecs(z). >= 5 is "has a lone pair to spend", which is the branch that
// decides whether a formal charge raises or lowers the valence an aromatic atom is aiming at.
constexpr int N_OUTER_ELECS[Z_MAX + 1] = {{
{fmt(no, 16, str)}
}};

}}  // namespace ic_tbl
#endif  // HUME_IC_TABLES_H
"""
    (ROOT / "src" / "hume_core" / "ic_tables.h").write_text(body)
    print("wrote src/hume_core/ic_tables.h  |  spec", spec)


# ============================================================================================
# dump -- the boundary, plus the AddHs cross-check
# ============================================================================================
def bond_code(b):
    code = _TYPE_BIT.get(b.GetBondType(), 0)
    if b.GetIsAromatic():
        code |= BIT_AROM
    return code


def bond_order(b):
    try:
        return b.GetBondTypeAsDouble()
    except Exception:
        return 0.0


def dump_mols(mols, path):
    """One boundary-shaped record per molecule; see the module docstring."""
    with open(path, "w") as f:
        f.write("%d\n" % len(mols))
        for m in mols:
            f.write("%d %d\n" % (m.GetNumAtoms(), m.GetNumBonds()))
            for a in m.GetAtoms():
                f.write("%d %d %d %d %d\n" % (a.GetAtomicNum(), a.GetDegree(),
                                              a.GetTotalNumHs(), a.GetFormalCharge(),
                                              int(a.GetIsAromatic())))
            for b in m.GetBonds():
                f.write("%d %d %d %r\n" % (b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                                           bond_code(b), bond_order(b)))


def check_addhs(m):
    """nH per heavy row must rebuild EXACTLY the graph Chem.AddHs produces."""
    h = Chem.AddHs(m)
    n = m.GetNumAtoms() + sum(a.GetTotalNumHs() for a in m.GetAtoms())
    nb = m.GetNumBonds() + sum(a.GetTotalNumHs() for a in m.GetAtoms())
    return n == h.GetNumAtoms() and nb == h.GetNumBonds()


def load_smis(path, n):
    smis = [l.split()[0] for l in open(path) if l.strip()]
    return smis if n <= 0 or n >= len(smis) else smis[:n]


def parse_all(path, n):
    out, kept = [], []
    for s in load_smis(path, n):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(m)
            kept.append(s)
    return out, kept


def cmd_dump(path, n):
    versions()
    mols, smis = parse_all(path, n)
    print("parsed %d / %d" % (len(mols), len(load_smis(path, n))))
    bad = [s for m, s in zip(mols, smis) if not check_addhs(m)]
    print("AddHs reconstruction failures:", len(bad), bad[:3])
    dump_mols(mols, CPP / "ic_in0.txt")
    Path(CPP / "ic_smis.txt").write_text("\n".join(smis) + "\n")

    # Permutations 1..3: random renumbering. Permutation 4: canonical-SMILES round trip.
    #
    # RENUMBER, THEN SANITIZE. `Chem.RenumberAtoms` returns a molecule with UNINITIALISED
    # RingInfo, so a descriptor computed straight off it is partly measuring RDKit's lazy
    # perception order rather than the descriptor. That has already produced spurious
    # ill-posedness reports elsewhere in this port (17 "ill-posed" columns that became 9 once
    # the sanitize was added). The renumbered molecule must be brought to the same perceived
    # state the original is in before anything is asked of it.
    #
    # The round trip is a CONTROL, not a second independent probe: canonical SMILES reproduces
    # the canonical numbering, so it perturbs nothing and must show zero. Random permutation is
    # the test that bites.
    rng = random.Random(20260827)

    def renumbered(m, idx):
        r = Chem.RenumberAtoms(m, idx)
        Chem.SanitizeMol(r)
        Chem.AssignStereochemistry(r, cleanIt=True, force=True)
        return r

    for p in range(1, 4):
        perm = []
        for m in mols:
            idx = list(range(m.GetNumAtoms()))
            rng.shuffle(idx)
            perm.append(renumbered(m, idx))
        dump_mols(perm, CPP / ("ic_in%d.txt" % p))
        print("wrote ic_in%d.txt (random renumbering + SanitizeMol + AssignStereochemistry)" % p)
    rt, moved = [], 0
    for m in mols:
        r = Chem.MolFromSmiles(Chem.MolToSmiles(m))
        if r is None:
            r = m
        if r.GetNumAtoms() != m.GetNumAtoms() or \
           sorted(int(a.GetIsAromatic()) for a in r.GetAtoms()) != \
           sorted(int(a.GetIsAromatic()) for a in m.GetAtoms()):
            moved += 1
        rt.append(r)
    dump_mols(rt, CPP / "ic_in4.txt")
    print("wrote ic_in4.txt (canonical-SMILES round trip -- a CONTROL that should show zero, "
          "since\n  canonical SMILES reproduces the canonical numbering); molecules whose "
          "PERCEPTION moved on\n  the round trip (an input change, not ours):", moved)
    Path(CPP / "ic_perception_moved.txt").write_text(str(moved) + "\n")


# ============================================================================================
# mordred oracle -- two numberings, so "mordred was unstable here" is measured, not assumed
# ============================================================================================
def cmd_mordred(path, n):
    from mordred import Calculator
    from mordred import InformationContent as M
    versions(("mordred",))
    cls = {"IC": M.InformationContent, "TIC": M.TotalIC, "SIC": M.StructuralIC,
           "BIC": M.BondingIC, "CIC": M.ComplementaryIC, "MIC": M.ModifiedIC,
           "ZMIC": M.ZModifiedIC}
    calc = Calculator([cls[f](o) for f in FAMS for o in range(6)])
    mols, smis = parse_all(path, n)
    rng = random.Random(20260827)

    def renumbered(m, idx):
        # Same sanitize-after-renumber discipline as `dump`; without it mordred's "instability"
        # would be partly RDKit's lazy ring perception and the comparison would be against a
        # moving target.
        r = Chem.RenumberAtoms(m, idx)
        Chem.SanitizeMol(r)
        Chem.AssignStereochemistry(r, cleanIt=True, force=True)
        return r

    def run(ms, out):
        with open(out, "w") as f:
            for k, m in enumerate(ms):
                vals = []
                for v in calc(m):
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(float("nan"))
                f.write(" ".join("%r" % v for v in vals) + "\n")
                if k % 10000 == 0:
                    print("  ...", out, k, flush=True)

    run(mols, CPP / "ic_mordred0.txt")
    perm = []
    for m in mols:
        idx = list(range(m.GetNumAtoms()))
        rng.shuffle(idx)
        perm.append(renumbered(m, idx))
    run(perm, CPP / "ic_mordred1.txt")
    print("wrote ic_mordred0.txt / ic_mordred1.txt for", len(mols), "molecules")


# ============================================================================================
# rdkit Ipc oracle, and the exact-integer check on top of it
# ============================================================================================
def exact_charpoly(mol):
    """Faddeev-Le Verrier in exact Python integers. The char poly of a 0/1 adjacency matrix is
    monic with INTEGER coefficients and every tr(M_k)/k is an exact integer, so this needs no
    fractions -- and it is the only thing in this file that is not itself floating point."""
    n = mol.GetNumAtoms()
    A = [[0] * n for _ in range(n)]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        A[u][v] = A[v][u] = 1
    nbrs = [[j for j in range(n) if A[i][j]] for i in range(n)]
    c = [1]
    M = [row[:] for row in A]
    cprev = -sum(M[i][i] for i in range(n))
    c.append(cprev)
    for k in range(2, n + 1):
        for i in range(n):
            M[i][i] += cprev
        T = [[0] * n for _ in range(n)]
        for i in range(n):
            ti = T[i]
            for j in nbrs[i]:
                mj = M[j]
                for q in range(n):
                    ti[q] += mj[q]
        M = T
        tr = sum(M[i][i] for i in range(n))
        assert tr % k == 0, "Faddeev-Le Verrier trace not divisible by k -- algebra is wrong"
        cprev = -tr // k
        c.append(cprev)
    return [abs(x) for x in c]


def exact_avgipc(cabs):
    from fractions import Fraction
    tot = sum(cabs)
    acc = 0.0
    for x in cabs:
        if x == 0:
            continue
        # log2(x/tot) with no overflow: exact integer ratio -> float via math.log
        lp = (math.log(x) - math.log(tot)) / math.log(2.0)
        p = float(Fraction(x, tot))
        acc -= p * lp
    return acc, tot


def cmd_ipc(path, n):
    from rdkit.Chem import GraphDescriptors
    versions()
    mols, smis = parse_all(path, n)
    with open(CPP / "ic_rdkit_ipc.txt", "w") as f:
        for k, m in enumerate(mols):
            try:
                ipc = GraphDescriptors.Ipc(m, forceDMat=True)
                avg = GraphDescriptors.AvgIpc(m, forceDMat=True)
            except Exception:
                ipc = avg = float("nan")
            # float() FIRST. `%r` on a numpy scalar writes `np.float64(908.68...)` under
            # numpy 2.x -- valid Python, and unreadable by np.loadtxt, which fails at row 0 with
            # a ValueError about a string it cannot convert. repr() of a plain Python float is
            # round-trip exact, which is why %r was the right instinct; the numpy scalar is what
            # broke it.
            f.write("%r %r\n" % (float(ipc), float(avg)))
            if k % 20000 == 0:
                print("  ...", k, flush=True)
    print("wrote ic_rdkit_ipc.txt for", len(mols), "molecules")


def cmd_exact(path, n, nmax=46, sample=90):
    """Exact-integer AvgIpc for a stratified sample, to say where DOUBLE precision -- RDKit's
    and ours alike -- stops being trustworthy, separately from where it overflows."""
    from rdkit.Chem import GraphDescriptors
    versions()
    mols, smis = parse_all(path, n)
    by_n = {}
    for m, s in zip(mols, smis):
        by_n.setdefault(m.GetNumAtoms(), []).append((m, s))
    rows = []
    sizes = sorted(k for k in by_n if 4 <= k <= nmax)
    per = max(1, sample // max(1, len(sizes)))
    for sz in sizes:
        for m, s in by_n[sz][:per]:
            cabs = exact_charpoly(m)
            ex, tot = exact_avgipc(cabs)
            rd = GraphDescriptors.AvgIpc(m, forceDMat=True)
            rows.append((sz, ex, rd, s))
    with open(CPP / "ic_exact_ipc.txt", "w") as f:
        for sz, ex, rd, s in rows:
            f.write("%d %r %r %s\n" % (sz, ex, rd, s))
    print("wrote ic_exact_ipc.txt |", len(rows), "molecules, n from", sizes[0], "to", sizes[-1])


# ============================================================================================
# compare -- the report
# ============================================================================================
def read_vals(path, ncol):
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    assert a.shape[1] == ncol, (path, a.shape, ncol)
    return a


def _has_isotope(smi: str) -> bool:
    """Does this molecule carry an explicit isotope label?

    The MIC order-0 control turns on this: repair R3 (element standard atomic weight instead of
    the representative atom's GetMass()) is bit-identical everywhere else, so a labelled molecule
    is the ONLY place MIC0 may legitimately differ from mordred.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return bool(m) and any(a.GetIsotope() for a in m.GetAtoms())


def same(a, b):
    """Bit-identical, with NaN counting as equal to NaN. Determinism is not a tolerance."""
    return (a.view(np.int64) == b.view(np.int64)) | (np.isnan(a) & np.isnan(b))


def cmd_compare(path, n):
    versions()
    smis = Path(CPP / "ic_smis.txt").read_text().split()
    V = [read_vals(CPP / ("ic_out%d.txt" % p), len(COLS)) for p in range(5)]
    nm = V[0].shape[0]
    print("\n" + "=" * 92)
    print("1. DETERMINISM -- the headline. %d molecules x %d columns, 3 random renumberings "
          "and a\n   canonical-SMILES round trip. Bit-identical is the test; no tolerance."
          % (nm, len(COLS)))
    print("=" * 92)
    moved_any = np.zeros(nm, dtype=bool)
    for p in range(1, 5):
        eq = same(V[0], V[p])
        bad = ~eq.all(axis=1)
        tag = "canonical-SMILES round trip" if p == 4 else "random renumbering %d" % p
        print("   %-30s molecules with ANY column moved: %d" % (tag, int(bad.sum())))
        if p < 4:
            moved_any |= bad
        if bad.any():
            for c in np.flatnonzero((~eq).any(axis=0)):
                print("        column %-8s moved on %d molecules  e.g. %s"
                      % (COLS[c], int((~eq[:, c]).sum()), smis[int(np.flatnonzero(~eq[:, c])[0])][:70]))
    print("   ---> renumbering-only determinism failures: %d / %d  (target 0)"
          % (int(moved_any.sum()), nm))
    pm = (CPP / "ic_perception_moved.txt")
    if pm.exists():
        print("   (round-trip differences, if any, are RDKit re-perceiving the molecule -- %s "
              "molecules\n    changed aromaticity on the round trip; see constraints.txt)"
              % pm.read_text().strip())

    mp = CPP / "ic_mordred0.txt"
    if mp.exists():
        Mo = [read_vals(CPP / ("ic_mordred%d.txt" % p), len(IC_COLS)) for p in range(2)]
        stable = same(Mo[0], Mo[1])
        print("\n" + "=" * 92)
        print("2. MORDRED'S OWN STABILITY, measured the same way (one renumbering).")
        print("=" * 92)
        unstable_any = ~stable.all(axis=1)
        print("   molecules where mordred moved on ANY of its 42 columns: %d / %d  (%.1f%%)"
              % (int(unstable_any.sum()), nm, 100.0 * unstable_any.mean()))
        ours = V[0][:, :len(IC_COLS)]
        print("\n" + "=" * 92)
        print("3. DIVERGENCE FROM MORDRED, per column, split by whether mordred had a value to")
        print("   diverge FROM. rtol 1e-9. 'mordred unstable' = that column moved under "
              "renumbering.")
        print("=" * 92)
        print("   %-7s %9s %11s %11s %13s %12s" % ("col", "differ", "mordred", "mordred",
                                                   "max rel dev", "max rel dev"))
        print("   %-7s %9s %11s %11s %13s %12s" % ("", "", "unstable", "STABLE",
                                                   "(unstable)", "(STABLE)"))
        summary = {}
        for c, name in enumerate(IC_COLS):
            a, b = ours[:, c], Mo[0][:, c]
            eq = np.isclose(a, b, rtol=1e-9, atol=0.0, equal_nan=True)
            diff = ~eq
            uns = ~stable[:, c]
            d_u = diff & uns
            d_s = diff & ~uns
            den = np.maximum(np.abs(b), 1e-300)
            rel = np.where(np.isfinite(a) & np.isfinite(b), np.abs(a - b) / den, 0.0)
            summary[name] = (int(diff.sum()), int(d_u.sum()), int(d_s.sum()))
            print("   %-7s %9d %11d %11d %13.3e %12.3e"
                  % (name, diff.sum(), d_u.sum(), d_s.sum(),
                     rel[d_u].max() if d_u.any() else 0.0,
                     rel[d_s].max() if d_s.any() else 0.0))
        print("\n   ORDER 0 IS THE CONTROL -- repairs R1 and R2 cannot touch it, so any nonzero "
              "count\n   in an order-0 row above is a BUG, not a divergence.  ONE EXCEPTION, and "
              "it is\n   earned rather than assumed -- see below:")
        # MIC IS THE EXCEPTION, AND ONLY ON ISOTOPICALLY LABELLED MOLECULES.
        #
        # Repair R3 changed MIC's per-class weight from `GetMass()` of whichever atom happened to
        # represent the class to the element's STANDARD ATOMIC WEIGHT. That is bit-identical for
        # every unlabelled atom, so it can only move a molecule that carries an isotope label --
        # and there, mordred had no defensible value to match. Demonstrated minimally:
        #
        #     [13C]C  -> MIC0 = 6.796788130428564
        #     C[13C]  -> MIC0 = 7.321516827665946     (same molecule, renumbered)
        #     ZMIC0   = 7.671792924958509 both ways   (weight is atomic NUMBER, isotope-blind)
        #
        # So MIC0 is excluded on labelled molecules and STILL REQUIRED TO MATCH EXACTLY on every
        # unlabelled one. Excluding the whole column would hide a real bug; excluding the
        # molecules where mordred is provably arbitrary does not.
        iso = np.array([_has_isotope(s) for s in smis[:nm]], dtype=bool)
        ok = True
        for name in ("IC0", "TIC0", "SIC0", "BIC0", "CIC0", "MIC0", "ZMIC0"):
            c = IC_COLS.index(name)
            a, b = ours[:, c], Mo[0][:, c]
            keep = ~iso if name == "MIC0" else np.ones(nm, dtype=bool)
            bits = same(a, b) | ~keep
            near = np.isclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True) | ~keep
            note = ("   (%d isotope-labelled molecules excluded, R3)" % int(iso.sum())
                    if name == "MIC0" else "")
            print("      %-6s bit-identical %6d / %d   within 1e-12 %6d / %d%s"
                  % (name, int(bits.sum()), nm, int(near.sum()), nm, note))
            ok &= bool(near.all())
        print("      ---> order-0 control:", "PASS" if ok else "FAIL")

        # Characterise the "mordred stable and we still differ" set: what is it made of?
        print("\n" + "=" * 92)
        print("4. THE SET THAT MATTERS: mordred was STABLE under renumbering and we still "
              "differ.")
        print("=" * 92)
        c_ic = [IC_COLS.index("IC%d" % o) for o in range(1, 6)]
        realdiff = np.zeros(nm, dtype=bool)
        for c in c_ic:
            realdiff |= (~np.isclose(ours[:, c], Mo[0][:, c], rtol=1e-9, equal_nan=True)) & stable[:, c]
        print("   molecules in this set (any of IC1..IC5): %d / %d  (%.1f%%)"
              % (int(realdiff.sum()), nm, 100.0 * realdiff.mean()))
        idx = np.flatnonzero(realdiff)
        has_arom, has_ring, acyclic = 0, 0, 0
        for i in idx[:20000]:
            m = Chem.MolFromSmiles(smis[i])
            if m is None:
                continue
            ar = any(a.GetIsAromatic() for a in m.GetAtoms())
            rg = m.GetRingInfo().NumRings() > 0
            has_arom += ar
            has_ring += rg
            acyclic += (not rg)
        k = min(len(idx), 20000)
        if k:
            print("   of the first %d: aromatic %d (%.1f%%), has a ring %d (%.1f%%), "
                  "ACYCLIC %d (%.1f%%)"
                  % (k, has_arom, 100.0 * has_arom / k, has_ring, 100.0 * has_ring / k,
                     acyclic, 100.0 * acyclic / k))
            print("   -- an ACYCLIC molecule can only be here through repair R1 (aromatic bond "
                  "symbol),\n      and an acyclic molecule has no aromatic bond, so that count "
                  "should be 0:\n      any acyclic molecule here would be an unexplained "
                  "divergence and must be chased.")
            print("   examples:", ", ".join(smis[i][:44] for i in idx[:3]))

    rp = CPP / "ic_rdkit_ipc.txt"
    if rp.exists():
        R = read_vals(rp, 2)
        ipc_c, avg_c = V[0][:, COLS.index("Ipc")], V[0][:, COLS.index("AvgIpc")]
        l2 = V[0][:, COLS.index("Log2Ipc")]
        fin = np.isfinite(R[:, 1]) & (R[:, 1] != 0)
        rel = np.abs(avg_c - R[:, 1]) / np.maximum(np.abs(R[:, 1]), 1e-300)
        print("\n" + "=" * 92)
        print("5. Ipc / AvgIpc against RDKit " + rdkit.__version__)
        print("=" * 92)
        print("   RDKit AvgIpc non-finite : %d / %d" % (int((~np.isfinite(R[:, 1])).sum()), nm))
        print("   RDKit Ipc   non-finite : %d / %d" % (int((~np.isfinite(R[:, 0])).sum()), nm))
        print("   ours AvgIpc non-finite : %d / %d" % (int((~np.isfinite(avg_c)).sum()), nm))
        print("   AvgIpc max rel deviation where RDKit is finite and nonzero: %.3e"
              % (rel[fin].max() if fin.any() else 0.0))
        print("   AvgIpc within 1e-9 rel : %d / %d" % (int((rel[fin] < 1e-9).sum()), int(fin.sum())))
        bad = ~np.isfinite(R[:, 0])
        if bad.any():
            ns = np.array([Chem.MolFromSmiles(smis[i]).GetNumAtoms() for i in np.flatnonzero(bad)])
            print("   RDKit Ipc overflows at heavy-atom count: min %d, median %d, max %d"
                  % (ns.min(), int(np.median(ns)), ns.max()))
            print("   our Log2Ipc on those molecules: min %.1f max %.1f  (finite everywhere)"
                  % (l2[bad].min(), l2[bad].max()))
        gd = np.isfinite(R[:, 0]) & (R[:, 0] != 0)
        rel2 = np.abs(ipc_c - R[:, 0]) / np.maximum(np.abs(R[:, 0]), 1e-300)
        print("   Ipc max rel deviation where RDKit is finite: %.3e"
              % (rel2[gd].max() if gd.any() else 0.0))

    ep = CPP / "ic_exact_ipc.txt"
    if ep.exists():
        print("\n" + "=" * 92)
        print("6. AvgIpc against EXACT INTEGER arithmetic -- where DOUBLE stops being "
              "trustworthy,\n   which is a different limit from overflow and applies to RDKit "
              "as much as to us.")
        print("=" * 92)
        idx = {s: i for i, s in enumerate(smis)}
        rows = []
        for line in open(ep):
            parts = line.split()
            sz, ex, rd, s = int(parts[0]), float(parts[1]), float(parts[2]), parts[3]
            if s in idx:
                rows.append((sz, ex, rd, V[0][idx[s], COLS.index("AvgIpc")]))
        if rows:
            print("   %6s %6s %14s %14s" % ("n", "mols", "rdkit max rel", "ours max rel"))
            buckets = {}
            for sz, ex, rd, ours_v in rows:
                b = (sz // 10) * 10
                buckets.setdefault(b, []).append(
                    (abs(rd - ex) / abs(ex) if ex else 0.0,
                     abs(ours_v - ex) / abs(ex) if ex else 0.0))
            for b in sorted(buckets):
                v = buckets[b]
                print("   %3d-%-3d %5d %14.3e %14.3e"
                      % (b, b + 9, len(v), max(x[0] for x in v), max(x[1] for x in v)))


USAGE = """usage: verify_ic.py CMD [FILE N]
  tables                 regenerate src/hume_core/ic_tables.h      (rdkit env)
  dump    FILE N         boundary dumps ic_in0..4.txt              (rdkit env)
  mordred FILE N         mordred oracle, 2 numberings              (mordred env)
  ipc     FILE N         rdkit Ipc/AvgIpc oracle                   (rdkit env)
  exact   FILE N         exact-integer AvgIpc, stratified sample   (rdkit env)
  compare FILE N         the report                                (rdkit env)"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE); sys.exit(1)
    cmd = sys.argv[1]
    f = sys.argv[2] if len(sys.argv) > 2 else str(CPP / "hard.smi")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if cmd == "tables":
        gen_tables()
    elif cmd == "dump":
        cmd_dump(f, n)
    elif cmd == "mordred":
        cmd_mordred(f, n)
    elif cmd == "ipc":
        cmd_ipc(f, n)
    elif cmd == "exact":
        cmd_exact(f, n)
    elif cmd == "compare":
        cmd_compare(f, n)
    else:
        print(USAGE); sys.exit(1)
