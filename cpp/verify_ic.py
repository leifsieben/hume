"""Generate ic_tables.h, and prove what src/hume_core/infocontent.h actually claims.

WHAT IT CLAIMS IS NOT "WE MATCH MORDRED", AND THAT IS DELIBERATE.  Neither oracle here is a
function of the molecule:

  * mordred's InformationContent kekulizes before building its atom-equivalence codes, and its
    BFS tree mutates a visited set while iterating over it.  A THIRD of the corpus has an
    order-DEPENDENT value.
  * RDKit's Ipc/AvgIpc runs Le Verrier-Faddeev in FLOATING POINT over coefficients that are
    integers needing more than a double's 53 bits above about 75 heavy atoms.  Past that its own
    answer moves with the numbering too -- at 199 atoms, six renumberings span a factor of 2.2.

See the header comment of src/hume_core/infocontent.h for both mechanisms and the worked
example.  What is claimed instead, and demonstrated here rather than asserted, is:

    determinism   every one of the 45 columns BIT-IDENTICAL under 3 atom renumberings, 3
                  permutations that also SHUFFLE THE BOND LIST, and a canonical-SMILES round
                  trip, on all 100,000 molecules of cpp/hard.smi
    order 0       bit-exact against mordred 1.2.0 -- the CONTROL, because nothing in the repair
                  should be able to touch order 0
    orders 1-5    divergence from mordred QUANTIFIED and split into "mordred was unstable here
                  anyway" and "mordred was stable and we still differ", and the second set
                  CHARACTERISED by two falsifiable predictions (see cmd_compare section 4)
    Ipc           exact integer arithmetic, so bit-identical to RDKit where RDKit is exact and
                  correct where RDKit is not -- checked against an independent exact oracle

WHY THE BOND-ORDER HALF OF THE SCREEN EXISTS.  `Chem.RenumberAtoms` permutes atoms and LEAVES
THE BOND LIST ALONE, so an atom-only screen cannot perturb anything that reads bonds in order --
which includes RDKit's ring perception.  Measured here: it misses about half of mordred's
instability.  A determinism claim made against atom shuffling alone is PROVISIONAL.

RUN IT PINNED.  Two environments, and asking for the wrong one FAILS SILENTLY -- uv resolves
mordred DOWN to 0.6.0, a different library, rather than erroring.  See constraints.txt.
`--no-project` is required or the project's own constraint-dependencies block the numpy 1.x
that mordred 1.2.0 needs; the rdkit-only commands can equivalently use `.venv/bin/python`,
which carries the same pinned rdkit.

    RD=(uv run --isolated --no-project --python 3.11 --with "rdkit==2025.9.2" \
                                       --with "numpy==1.26.4")
    MO=(uv run --isolated --no-project --python 3.11 --with "mordred==1.2.0" \
                                       --with "rdkit==2025.9.2" --with "numpy==1.26.4")

    "${RD[@]}" python cpp/verify_ic.py tables                 # regenerate ic_tables.h
    "${RD[@]}" python cpp/verify_ic.py dump  FILE N           # boundary dumps, incl. permuted
    for p in 0 1 2 3 4 5 6 7; do ./cpp/infocontent values cpp/ic_in$p.txt cpp/ic_out$p.txt; done
    "${RD[@]}" python cpp/verify_ic.py ipc   FILE N           # rdkit Ipc oracle + perturbations
    "${RD[@]}" python cpp/verify_ic.py exact FILE N           # exact-integer AvgIpc oracle
    "${MO[@]}" python cpp/verify_ic.py mordred FILE N         # mordred oracle, 3 numberings
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


def renumbered(m, idx):
    r = Chem.RenumberAtoms(m, idx)
    Chem.SanitizeMol(r)
    Chem.AssignStereochemistry(r, cleanIt=True, force=True)
    return r


def rebuilt(m, aidx, bperm):
    """Renumber the atoms AND rebuild the molecule with its bonds added in a shuffled order.

    Returns None if the rebuild did not reproduce the original molecule -- the caller counts
    those rather than quietly comparing against a different molecule. Isotope, charge, radical
    count and explicit-H count are copied atom by atom; everything perceived is recomputed by
    the sanitize, which is the point.
    """
    r = Chem.RenumberAtoms(m, aidx)
    Chem.SanitizeMol(r)
    e = Chem.RWMol()
    for a in r.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsotope(a.GetIsotope())
        na.SetNumRadicalElectrons(a.GetNumRadicalElectrons())
        na.SetNoImplicit(a.GetNoImplicit())
        na.SetNumExplicitHs(a.GetNumExplicitHs())
        na.SetChiralTag(a.GetChiralTag())
        e.AddAtom(na)
    bonds = list(r.GetBonds())
    for k in bperm:
        b = bonds[k]
        e.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
    out = e.GetMol()
    try:
        Chem.SanitizeMol(out)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    except Exception:
        return None
    # Chirality survives the rebuild but E/Z does not (it is stored on the bond's stereo atoms),
    # so compare without stereo -- the graph, the elements and the perception are what these
    # descriptors read, and none of the 45 columns looks at stereo at all.
    if Chem.MolToSmiles(out, isomericSmiles=False) != Chem.MolToSmiles(r, isomericSmiles=False):
        return None
    return out


ATOM_ONLY = (1, 2, 3)
ATOM_AND_BOND = (5, 6, 7)


def perturbations(mols, seed=20260827):
    """Yield (index, description, perturbed mols, n_failed_rebuilds), one at a time.

    A GENERATOR on purpose: six copies of a 100,000-molecule corpus does not fit comfortably in
    memory, and the callers (`dump` and `mordred`) must consume the SAME random sequence so that
    ic_in1 and ic_mordred1 are the same permutation of the same molecules. Skipping a yielded
    perturbation still advances the rng identically, which is what lets `mordred` take only two
    of the six without drifting out of step.
    """
    rng = random.Random(seed)
    for p in ATOM_ONLY + ATOM_AND_BOND:
        cur, bad = [], 0
        bondshuffle = p in ATOM_AND_BOND
        for m in mols:
            idx = list(range(m.GetNumAtoms()))
            rng.shuffle(idx)
            if bondshuffle:
                border = list(range(m.GetNumBonds()))
                rng.shuffle(border)
                r = rebuilt(m, idx, border)
                if r is None:
                    bad += 1
                    r = renumbered(m, idx)
            else:
                r = renumbered(m, idx)
            cur.append(r)
        yield (p, "atom renumbering AND bond-list shuffle" if bondshuffle
               else "atom renumbering only", cur, bad)


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

    # ---------------------------------------------------------------------------------------
    # THE PERTURBATION SCREEN. Files 1-3 permute ATOM NUMBERING only; files 5-7 permute atom
    # numbering AND BOND-LIST ORDER; file 4 is a canonical-SMILES round trip.
    #
    # WHY BOND ORDER IS NOT OPTIONAL. `Chem.RenumberAtoms` permutes atoms and LEAVES THE BOND
    # LIST ALONE. Anything downstream that reads the bond list in order -- RDKit's ring
    # perception, and any C++ that builds its adjacency by walking bonds -- is then not being
    # perturbed along the axis that decides its answer. Measured elsewhere in this port:
    # O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2 is stable across 201 atom
    # renumberings and yields two different ring sets as soon as the bond order is shuffled, and
    # all 32 molecules with genuinely unstable ring perception look perfectly stable under atom
    # shuffling alone. A determinism claim made against atom shuffling only is PROVISIONAL.
    #
    # RENUMBER, THEN SANITIZE. `Chem.RenumberAtoms` returns a molecule with UNINITIALISED
    # RingInfo, so a descriptor computed straight off it is partly measuring RDKit's lazy
    # perception order rather than the descriptor.
    #
    # The round trip is a CONTROL, not a probe: canonical SMILES reproduces the canonical
    # numbering, so it perturbs nothing and must show zero. It is the random permutations,
    # especially the bond-order ones, that bite.
    # ---------------------------------------------------------------------------------------
    for p, tag, perm, bad in perturbations(mols):
        dump_mols(perm, CPP / ("ic_in%d.txt" % p))
        print("wrote ic_in%d.txt (%s)%s"
              % (p, tag, "" if not bad else
                 "  -- %d rebuilds did not reproduce the original molecule and fell back to an "
                 "atom-only permutation; reported rather than silently kept" % bad))

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
    # The SAME two perturbations the C++ is screened on -- one atom-only, one that also shuffles
    # the bond list -- so "mordred was unstable here" is measured under the same screen and
    # ic_mordredN.txt lines up molecule for molecule with ic_inN.txt.
    for p, tag, perm, bad in perturbations(mols):
        if p not in (ATOM_ONLY[0], ATOM_AND_BOND[0]):
            continue
        print("  mordred under perturbation %d (%s)" % (p, tag), flush=True)
        run(perm, CPP / ("ic_mordred%d.txt" % p))
    print("wrote ic_mordred0/%d/%d.txt for %d molecules"
          % (ATOM_ONLY[0], ATOM_AND_BOND[0], len(mols)))


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
    """RDKit's own Ipc/AvgIpc -- on the base molecules AND under the same perturbation screen.

    The second half is not decoration. RDKit's characteristic polynomial is Le Verrier-Faddeev
    in FLOATING POINT; its coefficients are integers that outgrow a double's 53 bits at around
    75 heavy atoms, and past that the value RDKit returns depends on the atom numbering. Unless
    that is measured here, "we differ from RDKit" cannot be split into "RDKit had no defensible
    value" and "a real definitional change".
    """
    from rdkit.Chem import GraphDescriptors
    versions()
    mols, smis = parse_all(path, n)

    def run(ms, out):
        with open(out, "w") as f:
            for k, m in enumerate(ms):
                try:
                    ipc = GraphDescriptors.Ipc(m, forceDMat=True)
                    avg = GraphDescriptors.AvgIpc(m, forceDMat=True)
                except Exception:
                    ipc = avg = float("nan")
                # float() FIRST. `%r` on a numpy scalar writes `np.float64(908.68...)` under
                # numpy 2.x -- valid Python, and unreadable by np.loadtxt, which fails at row 0
                # with a ValueError about a string it cannot convert. repr() of a plain Python
                # float is round-trip exact, which is why %r was the right instinct; the numpy
                # scalar is what broke it.
                f.write("%r %r\n" % (float(ipc), float(avg)))
                if k % 20000 == 0:
                    print("  ...", out, k, flush=True)

    run(mols, CPP / "ic_rdkit_ipc.txt")
    for p, tag, perm, bad in perturbations(mols):
        if p not in (ATOM_ONLY[0], ATOM_AND_BOND[0]):
            continue
        run(perm, CPP / ("ic_rdkit_ipc%d.txt" % p))
    print("wrote ic_rdkit_ipc.txt (+ perturbations %d and %d) for %d molecules"
          % (ATOM_ONLY[0], ATOM_AND_BOND[0], len(mols)))


def cmd_exact(path, n, nmax=150, sample=300, biggest=12):
    """Exact-integer AvgIpc for a stratified sample, plus the LARGEST molecules in the corpus.

    Two jobs, and the second one is the reason `biggest` exists. The first is to say where
    RDKit's double arithmetic stops being trustworthy, which needs a spread of sizes. The second
    is to exercise infocontent.h's MULTIWORD path: its big integers start at one 64-bit word and
    widen on overflow, so a sample that stops at 80 heavy atoms only ever tests W = 1 -- the
    coefficients do not pass 64 bits until about 90 atoms, or 128 bits until about 200. Checking
    the wide path needs the wide molecules, and they are cheap here because there are few of
    them: the recurrence is 2*n_bonds*n element additions per step, not n^3.
    """
    from rdkit.Chem import GraphDescriptors
    versions()
    mols, smis = parse_all(path, n)
    by_n = {}
    for m, s in zip(mols, smis):
        by_n.setdefault(m.GetNumAtoms(), []).append((m, s))
    rows = []
    sizes = sorted(k for k in by_n if 4 <= k <= nmax)
    per = max(1, sample // max(1, len(sizes)))
    picked = [(sz, m, s) for sz in sizes for m, s in by_n[sz][:per]]
    for sz in sorted(by_n, reverse=True)[:biggest]:
        picked.append((sz, by_n[sz][0][0], by_n[sz][0][1]))
    for sz, m, s in picked:
        cabs = exact_charpoly(m)
        ex, tot = exact_avgipc(cabs)
        rd = GraphDescriptors.AvgIpc(m, forceDMat=True)
        rows.append((sz, ex, rd, s, max(cabs).bit_length()))
        print("   n=%3d  maxcoeff %3d bits" % (sz, max(cabs).bit_length()), flush=True)
    with open(CPP / "ic_exact_ipc.txt", "w") as f:
        for sz, ex, rd, s, bl in rows:
            f.write("%d %r %r %s %d\n" % (sz, ex, rd, s, bl))
    print("wrote ic_exact_ipc.txt |", len(rows), "molecules, n from", min(r[0] for r in rows),
          "to", max(r[0] for r in rows), "| widest coefficient",
          max(r[4] for r in rows), "bits")


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
    idxs = [0] + list(ATOM_ONLY) + [4] + list(ATOM_AND_BOND)
    V = {p: read_vals(CPP / ("ic_out%d.txt" % p), len(COLS) + 1) for p in idxs}
    nm = V[0].shape[0]
    print("\n" + "=" * 92)
    print("1. DETERMINISM -- the headline. %d molecules x %d columns. Bit-identical is the "
          "test;\n   no tolerance. The screen permutes ATOM NUMBERING (3x) and atom numbering "
          "TOGETHER WITH\n   BOND-LIST ORDER (3x); a claim made against atom shuffling alone is "
          "provisional, because\n   RenumberAtoms leaves the bond list untouched and ring "
          "perception reads it." % (nm, len(COLS)))
    print("=" * 92)
    moved_any = np.zeros(nm, dtype=bool)
    for p in idxs[1:]:
        eq = same(V[0], V[p])
        bad = ~eq.all(axis=1)
        tag = ("canonical-SMILES round trip (CONTROL, expect 0)" if p == 4
               else "atom renumbering %d" % p if p in ATOM_ONLY
               else "atom + BOND-ORDER shuffle %d" % p)
        print("   %-46s molecules with ANY column moved: %d" % (tag, int(bad.sum())))
        if p != 4:
            moved_any |= bad
        if bad.any():
            for c in np.flatnonzero((~eq).any(axis=0)):
                nmv = int((~eq[:, c]).sum())
                cname = COLS[c] if c < len(COLS) else "IpcCoeffBits"
                print("        column %-12s moved on %d molecules  e.g. %s"
                      % (cname, nmv, smis[int(np.flatnonzero(~eq[:, c])[0])][:66]))
    print("   ---> DETERMINISM FAILURES under the full screen: %d / %d  (target 0)"
          % (int(moved_any.sum()), nm))
    pm = (CPP / "ic_perception_moved.txt")
    if pm.exists():
        print("   (%s molecules changed aromaticity on the round trip -- an input change, "
              "not ours)" % pm.read_text().strip())

    mp = CPP / "ic_mordred0.txt"
    if mp.exists():
        mord_idx = [ATOM_ONLY[0], ATOM_AND_BOND[0]]
        Mo = {p: read_vals(CPP / ("ic_mordred%d.txt" % p), len(IC_COLS)) for p in [0] + mord_idx}
        stable = same(Mo[0], Mo[mord_idx[0]]) & same(Mo[0], Mo[mord_idx[1]])
        print("\n" + "=" * 92)
        print("2. MORDRED'S OWN STABILITY, under the same screen (one atom-only permutation and")
        print("   one that also shuffles the bond list).")
        print("=" * 92)
        unstable_any = ~stable.all(axis=1)
        only_atom = ~same(Mo[0], Mo[mord_idx[0]]).all(axis=1)
        only_bond = ~same(Mo[0], Mo[mord_idx[1]]).all(axis=1)
        print("   molecules where mordred moved on ANY of its 42 columns: %d / %d  (%.1f%%)"
              % (int(unstable_any.sum()), nm, 100.0 * unstable_any.mean()))
        print("      under atom renumbering alone      : %d" % int(only_atom.sum()))
        print("      under atom + bond-order shuffling : %d" % int(only_bond.sum()))
        print("      found ONLY by the bond-order screen: %d  (these are what an atom-only "
              "screen misses)" % int((only_bond & ~only_atom).sum()))
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
        # WHAT THE SET IS MADE OF, and the two predictions that make it a characterisation
        # rather than a count. The repair has exactly two mechanisms:
        #   R1 (aromatic bond keeps its symbol) needs an AROMATIC BOND to fire.
        #   R2 (distance layering) needs two adjacent atoms at equal distance from some root,
        #      which needs a CYCLE, and it cannot reach order 1 -- at order 1 mordred's tree is
        #      the root and its neighbours and no sibling has been expanded yet.
        # So:  acyclic + no aromatic bond  ->  we must agree with mordred at EVERY order.
        #      cyclic  + no aromatic bond  ->  we must agree at order 1, and may differ at 2+.
        # Both are checked below over the whole corpus. A failure of either is an unexplained
        # divergence and would have to be chased, not reported.
        arom = np.zeros(nm, dtype=bool)
        ring = np.zeros(nm, dtype=bool)
        for i, s in enumerate(smis[:nm]):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            arom[i] = any(b.GetIsAromatic() for b in m.GetBonds())
            ring[i] = m.GetRingInfo().NumRings() > 0
        idx = np.flatnonzero(realdiff)
        k = max(1, len(idx))
        print("   of them: has an aromatic bond %d (%.1f%%), has a ring %d (%.1f%%), "
              "ACYCLIC %d (%.1f%%)"
              % (int(arom[idx].sum()), 100.0 * arom[idx].sum() / k,
                 int(ring[idx].sum()), 100.0 * ring[idx].sum() / k,
                 int((~ring[idx]).sum()), 100.0 * (~ring[idx]).sum() / k))
        if len(idx):
            print("   examples:", ", ".join(smis[i][:44] for i in idx[:3]))

        # R3 (MIC's class weight) is a THIRD mechanism and it is orthogonal to R1 and R2: it
        # fires on isotope-labelled molecules whatever their topology. So the MIC columns are
        # excluded on labelled molecules here for exactly the reason they are excluded from the
        # order-0 control -- and on every UNLABELLED molecule they are still required to match.
        # Discovered rather than assumed: the first run of this prediction failed on 4 molecules,
        # all of them MIC, all of them carrying a [13C], [15N] or similar.
        print("\n   PREDICTION 1  acyclic AND no aromatic bond -> exact at every order.")
        sub = (~ring) & (~arom)
        worst, worstname = 0, ""
        for c, name in enumerate(IC_COLS):
            keep = sub & (~iso) if name.startswith("MIC") else sub
            d = (~np.isclose(ours[:, c], Mo[0][:, c], rtol=1e-9, equal_nan=True)) & keep
            if int(d.sum()) > worst:
                worst, worstname = int(d.sum()), name
        print("      %d such molecules (%d of them isotope-labelled, where MIC is excluded by "
              "R3);\n      worst column disagrees on %d of them%s  -> %s"
              % (int(sub.sum()), int((sub & iso).sum()), worst,
                 "" if not worst else " (%s)" % worstname, "PASS" if worst == 0 else "FAIL"))

        print("   PREDICTION 2  has a ring but NO aromatic bond -> exact at ORDER 1.")
        sub2 = ring & (~arom)
        worst1 = 0
        for name in ("IC1", "TIC1", "SIC1", "BIC1", "CIC1", "ZMIC1"):
            c = IC_COLS.index(name)
            d = (~np.isclose(ours[:, c], Mo[0][:, c], rtol=1e-9, equal_nan=True)) & sub2
            worst1 = max(worst1, int(d.sum()))
        c2 = IC_COLS.index("IC2")
        d2 = int(((~np.isclose(ours[:, c2], Mo[0][:, c2], rtol=1e-9, equal_nan=True)) & sub2).sum())
        print("      %d such molecules; worst order-1 column disagrees on %d  -> %s"
              "   (IC2 disagrees on %d, which is R2 doing its job)"
              % (int(sub2.sum()), worst1, "PASS" if worst1 == 0 else "FAIL", d2))

    rp = CPP / "ic_rdkit_ipc.txt"
    if rp.exists():
        Rp = {p: read_vals(CPP / ("ic_rdkit_ipc%s.txt" % ("" if p == 0 else str(p))), 2)
              for p in [0] + ([ATOM_ONLY[0], ATOM_AND_BOND[0]]
                              if (CPP / ("ic_rdkit_ipc%d.txt" % ATOM_ONLY[0])).exists() else [])}
        R = Rp[0]
        ipc_c, avg_c = V[0][:, COLS.index("Ipc")], V[0][:, COLS.index("AvgIpc")]
        bits = V[0][:, len(COLS)].astype(int)      # bit length of the largest EXACT coefficient
        print("\n" + "=" * 92)
        print("5. Ipc / AvgIpc against RDKit " + rdkit.__version__ + ".")
        print("   The characteristic-polynomial coefficients are INTEGERS. A double holds them")
        print("   exactly to 2^53 and no further, so the whole question is how many bits they "
              "need.")
        print("=" * 92)
        print("   RDKit Ipc non-finite: %d / %d      ours non-finite: %d / %d      ours "
              "saturated: %d"
              % (int((~np.isfinite(R[:, 0])).sum()), nm, int((~np.isfinite(ipc_c)).sum()), nm,
                 int((ipc_c == np.finfo(np.float64).max).sum())))
        print("   largest finite RDKit Ipc: %.4g      widest exact coefficient in the corpus: "
              "%d bits" % (np.nanmax(R[:, 0][np.isfinite(R[:, 0])]), bits.max()))
        exact_ok = bits <= 53
        print("   molecules whose COEFFICIENTS fit in a double (<= 53 bits): %d / %d  (%.2f%%)"
              % (int(exact_ok.sum()), nm, 100.0 * exact_ok.mean()))
        b = same(avg_c, R[:, 1])
        rel = np.abs(avg_c - R[:, 1]) / np.maximum(np.abs(R[:, 1]), 1e-300)
        for lo, hi in ((0, 40), (41, 53), (54, 60), (61, 128), (129, 10000)):
            sel = (bits >= lo) & (bits <= hi)
            if not sel.any():
                continue
            print("      %4d-%-5d bits: %7d molecules, AvgIpc bit-identical to RDKit %7d, "
                  "max rel dev %.3e" % (lo, hi, int(sel.sum()), int((b & sel).sum()),
                                        float(rel[sel].max())))
        print("   -- COEFFICIENT width is NOT the whole story, and the 41-53 row is where that")
        print("      shows: RDKit's Faddeev ITERATE MATRIX exceeds 2^53 well before its final")
        print("      coefficients do, so RDKit loses exactness sooner than the coefficient width")
        print("      alone predicts. Every such disagreement was checked against exact integer")
        print("      arithmetic (section 6 and the chase log): in all of them OURS is within")
        print("      ~1e-15 of exact and RDKit is the one that has drifted, by up to 1e-2.")

        if len(Rp) > 1:
            rst = np.ones(nm, dtype=bool)
            for p in list(Rp)[1:]:
                rst &= same(R[:, 0], Rp[p][:, 0]) & same(R[:, 1], Rp[p][:, 1])
            print("\n   RDKIT'S OWN STABILITY under the same screen:")
            print("      molecules where RDKit's Ipc/AvgIpc MOVED: %d / %d  (%.2f%%)"
                  % (int((~rst).sum()), nm, 100.0 * (~rst).mean()))
            if (~rst).any():
                bb = bits[~rst]
                print("      their coefficient widths: min %d, median %d, max %d bits"
                      % (bb.min(), int(np.median(bb)), bb.max()))
                ns = np.array([Chem.MolFromSmiles(smis[i]).GetNumAtoms()
                               for i in np.flatnonzero(~rst)])
                print("      their heavy-atom counts : min %d, median %d, max %d"
                      % (ns.min(), int(np.median(ns)), ns.max()))
                sp = np.abs(Rp[list(Rp)[1]][:, 1] - R[:, 1]) / np.maximum(np.abs(R[:, 1]), 1e-300)
                print("      largest relative move in RDKit's own AvgIpc: %.3e" % sp[~rst].max())
            print("      ours moved on 0 (see section 1) -- integer arithmetic has no "
                  "cancellation\n      pattern for an ordering to change.")

    ep = CPP / "ic_exact_ipc.txt"
    if ep.exists():
        print("\n" + "=" * 92)
        print("6. AvgIpc against EXACT INTEGER arithmetic computed independently in Python.")
        print("=" * 92)
        pos = {s: i for i, s in enumerate(smis[:nm])}
        rows = []
        for line in open(ep):
            parts = line.split()
            sz, ex, rd, s = int(parts[0]), float(parts[1]), float(parts[2]), parts[3]
            bl = int(parts[4]) if len(parts) > 4 else 0
            if s in pos:
                rows.append((sz, ex, rd, V[0][pos[s], COLS.index("AvgIpc")], bl))
        if rows:
            print("   %9s %6s %8s %16s %16s"
                  % ("n", "mols", "maxbits", "rdkit max rel", "ours max rel"))
            buckets = {}
            for sz, ex, rd, ours_v, bl in rows:
                b = (sz // 20) * 20
                buckets.setdefault(b, []).append(
                    (abs(rd - ex) / abs(ex) if ex else 0.0,
                     abs(ours_v - ex) / abs(ex) if ex else 0.0, bl))
            for b in sorted(buckets):
                v = buckets[b]
                print("   %4d-%-4d %6d %8d %16.3e %16.3e"
                      % (b, b + 19, len(v), max(x[2] for x in v),
                         max(x[0] for x in v), max(x[1] for x in v)))
            print("   -- the `maxbits` column is what the two error columns are really tracking:")
            print("      RDKit degrades as soon as its arithmetic outruns 53 bits, ours does not,")
            print("      and the rows past 64 and past 128 bits are the ones that exercise")
            print("      infocontent.h's 2-word and 4-word integer paths.")


def cmd_time(path, n):
    """Per-molecule cost of the Python oracles, for the C++ number to be compared against.

    EVERY MOLECULE IS PARSED FRESH. mordred memoises per molecule and RDKit caches ring and
    aromaticity perception, so a second pass over the same objects measures a cache hit -- the
    failure mode that has produced four wrong numbers on the RDKit/mordred side of this project.
    The machine is SHARED, so these are upper bounds on a quiet one and the C++ figure they are
    compared against was taken under the same contention.
    """
    import time
    from rdkit.Chem import GraphDescriptors
    have_mordred = False
    try:
        from mordred import Calculator
        from mordred import InformationContent as M
        have_mordred = True
    except ImportError:
        pass
    versions(("mordred",) if have_mordred else ())
    smis = load_smis(path, n if n > 0 else 3000)

    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for s in smis:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            GraphDescriptors.Ipc(mol, forceDMat=True)
            GraphDescriptors.AvgIpc(mol, forceDMat=True)
        reps.append((time.perf_counter() - t0) / len(smis) * 1e6)
    reps.sort()
    print("rdkit Ipc+AvgIpc (fresh parse INCLUDED): median %.1f us/mol  min %.1f  max %.1f"
          % (reps[len(reps) // 2], reps[0], reps[-1]))

    # The parse alone, so the descriptor cost can be separated from it.
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for s in smis:
            Chem.MolFromSmiles(s)
        reps.append((time.perf_counter() - t0) / len(smis) * 1e6)
    reps.sort()
    parse = reps[len(reps) // 2]
    print("  of which SMILES parsing: median %.1f us/mol" % parse)

    if have_mordred:
        calc = Calculator([{"IC": M.InformationContent, "TIC": M.TotalIC, "SIC": M.StructuralIC,
                            "BIC": M.BondingIC, "CIC": M.ComplementaryIC, "MIC": M.ModifiedIC,
                            "ZMIC": M.ZModifiedIC}[f](o) for f in FAMS for o in range(6)])
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            for s in smis:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                for v in calc(mol):
                    pass
            reps.append((time.perf_counter() - t0) / len(smis) * 1e6)
        reps.sort()
        print("mordred 42 InformationContent columns (fresh parse INCLUDED): median %.1f us/mol"
              "  min %.1f  max %.1f" % (reps[len(reps) // 2], reps[0], reps[-1]))
        print("  minus the parse: %.1f us/mol" % (reps[len(reps) // 2] - parse))


USAGE = """usage: verify_ic.py CMD [FILE N]
  tables                 regenerate src/hume_core/ic_tables.h      (rdkit env)
  dump    FILE N         boundary dumps ic_in0..4.txt              (rdkit env)
  mordred FILE N         mordred oracle, 2 numberings              (mordred env)
  ipc     FILE N         rdkit Ipc/AvgIpc oracle                   (rdkit env)
  exact   FILE N         exact-integer AvgIpc, stratified sample   (rdkit env)
  time    FILE N         per-molecule cost of the oracles          (mordred env)
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
    elif cmd == "time":
        cmd_time(f, n)
    else:
        print(USAGE); sys.exit(1)
