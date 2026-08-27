"""Verify src/hume_core/constit.h against mordred 1.2.0 and rdkit 2025.09.2, per column, on
100,000 molecules of cpp/hard.smi.

    # 1. dump the boundary rows AND the oracle's answers, IN THE PINNED ENV
    uv run --isolated --no-project --python 3.11 --with "mordred==1.2.0" \
           --with "rdkit==2025.9.2" --with "numpy==1.26.4" --with "networkx" \
           python cpp/verify_constit.py --dump 100000
    # 2. run the C++
    c++ -O2 -std=c++17 -o cpp/constit cpp/constit.cpp && ./cpp/constit cpp/constit_in.txt cpp/constit_cpp.txt
    # 3. compare (needs nothing but python)
    python cpp/verify_constit.py --compare

WHY THE INPUTS COME FROM RDKIT AND NOT FROM THE OTHER HEADERS.  constit.h takes seven values it
does not compute -- MolLogP, MolMR, TPSA, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds, and the
aromatic/aliphatic ring counts -- from headers that are each separately verified BIT-EXACT against
rdkit on this same corpus (vsa_bins.h 66/66 columns, frag_matcher.h all-exact, ringcount.h 49/49).
Feeding rdkit's own answers here therefore feeds the identical bits and isolates the arithmetic
under test.  The ONE place that is not true is the ring counts, and this file handles it
explicitly rather than papering over it: see RINGS below.

RINGS, AND THE ONE PLACE THE SHIPPING ANSWER DIFFERS FROM MORDRED'S ON PURPOSE.  mordred's
`Rings()` is `Chem.GetSymmSSSR`; the boundary ships `src/hume/_rings.py:rings_for`, the REPAIRED
ring set (PORT_STATUS: 22 of 100,000 molecules move under atom+bond shuffling before the repair
and 0 after, and the repair changes rdkit's answer on 32, every one independently confirmed
unstable).  The two differ on roughly 1 in 3,000 molecules of cpp/hard.smi.  Two columns read
rings -- `fMF` directly and `Vabc` through naRing/nARing -- so this script dumps BOTH ring sets
and reports the disagreements SEPARATELY, split into "the C++ is wrong" and "the ring set is
deliberately different here".  Do not let the second population hide inside the first.

THE ORACLE IS PINNED AND ASKING WRONGLY FAILS SILENTLY.  `--with "mordred==1.2.0"` next to numpy 2
does NOT error: uv resolves mordred DOWN to 0.6.0, a different library with different columns.
`versions()` prints what actually loaded and checks a numeric canary, because a version banner is
not evidence -- a process can print 2025.09.2 and compute another version's numbers out of an
unlinked-but-still-mapped dylib.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

CPP = Path(__file__).resolve().parent
IN = CPP / "constit_in.txt"
REF = CPP / "constit_ref.txt"
GOT = CPP / "constit_cpp.txt"
META = CPP / "constit_meta.json"

# constit.h's col_name() order.  The C++ writes its own header line and --compare asserts the two
# agree, so this list cannot drift out of step silently.
COLS = [
    "C1SP1", "C2SP1", "C1SP2", "C2SP2", "C3SP2", "C1SP3", "C2SP3", "C3SP3", "C4SP3",
    "nH", "nB", "nC", "nN", "nO", "nS", "nCl", "nBr",
    "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM", "nBondsKD",
    "Kier1", "Kier2", "Kier3",
    "MDEC-22", "MDEC-23", "MDEC-33",
    "RNCG", "RPCG",
    "Lipinski", "GhoseFilter",
    "nAcid", "nBase",
    "Vabc", "RotRatio", "bpol", "FilterItLogS", "fMF", "fragCpx",
    "qed", "SPS",
]
MORDRED_COLS = [c for c in COLS if c not in ("qed", "SPS")]

# The six columns of this census block that constit.h does NOT implement because they are already
# computed and verified in src/hume_core/vsa_bins.h.  Listed so the report can say so out loud.
ELSEWHERE = {
    "TopoPSA": "vsa_bins.h C_TOPOPSA",
    "TPSA": "vsa_bins.h C_TPSA",
    "SLogP": "vsa_bins.h C_MOLLOGP (alias: mordred/SLogP.py is `Crippen.MolLogP(mol)`)",
    "PEOE_VSA11": "vsa_bins.h PEOE_VSA11 (MoeType resolves to the same function by getattr)",
    "SMR_VSA1": "vsa_bins.h SMR_VSA1",
    "EState_VSA1": "vsa_bins.h EState_VSA1",
}


# ---------------------------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------------------------
def versions():
    import numpy
    import rdkit
    import mordred
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    print("RESOLVED: mordred %s  rdkit %s  numpy %s  python %s"
          % (mordred.__version__, rdkit.__version__, numpy.__version__, sys.version.split()[0]))
    assert mordred.__version__ == "1.2.0", "mordred resolved to %s -- see the module docstring" % mordred.__version__
    v = rdMolDescriptors.CalcTPSA(Chem.MolFromSmiles("c1ccccc1O"))
    assert abs(v - 20.23) < 1e-12, "rdkit canary failed: %r" % v
    n = rdMolDescriptors.CalcNumHBD(Chem.MolFromSmiles("NC(=O)c1ccc(O)cc1"))
    assert n == 2, "rdkit HBD canary failed: %r" % n
    return {"mordred": mordred.__version__, "rdkit": rdkit.__version__,
            "numpy": numpy.__version__, "python": sys.version.split()[0]}


def bond_code(b):
    """src/hume/_extract.py's fifth bond_i column, reproduced field for field."""
    from rdkit import Chem
    bits = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 4}
    return bits.get(b.GetBondType(), 0) | (8 if b.GetIsAromatic() else 0)


def hyb_enum_check():
    """constit.h hardcodes rdkit's HybridizationType values.  Check them, do not trust them."""
    from rdkit import Chem
    H = Chem.HybridizationType
    want = {"UNSPECIFIED": 0, "S": 1, "SP": 2, "SP2": 3, "SP3": 4, "SP3D": 6, "SP3D2": 7}
    for name, v in want.items():
        got = int(getattr(H, name))
        assert got == v, "HybridizationType.%s is %d, constit.h assumes %d" % (name, got, v)


def qed_ads_check():
    """constit.h carries QED's ADS parameter rows and WEIGHT_MEAN as literals.  Read them back out
    of the running rdkit and refuse to continue if they moved."""
    from rdkit.Chem import QED
    rows = [QED.adsParameters[k] for k in
            ("MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS")]
    hard = [
        (2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561),
        (3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604),
        (2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046),
        (1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616),
        (1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167),
        (0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403),
        (3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610),
        (0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140),
    ]
    for r, h in zip(rows, hard):
        assert tuple(r) == h, "QED ADS parameters moved: %r vs %r" % (tuple(r), h)
    assert tuple(QED.WEIGHT_MEAN) == (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95)


def logs_predicate_check(mols):
    """Every one of FilterItLogS's sixteen SMARTS is implemented in constit.h as a predicate over
    the boundary columns.  Check each ONE PATTERN AT A TIME against rdkit's own matcher, atom by
    atom -- a per-column check would let two errors cancel."""
    from rdkit import Chem
    from mordred.LogS import _smarts_logs
    pats = [(s, Chem.MolFromSmarts(s)) for s in _smarts_logs]

    def pred(a, p):
        z, h, X, v = a.GetAtomicNum(), a.GetTotalNumHs(True), a.GetTotalDegree(), a.GetTotalValence()
        # SMARTS lowercase `h` (patterns 10 and 11) is a DIFFERENT primitive from uppercase `H`:
        # it is GetTotalNumHs(False), the boundary's `nH` column.  Reading both as the total count
        # is wrong on every aromatic carbon carrying a [2H].
        himp = a.GetTotalNumHs(False)
        ar, R = a.GetIsAromatic(), a.IsInRing()
        return [
            z == 7 and not ar and h == 0 and X == 3 and v == 3,
            z == 7 and not ar and h == 2 and X == 3 and v == 3,
            z == 7 and ar and h == 0 and X == 3,
            z == 8 and not ar and h == 0 and X == 2 and v == 2,
            z == 8 and not ar and h == 0 and X == 1 and v == 2,
            z == 8 and not ar and h == 1 and X == 2 and v == 2,
            z == 6 and not ar and h == 2 and not R,
            z == 6 and not ar and h == 3 and not R,
            z == 6 and not ar and h == 0 and R,
            z == 6 and not ar and h == 2 and R,
            z == 6 and ar and himp == 0,
            z == 6 and ar and himp == 1,
            z == 9 and not ar,
            z == 17 and not ar,
            z == 35 and not ar,
            z == 53 and not ar,
        ][p]

    bad = [0] * len(pats)
    for m in mols:
        for p, (s, q) in enumerate(pats):
            ref = {i for (i,) in m.GetSubstructMatches(q)}
            got = {a.GetIdx() for a in m.GetAtoms() if pred(a, p)}
            bad[p] += len(ref ^ got)
    for p, (s, _) in enumerate(pats):
        print("   LogS %-14s mismatching atoms: %d" % (s, bad[p]))
    return sum(bad)


def qed_hba_check(mols):
    """constit.h's `qedHBA` hand-implements QED's eleven acceptor SMARTS.  Check the SUM the way
    rdkit computes it -- `sum(len(matches) for pattern if HasSubstructMatch(pattern))` -- so an
    error in one pattern cannot be cancelled by an error in another only at the atom level."""
    from rdkit import Chem
    from rdkit.Chem import QED
    bad = 0
    for m in mols:
        ref = sum(len(m.GetSubstructMatches(p)) for p in QED.Acceptors if m.HasSubstructMatch(p))
        got = 0
        for a in m.GetAtoms():
            z, h, X = a.GetAtomicNum(), a.GetTotalNumHs(True), a.GetTotalDegree()
            v, c, ar = a.GetTotalValence(), a.GetFormalCharge(), a.GetIsAromatic()
            hit = ((z == 8 and ar and h == 0 and X == 2) or
                   (z == 8 and not ar and h == 1 and X == 2 and v == 2) or
                   (z == 8 and not ar and h == 0 and X == 2 and v == 2) or
                   (z == 8 and not ar and h == 0 and X == 1 and v == 2) or
                   (z == 8 and not ar and c == -1 and X == 1) or
                   (z == 16 and not ar and h == 0 and X == 2 and v == 2) or
                   (z == 16 and not ar and h == 0 and X == 1 and v == 2) or
                   (z == 16 and not ar and c == -1 and X == 1) or
                   (z == 7 and ar and h == 0 and X == 2) or
                   (z == 7 and not ar and h == 0 and X == 1 and v == 3))
            if not hit and z == 7 and not ar and c == 0 and X == 3 and v == 3:
                amide = False
                for b1 in a.GetBonds():
                    nb = b1.GetOtherAtom(a)
                    if nb.GetIsAromatic() or nb.GetAtomicNum() not in (6, 16):
                        continue
                    # `N[C,S]=O` -- the first bond is rdkit's DEFAULT SingleOrAromatic query.
                    if b1.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
                        continue
                    for b2 in nb.GetBonds():
                        o = b2.GetOtherAtom(nb)
                        if (b2.GetBondType() == Chem.BondType.DOUBLE and o.GetAtomicNum() == 8
                                and not o.GetIsAromatic()):
                            amide = True
                if not amide:
                    hit = True
            got += bool(hit)
        bad += ref != got
    return bad


def qed_arom_check(mols):
    """constit.h computes QED's AROM term as the CYCLOMATIC NUMBER of the subgraph left after
    deleting `[$([A;R][!a])]`, instead of editing a molecule and re-perceiving its SSSR."""
    from rdkit import Chem
    from rdkit.Chem import QED
    bad = 0
    for m in mols:
        ref = len(Chem.GetSSSR(Chem.DeleteSubstructs(Chem.Mol(m), QED.AliphaticRings)))
        # The bond in `[$([A;R][!a])]` is rdkit's DEFAULT SingleOrAromatic query, not `~`.
        # Reading it as "any bond" gives a different atom set on 62 of 3,000 molecules.
        keep = []
        for a in m.GetAtoms():
            k = True
            if not a.GetIsAromatic() and a.IsInRing():
                for b in a.GetBonds():
                    t = b.GetBondType()
                    if t not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
                        continue
                    if not b.GetOtherAtom(a).GetIsAromatic():
                        k = False
                        break
            keep.append(k)
        idx = {}
        for a in m.GetAtoms():
            if keep[a.GetIdx()]:
                idx[a.GetIdx()] = len(idx)
        V = len(idx)
        uf = list(range(V))

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x
        E = 0
        for b in m.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u not in idx or v not in idx:
                continue
            E += 1
            a2, b2 = find(idx[u]), find(idx[v])
            if a2 != b2:
                uf[a2] = b2
        C = sum(1 for i in range(V) if find(i) == i)
        bad += (E - V + C) != ref
    return bad


def acidbase_hgraph_check(mols):
    """constit.h matches nAcid/nBase on the HEAVY molecule; mordred matches on the H-ADDED one.
    Measure the equivalence rather than assume it."""
    from rdkit import Chem
    acid = ["[O;H1]-[C,S,P]=O", "[*;-;!$(*~[*;+])]", "[NH](S(=O)=O)C(F)(F)F", "n1nnnc1"]
    base = ["[NH2]-[CX4]", "[NH](-[CX4])-[CX4]", "N(-[CX4])(-[CX4])-[CX4]",
            "[*;+;!$(*~[*;-])]", "N=C-N", "N-C=N"]
    pa = Chem.MolFromSmarts("[" + ",".join("$(" + s + ")" for s in acid) + "]")
    pb = Chem.MolFromSmarts("[" + ",".join("$(" + s + ")" for s in base) + "]")
    da = db = 0
    for m in mols:
        h = Chem.AddHs(m)
        da += len(h.GetSubstructMatches(pa)) != len(m.GetSubstructMatches(pa))
        db += len(h.GetSubstructMatches(pb)) != len(m.GetSubstructMatches(pb))
    return da, db


def bond_arom_check(mols):
    """nBondsA is `GetIsAromatic() or bondType == AROMATIC` and the boundary carries only the
    flag.  Count the bonds on which the two disagree, in both directions."""
    from rdkit import Chem
    a = b = 0
    for m in mols:
        for bd in m.GetBonds():
            t = bd.GetBondType() == Chem.BondType.AROMATIC
            f = bd.GetIsAromatic()
            a += t and not f
            b += f and not t
    return a, b


def ring_counts(rings, mol):
    """mordred RingCount(None,False,False,arom,None) over a given ring set: an AROMATIC ring is
    one all of whose atoms are aromatic.  This is ringcount.h's own predicate, applied to whichever
    ring set is handed in, so naRing/nARing and fMF read ONE ring perception."""
    na = nA = 0
    for r in rings:
        if all(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in r):
            na += 1
        else:
            nA += 1
    return na, nA


def cmd_dump(n):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors as rdmd, Crippen, MolSurf, QED, rdPartialCharges
    from rdkit.Chem import rdmolops
    from rdkit.Chem.SpacialScore import SPS
    from mordred import Calculator, descriptors as mdesc
    RDLogger.DisableLog("rdApp.*")

    meta = versions()
    hyb_enum_check()
    qed_ads_check()

    sys.path.insert(0, str(CPP.parent / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rings", CPP.parent / "src/hume/_rings.py")
    ringmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ringmod)

    smis = [l.split()[0] for l in open(CPP / "hard.smi") if l.strip()][:n]
    mols = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    print("parsed %d / %d" % (len(mols), len(smis)))

    # --- the pattern-level checks, on a sample big enough to be meaningful and small enough to run
    NCHK = min(len(mols), 20000)
    print("\nLogS predicate check (%d molecules, pattern by pattern, atom by atom):" % NCHK)
    nbad = logs_predicate_check(mols[:NCHK])
    print("   TOTAL mismatching atoms over all 16 patterns: %d" % nbad)
    qh = qed_hba_check(mols[:NCHK])
    print("QED HBA (11 acceptor SMARTS) molecules differing from rdkit's matcher: %d / %d" % (qh, NCHK))
    qa = qed_arom_check(mols[:NCHK])
    print("QED AROM as a cyclomatic number vs DeleteSubstructs+GetSSSR: %d / %d differ" % (qa, NCHK))
    da, db = acidbase_hgraph_check(mols[:NCHK])
    print("AcidBase H-added vs heavy (%d molecules): nAcid differs on %d, nBase on %d"
          % (NCHK, da, db))
    ba, bb = bond_arom_check(mols[:NCHK])
    print("bond aromatic FLAG vs TYPE (%d molecules): type&!flag %d, flag&!type %d" % (NCHK, ba, bb))
    meta.update(n_checked=NCHK, logs_pattern_mismatches=nbad, qed_hba_mismatches=qh,
                qed_arom_mismatches=qa, acidbase_hgraph_diff=[da, db],
                bond_arom_flag_vs_type=[ba, bb])

    # --- mordred, restricted to the columns under test
    full = Calculator(mdesc, ignore_3D=True)
    want = set(MORDRED_COLS)
    keep = [d for d in full.descriptors if str(d) in want]
    got = {str(d) for d in keep}
    assert not (want - got), "not in the mordred preset: %s" % sorted(want - got)
    calc = Calculator(keep, ignore_3D=True)
    order = [str(d) for d in calc.descriptors]

    fin = open(IN, "w")
    fref = open(REF, "w")
    fin.write("%d\n" % len(mols))
    fref.write(" ".join(COLS) + "\n")

    ring_differs = []
    for k, m in enumerate(mols):
        n_at, n_b = m.GetNumAtoms(), m.GetNumBonds()
        rings = [list(r) for r in ringmod.rings_for(m)]
        raw = [sorted(list(r)) for r in Chem.GetSymmSSSR(m)]
        if sorted(sorted(r) for r in rings) != sorted(raw):
            ring_differs.append(k)
        nhadd = sum(a.GetTotalNumHs(False) for a in m.GetAtoms())

        h = Chem.AddHs(m)
        try:
            rdPartialCharges.ComputeGasteigerCharges(h)
            hchg = [a.GetDoubleProp("_GasteigerCharge") + a.GetDoubleProp("_GasteigerHCharge")
                    for a in h.GetAtoms()]
            if not all(math.isfinite(c) for c in hchg):
                hchg = None
        except Exception:
            hchg = None

        st = {i for i, _ in Chem.FindMolChiralCenters(
            m, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)}
        cp = Chem.Mol(m)
        rdmolops.FindPotentialStereoBonds(cp)
        sb = [1 if (b.GetBondType() == Chem.BondType.DOUBLE and
                    b.GetStereo() != Chem.BondStereo.STEREONONE) else 0 for b in cp.GetBonds()]

        na, nA = ring_counts(rings, m)
        alerts = sum(1 for a in QED.StructuralAlerts if m.HasSubstructMatch(a))

        fin.write("%d %d %d %d %d\n" % (n_at, n_b, len(rings), nhadd, 1 if hchg else 0))
        for a in m.GetAtoms():
            cip = 0
            if a.HasProp("_CIPCode"):
                cip = 1 if a.GetProp("_CIPCode") == "R" else -1
            fin.write("%d %d %d %d %d %d %d %d %d %d %.17g\n" % (
                a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs(False), a.GetFormalCharge(),
                int(a.GetHybridization()), int(a.GetIsAromatic()), int(a.IsInRing()), cip,
                m.GetRingInfo().NumAtomRings(a.GetIdx()), a.GetTotalValence(), a.GetMass()))
        for b in m.GetBonds():
            fin.write("%d %d %d %d %d %.17g\n" % (
                b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetIsConjugated()),
                int(b.IsInRing()), bond_code(b), b.GetBondTypeAsDouble()))
        for r in rings:
            fin.write("%d %s\n" % (len(r), " ".join(str(int(i)) for i in r)))
        if hchg:
            fin.write(" ".join("%.17g" % c for c in hchg) + "\n")
        fin.write(" ".join("1" if a.GetIdx() in st else "0" for a in m.GetAtoms()) + "\n")
        fin.write(" ".join(str(x) for x in sb) + "\n")
        fin.write("%.17g %.17g %.17g %.17g %.17g %d %d %d %d\n" % (
            Crippen.MolLogP(m), Crippen.MolMR(m), MolSurf.TPSA(m), float(na), float(nA),
            rdmd.CalcNumHBD(m), rdmd.CalcNumHBA(m), rdmd.CalcNumRotatableBonds(m), alerts))

        vals = {}
        for d, v in zip(calc.descriptors, calc(m)):
            try:
                vals[str(d)] = float(v)
            except Exception:
                vals[str(d)] = float("nan")
        try:
            vals["qed"] = float(QED.qed(m))
        except Exception:
            vals["qed"] = float("nan")
        try:
            vals["SPS"] = float(SPS(m))
        except Exception:
            vals["SPS"] = float("nan")
        fref.write(" ".join("%.17g" % vals[c] for c in COLS) + "\n")
        if (k + 1) % 5000 == 0:
            print("  %d / %d" % (k + 1, len(mols)))
    fin.close()
    fref.close()
    meta["n_molecules"] = len(mols)
    meta["ring_set_differs"] = ring_differs
    META.write_text(json.dumps(meta, indent=1))
    print("\nwrote %s, %s, %s" % (IN, REF, META))
    print("ring sets differ from Chem.GetSymmSSSR on %d of %d molecules"
          % (len(ring_differs), len(mols)))


# ---------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------
def cmd_compare():
    meta = json.loads(META.read_text())
    ringdiff = set(meta.get("ring_set_differs", []))
    fr, fg = open(REF), open(GOT)
    hdr_ref = fr.readline().split()
    hdr_got = fg.readline().split()
    assert hdr_ref == COLS, "reference header drifted"
    assert hdr_got == COLS, "the C++ and this script disagree about the column order:\n  %s\n  %s" % (
        hdr_got, COLS)

    nbad = [0] * len(COLS)
    nbad_ring = [0] * len(COLS)
    worst = [0.0] * len(COLS)
    worst_row = [-1] * len(COLS)
    exact = [0] * len(COLS)
    nrow = 0
    for row, (lr, lg) in enumerate(zip(fr, fg)):
        a = [float(x) for x in lr.split()]
        b = [float(x) for x in lg.split()]
        nrow += 1
        for c in range(len(COLS)):
            x, y = a[c], b[c]
            if math.isnan(x) and math.isnan(y):
                exact[c] += 1
                continue
            if x == y:
                exact[c] += 1
                continue
            if math.isnan(x) or math.isnan(y):
                d = float("inf")
            else:
                s = max(abs(x), abs(y))
                d = abs(x - y) / s if s else abs(x - y)
            nbad[c] += 1
            if row in ringdiff:
                nbad_ring[c] += 1
            if d > worst[c]:
                worst[c] = d
                worst_row[c] = row
    print("RESOLVED (from the process that produced the reference): "
          "mordred %s  rdkit %s  numpy %s  python %s"
          % (meta["mordred"], meta["rdkit"], meta["numpy"], meta["python"]))
    print("molecules: %d\n" % nrow)
    print("%-14s %9s %9s %12s %s" % ("column", "exact", "differ", "max rel dev", "note"))
    tot_bad = 0
    for c, name in enumerate(COLS):
        note = ""
        if nbad[c]:
            tot_bad += 1
            if nbad_ring[c] == nbad[c]:
                note = "ALL on molecules where the SHIPPING ring set differs from GetSymmSSSR"
            elif nbad_ring[c]:
                note = "%d of them on ring-set-differs molecules" % nbad_ring[c]
            note += "  (worst at row %d)" % worst_row[c]
        print("%-14s %9d %9d %12.4g  %s" % (name, exact[c], nbad[c], worst[c], note))
    print("\ncolumns not bit-exact: %d of %d" % (tot_bad, len(COLS)))
    print("\nALSO IN THIS CENSUS BLOCK, ALREADY IMPLEMENTED AND VERIFIED ELSEWHERE "
          "(not recomputed by constit.h):")
    for k, v in ELSEWHERE.items():
        print("   %-12s -> %s" % (k, v))
    print("\npattern-level checks from the dump run (%s molecules):" % meta.get("n_checked"))
    print("   LogS 16 patterns, mismatching atoms vs rdkit's matcher: %s"
          % meta.get("logs_pattern_mismatches"))
    print("   QED HBA, molecules differing from rdkit's matcher:      %s"
          % meta.get("qed_hba_mismatches"))
    print("   QED AROM cyclomatic vs DeleteSubstructs+GetSSSR:        %s"
          % meta.get("qed_arom_mismatches"))
    print("   AcidBase H-added vs heavy graph, molecules differing:   %s"
          % meta.get("acidbase_hgraph_diff"))
    print("   bond aromatic TYPE vs FLAG, disagreeing bonds:          %s"
          % meta.get("bond_arom_flag_vs_type"))
    print("   shipping ring set != Chem.GetSymmSSSR on %d molecules" % len(ringdiff))


def cmd_bench(n, reps):
    """Time the ORACLE arm on the same molecules, so the C++ number has something to be against.

    WHAT IS AND IS NOT IN THE CLOCK.  The molecules are parsed once, before timing.  The mordred
    arm times `calc(m)` for the 41 mordred columns plus `QED.qed(m)` and `SPS(m)` -- i.e. the same
    43 columns constit.h emits, and nothing else.  It does NOT include the seven values constit.h
    takes as inputs (MolLogP, MolMR, TPSA, HBD, HBA, rotatable bonds, ring counts), because the
    C++ arm does not compute those either; they are on some other header's bill on both sides.
    Reporting a ratio that quietly charges mordred for work the C++ arm never does is the mistake
    bench_e2e.py's `report` exists to refuse.

    THE MACHINE.  The spread over reps is reported, not just the mean.  A contended box shows up
    as a large SD and the number should be read as an ordering, not as a measurement.
    """
    import time
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED
    from rdkit.Chem.SpacialScore import SPS
    from mordred import Calculator, descriptors as mdesc
    RDLogger.DisableLog("rdApp.*")
    versions()
    print("load average: %s" % (os.getloadavg(),))

    smis = [l.split()[0] for l in open(CPP / "hard.smi") if l.strip()][:n]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    full = Calculator(mdesc, ignore_3D=True)
    calc = Calculator([d for d in full.descriptors if str(d) in set(MORDRED_COLS)], ignore_3D=True)

    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for m in mols:
            calc(m)
            QED.qed(m)
            SPS(m)
        per.append((time.perf_counter() - t0) * 1e6 / len(mols))
    mean = sum(per) / len(per)
    sd = (sum((x - mean) ** 2 for x in per) / (len(per) - 1)) ** 0.5 if len(per) > 1 else 0.0
    print("mordred+rdkit  %d molecules x %d reps: %.1f +/- %.1f us/mol  (the same 43 columns)"
          % (len(mols), reps, mean, sd))
    print("load average after: %s" % (os.getloadavg(),))


def cmd_vabc(n=0):
    """THE DECISIVE TEST FOR THE ONE COLUMN THAT IS NOT BIT-EXACT.

    `Vabc` disagrees with mordred on exactly the molecules where the shipping REPAIRED ring set
    differs from `Chem.GetSymmSSSR`.  That is only a defensible divergence if mordred's own answer
    on those molecules is not a function of the molecule -- so this re-runs mordred's Vabc on each
    of them under atom renumbering AND bond-list shuffling and reports how many distinct answers
    it gives.  A molecule that yields two or more answers has no value to be exact against; one
    that yields exactly one is a real disagreement and must be reported as such.
    """
    import json
    from rdkit import Chem, RDLogger
    from mordred import Calculator, descriptors as mdesc
    sys.path.insert(0, str(CPP))
    from screen_constit import rebuilt_same_molecule, renumbered
    import random
    RDLogger.DisableLog("rdApp.*")
    versions()
    meta = json.loads(META.read_text())
    rows = set(meta["ring_set_differs"])
    smis = [l.split()[0] for l in open(CPP / "hard.smi") if l.strip()][:meta["n_molecules"]]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    full = Calculator(mdesc, ignore_3D=True)
    calc = Calculator([d for d in full.descriptors if str(d) == "Vabc"], ignore_3D=True)

    def vabc(m):
        try:
            return float(list(calc(m))[0])
        except Exception:
            return float("nan")

    rng = random.Random(20260827)
    stable, unstable = [], []
    for k in sorted(rows):
        m = mols[k]
        seen = {round(vabc(m), 9)}
        for _ in range(24):
            idx = list(range(m.GetNumAtoms()))
            rng.shuffle(idx)
            bo = list(range(m.GetNumBonds()))
            rng.shuffle(bo)
            r = rebuilt_same_molecule(m, idx, bo) or renumbered(m, idx)
            seen.add(round(vabc(r), 9))
        (unstable if len(seen) > 1 else stable).append((k, sorted(seen)))
    print("\nmordred Vabc under 24 atom+bond perturbations, on the %d molecules where the "
          "shipping\nring set differs from Chem.GetSymmSSSR:" % len(rows))
    print("   molecules giving MORE THAN ONE answer (ill-posed, nothing to be exact against): %d"
          % len(unstable))
    print("   molecules giving exactly one answer (a real disagreement):                      %d"
          % len(stable))
    for k, v in unstable[:6]:
        print("      row %-7d mordred answers: %s" % (k, v))
    for k, v in stable[:6]:
        print("      STABLE row %-7d mordred answer: %s   %s" % (k, v, smis[k][:70]))


if __name__ == "__main__":
    if "--vabc" in sys.argv:
        cmd_vabc()
    elif "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        cmd_bench(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 2000,
                  int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else 5)
    elif "--dump" in sys.argv:
        i = sys.argv.index("--dump")
        cmd_dump(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 100000)
    elif "--compare" in sys.argv:
        cmd_compare()
    else:
        print(__doc__)
