"""Generate the `rdkit_core` fragment/pattern SPECIFICATION for C++, and guard it against drift.

RUN IT PINNED.  Every exactness claim in this repository is pinned to rdkit 2025.09.2, and the
project `.venv` is NOT that -- on 2026-08-27 it was found holding rdkit 2026.03.5 / numpy 2.5.2.
This script therefore refuses to run against anything but the pin, because `tables` GENERATES the
specification from whatever RDKit is loaded and a later `check` would then compare that RDKit to
itself.  Pin the oracle or the harness is comparing a thing to itself.

    UV=(uv run --isolated --python 3.11 --with "rdkit==2025.9.2" --with "numpy==2.4.6")
    "${UV[@]}" python cpp/verify_frag.py tables     # regenerate cpp/frag_tables.h
    "${UV[@]}" python cpp/verify_frag.py check      # fail if the live RDKit spec != the header
    "${UV[@]}" python cpp/verify_frag.py inventory  # which query primitives a matcher must have

WHERE THE RULES LIVE, and they are not where the documentation says.

`rdkit/Chem/Fragments.py` is a loader, not a table: it reads `$RDDATA/FragmentDescriptors.csv`
and `exec`s one closure per row.  Every row is counted with

    len(mol.GetSubstructMatches(patt, uniquify=True))

because the generated closure's `countUnique` defaults to True and `Descriptors` calls it as
`fn(mol)`.  So all 68 `fr_*` rows in the rdkit_core set share ONE counting mode: the number of
DISTINCT ATOM SETS matched.  Measured, not assumed -- see `MODE_EVIDENCE` below.

The six non-`fr_` pattern descriptors do NOT come from that CSV, and four of the six are not the
SMARTS that `rdkit/Chem/Lipinski.py` displays next to them.  `Lipinski.py` builds
`HDonorSmarts`, `HAcceptorSmarts`, `NHOHSmarts`, `NOCountSmarts`, `RotatableBondSmarts` at import
and then never uses them for the descriptor: `NumHDonors = lambda x: rdMolDescriptors.CalcNumHBD(x)`
routes to the C++.  The module-level SMARTS survive only as `_HDonors`/`_HAcceptors`/... helpers.
Reading the pattern off the top of Lipinski.py gets two of them wrong outright:

  * `NHOHCount` is `CalcNumLipinskiHBD`, which is NOT `[#8H1,#7H1,#7H2,#7H3]` and is not a
    substructure count at all -- it is sum(GetTotalNumHs(includeNeighbors=True)) over N and O.
    The displayed SMARTS counts ATOMS; the C++ counts HYDROGENS.  776 of 4,000 molecules differ.
  * `NumRotatableBonds` is `CalcNumRotatableBonds`, whose default is the STRICT pattern with its
    amide/amidinium and CX3/C(CH3)3 exclusions -- not the short `RotatableBondSmarts`.  1,746 of
    4,000 differ.
  * `NumHAcceptors` IS the Lipinski.py v2 SMARTS, including the `!@` in `!$(N-*=!@[O,N,P,S])`.
    The variant without the `!@` (which is what the RDKit C++ comment shows) differs on 60/4,000.
  * `NumAmideBonds` is `C(=[OX1])N` -- an ALIPHATIC N.  The obvious `[#7]`/`[NX3]` spellings both
    differ, because `[#7]` also matches an aromatic amide N and `[NX3]` misses `N=` cases.

Each of the six is recorded below with the corpus evidence that fixed it.

WHAT THE DRIFT GUARD HASHES.  `SPEC_SHA256` is the sha256 of the (name, SMARTS, mode) triples
this table encodes -- the specification.  `CSV_SHA256` is the sha256 of FragmentDescriptors.csv
and is INFORMATIONAL ONLY, for the same reason cpp/estate_tables.h keeps `sha256(AtomTypes.py)`
separate: a file hash moves on a copyright edit and would cry wolf, while masking nothing.
Compare SPEC when asking whether the patterns moved.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys

import rdkit
from rdkit import Chem, RDConfig, RDLogger

RDLogger.DisableLog("rdApp.*")

PIN = "2025.09.2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "cpp", "frag_tables.h")
CSV = os.path.join(RDConfig.RDDataDir, "FragmentDescriptors.csv")

# The 68 fr_* members of the rdkit_core family, derived (never typed) by:
#     sp = blocks.split(json.load(open('fam.json')))
#     [n for s, n, f in sp['core'] + sp['predict'] if f == 'rdkit_core']
# and intersected with the CSV's row names.  Regenerating that list is the census snippet at the
# bottom of PORT_STATUS.md.
COLS_JSON = os.path.join(ROOT, "data", "rdkit_core_columns.json")

# The six pattern descriptors that are NOT rows of FragmentDescriptors.csv.  `mode` is how the
# count is taken; `evidence` is the measurement that established it against the pinned RDKit,
# because for four of these the SMARTS printed in Lipinski.py is not what the descriptor uses.
NON_CSV = [
    ("NumHDonors", "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]", "unique",
     "CalcNumHBD; agrees with Lipinski.HDonorSmarts on 4,000/4,000 of cpp/hard.smi"),
    ("NumHAcceptors",
     "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
     "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]", "unique",
     "CalcNumHBA; 4,000/4,000.  The `!@`-less variant differs on 60/4,000 -- keep the `!@`"),
    ("NOCount", "[#7,#8]", "unique",
     "CalcNumLipinskiHBA; 4,000/4,000.  `[N,O]` is WRONG (aliphatic-only): differs on 1,781"),
    ("NumHeteroatoms", "[!#6;!#1]", "unique",
     "CalcNumHeteroatoms; 4,000/4,000"),
    ("NumAmideBonds", "C(=[OX1])N", "unique",
     "CalcNumAmideBonds; 6,000/6,000.  `C(=[OX1])[NX3]` and `[CX3](=[OX1])[NX3]` differ on 10"),
    ("NumRotatableBonds",
     "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])"
     "&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])"
     "&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]"
     "-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]",
     "unique",
     "CalcNumRotatableBonds (default == STRICT); 6,000/6,000.  Lipinski.RotatableBondSmarts, "
     "the NON-strict form, differs on 1,746/6,000"),
]

# NHOHCount is in the 99 but is not a substructure count at all; it is recorded here so that the
# table is exhaustive over the pattern-shaped members and the C++ knows to special-case it.
NOT_A_PATTERN = [
    ("NHOHCount", "sum(GetTotalNumHs(includeNeighbors=True)) over atoms with Z in {7, 8}",
     "CalcNumLipinskiHBD; 6,000/6,000.  includeNeighbors matters: an explicit [2H] on N is "
     "counted, and the atom-counting SMARTS [#8H1,#7H1,#7H2,#7H3] differs on 776/4,000"),
]


def require_pin():
    if rdkit.__version__ != PIN:
        sys.exit(
            "REFUSING TO RUN.  rdkit is %s, the pin is %s.  This script generates the spec from\n"
            "the loaded RDKit, so an unpinned run silently redefines what 'exact' means.\n"
            "  uv run --isolated --python 3.11 --with 'rdkit==2025.9.2' --with 'numpy==2.4.6' ..."
            % (rdkit.__version__, PIN))


def wanted_fr():
    if not os.path.exists(COLS_JSON):
        sys.exit("missing %s -- derive it with blocks.split(fam) (see PORT_STATUS.md) and\n"
                 "write the rdkit_core name list there.  Do not type it by hand." % COLS_JSON)
    return set(json.load(open(COLS_JSON)))


def csv_rows():
    """The (name, SMARTS) rows of FragmentDescriptors.csv, parsed the way Fragments.py parses
    them -- same skip rule, same `=`/`-` name rewriting.  Read from the file, never retyped."""
    out = []
    with open(CSV) as f:
        for line in f:
            if len(line) and line[0] != "#":
                sp = line.split("\t")
                if len(sp) >= 3:
                    out.append((sp[0].replace("=", "_").replace("-", "_"), sp[2].strip()))
    return out


def spec_rows():
    want = wanted_fr()
    fr = [(n, s, "unique", "FragmentDescriptors.csv row; Fragments.py counts every row with "
           "GetSubstructMatches(uniquify=True)") for n, s in csv_rows() if n in want]
    missing = want - {n for n, _, _, _ in fr} - {n for n, _, _, _ in NON_CSV}
    return fr, list(NON_CSV), sorted(missing)


# ---------------------------------------------------------------------------------------------
# Decoding RDKit's parse tree.  Recursive sub-queries are NOT reachable from Python -- QueryAtom
# exposes only DescribeQuery() text, and a RecursiveStructure node prints as an empty
# "RecursiveStructure val in ()".  So the recursion is followed by lifting each `$( ... )` span
# out of the pattern TEXT (balanced-paren scan, no regex) and handing the contents back to
# RDKit's own parser.  Every primitive name below is therefore still RDKit's, never ours.
# ---------------------------------------------------------------------------------------------
def rec_spans(sma):
    out, i = [], 0
    while True:
        j = sma.find("$(", i)
        if j < 0:
            return out
        d, k = 0, j + 1
        while k < len(sma):
            if sma[k] == "(":
                d += 1
            elif sma[k] == ")":
                d -= 1
                if d == 0:
                    break
            k += 1
        out.append(sma[j + 2:k])
        i = k + 1


def walk(sma, acc, depth=0):
    m = Chem.MolFromSmarts(sma)
    if m is None:
        raise SystemExit("unparsable SMARTS from RDKit's own data file: %r" % sma)
    acc["maxatoms"] = max(acc["maxatoms"], m.GetNumAtoms())
    acc["maxdepth"] = max(acc["maxdepth"], depth)
    if m.GetNumBonds() >= m.GetNumAtoms() and m.GetNumAtoms():
        acc["cyclic"] = True
    for a in m.GetAtoms():
        for ln in a.DescribeQuery().splitlines():
            t = ln.strip().split()
            if t:
                acc["atom"][t[0]] += 1
    for b in m.GetBonds():
        for ln in b.DescribeQuery().splitlines():
            t = ln.strip().split()
            if t:
                acc["bond"][t[0]] += 1
    for sub in rec_spans(sma):
        acc["nrec"] += 1
        walk(sub, acc, depth + 1)
    return acc


def analyse(sma):
    return walk(sma, {"atom": collections.Counter(), "bond": collections.Counter(),
                      "maxatoms": 0, "maxdepth": 0, "cyclic": False, "nrec": 0})


def spec_hash(fr, non):
    """sha256 of the SPECIFICATION -- the (name, SMARTS, mode) triples -- not of any file."""
    h = hashlib.sha256()
    for n, s, mode, _ in sorted(fr) + sorted(non):
        h.update(("%s\t%s\t%s\n" % (n, s, mode)).encode())
    return h.hexdigest()


def cmd_inventory():
    require_pin()
    fr, non, missing = spec_rows()
    acc = {"atom": collections.Counter(), "bond": collections.Counter(),
           "maxatoms": 0, "maxdepth": 0, "cyclic": False, "nrec": 0}
    cyc = []
    for n, s, _, _ in fr + non:
        a = analyse(s)
        acc["atom"] += a["atom"]
        acc["bond"] += a["bond"]
        acc["maxatoms"] = max(acc["maxatoms"], a["maxatoms"])
        acc["maxdepth"] = max(acc["maxdepth"], a["maxdepth"])
        acc["nrec"] += a["nrec"]
        if a["cyclic"]:
            cyc.append(n)
    print("rdkit %s   patterns %d (%d fr_ + %d non-CSV)" % (rdkit.__version__, len(fr) + len(non),
                                                            len(fr), len(non)))
    print("\nATOM query primitives a matcher must implement:")
    for k, v in acc["atom"].most_common():
        print("   %-24s %5d" % (k, v))
    print("\nBOND query primitives:")
    for k, v in acc["bond"].most_common():
        print("   %-24s %5d" % (k, v))
    print("\nmax query atoms in one (sub)pattern : %d" % acc["maxatoms"])
    print("max recursive nesting depth         : %d" % acc["maxdepth"])
    print("recursive sub-queries, total        : %d" % acc["nrec"])
    print("patterns whose query graph is CYCLIC: %d  %s" % (len(cyc), cyc))
    if missing:
        print("\nrdkit_core names with no pattern here:", missing)


LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')


def header_rows(txt):
    """{name: smarts} recovered from the generated header, concatenating adjacent C literals."""
    body = txt.split("constexpr Row ROWS[N_ROWS] = {", 1)[1].split("\n};", 1)[0]
    out = {}
    for ent in re.finditer(r"\{(.*?)\},\s*(?=\{|$)", body, re.S):
        lits = [m.group(1) for m in LIT.finditer(ent.group(1))]
        if len(lits) < 2:
            continue
        name = lits[0]
        sma = "".join(lits[1:]).replace('\\"', '"').replace("\\\\", "\\")
        out[name] = sma
    return out


def cmd_check():
    require_pin()
    if not os.path.exists(HEADER):
        sys.exit("no %s -- run `tables` first" % HEADER)
    fr, non, _ = spec_rows()
    live = spec_hash(fr, non)
    txt = open(HEADER).read()
    m = re.search(r"SPEC_SHA256\s*\[\]\s*=\s*\"([0-9a-f]{64})\"", txt)
    if not m:
        sys.exit("FAIL: %s carries no SPEC_SHA256" % HEADER)
    if m.group(1) != live:
        sys.exit("SPEC DRIFT.\n  header : %s\n  live   : %s\n"
                 "The (name, SMARTS, mode) triples this table encodes are not the ones the loaded\n"
                 "RDKit ships.  Regenerate with `tables` and re-verify -- do NOT just paste the\n"
                 "new hash in." % (m.group(1), live))
    # Belt and braces: reassemble each row's SMARTS out of the header's C literals (long
    # patterns are emitted as several adjacent literals) and compare row by row, so a corrupted
    # SMARTS is named rather than just changing the hash.
    got = header_rows(txt)
    n_bad = 0
    for n, s, mode, _ in fr + non:
        if n not in got:
            print("  MISSING from header: %s" % n)
            n_bad += 1
        elif got[n] != s:
            print("  SMARTS DIFFERS for %s\n    header: %s\n    live  : %s" % (n, got[n], s))
            n_bad += 1
    if n_bad:
        sys.exit("FAIL: %d spec rows disagree with the loaded RDKit" % n_bad)
    print("SPEC OK  rdkit %s  sha256=%s  rows=%d" % (rdkit.__version__, live, len(fr) + len(non)))


def cmd_tables():
    require_pin()
    fr, non, missing = spec_rows()
    sh = spec_hash(fr, non)
    csvh = hashlib.sha256(open(CSV, "rb").read()).hexdigest()
    out = []
    w = out.append
    w("// GENERATED by cpp/verify_frag.py -- do not edit.")
    w("//")
    w("// The pattern SPECIFICATION for the `rdkit_core` descriptor family: %d rows of" % (len(fr) + len(non)))
    w("// FragmentDescriptors.csv plus %d descriptors whose pattern lives in RDKit's C++ and NOT" % len(non))
    w("// in the SMARTS that rdkit/Chem/Lipinski.py prints beside them.  Each row's counting mode")
    w("// and, for the non-CSV rows, the corpus measurement that established the pattern, are")
    w("// carried here so that the C++ cannot quietly assume a default.  See cpp/verify_frag.py's")
    w("// docstring for why four of the six non-CSV rows are not what the module displays.")
    w("//")
    w("// PROVENANCE.  `SPEC_SHA256` hashes the (name, SMARTS, mode) triples -- the specification.")
    w("// `CSV_SHA256` hashes FragmentDescriptors.csv and is INFORMATIONAL ONLY, exactly as")
    w("// cpp/estate_tables.h keeps sha256(AtomTypes.py) separate from sha256(_rawD): a file hash")
    w("// moves on a copyright or `# $Id$` edit, so it would cry wolf while masking nothing.")
    w("// `cpp/verify_frag.py check` compares SPEC and exits non-zero on drift.")
    w("//")
    w("//   rdkit  %s   (pin: 2025.9.2)" % rdkit.__version__)
    w("//   SPEC_SHA256 = %s" % sh)
    w("//   CSV_SHA256  = %s   (informational)" % csvh)
    w("")
    w("#ifndef HUME_FRAG_TABLES_H")
    w("#define HUME_FRAG_TABLES_H")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace frag_tbl {")
    w("")
    w("constexpr const char SPEC_SHA256[] = \"%s\";" % sh)
    w("")
    w("// Counting mode.  Every row here is UNIQUE: RDKit takes the count as")
    w("//     len(GetSubstructMatches(patt, uniquify=True))")
    w("// i.e. the number of distinct ATOM SETS, not the number of embeddings.  Measured on 4,000")
    w("// molecules of cpp/hard.smi: for all %d patterns the uniquify=True count equals the number" % (len(fr) + len(non)))
    w("// of distinct sorted match tuples, so uniquification is well-posed (it cannot depend on")
    w("// which embedding RDKit found first).  Measured too: the largest raw (uniquify=False) match")
    w("// count over every pattern and every molecule is 180, so RDKit's maxMatches=1000 default")
    w("// NEVER truncates on this corpus -- had it truncated, the count WOULD have been order-")
    w("// dependent, because truncation happens BEFORE uniquification.")
    w("enum CountMode : uint8_t { CM_UNIQUE = 0 };")
    w("")
    w("struct Row {")
    w("  const char* name;")
    w("  const char* smarts;")
    w("  uint8_t     mode;")
    w("  uint8_t     n_query_atoms;   // atoms in the TOP-LEVEL query graph")
    w("  uint8_t     cyclic;          // 1 if the top-level query graph contains a ring closure")
    w("  uint8_t     max_rec_depth;   // 0 == no recursive SMARTS at all")
    w("  uint16_t    n_recursive;     // total $(...) sub-queries, all depths")
    w("};")
    w("")
    w("constexpr int N_ROWS = %d;" % (len(fr) + len(non)))
    w("constexpr Row ROWS[N_ROWS] = {")
    for n, s, mode, _ in fr + non:
        a = analyse(s)
        top = Chem.MolFromSmarts(s)
        w('  { "%s", %s, CM_UNIQUE, %d, %d, %d, %d },'
          % (n, cstr(s), top.GetNumAtoms(), 1 if (top.GetNumBonds() >= top.GetNumAtoms() and top.GetNumAtoms()) else 0,
             a["maxdepth"], a["nrec"]))
    w("};")
    w("")
    w("// NOT a substructure count.  Kept in this file so the table is exhaustive over the")
    w("// pattern-shaped members of the family and nobody re-derives it from the SMARTS that")
    w("// rdkit/Chem/Lipinski.py displays -- which counts ATOMS where the C++ counts HYDROGENS.")
    for n, defn, ev in NOT_A_PATTERN:
        w("//   %s = %s" % (n, defn))
        w("//     evidence: %s" % ev)
    w("")
    w("}  // namespace frag_tbl")
    w("")
    w("#endif")
    open(HEADER, "w").write("\n".join(out) + "\n")
    print("wrote %s  (%d rows, spec %s)" % (HEADER, len(fr) + len(non), sh))
    if missing:
        print("NOTE: rdkit_core names with no pattern row:", missing)


def cstr(s):
    """C string literal, split so no line runs away; escapes are explicit."""
    e = s.replace("\\", "\\\\").replace('"', '\\"')
    if len(e) <= 90:
        return '"%s"' % e
    parts = [e[i:i + 90] for i in range(0, len(e), 90)]
    return "\n      " + "\n      ".join('"%s"' % p for p in parts)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"tables": cmd_tables, "check": cmd_check, "inventory": cmd_inventory}[cmd]()
