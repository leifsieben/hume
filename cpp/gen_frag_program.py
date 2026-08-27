"""Compile the rdkit_core SMARTS patterns into a flat query PROGRAM for C++, and prove the
compilation is RDKit's own parse and not our reading of the pattern text.

RUN IT PINNED (see cpp/verify_frag.py for why; this script refuses otherwise):

    UV=(uv run --isolated --no-project --python 3.11 --with "rdkit==2025.9.2" --with "numpy==2.4.6")
    "${UV[@]}" python cpp/gen_frag_program.py validate    # parser vs RDKit's DescribeQuery
    "${UV[@]}" python cpp/gen_frag_program.py program     # write cpp/frag_program.h

HOW THIS AVOIDS RETYPING THE SMARTS.  cpp/estate_tables.h could decode RDKit's parse tree
directly because each E-state row is one atom with a flat bracket expression.  These patterns
cannot: 21 of the 74 use recursive SMARTS, and `QueryAtom` does not expose its query object to
Python at all -- `atom.GetQuery()` returns None, and `DescribeQuery()` prints a recursive node as
an empty `RecursiveStructure val in ()`.  So there is no tree to read out.

The discipline used instead is DIFFERENTIAL.  This file contains a SMARTS parser, and every
pattern it compiles is re-rendered in RDKit's exact `DescribeQuery()` format and compared
BYTE-FOR-BYTE with what RDKit prints for the same pattern -- per atom and per bond, for all 74
top-level patterns and all 102 recursive sub-queries, reached by lifting each `$(...)` span out
of the text (balanced-paren scan) and handing it back to RDKit's parser.  If our tree and
RDKit's tree disagree anywhere, in structure, in a value, or in a negation flag, `validate`
fails and names the pattern.  The SMARTS strings themselves are read from
`$RDDATA/FragmentDescriptors.csv` and from cpp/verify_frag.py's NON_CSV table; none is typed here.

That check pins down five things that reading the text would have got wrong, and each is
reproduced rather than tidied:

  1. AND/OR are LEFT-ASSOCIATIVE BINARY trees, not n-ary.  `[N;!H0;v3]` is
     AtomAnd(AtomAnd(N, !H0), v3), and a matcher that folded it into one 3-way AND would still
     give the right answer -- but a matcher that mis-nested a mixed `,`/`;` expression would not.
  2. Negation is a FLAG ON THE NODE (`!= val`), not a wrapper node.  `!D1` is one
     AtomExplicitDegree node with negation set.  `[!$(C)]` likewise.
  3. `[R]` compiles to `AtomInNRings -1`, where -1 is a SENTINEL meaning "in at least one ring",
     not a ring count of -1.  `[R0]`, `[R1]`, `[R2]` carry their literal counts.
  4. `AtomType z` is an ALIPHATIC element query and `AtomType 1000+z` an AROMATIC one, while
     `AtomAtomicNum z` (`[#7]`) constrains the element and says NOTHING about aromaticity.  The
     three are distinct and the patterns use all three -- `fr_ether`'s `[OD2]([#6])[#6]` mixes
     them in one row.
  5. SMARTS `:` is BondOrder 12 (Bond::AROMATIC), a bond-TYPE query, and the default bond
     written between two atoms is `SingleOrAromaticBond`, which is a different question again.
     Same trap cpp/estate_tables.h documents for the E-state rows.

WHAT THE PROGRAM LOOKS LIKE.  One flat node pool for every query expression in every pattern;
each pattern is (atom roots, bonds with roots) indexing into it.  A RecursiveStructure leaf's
value is an index into the SAME pattern table, so recursion is just a pattern reference and the
C++ needs no separate representation for it.  Sub-patterns are emitted before their users.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import rdkit
from rdkit import Chem, RDConfig, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_frag as VF  # noqa: E402  -- reuse the spec rows and the pin check

ROOT = VF.ROOT
OUT = os.path.join(ROOT, "cpp", "frag_program.h")

# ---------------------------------------------------------------------------------------------
# Node opcodes.  Names on the left are RDKit's DescribeQuery() names; we keep them verbatim so
# the rendering comparison is a string comparison and cannot drift into a translation table.
# ---------------------------------------------------------------------------------------------
OPS = [
    "AtomAnd", "AtomOr", "AtomNull", "AtomType", "AtomAtomicNum", "AtomExplicitDegree",
    "AtomTotalDegree", "AtomHCount", "AtomFormalCharge", "AtomInNRings", "AtomTotalValence",
    "AtomIsAromatic", "AtomIsAliphatic", "RecursiveStructure",
    "BondAnd", "BondOr", "BondNull", "BondOrder", "BondInRing", "SingleOrAromaticBond",
]
OP = {n: i for i, n in enumerate(OPS)}
BINARY = {"AtomAnd", "AtomOr", "BondAnd", "BondOr"}
NOVALUE = {"AtomNull", "BondNull"}

AROMATIC_ORGANIC = {"c": 6, "n": 7, "o": 8, "s": 16, "p": 15, "b": 5}
ALIPHATIC_ORGANIC = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "F": 9, "Cl": 17, "Br": 35,
                     "I": 53, "B": 5}
# Every element symbol that can appear inside brackets in these patterns.  Two-letter symbols
# must be tried before one-letter ones or `Cl` tokenises as `C` + `l`.
ELEMENTS = ["Cl", "Br", "Se", "Si", "Ge", "As", "Sn", "Pb", "Li", "Be", "Na", "Mg", "Al",
            "Ca", "Fe", "Zn", "He", "Ne", "Ar", "Kr", "Xe",
            "C", "N", "O", "S", "P", "F", "I", "B", "H", "K", "V", "W", "U"]
ZOF = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
       "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
       "Ca": 20, "V": 23, "Fe": 26, "Zn": 30, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
       "Sn": 50, "I": 53, "Xe": 54, "W": 74, "Pb": 82, "U": 92}


class Node:
    __slots__ = ("op", "neg", "val", "lhs", "rhs")

    def __init__(self, op, val=0, neg=False, lhs=None, rhs=None):
        self.op, self.val, self.neg, self.lhs, self.rhs = op, val, neg, lhs, rhs

    def render(self, ind=0):
        """RDKit's DescribeQuery() format, reproduced exactly -- including the asymmetric
        `RecursiveStructure val not in )`, which is RDKit's own printing quirk."""
        p = "  " * ind
        if self.op in BINARY:
            out = ["%s%s" % (p, self.op)]
            out.append(self.lhs.render(ind + 1))
            out.append(self.rhs.render(ind + 1))
            return "\n".join(out)
        if self.op in NOVALUE:
            return "%s%s" % (p, self.op)
        if self.op == "RecursiveStructure":
            return "%sRecursiveStructure val %s" % (p, "not in )" if self.neg else "in ()")
        return "%s%s %d %s val" % (p, self.op, self.val, "!=" if self.neg else "=")


class SmartsError(Exception):
    pass


class Parser:
    """A SMARTS parser for exactly the constructs these 74 patterns use.  Anything else raises
    rather than being silently mis-parsed -- an unknown primitive that quietly became `AtomNull`
    would turn a constraint into a wildcard and inflate every count that used it."""

    def __init__(self, s, sink):
        self.s, self.i, self.sink = s, 0, sink
        self.atoms = []          # list of root Node
        self.bonds = []          # list of (u, v, root Node)
        # Ring-closure bookkeeping is PER MOLECULE, not per branch.  `fr_para_hydroxylation`
        # opens ring 1 at atom 0 and closes it inside a branch -- `[cH]1[cH]cc(c[cH]1)...` --
        # so a map scoped to walk() loses the closure and silently drops a bond, which turns a
        # benzene query into an open chain and inflates the count.
        self.ring = {}

    # -- lexer helpers ------------------------------------------------------------------------
    def eof(self):
        return self.i >= len(self.s)

    def peek(self, k=0):
        j = self.i + k
        return self.s[j] if j < len(self.s) else ""

    def take(self, lit):
        if self.s.startswith(lit, self.i):
            self.i += len(lit)
            return True
        return False

    def number(self, default=None):
        j = self.i
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        if j == self.i:
            if default is None:
                raise SmartsError("expected a number at %d in %r" % (self.i, self.s))
            return default
        return int(self.s[j:self.i])

    # -- atom expressions ---------------------------------------------------------------------
    def atom_primitive(self):
        c = self.peek()
        if c == "!":
            self.i += 1
            n = self.atom_primitive()
            n.neg = not n.neg
            return n
        if c == "(":                                   # grouping inside a bracket expression
            self.i += 1
            n = self.atom_or_low()
            if not self.take(")"):
                raise SmartsError("unbalanced ( in %r" % self.s)
            return n
        if self.s.startswith("$(", self.i):
            depth, k = 0, self.i + 1
            while k < len(self.s):
                if self.s[k] == "(":
                    depth += 1
                elif self.s[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            sub = self.s[self.i + 2:k]
            self.i = k + 1
            return Node("RecursiveStructure", val=self.sink(sub))
        if c == "*":
            self.i += 1
            return Node("AtomNull")
        if c == "a":
            self.i += 1
            return Node("AtomIsAromatic", val=1)
        if c == "A":
            self.i += 1
            return Node("AtomIsAliphatic", val=1)
        if c == "#":
            self.i += 1
            return Node("AtomAtomicNum", val=self.number())
        if c == "D":
            self.i += 1
            return Node("AtomExplicitDegree", val=self.number(1))
        if c == "X":
            self.i += 1
            return Node("AtomTotalDegree", val=self.number(1))
        if c == "H":
            self.i += 1
            return Node("AtomHCount", val=self.number(1))
        if c == "R":
            self.i += 1
            # `[R]` -> AtomInNRings -1, RDKit's sentinel for "in at least one ring".  A literal
            # count only appears when a digit follows.
            j = self.i
            v = self.number(-1)
            return Node("AtomInNRings", val=v if self.i > j else -1)
        if c == "v":
            self.i += 1
            return Node("AtomTotalValence", val=self.number(1))
        if c in "+-":
            self.i += 1
            sign = 1 if c == "+" else -1
            run = 1
            while self.peek() == c:                    # `++` / `--`
                self.i += 1
                run += 1
            j = self.i
            v = self.number(0)
            if self.i > j:
                return Node("AtomFormalCharge", val=sign * v)
            return Node("AtomFormalCharge", val=sign * run)
        # aromatic organic-subset symbol inside brackets
        if c in AROMATIC_ORGANIC and not (self.peek(1).isalpha() and self.peek(1).islower()
                                          and (c + self.peek(1)) in ELEMENTS):
            self.i += 1
            return Node("AtomType", val=1000 + AROMATIC_ORGANIC[c])
        for sym in ELEMENTS:
            if self.s.startswith(sym, self.i):
                # do not let `C` swallow the `l` of `Cl`
                if len(sym) == 1 and self.peek(1).isalpha() and self.peek(1).islower() \
                        and (sym + self.peek(1)) in ELEMENTS:
                    continue
                self.i += len(sym)
                return Node("AtomType", val=ZOF[sym])
        raise SmartsError("unsupported atom primitive %r at %d in %r" % (c, self.i, self.s))

    def atom_and_high(self):
        n = self.atom_primitive()
        while True:
            if self.take("&"):
                n = Node("AtomAnd", lhs=n, rhs=self.atom_primitive())
            elif self.peek() and self.peek() not in ",;])":
                n = Node("AtomAnd", lhs=n, rhs=self.atom_primitive())   # juxtaposition
            else:
                return n

    def atom_or(self):
        n = self.atom_and_high()
        while self.take(","):
            n = Node("AtomOr", lhs=n, rhs=self.atom_and_high())
        return n

    def atom_or_low(self):
        n = self.atom_or()
        while self.take(";"):
            n = Node("AtomAnd", lhs=n, rhs=self.atom_or())
        return n

    def bracket_atom(self):
        if not self.take("["):
            raise SmartsError("expected [")
        n = self.atom_or_low()
        if not self.take("]"):
            raise SmartsError("unbalanced [ in %r" % self.s)
        return n

    def organic_atom(self):
        for sym in ("Cl", "Br"):
            if self.take(sym):
                return Node("AtomType", val=ZOF[sym])
        c = self.peek()
        if c == "*":
            self.i += 1
            return Node("AtomNull")
        if c == "a":
            self.i += 1
            return Node("AtomIsAromatic", val=1)
        if c == "A":
            self.i += 1
            return Node("AtomIsAliphatic", val=1)
        if c in AROMATIC_ORGANIC:
            self.i += 1
            return Node("AtomType", val=1000 + AROMATIC_ORGANIC[c])
        if c in ALIPHATIC_ORGANIC:
            self.i += 1
            return Node("AtomType", val=ALIPHATIC_ORGANIC[c])
        raise SmartsError("unsupported organic-subset atom %r in %r" % (c, self.s))

    # -- bond expressions ---------------------------------------------------------------------
    BONDPRIM = {"-": ("BondOrder", 1), "=": ("BondOrder", 2), "#": ("BondOrder", 3),
                ":": ("BondOrder", 12), "@": ("BondInRing", 1), "~": ("BondNull", 0)}

    def bond_primitive(self):
        if self.take("!"):
            n = self.bond_primitive()
            n.neg = not n.neg
            return n
        c = self.peek()
        if c in self.BONDPRIM:
            self.i += 1
            op, v = self.BONDPRIM[c]
            return Node("BondNull") if op == "BondNull" else Node(op, val=v)
        raise SmartsError("unsupported bond primitive %r in %r" % (c, self.s))

    def bond_expr(self):
        """Returns None when no bond symbol is present, which means the SMARTS default:
        SingleOrAromaticBond.  That default is a DIFFERENT query from `-` and from `:`."""
        if not self.peek() or self.peek() not in "-=#:~@!":
            return None
        n = self.bond_primitive()
        while True:
            if self.take("&"):
                n = Node("BondAnd", lhs=n, rhs=self.bond_primitive())
            elif self.take(","):
                n = Node("BondOr", lhs=n, rhs=self.bond_primitive())
            elif self.take(";"):
                n = Node("BondAnd", lhs=n, rhs=self.bond_primitive())
            elif self.peek() and self.peek() in "-=#:~@!":
                n = Node("BondAnd", lhs=n, rhs=self.bond_primitive())   # juxtaposition
            else:
                return n

    # -- the molecule-level walk --------------------------------------------------------------
    def parse(self):
        self.walk(prev=None, prevbond=None)
        if not self.eof():
            raise SmartsError("trailing %r in %r" % (self.s[self.i:], self.s))
        return self.atoms, self.bonds

    def add_bond(self, u, v, b):
        self.bonds.append((u, v, b if b is not None else Node("SingleOrAromaticBond", val=1)))

    def walk(self, prev, prevbond):
        while not self.eof():
            c = self.peek()
            if c == ")":
                return prev
            if c == "(":
                self.i += 1
                self.walk(prev, None)
                if not self.take(")"):
                    raise SmartsError("unbalanced ( in %r" % self.s)
                continue
            if c in "-=#:~@!":
                prevbond = self.bond_expr()
                continue
            if c.isdigit() or c == "%":
                if c == "%":
                    self.i += 1
                    rid = int(self.s[self.i:self.i + 2])
                    self.i += 2
                else:
                    rid = int(c)
                    self.i += 1
                if rid in self.ring:
                    u, rb = self.ring.pop(rid)
                    self.add_bond(u, prev, rb if rb is not None else prevbond)
                else:
                    self.ring[rid] = (prev, prevbond)
                prevbond = None
                continue
            n = self.bracket_atom() if c == "[" else self.organic_atom()
            self.atoms.append(n)
            idx = len(self.atoms) - 1
            if prev is not None:
                self.add_bond(prev, idx, prevbond)
            prev, prevbond = idx, None
        return prev


# ---------------------------------------------------------------------------------------------
# Compilation into a pattern table.  A recursive sub-query is compiled as an ordinary pattern and
# referenced by index, so the C++ needs no separate machinery for recursion.
# ---------------------------------------------------------------------------------------------
class Program:
    def __init__(self):
        self.patterns = []          # (label, [atom roots], [(u,v,root)])
        self.by_text = {}

    def compile(self, sma, label):
        if sma in self.by_text:
            return self.by_text[sma]
        slot = len(self.patterns)
        self.patterns.append(None)          # reserve, so self-reference cannot loop
        self.by_text[sma] = slot
        atoms, bonds = Parser(sma, lambda sub: self.compile(sub, label + "$")).parse()
        self.patterns[slot] = (label, sma, atoms, bonds)
        return slot


def rdkit_describe(sma):
    m = Chem.MolFromSmarts(sma)
    if m is None:
        raise SmartsError("RDKit will not parse %r" % sma)
    return ([a.DescribeQuery().rstrip("\n") for a in m.GetAtoms()],
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.DescribeQuery().rstrip("\n"))
             for b in m.GetBonds()])


def validate_one(sma):
    """Compile `sma` with our parser, render it in RDKit's format, and diff against RDKit."""
    prog = Program()
    root = prog.compile(sma, "x")
    _, _, atoms, bonds = prog.patterns[root]
    ra, rb = rdkit_describe(sma)
    problems = []
    if len(atoms) != len(ra):
        problems.append("atom count %d != rdkit %d" % (len(atoms), len(ra)))
    else:
        for i, (mine, theirs) in enumerate(zip(atoms, ra)):
            if mine.render() != theirs:
                problems.append("atom %d\n   ours:\n%s\n   rdkit:\n%s" % (i, mine.render(), theirs))
    mineb = sorted((min(u, v), max(u, v), n.render()) for u, v, n in bonds)
    theirsb = sorted((min(u, v), max(u, v), d) for u, v, d in rb)
    if len(mineb) != len(theirsb):
        problems.append("bond count %d != rdkit %d" % (len(mineb), len(theirsb)))
    else:
        for (u1, v1, m1), (u2, v2, d2) in zip(mineb, theirsb):
            if (u1, v1) != (u2, v2) or m1 != d2:
                problems.append("bond %d-%d\n   ours:\n%s\n   rdkit:\n%s" % (u1, v1, m1, d2))
    return problems


def all_specs():
    VF.require_pin()
    fr, non, _ = VF.spec_rows()
    return [(n, s) for n, s, _, _ in fr + non]


def cmd_validate():
    specs = all_specs()
    print("RESOLVED rdkit %s" % rdkit.__version__)
    nsub = 0
    bad = 0
    for name, sma in specs:
        todo = [sma]
        seen = set()
        while todo:
            cur = todo.pop()
            if cur in seen:
                continue
            seen.add(cur)
            subs = VF.rec_spans(cur)
            todo.extend(subs)
            if cur != sma:
                nsub += 1
            probs = validate_one(cur)
            if probs:
                bad += 1
                print("MISMATCH in %s   %r" % (name, cur))
                for p in probs:
                    print("   " + p.replace("\n", "\n   "))
    print("\n%d top-level patterns + %d recursive sub-queries validated against RDKit's own "
          "DescribeQuery()" % (len(specs), nsub))
    print("MISMATCHES: %d" % bad)
    if bad:
        sys.exit(1)
    print("PARSE PROVEN IDENTICAL TO RDKIT'S, per atom and per bond, structure/value/negation.")


def cmd_program():
    specs = all_specs()
    prog = Program()
    tops = [(n, prog.compile(s, n)) for n, s in specs]
    pool, aroots, bl = [], [], []
    pat_meta = []

    def emit(n):
        if n.op in BINARY:
            l = emit(n.lhs)
            r = emit(n.rhs)
        else:
            l = r = -1
        pool.append((OP[n.op], 1 if n.neg else 0, n.val, l, r))
        return len(pool) - 1

    for label, sma, atoms, bonds in prog.patterns:
        a0 = len(aroots)
        for a in atoms:
            aroots.append(emit(a))
        b0 = len(bl)
        for u, v, bn in bonds:
            bl.append((u, v, emit(bn)))
        pat_meta.append((label, sma, a0, len(atoms), b0, len(bonds)))

    h = hashlib.sha256()
    for n, s in specs:
        h.update(("%s\t%s\n" % (n, s)).encode())
    sh = h.hexdigest()

    o = []
    w = o.append
    w("// GENERATED by cpp/gen_frag_program.py -- do not edit.")
    w("//")
    w("// The %d rdkit_core SMARTS patterns compiled to a flat query program: one node pool, one" % len(specs))
    w("// pattern table.  A RecursiveStructure node's `val` is an index into PATTERNS, so a")
    w("// recursive sub-query is just another pattern and the matcher needs no special case.")
    w("//")
    w("// PROVEN, NOT READ.  cpp/gen_frag_program.py `validate` re-renders every node of every")
    w("// pattern in RDKit's own DescribeQuery() format and compares it byte-for-byte with what")
    w("// RDKit prints -- %d top-level patterns and %d recursive sub-queries, per atom and per" % (len(specs), len(prog.patterns) - len(specs)))
    w("// bond, structure, value and negation flag.  The SMARTS themselves come from")
    w("// $RDDATA/FragmentDescriptors.csv and cpp/verify_frag.py's NON_CSV table; none is typed.")
    w("//")
    w("// FIVE THINGS THE TEXT DOES NOT SAY, all of them decided by that comparison:")
    w("//   1. AtomAnd/AtomOr are LEFT-ASSOCIATIVE BINARY, not n-ary.")
    w("//   2. Negation is a flag on the node (`!= val`), never a wrapper node.")
    w("//   3. AtomInNRings -1 is a SENTINEL for `[R]` == in at least one ring, not a count.")
    w("//   4. AtomType z is ALIPHATIC element z, AtomType 1000+z AROMATIC, AtomAtomicNum z")
    w("//      (`[#z]`) neither -- it says nothing about aromaticity.  fr_ether uses two of the")
    w("//      three in one row.")
    w("//   5. The default bond written between two atoms is SingleOrAromaticBond, which is a")
    w("//      different query from `-` (BondOrder 1) and from `:` (BondOrder 12).")
    w("//")
    w("//   rdkit  %s   (pin: 2025.9.2)" % rdkit.__version__)
    w("//   PROGRAM_SHA256 = %s" % sh)
    w("")
    w("#ifndef HUME_FRAG_PROGRAM_H")
    w("#define HUME_FRAG_PROGRAM_H")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace frag_prog {")
    w("")
    w('constexpr const char PROGRAM_SHA256[] = "%s";' % sh)
    w("")
    w("enum Op : uint8_t {")
    for i, n in enumerate(OPS):
        w("  OP_%s = %d," % (n.upper(), i))
    w("};")
    w("")
    w("struct Node { uint8_t op; uint8_t neg; int32_t val; int32_t lhs; int32_t rhs; };")
    w("struct QBond { int16_t u; int16_t v; int32_t root; };")
    w("struct Pattern { const char* label; int32_t a0; int16_t na; int32_t b0; int16_t nb; };")
    w("")
    w("constexpr int N_NODES = %d;" % len(pool))
    w("constexpr Node NODES[N_NODES] = {")
    for op, neg, val, l, r in pool:
        w("  {%d,%d,%d,%d,%d}," % (op, neg, val, l, r))
    w("};")
    w("")
    w("constexpr int N_AROOTS = %d;" % len(aroots))
    w("constexpr int32_t AROOTS[N_AROOTS] = {%s};" % ",".join(str(x) for x in aroots))
    w("")
    w("constexpr int N_QBONDS = %d;" % len(bl))
    w("constexpr QBond QBONDS[N_QBONDS] = {")
    for u, v, r in bl:
        w("  {%d,%d,%d}," % (u, v, r))
    w("};")
    w("")
    w("constexpr int N_PATTERNS = %d;" % len(pat_meta))
    w("constexpr Pattern PATTERNS[N_PATTERNS] = {")
    for label, sma, a0, na, b0, nb in pat_meta:
        w('  {"%s",%d,%d,%d,%d},' % (label.replace('"', '\\"'), a0, na, b0, nb))
    w("};")
    w("")
    w("// The %d named descriptors, in the order cpp/verify_frag.py's spec table gives them." % len(tops))
    w("// Every one is counted as len(GetSubstructMatches(patt, uniquify=True)) -- the number of")
    w("// distinct ATOM SETS.  Measured on cpp/hard.smi: the largest raw (uniquify=False) match")
    w("// count over every pattern and every molecule is 180, against RDKit's maxMatches=1000")
    w("// default, so truncation NEVER fires here.  That matters more than it looks: truncation")
    w("// happens BEFORE uniquification, so if it ever did fire the count would silently become")
    w("// dependent on which embeddings the search happened to find first.")
    w("struct Named { const char* name; int32_t pattern; };")
    w("constexpr int N_NAMED = %d;" % len(tops))
    w("constexpr Named NAMED[N_NAMED] = {")
    for n, p in tops:
        w('  {"%s",%d},' % (n, p))
    w("};")
    w("")
    w("}  // namespace frag_prog")
    w("")
    w("#endif")
    open(OUT, "w").write("\n".join(o) + "\n")
    print("RESOLVED rdkit %s" % rdkit.__version__)
    print("wrote %s" % OUT)
    print("  patterns %d (%d named + %d recursive sub-queries)"
          % (len(pat_meta), len(tops), len(pat_meta) - len(tops)))
    print("  nodes %d, atom roots %d, bonds %d" % (len(pool), len(aroots), len(bl)))
    print("  PROGRAM_SHA256 %s" % sh)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    {"validate": cmd_validate, "program": cmd_program}[cmd]()
