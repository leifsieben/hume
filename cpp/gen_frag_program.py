"""Compile a set of SMARTS patterns into a flat query PROGRAM for C++, and prove the compilation
is RDKit's own parse and not our reading of the pattern text.

TWO SPEC SETS GO THROUGH THIS ONE GENERATOR, and that is the point rather than a convenience:

    frag   the 74 `rdkit_core` fragment/pattern descriptors   -> cpp/frag_program.h
    qed    rdkit.Chem.QED.StructuralAlertSmarts, all 116      -> cpp/qed_alert_program.h

They compile to the SAME node/pattern representation and are matched by the SAME evaluator in
src/hume_core/frag_matcher.h, which takes its tables as a bound `Program` reference rather than
reading one namespace's at global scope.  Two subgraph-isomorphism implementations would be two
things to verify and two places for a divergence to hide.

RUN IT PINNED (see cpp/verify_frag.py for why; this script refuses otherwise):

    UV=(uv run --isolated --no-project --python 3.11 --with "rdkit==2025.9.2" --with "numpy==2.4.6")
    "${UV[@]}" python cpp/gen_frag_program.py validate [frag|qed|all]  # vs RDKit's DescribeQuery
    "${UV[@]}" python cpp/gen_frag_program.py program  [frag|qed|all]  # write the header(s)
    "${UV[@]}" python cpp/gen_frag_program.py check    [frag|qed|all]  # SPEC drift guard

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

WHAT THE QED ALERT SET NEEDED THAT THE FRAGMENT SET DID NOT, all four found by running `validate`
on it rather than by reading the SMARTS:

  1. ISOTOPE.  `[15N]` is AtomAnd(AtomType 7, AtomIsotope 15) -- the isotope is folded into the
     symbol as ONE primitive, so `[15NH2+]` nests as ((N & iso) & H2) & +1 and not any other way.
     Alerts 112-115 are `[15N]` `[13C]` `[18O]` `[34S]` and are nothing else.  This is the one
     addition that needed a new BOUNDARY column: `getIsotope()` is not derivable from Z, and the
     mass column only says "labelled", not "labelled with what".
  2. `!r` -> AtomInRing, which is NOT AtomInNRings.  `[C!r]` is AtomAnd(AtomType 6,
     AtomInRing 1 != val).  Alerts 67 and 88.  (`r<n>` is AtomMinRingSize, a third primitive
     again; no spec here uses it and the parser raises rather than guessing.)
  3. COMPONENT-LEVEL `.`.  `F.F.F.F` and `C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]` are ONE query
     with four / nine atoms and no bond joining the components.  Nothing is needed in the matcher
     for this -- `buildPlan()` already walks every connected component of the query graph -- only
     in this parser, which previously treated `.` as an unknown atom symbol.
  4. THE ELEMENT SYMBOL IS NOT ALWAYS `AtomType`.  Only the SMARTS organic subset
     {B,C,N,O,P,S,F,Cl,Br,I} compiles to AtomType (an ALIPHATIC element query); every other
     bracketed symbol compiles to AtomAtomicNum, which says nothing about aromaticity.  `[Si]`
     is AtomAtomicNum 14 while `[si]` is AtomType 1014.  That is the same quirk
     cpp/estate_tables.h records for `[SeD2H0]`, and here it is load-bearing: alert 20 lists 39
     metals and metalloids, all of them AtomAtomicNum.

`~` (BondNull) and `@` (BondInRing) were ALREADY in this parser and already implemented in
frag_matcher.h; the alert set is simply the first spec that exercises them.  Two-letter element
symbols are now matched LONGEST-FIRST and before the single-letter primitives, because `[Ru]`,
`[Ba]`, `[Ho]`, `[Nb]` and `[Hf]` would otherwise tokenise as `R`+`u`, `B`+`a`, `H`+`o` and so on
-- RDKit reads all five as elements.

ONE CONSTRUCT IS DELIBERATELY REFUSED RATHER THAN GUESSED.  A bracket whose whole content is a
bare `H` (optionally with an isotope or a charge) is RDKit's HYDROGEN ATOM, `[H]` ->
AtomAtomicNum 1, while `H` anywhere else in a bracket expression is an H COUNT -- `[H,C]` is
AtomOr(AtomHCount 1, AtomType 6) and `[H1]` is AtomHCount 1.  Neither spec set uses the atom
form, so the parser raises on it instead of carrying a rule nothing here validates.
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

# ---------------------------------------------------------------------------------------------
# Node opcodes.  Names on the left are RDKit's DescribeQuery() names; we keep them verbatim so
# the rendering comparison is a string comparison and cannot drift into a translation table.
# ---------------------------------------------------------------------------------------------
OPS = [
    "AtomAnd", "AtomOr", "AtomNull", "AtomType", "AtomAtomicNum", "AtomExplicitDegree",
    "AtomTotalDegree", "AtomHCount", "AtomFormalCharge", "AtomInNRings", "AtomTotalValence",
    "AtomIsAromatic", "AtomIsAliphatic", "RecursiveStructure",
    "BondAnd", "BondOr", "BondNull", "BondOrder", "BondInRing", "SingleOrAromaticBond",
    # Added for the QED structural alerts.  APPENDED, never inserted: the opcode number is
    # baked into cpp/frag_program.h, so renumbering the existing twenty would silently
    # reinterpret every node of the already-verified fragment program.
    "AtomIsotope", "AtomInRing",
]
OP = {n: i for i, n in enumerate(OPS)}
BINARY = {"AtomAnd", "AtomOr", "BondAnd", "BondOr"}
NOVALUE = {"AtomNull", "BondNull"}

# THE ORGANIC SUBSET IS THE WHOLE OF THE `AtomType` SET, and everything else is `AtomAtomicNum`.
# `[C]` is AtomType 6 (an ALIPHATIC carbon query); `[Si]` is AtomAtomicNum 14 and constrains the
# element only.  Their lowercase forms are AtomType 1000+z either way -- `[si]` is AtomType 1014.
# `cmd_validate` asserts this classification element by element against RDKit's own parse, so it
# is a claim under test rather than a comment.
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

# The aromatic (lowercase) symbols RDKit's SMARTS parser accepts, and the element they name.
AROMATIC = {"b": 5, "c": 6, "n": 7, "o": 8, "p": 15, "s": 16,
            "se": 34, "as": 33, "te": 52, "si": 14}

# Symbol -> atomic number, for every symbol either spec set can name inside a bracket.  Hand
# written, and then checked against `Chem.GetPeriodicTable()` in `cmd_validate` -- a typo here
# would be an element query for the wrong element, which no pattern-level diff would catch if the
# same typo reached both sides.
ZOF = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
       "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
       "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
       "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
       "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
       "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
       "Cs": 55, "Ba": 56, "La": 57, "Ho": 67, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76,
       "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "U": 92}

# Every symbol the tokeniser will try, LONGEST FIRST.  Two-letter symbols must beat one-letter
# ones twice over: `Cl` must not tokenise as `C` + `l`, and `[Ru]` / `[Ba]` / `[Ho]` / `[Nb]` /
# `[Hf]` must not tokenise as the single-letter primitives `R` / `B`+`a` / `H`+`o` / `N`+`b`.
# RDKit reads all of those as elements.
#
# THE BARE `H` IS EXCLUDED, and that exclusion is the whole of the hydrogen rule this file
# implements.  `H` in a general bracket expression is an H COUNT, not the element: `[nH]` is
# AtomAnd(AtomType 1007, AtomHCount 1) and `[H1]` is AtomHCount 1.  RDKit reads `H` as the
# ELEMENT only in its `hydrogen_atom` production -- a bracket that is nothing but `H` with an
# optional isotope and charge -- and `bracket_atom()` below refuses that form outright rather
# than reproducing a rule no spec here exercises.  The two-letter symbols starting with H (`He`,
# `Hf`, `Hg`, `Ho`) are unaffected: they cannot match an `H` that is followed by `]`, a digit or
# an operator.
SYMBOLS = sorted((set(ZOF) | set(AROMATIC)) - {"H"}, key=lambda s: (-len(s), s))

# A bracket whose entire content is a bare hydrogen atom; see SYMBOLS.
BARE_H = re.compile(r"^\d*H(?:[+-]+\d*)?$")


def symbol_node(sym):
    """The node RDKit builds for a bare element symbol inside a bracket."""
    if sym in AROMATIC:
        return Node("AtomType", val=1000 + AROMATIC[sym])
    if sym in ORGANIC_SUBSET:
        return Node("AtomType", val=ZOF[sym])
    return Node("AtomAtomicNum", val=ZOF[sym])


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
    def symbol(self):
        """The longest element / aromatic symbol at the cursor, or None.  Longest-first: see
        SYMBOLS for the five two-letter symbols that would otherwise tokenise as a one-letter
        primitive plus a stray character."""
        for sym in SYMBOLS:
            if self.s.startswith(sym, self.i):
                self.i += len(sym)
                return sym
        return None

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
        # ISOTOPE.  Leading digits bind to the symbol that follows and the pair is ONE primitive:
        # `[15N]` is AtomAnd(AtomType 7, AtomIsotope 15), so `[15NH2+]` nests as
        # ((N & iso) & H2) & +1.  With `*` the symbol contributes nothing and RDKit emits the
        # isotope node alone -- `[0*]` is AtomIsotope 0 and no AtomNull.
        if c.isdigit():
            iso = self.number()
            sym = self.symbol()
            if sym is None:
                if not self.take("*"):
                    raise SmartsError("isotope %d with no element symbol at %d in %r"
                                      % (iso, self.i, self.s))
                return Node("AtomIsotope", val=iso)
            return Node("AtomAnd", lhs=symbol_node(sym), rhs=Node("AtomIsotope", val=iso))
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
        if c == "#":
            self.i += 1
            return Node("AtomAtomicNum", val=self.number())
        # ELEMENT SYMBOLS BEFORE THE SINGLE-LETTER PRIMITIVES.  `[As]` is arsenic, not
        # `A`(aliphatic) + `s`; `[Ru]` is ruthenium, not `R`(ring) + `u`; `[Ba]`, `[Ho]`, `[Nb]`
        # and `[Hf]` are the same trap.  A one-letter symbol cannot shadow a primitive here --
        # the only overlap in the whole table is `H`, which SYMBOLS excludes for exactly that
        # reason -- and a primitive cannot shadow a symbol because `A;`, `R]`, `R1`, `H2` and
        # friends match no two-letter symbol.
        sym = self.symbol()
        if sym is not None:
            return symbol_node(sym)
        if c == "a":
            self.i += 1
            return Node("AtomIsAromatic", val=1)
        if c == "A":
            self.i += 1
            return Node("AtomIsAliphatic", val=1)
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
        if c == "r":
            self.i += 1
            # `r` alone -> AtomInRing, a BOOLEAN and a DIFFERENT primitive from `[R]`'s
            # AtomInNRings: `[C!r]` is AtomAnd(AtomType 6, AtomInRing 1 != val).  `r<n>` is
            # AtomMinRingSize, a third primitive again -- no spec here uses it, so it raises
            # rather than being guessed at.
            if self.peek().isdigit():
                raise SmartsError("ring-size query `r%s` needs AtomMinRingSize, which no spec "
                                  "here exercises and this generator does not emit (%r)"
                                  % (self.peek(), self.s))
            return Node("AtomInRing", val=1)
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
        j = self.i
        depth = 0
        while j < len(self.s) and (self.s[j] != "]" or depth):
            if self.s[j] == "(":
                depth += 1
            elif self.s[j] == ")":
                depth -= 1
            j += 1
        if BARE_H.match(self.s[self.i:j]):
            raise SmartsError(
                "bracket %r is RDKit's HYDROGEN ATOM production (`[H]` -> AtomAtomicNum 1), not "
                "an H count.  Neither spec set uses it, so this generator refuses it rather than "
                "carrying an unvalidated rule; see the module docstring." % self.s[self.i - 1:j + 1])
        n = self.atom_or_low()
        if not self.take("]"):
            raise SmartsError("unbalanced [ in %r" % self.s)
        return n

    def organic_atom(self):
        """An atom written WITHOUT brackets, where only the organic subset is legal.  Every one
        of these is an AtomType -- the AtomAtomicNum spelling needs a bracket."""
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
        if c in AROMATIC:
            self.i += 1
            return Node("AtomType", val=1000 + AROMATIC[c])
        if c in ORGANIC_SUBSET:
            self.i += 1
            return Node("AtomType", val=ZOF[c])
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
            # COMPONENT SEPARATOR.  `F.F.F.F` is ONE query of four atoms with no bond between
            # them; the next atom simply has no predecessor.  Atom numbering runs straight
            # through, which is what makes the per-atom diff against RDKit line up.  Ring-closure
            # bookkeeping deliberately survives a `.` -- `C1.C1` is a bonded pair in SMARTS.
            # Nothing is needed in the matcher: buildPlan() already starts a fresh scan at the
            # root of every connected component of the query graph.
            if c == ".":
                self.i += 1
                prev, prevbond = None, None
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


def qed_specs():
    """rdkit.Chem.QED's 116 structural alerts, READ OUT OF THE MODULE, never transcribed.

    They are a bare list with no names of their own, so the name is the INDEX -- which is also
    the only stable identifier they have.  `properties()` counts them as
    `sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert))`, i.e. a BOOLEAN per
    alert, not a match count; that is why the C++ calls `hasMatch()` and not `matchCount()`.
    """
    VF.require_pin()
    from rdkit.Chem import QED
    return [("alert_%03d" % i, s) for i, s in enumerate(QED.StructuralAlertSmarts)]


# name -> (spec function, output header, C++ namespace, include guard, one-line description)
SPEC_SETS = {
    "frag": (all_specs, os.path.join(ROOT, "cpp", "frag_program.h"), "frag_prog",
             "HUME_FRAG_PROGRAM_H",
             "the rdkit_core fragment/pattern descriptors"),
    "qed": (qed_specs, os.path.join(ROOT, "cpp", "qed_alert_program.h"), "qed_prog",
            "HUME_QED_ALERT_PROGRAM_H",
            "rdkit.Chem.QED.StructuralAlertSmarts, the ALERTS term of `qed`"),
}


def spec_hash(specs):
    """sha256 of the SPECIFICATION -- the (name, SMARTS) pairs -- and of no file.

    House rule 6: `sha256(QED.py)` would move on a docstring edit and cry wolf, while a spec that
    moved inside an unchanged file would slip past a hash of the wrong thing.  The fragment spec
    DID move between rdkit 2025.09.2 and 2026.03.5, so this is not hypothetical.
    """
    h = hashlib.sha256()
    for n, s in specs:
        h.update(("%s\t%s\n" % (n, s)).encode())
    return h.hexdigest()


def _which(argv_i=2):
    """Which spec set(s) a command applies to.  Default `all`, so a bare `validate` still checks
    everything and cannot pass by silently examining only half the repo."""
    which = sys.argv[argv_i] if len(sys.argv) > argv_i else "all"
    if which == "all":
        return list(SPEC_SETS)
    if which not in SPEC_SETS:
        sys.exit("unknown spec set %r; expected one of %s or `all`"
                 % (which, ", ".join(SPEC_SETS)))
    return [which]


def check_element_tables():
    """Two claims this file makes about elements, asserted against RDKit rather than commented.

    1. ZOF's symbol -> atomic number is RDKit's.  A typo here is an element query for the WRONG
       element, and the per-pattern diff below would not catch it if the same symbol appeared on
       both sides of the comparison.
    2. ORGANIC_SUBSET is exactly the set of bracketed symbols RDKit compiles to `AtomType`;
       every other symbol compiles to `AtomAtomicNum`, which says nothing about aromaticity.
       This is the `[SeD2H0]` quirk cpp/estate_tables.h records, generalised and put under test.
    """
    pt = Chem.GetPeriodicTable()
    bad = 0
    for sym, z in sorted(ZOF.items()):
        if pt.GetAtomicNumber(sym) != z:
            print("  ZOF[%r] = %d, RDKit says %d" % (sym, z, pt.GetAtomicNumber(sym)))
            bad += 1
    for sym in sorted(ZOF):
        m = Chem.MolFromSmarts("[%s]" % sym)
        got = m.GetAtomWithIdx(0).DescribeQuery().strip().split()[0]
        want = "AtomType" if sym in ORGANIC_SUBSET else "AtomAtomicNum"
        if got != want:
            print("  [%s] is %s to RDKit, %s here" % (sym, got, want))
            bad += 1
    for sym, z in sorted(AROMATIC.items()):
        m = Chem.MolFromSmarts("[%s]" % sym)
        d = m.GetAtomWithIdx(0).DescribeQuery().strip()
        if d != "AtomType %d = val" % (1000 + z):
            print("  [%s] is %r to RDKit, AtomType %d here" % (sym, d, 1000 + z))
            bad += 1
    print("element tables: %d symbols, %d aromatic forms, MISMATCHES: %d"
          % (len(ZOF), len(AROMATIC), bad))
    return bad


def cmd_validate():
    which = _which()
    print("RESOLVED rdkit %s" % rdkit.__version__)
    VF.require_pin()
    total_bad = check_element_tables()
    for w in which:
        print("\n--- %s: %s ---" % (w, SPEC_SETS[w][4]))
        total_bad += validate_set(SPEC_SETS[w][0]())
    if total_bad:
        sys.exit(1)


def validate_set(specs):
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
    if not bad:
        print("PARSE PROVEN IDENTICAL TO RDKIT'S, per atom and per bond, "
              "structure/value/negation.")
    return bad


def cstr(s):
    """C string literal, split so no line runs away; escapes are explicit.  Same shape as
    cpp/verify_frag.py's, so the two headers' SMARTS rows read back the same way."""
    e = s.replace("\\", "\\\\").replace('"', '\\"')
    if len(e) <= 90:
        return '"%s"' % e
    parts = [e[i:i + 90] for i in range(0, len(e), 90)]
    return "\n      " + "\n      ".join('"%s"' % p for p in parts)


TYPES = os.path.join(ROOT, "cpp", "frag_prog_types.h")


def cmd_types():
    """The opcode enum and the record layouts, in ONE header both programs include.

    They used to be declared inside `namespace frag_prog` in the generated fragment header, which
    was fine while there was one program.  With two, a matcher that took `frag_prog::Node*` could
    only be handed `qed_prog`'s table by casting between two distinct-but-identical structs --
    which is exactly the kind of thing that works until a compiler decides otherwise.  One set of
    types, generated from the same OPS list the compiler above indexes, so an opcode cannot mean
    one number here and another there.
    """
    o = []
    w = o.append
    w("// GENERATED by cpp/gen_frag_program.py -- do not edit.")
    w("//")
    w("// The query-program record layout and opcode numbering, shared by every program this")
    w("// generator emits (cpp/frag_program.h, cpp/qed_alert_program.h) and by the one evaluator")
    w("// that runs them, src/hume_core/frag_matcher.h.")
    w("//")
    w("// OPCODES ARE APPEND-ONLY.  The number is baked into every generated header; inserting one")
    w("// in the middle would silently reinterpret every node of an already-verified program.")
    w("//")
    w("//   rdkit  %s   (pin: 2025.9.2)" % rdkit.__version__)
    w("")
    w("#ifndef HUME_FRAG_PROG_TYPES_H")
    w("#define HUME_FRAG_PROG_TYPES_H")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace frag_prog_types {")
    w("")
    w("enum Op : uint8_t {")
    for i, n in enumerate(OPS):
        w("  OP_%s = %d," % (n.upper(), i))
    w("};")
    w("constexpr int N_OPS = %d;" % len(OPS))
    w("")
    w("struct Node { uint8_t op; uint8_t neg; int32_t val; int32_t lhs; int32_t rhs; };")
    w("struct QBond { int16_t u; int16_t v; int32_t root; };")
    w("struct Pattern { const char* label; int32_t a0; int16_t na; int32_t b0; int16_t nb; };")
    w("struct Named { const char* name; int32_t pattern; };")
    w("struct Spec { const char* name; const char* smarts; };")
    w("")
    w("// A whole compiled program, as one bindable value.  `fragmatch::Matcher` holds a pointer")
    w("// to this and to nothing else global, which is what lets one evaluator run both sets.")
    w("struct Program {")
    w("  const char*    name;")
    w("  const char*    spec_sha256;")
    w("  const Node*    nodes;")
    w("  const int32_t* aroots;")
    w("  const QBond*   qbonds;")
    w("  const Pattern* patterns;")
    w("  int            n_patterns;")
    w("  const Named*   named;")
    w("  int            n_named;")
    w("};")
    w("")
    w("}  // namespace frag_prog_types")
    w("")
    w("#endif")
    open(TYPES, "w").write("\n".join(o) + "\n")
    print("wrote %s  (%d opcodes)" % (TYPES, len(OPS)))


# Per-spec-set header preamble: the facts that are true of THAT pattern set and not the other.
PREAMBLE = {
    "frag": [
        "The %(nspec)d rdkit_core SMARTS patterns compiled to a flat query program: one node pool,",
        "one pattern table.  A RecursiveStructure node's `val` is an index into PATTERNS, so a",
        "recursive sub-query is just another pattern and the matcher needs no special case.",
        "",
        "The SMARTS come from $RDDATA/FragmentDescriptors.csv and cpp/verify_frag.py's NON_CSV",
        "table; none is typed here.  Every one is counted as",
        "len(GetSubstructMatches(patt, uniquify=True)) -- the number of distinct ATOM SETS.",
        "Measured on cpp/hard.smi: the largest raw (uniquify=False) match count over every pattern",
        "and every molecule is 180, against RDKit's maxMatches=1000 default, so truncation NEVER",
        "fires here.  That matters more than it looks -- truncation happens BEFORE uniquification,",
        "so if it ever did fire the count would silently become dependent on which embeddings the",
        "search happened to find first.",
    ],
    "qed": [
        "rdkit.Chem.QED.StructuralAlertSmarts -- all %(nspec)d of them -- compiled to a flat query",
        "program in the same representation as cpp/frag_program.h and run by the same evaluator.",
        "This is the last input `qed` was missing: seven of QED's eight properties are computed",
        "exactly in src/hume_core/constit.h and the eighth, ALERTS, is the count of these patterns",
        "that match.",
        "",
        "COUNTED AS A BOOLEAN PER PATTERN, NOT AS A MATCH COUNT.  rdkit/Chem/QED.py:",
        "    ALERTS=sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert))",
        "so the C++ calls hasMatch() and stops at the first embedding.  A pattern that matched a",
        "molecule 40 times contributes 1, exactly as one that matched it once.  That also means",
        "RDKit's maxMatches truncation cannot reach this number at all, unlike the fragment",
        "counts above.",
        "",
        "THE MOLECULE IS `Chem.RemoveHs(mol)`: QED.properties() calls it before anything else, and",
        "that is the graph src/hume/_extract.py already puts at the boundary.",
        "",
        "The names are INDICES because the alerts have no names in RDKit -- StructuralAlertSmarts",
        "is a bare list.  The index is their only stable identifier, and SPEC below carries the",
        "SMARTS text so the drift guard can name a row that moved rather than only a hash that did.",
    ],
}


def build_program(specs):
    prog = Program()
    tops = [(n, prog.compile(s, n)) for n, s in specs]
    pool, aroots, bl, pat_meta = [], [], [], []

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
    return tops, pool, aroots, bl, pat_meta, len(prog.patterns)


def cmd_program():
    VF.require_pin()
    cmd_types()
    for which in _which():
        specfn, out, ns, guard, desc = SPEC_SETS[which]
        specs = specfn()
        tops, pool, aroots, bl, pat_meta, npat = build_program(specs)
        sh = spec_hash(specs)

        o = []
        w = o.append
        w("// GENERATED by cpp/gen_frag_program.py `program %s` -- do not edit." % which)
        w("//")
        for line in PREAMBLE[which]:
            w(("// " + line % {"nspec": len(specs)}).rstrip())
        w("//")
        w("// PROVEN, NOT READ.  cpp/gen_frag_program.py `validate` re-renders every node of every")
        w("// pattern in RDKit's own DescribeQuery() format and compares it byte-for-byte with what")
        w("// RDKit prints -- %d top-level patterns and %d recursive sub-queries, per atom and per"
          % (len(specs), npat - len(specs)))
        w("// bond, structure, value and negation flag.")
        w("//")
        w("// SIX THINGS THE TEXT DOES NOT SAY, all of them decided by that comparison:")
        w("//   1. AtomAnd/AtomOr are LEFT-ASSOCIATIVE BINARY, not n-ary.")
        w("//   2. Negation is a flag on the node (`!= val`), never a wrapper node.")
        w("//   3. AtomInNRings -1 is a SENTINEL for `[R]` == in at least one ring, not a count --")
        w("//      and AtomInRing (`r`) is a DIFFERENT primitive again, a plain boolean.")
        w("//   4. AtomType z is ALIPHATIC element z, AtomType 1000+z AROMATIC, AtomAtomicNum z")
        w("//      neither -- it says nothing about aromaticity.  Only the organic subset")
        w("//      {B,C,N,O,P,S,F,Cl,Br,I} spells as AtomType at all: `[Si]` is AtomAtomicNum 14")
        w("//      while `[si]` is AtomType 1014.")
        w("//   5. The default bond written between two atoms is SingleOrAromaticBond, which is a")
        w("//      different query from `-` (BondOrder 1) and from `:` (BondOrder 12).")
        w("//   6. An isotope binds to its symbol as ONE primitive: `[15N]` is")
        w("//      AtomAnd(AtomType 7, AtomIsotope 15), so `[15NH2+]` nests ((N & iso) & H) & +.")
        w("//")
        w("// THE DRIFT GUARD IS ON THE SPEC, NOT ON A FILE.  SPEC_SHA256 hashes the (name, SMARTS)")
        w("// pairs below.  `cpp/gen_frag_program.py check %s` recomputes it from the loaded RDKit"
          % which)
        w("// and also diffs the SMARTS row by row, so a moved pattern is NAMED and not merely")
        w("// signalled.  House rule 6: a hash of the source file would move on a docstring edit")
        w("// and cry wolf, and the fragment spec really did move between rdkit 2025.09.2 and")
        w("// 2026.03.5.")
        w("//")
        w("//   rdkit  %s   (pin: 2025.9.2)" % rdkit.__version__)
        w("//   SPEC_SHA256 = %s" % sh)
        w("")
        w("#ifndef %s" % guard)
        w("#define %s" % guard)
        w("")
        w("#include <cstdint>")
        w("")
        w('#include "frag_prog_types.h"')
        w("")
        w("namespace %s {" % ns)
        w("")
        w("using frag_prog_types::Node;")
        w("using frag_prog_types::QBond;")
        w("using frag_prog_types::Pattern;")
        w("using frag_prog_types::Named;")
        w("using frag_prog_types::Spec;")
        w("")
        w('constexpr const char SPEC_SHA256[] = "%s";' % sh)
        w("// Kept under its old name too: cpp/frag_program.h shipped this as PROGRAM_SHA256 over")
        w("// the same bytes, and anything quoting the old identifier should keep resolving.")
        w("constexpr const char* PROGRAM_SHA256 = SPEC_SHA256;")
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
        w("constexpr int N_NAMED = %d;" % len(tops))
        w("constexpr Named NAMED[N_NAMED] = {")
        for n, p in tops:
            w('  {"%s",%d},' % (n, p))
        w("};")
        w("")
        w("// The SPECIFICATION SPEC_SHA256 hashes, carried so the drift guard can name the row")
        w("// that moved.  Read back by `check`; not used by the matcher.")
        w("constexpr Spec SPEC[N_NAMED] = {")
        for n, s in specs:
            w('  { "%s", %s },' % (n, cstr(s)))
        w("};")
        w("")
        w("// The whole program as one bindable value; see frag_prog_types.h.")
        w("constexpr frag_prog_types::Program PROGRAM = {")
        w('  "%s", SPEC_SHA256, NODES, AROOTS, QBONDS, PATTERNS, N_PATTERNS, NAMED, N_NAMED' % ns)
        w("};")
        w("")
        w("}  // namespace %s" % ns)
        w("")
        w("#endif")
        open(out, "w").write("\n".join(o) + "\n")
        print("wrote %s" % out)
        print("  patterns %d (%d named + %d recursive sub-queries)"
              % (len(pat_meta), len(tops), len(pat_meta) - len(tops)))
        print("  nodes %d, atom roots %d, bonds %d" % (len(pool), len(aroots), len(bl)))
        print("  SPEC_SHA256 %s" % sh)
    print("RESOLVED rdkit %s" % rdkit.__version__)


# The header's own SPEC table, read back so a corrupted row is NAMED rather than only hashed.
_LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')


def header_spec(txt):
    body = txt.split("constexpr Spec SPEC[N_NAMED] = {", 1)[1].split("\n};", 1)[0]
    out = []
    for ent in re.finditer(r"\{(.*?)\},\s*(?=\{|$)", body, re.S):
        lits = [m.group(1) for m in _LIT.finditer(ent.group(1))]
        if len(lits) < 2:
            continue
        out.append((lits[0],
                    "".join(lits[1:]).replace('\\"', '"').replace("\\\\", "\\")))
    return out


def header_arrays(txt):
    """The generated data arrays, read back out of the header as plain integer tuples.

    WHY THE HASH ALONE IS NOT ENOUGH, and this is the gap cpp/frag_tables.h's guard still has:
    SPEC_SHA256 covers the (name, SMARTS) pairs, which is the right thing to hash for DRIFT -- it
    answers "did RDKit's specification move?".  It says nothing about whether the COMPILED
    program in the same file is the compilation of those pairs.  A hand-edited node would pass a
    spec hash forever.  So `check` also recompiles the live spec and compares NODES / AROOTS /
    QBONDS / PATTERNS element by element with what the header ships.
    """
    def ints(section, per):
        body = txt.split(section, 1)[1].split("\n};", 1)[0]
        return [tuple(int(x) for x in mm.group(1).split(",")[:per])
                for mm in re.finditer(r"\{([-0-9,]+)\}", body)]
    nodes = ints("constexpr Node NODES[N_NODES] = {", 5)
    qbonds = ints("constexpr QBond QBONDS[N_QBONDS] = {", 3)
    aroots = [int(x) for x in
              txt.split("AROOTS[N_AROOTS] = {", 1)[1].split("}", 1)[0].split(",") if x.strip()]
    pbody = txt.split("constexpr Pattern PATTERNS[N_PATTERNS] = {", 1)[1].split("\n};", 1)[0]
    pats = [tuple(int(x) for x in mm.group(1).split(","))
            for mm in re.finditer(r'\{"(?:[^"\\]|\\.)*",([-0-9,]+)\}', pbody)]
    nbody = txt.split("constexpr Named NAMED[N_NAMED] = {", 1)[1].split("\n};", 1)[0]
    named = [(mm.group(1), int(mm.group(2)))
             for mm in re.finditer(r'\{"((?:[^"\\]|\\.)*)",([-0-9]+)\}', nbody)]
    return nodes, aroots, qbonds, pats, named


def cmd_check():
    """THE DRIFT GUARD.  Recompute the spec hash from the LOADED RDKit and compare, then diff the
    SMARTS row by row AND the compiled program array by array.  Exits non-zero on any
    disagreement."""
    VF.require_pin()
    print("RESOLVED rdkit %s" % rdkit.__version__)
    bad = 0
    for which in _which():
        specfn, out, ns, _, _ = SPEC_SETS[which]
        specs = specfn()
        live = spec_hash(specs)
        if not os.path.exists(out):
            print("%s: MISSING %s -- run `program %s` first" % (which, out, which))
            bad += 1
            continue
        txt = open(out).read()
        m = re.search(r'SPEC_SHA256\s*\[\]\s*=\s*"([0-9a-f]{64})"', txt)
        if not m:
            print("%s: %s carries no SPEC_SHA256" % (which, out))
            bad += 1
            continue
        n_bad = 0
        if m.group(1) != live:
            print("%s: SPEC DRIFT\n  header : %s\n  live   : %s" % (which, m.group(1), live))
            n_bad += 1
        got = dict(header_spec(txt))
        for n, s in specs:
            if n not in got:
                print("  MISSING from header: %s" % n)
                n_bad += 1
            elif got[n] != s:
                print("  SMARTS DIFFERS for %s\n    header: %s\n    live  : %s" % (n, got[n], s))
                n_bad += 1
        extra = sorted(set(got) - {n for n, _ in specs})
        if extra:
            print("  IN THE HEADER AND NOT IN THE LOADED RDKIT: %s" % extra)
            n_bad += len(extra)
        # ... and the compiled program really is the compilation of those SMARTS.
        tops, pool, aroots, bl, pat_meta, _ = build_program(specs)
        want = ([(OP_[0], OP_[1], OP_[2], OP_[3], OP_[4]) for OP_ in pool],
                aroots,
                [(u, v, r) for u, v, r in bl],
                [(a0, na, b0, nb) for _, _, a0, na, b0, nb in pat_meta],
                tops)
        try:
            have = header_arrays(txt)
        except Exception as e:                                        # noqa: BLE001
            print("  UNREADABLE program arrays in %s: %s" % (out, e))
            have = None
            n_bad += 1
        if have is not None:
            for label, h, wv in zip(("NODES", "AROOTS", "QBONDS", "PATTERNS", "NAMED"),
                                    have, want):
                if list(h) != list(wv):
                    where = next((k for k in range(min(len(h), len(wv))) if h[k] != wv[k]), None)
                    print("  COMPILED %s DIFFERS: header %d entries, live %d%s"
                          % (label, len(h), len(wv),
                             "" if where is None else
                             "; first at %d: header %r live %r" % (where, h[where], wv[where])))
                    n_bad += 1
        if n_bad:
            print("%s: FAIL -- %d disagreement(s).  Regenerate with `program %s` and RE-VERIFY; "
                  "do not paste the new hash in." % (which, n_bad, which))
        else:
            print("%-5s SPEC OK  rows=%-4d sha256=%s" % (which, len(specs), live))
        bad += n_bad
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    {"validate": cmd_validate, "program": cmd_program, "check": cmd_check,
     "types": cmd_types}[cmd]()
