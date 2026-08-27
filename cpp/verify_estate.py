"""Generate the E-state atom-type table for C++, and prove the C++ typer reproduces it EXACTLY.

RUN IT PINNED. Every exactness claim in this repository is pinned to rdkit 2025.09.2. A floating
`uv run --with rdkit` resolved to 2026.03.5 on 2026-08-27, and that is not a harmless difference
of convenience: `tables` GENERATES the specification from whatever RDKit is loaded and `verify`
then compares the C++ against THAT SAME RDKit, so both sides move together and the check goes
blind. Pin the oracle or the harness is comparing a thing to itself.

    UV=(uv run --with "rdkit==2025.9.2")
    "${UV[@]}" python cpp/verify_estate.py tables         # regenerate cpp/estate_tables.h
    "${UV[@]}" python cpp/verify_estate.py stress         # build the probe corpus
    "${UV[@]}" python cpp/verify_estate.py mols FILE N    # dump a corpus
    "${UV[@]}" python cpp/verify_estate.py verify         # build the C++ and check per ATOM

`columns` and `time` additionally need mordred, and mordred 1.2.0 REQUIRES numpy 1.x -- so it
cannot share an environment with the project's numpy 2.4.6 pin, and asking for both silently
resolves mordred down to 0.6.0, a different oracle again. Give it its own environment and let
numpy fall where mordred needs it:

    uv run --with "mordred==1.2.0" --with "rdkit==2025.9.2" python cpp/verify_estate.py columns
    uv run --with "mordred==1.2.0" --with "rdkit==2025.9.2" python cpp/verify_estate.py time

`all` runs everything, and REFUSES to start unless rdkit is the pinned version.

WHERE THE RULES ACTUALLY LIVE.  mordred/EState.py does NOT contain a typer.  Its `EStateCache`
is four lines and both halves are RDKit's:

    return EState.TypeAtoms(self.mol), EState.EStateIndices(self.mol)

so the 79 patterns are `rdkit.Chem.EState.AtomTypes._rawD`, a list of (name, SMARTS) taken from
Hall & Kier, JCICS 35 1039-1045 (1995) Table 1, with three rows RDKit marks `# mod`.
`AtomTypes.TypeAtoms` runs each pattern with `GetSubstructMatches(patt, uniquify=0)` and appends
the pattern NAME to the list for `match[0]`.  An atom therefore ends up with a TUPLE of names --
possibly empty, possibly longer than one -- and the tuples are what mordred aggregates:

    N<t> = reduce(add, types).count(t)          -- a pure count of atoms whose tuple holds t
    S<t> = sum(index[i] for i where t in types[i])
    MAX/MIN<t> = max/min of the same, NaN when the type is absent

so **the 29 `N*` columns need the TYPER ONLY and never touch the E-state index**, while the 21
`S*` columns pair the typer with the index `estate_from()` in src/hume_core/hume_blocks.h already
computes.  That claim is re-checked mechanically by `columns()` below, not asserted.

WHAT THE TABLE GENERATOR DOES, AND WHY IT IS NOT A PASTED TABLE.  Retyping 79 SMARTS by eye is
how this repository has lost rounds before.  `gen_tables()` instead hands each pattern to RDKit's
own SMARTS parser and decodes the resulting QUERY TREE via `DescribeQuery()`, converting it to
disjunctive normal form over the four leaf kinds that appear (`AtomType`, `AtomAtomicNum`,
`AtomExplicitDegree`, `AtomHCount`) plus the bond queries (`BondOrder n`, `BondNull`, `BondOr`).
The decode is total: `alt_from_conj()` and `decode_bond_leaf()` raise on any primitive they do
not account for, so a future RDKit that adds one to any pattern fails here loudly rather than
being silently ignored.
The header records the resolved rdkit/numpy/mordred versions, the pin it was generated under, and
TWO hashes: `spec`, the sha256 of the 79 (name, SMARTS) pairs, and `file`, the sha256 of
AtomTypes.py. Only `spec` answers "did the patterns move". The file hash is informational because
it moves on edits that mean nothing -- rdkit 2025.09.2 -> 2026.03.5 changes it by deleting a
`# $Id$` RCS keyword line and changes not one pattern, which was checked by diffing every decoded
parse tree between the two versions, not by trusting the hash.

FOUR THINGS THE SMARTS MEAN THAT A CAREFUL READER STILL GETS WRONG.  Each is decoded from the
parse tree, not from the pattern text, and each is exercised by the stress corpus:

1.  `[SeD2H0]`, `[SiD1H3]`, `[GeD1H3]`, `[AsD1H2]`, `[SnD1H3]`, `[PbD1H3]`, `[LiD1]`, `[BeD2]`
    parse to **element-number** queries (`AtomAtomicNum 34`), which carry NO aromaticity
    constraint, while `[CD1H3]`, `[ND1H2]`, `[OD1H]`, `[SD1H0]`, `[BD2H]`, `[PD1H2]`, `[FD1]`,
    `[ClD1]`, `[BrD1]`, `[ID1]` parse to **aliphatic** AtomType queries.  So `aaSe` genuinely
    fires on selenophene's aromatic `[se]` (verified: `c1cc[se]c1` -> `aaSe`), and the seemingly
    parallel `aaS` reaches thiophene only through its explicit `s` alternative.

2.  THE MISSING SEMICOLON.  `aaCH`/`aasC`/`aaaC` are written `[C,c;D2H0]` -- the `;` binds the
    degree and H count to BOTH alternatives.  `aaNH`/`aaN`/`aasN`/`aaO`/`aaS` are written
    `[N,nD2H0]`, which parses as `N` OR (`n` AND `D2` AND `H0`): the ALIPHATIC alternative is
    left completely unconstrained.  An aliphatic N carrying two AROMATIC-typed bonds would
    therefore be typed `aaN`, `aaNH` and possibly `aasN` all at once.  Reproduced exactly rather
    than "cleaned up"; no molecule in either corpus reaches it, which is why it is stated here.

3.  `:` PARSES AS A BOND-ORDER QUERY, NOT AN AROMATIC FLAG.  `DescribeQuery()` reports
    `BondOrder 12 = val`, i.e. `getBondType() == AROMATIC`.  That is a different question from
    `getIsAromatic()`, which is what cpp/crippen.cpp's bond code answers, and the two disagree on
    a SINGLE-typed bond carrying the aromatic flag.  The dump below therefore encodes the bond
    TYPE only, as four mutually exclusive bits, and anything else (DATIVE -- 22 of them in the
    first 20k of cpp/hard.smi) gets code 0, which only `~` matches.

4.  THE THREE `# mod` ROWS NEED NEIGHBOUR IDENTITY.  `ddsN` is
    `[ND3H0](~[OD1H0])(~[OD1H0])-,:*` and `ddssS` is `[SD4H0](~[OD1H0])(~[OD1H0])(-*)-*`: two of
    the branches are not `*` but a TERMINAL, H-FREE OXYGEN reached by ANY bond.  The bond order
    to those oxygens is deliberately unconstrained, so `CN(=O)=O` and `C[N+](=O)[O-]` both type
    the nitrogen `ddsN` even though one has two double bonds and the other one double and one
    single.  `aasN`'s modification is different -- it is the `-,:` last bond, which lets a
    bridgehead aromatic nitrogen (indolizine) match, and it needs no neighbour identity.  So
    exactly TWO of the 79 patterns look past the central atom's own bonds.

SUBSTRUCTURE MATCHING IS A PERFECT MATCHING, NOT A COUNT.  Every pattern's branch count equals
its `D`, and distinct query atoms must map to distinct target atoms, so "does atom i match" is
"is there a system of distinct representatives pairing the D branch specs with the D bonds".
Counting bonds per mask independently is wrong the moment two branch specs overlap, which is
exactly `ddssS` on `[O-][S+](=O)(=O)[O-]`-shaped sulfur where four neighbours are all terminal
oxygens.  `decode_pattern()` asserts the branch-count == D premise on every row.

AN ATOM CAN HAVE MORE THAN ONE TYPE, AND TWO IN cpp/hard.smi DO.  `[O-]N([O-])N=...` is a
nitrogen with three SINGLE bonds, two of them to terminal H-free oxygens, so it satisfies `sssN`
(three single bonds) AND `ddsN` (`~[OD1H0]` twice, and `~` does not care that the bonds are
single).  mordred counts it under BOTH `NsssN` and `NddsN`.  That is why the reference is the
whole type TUPLE in pattern order and not one winning row -- an atom typer that returns a single
best match is wrong here, and wrong in a way a column total would hide.

CACHING.  mordred memoises per molecule and RDKit caches ring/aromaticity perception, so timing
or verifying a second pass over the same `Mol` objects measures a cache hit.  Every mordred call
below is given a FRESHLY PARSED molecule.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import EState as _EState
from rdkit.Chem.EState import AtomTypes

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Bond TYPE as four mutually exclusive bits; see note 3 in the module docstring. Code 0 means
# "some other bond type" (DATIVE, unspecified, ...) and is matched only by `~`.
BT = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
      Chem.BondType.TRIPLE: 4, Chem.BondType.AROMATIC: 8}
# RDKit BondType enum values as they appear in DescribeQuery's `BondOrder n = val`.
BOND_ORDER_VALUE = {1: 1, 2: 2, 3: 4, 12: 8}

AROM_ALI, AROM_ARO, AROM_ANY = 0, 1, 2


# ---------------------------------------------------------------------------------------------
# decoding RDKit's own parse tree
# ---------------------------------------------------------------------------------------------
class Node:
    __slots__ = ("text", "kids")

    def __init__(self, text):
        self.text, self.kids = text, []


def parse_describe(s: str) -> Node:
    """`DescribeQuery()` emits a 2-space-indented tree; turn it back into one."""
    root, stack = None, []
    for raw in s.split("\n"):
        if not raw.strip():
            continue
        depth = (len(raw) - len(raw.lstrip(" "))) // 2
        n = Node(raw.strip())
        if depth == 0:
            root, stack = n, [n]
        else:
            del stack[depth:]
            stack[-1].kids.append(n)
            stack.append(n)
    return root


def atom_dnf(n: Node) -> list[list[str]]:
    """-> list of conjunctions, each a list of leaf strings. AND distributes over OR."""
    head = n.text.split()[0]
    if head == "AtomOr":
        out = []
        for k in n.kids:
            out += atom_dnf(k)
        return out
    if head == "AtomAnd":
        out = [[]]
        for k in n.kids:
            out = [a + b for a in out for b in atom_dnf(k)]
        return out
    return [[n.text]]


def alt_from_conj(conj: list[str]) -> tuple[int, int, int, int]:
    """One conjunction -> (z, arom, d, h) with -1 for 'unconstrained'."""
    z, arom, d, h = -1, AROM_ANY, -1, -1
    for leaf in conj:
        f = leaf.split()
        if f[0] == "AtomType":                       # atomicNum + 1000 * isAromatic
            v = int(f[1])
            assert z in (-1, v % 1000), f"two element constraints in {conj}"
            z, arom = v % 1000, AROM_ARO if v >= 1000 else AROM_ALI
        elif f[0] == "AtomAtomicNum":                # element only, aromaticity-agnostic
            z = int(f[1])
        elif f[0] == "AtomExplicitDegree":
            d = int(f[1])
        elif f[0] == "AtomHCount":
            h = int(f[1])
        elif f[0] == "AtomNull":
            pass
        else:
            raise AssertionError(f"unhandled atom primitive {leaf!r}")
        assert leaf.endswith("= val") or f[0] == "AtomNull", f"negated/ranged primitive {leaf!r}"
    return z, arom, d, h


def decode_atom(a) -> list[tuple[int, int, int, int]]:
    alts = [alt_from_conj(c) for c in atom_dnf(parse_describe(a.DescribeQuery()))]
    # `[C,c;D2H]` decodes to two alts differing only in aromaticity; fold them back into one.
    merged = []
    for alt in alts:
        for i, m in enumerate(merged):
            if m[0] == alt[0] and m[2:] == alt[2:] and {m[1], alt[1]} == {AROM_ALI, AROM_ARO}:
                merged[i] = (m[0], AROM_ANY, m[2], m[3])
                break
        else:
            merged.append(alt)
    assert 1 <= len(merged) <= 2, f"{len(merged)} alternatives, table holds 2"
    return merged


def decode_bond(b) -> int:
    """-> mask of allowed bond TYPE bits; 0 means `~` (any bond at all)."""
    n = parse_describe(b.DescribeQuery())
    if n.text == "BondNull":
        return 0
    if n.text == "BondOr":
        m = 0
        for k in n.kids:
            m |= decode_bond_leaf(k.text)
        return m
    return decode_bond_leaf(n.text)


def decode_bond_leaf(t: str) -> int:
    f = t.split()
    assert f[0] == "BondOrder" and t.endswith("= val"), f"unhandled bond primitive {t!r}"
    return BOND_ORDER_VALUE[int(f[1])]


def numpy_version() -> str:
    try:
        import numpy
        return numpy.__version__
    except Exception:                                                    # pragma: no cover
        return "?"


def spec_sha(rawD) -> str:
    """sha256 of the SPECIFICATION -- the (name, SMARTS) pairs -- not of the file holding it.

    A whole-file hash is the wrong guard: rdkit 2025.09.2 -> 2026.03.5 changes AtomTypes.py's
    hash by deleting a `# $Id$` RCS keyword line and changes not one pattern. This hash moves
    only when a pattern does, which is the question the C++ selfCheck() is actually asking.
    """
    blob = "\n".join(f"{n}\t{s}" for n, s in rawD).encode()
    return hashlib.sha256(blob).hexdigest()


NBR_QUERIES: list[tuple[int, int, int, int]] = []      # index 0 is always `*`


def nbr_index(alts) -> int:
    assert len(alts) == 1, "branch atom with alternatives is not in the table shape"
    a = alts[0]
    if a not in NBR_QUERIES:
        NBR_QUERIES.append(a)
    return NBR_QUERIES.index(a)


def decode_pattern(sma: str):
    p = Chem.MolFromSmarts(sma)
    assert p is not None, sma
    central = decode_atom(p.GetAtomWithIdx(0))
    branches = []
    for b in p.GetBonds():
        assert b.GetBeginAtomIdx() == 0, f"{sma}: branch not rooted at atom 0"
        branches.append((decode_bond(b), nbr_index(decode_atom(p.GetAtomWithIdx(b.GetEndAtomIdx())))))
    assert p.GetNumAtoms() == len(branches) + 1, f"{sma}: not a star"
    # Every pattern's branch count equals its degree constraint, which is what makes "does atom i
    # match" a perfect matching of D branches onto D bonds rather than an open-ended search.
    for z, arom, d, h in central:
        assert d in (-1, len(branches)), f"{sma}: D={d} but {len(branches)} branches"
    return central, branches


def gen_tables() -> None:
    src = Path(AtomTypes.__file__.replace(".pyc", ".py"))
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    mv = _mordred_version()
    NBR_QUERIES.clear()
    NBR_QUERIES.append((-1, AROM_ANY, -1, -1))                           # `*`
    rows = []
    for name, sma in AtomTypes._rawD:
        central, branches = decode_pattern(sma)
        rows.append((name, sma, central, branches))
    maxb = max(len(b) for _, _, _, b in rows)
    maxa = max(len(c) for _, _, c, _ in rows)

    def alt(a):
        return "{%3d, %d, %2d, %2d}" % a

    out = [
        "// GENERATED by cpp/verify_estate.py -- do not edit.",
        "//",
        "// The 79 E-state atom-type patterns of Hall & Kier JCICS 35 1039-1045 (1995) Table 1, as",
        "// RDKit ships them in rdkit/Chem/EState/AtomTypes.py (`_rawD`) and as mordred/EState.py",
        "// consumes them through `EState.TypeAtoms`. Each row is DECODED FROM RDKit'S OWN PARSE",
        "// TREE (`DescribeQuery()`), not retyped, so a pattern that changes upstream changes this",
        "// file visibly and trips src/hume_core/estate_typer.h's selfCheck() at load.",
        "//",
        "// PROVENANCE. `spec` is the sha256 of the 79 (name, SMARTS) pairs themselves, which is",
        "// what this table encodes; `file` is the sha256 of the whole AtomTypes.py, which is",
        "// INFORMATIONAL ONLY because it moves on changes that mean nothing. Between rdkit",
        "// 2025.09.2 and 2026.03.5 the file hash changes and the spec hash does not: the sole",
        "// diff is a deleted `# $Id$` RCS keyword line. Compare `spec` when asking whether the",
        "// patterns moved; `file` only tells you the shipped module was touched.",
        "//",
        "// Regenerate under THIS PIN, not a floating one -- every exactness claim in this repo",
        "// is pinned to rdkit 2025.09.2, and an unpinned `--with rdkit` silently becomes a",
        "// different oracle that the exactness check cannot see, because both sides move together:",
        "//     uv run --with 'rdkit==2025.9.2' python cpp/verify_estate.py tables",
        "//",
        f"//   rdkit   {rdkit.__version__}   (pin: 2025.9.2)",
        f"//   numpy   {numpy_version()}",
        f"//   mordred {mv}",
        f"//   spec    sha256(_rawD) = {spec_sha(AtomTypes._rawD)}",
        f"//   file    sha256(AtomTypes.py) = {sha}   (informational; moves on cosmetic edits)",
        "",
        "#ifndef HUME_ESTATE_TABLES_H",
        "#define HUME_ESTATE_TABLES_H",
        "",
        "#include <cstdint>",
        "",
        "namespace estate_tbl {",
        "",
        f"constexpr int N_ROWS = {len(rows)};",
        f"constexpr int MAX_BRANCH = {maxb};",
        f"constexpr int MAX_ALT = {maxa};",
        "",
        "// Bond TYPE bits, mutually exclusive. SMARTS `:` is a BondOrder==AROMATIC query, NOT a",
        "// getIsAromatic() query, so this is the bond TYPE and nothing else. Mask 0 means `~`.",
        "enum : uint8_t { BT_SINGLE = 1, BT_DOUBLE = 2, BT_TRIPLE = 4, BT_AROMATIC = 8 };",
        "enum : uint8_t { BM_ANY = 0 };",
        "",
        "// Aromaticity policy of an element primitive. `[C]` is ALI, `[c]` is ARO, and `[#34]`",
        "// -- which is what `[Se]` parses to -- is ANY.",
        "enum : uint8_t { AROM_ALI = 0, AROM_ARO = 1, AROM_ANY = 2 };",
        "",
        "// One conjunct of the central atom's bracket expression. d/h == -1 means unconstrained,",
        "// which is not decoration: `[N,nD2H0]` leaves its aliphatic alternative wide open.",
        "struct Alt { int16_t z; uint8_t arom; int8_t d; int8_t h; };",
        "struct Branch { uint8_t bond; uint8_t nq; };",
        "struct Row {",
        "  const char* name;",
        "  const char* smarts;",
        "  uint8_t nalt;",
        f"  Alt alt[{maxa}];",
        "  uint8_t nbranch;",
        f"  Branch br[{maxb}];",
        "};",
        "",
        "// Branch atom queries. Index 0 is `*`; index 1 is the terminal H-free oxygen that ddsN",
        "// and ddssS need, and is the ONLY place neighbour identity enters the typer.",
        f"constexpr int N_NBRQ = {len(NBR_QUERIES)};",
        "constexpr Alt NBRQ[N_NBRQ] = {",
    ]
    for a in NBR_QUERIES:
        out.append(f"  {alt(a)},")
    out += ["};", "", "constexpr Row ROWS[N_ROWS] = {"]
    for name, sma, central, branches in rows:
        alts = ", ".join(alt(a) for a in central) + ", {0,0,0,0}" * (maxa - len(central))
        brs = ", ".join("{%d, %d}" % b for b in branches) + ", {0,0}" * (maxb - len(branches))
        out.append(f'  {{ "{name}", "{sma}", {len(central)}, {{{alts}}}, '
                   f"{len(branches)}, {{{brs}}} }},")
    out += ["};", "", "}  // namespace estate_tbl", "", "#endif", ""]
    p = HERE / "estate_tables.h"
    p.write_text("\n".join(out))
    print(f"wrote {p}  |  {len(rows)} patterns, {len(NBR_QUERIES)} branch queries, "
          f"rdkit {rdkit.__version__}, mordred {mv}")
    print(f"  sha256(AtomTypes.py) = {sha}")
    ncent = sum(1 for _, _, c, _ in rows if len(c) > 1)
    nnbr = sum(1 for _, _, _, b in rows if any(q for _, q in b))
    print(f"  {ncent} patterns have a two-alternative central atom, "
          f"{nnbr} need neighbour identity")


# ---------------------------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------------------------
# cpp/hard.smi is 100k adversarial molecules and still leaves 25 of the 79 patterns cold: every
# Be/Ge/Sn/Pb row, most Si/As/Se rows, sLi, ssssB, ssPH. A pattern that never fires is UNTESTED,
# not passing, so these exist to make the coverage report honest. Each entry is a molecule whose
# named atom must reach the named pattern; `stress()` checks RDKit agrees before the C++ sees it.
STRESS = [
    ("[Li]C", "sLi"), ("CC[Li]", "sLi"), ("[Li]c1ccccc1", "sLi"),
    ("C[Be]C", "ssBe"), ("[Be](C)C", "ssBe"), ("Cl[Be]Cl", "ssBe"),
    ("C[Be-2](C)(C)C", "ssssBe"), ("[Be-2](Cl)(Cl)(Cl)Cl", "ssssBe"),
    ("B(C)(C)C", "sssB"), ("C[BH]C", "ssBH"), ("F[BH]F", "ssBH"), ("C[BH]O", "ssBH"),
    ("[B-](C)(C)(C)C", "ssssB"), ("[B-](F)(F)(F)F", "ssssB"),
    ("[SiH3]C", "sSiH3"), ("C[SiH2]C", "ssSiH2"), ("C[SiH](C)C", "sssSiH"),
    ("C[Si](C)(C)C", "ssssSi"), ("O[Si](O)(O)O", "ssssSi"),
    ("[PH2]C", "sPH2"), ("C[PH]C", "ssPH"), ("CP(C)C", "sssP"),
    ("CP(C)(C)=O", "dsssP"), ("O=P(O)(O)O", "dsssP"), ("FP(F)(F)(F)F", "sssssP"),
    ("[GeH3]C", "sGeH3"), ("C[GeH2]C", "ssGeH2"), ("C[GeH](C)C", "sssGeH"),
    ("C[Ge](C)(C)C", "ssssGe"), ("Cl[Ge](Cl)(Cl)Cl", "ssssGe"),
    ("[AsH2]C", "sAsH2"), ("C[AsH]C", "ssAsH"), ("C[As](C)C", "sssAs"),
    ("C[As](C)(C)=O", "sssdAs"), ("O=[As](O)(O)O", "sssdAs"),
    ("F[As](F)(F)(F)F", "sssssAs"), ("C[As](C)(C)(C)C", "sssssAs"),
    ("[SeH]C", "sSeH"), ("C=[Se]", "dSe"), ("O=[Se]", "dSe"), ("C[Se]C", "ssSe"),
    ("c1cc[se]c1", "aaSe"), ("c1ccc2[se]ccc2c1", "aaSe"),
    ("C[Se](C)=O", "dssSe"), ("C[Se](C)(=O)=O", "ddssSe"), ("O=[Se](=O)(O)O", "ddssSe"),
    ("[SnH3]C", "sSnH3"), ("C[SnH2]C", "ssSnH2"), ("C[SnH](C)C", "sssSnH"),
    ("C[Sn](C)(C)C", "ssssSn"), ("CCCC[Sn](CCCC)(CCCC)CCCC", "ssssSn"),
    ("[PbH3]C", "sPbH3"), ("C[PbH2]C", "ssPbH2"), ("C[PbH](C)C", "sssPbH"),
    ("C[Pb](C)(C)C", "ssssPb"), ("CC[Pb](CC)(CC)CC", "ssssPb"),
    # -- the three `# mod` rows, in both their charged and neutral forms -----------------------
    ("C[N+](=O)[O-]", "ddsN"), ("CN(=O)=O", "ddsN"), ("O=[N+]([O-])c1ccccc1", "ddsN"),
    ("CS(=O)(=O)C", "ddssS"), ("NS(=O)(=O)c1ccccc1", "ddssS"),
    ("[O-][S+](=O)([O-])C", "ddssS"), ("OS(=O)(=O)O", "ddssS"),
    ("Cn1cccc1", "aasN"), ("c1ccn2cccc2c1", "aasN"), ("c1ccc2[nH]ccc2c1", "aaNH"),
    # -- rare-but-real rows that hard.smi reaches thinly ---------------------------------------
    ("[NH4+]", None), ("C[NH3+]", "sNH3"), ("C[NH2+]C", "ssNH2"), ("C=N", "dNH"),
    ("C[N+](C)(C)C", "ssssN"), ("C#N", "tN"), ("C=C=C", "ddC"), ("CC#CC", "tsC"),
    ("c1ccc2ccccc2c1", "aaaC"), ("C=S", "dS"), ("CS(C)=O", "dssS"), ("CS", "sSH"),
    ("c1ccoc1", "aaO"), ("c1ccsc1", "aaS"), ("c1ccncc1", "aaN"),
    # -- things that must type to NOTHING, and things that stress the graph --------------------
    ("[Xe]", None), ("[U]", None), ("[Te]", None), ("c1cc[te]c1", None), ("[H][H]", None),
    ("[2H]C([2H])([2H])O", None), ("[13CH4]", None), ("[C]", None), ("[CH3]", None),
    ("[O-][n+]1ccccc1", None), ("[Na+].[Cl-]", None), ("O=[N+]([O-])[O-]", None),
    ("C1=CC2=CC=C3C=CC4=CC=C1N2[Fe]34", None), ("[Fe+2].[C-]#N", None),
    ("O=S=O", None), ("[O-]S(=O)(=O)[O-]", None), ("F[Se](F)(F)F", None),
]


def bond_code(b) -> int:
    return BT.get(b.GetBondType(), 0)


def dump_mols(smis: list[str], path: Path, tag: str) -> list[str]:
    """`path`: n_mols / per mol: `nat nbond` + atom lines + bond lines + a type line per atom.

    Atom line: Z  degree  nH_prop  aromatic
      `nH_prop` is GetTotalNumHs(False). SMARTS `H` is that PLUS neighbouring H ATOMS, which the
      C++ derives from the graph -- the same convention cpp/export_crippen.py established, and
      the same one src/hume_core/hume_blocks.h's `nH` array already carries.
    Bond line: u  v  typecode   (see BT above; 0 == some other type, e.g. DATIVE)
    Type line: k  i0 i1 ...     RDKit's OWN answer, as INDICES into _rawD, in pattern order.

    Indices, not names: comparing the index is comparing the decision the algorithm made, and it
    keeps the per-pattern coverage histogram on the reference side honest.
    """
    AtomTypes.BuildPatts()
    idx_of = {n: i for i, (n, _) in enumerate(AtomTypes._rawD)}
    out, kept = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        T = AtomTypes.TypeAtoms(m)                                    # fresh mol: no cache reuse
        atoms = [f"{a.GetAtomicNum()} {a.GetDegree()} {a.GetTotalNumHs()} {int(a.GetIsAromatic())}"
                 for a in m.GetAtoms()]
        bonds = [f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} {bond_code(b)}" for b in m.GetBonds()]
        types = [f"{len(t)} " + " ".join(str(idx_of[x]) for x in t) for t in T]
        out.append("\n".join([f"{m.GetNumAtoms()} {m.GetNumBonds()}"] + atoms + bonds + types))
        kept.append(s)
    path.write_text(f"{len(kept)}\n" + "\n".join(out) + "\n")
    nat = sum(int(o.split("\n", 1)[0].split()[0]) for o in out)
    print(f"wrote {path}  |  {len(kept):,} molecules, {nat:,} atoms ({tag})")
    return kept


def stress() -> None:
    """Build the probe corpus, and first check RDKit itself agrees with what each probe claims."""
    AtomTypes.BuildPatts()
    bad = []
    for smi, want in STRESS:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad.append(f"  {smi!r} does not parse")
            continue
        got = {x for t in AtomTypes.TypeAtoms(m) for x in t}
        if want is not None and want not in got:
            bad.append(f"  {smi!r} claims {want} but RDKit gives {sorted(got)}")
    if bad:
        raise SystemExit("stress corpus is wrong about RDKit:\n" + "\n".join(bad))
    dump_mols([s for s, _ in STRESS], HERE / "estate_stress.txt", "stress probes")


# ---------------------------------------------------------------------------------------------
# the 50 columns
# ---------------------------------------------------------------------------------------------
def hume_estate_columns() -> list[str]:
    """The 50 EState columns HUME's dedupe actually keeps, from blocks.py -- not a pasted list."""
    sys.path.insert(0, str(ROOT))
    import blocks
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    fam = {str(d): type(d).__module__.split(".")[-1] for d in calc.descriptors}
    return sorted(n for s, n, f in blocks.split(fam)["predict"] if f == "EState")


def columns(n_mol: int = 2000) -> int:
    from mordred import Calculator
    from mordred.EState import AtomTypeEState

    names = [n for n, _ in AtomTypes._rawD]
    want = hume_estate_columns()
    print(f"\nHUME keeps {len(want)} EState columns in PREDICT: "
          f"{sum(1 for c in want if c[0] == 'N')} counts (typer only), "
          f"{sum(1 for c in want if c[0] == 'S')} sums (typer + index)")

    smis = (HERE / "estate_cpp.smi").read_text().split()[:n_mol]
    lines = (HERE / "estate_cpp_types.txt").read_text().strip().split("\n")
    assert len(lines) >= len(smis), "type dump shorter than the smiles list"

    calc = Calculator([AtomTypeEState("count" if c[0] == "N" else "sum", c[1:]) for c in want])
    nbad = 0
    for k, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)                                   # fresh: mordred memoises
        ref = list(calc(m))
        f = lines[k].split()
        na = int(f[0])
        pos, per_atom = 1, []
        for _ in range(na):
            c = int(f[pos]); pos += 1
            per_atom.append([names[int(x)] for x in f[pos:pos + c]]); pos += c
        idx = _EState.EStateIndices(m)
        for col, r in zip(want, ref):
            t = col[1:]
            hit = [i for i in range(na) if t in per_atom[i]]
            v = len(hit) if col[0] == "N" else float(sum(idx[i] for i in hit))
            if isinstance(r, (int, float)) and not (isinstance(r, float) and math.isnan(r)):
                ok = (v == r) if col[0] == "N" else (v == r or abs(v - r) <= 1e-12 * max(1.0, abs(r)))
            else:
                ok = False
            if not ok:
                nbad += 1
                if nbad <= 20:
                    print(f"  MISMATCH mol {k} {smi} {col}: got {v!r} want {r!r}")
    tot = len(smis) * len(want)
    print(f"  {tot - nbad:,} / {tot:,} column values exact over {len(smis):,} molecules "
          f"({len(want)} columns)")
    return 1 if nbad else 0


def timing(n_mol: int = 2000) -> int:
    """What the 50 columns cost in mordred, measured in ALTERNATING PAIRS on a shared machine.

    Three arms, interleaved rep by rep so a load spike hits all three rather than whichever ran
    last: mordred's 50 EState columns, RDKit's TypeAtoms alone (the 79 SMARTS passes, which is
    the part this port replaces), and RDKit's EStateIndices alone (which HUME already computes
    natively). All three are handed a FRESH COPY of the molecule -- `Chem.Mol(m)` -- so nothing
    measures a memoised second pass, and none of them pays SMILES parsing, because the C++ this
    is compared against does not either.
    """
    import time as _t

    from mordred import Calculator
    from mordred.EState import AtomTypeEState

    want = hume_estate_columns()
    mols = [m for m in (Chem.MolFromSmiles(s) for s in
                        (HERE / "estate_mols.smi").read_text().split()[:n_mol]) if m is not None]
    calc = Calculator([AtomTypeEState("count" if c[0] == "N" else "sum", c[1:]) for c in want])
    AtomTypes.BuildPatts()

    def arm_mordred():
        for m in mols:
            calc(Chem.Mol(m))

    def arm_typer():
        for m in mols:
            AtomTypes.TypeAtoms(Chem.Mol(m))

    def arm_index():
        for m in mols:
            _EState.EStateIndices(Chem.Mol(m))

    arms = [("mordred, the 50 columns", arm_mordred),
            ("rdkit TypeAtoms alone (what this port replaces)", arm_typer),
            ("rdkit EStateIndices alone (hume already has this)", arm_index)]
    reps = {k: [] for k, _ in arms}
    for _ in range(5):
        for k, fn in arms:
            t0 = _t.perf_counter(); fn(); t1 = _t.perf_counter()
            reps[k].append((t1 - t0) * 1e6 / len(mols))
    print(f"\ncost of the same work in python, {len(mols):,} molecules, 5 alternating reps, "
          f"no SMILES parsing, fresh Mol copy each call   CONTENDED")
    for k, _ in arms:
        v = sorted(reps[k])
        print(f"  {k:52s} median {v[len(v)//2]:9.1f} us/mol   min {v[0]:9.1f}")
    return 0


# ---------------------------------------------------------------------------------------------
def build_and_run(args: list[str]) -> int:
    exe = HERE / "estate_typer"
    src = HERE / "estate_typer.cpp"
    if not exe.exists() or exe.stat().st_mtime < max(
            src.stat().st_mtime, (HERE / "estate_tables.h").stat().st_mtime,
            (ROOT / "src" / "hume_core" / "estate_typer.h").stat().st_mtime):
        cmd = ["c++", "-O3", "-std=c++17", "-Wall", "-Wextra", str(src), "-o", str(exe)]
        print("$ " + " ".join(cmd))
        if subprocess.call(cmd) != 0:
            return 2
    print("$ " + " ".join([str(exe)] + args))
    return subprocess.call([str(exe)] + args, cwd=str(ROOT))


# Every exactness claim in this repository is pinned here. `tables` generates the specification
# from the loaded RDKit and `verify` compares against the same one, so an unpinned run is a
# harness checking itself -- see the module docstring.
PINNED_RDKIT = "2025.09.2"


def banner(strict: bool) -> None:
    print(f"rdkit {rdkit.__version__}   numpy {numpy_version()}   "
          f"mordred {_mordred_version()}   (rdkit pin: {PINNED_RDKIT})")
    if rdkit.__version__ == PINNED_RDKIT:
        return
    msg = (f"rdkit is {rdkit.__version__}, not the pinned {PINNED_RDKIT}. Generating or verifying "
           f"against a floating oracle makes the exactness check blind, because the table and "
           f"the reference move together. Re-run with: "
           f"uv run --with 'rdkit=={PINNED_RDKIT.replace('.09.', '.9.')}' ...")
    if strict:
        raise SystemExit("REFUSING: " + msg)
    print("  WARNING: " + msg)


def _mordred_version() -> str:
    try:
        import mordred
        return mordred.__version__
    except Exception:
        return "-"


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    banner(strict=cmd in ("tables", "all"))
    if cmd == "tables":
        gen_tables(); return 0
    if cmd == "stress":
        stress(); return 0
    if cmd == "mols":
        src = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "hard.smi")
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
        kept = dump_mols(Path(src).read_text().split()[:n], HERE / "estate_mols.txt", src)
        (HERE / "estate_mols.smi").write_text("\n".join(kept) + "\n")
        return 0
    if cmd == "columns":
        return columns(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    if cmd == "time":
        return timing(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    if cmd == "all":
        gen_tables()
        stress()
        kept = dump_mols(Path(HERE / "hard.smi").read_text().split()[:100_000],
                         HERE / "estate_mols.txt", "cpp/hard.smi")
        (HERE / "estate_mols.smi").write_text("\n".join(kept) + "\n")
        rc = build_and_run(["verify", "cpp/estate_stress.txt"])
        rc |= build_and_run(["verify", "cpp/estate_mols.txt"])
        # Feed the C++ its own types back out for the column check.
        (HERE / "estate_cpp.smi").write_text("\n".join(kept[:2000]) + "\n")
        dump_mols(kept[:2000], HERE / "estate_cpp_in.txt", "column subset")
        rc |= build_and_run(["dump", "cpp/estate_cpp_in.txt", "cpp/estate_cpp_types.txt"])
        rc |= columns(2000)
        rc |= build_and_run(["bench", "cpp/estate_mols.txt"])
        rc |= timing(2000)
        return rc
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
