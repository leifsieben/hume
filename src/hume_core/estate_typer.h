// The E-state atom typer, as a header the extension can call.
//
// WHAT THIS IS AND WHY IT EXISTS. The E-state INDEX is already native -- `estate_from()` in
// hume_blocks.h, verified exact. What was missing is the TYPER: the 79 SMARTS patterns that
// decide whether an atom is an `sCH3`, a `dsCH`, an `aaN` or nothing at all. mordred's 50
// surviving `EState` columns are aggregations over those type strings:
//
//     N<t> = number of atoms whose type tuple contains t      <- TYPER ONLY, no index at all
//     S<t> = sum of the E-state index over those atoms        <- typer + the existing index
//
// so 29 of the 50 need nothing from this file's neighbours. The family sat in PREDICT because
// nobody had ported this, not because it is expensive: see blocks.py's classify().
//
// WHERE THE RULES COME FROM. mordred/EState.py has no typer -- `EStateCache.calculate` is
// `return EState.TypeAtoms(self.mol), EState.EStateIndices(self.mol)`. So the specification is
// `rdkit/Chem/EState/AtomTypes.py`, whose `_rawD` is 79 (name, SMARTS) pairs, and `TypeAtoms`,
// which runs each pattern with `GetSubstructMatches(patt, uniquify=0)` and records the pattern
// name for `match[0]`, dropping duplicates. An atom therefore gets a possibly-empty,
// possibly-multiple TUPLE of names, in pattern order. That is what this file reproduces.
//
// FIVE THINGS THE PATTERNS MEAN THAT THEIR NAMES DO NOT SAY. All five are decoded from RDKit's
// own parse tree by cpp/verify_estate.py, not inferred from the type strings:
//
//   1. `[SeD2H0]` and friends (Li, Be, Si, Ge, As, Se, Sn, Pb) parse to ELEMENT-NUMBER queries
//      with NO aromaticity constraint, while C/N/O/S/B/P/F/Cl/Br/I parse to ALIPHATIC ones. So
//      `aaSe` really does fire on selenophene's aromatic `[se]`, and `aaS` reaches thiophene
//      only through the explicit `s` in its `[S,sD2H0]`.
//   2. `aaNH`/`aaN`/`aasN`/`aaO`/`aaS` are written `[N,nD2H0]` -- no semicolon. That parses as
//      `N` OR (`n` AND `D2` AND `H0`): the ALIPHATIC alternative carries no degree or H
//      constraint whatsoever. `aaCH`/`aasC`/`aaaC` use `[C,c;D2H0]` and do not have this. The
//      difference is reproduced, not tidied up; it is why five rows carry two Alts.
//   3. SMARTS `:` is a BOND-ORDER query (`getBondType() == AROMATIC`), not `getIsAromatic()`.
//      This is a DIFFERENT question from the one cpp/crippen.cpp's bond code answers, and the
//      two disagree on a SINGLE-typed bond that carries the aromatic flag. `btype` below is the
//      bond TYPE and only the bond type.
//   4. `ddsN` and `ddssS` are the only patterns needing NEIGHBOUR IDENTITY: two of their
//      branches are `~[OD1H0]`, a terminal H-free oxygen reached by ANY bond. The bond order is
//      deliberately unconstrained, which is how `CN(=O)=O` and `C[N+](=O)[O-]` both type `ddsN`.
//      (`aasN` is also marked `# mod` upstream, but its modification is the `-,:` last bond.)
//   5. SMARTS is SUBSTRUCTURE matching with DISTINCT query atoms, so an atom matches when its
//      bonds admit a system of distinct representatives for the branch specs -- NOT when it has
//      enough bonds of each kind counted independently. The two differ as soon as two branch
//      specs overlap, which `ddssS` does on a sulfur whose four neighbours are all terminal
//      oxygens. `assign()` below is that matching; it is a search, not a tally.
//
// HOW IT IS KEPT HONEST. The 79 patterns live in the GENERATED cpp/estate_tables.h, decoded from
// RDKit's parse tree, carrying the RDKit version and the sha256 of AtomTypes.py. This file
// duplicates only the central-atom predicate, once per row, stored beside the SMARTS string it
// implements -- and `selfCheck()` does not merely compare those strings. It evaluates every
// hand-written predicate against the generated Alt spec over the ENTIRE input domain
// (z 0..95 x aromatic x degree 0..9 x H 0..5, 11,520 states per row), so a predicate that
// disagrees with the table anywhere at all fails at load rather than on some molecule nobody
// tried. It also proves the z-dispatch never hides a row that would have matched.
//
// WHAT THE CALLER MUST SUPPLY -- all five already exist in _extract.py / bindings.cpp:
//
//   z      atomic number                     GetAtomicNum()
//   arom   GetIsAromatic()                   (used only by the element primitives, see 1/2)
//   nH     GetTotalNumHs(False)              hume_blocks.h's `nH`; `finish()` turns it into the
//                                            SMARTS H count by adding neighbouring H ATOMS
//   start/nbr   CSR adjacency, from which SMARTS `D` == degree is start[i+1]-start[i]
//   btype  bond TYPE bits, per half-edge     BT_SINGLE/DOUBLE/TRIPLE/AROMATIC, 0 for anything
//                                            else (DATIVE occurs in cpp/hard.smi). NOT the same
//                                            as _extract.py's `bcode`; use btypeFromBcode().
//
// HOW TO CALL IT (this file is not wired into bindings.cpp or hume_blocks.h on purpose).
// Every input is already at the boundary as of the (n_atoms, 2) / (n_bonds, 5) layout --
// `atom_i` columns A_Z, A_DEG, A_NH, A_AROM and `bond_i` columns B_U, B_V, B_CODE. The fill is
// crippen_fill() with `chg`/`tx` dropped and one line changed, so the cheapest wiring is to
// build both typers' Mol in the same pass over the same rows:
//
//     esttyper::selfCheck();                       // once, at module load, next to criptyper's
//     esttyper::Mol em; em.alloc(n, 2 * nb);
//     em.z[i] = r[A_Z]; em.arom[i] = r[A_AROM]; em.nH[i] = r[A_NH];   // CSR exactly as crippen
//     em.btype[e] = esttyper::btypeFromBcode((uint8_t)r[B_CODE]);     // NOT the raw bcode
//     em.finish();                                 // derives the SMARTS H count
//     esttyper::aggregate(em, S.data(), N79, S79); // S = estate_from()'s per-atom index
//
// A_DEG is not stored: SMARTS `D` is the CSR degree, which `alloc`+`start` already give, and
// reading both would be two sources for one number. The harness asserts they agree.
//
// `N79[t]` is then mordred's `N<name>` and `S79[t]` its `S<name>` for
// `estate_tbl::ROWS[t].name`, for all 79 types in one pass over the molecule. Pass `nullptr`
// for the index and for S79 if only the 29 count columns are wanted -- they never read it.
#ifndef HUME_ESTATE_TYPER_H
#define HUME_ESTATE_TYPER_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// The patterns themselves. Generated by cpp/verify_estate.py from RDKit's own parsed queries;
// included from cpp/ rather than copied so there is exactly one of it in the repository.
#include "../../cpp/estate_tables.h"

namespace esttyper {

using estate_tbl::Alt;
using estate_tbl::Row;
using estate_tbl::ROWS;
using estate_tbl::NBRQ;
using estate_tbl::N_ROWS;
using estate_tbl::AROM_ANY;

// Bond TYPE bits. Re-exported so a caller need not include the generated header by name.
enum : uint8_t {
  BT_SINGLE   = estate_tbl::BT_SINGLE,
  BT_DOUBLE   = estate_tbl::BT_DOUBLE,
  BT_TRIPLE   = estate_tbl::BT_TRIPLE,
  BT_AROMATIC = estate_tbl::BT_AROMATIC,
};

// The most rows any single atom can satisfy at once is bounded by the rows sharing its element
// (nitrogen has 14). Sized generously and checked in selfCheck().
constexpr int MAX_TYPES = 16;

// _extract.py and cpp/export_crippen.py already build a 4-bit `bcode` per bond, but its top bit
// is `getIsAromatic()` -- and this file needs "bond type IS aromatic", which is a different
// question (see note 3 above). They coincide on 3,090,888 of cpp/hard.smi's 3,090,892 bonds; the
// four exceptions are TRIPLE bonds carrying the aromatic flag, which bcode reports as 4|8 and
// which `:` must NOT match. This recovery is exact whenever an order bit is set, and it is
// verified equal to the bond-type code on all 3,090,892 bonds of cpp/hard.smi. The single case
// it cannot recover is an AROMATIC-TYPED bond whose flag is false, which occurs 0 times there
// but is not impossible; if _extract.py ever gains a fifth bit meaning "type == AROMATIC"
// (bit 16, leaving bits 1..8 untouched so criptyper is unaffected), read that instead.
static inline uint8_t btypeFromBcode(uint8_t bcode) {
  const uint8_t ord = bcode & (uint8_t)(BT_SINGLE | BT_DOUBLE | BT_TRIPLE);
  return ord ? ord : (uint8_t)(bcode & BT_AROMATIC);
}

// ---------------------------------------------------------------------------------------------
// molecule
// ---------------------------------------------------------------------------------------------
struct Mol {
  int n = 0;
  std::vector<uint8_t>  z, arom, nH, sh;     // nH = GetTotalNumHs(False); sh = SMARTS H
  std::vector<int32_t>  start;               // CSR, size n+1
  std::vector<int32_t>  nbr;
  std::vector<uint8_t>  btype;               // per half-edge, parallel to nbr

  void alloc(int nn, int nb2) {
    n = nn;
    z.assign(nn, 0); arom.assign(nn, 0); nH.assign(nn, 0); sh.assign(nn, 0);
    start.assign(nn + 1, 0); nbr.assign(nb2, 0); btype.assign(nb2, 0);
  }

  // SMARTS `H` is GetTotalNumHs(TRUE): the implicit/property count PLUS neighbouring H ATOMS.
  // MolFromSmiles keeps an H atom in the graph whenever removeHs cannot fold it away (isotopes,
  // H2, charged H), and cpp/hard.smi contains those, so the difference is live.
  void finish() {
    for (int i = 0; i < n; ++i) {
      int extra = 0;
      for (int e = start[i]; e < start[i + 1]; ++e) if (z[nbr[e]] == 1) ++extra;
      sh[i] = (uint8_t)(nH[i] + extra);
    }
  }

  int deg(int i) const { return start[i + 1] - start[i]; }
};

// ---------------------------------------------------------------------------------------------
// central-atom predicates -- one per pattern, each stored beside its SMARTS
//
// The whole decision is a function of (z, aromatic, degree, H count), which is why selfCheck()
// can enumerate the entire domain and prove each predicate equals the generated table's Alt
// spec. `ali` is an ALIPHATIC element primitive (`[C]`), `ele` an element-number one (`[#34]`);
// which of the two a symbol parses to is not guessable from the symbol, see note 1 above.
// ---------------------------------------------------------------------------------------------
#define C_(name) static inline bool name(int z, int arom, int d, int h)

static inline bool ali(int z, int arom, int Z) { return z == Z && !arom; }
static inline bool aro(int z, int arom, int Z) { return z == Z && arom; }

C_(c_sLi)     { (void)arom; (void)h; return z ==  3 && d == 1; }                    // [#3&D1]
C_(c_ssBe)    { (void)arom; (void)h; return z ==  4 && d == 2; }                    // [#4&D2]
C_(c_ssssBe)  { (void)arom; (void)h; return z ==  4 && d == 4; }                    // [#4&D4]
C_(c_ssBH)    { return ali(z, arom,  5) && d == 2 && h == 1; }                      // [B&D2&H1]
C_(c_sssB)    { (void)h; return ali(z, arom,  5) && d == 3; }                       // [B&D3]
C_(c_ssssB)   { (void)h; return ali(z, arom,  5) && d == 4; }                       // [B&D4]
C_(c_sCH3)    { return ali(z, arom,  6) && d == 1 && h == 3; }
C_(c_dCH2)    { return ali(z, arom,  6) && d == 1 && h == 2; }
C_(c_ssCH2)   { return ali(z, arom,  6) && d == 2 && h == 2; }
C_(c_tCH)     { return ali(z, arom,  6) && d == 1 && h == 1; }
C_(c_dsCH)    { return ali(z, arom,  6) && d == 2 && h == 1; }
C_(c_aaCH)    { (void)arom; return z == 6 && d == 2 && h == 1; }                    // [C,c;D2&H1]
C_(c_sssCH)   { return ali(z, arom,  6) && d == 3 && h == 1; }
C_(c_ddC)     { return ali(z, arom,  6) && d == 2 && h == 0; }
C_(c_tsC)     { return ali(z, arom,  6) && d == 2 && h == 0; }
C_(c_dssC)    { return ali(z, arom,  6) && d == 3 && h == 0; }
C_(c_aasC)    { (void)arom; return z == 6 && d == 3 && h == 0; }                    // [C,c;D3&H0]
C_(c_aaaC)    { (void)arom; return z == 6 && d == 3 && h == 0; }                    // [C,c;D3&H0]
C_(c_ssssC)   { return ali(z, arom,  6) && d == 4 && h == 0; }
C_(c_sNH3)    { return ali(z, arom,  7) && d == 1 && h == 3; }
C_(c_sNH2)    { return ali(z, arom,  7) && d == 1 && h == 2; }
C_(c_ssNH2)   { return ali(z, arom,  7) && d == 2 && h == 2; }
C_(c_dNH)     { return ali(z, arom,  7) && d == 1 && h == 1; }
C_(c_ssNH)    { return ali(z, arom,  7) && d == 2 && h == 1; }
// [N,n&D2&H1] -- no semicolon, so the aliphatic branch is unconstrained. See note 2.
C_(c_aaNH)    { return ali(z, arom, 7) || (aro(z, arom, 7) && d == 2 && h == 1); }
C_(c_tN)      { return ali(z, arom,  7) && d == 1 && h == 0; }
C_(c_sssNH)   { return ali(z, arom,  7) && d == 3 && h == 1; }
C_(c_dsN)     { return ali(z, arom,  7) && d == 2 && h == 0; }
C_(c_aaN)     { return ali(z, arom, 7) || (aro(z, arom, 7) && d == 2 && h == 0); }  // [N,n&D2&H0]
C_(c_sssN)    { return ali(z, arom,  7) && d == 3 && h == 0; }
C_(c_ddsN)    { return ali(z, arom,  7) && d == 3 && h == 0; }
C_(c_aasN)    { return ali(z, arom, 7) || (aro(z, arom, 7) && d == 3 && h == 0); }  // [N,n&D3&H0]
C_(c_ssssN)   { return ali(z, arom,  7) && d == 4 && h == 0; }
C_(c_sOH)     { return ali(z, arom,  8) && d == 1 && h == 1; }
C_(c_dO)      { return ali(z, arom,  8) && d == 1 && h == 0; }
C_(c_ssO)     { return ali(z, arom,  8) && d == 2 && h == 0; }
C_(c_aaO)     { return ali(z, arom, 8) || (aro(z, arom, 8) && d == 2 && h == 0); }  // [O,o&D2&H0]
C_(c_sF)      { (void)h; return ali(z, arom,  9) && d == 1; }                       // [F&D1]
C_(c_sSiH3)   { (void)arom; return z == 14 && d == 1 && h == 3; }                   // [#14&D1&H3]
C_(c_ssSiH2)  { (void)arom; return z == 14 && d == 2 && h == 2; }
C_(c_sssSiH)  { (void)arom; return z == 14 && d == 3 && h == 1; }
C_(c_ssssSi)  { (void)arom; return z == 14 && d == 4 && h == 0; }
C_(c_sPH2)    { return ali(z, arom, 15) && d == 1 && h == 2; }
C_(c_ssPH)    { return ali(z, arom, 15) && d == 2 && h == 1; }
C_(c_sssP)    { return ali(z, arom, 15) && d == 3 && h == 0; }
C_(c_dsssP)   { return ali(z, arom, 15) && d == 4 && h == 0; }
C_(c_sssssP)  { return ali(z, arom, 15) && d == 5 && h == 0; }
C_(c_sSH)     { return ali(z, arom, 16) && d == 1 && h == 1; }
C_(c_dS)      { return ali(z, arom, 16) && d == 1 && h == 0; }
C_(c_ssS)     { return ali(z, arom, 16) && d == 2 && h == 0; }
C_(c_aaS)     { return ali(z, arom, 16) || (aro(z, arom, 16) && d == 2 && h == 0); }// [S,s&D2&H0]
C_(c_dssS)    { return ali(z, arom, 16) && d == 3 && h == 0; }
C_(c_ddssS)   { return ali(z, arom, 16) && d == 4 && h == 0; }
C_(c_sCl)     { (void)h; return ali(z, arom, 17) && d == 1; }                       // [Cl&D1]
C_(c_sGeH3)   { (void)arom; return z == 32 && d == 1 && h == 3; }                   // [#32&D1&H3]
C_(c_ssGeH2)  { (void)arom; return z == 32 && d == 2 && h == 2; }
C_(c_sssGeH)  { (void)arom; return z == 32 && d == 3 && h == 1; }
C_(c_ssssGe)  { (void)arom; return z == 32 && d == 4 && h == 0; }
C_(c_sAsH2)   { (void)arom; return z == 33 && d == 1 && h == 2; }                   // [#33&D1&H2]
C_(c_ssAsH)   { (void)arom; return z == 33 && d == 2 && h == 1; }
C_(c_sssAs)   { (void)arom; return z == 33 && d == 3 && h == 0; }
C_(c_sssdAs)  { (void)arom; return z == 33 && d == 4 && h == 0; }
C_(c_sssssAs) { (void)arom; return z == 33 && d == 5 && h == 0; }
C_(c_sSeH)    { (void)arom; return z == 34 && d == 1 && h == 1; }                   // [#34&D1&H1]
C_(c_dSe)     { (void)arom; return z == 34 && d == 1 && h == 0; }
C_(c_ssSe)    { (void)arom; return z == 34 && d == 2 && h == 0; }
C_(c_aaSe)    { (void)arom; return z == 34 && d == 2 && h == 0; }                   // aromatic se
C_(c_dssSe)   { (void)arom; return z == 34 && d == 3 && h == 0; }
C_(c_ddssSe)  { (void)arom; return z == 34 && d == 4 && h == 0; }
C_(c_sBr)     { (void)h; return ali(z, arom, 35) && d == 1; }                       // [Br&D1]
C_(c_sSnH3)   { (void)arom; return z == 50 && d == 1 && h == 3; }                   // [#50&D1&H3]
C_(c_ssSnH2)  { (void)arom; return z == 50 && d == 2 && h == 2; }
C_(c_sssSnH)  { (void)arom; return z == 50 && d == 3 && h == 1; }
C_(c_ssssSn)  { (void)arom; return z == 50 && d == 4 && h == 0; }
C_(c_sI)      { (void)h; return ali(z, arom, 53) && d == 1; }                       // [I&D1]
C_(c_sPbH3)   { (void)arom; return z == 82 && d == 1 && h == 3; }                   // [#82&D1&H3]
C_(c_ssPbH2)  { (void)arom; return z == 82 && d == 2 && h == 2; }
C_(c_sssPbH)  { (void)arom; return z == 82 && d == 3 && h == 1; }
C_(c_ssssPb)  { (void)arom; return z == 82 && d == 4 && h == 0; }

#undef C_

typedef bool (*CentralFn)(int, int, int, int);

// The predicate table. Each entry names the pattern and repeats the SMARTS it implements, and
// selfCheck() asserts BOTH against estate_tables.h -- the string so a pattern that is edited
// upstream is caught, and the behaviour so a predicate that was mistyped is caught.
struct Pred { const char* name; const char* smarts; CentralFn fn; };

static const Pred PRED[] = {
  {"sLi",     "[LiD1]-*",                          c_sLi},
  {"ssBe",    "[BeD2](-*)-*",                      c_ssBe},
  {"ssssBe",  "[BeD4](-*)(-*)(-*)-*",              c_ssssBe},
  {"ssBH",    "[BD2H](-*)-*",                      c_ssBH},
  {"sssB",    "[BD3](-*)(-*)-*",                   c_sssB},
  {"ssssB",   "[BD4](-*)(-*)(-*)-*",               c_ssssB},
  {"sCH3",    "[CD1H3]-*",                         c_sCH3},
  {"dCH2",    "[CD1H2]=*",                         c_dCH2},
  {"ssCH2",   "[CD2H2](-*)-*",                     c_ssCH2},
  {"tCH",     "[CD1H]#*",                          c_tCH},
  {"dsCH",    "[CD2H](=*)-*",                      c_dsCH},
  {"aaCH",    "[C,c;D2H](:*):*",                   c_aaCH},
  {"sssCH",   "[CD3H](-*)(-*)-*",                  c_sssCH},
  {"ddC",     "[CD2H0](=*)=*",                     c_ddC},
  {"tsC",     "[CD2H0](#*)-*",                     c_tsC},
  {"dssC",    "[CD3H0](=*)(-*)-*",                 c_dssC},
  {"aasC",    "[C,c;D3H0](:*)(:*)-*",              c_aasC},
  {"aaaC",    "[C,c;D3H0](:*)(:*):*",              c_aaaC},
  {"ssssC",   "[CD4H0](-*)(-*)(-*)-*",             c_ssssC},
  {"sNH3",    "[ND1H3]-*",                         c_sNH3},
  {"sNH2",    "[ND1H2]-*",                         c_sNH2},
  {"ssNH2",   "[ND2H2](-*)-*",                     c_ssNH2},
  {"dNH",     "[ND1H]=*",                          c_dNH},
  {"ssNH",    "[ND2H](-*)-*",                      c_ssNH},
  {"aaNH",    "[N,nD2H](:*):*",                    c_aaNH},
  {"tN",      "[ND1H0]#*",                         c_tN},
  {"sssNH",   "[ND3H](-*)(-*)-*",                  c_sssNH},
  {"dsN",     "[ND2H0](=*)-*",                     c_dsN},
  {"aaN",     "[N,nD2H0](:*):*",                   c_aaN},
  {"sssN",    "[ND3H0](-*)(-*)-*",                 c_sssN},
  {"ddsN",    "[ND3H0](~[OD1H0])(~[OD1H0])-,:*",   c_ddsN},
  {"aasN",    "[N,nD3H0](:*)(:*)-,:*",             c_aasN},
  {"ssssN",   "[ND4H0](-*)(-*)(-*)-*",             c_ssssN},
  {"sOH",     "[OD1H]-*",                          c_sOH},
  {"dO",      "[OD1H0]=*",                         c_dO},
  {"ssO",     "[OD2H0](-*)-*",                     c_ssO},
  {"aaO",     "[O,oD2H0](:*):*",                   c_aaO},
  {"sF",      "[FD1]-*",                           c_sF},
  {"sSiH3",   "[SiD1H3]-*",                        c_sSiH3},
  {"ssSiH2",  "[SiD2H2](-*)-*",                    c_ssSiH2},
  {"sssSiH",  "[SiD3H1](-*)(-*)-*",                c_sssSiH},
  {"ssssSi",  "[SiD4H0](-*)(-*)(-*)-*",            c_ssssSi},
  {"sPH2",    "[PD1H2]-*",                         c_sPH2},
  {"ssPH",    "[PD2H1](-*)-*",                     c_ssPH},
  {"sssP",    "[PD3H0](-*)(-*)-*",                 c_sssP},
  {"dsssP",   "[PD4H0](=*)(-*)(-*)-*",             c_dsssP},
  {"sssssP",  "[PD5H0](-*)(-*)(-*)(-*)-*",         c_sssssP},
  {"sSH",     "[SD1H1]-*",                         c_sSH},
  {"dS",      "[SD1H0]=*",                         c_dS},
  {"ssS",     "[SD2H0](-*)-*",                     c_ssS},
  {"aaS",     "[S,sD2H0](:*):*",                   c_aaS},
  {"dssS",    "[SD3H0](=*)(-*)-*",                 c_dssS},
  {"ddssS",   "[SD4H0](~[OD1H0])(~[OD1H0])(-*)-*", c_ddssS},
  {"sCl",     "[ClD1]-*",                          c_sCl},
  {"sGeH3",   "[GeD1H3](-*)",                      c_sGeH3},
  {"ssGeH2",  "[GeD2H2](-*)-*",                    c_ssGeH2},
  {"sssGeH",  "[GeD3H1](-*)(-*)-*",                c_sssGeH},
  {"ssssGe",  "[GeD4H0](-*)(-*)(-*)-*",            c_ssssGe},
  {"sAsH2",   "[AsD1H2]-*",                        c_sAsH2},
  {"ssAsH",   "[AsD2H1](-*)-*",                    c_ssAsH},
  {"sssAs",   "[AsD3H0](-*)(-*)-*",                c_sssAs},
  {"sssdAs",  "[AsD4H0](=*)(-*)(-*)-*",            c_sssdAs},
  {"sssssAs", "[AsD5H0](-*)(-*)(-*)(-*)-*",        c_sssssAs},
  {"sSeH",    "[SeD1H1]-*",                        c_sSeH},
  {"dSe",     "[SeD1H0]=*",                        c_dSe},
  {"ssSe",    "[SeD2H0](-*)-*",                    c_ssSe},
  {"aaSe",    "[SeD2H0](:*):*",                    c_aaSe},
  {"dssSe",   "[SeD3H0](=*)(-*)-*",                c_dssSe},
  {"ddssSe",  "[SeD4H0](=*)(=*)(-*)-*",            c_ddssSe},
  {"sBr",     "[BrD1]-*",                          c_sBr},
  {"sSnH3",   "[SnD1H3]-*",                        c_sSnH3},
  {"ssSnH2",  "[SnD2H2](-*)-*",                    c_ssSnH2},
  {"sssSnH",  "[SnD3H1](-*)(-*)-*",                c_sssSnH},
  {"ssssSn",  "[SnD4H0](-*)(-*)(-*)-*",            c_ssssSn},
  {"sI",      "[ID1]-*",                           c_sI},
  {"sPbH3",   "[PbD1H3]-*",                        c_sPbH3},
  {"ssPbH2",  "[PbD2H2](-*)-*",                    c_ssPbH2},
  {"sssPbH",  "[PbD3H1](-*)(-*)-*",                c_sssPbH},
  {"ssssPb",  "[PbD4H0](-*)(-*)(-*)-*",            c_ssssPb},
};

// ---------------------------------------------------------------------------------------------
// matching
// ---------------------------------------------------------------------------------------------
constexpr int ZMAX = 96;                       // Pb is 82; nothing above this can match a row

struct Dispatch {
  uint8_t n[ZMAX] = {0};
  uint8_t row[ZMAX][16] = {{0}};
  uint8_t uniform[N_ROWS] = {0};               // every branch spec identical -> count, not search
};

// Built ON FIRST USE, not by selfCheck(). Making the table a side effect of the checker would
// mean a caller who forgot to check got an empty dispatch and silently typed nothing -- the
// exact failure mode this file exists to make impossible.
inline Dispatch buildDispatch();
inline const Dispatch& dispatch() { static const Dispatch d = buildDispatch(); return d; }

static inline bool bondOK(uint8_t mask, uint8_t code) { return mask == 0 || (code & mask) != 0; }

static inline bool altOK(const Alt& a, int z, int arom, int d, int h) {
  if (a.z >= 0 && z != a.z) return false;
  if (a.arom != AROM_ANY && arom != (int)a.arom) return false;
  if (a.d >= 0 && d != a.d) return false;
  if (a.h >= 0 && h != a.h) return false;
  return true;
}

static inline bool nbrOK(const Mol& m, int j, uint8_t nq) {
  if (nq == 0) return true;                    // `*`
  return altOK(NBRQ[nq], m.z[j], m.arom[j], m.deg(j), m.sh[j]);
}

// System of distinct representatives: branch b picks an unused incident bond that satisfies both
// its bond mask and its neighbour query. Depth is at most 5 (sssssP/sssssAs), and `chosen` is
// scanned linearly rather than held as a bitmask so there is no cap on the atom's degree.
static bool assign(const Mol& m, int i, const Row& r, int b, int* chosen) {
  if (b >= r.nbranch) return true;
  const int s = m.start[i], e = m.start[i + 1];
  for (int k = s; k < e; ++k) {
    bool taken = false;
    for (int q = 0; q < b; ++q) if (chosen[q] == k) { taken = true; break; }
    if (taken) continue;
    if (!bondOK(r.br[b].bond, m.btype[k])) continue;
    if (!nbrOK(m, m.nbr[k], r.br[b].nq)) continue;
    chosen[b] = k;
    if (assign(m, i, r, b + 1, chosen)) return true;
  }
  return false;
}

static inline bool branchesOK(const Mol& m, int i, int rowIdx) {
  const Row& r = ROWS[rowIdx];
  if (dispatch().uniform[rowIdx]) {
    // All branch specs equal, so a matching exists iff enough bonds satisfy the one spec.
    int want = r.nbranch, got = 0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k)
      if (bondOK(r.br[0].bond, m.btype[k]) && nbrOK(m, m.nbr[k], r.br[0].nq) && ++got >= want)
        return true;
    return false;
  }
  int chosen[estate_tbl::MAX_BRANCH];
  return assign(m, i, r, 0, chosen);
}

// Every pattern atom `i` matches, as row indices IN PATTERN ORDER -- the same order
// AtomTypes.TypeAtoms builds its tuple in. Returns how many were written to `out`.
inline int typeAtom(const Mol& m, int i, uint8_t* out) {
  const int z = m.z[i];
  if (z >= ZMAX) return 0;
  const Dispatch& D = dispatch();
  const int arom = m.arom[i], d = m.deg(i), h = m.sh[i];
  int k = 0;
  for (int t = 0; t < D.n[z]; ++t) {
    const int r = D.row[z][t];
    if (!PRED[r].fn(z, arom, d, h)) continue;
    if (!branchesOK(m, i, r)) continue;
    out[k++] = (uint8_t)r;
  }
  return k;
}

// mordred's whole EState family in one pass: N[t] is `N<name>`, S[t] is `S<name>`.
// `estate` is the per-atom index -- hume_blocks.h's estate_from() output. Pass nullptr if only
// the 29 count columns are wanted; they never touch the index.
inline void aggregate(const Mol& m, const double* estate, int32_t* N, double* S) {
  for (int t = 0; t < N_ROWS; ++t) { N[t] = 0; if (S) S[t] = 0.0; }
  uint8_t hit[MAX_TYPES];
  for (int i = 0; i < m.n; ++i) {
    const int k = typeAtom(m, i, hit);
    for (int q = 0; q < k; ++q) {
      ++N[hit[q]];
      if (S && estate) S[hit[q]] += estate[i];
    }
  }
}

inline Dispatch buildDispatch() {
  Dispatch D;
  if ((int)(sizeof(PRED) / sizeof(PRED[0])) != N_ROWS)
    throw std::runtime_error("estate_typer: predicate count != estate_tables.h N_ROWS");
  for (int r = 0; r < N_ROWS; ++r) {
    for (int z = 0; z < ZMAX; ++z) {
      // "Can this row ever accept element z?" -- asked of the PREDICATE over the whole domain,
      // so the bucket cannot disagree with what typeAtom() will actually evaluate.
      bool any = false;
      for (int a = 0; a < 2 && !any; ++a)
        for (int d = 0; d <= 9 && !any; ++d)
          for (int h = 0; h <= 5 && !any; ++h) any = PRED[r].fn(z, a, d, h);
      if (!any) continue;
      if (D.n[z] >= 16) throw std::runtime_error("estate_typer: dispatch bucket overflow");
      D.row[z][D.n[z]++] = (uint8_t)r;
    }
    // A row is "uniform" when all its branch specs are identical, which turns the perfect
    // matching into a plain count. Most rows are; the rest go through assign().
    bool u = true;
    for (int b = 1; b < ROWS[r].nbranch; ++b)
      if (ROWS[r].br[b].bond != ROWS[r].br[0].bond || ROWS[r].br[b].nq != ROWS[r].br[0].nq) u = false;
    D.uniform[r] = (uint8_t)u;
  }
  return D;
}

// ---------------------------------------------------------------------------------------------
// selfCheck -- the guard against silent drift, run once at module load
// ---------------------------------------------------------------------------------------------
inline void selfCheck() {
  static bool done = false;
  if (done) return;

  if ((int)(sizeof(PRED) / sizeof(PRED[0])) != N_ROWS)
    throw std::runtime_error("estate_typer: predicate count != estate_tables.h N_ROWS");

  for (int r = 0; r < N_ROWS; ++r) {
    if (std::strcmp(PRED[r].name, ROWS[r].name) || std::strcmp(PRED[r].smarts, ROWS[r].smarts))
      throw std::runtime_error(std::string("estate_typer: row ") + std::to_string(r) + " is '" +
                               PRED[r].name + " " + PRED[r].smarts + "' but estate_tables.h says '" +
                               ROWS[r].name + " " + ROWS[r].smarts + "' -- upstream changed");
    for (int q = 0; q < r; ++q)
      if (!std::strcmp(PRED[r].name, PRED[q].name))
        throw std::runtime_error("estate_typer: duplicate pattern name, TypeAtoms dedups by name");
  }

  // EXHAUSTIVE equivalence of every hand-written predicate with the generated Alt spec over the
  // whole input domain. This is what makes the duplication safe: a mistyped degree or a missed
  // aromaticity constraint cannot survive to a molecule, because there is no molecule state
  // outside this box that the central predicate can see.
  for (int r = 0; r < N_ROWS; ++r) {
    for (int z = 0; z < ZMAX; ++z)
      for (int a = 0; a < 2; ++a)
        for (int d = 0; d <= 9; ++d)
          for (int h = 0; h <= 5; ++h) {
            bool tab = false;
            for (int q = 0; q < ROWS[r].nalt; ++q)
              if (altOK(ROWS[r].alt[q], z, a, d, h)) { tab = true; break; }
            if (PRED[r].fn(z, a, d, h) != tab) {
              char buf[256];
              std::snprintf(buf, sizeof buf,
                            "estate_typer: %s (%s) disagrees with estate_tables.h at "
                            "z=%d arom=%d D=%d H=%d: predicate says %d, table says %d",
                            PRED[r].name, PRED[r].smarts, z, a, d, h,
                            (int)PRED[r].fn(z, a, d, h), (int)tab);
              throw std::runtime_error(buf);
            }
          }
  }

  // The z-dispatch is derived FROM the predicates, so it cannot list a row the predicate would
  // reject -- but it could OMIT one. Prove it does not: over the same exhaustive domain, every
  // row whose central predicate passes must be reachable from DISPATCH[z].
  const Dispatch& D = dispatch();
  for (int z = 0; z < ZMAX; ++z)
    for (int a = 0; a < 2; ++a)
      for (int d = 0; d <= 9; ++d)
        for (int h = 0; h <= 5; ++h)
          for (int r = 0; r < N_ROWS; ++r) {
            if (!PRED[r].fn(z, a, d, h)) continue;
            bool listed = false;
            for (int t = 0; t < D.n[z] && !listed; ++t) listed = (D.row[z][t] == r);
            if (!listed)
              throw std::runtime_error(std::string("estate_typer: dispatch drops ") +
                                       PRED[r].name + " for element " + std::to_string(z));
          }
  for (int z = 0; z < ZMAX; ++z)
    if (D.n[z] > MAX_TYPES)
      throw std::runtime_error("estate_typer: MAX_TYPES too small for element " + std::to_string(z));

  // NBRQ[0] must be the unconstrained `*`, or nbrOK's fast path is wrong.
  if (!(NBRQ[0].z < 0 && NBRQ[0].arom == AROM_ANY && NBRQ[0].d < 0 && NBRQ[0].h < 0))
    throw std::runtime_error("estate_typer: NBRQ[0] is not the wildcard branch atom");

  done = true;
}

}  // namespace esttyper

#endif
