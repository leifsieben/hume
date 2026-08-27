// The VSA-binning descriptor family, header-only C++17.
//
// WHAT THIS COMPUTES -- 62 of the 865, plus four more that dedupe dropped but that fall out of
// the same pass for free:
//
//     SlogP_VSA1-12   SMR_VSA1-10   PEOE_VSA1-14   EState_VSA1-11   VSA_EState1-10
//     MolLogP  MolMR  TPSA  TopoPSA  LabuteASA
//     MaxEStateIndex  MinEStateIndex  MaxAbsEStateIndex  MinAbsEStateIndex
//
// THEY ARE ONE MECHANISM, NOT SEVEN.  Every `*_VSA` column is the same three lines: take a
// per-atom CONTRIBUTION vector, take a per-atom PROPERTY vector, and for each atom add the
// contribution into the bucket the property falls in.  Only the (contribution, property, edges)
// triple changes:
//
//     column          contribution         property            edges
//     SlogP_VSA*      Labute ASA           Crippen logP        LOGP_BINS   (11 -> 12 bins)
//     SMR_VSA*        Labute ASA           Crippen MR          MR_BINS     ( 9 -> 10 bins)
//     PEOE_VSA*       Labute ASA           Gasteiger charge    CHG_BINS    (13 -> 14 bins)
//     EState_VSA*     Labute ASA           E-state index       ESTATE_BINS (10 -> 11 bins)
//     VSA_EState*     E-state index        Labute ASA          VSA_BINS    ( 9 -> 10 bins)
//
// The last row is not a typo: VSA_EState swaps the roles, summing E-STATE into bins of SURFACE
// AREA.  rdkit/Chem/EState/EState_VSA.py has the two loops side by side and they differ only in
// which vector is indexed by which.
//
// WHICH SIDE OF A BIN EDGE.  Both implementations agree, and both are "upper": RDKit's C++
// (Code/GraphMol/Descriptors/MolSurf.cpp, assignContribsToBins) uses
//     std::upper_bound(bins.begin(), bins.end(), bVal) - bins.begin()
// and RDKit's Python (MolSurf.py, EState_VSA.py) uses `bisect.bisect_right(bins, prop)`, which
// is the same function.  So a value sitting EXACTLY on an edge goes into the HIGHER bin -- the
// same convention as numpy.digitize(right=False), which is worth stating because the opposite
// convention is equally common and the difference is invisible until it isn't.  This is not
// hypothetical here: the Wildman-Crippen table contains four rows whose logP contribution is
// EXACTLY 0.0 (C2 `[CH](C)(C)C`, C2 `[C](C)(C)(C)C`, C14 `[c][#9]`, C17 `[c][#53]`) and 0.0 is
// an edge of LOGP_BINS, so every branched aliphatic carbon and every aryl fluoride in the corpus
// is an on-edge test.  cpp/verify_vsa.py counts them.
//
// AND THE OTHER HALF OF THE EDGE QUESTION, which is what happens to an atom NEAR an edge.  A
// last-ULP wobble in a per-atom contribution is harmless as a number and lethal as a bin
// assignment: crossing an edge moves a whole atom's surface area from one column to the next.
// Measured rather than assumed -- 3,000 molecules x 3 random renumberings (with SanitizeMol
// after Chem.RenumberAtoms, which otherwise hands back uninitialised RingInfo and manufactures
// ghosts):
//     Labute ASA, Crippen logP, Crippen MR, Gasteiger charge   bit-identical, 0 bin flips
//     E-state index                                            max |delta| 1.6e-14, 0 bin flips
// The four stable vectors cannot flip a bin at all.  For the E-state index the guarantee is the
// margin: cpp/vsa verify reports how close the nearest non-exact atom in the corpus gets to an
// edge, and the wobble is orders of magnitude below it.  Note that the MOLECULE-level LabuteASA
// sum DOES move in the last ULP under renumbering -- that is summation order over atoms and does
// not touch any bin assignment, because the binning reads the per-atom vector, not the sum.
//
// WHAT IS ACTUALLY NEW HERE IS LABUTE ASA.  See labute_contribs() for the derivation; everything
// else in this header reuses machinery that is already verified:
//   Crippen (logP, MR) per atom     src/hume_core/crippen_typer.h   (exact on 2,866,100 atoms)
//   Gasteiger charge                 arrives at the boundary, atom_d column 1
//   E-state index                    estate_indices() below -- see the note there on why this is
//                                    a second copy rather than a call into hume_blocks.h
//
// PROVENANCE AND DRIFT.  All the numbers live in cpp/vsa_tables.h, generated from a pinned RDKit
// by cpp/gen_vsa_tables.py.  self_check() recomputes the sha256 of the table's own numbers and
// compares it against the literal the generator recorded, so a hand-edited bin edge is an
// exception at load rather than a wrong number in column 41.  The hash is over the SPEC (the
// IEEE-754 bit patterns) and not over the file, because a file hash fires on a comment edit and
// stays quiet on a plausible-looking edit to a value.
//
// This header does NO chemistry perception of its own.  Aromaticity, ring membership, degree,
// hydrogen counts and Gasteiger charges are all inherited from the boundary exactly as RDKit
// perceived them; the one graph question it answers for itself is "is this atom in a
// three-membered ring", which TPSA needs and the boundary's boolean `ring` column cannot answer.
#ifndef HUME_VSA_BINS_H
#define HUME_VSA_BINS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../cpp/vsa_tables.h"
#include "crippen_typer.h"

namespace vsabin {

// =============================================================================================
// the molecule, exactly as src/hume_core/bindings.cpp already has it
//
// Field for field this is the boundary layout described in src/hume/_extract.py:
//     atom_i (n_atoms, 10) = Z, deg, nH, fchg, hyb, arom, ring, cip, nring, tval
//     atom_d (n_atoms, 2) = mass, gasteiger
//     bond_i (n_bonds, 5) = u, v, conjugated, in-ring, SMARTS bond code
// Only the columns this family reads are kept: Z, deg, nH, fchg, arom from atom_i, the Gasteiger
// charge from atom_d, and all three bond columns.  `gast` may be NaN and MUST be left that way --
// see bin_of() below.
//
// NEITHER `ring` NOR `nring` IS ENOUGH FOR TPSA, which asks isAtomInRingOfSize(i, 3).  `ring` is
// a boolean and `nring` is RingInfo::numAtomRings, a COUNT -- neither carries a ring SIZE.  So
// in_ring3() below answers it from the graph instead.  That is safe here in a way that
// re-perceiving aromaticity would not be: a three-membered ring is a triangle, triangle
// membership is a numbering-independent property of the graph, and the corpus check compares it
// against RDKit's SSSR-derived answer on every atom rather than assuming they agree.  If TPSA is
// ever wanted without any C++ ring reasoning at all, the column to add is
// isAtomInRingOfSize(i, 3), not another ring count.
// =============================================================================================
struct Mol {
  int n = 0, nb = 0;
  std::vector<int32_t> z, deg, nH, fchg, arom;
  std::vector<double> gast;
  std::vector<int32_t> bu, bv, bcode;

  void alloc(int nn, int nbb) {
    n = nn; nb = nbb;
    z.resize(nn); deg.resize(nn); nH.resize(nn); fchg.resize(nn); arom.resize(nn);
    gast.resize(nn);
    bu.resize(nbb); bv.resize(nbb); bcode.resize(nbb);
  }
};

// SMARTS bond-code bits, the fifth bond_i column.  Identical to criptyper's, restated here so
// this header does not silently depend on the Crippen enum keeping its values.
enum : int32_t { BC_SINGLE = 1, BC_DOUBLE = 2, BC_TRIPLE = 4, BC_AROM = 8 };

// =============================================================================================
// sha256, so the drift guard can fire without Python in the process
//
// 90 lines of well-known arithmetic.  It exists for one call: hashing the numbers in
// cpp/vsa_tables.h at load and comparing against the literal cpp/gen_vsa_tables.py recorded.
// =============================================================================================
namespace detail {

struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint8_t buf[64] = {};
  size_t len = 0, total = 0;

  static uint32_t ror(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }

  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
             ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const char* s, size_t n) {
    total += n;
    while (n) {
      size_t k = std::min(n, (size_t)64 - len);
      std::memcpy(buf + len, s, k);
      len += k; s += k; n -= k;
      if (len == 64) { block(buf); len = 0; }
    }
  }

  std::string hex() {
    uint64_t bitlen = (uint64_t)total * 8;
    uint8_t pad = 0x80;
    update((const char*)&pad, 1);
    uint8_t zero = 0;
    while (len != 56) update((const char*)&zero, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; ++i) be[i] = (uint8_t)(bitlen >> (56 - 8 * i));
    update((const char*)be, 8);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};

// The canonical rendering cpp/gen_vsa_tables.py:canonical() defines.  Bit patterns, not decimal
// text, so that "1.29" cannot hash differently in Python and in C++ because of a formatting rule.
inline void emit(std::string& s, const char* name, const double* v, int n) {
  char line[64];
  for (int i = 0; i < n; ++i) {
    uint64_t bits;
    std::memcpy(&bits, &v[i], 8);
    std::snprintf(line, sizeof line, "%s[%d]=0x%016llx\n", name, i, (unsigned long long)bits);
    s += line;
  }
}

inline std::string spec_string() {
  std::string s;
  emit(s, "RB0", vsa_tbl::RB0, vsa_tbl::MAX_Z + 1);
  // NOUTER is stored as int and hashed as double, matching float(v) on the Python side.
  {
    double tmp[vsa_tbl::MAX_Z + 1];
    for (int z = 0; z <= vsa_tbl::MAX_Z; ++z) tmp[z] = (double)vsa_tbl::NOUTER[z];
    emit(s, "NOUTER", tmp, vsa_tbl::MAX_Z + 1);
  }
  emit(s, "BOND_SCALE", vsa_tbl::BOND_SCALE, 4);
  emit(s, "LOGP_BINS", vsa_tbl::LOGP_BINS, vsa_tbl::N_LOGP_BINS);
  emit(s, "MR_BINS", vsa_tbl::MR_BINS, vsa_tbl::N_MR_BINS);
  emit(s, "CHG_BINS", vsa_tbl::CHG_BINS, vsa_tbl::N_CHG_BINS);
  emit(s, "ESTATE_BINS", vsa_tbl::ESTATE_BINS, vsa_tbl::N_ESTATE_BINS);
  emit(s, "VSA_BINS", vsa_tbl::VSA_BINS, vsa_tbl::N_VSA_BINS);
  return s;
}

}  // namespace detail

// The drift verdict, recorded at load rather than thrown: an exception escaping a static
// constructor is std::terminate, and a shared library that aborts the interpreter on import is a
// worse failure than one that explains itself.  crippen_typer.h uses the same shape.
inline std::string& drift() { static std::string d; return d; }

inline void self_check() {
  detail::Sha256 s;
  const std::string spec = detail::spec_string();
  s.update(spec.data(), spec.size());
  const std::string got = s.hex();
  if (got != vsa_tbl::SPEC_SHA256)
    drift() = "hume: cpp/vsa_tables.h no longer hashes to the spec its generator recorded. "
              "expected " + std::string(vsa_tbl::SPEC_SHA256) + " but the numbers in the table "
              "now hash to " + got + ". A bin edge, a covalent radius or an outer-electron count "
              "has been edited by hand. Regenerate with cpp/gen_vsa_tables.py under the pinned "
              "rdkit; do NOT update the literal to match.";
}

struct Init { Init() { self_check(); } };
inline const Init INIT;

// Call once from somewhere that can report -- pybind11 turns this into an ImportError.
inline void check() { if (!drift().empty()) throw std::runtime_error(drift()); }

// =============================================================================================
// Labute's approximate surface area, per atom
//
// WHAT IT ACTUALLY COMPUTES, stated precisely.  Every atom is a sphere of radius Rb0(Z) -- the
// COVALENT bond radius, NOT the van der Waals radius, which is the single most surprising thing
// about this descriptor.  The area of atom i is the area of its own sphere minus the caps cut
// off it by each bonded neighbour:
//
//     Vi[i] = pi * Ri * (4*Ri - sum over neighbours j of [ Rj^2 - (Ri - dij)^2 / dij ])
//
// where dij, the assumed internuclear distance, is NOT a geometry -- there are no coordinates
// anywhere in this calculation.  It is
//
//     dij = min( max( |Ri - Rj|, Ri + Rj - s(bond) ), Ri + Rj )
//
// i.e. the spheres are placed just touching and then pushed together by a fixed amount s that
// depends only on the bond: 0 for a single bond, 0.2 for a double, 0.3 for a triple, and 0.1
// for an aromatic one.  The min/max clamp keeps dij between "one sphere's centre on the other's
// surface" and "just touching", so a cap can never exceed a hemisphere.  So Labute ASA is a
// TOPOLOGICAL quantity dressed as a geometric one: it is a function of (elements, bond orders)
// only, which is exactly why it is cheap and why it is stable under renumbering.
//
// THE HYDROGEN TERM IS ONE SPHERE FOR THE WHOLE MOLECULE, AND IT IGNORES HOW MANY HYDROGENS
// THERE ARE.  This is the quirk that costs people reconstruction.  With includeHs set (which is
// what every caller in RDKit uses), the code walks EVERY heavy atom, subtracts an H-sized cap
// from it, and accumulates the reciprocal caps into a SINGLE scalar `hContrib` that is then
// turned into one hydrogen-sized sphere's area.  A methyl group and a quaternary carbon each
// remove exactly one H cap from themselves.  So:
//
//     LabuteASA = sum(Vi) + hContrib      -- and _CalcLabuteASAContribs returns hContrib
//                                            SEPARATELY, as a second element, not inside the
//                                            length-N vector.  Every `*_VSA` column bins only
//                                            the length-N part, so hContrib appears in
//                                            LabuteASA and in NOTHING else.
//
// TWO MORE UPSTREAM DETAILS REPRODUCED RATHER THAN TIDIED:
//
//   1. `if (fabs(hContrib) > 1e-4)`.  When the accumulated H term is negligible RDKit leaves
//      hContrib as the RAW accumulator and does not add it to the total -- it does not zero it.
//      Reproduced; it matters only for an empty molecule, but "matters only for" is how the
//      other two bugs in this repo got in.
//   2. `Vi[i] = M_PI * Ri * (4.*Ri - Vi[i])` is the C++ form.  MolSurf.py's _pyLabuteHelper
//      writes the algebraically equal `4*pi*Ri**2 - pi*Ri*Vi[i]`, which rounds DIFFERENTLY.
//      The C++ form is the one that runs: _LabuteHelper calls
//      rdMolDescriptors._CalcLabuteASAContribs, and _pyLabuteHelper is dead code.  Using the
//      Python form here loses bit-exactness on roughly half of all atoms.
//
// WHAT THE BOUNDARY CANNOT QUITE SAY.  RDKit indexes bondScaleFacts by the raw Bond::BondType
// enum and guards it with `type < 4`, so UNSPECIFIED(0) shortens by 0.1 while QUADRUPLE(4) and
// everything above it -- including DATIVE(17) -- shortens by 0.  The boundary's SMARTS bond code
// keeps only {SINGLE, DOUBLE, TRIPLE} plus an aromatic flag, so code 0 conflates UNSPECIFIED
// with "type >= 4".  This header resolves code 0 as "type >= 4" (shorten by 0), which is exact
// for every molecule that came from a SMILES parse: MolFromSmiles never emits UNSPECIFIED.  A
// census of cpp/hard.smi finds SINGLE, DOUBLE, TRIPLE, AROMATIC and DATIVE and no UNSPECIFIED.
// If molecules ever arrive from a mol block, add the raw bond type as a sixth bond_i column;
// see the wiring note at the bottom of cpp/verify_vsa.py.
// =============================================================================================

// -> per-atom contributions in `Vi` (length m.n), the separate hydrogen term in `hContrib`,
//    and the total (which is sum(Vi) + hContrib) as the return value.
inline double labute_contribs(const Mol& m, double* Vi, double& hContrib) {
  for (int i = 0; i < m.n; ++i) Vi[i] = 0.0;

  for (int b = 0; b < m.nb; ++b) {
    const int u = m.bu[b], v = m.bv[b];
    const double Ri = vsa_tbl::RB0[m.z[u]], Rj = vsa_tbl::RB0[m.z[v]];
    const int32_t code = m.bcode[b];
    double bij = Ri + Rj;
    if (code & BC_AROM) {
      bij -= vsa_tbl::BOND_SCALE[0];              // aromatic FLAG wins over the order
    } else if (code & BC_SINGLE) {
      bij -= vsa_tbl::BOND_SCALE[1];
    } else if (code & BC_DOUBLE) {
      bij -= vsa_tbl::BOND_SCALE[2];
    } else if (code & BC_TRIPLE) {
      bij -= vsa_tbl::BOND_SCALE[3];
    }                                             // else: type >= 4, no shortening
    const double dij = std::min(std::max(std::fabs(Ri - Rj), bij), Ri + Rj);
    Vi[u] += std::fma(Rj, Rj, -((Ri - dij) * (Ri - dij) / dij));
    Vi[v] += std::fma(Ri, Ri, -((Rj - dij) * (Rj - dij) / dij));
  }

  hContrib = 0.0;
  {
    const double Rj = vsa_tbl::RB0[1];
    for (int i = 0; i < m.n; ++i) {
      const double Ri = vsa_tbl::RB0[m.z[i]];
      const double bij = Ri + Rj;
      const double dij = std::min(std::max(std::fabs(Ri - Rj), bij), Ri + Rj);
      Vi[i] += std::fma(Rj, Rj, -((Ri - dij) * (Ri - dij) / dij));
      hContrib += std::fma(Ri, Ri, -((Rj - dij) * (Rj - dij) / dij));
    }
  }

  double res = 0.0;
  for (int i = 0; i < m.n; ++i) {
    const double Ri = vsa_tbl::RB0[m.z[i]];
    Vi[i] = M_PI * Ri * (4.0 * Ri - Vi[i]);
    res += Vi[i];
  }
  if (std::fabs(hContrib) > 1e-4) {              // see quirk (1) above: NOT an else-zero
    const double Rj = vsa_tbl::RB0[1];
    hContrib = M_PI * Rj * (4.0 * Rj - hContrib);
    res += hContrib;
  }
  return res;
}

// =============================================================================================
// the E-state index
//
// WHY A SECOND COPY.  src/hume_core/hume_blocks.h::estate_from() computes the same quantity, but
// it seeds the accumulator with the intrinsic state (`S[i] = I[i]` and then `S[i] += t`) whereas
// rdkit/Chem/EState/EState.py accumulates the pair terms into a ZERO vector and adds the
// intrinsic state at the very end (`res = accum + Is`).  Those are algebraically identical and
// they round differently, and every VSA_EState column is a SUM of these numbers, so the
// difference does not average out.  This copy matches RDKit's association order exactly.  It is
// also the reason MaxEStateIndex and friends are computed here rather than read from the blocks.
//
// MEASURED, because "rounds differently" is the kind of claim that is usually wrong.  Both orders
// were run against rdkit.Chem.EState.EState.EStateIndices over the first 3,000 molecules of
// cpp/hard.smi, 86,654 atoms:
//     seed with Is, then accumulate  (estate_from's order)   22,482 / 86,654 bit-exact, |d| 1.4e-14
//     accumulate into zero, add Is last (RDKit's order)      86,654 / 86,654 bit-exact
// The smallest molecule that separates them is OCC(O)COc1ccc(Cl)cc1.  74% of atoms differ in the
// last bits, so this is not an edge case to shrug at -- it would have cost bit-exactness on every
// one of the 21 EState_VSA / VSA_EState / *EStateIndex columns.  hume_blocks.h's estate_from() is
// not wrong; it is answering with a different association, and only one association reproduces
// RDKit's bits.  Left alone deliberately: it is another agent's file and its own callers are
// verified against it.
//
// `dist` is the heavy-atom topological distance matrix as an int, with -1 for unreachable.
// RDKit uses GetDistanceMatrix(useBO=0, useAtomWts=0), which stores 1e8 for unreachable and is
// then filtered by `p < 1e6`; -1 here means the same thing and is filtered the same way.
// =============================================================================================
inline void estate_indices(const Mol& m, const std::vector<int32_t>& dist, double* out) {
  std::vector<double> Is(m.n, 0.0);
  for (int i = 0; i < m.n; ++i) {
    const int d = m.deg[i];
    if (d > 0) {
      const double dv = (double)vsa_tbl::NOUTER[m.z[i]] - (double)m.nH[i];
      const int Z = m.z[i];
      const int N = Z <= 2 ? 1 : Z <= 10 ? 2 : Z <= 18 ? 3 : Z <= 36 ? 4 : Z <= 54 ? 5
                    : Z <= 86 ? 6 : 7;
      Is[i] = (4.0 / (double)(N * N) * dv + 1.0) / (double)d;
    }
  }
  for (int i = 0; i < m.n; ++i) out[i] = 0.0;          // accumulate into ZERO, add Is at the end
  for (int i = 0; i < m.n; ++i)
    for (int j = i + 1; j < m.n; ++j) {
      const int32_t d = dist[(size_t)i * m.n + j];
      if (d < 0) continue;                              // unreachable: RDKit's `p < 1e6` filter
      const double p = (double)d + 1.0;
      const double t = (Is[i] - Is[j]) / (p * p);
      out[i] += t;
      out[j] -= t;
    }
  for (int i = 0; i < m.n; ++i) out[i] += Is[i];
}

// ---------------------------------------------------------------------------------------------
// the same index, computed so that the ANSWER DOES NOT DEPEND ON THE ATOM NUMBERING
//
// WHY THIS EXISTS.  estate_indices() above reproduces RDKit bit for bit, and RDKit's answer for
// this quantity is NOT a function of the molecule.  The pair terms t_ij = (I_i - I_j)/(d_ij+1)^2
// depend only on the pair, so renumbering the atoms leaves the MULTISET of addends for each atom
// unchanged and only reorders the additions -- but floating-point addition is not associative, so
// the last bits move.  Measured: 1.6e-14 over 3,000 molecules x 3 renumberings.
//
// That would be a harmless rounding wobble if the value were then reported.  It is not: it is
// BINNED, and bin assignment is discontinuous.  On cpp/hard.smi, SEVEN atoms in five molecules
// have an E-state index whose EXACT RATIONAL VALUE IS EXACTLY A BIN EDGE:
//
//     mol  6901  atom 16   29/100  = 0.29     C=C1CC[C@H]2C(=C)CC[C@H]3C(=C)C(=O)O[C@@H]3[C@@H]12.CC(C)(C)C(=O)Nc1nc...
//     mol 45497  atom 16   29/100  = 0.29     (same sesquiterpene lactone, different partner)
//     mol 55852  atom 16   29/100  = 0.29     (ditto)
//     mol 66215  atom  9  233/200  = 1.165    CC1CNC(=O)N1c1ncc([N+](=O)[O-])s1
//     mol 98316  atom  6   77/50   = 1.54     CCCCOP(C)(=O)OCCCC
//     mol 69444  atom  4   41/20   = 2.05     CCC1(c2ccccc2)CN(C)C1=O.O=C1c2ccccc2CCC1CN1CCCCC1
//     mol 86106  atom  4   41/20   = 2.05     CCC1(c2ccccc2)CN(C)C1=O
//
// This is not a coincidence to be shrugged at.  An E-state index is a rational with a small
// denominator -- degrees, and squared topological distances -- and the bin edges were rounded to
// two or three decimals, so exact hits are common rather than rare.  For four of the seven
// (0.29 and 1.165) RDKit's binary summation lands on EITHER SIDE of the correctly rounded value
// depending on the numbering, and the column then moves by a WHOLE ATOM'S SURFACE AREA:
//
//     mol  6901   EState_VSA2 = 42.383 or 36.465,  EState_VSA3 = 23.854 or 29.772
//     mol 66215   EState_VSA4 = 11.337 or 17.534,  EState_VSA5 = 11.097 or  4.900
//
// WHAT THIS FUNCTION DOES INSTEAD.  Neumaier-compensated summation of the same addends, so the
// result is the correctly rounded value of their exact sum and therefore independent of the order
// they arrive in.  The seven atoms above then get exactly their rational value, which IS the edge,
// and upper-bound binning sends them deterministically to the higher bin -- the same answer
// RDKit gives whenever its rounding happens to land on the edge rather than below it.
//
// IT IS NOT THE DEFAULT, deliberately.  Switching it on diverges from RDKit on the molecules
// listed above, and "bit-exact against the pinned oracle" is the property the rest of this repo
// is built on; that trade is the project owner's to make, not this header's.  Build with
// -DHUME_VSA_WELLPOSED_ESTATE to select it.
// ---------------------------------------------------------------------------------------------
inline void neumaier_add(double& sum, double& comp, double v) {
  const double t = sum + v;
  comp += (std::fabs(sum) >= std::fabs(v)) ? ((sum - t) + v) : ((v - t) + sum);
  sum = t;
}

inline void estate_indices_wellposed(const Mol& m, const std::vector<int32_t>& dist, double* out,
                                     std::vector<double>& comp) {
  std::vector<double> Is(m.n, 0.0);
  for (int i = 0; i < m.n; ++i) {
    const int d = m.deg[i];
    if (d > 0) {
      const double dv = (double)vsa_tbl::NOUTER[m.z[i]] - (double)m.nH[i];
      const int Z = m.z[i];
      const int N = Z <= 2 ? 1 : Z <= 10 ? 2 : Z <= 18 ? 3 : Z <= 36 ? 4 : Z <= 54 ? 5
                    : Z <= 86 ? 6 : 7;
      Is[i] = (4.0 / (double)(N * N) * dv + 1.0) / (double)d;
    }
  }
  comp.assign(m.n, 0.0);
  for (int i = 0; i < m.n; ++i) out[i] = 0.0;
  for (int i = 0; i < m.n; ++i)
    for (int j = i + 1; j < m.n; ++j) {
      const int32_t d = dist[(size_t)i * m.n + j];
      if (d < 0) continue;
      const double p = (double)d + 1.0;
      const double t = (Is[i] - Is[j]) / (p * p);
      neumaier_add(out[i], comp[i], t);
      neumaier_add(out[j], comp[j], -t);
    }
  for (int i = 0; i < m.n; ++i) {
    neumaier_add(out[i], comp[i], Is[i]);
    out[i] += comp[i];
  }
}

// =============================================================================================
// binning
//
// std::upper_bound is bisect_right: the index of the first edge STRICTLY GREATER than x, so a
// value equal to an edge lands in the higher bin.  Written out rather than called through
// std::upper_bound only so the NaN behaviour is explicit -- see below.
// =============================================================================================
inline int bin_of(const double* edges, int n, double x) {
  // NaN COMPARES FALSE AGAINST EVERYTHING, so every `edges[k] > x` test fails and the answer is
  // n -- the FINAL bin.  RDKit does not special-case this and neither does this: a Gasteiger
  // charge for an element with no PEOE parameters (Sn, Te and friends) is NaN, and it lands in
  // PEOE_VSA14.  Clamping it to 0.0 would put it in PEOE_VSA8 instead and would be wrong.
  // std::upper_bound on a NaN is undefined behaviour (the range stops being partitioned), so
  // this is a linear scan; n is at most 13.
  int k = 0;
  while (k < n && !(edges[k] > x)) ++k;
  return k;
}

inline void bin_add(const double* prop, const double* contrib, int n, const double* edges,
                    int nedges, double* acc) {
  for (int i = 0; i < n; ++i) acc[bin_of(edges, nedges, prop[i])] += contrib[i];
}

// =============================================================================================
// TPSA, and mordred's variant
//
// rdkit's TPSA is Code/GraphMol/Descriptors/MolSurf.cpp:getTPSAAtomContribs with the default
// includeSandP=false, i.e. nitrogen and oxygen only.  mordred's TopoPSA (the `no_only=False`
// preset, which is the one that survived dedupe) is `rdMolDescriptors.CalcTPSA(mol)` PLUS its
// OWN phosphorus and sulfur table -- and mordred's table is NOT rdkit's includeSandP=true path.
// The two differ in what they count:
//
//     rdkit    nNbrs = getDegree() minus H neighbours, and the bond counts EXCLUDE bonds to H
//     mordred  matches an exact multiset of ALL incident bonds, H bonds INCLUDED, against a
//              literal dict, and returns 0.0 for a charged or (for P) aromatic atom
//
// So a phosphorus with an explicit H neighbour is scored differently by the two.  Both are
// implemented below, separately, because both are shipped columns.
// =============================================================================================

// nNbrs/nHs/bond-type counts, shared by both TPSA variants.  `hnbr` is the number of neighbouring
// atoms with Z == 1, which both variants need and neither derives the same way.
struct TpsaCounts {
  std::vector<int> nNbrs, nHs, nSing, nDoub, nTrip, nArom, nOther, hnbr;
  // mordred counts every incident bond, including bonds to hydrogen
  std::vector<int> aSing, aDoub, aTrip, aArom, aOther;
};

inline void tpsa_counts(const Mol& m, TpsaCounts& c) {
  c.nNbrs.assign(m.n, 0); c.nHs.assign(m.n, 0);
  c.nSing.assign(m.n, 0); c.nDoub.assign(m.n, 0); c.nTrip.assign(m.n, 0);
  c.nArom.assign(m.n, 0); c.nOther.assign(m.n, 0); c.hnbr.assign(m.n, 0);
  c.aSing.assign(m.n, 0); c.aDoub.assign(m.n, 0); c.aTrip.assign(m.n, 0);
  c.aArom.assign(m.n, 0); c.aOther.assign(m.n, 0);
  for (int b = 0; b < m.nb; ++b) {
    const int u = m.bu[b], v = m.bv[b];
    const int32_t code = m.bcode[b];
    // mordred's _bond_type_count: aromatic flag first, then the order, over ALL bonds.
    for (int e = 0; e < 2; ++e) {
      const int a = e ? v : u;
      if (code & BC_AROM) c.aArom[a]++;
      else if (code & BC_SINGLE) c.aSing[a]++;
      else if (code & BC_DOUBLE) c.aDoub[a]++;
      else if (code & BC_TRIPLE) c.aTrip[a]++;
      else c.aOther[a]++;
    }
    // rdkit's loop: a bond to hydrogen is not a bond, it is a hydrogen.
    if (m.z[u] == 1) { c.nNbrs[v] -= 1; c.nHs[v] += 1; c.hnbr[v] += 1; }
    else if (m.z[v] == 1) { c.nNbrs[u] -= 1; c.nHs[u] += 1; c.hnbr[u] += 1; }
    else if (code & BC_AROM) { c.nArom[u]++; c.nArom[v]++; }
    else if (code & BC_SINGLE) { c.nSing[u]++; c.nSing[v]++; }
    else if (code & BC_DOUBLE) { c.nDoub[u]++; c.nDoub[v]++; }
    else if (code & BC_TRIPLE) { c.nTrip[u]++; c.nTrip[v]++; }
    else { c.nOther[u]++; c.nOther[v]++; }        // DATIVE and friends: rdkit's `default: break`
  }
  for (int i = 0; i < m.n; ++i) {
    c.nHs[i] += m.nH[i];
    c.nNbrs[i] += m.deg[i];
    // AN IMPLICIT HYDROGEN IS A SINGLE BOND, for the `a*` counters only.
    //
    // mordred's TopoPSA descriptor sets `explicit_hydrogens = True`, so _bond_type_count runs
    // over atom.GetBonds() of the H-ADDED molecule and every hydrogen contributes a SINGLE. Our
    // graph is heavy-atom with an implicit count, so the H bonds have to be put back or the
    // multiset never matches and the atom silently scores 0.0.
    //
    // This is not cosmetic: it was worth 38.80 per thiol sulfur. The corpus molecule that
    // exposed it is a four-cysteine peptide, off by exactly 4 x 38.80 = 155.2.
    //
    // It also explains a key that looks wrong and is not: phosphorus wants
    // {SINGLE: 3, DOUBLE: 1} WITH nH == 1, which is two heavy singles plus the hydrogen. Adding
    // nH here is what makes that key reachable, not what breaks it.
    //
    // Only the `a*` counters get this. rdkit's own TPSA loop treats a bond to hydrogen as a
    // hydrogen and not as a bond, which is the `n*` counters above -- the two conventions
    // genuinely differ and both are needed in this function.
    c.aSing[i] += m.nH[i];
  }
}

// "is atom i in a ring of size 3", which the boundary's boolean `ring` column cannot answer.
// A three-membered ring is a triangle in the graph, so this is "do two of my neighbours share a
// bond".  RDKit answers the same question off its SSSR ring info; cpp/verify_vsa.py checks the
// two agree on every nitrogen and oxygen of the corpus rather than taking it on trust, because
// SSSR is not obliged to contain every smallest cycle.
inline void in_ring3(const Mol& m, const std::vector<int32_t>& start,
                     const std::vector<int32_t>& nbr, std::vector<uint8_t>& out) {
  out.assign(m.n, 0);
  for (int i = 0; i < m.n; ++i)
    for (int p = start[i]; p < start[i + 1] && !out[i]; ++p)
      for (int q = p + 1; q < start[i + 1] && !out[i]; ++q) {
        const int a = nbr[p], b = nbr[q];
        for (int r = start[a]; r < start[a + 1]; ++r)
          if (nbr[r] == b) { out[i] = 1; break; }
      }
}

inline double tpsa(const Mol& m, const TpsaCounts& c, const std::vector<uint8_t>& r3) {
  double res = 0.0;
  for (int i = 0; i < m.n; ++i) {
    const int Z = m.z[i];
    if (Z != 7 && Z != 8) continue;
    const int nH = c.nHs[i], chg = m.fchg[i], nb = c.nNbrs[i];
    const int nS = c.nSing[i], nD = c.nDoub[i], nT = c.nTrip[i], nA = c.nArom[i];
    const bool r = r3[i] != 0;
    double tmp = -1;
    if (Z == 7) {
      switch (nb) {
        case 1:
          if (nH == 0 && chg == 0 && nT == 1) tmp = 23.79;
          else if (nH == 1 && chg == 0 && nD == 1) tmp = 23.85;
          else if (nH == 2 && chg == 0 && nS == 1) tmp = 26.02;
          else if (nH == 2 && chg == 1 && nD == 1) tmp = 25.59;
          else if (nH == 3 && chg == 1 && nS == 1) tmp = 27.64;
          break;
        case 2:
          if (nH == 0 && chg == 0 && nS == 1 && nD == 1) tmp = 12.36;
          else if (nH == 0 && chg == 0 && nT == 1 && nD == 1) tmp = 13.60;
          else if (nH == 1 && chg == 0 && nS == 2 && r) tmp = 21.94;
          else if (nH == 1 && chg == 0 && nS == 2 && !r) tmp = 12.03;
          else if (nH == 0 && chg == 1 && nT == 1 && nS == 1) tmp = 4.36;
          else if (nH == 1 && chg == 1 && nD == 1 && nS == 1) tmp = 13.97;
          else if (nH == 2 && chg == 1 && nS == 2) tmp = 16.61;
          else if (nH == 0 && chg == 0 && nA == 2) tmp = 12.89;
          else if (nH == 1 && chg == 0 && nA == 2) tmp = 15.79;
          else if (nH == 1 && chg == 1 && nA == 2) tmp = 14.14;
          break;
        case 3:
          if (nH == 0 && chg == 0 && nS == 3 && r) tmp = 3.01;
          else if (nH == 0 && chg == 0 && nS == 3 && !r) tmp = 3.24;
          else if (nH == 0 && chg == 0 && nS == 1 && nD == 2) tmp = 11.68;
          else if (nH == 0 && chg == 1 && nS == 2 && nD == 1) tmp = 3.01;
          else if (nH == 1 && chg == 1 && nS == 3) tmp = 4.44;
          else if (nH == 0 && chg == 0 && nA == 3) tmp = 4.41;
          else if (nH == 0 && chg == 0 && nS == 1 && nA == 2) tmp = 4.93;
          else if (nH == 0 && chg == 0 && nD == 1 && nA == 2) tmp = 8.39;
          else if (nH == 0 && chg == 1 && nA == 3) tmp = 4.10;
          else if (nH == 0 && chg == 1 && nS == 1 && nA == 2) tmp = 3.88;
          break;
        case 4:
          if (nH == 0 && nS == 4 && chg == 1) tmp = 0.0;
          break;
        default: break;
      }
      if (tmp < 0.0) { tmp = 30.5 - nb * 8.2 + nH * 1.5; if (tmp < 0) tmp = 0.0; }
    } else {
      switch (nb) {
        case 1:
          if (nH == 0 && chg == 0 && nD == 1) tmp = 17.07;
          else if (nH == 1 && chg == 0 && nS == 1) tmp = 20.23;
          else if (nH == 0 && chg == -1 && nS == 1) tmp = 23.06;
          break;
        case 2:
          if (nH == 0 && chg == 0 && nS == 2 && r) tmp = 12.53;
          else if (nH == 0 && chg == 0 && nS == 2 && !r) tmp = 9.23;
          else if (nH == 0 && chg == 0 && nA == 2) tmp = 13.14;
          break;
        default: break;
      }
      if (tmp < 0.0) { tmp = 28.5 - nb * 8.6 + nH * 1.5; if (tmp < 0) tmp = 0.0; }
    }
    res += tmp;
  }
  return res;
}

// mordred/TopoPSA.py, the `no_only=False` preset.  Its dict equality is an EXACT multiset match,
// so any bond it has no key for (a dative bond, say) makes every branch fail and the atom scores
// 0.0.  `aOther` carries that.
//
// TAKES THE TPSA BASE AND ACCUMULATES INTO IT, IN ATOM ORDER, because mordred does:
//
//     tpsa = CalcTPSA(mol);  for atom in mol.GetAtoms(): tpsa += contrib(atom)
//
// Summing the contributions separately and adding once at the end is the same value in exact
// arithmetic and a DIFFERENT one in floating point -- (base + c1) + c2 associates differently
// from base + (c1 + c2).  That association alone left 1,232 of 100,000 molecules off by up to
// 2.3e-13.  Matching the order costs nothing and makes the column bit-exact instead of
// approximately right, which is the standard everything else in this file is held to.
inline double topopsa_sp(const Mol& m, const TpsaCounts& c, double add) {
  for (int i = 0; i < m.n; ++i) {
    const int Z = m.z[i];
    if (Z != 15 && Z != 16) continue;
    const int nH = c.nHs[i];
    const int s = c.aSing[i], d = c.aDoub[i], t = c.aTrip[i], a = c.aArom[i], o = c.aOther[i];
    if (m.fchg[i] != 0) continue;                       // mordred: charged -> 0.0
    if (t || o) continue;                               // no key in any of mordred's dicts
    if (Z == 15) {
      if (m.arom[i]) continue;                          // mordred checks GetIsAromatic for P
      if (a) continue;
      if (nH == 1 && s == 3 && d == 1) add += 23.47;
      else if (nH == 0) {
        if (s == 3 && d == 0) add += 13.59;
        else if (s == 1 && d == 1) add += 34.14;
        else if (s == 3 && d == 1) add += 9.81;
      }
    } else {
      if (m.arom[i]) {
        if (nH == 0 && s == 0) {
          if (a == 2 && d == 0) add += 28.24;
          else if (a == 2 && d == 1) add += 21.70;
        }
      } else {
        if (a) continue;
        if (nH == 1 && s == 2 && d == 0) add += 38.80;
        else if (nH == 0) {
          if (s == 2 && d == 0) add += 25.30;
          else if (s == 0 && d == 1) add += 32.09;
          else if (s == 2 && d == 1) add += 19.21;
          else if (s == 2 && d == 2) add += 8.38;
        }
      }
    }
  }
  return add;
}

// =============================================================================================
// the row
// =============================================================================================
constexpr int N_SLOGP = vsa_tbl::N_LOGP_BINS + 1;      // 12
constexpr int N_SMR = vsa_tbl::N_MR_BINS + 1;          // 10
constexpr int N_PEOE = vsa_tbl::N_CHG_BINS + 1;        // 14
constexpr int N_ESTATE_VSA = vsa_tbl::N_ESTATE_BINS + 1;  // 11
constexpr int N_VSA_ESTATE = vsa_tbl::N_VSA_BINS + 1;     // 10

// Column order of out[]; N_COLS values per molecule.
enum {
  C_SLOGP = 0,
  C_SMR = C_SLOGP + N_SLOGP,
  C_PEOE = C_SMR + N_SMR,
  C_ESTATE_VSA = C_PEOE + N_PEOE,
  C_VSA_ESTATE = C_ESTATE_VSA + N_ESTATE_VSA,
  C_MOLLOGP = C_VSA_ESTATE + N_VSA_ESTATE,
  C_MOLMR,
  C_TPSA,
  C_TOPOPSA,
  C_LABUTE_ASA,
  C_MAX_ES,
  C_MIN_ES,
  C_MAXABS_ES,
  C_MINABS_ES,
  N_COLS
};

// Scratch reused across molecules so the timed loop does not allocate.
struct Work {
  std::vector<double> asa, logp, mr, es;
  std::vector<int32_t> start, nbr, dist, q;
  std::vector<uint8_t> r3;
  TpsaCounts tc;
  criptyper::Mol cm, ch;                 // heavy-atom mol, and the H-ADDED one MolLogP needs
  std::vector<int32_t> cur;
  std::vector<double> hlogp, hmr, comp;
};

// CSR adjacency + the all-pairs heavy-atom BFS distance matrix (-1 = unreachable).
inline void build_graph(const Mol& m, Work& W) {
  W.start.assign(m.n + 1, 0);
  for (int b = 0; b < m.nb; ++b) { W.start[m.bu[b] + 1]++; W.start[m.bv[b] + 1]++; }
  for (int i = 0; i < m.n; ++i) W.start[i + 1] += W.start[i];
  W.nbr.resize(2 * (size_t)m.nb);
  W.cur.assign(W.start.begin(), W.start.end() - 1);
  for (int b = 0; b < m.nb; ++b) {
    W.nbr[W.cur[m.bu[b]]++] = m.bv[b];
    W.nbr[W.cur[m.bv[b]]++] = m.bu[b];
  }
  W.dist.assign((size_t)m.n * m.n, -1);
  W.q.resize(m.n);
  for (int s = 0; s < m.n; ++s) {
    int32_t* d = &W.dist[(size_t)s * m.n];
    int head = 0, tail = 0;
    d[s] = 0; W.q[tail++] = s;
    while (head < tail) {
      const int u = W.q[head++];
      for (int p = W.start[u]; p < W.start[u + 1]; ++p) {
        const int v = W.nbr[p];
        if (d[v] < 0) { d[v] = d[u] + 1; W.q[tail++] = v; }
      }
    }
  }
}

// Fill a criptyper::Mol from the boundary arrays.  Byte for byte what
// src/hume_core/bindings.cpp:crippen_fill() does; kept here so this header is standalone.
inline void fill_crippen(const Mol& m, criptyper::Mol& c, std::vector<int32_t>& cur) {
  c.alloc(m.n, 2 * m.nb);
  for (int i = 0; i < m.n; ++i) {
    c.z[i] = (uint8_t)(m.z[i] > 255 ? 255 : m.z[i]);
    c.arom[i] = (uint8_t)m.arom[i];
    c.chg[i] = (int8_t)m.fchg[i];
    c.tx[i] = (uint8_t)(m.deg[i] + m.nH[i]);
    c.sh[i] = (uint8_t)m.nH[i];
  }
  for (int b = 0; b < m.nb; ++b) { c.start[m.bu[b] + 1]++; c.start[m.bv[b] + 1]++; }
  for (int i = 0; i < m.n; ++i) c.start[i + 1] += c.start[i];
  cur.assign(c.start.begin(), c.start.end() - 1);
  for (int b = 0; b < m.nb; ++b) {
    const uint8_t code = (uint8_t)m.bcode[b];
    c.nbr[cur[m.bu[b]]] = m.bv[b]; c.bcode[cur[m.bu[b]]++] = code;
    c.nbr[cur[m.bv[b]]] = m.bu[b]; c.bcode[cur[m.bv[b]]++] = code;
  }
  for (int i = 0; i < m.n; ++i) {
    int extra = 0;
    for (int e = c.start[i]; e < c.start[i + 1]; ++e) if (c.z[c.nbr[e]] == 1) extra++;
    c.sh[i] = (uint8_t)(c.sh[i] + extra);
  }
}

// The H-ADDED molecule MolLogP and MolMR sum over.
//
// Code/GraphMol/Descriptors/Crippen.cpp:calcCrippenDescriptors() calls MolOps::addHs(mol) before
// summing -- and NOTHING else in this family does, which is why MolLogP is not simply the sum of
// the vector SlogP_VSA bins.  addHs appends the new hydrogens AFTER all original atoms, grouped
// by the atom they attach to, and the sum is taken in that order; reproduced exactly, because
// floating-point addition is not associative and a 62-column exactness claim is a bit-for-bit
// claim.
//
// The derived SMARTS quantities are INVARIANT under addHs and this is worth stating because it
// looks like it cannot be: for a heavy atom, X = degree + totalNumHs gains nH from the degree and
// loses nH from the H count, and H = totalNumHs + hydrogen NEIGHBOURS moves the same nH from the
// first term to the second.  So `tx` and `sh` are copied across unchanged.  Each new hydrogen has
// X = 1 and H = 0.
inline void fill_crippen_hadded(const Mol& m, const criptyper::Mol& src, criptyper::Mol& c,
                                std::vector<int32_t>& cur) {
  int extra = 0;
  for (int i = 0; i < m.n; ++i) extra += m.nH[i];
  const int n = m.n + extra, nb = m.nb + extra;
  c.alloc(n, 2 * nb);
  for (int i = 0; i < m.n; ++i) {
    c.z[i] = src.z[i]; c.arom[i] = src.arom[i]; c.chg[i] = src.chg[i];
    c.tx[i] = src.tx[i]; c.sh[i] = src.sh[i];
  }
  std::vector<int32_t> hu(extra), hv(extra);
  int k = m.n, e = 0;
  for (int i = 0; i < m.n; ++i)
    for (int t = 0; t < m.nH[i]; ++t) {
      c.z[k] = 1; c.arom[k] = 0; c.chg[k] = 0; c.tx[k] = 1; c.sh[k] = 0;
      hu[e] = i; hv[e] = k; ++e; ++k;
    }
  for (int b = 0; b < m.nb; ++b) { c.start[m.bu[b] + 1]++; c.start[m.bv[b] + 1]++; }
  for (int b = 0; b < extra; ++b) { c.start[hu[b] + 1]++; c.start[hv[b] + 1]++; }
  for (int i = 0; i < n; ++i) c.start[i + 1] += c.start[i];
  cur.assign(c.start.begin(), c.start.end() - 1);
  for (int b = 0; b < m.nb; ++b) {
    const uint8_t code = (uint8_t)m.bcode[b];
    c.nbr[cur[m.bu[b]]] = m.bv[b]; c.bcode[cur[m.bu[b]]++] = code;
    c.nbr[cur[m.bv[b]]] = m.bu[b]; c.bcode[cur[m.bv[b]]++] = code;
  }
  for (int b = 0; b < extra; ++b) {                  // new C-H bonds: SINGLE, not aromatic
    c.nbr[cur[hu[b]]] = hv[b]; c.bcode[cur[hu[b]]++] = criptyper::B_SINGLE;
    c.nbr[cur[hv[b]]] = hu[b]; c.bcode[cur[hv[b]]++] = criptyper::B_SINGLE;
  }
}

// Compute all N_COLS values for one molecule.
inline void vsa_row(const Mol& m, Work& W, double* out) {
  if (!drift().empty()) throw std::runtime_error(drift());
  for (int i = 0; i < N_COLS; ++i) out[i] = 0.0;

  build_graph(m, W);

  W.asa.resize(m.n);
  double hContrib = 0.0;
  const double asa_total = labute_contribs(m, W.asa.data(), hContrib);
  out[C_LABUTE_ASA] = asa_total;

  fill_crippen(m, W.cm, W.cur);
  W.logp.resize(m.n); W.mr.resize(m.n);
  criptyper::contribs(W.cm, W.logp.data(), W.mr.data());

  W.es.resize(m.n);
#ifdef HUME_VSA_WELLPOSED_ESTATE
  estate_indices_wellposed(m, W.dist, W.es.data(), W.comp);
#else
  estate_indices(m, W.dist, W.es.data());
#endif

  bin_add(W.logp.data(), W.asa.data(), m.n, vsa_tbl::LOGP_BINS, vsa_tbl::N_LOGP_BINS,
          out + C_SLOGP);
  bin_add(W.mr.data(), W.asa.data(), m.n, vsa_tbl::MR_BINS, vsa_tbl::N_MR_BINS, out + C_SMR);
  bin_add(m.gast.data(), W.asa.data(), m.n, vsa_tbl::CHG_BINS, vsa_tbl::N_CHG_BINS, out + C_PEOE);
  bin_add(W.es.data(), W.asa.data(), m.n, vsa_tbl::ESTATE_BINS, vsa_tbl::N_ESTATE_BINS,
          out + C_ESTATE_VSA);
  bin_add(W.asa.data(), W.es.data(), m.n, vsa_tbl::VSA_BINS, vsa_tbl::N_VSA_BINS,
          out + C_VSA_ESTATE);

  // MolLogP / MolMR over the H-added molecule, summed in addHs order.
  fill_crippen_hadded(m, W.cm, W.ch, W.cur);
  W.hlogp.resize(W.ch.n); W.hmr.resize(W.ch.n);
  criptyper::contribs(W.ch, W.hlogp.data(), W.hmr.data());
  double lp = 0.0, mr = 0.0;
  for (int i = 0; i < W.ch.n; ++i) lp += W.hlogp[i];
  for (int i = 0; i < W.ch.n; ++i) mr += W.hmr[i];
  out[C_MOLLOGP] = lp;
  out[C_MOLMR] = mr;

  tpsa_counts(m, W.tc);
  in_ring3(m, W.start, W.nbr, W.r3);
  const double p = tpsa(m, W.tc, W.r3);
  out[C_TPSA] = p;
  out[C_TOPOPSA] = topopsa_sp(m, W.tc, p);

  // max/min over an EMPTY vector: rdkit's `max(EStateIndices(mol))` raises on a zero-atom
  // molecule, so there is no upstream answer to match.  0.0 is this header's choice and the
  // corpus contains no such molecule; noted rather than hidden.
  if (m.n > 0) {
    double mx = W.es[0], mn = W.es[0], amx = std::fabs(W.es[0]), amn = std::fabs(W.es[0]);
    for (int i = 1; i < m.n; ++i) {
      mx = std::max(mx, W.es[i]);
      mn = std::min(mn, W.es[i]);
      amx = std::max(amx, std::fabs(W.es[i]));
      amn = std::min(amn, std::fabs(W.es[i]));
    }
    out[C_MAX_ES] = mx; out[C_MIN_ES] = mn;
    out[C_MAXABS_ES] = amx; out[C_MINABS_ES] = amn;
  }
}

// Names in out[] order.  cpp/verify_vsa.py reads these so the two sides cannot disagree on which
// column is which.
inline const char* col_name(int i) {
  static char buf[16][32];
  static const char* fixed[] = {"MolLogP", "MolMR", "TPSA", "TopoPSA", "LabuteASA",
                                "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex",
                                "MinAbsEStateIndex"};
  if (i >= C_MOLLOGP) return fixed[i - C_MOLLOGP];
  const char* stem; int k;
  if (i < C_SMR) { stem = "SlogP_VSA"; k = i - C_SLOGP; }
  else if (i < C_PEOE) { stem = "SMR_VSA"; k = i - C_SMR; }
  else if (i < C_ESTATE_VSA) { stem = "PEOE_VSA"; k = i - C_PEOE; }
  else if (i < C_VSA_ESTATE) { stem = "EState_VSA"; k = i - C_ESTATE_VSA; }
  else { stem = "VSA_EState"; k = i - C_VSA_ESTATE; }
  static int slot = 0;
  char* b = buf[slot = (slot + 1) % 16];
  std::snprintf(b, 32, "%s%d", stem, k + 1);
  return b;
}

}  // namespace vsabin

#endif  // HUME_VSA_BINS_H
