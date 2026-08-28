// The last of RDKit's `rdkit_core` family: 21 columns that are not substructure counts and so do
// not belong in frag_matcher.h -- thirteen ring predicates, two whole-molecule scalars, RDKit's
// Kier flexibility index, the three Morgan fingerprint densities, and the two atom-stereo counts.
//
// SPECIFICATION IS THE C++ SOURCE, read at rdkit 2025.09.2 (house rule 1):
//   Code/GraphMol/Descriptors/Lipinski.cpp                  the 13 ring predicates, FractionCSP3,
//                                                           the two atom-stereo counts
//   Code/GraphMol/MolProps.cpp                              getAvgMolWt(onlyHeavy)
//   Code/GraphMol/Descriptors/ConnectivityDescriptors.cpp   calcPhi
//   Code/GraphMol/Fingerprints/MorganGenerator.cpp          the environment generator
//   Code/GraphMol/Fingerprints/FingerprintUtil.cpp          getConnectivityInvariants
//   Code/RDGeneral/hash/hash.hpp                            gboost's 32-bit hash_combine
// Every one of them is transcribed below, including the two places where upstream reads an atom
// twice or breaks out of a loop early; those are noted where they occur.
//
// -------------------------------------------------------------------------------------------
// THE RING PREDICATES NEED BOND RINGS, AND THE BOUNDARY CARRIES ATOM RINGS. Eleven of the
// thirteen iterate `RingInfo::bondRings()`; only `NumHeterocycles` and `NumSpiroAtoms` read
// `atomRings()`. This file derives a ring's bonds from its ATOM SET -- the bonds whose two
// endpoints are both in the set -- rather than asking for a second CSR at the boundary.
//
// That is exact, not an approximation, and it was measured rather than argued. A ring in a
// minimum cycle basis is chordless (a chord would express it as the sum of two shorter cycles,
// so it could not be in a minimum basis), and `symmetrizeSSSR` only adds further minimal cycles.
// Measured over all 100,000 molecules of cpp/hard.smi, on BOTH ring sets -- RDKit's raw
// `AtomRings()` and the repaired `_rings.rings_for()` one this file is actually fed:
//
//     rings whose induced edge count != ring size            0
//     derived bond set != RDKit's own BondRings()[i]         0
//
// so the derivation reproduces RDKit's bond rings exactly and needs no extra boundary array. The
// `compute()` below still counts what it derives and refuses a ring whose induced subgraph is
// not a cycle, so a future corpus that breaks the argument fails loudly instead of silently.
//
// THE RING SET IS THE REPAIRED ONE, which is a DELIBERATE DIVERGENCE FROM RDKit on 32 of 100,000
// molecules. `Chem.GetSymmSSSR` is not a function of the molecular graph (PORT_STATUS.md house
// rule 1 and src/hume/_rings.py); the same repair that makes mordred's 49 RingCount columns
// deterministic necessarily moves RDKit's 13 on the same molecules. Measured, per column, over
// the whole corpus:
//
//     RingCount 32, NumAliphaticRings 32, NumAliphaticCarbocycles 19, NumAliphaticHeterocycles 13,
//     NumHeterocycles 13, NumSaturatedRings 8, NumBridgeheadAtoms 7, NumSaturatedCarbocycles 5,
//     NumSaturatedHeterocycles 3; the other four never move.
//
// Taking RDKit's raw rings here instead would make these 13 columns exact against RDKit and
// INCONSISTENT with the 49 mordred ring columns computed from the same molecule in the same row
// -- two ring sets in one feature vector. cpp/verify_wiring.py grades the two populations
// separately, exactly as it already does for `constit`.
// -------------------------------------------------------------------------------------------
#ifndef HUME_RDKCORE_H
#define HUME_RDKCORE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rdkcore {

static constexpr int N_COLS = 21;

enum {
  C_RINGCOUNT = 0,
  C_NAROMRINGS,
  C_NALIPHRINGS,
  C_NSATRINGS,
  C_NAROMCARBO,
  C_NAROMHETERO,
  C_NALIPHCARBO,
  C_NALIPHHETERO,
  C_NSATCARBO,
  C_NSATHETERO,
  C_NHETEROCYCLES,
  C_NBRIDGEHEAD,
  C_NSPIRO,
  C_HEAVYMOLWT,
  C_FRACCSP3,
  C_PHI,
  C_FPD1,
  C_FPD2,
  C_FPD3,
  // APPENDED, AND THAT IS DELIBERATE: these two are the LAST columns of the LAST family in
  // bindings.cpp's layout, so every pre-existing column of hume.ALL_COLUMNS keeps the index it
  // had and an A/B of the extension across this change compares like with like rather than a
  // shifted row.
  C_NATOMSTEREO,
  C_NUNSPECATOMSTEREO,
};

inline const char *col_name(int c) {
  static const char *N[N_COLS] = {
      "RingCount",          "NumAromaticRings",     "NumAliphaticRings",
      "NumSaturatedRings",  "NumAromaticCarbocycles", "NumAromaticHeterocycles",
      "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumSaturatedCarbocycles",
      "NumSaturatedHeterocycles", "NumHeterocycles",  "NumBridgeheadAtoms",
      "NumSpiroAtoms",      "HeavyAtomMolWt",       "FractionCSP3",
      "Phi",                "FpDensityMorgan1",     "FpDensityMorgan2",
      "FpDensityMorgan3",   "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters"};
  if (c < 0 || c >= N_COLS) throw std::out_of_range("rdkcore::col_name");
  return N[c];
}

//! One molecule, in the boundary's own quantities. Nothing here is perceived by this file.
struct Mol {
  int n = 0, nb = 0;
  std::vector<int> z;       // GetAtomicNum()
  std::vector<int> deg;     // GetDegree()          -- heavy-graph degree, explicit H included
  std::vector<int> nH;      // GetTotalNumHs(false)
  std::vector<int> fchg;    // GetFormalCharge()
  std::vector<int> nring;   // RingInfo::NumAtomRings(i)
  std::vector<double> mass; // GetMass()
  std::vector<double> aw;   // PeriodicTable::getAtomicWeight(z), for Morgan's deltaMass
  std::vector<int> bu, bv;
  std::vector<int> barom;   // Bond::getIsAromatic()
  std::vector<int> btype;   // (int)Bond::getBondType(): SINGLE 1, DOUBLE 2, TRIPLE 3, AROMATIC 12
  std::vector<int> ring_off, ring_at;   // the ring CSR, atom indices LOCAL to this molecule
  // THE LEGACY STEREO PERCEPTION, and it is not the one `SPS` reads. `chirposs[i]` is
  // hasProp("_ChiralityPossible") -- set by MolOps::assignStereochemistry(cleanIt, force,
  // flagPossible) -- and `ctag[i]` is (int)Atom::getChiralTag(), CHI_UNSPECIFIED == 0. Both come
  // across the boundary from RDKit rather than being perceived here: potential-stereo perception
  // is a real subsystem (ranking, para-stereo, ring stereo) and a second implementation of it
  // would be a second answer, which is the argument that already kept hybridisation and CIP on
  // RDKit's side of the line.
  std::vector<int> chirposs;
  std::vector<int> ctag;

  int n_rings() const { return ring_off.empty() ? 0 : (int)ring_off.size() - 1; }

  void alloc(int natoms, int nbonds) {
    n = natoms;
    nb = nbonds;
    z.assign(n, 0); deg.assign(n, 0); nH.assign(n, 0); fchg.assign(n, 0);
    nring.assign(n, 0); mass.assign(n, 0.0); aw.assign(n, 0.0);
    chirposs.assign(n, 0); ctag.assign(n, 0);
    bu.assign(nb, 0); bv.assign(nb, 0); barom.assign(nb, 0); btype.assign(nb, 0);
    ring_off.assign(1, 0);
    ring_at.clear();
  }
  void add_ring(const int *atoms, int k) {
    for (int q = 0; q < k; ++q) ring_at.push_back(atoms[q]);
    ring_off.push_back((int)ring_at.size());
  }
};

//! Reused across molecules so the hot loop makes no allocation.
struct Scratch {
  std::vector<int> astamp, bstamp;       // per atom / per bond ring marker
  std::vector<int> istart, inbr, ibnd;   // atom -> incident bonds, CSR
  std::vector<int> rb_off, rb_at;        // the derived BOND rings, CSR
  std::vector<int> cnt;                  // bridgehead endpoint counts
  std::vector<std::uint8_t> hit;         // per-atom "already collected" flag
  // Morgan
  std::vector<std::uint32_t> cur, nxt, inv_pair;
  std::vector<std::uint64_t> anbh, rnbh;      // per-atom environment bitsets, `words` per atom
  std::vector<std::uint64_t> seen_w, seen_h;
  std::vector<std::uint8_t> dead;
  std::vector<int> order;
  std::vector<std::uint32_t> codes;
  std::vector<std::pair<int, std::uint32_t>> nbrinv;
  std::vector<int> layer_end;
};

// --------------------------------------------------------------------------------------------
// gboost's 32-bit hash, transcribed from Code/RDGeneral/hash/hash.hpp.
//
// `std::hash_result_t` is `std::uint32_t` there -- Landrum's portability edit -- so every seed,
// every intermediate and every fingerprint key is 32 bits wide, on every platform.
// `hash_value_signed` / `hash_value_unsigned` both reduce to a plain two's-complement cast for a
// 32-bit input: their loop runs `(digits - 1) / 32` times, which is 0 for both int32 and uint32.
// --------------------------------------------------------------------------------------------
static inline void hash_combine(std::uint32_t &seed, std::uint32_t h) {
  seed ^= (std::uint32_t)(h + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}

//! gboost::hash_value(std::pair<int32_t, uint32_t>) -- seed 0, both members folded in.
static inline std::uint32_t hash_pair(std::int32_t a, std::uint32_t b) {
  std::uint32_t s = 0;
  hash_combine(s, (std::uint32_t)a);
  hash_combine(s, b);
  return s;
}

// --------------------------------------------------------------------------------------------
// The environment bitsets.
//
// boost::dynamic_bitset's operator< compares BLOCKS from the highest index down, numerically --
// i.e. it orders two equal-length bit strings as big-endian unsigned numbers with bit i worth
// 2^i. That ordering does not depend on the block width, so the 64-bit words used here induce
// exactly the same order as boost's `unsigned long` blocks, and `allNeighborhoodsThisRound` is
// sorted the same way RDKit sorts it. The order matters: it decides which of two atoms sharing
// an environment contributes its bit and which is marked dead.
// --------------------------------------------------------------------------------------------
static inline int bit_words(int nb) { return (nb + 63) / 64; }

static inline int bitset_cmp(const std::uint64_t *a, const std::uint64_t *b, int w) {
  for (int i = w - 1; i >= 0; --i) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}

// --------------------------------------------------------------------------------------------
// out must have room for N_COLS doubles.
//
// `kappa1` / `kappa2` are the 182-block row's own Kappa1 and Kappa2 -- RDKit's, already verified
// bit-exact -- passed in rather than recomputed, because calcPhi's P2 is the same
// findAllPathsOfLengthN(mol, 2) the block row has already paid for. See the wiring note in
// src/hume_core/bindings.cpp.
// --------------------------------------------------------------------------------------------
inline void compute(const Mol &m, double kappa1, double kappa2, double *out, Scratch &S) {
  for (int c = 0; c < N_COLS; ++c) out[c] = 0.0;

  // ---- the two whole-molecule scalars, and Phi ---------------------------------------------
  // getAvgMolWt(mol, onlyHeavy=true) is a bare sum of getMass() over the atoms that are not
  // hydrogen -- NO hydrogen mass is added at all, and no multiplication happens. That matters:
  // the onlyHeavy=false path adds `getTotalNumHs() * atomicWeight(1)` per atom, which is a
  // different floating-point expression from summing an AddHs molecule's H masses one at a time,
  // and the two disagree in the last bits on most molecules. This column is neither of those.
  // The accumulation order is ATOM INDEX ORDER, which is RDKit's, and it is load-bearing at the
  // last bit.
  //
  // THE TWO "HEAVY" TESTS ARE NOT THE SAME TEST, and a dummy atom is what separates them.
  // getAvgMolWt(onlyHeavy) skips an atom when `getAtomicNum() != 1` is false, so a dummy (Z = 0)
  // CONTRIBUTES -- its mass is 0.0, so the sum is unchanged, but it is in the loop.
  // ROMol::getNumHeavyAtoms() counts `getAtomicNum() > 1`, so a dummy is NOT a heavy atom, and
  // that is the count `calcPhi` and `_FingerprintDensity` divide by. On `*CCO` RDKit gives 3, not
  // 4. Reproduced as two separate accumulators rather than one.
  double amw = 0.0;
  int heavy = 0, nC = 0, nCSP3 = 0;
  for (int i = 0; i < m.n; ++i) {
    if (m.z[i] != 1) amw += m.mass[i];
    if (m.z[i] > 1) ++heavy;
    if (m.z[i] == 6) {
      ++nC;
      if (m.deg[i] + m.nH[i] == 4) ++nCSP3;   // getTotalDegree()
    }
  }
  out[C_HEAVYMOLWT] = amw;
  out[C_FRACCSP3] = nC ? (double)nCSP3 / (double)nC : 0.0;
  out[C_PHI] = heavy ? kappa1 * kappa2 / (double)heavy : 0.0;

  // ---- the two stereo counts ---------------------------------------------------------------
  // Code/GraphMol/Descriptors/Lipinski.cpp, numAtomStereoCenters / numUnspecifiedAtomStereoCenters,
  // in full: count the atoms carrying `_ChiralityPossible`, and for the second also require
  // `getChiralTag() == CHI_UNSPECIFIED`. There is no arithmetic here and no perception -- the
  // whole descriptor is those two atom properties, which is why these columns cost the pickle
  // path nothing once molpickle.h stopped skipping the bytes that already held them.
  //
  // THIS IS THE LEGACY PERCEPTION AND `SPS` USES THE NEW ONE. The `_ChiralityPossible` set and
  // the `FindPotentialStereo` set differ on 262 of 4,000 corpus molecules, so the two must not be
  // wired from one input; src/hume_core/constit.h's `sps()` takes its own pair of arrays.
  //
  // NOTE WHAT UPSTREAM DOES NOT DO: it does not run the perception. Both functions throw unless
  // the molecule already has `_StereochemDone`, so the answer is a function of whatever
  // assignStereochemistry the caller last ran -- and `flagPossibleStereoCenters=False` (RDKit's
  // default) leaves the flag cleared and the count at 0. Reproduced as a boundary contract
  // instead: src/hume/_extract.py passes the argument on both paths.
  {
    int n_stereo = 0, n_unspec = 0;
    for (int i = 0; i < m.n; ++i) {
      if (!m.chirposs[i]) continue;
      ++n_stereo;
      if (m.ctag[i] == 0) ++n_unspec;      // Atom::CHI_UNSPECIFIED
    }
    out[C_NATOMSTEREO] = (double)n_stereo;
    out[C_NUNSPECATOMSTEREO] = (double)n_unspec;
  }

  // ---- the ring predicates -----------------------------------------------------------------
  const int R = m.n_rings();
  out[C_RINGCOUNT] = (double)R;
  if (R) {
    if ((int)S.astamp.size() < m.n) S.astamp.assign(m.n, -1);
    else std::fill(S.astamp.begin(), S.astamp.begin() + m.n, -1);
    if ((int)S.bstamp.size() < m.nb) S.bstamp.assign(m.nb, -1);
    else std::fill(S.bstamp.begin(), S.bstamp.begin() + m.nb, -1);
    if ((int)S.hit.size() < m.n) S.hit.assign(m.n, 0);
    else std::fill(S.hit.begin(), S.hit.begin() + m.n, 0);
    if ((int)S.cnt.size() < m.n) S.cnt.assign(m.n, 0);
    else std::fill(S.cnt.begin(), S.cnt.begin() + m.n, 0);

    // atom -> incident bonds, CSR, counting-sorted.
    S.istart.assign(m.n + 1, 0);
    for (int b = 0; b < m.nb; ++b) { S.istart[m.bu[b] + 1]++; S.istart[m.bv[b] + 1]++; }
    for (int i = 0; i < m.n; ++i) S.istart[i + 1] += S.istart[i];
    S.inbr.resize(2 * m.nb);
    S.ibnd.resize(2 * m.nb);
    {
      std::vector<int> cur(S.istart.begin(), S.istart.end() - 1);
      for (int b = 0; b < m.nb; ++b) {
        S.inbr[cur[m.bu[b]]] = m.bv[b]; S.ibnd[cur[m.bu[b]]++] = b;
        S.inbr[cur[m.bv[b]]] = m.bu[b]; S.ibnd[cur[m.bv[b]]++] = b;
      }
    }

    // Derive each ring's BOND list from its ATOM set. See the header note for why this is exact.
    S.rb_off.assign(1, 0);
    S.rb_at.clear();
    for (int r = 0; r < R; ++r) {
      const int b0 = m.ring_off[r], e0 = m.ring_off[r + 1];
      for (int q = b0; q < e0; ++q) S.astamp[m.ring_at[q]] = r;
      int k = 0;
      for (int q = b0; q < e0; ++q) {
        const int i = m.ring_at[q];
        for (int e = S.istart[i]; e < S.istart[i + 1]; ++e) {
          const int b = S.ibnd[e];
          if (S.astamp[S.inbr[e]] != r || S.bstamp[b] == r) continue;
          S.bstamp[b] = r;
          S.rb_at.push_back(b);
          ++k;
        }
      }
      if (k != e0 - b0)
        throw std::runtime_error(
            "rdkcore: ring " + std::to_string(r) + " has " + std::to_string(k) +
            " induced bonds over " + std::to_string(e0 - b0) +
            " atoms -- its induced subgraph is not a cycle, so RDKit's bondRings() cannot be "
            "derived from the atom set for this molecule");
      S.rb_off.push_back((int)S.rb_at.size());
    }

    for (int r = 0; r < R; ++r) {
      bool allArom = true, allSat = true, anyAliph = false, hasHet = false;
      for (int q = S.rb_off[r]; q < S.rb_off[r + 1]; ++q) {
        const int b = S.rb_at[q];
        const bool ar = m.barom[b] != 0;
        if (!ar) { allArom = false; anyAliph = true; }
        if (m.btype[b] != 1 || ar) allSat = false;      // getBondType() != SINGLE || isAromatic
        // upstream reads each atom twice here ("kind of doofy", its own comment) and asks the
        // question of the BOND ENDPOINTS rather than of the ring's atom set. On a cycle the two
        // are the same set; the transcription follows the source.
        if (m.z[m.bu[b]] != 6 || m.z[m.bv[b]] != 6) hasHet = true;
      }
      out[C_NAROMRINGS] += allArom;
      out[C_NALIPHRINGS] += anyAliph;
      out[C_NSATRINGS] += allSat;
      out[C_NAROMCARBO] += allArom && !hasHet;
      out[C_NAROMHETERO] += allArom && hasHet;
      out[C_NALIPHCARBO] += anyAliph && !hasHet;
      out[C_NALIPHHETERO] += anyAliph && hasHet;
      out[C_NSATCARBO] += allSat && !hasHet;
      out[C_NSATHETERO] += allSat && hasHet;
      // calcNumHeterocycles is the one predicate over the ATOM ring.
      bool het = false;
      for (int q = m.ring_off[r]; q < m.ring_off[r + 1]; ++q)
        if (m.z[m.ring_at[q]] != 6) { het = true; break; }
      out[C_NHETEROCYCLES] += het;
    }

    // Spiro: pairs of ATOM rings meeting in exactly one atom. Bridgehead: pairs of BOND rings
    // sharing more than one bond, and then the atoms that one of those shared bonds touches
    // exactly once -- the ends of the shared path. Both collect a SET of atoms and return its
    // size, so a molecule's atom can be counted once however many pairs nominate it.
    int n_spiro = 0, n_bridge = 0;
    for (int i = 0; i < R; ++i) {
      for (int q = m.ring_off[i]; q < m.ring_off[i + 1]; ++q) S.astamp[m.ring_at[q]] = R + i;
      for (int j = i + 1; j < R; ++j) {
        int shared = 0, which = -1;
        for (int q = m.ring_off[j]; q < m.ring_off[j + 1] && shared < 2; ++q) {
          const int a = m.ring_at[q];
          if (S.astamp[a] == R + i) { ++shared; which = a; }
        }
        if (shared == 1 && !S.hit[which]) { S.hit[which] = 1; ++n_spiro; }
      }
    }
    std::fill(S.hit.begin(), S.hit.begin() + m.n, 0);
    for (int i = 0; i < R; ++i) {
      for (int q = S.rb_off[i]; q < S.rb_off[i + 1]; ++q) S.bstamp[S.rb_at[q]] = R + i;
      for (int j = i + 1; j < R; ++j) {
        int shared = 0;
        for (int q = S.rb_off[j]; q < S.rb_off[j + 1]; ++q)
          if (S.bstamp[S.rb_at[q]] == R + i) ++shared;
        if (shared <= 1) continue;
        for (int q = S.rb_off[j]; q < S.rb_off[j + 1]; ++q) {
          const int b = S.rb_at[q];
          if (S.bstamp[b] != R + i) continue;
          S.cnt[m.bu[b]]++;
          S.cnt[m.bv[b]]++;
        }
        for (int q = S.rb_off[j]; q < S.rb_off[j + 1]; ++q) {
          const int b = S.rb_at[q];
          if (S.bstamp[b] != R + i) continue;
          for (int e = 0; e < 2; ++e) {
            const int a = e ? m.bv[b] : m.bu[b];
            if (S.cnt[a] == 1 && !S.hit[a]) { S.hit[a] = 1; ++n_bridge; }
          }
        }
        for (int q = S.rb_off[j]; q < S.rb_off[j + 1]; ++q) {
          const int b = S.rb_at[q];
          if (S.bstamp[b] != R + i) continue;
          S.cnt[m.bu[b]] = 0;
          S.cnt[m.bv[b]] = 0;
        }
      }
    }
    out[C_NSPIRO] = (double)n_spiro;
    out[C_NBRIDGEHEAD] = (double)n_bridge;
  }

  // ---- the three Morgan densities ----------------------------------------------------------
  // ONE PASS AT RADIUS 3 ANSWERS ALL THREE. The generator's state (dead atoms, seen environments,
  // current invariants, atom neighbourhoods) evolves without ever looking at the radius; the
  // radius only bounds the loop. So the environment list for radius 1 is a PREFIX of the one for
  // radius 3, and the three densities are three prefix cardinalities of one run.
  if (heavy) {
    const int n = m.n, nb = m.nb, W = bit_words(nb > 0 ? nb : 1);
    S.cur.assign(n, 0);
    S.nxt.assign(n, 0);
    S.dead.assign(n, 0);
    S.anbh.assign((std::size_t)n * W, 0);
    S.rnbh.assign((std::size_t)n * W, 0);
    S.seen_w.clear();
    S.seen_h.clear();
    S.codes.clear();
    S.layer_end.clear();

    // atom -> incident bonds, again: the ring block above only builds it when R > 0.
    S.istart.assign(n + 1, 0);
    for (int b = 0; b < nb; ++b) { S.istart[m.bu[b] + 1]++; S.istart[m.bv[b] + 1]++; }
    for (int i = 0; i < n; ++i) S.istart[i + 1] += S.istart[i];
    S.inbr.resize(2 * (std::size_t)nb);
    S.ibnd.resize(2 * (std::size_t)nb);
    {
      std::vector<int> c(S.istart.begin(), S.istart.end() - 1);
      for (int b = 0; b < nb; ++b) {
        S.inbr[c[m.bu[b]]] = m.bv[b]; S.ibnd[c[m.bu[b]]++] = b;
        S.inbr[c[m.bv[b]]] = m.bu[b]; S.ibnd[c[m.bv[b]]++] = b;
      }
    }

    // getConnectivityInvariants(mol, invars, includeRingMembership=true). The components vector
    // is a std::vector<uint32_t>, so `getFormalCharge()` and `deltaMass` -- both `int` -- are
    // converted to unsigned before they are hashed; a negative charge therefore contributes its
    // two's-complement pattern, which is reproduced by the cast here.
    for (int i = 0; i < n; ++i) {
      std::uint32_t comp[6];
      int k = 0;
      comp[k++] = (std::uint32_t)m.z[i];
      comp[k++] = (std::uint32_t)(m.deg[i] + m.nH[i]);            // getTotalDegree()
      // getTotalNumHs(true) -- implicit + explicit Hs PLUS neighbouring hydrogen ATOMS. On a
      // molecule from SMILES the second term is zero, but [2H]C([2H])([2H])O is in this corpus.
      int th = m.nH[i];
      for (int e = S.istart[i]; e < S.istart[i + 1]; ++e)
        if (m.z[S.inbr[e]] == 1) ++th;
      comp[k++] = (std::uint32_t)th;
      comp[k++] = (std::uint32_t)(std::int32_t)m.fchg[i];
      comp[k++] = (std::uint32_t)(std::int32_t)(int)(m.mass[i] - m.aw[i]);   // deltaMass
      if (m.nring[i]) comp[k++] = 1u;
      std::uint32_t s = 0;
      for (int q = 0; q < k; ++q) hash_combine(s, comp[q]);
      S.cur[i] = s;
    }
    for (int i = 0; i < n; ++i) S.codes.push_back(S.cur[i]);   // layer 0: every atom
    S.layer_end.push_back((int)S.codes.size());

    for (int layer = 0; layer < 3; ++layer) {
      S.rnbh = S.anbh;
      S.order.clear();
      for (int idx = 0; idx < n; ++idx) {
        if (S.dead[idx]) continue;
        if (m.deg[idx] == 0) { S.dead[idx] = 1; continue; }
        std::uint64_t *env = S.rnbh.data() + (std::size_t)idx * W;
        S.nbrinv.clear();
        for (int e = S.istart[idx]; e < S.istart[idx + 1]; ++e) {
          const int b = S.ibnd[e], o = S.inbr[e];
          env[b >> 6] |= 1ull << (b & 63);
          const std::uint64_t *oe = S.anbh.data() + (std::size_t)o * W;
          for (int q = 0; q < W; ++q) env[q] |= oe[q];
          S.nbrinv.emplace_back(m.btype[b], S.cur[o]);
        }
        std::sort(S.nbrinv.begin(), S.nbrinv.end());
        std::uint32_t invar = (std::uint32_t)layer;
        hash_combine(invar, S.cur[idx]);
        for (const auto &p : S.nbrinv) hash_combine(invar, hash_pair(p.first, p.second));
        S.nxt[idx] = invar;
        S.order.push_back(idx);
      }
      // std::sort over (environment, invariant, atom index). The atom index makes every tuple
      // distinct, so the unstable sort is still deterministic -- and the ORDER decides which of
      // two atoms sharing an environment contributes a bit and which one dies.
      const std::uint64_t *R0 = S.rnbh.data();
      const std::uint32_t *NX = S.nxt.data();
      std::sort(S.order.begin(), S.order.end(), [&](int a, int b) {
        const int c = bitset_cmp(R0 + (std::size_t)a * W, R0 + (std::size_t)b * W, W);
        if (c) return c < 0;
        if (NX[a] != NX[b]) return NX[a] < NX[b];
        return a < b;
      });
      for (int idx : S.order) {
        const std::uint64_t *env = R0 + (std::size_t)idx * W;
        std::uint64_t h = 1469598103934665603ull;
        for (int q = 0; q < W; ++q) { h ^= env[q]; h *= 1099511628211ull; }
        bool found = false;
        for (std::size_t t = 0; t < S.seen_h.size(); ++t) {
          if (S.seen_h[t] != h) continue;
          if (!std::memcmp(S.seen_w.data() + t * W, env, (std::size_t)W * 8)) {
            found = true;
            break;
          }
        }
        if (!found) {
          S.codes.push_back(S.nxt[idx]);
          S.seen_h.push_back(h);
          S.seen_w.insert(S.seen_w.end(), env, env + W);
        } else {
          S.dead[idx] = 1;
        }
      }
      S.cur.swap(S.nxt);
      std::fill(S.nxt.begin(), S.nxt.end(), 0u);
      S.anbh.swap(S.rnbh);
      S.layer_end.push_back((int)S.codes.size());
    }

    // len(GetNonzeroElements()) is the number of DISTINCT 32-bit codes, and the radius-r answer
    // is the distinct count of the first layer_end[r] of them.
    std::vector<std::uint32_t> tmp;
    for (int r = 1; r <= 3; ++r) {
      tmp.assign(S.codes.begin(), S.codes.begin() + S.layer_end[r]);
      std::sort(tmp.begin(), tmp.end());
      const int u = (int)(std::unique(tmp.begin(), tmp.end()) - tmp.begin());
      out[C_FPD1 + r - 1] = (double)u / (double)heavy;
    }
  }
  // heavy == 0 -> _FingerprintDensity returns 0.0 (an explicit branch in Descriptors.py), which
  // is the zero this row already carries.
}

}  // namespace rdkcore

#endif  // HUME_RDKCORE_H
