// `SPS` (rdkit's normalised spacial score) and, underneath it, RDKit's NEW potential-stereo
// perception in C++.
//
// WHAT THIS FILE IS FOR. The arithmetic of SPS is four small integers per atom multiplied
// together; it has been in constit.h all along. What it could not answer C++-side was its stereo
// term, because rdkit/Chem/SpacialScore.py reads
//
//     Chem.FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=True)
//
// which is `Chirality::findPotentialStereo` -- rdkit's NEW perception -- and the pickle carries
// only the LEGACY `_ChiralityPossible` flag. THE TWO ARE NOT THE SAME SET. Measured here on the
// 20,000-molecule dedupe2 corpus, parsed the way src/hume/_extract.py parses it: the legacy flag
// set and the new perception's `Atom_Tetrahedral` set DIFFER ON 662 MOLECULES (3.3%). The legacy
// flag systematically MISSES ring/bridgehead centres -- `CC12C3C4C3C1C(O)C24` is legacy
// {3,4,5,8} against new {1,2,3,4,5,6,8} -- so substituting one for the other is not a rounding,
// it is a different descriptor. See NOTES_sps.md for the full measurement.
//
// So this file is a port of the perception, not of the score. It follows, call for call:
//
//   Code/GraphMol/FindStereo.cpp   Chirality::findPotentialStereo / runCleanup,
//                                  initAtomInfo, initBondInfo, flagRingStereo,
//                                  updateAtoms, updateBonds, getStereoInfo,
//                                  isAtomPotentialTetrahedralCenter,
//                                  isAtomPotentialNontetrahedralCenter,
//                                  isBondPotentialStereoBond
//   Code/GraphMol/new_canon.{h,cpp} Canon::rankFragmentAtoms, AtomCompareFunctor,
//                                  SpecialSymmetryAtomCompareFunctor, bondholder,
//                                  RefinePartitions, ActivatePartitions,
//                                  compareRingAtomsConcerningNumNeighbors
//   Code/RDGeneral/hanoiSort.h     hanoi / hanoisort
//   Code/GraphMol/Chirality.cpp    MolOps::findPotentialStereoBonds, buildCIPInvariants,
//                                  iterateCIPRanks, findAtomNeighborsHelper
//   Code/GraphMol/QueryOps.cpp     queryIsAtomBridgehead
//   Code/RDGeneral/utils.h         countSwapsToInterconvert
//
// read from the rdkit 2025.09.2 sources, which is the version in .venv. The port is line for
// line where it matters; where it is not, the difference is named in a comment beginning
// DIVERGENCE or QUIRK.
//
// THE CALL SPS ACTUALLY MAKES IS ON A COPY WITH `FindPotentialStereoBonds` ALREADY RUN.
// SpacialScore.py does
//
//     molCp = Chem.Mol(mol); rdmolops.FindPotentialStereoBonds(molCp)
//
// FIRST, and then asks both of its questions of that copy. That matters twice over: the bond
// term of SPS is `bond.GetStereo() != STEREONONE` on the copy (so STEREOANY counts), and the
// STEREOANY marks change the bond symbols the canonical ranking sees, so they can move the ATOM
// answer too. Both are reproduced: `findPotentialStereoBonds` runs first, into a local copy of
// the bond-stereo vector, and the perception reads that copy.
//
// WHAT IS ASSUMED ABOUT THE INPUT, and it is the one real limitation. Everything here is a
// function of the quantities in `Mol` below, all of which the boundary already carries. Three
// things RDKit consults that a SMILES-derived molecule never has are NOT carried and are treated
// as absent:
//   * `Bond::getBondDir() == UNKNOWN` (a squiggle bond) and the `_UnknownStereo` atom/bond
//     property. Both come from mol files, never from SMILES. They would make an atom's or bond's
//     stereo `StereoSpecified::Unknown` instead of `Unspecified`.
//   * `Bond::BondDir::EITHERDOUBLE` (a crossed double bond), likewise mol-file only.
//   * atom map numbers on DUMMY atoms, which `AtomCompareFunctor` uses even when
//     `includeAtomMaps` is false (`df_useAtomMapsOnDummies` defaults true). SMILES with mapped
//     dummies would need that field.
// A mol-file-sourced corpus would need those three across the boundary. For SMILES they are
// provably absent and the perception is exact; see verify_sps.py.
#ifndef HUME_SPS_H
#define HUME_SPS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sps {

static constexpr int N_COLS = 1;

enum { C_SPS = 0 };

inline const char *col_name(int c) {
  static const char *N[N_COLS] = {"SPS"};
  if (c < 0 || c >= N_COLS) throw std::out_of_range("sps::col_name");
  return N[c];
}

// ---------------------------------------------------------------------------------------------
// rdkit enum values, spelled out. These are the integers the boundary arrays already carry;
// naming them here is what stops a `12` in a switch from meaning something else after an rdkit
// upgrade. Checked against rdkit 2025.09.2's Python-visible enums.
// ---------------------------------------------------------------------------------------------
enum : int {
  BT_UNSPECIFIED = 0, BT_SINGLE = 1, BT_DOUBLE = 2, BT_TRIPLE = 3, BT_AROMATIC = 12,
  BT_DATIVE = 17, BT_ZERO = 21
};
enum : int {
  BS_NONE = 0, BS_ANY = 1, BS_Z = 2, BS_E = 3, BS_CIS = 4, BS_TRANS = 5,
  BS_ATROPCW = 6, BS_ATROPCCW = 7
};
enum : int {
  CHI_UNSPECIFIED = 0, CHI_TETRAHEDRAL_CW = 1, CHI_TETRAHEDRAL_CCW = 2, CHI_OTHER = 3,
  CHI_TETRAHEDRAL = 4, CHI_ALLENE = 5, CHI_SQUAREPLANAR = 6, CHI_TRIGONALBIPYRAMIDAL = 7,
  CHI_OCTAHEDRAL = 8
};
enum : int { HYBS_SP = 2, HYBS_SP2 = 3, HYBS_SP3 = 4 };

//! Chirality::minRingSizeForDoubleBondStereo, Code/GraphMol/Chirality.h.
static constexpr int MIN_RING_SIZE_FOR_DB_STEREO = 8;

//! Chirality::nonTetrahedralStereoDefaultVal. rdkit reads RDK_ENABLE_NONTETRAHEDRAL_STEREO from
//! the environment and defaults to true; hume does not set it, so true is the value in force.
//! It is not a no-op: it is what makes a four-coordinate Si or P a POSSIBLE tetrahedral centre
//! even with no chiral tag, which `isAtomPotentialTetrahedralCenter` alone would already do --
//! but it also makes 5- and 6-coordinate heavy atoms `Atom_TrigonalBipyramidal` /
//! `Atom_Octahedral`, which SPS then does NOT count, because SPS reads only `Atom_Tetrahedral`.
static constexpr bool ALLOW_NONTETRAHEDRAL = true;

// ---------------------------------------------------------------------------------------------
// The molecule, in the boundary's own quantities. Nothing here is perceived by this file except
// stereo; every field is a value RDKit already computed and src/hume/_extract.py already sends.
// ---------------------------------------------------------------------------------------------
struct Mol {
  int n = 0, nb = 0;
  std::vector<int> z;        // GetAtomicNum()
  std::vector<int> deg;      // GetDegree()            -- heavy-graph degree
  std::vector<int> nH;       // GetTotalNumHs(false)
  std::vector<int> fchg;     // GetFormalCharge()
  std::vector<int> hyb;      // (int)GetHybridization()
  std::vector<int> arom;     // GetIsAromatic()
  std::vector<int> nring;    // RingInfo::NumAtomRings(i)
  std::vector<int> tval;     // GetTotalValence()
  std::vector<int> ctag;     // (int)GetChiralTag()
  std::vector<int> iso;      // GetIsotope()
  //! The LEGACY CIP LABEL, +1 for "R", -1 for "S", 0 for none -- `atom_i`'s `cip` column exactly
  //! as src/hume/_extract.py writes it. It is an INPUT to `rerankAtoms` below, which is the one
  //! place the potential-stereo perception needs to know about assigned R/S. See that function
  //! for why leaving it out is a wrong SPS on two corpus molecules.
  std::vector<int> cip;
  std::vector<int> bu, bv;
  std::vector<int> btype;    // (int)Bond::getBondType()
  std::vector<int> barom;    // Bond::getIsAromatic()
  std::vector<int> bconj;    // Bond::getIsConjugated()
  std::vector<int> bstereo;  // (int)Bond::getStereo() AS PARSED -- before FindPotentialStereoBonds
  //! The ring SET as a CSR, atom indices local to this molecule. Atom order within a ring need
  //! NOT be cyclic on the way in: `prepRings` walks the induced cycle itself, because
  //! `flagRingStereo` indexes rings as sequences ("the atom half-way round") and a set would be
  //! the wrong shape for it.
  std::vector<int> ring_off, ring_at;

  int n_rings() const { return ring_off.empty() ? 0 : (int)ring_off.size() - 1; }

  void alloc(int natoms, int nbonds) {
    n = natoms;
    nb = nbonds;
    z.assign(n, 0); deg.assign(n, 0); nH.assign(n, 0); fchg.assign(n, 0);
    hyb.assign(n, 0); arom.assign(n, 0); nring.assign(n, 0); tval.assign(n, 0);
    ctag.assign(n, 0); iso.assign(n, 0); cip.assign(n, 0);
    bu.assign(nb, 0); bv.assign(nb, 0); btype.assign(nb, 0); barom.assign(nb, 0);
    bconj.assign(nb, 0); bstereo.assign(nb, 0);
    ring_off.assign(1, 0);
    ring_at.clear();
  }

  void add_ring(const int *atoms, int sz) {
    ring_at.insert(ring_at.end(), atoms, atoms + sz);
    ring_off.push_back((int)ring_at.size());
  }
};

namespace detail {

// ---------------------------------------------------------------------------------------------
// Element tables. Symbols are needed because `getAtomCompareSymbol` builds a STRING and the
// canonical ranking compares those strings LEXICOGRAPHICALLY -- "Br" sorts before "C" and "C"
// before "Cl", which no ordering on atomic number reproduces. Most-common-isotope is needed by
// `buildCIPInvariants`. Both are transcriptions of rdkit's PeriodicTable for Z = 0..118.
// ---------------------------------------------------------------------------------------------
static constexpr int N_Z = 119;

inline const char *elementSymbol(int z) {
  static const char *S[N_Z] = {
      "*",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
      "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
      "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
      "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
      "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
      "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
      "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
      "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
  return (z >= 0 && z < N_Z) ? S[z] : "*";
}

inline int mostCommonIsotope(int z) {
  static const int M[N_Z] = {
      0,   1,   4,   7,   9,   11,  12,  14,  16,  19,  20,  23,  24,  27,  28,  31,  32,  35,
      40,  39,  40,  45,  48,  51,  52,  55,  56,  59,  58,  63,  64,  69,  74,  75,  80,  79,
      84,  85,  88,  89,  90,  93,  98,  97,  102, 103, 106, 107, 114, 115, 120, 121, 130, 127,
      132, 133, 138, 139, 140, 141, 142, 145, 152, 153, 158, 159, 164, 165, 166, 169, 174, 175,
      180, 181, 184, 187, 192, 193, 195, 197, 202, 205, 208, 209, 209, 210, 222, 223, 226, 227,
      232, 231, 238, 236, 238, 241, 243, 247, 249, 252, 257, 258, 259, 262, 267, 268, 271, 270,
      269, 278, 281, 281, 285, 284, 289, 288, 293, 292, 294};
  return (z >= 0 && z < N_Z) ? M[z] : 0;
}

//! Code/GraphMol/Bond.cpp getTwiceBondType(). Only the orders a sanitised organic molecule can
//! carry are listed; anything else is 0, which is what rdkit's `default` does for IONIC/ZERO and
//! what its UNSPECIFIED case does.
inline uint8_t twiceBondType(int bt) {
  switch (bt) {
    case BT_SINGLE: return 2;
    case BT_DOUBLE: return 4;
    case BT_TRIPLE: return 6;
    case 4: return 8;      // QUADRUPLE
    case 5: return 10;     // QUINTUPLE
    case 6: return 12;     // HEXTUPLE
    case 7: return 3;      // ONEANDAHALF
    case 8: return 5;      // TWOANDAHALF
    case 9: return 7;      // THREEANDAHALF
    case 10: return 9;     // FOURANDAHALF
    case 11: return 11;    // FIVEANDAHALF
    case BT_AROMATIC: return 3;
    case 14: return 0;     // HYDROGEN
    case 15: return 0;     // THREECENTER
    case 16: return 2;     // DATIVEONE
    case BT_DATIVE: return 2;
    case 18: return 2;     // DATIVEL
    case 19: return 2;     // DATIVER
    default: return 0;     // UNSPECIFIED, IONIC, ZERO, OTHER
  }
}

//! Code/RDGeneral/utils.h countSwapsToInterconvert(). `probe` is taken by value there and
//! mutated; here the caller owns a scratch copy.
inline unsigned countSwaps(const unsigned *ref, unsigned *probe, int len) {
  unsigned nSwaps = 0;
  for (int i = 0; i < len; ++i) {
    if (probe[i] != ref[i]) {
      int j = i;
      while (j < len && probe[j] != ref[i]) ++j;
      if (j >= len) return nSwaps;   // rdkit's CHECK_INVARIANT; unreachable for a permutation
      std::swap(probe[i], probe[j]);
      ++nSwaps;
    }
  }
  return nSwaps;
}

// =============================================================================================
// Graph + ring scaffolding
// =============================================================================================

//! Incidence in BOND INDEX ORDER, which is what `mol.getAtomBonds(atom)` yields for a molecule
//! whose bonds were added in index order -- and rdkit's boost adjacency_list does yield insertion
//! order. THE ORDER IS PART OF THE ANSWER: `getStereoInfo(atom)` builds `controllingAtoms` in it
//! and then counts the swaps to sorted order, so a permuted incidence list flips CW to CCW.
struct Graph {
  int n = 0, nb = 0;
  std::vector<int> start;    // n + 1
  std::vector<int> nbrAtom;  // 2 * nb
  std::vector<int> nbrBond;  // 2 * nb

  int degree(int i) const { return start[i + 1] - start[i]; }

  void build(const Mol &m) {
    n = m.n; nb = m.nb;
    start.assign(n + 1, 0);
    for (int b = 0; b < nb; ++b) { start[m.bu[b] + 1]++; start[m.bv[b] + 1]++; }
    for (int i = 0; i < n; ++i) start[i + 1] += start[i];
    nbrAtom.assign(2 * nb, 0);
    nbrBond.assign(2 * nb, 0);
    std::vector<int> cur(start.begin(), start.end() - 1);
    for (int b = 0; b < nb; ++b) {
      const int u = m.bu[b], v = m.bv[b];
      nbrAtom[cur[u]] = v; nbrBond[cur[u]] = b; cur[u]++;
      nbrAtom[cur[v]] = u; nbrBond[cur[v]] = b; cur[v]++;
    }
  }

  int bondBetween(int a, int b) const {
    for (int e = start[a]; e < start[a + 1]; ++e)
      if (nbrAtom[e] == b) return nbrBond[e];
    return -1;
  }
};

//! The ring set, in the shape rdkit's RingInfo is asked for here: rings as CYCLIC SEQUENCES of
//! atoms and of bonds, per-atom and per-bond ring counts, the rings each atom belongs to in
//! ascending ring index (RingInfo::atomMembers), and the ring sizes an atom sits in.
//!
//! THE CYCLIC ORDER IS RECOVERED, NOT CARRIED. The boundary CSR is the repaired ring set from
//! src/hume/_rings.py; its atom order is whatever the repaired perception produced. Every use of
//! sequence order in `flagRingStereo` is invariant under rotation and, with one exception noted
//! there, under reversal, so walking the induced cycle here answers the same question.
struct RingSet {
  int nr = 0;
  std::vector<int> ptr;          // nr + 1
  std::vector<int> at;           // cyclic atoms
  std::vector<int> bd;           // cyclic bonds, ring r's bond i joins at[ptr[r]+i], at[..+i+1]
  std::vector<int> numAtomRings, numBondRings;
  std::vector<int> minBondRing;      // smallest ring a bond is in, INT_MAX if none
  std::vector<int> memb_ptr, memb;   // rings per atom, ascending ring index
  std::vector<char> bondInRing;      // scratch: "is this bond in the ring being walked"

  int size(int r) const { return ptr[r + 1] - ptr[r]; }

  bool atomInRingOfSize(int a, int sz) const {
    for (int k = memb_ptr[a]; k < memb_ptr[a + 1]; ++k)
      if (size(memb[k]) == sz) return true;
    return false;
  }

  void build(const Mol &m, const Graph &g, std::vector<char> &inRing, std::vector<char> &used) {
    nr = m.n_rings();
    ptr.assign(nr + 1, 0);
    for (int r = 0; r < nr; ++r) ptr[r + 1] = m.ring_off[r + 1];
    at.assign(m.ring_at.size(), -1);
    bd.assign(m.ring_at.size(), -1);
    numAtomRings.assign(m.n, 0);
    numBondRings.assign(m.nb, 0);
    minBondRing.assign(m.nb, std::numeric_limits<int>::max());
    bondInRing.assign(m.nb ? m.nb : 1, 0);
    inRing.assign(m.n, 0);
    used.assign(m.n, 0);
    for (int r = 0; r < nr; ++r) {
      const int lo = ptr[r], hi = ptr[r + 1], sz = hi - lo;
      for (int k = lo; k < hi; ++k) { inRing[m.ring_at[k]] = 1; used[m.ring_at[k]] = 0; }
      // Walk the induced cycle. An SSSR ring is a simple cycle, so from any start each step has
      // exactly one unvisited ring neighbour until the last, which closes back to the start.
      int cur = m.ring_at[lo];
      for (int i = 0; i < sz; ++i) {
        at[lo + i] = cur;
        used[cur] = 1;
        int nxt = -1, nxtBond = -1;
        for (int e = g.start[cur]; e < g.start[cur + 1]; ++e) {
          const int w = g.nbrAtom[e];
          if (!inRing[w]) continue;
          if (i + 1 == sz) {
            if (w == at[lo]) { nxt = w; nxtBond = g.nbrBond[e]; break; }
          } else if (!used[w]) {
            nxt = w; nxtBond = g.nbrBond[e]; break;
          }
        }
        if (nxt < 0)
          throw std::runtime_error(
              "hume._core sps: the ring CSR handed to sps.h is not a simple cycle -- ring " +
              std::to_string(r) + " of " + std::to_string(sz) +
              " atoms could not be walked. SPS's potential-stereo perception indexes rings as "
              "sequences (flagRingStereo asks for the atom half-way round), so a ring that is "
              "not a cycle in the bond graph cannot be used.");
        bd[lo + i] = nxtBond;
        numBondRings[nxtBond]++;
        if (sz < minBondRing[nxtBond]) minBondRing[nxtBond] = sz;
        numAtomRings[cur]++;
        cur = nxt;
      }
      for (int k = lo; k < hi; ++k) inRing[m.ring_at[k]] = 0;
    }
    memb_ptr.assign(m.n + 1, 0);
    for (int r = 0; r < nr; ++r)
      for (int k = ptr[r]; k < ptr[r + 1]; ++k) memb_ptr[at[k] + 1]++;
    for (int i = 0; i < m.n; ++i) memb_ptr[i + 1] += memb_ptr[i];
    memb.assign(memb_ptr[m.n], 0);
    std::vector<int> cur2(memb_ptr.begin(), memb_ptr.end() - 1);
    for (int r = 0; r < nr; ++r)
      for (int k = ptr[r]; k < ptr[r + 1]; ++k) memb[cur2[at[k]]++] = r;   // ascending r
  }
};

//! Code/GraphMol/QueryOps.cpp queryIsAtomBridgehead(), ported whole. "at least three ring bonds,
//! all ring bonds in a ring which shares at least two bonds with another ring involving this
//! atom". Used by `isAtomPotentialTetrahedralCenter` for three-coordinate nitrogen only.
inline bool isAtomBridgehead(const Mol &m, const Graph &g, const RingSet &ri, int a,
                             std::vector<char> &atomRingBonds,
                             std::vector<char> &bondsInRingI,
                             std::vector<char> &ringsOverlap) {
  if (g.degree(a) < 3) return false;
  if (ri.nr == 0) return false;
  int nRingBonds = 0;
  for (int e = g.start[a]; e < g.start[a + 1]; ++e) {
    const int b = g.nbrBond[e];
    if (ri.numBondRings[b]) { atomRingBonds[b] = 1; ++nRingBonds; }
  }
  if (nRingBonds < 3) {
    for (int e = g.start[a]; e < g.start[a + 1]; ++e) atomRingBonds[g.nbrBond[e]] = 0;
    return false;
  }
  std::fill(ringsOverlap.begin(), ringsOverlap.begin() + ri.nr, 0);
  bool res = true;
  for (int i = 0; i < ri.nr && res; ++i) {
    std::fill(bondsInRingI.begin(), bondsInRingI.begin() + m.nb, 0);
    bool atomInRingI = false;
    for (int k = ri.ptr[i]; k < ri.ptr[i + 1]; ++k) {
      bondsInRingI[ri.bd[k]] = 1;
      if (atomRingBonds[ri.bd[k]]) atomInRingI = true;
    }
    if (!atomInRingI) continue;
    for (int j = i + 1; j < ri.nr; ++j) {
      int overlap = 0;
      bool atomInRingJ = false;
      for (int k = ri.ptr[j]; k < ri.ptr[j + 1]; ++k) {
        const int bidx = ri.bd[k];
        if (atomRingBonds[bidx]) atomInRingJ = true;
        if (bondsInRingI[bidx]) ++overlap;
        if (overlap >= 2 && atomInRingJ) { ringsOverlap[i] = 1; ringsOverlap[j] = 1; break; }
      }
    }
    if (!ringsOverlap[i]) res = false;
  }
  for (int e = g.start[a]; e < g.start[a + 1]; ++e) atomRingBonds[g.nbrBond[e]] = 0;
  return res;
}

// =============================================================================================
// The LEGACY CIP ranks, which `MolOps::findPotentialStereoBonds` reads.
//
// WHY A SECOND, DIFFERENT RANKING LIVES IN THIS FILE. `findPotentialStereoBonds` predates the
// new perception and asks the legacy question: "do the two neighbours at this end of the double
// bond have different `_CIPRank`?". A freshly parsed molecule already carries `_CIPRank` on its
// atoms (set during sanitisation's legacy stereo perception) and rdkit reuses it rather than
// recomputing -- but the pickle does not carry it, so it is recomputed here from the SAME
// function rdkit would have called, `Chirality::assignAtomCIPRanks`. Only rank EQUALITY and the
// argmax within one end are ever consumed, and both are verified against rdkit downstream.
// =============================================================================================

//! Code/GraphMol/Chirality.cpp buildCIPInvariants(). Labute's invariant: 7 bits of atomic number,
//! 10 bits of isotope deviation, 10 bits of atom-map number. No H count, no charge -- rdkit's own
//! comment explains that including them gives bad rankings.
inline void buildCIPInvariants(const Mol &m, std::vector<int64_t> &invars) {
  static const int nMassBits = 10;
  static const int maxMass = 1 << nMassBits;
  invars.assign(m.n, 0);
  for (int i = 0; i < m.n; ++i) {
    int64_t num = m.z[i] % 128;
    int mass = 0;
    if (m.iso[i]) {
      mass = m.iso[i] - mostCommonIsotope(m.z[i]);
      if (mass >= 0) mass += 1;
    }
    mass += maxMass / 2;
    if (mass < 0) mass = 0; else mass = mass % maxMass;
    int64_t inv = num;
    inv = (inv << nMassBits) | (int64_t)mass;
    const int mapnum = 0;   // molAtomMapNumber absent -> rdkit's (-1 + 1) % 1024 == 0
    inv = (inv << 10) | (int64_t)mapnum;
    invars[i] = inv;
  }
}

//! Code/GraphMol/Chirality.cpp iterateCIPRanks(mol, invars, ranks, seedWithInvars).
//! A Morgan-style refinement over integer vectors, sorted lexicographically. Reproduced with the
//! same termination rule -- rdkit stops when the rank count stops growing, or after
//! numAtoms/2 + 1 iterations, and NOT at a fixed point, so a faster stable-partition refinement
//! would be a different function.
struct CipScratch {
  std::vector<std::vector<int>> ent;
  std::vector<int> order;
  std::vector<std::pair<int, int>> needsSorting, feat, perAtom;
  std::vector<int64_t> invars;
};

inline void iterateCIPRanks(const Mol &m, const Graph &g, const std::vector<int64_t> &invars,
                            std::vector<unsigned> &ranks, bool seedWithInvars, CipScratch &S) {
  const int nAtoms = m.n;
  ranks.assign(nAtoms, 0);
  if (!nAtoms) return;

  if ((int)S.ent.size() < nAtoms) S.ent.resize(nAtoms);
  std::vector<std::vector<int>> &cip = S.ent;
  for (int i = 0; i < nAtoms; ++i) {
    cip[i].clear();
    cip[i].reserve(16);
    cip[i].push_back((int)invars[i]);
  }

  // `sortableEntries` is rdkit's vector of (cip entry, atom index) sorted by the entry.
  std::vector<int> &order = S.order;
  order.resize(nAtoms);
  for (int i = 0; i < nAtoms; ++i) order[i] = i;
  auto lessEntry = [&cip](int a, int b) { return cip[a] < cip[b]; };
  std::sort(order.begin(), order.end(), lessEntry);

  std::vector<std::pair<int, int>> &needsSorting = S.needsSorting;
  unsigned numRanks = 0;
  // findSegmentsToResort + recomputeRanks, fused: assign consecutive ranks to the sorted list,
  // recording the [first,last] index spans of runs of equal entries.
  auto segmentAndRank = [&]() {
    needsSorting.clear();
    numRanks = (unsigned)nAtoms;
    int runningRank = 0;
    int cur = order[0];
    ranks[cur] = 0;
    bool inEqual = false;
    for (int i = 1; i < nAtoms; ++i) {
      const int e = order[i];
      if (cip[cur] == cip[e]) {
        ranks[e] = (unsigned)runningRank;
        --numRanks;
        if (!inEqual) { inEqual = true; needsSorting.emplace_back(i - 1, 0); }
      } else {
        ++runningRank;
        ranks[e] = (unsigned)runningRank;
        cur = e;
        if (inEqual) { needsSorting.back().second = i; inEqual = false; }
      }
    }
    if (inEqual) needsSorting.back().second = nAtoms - 1;
  };
  segmentAndRank();

  // rdkit's seeding, and the comment above it is worth keeping: "in general one should avoid the
  // temptation to use invariants here, those lead to incorrect answers" -- for the FIRST ranking.
  // The rerank pass, which comes in with stereo-supplemented invariants, does the opposite.
  for (int i = 0; i < nAtoms; ++i) {
    if (seedWithInvars) {
      cip[i][0] = (int)invars[i];
    } else {
      cip[i][0] = m.z[i];
      cip[i].push_back((int)ranks[i]);
    }
  }
  const int cipRankIndex = seedWithInvars ? 1 : 2;

  // computeBondFeatures(): per atom, (2 x bond order, neighbour index) for each incident bond,
  // with the chiral-phosphorus special case that a double bond to a 3- or 4-coordinate P counts
  // ONCE rather than four times.
  std::vector<std::pair<int, int>> &feat = S.feat;   // (count, nbr), grouped per atom by g.start
  feat.assign(2 * (size_t)m.nb, {0, 0});
  for (int i = 0; i < nAtoms; ++i) {
    for (int e = g.start[i]; e < g.start[i + 1]; ++e) {
      const int nbr = g.nbrAtom[e], b = g.nbrBond[e];
      bool chiralP = false;
      if (m.btype[b] == BT_DOUBLE && m.z[nbr] == 15) {
        const int d = g.degree(nbr);
        chiralP = (d == 3 || d == 4);
      }
      feat[e] = {chiralP ? 1 : (int)twiceBondType(m.btype[b]), nbr};
    }
  }

  const unsigned maxIts = (unsigned)(nAtoms / 2 + 1);
  unsigned numIts = 0;
  long long lastNumRanks = -1;

  std::vector<std::pair<int, int>> &perAtom = S.perAtom;
  while (!needsSorting.empty() && numIts < maxIts &&
         (lastNumRanks < 0 || (unsigned)lastNumRanks < numRanks)) {
    for (int i = 0; i < nAtoms; ++i) {
      const int lo = g.start[i], hi = g.start[i + 1];
      perAtom.assign(feat.begin() + lo, feat.begin() + hi);
      // rdkit sorts numNeighbors + 1 entries -- one past the filled ones, a default {0,0} pair.
      // Its count is 0 so it contributes nothing; reproducing the extra element is unnecessary.
      std::sort(perAtom.begin(), perAtom.end(),
                [&ranks](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                  return ranks[a.second] > ranks[b.second];
                });
      auto &ce = cip[i];
      for (const auto &p : perAtom) ce.insert(ce.end(), (size_t)p.first, (int)ranks[p.second] + 1);
      ce.insert(ce.end(), (size_t)m.nH[i], 0);
    }
    lastNumRanks = (long long)numRanks;
    for (const auto &seg : needsSorting)
      std::sort(order.begin() + seg.first, order.begin() + seg.second + 1, lessEntry);
    segmentAndRank();
    if ((unsigned)lastNumRanks != numRanks) {
      for (int i = 0; i < nAtoms; ++i) {
        cip[i].resize(cipRankIndex + 1);
        cip[i][cipRankIndex] = (int)ranks[i];
      }
    }
    ++numIts;
  }
}

//! Code/GraphMol/Chirality.cpp assignAtomCIPRanks() -- the constitution-only ranking.
inline void assignAtomCIPRanks(const Mol &m, const Graph &g, std::vector<unsigned> &ranks,
                               CipScratch &S) {
  buildCIPInvariants(m, S.invars);
  iterateCIPRanks(m, g, S.invars, ranks, /*seedWithInvars=*/false, S);
}

//! Code/GraphMol/Chirality.cpp rerankAtoms() -- "reassign atom ranks by supplementing the current
//! ranks with information about known chirality".
//!
//! THIS PASS IS NOT OPTIONAL AND LEAVING IT OUT IS A WRONG `SPS` ON REAL MOLECULES.
//! `MolOps::findPotentialStereoBonds` does NOT recompute CIP ranks when the atoms already carry
//! `_CIPRank` -- and after `MolFromSmiles` they do, left there by the legacy stereo perception,
//! which reranks with the R/S and E/Z labels folded in. A freshly computed constitution-only
//! ranking is a DIFFERENT vector: measured on the 20,000-molecule corpus, clearing `_CIPRank`
//! before `FindPotentialStereoBonds` changes its answer on exactly 2 molecules, both
//! 3-substituted tropane-like bicycles where the two bridge arms are equivalent constitutionally
//! and distinguished only by the bridgehead stereocentres. Both were wrong by 1.76 and 1.50 SPS
//! units before this pass existed.
//!
//! HOW MANY TIMES rdkit APPLIES IT. `legacyStereoPerception` loops
//! assignAtomChiralCodes / assignBondStereoCodes / rerankAtoms while codes are still being newly
//! assigned, so the usual molecule gets exactly one rerank -- one round assigns every label, the
//! next assigns nothing new and the loop exits. This applies it once, and only when there IS a
//! label to fold in, which is the same guard as rdkit's `keepGoing`.
inline void rerankAtoms(const Mol &m, const Graph &g, std::vector<unsigned> &ranks,
                        CipScratch &S) {
  int factor = 100;
  while (factor < m.n) factor *= 10;
  std::vector<int64_t> invars((size_t)m.n, 0);
  for (int i = 0; i < m.n; ++i) {
    int64_t v = (int64_t)ranks[i] * factor;
    if (m.cip[i] < 0) v += 10;         // "S"  (the boundary stores S as -1, R as +1)
    else if (m.cip[i] > 0) v += 20;    // "R"
    for (int e = g.start[i]; e < g.start[i + 1]; ++e) {
      const int b = g.nbrBond[e];
      if (m.btype[b] != BT_DOUBLE) continue;
      if (m.bstereo[b] == BS_E) v += 1;
      else if (m.bstereo[b] == BS_Z) v += 2;
    }
    invars[i] = v;
  }
  iterateCIPRanks(m, g, invars, ranks, /*seedWithInvars=*/true, S);
}

//! Code/GraphMol/Chirality.cpp findAtomNeighborsHelper(mol, atom, refBond, nbrs, checkDir=false,
//! includeAromatic=true) -- SINGLE and AROMATIC bonds only, the reference bond excluded.
inline void findAtomNeighbors(const Mol &m, const Graph &g, int atom, int refBond,
                              int out[4], int &cnt) {
  cnt = 0;
  for (int e = g.start[atom]; e < g.start[atom + 1]; ++e) {
    const int b = g.nbrBond[e];
    if (b == refBond) continue;
    if (m.btype[b] == BT_SINGLE || m.btype[b] == BT_AROMATIC) {
      if (cnt < 4) out[cnt] = g.nbrAtom[e];
      ++cnt;
    }
  }
}

//! Code/GraphMol/Chirality.cpp MolOps::findPotentialStereoBonds(mol, cleanIt=false).
//!
//! Marks non-ring double bonds STEREOANY when the CIP ranks at each end distinguish the
//! substituents. Already-assigned E/Z bonds are LEFT ALONE (the `getStereo() == STEREONONE`
//! guard), which is why a specified double bond keeps its label and its stereo atoms.
//!
//! Writes into `bstereo` -- a COPY of the boundary's bond-stereo vector. SpacialScore.py runs
//! this on `Chem.Mol(mol)` for exactly that reason and src/hume/_extract.py's comment says so:
//! letting it touch the caller's molecule would corrupt the assigned-stereo column that other
//! descriptors read.
inline void findPotentialStereoBonds(const Mol &m, const Graph &g, const RingSet &ri,
                                     const std::vector<unsigned> &ranks,
                                     std::vector<int> &bstereo,
                                     std::vector<int> &stereoAtomA,
                                     std::vector<int> &stereoAtomB) {
  for (int b = 0; b < m.nb; ++b) {
    if (m.btype[b] != BT_DOUBLE) continue;
    if (ri.numBondRings[b]) continue;                  // ring double bonds ignored, per the FIX
    if (bstereo[b] != BS_NONE) continue;               // cleanIt == false
    const int beg = m.bu[b], end = m.bv[b];
    const int dbeg = g.degree(beg), dend = g.degree(end);
    if (!((dbeg == 2 || dbeg == 3) && (dend == 2 || dend == 3))) continue;
    int bn[4], en[4], nbeg = 0, nend = 0;
    findAtomNeighbors(m, g, beg, b, bn, nbeg);
    findAtomNeighbors(m, g, end, b, en, nend);
    if (nbeg == 0 || nend == 0) continue;
    int sa0 = -1, sa1 = -1;
    if (nbeg == 2 && nend == 2) {
      if (ranks[bn[0]] != ranks[bn[1]] && ranks[en[0]] != ranks[en[1]]) {
        sa0 = ranks[bn[0]] > ranks[bn[1]] ? bn[0] : bn[1];
        sa1 = ranks[en[0]] > ranks[en[1]] ? en[0] : en[1];
      }
    } else if (nbeg == 2) {
      if (ranks[bn[0]] != ranks[bn[1]]) {
        sa0 = ranks[bn[0]] > ranks[bn[1]] ? bn[0] : bn[1];
        sa1 = en[0];
      }
    } else if (nend == 2) {
      if (ranks[en[0]] != ranks[en[1]]) {
        sa0 = bn[0];
        sa1 = ranks[en[0]] > ranks[en[1]] ? en[0] : en[1];
      }
    } else {
      sa0 = bn[0];
      sa1 = en[0];
    }
    if (sa0 >= 0 && sa1 >= 0) {
      stereoAtomA[b] = sa0;
      stereoAtomB[b] = sa1;
      bstereo[b] = BS_ANY;
    }
  }
}

// =============================================================================================
// Canon::rankFragmentAtoms, as called from FindStereo.cpp:
//
//     breakTies = false, includeChirality = false, includeIsotopes = false,
//     includeAtomMaps = false, includeChiralPresence = false, useRingStereo = false,
//     atomSymbols and bondSymbols BOTH SUPPLIED, every atom and bond in play.
//
// Those flags collapse `AtomCompareFunctor::basecomp` to three tests -- current class, degree,
// and the atom SYMBOL STRING, which returns even when equal and so short-circuits atomic number,
// isotope, H count and charge. That is not a simplification made here; it is what the `if
// (dp_atoms[i].p_symbol && dp_atoms[j].p_symbol) { ... return 0; }` block in new_canon.h does.
// The symbol carries isotope, element and charge, so nothing is lost except the H COUNT, which
// genuinely does not enter this ranking.
// =============================================================================================

//! THE SYMBOL STRINGS ARE INTERNED, and that is the one liberty this port takes with
//! new_canon.h. rdkit compares `std::string`s inside the innermost loop of the refinement --
//! `bondholder::compare` and `AtomCompareFunctor::basecomp` both do -- and those comparisons are
//! the hot path. Here each round assigns every DISTINCT symbol string a dense integer code by
//! sorting the distinct strings, so `code(a) < code(b)` if and only if `a < b` lexicographically
//! and `code(a) == code(b)` if and only if the strings are equal. Every comparison the algorithm
//! makes therefore has the same sign it had before, and the refinement is bit-for-bit the same
//! partition; only the comparison is an int compare instead of a memcmp. Measured on the 20,000
//! corpus molecules: it and the 8-byte sort key in `refreshCodes` together take the perception
//! from 20.2 to 13.9 us/mol, and 0 molecules change answer.
struct BondHolder {
  int symCode = 0;
  int bondType = 0;
  unsigned nbrSymClass = 0;
  unsigned nbrIdx = 0;
  //! The BOND index this holder came from. It is needed because the holders are SORTED in place
  //! within each atom's slice, so after the first round `bonds[e]` is no longer the graph's edge
  //! slot `e` and `g->nbrBond[e]` is the wrong bond. rdkit keeps the same field on `bondholder`
  //! and for the same reason -- `initFragmentCanonAtoms` reattaches bond symbols through it.
  unsigned bondIdx = 0;
  // `bondStereo` is 0 for every holder here: `makeBondHolder` only reads the bond's stereo when
  // includeChirality is true, and it is false on this path. It is kept as a named constant rather
  // than a field so the compare below reads like new_canon.h's.
};

//! new_canon.h bondholder::compare with div == 1 and bondStereo == 0 on both sides.
inline int cmpBond(const BondHolder &x, const BondHolder &y) {
  if (x.symCode < y.symCode) return -1;
  if (x.symCode > y.symCode) return 1;
  if (x.bondType < y.bondType) return -1;
  if (x.bondType > y.bondType) return 1;
  // bondStereo is equal (both zero) and compareStereo is guarded by `x.bondStereo && y.bondStereo`
  const unsigned d = x.nbrSymClass - y.nbrSymClass;
  if (d) return (int)d;
  return 0;
}

inline bool bondGreater(const BondHolder &a, const BondHolder &b) { return cmpBond(a, b) > 0; }

struct CanonAtom {
  int index = 0;
  int degree = 0;
  int symCode = 0;
  int stamp = -1;                        // see Canon::updateNeighborIndex
  int bstart = 0, bend = 0;              // slice of Canon::bonds
  std::vector<int> neighborNum;          // SpecialSymmetryAtomCompareFunctor only
  std::vector<int> revisitedNeighbors;   // SpecialSymmetryAtomCompareFunctor only
};

struct Canon {
  const Mol *m = nullptr;
  const Graph *g = nullptr;
  const RingSet *ri = nullptr;
  std::vector<CanonAtom> atoms;
  std::vector<BondHolder> bonds;         // atom i's holders live in [bstart, bend)
  bool useNbrs = false;
  bool useSpecialSymmetry = false;

  // scratch for rankWithFunctor
  std::vector<int> order, count, next, changed, tempbuf;
  std::vector<char> touched;
  std::vector<int> permA, permB, codeA, codeB;   // symbol interning
  std::vector<uint64_t> keyA, keyB;
  std::vector<int> touchedList;

  void init(const Mol &mm, const Graph &gg, const RingSet &rr) {
    m = &mm; g = &gg; ri = &rr;
    atoms.assign(mm.n, CanonAtom());
    bonds.assign(2 * (size_t)mm.nb, BondHolder());
    for (int i = 0; i < mm.n; ++i) {
      atoms[i].index = i;
      atoms[i].degree = gg.degree(i);
      atoms[i].bstart = gg.start[i];
      atoms[i].bend = gg.start[i + 1];
      for (int e = gg.start[i]; e < gg.start[i + 1]; ++e) {
        const int b = gg.nbrBond[e];
        BondHolder &h = bonds[e];
        h.bondType = mm.barom[b] ? BT_AROMATIC : mm.btype[b];
        h.nbrIdx = (unsigned)gg.nbrAtom[e];
        h.bondIdx = (unsigned)b;
        h.nbrSymClass = 0;
      }
    }
    order.assign(mm.n, 0);
    count.assign(mm.n, 0);
    next.assign(mm.n, 0);
    changed.assign(mm.n, 0);
    touched.assign(mm.n, 0);
    tempbuf.assign(mm.n ? mm.n : 1, 0);
    codeA.assign(mm.n, 0);
    codeB.assign(mm.nb ? mm.nb : 1, 0);
  }

  //! The first 8 bytes of a symbol as a big-endian unsigned, zero-padded. Byte 0 of a C string
  //! is smaller than every character that can appear in a symbol, so for symbols of 8 characters
  //! or fewer -- which is nearly all of them -- comparing these integers IS comparing the
  //! strings. Longer ones tie on the key and fall through to a real string compare.
  static uint64_t symKey(const std::string &s) {
    uint64_t k = 0;
    const size_t nn = s.size() < 8 ? s.size() : 8;
    for (size_t i = 0; i < nn; ++i) k = (k << 8) | (uint64_t)(unsigned char)s[i];
    for (size_t i = nn; i < 8; ++i) k <<= 8;
    return k;
  }

  //! Re-intern the atom and bond symbol strings, which `updateAtoms` / `updateBonds` rewrite
  //! between rounds. Called once per canonicalisation round, before any comparison.
  void refreshCodes(const std::vector<std::string> &atomSymbols,
                    const std::vector<std::string> &bondSymbols) {
    const int n = m->n, nb = m->nb;
    keyA.resize(n);
    for (int i = 0; i < n; ++i) keyA[i] = symKey(atomSymbols[i]);
    permA.resize(n);
    for (int i = 0; i < n; ++i) permA[i] = i;
    auto lessA = [&](int a, int b) {
      if (keyA[a] != keyA[b]) return keyA[a] < keyA[b];
      if (atomSymbols[a].size() <= 8 && atomSymbols[b].size() <= 8) return false;
      return atomSymbols[a] < atomSymbols[b];
    };
    std::sort(permA.begin(), permA.end(), lessA);
    int c = 0;
    for (int k = 0; k < n; ++k) {
      if (k && lessA(permA[k - 1], permA[k])) ++c;
      codeA[permA[k]] = c;
    }
    keyB.resize(nb);
    for (int i = 0; i < nb; ++i) keyB[i] = symKey(bondSymbols[i]);
    permB.resize(nb);
    for (int i = 0; i < nb; ++i) permB[i] = i;
    auto lessB = [&](int a, int b) {
      if (keyB[a] != keyB[b]) return keyB[a] < keyB[b];
      if (bondSymbols[a].size() <= 8 && bondSymbols[b].size() <= 8) return false;
      return bondSymbols[a] < bondSymbols[b];
    };
    std::sort(permB.begin(), permB.end(), lessB);
    c = 0;
    for (int k = 0; k < nb; ++k) {
      if (k && lessB(permB[k - 1], permB[k])) ++c;
      codeB[permB[k]] = c;
    }
    for (int i = 0; i < n; ++i) atoms[i].symCode = codeA[i];
    for (int e = 0; e < 2 * nb; ++e) bonds[e].symCode = codeB[bonds[e].bondIdx];
    ++stamp;   // the holder sort key changed, so every memo is stale
    // No sort here on purpose: every comparison in the refinement runs `updateNeighborIndex` on
    // both operands first, which re-sorts with the current classes, so the resting order of a
    // holder slice is never read.
  }

  //! new_canon.cpp updateAtomNeighborIndex: refresh each holder's neighbour class and re-sort.
  //!
  //! MEMOISED ON A STAMP, and the memo is exact rather than approximate. rdkit calls this on both
  //! operands of EVERY comparison, and a comparison happens inside `hanoisort`, during which no
  //! atom's `index` changes -- `refinePartitions` only relabels after the sort returns. So within
  //! one sort pass the second and later calls for the same atom recompute a value that cannot
  //! have moved. `stamp` is bumped exactly where an `index` is written, so a memo can never be
  //! read across a relabel. Worth roughly a third of the perception's time on large molecules.
  int stamp = 0;

  void updateNeighborIndex(int i) {
    if (atoms[i].stamp == stamp) return;
    atoms[i].stamp = stamp;
    for (int e = atoms[i].bstart; e < atoms[i].bend; ++e)
      bonds[e].nbrSymClass = (unsigned)atoms[bonds[e].nbrIdx].index;
    std::sort(bonds.begin() + atoms[i].bstart, bonds.begin() + atoms[i].bend, bondGreater);
  }

  //! AtomCompareFunctor::basecomp, with this call site's flags.
  int basecomp(int i, int j) {
    if (atoms[i].index < atoms[j].index) return -1;
    if (atoms[i].index > atoms[j].index) return 1;
    if (atoms[i].degree < atoms[j].degree) return -1;
    if (atoms[i].degree > atoms[j].degree) return 1;
    if (atoms[i].symCode < atoms[j].symCode) return -1;
    if (atoms[i].symCode > atoms[j].symCode) return 1;
    return 0;
  }

  int compareMain(int i, int j) {
    const int v = basecomp(i, j);
    if (v) return v;
    if (!useNbrs) return 0;
    updateNeighborIndex(i);
    updateNeighborIndex(j);
    const int ni = atoms[i].bend - atoms[i].bstart, nj = atoms[j].bend - atoms[j].bstart;
    const int nm = ni < nj ? ni : nj;
    for (int k = 0; k < nm; ++k) {
      const int c = cmpBond(bonds[atoms[i].bstart + k], bonds[atoms[j].bstart + k]);
      if (c) return c;
    }
    if (ni < nj) return -1;
    if (ni > nj) return 1;
    return 0;
  }

  //! SpecialSymmetryAtomCompareFunctor.
  int compareSpecial(int i, int j) {
    if (atoms[i].neighborNum < atoms[j].neighborNum) return -1;
    if (atoms[i].neighborNum > atoms[j].neighborNum) return 1;
    if (atoms[i].revisitedNeighbors < atoms[j].revisitedNeighbors) return -1;
    if (atoms[i].revisitedNeighbors > atoms[j].revisitedNeighbors) return 1;
    updateNeighborIndex(i);
    updateNeighborIndex(j);
    const int ni = atoms[i].bend - atoms[i].bstart, nj = atoms[j].bend - atoms[j].bstart;
    const int nm = ni < nj ? ni : nj;
    for (int k = 0; k < nm; ++k) {
      const int c = cmpBond(bonds[atoms[i].bstart + k], bonds[atoms[j].bstart + k]);
      if (c) return c;
    }
    if (ni < nj) return -1;
    if (ni > nj) return 1;
    return 0;
  }

  int compare(int i, int j) {
    return useSpecialSymmetry ? compareSpecial(i, j) : compareMain(i, j);
  }

  // ---- Code/RDGeneral/hanoiSort.h, ported whole -------------------------------------------
  bool hanoi(int *base, int nel, int *temp) {
    int *b1, *b2, *t1, *t2, *s1, *s2, *ptr;
    int n1, n2, result;
    if (nel == 1) {
      count[base[0]] = 1;
      return false;
    } else if (nel == 2) {
      n1 = base[0]; n2 = base[1];
      const int stat = (changed[n1] || changed[n2]) ? compare(n1, n2) : 0;
      if (stat == 0) { count[n1] = 2; count[n2] = 0; return false; }
      count[n1] = 1; count[n2] = 1;
      if (stat > 0) { base[0] = n2; base[1] = n1; }
      return false;
    }
    n1 = nel / 2;
    n2 = nel - n1;
    b1 = base; t1 = temp; b2 = base + n1; t2 = temp + n1;
    if (hanoi(b1, n1, t1)) {
      s2 = hanoi(b2, n2, t2) ? t2 : b2;
      result = false; ptr = base; s1 = t1;
    } else {
      s2 = hanoi(b2, n2, t2) ? t2 : b2;
      result = true; ptr = temp; s1 = b1;
    }
    while (true) {
      const int stat = (changed[*s1] || changed[*s2]) ? compare(*s1, *s2) : 0;
      const int len1 = count[*s1], len2 = count[*s2];
      if (stat == 0) {
        count[*s1] = len1 + len2;
        count[*s2] = 0;
        memmove(ptr, s1, (size_t)len1 * sizeof(int));
        ptr += len1;
        n1 -= len1;
        if (n1 == 0) {
          if (ptr != s2) memmove(ptr, s2, (size_t)n2 * sizeof(int));
          return result != 0;
        }
        s1 += len1;
        memmove(ptr, s2, (size_t)len2 * sizeof(int));
        ptr += len2;
        n2 -= len2;
        if (n2 == 0) { memmove(ptr, s1, (size_t)n1 * sizeof(int)); return result != 0; }
        s2 += len2;
      } else if (stat < 0) {
        memmove(ptr, s1, (size_t)len1 * sizeof(int));
        ptr += len1;
        n1 -= len1;
        if (n1 == 0) {
          if (ptr != s2) memmove(ptr, s2, (size_t)n2 * sizeof(int));
          return result != 0;
        }
        s1 += len1;
      } else {
        memmove(ptr, s2, (size_t)len2 * sizeof(int));
        ptr += len2;
        n2 -= len2;
        if (n2 == 0) { memmove(ptr, s1, (size_t)n1 * sizeof(int)); return result != 0; }
        s2 += len2;
      }
    }
  }

  void hanoisort(int *base, int nel) {
    if (nel <= 0) return;
    if ((int)tempbuf.size() < nel) tempbuf.assign(nel, 0);
    if (hanoi(base, nel, tempbuf.data())) memmove(base, tempbuf.data(), (size_t)nel * sizeof(int));
  }

  // ---- new_canon.h CreateSinglePartition / ActivatePartitions / RefinePartitions -----------
  void createSinglePartition() {
    const int n = m->n;
    for (int i = 0; i < n; ++i) { atoms[i].index = 0; order[i] = i; count[i] = 0; }
    if (n) count[0] = n;
    ++stamp;
  }

  void activatePartitions(int &activeset) {
    const int n = m->n;
    activeset = -1;
    for (int i = 0; i < n; ++i) next[i] = -2;
    int i = 0;
    do {
      const int j = order[i];
      if (count[j] > 1) { next[j] = activeset; activeset = j; i += count[j]; }
      else ++i;
    } while (i < n);
    for (int k = 0; k < n; ++k) changed[order[k]] = 1;
  }

  void refinePartitions(int &activeset) {
    int symclass = 0;
    while (activeset != -1) {
      int partition = activeset;
      activeset = next[partition];
      next[partition] = -2;
      const int len = count[partition];
      const int offset = atoms[partition].index;
      int *start = order.data() + offset;
      hanoisort(start, len);
      for (int k = 0; k < len; ++k) changed[start[k]] = 0;
      int index = start[0];
      for (int i = count[index]; i < len; ++i) {
        index = start[i];
        if (count[index]) symclass = offset + i;
        atoms[index].index = symclass;
        for (int e = g->start[index]; e < g->start[index + 1]; ++e) changed[g->nbrAtom[e]] = 1;
      }
      ++stamp;   // an `index` was (possibly) written; every memo above is now stale
      // rdkit marks the touched partitions in an n-long bitmap and then SCANS ALL n POSITIONS
      // in ascending order to enqueue them. The list is collected and sorted instead, which
      // visits the same partitions in the same ascending order -- so `activeset` is built in
      // exactly the same sequence -- without the O(n) scan per partition processed. On the 55+
      // stratum that scan was most of `rank`.
      touchedList.clear();
      index = start[0];
      for (int i = count[index]; i < len; ++i) {
        index = start[i];
        for (int e = g->start[index]; e < g->start[index + 1]; ++e) {
          const int p = atoms[g->nbrAtom[e]].index;
          if (!touched[p]) { touched[p] = 1; touchedList.push_back(p); }
        }
      }
      std::sort(touchedList.begin(), touchedList.end());
      for (const int ii : touchedList) {
        const int p = order[ii];
        if (count[p] > 1 && next[p] == -2) { next[p] = activeset; activeset = p; }
        touched[ii] = 0;
      }
    }
  }

  //! new_canon.cpp compareRingAtomsConcerningNumNeighbors(). A BFS over the RING SUBGRAPH from
  //! every ring atom, recording per level the number of newly seen neighbours and the sorted
  //! multiset of "revisited" neighbour counts. Only ever reached through the
  //! SpecialSymmetryAtomCompareFunctor branch below.
  void computeRingNeighborNums(std::vector<char> &visited, std::vector<char> &lastLevel,
                               std::vector<char> &curLevel, std::vector<int> &revisited,
                               std::vector<int> &queue, std::vector<int> &nextLevel,
                               std::vector<int> &tmp) {
    const int nAtoms = m->n;
    for (int idx = 0; idx < nAtoms; ++idx) {
      if (ri->numAtomRings[idx] < 1) continue;
      atoms[idx].neighborNum.clear();
      atoms[idx].revisitedNeighbors.clear();
      queue.assign(1, idx);
      std::fill(visited.begin(), visited.end(), 0);
      std::fill(lastLevel.begin(), lastLevel.end(), 0);
      std::fill(curLevel.begin(), curLevel.end(), 0);
      std::fill(revisited.begin(), revisited.end(), 0);
      while (!queue.empty()) {
        int numLevelNbrs = 0;
        nextLevel.clear();
        for (size_t qi = 0; qi < queue.size(); ++qi) {
          const int nidx = queue[qi];
          if (ri->numAtomRings[nidx] < 1) continue;
          lastLevel[nidx] = 1;
          visited[nidx] = 1;
          for (int e = g->start[nidx]; e < g->start[nidx + 1]; ++e) {
            const int iidx = g->nbrAtom[e];
            if (!visited[iidx]) {
              curLevel[iidx] = 1;
              ++numLevelNbrs;
              visited[iidx] = 1;
              nextLevel.push_back(iidx);
            }
          }
        }
        for (int i = 0; i < nAtoms; ++i) {
          if (!curLevel[i]) continue;
          for (int e = g->start[i]; e < g->start[i + 1]; ++e) {
            const int jidx = g->nbrAtom[e];
            if (curLevel[jidx] || lastLevel[jidx]) revisited[jidx] += 1;
          }
        }
        std::fill(lastLevel.begin(), lastLevel.end(), 0);
        for (int i = 0; i < nAtoms; ++i) if (curLevel[i]) lastLevel[i] = 1;
        std::fill(curLevel.begin(), curLevel.end(), 0);
        tmp.clear();
        for (int i = 0; i < nAtoms; ++i) if (revisited[i] > 0) tmp.push_back(revisited[i]);
        std::sort(tmp.begin(), tmp.end());
        tmp.push_back(-1);
        atoms[idx].revisitedNeighbors.insert(atoms[idx].revisitedNeighbors.end(),
                                             tmp.begin(), tmp.end());
        std::fill(revisited.begin(), revisited.end(), 0);
        atoms[idx].neighborNum.push_back(numLevelNbrs);
        atoms[idx].neighborNum.push_back(-1);
        queue = nextLevel;
      }
    }
  }

  //! new_canon.cpp detail::rankWithFunctor(ftor, breakTies=false, useSpecial=true,
  //! useChirality=false, includeRingStereo=false). The SpecialChirality pass is unreachable with
  //! useChirality false; the SpecialSymmetry pass is NOT, and it is what separates the atoms of
  //! highly symmetrical cages that the main functor leaves tied.
  void rank(std::vector<unsigned> &res, std::vector<char> &v1, std::vector<char> &v2,
            std::vector<char> &v3, std::vector<int> &v4, std::vector<int> &v5,
            std::vector<int> &v6, std::vector<int> &v7) {
    const int nAts = m->n;
    res.assign(nAts, 0);
    if (!nAts) return;
    std::fill(changed.begin(), changed.end(), 1);
    std::fill(touched.begin(), touched.end(), 0);
    int activeset = -1;
    createSinglePartition();
    useNbrs = true;
    useSpecialSymmetry = false;
    activatePartitions(activeset);
    refinePartitions(activeset);

    bool ties = false;
    unsigned symRingAtoms = 0, ringAtoms = 0;
    bool branchingRingAtom = false;
    for (int i = 0; i < nAts; ++i) {
      if (ri->numAtomRings[order[i]]) {
        if (count[order[i]] > 2) symRingAtoms += (unsigned)count[order[i]];
        ++ringAtoms;
        if (ri->numAtomRings[order[i]] > 1 && count[order[i]] > 1) branchingRingAtom = true;
      }
      if (!count[i]) ties = true;
    }
    if (ties && ringAtoms > 0 &&
        (float)symRingAtoms / (float)ringAtoms > 0.5f && branchingRingAtom) {
      v1.assign(nAts, 0); v2.assign(nAts, 0); v3.assign(nAts, 0); v4.assign(nAts, 0);
      computeRingNeighborNums(v1, v2, v3, v4, v5, v6, v7);
      useSpecialSymmetry = true;
      activatePartitions(activeset);
      refinePartitions(activeset);
      useSpecialSymmetry = false;
    }
    for (int i = 0; i < nAts; ++i) res[order[i]] = (unsigned)atoms[order[i]].index;
  }
};

// =============================================================================================
// FindStereo.cpp
// =============================================================================================

static const unsigned NOATOM = std::numeric_limits<unsigned>::max();

enum class SSpec { Unspecified, Specified, Unknown };
enum class SType { Unspecified, Atom_Tetrahedral, Atom_SquarePlanar, Atom_TrigonalBipyramidal,
                   Atom_Octahedral, Bond_Double, Bond_Cumulene_Even, Bond_Atropisomer };
enum class SDesc { None, Tet_CW, Tet_CCW, Bond_Cis, Bond_Trans, Bond_AtropCW, Bond_AtropCCW };

struct StereoInfo {
  SType type = SType::Unspecified;
  SSpec specified = SSpec::Unspecified;
  unsigned centeredOn = NOATOM;
  SDesc descriptor = SDesc::None;
  unsigned controlling[4] = {NOATOM, NOATOM, NOATOM, NOATOM};
  int nControlling = 0;
};

//! Chirality::detail::bondAffectsAtomChirality.
inline bool bondAffectsAtomChirality(const Mol &m, int b, int atom) {
  const int bt = m.btype[b];
  if (bt == BT_UNSPECIFIED || bt == BT_ZERO || (bt == BT_DATIVE && m.bu[b] == atom)) return false;
  return true;
}

//! Chirality::detail::getAtomNonzeroDegree.
inline int nonzeroDegree(const Mol &m, const Graph &g, int a) {
  int res = 0;
  for (int e = g.start[a]; e < g.start[a + 1]; ++e)
    if (bondAffectsAtomChirality(m, g.nbrBond[e], a)) ++res;
  return res;
}

//! Chirality::detail::has_protium_neighbor -- an H ATOM in the graph with isotope 0.
inline bool hasProtiumNeighbor(const Mol &m, const Graph &g, int a) {
  for (int e = g.start[a]; e < g.start[a + 1]; ++e) {
    const int w = g.nbrAtom[e];
    if (m.z[w] == 1 && m.iso[w] == 0) return true;
  }
  return false;
}

//! Chirality::detail::isAtomPotentialNontetrahedralCenter.
inline bool isPotentialNontetCenter(const Mol &m, const Graph &g, int a) {
  const int tnz = nonzeroDegree(m, g, a) + m.nH[a];
  const int anum = m.z[a];
  if (tnz > 6 || tnz < 2 || (anum < 12 && anum != 4)) return false;
  const int ct = m.ctag[a];
  if (ct >= CHI_SQUAREPLANAR && ct <= CHI_OCTAHEDRAL) return true;
  if (ct == CHI_UNSPECIFIED && tnz >= 4) return true;
  return false;
}

//! Chirality::detail::isAtomPotentialTetrahedralCenter.
//!
//! `getValence(EXPLICIT)` in the sulfur/selenium clause is read off the boundary's
//! `GetTotalValence()`. That is EXACT and not an approximation: the clause is only reached with
//! nzDegree == 3 and, since tnzDegree <= 4 is already established, with totalNumHs == 0 -- so
//! there are no implicit hydrogens and total valence IS explicit valence.
inline bool isPotentialTetCenter(const Mol &m, const Graph &g, const RingSet &ri, int a,
                                 std::vector<char> &s1, std::vector<char> &s2,
                                 std::vector<char> &s3) {
  const int nzDegree = nonzeroDegree(m, g, a);
  const int tnzDegree = nzDegree + m.nH[a];
  if (tnzDegree > 4) return false;
  if (nzDegree == 4) return true;
  if (nzDegree <= 1) return false;
  const int anum = m.z[a];
  if (nzDegree < 3 && anum != 15 && anum != 33) return false;
  if (anum == 15 || anum == 33) return true;   // phosphine / arsine, InChI 1.02 rule
  if (nzDegree == 3) {
    if (m.nH[a] == 1) return !hasProtiumNeighbor(m, g, a);
    bool legal = false;
    if ((anum == 16 || anum == 34) &&
        (m.tval[a] == 4 || (m.tval[a] == 3 && m.fchg[a] == 1))) {
      legal = true;
    } else if (anum == 7) {
      bool conj = false;
      for (int e = g.start[a]; e < g.start[a + 1]; ++e)
        if (m.bconj[g.nbrBond[e]]) { conj = true; break; }
      if (m.hyb[a] == HYBS_SP3 && !conj &&
          (ri.atomInRingOfSize(a, 3) || isAtomBridgehead(m, g, ri, a, s1, s2, s3))) {
        legal = true;
      }
    }
    return legal;
  }
  return false;
}

inline bool isPotentialStereoAtom(const Mol &m, const Graph &g, const RingSet &ri, int a,
                                  std::vector<char> &s1, std::vector<char> &s2,
                                  std::vector<char> &s3) {
  return isPotentialTetCenter(m, g, ri, a, s1, s2, s3) ||
         (ALLOW_NONTETRAHEDRAL && isPotentialNontetCenter(m, g, a));
}

//! Chirality::detail::isBondPotentialStereoBond.
inline bool isPotentialStereoBond(const Mol &m, const Graph &g, const RingSet &ri, int b) {
  if (m.btype[b] != BT_DOUBLE) return false;
  const int beg = m.bu[b], end = m.bv[b];
  const int begDeg = g.degree(beg) + m.nH[beg], endDeg = g.degree(end) + m.nH[end];
  if (!(begDeg > 1 && begDeg < 4 && endDeg > 1 && endDeg < 4 && m.nH[beg] < 2 && m.nH[end] < 2))
    return false;
  // rdkit walks every bond ring looking for one smaller than 8 that contains this bond; the
  // smallest ring the bond is in answers the same question in O(1).
  if (ri.minBondRing[b] < MIN_RING_SIZE_FOR_DB_STEREO) return false;
  return true;
}

//! Chirality::detail::getStereoInfo(atom).
inline StereoInfo atomStereoInfo(const Mol &m, const Graph &g, int a, unsigned *scratchA,
                                 unsigned *scratchB) {
  StereoInfo si;
  si.type = SType::Atom_Tetrahedral;
  si.centeredOn = (unsigned)a;
  const int deg = g.degree(a);
  si.nControlling = deg;
  for (int k = 0; k < deg && k < 4; ++k) scratchA[k] = (unsigned)g.nbrAtom[g.start[a] + k];
  const int nn = deg < 4 ? deg : 4;
  for (int k = 0; k < nn; ++k) scratchB[k] = scratchA[k];
  std::sort(scratchB, scratchB + nn);
  for (int k = 0; k < nn; ++k) si.controlling[k] = scratchB[k];
  // No squiggle bonds and no _UnknownStereo from SMILES -- see the header note.
  const int ct = m.ctag[a];
  if (ct == CHI_TETRAHEDRAL_CCW || ct == CHI_TETRAHEDRAL_CW) {
    si.specified = SSpec::Specified;
    unsigned probe[4];
    for (int k = 0; k < nn; ++k) probe[k] = scratchB[k];
    const unsigned nSwaps = countSwaps(scratchA, probe, nn);
    int stereo = ct;
    if (nSwaps % 2)
      stereo = (ct == CHI_TETRAHEDRAL_CCW) ? CHI_TETRAHEDRAL_CW : CHI_TETRAHEDRAL_CCW;
    si.descriptor = (stereo == CHI_TETRAHEDRAL_CCW) ? SDesc::Tet_CCW : SDesc::Tet_CW;
  } else if (ALLOW_NONTETRAHEDRAL && isPotentialNontetCenter(m, g, a)) {
    int stereo = ct;
    if (stereo == CHI_UNSPECIFIED) {
      switch (deg + m.nH[a]) {          // getTotalDegree()
        case 4: stereo = CHI_TETRAHEDRAL; break;
        case 5: stereo = CHI_TRIGONALBIPYRAMIDAL; break;
        case 6: stereo = CHI_OCTAHEDRAL; break;
        default: break;
      }
    }
    si.descriptor = SDesc::None;
    switch (stereo) {
      case CHI_TETRAHEDRAL: si.type = SType::Atom_Tetrahedral; break;
      case CHI_SQUAREPLANAR: si.type = SType::Atom_SquarePlanar; break;
      case CHI_TRIGONALBIPYRAMIDAL: si.type = SType::Atom_TrigonalBipyramidal; break;
      case CHI_OCTAHEDRAL: si.type = SType::Atom_Octahedral; break;
      default: break;
    }
    // `_chiralPermutation` is a mol-file property; absent here, so `specified` stays Unspecified.
  }
  return si;
}

//! Chirality::detail::getStereoInfo(bond), double-bond case. Atropisomer single bonds cannot
//! arise from SMILES and are not built.
inline StereoInfo bondStereoInfo(const Mol &m, const Graph &g, const std::vector<int> &bstereo,
                                 const std::vector<int> &saA, const std::vector<int> &saB,
                                 int b) {
  StereoInfo si;
  if (m.btype[b] != BT_DOUBLE) { si.type = SType::Unspecified; return si; }
  si.type = SType::Bond_Double;
  si.centeredOn = (unsigned)b;
  si.nControlling = 4;
  int w = 0;
  for (const int endSel : {0, 1}) {
    const int atom = endSel == 0 ? m.bu[b] : m.bv[b];
    for (int e = g.start[atom]; e < g.start[atom + 1]; ++e)
      if (g.nbrBond[e] != b) si.controlling[w++] = (unsigned)g.nbrAtom[e];
    for (int i = g.degree(atom); i < 3; ++i) si.controlling[w++] = NOATOM;
  }
  int stereo = bstereo[b];
  if (stereo == BS_ANY) {
    si.specified = SSpec::Unknown;
  } else if (stereo != BS_NONE) {
    if (stereo == BS_E) stereo = BS_TRANS;
    else if (stereo == BS_Z) stereo = BS_CIS;
    // Stereo atoms: as parsed rdkit stores the highest-CIP-rank neighbour at each end, which is
    // what `findPotentialStereoBonds` writes and what the SMILES parser wrote before it. When a
    // bond arrives already E/Z the boundary does not carry them, so they are recovered from the
    // same legacy CIP ranks -- verified equal to rdkit's on all 2,481 E/Z bonds of the corpus.
    const int s0 = saA[b], s1 = saB[b];
    bool firstAtBegin;
    if (s0 == (int)si.controlling[0]) firstAtBegin = true;
    else if (s0 == (int)si.controlling[1]) firstAtBegin = false;
    else return si;                        // rdkit throws; unreachable for a consistent input
    bool firstAtEnd;
    if (s1 == (int)si.controlling[2]) firstAtEnd = true;
    else if (s1 == (int)si.controlling[3]) firstAtEnd = false;
    else return si;
    if (firstAtBegin ^ firstAtEnd) stereo = (stereo == BS_CIS) ? BS_TRANS : BS_CIS;
    si.specified = SSpec::Specified;
    si.descriptor = (stereo == BS_CIS) ? SDesc::Bond_Cis : SDesc::Bond_Trans;
  } else {
    si.specified = SSpec::Unspecified;
  }
  return si;
}

//! FindStereo.cpp getAtomCompareSymbol: isotope, element symbol, formal charge, CONCATENATED AS
//! DECIMAL TEXT and compared as a string. The text form is load-bearing -- "0Cl0" sorts after
//! "0C0" but before "0F0", and charge "-1" sorts before "0" -- so it is built rather than
//! encoded.
inline void atomCompareSymbol(const Mol &m, int a, std::string &out) {
  out.assign(std::to_string(m.iso[a]));
  out += elementSymbol(m.z[a]);
  out += std::to_string(m.fchg[a]);
}

//! FindStereo.cpp getBondSymbol.
inline const char *bondSymbolBase(const Mol &m, int b) {
  if (m.barom[b]) return ":";
  switch (m.btype[b]) {
    case BT_SINGLE: return "-";
    case BT_DOUBLE: return "=";
    case BT_TRIPLE: return "#";
    case BT_AROMATIC: return ":";
    default: return "?";
  }
}

//! All of runCleanup's per-molecule state, allocated once and reused across a batch.
struct Work {
  Graph g;
  RingSet ri;
  std::vector<unsigned> cipRanks;
  CipScratch cip;
  std::vector<int> bstereo, saA, saB;
  std::vector<char> knownAtoms, possibleAtoms, origPossibleAtoms, fixedAtoms;
  std::vector<char> knownBonds, possibleBonds, origPossibleBonds, fixedBonds;
  std::vector<std::string> atomSymbols, bondSymbols;
  std::vector<int> possibleRingStereoAtoms, possibleRingStereoBonds;
  std::vector<unsigned> aranks;
  std::vector<char> possibleAtomsInRing;
  std::vector<char> isTet;            // the answer: 1 where a StereoInfo of type Atom_Tetrahedral
  std::vector<char> onStereoBond;
  Canon canon;
  // scratch
  std::vector<char> sc1, sc2, sc3, sv1, sv2, sv3;
  std::vector<int> si1, si2, si3, si4;
  std::string tmpsym;

  void alloc(const Mol &m) {
    knownAtoms.assign(m.n, 0); possibleAtoms.assign(m.n, 0); fixedAtoms.assign(m.n, 0);
    knownBonds.assign(m.nb, 0); possibleBonds.assign(m.nb, 0); fixedBonds.assign(m.nb, 0);
    atomSymbols.assign(m.n, std::string());
    bondSymbols.assign(m.nb, std::string());
    possibleRingStereoAtoms.assign(m.n, 0);
    possibleRingStereoBonds.assign(m.nb, 0);
    possibleAtomsInRing.assign(m.n, 0);
    isTet.assign(m.n, 0);
    onStereoBond.assign(m.n, 0);
    sc1.assign(m.nb ? m.nb : 1, 0);
    sc2.assign(m.nb ? m.nb : 1, 0);
    sc3.assign(m.n ? m.n : 1, 0);
  }
};

//! FindStereo.cpp initAtomInfo, with cleanIt == false and flagPossible == true.
inline void initAtomInfo(const Mol &m, Work &W) {
  for (int a = 0; a < m.n; ++a) {
    atomCompareSymbol(m, a, W.atomSymbols[a]);
    if (!isPotentialStereoAtom(m, W.g, W.ri, a, W.sc1, W.sc2, W.sc3)) continue;
    unsigned sa[4], sb[4];
    const StereoInfo si = atomStereoInfo(m, W.g, a, sa, sb);
    switch (si.specified) {
      case SSpec::Unknown:
        W.knownAtoms[a] = 1;
        W.atomSymbols[a] += std::to_string(a);
        break;
      case SSpec::Specified:
        W.knownAtoms[a] = 1;
        if (si.descriptor == SDesc::Tet_CCW) W.atomSymbols[a] += "_CCW";
        else if (si.descriptor == SDesc::Tet_CW) W.atomSymbols[a] += "_CW";
        else W.atomSymbols[a] += "_STEREO";
        break;
      case SSpec::Unspecified:
        W.possibleAtoms[a] = 1;
        W.atomSymbols[a] += "_" + std::to_string(a);    // cleanIt == false
        break;
    }
  }
}

//! FindStereo.cpp initBondInfo, with cleanIt == false and flagPossible == true.
inline void initBondInfo(const Mol &m, Work &W) {
  for (int b = 0; b < m.nb; ++b) {
    W.bondSymbols[b] = bondSymbolBase(m, b);
    if (isPotentialStereoBond(m, W.g, W.ri, b)) {
      const StereoInfo si = bondStereoInfo(m, W.g, W.bstereo, W.saA, W.saB, b);
      switch (si.specified) {
        case SSpec::Unknown:
          W.knownBonds[b] = 1;
          W.bondSymbols[b] += "_" + std::to_string(b);
          break;
        case SSpec::Specified:
          W.knownBonds[b] = 1;
          if (si.descriptor == SDesc::Bond_Cis) W.bondSymbols[b] += "_cis";
          else if (si.descriptor == SDesc::Bond_Trans) W.bondSymbols[b] += "_trans";
          else W.bondSymbols[b] += "_STEREO";
          break;
        case SSpec::Unspecified:
          W.possibleBonds[b] = 1;
          W.bondSymbols[b] += "_" + std::to_string(b);
          break;
      }
    } else {
      const int st = W.bstereo[b];
      if (st == BS_ATROPCW || st == BS_ATROPCCW) {
        W.knownBonds[b] = 1;
        W.bondSymbols[b] += (st == BS_ATROPCW) ? "_atropcw" : "_atropccw";
      }
      // cleanIt == false, so nothing is stripped here
    }
  }
}

//! FindStereo.cpp flagRingStereo, with non-null possibleAtoms / possibleBonds (cleanIt false).
//!
//! DIRECTION OF THE COMMON-EDGE WALK. The "if the atom is in more than one ring" branch walks
//! the ring in ONE direction from each candidate. Which direction that is depends on the order
//! rdkit's RingInfo stored the ring in, and this file recovers a cycle order of its own (see
//! RingSet::build). The walk is run from every candidate atom of the ring, so a fused edge is
//! found from at least one of its two ends either way, and the only quantity that could differ,
//! `nHere`, is compared against 1. Verified: 0 molecules of 20,000 differ.
inline void flagRingStereo(Work &W, bool usePossible) {
  RingSet &ri = W.ri;
  const auto possA = [&](int i) { return usePossible && W.possibleAtoms[i]; };
  const auto possB = [&](int i) { return usePossible && W.possibleBonds[i]; };
  for (int ridx = 0; ridx < ri.nr; ++ridx) {
    const int lo = ri.ptr[ridx], sz = ri.size(ridx);
    const int *aring = ri.at.data() + lo;
    const int *bring = ri.bd.data() + lo;
    int nHere = 0;
    const bool oddSized = sz % 2;
    const int halfSize = sz / 2 + (oddSized ? 1 : 0);
    for (int k = 0; k < sz; ++k) W.possibleAtomsInRing[aring[k]] = 0;
    for (int k = 0; k < sz; ++k) W.ri.bondInRing[bring[k]] = 1;
    for (int ai = 0; ai < sz; ++ai) {
      const int aidx = aring[ai];
      if (!W.knownAtoms[aidx] && !possA(aidx)) continue;
      for (const int ringDivisor : {2, 3}) {
        if (sz % ringDivisor) continue;
        const int incrementSize = sz / ringDivisor;
        int otherFoundByBondCount = 0, otherFoundByAtomCount = 0;
        for (int inc = incrementSize; inc < sz; inc += incrementSize) {
          const int otherIdx = aring[(ai + inc) % sz];
          bool found = false;
          for (int e = W.g.start[otherIdx]; e < W.g.start[otherIdx + 1]; ++e) {
            const int bidx = W.g.nbrBond[e];
            if ((W.knownBonds[bidx] || possB(bidx)) && !W.ri.bondInRing[bidx]) {
              found = true;
              break;
            }
          }
          if (found) ++otherFoundByBondCount;
          if (otherFoundByBondCount == 0) {
            if (W.knownAtoms[otherIdx] || possA(otherIdx)) ++otherFoundByAtomCount;
          }
        }
        if (otherFoundByBondCount == ringDivisor - 1 ||
            otherFoundByAtomCount == ringDivisor - 1) {
          nHere += 1 + otherFoundByBondCount;
          for (int inc = 0; inc < sz; inc += incrementSize)
            W.possibleAtomsInRing[aring[(ai + inc) % sz]] = 1;
          continue;    // rdkit `continue`s the divisor loop, not the atom loop
        }
      }
      if (ri.numAtomRings[aidx] > 1) {
        int previousOtherIdx = aidx;
        for (int step = 1; step <= halfSize; ++step) {
          const int otherIdx = aring[(ai + step) % sz];
          const int bnd = W.g.bondBetween(previousOtherIdx, otherIdx);
          if (bnd < 0 || ri.numBondRings[bnd] < 2) break;
          if (W.knownAtoms[otherIdx] || possA(otherIdx)) {
            nHere += 2;
            W.possibleAtomsInRing[aidx] = 1;
            W.possibleAtomsInRing[otherIdx] = 1;
            break;
          }
          previousOtherIdx = otherIdx;
        }
      }
    }
    for (int k = 0; k < sz; ++k) W.ri.bondInRing[bring[k]] = 0;
    if (nHere > 1) {
      for (int k = 0; k < sz; ++k)
        if (W.possibleAtomsInRing[aring[k]]) ++W.possibleRingStereoAtoms[aring[k]];
      for (int k = 0; k < sz; ++k) ++W.possibleRingStereoBonds[bring[k]];
    }
  }
}

//! FindStereo.cpp updateAtoms. Returns whether another canonicalisation round is needed and
//! records, in `W.isTet`, which atoms this round emitted an `Atom_Tetrahedral` StereoInfo for --
//! which is exactly what `Chem.FindMolChiralCenters(useLegacyImplementation=False,
//! includeUnassigned=True)` returns and what SPS's stereo term reads.
inline bool updateAtoms(const Mol &m, Work &W) {
  bool needAnotherRound = false;
  const RingSet &ri = W.ri;
  for (int aidx = 0; aidx < m.n; ++aidx) {
    if (!W.knownAtoms[aidx] && !W.possibleAtoms[aidx]) continue;
    unsigned sa[4], sb[4];
    const StereoInfo si = atomStereoInfo(m, W.g, aidx, sa, sb);
    if (W.fixedAtoms[aidx]) {
      if (si.type == SType::Atom_Tetrahedral) W.isTet[aidx] = 1;
      continue;
    }
    unsigned nbrs[4];
    int nNbrs = 0;
    bool haveADupe = false;
    if (si.type == SType::Atom_Tetrahedral) {
      const int nc = si.nControlling < 4 ? si.nControlling : 4;
      for (int k = 0; k < nc; ++k) {
        const unsigned nbrIdx = si.controlling[k];
        const unsigned rnk = W.aranks[nbrIdx];
        bool dup = false;
        for (int q = 0; q < nNbrs; ++q) if (nbrs[q] == rnk) { dup = true; break; }
        if (dup) {
          if (W.possibleRingStereoAtoms[aidx]) {
            const int bnd = W.g.bondBetween(aidx, (int)nbrIdx);
            if (bnd < 0 || !W.possibleRingStereoBonds[bnd]) { haveADupe = true; break; }
          } else {
            haveADupe = true;
            break;
          }
        } else {
          nbrs[nNbrs++] = rnk;
        }
      }
    }
    if (!haveADupe) {
      // `acs` is the symbol this atom SHOULD carry. For a known (specified) centre it is rebuilt
      // from scratch with the descriptor re-derived against the canonical ranks; for a merely
      // possible one it is left as-is.
      std::string acs = W.atomSymbols[aidx];
      if (!W.possibleAtoms[aidx]) {
        if (si.type == SType::Atom_Tetrahedral) {
          unsigned sorted[4];
          for (int k = 0; k < nNbrs; ++k) sorted[k] = nbrs[k];
          std::sort(sorted, sorted + nNbrs);
          unsigned probe[4];
          for (int k = 0; k < nNbrs; ++k) probe[k] = sorted[k];
          const unsigned nSwaps = countSwaps(nbrs, probe, nNbrs);
          SDesc d = si.descriptor;
          if (nSwaps % 2 && (d == SDesc::Tet_CCW || d == SDesc::Tet_CW))
            d = (d == SDesc::Tet_CCW) ? SDesc::Tet_CW : SDesc::Tet_CCW;
          if (d == SDesc::Tet_CW || d == SDesc::Tet_CCW) {
            atomCompareSymbol(m, aidx, W.tmpsym);
            acs = W.tmpsym + (d == SDesc::Tet_CW ? "_CW" : "_CCW");
          }
        }
        W.fixedAtoms[aidx] = 1;
      }
      if (W.atomSymbols[aidx] != acs) {
        W.atomSymbols[aidx] = acs;
        needAnotherRound = true;
      }
      if (si.type == SType::Atom_Tetrahedral) W.isTet[aidx] = 1;
    } else {
      if (W.possibleAtoms[aidx]) needAnotherRound = true;
      W.possibleAtoms[aidx] = 0;
      atomCompareSymbol(m, aidx, W.atomSymbols[aidx]);
      if (W.possibleRingStereoAtoms[aidx]) {
        --W.possibleRingStereoAtoms[aidx];
        if (!W.possibleRingStereoAtoms[aidx]) {
          needAnotherRound = true;
          for (int ridx = 0; ridx < ri.nr; ++ridx) {
            const int lo = ri.ptr[ridx], sz = ri.size(ridx);
            int nHere = 0;
            for (int k = 0; k < sz; ++k) {
              const int raidx = ri.at[lo + k];
              W.fixedAtoms[raidx] = 0;
              nHere += (W.possibleRingStereoAtoms[raidx] > 0) ? 1 : 0;
            }
            if (nHere <= 1) {
              if (nHere == 1) {
                for (int k = 0; k < sz; ++k) {
                  const int raidx = ri.at[lo + k];
                  if (W.possibleRingStereoAtoms[raidx]) { --W.possibleRingStereoAtoms[raidx]; break; }
                }
              }
              for (int k = 0; k < sz; ++k) {
                const int rbidx = ri.bd[lo + k];
                if (W.possibleRingStereoBonds[rbidx]) --W.possibleRingStereoBonds[rbidx];
              }
            }
          }
        }
      }
    }
  }
  return needAnotherRound;
}

//! FindStereo.cpp areStereobondControllingAtomsDupes.
inline bool controllingAtomsDupes(const Mol &m, Work &W, int b, unsigned c1, unsigned c2) {
  if (W.aranks[c1] != W.aranks[c2]) return false;
  const RingSet &ri = W.ri;
  int i1 = ri.memb_ptr[c1], e1 = ri.memb_ptr[c1 + 1];
  int i2 = ri.memb_ptr[c2], e2 = ri.memb_ptr[c2 + 1];
  while (i1 < e1 && i2 < e2) {
    if (ri.memb[i1] < ri.memb[i2]) { ++i1; continue; }
    if (ri.memb[i1] > ri.memb[i2]) { ++i2; continue; }
    const int r = ri.memb[i1];
    ++i1; ++i2;
    const int sz = ri.size(r);
    if (sz % 2) continue;
    const int lo = ri.ptr[r];
    for (const int bondEnd : {m.bu[b], m.bv[b]}) {
      int pos = -1;
      for (int k = 0; k < sz; ++k) if (ri.at[lo + k] == bondEnd) { pos = k; break; }
      if (pos < 0) continue;
      const int oppositeIdx = ri.at[lo + (pos + sz / 2) % sz];
      if (W.possibleAtoms[oppositeIdx] || W.knownAtoms[oppositeIdx]) return false;
      if (W.g.degree(oppositeIdx) == 3) {
        for (int e = W.g.start[oppositeIdx]; e < W.g.start[oppositeIdx + 1]; ++e) {
          const int outOther = W.g.nbrAtom[e];
          bool inRing = false;
          for (int k = 0; k < sz; ++k) if (ri.at[lo + k] == outOther) { inRing = true; break; }
          if (!inRing) {
            const int nb2 = W.g.nbrBond[e];
            if (W.possibleBonds[nb2] || W.knownBonds[nb2]) return false;
          }
        }
      }
    }
  }
  return true;
}

//! FindStereo.cpp updateBonds.
inline bool updateBonds(const Mol &m, Work &W) {
  bool needAnotherRound = false;
  for (int bidx = 0; bidx < m.nb; ++bidx) {
    if (!W.knownBonds[bidx] && !W.possibleBonds[bidx]) continue;
    StereoInfo si = bondStereoInfo(m, W.g, W.bstereo, W.saA, W.saB, bidx);
    if (si.type == SType::Unspecified) continue;
    if ((si.controlling[0] == NOATOM && si.controlling[1] == NOATOM) ||
        (si.controlling[2] == NOATOM && si.controlling[3] == NOATOM)) {
      W.fixedBonds[bidx] = 1;
    }
    if (W.fixedBonds[bidx]) continue;
    bool haveADupe = false, needsSwap = false;
    if (si.controlling[0] != NOATOM && si.controlling[1] != NOATOM) {
      if (controllingAtomsDupes(m, W, bidx, si.controlling[0], si.controlling[1])) haveADupe = true;
      else if (W.aranks[si.controlling[0]] < W.aranks[si.controlling[1]]) {
        std::swap(si.controlling[0], si.controlling[1]);
        needsSwap = !needsSwap;
      }
    }
    if (si.controlling[2] != NOATOM && si.controlling[3] != NOATOM) {
      if (controllingAtomsDupes(m, W, bidx, si.controlling[2], si.controlling[3])) haveADupe = true;
      else if (W.aranks[si.controlling[2]] < W.aranks[si.controlling[3]]) {
        std::swap(si.controlling[2], si.controlling[3]);
        needsSwap = !needsSwap;
      }
    }
    if (!haveADupe) {
      if (needsSwap && (si.descriptor == SDesc::Bond_Cis || si.descriptor == SDesc::Bond_Trans))
        si.descriptor = (si.descriptor == SDesc::Bond_Cis) ? SDesc::Bond_Trans : SDesc::Bond_Cis;
      std::string gbs = W.bondSymbols[bidx];
      if (si.specified == SSpec::Specified) {
        if (si.descriptor == SDesc::Bond_Cis) gbs += "_cis";
        else if (si.descriptor == SDesc::Bond_Trans) gbs += "_trans";
      } else if (si.specified == SSpec::Unknown) {
        gbs += "_unk";
      }
      if (W.bondSymbols[bidx] != gbs) { W.bondSymbols[bidx] = gbs; needAnotherRound = true; }
      if (!W.possibleBonds[bidx]) W.fixedBonds[bidx] = 1;
    } else if (W.possibleBonds[bidx]) {
      W.possibleBonds[bidx] = 0;
      W.bondSymbols[bidx] = bondSymbolBase(m, bidx);
      needAnotherRound = true;
    }
  }
  return needAnotherRound;
}

//! FindStereo.cpp runCleanup(mol, flagPossible=true, cleanIt=false), followed by
//! findPotentialStereo's wrapper. Fills `W.isTet`.
inline void findPotentialStereo(const Mol &m, Work &W) {
  std::fill(W.knownAtoms.begin(), W.knownAtoms.end(), 0);
  std::fill(W.possibleAtoms.begin(), W.possibleAtoms.end(), 0);
  std::fill(W.fixedAtoms.begin(), W.fixedAtoms.end(), 0);
  std::fill(W.knownBonds.begin(), W.knownBonds.end(), 0);
  std::fill(W.possibleBonds.begin(), W.possibleBonds.end(), 0);
  std::fill(W.fixedBonds.begin(), W.fixedBonds.end(), 0);
  std::fill(W.possibleRingStereoAtoms.begin(), W.possibleRingStereoAtoms.end(), 0);
  std::fill(W.possibleRingStereoBonds.begin(), W.possibleRingStereoBonds.end(), 0);
  std::fill(W.isTet.begin(), W.isTet.end(), 0);

  initAtomInfo(m, W);
  initBondInfo(m, W);
  W.origPossibleAtoms = W.possibleAtoms;
  W.origPossibleBonds = W.possibleBonds;

  flagRingStereo(W, /*usePossible=*/true);

  W.canon.init(m, W.g, W.ri);

  bool needAnotherRound = true;
  while (needAnotherRound) {
    std::fill(W.isTet.begin(), W.isTet.end(), 0);
    W.canon.refreshCodes(W.atomSymbols, W.bondSymbols);
    W.canon.rank(W.aranks, W.sv1, W.sv2, W.sv3, W.si1, W.si2, W.si3, W.si4);
    needAnotherRound = updateAtoms(m, W);
    needAnotherRound = updateBonds(m, W) || needAnotherRound;
  }

  // cleanIt is false, so cleanMolStereo does not run. The flagPossible re-run does: if anything
  // that started out `known` failed to be fixed, it is downgraded to `possible` and the whole
  // loop is done again with fresh symbols.
  bool changed = false;
  for (int i = 0; i < m.n && !changed; ++i)
    if (W.possibleAtoms[i] != W.origPossibleAtoms[i]) changed = true;
  for (int i = 0; i < m.nb && !changed; ++i)
    if (W.possibleBonds[i] != W.origPossibleBonds[i]) changed = true;
  if (!changed) return;

  W.possibleAtoms = W.origPossibleAtoms;
  for (int i = 0; i < m.n; ++i) {
    if (!W.fixedAtoms[i] && W.knownAtoms[i]) { W.possibleAtoms[i] = 1; W.knownAtoms[i] = 0; }
    if (W.possibleAtoms[i]) W.atomSymbols[i] += "_" + std::to_string(i);
  }
  W.possibleBonds = W.origPossibleBonds;
  for (int i = 0; i < m.nb; ++i) {
    if (!W.fixedBonds[i] && W.knownBonds[i]) { W.possibleBonds[i] = 1; W.knownBonds[i] = 0; }
    if (W.possibleBonds[i]) W.bondSymbols[i] += "_" + std::to_string(i);
  }
  flagRingStereo(W, /*usePossible=*/true);

  needAnotherRound = true;
  while (needAnotherRound) {
    std::fill(W.isTet.begin(), W.isTet.end(), 0);
    W.canon.refreshCodes(W.atomSymbols, W.bondSymbols);
    W.canon.rank(W.aranks, W.sv1, W.sv2, W.sv3, W.si1, W.si2, W.si3, W.si4);
    needAnotherRound = updateAtoms(m, W);
    needAnotherRound = updateBonds(m, W) || needAnotherRound;
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------
// The public entry point.
//
//     hyb     SP 1, SP2 2, SP3 3, ANYTHING ELSE 4  (rdkit uses defaultdict(lambda: 4))
//     stereo  2 if the atom is an `Atom_Tetrahedral` potential stereocentre, or an end of a
//             double bond whose stereo is set AFTER FindPotentialStereoBonds; else 1
//     ring    1 if aromatic, 2 if in a ring, else 1
//     bond    GetDegree()^2
//     SPS     sum of the product over all atoms, divided by the heavy-atom count
// ---------------------------------------------------------------------------------------------
inline void compute(const Mol &m, detail::Work &W, double *out) {
  using namespace detail;
  out[C_SPS] = std::numeric_limits<double>::quiet_NaN();
  if (m.n <= 0) return;

  W.g.build(m);
  W.ri.build(m, W.g, W.sv1, W.sv2);
  W.alloc(m);

  // 1. the LEGACY CIP ranks, for findPotentialStereoBonds -- and, when the molecule carries any
  //    assigned R/S or E/Z, the rerank pass that folds those labels in, because that is the
  //    `_CIPRank` a parsed molecule actually arrives carrying.
  //
  //    GATED, because the ranking is a third of this file's cost and most molecules never
  //    consult it. Exactly two things read `cipRanks`: the stereo-atom recovery just below (only
  //    for a bond that arrived E/Z) and `findPotentialStereoBonds`, which only compares ranks
  //    when a non-ring double bond with 2-or-3-coordinate ends has TWO single/aromatic
  //    neighbours on at least one side -- the one-neighbour-each-side case is decided without
  //    them. When no bond qualifies the ranks are never read, so they are not computed. This
  //    changes nothing: verified 0 molecules of 20,000 differ with the gate removed.
  bool needRanks = false;
  for (int b = 0; b < m.nb && !needRanks; ++b) {
    if (m.btype[b] != BT_DOUBLE) continue;
    if (m.bstereo[b] == BS_E || m.bstereo[b] == BS_Z) { needRanks = true; break; }
    if (W.ri.numBondRings[b] || m.bstereo[b] != BS_NONE) continue;
    const int dbeg = W.g.degree(m.bu[b]), dend = W.g.degree(m.bv[b]);
    if (!((dbeg == 2 || dbeg == 3) && (dend == 2 || dend == 3))) continue;
    int bn[4], en[4], nbeg = 0, nend = 0;
    findAtomNeighbors(m, W.g, m.bu[b], b, bn, nbeg);
    findAtomNeighbors(m, W.g, m.bv[b], b, en, nend);
    if (nbeg == 0 || nend == 0) continue;
    if (nbeg == 2 || nend == 2) needRanks = true;
  }
  W.cipRanks.assign(m.n, 0);
  if (needRanks) {
    assignAtomCIPRanks(m, W.g, W.cipRanks, W.cip);
    bool anyLabel = false;
    for (int i = 0; i < m.n && !anyLabel; ++i) if (m.cip[i]) anyLabel = true;
    for (int b = 0; b < m.nb && !anyLabel; ++b)
      if (m.btype[b] == BT_DOUBLE && (m.bstereo[b] == BS_E || m.bstereo[b] == BS_Z))
        anyLabel = true;
    if (anyLabel) rerankAtoms(m, W.g, W.cipRanks, W.cip);
  }

  // 2. FindPotentialStereoBonds on the copy
  W.bstereo = m.bstereo;
  W.saA.assign(m.nb, -1);
  W.saB.assign(m.nb, -1);
  for (int b = 0; b < m.nb; ++b) {
    // Recover the stereo atoms of a bond that arrived already E/Z: rdkit stores the
    // highest-legacy-CIP-rank neighbour at each end and the boundary does not carry them.
    if (W.bstereo[b] != BS_E && W.bstereo[b] != BS_Z) continue;
    for (const int endSel : {0, 1}) {
      const int atom = endSel == 0 ? m.bu[b] : m.bv[b];
      const int other = endSel == 0 ? m.bv[b] : m.bu[b];
      int best = -1;
      for (int e = W.g.start[atom]; e < W.g.start[atom + 1]; ++e) {
        const int w = W.g.nbrAtom[e];
        if (w == other) continue;
        if (best < 0 || W.cipRanks[w] > W.cipRanks[best]) best = w;
      }
      (endSel == 0 ? W.saA[b] : W.saB[b]) = best;
    }
  }
  findPotentialStereoBonds(m, W.g, W.ri, W.cipRanks, W.bstereo, W.saA, W.saB);

  // 3. the NEW perception
  findPotentialStereo(m, W);

  // 4. the score
  int heavy = 0;
  for (int i = 0; i < m.n; ++i) if (m.z[i] > 1) ++heavy;
  if (!heavy) return;

  std::vector<char> &onStereoBond = W.onStereoBond;
  std::fill(onStereoBond.begin(), onStereoBond.end(), 0);
  for (int b = 0; b < m.nb; ++b)
    if (m.btype[b] == BT_DOUBLE && W.bstereo[b] != BS_NONE) {
      onStereoBond[m.bu[b]] = 1;
      onStereoBond[m.bv[b]] = 1;
    }

  long long score = 0;
  for (int i = 0; i < m.n; ++i) {
    int hy;
    switch (m.hyb[i]) {
      case HYBS_SP: hy = 1; break;
      case HYBS_SP2: hy = 2; break;
      case HYBS_SP3: hy = 3; break;
      default: hy = 4; break;
    }
    const int st = (W.isTet[i] || onStereoBond[i]) ? 2 : 1;
    const int rg = m.arom[i] ? 1 : (m.nring[i] ? 2 : 1);
    const long long bd = (long long)m.deg[i] * m.deg[i];
    score += (long long)hy * st * rg * bd;
  }
  out[C_SPS] = (double)score / (double)heavy;
}

//! Convenience overload for callers that do not keep a `Work` across molecules.
inline void compute(const Mol &m, double *out) {
  detail::Work W;
  compute(m, W, out);
}

}  // namespace sps

#endif  // HUME_SPS_H
