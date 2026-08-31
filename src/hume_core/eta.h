// mordred's ExtendedTopochemicalAtom block: the 29 columns HUME was missing.
//
//     ETA_alpha  ETA_shape_{p,y,x}  ETA_beta{,_s,_ns,_ns_d}
//     ETA_eta{,_L,_R,_F,_FL,_B,_BR}  ETA_dAlpha_{A,B}
//     ETA_epsilon_{1..5}  ETA_dEpsilon_{A,B,C,D}  ETA_dBeta  ETA_psi_1  ETA_dPsi_B
//
// SPECIFICATION is mordred/ExtendedTopochemicalAtom.py plus mordred/_atomic_property.py at
// 1.2.0, read line for line, and mordred/_base/{context,calculator}.py for the failure
// semantics. Every number below is one of those files' arithmetic, not a paper's formula.
//
// ------------------------------------------------------------------------------------------
// THE SHAPE OF THE FAMILY: FOUR PRIMITIVES, TWENTY-NINE COLUMNS
// ------------------------------------------------------------------------------------------
// mordred defines these as 29 Descriptor classes with a dependency graph between them, and its
// Calculator memoises the shared nodes.  Written out, the whole block is four per-atom vectors
// and two graphs:
//
//     core_i    (Z_i - Z^v_i) / (Z^v_i (PN_i - 1))          a function of Z ALONE
//     eps_i     0.3 Z^v_i - core_i                          a function of Z ALONE
//     beta_i    sigma + non-sigma + delta contributions     needs the bonds
//     gamma_i   core_i / beta_i                             NaN when beta_i == 0
//
// on the molecule itself, and the same four on the REFERENCE ALKANE -- same skeleton, every
// heavy atom a carbon, every bond single -- where they collapse to constants:
//
//     core = 0.5,  eps = 0.3*4 - 0.5,  beta_i = 0.5 * deg_i,  gamma_i = 1 / deg_i
//
// so eta_R and eta_RL need no second molecule to be built, only the skeleton's degrees.  Every
// column here is then a two-line combination of those, and this file computes each primitive
// exactly once per molecule.
//
// ETA_eta_RL IS THE RANDIC INDEX, AND THAT IS WHY IT WAS DEDUPLICATED.  It is
// sum over adjacent pairs of sqrt(gamma_i gamma_j) on the reference alkane, and with
// gamma_i = 1/deg_i that is sum over edges 1/sqrt(deg_i deg_j) -- the 1975 connectivity index,
// nothing else.  It is not near-constant and it is not a coincidence: on the 20,000-molecule
// corpus it is BIT-IDENTICAL to mordred's own `Xp-1d` on all 19,896 molecules where both are
// finite (max |diff| exactly 0.0), and 0.9999 against RDKit's `Chi1`.  The >=0.99 correlations
// the dedupe review flagged are a genuine algebraic identity with a column from another family.
// It survives here only as an INTERMEDIATE, because ETA_eta_B / ETA_eta_BR are affine in it.
// See NOTES_eta.md.
//
// ------------------------------------------------------------------------------------------
// THE GRAPH.  `explicit_hydrogens = False` for every descriptor in this file except
// EtaEpsilon(1), (3), (4), (5), so the primary molecule is `Chem.RemoveHs(mol)` -- which for a
// molecule that came out of `Chem.MolFromSmiles` is the boundary's own graph, unchanged.  That
// is checked, not assumed: `RemoveHs(m, updateExplicitCount=True)` moves the atom count on 0 of
// the 20,000 corpus molecules.
//
// BUT `RemoveHs` IS NOT `AddHs` INVERTED, AND 558 CORPUS MOLECULES HAVE A HYDROGEN ATOM IN THE
// HEAVY GRAPH.  RDKit keeps an explicit `[H]` that defines double-bond stereochemistry, so
// `[H]/N=C(...)...` arrives with a real hydrogen ATOM among its 15 atoms and mordred counts it:
// `AETA_alpha` for that molecule is alpha/15, not alpha/14.  Every place below that says
// "atom count" therefore means N INCLUDING SUCH HYDROGENS, and every place that says "reference
// skeleton" means the non-hydrogen atoms only (AlterMolecule skips `GetAtomicNum() == 1`).  The
// two counts are `m.n` and `A_R` and they are deliberately never mixed:
//
//     ETA_dAlpha_*   (alpha - 0.5*A_R) / n          <- BOTH counts in one expression
//     ETA_eta_B      sqrt(2) + 0.5*(n-3) - eta_RL   <- n from the heavy graph, eta_RL from the
//                                                      skeleton
//
// Getting this wrong is invisible on 97% of any drug-like corpus and then wrong by a few percent
// on the rest.
//
// ------------------------------------------------------------------------------------------
// KEKULE STRUCTURES: WHERE THEY BITE AND WHERE THEY DO NOT
// ------------------------------------------------------------------------------------------
// mordred sets `kekulize = True`, so its molecule has SINGLE/DOUBLE bond TYPES with the aromatic
// FLAGS still set (`Chem.Kekulize(m)`, clearAromaticFlags defaulting False).  The boundary
// carries the aromatic form, so the Kekule assignment has to be reconstructed here.
//
// FOR beta AND gamma IT IS FREE.  `get_eta_nonsigma_contribute` returns 0 for a SINGLE bond and
// y*f otherwise, and for an aromatic-flagged bond y is 2.0 with no reference to the endpoints.
// So an aromatic bond contributes 2.0 exactly when it took the double, and the per-atom sum only
// needs to know HOW MANY of atom i's aromatic bonds are double -- which is
//
//     takesDouble(i) = tval_i - nH_i - round(non-aromatic valence contributions) - n_aromatic_i
//
// straight off the boundary, exactly the invariant constit.h uses for `nBondsKD`.  It is in
// {0,1}, it is Kekule-INVARIANT, and it needs no matching.
//
// FOR ETA_epsilon_4 IT DOES NOT COME FREE, AND THE DEFINITION IS ILL-POSED THERE.  The saturated
// skeleton reduces C-C bonds to single and leaves every bond touching a heteroatom at its
// KEKULIZED order, so its hydrogen count depends on WHICH bonds took the double, not just how
// many per atom.  A perfect matching of the takesDouble atoms over the aromatic bonds is
// computed below (`KekuleMatcher`), deterministically, in atom-then-bond index order.  On 79 of
// the 20,000 corpus molecules more than one perfect matching exists with a DIFFERENT number of
// C-C doubles -- pyridazines, tetrazoles, fused triazoles -- and there the value depends on the
// Kekule structure the perceiver happened to pick.  RDKit's own choice is not a rule this file
// could follow: over those 79 it lands on the FEWEST C-C doubles 43 times, on the MOST 35 times
// and in between once, so there is nothing to copy short of its search order.  ETA_epsilon_4,
// ETA_dEpsilon_B and ETA_dEpsilon_C are therefore ill-posed on the Kekule axis; the measurement
// is in NOTES_eta.md.  As it happens the matching below agrees with RDKit on all 79 -- all three
// columns come out bit-identical to mordred on 19,895 of 19,896 -- but that is a coincidence of
// two similar depth-first searches, not a guarantee, and verify_eta.py's invariance screen shows
// the exposure directly: 14 cells of 12,000 move under renumbering, and none under a Kekule
// round trip.  Everything else in this file is invariant on both axes.
//
// ------------------------------------------------------------------------------------------
// THE TWO ALTERED MOLECULES ARE RECONSTRUCTED, NOT BUILT
// ------------------------------------------------------------------------------------------
// `AlterMolecule` builds an RWMol, sanitizes it and calls `Chem.AddHs`, so what the epsilon
// columns actually need from it is a HYDROGEN COUNT after RDKit's implicit-valence fill.  Both
// are closed forms of boundary integers, and both were checked against a verbatim port of
// AlterMolecule over all 20,000 corpus molecules:
//
//   REFERENCE ALKANE  every atom is a neutral carbon of degree deg_R, so
//                         H_R = 4*A_R - 2*B_R
//                     0 mismatches / 20,000.
//
//   SATURATED SKELETON  only C-C bonds change, and dropping a bond lowers the atom's explicit
//                     valence by exactly that bond's valence contribution, which RDKit then
//                     refills with hydrogen:
//                         H_S = sum over non-H atoms of ( nH_i + reduction_i + hbonds_i )
//                     3 mismatches / 20,000, all three of them free metal atoms whose bracket
//                     form carries `noImplicit` ([Cs], [Sr], [Rh]); two are disconnected and NaN
//                     anyway, the third is the single-atom molecule `[Cs]`.  That is the one
//                     molecule in the corpus this file cannot make exact; see NOTES_eta.md.
//
// ------------------------------------------------------------------------------------------
// ETA_eta_BR TAKES ITS RING COUNT FROM THE SAME PLACE RingCount DOES, AND THAT IS A DIVERGENCE
// ------------------------------------------------------------------------------------------
// EtaBranchingIndex(ring=True) adds `0.086 * RingCount()`, i.e. mordred's `nRing`, i.e.
// `len(Chem.GetSymmSSSR(mol))`.  `GetSymmSSSR` is not a function of the molecular graph -- RDKit
// symmetrises the SSSR basis and its own source admits it "may miss extra rings" depending on
// the order it sees the molecule in -- so src/hume/_rings.py already perceives rings on a
// canonically rebuilt skeleton and hands THAT set to RingCount.  `n_rings` here is the same
// number, deliberately, because two ring counts inside one package that disagree with each other
// is worse than one that disagrees with an unstable upstream.  The price is 14 molecules of
// 20,000 where ETA_eta_BR differs from mordred by exactly 0.086 -- the same 14 where HUME's
// `nRing` already differs, all of them small bridged cages (`CC12CC3C1OCCN23` and friends).
// Nothing else in this file reads `n_rings`.
//
// ------------------------------------------------------------------------------------------
// FLOATING POINT IS PART OF THE SPECIFICATION HERE, NOT AN AFTERTHOUGHT
// ------------------------------------------------------------------------------------------
// 1. eps for carbon is `0.3*4 - 0.5` = 0.7000000000000002, NOT 0.7.  ETA_epsilon_3 for a
//    molecule with 14 skeleton atoms is 0.4400000000000004 and not 0.44 because of it.
// 2. mordred sums the epsilons ATOM BY ATOM over `Chem.AddHs`'s atom order, which is the
//    original atoms followed by the appended hydrogens.  0.3 added k times is not 0.3*k, so the
//    hydrogen tail is a loop below and not a multiply.
// 3. `sum(sum(... for j) for i)` in EtaCompositeIndex is a per-row inner accumulator folded into
//    an outer one.  Reassociating it moves the last bits of ETA_eta / ETA_eta_R.
// 4. The beta sums are made of 0.5 / 0.75 / 1.0 / 1.5 / 2.0 / 3.0 / 4.0 only, so they are exact
//    in double at any association and need no such care.
//
// ------------------------------------------------------------------------------------------
// WIRING (bindings.cpp is not edited by this file's author -- see NOTES_eta.md)
// ------------------------------------------------------------------------------------------
//     eta::Mol m;
//     eta::build_from_rows(m, n, nb, ai, N_ATOM_INT, bi, N_BOND_INT, bd, n_rings);
//     eta::compute(m, out + OFF_ETA, W.es);
// It reads atom columns A_Z, A_DEG, A_NH, A_AROM, A_RING, A_TVAL and bond columns B_U, B_V,
// B_CODE, B_BTYPE plus the bond-order doubles, and `n_rings` is the molecule's SymmSSSR count --
// the same `ring_ptr` span RingCount is given, and the same number `nRing` reports.
#ifndef HUME_ETA_H
#define HUME_ETA_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace eta {

inline constexpr int N_COLS = 29;

// The order the 29 values are written in.  It is the order of `results/dedupe2/agent_groups.json`
// group C_eta, and bindings.cpp's offset enum must agree with it.
enum {
  C_ALPHA = 0, C_SHAPE_P, C_SHAPE_Y, C_SHAPE_X,
  C_BETA, C_BETA_S, C_BETA_NS, C_BETA_NS_D,
  C_ETA, C_ETA_L, C_ETA_R, C_ETA_F, C_ETA_FL, C_ETA_B, C_ETA_BR,
  C_DALPHA_A, C_DALPHA_B,
  C_EPS1, C_EPS2, C_EPS3, C_EPS4, C_EPS5,
  C_DEPS_A, C_DEPS_B, C_DEPS_C, C_DEPS_D,
  C_DBETA, C_PSI1, C_DPSI_B
};

inline const char *col_name(int i) {
  static const char *const NAMES[N_COLS] = {
      "ETA_alpha",     "ETA_shape_p",    "ETA_shape_y",    "ETA_shape_x",
      "ETA_beta",      "ETA_beta_s",     "ETA_beta_ns",    "ETA_beta_ns_d",
      "ETA_eta",       "ETA_eta_L",      "ETA_eta_R",      "ETA_eta_F",
      "ETA_eta_FL",    "ETA_eta_B",      "ETA_eta_BR",     "ETA_dAlpha_A",
      "ETA_dAlpha_B",  "ETA_epsilon_1",  "ETA_epsilon_2",  "ETA_epsilon_3",
      "ETA_epsilon_4", "ETA_epsilon_5",  "ETA_dEpsilon_A", "ETA_dEpsilon_B",
      "ETA_dEpsilon_C", "ETA_dEpsilon_D", "ETA_dBeta",     "ETA_psi_1",
      "ETA_dPsi_B"};
  if (i < 0 || i >= N_COLS)
    throw std::out_of_range("eta::col_name: column index " + std::to_string(i) +
                            " outside 0.." + std::to_string(N_COLS - 1));
  return NAMES[i];
}

// --------------------------------------------------------------------------------------------
// The two element tables mordred reads.
// --------------------------------------------------------------------------------------------
// RDKit's `PeriodicTable::getNouterElecs`, Z = 0..118, dumped from the pinned rdkit
// (`Chem.GetPeriodicTable().GetNOuterElecs(z)`).  mordred calls it for BOTH `Z^v` in
// get_core_count and the `0.3 Z^v` in get_eta_epsilon, and it is not the group number: it is 11
// for Cu and 15 for Tm.
inline constexpr int N_Z = 119;
inline constexpr int NOUTER[N_Z] = {
    0,  1,  2,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,  1,
    2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 2,  3,  4,  5,  6,  7,  8,  1,  2,  3,
    4,  5,  6,  7,  8,  9,  10, 11, 2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  3,
    4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 4,  5,  6,  7,  8,  9,  10, 11,
    2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  3,  4,  5,  6,  7,  8,  9,  10, 11,
    12, 13, 14, 15, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2};

// mordred's OWN period table -- `PeriodicTable(([1]*2)+([2]*8)+([3]*8)+([4]*18)+([5]*18)+
// ([6]*32)+([7]*32))`, indexed 1-based, np.nan outside 1..118.  It is a row number, so it is 4
// for every element from K to Kr including the d block.  Returned as a double because the nan is
// load-bearing: mordred does not raise there, it produces a nan core count that propagates.
inline double period_of(int z) {
  static const int CUM[7] = {2, 10, 18, 36, 54, 86, 118};
  if (z < 1) return std::nan("");
  for (int k = 0; k < 7; ++k)
    if (z <= CUM[k]) return (double)(k + 1);
  return std::nan("");
}

//! `_atomic_property.get_core_count`.  Z == 1 short-circuits to 0.0 BEFORE the division, which
//! is why hydrogen never divides by (PN - 1) == 0.  Helium does, and mordred raises
//! ZeroDivisionError there; NaN is returned instead and propagates through exactly the same set
//! of columns the exception would have failed (see the note in compute()).
inline double core_count(int z) {
  if (z == 1) return 0.0;
  if (z < 0 || z >= N_Z)
    throw std::runtime_error("eta: atomic number " + std::to_string(z) +
                             " is outside 0..118; RDKit's GetNOuterElecs refuses it too");
  const double PN = period_of(z);
  if (std::isnan(PN)) return std::nan("");
  const int Zv = NOUTER[z];
  const double den = (double)Zv * (PN - 1.0);
  if (den == 0.0) return std::nan("");   // helium: mordred raises ZeroDivisionError
  return (double)(z - Zv) / den;
}

//! `_atomic_property.get_eta_epsilon` = 0.3 * Z^v - core.  Hydrogen is 0.3 exactly; carbon is
//! 0.3*4 - 0.5 = 0.7000000000000002 and NOT 0.7 (see the floating-point note at the top).
//!
//! THE PRODUCT IS A SEPARATE STATEMENT ON PURPOSE, and so is every other `a +- b*c` in this
//! file.  clang under plain `-O3` contracts a multiply into an adjacent add, and the fused
//! result is MORE accurate than the one Python computed -- which makes it a different number.
//! Written as one expression, 59 of the first 300 corpus molecules get a last-bit different
//! ETA_epsilon_2, and eight other columns move with it; split, all 29 are bit-identical to
//! mordred under the same `-O3`.  The rule is to match the reference, not to maximise accuracy;
//! it is the same rule (and the same fix) as cpp/verify_wiring.py's.
inline double eta_epsilon(int z) {
  if (z < 0 || z >= N_Z)
    throw std::runtime_error("eta: atomic number " + std::to_string(z) + " is outside 0..118");
  const double scaled = 0.3 * (double)NOUTER[z];
  const double core = core_count(z);
  return scaled - core;
}

// --------------------------------------------------------------------------------------------
// the molecule, in the boundary's own terms
// --------------------------------------------------------------------------------------------
//! One molecule.  `z/deg/nh/tval/arom/inring` are atom_i columns A_Z, A_DEG, A_NH, A_TVAL,
//! A_AROM, A_RING; `bu/bv/bcode/btype` are bond_i columns B_U, B_V, B_CODE, B_BTYPE and `bord`
//! is the bond-order double array.  `n_rings` is len(Chem.GetSymmSSSR(mol)), which
//! EtaBranchingIndex needs as mordred's `RingCount()` with all defaults.
struct Mol {
  int n = 0, nb = 0, n_rings = 0;
  std::vector<int32_t> z, deg, nh, tval, arom, inring;
  std::vector<int32_t> bu, bv, bcode, btype;
  std::vector<double> bord;
  // CSR over all atoms.  `nbr[k]` is the neighbour and `nbond[k]` the bond joining them, and the
  // entries of one atom are in BOND INDEX ORDER -- which is `Atom::GetBonds()`'s order, the order
  // mordred's per-atom beta sums walk.  (Those sums are exact in double at any association, so
  // this is for readability rather than for the last bit.)
  std::vector<int32_t> start, nbr, nbond;
};

//! Scratch reused across molecules so a batch does no per-molecule allocation.
struct Work {
  std::vector<double> core, eps, bsig, bnsig, bdelta, gamma, gammaR;
  std::vector<int32_t> takes, dist, distR, degR, compact, bfs;
  std::vector<char> kekdbl, used;
};

inline void build_from_rows(Mol &m, int n, int nb, const int32_t *arows, int astride,
                            const int32_t *brows, int bstride, const double *bordv,
                            int n_rings) {
  m.n = n;
  m.nb = nb;
  m.n_rings = n_rings;
  m.z.resize(n); m.deg.resize(n); m.nh.resize(n);
  m.tval.resize(n); m.arom.resize(n); m.inring.resize(n);
  for (int i = 0; i < n; ++i) {
    const int32_t *r = arows + (std::size_t)i * astride;
    m.z[i] = r[0];        // A_Z
    m.deg[i] = r[1];      // A_DEG
    m.nh[i] = r[2];       // A_NH
    m.arom[i] = r[5];     // A_AROM
    m.inring[i] = r[6];   // A_RING
    m.tval[i] = r[9];     // A_TVAL
  }
  m.bu.resize(nb); m.bv.resize(nb); m.bcode.resize(nb); m.btype.resize(nb); m.bord.resize(nb);
  m.start.assign(n + 1, 0);
  for (int b = 0; b < nb; ++b) {
    const int32_t *r = brows + (std::size_t)b * bstride;
    m.bu[b] = r[0];       // B_U
    m.bv[b] = r[1];       // B_V
    m.bcode[b] = r[4];    // B_CODE
    m.btype[b] = r[5];    // B_BTYPE
    m.bord[b] = bordv[b];
    if (m.bu[b] < 0 || m.bu[b] >= n || m.bv[b] < 0 || m.bv[b] >= n)
      throw std::runtime_error("eta: bond " + std::to_string(b) + " names atom " +
                               std::to_string(m.bu[b]) + "-" + std::to_string(m.bv[b]) +
                               " outside 0.." + std::to_string(n - 1));
    m.start[m.bu[b] + 1]++;
    m.start[m.bv[b] + 1]++;
  }
  for (int i = 0; i < n; ++i) m.start[i + 1] += m.start[i];
  m.nbr.assign(2 * (std::size_t)nb, 0);
  m.nbond.assign(2 * (std::size_t)nb, 0);
  std::vector<int32_t> cur(m.start.begin(), m.start.end() - 1);
  for (int b = 0; b < nb; ++b) {
    m.nbr[cur[m.bu[b]]] = m.bv[b];  m.nbond[cur[m.bu[b]]++] = b;
    m.nbr[cur[m.bv[b]]] = m.bu[b];  m.nbond[cur[m.bv[b]]++] = b;
  }
}

// --------------------------------------------------------------------------------------------
// RDKit's Bond::BondType integers, the only four this file has to name.  B_BTYPE carries the
// enum verbatim (SINGLE 1, DOUBLE 2, TRIPLE 3, AROMATIC 12, DATIVE 17, ...), which is what
// `get_eta_nonsigma_contribute` branches on -- it asks `is Chem.BondType.SINGLE` and then
// compares `GetBondTypeAsDouble()` against `Chem.BondType.TRIPLE`, an int-valued Boost enum, so
// the second test is "does this bond have order 3.0" and TRIPLE is the only type that does.
// --------------------------------------------------------------------------------------------
inline constexpr int BT_SINGLE = 1, BT_DOUBLE = 2, BT_TRIPLE = 3, BT_AROMATIC = 12;
inline constexpr int BC_AROM = 8;   // constit.h's bond-code bit for Bond::GetIsAromatic()

//! `Bond::getValenceContrib(atom)`, the same reading constit.h's nBondsKD reconstruction uses: a
//! DATIVE bond (bond code 0 -- none of SINGLE/DOUBLE/TRIPLE/aromatic-flagged) contributes 0 to
//! its DONOR and its order to its acceptor; everything else contributes its order to both.
inline double valence_contrib(const Mol &m, int e, int atom, double order) {
  if (m.bcode[e] == 0) return (m.bu[e] == atom) ? 0.0 : order;
  return order;
}

// --------------------------------------------------------------------------------------------
// the Kekule reconstruction
// --------------------------------------------------------------------------------------------
//! Perfect matching of the atoms that must carry a ring double bond, over the AROMATIC-type
//! bonds.  `takes` is 1 for those atoms.  Deterministic: it always extends the lowest-index
//! unmatched atom, trying its incident aromatic bonds in bond-index order.  `kekdbl[e]` comes
//! back 1 for the bonds that took the double.
//!
//! A perfect matching always exists -- RDKit kekulized this molecule to make the pickle -- so a
//! failure here means the boundary and the aromatic perception disagree, and it throws rather
//! than returning a silently wrong hydrogen count.
struct KekuleMatcher {
  const Mol &m;
  const std::vector<int32_t> &takes;
  std::vector<char> &kekdbl;
  std::vector<char> used;
  long budget;

  KekuleMatcher(const Mol &mm, const std::vector<int32_t> &t, std::vector<char> &k)
      : m(mm), takes(t), kekdbl(k), used((std::size_t)mm.n, 0), budget(4000000) {}

  bool extend(int from) {
    int u = -1;
    for (int i = from; i < m.n; ++i)
      if (takes[i] && !used[i]) { u = i; break; }
    if (u < 0) return true;
    if (--budget < 0)
      throw std::runtime_error(
          "eta: the Kekule matching for ETA_epsilon_4 exceeded its search budget; the aromatic "
          "system of this molecule is larger than this header is built for");
    used[u] = 1;
    for (int k = m.start[u]; k < m.start[u + 1]; ++k) {
      const int e = m.nbond[k], v = m.nbr[k];
      if (m.btype[e] != BT_AROMATIC) continue;
      if (!takes[v] || used[v]) continue;
      used[v] = 1;
      kekdbl[e] = 1;
      if (extend(u + 1)) return true;
      kekdbl[e] = 0;
      used[v] = 0;
    }
    used[u] = 0;
    return false;
  }
};

// --------------------------------------------------------------------------------------------
//! mordred's EtaCompositeIndex.calculate for BOTH the plain and the `local` variant in one pass.
//!
//! Transliterated including its association: `sum(sum(... for j) for i)` is an inner accumulator
//! per row i folded into an outer one, and reassociating it moves the last bits.  `local`
//! restricts to r == 1 and is the SAME term (r*r == 1), so the two sums share their square root
//! and differ only in which rows contribute -- which is why they are computed together.
//!
//! ONE BFS ROW AT A TIME, NEVER THE WHOLE MATRIX.  mordred materialises `Chem.GetDistanceMatrix`
//! and walks it twice per variant; nothing here needs row i after row i has been folded in, so
//! the n x n int array and the second traversal both go away.  That is a memory win and only a
//! small time one -- 55.7 -> 52.1 us/molecule on the 55+ stratum -- because what this loop costs
//! is one divide and one square root per atom PAIR, twice (this graph and the reference
//! skeleton), and mordred's expression admits no algebraic shortcut that rounds the same way.
//! See NOTES_eta.md for the one lever left.  The unreachable value is RDKit's 1e8 rather than
//! infinity, which matters only for a disconnected graph -- and `require_connected` has already
//! made those NaN.
//!
//! `map` is null (use every atom, compacted index == atom index) or a compacted-index map with
//! -1 for the atoms outside the subgraph -- which for the reference alkane is exactly the
//! hydrogen ATOMS, so the map alone defines that graph and no second element test is needed.
// --------------------------------------------------------------------------------------------
inline void composite_pair(const Mol &m, const std::vector<double> &g, const int32_t *map,
                           int nsub, std::vector<int32_t> &d, std::vector<int32_t> &q,
                           double &total, double &total_local) {
  const int UNREACH = 100000000;
  total = 0.0;
  total_local = 0.0;
  d.assign(nsub, UNREACH);
  q.resize(nsub);
  for (int src = 0; src < m.n; ++src) {
    const int i = map ? map[src] : src;
    if (i < 0) continue;
    std::fill(d.begin(), d.end(), UNREACH);
    d[i] = 0;
    int head = 0, tail = 0;
    q[tail++] = src;
    while (head < tail) {
      const int u = q[head++];
      const int du = d[map ? map[u] : u];
      for (int k = m.start[u]; k < m.start[u + 1]; ++k) {
        const int v = m.nbr[k];
        const int vv = map ? map[v] : v;
        if (vv < 0) continue;
        if (d[vv] == UNREACH) { d[vv] = du + 1; q[tail++] = v; }
      }
    }
    double inner = 0.0, inner_local = 0.0;
    for (int j = i + 1; j < nsub; ++j) {
      const int r = d[j];
      if (r == 0) continue;
      const double rr = (double)r;
      const double t = std::sqrt(g[i] * g[j] / (rr * rr));
      inner += t;
      if (r == 1) inner_local += t;
    }
    total += inner;
    total_local += inner_local;
  }
}

// --------------------------------------------------------------------------------------------
//! All 29 values for one molecule.  NaN exactly where mordred returns a missing value.
// --------------------------------------------------------------------------------------------
inline void compute(const Mol &m, double *out, Work &W) {
  const double NaN = std::nan("");
  const int n = m.n;
  for (int c = 0; c < N_COLS; ++c) out[c] = NaN;

  // `require_connected = True` on EtaBase: mordred hands back Missing(MultipleFragments()) for
  // every one of these 29 columns before `calculate` is ever reached.  104 of the 20,000 corpus
  // molecules are salts and take this path.
  if (n <= 0) return;
  {
    W.bfs.assign(n, 0);
    W.used.assign((std::size_t)n, 0);
    int head = 0, tail = 0;
    W.bfs[tail++] = 0;
    W.used[0] = 1;
    while (head < tail) {
      const int u = W.bfs[head++];
      for (int k = m.start[u]; k < m.start[u + 1]; ++k)
        if (!W.used[m.nbr[k]]) { W.used[m.nbr[k]] = 1; W.bfs[tail++] = m.nbr[k]; }
    }
    if (tail != n) return;
  }

  // ---- the per-atom primitives that need only Z ----
  W.core.assign(n, 0.0);
  W.eps.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    W.core[i] = core_count(m.z[i]);
    W.eps[i] = eta_epsilon(m.z[i]);
  }

  // ---- takesDouble, then the Kekule matching ----
  // takesDouble(i) is the number of atom i's AROMATIC-type bonds that kekulization turns into a
  // double.  Kekulization preserves every atom's valence and rewrites nothing but aromatic-type
  // bonds, so it is a difference of boundary integers.  It is in {0,1} on every molecule this
  // repo has seen; anything else means the boundary is not describing the molecule RDKit
  // kekulized, and is an error rather than a clamp.
  W.takes.assign(n, 0);
  W.kekdbl.assign((std::size_t)m.nb, 0);
  for (int i = 0; i < n; ++i) {
    int narom = 0;
    double nonarom = 0.0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
      const int e = m.nbond[k];
      if (m.btype[e] == BT_AROMATIC) {
        ++narom;
        if (!(m.bcode[e] & BC_AROM))
          throw std::runtime_error(
              "eta: bond " + std::to_string(e) + " has BondType AROMATIC but its aromatic flag "
              "is clear; get_eta_nonsigma_contribute reads both and this header assumes they "
              "agree");
      } else {
        nonarom += valence_contrib(m, e, i, m.bord[e]);
      }
    }
    if (narom == 0) continue;
    const int g = m.tval[i] - m.nh[i] - (int)std::floor(nonarom + 0.5) - narom;
    if (g < 0 || g > 1)
      throw std::runtime_error(
          "eta: atom " + std::to_string(i) + " (Z=" + std::to_string(m.z[i]) +
          ") wants " + std::to_string(g) + " aromatic double bonds; the Kekule reconstruction "
          "only holds for 0 or 1, so the boundary's valence or aromatic flags are inconsistent");
    W.takes[i] = g;
  }
  {
    KekuleMatcher km(m, W.takes, W.kekdbl);
    if (!km.extend(0))
      throw std::runtime_error(
          "eta: the atoms needing an aromatic double bond admit no perfect matching over this "
          "molecule's aromatic bonds; ETA_epsilon_4's saturated skeleton cannot be built");
  }
  // Every bond's order in mordred's KEKULIZED molecule.
  auto order_kek = [&](int e) -> double {
    if (m.btype[e] == BT_AROMATIC) return W.kekdbl[e] ? 2.0 : 1.0;
    return m.bord[e];
  };

  // ---- beta and gamma ----
  // get_eta_beta_sigma:   sum over NON-HYDROGEN neighbours of 0.5 / 0.75 by |d eps| <= 0.3.
  //   The filter is on the NEIGHBOUR, not on the atom, so the stray explicit hydrogens of the
  //   heavy graph DO get a beta of their own (0.75 towards the nitrogen they hang off) and DO
  //   contribute to ETA_beta_s.  Their core count is 0, so their gamma is 0 and they contribute
  //   nothing to ETA_eta -- but the difference between "gamma 0" and "gamma NaN" is the
  //   difference between a number and a missing value for the whole molecule.
  // get_eta_beta_non_sigma: sum over BONDS whose other end is not hydrogen.
  // get_eta_beta_delta:   0.5 for an atom that is neither aromatic nor in a ring, still has a
  //   lone pair (Z^v - total valence > 0) and touches an aromatic atom.
  W.bsig.assign(n, 0.0);
  W.bnsig.assign(n, 0.0);
  W.bdelta.assign(n, 0.0);
  W.gamma.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double bs = 0.0, bns = 0.0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
      const int e = m.nbond[k], j = m.nbr[k];
      if (m.z[j] == 1) continue;
      bs += (std::fabs(W.eps[j] - W.eps[i]) <= 0.3) ? 0.5 : 0.75;
      // `get_eta_nonsigma_contribute` returns 0.0 for a bond whose KEKULIZED type is SINGLE and
      // y*f for everything else -- so a DATIVE bond, order 1.0 but not type SINGLE, does count.
      const bool kek_single =
          (m.btype[e] == BT_SINGLE) || (m.btype[e] == BT_AROMATIC && !W.kekdbl[e]);
      if (!kek_single) {
        const double ok = order_kek(e);
        const double f = (ok == 3.0) ? 2.0 : 1.0;   // `GetBondTypeAsDouble() == BondType.TRIPLE`
        double y;
        if (m.bcode[e] & BC_AROM)               y = 2.0;
        else if (std::fabs(W.eps[i] - W.eps[j]) > 0.3) y = 1.5;
        else                                    y = 1.0;
        const double yf = y * f;   // separate statement: no FMA (see eta_epsilon's note)
        bns += yf;
      }
    }
    double d = 0.0;
    if (!m.arom[i] && !m.inring[i] && (NOUTER[m.z[i]] - m.tval[i]) > 0) {
      for (int k = m.start[i]; k < m.start[i + 1]; ++k)
        if (m.arom[m.nbr[k]]) { d = 0.5; break; }
    }
    W.bsig[i] = bs;
    W.bnsig[i] = bns;
    W.bdelta[i] = d;
    const double beta = bs + bns + d;
    W.gamma[i] = (beta == 0.0) ? NaN : W.core[i] / beta;
  }

  // ---- the reference alkane, as degrees rather than as a molecule ----
  // AlterMolecule(explicit_hydrogens=False): skip hydrogen ATOMS, make every survivor a carbon,
  // make every surviving bond single.  It refuses -- and every column built on it is Missing --
  // when any atom of any bond has degree > 4, because a carbon cannot take five bonds.  With
  // explicit hydrogens (EtaEpsilon(3)) the same test runs on `Chem.AddHs`'s degrees, which is
  // deg + nH.
  int A_R = 0, B_R = 0;
  W.compact.assign(n, -1);
  W.degR.assign(n, 0);
  for (int i = 0; i < n; ++i)
    if (m.z[i] != 1) W.compact[i] = A_R++;
  for (int b = 0; b < m.nb; ++b) {
    if (m.z[m.bu[b]] == 1 || m.z[m.bv[b]] == 1) continue;
    ++B_R;
    W.degR[m.bu[b]]++;
    W.degR[m.bv[b]]++;
  }
  bool ref_fail = false, ref_fail_h = false;
  for (int i = 0; i < n; ++i) {
    if (m.deg[i] > 4) ref_fail = true;
    if (m.deg[i] + m.nh[i] > 4) ref_fail_h = true;
  }

  // ---- alpha, the shape indices, the betas ----
  double alpha = 0.0;
  for (int i = 0; i < n; ++i) alpha += W.core[i];
  double alpha_R = 0.0;
  for (int i = 0; i < A_R; ++i) alpha_R += 0.5;   // core_count of a carbon, A_R times

  out[C_ALPHA] = alpha;
  {
    const int DEG[3] = {1, 3, 4};   // shape p, y, x
    for (int t = 0; t < 3; ++t) {
      double s = 0.0;
      for (int i = 0; i < n; ++i)
        if (m.deg[i] == DEG[t]) s += W.core[i];
      // mordred divides by alpha with Python's `/`: alpha == 0 is a ZeroDivisionError and a
      // missing value, not an infinity.
      out[C_SHAPE_P + t] = (alpha == 0.0) ? NaN : s / alpha;
    }
  }
  double beta_s = 0.0, beta_ns = 0.0, beta_nsd = 0.0, beta_all = 0.0;
  for (int i = 0; i < n; ++i) {
    const double bs = W.bsig[i] / 2.0;
    const double bns = W.bnsig[i] / 2.0 + W.bdelta[i];
    beta_s += bs;
    beta_ns += bns;
    beta_nsd += W.bdelta[i];
    beta_all += bs + bns;
  }
  out[C_BETA] = beta_all;
  out[C_BETA_S] = beta_s;
  out[C_BETA_NS] = beta_ns;
  out[C_BETA_NS_D] = beta_nsd;
  out[C_DBETA] = beta_ns - beta_s;

  // ---- the four composite indices ----
  double eta_v = 0.0, eta_L = 0.0;
  composite_pair(m, W.gamma, nullptr, n, W.dist, W.bfs, eta_v, eta_L);
  out[C_ETA] = eta_v;
  out[C_ETA_L] = eta_L;

  double eta_R = NaN, eta_RL = NaN;
  if (!ref_fail) {
    // gamma on the reference alkane.  Every atom is a carbon among carbons, so every sigma
    // contribution is 0.5, there are no non-single bonds and no delta term: beta_i = 0.5*deg_i
    // and gamma_i = 0.5 / (0.5 * deg_i).  Written that way rather than as 1/deg because that is
    // mordred's expression; both are exact here, and one of them is the specification.
    W.gammaR.assign(A_R, 0.0);
    for (int i = 0; i < n; ++i) {
      if (m.z[i] == 1) continue;
      double b = 0.0;
      for (int k = 0; k < W.degR[i]; ++k) b += 0.5;
      W.gammaR[W.compact[i]] = (b == 0.0) ? NaN : 0.5 / b;
    }
    composite_pair(m, W.gammaR, W.compact.data(), A_R, W.distR, W.bfs, eta_R, eta_RL);
    out[C_ETA_R] = eta_R;
    out[C_ETA_F] = eta_R - eta_v;
    out[C_ETA_FL] = eta_RL - eta_L;

    // EtaBranchingIndex.  N is the HEAVY GRAPH's atom count (stray hydrogens included), eta_RL
    // is the SKELETON's; N <= 1 is `self.fail(ValueError("single atom"))`.
    if (n > 1) {
      const double half = 0.5 * (double)(n - 3);
      const double eta_NL = (n == 2) ? 1.0 : (std::sqrt(2.0) + half);
      const double base = eta_NL - eta_RL;
      const double ring = 0.086 * (double)m.n_rings;
      out[C_ETA_B] = base + 0.0;      // mordred's `+ 0.086 * (None or 0)`
      out[C_ETA_BR] = base + ring;
    }

    out[C_DALPHA_A] = std::fmax((alpha - alpha_R) / (double)n, 0.0);
    out[C_DALPHA_B] = std::fmax((alpha_R - alpha) / (double)n, 0.0);
  }
  // `max(nan, 0.0)` in Python returns nan (every comparison against nan is False, so the first
  // argument stands).  std::fmax returns the NON-nan argument, so a nan alpha would come back
  // 0.0 here where mordred says missing.  Restore mordred's answer explicitly.
  if (std::isnan(alpha)) { out[C_DALPHA_A] = NaN; out[C_DALPHA_B] = NaN; }

  // ---- the five epsilons ----
  // Each is a mean over a DIFFERENT atom set, summed in that molecule's own atom order.  The
  // hydrogen tails are loops because 0.3 added k times is not 0.3*k.
  int nh_total = 0;
  for (int i = 0; i < n; ++i) nh_total += m.nh[i];

  {   // epsilon_1: Chem.AddHs(mol) -- original atoms then the appended hydrogens
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += W.eps[i];
    for (int h = 0; h < nh_total; ++h) s += 0.3;
    out[C_EPS1] = s / (double)(n + nh_total);
  }
  double eps2 = NaN;
  {   // epsilon_2: the heavy graph, hydrogen ATOMS included in both sum and count
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += W.eps[i];
    eps2 = s / (double)n;
    out[C_EPS2] = eps2;
  }
  double eps3 = NaN;
  if (!ref_fail_h) {   // epsilon_3: the reference alkane WITH hydrogens
    const int H_R = 4 * A_R - 2 * B_R;
    const double epsC = eta_epsilon(6);
    double s = 0.0;
    for (int i = 0; i < A_R; ++i) s += epsC;
    for (int h = 0; h < H_R; ++h) s += 0.3;
    eps3 = s / (double)(A_R + H_R);
    out[C_EPS3] = eps3;
  }
  double eps4 = NaN;
  {   // epsilon_4: the saturated skeleton -- same elements and charges, C-C bonds reduced to
      // single, every bond touching a heteroatom left at its kekulized order, then AddHs.
      // Its hydrogen count is reconstructed rather than re-sanitized; see the header note.
    double hs = 0.0, s = 0.0;
    for (int i = 0; i < n; ++i) {
      if (m.z[i] == 1) continue;
      s += W.eps[i];
      double implicit = (double)m.nh[i];
      for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
        const int e = m.nbond[k], j = m.nbr[k];
        const double vc = valence_contrib(m, e, i, order_kek(e));
        if (m.z[j] == 1)                          implicit += vc;          // the bond is dropped
        else if (m.z[i] == 6 && m.z[j] == 6)      implicit += vc - 1.0;    // reduced to single
      }
      hs += implicit;
    }
    const long H_S = std::lround(hs);
    if (H_S < 0 || std::fabs(hs - (double)H_S) > 1e-9)
      throw std::runtime_error(
          "eta: the saturated skeleton's hydrogen count came out as " + std::to_string(hs) +
          ", which is not a non-negative integer; ETA_epsilon_4's reconstruction has broken");
    for (long h = 0; h < H_S; ++h) s += 0.3;
    eps4 = s / (double)(A_R + H_S);
    out[C_EPS4] = eps4;
  }
  double eps5 = NaN;
  {   // epsilon_5: heavy atoms, plus only those hydrogens that are NOT bonded to carbon.
      // mordred asks `a.GetNeighbors()[0]` on the hydrogen-added molecule, so a hydrogen ATOM
      // already in the heavy graph is tested the same way as an appended one.
    double s = 0.0;
    long cnt = 0;
    for (int i = 0; i < n; ++i) {
      if (m.z[i] != 1) { s += W.eps[i]; ++cnt; continue; }
      if (m.start[i] == m.start[i + 1])
        throw std::runtime_error(
            "eta: atom " + std::to_string(i) + " is a hydrogen with no bonds; mordred's "
            "ETA_epsilon_5 indexes GetNeighbors()[0] and would raise IndexError");
      if (m.z[m.nbr[m.start[i]]] != 6) { s += W.eps[i]; ++cnt; }
    }
    long nh_noncarbon = 0;
    for (int i = 0; i < n; ++i)
      if (m.z[i] != 6) nh_noncarbon += m.nh[i];
    for (long h = 0; h < nh_noncarbon; ++h) { s += 0.3; ++cnt; }
    eps5 = (cnt == 0) ? NaN : s / (double)cnt;
    out[C_EPS5] = eps5;
  }
  out[C_DEPS_A] = out[C_EPS1] - eps3;
  out[C_DEPS_B] = out[C_EPS1] - eps4;
  out[C_DEPS_C] = eps3 - eps4;
  out[C_DEPS_D] = eps2 - eps5;

  // ---- psi ----
  {
    const double den = (double)n * eps2;
    const double psi = (den == 0.0) ? NaN : alpha / den;
    out[C_PSI1] = psi;
    out[C_DPSI_B] = std::isnan(psi) ? NaN : std::fmax(psi - 0.714, 0.0);
  }
}

}  // namespace eta

#endif
