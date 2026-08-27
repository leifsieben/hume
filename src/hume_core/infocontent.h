// InformationContent (33 surviving mordred columns) and Ipc / AvgIpc, as a header the
// extension can call.
//
// ============================================================================================
// READ THIS FIRST: THIS FILE DELIBERATELY DOES NOT REPRODUCE MORDRED, AND THAT IS THE POINT.
// ============================================================================================
//
// PORT_STATUS.md house rule 1 says: reproduce a QUIRK, diverge from an ILL-POSED DEFINITION,
// and the test is whether the upstream descriptor is a function of the molecule. mordred's
// InformationContent is NOT a function of the molecule. It has two independent defects, both
// established by measurement on cpp/hard.smi rather than by reading:
//
//   1. IT KEKULIZES BEFORE BUILDING THE ATOM-EQUIVALENCE CODES. `InformationContentBase` sets
//      `kekulize = True`, so mordred/_base/context.py hands `Ag.calculate` a mol on which
//      `Chem.Kekulize` has run, and `BFSTree.__init__` records `b.GetBondType()` from THAT mol.
//      An aromatic bond therefore enters the code as SINGLE or DOUBLE depending on which Kekule
//      structure the perceiver happened to pick, and the equivalence classes move with it.
//
//   2. ITS BFS TREE MUTATES A VISITED SET WHILE ITERATING OVER IT. `BFSTree._expand` is
//
//          for src, dst in list(tree.items()):
//              self.visited.add(src)
//              if dst is ():
//                  tree[src] = {n.GetIdx(): () for n in self.atoms[src][2]
//                               if n.GetIdx() not in self.visited}
//
//      Two SIBLINGS at the same depth that happen to be adjacent are both in `tree` but neither
//      is in `visited` when the loop starts. Whichever the dict yields FIRST is added to
//      `visited` and then claims the other as its CHILD; the second one, now looking at a
//      `visited` that contains the first, does not claim it back. Dict order here is insertion
//      order, which is `GetNeighbors()` order, which is atom numbering. So the tree -- and every
//      code built from it -- depends on how the atoms happen to be numbered.
//
// MEASURED CONSEQUENCE: 20.4% of the 100,000 molecules in cpp/hard.smi change at least one
// InformationContent column when the atoms are renumbered with `Chem.RenumberAtoms`. There is
// nothing there to be bit-exact against, and "we reproduce mordred" would be a claim about a
// coin flip.
//
// ------------------------------------------------------------------------------------------
// THE WORKED EXAMPLE. This is the evidence that what follows is a repair and not a discrepancy.
// ------------------------------------------------------------------------------------------
//
//   SMILES   ON=Cc1ccccn1     pyridine-2-carbaldehyde oxime. 9 heavy atoms, 15 with H added.
//                             It is the smallest molecule in cpp/hard.smi that flips.
//
//   numbering A -- exactly as parsed:  0=O 1=N 2=C 3=c 4=c 5=c 6=c 7=c 8=n
//       mordred  IC1 = 2.682588730501833     IC2 = 3.4565647621309532
//   numbering B -- ONE TRANSPOSITION, Chem.RenumberAtoms(mol, [5,1,2,3,4,0,6,7,8]), i.e. swap
//                  the oxime oxygen with one ring CH, then SanitizeMol
//       mordred  IC1 = 2.8159220638351665    IC2 = 3.5898980954642865
//
//   Same molecule, same SMILES, same mordred, same RDKit, one swapped label: two answers, 5%
//   apart. TIC1/SIC1/BIC1/CIC1/MIC1/ZMIC1 and the order-2 row all move with them.
//
//   NOTE WHICH DEFECT THIS IS. `IC1` moves, and defect 2 CANNOT reach order 1 -- at order 1 the
//   tree is the root and its neighbours and no sibling has been expanded yet. So this is
//   defect 1: pyridine has two Kekule structures, RDKit's choice between them depends on atom
//   order, and the chosen bond orders go straight into the codes. (Defect 2 needs a cycle to
//   put two adjacent atoms at the same depth, so it starts at order 2.)
//
//   THIS FILE returns IC1 = 2.8159220638351665 and IC2 = 3.5898980954642865 for BOTH numberings,
//   for all 36 transpositions, and for the canonical-SMILES round trip, because the aromatic
//   bonds never become single or double. Which of mordred's two values we land on is not the
//   point and is not always either of them; the point is that there is exactly one.
//
//   AND THE CONVERSE CONTROL: an ACYCLIC molecule has no aromatic bond (defect 1 cannot fire)
//   and cannot have two adjacent atoms at equal distance from any root (defect 2 cannot fire),
//   so on acyclic molecules this file must agree with mordred EXACTLY at every order. That is
//   checked on the corpus by cpp/verify_ic.py rather than argued.
//
// ------------------------------------------------------------------------------------------
// THE RESOLUTION IMPLEMENTED HERE -- decided by the project owner, recorded in PORT_STATUS.md
// ------------------------------------------------------------------------------------------
//
//   R1. AN AROMATIC BOND KEEPS ITS OWN BOND-TYPE SYMBOL. `SYM_AROMATIC` is a fifth symbol
//       alongside single / double / triple / other, rather than being kekulized into one of
//       them. Nothing about a Kekule choice can reach the codes.
//
//   R2. THE TREE IS LAYERED BY GRAPH DISTANCE. The children of a node at depth d are its
//       neighbours at distance d+1 from the ROOT -- the BFS DAG of shortest paths -- so a
//       sibling is never a child of a sibling and atom numbering cannot reach the codes either.
//       Leaves are nodes at depth `order` and dead ends short of it, exactly as mordred's
//       `()`-vs-`{}` termination gives.
//
//   R3 (found during this port, not previously recorded). MIC's PER-CLASS WEIGHT WAS ALSO
//       ILL-POSED. mordred builds `ad = {a: i for i, a in enumerate(atoms)}`, a dict keyed by
//       CODE, so `ad[k]` is the LAST atom index carrying code k; `ModifiedIC` then weights the
//       class by `GetMass()` of THAT atom. Every atom in a class shares an atomic number, so
//       this is invisible until isotopes appear -- and 5.0% of cpp/hard.smi carries an isotope
//       label, where the class weight becomes "the mass of whichever labelled atom was numbered
//       last". Resolution: the weight is the STANDARD ATOMIC WEIGHT OF THE ELEMENT, which is
//       bit-identical to `GetMass()` for every unlabelled atom and is therefore a change only
//       where mordred had no defensible value. ZMIC is not affected -- its weight is the atomic
//       NUMBER, which is constant within a class by construction.
//
// CONSEQUENCES, EXPECTED AND CORRECT:
//   * ORDER 0 IS UNCHANGED. It never builds a tree -- `Ag.calculate` short-circuits to
//     `[a.GetAtomicNum() for a in mol.GetAtoms()]` -- so R1 and R2 cannot touch it. It is the
//     CONTROL: if IC0/TIC0/SIC0/CIC0/MIC0/ZMIC0 do not match mordred, this file has a bug.
//   * ORDERS 1-5 DIFFER FROM MORDRED BY DESIGN. cpp/verify_ic.py quantifies the divergence and
//     splits it into "mordred was unstable here anyway" and "mordred was stable and we still
//     differ"; see its docstring for the numbers and the characterisation of the second set.
//   * WHAT IS CLAIMED INSTEAD OF EXACTNESS IS DETERMINISM, and it is demonstrated rather than
//     asserted: identical output for every column on every molecule under random renumbering
//     and canonical-SMILES round trip.
//
// ============================================================================================
// WHAT IS *NOT* CHANGED, AND THE ARITHMETIC THAT IS REPRODUCED EXACTLY
// ============================================================================================
//
// THE MOLECULE IS THE HYDROGEN-ADDED, and for B the KEKULIZED, ONE. `Descriptor` defaults to
// `explicit_hydrogens = True` and `InformationContentBase` sets `kekulize = True`, so
// context.py hands these descriptors `Chem.Kekulize(Chem.AddHs(mol))`. Every count below is
// over that graph: A is `GetNumAtoms()` WITH hydrogens, the atom code is
// `(GetAtomicNum(), GetDegree())` with hydrogens as real neighbours AND real nodes, and B is
// `sum(b.GetBondTypeAsDouble())` over the KEKULIZED bonds. Missing the AddHs makes every one of
// the 33 columns wrong; this was checked by transcribing mordred into Python and reproducing
// all 42 columns bit-for-bit before a line of C++ was written.
//
// B IS RECOVERED WITHOUT KEKULIZING, AND THE RECOVERY IS VERIFIED, NOT ASSUMED. We do not
// kekulize (R1), but BIC needs the kekulized bond-order sum, which for pyrrole (7) differs from
// the aromatic-form sum (7.5). `kekuleBondOrderSum()` below rebuilds it from valences:
//
//     B = sum(order of every non-aromatic bond) + n_aromatic_bonds + k
//     k = (number of aromatic atoms that still need a ring double bond) / 2
//
// where an aromatic atom needs one iff its accumulated valence is short of the default valence
// for its element and charge -- RDKit's own Kekulize candidate test. A DATIVE bond contributes
// 0 to its DONOR and its order to its ACCEPTOR (RDKit's `Bond::getValenceContrib`), which is
// the only place the direction of a boundary bond row matters and the one case that a
// symmetric accumulation gets wrong. VERIFIED: 100,000 / 100,000 molecules of cpp/hard.smi
// reproduce `sum(GetBondTypeAsDouble())` over the AddHs+Kekulize mol exactly, with the number
// of "needs a double bond" atoms even on every single one.
//
// THE SUMMATION ORDER IS NUMPY'S. mordred's `shannon_entropy` is `-np.sum(w * (a/N)*log2(a/N))`
// and numpy's `np.sum` is PAIRWISE, not sequential. `pairwiseSum()` below is a transcription of
// numpy's `pairwise_sum_DOUBLE` (sequential under 8, 8-way unrolled to 128, recursive split
// above). Without it the order-0 control is a 1-ulp story instead of a bit-exact one.
//
// ============================================================================================
// Ipc / AvgIpc -- rdkit/Chem/GraphDescriptors.py, a DIFFERENT graph and a DIFFERENT problem
// ============================================================================================
//
// *** OPEN BUG, DIAGNOSED 2026-08-27, NOT YET FIXED. Ipc / AvgIpc / Log2Ipc ONLY -- the 42
// *** mordred InformationContent columns below are unaffected and are deterministic.
//
// Our Ipc is NUMBERING-DEPENDENT on 56 of 2,000 corpus molecules (2.8%). RDKit's is NOT, and
// RDKit's is CORRECT. That combination rules out the comfortable explanation, so it is written
// down here rather than left as "a floating-point difference":
//
//   * RDKit stable: 6 random renumberings of the example below give ONE value, 3597222335.9511533.
//   * RDKit correct: exact integer Faddeev-Le Verrier over the same adjacency matrix gives
//     AvgIpc = 3.78859017853964 against RDKit's 3.7885901785396405 -- agreement to ~1e-15.
//   * So OURS is the one that moves, and ours is the one that is wrong.
//
//   example: CC(=O)OC1CC2(C)C3(CO3)C1OC1C=C(C)C(=O)C(O)C12CO.CCCCCCCCCCCCCCCCCCl.CO
//            44 heavy atoms, THREE FRAGMENTS.
//
// AND THE OVERFLOW STORY DOES NOT APPLY HERE. The largest |coefficient| for that molecule is 9
// DIGITS -- comfortably inside double -- so the power-of-two rescaling described below should
// never engage, and the accuracy budget is not the explanation. Suspect the rescaling triggering
// spuriously, or the disconnected-graph path: the char poly of a disconnected graph is the
// PRODUCT of its fragments' polynomials, and a fragment loop is exactly where an atom-order
// dependence would enter. Note 339 of the 2,000 are multi-fragment while only 56 are unstable,
// so being disconnected is not sufficient on its own -- size interacts with it.
//
// The determinism check that finds these lives in cpp/verify_ic.py ("1. DETERMINISM"), and the
// exact-integer oracle to check a candidate fix against is the Faddeev recurrence stated
// correctly: M_0 = 0, c_0 = 1, M_k = A M_{k-1} + c_{k-1} I, c_k = -tr(A M_k)/k. Getting that
// recurrence subtly wrong (seeding M_1 = A, or taking tr(M) instead of tr(A M)) produces
// plausible 24-digit coefficients and a confidently wrong answer.
//
//   adjMat = (GetDistanceMatrix(mol, 0) == 1)      <- HYDROGEN-SUPPRESSED, unlike everything
//   cPoly  = abs(CharacteristicPolynomial(mol, adjMat))       above
//   Ipc    = sum(cPoly) * InfoEntropy(cPoly)
//   AvgIpc =                InfoEntropy(cPoly)
//
// CharacteristicPolynomial is Le Verrier-Faddeev-Frame and returns n+1 coefficients, leading 1
// first. InfoEntropy is rdInfoTheory's C++ one: normalise by the sum, skip zero entries, natural
// log divided by log 2.
//
// THE NUMERICS. The coefficients are integers that grow roughly like the number of k-matchings,
// and BOTH the coefficients and the Faddeev iterate matrix overflow IEEE double for molecules
// that are entirely ordinary. Measured on cpp/hard.smi -- see cpp/verify_ic.py for the run:
//
//     RDKit's Ipc goes non-finite at (see verify log; smallest overflowing heavy-atom count and
//     the fraction of the corpus affected are printed by `verify_ic.py ipc`).
//
// WHAT THIS FILE DOES ABOUT IT, and it is not "return inf quietly":
//   * The Faddeev iterate and the running coefficient are RESCALED BY A POWER OF TWO whenever
//     they threaten to overflow. The recurrence M_k = A(M_{k-1} + c_{k-1} I), c_k = -tr(M_k)/k
//     is HOMOGENEOUS in the pair (M_{k-1}, c_{k-1}), so scaling both by s scales every later
//     coefficient by exactly s and nothing else changes. Each coefficient is therefore kept as
//     (mantissa, binary exponent) and log2|c_k| is always finite and always exact to the working
//     precision. Powers of two are used so the rescaling itself introduces no rounding.
//   * AVGIPC IS COMPUTED FROM THE LOGS, via a max-subtracted softmax, so it NEVER overflows and
//     is defined on every molecule in the corpus. AvgIpc is the column that survives dedupe.
//   * IPC = sum(cPoly) * AvgIpc is returned exactly when it is representable. Past that it
//     SATURATES AT DBL_MAX and the row's `ipcOverflow` flag is set -- it is never silently inf
//     -- and `Log2Ipc`, which is exact and finite everywhere, is emitted as a third column
//     beside it so no information is lost at the boundary.
//   * DOUBLE PRECISION IS ITSELF A LIMIT, separately from overflow: the coefficients are exact
//     integers with more than 53 bits long before they reach 1e308. cpp/verify_ic.py checks the
//     double answer against exact integer Faddeev-Le Verrier and reports the heavy-atom count at
//     which AvgIpc's relative error passes 1e-9. RDKit has the same limit and does not say so.
//
// ============================================================================================
// WHAT THE CALLER MUST SUPPLY -- all of it already exists at bindings.cpp's boundary
// ============================================================================================
//
// Boundary as of 2026-08-27: atom_i is (n_atoms, 9) -- Z, deg, nH, fchg, hyb, arom, ring, cip,
// nring. This file reads five of the nine and needs nothing that is not already there; the
// per-atom ring COUNT in the new column 8 is not used here.
//
//   per atom   z      GetAtomicNum()               atom_i[:, 0]
//              deg    GetDegree()   (heavy)        atom_i[:, 1]   (cross-check only)
//              nh     GetTotalNumHs(False)         atom_i[:, 2]
//              chg    GetFormalCharge()            atom_i[:, 3]
//              arom   GetIsAromatic()              atom_i[:, 5]
//   per bond   u, v   begin/end atom index         bond_i[:, 0], bond_i[:, 1]
//              code   SMARTS bond code             bond_i[:, 4]
//              order  GetBondTypeAsDouble()        bond_d
//
// `u` MUST BE THE BEGIN ATOM and `v` THE END ATOM, because that is what makes a dative bond's
// donor and acceptor distinguishable; _extract.py already fills them that way.
//
// The bond code is _extract.py's: bit 0 SINGLE, bit 1 DOUBLE, bit 2 TRIPLE, bit 3 the aromatic
// FLAG, and 0 for everything RDKit will not name (DATIVE and friends). A bond whose TYPE is
// AROMATIC is exactly `(code & 7) == 0 && (code & 8) != 0`, and that is the only bond that gets
// SYM_AROMATIC -- a TRIPLE bond carrying the aromatic flag, which cpp/mols.smi contains, is
// still a triple bond to mordred's kekulized `GetBondType()` and is still a triple bond here.
//
// Hydrogens are not in the boundary and do not need to be: `nh` per heavy atom is exactly what
// `Chem.AddHs` would add, and an isotopic [2H] that is already an atom in its own right arrives
// as an ordinary Z=1 heavy row and is not double counted. That correspondence is checked
// per molecule by cpp/verify_ic.py against the real `Chem.AddHs` graph.
//
//   #include "infocontent.h"
//   infoic::selfCheck();                       // once, at module load
//   infoic::Row r; infoic::compute(mol, r);    // 45 doubles, names in infoic::COLS
//
#ifndef HUME_INFOCONTENT_H
#define HUME_INFOCONTENT_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ic_tables.h"

namespace infoic {

// --------------------------------------------------------------------------------------------
// Columns. 7 families x 6 orders, then the three Ipc outputs. The 33 that survive data/
// dedupe.json are a SUBSET of the 42; computing the other 9 is free (they share every Ag) and
// dropping them would violate house rule 7 the moment the dedupe is rerun at another threshold.
// --------------------------------------------------------------------------------------------
enum { MAX_ORDER = 5, N_ORDERS = 6, N_FAM = 7, N_IC = N_FAM * N_ORDERS, N_COLS = N_IC + 3 };
enum { F_IC = 0, F_TIC, F_SIC, F_BIC, F_CIC, F_MIC, F_ZMIC };
enum { C_IPC = N_IC, C_AVGIPC, C_LOG2IPC };

inline const char *const *columnNames() {
  static const char *n[N_COLS] = {
      "IC0",   "IC1",   "IC2",   "IC3",   "IC4",   "IC5",
      "TIC0",  "TIC1",  "TIC2",  "TIC3",  "TIC4",  "TIC5",
      "SIC0",  "SIC1",  "SIC2",  "SIC3",  "SIC4",  "SIC5",
      "BIC0",  "BIC1",  "BIC2",  "BIC3",  "BIC4",  "BIC5",
      "CIC0",  "CIC1",  "CIC2",  "CIC3",  "CIC4",  "CIC5",
      "MIC0",  "MIC1",  "MIC2",  "MIC3",  "MIC4",  "MIC5",
      "ZMIC0", "ZMIC1", "ZMIC2", "ZMIC3", "ZMIC4", "ZMIC5",
      "Ipc",   "AvgIpc", "Log2Ipc"};
  return n;
}

struct Row {
  double v[N_COLS];
  bool ipcOverflow = false;   // Ipc saturated at DBL_MAX; Log2Ipc is still exact
};

// --------------------------------------------------------------------------------------------
// Bond symbols. The VALUES are arbitrary -- a code is a sorted multiset of paths and only
// equality of codes ever matters -- but they are kept distinct and stable so a dumped code is
// readable. SYM_AROMATIC is repair R1: it exists precisely so that no Kekule choice can reach
// the codes.
// --------------------------------------------------------------------------------------------
enum : uint8_t { SYM_OTHER = 0, SYM_SINGLE = 1, SYM_DOUBLE = 2, SYM_TRIPLE = 3,
                 SYM_AROMATIC = 4, SYM_NONE = 15 };

inline uint8_t symbolFromCode(int code) {
  if (code & 1) return SYM_SINGLE;
  if (code & 2) return SYM_DOUBLE;
  if (code & 4) return SYM_TRIPLE;
  if (code & 8) return SYM_AROMATIC;   // GetBondType() == AROMATIC; see the header note
  return SYM_OTHER;                    // DATIVE and anything else RDKit will not name
}

// --------------------------------------------------------------------------------------------
// Input: the HEAVY-ATOM graph, in exactly the shape bindings.cpp already has. Hydrogens are
// materialised inside compute(); see build().
// --------------------------------------------------------------------------------------------
struct Mol {
  int n = 0, nb = 0;
  std::vector<uint8_t> z, nh, arom;
  std::vector<int8_t> chg;
  std::vector<int32_t> bu, bv;
  std::vector<uint8_t> bcode;
  std::vector<double> bord;

  void alloc(int na, int nbonds) {
    n = na; nb = nbonds;
    z.assign(na, 0); nh.assign(na, 0); arom.assign(na, 0); chg.assign(na, 0);
    bu.assign(nbonds, 0); bv.assign(nbonds, 0); bcode.assign(nbonds, 0);
    bord.assign(nbonds, 0.0);
  }
};

// --------------------------------------------------------------------------------------------
// The hydrogen-added graph mordred actually descriptors. Heavy atoms keep their indices; the
// hydrogens `Chem.AddHs` would append are appended here in the same order (heavy atom 0's
// hydrogens first), which is not load bearing -- nothing downstream reads an index -- but makes
// a dump comparable with RDKit's atom-by-atom.
// --------------------------------------------------------------------------------------------
struct HGraph {
  int N = 0;
  std::vector<uint8_t> z, deg;
  std::vector<int32_t> start, nbr;
  std::vector<uint8_t> sym;

  void build(const Mol &m) {
    int nhtot = 0;
    for (int i = 0; i < m.n; i++) nhtot += m.nh[i];
    N = m.n + nhtot;
    z.assign(N, 1); deg.assign(N, 1);
    for (int i = 0; i < m.n; i++) z[i] = m.z[i];
    std::vector<int32_t> cnt(N, 0);
    for (int b = 0; b < m.nb; b++) { cnt[m.bu[b]]++; cnt[m.bv[b]]++; }
    for (int i = 0; i < m.n; i++) cnt[i] += m.nh[i];       // the C-H bonds
    for (int q = m.n; q < N; q++) cnt[q] = 1;              // every hydrogen is terminal
    start.assign(N + 1, 0);
    for (int i = 0; i < N; i++) start[i + 1] = start[i] + cnt[i];
    nbr.assign(start[N], 0); sym.assign(start[N], SYM_SINGLE);
    std::vector<int32_t> cur(start.begin(), start.end() - 1);
    for (int b = 0; b < m.nb; b++) {
      const uint8_t s = symbolFromCode(m.bcode[b]);
      const int u = m.bu[b], v = m.bv[b];
      nbr[cur[u]] = v; sym[cur[u]++] = s;
      nbr[cur[v]] = u; sym[cur[v]++] = s;
    }
    int h = m.n;
    for (int i = 0; i < m.n; i++)
      for (int q = 0; q < m.nh[i]; q++, h++) {
        nbr[cur[i]] = h; sym[cur[i]++] = SYM_SINGLE;
        nbr[cur[h]] = i; sym[cur[h]++] = SYM_SINGLE;
      }
    for (int i = 0; i < N; i++) deg[i] = (uint8_t)std::min(start[i + 1] - start[i], 255);
  }
};

// --------------------------------------------------------------------------------------------
// numpy's pairwise summation, transcribed from numpy/core/src/umath/loops_utils.h.src
// (`pairwise_sum_DOUBLE`). mordred's entropies go through `np.sum`, so reproducing the ORDER of
// the additions is what makes the order-0 control bit-exact rather than 1-ulp-ish. Any change
// here is a change to the reference, not an optimisation.
// --------------------------------------------------------------------------------------------
inline double pairwiseSum(const double *a, size_t n) {
  const size_t PW_BLOCKSIZE = 128;
  if (n < 8) {
    double res = 0.0;
    for (size_t i = 0; i < n; i++) res += a[i];
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    double r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
}

// mordred: shannon_entropy(a, w) = -np.sum(w * (a/N) * log2(a/N)), N = np.sum(a).
// `w` is scalar 1 for IC and a per-class vector for MIC / ZMIC. The temporaries are built in the
// same order numpy would build them.
inline double shannonEntropy(const double *a, const double *w, size_t k, double *scratch) {
  const double N = pairwiseSum(a, k);
  for (size_t j = 0; j < k; j++) {
    const double p = a[j] / N;
    scratch[j] = (w ? w[j] : 1.0) * (p * std::log2(p));
  }
  return -pairwiseSum(scratch, k);
}

// --------------------------------------------------------------------------------------------
// B: the KEKULIZED bond-order sum over the H-added mol, rebuilt without kekulizing. See the
// header note. Verified 100,000/100,000 against mordred's own value on cpp/hard.smi.
// --------------------------------------------------------------------------------------------
inline double kekuleBondOrderSum(const Mol &m) {
  double tot = 0.0;
  int narom = 0;
  std::vector<double> used(m.n, 0.0);
  for (int i = 0; i < m.n; i++) { tot += m.nh[i]; used[i] += m.nh[i]; }  // the C-H bonds
  for (int b = 0; b < m.nb; b++) {
    const int u = m.bu[b], v = m.bv[b], code = m.bcode[b];
    const bool aromatic = ((code & 7) == 0) && (code & 8);
    if (aromatic) { narom++; used[u] += 1.0; used[v] += 1.0; continue; }
    tot += m.bord[b];
    if (code == 0) {
      used[v] += m.bord[b];         // DATIVE: 0 to the donor (u), the order to the acceptor (v)
    } else {
      used[u] += m.bord[b];
      used[v] += m.bord[b];
    }
  }
  int need = 0;
  for (int i = 0; i < m.n; i++) {
    if (!m.arom[i]) continue;
    const int dv = ic_tbl::DEFAULT_VALENCE[m.z[i]];
    if (dv < 0) continue;                       // no default valence: cannot be a candidate
    const int ev = m.chg[i] == 0
                       ? dv
                       : (ic_tbl::N_OUTER_ELECS[m.z[i]] >= 5 ? dv + m.chg[i]
                                                             : dv - std::abs((int)m.chg[i]));
    if (used[i] < (double)ev) need++;
  }
  return tot + (double)narom + 0.5 * (double)need;
}

// --------------------------------------------------------------------------------------------
// The atom-equivalence codes, repaired. A code is the sorted multiset of the root-to-leaf paths
// of the DISTANCE-LAYERED tree (repair R2), each path a run of (Z, degree) nodes joined by bond
// SYMBOLS (repair R1), 0xFF-terminated and zero-padded so that concatenation is injective.
// --------------------------------------------------------------------------------------------
enum { KEY_BYTES = 24, MAX_PATHS = 65536 };
struct Key { uint8_t b[KEY_BYTES]; };
inline bool keyLess(const Key &x, const Key &y) { return std::memcmp(x.b, y.b, KEY_BYTES) < 0; }

struct Codes {
  // Blob per atom: sorted keys back to back. Atoms are grouped by comparing blobs.
  std::vector<uint8_t> arena;
  std::vector<int32_t> off;         // off[i] .. off[i+1]
};

class CodeBuilder {
 public:
  void reset(const HGraph &g) {
    g_ = &g;
    dist_.assign(g.N, -1);
    stamp_.assign(g.N, 0);
    epoch_ = 0;
    bfs_.reserve(g.N);
  }

  // All root-to-leaf paths of the depth-`order` layered tree rooted at `root`.
  void codeFor(int root, int order, std::vector<Key> &out) {
    out.clear();
    const HGraph &g = *g_;
    ++epoch_;
    bfs_.clear();
    bfs_.push_back(root);
    stamp_[root] = epoch_; dist_[root] = 0;
    for (size_t h = 0; h < bfs_.size(); h++) {
      const int u = bfs_[h];
      if (dist_[u] >= order) continue;
      for (int e = g.start[u]; e < g.start[u + 1]; e++) {
        const int v = g.nbr[e];
        if (stamp_[v] != epoch_) {
          stamp_[v] = epoch_; dist_[v] = dist_[u] + 1;
          bfs_.push_back(v);
        }
      }
    }
    Key k;
    std::memset(k.b, 0, KEY_BYTES);
    k.b[0] = g.z[root];
    k.b[1] = g.deg[root];
    walk(root, 0, order, 2, k, out);
    std::sort(out.begin(), out.end(), keyLess);
  }

 private:
  void walk(int u, int d, int order, int pos, Key &k, std::vector<Key> &out) {
    const HGraph &g = *g_;
    bool leaf = true;
    if (d < order) {
      for (int e = g.start[u]; e < g.start[u + 1]; e++) {
        const int v = g.nbr[e];
        if (stamp_[v] != epoch_ || dist_[v] != d + 1) continue;   // layered: children only
        leaf = false;
        if (pos + 3 > KEY_BYTES) throw std::runtime_error("infoic: path longer than KEY_BYTES");
        const uint8_t save0 = k.b[pos], save1 = k.b[pos + 1], save2 = k.b[pos + 2];
        k.b[pos] = g.sym[e];
        k.b[pos + 1] = g.z[v];
        k.b[pos + 2] = g.deg[v];
        walk(v, d + 1, order, pos + 3, k, out);
        k.b[pos] = save0; k.b[pos + 1] = save1; k.b[pos + 2] = save2;
      }
    }
    if (leaf) {
      if (out.size() >= MAX_PATHS) throw std::runtime_error("infoic: path explosion");
      Key t = k;
      t.b[pos] = 0xFF;                                   // terminator; Z, degree and symbol
      for (int q = pos + 1; q < KEY_BYTES; q++) t.b[q] = 0;   // never take the value 0xFF
      out.push_back(t);
    }
  }

  const HGraph *g_ = nullptr;
  std::vector<int32_t> dist_;
  std::vector<int32_t> stamp_;
  std::vector<int32_t> bfs_;
  int32_t epoch_ = 0;
};

// --------------------------------------------------------------------------------------------
// Ipc: Le Verrier-Faddeev-Frame with power-of-two rescaling, on the HYDROGEN-SUPPRESSED graph.
// Returns log2|c_k| for k = 0..n; zero coefficients come back as -infinity and are skipped by
// the entropy, which is what rdInfoTheory's `if (tPtr[i])` does.
// --------------------------------------------------------------------------------------------
inline void charPolyLog2Abs(const Mol &m, std::vector<double> &log2abs) {
  const int n = m.n;
  log2abs.assign(n + 1, -std::numeric_limits<double>::infinity());
  log2abs[0] = 0.0;                                     // leading coefficient is exactly 1
  if (n == 0) return;
  std::vector<double> M((size_t)n * n, 0.0), T((size_t)n * n, 0.0);
  for (int b = 0; b < m.nb; b++) {                      // adjacency: dMat == 1, so unweighted
    M[(size_t)m.bu[b] * n + m.bv[b]] = 1.0;
    M[(size_t)m.bv[b] * n + m.bu[b]] = 1.0;
  }
  double cprev = 0.0;                                   // M_1 = A, c_1 = -tr(A) = 0
  for (int i = 0; i < n; i++) cprev -= M[(size_t)i * n + i];
  int E = 0;                                            // running binary scale of (M, cprev)
  log2abs[1] = cprev == 0.0 ? -std::numeric_limits<double>::infinity() : std::log2(std::fabs(cprev));
  for (int k = 2; k <= n; k++) {
    double mx = std::fabs(cprev);
    for (size_t q = 0; q < M.size(); q++) mx = std::max(mx, std::fabs(M[q]));
    if (mx > 1.0e150) {
      // Powers of two only, so the rescale is exact. The recurrence is homogeneous in the pair
      // (M, c_prev), so this scales every LATER coefficient by the same factor and nothing else.
      const int e = std::ilogb(mx);
      const double s = std::ldexp(1.0, -e);
      for (size_t q = 0; q < M.size(); q++) M[q] *= s;
      cprev *= s;
      E += e;
    }
    for (int i = 0; i < n; i++) M[(size_t)i * n + i] += cprev;     // M + c_{k-1} I
    std::fill(T.begin(), T.end(), 0.0);                            // T = A * M
    for (int b = 0; b < m.nb; b++) {                    // A is the adjacency: one row-add per bond
      const int u = m.bu[b], v = m.bv[b];
      double *tu = &T[(size_t)u * n], *tv = &T[(size_t)v * n];
      const double *mu = &M[(size_t)u * n], *mv = &M[(size_t)v * n];
      for (int j = 0; j < n; j++) { tu[j] += mv[j]; tv[j] += mu[j]; }
    }
    M.swap(T);
    double tr = 0.0;
    for (int i = 0; i < n; i++) tr += M[(size_t)i * n + i];
    const double ck = -tr / (double)k;
    log2abs[k] = ck == 0.0 ? -std::numeric_limits<double>::infinity()
                           : std::log2(std::fabs(ck)) + (double)E;
    cprev = ck;
  }
}

// InfoEntropy over values given by their log2, plus log2 of their sum. Max-subtracted, so it is
// finite for every molecule regardless of how large the coefficients are.
inline void entropyFromLog2(const std::vector<double> &l, double &entropy, double &log2sum) {
  double mx = -std::numeric_limits<double>::infinity();
  for (double x : l) if (std::isfinite(x)) mx = std::max(mx, x);
  if (!std::isfinite(mx)) { entropy = 0.0; log2sum = -std::numeric_limits<double>::infinity(); return; }
  double t = 0.0;
  for (double x : l) if (std::isfinite(x)) t += std::exp2(x - mx);
  const double log2t = std::log2(t);
  double acc = 0.0;
  for (double x : l) {
    if (!std::isfinite(x)) continue;                 // rdInfoTheory skips zero entries
    const double lp = (x - mx) - log2t;              // log2 p
    acc -= std::exp2(lp) * lp;
  }
  entropy = acc;
  log2sum = mx + log2t;
}

// --------------------------------------------------------------------------------------------
// The whole row.
// --------------------------------------------------------------------------------------------
inline void compute(const Mol &m, Row &row, CodeBuilder *cb = nullptr) {
  for (int c = 0; c < N_COLS; c++) row.v[c] = std::numeric_limits<double>::quiet_NaN();
  row.ipcOverflow = false;

  HGraph g;
  g.build(m);
  const int A = g.N;
  const double B = kekuleBondOrderSum(m);
  const double log2A = std::log2((double)A);
  const double log2B = std::log2(B);

  CodeBuilder local;
  CodeBuilder &bld = cb ? *cb : local;
  bld.reset(g);

  std::vector<int32_t> ordIdx(A);
  std::vector<Key> paths;
  std::vector<uint8_t> arena;
  std::vector<int32_t> off(A + 1, 0);
  std::vector<double> cnt, wmass, wz, scratch;

  for (int order = 0; order <= MAX_ORDER; order++) {
    arena.clear();
    off.assign(A + 1, 0);
    if (order == 0) {
      // mordred short-circuits: the code IS the atomic number. Nothing here can be order
      // dependent, which is exactly why order 0 is the control.
      for (int i = 0; i < A; i++) { arena.push_back(g.z[i]); off[i + 1] = (int32_t)arena.size(); }
    } else {
      for (int i = 0; i < A; i++) {
        bld.codeFor(i, order, paths);
        for (const Key &k : paths) arena.insert(arena.end(), k.b, k.b + KEY_BYTES);
        off[i + 1] = (int32_t)arena.size();
      }
    }
    for (int i = 0; i < A; i++) ordIdx[i] = i;
    const uint8_t *ar = arena.data();
    const int32_t *of = off.data();
    std::sort(ordIdx.begin(), ordIdx.end(), [ar, of](int32_t x, int32_t y) {
      const int32_t lx = of[x + 1] - of[x], ly = of[y + 1] - of[y];
      const int c = std::memcmp(ar + of[x], ar + of[y], (size_t)std::min(lx, ly));
      return c != 0 ? c < 0 : lx < ly;
    });

    cnt.clear(); wmass.clear(); wz.clear();
    for (int a = 0; a < A;) {
      int b = a + 1;
      const int32_t la = of[ordIdx[a] + 1] - of[ordIdx[a]];
      while (b < A) {
        const int32_t lb = of[ordIdx[b] + 1] - of[ordIdx[b]];
        if (lb != la || std::memcmp(ar + of[ordIdx[a]], ar + of[ordIdx[b]], (size_t)la) != 0) break;
        b++;
      }
      const double c = (double)(b - a);
      // Every atom in a class shares the code's first byte, which is the root's atomic number,
      // so the class HAS an element -- that is what makes repair R3 well defined.
      const int z = ar[of[ordIdx[a]]];
      cnt.push_back(c);
      wmass.push_back(ic_tbl::ATOMIC_WEIGHT[z]);
      wz.push_back(c * (double)z);
      a = b;
    }
    const size_t k = cnt.size();
    scratch.assign(k, 0.0);
    const double ic = shannonEntropy(cnt.data(), nullptr, k, scratch.data());
    row.v[F_IC * N_ORDERS + order] = ic;
    row.v[F_TIC * N_ORDERS + order] = (double)A * ic;
    row.v[F_SIC * N_ORDERS + order] = ic / log2A;      // IEEE, as mordred's numpy division is
    row.v[F_BIC * N_ORDERS + order] = ic / log2B;
    row.v[F_CIC * N_ORDERS + order] = log2A - ic;
    row.v[F_MIC * N_ORDERS + order] = shannonEntropy(cnt.data(), wmass.data(), k, scratch.data());
    row.v[F_ZMIC * N_ORDERS + order] = shannonEntropy(cnt.data(), wz.data(), k, scratch.data());
  }

  // ---- Ipc -----------------------------------------------------------------------------
  std::vector<double> l;
  charPolyLog2Abs(m, l);
  double h = 0.0, log2sum = 0.0;
  entropyFromLog2(l, h, log2sum);
  row.v[C_AVGIPC] = h;
  row.v[C_LOG2IPC] = (h > 0.0) ? log2sum + std::log2(h)
                               : -std::numeric_limits<double>::infinity();
  if (h == 0.0) {
    row.v[C_IPC] = 0.0;                                  // a one-atom graph: entropy is exactly 0
  } else {
    const double ipc = std::exp2(log2sum) * h;
    if (std::isfinite(ipc)) {
      row.v[C_IPC] = ipc;
    } else {
      row.v[C_IPC] = std::numeric_limits<double>::max();   // SATURATES, and says so:
      row.ipcOverflow = true;                              // never a silent inf
    }
  }
}

// --------------------------------------------------------------------------------------------
// selfCheck -- the guard against silent drift, run once at module load.
//
// It checks the two things that are cheap to check and expensive to discover on a corpus: the
// generated tables are the ones this file was written against, and the layered code builder
// gives the SAME code for a molecule under an atom permutation that mordred's tree does not.
// The permutation case is the worked example from the header comment, in graph form, so a
// regression that reintroduces order dependence fails at load.
// --------------------------------------------------------------------------------------------
// ON=Cc1ccccn1 -- pyridine-2-carbaldehyde oxime, 9 heavy atoms, as parsed by RDKit 2025.09.2.
// This is the header comment's worked example, in graph form.
inline void buildWorkedExample(Mol &m) {
  const uint8_t Z[9] = {8, 7, 6, 6, 6, 6, 6, 6, 7};
  const uint8_t NH[9] = {1, 0, 1, 0, 1, 1, 1, 1, 0};
  const uint8_t AR[9] = {0, 0, 0, 1, 1, 1, 1, 1, 1};
  const int E[9][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 3}};
  const int CODE[9] = {1, 2, 1, 8, 8, 8, 8, 8, 8};      // 1 SINGLE, 2 DOUBLE, 8 AROMATIC flag
  const double ORD[9] = {1.0, 2.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
  m.alloc(9, 9);
  for (int i = 0; i < 9; i++) { m.z[i] = Z[i]; m.nh[i] = NH[i]; m.arom[i] = AR[i]; }
  for (int e = 0; e < 9; e++) {
    m.bu[e] = E[e][0]; m.bv[e] = E[e][1]; m.bcode[e] = (uint8_t)CODE[e]; m.bord[e] = ORD[e];
  }
}

inline void selfCheck() {
  if (ic_tbl::ATOMIC_WEIGHT[6] <= 12.0 || ic_tbl::ATOMIC_WEIGHT[6] >= 12.02)
    throw std::runtime_error("infoic: ic_tables.h atomic weights look wrong");
  if (ic_tbl::DEFAULT_VALENCE[6] != 4 || ic_tbl::DEFAULT_VALENCE[7] != 3 ||
      ic_tbl::DEFAULT_VALENCE[8] != 2)
    throw std::runtime_error("infoic: ic_tables.h default valences look wrong");
  if (ic_tbl::N_OUTER_ELECS[7] != 5 || ic_tbl::N_OUTER_ELECS[6] != 4)
    throw std::runtime_error("infoic: ic_tables.h outer-electron counts look wrong");

  // numpy pairwise summation. Three shapes, one per branch of numpy's function, each written
  // out longhand here so that a "simplification" of pairwiseSum that changes the ORDER of the
  // additions fails at load rather than as a 1-ulp mystery in the order-0 control.
  {
    double a[300];
    for (int i = 0; i < 300; i++) a[i] = 1.0 / (double)(i + 3);
    double s7 = 0.0;
    for (int i = 0; i < 7; i++) s7 += a[i];          // n < 8: plain sequential
    if (pairwiseSum(a, 7) != s7) throw std::runtime_error("infoic: pairwiseSum n<8");
    {                                                // 8 <= n <= 128: 8 accumulators, tail last
      double r[8];
      for (int j = 0; j < 8; j++) r[j] = a[j];
      double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      res += a[8];
      if (pairwiseSum(a, 9) != res) throw std::runtime_error("infoic: pairwiseSum 8<=n<=128");
    }
    {                                                // n > 128: split at n/2 rounded down to 8
      const size_t n = 300, n2 = (n / 2) - ((n / 2) % 8);
      if (pairwiseSum(a, n) != pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2))
        throw std::runtime_error("infoic: pairwiseSum n>128");
    }
  }

  // The worked example, as a graph, so the check needs no RDKit and cannot go stale against a
  // comment. ON=Cc1ccccn1 under the identity and under every one of its 36 transpositions:
  // mordred moves on the single swap (0 5), this must not move on any of them.
  Mol base;
  buildWorkedExample(base);
  double ref[N_COLS];
  int perm[9];
  for (int a = -1; a < 9; a++)
    for (int b = a + 1; b < 9; b++) {
      if (a < 0 && b > 0) break;                    // a == -1 is the identity, done once
      for (int i = 0; i < 9; i++) perm[i] = i;
      if (a >= 0) { perm[a] = b; perm[b] = a; }
      Mol m;
      m.alloc(base.n, base.nb);
      for (int i = 0; i < base.n; i++) {
        m.z[perm[i]] = base.z[i]; m.nh[perm[i]] = base.nh[i]; m.arom[perm[i]] = base.arom[i];
        m.chg[perm[i]] = base.chg[i];
      }
      for (int e = 0; e < base.nb; e++) {
        m.bu[e] = perm[base.bu[e]]; m.bv[e] = perm[base.bv[e]];
        m.bcode[e] = base.bcode[e]; m.bord[e] = base.bord[e];
      }
      Row r;
      compute(m, r);
      if (a < 0) {
        std::memcpy(ref, r.v, sizeof ref);
        // The value itself, so a change of DEFINITION is caught here and not only a change of
        // determinism. mordred's two answers are 2.682588730501833 and 2.8159220638351665.
        if (std::fabs(r.v[F_IC * N_ORDERS + 1] - 2.8159220638351665) > 1e-12)
          throw std::runtime_error("infoic: selfCheck IC1 for ON=Cc1ccccn1 moved");
      } else {
        for (int c = 0; c < N_IC; c++)
          if (std::memcmp(&ref[c], &r.v[c], sizeof(double)) != 0)
            throw std::runtime_error(std::string("infoic: order dependence reintroduced in ") +
                                     columnNames()[c]);
      }
    }
}

}  // namespace infoic

#endif  // HUME_INFOCONTENT_H
