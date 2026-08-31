// The 65 matrix-spectral columns: one eigensolver, eight matrices, eleven Burden diagonals.
//
//   SpAbs/SpMax/SpDiam/SpMAD/VE1-3/VR1-3  _A        adjacency matrix          10
//   BCUT{c,dv,d,s,Z,v,se,pe,are,p,i}-1h/l           Burden matrix             20
//   SpAbs/SpDiam/SpMAD/SM1/VE1-2/VR1-3    _Dz*      Barysz matrix x 6 props   30
//   SpAbs/SpDiam/SpMAD/VE1-2              _D        topological distance       5
//
// SPECIFICATION: mordred 1.2.0 `_matrix_attributes.py` (the ten aggregators), `BCUT.py`,
// `BaryszMatrix.py`, `AdjacencyMatrix.py`, `DistanceMatrix.py`, `_graph_matrix.py` and
// `_atomic_property.py`, plus networkx 2.8.8 `floyd_warshall_numpy` and numpy 1.26's pairwise
// summation. Every one of those was read; nothing here is from memory.
//
// ------------------------------------------------------------------------------------------
// WHAT IS REUSED RATHER THAN REBUILT
// ------------------------------------------------------------------------------------------
//
//   cpp/eigen_small.h   Householder tridiagonalisation (dsytd2, UPLO='U') + the Pal-Walker-Kahan
//                       QL/QR sweep (dsterf). BCUT2D already runs it; this header calls the SAME
//                       two stages, but keeps the WHOLE `d` array instead of only its two
//                       extremes, because SpAbs/SpMAD are functions of the entire spectrum.
//                       `hume_eig::extremal()` is used unchanged where only the extremes are
//                       wanted -- that is all 20 BCUT columns.
//   cpp/lu_small.h      reference dgetf2/dgetrs. The leading EIGENVECTOR (VE1/VR1 need it, and
//                       eigen_small returns eigenvalues only) comes from two steps of inverse
//                       iteration on the dense M - lambda*I. This costs n^3/3 against the
//                       reduction's 4n^3/3 and needs no second solver; writing a dstein-style
//                       tridiagonal inverse iteration plus a dormtr back-transform would have
//                       been a fourth linear-algebra kernel in the repository for no measured
//                       gain. NO NEW EIGENSOLVER IS ADDED BY THIS FILE.
//   cpp/ac_weights.h    all twelve mordred AtomicProperty getters, already verified bitwise by
//                       cpp/verify_ac.py. `explicit_hydrogens` is False for every descriptor
//                       here, so the SAME function is called on the heavy-atom graph: `d` is
//                       then the heavy degree and `dv`'s h is GetTotalNumHs() alone, which is
//                       exactly what mordred computes when its Context hands over RemoveHs(m).
//   topomisc.h          npPairwiseSum, the numpy 1.26 DOUBLE_pairwise_sum transliteration. Every
//                       SpAbs, SpMAD, SM1 and VE1 here is an `ndarray.sum()` or an `np.trace()`
//                       -- 25 of the 65 columns directly, plus the 6 derived from a VE1 -- and
//                       none of them is bit-exact unless that association is reproduced. The six
//                       SM1 columns ARE bit-exact, which is what shows the transliteration is
//                       right. There is one copy of it in the repository and this includes it
//                       rather than making a second.
//
// LANCZOS IS NOT USED, and the reason is measured rather than assumed: it was tried for BCUT2D
// and is 0.10x-0.63x the speed of the dense path at EVERY molecular size, with accuracy failing
// exactly where the speedup would have arrived. See the long note above bcut2d() in
// hume_blocks.h. Nothing about this family changes that arithmetic -- and seven of these
// matrices want the FULL spectrum, which Lanczos does not cheaply give.
//
// ------------------------------------------------------------------------------------------
// THE SEVEN UPSTREAM FACTS THAT ARE NOT GUESSABLE
// ------------------------------------------------------------------------------------------
//
// 1. EVERY COLUMN HERE IS NaN ON A DISCONNECTED MOLECULE. `MatrixAttributeBase` and `BCUTBase`
//    both set `require_connected = True`, and mordred's calculator short-circuits on
//    `n_frags != 1` BEFORE calculate() runs. 104 of the 20,000 corpus molecules are salts and
//    every one of these 65 cells is missing for them. The fragment count is derived here by one
//    BFS over the heavy-atom graph rather than carried across the boundary as a new field; see
//    `n_fragments` for why that cannot disagree with `Chem.GetMolFrags`, and for the check.
//
// 2. mordred's BURDEN MATRIX IS DENSE. `mat = 0.001 * np.ones((N, N))` -- every NON-bonded pair
//    is 0.001, not zero, and only then are the bonded pairs overwritten with
//    `GetBondTypeAsDouble()/10 (+0.01 if either end has degree 1)` and the diagonal with the
//    atomic property. This is NOT RDKit's BCUT2D Burden matrix (which is sparse, and whose
//    off-diagonal is a different function of the bond order); the two families share a name and
//    a paper and nothing else. hume_blocks.h's `bcut2d` is RDKit's and stays untouched.
//
// 3. mordred SOLVES THE SYMMETRIC BURDEN MATRIX WITH `np.linalg.eig`, THE UNSYMMETRIC SOLVER
//    (dgeev, Hessenberg + Francis QR), and then takes `.real`. That is a quirk -- deterministic,
//    same answer every time -- but it is NOT reproducible from outside: dgeev's rounding is not
//    a function anyone can transliterate. This header runs the symmetric path (dsytd2 + dsterf)
//    instead. Measured over all 20,000 molecules against a float64 mordred run: every one of the
//    20 BCUT columns agrees to better than 1e-9 relative on every cell mordred defines, worst
//    single cell 2.6e-12 (BCUTdv-1l), typical 1e-14.
//
// 4. THE BARYSZ OFF-DIAGONAL IS A FLOYD-WARSHALL SUM, AND THE ORDER OF THAT SUM IS PART OF THE
//    ANSWER. mordred hands the weighted graph to networkx's `floyd_warshall_numpy`, whose entire
//    body is
//        A = to_numpy_array(G, nonedge=inf); fill_diagonal(A, 0)
//        for i in range(n): A = np.minimum(A, A[i, :][None, :] + A[:, i][:, None])
//    Every entry is a sum of edge weights accumulated by that specific k-loop, so a Dijkstra
//    that finds the same path can still round differently. `floyd_warshall` below is that loop,
//    in that order. (Doing it in place is safe and is not an approximation: at step k, row k and
//    column k are provably unchanged by their own update because A[k][k] == 0.)
//
// 5. THE BARYSZ DIAGONAL IS WRITTEN AFTER THE SHORTEST PATHS, NOT BEFORE. `np.fill_diagonal(sp,
//    [1.0 - C/P[i]])` runs on the finished distance matrix, so the paths were computed with a
//    zero diagonal and the diagonal never participates in them. Getting this backwards changes
//    every off-diagonal entry of a molecule containing a heteroatom.
//
// 6. `rethrow_zerodiv` IS `np.errstate(divide="raise", invalid="raise")`, AND IT DECIDES THREE
//    COLUMNS' NaNs. Barysz's edge weight `C*C/(P_i*P_j*pi)` is inside it, so a property that is
//    zero on any bonded atom makes the WHOLE Barysz family missing for that molecule rather
//    than infinite. VE3 and VR3 are `np.log(0.1*A*VE1)` inside it, so VE1 == 0 is a MISSING
//    VALUE and not -inf -- which is exactly why VR3_A has one more NaN in the corpus than VR1_A
//    does (one single-atom molecule, no bonds, VR1 = 0.0). VR1 itself is NOT inside it, so a
//    negative eigenvector product there gives NaN and a zero gives +inf, both silently.
//
// 7. `np.linalg.eigh` RETURNS EIGENVALUES ASCENDING, and SpAbs / SpMAD are numpy `.sum()` calls
//    over arrays in that order. The spectrum is therefore SORTED here before it is summed, even
//    though SpAbs is mathematically order-free: pairwise summation is not.
//
// ------------------------------------------------------------------------------------------
// THE ONE INPUT THIS HEADER CANNOT GET FROM TODAY'S BOUNDARY
// ------------------------------------------------------------------------------------------
//
// mordred's `c` property on the HEAVY-atom graph is
//     GetDoubleProp("_GasteigerCharge") + GetDoubleProp("_GasteigerHCharge")
// -- the charge of the atom PLUS the charge RDKit assigned to its implicit hydrogens. `Mol.gast`
// in hume_blocks.h and `atom_d` column 1 in bindings.cpp carry `_GasteigerCharge` ALONE, which
// differs by up to 0.30 per atom (aspirin's carboxyl O: -0.4775 against -0.1809). The
// Autocorrelation `c` weight does not want the sum -- it runs on AddHs(m), where every hydrogen
// is its own atom and `_GasteigerHCharge` is cleared -- so nothing in HUME has needed it before.
//
// `Mol.at[i].c` here is the SUM, and the caller must supply it. `AC_C_MISSING` marks "RDKit
// could not charge this molecule", which is ac_weights.h's own existing convention and turns
// into the NaN that makes BCUTc-1h/-1l missing, exactly as mordred's AtomicProperty.calculate()
// does. The blob already carries `_GasteigerHCharge` (`_PICKLE_FLAGS` includes ComputedProps
// precisely because `_GasteigerCharge` is a computed property, and the H charge rides along);
// molpickle.h's reader skips it today. See NOTES_spectral.md for the exact change.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../../cpp/ac_weights.h"   // AtomRec, ac_weights(), the AC_* element tables, NW
#include "../../cpp/eigen_small.h"  // hume_eig::{Work, sytd2_upper, sterf_min_max, extremal}
#include "../../cpp/lu_small.h"     // hume_lin::{getf2, getrs_n}
#include "topomisc.h"               // topomisc::npPairwiseSum -- numpy's association

namespace spectral {

static constexpr int N_COLS = 65;

// The order is `results/dedupe2/agent_groups.json["B_spectral"]`, unchanged. bindings.cpp reads
// col_name(i) to place them; nothing here depends on the order except this table.
static const char *const COLS[N_COLS] = {
    "SpAbs_A", "SpMax_A", "SpDiam_A", "SpMAD_A", "VE1_A", "VE2_A", "VE3_A",
    "VR1_A", "VR2_A", "VR3_A",
    "BCUTc-1h", "BCUTc-1l", "BCUTdv-1h", "BCUTdv-1l", "BCUTd-1h", "BCUTd-1l",
    "BCUTs-1h", "BCUTs-1l", "BCUTZ-1h", "BCUTZ-1l", "BCUTv-1h", "BCUTv-1l",
    "BCUTse-1h", "BCUTse-1l", "BCUTpe-1l", "BCUTare-1l", "BCUTp-1h", "BCUTp-1l",
    "BCUTi-1h", "BCUTi-1l",
    "SpAbs_DzZ", "SpDiam_DzZ", "SpMAD_DzZ", "SM1_DzZ", "VE1_DzZ", "VE2_DzZ",
    "VR1_DzZ", "VR2_DzZ",
    "SpAbs_Dzv", "SpDiam_Dzv", "SpMAD_Dzv", "SM1_Dzv", "VE1_Dzv", "VE2_Dzv",
    "VR2_Dzv", "VR3_Dzv",
    "SpAbs_Dzse", "SpDiam_Dzse", "SpMAD_Dzse", "SM1_Dzse",
    "SM1_Dzare",
    "SpAbs_Dzp", "SpDiam_Dzp", "SpMAD_Dzp", "SM1_Dzp", "VE1_Dzp", "VE2_Dzp",
    "SpAbs_Dzi", "SpMAD_Dzi", "SM1_Dzi",
    "SpAbs_D", "SpDiam_D", "SpMAD_D", "VE1_D", "VE2_D",
};

inline const char *col_name(int i) { return COLS[i]; }

// Indices into ac_weights.h's twelve-wide per-atom property block. That order is fixed by
// verify_ac.py and autocorr.h's col_name(); it is NOT mordred's getter order, which is why
// these are named rather than written as literals at the call sites.
enum { P_c = 0, P_d = 1, P_dv = 2, P_i = 3, P_p = 4, P_v = 5,
       P_se = 6, P_pe = 7, P_are = 8, P_Z = 9, P_m = 10, P_s = 11 };

//! One molecule, heavy atoms only -- `Chem.RemoveHs(mol, updateExplicitCount=True)`, which is
//! what mordred's Context hands every descriptor in this file (all set explicit_hydrogens=False).
struct Mol {
  int n = 0, nb = 0;
  //! z / fc / nh / c, exactly ac_weights.h's record. `nh` is GetTotalNumHs(); `c` is
  //! `_GasteigerCharge + _GasteigerHCharge` (see the header note), or AC_C_MISSING.
  std::vector<AtomRec> at;
  std::vector<int> bu, bv;      //!< bond endpoints, in RDKit's bond index order
  std::vector<double> bord;     //!< GetBondTypeAsDouble(): 1, 1.5 (aromatic, NOT kekulised), 2, 3
  std::vector<std::vector<int>> adj;   //!< heavy-atom adjacency
};

//! Only ever grows; a resize() that shrinks and regrows value-initialises the new tail every
//! molecule, which is the trap BcutWork and hume_eig::Work both document.
struct Scratch {
  std::vector<double> M, B, prop, ev, tmp, vec, fw;
  std::vector<int> ipiv, q, comp;
  hume_eig::Work eig;
  void ensure(int n) {
    const std::size_t nn = (std::size_t)n * (std::size_t)n;
    if (M.size() < nn) M.resize(nn);
    if (B.size() < nn) B.resize(nn);
    if (fw.size() < nn) fw.resize(nn);
    if ((int)ev.size() < n) {
      ev.resize(n); tmp.resize(n); vec.resize(n);
      ipiv.resize(n); q.resize(n); comp.resize(n);
    }
  }
};

namespace detail {

inline double nan_() { return std::numeric_limits<double>::quiet_NaN(); }

//! Connected components of the HEAVY-atom graph.
//!
//! mordred asks `len(Chem.GetMolFrags(mol))` of the molecule as PARSED, before RemoveHs; this
//! counts components of the graph AFTER it. They cannot differ: a hydrogen that RemoveHs folds
//! away is by definition bonded to a heavy atom, so it was never its own fragment, and a
//! hydrogen-only fragment (`[H][H]`, a lone `[2H]`) has no heavy neighbour to fold into and
//! survives RemoveHs as its own component here too. Checked rather than argued -- the two counts
//! agree on 20,000 of 20,000 corpus molecules (verify_spectral.py's serialise path).
inline int n_fragments(const Mol &m, std::vector<int> &seen, std::vector<int> &q) {
  const int n = m.n;
  for (int i = 0; i < n; i++) seen[i] = 0;   // not assign(): Scratch only ever grows
  int nf = 0;
  for (int s = 0; s < n; s++) {
    if (seen[s]) continue;
    nf++;
    int head = 0, tail = 0;
    q[tail++] = s;
    seen[s] = 1;
    while (head < tail) {
      const int u = q[head++];
      for (int v : m.adj[u])
        if (!seen[v]) { seen[v] = 1; q[tail++] = v; }
    }
  }
  return nf;
}

//! `Chem.GetDistanceMatrix` on a connected graph: BFS from every atom, as doubles.
inline void topological_distances(const Mol &m, double *D, std::vector<int> &q) {
  const int n = m.n;
  const double UNSEEN = -1.0;
  for (std::size_t k = 0; k < (std::size_t)n * n; k++) D[k] = UNSEEN;
  for (int s = 0; s < n; s++) {
    double *d = D + (std::size_t)s * n;
    d[s] = 0.0;
    int head = 0, tail = 0;
    q[tail++] = s;
    while (head < tail) {
      const int u = q[head++];
      for (int v : m.adj[u])
        if (d[v] == UNSEEN) { d[v] = d[u] + 1.0; q[tail++] = v; }
    }
  }
}

//! networkx 2.8.8 floyd_warshall_numpy, k-loop and all. See fact 4 in the header.
inline void floyd_warshall(double *A, int n) {
  const double INF = std::numeric_limits<double>::infinity();
  for (int k = 0; k < n; k++) {
    const double *rk = A + (std::size_t)k * n;
    for (int p = 0; p < n; p++) {
      const double apk = A[(std::size_t)p * n + k];
      if (apk == INF) continue;          // min(A, inf + x) == A for every finite or infinite x
      double *rp = A + (std::size_t)p * n;
      for (int j = 0; j < n; j++) {
        const double cand = rk[j] + apk;  // numpy broadcasts A[k,:][None,:] + A[:,k][:,None]
        if (cand < rp[j]) rp[j] = cand;   // np.minimum
      }
    }
  }
}

//! The whole spectrum of a dense symmetric M, ASCENDING, into `ev`. eigen_small's own two
//! stages, with `d` kept instead of discarded -- see the header's reuse note.
//!
//! Returns false only if the QL/QR sweep fails to converge, which is eigen_small's own
//! `sterf_min_max` returning false; no Burden, Barysz, adjacency or distance matrix in the
//! 20,000-molecule corpus has done it.
inline bool spectrum(const double *M, int n, std::vector<double> &ev, hume_eig::Work &W) {
  if (n <= 0) return false;
  if ((int)ev.size() < n) ev.resize(n);
  if (n == 1) { ev[0] = M[0]; return true; }
  W.ensure(n);
  double *A = W.a.data();
  // For a symmetric matrix row-major and column-major are the same bytes; sytd2_upper reads the
  // upper triangle only, so this copies exactly that.
  for (int j = 0; j < n; j++)
    for (int k = 0; k <= j; k++) A[(std::size_t)j * n + k] = M[(std::size_t)j * n + k];
  hume_eig::sytd2_upper(A, n, n, W.d.data(), W.e.data(), W.tau.data(), W.wk.data());
  double lo, hi;
  if (!hume_eig::sterf_min_max(n, W.d.data(), W.e.data(), &lo, &hi)) return false;
  for (int i = 0; i < n; i++) ev[i] = W.d[i];
  std::sort(ev.begin(), ev.begin() + n);   // np.linalg.eigh returns ascending; fact 7
  return true;
}

//! Unit-2-norm eigenvector of the LARGEST eigenvalue, by inverse iteration on the dense matrix.
//!
//! WHY THIS IS WELL POSED, which is the question rule 4 of the contract asks. An eigenvector is
//! only defined up to the eigenspace, so a DEGENERATE leading eigenvalue would make VE1/VR1
//! depend on which basis LAPACK happened to return -- an ill-posed definition. It cannot happen
//! for any matrix in this file: all four (adjacency, topological distance, Barysz, Burden) are
//! irreducible on a connected molecule and become entrywise NON-NEGATIVE after a shift by a
//! multiple of the identity, which moves the whole spectrum and no eigenvector. Perron-Frobenius
//! then makes the spectral radius a SIMPLE eigenvalue with a strictly positive eigenvector, so
//! the eigenvector is unique up to sign -- and both VE1 (an absolute value) and VR1 (a product
//! over each bond's two entries) are invariant to that sign. Nothing here depends on atom
//! numbering, in exact arithmetic. HOW WIDELY SEPARATED that simple eigenvalue is, is a
//! different question, and the next paragraph is about the molecules whose answer is "by less
//! than one ulp".
//!
//! WHAT IS NOT WELL CONDITIONED, and it is a different statement: the SMALL entries of that
//! vector. A molecule built of two near-identical halves joined by a long flexible linker has a
//! Perron root whose separation from the next eigenvalue is EXPONENTIALLY small in the linker
//! length -- 8.9e-16 on corpus molecule 19279, i.e. below one ulp of lambda_max itself. The
//! eigenvalue is still simple, so nothing is ill-posed in exact arithmetic, but no double
//! precision eigenSOLVER can resolve which vector of that pair it is looking at. And VR1 raises
//! the PRODUCT of two entries to the -1/2, so a 1e-16 absolute error in a 1e-8 entry is a 1e-8
//! relative error in that bond's term; VR1_A reaches 1.1e16 on this corpus for that reason.
//!
//! THIS IS WHERE WE DIVERGE FROM MORDRED, AND THE DIVERGENCE IS AN IMPROVEMENT -- measured, not
//! claimed. mordred takes `eig.vec[:, argmax]` out of `numpy.linalg.eigh`, i.e. out of dsyevd's
//! divide-and-conquer, which on a numerically degenerate pair returns an ARBITRARY member of the
//! pair's span. Three separate demonstrations that its answer is not a function of the molecule,
//! all on corpus molecule 19279 and all running mordred's own code:
//!   * under nine random atom numberings, VE1_A lands anywhere in 4.244 .. 5.011 (18% spread)
//!     and VR1_A anywhere from 1.3e12 to NaN -- NaN because the mixed-sign vector makes some
//!     bond's v_i*v_j negative and `** -0.5` of a negative float is NaN;
//!   * VR1_A's last digits move with OMP_NUM_THREADS (1 vs 4 vs 8) on the neighbouring
//!     molecules 18115/18694/19817;
//!   * data/dedupe2/matrix.npz holds VR1_A = 1.302269e12 for this cell, and re-running mordred
//!     1.2.0 today on the same SMILES -- single process AND nproc=10, subset registry AND the
//!     full one -- returns NaN. Same code, same molecule, two different answers.
//!
//! Inverse iteration from the ALL-ONES start vector is not arbitrary in the same way: the true
//! Perron vector is strictly positive, so `ones` has an O(1) overlap with it and essentially
//! none with the antisymmetric partner that shares its eigenvalue, and the iteration therefore
//! selects the right member of the pair. Against a 60-digit mpmath power iteration (the true
//! Perron vector) on the five corpus molecules where the two implementations differ at all:
//!
//!     molecule   n   lambda1-lambda2   |ours - true|/true     |mordred - true|/true
//!                                       VE1_A     VR1_A         VE1_A     VR1_A
//!      19279     97      8.9e-16       3.8e-04   7.7e-04       2.9e-01   (NaN)
//!      18115     90      1.2e-13       3.8e-10   7.7e-10       2.7e-05   5.5e-05
//!      18694     94      1.9e-13       5.1e-09   1.0e-08       2.2e-05   4.6e-05
//!      19817     90      1.9e-13       1.2e-07   2.4e-07       3.5e-07   7.0e-07
//!      18726     57      2.5e-01       0.0e+00   3.0e-15       5.5e-16   6.9e-05
//!
//! Ours is better on nine of the ten cells and exact on the tenth; mordred is better on none.
//! The other 19,995 molecules agree with mordred inside float32, so this changes 5 molecules
//! x 6 columns out of 1,300,000.
//!
//! TWO SOLVES, AND THE COUNT IS MEASURED RATHER THAN ROUNDED UP -- iteration counts 1..4, run in
//! numpy against the same 60-digit truth. One is not enough: 18726 has a well-separated Perron
//! root but eigenvector entries down to 4.4e-13, and one solve leaves VR1 1.5e-3 off where two
//! leave it at 1.4e-15. THREE IS WORSE THAN TWO at the degenerate end: every extra solve injects
//! a little more of the antisymmetric partner, and 19279's VE1 error grows
//! 2.4e-4 -> 6.7e-4 -> 1.1e-3 -> 1.5e-3 over four iterations. Two is the minimum of that
//! trade-off, not a default.
//! Diagnostic counter: how many times the shift below has had to be nudged off the computed
//! eigenvalue because the factorisation came out exactly singular. A guard that never fires
//! proves nothing unless someone can read the count, so it is readable -- and it is a
//! function-local static inside an inline function, which is one object per program without
//! putting a global in a header. verify_spectral.py prints it.
inline long long &shift_retries() { static long long n = 0; return n; }

inline bool leading_vector(const double *M, int n, double lam, std::vector<double> &v,
                           std::vector<double> &Bs, std::vector<int> &ipiv) {
  if ((int)v.size() < n) v.resize(n);
  if (n == 1) { v[0] = 1.0; return true; }

  const double eps = std::numeric_limits<double>::epsilon();
  const double scale = std::max(std::fabs(lam), 1.0);

  for (int attempt = 0; attempt < 4; attempt++) {
    if (attempt > 0) shift_retries()++;
    // Nudging the shift only matters if M - lambda*I comes out EXACTLY singular in floating
    // point, which makes getf2 report a zero pivot and getrs_n divide by it. attempt 0 uses the
    // eigenvalue exactly as computed, because it is the near-singularity of M - sigma*I that
    // amplifies the wanted mode by 1/(lambda_max - sigma) ~ 1/(eps*||M||); moving sigma away
    // from it costs accuracy and is only done when the factorisation refuses.
    //
    // AND IT DOES REFUSE, which is why the counter exists rather than a comment saying it cannot
    // happen: `shift_retries()` is 62 over the 20,000-molecule corpus, on 59 molecules (three
    // need two nudges), from n = 2 to n = 64. The cause is exactly representable spectra on
    // small symmetric graphs -- for an isolated bond A = [[0,1],[1,0]], lambda_max is 1.0
    // exactly, A - I = [[-1,1],[1,-1]] and the second pivot is an exact zero. All 59 land inside
    // 1e-9 of mordred after the nudge; none of them is among the five molecules of the
    // divergence above.
    const double sigma = lam - (double)attempt * 8.0 * eps * scale;
    for (std::size_t k = 0; k < (std::size_t)n * n; k++) Bs[k] = M[k];
    for (int i = 0; i < n; i++) Bs[(std::size_t)i * n + i] -= sigma;
    if (hume_lin::getf2(n, Bs.data(), n, ipiv.data()) != 0) continue;   // exact zero pivot

    // A strictly positive start vector: the Perron vector is strictly positive, so this has an
    // O(1) overlap with it and no unlucky-start case exists.
    for (int i = 0; i < n; i++) v[i] = 1.0;
    bool bad = false;
    for (int it = 0; it < 2 && !bad; it++) {
      hume_lin::getrs_n(n, 1, Bs.data(), n, ipiv.data(), v.data(), n);
      double mx = 0.0;
      for (int i = 0; i < n; i++) {
        if (!std::isfinite(v[i])) { bad = true; break; }
        mx = std::max(mx, std::fabs(v[i]));
      }
      if (bad || mx == 0.0) { bad = true; break; }
      for (int i = 0; i < n; i++) v[i] /= mx;      // keep the norm sane between solves
    }
    if (bad) continue;

    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    const double nrm = std::sqrt(ss);
    if (!(nrm > 0.0) || !std::isfinite(nrm)) continue;
    // Sign is not observable: VE1 takes |v_i| and VR1 takes v_i*v_j. Fixed to positive anyway so
    // a debugging dump of this vector is comparable between runs.
    const double sgn = (v[0] < 0.0) ? -1.0 : 1.0;
    for (int i = 0; i < n; i++) v[i] = sgn * v[i] / nrm;
    return true;
  }
  return false;
}

//! The ten aggregators of `_matrix_attributes.py` for one matrix. Fields nobody asks for are
//! still computed -- they are O(n) on a spectrum that cost O(n^3), and branching on eleven
//! output flags would be more code than the arithmetic it saves.
struct Attrs {
  double spabs, spmax, spdiam, spmad, sm1, ve1, ve2, ve3, vr1, vr2, vr3;
};

inline void attrs_nan(Attrs &a) {
  const double q = nan_();
  a.spabs = a.spmax = a.spdiam = a.spmad = a.sm1 = q;
  a.ve1 = a.ve2 = a.ve3 = a.vr1 = a.vr2 = a.vr3 = q;
}

//! `np.log(0.1 * A * X)` under `np.errstate(divide="raise", invalid="raise")`: a non-positive
//! argument is a MISSING VALUE, not -inf and not NaN-by-arithmetic. Fact 6.
inline double log_index(int n, double x) {
  const double arg = 0.1 * (double)n * x;
  return (arg > 0.0) ? std::log(arg) : nan_();
}

inline void eval_matrix(const Mol &m, const double *M, bool want_vec, Attrs &a, Scratch &S) {
  const int n = m.n;
  attrs_nan(a);

  // SM1 = np.trace(matrix). numpy reduces the diagonal with the same pairwise sum it uses on a
  // contiguous array -- DOUBLE_pairwise_sum takes a stride -- so the diagonal is gathered and
  // handed to the same transliteration rather than added left to right.
  for (int i = 0; i < n; i++) S.tmp[i] = M[(std::size_t)i * n + i];
  a.sm1 = topomisc::npPairwiseSum(S.tmp.data(), n);

  if (!spectrum(M, n, S.ev, S.eig)) return;
  const double *w = S.ev.data();

  a.spmax = w[n - 1];                       // eig.val[argmax]; the array is ascending
  a.spdiam = a.spmax - w[0];                // SpMax - eig.val[argmin]
  for (int i = 0; i < n; i++) S.tmp[i] = std::fabs(w[i]);
  a.spabs = topomisc::npPairwiseSum(S.tmp.data(), n);
  const double mean = topomisc::npPairwiseSum(w, n) / (double)n;   // np.mean
  for (int i = 0; i < n; i++) S.tmp[i] = std::fabs(w[i] - mean);
  a.spmad = topomisc::npPairwiseSum(S.tmp.data(), n) / (double)n;  // SpAD / A

  if (!want_vec) return;
#ifndef SPECTRAL_WANT_EIGVEC
  // DEFAULT. Every VE*/VR* column was dropped in the cost triage (METHODS 5.2), so nothing
  // consumes the leading EIGENVECTOR any more and the inverse iteration -- with its lu_small.h
  // dgetf2/dgetrs factorisation -- leaves the kernel. Every retained spectral column is an
  // eigenVALUE aggregate, which sytd2/sterf above has already produced.
  // Measured: 264.2 -> 250.7 us/mol at the median molecule.
  return;
#else
  if (!leading_vector(M, n, a.spmax, S.vec, S.B, S.ipiv)) return;
#endif
  const double *v = S.vec.data();

  for (int i = 0; i < n; i++) S.tmp[i] = std::fabs(v[i]);
  a.ve1 = topomisc::npPairwiseSum(S.tmp.data(), n);
  a.ve2 = a.ve1 / (double)n;
  a.ve3 = log_index(n, a.ve1);

  // mordred's VR1 is a PYTHON for-loop over GetBonds() starting from the float 0.0 -- no numpy
  // reduction, so no pairwise association: strictly bond index order, left to right. `** -0.5`
  // on an np.float64 is libm's pow(x, -0.5), which is not bit-identical to 1/sqrt(x); a negative
  // product gives NaN and a zero product gives +inf, and both are kept (fact 6).
  double s = 0.0;
  for (int b = 0; b < m.nb; b++) s += std::pow(v[m.bu[b]] * v[m.bv[b]], -0.5);
  a.vr1 = s;
  a.vr2 = a.vr1 / (double)n;
  a.vr3 = log_index(n, a.vr1);
}

//! True when mordred's `AtomicProperty.calculate()` would succeed: it tests `np.isnan` and
//! NOTHING ELSE, so an infinite property value passes the gate and poisons the matrix instead.
inline bool prop_ok(const std::vector<double> &w, int n, int q) {
  for (int i = 0; i < n; i++)
    if (std::isnan(w[(std::size_t)i * NW + q])) return false;
  return true;
}

//! mordred's `AtomicProperty.carbon` -- `self.prop(Chem.Atom(6))`, i.e. the getter applied to a
//! bare carbon. Only the eight non-valence, non-charge properties can reach here (Barysz's
//! `get_properties()` excludes both classes), and for all of them it is one table row.
inline double carbon_value(int q) {
  switch (q) {
    case P_Z: return 6.0;
    case P_m: return AC_MASS[6];
    case P_v: return AC_VDWVOL[6];
    case P_se: return AC_SE[6];
    case P_pe: return AC_PE[6];
    case P_are: return AC_ARE[6];
    case P_p: return AC_POL[6];
    case P_i: return AC_IP[6];
    default: return nan_();
  }
}

//! The Barysz matrix of `BaryszMatrix.py`. Returns false when mordred's rethrow_zerodiv fires,
//! which makes the whole family missing for this property (fact 6).
//!
//! `need_paths == false` skips the O(n^3) Floyd-Warshall entirely: SM1 is `np.trace`, and the
//! trace is the diagonal, and the diagonal does not depend on the shortest paths (fact 5). The
//! zero-division scan still runs, because it is what decides whether SM1 exists at all.
inline bool barysz(const Mol &m, const std::vector<double> &w, int q, bool need_paths,
                   double *M) {
  const int n = m.n;
  const double C = carbon_value(q);
  const double CC = C * C;

  for (int b = 0; b < m.nb; b++) {
    const double den = w[(std::size_t)m.bu[b] * NW + q] * w[(std::size_t)m.bv[b] * NW + q] *
                       m.bord[b];
    if (den == 0.0) return false;            // np.errstate(divide="raise") -> MissingValue
  }
  if (need_paths) {
    const double INF = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < (std::size_t)n * n; k++) M[k] = INF;
    for (int i = 0; i < n; i++) M[(std::size_t)i * n + i] = 0.0;   // fill_diagonal(A, 0)
    for (int b = 0; b < m.nb; b++) {
      const double den = w[(std::size_t)m.bu[b] * NW + q] * w[(std::size_t)m.bv[b] * NW + q] *
                         m.bord[b];
      const double e = CC / den;
      // G.add_edge overwrites: a repeated pair would keep the LAST bond's weight. RDKit cannot
      // produce one, but the assignment is written the way networkx would resolve it.
      M[(std::size_t)m.bu[b] * n + m.bv[b]] = e;
      M[(std::size_t)m.bv[b] * n + m.bu[b]] = e;
    }
    floyd_warshall(M, n);
  } else {
    for (std::size_t k = 0; k < (std::size_t)n * n; k++) M[k] = 0.0;
  }
  // AFTER the paths, never before (fact 5).
  for (int i = 0; i < n; i++)
    M[(std::size_t)i * n + i] = 1.0 - C / w[(std::size_t)i * NW + q];
  return true;
}

}  // namespace detail

//! All N_COLS values for one molecule. `out` must have room for 65 doubles.
//
// TWENTY-NINE OF THE 65 SLOTS ARE NaN BY CONSTRUCTION AND MUST NOT BE WIRED. The cost triage
// (METHODS 5.2) dropped them; the code that computed them is still here behind
// -DSPECTRAL_WANT_EIGVEC and -DSPECTRAL_FIVE_BARYSZ_SPECTRA, so a reversal is a rebuild rather
// than a reimplementation. The `out[i] = NaN` fill below means a slot nothing writes reads back
// as NaN, not as stale buffer -- but the wiring must skip these names, not emit NaN columns:
//
//   eigenvector-derived (18): VE1/2/3_A  VR1/2/3_A  VE1/2_D
//                             VE1_DzZ VE2_DzZ VR1_DzZ VR2_DzZ
//                             VE1_Dzv VE2_Dzv VR2_Dzv VR3_Dzv  VE1_Dzp VE2_Dzp
//   redundant Barysz    (11): SpAbs/SpDiam/SpMAD_Dzv, _Dzse, _Dzp   SpAbs_Dzi  SpMAD_Dzi
//
// 36 columns are emitted.
inline void compute(const Mol &m, double *out, Scratch &S) {
  using namespace detail;
  const int n = m.n;
  const double NaN = nan_();
  for (int i = 0; i < N_COLS; i++) out[i] = NaN;
  if (n <= 0) return;
  S.ensure(n);
  if (n_fragments(m, S.comp, S.q) != 1) return;   // require_connected -- fact 1

  // All twelve mordred atomic properties for the heavy-atom graph, from the verified
  // transliteration Autocorrelation already uses.
  ac_weights(m.at, m.adj, S.prop);
  const std::vector<double> &w = S.prop;

  Attrs a;
  double *M = S.M.data();

  // ---------------------------------------------------------------- adjacency, columns 0..9
  // Chem.GetAdjacencyMatrix(useBO=False): 1 for a bond of any order, 0 otherwise.
  for (std::size_t k = 0; k < (std::size_t)n * n; k++) M[k] = 0.0;
  for (int b = 0; b < m.nb; b++) {
    M[(std::size_t)m.bu[b] * n + m.bv[b]] = 1.0;
    M[(std::size_t)m.bv[b] * n + m.bu[b]] = 1.0;
  }
  eval_matrix(m, M, /*want_vec=*/true, a, S);
  out[0] = a.spabs; out[1] = a.spmax; out[2] = a.spdiam; out[3] = a.spmad;
  out[4] = a.ve1;   out[5] = a.ve2;   out[6] = a.ve3;
  out[7] = a.vr1;   out[8] = a.vr2;   out[9] = a.vr3;

  // ---------------------------------------------------------------- Burden / BCUT, columns 10..29
  // The dense 0.001 background and the bond overwrites are the same for every property; only
  // the diagonal changes, so the off-diagonal half of the matrix is built ONCE and copied.
  // Fact 2: this is mordred's Burden matrix, not RDKit's.
  {
    double *base = S.fw.data();                    // reused as the Burden template
    for (std::size_t k = 0; k < (std::size_t)n * n; k++) base[k] = 0.001;
    for (int b = 0; b < m.nb; b++) {
      const int u = m.bu[b], v = m.bv[b];
      double bw = m.bord[b] / 10.0;
      if ((int)m.adj[u].size() == 1 || (int)m.adj[v].size() == 1) bw += 0.01;
      base[(std::size_t)u * n + v] = bw;
      base[(std::size_t)v * n + u] = bw;
    }
    // mordred's BCUT preset order: get_properties(valence=True, charge=True) yields the getters
    // in registration order c, dv, d, s, Z, m, v, se, pe, are, p, i. `m` is absent from this
    // group (deduplicated upstream) and pe/are keep only their `-1l` half.
    struct Slot { int q; int hi; int lo; };        // -1 == that half is not one of the 65
    static const Slot SL[11] = {
        {P_c, 10, 11}, {P_dv, 12, 13}, {P_d, 14, 15}, {P_s, 16, 17}, {P_Z, 18, 19},
        {P_v, 20, 21}, {P_se, 22, 23}, {P_pe, -1, 24}, {P_are, -1, 25}, {P_p, 26, 27},
        {P_i, 28, 29},
    };
    for (const Slot &sl : SL) {
      if (!prop_ok(w, n, sl.q)) continue;          // AtomicProperty.calculate() fails -> NaN
      for (std::size_t k = 0; k < (std::size_t)n * n; k++) M[k] = base[k];
      for (int i = 0; i < n; i++)
        M[(std::size_t)i * n + i] = w[(std::size_t)i * NW + sl.q];   // np.fill_diagonal(bmat, ps)
      double lo, hi;
      // Only the two extremes are wanted, which is exactly what eigen_small's one entry point
      // returns -- the same call BCUT2D makes. Fact 3: mordred asks dgeev for the whole
      // unsymmetric spectrum instead; the difference is measured in NOTES_spectral.md.
      if (!hume_eig::extremal(M, n, &lo, &hi, S.eig)) continue;
      if (sl.hi >= 0) out[sl.hi] = hi;             // np.sort(ev)[::-1][0]
      if (sl.lo >= 0) out[sl.lo] = lo;             // np.sort(ev)[::-1][-1]
    }
  }

  // ---------------------------------------------------------------- Barysz, columns 30..59
  // {property, wants Floyd-Warshall, wants the leading eigenvector, {out slots}}. Dzare is the
  // one that needs neither: SM1_Dzare is the trace and nothing else survived deduplication.
  {
    struct Bar {
      int q; bool paths; bool vec;
      int spabs, spdiam, spmad, sm1, ve1, ve2, vr1, vr2, vr3;
    };
#ifndef SPECTRAL_FIVE_BARYSZ_SPECTRA
    // DEFAULT. SpAbs/SpDiam/SpMAD are 0.982-0.999 correlated ACROSS the atomic-property
    // weightings -- the Barysz spectrum is dominated by topology and barely notices which
    // property sits on the diagonal. Only P_Z keeps its eigensolve; the other five fall back to
    // the trace, which is all SM1 needs and is the one aggregate that genuinely varies with the
    // property (pairwise |r| down to 0.282 against 0.982 for the other three).
    // Measured: 250.7 -> 209.3 us/mol at the median molecule, for 11 columns every one of which
    // is >= 0.982 correlated with a column that is kept. -DSPECTRAL_FIVE_BARYSZ_SPECTRA restores
    // all five, and the 11 columns with them.
    static const Bar BR[6] = {
        {P_Z,   true,  false, 30, 31, 32, 33, -1, -1, -1, -1, -1},
        {P_v,   false, false, -1, -1, -1, 41, -1, -1, -1, -1, -1},
        {P_se,  false, false, -1, -1, -1, 49, -1, -1, -1, -1, -1},
        {P_are, false, false, -1, -1, -1, 50, -1, -1, -1, -1, -1},
        {P_p,   false, false, -1, -1, -1, 54, -1, -1, -1, -1, -1},
        {P_i,   false, false, -1, -1, -1, 59, -1, -1, -1, -1, -1},
    };
#else
    static const Bar BR[6] = {
        {P_Z,   true,  true,  30, 31, 32, 33, 34, 35, 36, 37, -1},
        {P_v,   true,  true,  38, 39, 40, 41, 42, 43, -1, 44, 45},
        {P_se,  true,  false, 46, 47, 48, 49, -1, -1, -1, -1, -1},
        {P_are, false, false, -1, -1, -1, 50, -1, -1, -1, -1, -1},
        {P_p,   true,  true,  51, 52, 53, 54, 55, 56, -1, -1, -1},
        {P_i,   true,  false, 57, -1, 58, 59, -1, -1, -1, -1, -1},
    };
#endif
    for (const Bar &b : BR) {
      if (!prop_ok(w, n, b.q)) continue;
      if (!barysz(m, w, b.q, b.paths, M)) continue;
      if (!b.paths) {
        // Trace only. eval_matrix would spend an O(n^3) eigensolve on a matrix whose
        // off-diagonal was never built.
        for (int i = 0; i < n; i++) S.tmp[i] = M[(std::size_t)i * n + i];
        if (b.sm1 >= 0) out[b.sm1] = topomisc::npPairwiseSum(S.tmp.data(), n);
        continue;
      }
      eval_matrix(m, M, b.vec, a, S);
      if (b.spabs >= 0) out[b.spabs] = a.spabs;
      if (b.spdiam >= 0) out[b.spdiam] = a.spdiam;
      if (b.spmad >= 0) out[b.spmad] = a.spmad;
      if (b.sm1 >= 0) out[b.sm1] = a.sm1;
      if (b.ve1 >= 0) out[b.ve1] = a.ve1;
      if (b.ve2 >= 0) out[b.ve2] = a.ve2;
      if (b.vr1 >= 0) out[b.vr1] = a.vr1;
      if (b.vr2 >= 0) out[b.vr2] = a.vr2;
      if (b.vr3 >= 0) out[b.vr3] = a.vr3;
    }
  }

  // ---------------------------------------------------------------- distance, columns 60..64
  topological_distances(m, M, S.q);
  eval_matrix(m, M, /*want_vec=*/true, a, S);
  out[60] = a.spabs; out[61] = a.spdiam; out[62] = a.spmad;
  out[63] = a.ve1;   out[64] = a.ve2;
}

}  // namespace spectral
