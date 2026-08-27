// mordred's TopologicalCharge family: GGI1..GGI10, JGI1..JGI10, JGT10. All 21 survive dedupe.
//
// SPECIFICATION IS mordred/TopologicalCharge.py plus mordred/_graph_matrix.py, at mordred 1.2.0.
// Two matrices and four lines:
//
//     A  = Chem.GetAdjacencyMatrix(mol, useBO=False)          int32, 0/1
//     D  = Chem.GetDistanceMatrix(mol, useBO=False)           float64, 1e8 between components
//     D2 = D with every NONZERO entry raised to -2, diagonal zeroed
//     CT = A.dot(D2) - (A.dot(D2)).T                          the Galvez charge-term matrix
//
// then, per column, over the STRICT LOWER TRIANGLE only:
//
//     D  = D * np.tri(D.shape)   -> keeps row >= col; D[D == 0] = inf then kills the diagonal too
//     raw   (GGI_k)  sum of |CT_ij| over  d(i,j) == k
//     mean  (JGI_k)  sum of |CT_ij| / n_k  over the same pairs, n_k = how many there are
//     global(JGT10)  sum of |CT_ij| / n_{d(i,j)} over d(i,j) <= 10
//
// THE 1e8 IS NOT A SENTINEL WE MAY SKIP. RDKit writes 1e8 into D for a disconnected pair, so
// mordred's D2 carries 1e-16 there rather than 0, and those 1e-16 terms DO enter A.dot(D2) and
// therefore CT. They are excluded from the SELECTION (1e8 > 10) but not from the VALUE. On a
// salt this is the difference between agreeing with mordred and not. `BIG` below is that 1e8.
//
// HOUSE RULE 1 -- THIS FAMILY IS WELL-POSED BUT NOT BIT-REPRODUCIBLE, INCLUDING AGAINST ITSELF.
// Re-running MORDRED on a randomly renumbered copy of the same molecule moves 20 of these 21
// columns. On all 100,000 molecules of cpp/hard.smi (3 random renumberings plus a
// canonical-SMILES round trip), GGI2..GGI10 / JGI1..JGI10 / JGT10 each changed on 21% to 70% of
// molecules -- JGI4 on 70,043 of them -- at a relative size never exceeding 2.9e-15. GGI1 never
// moved, on any molecule.
//
// That is NOT ill-posedness in the PORT_STATUS.md sense -- the quantity is a permutation
// invariant, a sum of absolute values over an unordered set of atom pairs, and the mathematics
// does not depend on the numbering. What moves is the FLOATING-POINT SUMMATION ORDER: `A.dot(D2)`
// is a BLAS dgemm (Accelerate, on this machine), whose accumulation order over an atom's
// neighbours is a property of the kernel's blocking, not of the source order. CT = M - M^T then
// subtracts two nearly equal sums, so a 1-ulp difference in M lands as a relative difference in
// CT. There is no definition to repair and nothing to diverge from; there is a last-digit
// tolerance to state out loud, which is what cpp/verify_topo3.py reports.
//
// WHAT THIS FILE DOES ABOUT IT. Every summation that CAN be pinned down is pinned down:
//
//   * M[i][j] accumulates over i's neighbours in ASCENDING NEIGHBOUR INDEX. Measured against
//     Accelerate's dgemm on 219,458 matrix entries, that agrees on 99.99%; a 4-way-unrolled
//     order agrees on 99.61% and an exactly-rounded sum on 97.69%, so plain ascending order is
//     the closest reachable choice, not merely the simplest.
//   * every outer reduction uses pairwise_sum(), which is numpy's own pairwise algorithm
//     (block 128, 8 accumulators) transcribed. Verified to reproduce ndarray.sum() BIT-EXACTLY
//     on this platform for lengths 0..300, 1000, 5000, 20000 and 100003.
//   * the values are laid out in ROW-MAJOR (i ascending, then j ascending) order, which is the
//     order numpy's boolean-mask selection CT[f] produces, so the pairwise tree is the same tree.
//
// NO UPSTREAM TABLE OR CONSTANT IS INVOLVED, so there is no drift guard here: the family is a
// BFS, a reciprocal square and three reductions. The one magic number, 1e8, is RDKit's
// disconnected-distance sentinel and is asserted against RDKit by the harness rather than
// trusted (cpp/verify_topo3.py checks GetDistanceMatrix on a two-fragment molecule).
//
// WHAT THE CALLER MUST SUPPLY: the CSR adjacency of `Chem.RemoveHs(mol)`. UNLIKE PathCount,
// hydrogen ATOMS are NOT excluded -- GetAdjacencyMatrix and GetDistanceMatrix see every atom in
// the H-suppressed graph, isotopic [2H] included. The two families genuinely read different
// graphs; see the note in pathcount.h.
#ifndef HUME_TOPOCHARGE_H
#define HUME_TOPOCHARGE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace topocharge {

static constexpr int N_COLS = 21;   // GGI1..10, JGI1..10, JGT10
static constexpr int ORDER = 10;

// RDKit's GetDistanceMatrix entry for a pair in different fragments.
static constexpr double BIG = 1e8;

static inline const char *col_name(int c) {
  static const char *N[N_COLS] = {"GGI1", "GGI2", "GGI3", "GGI4", "GGI5", "GGI6", "GGI7",
                                  "GGI8", "GGI9", "GGI10", "JGI1", "JGI2", "JGI3", "JGI4",
                                  "JGI5", "JGI6", "JGI7", "JGI8", "JGI9", "JGI10", "JGT10"};
  return N[c];
}

struct Mol {
  int n = 0;
  std::vector<int32_t> start;  // n + 1
  std::vector<int32_t> nbr;    // 2 * n_bonds, ASCENDING within each atom (see the header note)
};

// CSR straight off the boundary's `bond_i` rows, with the neighbour lists ASCENDING.
//
// The sort is not cosmetic and must not be dropped for a counting sort by bond index: ascending
// neighbour order is the accumulation order for M = A.D2, and it is the one measured closest to
// Accelerate's dgemm (99.99% of matrix entries, against 99.61% for a 4-way-unrolled order and
// 97.69% for an exactly rounded sum). bindings.cpp's crippen_fill() builds a DIFFERENT CSR, in
// bond order, which is correct there because every Crippen predicate is existential over
// neighbours and this one is not. Do not share the two.
//
// `rows` is bond_i, `stride` its column count (5 at the (n_bonds, 5) layout); cu/cv are the
// column indices of the two endpoints (0 and 1).
inline void build(Mol &m, int n, int nb, const int32_t *rows, int stride, int cu, int cv) {
  m.n = n;
  m.start.assign(n + 1, 0);
  for (int b = 0; b < nb; ++b) {
    m.start[rows[(ptrdiff_t)b * stride + cu] + 1]++;
    m.start[rows[(ptrdiff_t)b * stride + cv] + 1]++;
  }
  for (int i = 0; i < n; ++i) m.start[i + 1] += m.start[i];
  m.nbr.assign(m.start[n], 0);
  std::vector<int32_t> cur(m.start.begin(), m.start.end() - 1);
  for (int b = 0; b < nb; ++b) {
    const int u = rows[(ptrdiff_t)b * stride + cu], v = rows[(ptrdiff_t)b * stride + cv];
    m.nbr[cur[u]++] = v;
    m.nbr[cur[v]++] = u;
  }
  for (int i = 0; i < n; ++i) std::sort(m.nbr.begin() + m.start[i], m.nbr.begin() + m.start[i + 1]);
}

struct Scratch {
  std::vector<double> D2;    // n x n
  std::vector<double> M;     // n x n
  std::vector<int32_t> dist; // n x n, -1 for unreachable
  std::vector<int32_t> q;
  std::vector<double> bucket[ORDER + 1];  // |CT| per distance, row-major
  std::vector<double> flat;               // |CT| / n_d over d <= 10, row-major
  std::vector<double> tmp;
};

// -------------------------------------------------------------------------------------------
// numpy's pairwise summation, transcribed from numpy/_core/src/umath/loops_utils.h
// (`@TYPE@_pairwise_sum`): naive below 8, an 8-accumulator unrolled block up to 128, and a
// recursive split above that with the halves kept on a multiple of the unroll factor. This is
// not "a" pairwise sum -- the block size, the accumulator count and the (((r0+r1)+(r2+r3)) +
// ((r4+r5)+(r6+r7))) reduction tree all have to match or the last bit does not.
// -------------------------------------------------------------------------------------------
inline double pairwise_sum(const double *a, size_t n) {
  if (n < 8) {
    double r = 0.0;
    for (size_t i = 0; i < n; ++i) r += a[i];
    return r;
  }
  if (n <= 128) {
    double r[8] = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
      r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// out must have room for N_COLS doubles.
inline void compute(const Mol &m, double *out, Scratch &S) {
  for (int c = 0; c < N_COLS; ++c) out[c] = 0.0;
  const int n = m.n;
  if (n < 2) return;

  const size_t nn = (size_t)n * (size_t)n;
  if (S.D2.size() < nn) { S.D2.resize(nn); S.M.resize(nn); S.dist.resize(nn); }
  S.q.resize(n);

  // ---- BFS from every atom -> D, then D2 ----------------------------------------------------
  // D2[i][j] = d^-2 for d != 0, 0 on the diagonal. mordred writes `D2[D2 != 0] **= -2`; for
  // float64 and an integer exponent that is bit-identical to 1/(d*d) (checked for d = 1..39 and
  // for the 1e8 sentinel), so the reciprocal is used rather than a pow() call.
  for (int s = 0; s < n; ++s) {
    int32_t *dr = &S.dist[(size_t)s * n];
    double *d2r = &S.D2[(size_t)s * n];
    for (int i = 0; i < n; ++i) dr[i] = -1;
    int head = 0, tail = 0;
    dr[s] = 0;
    S.q[tail++] = s;
    while (head < tail) {
      const int u = S.q[head++];
      for (int e = m.start[u]; e < m.start[u + 1]; ++e) {
        const int v = m.nbr[e];
        if (dr[v] < 0) { dr[v] = dr[u] + 1; S.q[tail++] = v; }
      }
    }
    for (int i = 0; i < n; ++i) {
      if (i == s) { d2r[i] = 0.0; continue; }
      const double d = dr[i] < 0 ? BIG : (double)dr[i];
      d2r[i] = 1.0 / (d * d);
    }
  }

  // ---- M = A . D2, accumulated over neighbours in ascending index order ---------------------
  for (int i = 0; i < n; ++i) {
    double *mr = &S.M[(size_t)i * n];
    for (int j = 0; j < n; ++j) mr[j] = 0.0;
    for (int e = m.start[i]; e < m.start[i + 1]; ++e) {
      const double *d2r = &S.D2[(size_t)m.nbr[e] * n];
      for (int j = 0; j < n; ++j) mr[j] += d2r[j];
    }
  }

  // ---- select the strict lower triangle, bucketed by distance -------------------------------
  for (int k = 0; k <= ORDER; ++k) S.bucket[k].clear();
  S.flat.clear();
  size_t cnt[ORDER + 1] = {0};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      const int d = S.dist[(size_t)i * n + j];
      if (d >= 1 && d <= ORDER) ++cnt[d];
    }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      const int d = S.dist[(size_t)i * n + j];
      if (d < 1 || d > ORDER) continue;
      const double ct = S.M[(size_t)i * n + j] - S.M[(size_t)j * n + i];
      const double a = std::fabs(ct);
      S.bucket[d].push_back(a);
      S.flat.push_back(a / (double)cnt[d]);
    }

  for (int k = 1; k <= ORDER; ++k) {
    const std::vector<double> &b = S.bucket[k];
    out[k - 1] = pairwise_sum(b.data(), b.size());
    if (b.empty()) { out[ORDER + k - 1] = 0.0; continue; }
    const double c = (double)cnt[k];
    S.tmp.resize(b.size());
    for (size_t t = 0; t < b.size(); ++t) S.tmp[t] = b[t] / c;
    out[ORDER + k - 1] = pairwise_sum(S.tmp.data(), b.size());
  }
  out[2 * ORDER] = pairwise_sum(S.flat.data(), S.flat.size());
}

}  // namespace topocharge

#endif  // HUME_TOPOCHARGE_H
