// The 15 small mordred columns that share one graph and no machinery with Chi:
//
//     WalkCount        6   MWC03 MWC05 MWC08 SRW05 SRW07 TSRW10
//     Constitutional   4   Sp MZ Mv Mp
//     TopologicalIndex 2   Diameter TopoShapeIndex
//     WienerIndex      2   WPath WPol
//     ABCIndex         1   ABCGG
//
// SPECIFICATIONS are mordred/{WalkCount,Constitutional,TopologicalIndex,WienerIndex,ABCIndex}.py
// at 1.2.0, plus mordred/_graph_matrix.py, which is where all five families get their matrices.
//
// THERE ARE THREE DIFFERENT GRAPHS IN THESE 55 COLUMNS AND NOTHING SAYS SO OUT LOUD.
// `Descriptor.explicit_hydrogens` DEFAULTS TO True, and mordred's Context hands a descriptor
// `Chem.AddHs(mol)` or `Chem.RemoveHs(mol, updateExplicitCount=True)` accordingly. Four of the
// families here set it False; **Constitutional does not set it at all**, so:
//
//     Sp, MZ, Mv, Mp                      -> Chem.AddHs(mol)      HYDROGENS ARE REAL ATOMS
//     WalkCount / TopologicalIndex /
//     WienerIndex / ABCIndex / Chi        -> Chem.RemoveHs(mol)
//     Chi's SUBGRAPH ENUMERATION          -> RemoveHs, and then useHs=False on top (chi.h note 1)
//
// This is not a footnote: `MZ` for aspirin is 0.7460, which is sum(Z)/Z_C over TWENTY-ONE atoms,
// not the 1.1026 you get over the thirteen heavy ones. Getting it wrong is a 48% error, not a
// last-bit one, and it is invisible in any test that only uses hydrocarbon-free molecules.
// Constitutional needs no bonds at all, so the hydrogen-added graph never has to be built: the
// property array is the heavy atoms in their original order followed by `sum(GetTotalNumHs())`
// copies of the hydrogen value, which is exactly the atom order `Chem.AddHs` produces (it
// appends) and exactly the array numpy's pairwise sum then walks.
//
// ------------------------------------------------------------------------------------------
// FIVE UPSTREAM FACTS, EACH ONE LOAD-BEARING
// ------------------------------------------------------------------------------------------
//
// 1. `ABCGG` IS BROKEN IN THE PINNED ORACLE AND RETURNS AN ERROR FOR EVERY MOLECULE.
//    `ABCIndex.py` ends with `return np.float(np.sum(...))`, and `np.float` was REMOVED in numpy
//    1.24. Under the pin (numpy 1.26.4, forced by mordred 1.2.0's `numpy==1.*`) the descriptor
//    raises `AttributeError` on every input, so there is no oracle to be exact against as
//    shipped. `np.float` was never anything but a deprecated alias for the builtin `float`, so
//    the FIX IS A NO-OP ON THE VALUE: cpp/verify_chiwalk.py restores the alias and compares
//    against the resulting numbers, exactly as verify_topo3.py restores `np.product` for
//    MolecularDistanceEdge. This is a dead alias, not an ill-posed definition -- there is one
//    intended answer and it does not depend on numbering.
//
// 2. `np.sum(<generator>)` IS NOT numpy's sum. numpy's `fromnumeric.sum` special-cases a
//    generator argument and delegates to the BUILTIN `sum`, which starts from the int 0 and adds
//    left to right. So ABCGG accumulates strictly in BOND INDEX ORDER with no pairwise
//    reassociation -- the opposite of what the surrounding numpy code would suggest.
//
// 3. ...WHEREAS `Constitutional` REALLY IS numpy's sum: `np.sum(P / carbon)` over a contiguous
//    float64 array, which is numpy's PAIRWISE summation (8 accumulators, 128-element blocks,
//    recursive halving on a multiple of 8). Its terms are not dyadic, so `Sp`/`Mp`/`Mv` are only
//    bit-exact if that association is reproduced. `npPairwiseSum` below is a transliteration of
//    numpy 1.26's `DOUBLE_pairwise_sum`; verify_chiwalk.py checks it against `np.sum` directly.
//    `MZ` would survive a naive sum (atomic numbers over 6 are dyadic-ish in practice) but is
//    computed the same way rather than special-cased.
//
// 4. UNREACHABLE PAIRS ARE 1e8, NOT INFINITY, AND 17,050 OF cpp/hard.smi's 100,000 MOLECULES ARE
//    DISCONNECTED. RDKit's `GetDistanceMatrix` fills cross-fragment entries with 1e8 and mordred
//    passes them straight through, so `Diameter` is literally 100000000 for a salt and `WPath` is
//    dominated by the fragment count. Both are reproduced -- they are quirks (same answer every
//    time, a function of the molecule), not ill-posedness. `int(0.5 * D.sum())` stays exact
//    because every entry is an integer and the total stays under 2^53 for any molecule here.
//
// 5. `WalkCount`'s MATRICES ARE INTEGER. `Chem.GetAdjacencyMatrix` returns int32 and
//    `An.dot(A1)` keeps int32, so the powers, their sums and their traces are exact integers,
//    and int64 here reproduces them with room to spare -- the largest A^10 entry over the 50
//    biggest molecules of the corpus is 7,701 against int32's 2.1e9, so mordred's own int32 does
//    not overflow either and there is no wraparound quirk to copy. Only the final `np.log` is
//    floating point, on a value both sides agree on bit-for-bit.
//
//    THE TRACES ARE NOT COMPUTED BY BUILDING A^10. `trace(A^2p) = ||A^p||_F^2` and
//    `trace(A^(2p+1)) = <A^p, A^(p+1)>_F` for symmetric A, so powers up to A^5 give every trace
//    TSRW10 needs; and the three `sum(A^k)` values come from a vector iteration `1 -> A1 -> ...`
//    that never forms a matrix at all. Identical integers, four sparse-times-dense products
//    instead of nine.
//
// HOUSE RULE 1: all 15 are functions of the molecule -- integer counts, an order-independent
// integer matrix arithmetic, and one summation order fixed by RDKit's bond indices rather than
// by a perception. `cpp/verify_chiwalk.py perm` re-runs MORDRED on atom+bond-shuffled and
// Kekule/aromatic-toggled copies and reports what moves.
//
// WHAT THE CALLER MUST SUPPLY: `atom_i` (Z, degree, nH, formal charge) and `bond_i` (u, v) of
// `Chem.RemoveHs(mol, updateExplicitCount=True)`. Use build_from_rows().
#ifndef HUME_TOPOMISC_H
#define HUME_TOPOMISC_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "../../cpp/chiwalk_tables.h"

namespace topomisc {

static constexpr int N_COLS = 15;
static const char *const COLS[N_COLS] = {
    "MWC03", "MWC05", "MWC08", "SRW05", "SRW07", "TSRW10",
    "Sp", "MZ", "Mv", "Mp",
    "Diameter", "TopoShapeIndex",
    "WPath", "WPol",
    "ABCGG",
};

// RDKit's Code/GraphMol/MolOps.cpp fills unreachable pairs of getDistanceMat with this.
static constexpr int32_t LOCAL_INF = 100000000;

struct Mol {
  int n = 0;
  int nh_total = 0;              // sum of GetTotalNumHs() -- the atoms Chem.AddHs would append
  std::vector<int32_t> z;
  std::vector<int32_t> bu, bv;   // bond index order; orientation is irrelevant in this header
  std::vector<int32_t> start, nbr;  // CSR adjacency over ALL atoms
};

struct Scratch {
  std::vector<int32_t> dist;        // n x n
  std::vector<int32_t> bfs;
  std::vector<int64_t> mp, mq;      // n x n integer matrix powers
  std::vector<int64_t> vec, vec2;
  std::vector<double> prop;
};

inline void build_from_rows(Mol &m, int n, int nb, const int32_t *arows, int astride,
                            const int32_t *brows, int bstride) {
  m.n = n;
  m.z.resize(n);
  m.nh_total = 0;
  for (int i = 0; i < n; ++i) {
    m.z[i] = arows[(size_t)i * astride + 0];
    m.nh_total += arows[(size_t)i * astride + 2];   // nH = GetTotalNumHs(), for Constitutional
  }
  m.bu.resize(nb);
  m.bv.resize(nb);
  m.start.assign(n + 1, 0);
  for (int b = 0; b < nb; ++b) {
    m.bu[b] = brows[(size_t)b * bstride + 0];
    m.bv[b] = brows[(size_t)b * bstride + 1];
    m.start[m.bu[b] + 1]++;
    m.start[m.bv[b] + 1]++;
  }
  for (int i = 0; i < n; ++i) m.start[i + 1] += m.start[i];
  m.nbr.resize(m.start[n]);
  std::vector<int32_t> fill(m.start.begin(), m.start.end() - 1);
  for (int b = 0; b < nb; ++b) {
    m.nbr[fill[m.bu[b]]++] = m.bv[b];
    m.nbr[fill[m.bv[b]]++] = m.bu[b];
  }
}

// numpy 1.26 DOUBLE_pairwise_sum (numpy/core/src/umath/loops_arithm_fp.dispatch.c.src), unit
// stride. Eight accumulators inside a 128-element block, recursive halving above it, with the
// split forced to a multiple of 8. The association is the whole point; do not "simplify" this.
inline double npPairwiseSum(const double *a, int64_t n) {
  static constexpr int64_t PW_BLOCKSIZE = 128;
  if (n < 8) {
    double res = 0.0;
    for (int64_t i = 0; i < n; ++i) res += a[i];
    return res;
  } else if (n <= PW_BLOCKSIZE) {
    double r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[k];
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
      r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return npPairwiseSum(a, n2) + npPairwiseSum(a + n2, n - n2);
}

namespace detail {

inline void distanceMatrix(const Mol &m, Scratch &S) {
  const int n = m.n;
  S.dist.assign((size_t)n * n, LOCAL_INF);
  S.bfs.resize(n);
  for (int s = 0; s < n; ++s) {
    int32_t *d = &S.dist[(size_t)s * n];
    d[s] = 0;
    int head = 0, tail = 0;
    S.bfs[tail++] = s;
    while (head < tail) {
      const int u = S.bfs[head++];
      const int du = d[u];
      for (int k = m.start[u]; k < m.start[u + 1]; ++k) {
        const int v = m.nbr[k];
        if (d[v] == LOCAL_INF) { d[v] = du + 1; S.bfs[tail++] = v; }
      }
    }
  }
}

// A^p for p = 1..5 as dense int64, yielding trace(A^k) for k = 1..10 through
// trace(A^2p) = ||A^p||^2 and trace(A^2p+1) = <A^p, A^(p+1)>.
inline void walkTraces(const Mol &m, Scratch &S, int64_t tr[11], int64_t sums[11]) {
  const int n = m.n;
  const size_t nn = (size_t)n * n;
  S.mp.assign(nn, 0);
  S.mq.assign(nn, 0);
  for (int b = 0; b < (int)m.bu.size(); ++b) {
    S.mp[(size_t)m.bu[b] * n + m.bv[b]] += 1;
    S.mp[(size_t)m.bv[b] * n + m.bu[b]] += 1;
  }
  // tr[1] = <A^0, A^1> = trace(A) = 0 for a graph without self loops, but read it off anyway.
  tr[1] = 0;
  for (int i = 0; i < n; ++i) tr[1] += S.mp[(size_t)i * n + i];
  {
    int64_t f = 0;
    for (size_t t = 0; t < nn; ++t) f += S.mp[t] * S.mp[t];
    tr[2] = f;
  }
  for (int p = 2; p <= 5; ++p) {
    // mq = A * mp
    std::fill(S.mq.begin(), S.mq.end(), (int64_t)0);
    for (int i = 0; i < n; ++i) {
      int64_t *dst = &S.mq[(size_t)i * n];
      for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
        const int64_t *src = &S.mp[(size_t)m.nbr[k] * n];
        for (int j = 0; j < n; ++j) dst[j] += src[j];
      }
    }
    int64_t cross = 0, frob = 0;
    for (size_t t = 0; t < nn; ++t) { cross += S.mp[t] * S.mq[t]; frob += S.mq[t] * S.mq[t]; }
    tr[2 * p - 1] = cross;      // <A^(p-1), A^p> = trace(A^(2p-1))
    tr[2 * p] = frob;           // ||A^p||^2     = trace(A^(2p))
    S.mp.swap(S.mq);
  }
  // sum(A^k) = 1^T A^k 1, by vector iteration -- no matrix needed.
  S.vec.assign(n, 1);
  S.vec2.resize(n);
  for (int k = 1; k <= 10; ++k) {
    for (int i = 0; i < n; ++i) {
      int64_t acc = 0;
      for (int t = m.start[i]; t < m.start[i + 1]; ++t) acc += S.vec[m.nbr[t]];
      S.vec2[i] = acc;
    }
    S.vec.swap(S.vec2);
    int64_t s = 0;
    for (int i = 0; i < n; ++i) s += S.vec[i];
    sums[k] = s;
  }
}

}  // namespace detail

inline void compute(const Mol &m, double *out, Scratch &S) {
  const double NANV = std::numeric_limits<double>::quiet_NaN();
  const int n = m.n;

  // ---- WalkCount ---------------------------------------------------------------------------
  int64_t tr[11] = {0}, sums[11] = {0};
  if (n > 0) detail::walkTraces(m, S, tr, sums);
  out[0] = std::log((double)(sums[3] + 1));    // MWC03
  out[1] = std::log((double)(sums[5] + 1));    // MWC05
  out[2] = std::log((double)(sums[8] + 1));    // MWC08
  out[3] = std::log((double)(tr[5] + 1));      // SRW05
  out[4] = std::log((double)(tr[7] + 1));      // SRW07
  // TSRW10: mordred recurses TSRW_k = TSRW_(k-1) + SRW_k down to TSRW_1 = nAtoms + SRW_1, so the
  // accumulation runs from order 1 UPWARDS and the order matters. SRW_1 is log(1) = 0.
  {
    double t = (double)n;
    for (int k = 1; k <= 10; ++k) t = t + std::log((double)(tr[k] + 1));
    out[5] = t;
  }

  // ---- Constitutional ----------------------------------------------------------------------
  // `AtomicProperty.calculate` FAILS THE WHOLE COLUMN if any atom's table entry is NaN, so a
  // single unparameterised element makes Sp/Mv/Mp NaN rather than skipping that atom. Letting
  // NaN flow through the sum reproduces that, for that reason.
  // Chem.AddHs appends, so the array is the given atoms in their own order followed by nh_total
  // hydrogens; `nh` is the atom count the ConstitutionalMean divisor uses.
  const int nh = n + m.nh_total;
  S.prop.resize(nh > 0 ? nh : 1);
  {
    for (int i = 0; i < n; ++i)
      S.prop[i] = chiwalk_tables::pol94(m.z[i]) / chiwalk_tables::CARBON_POL;
    for (int i = n; i < nh; ++i) S.prop[i] = chiwalk_tables::POL94[1] / chiwalk_tables::CARBON_POL;
    const double sp = npPairwiseSum(S.prop.data(), nh);
    out[6] = sp;                                        // Sp
    out[9] = nh > 0 ? sp / (double)nh : NANV;           // Mp
    for (int i = 0; i < n; ++i) S.prop[i] = (double)m.z[i] / chiwalk_tables::CARBON_Z;
    for (int i = n; i < nh; ++i) S.prop[i] = 1.0 / chiwalk_tables::CARBON_Z;
    out[7] = nh > 0 ? npPairwiseSum(S.prop.data(), nh) / (double)nh : NANV;   // MZ
    for (int i = 0; i < n; ++i)
      S.prop[i] = chiwalk_tables::vdwVol(m.z[i]) / chiwalk_tables::CARBON_VDW_VOL;
    for (int i = n; i < nh; ++i)
      S.prop[i] = chiwalk_tables::VDW_VOL[1] / chiwalk_tables::CARBON_VDW_VOL;
    out[8] = nh > 0 ? npPairwiseSum(S.prop.data(), nh) / (double)nh : NANV;   // Mv
  }

  // ---- distance-matrix families ------------------------------------------------------------
  detail::distanceMatrix(m, S);
  int64_t wsum = 0, wpol = 0;
  int32_t diam = 0, radius = LOCAL_INF;
  for (int i = 0; i < n; ++i) {
    const int32_t *d = &S.dist[(size_t)i * n];
    int32_t ecc = 0;
    for (int j = 0; j < n; ++j) {
      wsum += d[j];
      if (d[j] == 3) ++wpol;
      if (d[j] > ecc) ecc = d[j];
    }
    if (ecc > diam) diam = ecc;
    if (ecc < radius) radius = ecc;
  }
  if (n == 0) { diam = 0; radius = 0; }
  out[10] = (double)diam;                              // Diameter
  // TopoShapeIndex = (D - R) / R under np.errstate(divide/invalid = raise), i.e. NaN when R == 0.
  out[11] = radius == 0 ? NANV : ((double)diam - (double)radius) / (double)radius;
  out[12] = (double)(wsum / 2);                        // WPath  = int(0.5 * D.sum())
  out[13] = (double)(wpol / 2);                        // WPol   = int(0.5 * (D == 3).sum())

  // ---- ABCGG -------------------------------------------------------------------------------
  // Accumulated with the BUILTIN sum in bond index order (note 2), starting from an integer zero.
  {
    double acc = 0.0;
    for (size_t b = 0; b < m.bu.size(); ++b) {
      const int32_t *du = &S.dist[(size_t)m.bu[b] * n];
      const int32_t *dv = &S.dist[(size_t)m.bv[b] * n];
      int64_t nu = 0, nv = 0;
      for (int k = 0; k < n; ++k) { nu += du[k] < dv[k]; nv += dv[k] < du[k]; }
      acc += std::sqrt(((double)nu + (double)nv - 2.0) / ((double)nu * (double)nv));
    }
    out[14] = acc;
  }
}

}  // namespace topomisc

#endif  // HUME_TOPOMISC_H
