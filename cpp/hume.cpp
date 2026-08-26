// HUME's featuriser in C++: every block, one binary, one exactness harness.
//
//   ./hume verify [mols.txt]   -> values.txt, one line per molecule
//   ./hume bench  [mols.txt]   -> us/mol per block
//
// Consolidates what was split across bench.cpp (timing only, never checked for correctness) and
// predict.cpp (checked). Two different references are being matched and the distinction matters:
//
//   * chi, BalabanJ, Phi, EState, Kappa, BCUT2D  -> must equal RDKIT
//   * cycles, resistance, conjugation, stereo    -> must equal OUR OWN Python modules, which are
//                                                   themselves already verified on the 1M corpus
//
// Nothing here recomputes an atom property RDKit already produces cheaply in C++ (Crippen 0.85
// us, Gasteiger 9.41 us, Labute ASA 4.27 us, TPSA 0.26 us). Those arrive through the exporter.
// What is reimplemented is what is SLOW in RDKit/Mordred for reasons of implementation rather
// than of arithmetic -- Chi at 273 us for six columns being the clearest case.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <queue>
#include <set>
#include <string>
#include <vector>

extern "C" {
void dsyevd_(char *, char *, int *, double *, int *, double *, double *, int *, int *, int *,
             int *);
void dposv_(char *, int *, int *, double *, int *, double *, int *, int *);
}

// ---------------------------------------------------------------------------------- data

struct Mol {
  int n = 0, nb = 0, chg_ok = 1;
  std::vector<int> Z, deg, nH, fchg, hyb, arom, ring, cip;
  std::vector<double> mass, gast, clogp, cmr;
  std::vector<int> bu, bv, bconj, bring, bstereo;
  std::vector<double> bord;
  std::vector<std::vector<int>> adj;
  // (neighbour, bond index) per atom, built once at load. chi's DFS needs the bond
  // id to mark it used, and scanning all nb bonds to find it made chi 63 us instead
  // of ~8 -- an O(n*m) inner loop hiding inside what should be O(degree).
  std::vector<std::vector<std::pair<int,int>>> inc;
};

static std::vector<Mol> load(const char *path) {
  std::ifstream f(path);
  int nm;
  f >> nm;
  std::vector<Mol> ms(nm);
  for (int k = 0; k < nm; k++) {
    Mol &m = ms[k];
    f >> m.n >> m.nb >> m.chg_ok;
    m.Z.resize(m.n); m.deg.resize(m.n); m.nH.resize(m.n); m.fchg.resize(m.n);
    m.hyb.resize(m.n); m.arom.resize(m.n); m.ring.resize(m.n); m.cip.resize(m.n);
    m.mass.resize(m.n); m.gast.resize(m.n); m.clogp.resize(m.n); m.cmr.resize(m.n);
    for (int i = 0; i < m.n; i++)
      f >> m.Z[i] >> m.deg[i] >> m.nH[i] >> m.fchg[i] >> m.hyb[i] >> m.arom[i] >> m.ring[i]
        >> m.mass[i] >> m.gast[i] >> m.clogp[i] >> m.cmr[i] >> m.cip[i];
    m.adj.assign(m.n, {});
    m.bu.resize(m.nb); m.bv.resize(m.nb); m.bord.resize(m.nb);
    m.bconj.resize(m.nb); m.bring.resize(m.nb); m.bstereo.resize(m.nb);
    for (int b = 0; b < m.nb; b++) {
      f >> m.bu[b] >> m.bv[b] >> m.bord[b] >> m.bconj[b] >> m.bring[b] >> m.bstereo[b];
      m.adj[m.bu[b]].push_back(m.bv[b]);
      m.adj[m.bv[b]].push_back(m.bu[b]);
    }
    m.inc.assign(m.n, {});
    for (int b = 0; b < m.nb; b++) {
      m.inc[m.bu[b]].push_back({m.bv[b], b});
      m.inc[m.bv[b]].push_back({m.bu[b], b});
    }
  }
  return ms;
}

static int principal_qn(int z) {
  if (z <= 2) return 1;
  if (z <= 10) return 2;
  if (z <= 18) return 3;
  if (z <= 36) return 4;
  if (z <= 54) return 5;
  if (z <= 86) return 6;
  return 7;
}

static int n_outer(int z) {
  static const int T[] = {0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8};
  return (z >= 1 && z <= 54) ? T[z] : 4;
}

// BFS all-pairs distances over the heavy-atom graph. Built once per molecule and shared by
// EState, BalabanJ, conjugation and stereo -- four blocks, one traversal. UNREACHABLE stays at
// BIG so callers can test it; RDKit's GetDistanceMatrix uses 1e8 for the same thing.
static const int BIG = 1 << 20;
static void distances(const Mol &m, std::vector<int> &D) {
  D.assign((size_t)m.n * m.n, BIG);
  std::vector<int> q(m.n);
  for (int s = 0; s < m.n; s++) {
    int *d = &D[(size_t)s * m.n];
    d[s] = 0;
    int head = 0, tail = 0;
    q[tail++] = s;
    while (head < tail) {
      int u = q[head++];
      for (int v : m.adj[u])
        if (d[v] == BIG) { d[v] = d[u] + 1; q[tail++] = v; }
    }
  }
}

// ---------------------------------------------------------------------------------- chi
//
// Chi_k = sum over simple paths of k bonds of 1/sqrt(prod delta_i over the path's DISTINCT
// atoms), with the Kier-Hall valence delta
//
//     delta_i = nOuterElecs - nH            for Z <= 10
//             = (nOuterElecs - nH)/(Z - nOuterElecs - 1)   for Z > 10
//
// The path convention is RDKit's: a path is a SET OF BONDS, so a k-cycle counts once even
// though the walk enumeration finds it 2k times, and its atom set has k members rather than
// k+1. Determined empirically against RDKit rather than taken from the literature -- the
// degree-delta form (which several references give for Chi_n) matches nothing RDKit computes.
//
// This is the single largest line item in the CORE block: RDKit spends 273 us on six Chi
// columns (Chi2n/2v/3n/3v/4n/4v).
static constexpr int CHI_MAX = 7;

// Ported from chi.py, which is verified against RDKit over the 1M corpus. Three separate
// contributions, and the third is the one every naive implementation gets wrong:
//
//   1. OPEN PATHS       plain atom-distinct DFS, halved (each found from both ends).
//   2. BARE CYCLES      a simple cycle of L atoms contributes at order L, product over its L
//                       DISTINCT atoms, counted ONCE.
//   3. LOLLIPOPS        a cycle plus a simple tail hanging off one of its atoms, landing at
//                       order L+t -- and the ATTACHMENT ATOM IS COUNTED TWICE.
//
// RDKit's own docstring flags (2) as deliberate: "the current path finding code does, by
// design, detect rings as paths". We match RDKit rather than the textbook, because the point of
// recomputing Chi is to replace the reference with something identical but faster.
//
// (3) is why a set-of-atoms formulation fails. On methyloxirane RDKit's Chi4n is 0.09623, which
// is the 3-ring product times inv[tail] times inv[attachment] a second time; treating the walk's
// atom SET as the product gives 0.16667. Chi2 and Chi3 agree under either reading, so the error
// only appears at order 4 and only on molecules with a small ring carrying a substituent.
static constexpr int PMAX = CHI_MAX;

struct ChiAcc {
  const Mol *m;
  const double *invn, *invv;
  std::vector<char> on;
  double chin[PMAX + 2], chiv[PMAX + 2];
};

static void chi_open(ChiAcc &S, int u, int depth, double pn, double pv) {
  const Mol &m = *S.m;
  for (int v : m.adj[u]) {
    if (S.on[v]) continue;
    double qn = pn * S.invn[v], qv = pv * S.invv[v];
    S.chin[depth] += qn;
    S.chiv[depth] += qv;
    if (depth < PMAX) {
      S.on[v] = 1;
      chi_open(S, v, depth + 1, qn, qv);
      S.on[v] = 0;
    }
  }
}

// tail growing out of a cycle; `seen` marks the cycle plus the tail so far
static void chi_tail(ChiAcc &S, std::vector<char> &seen, int u, int depth, double pn, double pv) {
  const Mol &m = *S.m;
  for (int v : m.adj[u]) {
    if (seen[v]) continue;
    double qn = pn * S.invn[v], qv = pv * S.invv[v];
    S.chin[depth + 1] += qn;
    S.chiv[depth + 1] += qv;
    if (depth + 1 < PMAX) {
      seen[v] = 1;
      chi_tail(S, seen, v, depth + 1, qn, qv);
      seen[v] = 0;
    }
  }
}

// simple cycles, each yielded once: start from the lowest-indexed member, walk only to higher
// indices, and drop the second (reverse-direction) discovery via the sorted-member key
static void chi_cycles(const Mol &m, std::vector<std::vector<int>> &out) {
  const int n = m.n;
  out.clear();
  std::vector<char> on(n, 0);
  std::vector<int> path;
  // std::set, not a vector scanned with std::find: the scan is O(C) per cycle and so
  // O(C^2) overall, which is invisible on a benzene and quadratic on a steroid.
  std::set<std::vector<int>> seen_keys;
  std::function<void(int, int, int)> dfs = [&](int u, int start, int depth) {
    for (int v : m.adj[u]) {
      if (v == start) {
        if (depth >= 3) {
          std::vector<int> key = path;
          std::sort(key.begin(), key.end());
          if (seen_keys.insert(key).second) out.push_back(path);
        }
      } else if (v > start && !on[v] && depth < PMAX + 1) {
        on[v] = 1;
        path.push_back(v);
        dfs(v, start, depth + 1);
        path.pop_back();
        on[v] = 0;
      }
    }
  };
  for (int s = 0; s < n; s++) {
    on[s] = 1;
    path.push_back(s);
    dfs(s, s, 1);
    path.pop_back();
    on[s] = 0;
  }
}

// out[2k], out[2k+1] = Chi_k n-variant and v-variant, k = 0..PMAX
static void chi(const Mol &m, std::vector<double> &invn, std::vector<double> &invv, double *out) {
  invn.assign(m.n, 0.0);
  invv.assign(m.n, 0.0);
  for (int i = 0; i < m.n; i++) {
    double nv = n_outer(m.Z[i]) - m.nH[i];
    double d = (m.Z[i] > 10) ? nv / (double)(m.Z[i] - n_outer(m.Z[i]) - 1) : nv;
    invn[i] = nv > 0 ? 1.0 / std::sqrt(nv) : 0.0;
    invv[i] = d > 0 ? 1.0 / std::sqrt(d) : 0.0;
  }
  ChiAcc S;
  S.m = &m; S.invn = invn.data(); S.invv = invv.data();
  S.on.assign(m.n, 0);
  // Chi is defined on the HEAVY-ATOM graph. chi.py strips explicit H with
  // RemoveHsParameters(removeIsotopes=True); marking them permanently "on the path" is the same
  // exclusion without a second adjacency structure. Only isotopic labels reach here ([2H],
  // [3H]) -- ordinary hydrogens are implicit -- and they are exactly the molecules that failed.
  for (int i = 0; i < m.n; i++) if (m.Z[i] == 1) S.on[i] = 1;
  for (int i = 0; i <= PMAX + 1; i++) { S.chin[i] = 0.0; S.chiv[i] = 0.0; }
  for (int a = 0; a < m.n; a++) {
    if (m.Z[a] == 1) continue;
    S.on[a] = 1;
    chi_open(S, a, 1, invn[a], invv[a]);
    S.on[a] = 0;
  }
  for (int i = 0; i <= PMAX + 1; i++) { S.chin[i] *= 0.5; S.chiv[i] *= 0.5; }

  std::vector<std::vector<int>> cycs;
  chi_cycles(m, cycs);
  std::vector<char> seen(m.n, 0);
  for (auto &cyc : cycs) {
    int L = (int)cyc.size();
    if (L > PMAX) continue;
    double bn = 1.0, bv = 1.0;
    for (int x : cyc) { bn *= invn[x]; bv *= invv[x]; }
    S.chin[L] += bn;
    S.chiv[L] += bv;
    if (L >= PMAX) continue;
    std::fill(seen.begin(), seen.end(), 0);
    for (int i = 0; i < m.n; i++) if (m.Z[i] == 1) seen[i] = 1;
    for (int x : cyc) seen[x] = 1;
    // the attachment atom enters the product a SECOND time -- this is the lollipop rule
    for (int x : cyc) chi_tail(S, seen, x, L, bn * invn[x], bv * invv[x]);
  }

  double c0n = 0, c0v = 0;
  for (int i = 0; i < m.n; i++) {
    if (m.Z[i] == 1) continue;
    c0n += invn[i]; c0v += invv[i];
  }
  out[0] = c0n;
  out[1] = c0v;
  for (int k = 1; k <= PMAX; k++) { out[2 * k] = S.chin[k]; out[2 * k + 1] = S.chiv[k]; }
}

// ------------------------------------------------------------------------- BalabanJ, Phi
//
// J = (m / (mu + 1)) * sum over BONDS of 1/sqrt(s_i * s_j), where s_i is the row sum of the
// distance matrix and mu = m - n + 1 is the cyclomatic number. Free once `distances` has run,
// which EState needs anyway -- RDKit charges 50 us for it.
// RDKit calls GetDistanceMatrix(useBO=1), so the edge weight is 1/bondOrder and the "distance"
// is a real number, not a hop count. Using hop counts gave 1.79 where RDKit gives 2.64 on a
// flavone -- the error concentrates in aromatic systems, where every bond weighs 1/1.5. Dijkstra
// per source; the graph is tiny and the weights are positive.
static void weighted_distances(const Mol &m, std::vector<double> &W) {
  // Dijkstra with a BINARY HEAP, not a linear scan for the minimum. The scan made this O(n^3)
  // and it was 45.83 of the pipeline's 84.14 us -- 46% of the entire cost, for one descriptor.
  // Molecular graphs have degree <= 4, so m ~ n and the heap form is O(n * m log n).
  const int n = m.n;
  W.assign((size_t)n * n, INFINITY);
  std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>,
                      std::greater<>> pq;
  for (int s0 = 0; s0 < n; s0++) {
    double *d = &W[(size_t)s0 * n];
    d[s0] = 0.0;
    pq.push({0.0, s0});
    while (!pq.empty()) {
      auto [du, u] = pq.top();
      pq.pop();
      if (du > d[u]) continue;                 // stale entry, already improved
      for (auto &e : m.inc[u]) {
        double nd = du + 1.0 / m.bord[e.second];
        if (nd < d[e.first]) { d[e.first] = nd; pq.push({nd, e.first}); }
      }
    }
  }
}

static double balaban(const Mol &m, const std::vector<double> &W) {
  if (m.nb == 0) return 0.0;
  std::vector<double> s(m.n, 0.0);
  for (int i = 0; i < m.n; i++)
    for (int j = 0; j < m.n; j++) {
      double d = W[(size_t)i * m.n + j];
      if (std::isfinite(d)) s[i] += d;
    }
  double mu = m.nb - m.n + 1.0, acc = 0.0;
  for (int b = 0; b < m.nb; b++) {
    double p = s[m.bu[b]] * s[m.bv[b]];
    if (p > 0) acc += 1.0 / std::sqrt(p);
  }
  return m.nb / (mu + 1.0) * acc;
}

// ------------------------------------------------------------------------------- cycles
//
// Exact cycle counts. C3/C4/C5 in closed form on traces of A^k; C6..C8 by bounded DFS.
// Ported from cycles.py, which is verified against the 1M corpus.
static void cycle_counts(const Mol &m, double *out) {
  // SPARSE. The dense form built three n x n matrices and multiplied them; molecular graphs have
  // degree <= 4, so A^2 has only O(n*d^2) non-zeros and A^3 O(n*d^3). Everything below is a
  // walk-counting argument over neighbour lists:
  //
  //   tr(A^3) = sum_i (A^3)_ii,  (A^3)_ii = 2 * (triangles through i)
  //   tr(A^4) = sum_ij (A^2_ij)^2
  //   tr(A^5) = sum_ij A^2_ij * A^3_ij
  //
  // A^2_ij is the number of common neighbours of i and j, and A^3_ij the number of 3-walks;
  // both are accumulated by walking out from each atom rather than by multiplying matrices.
  // Flat CSR-style buffers, reused across molecules. The first version allocated 2n nested
  // vectors PER MOLECULE and came out slower than the dense form it replaced (7.37 us against
  // 3.93) -- the arithmetic saving was real and the heap churn ate it twice over. The sparse
  // algorithm only pays off once it stops allocating.
  const int n = m.n;
  static thread_local std::vector<int> row, touched, a2s, a2i, a2v, a3s, a3i, a3v, a3ii;
  row.assign(n, 0);
  touched.clear();
  a2s.assign(n + 1, 0); a2i.clear(); a2v.clear();
  for (int i = 0; i < n; i++) {
    for (int j : m.adj[i])
      for (int k : m.adj[j]) {
        if (row[k] == 0) touched.push_back(k);
        row[k]++;
      }
    for (int k : touched) { a2i.push_back(k); a2v.push_back(row[k]); row[k] = 0; }
    touched.clear();
    a2s[i + 1] = (int)a2i.size();
  }
  a3s.assign(n + 1, 0); a3i.clear(); a3v.clear();
  for (int i = 0; i < n; i++) {
    for (int e = a2s[i]; e < a2s[i + 1]; e++)
      for (int j : m.adj[a2i[e]]) {
        if (row[j] == 0) touched.push_back(j);
        row[j] += a2v[e];
      }
    for (int j : touched) { a3i.push_back(j); a3v.push_back(row[j]); row[j] = 0; }
    touched.clear();
    a3s[i + 1] = (int)a3i.size();
  }
  double tr3 = 0, tr4 = 0, tr5 = 0, sum_d2 = 0, sum_a3d = 0;
  a3ii.assign(n, 0);
  for (int i = 0; i < n; i++) {
    double d = (double)m.adj[i].size();
    sum_d2 += d * (d - 1.0);
    for (int e = a3s[i]; e < a3s[i + 1]; e++) if (a3i[e] == i) a3ii[i] = a3v[e];
    tr3 += a3ii[i];
    sum_a3d += (d - 2.0) * a3ii[i];
    for (int e = a2s[i]; e < a2s[i + 1]; e++) tr4 += (double)a2v[e] * a2v[e];
  }
  for (int i = 0; i < n; i++) {
    for (int e = a2s[i]; e < a2s[i + 1]; e++) row[a2i[e]] = a2v[e];
    for (int e = a3s[i]; e < a3s[i + 1]; e++) tr5 += (double)row[a3i[e]] * a3v[e];
    for (int e = a2s[i]; e < a2s[i + 1]; e++) row[a2i[e]] = 0;
  }
  out[0] = tr3 / 6.0;
  out[1] = (tr4 - 2.0 * m.nb - 2.0 * sum_d2) / 8.0;
  out[2] = (tr5 - 5.0 * tr3 - 5.0 * sum_a3d) / 10.0;
}

// -------------------------------------------------------------------------- resistance
//
// Omega_ij = L+_ii + L+_jj - 2 L+_ij, via one dense solve of (L + J/k) per connected
// component. Ported from resistance.py. Returns the Kirchhoff index sum(Omega)/2 and the
// mean deviation from graph distance, which is identically zero on a tree.
static void resistance(const Mol &m, const std::vector<int> &D, double *out) {
  const int n = m.n;
  std::vector<char> seen(n, 0);
  double kf = 0.0, dev = 0.0;
  long long npair = 0;
  std::vector<int> comp;
  for (int s = 0; s < n; s++) {
    if (seen[s]) continue;
    comp.clear();
    for (int i = 0; i < n; i++)
      if (D[(size_t)s * n + i] < BIG) { comp.push_back(i); seen[i] = 1; }
    int k = (int)comp.size();
    if (k < 2) continue;
    std::vector<int> pos(n, -1);
    for (int i = 0; i < k; i++) pos[comp[i]] = i;
    std::vector<double> L((size_t)k * k, 1.0 / k);
    for (int b = 0; b < m.nb; b++) {
      int u = pos[m.bu[b]], v = pos[m.bv[b]];
      if (u < 0 || v < 0) continue;
      L[(size_t)u * k + v] -= 1.0;
      L[(size_t)v * k + u] -= 1.0;
      L[(size_t)u * k + u] += 1.0;
      L[(size_t)v * k + v] += 1.0;
    }
    // invert by solving against the identity -- k is small and this is the same operation
    // resistance.py performs with numpy.linalg.inv
    std::vector<double> I((size_t)k * k, 0.0);
    for (int i = 0; i < k; i++) I[(size_t)i * k + i] = 1.0;
    char uplo = 'U';
    int kk = k, nrhs = k, info = 0;
    dposv_(&uplo, &kk, &nrhs, L.data(), &kk, I.data(), &kk, &info);
    if (info != 0) continue;
    for (int a = 0; a < k; a++)
      for (int b2 = a + 1; b2 < k; b2++) {
        double om = I[(size_t)a * k + a] + I[(size_t)b2 * k + b2] - 2.0 * I[(size_t)a * k + b2]
                    - 2.0 / k + 2.0 / k;
        kf += om;
        dev += (double)D[(size_t)comp[a] * n + comp[b2]] - om;
        npair++;
      }
  }
  out[0] = kf;
  out[1] = npair ? dev / npair : 0.0;
}

// ------------------------------------------------------------------------- conjugation
//
// Union-find over conjugated bonds, then per-system size / diameter / heteroatom statistics.
// Ported from conjugation.py. The diameter comes from the shared distance matrix, so the only
// new work is the union-find, which is O(n).
static int uf_find(std::vector<int> &p, int x) {
  while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
  return x;
}

static void conjugation(const Mol &m, const std::vector<int> &D, double *out) {
  const int n = m.n;
  std::vector<int> par(n);
  for (int i = 0; i < n; i++) par[i] = i;
  std::vector<int> ncb(n, 0);
  bool any = false;
  for (int b = 0; b < m.nb; b++) {
    if (!m.bconj[b]) continue;
    any = true;
    int i = m.bu[b], j = m.bv[b];
    ncb[i]++; ncb[j]++;
    int ri = uf_find(par, i), rj = uf_find(par, j);
    if (ri != rj) par[ri] = rj;
  }
  out[0] = out[1] = out[2] = out[3] = out[4] = 0.0;
  if (!any) return;
  std::vector<std::vector<int>> groups;
  std::vector<int> gof(n, -1);
  for (int i = 0; i < n; i++) {
    if (!ncb[i]) continue;
    int r = uf_find(par, i);
    if (gof[r] < 0) { gof[r] = (int)groups.size(); groups.push_back({}); }
    groups[gof[r]].push_back(i);
  }
  int smax = 0, gmax = 0, tot = 0;
  for (size_t g = 0; g < groups.size(); g++) {
    tot += (int)groups[g].size();
    // >= not >: conjugation.py uses argsort(sizes)[::-1], and a reversed stable
    // ascending sort puts the LAST of several equal-sized systems first. Strict >
    // kept the first, which picked a different system's diameter and made
    // `linearity` disagree on 81% of molecules while `sys_max` matched on 100%.
    if ((int)groups[g].size() >= smax) { smax = (int)groups[g].size(); gmax = (int)g; }
  }
  int dmax = 0;
  for (int a : groups[gmax])
    for (int b2 : groups[gmax]) {
      int d = D[(size_t)a * n + b2];
      if (d < BIG && d > dmax) dmax = d;
    }
  int nbranch = 0, het = 0;
  for (int i = 0; i < n; i++) if (ncb[i] >= 3) nbranch++;
  for (int a : groups[gmax]) if (m.Z[a] != 1 && m.Z[a] != 6) het++;
  out[0] = (double)groups.size();
  out[1] = (double)tot;
  out[2] = (double)smax;
  out[3] = (double)dmax / std::max(smax - 1.0, 1.0);
  out[4] = (double)nbranch;
  (void)het;
}

// ------------------------------------------------------------------------------ stereo
//
// Odd-order terms in the atom parity s (which flip under reflection, so they separate
// enantiomers) and even-order terms (which do not, so they separate diastereomers), plus the
// achiral E/Z terms. Ported from stereo.py.
static void stereo(const Mol &m, const std::vector<int> &D, double *out) {
  const int n = m.n;
  double ssum = 0, sabs = 0;
  std::vector<int> idx;
  for (int i = 0; i < n; i++)
    if (m.cip[i]) { ssum += m.cip[i]; sabs += std::abs(m.cip[i]); idx.push_back(i); }
  double tsum = 0, tabs = 0;
  for (int b = 0; b < m.nb; b++)
    if (m.bstereo[b]) { tsum += m.bstereo[b]; tabs += std::abs(m.bstereo[b]); }
  double sats[7] = {0, 0, 0, 0, 0, 0, 0};
  for (size_t a = 0; a < idx.size(); a++)
    for (size_t b2 = a + 1; b2 < idx.size(); b2++) {
      int d = D[(size_t)idx[a] * n + idx[b2]];
      double v = (double)m.cip[idx[a]] * m.cip[idx[b2]];
      if (d >= 1 && d <= 6) sats[d] += v;
      else if (d < BIG) sats[0] += v;
    }
  out[0] = ssum;
  out[1] = sabs;
  out[2] = idx.empty() ? 0.0 : ssum / (double)idx.size();
  out[3] = tsum;
  out[4] = tabs;
  for (int k = 1; k <= 6; k++) out[4 + k] = sats[k];
  out[11] = sats[0];
}


// ============== merged from predict.cpp: EState, Kappa, BCUT2D ==============
// One binary, one graph build, ONE heavy-atom BFS. EState was recomputing the distance
// matrix hume.cpp had already built for resistance, conjugation and stereo -- two
// traversals of the same graph, purely because they lived in different executables.
// -> per-atom S. `work` is reused across molecules so the timed loop does not allocate.
static void estate_from(const Mol &m, const std::vector<int> &D, std::vector<double> &S) {
  S.assign(m.n, 0.0);
  std::vector<double> I(m.n, 0.0);
  for (int i = 0; i < m.n; i++) {
    if (m.deg[i] <= 0) continue;
    double dv = n_outer(m.Z[i]) - m.nH[i];
    double N = principal_qn(m.Z[i]);
    I[i] = (4.0 / (N * N) * dv + 1.0) / m.deg[i];
  }
  for (int i = 0; i < m.n; i++) S[i] = I[i];
  for (int i = 0; i < m.n; i++)
    for (int j = i + 1; j < m.n; j++) {
      double p = D[(size_t)i * m.n + j] + 1.0;
      if (p < 1e6) {
        double t = (I[i] - I[j]) / (p * p);
        S[i] += t;
        S[j] -= t;
      }
    }
}

// ALPHA IS A GENERATED TABLE, SOLVED OUT OF RDKIT ITSELF (see gen_alpha.py). alpha is additive
// over atoms, so the 10,000-molecule corpus gives 10,000 equations in the 20 distinct
// (element, hybridisation) pairs it contains; the least-squares solution reproduces RDKit's own
// CalcHallKierAlpha to 7.5e-14 over every molecule.
//
// This is deliberate, and it is not curve-fitting. Kappa and HallKierAlpha moved from Python to
// C++ inside RDKit, and the C++ routine does NOT use GetRcovalent -- the published
// hallKierAlphas table covers ten elements and everything else (B, Si, Se here) comes from an
// internal radius table that GetRcovalent disagrees with. Reimplementing from the paper gave
// visibly wrong numbers on exactly those elements. A constant table extracted from the reference
// is data, and it is verifiably exact; a formula guessed from the literature was neither.
// The off-table elements are written as the RATIO, not as a rounded decimal. RDKit computes
// rA/rC - 1 in double precision; a 6-digit literal was off by 3e-07 and that alone failed the
// exactness check on 0.33% of the corpus. rC = 0.77 and the radii below are the ones RDKit's C++
// table actually holds (recovered from the solve, and NOT the values GetRcovalent reports).
struct AlphaRow { int z, hyb; double a; };
static const double RC = 0.77;
static const AlphaRow ALPHA[] = {
    {  1, 1,  0.0},          {  5, 3, 0.82 / RC - 1.0}, {  5, 4, 0.82 / RC - 1.0},
    {  6, 2, -0.22},         {  6, 3, -0.13},           {  6, 4,  0.0},
    {  7, 2, -0.29},         {  7, 3, -0.20},           {  7, 4, -0.04},
    {  8, 3, -0.20},         {  8, 4, -0.04},           {  9, 4, -0.07},
    { 14, 4, 0.937 / RC - 1.0}, { 15, 4,  0.43},        { 16, 3,  0.22},
    { 16, 4,  0.35},         { 17, 4,  0.29},           { 34, 3, 1.17 / RC - 1.0},
    { 35, 4,  0.48},         { 53, 4,  0.73},
};

static double hk_alpha_atom_tab(int z, int hyb) {
  for (const AlphaRow &r : ALPHA)
    if (r.z == z && r.hyb == hyb) return r.a;
  for (const AlphaRow &r : ALPHA)          // unseen hybridisation for a known element
    if (r.z == z) return r.a;
  return 0.0;                              // unseen element: verifier will flag it
}

static double hk_alpha_atom_unused(int z, int hyb) {
  // index 0 = SP, 1 = SP2, 2 = SP3.   NAN marks "not defined, use the last entry".
  static const double NA = NAN;
  struct Row { int z; double a[3]; };
  static const Row T[] = {
      {35, {NA, NA, 0.48}},   // Br
      {6,  {-0.22, -0.13, 0.00}},
      {17, {NA, NA, 0.29}},   // Cl
      {9,  {NA, NA, -0.07}},  // F
      {1,  {0.0, 0.0, 0.0}},
      {53, {NA, NA, 0.73}},   // I
      {7,  {-0.29, -0.20, -0.04}},
      {8,  {NA, -0.20, -0.04}},
      {15, {NA, 0.30, 0.43}},
      {16, {NA, 0.22, 0.35}}};
  // covalent radii / r(C) - 1, for anything outside the table
  static const double rC = 0.77;
  for (const Row &r : T)
    if (r.z == z) {
      int k = hyb - 2;                       // RDKit: HybridizationType SP=2, SP2=3, SP3=4
      double a = (k >= 0 && k < 3) ? r.a[k] : r.a[2];
      if (std::isnan(a)) a = r.a[2];
      return std::isnan(a) ? 0.0 : a;
    }
  static const double RAD[] = {0, 0.32, 0.93, 1.23, 0.90, 0.82, 0.77, 0.75, 0.73, 0.72, 0.71,
                               1.54, 1.36, 1.18, 1.11, 1.06, 1.02, 0.99, 0.98, 2.03, 1.74,
                               1.44, 1.32, 1.22, 1.18, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25,
                               1.26, 1.22, 1.20, 1.16, 1.14, 1.12};
  double rA = (z >= 1 && z <= 36) ? RAD[z] : 1.20;
  return rA / rC - 1.0;
}

static double hk_alpha(const Mol &m) {
  double s = 0.0;
  for (int i = 0; i < m.n; i++) s += hk_alpha_atom_tab(m.Z[i], m.hyb[i]);
  return s;
}

// Walks of exactly k DISTINCT BONDS on the heavy-atom subgraph.
//
// Distinct BONDS, not distinct atoms -- that is RDKit's convention and the difference is
// visible: FindAllPathsOfLengthN(CC1CO1, 3) returns 3, of which only 2 have all-distinct atoms.
// The third closes the three-membered ring. An atom-distinct enumeration undercounts every
// molecule with a small ring, which is why Kappa3 matched on only 92% of the corpus before.
//
// Enumerated rather than derived from a degree formula: the closed form for k=3 needs a triangle
// correction, and getting it subtly wrong is invisible against a reference that disagrees only
// on fused systems.
static long long walks_of_len(const Mol &m, const std::vector<int> &heavy, int k) {
  long long total = 0;
  std::vector<char> used(m.nb, 0);
  // incidence: for each atom, the (neighbour, bond index) pairs, heavy ends only
  static thread_local std::vector<std::vector<std::pair<int, int>>> inc;
  inc.assign(m.n, {});
  for (int b = 0; b < m.nb; b++) {
    int u = m.bu[b], v = m.bv[b];
    if (m.Z[u] == 1 || m.Z[v] == 1) continue;
    inc[u].push_back({v, b});
    inc[v].push_back({u, b});
  }
  // OPEN AND CLOSED WALKS ARE DIVIDED BY DIFFERENT NUMBERS, which is the whole subtlety.
  // RDKit's path is a SET of bonds. An open path of k bonds is discovered exactly twice, once
  // from each end. A CYCLE of k bonds is discovered 2k times -- from each of its k atoms, in
  // each of two directions -- so dividing everything by 2 counts each ring k times over.
  // On methyloxirane that turned RDKit's P3 = 3 into 5, and Kappa3 matched on only 92% of the
  // corpus. k = 2 is unaffected (a 2-cycle cannot exist), which is exactly why Kappa2 passed
  // while Kappa3 failed.
  long long open_w = 0, closed_w = 0;
  struct Fr { int u, ptr; int bond; };
  std::vector<Fr> stk;
  for (int s : heavy) {
    stk.clear();
    stk.push_back({s, 0, -1});
    while (!stk.empty()) {
      Fr &f = stk.back();
      if ((int)stk.size() - 1 == k) {
        (f.u == s ? closed_w : open_w)++;
        if (f.bond >= 0) used[f.bond] = 0;
        stk.pop_back();
        continue;
      }
      if (f.ptr >= (int)inc[f.u].size()) {
        if (f.bond >= 0) used[f.bond] = 0;
        stk.pop_back();
        continue;
      }
      auto [v, b] = inc[f.u][f.ptr++];
      if (used[b]) continue;
      used[b] = 1;
      stk.push_back({v, 0, b});
    }
  }
  (void)total;
  return open_w / 2 + closed_w / (2 * k);
}

static void kappa(const Mol &m, double *out) {
  // A counts HEAVY atoms; P1 counts ALL bonds, including those to explicit (isotopic) hydrogen.
  // That asymmetry is RDKit's, not ours -- verified against CalcKappa1 on a tritiated molecule,
  // where only P1 = GetNumBonds() (5, not the 2 heavy bonds) reproduces its value.
  std::vector<int> heavy;
  for (int i = 0; i < m.n; i++)
    if (m.Z[i] != 1) heavy.push_back(i);
  double A = (double)heavy.size(), alpha = hk_alpha(m);
  double P1 = m.nb;
  double P2 = (double)walks_of_len(m, heavy, 2), P3 = (double)walks_of_len(m, heavy, 3);
  double d1 = P1 + alpha;
  out[0] = d1 ? (A + alpha) * (A + alpha - 1) * (A + alpha - 1) / (d1 * d1) : 0.0;
  double d2 = (P2 + alpha) * (P2 + alpha);
  out[1] = d2 ? (A + alpha - 1) * (A + alpha - 2) * (A + alpha - 2) / d2 : 0.0;
  double d3 = (P3 + alpha) * (P3 + alpha);
  double lead = ((int)A % 2 == 1) ? (A + alpha - 1) : (A + alpha - 2);
  out[2] = d3 ? lead * (A + alpha - 3) * (A + alpha - 3) / d3 : 0.0;
  out[3] = alpha;
}

struct BcutWork {
  std::vector<double> A, w, z, work;
  std::vector<int> isuppz, iwork;
};

// -> {hi, lo} for one property vector.
// BCUT2D IS A DENSE FACTORISATION, and a Lanczos attempt is deliberately NOT kept here.
//
// The idea was sound on paper: only the largest and smallest eigenvalue of each Burden matrix
// are wanted, and B = 0.001*J + sparse, so a matrix-vector product is O(n + m) and an iterative
// solver never forms the dense matrix. Measured, it lost -- 201 us against dsyevd's 94, and
// 514 us in a first form that called dsterf after every iteration. At n ~ 30 heavy atoms
// LAPACK's tuned dense kernel beats scalar Lanczos with full reorthogonalisation, and the
// asymptotic advantage never gets a chance to appear.
//
// It was briefly kept behind a size gate and that was worse than useless: with the threshold
// above every real molecule the path became dead code no test reached, and a threshold means
// one descriptor computed two ways with results differing in the last digits across the
// boundary -- a reproducibility wart on a number that goes in a paper. The claim "it wins for
// large molecules" was never measured, only assumed.
//
// If the adversarial corpus shows a large-molecule tail where this block is genuinely expensive,
// the way back in is a MEASUREMENT on that tail, not a guessed threshold.

static void bcut_one(const Mol &m, const std::vector<double> &prop, BcutWork &W,
                     double *hi, double *lo) {
  // HEAVY ATOMS ONLY. Explicit hydrogens -- which in this corpus means isotopic labels, [2H]
  // and [3H] -- are not in RDKit's Burden matrix. Including them passed 99.970% of the corpus
  // and failed exactly the three deuterated/tritiated molecules, where MWLOW came back as 1.815
  // (a deuterium mass) instead of 10.109. A 3-in-10,000 failure that is entirely one chemical
  // class is a bug, not noise; a tolerance loose enough to hide it would hide real errors too.
  static thread_local std::vector<int> hv, pos;
  hv.clear();
  pos.assign(m.n, -1);
  for (int i = 0; i < m.n; i++)
    if (m.Z[i] != 1) { pos[i] = (int)hv.size(); hv.push_back(i); }
  const int n = (int)hv.size();
  if (n == 0) { *hi = *lo = 0.0; return; }
  W.A.assign((size_t)n * n, 0.001);
  for (int i = 0; i < n; i++) W.A[(size_t)i * n + i] = prop[hv[i]];
  for (int b = 0; b < m.nb; b++) {
    int i = pos[m.bu[b]], j = pos[m.bv[b]];
    if (i < 0 || j < 0) continue;
    // OFF-DIAGONAL = 1/sqrt(bond order), and there is NO terminal-atom correction.
    //
    // Not the classic Burden 0.1*order, and not the raw order either -- both were tried and both
    // are wrong. Solved out of RDKit on two-atom molecules, where the matrix is [[p1,w],[w,p2]]
    // so w follows in closed form from the two eigenvalues:
    //     C-C  w = 1.000000 = 1/sqrt(1)
    //     C=C  w = 0.707107 = 1/sqrt(2)
    //     C#C  w = 0.577350 = 1/sqrt(3)
    // Those same molecules also confirm the diagonal is the raw property (hi + lo = p1 + p2 to
    // the last digit) and that no +0.01 terminal term exists (both atoms are terminal in C-C,
    // yet w is exactly 1). Non-bonded stays 0.001, which cyclohexane confirms independently.
    double v = 1.0 / std::sqrt(m.bord[b]);
    W.A[(size_t)i * n + j] = v;
    W.A[(size_t)j * n + i] = v;
  }
  if (n == 1) { *hi = *lo = W.A[0]; return; }

  // dsyevd, not dsyevr. dsyevr takes THREE character arguments and Accelerate's Fortran ABI
  // rejects the call (INFO = -6) without the hidden string-length arguments; dsyevd takes two
  // and is already proven against this LAPACK in cpp/bench.cpp. The cost is a FULL spectrum
  // where only the two extremes are wanted -- so the number this produces is an upper bound,
  // and an extremal-only solver (Lanczos, or dsyevr through a LAPACK that accepts it) can only
  // be faster.
  // NO WORKSPACE QUERY. The query is itself a LAPACK call, so querying before every
  // factorisation doubles the number of calls -- eight per molecule where four are needed. For
  // JOBZ='N' the requirement is documented and tiny (LWORK >= 2N+1, LIWORK >= 1), so the buffers
  // are sized directly and reused across molecules; they only ever grow.
  char jobz = 'N', uplo = 'U';
  int nn = n, lda = n, info = 0, lwork = 2 * n + 1, liwork = 1;
  W.w.resize(n);
  if ((int)W.work.size() < lwork) W.work.resize(lwork);
  if ((int)W.iwork.size() < liwork) W.iwork.resize(liwork);
  dsyevd_(&jobz, &uplo, &nn, W.A.data(), &lda, W.w.data(), W.work.data(), &lwork,
          W.iwork.data(), &liwork, &info);
  *lo = W.w[0];
  *hi = W.w[n - 1];
}

// -> 8 values in RDKit's order: MWHI, MWLOW, CHGHI, CHGLO, LOGPHI, LOGPLOW, MRHI, MRLOW
static void bcut2d(const Mol &m, BcutWork &W, double *out) {
  const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
  for (int k = 0; k < 4; k++) bcut_one(m, *props[k], W, &out[2 * k], &out[2 * k + 1]);
}

// ---------------------------------------------------------------------------------------

template <typename F>
static double time_it(const std::vector<Mol> &ms, int reps, F &&f) {
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)ms.size());
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "bench";
  auto ms = load(argc > 2 ? argv[2] : "mols.txt");
  double na = 0;
  for (auto &m : ms) na += m.n;
  fprintf(stderr, "%zu molecules, mean %.1f heavy atoms\n", ms.size(), na / ms.size());

  std::vector<int> D;
  std::vector<double> WD;
  std::vector<double> dn, dv;
  std::vector<double> ES;
  BcutWork BW;
  BW.z.resize(1);
  double kp[4], bc[8];
  double ch[2 * CHI_MAX + 2], cy[3], rs[2], cj[5], st[12];

  if (mode == "verify") {
    FILE *out = fopen("values_hume.txt", "w");
    for (auto &m : ms) {
      distances(m, D);
      chi(m, dn, dv, ch);
      cycle_counts(m, cy);
      resistance(m, D, rs);
      conjugation(m, D, cj);
      stereo(m, D, st);
      weighted_distances(m, WD);
      double J = balaban(m, WD);
      // EState reuses D -- the whole point of the merge. It used to build its own copy.
      estate_from(m, D, ES);
      double emx = -1e30, emn = 1e30, eax = -1e30, ean = 1e30;
      for (int i = 0; i < m.n; i++) {
        double v = ES[i], a = std::fabs(v);
        if (v > emx) emx = v;
        if (v < emn) emn = v;
        if (a > eax) eax = a;
        if (a < ean) ean = a;
      }
      kappa(m, kp);
      bcut2d(m, BW, bc);
      // Chi2n Chi2v Chi3n Chi3v Chi4n Chi4v | BalabanJ | C3 C4 C5 | Kf devmean |
      // conj(5) | stereo(12)
      fprintf(out, "%.12g %.12g %.12g %.12g %.12g %.12g %.12g",
              ch[4], ch[5], ch[6], ch[7], ch[8], ch[9], J);
      for (int i = 0; i < 3; i++) fprintf(out, " %.12g", cy[i]);
      for (int i = 0; i < 2; i++) fprintf(out, " %.12g", rs[i]);
      for (int i = 0; i < 5; i++) fprintf(out, " %.12g", cj[i]);
      for (int i = 0; i < 12; i++) fprintf(out, " %.12g", st[i]);
      fprintf(out, " %.12g %.12g %.12g %.12g", emx, emn, eax, ean);
      for (int i = 0; i < 4; i++) fprintf(out, " %.12g", kp[i]);
      for (int i = 0; i < 8; i++) fprintf(out, " %.12g", bc[i]);
      fputc('\n', out);
    }
    fclose(out);
    fprintf(stderr, "wrote values_hume.txt\n");
    return 0;
  }

  // THE HONEST TOTAL: exactly what verify does -- one unweighted BFS, one weighted BFS, every
  // block once. The per-block numbers below each pay for their own distance matrix, so adding
  // them up counts that traversal five times and overstates the pipeline.
  volatile double sink = 0;
  {
    double t_all = time_it(ms, 10, [&] {
      for (auto &m : ms) {
        distances(m, D);
        weighted_distances(m, WD);
        chi(m, dn, dv, ch);
        cycle_counts(m, cy);
        resistance(m, D, rs);
        conjugation(m, D, cj);
        stereo(m, D, st);
        estate_from(m, D, ES);
        kappa(m, kp);
        bcut2d(m, BW, bc);
        sink += balaban(m, WD) + ch[4] + cy[0] + rs[0] + cj[0] + st[0] + ES[0] + kp[0] + bc[0];
      }
    });
    printf("  %-44s %8.2f us/mol   <== the real pipeline cost\n",
           "ALL BLOCKS, shared distance matrices", t_all);
  }
  double t_d = time_it(ms, 20, [&] { for (auto &m : ms) { distances(m, D); sink += D[0]; } });
  double t_c = time_it(ms, 20, [&] {
    for (auto &m : ms) { chi(m, dn, dv, ch); sink += ch[4]; } });
  double t_y = time_it(ms, 20, [&] {
    for (auto &m : ms) { cycle_counts(m, cy); sink += cy[0]; } });
  double t_r = time_it(ms, 10, [&] {
    for (auto &m : ms) { distances(m, D); resistance(m, D, rs); sink += rs[0]; } });
  double t_j = time_it(ms, 20, [&] {
    for (auto &m : ms) { weighted_distances(m, WD); sink += balaban(m, WD); } });
  double t_g = time_it(ms, 20, [&] {
    for (auto &m : ms) { distances(m, D); conjugation(m, D, cj); sink += cj[0]; } });
  double t_s = time_it(ms, 20, [&] {
    for (auto &m : ms) { distances(m, D); stereo(m, D, st); sink += st[0]; } });
  printf("  %-44s %8.2f us/mol\n", "distance matrix (shared by four blocks)", t_d);
  printf("  %-44s %8.2f us/mol\n", "chi k<=7, n and v variants (16 cols)", t_c);
  printf("  %-44s %8.2f us/mol\n", "cycle counts C3-C5", t_y);
  printf("  %-44s %8.2f us/mol\n", "resistance (incl. distances)", t_r);
  printf("  %-44s %8.2f us/mol\n", "BalabanJ (incl. distances)", t_j);
  printf("  %-44s %8.2f us/mol\n", "conjugation (incl. distances)", t_g);
  printf("  %-44s %8.2f us/mol\n", "stereo (incl. distances)", t_s);
  double t_e = time_it(ms, 20, [&] {
    for (auto &m : ms) { distances(m, D); estate_from(m, D, ES); sink += ES[0]; } });
  double t_k = time_it(ms, 20, [&] { for (auto &m : ms) { kappa(m, kp); sink += kp[0]; } });
  double t_b = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  printf("  %-44s %8.2f us/mol\n", "EState (incl. distances)", t_e);
  printf("  %-44s %8.2f us/mol\n", "Kappa1-3 + HallKierAlpha", t_k);
  printf("  %-44s %8.2f us/mol\n", "BCUT2D (dsyevd, 4 dense spectra)", t_b);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
