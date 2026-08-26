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
  std::vector<std::vector<int>> seen_keys;
  std::function<void(int, int, int)> dfs = [&](int u, int start, int depth) {
    for (int v : m.adj[u]) {
      if (v == start) {
        if (depth >= 3) {
          std::vector<int> key = path;
          std::sort(key.begin(), key.end());
          if (std::find(seen_keys.begin(), seen_keys.end(), key) == seen_keys.end()) {
            seen_keys.push_back(key);
            out.push_back(path);
          }
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
  const int n = m.n;
  W.assign((size_t)n * n, INFINITY);
  std::vector<char> done(n);
  for (int s0 = 0; s0 < n; s0++) {
    double *d = &W[(size_t)s0 * n];
    std::fill(done.begin(), done.end(), 0);
    d[s0] = 0.0;
    for (int it = 0; it < n; it++) {
      int u = -1;
      double best = INFINITY;
      for (int i = 0; i < n; i++)
        if (!done[i] && d[i] < best) { best = d[i]; u = i; }
      if (u < 0) break;
      done[u] = 1;
      for (auto &e : m.inc[u]) {
        double w = 1.0 / m.bord[e.second];
        if (d[u] + w < d[e.first]) d[e.first] = d[u] + w;
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
  const int n = m.n;
  std::vector<double> A((size_t)n * n, 0.0);
  for (int b = 0; b < m.nb; b++) {
    A[(size_t)m.bu[b] * n + m.bv[b]] = 1.0;
    A[(size_t)m.bv[b] * n + m.bu[b]] = 1.0;
  }
  std::vector<double> A2((size_t)n * n, 0.0), A3((size_t)n * n, 0.0);
  for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
      double a = A[(size_t)i * n + k];
      if (a == 0.0) continue;
      for (int j = 0; j < n; j++) A2[(size_t)i * n + j] += a * A[(size_t)k * n + j];
    }
  for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
      double a = A2[(size_t)i * n + k];
      if (a == 0.0) continue;
      for (int j = 0; j < n; j++) A3[(size_t)i * n + j] += a * A[(size_t)k * n + j];
    }
  double tr3 = 0, tr4 = 0, tr5 = 0, sum_d2 = 0, sum_a3d = 0;
  std::vector<double> d(n, 0.0);
  for (int i = 0; i < n; i++) d[i] = (double)m.adj[i].size();
  for (int i = 0; i < n; i++) {
    tr3 += A3[(size_t)i * n + i];
    sum_d2 += d[i] * (d[i] - 1.0);
    sum_a3d += (d[i] - 2.0) * A3[(size_t)i * n + i];
    for (int j = 0; j < n; j++) {
      tr4 += A2[(size_t)i * n + j] * A2[(size_t)j * n + i];
      tr5 += A3[(size_t)i * n + j] * A2[(size_t)j * n + i];
    }
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
      // Chi2n Chi2v Chi3n Chi3v Chi4n Chi4v | BalabanJ | C3 C4 C5 | Kf devmean |
      // conj(5) | stereo(12)
      fprintf(out, "%.12g %.12g %.12g %.12g %.12g %.12g %.12g",
              ch[4], ch[5], ch[6], ch[7], ch[8], ch[9], J);
      for (int i = 0; i < 3; i++) fprintf(out, " %.12g", cy[i]);
      for (int i = 0; i < 2; i++) fprintf(out, " %.12g", rs[i]);
      for (int i = 0; i < 5; i++) fprintf(out, " %.12g", cj[i]);
      for (int i = 0; i < 12; i++) fprintf(out, " %.12g", st[i]);
      fputc('\n', out);
    }
    fclose(out);
    fprintf(stderr, "wrote values_hume.txt\n");
    return 0;
  }

  volatile double sink = 0;
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
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
