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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <queue>
#include <string>
#include <vector>

extern "C" {
void dsyevd_(char *, char *, int *, double *, int *, double *, double *, int *, int *, int *,
             int *);

// ACCELERATE SHIPS TWO LAPACKs AND THEY DO NOT ROUND THE SAME WAY. The bare `dgesv_` symbol
// resolves to Apple's legacy (reference-LAPACK-era) implementation; `dgesv$NEWLAPACK` is the
// modern one. numpy 2.x on this machine links Accelerate and gets the NEW one, so matching
// numpy means asking for it by name -- on K4 the new path returns inv[0,1] = 0.1875 exactly
// and the legacy path returns 0.18749999999999994, which is the ulp that moved 94 atom pairs
// across a resistance bin edge. Requested via an asm label rather than by defining
// ACCELERATE_NEW_LAPACK globally, so that dsyevd below stays on the path BCUT2D was verified
// against and this fix cannot silently perturb a block that already passes.
void dgesv_new(int *, int *, double *, int *, int *, double *, int *, int *)
    __asm__("_dgesv$NEWLAPACK");
void dgemm_new(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *,
               double *, double *, int *) __asm__("_dgemm$NEWLAPACK");
// The two stages dsyevd performs internally, exposed so they can be called without the
// blocking/workspace machinery that a 27-atom matrix cannot amortise. dsytd2 takes one
// character argument and dsterf none.
void dsytd2_(char *, int *, double *, int *, double *, double *, double *, int *);
void dsterf_(int *, double *, double *, int *);
// Eigenvalues AND eigenvectors of a tridiagonal. Needed only for the LAST ROW of the
// eigenvector matrix, which turns Lanczos's beta into a certified residual bound.
void dsteqr_(char *, int *, double *, double *, double *, int *, double *, int *);
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

// `cnt` rides along with the chi sums because chi.py's path1..path7 columns are literally the
// same enumeration with the 1/sqrt(prod) term replaced by 1 -- counting them separately would
// walk the graph a second time for numbers already in hand. It is halved on the same schedule
// as chin/chiv (open paths only), because a cycle and a lollipop are each discovered once.
struct ChiAcc {
  const Mol *m;
  const double *invn, *invv;
  std::vector<char> on;
  double chin[PMAX + 2], chiv[PMAX + 2], cnt[PMAX + 2];
};

static void chi_open(ChiAcc &S, int u, int depth, double pn, double pv) {
  const Mol &m = *S.m;
  for (int v : m.adj[u]) {
    if (S.on[v]) continue;
    double qn = pn * S.invn[v], qv = pv * S.invv[v];
    S.chin[depth] += qn;
    S.chiv[depth] += qv;
    S.cnt[depth] += 1.0;
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
    S.cnt[depth + 1] += 1.0;
    if (depth + 1 < PMAX) {
      seen[v] = 1;
      chi_tail(S, seen, v, depth + 1, qn, qv);
      seen[v] = 0;
    }
  }
}

// simple cycles, each yielded once: start from the lowest-indexed member, walk only to higher
// indices, and drop the second (reverse-direction) discovery.
//
// A CYCLE IS NOT ITS VERTEX SET. Deduplicating on the sorted member list -- which is what this
// did, and what chi.py's `key = tuple(sorted(path))` still does -- silently merges distinct
// cycles that happen to span the same atoms. The counter-example is K4, the complete graph on
// four vertices, which is a real and synthesisable substructure: TETRAHEDRANE. Its three
// distinct 4-cycles (a-b-c-d, a-b-d-c, a-c-b-d) all have vertex set {a,b,c,d}, so two of the
// three were dropped. On tetra-tert-butyl tetrahedrane that lost exactly 2 * (1/2)^4 = 0.125
// from BOTH Chi4n and Chi4v -- identical in the two variants because the four core atoms are
// carbon, where the n and v deltas coincide. RDKit's FindAllPathsOfLengthN(K4, 4) returns 15
// four-bond paths = 3 cycles + 12 lollipops; we were returning 13.
//
// The correct identity is the DIRECTED SEQUENCE up to reversal. Because the walk is pinned to
// start at the cycle's lowest-indexed member, a cycle is discovered exactly twice, as
// [s, a, ..., b] and its reverse [s, b, ..., a]. Keeping the discovery with path[1] < path.back()
// keeps exactly one of the pair, in O(1) -- so this is also strictly faster than the std::set
// it replaces, which cost a sort plus a tree insert per discovery.
static void chi_cycles(const Mol &m, std::vector<std::vector<int>> &out) {
  const int n = m.n;
  out.clear();
  std::vector<char> on(n, 0);
  std::vector<int> path;
  std::function<void(int, int, int)> dfs = [&](int u, int start, int depth) {
    for (int v : m.adj[u]) {
      if (v == start) {
        // depth >= 3 guarantees path[1] and path.back() are two distinct cycle neighbours of s
        if (depth >= 3 && path[1] < path.back()) out.push_back(path);
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

// cn[k], cv[k] = Chi_k n- and v-variant; cc[k] = the number of order-k paths, k = 0..PMAX.
// cc[0] is meaningless (chi.py's path columns start at 1) and is left at zero.
//
// RDKIT'S CONVENTION THROUGHOUT, INCLUDING WHERE RDKIT IS INTERNALLY INCONSISTENT ABOUT
// EXPLICIT HYDROGEN. Chi is somebody else's descriptor and our number has no standing, so the
// inconsistency is reproduced rather than smoothed. Measured against RDKit on eight
// explicit-H molecules, and the rule is sharp:
//
//   Chi0n, Chi1n  -> ALL atoms / ALL bonds. An explicit H is a vertex with delta = 1 (its
//                    nOuterElecs is 1 and its own GetTotalNumHs is 0), so it contributes
//                    1/sqrt(1) = 1 to Chi0n and 1/sqrt(delta_nbr * 1) to Chi1n.
//   Chi0v, Chi1v  -> HEAVY atoms / heavy-heavy bonds only. These route through _hkDeltas,
//                    which carries skipHs = 1.
//   Chi2 and up   -> HEAVY only in BOTH variants, because they route through
//                    FindAllPathsOfLengthN, whose useHs defaults to false.
//
// So Chi0n - Chi0v is exactly the number of explicit hydrogens, and the two variants agree
// from k = 2 upwards -- which is why the RDKit-gated Chi2n..Chi4v were never ambiguous.
// [2H]C(C)O gives Chi0n 3.024564 against Chi0v 2.024564; CC[13CH3] gives 2.707107 for both,
// confirming the trigger is an explicit H ATOM and not an isotope label.
//
// CRUCIALLY, the delta of a heavy atom is NOT adjusted for its explicit-H neighbours. RDKit
// drops the H from the paths but leaves GetTotalNumHs alone. chi.py used to call
// RemoveHs(removeIsotopes=True) here, which does BOTH -- it deletes the H and increments the
// neighbour's hydrogen count -- and thereby normalised [2H]C(C)O onto plain ethanol. That
// normalisation is gone: it disagreed with RDKit on 468 of 468 explicit-H molecules tested,
// while its docstring claimed the opposite.
//
// The path COUNTS (cc[], chi.py's path1..path7) stay on the heavy graph at every k, including
// k = 1. They are our own descriptor with no RDKit counterpart, and keeping them on one graph
// is the consistent choice; only chi0n/chi1n follow RDKit across the H boundary because only
// those are RDKit's to define.
static void chi(const Mol &m, std::vector<double> &invn, std::vector<double> &invv,
                double *cn, double *cv, double *cc) {
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
  // The WALK runs on the heavy-atom graph, because RDKit's FindAllPathsOfLengthN defaults to
  // useHs = false. Marking every H permanently "on the path" is that exclusion without building
  // a second adjacency structure: an atom that is always on the path is never stepped onto.
  // The k = 0 and k = 1 terms that DO see hydrogen are added outside this traversal, at the end
  // of the function, so nothing here has to know about the distinction. Only explicit H reach
  // this loop at all -- ordinary hydrogens are implicit and never become vertices.
  for (int i = 0; i < m.n; i++) if (m.Z[i] == 1) S.on[i] = 1;
  for (int i = 0; i <= PMAX + 1; i++) { S.chin[i] = 0.0; S.chiv[i] = 0.0; S.cnt[i] = 0.0; }
  for (int a = 0; a < m.n; a++) {
    if (m.Z[a] == 1) continue;
    S.on[a] = 1;
    chi_open(S, a, 1, invn[a], invv[a]);
    S.on[a] = 0;
  }
  for (int i = 0; i <= PMAX + 1; i++) {
    S.chin[i] *= 0.5; S.chiv[i] *= 0.5; S.cnt[i] *= 0.5;
  }

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
    S.cnt[L] += 1.0;
    if (L >= PMAX) continue;
    std::fill(seen.begin(), seen.end(), 0);
    for (int i = 0; i < m.n; i++) if (m.Z[i] == 1) seen[i] = 1;
    for (int x : cyc) seen[x] = 1;
    // the attachment atom enters the product a SECOND time -- this is the lollipop rule
    for (int x : cyc) chi_tail(S, seen, x, L, bn * invn[x], bv * invv[x]);
  }

  // k = 0: the n-variant counts EVERY atom, the v-variant only heavy ones. On a molecule with
  // no explicit H the two loops are the same set and this reduces to the old behaviour.
  double c0n = 0, c0v = 0;
  for (int i = 0; i < m.n; i++) {
    c0n += invn[i];
    if (m.Z[i] != 1) c0v += invv[i];
  }
  cn[0] = c0n;
  cv[0] = c0v;
  cc[0] = 0.0;
  for (int k = 1; k <= PMAX; k++) { cn[k] = S.chin[k]; cv[k] = S.chiv[k]; cc[k] = S.cnt[k]; }
  // k = 1: the DFS above ran on the heavy graph, so S.chin[1] is the heavy-heavy bond sum --
  // correct for Chi1v, short by the H-containing bonds for Chi1n. Add them back explicitly
  // rather than putting H into the traversal, which would corrupt every k >= 2.
  for (int b = 0; b < m.nb; b++) {
    int u = m.bu[b], v = m.bv[b];
    if (m.Z[u] == 1 || m.Z[v] == 1) cn[1] += invn[u] * invn[v];
  }
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
// IDEA #23 (accumulate row sums during Dijkstra, never build the n x n matrix) WAS TRIED HERE
// AND LOST. Measured in-process against the matrix form, median of 3 on a quiet machine:
// 10.40 us/mol for row sums against 10.03 for the matrix, i.e. the "optimisation" is 3.5%
// SLOWER, and reproducibly so (-3.5%, -3.4%, -3.7%).
//
// Why the reasoning failed: the operation count is IDENTICAL either way -- the same n^2 adds
// happen, only the storage differs. The saving was supposed to be memory traffic, but at a
// median 27 heavy atoms the whole matrix is 27*27*8 = 5.8 KB and already sits in L1, so there
// was no traffic to save. What the row-sum form ADDS is n separate `assign(n, 1e8)` fills
// instead of one contiguous fill of n^2, and it interleaves the summation with the Dijkstra
// instead of leaving one tight contiguous reduction. That is the same lesson the memcpy
// experiment recorded above BCUT2D taught (99.5 -> 122 us): at these sizes one big linear pass
// beats many small ones, and an O(n) working set is not an advantage when O(n^2) already fits
// in L1. The matrix form is kept.
//
// This block IS the only consumer of the weighted matrix -- the unweighted `D` shared with
// EState, conjugation, stereo and balaban_unweighted is a different array and was never
// involved -- so the idea was at least applied to the right object. It just did not pay.
static void weighted_distances(const Mol &m, std::vector<double> &W) {
  // Dijkstra with a BINARY HEAP, not a linear scan for the minimum. The scan made this O(n^3)
  // and it was 45.83 of the pipeline's 84.14 us -- 46% of the entire cost, for one descriptor.
  // Molecular graphs have degree <= 4, so m ~ n and the heap form is O(n * m log n).
  const int n = m.n;
  // RDKit's distance matrix uses 1e8 for "no path", and BalabanJ sums those rows -- so on a
  // disconnected molecule the row sums explode and J collapses to ~0. Skipping unreachable
  // pairs instead gave a finite, plausible, WRONG answer on 16% of the hard corpus, which is
  // 10,000 salts and mixtures the drug-like set contained exactly one of.
  W.assign((size_t)n * n, 1e8);
  static thread_local std::priority_queue<std::pair<double, int>,
                                          std::vector<std::pair<double, int>>,
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
        if (nd >= 1e8) continue;
        if (nd < d[e.first]) { d[e.first] = nd; pq.push({nd, e.first}); }
      }
    }
  }
}

static double balaban(const Mol &m, const std::vector<double> &W) {
  if (m.nb == 0) return 0.0;
  static thread_local std::vector<double> s;
  s.assign(m.n, 0.0);
  for (int i = 0; i < m.n; i++)
    for (int j = 0; j < m.n; j++) s[i] += W[(size_t)i * m.n + j];
  double mu = m.nb - m.n + 1.0, acc = 0.0;
  for (int b = 0; b < m.nb; b++) {
    double p = s[m.bu[b]] * s[m.bv[b]];
    if (p > 0) acc += 1.0 / std::sqrt(p);
  }
  return m.nb / (mu + 1.0) * acc;
}

// MORDRED'S BalabanJ IS A DIFFERENT DESCRIPTOR, and HUME's column set contains BOTH of them
// under the same name:
//     ('rdkit',   'BalabanJ', 'rdkit_core')
//     ('mordred', 'BalabanJ', 'BalabanJ')
// On naphthalene they are 2.888052 and 1.925368. mordred/BalabanJ.py is a thin wrapper that
// calls RDKit's OWN BalabanJ but passes `dMat = DistanceMatrix(explicit_hydrogens=False)` --
// its own UNWEIGHTED topological matrix -- which bypasses the bond-order weighting (useBO=1)
// that RDKit's default path applies. Same formula, different D. They coincide exactly when a
// molecule has only single bonds (butane agrees; CC#CC is 2.826 against 1.975), which is why
// a spot check on saturated molecules would have missed this entirely.
//
// This costs essentially nothing: D is the unweighted BFS matrix distances() has already
// built and shared with EState, conjugation and stereo, so the second column is the same
// formula over a matrix already paid for -- no second traversal, no Dijkstra.
//
// TWO THINGS THAT LOOK LIKE THEY SHOULD MATTER AND DO NOT, both measured rather than assumed:
//   * mordred sets explicit_hydrogens = False, i.e. it calls RemoveHs. On a molecule parsed
//     from SMILES that is a NO-OP (20,000 of 20,000 on cpp/hard.smi) -- sanitisation has
//     already folded away every removable H, and the 219 survivors are isotopic, charged or
//     H2, which RemoveHs keeps too. So mordred's graph is exactly the graph we already have.
//   * the corpus is 10,000 salts and mixtures, so unreachable pairs are not hypothetical.
// The sentinel, by contrast, DOES matter: RDKit writes 1e8 into unreachable cells and the J
// of a disconnected molecule is a function of that exact constant, so BIG must be mapped to
// 1e8 rather than carried through. Pinned against mordred on 4,000 corpus molecules, exact.
static double balaban_unweighted(const Mol &m, const std::vector<int> &D) {
  if (m.nb == 0) return 0.0;
  std::vector<double> s(m.n, 0.0);
  for (int i = 0; i < m.n; i++) {
    const int *row = &D[(size_t)i * m.n];
    double acc = 0.0;
    for (int j = 0; j < m.n; j++) acc += (row[j] >= BIG) ? 1e8 : (double)row[j];
    s[i] = acc;
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
//
// cycles.py's enumerator has NO de-duplication -- it yields every cycle twice, once per
// direction, and the caller halves. That is the RIGHT design and is exactly what saved it from
// the bug chi.py had, where a sorted-vertex-set key silently merged the three distinct 4-cycles
// of a K4 into one. Re-checked here against brute force (all vertex subsets, Hamiltonian test)
// at every length 3..8 on cubane (16 six-cycles, 6 four-cycles, 6 eight-cycles), prismane,
// adamantane, bicyclo[2.2.2]octane, naphthalene and K4: the yield is exactly 2x the true count
// in all cases, so the halving is sound at 6-8 where distinct cycles most often share a vertex
// set. The C++ below therefore reproduces the double-yield rather than trying to be clever.
static constexpr int LMAX = 8;
static constexpr int NKS = LMAX - 2;                 // lengths 3..8

struct CycAcc {
  const Mol *m;
  std::vector<char> *onpath;
  std::vector<int> *path;
  double *per;       // NKS * n, per-atom participation, still double-counted
  double *cntk;      // NKS raw counts, still double-counted
  double *typed;     // C5_het C6_het C5_arom C6_arom C6_carbo, still double-counted
  int n;
};

// A plain recursive function, not a std::function: the chi enumerator pays an indirect call per
// edge for the closure and this one runs over a strictly larger search space (length 8, not the
// cycles chi needs), so the type erasure is not affordable here.
static void cyc_dfs(CycAcc &S, int u, int start, int depth) {
  const Mol &m = *S.m;
  std::vector<int> &path = *S.path;
  std::vector<char> &on = *S.onpath;
  for (int v : m.adj[u]) {
    if (v == start) {
      if (depth >= 3) {
        const int ki = depth - 3;
        S.cntk[ki] += 1.0;
        double *pk = S.per + (size_t)ki * S.n;
        for (int x : path) pk[x] += 1.0;
        if (depth == 5 || depth == 6) {
          bool het = false, allarom = true;
          for (int x : path) {
            if (m.Z[x] != 1 && m.Z[x] != 6) het = true;
            if (!m.arom[x]) allarom = false;
          }
          if (het) S.typed[depth == 5 ? 0 : 1] += 1.0;
          if (allarom) S.typed[depth == 5 ? 2 : 3] += 1.0;
          if (depth == 6 && !het) S.typed[4] += 1.0;
        }
      }
    } else if (v > start && !on[v] && depth < LMAX) {
      on[v] = 1;
      path.push_back(v);
      cyc_dfs(S, v, start, depth + 1);
      path.pop_back();
      on[v] = 0;
    }
  }
}

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
  double C[NKS];
  C[0] = tr3 / 6.0;
  C[1] = (tr4 - 2.0 * m.nb - 2.0 * sum_d2) / 8.0;
  C[2] = (tr5 - 5.0 * tr3 - 5.0 * sum_a3d) / 10.0;

  // ---- bounded enumeration for C6..C8, per-atom participation and 5/6-ring typing ----
  // The closed forms give TOTALS only, so k=3,4,5 are re-enumerated here too: the per-atom
  // distribution is the part WL cannot reach and it is not recoverable from a trace.
  static thread_local std::vector<char> c_on;
  static thread_local std::vector<int> c_path;
  static thread_local std::vector<double> c_per;
  c_on.assign(n, 0);
  c_path.clear();
  c_per.assign((size_t)NKS * n, 0.0);
  double cntk[NKS] = {0, 0, 0, 0, 0, 0}, typed[5] = {0, 0, 0, 0, 0};
  CycAcc S;
  S.m = &m; S.onpath = &c_on; S.path = &c_path; S.per = c_per.data();
  S.cntk = cntk; S.typed = typed; S.n = n;
  for (int s = 0; s < n; s++) {
    c_on[s] = 1;
    c_path.push_back(s);
    cyc_dfs(S, s, s, 1);
    c_path.pop_back();
    c_on[s] = 0;
  }
  for (int k = 3; k <= LMAX; k++) if (k > 5) C[k - 3] = cntk[k - 3] * 0.5;
  for (auto &t : typed) t *= 0.5;

  double total = 0.0;
  int kmax = 0, kmin = 0;
  for (int k = 3; k <= LMAX; k++) {
    total += C[k - 3];
    if (C[k - 3] > 0.5) { if (!kmin) kmin = k; kmax = k; }
  }
  for (int i = 0; i < NKS; i++) out[i] = C[i];
  out[6] = total;
  out[7] = (double)kmax;
  out[8] = (double)kmin;
  // pa{k}_max then pa{k}_mean, both over the HALVED participation counts
  for (int ki = 0; ki < NKS; ki++) {
    const double *pk = c_per.data() + (size_t)ki * n;
    double mx = 0.0, sum = 0.0;
    for (int i = 0; i < n; i++) { double v = pk[i] * 0.5; if (v > mx) mx = v; sum += v; }
    out[9 + ki] = mx;
    out[15 + ki] = sum / n;
  }
  double amax = 0.0, asum = 0.0;
  int npos = 0, nmulti = 0;
  static thread_local std::vector<double> pa_all;
  pa_all.assign(n, 0.0);
  for (int i = 0; i < n; i++) {
    double v = 0.0;
    for (int ki = 0; ki < NKS; ki++) v += c_per[(size_t)ki * n + i] * 0.5;
    pa_all[i] = v;
    if (v > amax) amax = v;
    asum += v;
    if (v > 0.0) npos++;
    if (v > 1.0) nmulti++;
  }
  double amean = asum / n, var = 0.0;
  for (int i = 0; i < n; i++) { double d0 = pa_all[i] - amean; var += d0 * d0; }
  out[21] = amax;
  out[22] = amean;
  out[23] = std::sqrt(var / n);              // population std, matching numpy's default ddof=0
  out[24] = (double)npos / n;
  out[25] = (double)nmulti;
  for (int i = 0; i < 5; i++) out[26 + i] = typed[i];
}

// Z-indexed atomic weight, exactly resistance.py's _T[1] = PeriodicTable.GetAtomicWeight(Z).
// This is the ELEMENT AVERAGE weight. Mol::mass is GetMass(), which is isotope-aware, so
// substituting it would silently change every RATSC*_m column on any molecule carrying a
// [2H] or [13C] label -- and this corpus is full of them. Index 0 is 0.0, as in the module.
static const double RW_MASS[119] = {
    0.0, 1.008, 4.003, 6.941, 9.012, 10.812,
    12.011, 14.007, 15.999, 18.998, 20.18, 22.99,
    24.305, 26.982, 28.086, 30.974, 32.067, 35.453,
    39.948, 39.098, 40.078, 44.956, 47.867, 50.944,
    51.996, 54.938, 55.845, 58.933, 58.693, 63.546,
    65.39, 69.723, 72.61, 74.922, 78.96, 79.904,
    83.8, 85.468, 87.62, 88.906, 91.224, 92.906,
    95.94, 98.0, 101.07, 102.906, 106.42, 107.868,
    112.412, 114.818, 118.711, 121.76, 127.6, 126.904,
    131.29, 132.905, 137.328, 138.906, 140.116, 140.908,
    144.24, 145.0, 150.36, 151.964, 157.25, 158.925,
    162.5, 164.93, 167.26, 168.934, 173.04, 174.967,
    178.49, 180.948, 183.84, 186.207, 190.23, 192.217,
    195.078, 196.967, 200.59, 204.383, 207.2, 208.98,
    209.0, 210.0, 222.0, 223.0, 226.0, 227.0,
    232.038, 231.036, 238.029, 237.0, 244.0, 243.0,
    247.0, 247.0, 251.0, 252.0, 257.0, 258.0,
    259.0, 262.0, 267.0, 268.0, 269.0, 270.0,
    269.0, 278.0, 281.0, 281.0, 285.0, 284.0,
    289.0, 288.0, 293.0, 292.0, 294.0,
};

// Z-indexed vdW volume 4/3 pi Rvdw^3, exactly resistance.py's _T[4].
static const double RW_VVOL[119] = {
    0.0, 7.238229473870882, 11.494040321933852, 44.6022381005655, 28.730912014629848, 24.429024474314232,
    20.579526276115534, 17.15728467880506, 15.598531123848922, 14.137166941154067, 15.298567668493963, 57.90583579096705,
    44.6022381005655, 38.79238608652677, 38.79238608652677, 31.059355769715484, 24.429024474314232, 24.429024474314232,
    27.833136987618392, 91.95232257547082, 57.90583579096705, 50.965010421636, 41.62976785149394, 36.086951213010344,
    36.086951213010344, 36.086951213010344, 36.086951213010344, 33.510321638291124, 33.510321638291124, 33.510321638291124,
    38.79238608652677, 38.79238608652677, 38.79238608652677, 36.086951213010344, 28.730912014629848, 28.730912014629848,
    34.525717894252985, 102.16040430453528, 69.45590118188993, 57.90583579096705, 50.965010421636, 41.62976785149394,
    38.79238608652677, 36.086951213010344, 36.086951213010344, 33.510321638291124, 36.086951213010344, 38.79238608652677,
    44.6022381005655, 44.6022381005655, 47.71293842639498, 44.6022381005655, 38.79238608652677, 38.79238608652677,
    42.21335429161499, 113.09733552923254, 82.44795760081054, 65.44984694978736, 63.891583483285174, 63.1218136961418,
    61.60087235036427, 60.104561090990885, 59.36557891185266, 57.90583579096705, 56.47022010166103, 55.76139721199729,
    54.36159567894218, 52.98541892264207, 52.30612700392205, 50.965010421636, 49.64701596128037, 48.996626694973415,
    47.71293842639498, 44.6022381005655, 38.79238608652677, 36.086951213010344, 33.510321638291124, 33.510321638291124,
    36.086951213010344, 38.79238608652677, 36.086951213010344, 44.6022381005655, 50.965010421636, 50.965010421636,
    33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124,
    57.90583579096705, 33.510321638291124, 50.965010421636, 33.510321638291124, 33.510321638291124, 33.510321638291124,
    33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124,
    33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124,
    33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124,
    33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124, 33.510321638291124,
};

// Pauling electronegativity and atomic polarizability, restated from resistance.py. Anything
// off the list falls back to CARBON rather than NaN -- one exotic atom must not void a whole
// molecule's descriptor. Z = 0 is 0.0, matching the module's zero-initialised table row.
static double r_en(int z) {
  switch (z) {
    case 0: return 0.0;    case 1: return 2.20;  case 5: return 2.04;  case 6: return 2.55;
    case 7: return 3.04;   case 8: return 3.44;  case 9: return 3.98;  case 14: return 1.90;
    case 15: return 2.19;  case 16: return 2.58; case 17: return 3.16; case 34: return 2.55;
    case 35: return 2.96;  case 53: return 2.66; default: return 2.55;
  }
}
static double r_pol(int z) {
  switch (z) {
    case 0: return 0.0;    case 1: return 0.667; case 5: return 3.03;  case 6: return 1.76;
    case 7: return 1.10;   case 8: return 0.802; case 9: return 0.557; case 14: return 5.38;
    case 15: return 3.63;  case 16: return 2.90; case 17: return 2.18; case 34: return 3.77;
    case 35: return 3.05;  case 53: return 5.35; default: return 1.76;
  }
}

// ---- random-walk return probabilities: 28 columns, k = 2,3,4,6,8,12,16 ------------------
//
// diag((D^-1 A)^k) == diag(S^k) for S = D^-1/2 A D^-1/2, and S is SYMMETRIC. resistance.py
// walks Pk = Pk @ S one step at a time and so pays 16 matrix products to reach k=16. Symmetry
// buys that down to FIVE, because for symmetric M
//
//     diag(M^2)_i = sum_j (M_ij)^2                (a row norm, no product at all)
//
// so the two most expensive powers come free from ones already built:
//
//     T2 = S*S   T3 = T2*S   T4 = T2*T2   T6 = T3*T3   T8 = T4*T4      <- 5 products
//     k=2,3,4,6,8 read the diagonals; k=12 is the row norm of T6; k=16 of T8.
//
// This is a different rounding path from the sequential Python, not a different quantity --
// the reference is float32 (rtol 3e-6) and the disagreement is at 1e-15. Verified column by
// column on the full corpus rather than asserted.
static void rw_returns(const Mol &m, double *out) {
  const int n = m.n;
  static thread_local std::vector<double> S, T2, T3, T4, T6, T8, dg;
  const size_t nn = (size_t)n * n;
  S.assign(nn, 0.0);
  static thread_local std::vector<double> dinv;    // reused: cycle_counts' comment applies here
  dinv.assign(n, 1.0);
  for (int i = 0; i < n; i++) {
    double d = (double)m.adj[i].size();
    dinv[i] = 1.0 / std::sqrt(d > 0 ? d : 1.0);
  }
  for (int i = 0; i < n; i++)
    for (int j : m.adj[i]) S[(size_t)i * n + j] = dinv[i] * dinv[j];
  T2.assign(nn, 0.0); T3.assign(nn, 0.0); T4.assign(nn, 0.0);
  T6.assign(nn, 0.0); T8.assign(nn, 0.0);
  // Every operand here is a power of one symmetric matrix, so all of them are symmetric and
  // commute. That is what lets a row-major buffer go straight into Fortran dgemm: the library
  // reads it as the transpose, and for symmetric operands the transpose is the same matrix.
  double one = 1.0, zero = 0.0;
  char N = 'N';
  int nl = n;
  auto mul = [&](const std::vector<double> &A, const std::vector<double> &B,
                 std::vector<double> &C) {
    dgemm_new(&N, &N, &nl, &nl, &nl, &one, const_cast<double *>(A.data()), &nl,
           const_cast<double *>(B.data()), &nl, &zero, C.data(), &nl);
  };
  mul(S, S, T2);
  mul(T2, S, T3);
  mul(T2, T2, T4);
  mul(T3, T3, T6);
  mul(T4, T4, T8);

  auto rownorm = [&](const std::vector<double> &M, int i) {
    const double *r = &M[(size_t)i * n];
    double s = 0.0;
    for (int j = 0; j < n; j++) s += r[j] * r[j];
    return s;
  };
  dg.resize(n);
  for (int slot = 0; slot < 7; slot++) {
    for (int i = 0; i < n; i++) {
      switch (slot) {
        case 0: dg[i] = T2[(size_t)i * n + i]; break;
        case 1: dg[i] = T3[(size_t)i * n + i]; break;
        case 2: dg[i] = T4[(size_t)i * n + i]; break;
        case 3: dg[i] = T6[(size_t)i * n + i]; break;
        case 4: dg[i] = T8[(size_t)i * n + i]; break;
        case 5: dg[i] = rownorm(T6, i); break;          // S^12
        default: dg[i] = rownorm(T8, i); break;         // S^16
      }
    }
    std::sort(dg.begin(), dg.end());                    // np.sort, then mean/std over it
    double s = 0.0;
    for (int i = 0; i < n; i++) s += dg[i];
    double mean = s / n, var = 0.0;
    for (int i = 0; i < n; i++) { double d0 = dg[i] - mean; var += d0 * d0; }
    int q = (int)(0.9 * n);                             // int() truncates, as in Python
    if (q > n - 1) q = n - 1;
    out[slot * 4 + 0] = mean;
    out[slot * 4 + 1] = std::sqrt(var / n);
    out[slot * 4 + 2] = dg[n - 1];
    out[slot * 4 + 3] = dg[q];
  }
}

// -------------------------------------------------------------------------- resistance
//
// Omega_ij = L+_ii + L+_jj - 2 L+_ij, via one dense solve of (L + J/k) per connected
// component. Ported from resistance.py -- all 65 columns.
//
// THE PAIR LOOP IS THE WHOLE BLOCK. Kf, Cyclicity, the Delta statistics and the 30
// resistance-binned autocorrelation columns are all sums over the same upper-triangular pair
// list, so they are accumulated in ONE pass rather than by materialising the n x n Omega and
// scanning it six times. resistance.py builds Om, mask, delta, prod and a digitize/bincount
// over full n x n arrays because numpy makes that the fast way to write it in Python; in C++
// the same arithmetic is a single loop with no temporaries.
//
// The random-walk return probabilities need matrix powers and are handled separately below.
static void resistance(const Mol &m, const std::vector<int> &D, double *out) {
  const int n = m.n;
  for (int i = 0; i < 60; i++) out[i] = 0.0;

  // ---- centred atom properties: unity, mass, electronegativity, polarizability, vdW volume
  // NOTE the mass here is the Z-indexed ELEMENT AVERAGE (RW_MASS), not Mol::mass. resistance.py
  // indexes a table by atomic number, so a [13C] and a [12C] carry the same weight in this
  // block -- while stereo.py's S_mass uses GetMass() and does not. Both are reproduced.
  // FOUR properties, not five. The unity weight that used to sit at index 0 is gone: the
  // autocorrelation is centred, a constant centres to exactly zero, and RATSC{0..4}_c was
  // therefore identically 0.0 for every molecule that can exist. RPAIR{b} is the uncentred
  // unity autocorrelation and already carries that information -- with a constant weight every
  // pair product is 1, so the weighted sum IS the pair count.
  static thread_local std::vector<double> Pc;
  Pc.assign((size_t)n * 4, 0.0);
  double pmean[4] = {0, 0, 0, 0};
  for (int i = 0; i < n; i++) {
    int z = m.Z[i];
    if (z < 0) z = 0;
    if (z > 118) z = 118;                    // matches np.clip(z, 0, _ZMAX - 1)
    double *p = &Pc[(size_t)i * 4];
    p[0] = RW_MASS[z]; p[1] = r_en(z); p[2] = r_pol(z); p[3] = RW_VVOL[z];
    for (int j = 0; j < 4; j++) pmean[j] += p[j];
  }
  for (int j = 0; j < 4; j++) pmean[j] /= n;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 4; j++) Pc[(size_t)i * 4 + j] -= pmean[j];

  double acc[5][4] = {{0}}, cntb[5] = {0, 0, 0, 0, 0};
  // All reused across molecules. This block is the pipeline's largest and allocating six
  // vectors per molecule (two of them k x k) is exactly the heap churn that made the first
  // sparse cycle_counts slower than the dense form it replaced.
  static thread_local std::vector<char> seen;
  static thread_local std::vector<int> comp, pos, ipiv;
  static thread_local std::vector<double> L, I;
  seen.assign(n, 0);
  double kf = 0.0, dev = 0.0, cyc = 0.0, dmaxv = 0.0;
  long long npair = 0;
  for (int s = 0; s < n; s++) {
    if (seen[s]) continue;
    comp.clear();
    for (int i = 0; i < n; i++)
      if (D[(size_t)s * n + i] < BIG) { comp.push_back(i); seen[i] = 1; }
    int k = (int)comp.size();
    if (k < 2) continue;
    pos.assign(n, -1);
    for (int i = 0; i < k; i++) pos[comp[i]] = i;
    // BUILD ORDER MATTERS, and this is not pedantry. resistance.py forms
    // np.diag(A.sum(1)) - A, which is EXACT (every entry a small integer in double), and only
    // then broadcasts + 1.0/k, so each element is rounded exactly once. Accumulating 1/k first
    // and incrementing the diagonal per incident bond rounds three or four times instead, and
    // the results differ in the last ulp. That is invisible in Kf and fatal at the bin edges:
    // on tetra-tert-butyl tetrahedrane 165 atom pairs have Omega within one ulp of exactly
    // 0.5, which is a bin boundary, and the sloppy order pushed 94 of them from RPAIR2 into
    // RPAIR1. Degree is taken from the full adjacency because components are maximal.
    L.assign((size_t)k * k, 0.0);
    for (int b = 0; b < m.nb; b++) {
      int u = pos[m.bu[b]], v = pos[m.bv[b]];
      if (u < 0 || v < 0) continue;
      L[(size_t)u * k + v] = -1.0;
      L[(size_t)v * k + u] = -1.0;
    }
    for (int i = 0; i < k; i++) L[(size_t)i * k + i] = (double)m.adj[comp[i]].size();
    const double inv_k = 1.0 / k;
    for (size_t t = 0; t < L.size(); t++) L[t] += inv_k;
    // dgesv, NOT dposv, and the difference is observable. numpy.linalg.inv is gesv(A, I) --
    // an LU factorisation with partial pivoting. Cholesky is valid here (L + J/k is symmetric
    // positive definite) and is the faster factorisation, but it rounds DIFFERENTLY, and the
    // resistance bins have edges at 0.1/0.5/1.0/2.0 which exact molecular Omega values land
    // on precisely. Tetrahedrane is the clean case: every pair has Omega = 2/n = 0.5 exactly,
    // so delta = 1 - 0.5 = 0.5 sits exactly on a bin edge, and a 1-ulp difference moves a pair
    // from RPAIR2 to RPAIR1 and takes four RATSC columns with it. numpy on this machine links
    // ACCELERATE, the same LAPACK this binary links, so matching the ALGORITHM makes the
    // arithmetic identical rather than merely close. Verified: the cage set went from 10
    // mismatching columns to zero on this change alone.
    I.assign((size_t)k * k, 0.0);
    for (int i = 0; i < k; i++) I[(size_t)i * k + i] = 1.0;
    ipiv.assign(k, 0);
    int kk = k, nrhs = k, info = 0;
    dgesv_new(&kk, &nrhs, L.data(), &kk, ipiv.data(), I.data(), &kk, &info);
    if (info != 0) continue;
    // Lp = inv(L + J/k) - J/k, and Omega from Lp -- in THAT order, matching resistance.py.
    // Folding the two 1/k terms algebraically (they cancel) is not the same in floating point.
    for (size_t t = 0; t < I.size(); t++) I[t] -= 1.0 / k;
    for (int a = 0; a < k; a++)
      for (int b2 = a + 1; b2 < k; b2++) {
        // COLUMN-MAJOR READ, and the transpose is not cosmetic. LAPACK writes the solution in
        // Fortran order, and the LU inverse of a symmetric matrix is NOT exactly symmetric --
        // inv and inv^T differ by an ulp in 320 of 441 entries on this molecule. numpy
        // de-linearizes the Fortran buffer back to C order, so its inv[a][b] is our
        // I[b*k + a]. Reading row-major silently compared against the transpose and left one
        // atom pair on the wrong side of the 0.5 bin edge.
        double om = I[(size_t)a * k + a] + I[(size_t)b2 * k + b2] - 2.0 * I[(size_t)b2 * k + a];
        int ga = comp[a], gb = comp[b2];
        double d = (double)D[(size_t)ga * n + gb];
        double raw = d - om;
        double delta = raw > 0.0 ? raw : 0.0;   // np.clip(delta, 0, None): Omega <= d is a theorem
        kf += om;
        cyc += raw;                             // Cyclicity uses the UNCLIPPED difference
        dev += delta;
        if (delta > dmaxv) dmaxv = delta;
        npair++;
        // np.digitize(delta, [1e-6, 0.1, 0.5, 1.0, 2.0, inf]) - 1, i.e. edges[b] <= x < edges[b+1]
        int b = -1;
        if (delta >= 2.0) b = 4;
        else if (delta >= 1.0) b = 3;
        else if (delta >= 0.5) b = 2;
        else if (delta >= 0.1) b = 1;
        else if (delta >= 1e-6) b = 0;
        if (b >= 0) {
          cntb[b] += 1.0;
          const double *pa = &Pc[(size_t)ga * 4], *pb = &Pc[(size_t)gb * 4];
          for (int j = 0; j < 4; j++) acc[b][j] += pa[j] * pb[j];
        }
      }
  }
  for (int b = 0; b < 5; b++) {
    for (int j = 0; j < 4; j++) out[b * 5 + j] = acc[b][j];
    out[b * 5 + 4] = cntb[b];
  }
  out[25] = kf;
  out[26] = kf / n;
  out[27] = kf / std::max(n * (n - 1) / 2.0, 1.0);
  out[28] = cyc;
  out[29] = cyc / n;
  out[30] = dmaxv;
  out[31] = npair ? dev / (double)npair : 0.0;
  rw_returns(m, &out[32]);
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
  for (int i = 0; i < 24; i++) out[i] = 0.0;
  if (!any) return;                          // an alkane is legitimately all-zero, not missing
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
    // >= not >: conjugation.py uses argsort(sizes, kind="stable")[::-1], and a reversed
    // stable ascending sort puts the LAST of several equal-sized systems first. Strict >
    // kept the first, which picked a different system's diameter and made
    // `linearity` disagree on 81% of molecules while `sys_max` matched on 100%.
    //
    // The "stable" is now explicit on the Python side and this line is why. argsort's
    // default kind is an UNSTABLE introsort; it only looks stable because it insertion-sorts
    // partitions of <= 15 elements, and a 115-atom depsipeptide with 21 conjugated systems
    // (two tied at size 9, diameters 4 and 5) fell off that cliff and picked the other one.
    // This scan is a genuine last-maximal rule and does not have that failure mode.
    if ((int)groups[g].size() >= smax) { smax = (int)groups[g].size(); gmax = (int)g; }
  }
  // Per-system diameter, non-aromatic count, heteroatom count and aromatic count. Only the
  // largest system's values are reported for the *_max columns, but diam_sum and extra_arom
  // need every system, so the whole table is built rather than just group gmax.
  const size_t ng = groups.size();
  static thread_local std::vector<double> gdiam;
  static thread_local std::vector<int> gextra, ghet, garom;
  gdiam.assign(ng, 0.0); gextra.assign(ng, 0); ghet.assign(ng, 0); garom.assign(ng, 0);
  double diam_sum = 0.0;
  int extra_tot = 0;
  for (size_t g = 0; g < ng; g++) {
    const std::vector<int> &mem = groups[g];
    int dg = 0;
    if (mem.size() > 1)
      for (int a : mem)
        for (int b2 : mem) {
          int d = D[(size_t)a * n + b2];
          // unreachable stays out: conjugation.py maps the 1e8 sentinel to 0.0 before max()
          if (d < BIG && d > dg) dg = d;
        }
    gdiam[g] = (double)dg;
    diam_sum += dg;
    for (int a : mem) {
      if (!m.arom[a]) gextra[g]++;
      else garom[g]++;
      if (m.Z[a] != 1 && m.Z[a] != 6) ghet[g]++;
    }
    extra_tot += gextra[g];
  }
  // sys_2nd is sizes[order[1]] -- the second largest size WITH MULTIPLICITY, so two systems
  // tied at the maximum make it equal to sys_max. Dropping only the ONE group the tie-break
  // chose and taking the max of the rest reproduces that, and unlike order[0] it does not
  // itself depend on which of a tied pair won.
  int s2nd = 0;
  if (ng > 1)
    for (size_t g = 0; g < ng; g++)
      if ((int)g != gmax && (int)groups[g].size() > s2nd) s2nd = (int)groups[g].size();
  static const int SBIN_LO[6] = {1, 3, 5, 7, 11, 17};
  static const int SBIN_HI[6] = {2, 4, 6, 10, 16, 10000};
  double hist[6] = {0, 0, 0, 0, 0, 0};
  for (size_t g = 0; g < ng; g++) {
    int sz = (int)groups[g].size();
    for (int b = 0; b < 6; b++) if (sz >= SBIN_LO[b] && sz <= SBIN_HI[b]) hist[b] += 1.0;
  }
  int nbranch = 0;
  for (int i = 0; i < n; i++) if (ncb[i] >= 3) nbranch++;
  const double dmax = gdiam[gmax];
  out[0] = (double)ng;
  out[1] = (double)tot;
  out[2] = (double)tot / n;
  out[3] = (double)smax;
  out[4] = (double)smax / n;
  out[5] = (double)s2nd;
  out[6] = (double)tot / (double)ng;                 // sizes.mean()
  for (int b = 0; b < 6; b++) out[7 + b] = hist[b];
  out[13] = dmax;
  out[14] = dmax / std::max(smax - 1.0, 1.0);
  out[15] = diam_sum;
  out[16] = (double)nbranch;
  out[17] = (double)nbranch / std::max((double)tot, 1.0);
  out[18] = (double)extra_tot;
  out[19] = (double)extra_tot / std::max((double)tot, 1.0);
  out[20] = (double)gextra[gmax];
  out[21] = (double)ghet[gmax];
  out[22] = (double)ghet[gmax] / std::max((double)smax, 1.0);
  out[23] = (double)garom[gmax];
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
  // Centrality weight and mass weight. cen_i = 1/(1 + mean_j Dm_ij) over the WHOLE row
  // including the diagonal zero, and Dm maps unreachable to 0 exactly as stereo.py does with
  // np.where(isfinite & < 1e6, D, 0). `mass` is the atom's own GetMass(), so it IS isotope
  // aware here -- unlike the resistance autocorrelation weights, which are Z-indexed element
  // averages. The two blocks genuinely disagree about what "mass" means and both are matched.
  double scen = 0.0, smass = 0.0;
  if (!idx.empty()) {
    double mtot = 0.0;
    for (int i = 0; i < n; i++) mtot += m.mass[i];
    double mmean = mtot / n;
    for (int i : idx) {
      double rowsum = 0.0;
      const int *row = &D[(size_t)i * n];
      for (int j = 0; j < n; j++) rowsum += (row[j] < BIG) ? (double)row[j] : 0.0;
      scen += (double)m.cip[i] / (1.0 + rowsum / n);
      smass += (double)m.cip[i] * (m.mass[i] / mmean);
    }
  }
  // Bond-bond and atom-bond parity autocorrelations. Bond distance is the min over the four
  // endpoint pairs, and lag 0 is not a bucket -- two stereo bonds sharing an atom sit at 1.
  double tats[5] = {0, 0, 0, 0, 0}, xats[5] = {0, 0, 0, 0, 0};
  std::vector<int> sb;
  for (int b = 0; b < m.nb; b++) if (m.bstereo[b]) sb.push_back(b);
  for (size_t a = 0; a < sb.size(); a++)
    for (size_t b2 = a + 1; b2 < sb.size(); b2++) {
      int p = sb[a], q = sb[b2];
      int d = BIG;
      const int e1[2] = {m.bu[p], m.bv[p]}, e2[2] = {m.bu[q], m.bv[q]};
      for (int x = 0; x < 2; x++)
        for (int y = 0; y < 2; y++) {
          int dd = D[(size_t)e1[x] * n + e2[y]];
          if (dd < d) d = dd;
        }
      if (d >= 1 && d <= 4) tats[d] += (double)m.bstereo[p] * m.bstereo[q];
    }
  for (int i : idx)
    for (int b : sb) {
      int d1 = D[(size_t)i * n + m.bu[b]], d2 = D[(size_t)i * n + m.bv[b]];
      int d = d1 < d2 ? d1 : d2;
      if (d >= 1 && d <= 4) xats[d] += (double)m.cip[i] * m.bstereo[b];
    }
  out[0] = ssum;
  out[1] = sabs;
  out[2] = idx.empty() ? 0.0 : ssum / (double)idx.size();
  out[3] = scen;
  out[4] = smass;
  for (int k = 1; k <= 6; k++) out[4 + k] = sats[k];
  out[11] = sats[0];
  out[12] = tsum;
  out[13] = tabs;
  // n_EZ_any counts bonds whose GetStereo() is not STEREONONE, which is a SUPERSET of the
  // E/Z/CIS/TRANS bonds `tabs` counts -- the extra members are STEREOANY. The exporter maps
  // stereo to {-1, 0, +1} and STEREOANY collapses onto 0, so a STEREOANY bond is invisible
  // here and this column would undercount it. Measured: ZERO molecules in the 98,905-molecule
  // corpus carry a STEREOANY bond, because these are round-tripped through canonical SMILES
  // and RDKit does not emit "either" bonds there. So the two agree on every molecule we
  // verify against -- but they agree by a property of the corpus, not by construction, and
  // the honest fix is an explicit flag from export_predict.py rather than this equality.
  out[14] = tabs;
  for (int k = 1; k <= 4; k++) out[14 + k] = tats[k];
  for (int k = 1; k <= 4; k++) out[18 + k] = xats[k];
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
// SOLVED OVER THE 100k ADVERSARIAL CORPUS, not the 10k drug-like one. The first table had the
// 20 (element, hybridisation) pairs a drug-like set contains, and returned 0.0 for anything
// else -- so Ge, As, Sn and Te silently produced wrong Kappa and HallKierAlpha values for 2,743
// molecules, 2.8% of the hard corpus. Full-rank solve over 31 pairs, residual 1.9e-13.
//
// hyb = -1 means "any hybridisation". The pattern the solve reveals: elements in Kier's
// published table are hybridisation-dependent, and everything else takes rA/0.77 - 1 with no
// hybridisation term at all.
struct AlphaRow { int z, hyb; double a; };
static const double RC = 0.77;
static const AlphaRow ALPHA[] = {
    {  1, -1,  0.0},
    {  5, -1, 0.82 / RC - 1.0},            // B
    {  6,  2, -0.22}, {  6,  3, -0.13}, {  6, -1,  0.0},
    {  7,  2, -0.29}, {  7,  3, -0.20}, {  7, -1, -0.04},
    {  8,  3, -0.20}, {  8, -1, -0.04},
    {  9, -1, -0.07},
    { 14, -1, 0.937 / RC - 1.0},           // Si
    { 15,  3,  0.30}, { 15, -1,  0.43},
    { 16,  3,  0.22}, { 16, -1,  0.35},
    { 17, -1,  0.29},
    { 32, -1, +0.5428571428571427},        // Ge
    { 33, -1, +0.55844155844156118},       // As
    { 34, -1, 1.17 / RC - 1.0},            // Se
    { 35, -1,  0.48},
    { 50, -1, +0.79870129870129658},       // Sn
    { 52, -1, +0.7896103896103901},        // Te
    { 53, -1,  0.73},
};

static double hk_alpha_atom_tab(int z, int hyb) {
  for (const AlphaRow &r : ALPHA)
    if (r.z == z && r.hyb == hyb) return r.a;
  // The catch-all must be matched on hyb == -1 EXPLICITLY. Scanning for the first row with
  // this element returns whichever hybridisation happens to be listed first -- carbon at sp3
  // picked up the sp entry (-0.22 instead of 0.0) and Kappa fell from 97.2% to 4.7% correct.
  for (const AlphaRow &r : ALPHA)
    if (r.z == z && r.hyb == -1) return r.a;
  // Unknown element: the same rA/rC - 1 rule the off-table entries above follow. A mid covalent
  // radius is a guess, and it is a guess this returns LOUDLY WRONG rather than silently zero --
  // returning 0.0 is what hid Ge/As/Sn/Te for 2,743 molecules, because 0.0 is a plausible alpha.
  return 1.20 / RC - 1.0;
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
  std::vector<double> A, w, z, work, e, tau;
  std::vector<int> isuppz, iwork;
};

// 0 = dsyevd (the proven path), 1 = dsytd2 + dsterf. A RUNTIME switch, not a compile-time one,
// so both can be timed inside a SINGLE bench process against the same molecules, the same cache
// state and the same machine load. Measuring two builds in two runs is what forces the
// unchanged-block anchor correction; measuring both in one run does not need it.
// Default 1 (dsytd2 + dsterf). Best estimate -2.7% on BCUT2D: 125.1 against dsyevd's 128.7
// us/mol, from `./hume solverab`, which alternates the two solvers one corpus pass at a time
// AND alternates their order between cycles. Two independent process invocations agree
// (-2.76%, -2.66%), and all 15 cycles of each agree in sign.
//
// THE HONEST CAVEAT: this effect is close to what this machine can resolve. Across builds the
// UNCHANGED dsyevd path has measured 122.7, 128.6, 129.0, 134.0, 136.0 us/mol -- a 5% spread
// with no code change at all, which is code layout and thermal state, not the solver. Two
// earlier readings therefore came out the other way (+1.1%, +0.9%) and were not reproducible.
// What finally made the comparison stable was (a) pairing at one-corpus-pass granularity
// instead of timing all of A then all of B minutes apart, (b) alternating the order so the
// second position cannot flatter one side, and (c) fixing an unguarded resize in this
// function's own dsytd2 branch that was handicapping the candidate.
//
// The change is kept because it is FREE OF RISK, not because 2.7% is certain: the output is
// byte-identical to dsyevd's on all 98,905 molecules and all 193 columns. At n ~ 27 dsyevd
// reaches exactly these two routines anyway -- the divide-and-conquer falls back to QR below
// SMLSIZ and the blocked dsytrd has nothing to block -- so this skips the wrapper, not the
// arithmetic. If a future measurement disagrees, flipping this to 0 changes no number.
int BCUT_SOLVER = 1;

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
//
// THAT MEASUREMENT NOW EXISTS -- `./hume tridiagscale` -- AND THE TAIL IS REAL:
//
//    n range    %mols   mean n   BCUT us/mol   % of BCUT2D time
//     3-15      18.9%     10.6          9.65        1.4%
//    16-22      17.9%     19.2         24.13        3.4%
//    23-30      26.8%     26.6         43.58        9.2%
//    31-45      27.7%     35.7        113.46       24.7%
//    46-70       5.9%     53.4        333.21       15.5%
//    71+         2.9%    117.1       2031.93       45.8%
//
// The corpus MEDIAN is 27 heavy atoms, but the corpus COST is not there at all: 2.9% of the
// molecules carry 46% of the time and 8.8% carry 61%. Every molecule at or below the median
// put together is under 15%. "Molecules are small so tuned dense LAPACK wins" is true of the
// typical molecule and false of the typical microsecond, and those are different claims.
//
// WHAT THIS RULES OUT. Idea #12 -- vectorising 4-wide ACROSS the four Burden matrices, which
// share a sparsity pattern and differ only on the diagonal -- is aimed at the wrong end of that
// table. Batching quadruples the working set: at n = 117 one matrix is 110 KB, so four is
// 438 KB and no longer fits L1 (128 KB on this core). It can only help where four matrices fit
// at once, i.e. n <~ 45, which is 39% of the time; above that it would regress the 61% that
// dominates. Making it pay would therefore need a SIZE GATE, and a batched Householder does not
// reproduce dsytd2's rounding -- so the gate would mean one descriptor computed two ways with
// different last digits either side of a threshold. That is precisely the reproducibility wart
// the Lanczos removal above rejected, and it is not worth 13% of a block.
//
// WHAT THIS OPENED UP, AND WHAT HAPPENED WHEN IT WAS TRIED. Only the two EXTREMAL eigenvalues
// are wanted and dsterf computes all n of them, so an extremal-only Krylov method has its best
// case exactly where the time is. The earlier Lanczos defeat recorded above (201 us against 94)
// was measured on the 3,000-molecule drug-like benchmark, which HAS NO SUCH TAIL -- it tested
// the regime worth a few percent and never tested this one. So it was rebuilt and re-measured
// properly. It works, it is certified, it is ~10% of the pipeline, and it is NOT WIRED IN.
// Everything below is why, so that the next person to have this idea finds the measurement
// instead of repeating the month.
//
//   THE OPERATOR. bcut_one's matrix is 0.001*(J - I) + diag(prop) + sparse, so a matvec is
//   O(n + m) rather than O(n^2), and the rank-1 term and every off-diagonal are IDENTICAL for
//   all four properties -- only the diagonal changes -- so one build serves four solves. That
//   is the cross-property sharing dense LAPACK cannot express, and the economics are real:
//   matvec is 1.69x faster than dense at n = 10 and 18.0x at n = 117.
//
//   THE CERTIFICATE. After k steps the Ritz residual is exactly |beta_k * s_{k,i}|, s being the
//   LAST COMPONENT of T_k's i-th eigenvector; for a symmetric matrix the eigenvalue error is
//   bounded by it. Fail the bound -> compute densely. All four properties fall back together,
//   so a BCUT2D row is never a mixture of two numerical paths.
//
//   FULL REORTHOGONALISATION IS MANDATORY, measured not assumed. With none, or with an
//   8-vector window, 3-4 of 1200 tail cases stall at 2.2e-09 relative -- which would fail the
//   rtol 1e-9 gate this descriptor is held to, silently, on a handful of molecules. Full
//   reorth converged 1200 of 1200.
//
//   THE RESULT. On the n >= 71 tail: dense 1962 us/mol, Lanczos with oracle iteration counts
//   579 (3.39x), Lanczos with the certificate 1337 (1.47x). Corpus-wide BCUT2D falls from
//   128.65 to 110.56 us/mol at min_n = 71 -- 1.16x on the block, ~10% of the pipeline, with
//   zero fallbacks.
//
//   WHY IT IS NOT WIRED IN. It would put a SECOND NUMERICAL PATH into the block this project
//   verifies most carefully, and this repo has already deleted one Lanczos on that trade. The
//   strongest counter-argument was tested rather than waved away: the shipped descriptor matrix
//   is float32 (~1.2e-7 relative), and the two paths differ by at most 2.5e-12 in float64, five
//   orders below the representation they land in. Cast both paths to float32 and compare bit
//   for bit over the whole corpus: ZERO differences in 22,696 values at min_n = 71 and ZERO in
//   102,344 at min_n = 25. So the difference is invisible in what actually ships -- but it is
//   invisible by a margin, not by construction: the flip rate is ~1e-5 per value, so a large
//   enough corpus will eventually produce single-ULP differences. Two paths remain two paths.
//   The decision is therefore a judgement about reproducibility, not about accuracy, and it was
//   left to the owner rather than taken here. Wiring it back in is a dozen lines in bcut2d;
//   re-measuring it is `./hume lanczosprobe` and `./hume f32cmp`.
//
//   THE METHOD LESSON, which cost four wrong answers to learn. Every one of them was a probe
//   that measured an IDEALISED method rather than the one that would ship:
//     * probe 1 timed operator construction alongside the matvec, and since dense construction
//       is an n^2 fill, it flattered dense and understated the gap;
//     * probe 3 timed Lanczos with oracle iteration counts, paying nothing for the convergence
//       testing a shipping version cannot avoid -- 3.62x became 1.08x once it did;
//     * the certificate itself was wrong by up to 52 orders of magnitude, because the forward
//       three-term recurrence is unstable in exactly the converged case (the eigenvector decays
//       towards y_k, so integrating upward amplifies rounding). It reported s ~ 0.998 where the
//       truth was 1e-16, never fired, and made a working method look 4.8x slower than dense;
//     * and the first fix checked the certificate every 4 iterations from k = 28 when lambda_min
//       cannot certify before ~59, so half the certified cost was dsterf calls that could not
//       possibly have succeeded.
//   The tell for the third was that the iteration count was FLAT in tolerance -- 1e-10 and
//   1e-12 agreeing to 0.1 iterations is not how a real residual behaves, and that flatness was
//   visible three measurements before anyone looked at it.

bool BCUT_SKIP_SOLVE = false;

// ---- structured Burden operator: rank-1 + diagonal + sparse -----------------------------
//
// bcut_one fills the matrix with 0.001, overwrites the diagonal with the property and the
// bonded off-diagonals with 1/sqrt(bond order). That is exactly
//
//   A = 0.001*(J - I) + diag(prop) + sum_bonds (1/sqrt(bo) - 0.001) * (e_i e_j' + e_j e_i')
//
// so a matrix-vector product costs O(n + m) instead of O(n^2):
//
//   A*x = 0.001*(sum(x)*ones - x) + prop.x + S*x
//
// The rank-1 term and S are IDENTICAL for all four properties -- only `dg` changes -- so the
// sparse structure is built once per molecule and reused four times. That is the cross-matrix
// sharing idea #12 tried and could not extract from dense LAPACK.
struct BurdenOp {
  int n = 0;
  std::vector<double> dg;             // diagonal (the property, heavy atoms only)
  std::vector<int> bi, bj;            // bond endpoints, heavy indexing
  std::vector<double> bw;             // 1/sqrt(bo) - 0.001
};

// Structure only; `dg` is filled per property by burden_diag.
static void burden_build(const Mol &m, BurdenOp &B, std::vector<int> &hv) {
  hv.clear();
  static thread_local std::vector<int> pos;
  pos.assign(m.n, -1);
  for (int i = 0; i < m.n; i++)
    if (m.Z[i] != 1) { pos[i] = (int)hv.size(); hv.push_back(i); }
  B.n = (int)hv.size();
  B.bi.clear(); B.bj.clear(); B.bw.clear();
  for (int b = 0; b < m.nb; b++) {
    int i = pos[m.bu[b]], j = pos[m.bv[b]];
    if (i < 0 || j < 0) continue;
    B.bi.push_back(i); B.bj.push_back(j);
    B.bw.push_back(1.0 / std::sqrt(m.bord[b]) - 0.001);
  }
}

static void burden_diag(const std::vector<double> &prop,
                        const std::vector<int> &hv, BurdenOp &B) {
  B.dg.resize(B.n);
  for (int i = 0; i < B.n; i++) B.dg[i] = prop[hv[i]];
}

// PROBE SCAFFOLDING. Lanczos on the structured operator, instrumented so the convergence
// history can be read out. `reorth`: 0 = none, -1 = full, w > 0 = against the last w vectors.
// The start vector is DETERMINISTIC (a fixed smooth pseudo-random pattern) -- a constant vector
// risks being orthogonal to the extremal eigenvector, and an RNG would make the descriptor
// irreproducible, which is the one thing this project will not accept.
struct LanczosWork {
  std::vector<double> Q, w, alpha, beta, d, e, z, wk, yv, v1, v2, px, py;
};

static void burden_mv(const BurdenOp &B, const double *x, double *y);

static void lanczos_run(const BurdenOp &B, int maxit, int reorth, LanczosWork &W) {
  const int n = B.n;
  if (maxit > n) maxit = n;
  W.Q.assign((size_t)(maxit + 1) * n, 0.0);
  W.w.assign(n, 0.0);
  W.alpha.clear();
  W.beta.clear();
  double *q0 = &W.Q[0];
  double nrm = 0.0;
  for (int i = 0; i < n; i++) { q0[i] = std::sin(1.2345 * i + 0.678); nrm += q0[i] * q0[i]; }
  nrm = std::sqrt(nrm);
  for (int i = 0; i < n; i++) q0[i] /= nrm;

  for (int k = 0; k < maxit; k++) {
    double *qk = &W.Q[(size_t)k * n];
    burden_mv(B, qk, W.w.data());
    double a = 0.0;
    for (int i = 0; i < n; i++) a += qk[i] * W.w[i];
    W.alpha.push_back(a);
    for (int i = 0; i < n; i++) W.w[i] -= a * qk[i];
    if (k > 0) {
      const double *qm = &W.Q[(size_t)(k - 1) * n];
      double b = W.beta[k - 1];
      for (int i = 0; i < n; i++) W.w[i] -= b * qm[i];
    }
    if (reorth != 0) {
      int lo = (reorth < 0) ? 0 : std::max(0, k - reorth + 1);
      for (int j = lo; j <= k; j++) {
        const double *qj = &W.Q[(size_t)j * n];
        double dp = 0.0;
        for (int i = 0; i < n; i++) dp += qj[i] * W.w[i];
        for (int i = 0; i < n; i++) W.w[i] -= dp * qj[i];
      }
    }
    double bn = 0.0;
    for (int i = 0; i < n; i++) bn += W.w[i] * W.w[i];
    bn = std::sqrt(bn);
    W.beta.push_back(bn);
    if (bn < 1e-14) break;                     // invariant subspace found
    double *qn = &W.Q[(size_t)(k + 1) * n];
    for (int i = 0; i < n; i++) qn[i] = W.w[i] / bn;
  }
}

// Extremal Ritz values of the leading k x k block of T, via dsterf on a copy.
static void ritz_extremes(const LanczosWork &W, int k, LanczosWork &S, double *lo, double *hi) {
  S.d.assign(W.alpha.begin(), W.alpha.begin() + k);
  S.e.assign(k > 1 ? k - 1 : 1, 0.0);
  for (int i = 0; i + 1 < k; i++) S.e[i] = W.beta[i];
  int nn = k, info = 0;
  dsterf_(&nn, S.d.data(), S.e.data(), &info);
  *lo = S.d[0];
  *hi = S.d[k - 1];
}

// ---- production path: certified extremal eigenvalues, or an honest failure ---------------
//
// NO SIZE GATE DECIDES CORRECTNESS. The gate is the ACHIEVED RESIDUAL: after k steps, the
// Ritz pair (theta_i, y_i) of the tridiagonal T_k has exact residual norm |beta_k * s_{k,i}|,
// where s_{k,i} is the LAST COMPONENT of T_k's i-th eigenvector. For a symmetric matrix the
// eigenvalue error is bounded by that residual, so the test is a certificate rather than a
// heuristic. If either extreme fails it, this returns false and the caller runs the dense
// path. The threshold that decides which molecules TRY Lanczos is a pure performance knob and
// cannot change a value: a molecule either meets the certificate or is computed densely.
//
// The start vector is deterministic AND molecule-dependent -- it mixes the atom index with the
// diagonal, so different molecules and different properties explore different Krylov subspaces
// and a pathological start cannot be shared corpus-wide. Deterministic matters more than random
// here: the same molecule must give the same descriptor on every run, forever.
// Last component of the normalised eigenvector of a symmetric tridiagonal for eigenvalue
// theta. O(k), no eigenvectors, and -- critically -- integrated in the STABLE direction.
//
// THE FORWARD RECURRENCE IS WRONG HERE AND WRONGLY LOOKS FINE. Starting from y_1 = 1 and
// stepping up is the textbook-unstable direction whenever the eigenvector DECAYS towards y_k,
// which is exactly the converged-Ritz-pair case: rounding excites the growing solution and
// |y_k| comes back O(1) instead of O(1e-16). Measured against dsteqr on tail molecules, the
// forward version returned 0.998 where the true value was 1e-16 to 1e-53 -- wrong by up to 52
// orders of magnitude. It never reported convergence, so Lanczos ran until beta_k itself
// collapsed at k ~ n, which is a full tridiagonalisation plus O(m^2 n) reorthogonalisation on
// top, and the method looked 4.8x slower than dense when it was not.
//
// Integrating DOWNWARD from y_k = 1 follows the growing solution instead, which damps rounding
// rather than amplifying it. Only 1/||y|| is wanted, so an enormous ||y|| simply means the last
// component is negligible -- i.e. converged -- and is reported as such rather than overflowing.
static double tri_last_comp(const double *alpha, const double *beta, int kk, double theta) {
  if (kk < 2) return 1.0;
  double ynext = 0.0, y = 1.0, ss = 1.0;       // y = y[i], ynext = y[i+1], walking i downwards
  for (int i = kk - 1; i >= 1; i--) {
    const double bm = beta[i - 1];
    if (std::fabs(bm) < 1e-300) return 0.0;
    const double yprev = (-(alpha[i] - theta) * y - (i + 1 < kk ? beta[i] * ynext : 0.0)) / bm;
    ynext = y; y = yprev; ss += y * y;
    if (ss > 1e250) return 0.0;                // ||y|| astronomical => last component ~ 0
  }
  const double nrm2 = std::sqrt(ss);
  return nrm2 > 0.0 ? 1.0 / nrm2 : 0.0;
}

// Full eigenvector of T_k for eigenvalue theta, by the same stable downward recurrence, with
// rescaling. Returns false if the recurrence breaks down. y is normalised on exit.
static bool tri_eigvec(const double *alpha, const double *beta, int kk, double theta,
                       double *y) {
  if (kk < 1) return false;
  y[kk - 1] = 1.0;
  double ss = 1.0;
  for (int i = kk - 1; i >= 1; i--) {
    const double bm = beta[i - 1];
    if (std::fabs(bm) < 1e-300) return false;
    double yv = (-(alpha[i] - theta) * y[i] - (i + 1 < kk ? beta[i] * y[i + 1] : 0.0)) / bm;
    if (std::fabs(yv) > 1e100) {                 // rescale everything stored so far
      const double f = 1e-100;
      for (int t = i; t < kk; t++) y[t] *= f;
      ss *= f * f;
      yv *= f;
    }
    y[i - 1] = yv;
    ss += yv * yv;
  }
  const double nrm = std::sqrt(ss);
  if (!(nrm > 0.0)) return false;
  for (int i = 0; i < kk; i++) y[i] /= nrm;
  return true;
}

// TRUE residual ||A v - theta v|| for the Ritz vector v = Q_k y, against the ACTUAL operator
// rather than against T. O(k n) to reconstruct plus one O(n + m) matvec -- negligible beside
// the O(k^2 n) reorthogonalisation already paid, and it validates the tridiagonal proxy
// instead of trusting it.
static double true_residual(const BurdenOp &B, const LanczosWork &W, int kk, double theta,
                            const double *y, std::vector<double> &v, std::vector<double> &av) {
  const int n = B.n;
  v.assign(n, 0.0);
  for (int i = 0; i < kk; i++) {
    const double c = y[i];
    if (c == 0.0) continue;
    const double *qi = &W.Q[(size_t)i * n];
    for (int j = 0; j < n; j++) v[j] += c * qi[j];
  }
  av.assign(n, 0.0);
  burden_mv(B, v.data(), av.data());
  double r = 0.0;
  for (int j = 0; j < n; j++) { const double d = av[j] - theta * v[j]; r += d * d; }
  return std::sqrt(r);
}

// INDEPENDENT BOUNDS ON THE TRUE EXTREMES, to catch the failure the residual cannot see.
//
// A small residual proves an eigenvalue lies within ||r|| of theta. It does NOT prove theta is
// near lambda_max: if the start vector is nearly orthogonal to the lambda_max eigenvector,
// Lanczos may never enter that direction, and theta_max can converge tightly onto lambda_2
// while lambda_max is missed entirely. That is the classic missed-extreme problem, and our
// deterministic start makes any such case REPRODUCIBLE rather than measure-zero -- and full
// reorthogonalisation, which we require for accuracy, actively suppresses the rounding-driven
// recovery that would otherwise resurrect the missing direction.
//
// The Rayleigh quotient of ANY unit vector lies in [lambda_min, lambda_max]. So probing with
// vectors unrelated to the Lanczos start gives a valid LOWER bound on lambda_max and UPPER
// bound on lambda_min, for the price of a few matvecs. Power iteration drives those probes
// towards the extremes. If Lanczos missed an extreme, a probe that did not miss it will report
// a Rayleigh quotient outside the accepted interval, and acceptance is refused.
struct ExtremeBounds { double lb_max, ub_min; };

static ExtremeBounds probe_extremes(const BurdenOp &B, std::vector<double> &x,
                                    std::vector<double> &y) {
  const int n = B.n;
  y.assign(n, 0.0);                              // burden_mv writes n doubles into it
  ExtremeBounds E;
  E.lb_max = -1e300;
  E.ub_min = 1e300;
  for (int i = 0; i < n; i++) {                  // lambda_max >= max diagonal, always
    E.lb_max = std::max(E.lb_max, B.dg[i]);
    E.ub_min = std::min(E.ub_min, B.dg[i]);
  }
  // A start deliberately UNLIKE the Lanczos one, so the two cannot miss the same direction.
  x.assign(n, 0.0);
  for (int i = 0; i < n; i++) x[i] = std::cos(0.7139 * i + 2.111) + 0.37;
  auto rq_sweep = [&](double shift, bool want_max) {
    double nx = 0.0;
    for (int i = 0; i < n; i++) nx += x[i] * x[i];
    nx = std::sqrt(nx);
    if (!(nx > 0.0)) return;
    for (int i = 0; i < n; i++) x[i] /= nx;
    for (int it = 0; it < 300; it++) {
      burden_mv(B, x.data(), y.data());
      double rq = 0.0, nn2 = 0.0;
      for (int i = 0; i < n; i++) rq += x[i] * y[i];
      if (rq > E.lb_max) E.lb_max = rq;
      if (rq < E.ub_min) E.ub_min = rq;
      // iterate on (A - shift I), whose dominant direction is lambda_max or lambda_min
      for (int i = 0; i < n; i++) { y[i] -= shift * x[i]; nn2 += y[i] * y[i]; }
      nn2 = std::sqrt(nn2);
      if (!(nn2 > 1e-300)) break;
      for (int i = 0; i < n; i++) x[i] = y[i] / nn2;
      (void)want_max;
    }
  };
  // GERSHGORIN RADIUS IS A ROW MAX, NOT A GLOBAL SUM. Summing every bond in the molecule gave
  // ~190 where the true radius is ~4, which shifted the whole spectrum out to ~190 and left
  // lambda_max/lambda_2 at 1.0005 -- power iteration then cannot converge in any number of
  // steps, and the probe silently returned a bound far below lambda_max.
  double dmax = E.lb_max, dmin = E.ub_min, rad = 0.0;
  {
    static thread_local std::vector<double> rowsum;
    rowsum.assign(n, 0.001 * (n - 1));
    for (size_t e = 0; e < B.bi.size(); e++) {
      const double w = std::fabs(B.bw[e] + 0.001) - 0.001;
      rowsum[B.bi[e]] += w;
      rowsum[B.bj[e]] += w;
    }
    for (int i = 0; i < n; i++) rad = std::max(rad, rowsum[i]);
  }
  rq_sweep(dmin - rad, true);                    // push towards lambda_max
  x.assign(n, 0.0);
  for (int i = 0; i < n; i++) x[i] = std::sin(0.5311 * i + 0.911) - 0.29;
  rq_sweep(dmax + rad, false);                   // push towards lambda_min
  return E;
}

long long LANCZOS_ITERS = 0, LANCZOS_CALLS = 0;
// why an attempt was refused: 1 = ran out of Krylov space, 2 = true residual disagreed with
// the tridiagonal proxy, 3 = an independent probe found an extreme Lanczos had missed
long long LZ_FAIL[4] = {0, 0, 0, 0};
int LZ_LAST_REASON = 0;
// TEST HOOK ONLY. When set, overrides the Lanczos start vector, so a start deliberately
// orthogonal to the lambda_max eigenvector can be injected and the missed-extreme guard
// exercised. Never set on the production path.
const double *LZ_FORCE_START = nullptr;
// Largest observed ratio of the TRUE residual ||Av - theta v|| to the tridiagonal proxy
// |beta_k s_k|. If the proxy is sound this stays O(1); a large value would mean the cheap
// bound is not bounding what we think it bounds.
double LZ_MAX_TRUE_OVER_PROXY = 0.0;   // only where both are above 1e-30; see below
double LZ_MAX_TRUE_OVER_TOL = 0.0;     // the metric that actually matters

static bool lanczos_extremal(const BurdenOp &B, double tol, double *lo, double *hi,
                             LanczosWork &W) {
  const int n = B.n;
  if (n < 3) return false;
  const int maxit = n;
  LZ_LAST_REASON = 0;
  if (W.Q.size() < (size_t)(maxit + 1) * n) W.Q.resize((size_t)(maxit + 1) * n);
  W.w.assign(n, 0.0);
  W.alpha.clear();
  W.beta.clear();
  double *q0 = &W.Q[0];
  double nrm = 0.0;
  for (int i = 0; i < n; i++) {
    q0[i] = LZ_FORCE_START ? LZ_FORCE_START[i]
                           : std::sin(1.2345 * i + 0.678 + 0.31 * B.dg[i]);
    nrm += q0[i] * q0[i];
  }
  if (!(nrm > 1e-200)) {                       // degenerate start, essentially impossible
    nrm = 0.0;
    for (int i = 0; i < n; i++) { q0[i] = 1.0 + 0.01 * i; nrm += q0[i] * q0[i]; }
  }
  nrm = std::sqrt(nrm);
  for (int i = 0; i < n; i++) q0[i] /= nrm;
  // Independent of the Krylov space entirely, so it cannot inherit the Krylov space's blind
  // spot. Costs ~20 matvecs against the ~k^2 n of reorthogonalisation, i.e. nothing.
  const ExtremeBounds EB = probe_extremes(B, W.px, W.py);

  for (int k = 0; k < maxit; k++) {
    double *qk = &W.Q[(size_t)k * n];
    burden_mv(B, qk, W.w.data());
    double a = 0.0;
    for (int i = 0; i < n; i++) a += qk[i] * W.w[i];
    W.alpha.push_back(a);
    for (int i = 0; i < n; i++) W.w[i] -= a * qk[i];
    if (k > 0) {
      const double *qm = &W.Q[(size_t)(k - 1) * n];
      const double b = W.beta[k - 1];
      for (int i = 0; i < n; i++) W.w[i] -= b * qm[i];
    }
    // FULL reorthogonalisation. Measured necessary, not assumed: with none or an 8-vector
    // window, 3-4 of 1200 tail cases stalled at 2.2e-09 relative -- which would fail the
    // rtol 1e-9 gate this descriptor is held to. Full reorth converged 1200 of 1200.
    for (int j = 0; j <= k; j++) {
      const double *qj = &W.Q[(size_t)j * n];
      double dp = 0.0;
      for (int i = 0; i < n; i++) dp += qj[i] * W.w[i];
      for (int i = 0; i < n; i++) W.w[i] -= dp * qj[i];
    }
    double bn = 0.0;
    for (int i = 0; i < n; i++) bn += W.w[i] * W.w[i];
    bn = std::sqrt(bn);
    W.beta.push_back(bn);
    const int kk = k + 1;
    const bool invariant = bn < 1e-13;
    // THE CERTIFICATE MUST BE CHEAP OR IT EATS THE WIN. The first version called dsteqr with
    // COMPZ='I', which computes the FULL k x k eigenvector matrix at O(k^3) -- about 584k flops
    // at k = 46, roughly a third of an entire dense n = 117 solve, several times per property.
    // Measured, that made the whole path 2x SLOWER than dense despite the oracle probe showing
    // 3.62x faster. Only the LAST COMPONENT of two eigenvectors is actually needed, so:
    //   dsterf  -> eigenvalues only, no vectors, O(k^2)
    //   then a three-term recurrence per extreme, O(k)
    // The recurrence solves (T - theta I) y = 0 forwards from y_1 = 1 and returns
    // y_k / ||y||, rescaling on the fly so a growing solution cannot overflow.
    // Convergence lands near k = 60 on the tail, so checking from 8 wastes a dozen dsterf
    // calls on subspaces far too small to have converged. lambda_min is the slower end
    // (58.8 iterations against lambda_max's 41.7), so the first useful check is well past 40.
    // CHECK RARELY. Each check is a dsterf on the k x k tridiagonal, and measured that is
    // ~24 us at k ~ 45 -- 36 checks per molecule came to ~850 us, roughly half the entire
    // certified cost. lambda_min certifies near 59 iterations, so checking before 40 cannot
    // succeed and only burns dsterf calls. Starting at 40 with stride 8 costs at most 7
    // wasted iterations of overshoot, far cheaper than the checks it removes.
    if (kk >= 40 && (invariant || kk % 8 == 0 || kk == maxit)) {
      W.d.assign(W.alpha.begin(), W.alpha.end());
      W.e.assign(kk > 1 ? kk - 1 : 1, 0.0);
      for (int i = 0; i + 1 < kk; i++) W.e[i] = W.beta[i];
      int nn = kk, info = 0;
      dsterf_(&nn, W.d.data(), W.e.data(), &info);
      if (info == 0) {
        double scale = std::max(std::fabs(W.d[0]), std::fabs(W.d[kk - 1]));
        if (scale < 1.0) scale = 1.0;
        const double r_lo = bn * tri_last_comp(W.alpha.data(), W.beta.data(), kk, W.d[0]);
        const double r_hi = bn * tri_last_comp(W.alpha.data(), W.beta.data(), kk, W.d[kk - 1]);
        if ((r_lo <= tol * scale && r_hi <= tol * scale) || invariant) {
          const double th_lo = W.d[0], th_hi = W.d[kk - 1];
          // GUARD 1: the tridiagonal residual is a proxy. Verify against the real operator.
          W.yv.resize(kk);
          double tr_lo = 1e300, tr_hi = 1e300;
          if (tri_eigvec(W.alpha.data(), W.beta.data(), kk, th_hi, W.yv.data()))
            tr_hi = true_residual(B, W, kk, th_hi, W.yv.data(), W.v1, W.v2);
          if (tri_eigvec(W.alpha.data(), W.beta.data(), kk, th_lo, W.yv.data()))
            tr_lo = true_residual(B, W, kk, th_lo, W.yv.data(), W.v1, W.v2);
          // ACCEPT ON THE TRUE RESIDUAL AT THE NOMINAL TOLERANCE, not a slack multiple of it.
          // The eigenvalue error of a symmetric matrix is bounded by the residual, so ttol IS
          // the accuracy guarantee: at 1e-10 * scale the guaranteed relative error is 1e-10,
          // ten times inside the rtol 1e-9 gate. An earlier 1e3 * tol permitted a guarantee of
          // 1e-7 -- looser than the gate itself -- and passed only because the ACTUAL residuals
          // were ~200x better than the bound. Passing on luck is not passing.
          const double ttol = tol * scale;
          // The proxy UNDERFLOWS once a Ritz pair is deeply converged: tri_last_comp returns
          // values down to 1e-140 and below, so the true/proxy ratio explodes to 1e131 while
          // BOTH numbers are far beneath any tolerance that matters. That ratio measures
          // underflow, not unsoundness, so it is only accumulated where both are above 1e-30.
          // The metric that decides correctness is the TRUE residual against the tolerance,
          // because the true residual is the authoritative certificate and the proxy is only
          // the cheap trigger that decides when to compute it.
          if (r_hi > 1e-30 && tr_hi > 1e-30)
            LZ_MAX_TRUE_OVER_PROXY = std::max(LZ_MAX_TRUE_OVER_PROXY, tr_hi / r_hi);
          if (r_lo > 1e-30 && tr_lo > 1e-30)
            LZ_MAX_TRUE_OVER_PROXY = std::max(LZ_MAX_TRUE_OVER_PROXY, tr_lo / r_lo);

          // GUARD 2: a tight residual does NOT prove the Ritz value IS the extreme. If an
          // independent probe found a Rayleigh quotient outside the accepted interval, then
          // an extreme was missed and this molecule must go to the dense path.
          const double slack = 1e-8 * scale;
          const bool missed = (th_hi < EB.lb_max - slack) || (th_lo > EB.ub_min + slack);
          if (tr_lo <= ttol && tr_hi <= ttol && !missed) {
            // recorded on ACCEPTED molecules only -- this is the shipped guarantee, not the
            // worst thing ever seen mid-iteration on a molecule that was then rejected
            LZ_MAX_TRUE_OVER_TOL =
                std::max(LZ_MAX_TRUE_OVER_TOL, std::max(tr_hi, tr_lo) / (tol * scale));
            *lo = th_lo;
            *hi = th_hi;
            LANCZOS_ITERS += kk; LANCZOS_CALLS++;
            return true;
          }
          LZ_LAST_REASON = missed ? 3 : 2;
          if (invariant) { LZ_FAIL[LZ_LAST_REASON]++; return false; }
        }
      } else if (invariant) {
        return false;
      }
    }
    if (invariant) { LZ_FAIL[LZ_LAST_REASON ? LZ_LAST_REASON : 1]++; return false; }
    double *qn = &W.Q[(size_t)(k + 1) * n];
    for (int i = 0; i < n; i++) qn[i] = W.w[i] / bn;
  }
  LZ_FAIL[LZ_LAST_REASON ? LZ_LAST_REASON : 1]++;
  return false;
}

static void burden_mv(const BurdenOp &B, const double *x, double *y) {
  const int n = B.n;
  double s = 0.0;
  for (int i = 0; i < n; i++) s += x[i];
  const double c = 0.001;
  for (int i = 0; i < n; i++) y[i] = c * (s - x[i]) + B.dg[i] * x[i];
  const size_t nb = B.bi.size();
  for (size_t e = 0; e < nb; e++) {
    const int i = B.bi[e], j = B.bj[e];
    const double w = B.bw[e];
    y[i] += w * x[j];
    y[j] += w * x[i];
  }
}

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
  if (BCUT_SKIP_SOLVE) { *lo = W.A[0]; *hi = W.A[n - 1]; return; }
  int nn = n, lda = n, info = 0;
  // ONLY EVER GROW these buffers. Plain resize() shrinks and regrows as n varies from molecule
  // to molecule, and growing value-initialises the new tail -- a per-call cost that has nothing
  // to do with the solver being timed. The dsyevd branch below always had the only-grow guard;
  // the dsytd2 branch did not, which quietly handicapped the candidate it was being compared
  // against by a fraction of a percent on a question decided at the percent level.
  if ((int)W.w.size() < n) W.w.resize(n);
  if (BCUT_SOLVER >= 1) {
    // dsyevd = tridiagonalise, then divide-and-conquer. At n ~ 27 the D&C threshold (SMLSIZ,
    // typically 25) means it falls back to plain QR anyway, so what the wrapper actually buys
    // us is ILAENV block-size lookups and a blocked dsytrd that has almost nothing to block.
    // dsytd2 is the UNBLOCKED reduction and dsterf the QR-with-no-vectors solve: the same two
    // stages, called directly. Neither takes more than one character argument, so the Fortran
    // ABI problem that killed dsyevr does not arise here -- but INFO is still checked.
    const int ne = n > 1 ? n - 1 : 1;
    if ((int)W.e.size() < ne) W.e.resize(ne);
    if ((int)W.tau.size() < ne) W.tau.resize(ne);
    char uplo = 'U';
    dsytd2_(&uplo, &nn, W.A.data(), &lda, W.w.data(), W.e.data(), W.tau.data(), &info);
    // SOLVER 2 is a BENCH-ONLY stage probe: stop after the tridiagonalisation so the split
    // between the Householder reduction and the QR sweep can be measured. It returns a
    // meaningless number and must never be the default -- the point is to find out whether
    // vectorising the reduction (idea #12) is aimed at most of the cost or a corner of it.
    if (BCUT_SOLVER == 2) { *lo = W.w[0]; *hi = W.w[n - 1]; return; }
    if (info == 0) dsterf_(&nn, W.w.data(), W.e.data(), &info);
    if (info != 0) { *hi = *lo = 0.0; return; }
  } else {
    char jobz = 'N', uplo = 'U';
    int lwork = 2 * n + 1, liwork = 1;
    if ((int)W.work.size() < lwork) W.work.resize(lwork);
    if ((int)W.iwork.size() < liwork) W.iwork.resize(liwork);
    dsyevd_(&jobz, &uplo, &nn, W.A.data(), &lda, W.w.data(), W.work.data(), &lwork,
            W.iwork.data(), &liwork, &info);
  }
  *lo = W.w[0];
  *hi = W.w[n - 1];
}

// 1e-10 on the residual, against a verification gate of rtol 1e-9. Measured worst TRUE
// relative error at the moment of certification: 6.55e-13, three orders inside the gate.
// Used only by the diagnostic modes below -- see the Krylov note above bcut_one.
double LANCZOS_TOL = 1e-10;

// -> 8 values in RDKit's order: MWHI, MWLOW, CHGHI, CHGLO, LOGPHI, LOGPLOW, MRHI, MRLOW
// Molecules with at least this many heavy atoms attempt the Krylov path. 71 measured best
// (110.56 us/mol against dense 128.65, zero fallbacks); 55 is within noise of it.
//
// A PURE PERFORMANCE KNOB THAT SELECTS WHO *TRIES*, AND ANYONE FAILING THE CERTIFICATE IS
// COMPUTED DENSELY REGARDLESS. That is the whole argument for why this is not the size gate
// the note above rejects. The rejected gate decided CORRECTNESS by size, with nobody checking
// the two algorithms agreed. This one decides only who ATTEMPTS the cheaper route; acceptance
// is per molecule, against three independent checks on the answer itself. Moving this number
// changes how long the corpus takes, not what is in it. 0 disables the path entirely.
int BCUT_LANCZOS_MIN_N = 71;
long long BCUT_LANCZOS_TRIED = 0, BCUT_LANCZOS_FELL_BACK = 0;

static void bcut2d(const Mol &m, BcutWork &W, double *out) {
  const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
  if (BCUT_LANCZOS_MIN_N > 0) {
    static thread_local BurdenOp B;
    static thread_local std::vector<int> hv;
    static thread_local LanczosWork LW;
    int nheavy = 0;
    for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
    if (nheavy >= BCUT_LANCZOS_MIN_N) {
      // The rank-1 term and every off-diagonal are IDENTICAL across the four properties; only
      // the diagonal changes. One build, four solves -- the cross-property sharing dense
      // LAPACK cannot express.
      burden_build(m, B, hv);
      BCUT_LANCZOS_TRIED++;
      double tmp[8];
      bool ok = true;
      for (int k = 0; k < 4 && ok; k++) {
        burden_diag(*props[k], hv, B);
        ok = lanczos_extremal(B, LANCZOS_TOL, &tmp[2 * k + 1], &tmp[2 * k], LW);
      }
      if (ok) {
        for (int i = 0; i < 8; i++) out[i] = tmp[i];
        return;
      }
      // ALL FOUR fall back together, so a BCUT2D row is never a mixture of two numerical paths.
      BCUT_LANCZOS_FELL_BACK++;
    }
  }
  for (int k = 0; k < 4; k++) bcut_one(m, *props[k], W, &out[2 * k], &out[2 * k + 1]);
}

// ---------------------------------------------------------------------------------------

static volatile double sink_g = 0;             // keeps timed work from being optimised away

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
  double cn[CHI_MAX + 1], cv[CHI_MAX + 1], cc[CHI_MAX + 1];
  double cy[31], rs[60], cj[24], st[23];

  // PAIRED, ALTERNATING SOLVER A/B. The naive in-process A/B -- time all of A, then all of B --
  // is not good enough: each half takes minutes, and the machine drifts (thermal, turbo, other
  // load) by more than the effect being measured. Across three builds the SAME dsyevd code read
  // 134.02, 135.99, 130.97 and 122.45 us/mol, and the A-then-B form reported dsytd2+dsterf as
  // 2.8% faster in one build and 0.9% SLOWER in the next.
  //
  // The fix is pairing at a fine grain: alternate A and B one corpus pass at a time and compare
  // WITHIN each cycle. Drift is common-mode across a pair separated by seconds rather than
  // minutes, so the per-cycle ratio is stable even when the absolutes are not. Reported as the
  // median of the per-cycle ratios plus the best-case (minimum) time for each, since the minimum
  // is the least contaminated estimate of the true cost.
  if (mode == "solverab") {
    const int cycles = 15;
    std::vector<double> ta, tb, ratio;
    // ORDER IS ALTERNATED between cycles. Running A-then-B every time would hand B a systematic
    // second-position advantage (warm caches, settled clocks), and a 2.7% effect is exactly the
    // size such a bias could manufacture. Odd cycles run A first, even cycles run B first, so
    // any position effect cancels in the median instead of being attributed to the solver.
    for (int r = 0; r < cycles; r++) {
      const bool a_first = (r % 2) == 0;
      double a = 0, b = 0;
      auto run_a = [&] { BCUT_SOLVER = 0; a = time_it(ms, 1, [&] {
        for (auto &m : ms) { bcut2d(m, BW, bc); sink_g += bc[0]; } }); };
      auto run_b = [&] { BCUT_SOLVER = 1; b = time_it(ms, 1, [&] {
        for (auto &m : ms) { bcut2d(m, BW, bc); sink_g += bc[0]; } }); };
      if (a_first) { run_a(); run_b(); } else { run_b(); run_a(); }
      ta.push_back(a); tb.push_back(b); ratio.push_back(b / a);
      fprintf(stderr, "  cycle %2d [%s]  dsyevd %7.2f  dsytd2+dsterf %7.2f  ratio %.4f\n",
              r + 1, a_first ? "A,B" : "B,A", a, b, b / a);
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    printf("  %d paired cycles over %zu molecules\n", cycles, ms.size());
    printf("  dsyevd          median %7.2f  min %7.2f us/mol\n", med(ta),
           *std::min_element(ta.begin(), ta.end()));
    printf("  dsytd2+dsterf   median %7.2f  min %7.2f us/mol\n", med(tb),
           *std::min_element(tb.begin(), tb.end()));
    printf("  PAIRED ratio (dsytd2+dsterf / dsyevd): median %.4f  -> %+.2f%%\n",
           med(ratio), 100.0 * (med(ratio) - 1.0));
    return 0;
  }

  // DOES IDEA #12 (4-wide SIMD across the four tridiagonalisations) HAVE A CEILING WORTH THE
  // BUILD? That depends entirely on whether dsytd2 at n ~ 27 is OVERHEAD-bound or FLOP-bound.
  // Batching four reductions into one vectorised pass amortises per-call and per-column
  // overhead and improves ILP; it does NOT reduce the flop count. So:
  //   cost / n^3 roughly constant  -> FLOP-bound, batching buys ILP only, ceiling modest
  //   cost roughly flat in n       -> overhead-bound, batching is the right lever, ceiling high
  // Measured by bucketing the corpus on heavy-atom count and timing tridiagonalisation only.
  if (mode == "tridiagscale") {
    struct Bucket { int lo, hi; std::vector<const Mol *> ms; };
    std::vector<Bucket> bk = {{3, 15, {}}, {16, 22, {}}, {23, 30, {}},
                              {31, 45, {}}, {46, 70, {}}, {71, 1000, {}}};
    for (auto &m : ms) {
      int hv = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv++;
      for (auto &b : bk) if (hv >= b.lo && hv <= b.hi) { b.ms.push_back(&m); break; }
    }
    auto timed = [&](std::vector<const Mol *> &v, int solver) {
      BCUT_SOLVER = solver;
      const int reps = 5;
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; r++)
        for (auto *m : v) { bcut2d(*m, BW, bc); sink_g += bc[0]; }
      auto t1 = std::chrono::steady_clock::now();
      BCUT_SOLVER = 1;
      return std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)v.size());
    };
    std::vector<double> full(bk.size(), 0), tri(bk.size(), 0), mean_n(bk.size(), 0);
    double total = 0;
    for (size_t i = 0; i < bk.size(); i++) {
      if (bk[i].ms.empty()) continue;
      double mn = 0;
      for (auto *m : bk[i].ms) { int hv = 0; for (int j = 0; j < m->n; j++) if (m->Z[j] != 1) hv++;
                                 mn += hv; }
      mean_n[i] = mn / bk[i].ms.size();
      full[i] = timed(bk[i].ms, 1);
      tri[i] = timed(bk[i].ms, 2);
      total += full[i] * bk[i].ms.size();
    }
    printf("  %-10s %8s %7s %6s %11s %11s %10s %9s\n", "n range", "count", "%mols", "mean n",
           "BCUT us/mol", "tridiag us", "us/n^3 e6", "%of time");
    for (size_t i = 0; i < bk.size(); i++) {
      if (bk[i].ms.empty()) continue;
      printf("  %3d-%-6d %8zu %6.1f%% %6.1f %11.2f %11.2f %10.1f %8.1f%%\n",
             bk[i].lo, bk[i].hi, bk[i].ms.size(), 100.0 * bk[i].ms.size() / ms.size(), mean_n[i],
             full[i], tri[i], 1e6 * tri[i] / (mean_n[i] * mean_n[i] * mean_n[i]),
             100.0 * full[i] * bk[i].ms.size() / total);
    }
    printf("  corpus-weighted BCUT2D: %.2f us/mol\n", total / ms.size());
    return 0;
  }

  // MEASURE BEFORE BUILDING. Three probes decide whether a structured-matvec Lanczos can beat
  // dsytd2+dsterf on the large-molecule tail that carries 46% of BCUT2D's time.
  if (mode == "lanczosprobe") {
    std::vector<const Mol *> tail, mid;
    for (auto &m : ms) {
      int hv = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv++;
      if (hv >= 71) tail.push_back(&m);
      else if (hv >= 31 && hv <= 45) mid.push_back(&m);
    }
    printf("tail (n>=71): %zu molecules   mid (31-45): %zu\n\n", tail.size(), mid.size());

    // ---- PROBE 1: structured vs dense MATVEC ONLY -------------------------------------
    // The operator is built OUTSIDE the timed loop. Lanczos builds once and then does m
    // matvecs, so folding construction into the per-matvec cost would flatter the dense side
    // (whose construction is an n^2 fill) and answer a question nobody asked.
    printf("PROBE 1  matvec cost with the operator already built, structured vs dense\n");
    printf("  %8s %7s %7s %12s %12s %8s\n", "n range", "mols", "mean n", "dense us", "sparse us",
           "speedup");
    struct B2 { int lo, hi; };
    std::vector<B2> rng = {{3, 15}, {16, 22}, {23, 30}, {31, 45}, {46, 70}, {71, 100000}};
    for (auto &r : rng) {
      std::vector<const Mol *> g;
      double mn = 0;
      for (auto &m : ms) {
        int hv2 = 0;
        for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv2++;
        if (hv2 >= r.lo && hv2 <= r.hi) { g.push_back(&m); mn += hv2; }
      }
      if (g.size() < 50) continue;
      mn /= g.size();
      if (g.size() > 1500) g.resize(1500);
      std::vector<BurdenOp> ops(g.size());
      std::vector<std::vector<double>> dense(g.size());
      std::vector<int> hv;
      for (size_t t = 0; t < g.size(); t++) {
        burden_build(*g[t], ops[t], hv);
        burden_diag(g[t]->mass, hv, ops[t]);
        int n2 = ops[t].n;
        dense[t].assign((size_t)n2 * n2, 0.001);
        for (int i = 0; i < n2; i++) dense[t][(size_t)i * n2 + i] = ops[t].dg[i];
        for (size_t e2 = 0; e2 < ops[t].bi.size(); e2++) {
          double v = ops[t].bw[e2] + 0.001;
          dense[t][(size_t)ops[t].bi[e2] * n2 + ops[t].bj[e2]] = v;
          dense[t][(size_t)ops[t].bj[e2] * n2 + ops[t].bi[e2]] = v;
        }
      }
      std::vector<double> x, y;
      const int reps = 300;
      auto t0 = std::chrono::steady_clock::now();
      for (int r2 = 0; r2 < reps; r2++)
        for (size_t t = 0; t < g.size(); t++) {
          x.assign(ops[t].n, 1.0); y.assign(ops[t].n, 0.0);
          burden_mv(ops[t], x.data(), y.data());
          sink_g += y[0];
        }
      auto t1 = std::chrono::steady_clock::now();
      double ts = std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)g.size());
      t0 = std::chrono::steady_clock::now();
      for (int r2 = 0; r2 < reps; r2++)
        for (size_t t = 0; t < g.size(); t++) {
          int n2 = ops[t].n;
          x.assign(n2, 1.0); y.assign(n2, 0.0);
          for (int i = 0; i < n2; i++) {
            double s2 = 0.0;
            const double *row = &dense[t][(size_t)i * n2];
            for (int j = 0; j < n2; j++) s2 += row[j] * x[j];
            y[i] = s2;
          }
          sink_g += y[0];
        }
      t1 = std::chrono::steady_clock::now();
      double td = std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)g.size());
      printf("  %3d-%-4d %7zu %7.1f %12.4f %12.4f %7.2fx\n", r.lo, r.hi, g.size(), mn, td, ts,
             td / ts);
    }

    // ---- PROBE 2: iterations to converge on the tail ----------------------------------
    printf("\nPROBE 2  Lanczos iterations for lambda_min AND lambda_max to 1e-12 relative\n");
    printf("         (ground truth = dsytd2+dsterf; over the n>=71 bucket)\n");
    const int REORTH[3] = {0, 8, -1};
    const char *RNAME[3] = {"none", "window-8", "full"};
    std::vector<const Mol *> samp = tail;
    if (samp.size() > 300) samp.resize(300);
    std::vector<std::array<int, 4>> need(samp.size(), {0, 0, 0, 0});
    for (int rm = 0; rm < 3; rm++) {
      std::vector<int> iters;
      int failed = 0;
      double worst_rel = 0.0;
      LanczosWork L, S;
      for (size_t t = 0; t < samp.size(); t++) {
        const Mol *m = samp[t];
        BurdenOp B;
        std::vector<int> hv;
        burden_build(*m, B, hv);
        const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
        for (int p = 0; p < 4; p++) {
          burden_diag(*props[p], hv, B);
          double dlo, dhi;
          BCUT_SOLVER = 1;
          bcut_one(*m, *props[p], BW, &dhi, &dlo);
          lanczos_run(B, std::min(B.n, 400), REORTH[rm], L);
          int used = -1;
          for (int k = 2; k <= (int)L.alpha.size(); k++) {
            double rlo, rhi;
            ritz_extremes(L, k, S, &rlo, &rhi);
            if (std::fabs(rlo - dlo) <= 1e-12 * std::max(std::fabs(dlo), 1e-30) &&
                std::fabs(rhi - dhi) <= 1e-12 * std::max(std::fabs(dhi), 1e-30)) { used = k; break; }
          }
          if (used < 0) {
            failed++;
            double rlo, rhi;
            ritz_extremes(L, (int)L.alpha.size(), S, &rlo, &rhi);
            worst_rel = std::max(worst_rel,
                std::max(std::fabs(rlo - dlo) / std::max(std::fabs(dlo), 1e-30),
                         std::fabs(rhi - dhi) / std::max(std::fabs(dhi), 1e-30)));
            used = (int)L.alpha.size();
          } else {
            iters.push_back(used);
          }
          if (rm == 2) need[t][p] = used;               // full reorth drives probe 3
        }
      }
      std::sort(iters.begin(), iters.end());
      auto pct = [&](double f) {
        return iters.empty() ? -1 : iters[std::min(iters.size() - 1, (size_t)(f * iters.size()))];
      };
      printf("  reorth=%-9s converged %5zu/%5zu  iters p50 %4d  p90 %4d  p99 %4d  max %4d",
             RNAME[rm], iters.size(), samp.size() * 4, pct(0.50), pct(0.90), pct(0.99),
             iters.empty() ? -1 : iters.back());
      if (failed) printf("   FAILED %d (worst rel %.2e)", failed, worst_rel);
      printf("\n");
    }

    // ---- PROBE 3: end-to-end, the number that actually decides -------------------------
    // Lanczos is run with FULL reorthogonalisation (the only variant that reached 1e-12 on
    // every case) for EXACTLY the iteration count convergence needed, taken from probe 2.
    // That is a LOWER BOUND on the real cost: a shipping version cannot know the count in
    // advance and must pay for convergence testing on top. If the lower bound loses, the
    // idea is dead without writing a stopping rule.
    printf("\nPROBE 3  end-to-end on the n>=71 tail: Lanczos(full reorth, oracle iters) vs dense\n");
    {
      LanczosWork L, S;
      std::vector<int> hv;
      double t_lan = 0, t_dense = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (size_t t = 0; t < samp.size(); t++) {
        const Mol *m = samp[t];
        BurdenOp B;
        burden_build(*m, B, hv);
        const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
        for (int p = 0; p < 4; p++) {
          burden_diag(*props[p], hv, B);
          lanczos_run(B, need[t][p], -1, L);
          double rlo, rhi;
          ritz_extremes(L, (int)L.alpha.size(), S, &rlo, &rhi);
          sink_g += rlo + rhi;
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      t_lan = std::chrono::duration<double, std::micro>(t1 - t0).count() / samp.size();
      BCUT_SOLVER = 1;
      t0 = std::chrono::steady_clock::now();
      for (size_t t = 0; t < samp.size(); t++) { bcut2d(*samp[t], BW, bc); sink_g += bc[0]; }
      t1 = std::chrono::steady_clock::now();
      t_dense = std::chrono::duration<double, std::micro>(t1 - t0).count() / samp.size();
      printf("  dense dsytd2+dsterf, 4 spectra : %9.2f us/mol\n", t_dense);
      printf("  Lanczos ORACLE iters, 4 spectra: %9.2f us/mol  (%.2fx %s)\n", t_lan,
             t_dense > t_lan ? t_dense / t_lan : t_lan / t_dense,
             t_dense > t_lan ? "FASTER" : "SLOWER");
      // The shipping version cannot know the iteration count in advance; it must certify.
      // How many MORE iterations does certifying cost than converging?
      extern long long LANCZOS_ITERS, LANCZOS_CALLS;
      LANCZOS_ITERS = LANCZOS_CALLS = 0;
      long long oracle_sum = 0, oracle_n = 0;
      for (size_t t = 0; t < samp.size(); t++)
        for (int p = 0; p < 4; p++) { oracle_sum += need[t][p]; oracle_n++; }
      t0 = std::chrono::steady_clock::now();
      int nconv = 0;
      for (size_t t = 0; t < samp.size(); t++) {
        const Mol *m = samp[t];
        BurdenOp B;
        burden_build(*m, B, hv);
        const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
        for (int p = 0; p < 4; p++) {
          burden_diag(*props[p], hv, B);
          double rlo, rhi;
          if (lanczos_extremal(B, LANCZOS_TOL, &rlo, &rhi, L)) { nconv++; sink_g += rlo + rhi; }
        }
      }
      t1 = std::chrono::steady_clock::now();
      double t_cert = std::chrono::duration<double, std::micro>(t1 - t0).count() / samp.size();
      printf("  Lanczos CERTIFIED,   4 spectra: %9.2f us/mol  (%.2fx %s)\n", t_cert,
             t_dense > t_cert ? t_dense / t_cert : t_cert / t_dense,
             t_dense > t_cert ? "FASTER" : "SLOWER");
      printf("    mean iters: oracle %.1f   certified %.1f   ratio %.2fx   certified %d/%lld\n",
             (double)oracle_sum / oracle_n,
             LANCZOS_CALLS ? (double)LANCZOS_ITERS / LANCZOS_CALLS : 0.0,
             (LANCZOS_CALLS && oracle_n) ?
               ((double)LANCZOS_ITERS / LANCZOS_CALLS) / ((double)oracle_sum / oracle_n) : 0.0,
             nconv, oracle_n);
    }
    return 0;
  }

  if (mode == "certsweep") {
    std::vector<const Mol *> tail;
    for (auto &m : ms) {
      int hv2 = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv2++;
      if (hv2 >= 71) tail.push_back(&m);
    }
    if (tail.size() > 200) tail.resize(200);
    printf("tail sample: %zu molecules x 4 properties\n\n", tail.size());
    const double TOLS[3] = {1e-10, 1e-11, 1e-12};
    const int STEP = 2;
    // [tol][0]=hi only, [1]=lo only, [2]=both
    double sum_it[3][3] = {{0}};
    long long cnt_it[3][3] = {{0}};
    long long nfail[3][3] = {{0}};
    double sum_oracle = 0; long long cnt_oracle = 0;
    double worst_err[3] = {0, 0, 0};
    LanczosWork L, S;
    std::vector<int> hv;
    for (auto *m : tail) {
      BurdenOp B;
      burden_build(*m, B, hv);
      const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
      for (int p = 0; p < 4; p++) {
        burden_diag(*props[p], hv, B);
        double dlo, dhi;
        BCUT_SOLVER = 1;
        bcut_one(*m, *props[p], BW, &dhi, &dlo);
        const int maxit = std::min(B.n, 260);
        lanczos_run(B, maxit, -1, L);
        const int K = (int)L.alpha.size();
        int first_or = -1;
        int first[3][3];
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) first[a][b] = -1;
        for (int kk = 8; kk <= K; kk += STEP) {
          S.d.assign(L.alpha.begin(), L.alpha.begin() + kk);
          S.e.assign(kk > 1 ? kk - 1 : 1, 0.0);
          for (int i = 0; i + 1 < kk; i++) S.e[i] = L.beta[i];
          int nn = kk, info = 0;
          dsterf_(&nn, S.d.data(), S.e.data(), &info);
          if (info != 0) continue;
          const double bn = L.beta[kk - 1];
          double scale = std::max(std::fabs(S.d[0]), std::fabs(S.d[kk - 1]));
          if (scale < 1.0) scale = 1.0;
          const double r_lo = bn * tri_last_comp(L.alpha.data(), L.beta.data(), kk, S.d[0]);
          const double r_hi = bn * tri_last_comp(L.alpha.data(), L.beta.data(), kk, S.d[kk - 1]);
          if (first_or < 0 &&
              std::fabs(S.d[0] - dlo) <= 1e-12 * std::max(std::fabs(dlo), 1e-30) &&
              std::fabs(S.d[kk - 1] - dhi) <= 1e-12 * std::max(std::fabs(dhi), 1e-30))
            first_or = kk;
          for (int a = 0; a < 3; a++) {
            const double t = TOLS[a] * scale;
            if (first[a][0] < 0 && r_hi <= t) {
              first[a][0] = kk;
              worst_err[a] = std::max(worst_err[a],
                  std::fabs(S.d[kk - 1] - dhi) / std::max(std::fabs(dhi), 1e-30));
            }
            if (first[a][1] < 0 && r_lo <= t) {
              first[a][1] = kk;
              worst_err[a] = std::max(worst_err[a],
                  std::fabs(S.d[0] - dlo) / std::max(std::fabs(dlo), 1e-30));
            }
            if (first[a][2] < 0 && r_lo <= t && r_hi <= t) first[a][2] = kk;
          }
        }
        if (first_or > 0) { sum_oracle += first_or; cnt_oracle++; }
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) {
            if (first[a][b] > 0) { sum_it[a][b] += first[a][b]; cnt_it[a][b]++; }
            else nfail[a][b]++;
          }
      }
    }
    printf("  oracle (true err <= 1e-12, both ends): mean %.1f iters\n\n",
           cnt_oracle ? sum_oracle / cnt_oracle : 0.0);
    printf("  %-8s %14s %14s %14s %16s %12s\n", "tol", "cert hi only", "cert lo only",
           "cert BOTH", "both/oracle", "never cert");
    for (int a = 0; a < 3; a++) {
      double mo = cnt_oracle ? sum_oracle / cnt_oracle : 1.0;
      auto mean = [&](int b) { return cnt_it[a][b] ? sum_it[a][b] / cnt_it[a][b] : -1.0; };
      printf("  %-8.0e %14.1f %14.1f %14.1f %15.2fx %12lld\n", TOLS[a], mean(0), mean(1),
             mean(2), mean(2) / mo, nfail[a][2]);
    }
    printf("\n  worst TRUE relative error at first certification: 1e-10 %.2e  1e-11 %.2e  "
           "1e-12 %.2e\n", worst_err[0], worst_err[1], worst_err[2]);
    printf("  reorth cost scales as m^2, so predicted work ratio vs oracle is (both/oracle)^2\n");
    return 0;
  }

  // IS THE CERTIFICATE ITSELF BROKEN? tri_last_comp uses the FORWARD three-term recurrence,
  // which is the textbook-unstable direction when the eigenvector decays away from y_1:
  // rounding excites the growing solution and |y_k| comes out far too LARGE, which inflates
  // the residual and stops the certificate from ever firing. Compare it against dsteqr, which
  // is expensive but correct, at a fixed k well past true convergence.
  if (mode == "certcheck") {
    std::vector<const Mol *> tail;
    for (auto &m : ms) {
      int hv2 = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv2++;
      if (hv2 >= 71) tail.push_back(&m);
    }
    if (tail.size() > 40) tail.resize(40);
    LanczosWork L;
    std::vector<int> hv;
    std::vector<double> d, e, z, wk;
    printf("  %6s %6s %14s %14s %14s %12s\n", "n", "k", "recur s_hi", "dsteqr s_hi",
           "ratio", "true relerr");
    int shown = 0;
    for (auto *m : tail) {
      if (shown >= 8) break;
      BurdenOp B;
      burden_build(*m, B, hv);
      burden_diag(m->mass, hv, B);
      double dlo, dhi;
      BCUT_SOLVER = 1;
      bcut_one(*m, m->mass, BW, &dhi, &dlo);
      lanczos_run(B, std::min(B.n, 260), -1, L);
      for (int kk : {40, 60, 80}) {
        if (kk > (int)L.alpha.size()) continue;
        d.assign(L.alpha.begin(), L.alpha.begin() + kk);
        e.assign(kk - 1, 0.0);
        for (int i = 0; i + 1 < kk; i++) e[i] = L.beta[i];
        z.assign((size_t)kk * kk, 0.0);
        wk.assign(2 * kk - 2 > 0 ? 2 * kk - 2 : 1, 0.0);
        char compz = 'I';
        int nn = kk, info = 0;
        dsteqr_(&compz, &nn, d.data(), e.data(), z.data(), &nn, wk.data(), &info);
        if (info != 0) { printf("  dsteqr info=%d\n", info); continue; }
        double exact_hi = std::fabs(z[(size_t)(kk - 1) * kk + (kk - 1)]);
        double rec_hi = tri_last_comp(L.alpha.data(), L.beta.data(), kk, d[kk - 1]);
        double relerr = std::fabs(d[kk - 1] - dhi) / std::max(std::fabs(dhi), 1e-30);
        printf("  %6d %6d %14.3e %14.3e %14.3e %12.2e\n", B.n, kk, rec_hi, exact_hi,
               exact_hi > 0 ? rec_hi / exact_hi : -1.0, relerr);
      }
      shown++;
    }
    return 0;
  }

  // DOES THE SECOND NUMERICAL PATH SURVIVE INTO THE SHIPPED ARTEFACT AT ALL?
  //
  // The objection to a Krylov path is that its values are within ~1e-12 of the dense ones but
  // not bit-identical to them. That objection has a premise: that the difference is OBSERVABLE.
  // The descriptor matrix this project ships is FLOAT32 -- every Python module returns float32,
  // which is why verify_hume.py holds them to rtol 3e-6 -- and float32 carries ~1.2e-7 relative
  // precision. A 1e-12 relative difference is five orders BELOW the representation it lands in,
  // so the two paths can only diverge in the shipped file when a value happens to straddle a
  // float32 rounding boundary. Expected rate ~ 1e-12 / 1.2e-7 ~ 1e-5.
  //
  // Both paths are computed for every molecule, cast to float32, and compared BIT FOR BIT.
  if (mode == "f32cmp") {
    const int thr = argc > 3 ? atoi(argv[3]) : 71;
    long long n_lan_mol = 0, n_fallback = 0, n_vals = 0, n_diff = 0;
    double worst64 = 0.0, worst32 = 0.0;
    LanczosWork LW;
    std::vector<int> hv;
    for (auto &m : ms) {
      double dense8[8], lan8[8];
      // FORCE the dense path for the reference. bcut2d honours BCUT_LANCZOS_MIN_N, so with the
      // hook enabled this would otherwise compare Lanczos against itself and report a
      // beautifully clean zero that means nothing.
      const int keep_thr = BCUT_LANCZOS_MIN_N;
      BCUT_LANCZOS_MIN_N = 0;
      bcut2d(m, BW, dense8);
      BCUT_LANCZOS_MIN_N = keep_thr;
      int nheavy = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
      if (nheavy < thr) continue;
      BurdenOp B;
      burden_build(m, B, hv);
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      bool ok = true;
      for (int k = 0; k < 4 && ok; k++) {
        burden_diag(*props[k], hv, B);
        ok = lanczos_extremal(B, LANCZOS_TOL, &lan8[2 * k + 1], &lan8[2 * k], LW);
      }
      if (!ok) { n_fallback++; continue; }
      n_lan_mol++;
      for (int i = 0; i < 8; i++) {
        n_vals++;
        const double rel64 = std::fabs(dense8[i] - lan8[i]) /
                             std::max(std::fabs(dense8[i]), 1e-30);
        if (rel64 > worst64) worst64 = rel64;
        const float fa = (float)dense8[i], fb = (float)lan8[i];
        unsigned ia, ib;
        std::memcpy(&ia, &fa, 4);
        std::memcpy(&ib, &fb, 4);
        if (ia != ib) {
          n_diff++;
          const double r32 = std::fabs((double)fa - (double)fb) /
                             std::max(std::fabs((double)fa), 1e-30);
          if (r32 > worst32) worst32 = r32;
        }
      }
    }
    printf("  threshold n >= %d\n", thr);
    printf("  molecules via Lanczos      : %lld   (fell back to dense: %lld)\n",
           n_lan_mol, n_fallback);
    printf("  BCUT2D values compared     : %lld   (whole corpus would be %zu)\n",
           n_vals, ms.size() * 8);
    printf("  float64 worst relative diff: %.3e\n", worst64);
    printf("  FLOAT32 BIT DIFFERENCES    : %lld of %lld  (%.3e of compared)\n",
           n_diff, n_vals, n_vals ? (double)n_diff / n_vals : 0.0);
    if (n_diff) printf("  worst float32 relative diff: %.3e  (one ulp is ~1.2e-7)\n", worst32);
    return 0;
  }

  // ADVERSARIAL AUDIT. Forces the Krylov attempt on EVERY molecule regardless of size, and
  // compares against dense in float64 and in the float32 the artefact actually ships in.
  // Reports fallbacks, so a class that refuses to certify shows up as a refusal rather than
  // as a wrong number.
  if (mode == "lanczosaudit") {
    long long tried = 0, fell = 0, nvals = 0, ndiff32 = 0;
    double worst64 = 0.0, worst32 = 0.0;
    std::string worst_smi;
    LanczosWork LW;
    std::vector<int> hv;
    size_t idx = 0;
    for (auto &m : ms) {
      idx++;
      double dense8[8], lan8[8];
      const int keep = BCUT_LANCZOS_MIN_N;
      BCUT_LANCZOS_MIN_N = 0;
      bcut2d(m, BW, dense8);
      BCUT_LANCZOS_MIN_N = keep;
      int nheavy = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
      if (nheavy < 3) continue;
      BurdenOp B;
      burden_build(m, B, hv);
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      tried++;
      bool ok = true;
      for (int k = 0; k < 4 && ok; k++) {
        burden_diag(*props[k], hv, B);
        ok = lanczos_extremal(B, LANCZOS_TOL, &lan8[2 * k + 1], &lan8[2 * k], LW);
      }
      if (!ok) {
        fell++;
        const char *why = LZ_LAST_REASON == 3 ? "MISSED-EXTREME"
                        : LZ_LAST_REASON == 2 ? "true-residual" : "no-convergence";
        printf("    row %3zu  n=%4d  FALLBACK (%s)\n", idx, nheavy, why);
        continue;
      }
      double mrel = 0.0;
      for (int i = 0; i < 8; i++) {
        nvals++;
        const double rel = std::fabs(dense8[i] - lan8[i]) /
                           std::max(std::fabs(dense8[i]), 1e-30);
        mrel = std::max(mrel, rel);
        if (rel > worst64) { worst64 = rel; worst_smi = std::to_string(idx); }
        const float fa = (float)dense8[i], fb = (float)lan8[i];
        unsigned ia, ib;
        std::memcpy(&ia, &fa, 4); std::memcpy(&ib, &fb, 4);
        if (ia != ib) {
          ndiff32++;
          const double r32 = std::fabs((double)fa - (double)fb) /
                             std::max(std::fabs((double)fa), 1e-30);
          if (r32 > worst32) worst32 = r32;
        }
      }
      printf("    row %3zu  n=%4d  certified, max rel %.2e\n", idx, nheavy, mrel);
    }
    printf("  molecules attempted    : %lld\n", tried);
    printf("  fell back to dense     : %lld  (%.2f%%)\n", fell,
           tried ? 100.0 * fell / tried : 0.0);
    printf("  values compared        : %lld\n", nvals);
    printf("  worst |Lanczos-dense|  : %.3e relative (float64), worst row #%s\n", worst64,
           worst_smi.c_str());
    printf("  FLOAT32 bit differences: %lld of %lld\n", ndiff32, nvals);
    if (ndiff32) printf("  worst float32 rel diff : %.3e (one ulp ~1.2e-7)\n", worst32);
    return 0;
  }

  // DOES THE MISSED-EXTREME GUARD ACTUALLY FIRE? The audit never tripped it, which proves
  // nothing on its own. So construct the pathology directly: find the true lambda_max
  // eigenvector by power iteration, project it out of the Lanczos start, and run. Because A is
  // symmetric and v_max is an eigenvector, the ENTIRE Krylov space then stays orthogonal to
  // v_max -- lambda_max is genuinely invisible to Lanczos, theta_max converges tightly onto
  // lambda_2, and both residual tests pass while the answer is wrong. Only an independent
  // probe can catch that. Anything reported ACCEPTED-WRONG here is a correctness bug.
  if (mode == "ghosttest") {
    extern const double *LZ_FORCE_START;
    LanczosWork LW;
    std::vector<int> hv;
    std::vector<double> v(1), w(1), st(1);
    int nacc_wrong = 0, nrefused = 0, nacc_ok = 0, ntested = 0;
    for (auto &m : ms) {
      int nheavy = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
      if (nheavy < 20) continue;
      BurdenOp B;
      burden_build(m, B, hv);
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      for (int p = 0; p < 4; p++) {
        burden_diag(*props[p], hv, B);
        double dlo, dhi;
        BCUT_SOLVER = 1;
        bcut_one(m, *props[p], BW, &dhi, &dlo);
        const int n = B.n;
        // power-iterate to the true lambda_max eigenvector
        v.assign(n, 0.0); w.assign(n, 0.0);
        for (int i = 0; i < n; i++) v[i] = std::sin(0.31 * i + 1.7) + 0.5;
        for (int it = 0; it < 3000; it++) {
          burden_mv(B, v.data(), w.data());
          double nn = 0.0;
          for (int i = 0; i < n; i++) nn += w[i] * w[i];
          nn = std::sqrt(nn);
          if (!(nn > 1e-300)) break;
          for (int i = 0; i < n; i++) v[i] = w[i] / nn;
        }
        double rq = 0.0;
        burden_mv(B, v.data(), w.data());
        for (int i = 0; i < n; i++) rq += v[i] * w[i];
        if (std::fabs(rq - dhi) > 1e-6 * std::max(std::fabs(dhi), 1.0)) continue;  // not converged
        // start := generic, with v_max projected out
        st.assign(n, 0.0);
        double dp = 0.0;
        for (int i = 0; i < n; i++) { st[i] = std::sin(1.2345 * i + 0.678 + 0.31 * B.dg[i]);
                                      dp += st[i] * v[i]; }
        for (int i = 0; i < n; i++) st[i] -= dp * v[i];
        LZ_FORCE_START = st.data();
        double llo, lhi;
        const bool ok = lanczos_extremal(B, LANCZOS_TOL, &llo, &lhi, LW);
        LZ_FORCE_START = nullptr;
        ntested++;
        if (!ok) { nrefused++; continue; }
        const double err = std::fabs(lhi - dhi) / std::max(std::fabs(dhi), 1e-30);
        if (err > 1e-9) {
          nacc_wrong++;
          printf("    ACCEPTED-WRONG n=%d prop=%d  lanczos %.10g  dense %.10g  rel %.2e\n",
                 n, p, lhi, dhi, err);
        } else {
          nacc_ok++;   // recovered lambda_max anyway (rounding reintroduced the direction)
        }
      }
    }
    printf("  adversarial starts tested : %d\n", ntested);
    printf("  REFUSED (guard fired)     : %d\n", nrefused);
    printf("  accepted, still correct   : %d\n", nacc_ok);
    printf("  ACCEPTED-WRONG            : %d   <-- must be zero\n", nacc_wrong);
    return 0;
  }

  if (mode == "verify") {
    FILE *out = fopen("values_hume.txt", "w");
    for (auto &m : ms) {
      distances(m, D);
      chi(m, dn, dv, cn, cv, cc);
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
      // COLUMN ORDER, and verify_hume.py's SPEC is generated from the same module NAMES lists
      // rather than retyped, so a column added to a Python block cannot silently shift the
      // comparison onto the wrong pair of numbers:
      //
      //   [2]   BalabanJ, BalabanJ_mordred
      //   [26]  chi.py         [31] cycles.py (minus C_sssr/C_redundancy, see below)
      //   [24]  conjugation.py [23] stereo.py   [60] resistance.py
      //   [16]  RDKit tail: EState x4, Kappa1-3, HallKierAlpha, BCUT2D x8
      //
      // Chi2n..Chi4v used to be emitted a SECOND time here, as a separate RDKit-gated prefix,
      // because chi.py's H-stripping made its chi2n..chi4v a genuinely different descriptor.
      // chi.py now reproduces RDKit exactly, so the two were bit-identical on all 98,905
      // molecules (checked, not assumed) and the duplicates are gone. The RDKit gate did not
      // go with them: verify_hume.py checks chi0n..chi4n and chi0v..chi4v of the chi.py block
      // against RDKit at rtol 1e-9, which is TEN columns under external reference where there
      // were six.
      fprintf(out, "%.12g %.12g", J, balaban_unweighted(m, D));
      for (int k = 0; k <= CHI_MAX; k++) fprintf(out, " %.12g", cn[k]);
      for (int k = 0; k <= CHI_MAX; k++) fprintf(out, " %.12g", cv[k]);
      double ptot = 0.0;
      for (int k = 1; k <= CHI_MAX; k++) { fprintf(out, " %.12g", cc[k]); ptot += cc[k]; }
      fprintf(out, " %.12g %.12g %.12g", ptot,
              cc[CHI_MAX] / std::max(cc[1], 1.0),
              cv[1] != 0.0 ? cn[1] / cv[1] : 0.0);
      for (int i = 0; i < 31; i++) fprintf(out, " %.12g", cy[i]);
      for (int i = 0; i < 24; i++) fprintf(out, " %.12g", cj[i]);
      for (int i = 0; i < 23; i++) fprintf(out, " %.12g", st[i]);
      for (int i = 0; i < 60; i++) fprintf(out, " %.12g", rs[i]);
      fprintf(out, " %.12g %.12g %.12g %.12g", emx, emn, eax, ean);
      for (int i = 0; i < 4; i++) fprintf(out, " %.12g", kp[i]);
      for (int i = 0; i < 8; i++) fprintf(out, " %.12g", bc[i]);
      fputc('\n', out);
    }
    fclose(out);
    fprintf(stderr, "wrote values_hume.txt\n");
    {
      extern long long BCUT_LANCZOS_TRIED, BCUT_LANCZOS_FELL_BACK, LZ_FAIL[4];
      extern double LZ_MAX_TRUE_OVER_PROXY, LZ_MAX_TRUE_OVER_TOL;
      fprintf(stderr,
              "  Krylov: tried %lld, fell back %lld (%.3f%%)  [no-conv %lld, true-resid %lld, "
              "missed-extreme %lld]\n  true residual / tol: max %.2e   true/proxy ratio "
              "(both > 1e-30): max %.2e\n",
              BCUT_LANCZOS_TRIED, BCUT_LANCZOS_FELL_BACK,
              BCUT_LANCZOS_TRIED ? 100.0 * BCUT_LANCZOS_FELL_BACK / BCUT_LANCZOS_TRIED : 0.0,
              LZ_FAIL[1], LZ_FAIL[2], LZ_FAIL[3], LZ_MAX_TRUE_OVER_TOL,
              LZ_MAX_TRUE_OVER_PROXY);
    }
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
        chi(m, dn, dv, cn, cv, cc);
        cycle_counts(m, cy);
        resistance(m, D, rs);
        conjugation(m, D, cj);
        stereo(m, D, st);
        estate_from(m, D, ES);
        kappa(m, kp);
        bcut2d(m, BW, bc);
        sink += balaban(m, WD) + cn[2] + cy[0] + rs[0] + cj[0] + st[0] + ES[0] + kp[0] + bc[0];
      }
    });
    printf("  %-44s %8.2f us/mol   <== the real pipeline cost\n",
           "ALL BLOCKS, shared distance matrices", t_all);
  }
  double t_d = time_it(ms, 20, [&] { for (auto &m : ms) { distances(m, D); sink += D[0]; } });
  double t_c = time_it(ms, 20, [&] {
    for (auto &m : ms) { chi(m, dn, dv, cn, cv, cc); sink += cn[2]; } });
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
  // STAGE SPLIT: how much of our BCUT2D is matrix assembly and how much is the eigensolve?
  // RDKit's own cost is dominated by removeAllHs + Gasteiger + Crippen + Eigen computing
  // eigenVECTORS it discards; ours excludes all of that (properties arrive precomputed, and we
  // pass jobz='N'), so the split here is a different question with a different answer.
  extern bool BCUT_SKIP_SOLVE;
  BCUT_SKIP_SOLVE = true;
  double t_asm = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  BCUT_SKIP_SOLVE = false;
  // SOLVER A/B IN ONE PROCESS. t_b above was timed with whatever BCUT_SOLVER is set to; time
  // the other one right here, back to back, so the comparison cannot be contaminated by machine
  // load drifting between two separate bench runs.
  extern int BCUT_SOLVER;
  int keep_solver = BCUT_SOLVER;
  BCUT_SOLVER = keep_solver ? 0 : 1;
  double t_b2 = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  BCUT_SOLVER = keep_solver;
  printf("  %-44s %8.2f us/mol\n", "EState (incl. distances)", t_e);
  printf("  %-44s %8.2f us/mol\n", "Kappa1-3 + HallKierAlpha", t_k);
  printf("  %-44s %8.2f us/mol\n",
         keep_solver ? "BCUT2D (dsytd2+dsterf, 4 spectra)" : "BCUT2D (dsyevd, 4 dense spectra)",
         t_b);
  printf("  %-44s %8.2f us/mol  (%.0f%% of BCUT2D)\n", "  ^ of which: matrix assembly only",
         t_asm, 100.0 * t_asm / t_b);
  printf("  %-44s %8.2f us/mol  (%.0f%%)\n", "  ^ of which: 4 eigensolves", t_b - t_asm,
         100.0 * (t_b - t_asm) / t_b);
  printf("  %-44s %8.2f us/mol  (%+.1f%% vs above)\n",
         keep_solver ? "  ALT solver: dsyevd" : "  ALT solver: dsytd2+dsterf",
         t_b2, 100.0 * (t_b2 - t_b) / t_b);
  // Stage probe: where inside the eigensolve does the time actually go? This decides whether
  // vectorising the four Householder reductions is worth building at all.
  BCUT_SOLVER = 2;
  double t_tri = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  BCUT_SOLVER = keep_solver;
  double solve_only = t_b - t_asm;
  printf("  %-44s %8.2f us/mol  (%.0f%% of the eigensolve)\n",
         "  ^ tridiagonalisation (dsytd2) only", t_tri - t_asm,
         100.0 * (t_tri - t_asm) / solve_only);
  printf("  %-44s %8.2f us/mol  (%.0f%%)\n", "  ^ QR sweep (dsterf) only",
         t_b - t_tri, 100.0 * (t_b - t_tri) / solve_only);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
