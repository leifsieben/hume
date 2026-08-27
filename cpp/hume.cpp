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
// WHAT THIS OPENS UP. Only the two EXTREMAL eigenvalues are wanted and dsterf computes all n of
// them. The asymptotic case for an extremal-only Krylov method is strongest exactly where the
// time is -- n ~ 117, where the tail sits. The recorded Lanczos defeat (201 us against dsyevd's
// 94) was measured on the 3,000-molecule drug-like benchmark, which has no such tail, so it
// tested the regime that accounts for a few percent of the cost and never tested this one.
// Re-running that comparison against the 71+ bucket is the highest-value experiment left here.

bool BCUT_SKIP_SOLVE = false;

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

// -> 8 values in RDKit's order: MWHI, MWLOW, CHGHI, CHGLO, LOGPHI, LOGPLOW, MRHI, MRLOW
static void bcut2d(const Mol &m, BcutWork &W, double *out) {
  const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
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
