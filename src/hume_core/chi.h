// mordred's Chi family: the 40 columns that survive data/dedupe.json.
//
//     path          21   AXp-1d..7d, AXp-0dv..7dv, Xp-2d/4d/5d/6d, Xp-5dv/6dv
//     cluster        7   Xc-3d..6d, Xc-3dv..5dv
//     path-cluster   5   Xpc-4d/5d, Xpc-4dv/5dv/6dv
//     chain          7   Xch-4d..7d, Xch-5dv..7dv
//
// MORDRED'S Chi IS NOT RDKIT'S Chi AND THE TWO SHARE NO CODE. RDKit's `chi0n..chi4v` (already in
// src/hume_core/hume_blocks.h) are Kier-Hall connectivity indices over PATHS. mordred enumerates
// connected SUBGRAPHS of a fixed number of BONDS -- branched ones included -- and sorts each into
// one of four shapes. Nothing in hume_blocks.h computes the cluster, path-cluster or chain
// variants, and nothing here supersedes hume_blocks.h.
//
// SPECIFICATION IS mordred/Chi.py at 1.2.0. `ChiCache.calculate()` is:
//
//     for bonds in Chem.FindAllSubgraphsOfLengthN(self.mol, order):   # list of BOND indices
//         dfs.reset(bonds); typ = dfs(); nodes = dfs.nodes
//         <append `nodes` to one of four lists, by `typ`>
//
// and `Chi.calculate()` is then, for the list its own type selects,
//
//     x = 0.0
//     for nodes in node_sets:
//         c = 1
//         for node in nodes: c *= P[node]
//         if c <= 0: FAIL          -> the whole column is NaN, not just this term
//         x += c ** -0.5
//     if averaged: x /= len(node_sets)      -> ZeroDivisionError, i.e. NaN, when there are none
//
// ------------------------------------------------------------------------------------------
// FIVE THINGS THAT ARE NOT IN THE DOCUMENTATION AND EACH OF WHICH BREAKS EXACTNESS ALONE
// ------------------------------------------------------------------------------------------
//
// 1. `Chem.FindAllSubgraphsOfLengthN(mol, N)` TAKES useHs=False BY DEFAULT and mordred never
//    passes it -- the identical trap that made the first PathCount port wrong (see pathcount.h).
//    Bonds incident on a hydrogen ATOM are invisible to the enumeration. `Chem.RemoveHs` keeps
//    isotopic hydrogen, so on cpp/hard.smi the graph genuinely still contains some.
//
//    But the ORDER-0 case does NOT go through the enumeration: mordred builds it as
//    `[{a.GetIdx()} for a in self.mol.GetAtoms()]`, over EVERY atom, hydrogens included. A `[2H]`
//    has `dv == 0` (get_valence_electrons returns 0 for N==1), so `c <= 0` fires and **AXp-0dv is
//    NaN for every molecule carrying an explicit hydrogen** while every other Chi column ignores
//    it. Two different atom sets in one family; both are reproduced below.
//
// 2. THE SUM IS ORDER-SENSITIVE AND SO IS THE PRODUCT, so this port reproduces RDKit's
//    enumeration order bond-for-bond rather than just its set of subgraphs. `x += c ** -0.5` adds
//    irrational terms, so a permuted enumeration moves the last bits; and for the `dv` property
//    `c *= P[node]` multiplies non-dyadic rationals (chlorine's dv is (7-h)/9), so even the
//    PRODUCT depends on the node order. `enumerate()` below is a transliteration of RDKit's
//    `findAllSubgraphsOfLengthN` + `Subgraphs::recurseWalk` + `Subgraphs::getNbrsList`
//    (Code/GraphMol/Subgraphs/Subgraphs.cpp). The ORDERING RULES were pinned down first by
//    reimplementing them in Python and diffing against `Chem.FindAllSubgraphsOfLengthN` AS A LIST
//    OF LISTS -- order included, not as a set -- over 300 corpus molecules x orders {1,3,5,7}:
//    1,200 comparisons, 0 mismatches. The evidence for the C++ itself is that all 40 columns come
//    out bit-exact on 100,000 molecules, which a wrong enumeration order would not survive. Two
//    facts make the order reproducible from the boundary alone:
//
//      * `nbrs` is a `std::map<int, INT_VECT>`, so the start bonds are visited in ASCENDING BOND
//        INDEX. `bond_i`'s row order is RDKit's bond index order.
//      * an atom's out-edge list is in ascending bond index (boost `vecS`, bonds appended in
//        index order); verified on every atom of 300 corpus molecules. So `nbrs[b]` for
//        b = (p,q) is [heavy bonds of min(p,q) except b, ascending] ++ [same for max(p,q)].
//
//    `dfs.nodes` is `list(defaultdict.keys())`, i.e. first-appearance order over the subgraph's
//    bonds taken as (GetBeginAtomIdx, GetEndAtomIdx) pairs -- NOT sorted, so the boundary's
//    begin/end orientation is load-bearing too.
//
// 3. `c ** -0.5` IS NOT `1.0 / sqrt(c)`. numpy's float64 power is libm `pow`, and
//    `pow(c, -0.5)` differs from `1/sqrt(c)` in the last bit on 51,507 of the first 199,999
//    integers -- 26%. Every term below uses std::pow for that reason and for no other.
//
// 4. THE SHAPE TEST IS ORDER-INDEPENDENT EVEN THOUGH MORDRED'S CODE IS NOT. mordred classifies by
//    running a recursive DFS over `self.neighbors[u]`, which is a Python SET -- an iteration order
//    that is an implementation detail. It does not matter, and here is the proof, which is why
//    this port uses a closed form instead of transliterating the DFS:
//      * `degrees` ends up as the set of subgraph degrees over all nodes; every node is visited
//        because a subgraph from this enumeration is connected. Order-independent.
//      * `is_chain` is set exactly when the DFS meets an edge whose endpoint is already visited
//        and which is not already a tree edge -- i.e. exactly when the subgraph has a cycle. For
//        a CONNECTED subgraph with `order` edges that is just `order >= |nodes|`.
//    So: chain if order >= |nodes|; else path if degrees subset {1,2}; else path-cluster if 2 is
//    a degree; else cluster. Same four buckets, no set iteration.
//
// 5. THE FAILURE IS PER COLUMN, NOT PER TERM. `self.fail()` aborts the descriptor, so one
//    subgraph with a non-positive product makes that (shape, order, property) column NaN while
//    its siblings at the same order keep real values.
//
// ------------------------------------------------------------------------------------------
// HOUSE RULE 1
// ------------------------------------------------------------------------------------------
// These 40 columns are NOT invariant under renumbering, and the reason is arithmetic rather than
// structural: the MULTISET of terms is a graph invariant, but the ORDER in which mordred adds
// them is not, so the last bits move. That is the TopologicalCharge situation, not the
// InformationContent situation -- there is no second answer being chosen between, only a
// summation order. This port matches mordred bit-for-bit ON A GIVEN NUMBERING because it
// reproduces that order; the size of the wobble mordred shows against ITSELF under atom+bond
// shuffling is reported by `cpp/verify_chiwalk.py perm`. No divergence from mordred is taken.
//
// WHAT THE CALLER MUST SUPPLY: `Chem.RemoveHs(mol)`, as `atom_i` (Z, degree, nH, formal charge)
// and `bond_i` (u = GetBeginAtomIdx, v = GetEndAtomIdx) in RDKit's bond index order. Use
// build_from_rows().
#ifndef HUME_CHI_H
#define HUME_CHI_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "../../cpp/chiwalk_tables.h"

// THE NAMESPACE IS `chisub`, NOT `chi`, AND IT HAS TO BE: hume_blocks.h already defines a
// file-scope `static void chi(...)` -- RDKit's Kier-Hall chi0n..chi4v -- and a namespace of that
// name is a redefinition the moment bindings.cpp includes both. `sub` is for the SUBGRAPH
// enumeration, which is the whole difference between the two families.
namespace chisub {

static constexpr int N_COLS = 40;
static constexpr int OMAX = 7;    // Xch-7d / AXp-7dv are the deepest columns that survive dedupe

// mordred's ChiType, same integer values (they index the accumulators below).
enum Shape { CHAIN = 0, PATH = 1, PATH_CLUSTER = 2, CLUSTER = 3, N_SHAPE = 4 };
enum Prop { D = 0, DV = 1, N_PROP = 2 };

// The 40 survivors in mordred's own preset() order:
//     chain 3..7 x {d, dv}; cluster 3..6 x {d, dv}; path_cluster 4..6 x {d, dv};
//     path 0..7 x {d, dv} x {raw, averaged}
// filtered to data/dedupe.json. cpp/verify_chiwalk.py::check_spec() asserts that every name here
// is str() of a live mordred Chi built from the same (shape, order, prop, averaged) tuple.
struct Spec { const char *name; uint8_t shape, order, prop, averaged; };
static const Spec COLS[N_COLS] = {
    {"Xch-4d", CHAIN, 4, D, 0},   {"Xch-5d", CHAIN, 5, D, 0},
    {"Xch-6d", CHAIN, 6, D, 0},   {"Xch-7d", CHAIN, 7, D, 0},
    {"Xch-5dv", CHAIN, 5, DV, 0}, {"Xch-6dv", CHAIN, 6, DV, 0}, {"Xch-7dv", CHAIN, 7, DV, 0},

    {"Xc-3d", CLUSTER, 3, D, 0},   {"Xc-4d", CLUSTER, 4, D, 0},
    {"Xc-5d", CLUSTER, 5, D, 0},   {"Xc-6d", CLUSTER, 6, D, 0},
    {"Xc-3dv", CLUSTER, 3, DV, 0}, {"Xc-4dv", CLUSTER, 4, DV, 0}, {"Xc-5dv", CLUSTER, 5, DV, 0},

    {"Xpc-4d", PATH_CLUSTER, 4, D, 0},   {"Xpc-5d", PATH_CLUSTER, 5, D, 0},
    {"Xpc-4dv", PATH_CLUSTER, 4, DV, 0}, {"Xpc-5dv", PATH_CLUSTER, 5, DV, 0},
    {"Xpc-6dv", PATH_CLUSTER, 6, DV, 0},

    {"Xp-2d", PATH, 2, D, 0}, {"Xp-4d", PATH, 4, D, 0},
    {"Xp-5d", PATH, 5, D, 0}, {"Xp-6d", PATH, 6, D, 0},
    {"AXp-1d", PATH, 1, D, 1}, {"AXp-2d", PATH, 2, D, 1}, {"AXp-3d", PATH, 3, D, 1},
    {"AXp-4d", PATH, 4, D, 1}, {"AXp-5d", PATH, 5, D, 1}, {"AXp-6d", PATH, 6, D, 1},
    {"AXp-7d", PATH, 7, D, 1},

    {"Xp-5dv", PATH, 5, DV, 0}, {"Xp-6dv", PATH, 6, DV, 0},
    {"AXp-0dv", PATH, 0, DV, 1}, {"AXp-1dv", PATH, 1, DV, 1}, {"AXp-2dv", PATH, 2, DV, 1},
    {"AXp-3dv", PATH, 3, DV, 1}, {"AXp-4dv", PATH, 4, DV, 1}, {"AXp-5dv", PATH, 5, DV, 1},
    {"AXp-6dv", PATH, 6, DV, 1}, {"AXp-7dv", PATH, 7, DV, 1},
};

struct Mol {
  int n = 0;
  std::vector<int32_t> z;       // GetAtomicNum()
  std::vector<double> p[N_PROP];  // [D] sigma electrons, [DV] valence electrons
  std::vector<int32_t> bu, bv;  // GetBeginAtomIdx() / GetEndAtomIdx(), RDKit bond index order
};

// mordred/_atomic_property.py::get_valence_electrons. NOTE that the formal charge is subtracted
// from BOTH the outer-electron count and the atomic number -- it cancels out of the numerator's
// `Zv` only, not out of the denominator. That is upstream's arithmetic, reproduced verbatim.
// `nh` is `GetTotalNumHs()` (the property, not the neighbours) and `he` is the number of
// NEIGHBOURING hydrogen ATOMS; mordred adds them, so an explicit [2H] counts twice only if RDKit
// reports it in both, which it does not.
inline double valenceElectrons(int z, int fchg, int nh, int he) {
  if (z == 1) return 0.0;
  const int Zv = chiwalk_tables::nOuterElecs(z) - fchg;
  const int Z = z - fchg;
  const int denom = Z - Zv - 1;
  if (denom == 0) return std::numeric_limits<double>::quiet_NaN();  // unreachable for z > 1
  return (double)(Zv - nh - he) / (double)denom;
}

struct Scratch {
  // The `nbrs` CSR of RDKit's getNbrsList, over HEAVY-HEAVY bonds only.
  std::vector<int32_t> nb_start, nb_list;
  std::vector<int32_t> hb;          // compact index -> original bond index
  std::vector<uint8_t> forbidden;
  std::vector<int32_t> undo;
  std::vector<int32_t> cands[OMAX + 2];
  int32_t path[OMAX + 1];
  // The current subgraph's node set, maintained INCREMENTALLY down the recursion rather than
  // rebuilt per subgraph. `nodes` is in mordred's first-appearance order by construction -- a
  // bond is appended to the path and its endpoints are appended here at the same moment -- so
  // `prod` is the identical sequence of float64 multiplications the batch form would perform,
  // which is what keeps the `dv` products bit-identical (see note 2). `ndeg2` / `ndeg3` are the
  // two facts the shape test needs, kept as counters so the test is O(1).
  int32_t nodes[OMAX + 2];
  int32_t deg[OMAX + 2];
  int nn = 0, ndeg2 = 0, ndeg3 = 0;
  double prod[N_PROP];
  // accumulators, indexed [prop][shape][order]
  double sum[N_PROP][N_SHAPE][OMAX + 1];
  int64_t cnt[N_SHAPE][OMAX + 1];
  uint8_t bad[N_PROP][N_SHAPE][OMAX + 1];   // a non-positive product was seen -> column is NaN
  // a tiny memo for pow(k, -0.5) on integer products, which is what the `d` property always
  // produces. Purely a speed device: every hit returns the identical double std::pow would.
  static constexpr int MEMO = 1 << 16;
  std::vector<double> memo;
};

inline void build_from_rows(Mol &m, int n, int nb, const int32_t *arows, int astride,
                            const int32_t *brows, int bstride) {
  m.n = n;
  m.z.resize(n);
  m.p[D].assign(n, 0.0);
  m.p[DV].resize(n);
  m.bu.resize(nb);
  m.bv.resize(nb);
  for (int i = 0; i < n; ++i) m.z[i] = arows[(size_t)i * astride + 0];
  for (int b = 0; b < nb; ++b) {
    m.bu[b] = brows[(size_t)b * bstride + 0];
    m.bv[b] = brows[(size_t)b * bstride + 1];
  }
  // `d` = get_sigma_electrons = neighbours with atomic number != 1, counted over the graph as
  // given. NOT `atom_i`'s GetDegree() column, which also counts explicit hydrogen neighbours.
  std::vector<int32_t> he(n, 0);
  for (int b = 0; b < nb; ++b) {
    const int u = m.bu[b], v = m.bv[b];
    if (m.z[v] != 1) m.p[D][u] += 1.0; else he[u] += 1;
    if (m.z[u] != 1) m.p[D][v] += 1.0; else he[v] += 1;
  }
  for (int i = 0; i < n; ++i)
    m.p[DV][i] = valenceElectrons(m.z[i], arows[(size_t)i * astride + 3],
                                  arows[(size_t)i * astride + 2], he[i]);
}

namespace detail {

// RDKit Subgraphs::getNbrsList(mol, useHs=false, nbrs), rewritten as a CSR keyed by a COMPACT
// index that preserves ascending original bond index -- which is what makes the std::map<int,...>
// iteration order of findAllSubgraphsOfLengthN reproducible without a map.
inline void buildNbrs(const Mol &m, Scratch &S) {
  const int nb = (int)m.bu.size();
  S.hb.clear();
  std::vector<int32_t> cidx(nb, -1);
  for (int b = 0; b < nb; ++b)
    if (m.z[m.bu[b]] != 1 && m.z[m.bv[b]] != 1) { cidx[b] = (int32_t)S.hb.size(); S.hb.push_back(b); }
  const int M = (int)S.hb.size();

  // incident heavy-heavy bonds per atom, ascending original bond index (the boost out-edge order)
  std::vector<int32_t> astart(m.n + 1, 0), alist;
  for (int b = 0; b < nb; ++b)
    if (cidx[b] >= 0) { astart[m.bu[b] + 1]++; astart[m.bv[b] + 1]++; }
  for (int i = 0; i < m.n; ++i) astart[i + 1] += astart[i];
  alist.resize(astart[m.n]);
  {
    std::vector<int32_t> fill(astart.begin(), astart.end() - 1);
    for (int b = 0; b < nb; ++b)
      if (cidx[b] >= 0) { alist[fill[m.bu[b]]++] = cidx[b]; alist[fill[m.bv[b]]++] = cidx[b]; }
  }

  S.nb_start.assign(M + 1, 0);
  for (int c = 0; c < M; ++c) {
    const int b = S.hb[c];
    const int p = m.bu[b] < m.bv[b] ? m.bu[b] : m.bv[b];
    const int q = m.bu[b] < m.bv[b] ? m.bv[b] : m.bu[b];
    S.nb_start[c + 1] = (astart[p + 1] - astart[p] - 1) + (astart[q + 1] - astart[q] - 1);
  }
  for (int c = 0; c < M; ++c) S.nb_start[c + 1] += S.nb_start[c];
  S.nb_list.resize(S.nb_start[M]);
  for (int c = 0; c < M; ++c) {
    const int b = S.hb[c];
    const int p = m.bu[b] < m.bv[b] ? m.bu[b] : m.bv[b];
    const int q = m.bu[b] < m.bv[b] ? m.bv[b] : m.bu[b];
    int w = S.nb_start[c];
    for (int k = astart[p]; k < astart[p + 1]; ++k) if (alist[k] != c) S.nb_list[w++] = alist[k];
    for (int k = astart[q]; k < astart[q + 1]; ++k) if (alist[k] != c) S.nb_list[w++] = alist[k];
  }
  S.forbidden.assign(M, 0);
}

inline double powNegHalf(double c, Scratch &S) {
  // Integer products (always, for the `d` property) come out of a memo; everything else calls
  // libm. The memo stores exactly what std::pow returns, so this changes speed and not values.
  if (c > 0 && c < (double)Scratch::MEMO && c == (double)(int64_t)c) {
    const int k = (int)c;
    double v = S.memo[k];
    if (v == 0.0) { v = std::pow(c, -0.5); S.memo[k] = v; }
    return v;
  }
  return std::pow(c, -0.5);
}

// Append one bond's endpoints to the running node set. The two properties are multiplied in
// exactly at the moment a node first appears, which IS mordred's `for node in dfs.nodes:
// c *= P[node]` order -- `dfs.nodes` is a defaultdict's key order over the same (begin, end)
// pairs in the same bond order.
inline void addBond(const Mol &m, Scratch &S, int b) {
  const int ends[2] = {m.bu[b], m.bv[b]};
  for (int e = 0; e < 2; ++e) {
    int j = 0;
    for (; j < S.nn; ++j) if (S.nodes[j] == ends[e]) break;
    if (j == S.nn) {
      S.nodes[S.nn] = ends[e];
      S.deg[S.nn] = 1;
      ++S.nn;
      S.prod[D] *= m.p[D][ends[e]];
      S.prod[DV] *= m.p[DV][ends[e]];
    } else {
      const int d = ++S.deg[j];
      if (d == 2) ++S.ndeg2;
      else if (d == 3) { --S.ndeg2; ++S.ndeg3; }
    }
  }
}

// Classify the current subgraph (`order` bonds) and accumulate. See note 4 for why the four
// buckets are a closed form rather than mordred's set-iterating DFS.
inline void emit(Scratch &S, int order) {
  int shape;
  if (order >= S.nn) {
    shape = CHAIN;                       // connected, `order` edges, nn vertices -> has a cycle
  } else {
    shape = S.ndeg3 == 0 ? PATH : (S.ndeg2 > 0 ? PATH_CLUSTER : CLUSTER);
  }
  S.cnt[shape][order] += 1;
  for (int pr = 0; pr < N_PROP; ++pr) {
    const double c = S.prod[pr];
    if (c <= 0) { S.bad[pr][shape][order] = 1; continue; }
    S.sum[pr][shape][order] += powNegHalf(c, S);
  }
}

// RDKit Subgraphs::recurseWalkRange -- the MtoN form, used here in place of seven separate
// recurseWalk(targetLen = 1..7) traversals.
//
// THAT SUBSTITUTION IS EXACT, NOT AN APPROXIMATION, and RDKit's own source is the argument: the
// two functions differ only in WHEN they record (`recurseWalk` at depth == targetLen,
// `recurseWalkRange` on entry at every depth in range) and in where they stop. The `forbidden`
// evolution at depth d does not depend on the pruning depth at all, because every bit a child
// sets is undone when the child returns -- upstream by passing the bitset by value, here by the
// undo log. So the depth-k entries of one depth-7 traversal are the same subgraphs, in the same
// pre-order sequence, as findAllSubgraphsOfLengthN(k) produces. mordred calls the LengthN form
// seven times; we call the range form once and slice it, and the 100,000-molecule bit-exactness
// is what closes the argument. Costs about a third less than seven passes.
//
// `forbidden` is passed BY VALUE upstream, so bits set at this level must survive the rest of
// this level's while-loop but must NOT survive the return to the caller -- which is what the undo
// log implements. Getting that backwards silently drops subgraphs.
inline void recurseWalk(const Mol &m, Scratch &S, int depth) {
  emit(S, depth);                          // recurseWalkRange records on ENTRY, at every depth
  if (depth == OMAX) return;
  std::vector<int32_t> &cands = S.cands[depth];
  while (!cands.empty()) {
    const int next = cands.back();
    cands.pop_back();
    if (S.forbidden[next]) continue;
    S.forbidden[next] = 1;
    S.undo.push_back(next);              // undone by the CALLER, not by this level
    std::vector<int32_t> &ts = S.cands[depth + 1];
    ts.assign(cands.begin(), cands.end());
    for (int k = S.nb_start[next]; k < S.nb_start[next + 1]; ++k)
      if (!S.forbidden[S.nb_list[k]]) ts.push_back(S.nb_list[k]);
    S.path[depth] = next;
    // The node set is at most OMAX+1 entries, so its undo is a memcpy of a handful of ints on
    // this frame rather than a stack of edits. `nodes` beyond `nn` needs no restoring: it is
    // append-only and anything above `nn` is dead.
    const int save_nn = S.nn, save_d2 = S.ndeg2, save_d3 = S.ndeg3;
    const double save_p0 = S.prod[D], save_p1 = S.prod[DV];
    int32_t save_deg[OMAX + 2];
    for (int j = 0; j < save_nn; ++j) save_deg[j] = S.deg[j];
    addBond(m, S, S.hb[next]);
    const size_t mark = S.undo.size();
    recurseWalk(m, S, depth + 1);
    while (S.undo.size() > mark) { S.forbidden[S.undo.back()] = 0; S.undo.pop_back(); }
    S.nn = save_nn; S.ndeg2 = save_d2; S.ndeg3 = save_d3;
    S.prod[D] = save_p0; S.prod[DV] = save_p1;
    for (int j = 0; j < save_nn; ++j) S.deg[j] = save_deg[j];
  }
}

// RDKit findAllSubgraphsOfLengthsMtoN(mol, 1, OMAX, useHs=false). Start bonds in ascending index;
// each start bond is forbidden for every LATER start bond and that is never undone.
inline void enumerate(const Mol &m, Scratch &S) {
  const int M = (int)S.hb.size();
  std::fill(S.forbidden.begin(), S.forbidden.end(), (uint8_t)0);
  S.undo.clear();
  for (int c = 0; c < M; ++c) {
    if (S.forbidden[c]) continue;
    S.forbidden[c] = 1;
    S.path[0] = c;
    S.nn = 0; S.ndeg2 = 0; S.ndeg3 = 0;
    S.prod[D] = 1.0; S.prod[DV] = 1.0;    // mordred starts `c` at the int 1
    detail::addBond(m, S, S.hb[c]);
    std::vector<int32_t> &cs = S.cands[1];
    cs.assign(S.nb_list.begin() + S.nb_start[c], S.nb_list.begin() + S.nb_start[c + 1]);
    recurseWalk(m, S, 1);
    while (!S.undo.empty()) { S.forbidden[S.undo.back()] = 0; S.undo.pop_back(); }
  }
}

}  // namespace detail

inline void compute(const Mol &m, double *out, Scratch &S) {
  const double NANV = std::numeric_limits<double>::quiet_NaN();
  if (S.memo.size() != (size_t)Scratch::MEMO) S.memo.assign(Scratch::MEMO, 0.0);
  for (int pr = 0; pr < N_PROP; ++pr)
    for (int s = 0; s < N_SHAPE; ++s)
      for (int o = 0; o <= OMAX; ++o) { S.sum[pr][s][o] = 0.0; S.bad[pr][s][o] = 0; }
  for (int s = 0; s < N_SHAPE; ++s)
    for (int o = 0; o <= OMAX; ++o) S.cnt[s][o] = 0;

  // ORDER 0 IS A DIFFERENT ATOM SET. mordred bypasses the enumeration entirely and uses every
  // atom of the molecule, hydrogens included -- see note 1 at the top.
  S.cnt[PATH][0] = m.n;
  for (int pr = 0; pr < N_PROP; ++pr)
    for (int i = 0; i < m.n; ++i) {
      const double c = m.p[pr][i];
      if (c <= 0) { S.bad[pr][PATH][0] = 1; continue; }
      S.sum[pr][PATH][0] += std::pow(c, -0.5);
    }

  detail::buildNbrs(m, S);
  detail::enumerate(m, S);

  for (int col = 0; col < N_COLS; ++col) {
    const Spec &sp = COLS[col];
    if (S.bad[sp.prop][sp.shape][sp.order]) { out[col] = NANV; continue; }
    double x = S.sum[sp.prop][sp.shape][sp.order];
    if (sp.averaged) {
      const int64_t k = S.cnt[sp.shape][sp.order];
      out[col] = k == 0 ? NANV : x / (double)k;   // mordred: ZeroDivisionError -> Error -> NaN
    } else {
      out[col] = x;
    }
  }
}

}  // namespace chisub

#endif  // HUME_CHI_H
