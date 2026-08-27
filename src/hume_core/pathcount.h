// mordred's PathCount family: the 11 columns that survive data/dedupe.json --
// MPC4, MPC6, MPC9 and piPC1..piPC6, piPC8, piPC10.
//
// SPECIFICATION IS mordred/PathCount.py at mordred 1.2.0. `PathCountCache.calculate()` is:
//
//     for path in Chem.FindAllPathsOfLengthN(self.mol, self._order):   # bond indices
//         reconstruct the atom sequence; if any atom REPEATS, drop the path entirely
//         else  l += 1;  pi *= product of GetBondTypeAsDouble() along the path
//
// and then `MPC_k = l`, `piPC_k = log(pi + 1)`.
//
// SO MPC_k IS THE NUMBER OF SIMPLE PATHS OF k BONDS, each counted once, and piPC_k is the log of
// one plus the sum over those same paths of the product of their bond orders. RDKit's path finder
// enumerates BOND-distinct walks, not atom-distinct ones -- its own docstring says "the current
// path finding code does, by design, detect rings as paths" -- so on cyclopropane
// FindAllPathsOfLengthN(mol, 3) returns the triangle 0-1-2-0, and mordred's `if i in aids: break`
// throws it away through the for/else. The enumeration below is atom-distinct from the start,
// which is the same set; verified column-for-column on 100,000 molecules.
//
// THE ONE THING THAT IS NOT IN THE MORDRED FILE AT ALL, AND THE ONLY REASON THIS PORT WAS WRONG
// ON ITS FIRST TRY:
//
//     Chem.FindAllPathsOfLengthN(mol, N)  TAKES useHs=False BY DEFAULT.
//
// mordred never passes it, so HYDROGEN ATOMS ARE INVISIBLE TO PathCount -- while RingCount and
// TopologicalCharge, sharing the same `Chem.RemoveHs(mol)` graph, see them. `Chem.RemoveHs` keeps
// isotopic hydrogen, so on cpp/hard.smi the graphs genuinely differ. The molecule that
// discriminates the two readings is
//
//     [2H]Cc1cc(C[2H])c2c(C(=O)OC)cn(C)cc1-2.[SiH3]
//
// on which counting the [2H] atoms puts MPC9 out by 50% and piPC10 by 8%. `build()` below drops
// every half-edge incident on a Z==1 atom for that reason, and for that reason only.
//
// HOUSE RULE 1: WELL-POSED AND BIT-STABLE. All 11 columns were re-evaluated with MORDRED on
// randomly renumbered copies and on canonical-SMILES round trips; not one moved on any molecule.
// They are also exactly reproducible in a way TopologicalCharge is not, and for a reason worth
// writing down: every path weight is a product of at most 10 factors drawn from
// GetBondTypeAsDouble(), i.e. from {0, 1, 1.5, 2, 2.5, 3, ...}, so every weight is an integer
// multiple of 2^-10 with a small integer part. Sums of such numbers stay exactly representable
// in float64 far beyond any count this corpus reaches, so `pi` is INDEPENDENT OF SUMMATION
// ORDER -- and the final log() is one libm call on a value both sides agree on bit-for-bit.
// The halving is exact for the same reason. This is why piPC is bit-exact and JGI is not.
//
// NO UPSTREAM TABLE IS INVOLVED -- no drift guard. The bond ORDERS are RDKit's own
// GetBondTypeAsDouble() arriving through the boundary's `bond_d` column, not a transcribed table
// of bond types; _extract.py already memoises them from RDKit's own answer for exactly this
// reason. The only constants here are the nine path orders and the column list.
//
// WHAT THE CALLER MUST SUPPLY: `Chem.RemoveHs(mol)`'s bonds with their GetBondTypeAsDouble(), and
// the atomic numbers (used only to find Z==1). Use build().
#ifndef HUME_PATHCOUNT_H
#define HUME_PATHCOUNT_H

#include <cmath>
#include <cstdint>
#include <vector>

namespace pathcount {

static constexpr int N_COLS = 11;
static constexpr int PMAX = 10;   // piPC10 is the deepest column that survives dedupe

// mordred's preset order, filtered to the dedupe survivors: the three raw counts first, then the
// eight log-pi columns. `pi` selects which of PathCountCache's two accumulators is read.
struct Spec { const char *name; int order; int pi; };
static const Spec COLS[N_COLS] = {
    {"MPC4", 4, 0},   {"MPC6", 6, 0},   {"MPC9", 9, 0},
    {"piPC1", 1, 1},  {"piPC2", 2, 1},  {"piPC3", 3, 1},  {"piPC4", 4, 1},
    {"piPC5", 5, 1},  {"piPC6", 6, 1},  {"piPC8", 8, 1},  {"piPC10", 10, 1},
};

struct Mol {
  int n = 0;
  std::vector<int32_t> start;   // n + 1
  std::vector<int32_t> nbr;     // CSR, hydrogen half-edges already removed
  std::vector<double> border;   // GetBondTypeAsDouble(), parallel to nbr
};

struct Scratch {
  std::vector<uint8_t> on;
  long long cnt[PMAX + 1];
  double w[PMAX + 1];
};

// CSR build that applies the useHs=False rule ONCE, here, so no caller can forget it.
// u/v/order are the bonds of Chem.RemoveHs(mol); z is GetAtomicNum() per atom.
inline void build(Mol &m, int n, int nb, const int32_t *u, const int32_t *v, const double *order,
                  const int32_t *z) {
  m.n = n;
  m.start.assign(n + 1, 0);
  std::vector<uint8_t> isH(n, 0);
  for (int i = 0; i < n; ++i) isH[i] = (z[i] == 1);
  for (int b = 0; b < nb; ++b) {
    if (isH[u[b]] || isH[v[b]]) continue;
    m.start[u[b] + 1]++;
    m.start[v[b] + 1]++;
  }
  for (int i = 0; i < n; ++i) m.start[i + 1] += m.start[i];
  m.nbr.assign(m.start[n], 0);
  m.border.assign(m.start[n], 0.0);
  std::vector<int32_t> cur(m.start.begin(), m.start.end() - 1);
  for (int b = 0; b < nb; ++b) {
    if (isH[u[b]] || isH[v[b]]) continue;
    m.nbr[cur[u[b]]] = v[b]; m.border[cur[u[b]]++] = order[b];
    m.nbr[cur[v[b]]] = u[b]; m.border[cur[v[b]]++] = order[b];
  }
}

// Same, straight off the boundary's strided rows, so no caller has to unpack `bond_i` into
// three parallel arrays and get a column index wrong. `brows` is bond_i with `bstride` columns
// and the endpoints at cu/cv; `arows` is atom_i with `astride` columns and Z at cz; `order` is
// the boundary's bond_d.
inline void build_from_rows(Mol &m, int n, int nb, const int32_t *brows, int bstride, int cu,
                            int cv, const double *order, const int32_t *arows, int astride,
                            int cz) {
  std::vector<int32_t> u(nb), v(nb), z(n);
  for (int b = 0; b < nb; ++b) {
    u[b] = brows[(ptrdiff_t)b * bstride + cu];
    v[b] = brows[(ptrdiff_t)b * bstride + cv];
  }
  for (int i = 0; i < n; ++i) z[i] = arows[(ptrdiff_t)i * astride + cz];
  build(m, n, nb, u.data(), v.data(), order, z.data());
}

static void dfs(const Mol &m, Scratch &S, int u, int depth, double prod) {
  for (int e = m.start[u]; e < m.start[u + 1]; ++e) {
    const int v = m.nbr[e];
    if (S.on[v]) continue;
    const double p = prod * m.border[e];
    S.cnt[depth] += 1;
    S.w[depth] += p;
    if (depth < PMAX) {
      S.on[v] = 1;
      dfs(m, S, v, depth + 1, p);
      S.on[v] = 0;
    }
  }
}

// out must have room for N_COLS doubles.
inline void compute(const Mol &m, double *out, Scratch &S) {
  for (int k = 0; k <= PMAX; ++k) { S.cnt[k] = 0; S.w[k] = 0.0; }
  S.on.assign(m.n, 0);
  for (int s = 0; s < m.n; ++s) {
    S.on[s] = 1;
    dfs(m, S, s, 1, 1.0);
    S.on[s] = 0;
  }
  // Every path is discovered from both of its ends, so both accumulators are halved. Halving is
  // exact in binary floating point, and cnt is even by construction.
  for (int c = 0; c < N_COLS; ++c) {
    const int k = COLS[c].order;
    out[c] = COLS[c].pi ? std::log(S.w[k] * 0.5 + 1.0) : (double)(S.cnt[k] / 2);
  }
}

}  // namespace pathcount

#endif  // HUME_PATHCOUNT_H
