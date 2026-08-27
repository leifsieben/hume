// A SMARTS substructure matcher, sized to the `rdkit_core` fragment patterns and no larger.
//
// WHY A REAL MATCHER, when crippen_typer.h and estate_typer.h deliberately avoid one.  Those two
// decode a fixed pattern set into predicates over (element, aromaticity, degree, H-count, charge)
// plus a one-step neighbour query, because every E-state and Crippen pattern is one central atom
// with a flat bracket expression and at most a branch.  The fragment patterns are not that.
// Measured from RDKit's own parse trees by cpp/gen_frag_program.py:
//
//     * 17 of the 74 patterns have a CYCLIC query graph -- `c1ccccc1`, `n1ccccc1`, `O1CC1`,
//       `[c&R2]12[c&R1]...`.  A ring closure cannot be checked by walking outward from a centre;
//       it is subgraph isomorphism.
//     * 21 patterns use RECURSIVE SMARTS, nested to depth 3, 102 occurrences over 76 distinct
//       sub-queries.  `fr_phos_acid` alone has 20.
//     * the largest single (sub)pattern is 18 query atoms.
//     * 19 distinct query primitives appear, including ring-membership counts (`[R1]`, `[R2]`)
//       and bond-in-ring (`-!@`), which neither existing typer has any notion of.
//
// So this file implements the general thing: a query-tree evaluator over the 20 opcodes in
// cpp/frag_program.h, plus backtracking subgraph isomorphism, plus memoised recursive queries.
//
// WHAT "COUNT" MEANS, and it is the one thing most likely to be got wrong.  RDKit counts every
// one of these patterns as
//
//     len(mol.GetSubstructMatches(patt, uniquify=True))
//
// which is the number of distinct ATOM SETS, not the number of embeddings.  Two different
// mappings of the query onto the same set of molecule atoms count once.  `matchCount()` below
// therefore collects sorted atom-index tuples into a set.  Measured on cpp/hard.smi: for all 74
// patterns the uniquify=True count equals the number of distinct sorted match tuples, so
// uniquification here is well-posed and cannot depend on search order.
//
// AND THE TRAP THAT ISN'T SPRUNG, recorded because it would be near-impossible to spot later.
// `GetSubstructMatches` defaults to `maxMatches=1000`, and TRUNCATION HAPPENS BEFORE
// UNIQUIFICATION.  If a pattern ever exceeded 1000 raw embeddings on some molecule, RDKit's
// count would silently become a function of which embeddings the search found first -- i.e.
// order-dependent, i.e. not a function of the molecule at all.  Measured: the largest raw
// (uniquify=False) embedding count over every pattern and every one of the 100,000 molecules in
// cpp/hard.smi is 180.  There is a 5.5x margin.  `matchCount()` asserts if it ever sees 1000, so
// that the day a corpus does spring this trap it fails loudly instead of quietly disagreeing.
//
// WHAT THE CALLER MUST SUPPLY.  Every column is at the boundary as of the (n_atoms, 10) `atom_i`
// / (n_bonds, 5) `bond_i` layout -- `tval` is the tenth, added for this family:
//
//   z      A_Z      GetAtomicNum()
//   deg    A_DEG    GetDegree()          -- SMARTS `D`
//   nH     A_NH     GetTotalNumHs(False)
//   fchg   A_FCHG   GetFormalCharge()
//   arom   A_AROM   GetIsAromatic()
//   nring  A_NRING  RingInfo::NumAtomRings(i)   -- SMARTS `R<n>`; the BOOLEAN A_RING cannot
//                                                  answer `[R1]` vs `[R2]`
//   tval   A_TVAL   GetTotalValence()    -- SMARTS `v`; see below
//   bond   B_U/B_V/B_RING/B_CODE
//
// `border` is NOT a boundary column and must not become one: it is RDKit's Bond::BondType integer
// (AROMATIC = 12), and `esttyper::btypeFromBcode()` already turns `bond_i`'s B_CODE into exactly
// that number -- verified equal on all 3,090,892 bonds of cpp/hard.smi.  One converter, not two.
//
// and two are DERIVED here rather than carried, because they are exact functions of the above
// and a second source would be a second thing to keep in step.  Both verified over 575,571 atoms
// of cpp/hard.smi against RDKit, 0 mismatches:
//
//   X (SMARTS total degree) = deg + nH                 == GetTotalDegree()
//   H (SMARTS H count)      = nH + #neighbours with z==1   == GetTotalNumHs(True)
//
// THE ONE COLUMN THAT IS NOT DERIVABLE is SMARTS `v`, total valence, and it is why the boundary
// grew a tenth `atom_i` column rather than this file growing a reconstruction.  The obvious one
// -- round(sum of incident bond orders) + nH -- is WRONG on 11,238 of those same 575,571 atoms,
// because RDKit folds aromatic bond contributions and hydrogens together under its own rounding
// rule.  Pyrrole's `[nH]`: two aromatic bonds sum to 3.0 and it carries one H, and RDKit's total
// valence is 3, not 4.  `v` is used by `fr_Imine` (`[Nv3]`), `NumHDonors` (`v3`, `v4`) and
// `NumHAcceptors` (`v2`, `v3`), so it cannot be dropped either.  It reaches the boundary as
// `Atom.GetTotalValence()` on the reference path and as the pickle's own explicit + implicit
// valence on the fast path; the two agree column-wise on both corpora (cpp/verify_molpickle.py).
//
// SMARTS BOND SEMANTICS, same trap cpp/estate_tables.h documents.  `:` is BondOrder 12
// (Bond::AROMATIC), a bond TYPE query, not a getIsAromatic() query; and the default bond written
// between two atoms is `SingleOrAromaticBond`, which is a third thing again.  On this corpus the
// two questions never disagree -- bondType==AROMATIC matched GetIsAromatic() on 155,276 of
// 155,276 bonds -- but they are different questions and are kept distinct here.
#ifndef HUME_FRAG_MATCHER_H
#define HUME_FRAG_MATCHER_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../cpp/frag_program.h"

namespace fragmatch {

using frag_prog::NODES;
using frag_prog::AROOTS;
using frag_prog::QBONDS;
using frag_prog::PATTERNS;
using frag_prog::Node;

// Bond order codes, as RDKit's Bond::BondType numbers them.  AROMATIC is 12; that is not a
// typo and not a bitmask.
enum : int { BO_SINGLE = 1, BO_DOUBLE = 2, BO_TRIPLE = 3, BO_AROMATIC = 12 };

struct Mol {
  int n = 0, nb = 0;
  std::vector<int> z, deg, nH, fchg, arom, nring, tval;
  std::vector<int> hcount, tdeg;                  // derived: SMARTS H and X
  std::vector<int> bu, bv, border, bring;
  std::vector<int> start, nbr, nbond;             // CSR adjacency + parallel bond index

  void alloc(int na, int nbonds) {
    n = na; nb = nbonds;
    z.assign(na, 0); deg.assign(na, 0); nH.assign(na, 0); fchg.assign(na, 0);
    arom.assign(na, 0); nring.assign(na, 0); tval.assign(na, 0);
    hcount.assign(na, 0); tdeg.assign(na, 0);
    bu.assign(nbonds, 0); bv.assign(nbonds, 0); border.assign(nbonds, 0); bring.assign(nbonds, 0);
  }

  // Build CSR from the bond list, then derive the two SMARTS quantities that are exact
  // functions of the boundary columns (see the header comment for the evidence).
  void finish() {
    std::vector<int> cnt(n + 1, 0);
    for (int e = 0; e < nb; ++e) { cnt[bu[e]]++; cnt[bv[e]]++; }
    start.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) start[i + 1] = start[i] + cnt[i];
    nbr.assign(start[n], 0); nbond.assign(start[n], 0);
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (int e = 0; e < nb; ++e) {
      nbr[fill[bu[e]]] = bv[e]; nbond[fill[bu[e]]++] = e;
      nbr[fill[bv[e]]] = bu[e]; nbond[fill[bv[e]]++] = e;
    }
    for (int i = 0; i < n; ++i) {
      int hn = 0;
      for (int k = start[i]; k < start[i + 1]; ++k) if (z[nbr[k]] == 1) ++hn;
      hcount[i] = nH[i] + hn;      // GetTotalNumHs(True)
      tdeg[i] = deg[i] + nH[i];    // GetTotalDegree()
    }
  }

  int bondBetween(int a, int b) const {
    for (int k = start[a]; k < start[a + 1]; ++k) if (nbr[k] == b) return nbond[k];
    return -1;
  }
};

// ---------------------------------------------------------------------------------------------
// Query evaluation.
//
// Recursive queries are memoised per (pattern, atom) in a byte cache: 0 unknown, 1 yes, 2 no.
// Without it `fr_phos_acid`, whose 20 recursive sub-queries sit under an OR that is retried at
// every candidate atom, re-runs the same sub-search thousands of times per molecule.
// ---------------------------------------------------------------------------------------------
class Matcher {
 public:
  Matcher() : mol_(0) {}
  explicit Matcher(const Mol& m) : mol_(0) { bind(m); }

  // Point at a new molecule and clear the recursive-query cache.  Exists so a batch loop can
  // hold ONE Matcher and pay for the cache's storage once: `assign` reuses the vector's capacity,
  // where a fresh Matcher per molecule mallocs N_PATTERNS * n bytes every time.  The cache is
  // cleared, not merely resized -- a stale `yes` from the previous molecule would be a wrong
  // count with no symptom.
  void bind(const Mol& m) {
    mol_ = &m;
    cache_.assign((size_t)frag_prog::N_PATTERNS * (size_t)m.n, 0);
  }

  int matchCount(int pat) {
    std::vector<std::vector<int> > out;
    enumerate(pat, /*firstOnly=*/false, /*anchor=*/-1, out);
    // uniquify=True: distinct sorted ATOM SETS, not distinct embeddings.
    for (size_t i = 0; i < out.size(); ++i) std::sort(out[i].begin(), out[i].end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return (int)out.size();
  }

  bool atomMatchesPattern(int pat, int atom) {
    std::vector<std::vector<int> > out;
    return enumerate(pat, /*firstOnly=*/true, /*anchor=*/atom, out);
  }

 private:
  const Mol* mol_;
  std::vector<uint8_t> cache_;

  // -- atom query tree --------------------------------------------------------------------
  bool evalAtom(int node, int a) {
    const Node& q = NODES[node];
    bool r;
    switch (q.op) {
      case frag_prog::OP_ATOMAND: r = evalAtom(q.lhs, a) && evalAtom(q.rhs, a); break;
      case frag_prog::OP_ATOMOR:  r = evalAtom(q.lhs, a) || evalAtom(q.rhs, a); break;
      case frag_prog::OP_ATOMNULL: r = true; break;
      // AtomType fuses element and aromaticity: 1000+z is aromatic z, plain z is aliphatic z.
      case frag_prog::OP_ATOMTYPE:
        r = (mol_->z[a] + 1000 * (mol_->arom[a] ? 1 : 0)) == q.val; break;
      // AtomAtomicNum (`[#7]`) constrains the element and says NOTHING about aromaticity.
      case frag_prog::OP_ATOMATOMICNUM: r = mol_->z[a] == q.val; break;
      case frag_prog::OP_ATOMEXPLICITDEGREE: r = mol_->deg[a] == q.val; break;
      case frag_prog::OP_ATOMTOTALDEGREE: r = mol_->tdeg[a] == q.val; break;
      case frag_prog::OP_ATOMHCOUNT: r = mol_->hcount[a] == q.val; break;
      case frag_prog::OP_ATOMFORMALCHARGE: r = mol_->fchg[a] == q.val; break;
      // -1 is RDKit's sentinel for `[R]` == "in at least one ring", NOT a ring count of -1.
      case frag_prog::OP_ATOMINNRINGS:
        r = (q.val < 0) ? (mol_->nring[a] != 0) : (mol_->nring[a] == q.val); break;
      case frag_prog::OP_ATOMTOTALVALENCE: r = mol_->tval[a] == q.val; break;
      case frag_prog::OP_ATOMISAROMATIC: r = (mol_->arom[a] ? 1 : 0) == q.val; break;
      case frag_prog::OP_ATOMISALIPHATIC: r = (mol_->arom[a] ? 0 : 1) == q.val; break;
      case frag_prog::OP_RECURSIVESTRUCTURE: r = recursive(q.val, a); break;
      default:
        std::fprintf(stderr, "fragmatch: atom opcode %d unimplemented\n", (int)q.op);
        std::abort();
    }
    return q.neg ? !r : r;
  }

  // -- bond query tree --------------------------------------------------------------------
  bool evalBond(int node, int e) {
    const Node& q = NODES[node];
    bool r;
    switch (q.op) {
      case frag_prog::OP_BONDAND: r = evalBond(q.lhs, e) && evalBond(q.rhs, e); break;
      case frag_prog::OP_BONDOR:  r = evalBond(q.lhs, e) || evalBond(q.rhs, e); break;
      case frag_prog::OP_BONDNULL: r = true; break;
      case frag_prog::OP_BONDORDER: r = mol_->border[e] == q.val; break;
      case frag_prog::OP_BONDINRING: r = (mol_->bring[e] ? 1 : 0) == q.val; break;
      // RDKit's queryBondIsSingleOrAromatic, on the bond TYPE.  This is the default bond
      // written between two SMARTS atoms and is a different query from `-` and from `:`.
      case frag_prog::OP_SINGLEORAROMATICBOND:
        r = (mol_->border[e] == BO_SINGLE || mol_->border[e] == BO_AROMATIC); break;
      default:
        std::fprintf(stderr, "fragmatch: bond opcode %d unimplemented\n", (int)q.op);
        std::abort();
    }
    return q.neg ? !r : r;
  }

  bool recursive(int pat, int a) {
    size_t key = (size_t)pat * (size_t)mol_->n + (size_t)a;
    uint8_t& c = cache_[key];
    if (c) return c == 1;
    std::vector<std::vector<int> > dummy;
    bool ok = enumerate(pat, /*firstOnly=*/true, /*anchor=*/a, dummy);
    c = ok ? 1 : 2;
    return ok;
  }

  // -- subgraph isomorphism ---------------------------------------------------------------
  // Query atoms are visited in an order in which every atom after the first of its connected
  // component has an already-placed neighbour, so candidates come from that neighbour's
  // adjacency rather than from a scan of the whole molecule.
  struct Plan {
    std::vector<int> order;       // query atom indices, in visit order
    std::vector<int> parent;      // an earlier query atom adjacent to this one, or -1
    std::vector<int> parentBond;  // QBONDS index joining them, or -1
    // every query bond whose BOTH ends are placed at or before this step, for late checking
    std::vector<std::vector<int> > checks;
  };

  // A Plan depends only on the PATTERN, never on the molecule, so it is built once for the
  // process and shared.  Rebuilding it per enumerate() call cost 9 ms/mol -- the recursive
  // queries call enumerate() once per (sub-pattern, atom) cache miss, so `fr_phos_acid` alone
  // rebuilt 20 plans per candidate atom.
  static const Plan& plan(int pat) {
    static std::vector<Plan> cache;
    static std::vector<char> built;
    if (cache.empty()) {
      cache.resize(frag_prog::N_PATTERNS);
      built.assign(frag_prog::N_PATTERNS, 0);
    }
    if (!built[pat]) { cache[pat] = buildPlan(pat); built[pat] = 1; }
    return cache[pat];
  }

  static Plan buildPlan(int pat) {
    const frag_prog::Pattern& P = PATTERNS[pat];
    int na = P.na;
    std::vector<std::vector<std::pair<int, int> > > adj(na);   // (other query atom, qbond idx)
    for (int k = 0; k < P.nb; ++k) {
      const frag_prog::QBond& b = QBONDS[P.b0 + k];
      adj[b.u].push_back(std::make_pair((int)b.v, P.b0 + k));
      adj[b.v].push_back(std::make_pair((int)b.u, P.b0 + k));
    }
    Plan pl;
    std::vector<int> pos(na, -1);
    std::vector<char> seen(na, 0);
    for (int s = 0; s < na; ++s) {
      if (seen[s]) continue;
      // DFS from s so each new atom attaches to one already placed
      std::vector<int> stack(1, s);
      seen[s] = 1;
      pl.order.push_back(s); pl.parent.push_back(-1); pl.parentBond.push_back(-1);
      pos[s] = (int)pl.order.size() - 1;
      while (!stack.empty()) {
        int cur = stack.back(); stack.pop_back();
        for (size_t k = 0; k < adj[cur].size(); ++k) {
          int nx = adj[cur][k].first;
          if (seen[nx]) continue;
          seen[nx] = 1;
          pl.order.push_back(nx); pl.parent.push_back(cur); pl.parentBond.push_back(adj[cur][k].second);
          pos[nx] = (int)pl.order.size() - 1;
          stack.push_back(nx);
        }
      }
    }
    // Ring-closure bonds: any query bond not used as a tree edge must be verified once both of
    // its ends are placed.  This is what makes the 17 cyclic patterns correct.
    pl.checks.assign(pl.order.size(), std::vector<int>());
    std::vector<char> used(P.nb, 0);
    for (size_t st = 0; st < pl.order.size(); ++st)
      if (pl.parentBond[st] >= 0) used[pl.parentBond[st] - P.b0] = 1;
    for (int k = 0; k < P.nb; ++k) {
      if (used[k]) continue;
      const frag_prog::QBond& b = QBONDS[P.b0 + k];
      int st = std::max(pos[b.u], pos[b.v]);
      pl.checks[st].push_back(P.b0 + k);
    }
    return pl;
  }

  bool enumerate(int pat, bool firstOnly, int anchor, std::vector<std::vector<int> >& out) {
    const frag_prog::Pattern& P = PATTERNS[pat];
    if (P.na == 0) return false;
    const Plan& pl = plan(pat);
    std::vector<int> map(P.na, -1);
    std::vector<char> used(mol_->n, 0);
    return step(pat, pl, 0, map, used, firstOnly, anchor, out);
  }

  bool step(int pat, const Plan& pl, size_t si, std::vector<int>& map, std::vector<char>& used,
            bool firstOnly, int anchor, std::vector<std::vector<int> >& out) {
    const frag_prog::Pattern& P = PATTERNS[pat];
    if (si == pl.order.size()) {
      out.push_back(map);
      if (out.size() >= 1000u && !firstOnly) {
        // See the header: RDKit truncates at 1000 BEFORE uniquifying, so a corpus that reaches
        // this bound makes RDKit's own answer order-dependent.  Measured max on cpp/hard.smi is
        // 180.  Fail loudly rather than disagree quietly.
        std::fprintf(stderr, "fragmatch: pattern %s hit 1000 raw embeddings -- RDKit's "
                             "maxMatches default would truncate here and its count would become "
                             "order-dependent\n", P.label);
        std::abort();
      }
      return true;
    }
    int qa = pl.order[si];
    int qroot = AROOTS[P.a0 + qa];
    bool any = false;

    // Candidate molecule atoms for this query atom.
    if (pl.parent[si] < 0) {
      // component root: anchored (recursive query / atomMatchesPattern) or a full scan
      int lo = 0, hi = mol_->n;
      if (si == 0 && anchor >= 0) { lo = anchor; hi = anchor + 1; }
      for (int a = lo; a < hi; ++a) {
        if (used[a] || !evalAtom(qroot, a)) continue;
        map[qa] = a; used[a] = 1;
        if (checkClosures(pat, pl, si, map)) {
          if (step(pat, pl, si + 1, map, used, firstOnly, anchor, out)) {
            any = true;
            if (firstOnly) { used[a] = 0; map[qa] = -1; return true; }
          }
        }
        used[a] = 0; map[qa] = -1;
      }
    } else {
      int pa = map[pl.parent[si]];
      const frag_prog::QBond& pb = QBONDS[pl.parentBond[si]];
      for (int k = mol_->start[pa]; k < mol_->start[pa + 1]; ++k) {
        int a = mol_->nbr[k];
        if (used[a] || !evalBond(pb.root, mol_->nbond[k]) || !evalAtom(qroot, a)) continue;
        map[qa] = a; used[a] = 1;
        if (checkClosures(pat, pl, si, map)) {
          if (step(pat, pl, si + 1, map, used, firstOnly, anchor, out)) {
            any = true;
            if (firstOnly) { used[a] = 0; map[qa] = -1; return true; }
          }
        }
        used[a] = 0; map[qa] = -1;
      }
    }
    return any;
  }

  bool checkClosures(int pat, const Plan& pl, size_t si, const std::vector<int>& map) {
    for (size_t k = 0; k < pl.checks[si].size(); ++k) {
      const frag_prog::QBond& b = QBONDS[pl.checks[si][k]];
      int e = mol_->bondBetween(map[b.u], map[b.v]);
      if (e < 0 || !evalBond(b.root, e)) return false;
    }
    return true;
  }
};

// Count every named descriptor's pattern on one molecule.  `out` must have N_NAMED slots.
//
// The overload taking a Matcher& is what a batch loop should call: it reuses the matcher's
// recursive-query cache allocation across molecules instead of mallocing one per molecule.  Both
// spellings run identical arithmetic -- `bind()` clears the cache, so the second molecule cannot
// see the first one's answers.
inline void countAll(const Mol& m, Matcher& mt, int* out) {
  mt.bind(m);
  for (int i = 0; i < frag_prog::N_NAMED; ++i)
    out[i] = mt.matchCount(frag_prog::NAMED[i].pattern);
}

inline void countAll(const Mol& m, int* out) {
  Matcher mt;
  countAll(m, mt, out);
}

// ---------------------------------------------------------------------------------------------
// Two rdkit_core columns that are NOT substructure counts but are exact functions of the same
// inputs, so they ride along rather than needing a second pass over the molecule.
//
// NHOHCount is the one most likely to be got wrong from the documentation.  It is
// `rdMolDescriptors.CalcNumLipinskiHBD`, and rdkit/Chem/Lipinski.py displays
// `NHOHSmarts = [#8H1,#7H1,#7H2,#7H3]` a few lines above it -- but never uses it for the
// descriptor.  That SMARTS counts ATOMS; the C++ counts HYDROGENS, i.e. it sums
// GetTotalNumHs(includeNeighbors=True) over N and O.  The two differ on 776 of 4,000 molecules
// of cpp/hard.smi.  `includeNeighbors` is load-bearing: an explicit [2H] on an N is counted.
inline int nhohCount(const Mol& m) {
  int s = 0;
  for (int i = 0; i < m.n; ++i)
    if (m.z[i] == 7 || m.z[i] == 8) s += m.hcount[i];
  return s;
}

// HeavyAtomCount is Mol::getNumHeavyAtoms() -- atoms with z != 1.  Note this counts an explicit
// deuterium as light, because RDKit keys on atomic number and not on isotope.
inline int heavyAtomCount(const Mol& m) {
  int s = 0;
  for (int i = 0; i < m.n; ++i) if (m.z[i] != 1) ++s;
  return s;
}

}  // namespace fragmatch

#endif
