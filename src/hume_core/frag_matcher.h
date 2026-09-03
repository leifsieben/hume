// A SMARTS substructure matcher: ONE evaluator, run against whichever compiled query program it
// is bound to.
//
// TWO PROGRAMS GO THROUGH IT TODAY, and that is the reason `Matcher` takes a bound
// `frag_prog_types::Program` reference rather than reading one namespace's tables at global
// scope, which is how this file started:
//
//     cpp/frag_program.h        74 `rdkit_core` fragment/pattern descriptors, counted as
//                               len(GetSubstructMatches(patt, uniquify=True))
//     cpp/qed_alert_program.h   rdkit.Chem.QED.StructuralAlertSmarts, all 116, counted as a
//                               BOOLEAN per pattern -- QED.py sums HasSubstructMatch, so
//                               `hasMatch()` and not `matchCount()`
//
// A second subgraph-isomorphism implementation for the alerts would have been two things to
// verify and two places for a divergence to hide.  Both programs are compiled by the same
// cpp/gen_frag_program.py, from the same parser, validated the same way against RDKit's own
// DescribeQuery() parse tree.
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
//                                                  answer `[R1]` vs `[R2]`.  It also answers
//                                                  SMARTS `r` (AtomInRing), which is `!= 0` of
//                                                  the same count and is RDKit's own definition
//   tval   A_TVAL   GetTotalValence()    -- SMARTS `v`; see below
//   iso    A_ISO    GetIsotope()         -- SMARTS `[15N]`; see below
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
// THE FIRST COLUMN THAT IS NOT DERIVABLE is SMARTS `v`, total valence, and it is why the boundary
// grew a tenth `atom_i` column rather than this file growing a reconstruction.  The obvious one
// -- round(sum of incident bond orders) + nH -- is WRONG on 11,238 of those same 575,571 atoms,
// because RDKit folds aromatic bond contributions and hydrogens together under its own rounding
// rule.  Pyrrole's `[nH]`: two aromatic bonds sum to 3.0 and it carries one H, and RDKit's total
// valence is 3, not 4.  `v` is used by `fr_Imine` (`[Nv3]`), `NumHDonors` (`v3`, `v4`) and
// `NumHAcceptors` (`v2`, `v3`), so it cannot be dropped either.  It reaches the boundary as
// `Atom.GetTotalValence()` on the reference path and as the pickle's own explicit + implicit
// valence on the fast path; the two agree column-wise on both corpora (cpp/verify_molpickle.py).
//
// THE SECOND IS SMARTS ISOTOPE, and the QED alert set is what needed it: alerts 112-115 are
// `[15N]`, `[13C]`, `[18O]` and `[34S]` and nothing else.  The boundary's `mass` column can say an
// atom IS labelled -- constit.h's `exactMolWt` uses exactly that test, `mass != AWEIGHT[z]` -- but
// not labelled with WHAT, and recovering the mass number by searching RDKit's isotope-mass table
// backwards would put an injectivity argument where a value RDKit already has would do.  So it
// became the ELEVENTH `atom_i` column, the same call the tval addition made.  It is FREE on the
// fast path: molpickle.h was already decoding atom property-flag bit 8 into a local to compute
// `mass`, and now writes that local out instead of dropping it.
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
#include <memory>
#include <string>
#include <mutex>
#include <vector>

#include "../../cpp/frag_prog_types.h"

namespace fragmatch {

using frag_prog_types::Node;
using frag_prog_types::QBond;
using frag_prog_types::Pattern;
using frag_prog_types::Named;
using frag_prog_types::Program;
namespace ops = frag_prog_types;

// Bond order codes, as RDKit's Bond::BondType numbers them.  AROMATIC is 12; that is not a
// typo and not a bitmask.
enum : int { BO_SINGLE = 1, BO_DOUBLE = 2, BO_TRIPLE = 3, BO_AROMATIC = 12 };

struct Mol {
  int n = 0, nb = 0;
  std::vector<int> z, deg, nH, fchg, arom, nring, tval, iso;
  std::vector<int> hcount, tdeg;                  // derived: SMARTS H and X
  std::vector<int> bu, bv, border, bring;
  std::vector<int> start, nbr, nbond;             // CSR adjacency + parallel bond index

  void alloc(int na, int nbonds) {
    n = na; nb = nbonds;
    z.assign(na, 0); deg.assign(na, 0); nH.assign(na, 0); fchg.assign(na, 0);
    arom.assign(na, 0); nring.assign(na, 0); tval.assign(na, 0); iso.assign(na, 0);
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

// The visit order for one pattern's query atoms.  A Plan depends only on the PATTERN, never on
// the molecule, so it is built once per (program, pattern) for the life of the process.
// Rebuilding it per enumerate() call cost 9 ms/mol -- the recursive queries call enumerate() once
// per (sub-pattern, atom) cache miss, so `fr_phos_acid` alone rebuilt 20 plans per candidate atom.
struct Plan {
  std::vector<int> order;       // query atom indices, in visit order
  std::vector<int> parent;      // an earlier query atom adjacent to this one, or -1
  std::vector<int> parentBond;  // QBONDS index joining them, or -1
  // every query bond whose BOTH ends are placed at or before this step, for late checking
  std::vector<std::vector<int> > checks;
};

class Matcher {
 public:
  Matcher() : prog_(0), plans_(0), mol_(0) {}
  explicit Matcher(const Program& p) : prog_(0), plans_(0), mol_(0) { bindProgram(p); }

  // Point at a compiled query program.  Cheap and idempotent; the per-pattern search plans are
  // built once per program for the life of the process (see `planCache`), so a batch loop can
  // bind at construction and never touch this again.
  void bindProgram(const Program& p) {
    prog_ = &p;
    plans_ = &planCache(p);
  }

  // Point at a new molecule and clear the recursive-query cache.  Exists so a batch loop can
  // hold ONE Matcher and pay for the cache's storage once: `assign` reuses the vector's capacity,
  // where a fresh Matcher per molecule mallocs n_patterns * n bytes every time.  The cache is
  // cleared, not merely resized -- a stale `yes` from the previous molecule would be a wrong
  // count with no symptom.
  void bind(const Mol& m) {
    mol_ = &m;
    cache_.assign((size_t)prog_->n_patterns * (size_t)m.n, 0);
  }

  const Program& program() const { return *prog_; }

  int matchCount(int pat) {
    std::vector<std::vector<int> > out;
    enumerate(pat, /*firstOnly=*/false, /*anchor=*/-1, out);
    // uniquify=True: distinct sorted ATOM SETS, not distinct embeddings.
    for (size_t i = 0; i < out.size(); ++i) std::sort(out[i].begin(), out[i].end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return (int)out.size();
  }

  // RDKit's `HasSubstructMatch`: does ANY embedding exist.  This is what QED's ALERTS term wants
  // -- `sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert))` counts patterns,
  // not embeddings -- and it stops at the first one, so the 1000-embedding trap `matchCount()`
  // guards against cannot arise here at all.
  bool hasMatch(int pat) {
    std::vector<std::vector<int> > out;
    return enumerate(pat, /*firstOnly=*/true, /*anchor=*/-1, out);
  }

  bool atomMatchesPattern(int pat, int atom) {
    std::vector<std::vector<int> > out;
    return enumerate(pat, /*firstOnly=*/true, /*anchor=*/atom, out);
  }

 private:
  const Program* prog_;
  const std::vector<Plan>* plans_;
  const Mol* mol_;
  std::vector<uint8_t> cache_;

  // -- atom query tree --------------------------------------------------------------------
  bool evalAtom(int node, int a) {
    const Node& q = prog_->nodes[node];
    bool r;
    switch (q.op) {
      case ops::OP_ATOMAND: r = evalAtom(q.lhs, a) && evalAtom(q.rhs, a); break;
      case ops::OP_ATOMOR:  r = evalAtom(q.lhs, a) || evalAtom(q.rhs, a); break;
      case ops::OP_ATOMNULL: r = true; break;
      // AtomType fuses element and aromaticity: 1000+z is aromatic z, plain z is aliphatic z.
      case ops::OP_ATOMTYPE:
        r = (mol_->z[a] + 1000 * (mol_->arom[a] ? 1 : 0)) == q.val; break;
      // AtomAtomicNum (`[#7]`, and EVERY bracketed symbol outside the organic subset -- `[Si]`,
      // `[Hg]`, `[Se]`) constrains the element and says NOTHING about aromaticity.
      case ops::OP_ATOMATOMICNUM: r = mol_->z[a] == q.val; break;
      case ops::OP_ATOMEXPLICITDEGREE: r = mol_->deg[a] == q.val; break;
      case ops::OP_ATOMTOTALDEGREE: r = mol_->tdeg[a] == q.val; break;
      case ops::OP_ATOMHCOUNT: r = mol_->hcount[a] == q.val; break;
      case ops::OP_ATOMFORMALCHARGE: r = mol_->fchg[a] == q.val; break;
      // -1 is RDKit's sentinel for `[R]` == "in at least one ring", NOT a ring count of -1.
      case ops::OP_ATOMINNRINGS:
        r = (q.val < 0) ? (mol_->nring[a] != 0) : (mol_->nring[a] == q.val); break;
      case ops::OP_ATOMTOTALVALENCE: r = mol_->tval[a] == q.val; break;
      case ops::OP_ATOMISAROMATIC: r = (mol_->arom[a] ? 1 : 0) == q.val; break;
      case ops::OP_ATOMISALIPHATIC: r = (mol_->arom[a] ? 0 : 1) == q.val; break;
      // SMARTS isotope.  `getIsotope()` is 0 when unset, which is exactly what `[0*]` asks
      // about, so there is no sentinel here and the comparison is the whole predicate.
      case ops::OP_ATOMISOTOPE: r = mol_->iso[a] == q.val; break;
      // SMARTS `r`, a BOOLEAN, and a different primitive from `[R]`'s AtomInNRings above.
      // RDKit's queryIsAtomInRing is literally `numAtomRings(idx) != 0`, so the ring COUNT
      // already at the boundary answers it and no second ring perception is involved.
      case ops::OP_ATOMINRING: r = ((mol_->nring[a] != 0) ? 1 : 0) == q.val; break;
      case ops::OP_RECURSIVESTRUCTURE: r = recursive(q.val, a); break;
      default:
        // Thrown, not aborted, for the reason at the 1000-embedding site: a library must not
        // kill the host process, and the caller can be told which molecule by index.
        throw std::runtime_error("fragmatch: atom opcode " + std::to_string((int)q.op) +
                                 " unimplemented");
    }
    return q.neg ? !r : r;
  }

  // -- bond query tree --------------------------------------------------------------------
  bool evalBond(int node, int e) {
    const Node& q = prog_->nodes[node];
    bool r;
    switch (q.op) {
      case ops::OP_BONDAND: r = evalBond(q.lhs, e) && evalBond(q.rhs, e); break;
      case ops::OP_BONDOR:  r = evalBond(q.lhs, e) || evalBond(q.rhs, e); break;
      // SMARTS `~`, the any-bond query.  Exercised for the first time by the QED alerts.
      case ops::OP_BONDNULL: r = true; break;
      case ops::OP_BONDORDER: r = mol_->border[e] == q.val; break;
      // SMARTS `@` / `!@`.  Also first exercised by the QED alerts, alerts 43 and 103.
      case ops::OP_BONDINRING: r = (mol_->bring[e] ? 1 : 0) == q.val; break;
      // RDKit's queryBondIsSingleOrAromatic, on the bond TYPE.  This is the default bond
      // written between two SMARTS atoms and is a different query from `-` and from `:`.
      case ops::OP_SINGLEORAROMATICBOND:
        r = (mol_->border[e] == BO_SINGLE || mol_->border[e] == BO_AROMATIC); break;
      default:
        // Thrown, not aborted, for the reason at the 1000-embedding site: a library must not
        // kill the host process, and the caller can be told which molecule by index.
        throw std::runtime_error("fragmatch: bond opcode " + std::to_string((int)q.op) +
                                 " unimplemented");
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
  //
  // THE COMPONENT LOOP IS WHAT MAKES `.` WORK and it predates the QED alerts by accident: the
  // outer `for (int s = 0; s < na; ++s)` restarts at the root of every connected component of the
  // QUERY graph, so a pattern like `F.F.F.F` or QED alert 91's three disconnected esters needs
  // nothing new here.  Distinctness across components comes from `used[]`, which is exactly what
  // RDKit's matcher requires too.

  // ONE PLAN SET PER PROGRAM, resolved once per `bindProgram` and then held by pointer.  It used
  // to be a single function-local static indexed by pattern, which was correct while there was
  // one program and would silently serve `frag_prog`'s plan for `qed_prog`'s pattern 7 the moment
  // there were two.  The registry is process-lifetime by design -- a Plan is derived purely from
  // a constexpr table, so there is nothing to invalidate.
  // THREAD SAFETY. This registry is read and MUTATED lazily, and bindings.cpp now runs the
  // per-molecule row loop on every core. Two threads reaching a cold cache together raced on
  // `reg.push_back` -- a vector reallocating under a concurrent scan -- which showed up as an
  // intermittent process abort, roughly one run in four, with no message and a python traceback
  // pointing at the featurise call. It reproduced only with a fresh process, because the race
  // window is the FIRST call and the registry is process-lifetime by design.
  //
  // The mutex is taken once per bindProgram, i.e. a couple of times per molecule against ~700 us
  // of work, so the cost is not measurable. A lock-free fast path would still be UB: the scan
  // reads the same vector another thread may be reallocating.
  static const std::vector<Plan>& planCache(const Program& p) {
    static std::vector<std::pair<const Program*, std::shared_ptr<std::vector<Plan> > > > reg;
    static std::mutex mu;
    std::lock_guard<std::mutex> guard(mu);
    for (size_t i = 0; i < reg.size(); ++i)
      if (reg[i].first == &p) return *reg[i].second;
    std::shared_ptr<std::vector<Plan> > v(new std::vector<Plan>((size_t)p.n_patterns));
    for (int k = 0; k < p.n_patterns; ++k) (*v)[k] = buildPlan(p, k);
    reg.push_back(std::make_pair(&p, v));
    return *v;
  }

  static Plan buildPlan(const Program& prog, int pat) {
    const Pattern& P = prog.patterns[pat];
    int na = P.na;
    std::vector<std::vector<std::pair<int, int> > > adj(na);   // (other query atom, qbond idx)
    for (int k = 0; k < P.nb; ++k) {
      const QBond& b = prog.qbonds[P.b0 + k];
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
      const QBond& b = prog.qbonds[P.b0 + k];
      int st = std::max(pos[b.u], pos[b.v]);
      pl.checks[st].push_back(P.b0 + k);
    }
    return pl;
  }

  bool enumerate(int pat, bool firstOnly, int anchor, std::vector<std::vector<int> >& out) {
    const Pattern& P = prog_->patterns[pat];
    if (P.na == 0) return false;
    const Plan& pl = (*plans_)[pat];
    std::vector<int> map(P.na, -1);
    std::vector<char> used(mol_->n, 0);
    return step(pat, pl, 0, map, used, firstOnly, anchor, out);
  }

  bool step(int pat, const Plan& pl, size_t si, std::vector<int>& map, std::vector<char>& used,
            bool firstOnly, int anchor, std::vector<std::vector<int> >& out) {
    const Pattern& P = prog_->patterns[pat];
    if (si == pl.order.size()) {
      out.push_back(map);
      if (out.size() >= 1000u && !firstOnly) {
        // See the header: RDKit truncates at 1000 BEFORE uniquifying, so a corpus that reaches
        // this bound makes RDKit's own answer order-dependent.  Measured max on cpp/hard.smi is
        // 180.  Fail loudly rather than disagree quietly.
        // THROW, DO NOT KILL THE PROCESS. The bound itself is right -- RDKit truncates at
        // maxMatches=1000 BEFORE uniquifying, so past it RDKit's own count is order-dependent
        // and agreeing with it is not possible. What was wrong was the mechanism: a hard process
        // kill cannot be caught, so molhume.featurize(on_error="nan") could not honour its
        // contract, no index identified the molecule, and every other molecule in the process
        // died with it. Reported from a 35M-molecule run where one molecule ended the job.
        //
        // The measured max on cpp/hard.smi was 180, a 5.5x margin, and a large enough corpus
        // went straight through it: an unbranched chain of about 600 carbons is enough.
        throw std::runtime_error(
            std::string("fragmatch: pattern ") + P.label + " hit 1000 raw embeddings on this "
            "molecule. RDKit's maxMatches default truncates before uniquifying, so its own count "
            "would be order-dependent here, and this library will not agree quietly with a "
            "number that is not well defined.");
      }
      return true;
    }
    int qa = pl.order[si];
    int qroot = prog_->aroots[P.a0 + qa];
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
      const QBond& pb = prog_->qbonds[pl.parentBond[si]];
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
      const QBond& b = prog_->qbonds[pl.checks[si][k]];
      int e = mol_->bondBetween(map[b.u], map[b.v]);
      if (e < 0 || !evalBond(b.root, e)) return false;
    }
    return true;
  }
};

// Count every named pattern of the BOUND program on one molecule.  `out` must have
// `mt.program().n_named` slots.
//
// The Matcher& is what a batch loop should hold: it reuses the recursive-query cache allocation
// across molecules instead of mallocing one per molecule, and it resolves the program's plan set
// once. `bind()` clears the cache, so the second molecule cannot see the first one's answers.
inline void countAll(const Mol& m, Matcher& mt, int* out) {
  mt.bind(m);
  const Program& p = mt.program();
  for (int i = 0; i < p.n_named; ++i) out[i] = mt.matchCount(p.named[i].pattern);
}

// The QED ALERTS term: how many of the bound program's named patterns match AT ALL.  rdkit's
// QED.py sums `HasSubstructMatch` over the 116 alerts, so this is a count of PATTERNS and not of
// embeddings -- see `Matcher::hasMatch`.
inline int countMatching(const Mol& m, Matcher& mt) {
  mt.bind(m);
  const Program& p = mt.program();
  int n = 0;
  for (int i = 0; i < p.n_named; ++i) n += mt.hasMatch(p.named[i].pattern) ? 1 : 0;
  return n;
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
