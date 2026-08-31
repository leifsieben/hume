// mordred's RingCount family: the 49 columns that survive data/dedupe.json, in preset order.
//
// SPECIFICATION IS mordred/RingCount.py, read at mordred 1.2.0. Three classes, and the whole
// family is 20 lines of predicate on top of them:
//
//     Rings.calculate()      -> [frozenset(s) for s in Chem.GetSymmSSSR(mol)]
//     FusedRings.calculate() -> for each connected component of the graph on rings with an edge
//                               when |Ri & Rj| >= 2, the frozenset UNION of that component's
//                               atoms. Components of size 1 are NOT in the graph at all
//                               (networkx.add_edge only), so an unfused ring is not a fused ring.
//                               Returns [] outright when there are fewer than two rings.
//     RingCount.calculate()  -> count the rings (or fused systems) passing three predicates:
//                               order   len(R) == order, or >= order when `greater`
//                               arom    all(atom.GetIsAromatic() for atom in R)
//                               hetero  any(atom.GetAtomicNum() != 6 for atom in R)
//
// WHAT `len(R)` MEANS, AND WHY IT IS THE ATOM COUNT. mordred turns every ring into a frozenset
// of ATOM indices before it measures anything, so "order" is the number of DISTINCT ATOMS, not
// the number of bonds. For an SSSR ring the two agree. For a FUSED SYSTEM they do not, and that
// is the whole reason `nFRing` starts at order 4 while `nRing` starts at 3: naphthalene's fused
// system has 10 atoms and 11 bonds, and mordred counts it as a 10.
//
// A CYCLE IS NOT ITS VERTEX SET -- and this file never has to make that mistake, because it does
// not enumerate cycles. It consumes RDKit's ring list verbatim, exactly as mordred does, so two
// distinct rings that happen to share a vertex set stay two entries: `Rings` is a LIST of
// frozensets, never a set of them, so mordred does not deduplicate either. `ring_size()` below
// still counts DISTINCT atoms rather than trusting the stored length, because that, and not the
// list length, is what `len(frozenset(...))` computes.
//
// THIS FILE DOES NO AROMATICITY OR RING PERCEPTION OF ITS OWN. Both PORT_STATUS.md 1b repairs
// (ring sulfur with an exocyclic double bond cannot be aromatic; "a bond in an all-aromatic ring
// is aromatic" must run after perception) are therefore inapplicable here: the `arom` flag is
// RDKit's `GetIsAromatic()` arriving from the boundary, and the ring list is RDKit's own
// `GetSymmSSSR` / `RingInfo::atomRings()`, which is the object mordred asks for by name.
//
// HOUSE RULE 1: THE ARITHMETIC HERE IS WELL-POSED; ITS INPUT IS NOT. Everything below is a
// deterministic function of (ring list, z, arom) -- give it the same three and it gives the same
// 49 numbers, always. But `Chem.GetSymmSSSR` is NOT a function of the molecular graph. RDKit's
// `symmetrizeSSSR` adds symmetry-equivalent extra rings to the SSSR basis and its own source
// admits it "may miss extra rings that would need to swap two (or three...) rings to be
// included"; whether it misses one depends on the ORDER RDKit sees the molecule in. Measured
// with mordred on cpp/hard.smi:
//
//     25 of 100,000 molecules move at least one of these 49 columns under atom renumbering
//     five columns move: nARing (25), nG12Ring (11), n6Ring (6), n7Ring (6), n6ARing (6)
//     the other 44 never moved, and neither did any of the 32 columns of the other two families
//     C1=CC2C3C(C=C1)C23  gives sizes (3,3,7) on 33 of 60 numberings and (3,3,7,7) on 27
//
// and the basis itself is stable -- what flips is a single symmetry-equivalent extra ring, of a
// size already present. Brute force over every simple cycle confirms the larger answer is the
// RELEVANT-CYCLE set (the cycles that lie in some minimum cycle basis) on all seven probes: the
// well-posed object symmetrizeSSSR is reaching for, reached only sometimes.
//
// THE REPAIR IS TO THE SELECTION, NOT TO THE QUANTITY, and it lives at the boundary rather than
// here: `canon_rings()` in cpp/verify_topo3.py perceives the rings on a SKELETON rebuilt in
// canonical order and hands the result in. 0 of 100,000 molecules move afterwards. Two traps it
// had to survive, both worth knowing before touching any other ring-consuming family:
//
//   * `Chem.RenumberAtoms` leaves RingInfo UNINITIALISED. Without a following SanitizeMol you
//     measure lazy-perception order and get phantom ill-posedness.
//   * `Chem.RenumberAtoms` permutes ATOMS and leaves the BOND LIST order alone -- and ring
//     perception reads the bond list. So an atom-only renumbering screen UNDER-SAMPLES: on
//     `O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2` RDKit is stable over 201 atom
//     renumberings and gives two different answers once the bond order is shuffled too.
//     Canonical atom ranks ALONE therefore do not fix this; the bond order must be canonicalised
//     as well, which is why canon_rings() rebuilds the graph instead of renumbering it.
//
// THE REPAIR IS GATED, because it costs 104 us/mol against 5.1 for reading RingInfo -- five times
// all 81 columns. `gate()` in cpp/verify_topo3.py fires on a molecule with a ring of 7+ atoms, an
// atom in 3+ rings, or a ring system carrying 3+ independent cycles: 21.3% of cpp/hard.smi,
// 26.1 us/mol amortised, where the minimal gate that still covers every affected molecule fires
// on 8.5%. The extra is deliberate margin -- hexaprismane
// (C12C3C4C5C6C1C1C6C5C4C3C21) has every ring at 4 atoms and no atom in more than 2, and only the
// cyclomatic clause catches it. RESIDUAL RISK, in one sentence: a molecule slips past only if its
// ring set is ambiguous while every ring has at most 6 atoms, no atom lies in more than 2 rings
// and no ring system has more than 2 independent cycles, and the consequence is a
// numbering-dependent value for that molecule's five sensitive columns -- not a wrong value
// anywhere else. `verify_topo3.py gatecheck` is the standing guard: it runs the repair
// unconditionally over all 100,000 and asserts the gated pipeline is identical (currently 0/100,000
// disagreements), so a future corpus that defeats the gate fails loudly instead of silently.
//
// WHAT THE CALLER MUST SUPPLY.
//
//   z[i]        GetAtomicNum()      only ever compared against 6
//   arom[i]     GetIsAromatic()     RDKit's perceived flag, not re-derived
//   ring_off / ring_at   RDKit's ring list in CSR form: ring r is
//                        ring_at[ring_off[r] .. ring_off[r+1]).
//
// THE `nring` COLUMN OF THE (n_atoms, 10) `atom_i` CANNOT SERVE THIS FAMILY, and the reason is
// not preference. `RingInfo.NumAtomRings` is a per-ATOM count; every one of these 49 columns is a
// predicate on a RING -- its size, whether ALL of its atoms are aromatic, whether ANY is a
// heteroatom -- and the 28 fused columns additionally need |Ri & Rj| for every pair of rings, to
// build the fusion graph. None of that is recoverable from per-atom counts: benzene and
// cyclohexane give identical `nring` vectors and differ on 6 of the 49. So this header needs the
// ring SET, which is why it takes one. It does NOT re-perceive anything -- the set comes from the
// same single RDKit ring perception `nring` is filled from, so there is still only one perception
// and only one chance to disagree.
//
// The rings must be the ones mordred asks for, which is `Chem.GetSymmSSSR(mol)` -- the
// SYMMETRISED SSSR, not the plain SSSR and not all cycles. On a sanitised molecule that is
// already cached: RDKit's sanitisation runs SANITIZE_SYMMRINGS, so `mol.GetRingInfo().AtomRings()`
// IS the symmetrised SSSR and asking for it costs no ring perception. Verified equal on all
// 100,000 molecules of cpp/hard.smi. Cubane is the molecule that discriminates symmetrised SSSR
// (6 four-rings) from the plain SSSR (5); C12C3C1C23 is the one that discriminates a ring LIST
// from a ring SET (four triangles on four atoms).
//
// THE MOLECULE IS `Chem.RemoveHs(mol)`, not the molecule as parsed -- mordred's Context does that
// for every descriptor with `explicit_hydrogens = False`. On cpp/hard.smi the two coincide (plain
// RemoveHs keeps isotopic [2H]/[3H], which is all the explicit hydrogen this corpus carries), but
// the contract is RemoveHs and the caller should honour it rather than rely on the coincidence.
#ifndef HUME_RINGCOUNT_H
#define HUME_RINGCOUNT_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ringcount {

static constexpr int N_COLS = 52;   // 49 + the three the emit filter found missing

// arom / hetero are tri-state, matching mordred's `None / True / False`.
enum { ANY = -1, NO = 0, YES = 1 };

struct Spec {
  const char *name;   // mordred's own __str__ for this parameter tuple
  int order;          // -1 == None
  int greater;        // bool
  int fused;          // bool
  int arom;           // ANY / YES / NO
  int hetero;         // ANY / YES / NO
};

// The dedupe survivors, IN mordred's preset() ORDER.
//
// THREE WERE MISSING AND THE EMIT FILTER FOUND THEM. Applying the deduplication to the emitted
// set turned up `nAHRing`, `n8FARing` and `nG12FaHRing` as columns the dedup had SELECTED and
// this table did not supply -- the build computed only their case partners `naHRing`, `n8FaRing`
// and `nG12FAHRing`, which are different quantities (lowercase `a` is AROMATIC, uppercase `A` is
// ALIPHATIC). They are inserted in preset order: `nAHRing` heads the unfused/aliphatic/hetero
// group the way `naHRing` heads the aromatic one, `n8FARing` sorts by size before `n9FARing`,
// and `nG12FaHRing` closes the fused/aromatic/hetero group the way every other G12 entry closes
// its own. selfCheck() re-derives all three tuples from the name and refuses to load on a
// mismatch, so a wrong guess here is a load error and not a wrong column. Generated from mordred's own
// RingCount.preset() and filtered against data/dedupe.json; selfCheck() re-derives every
// parameter tuple from the name string and refuses to load if the two disagree.
static const Spec COLS[N_COLS] = {
    {"n3Ring", 3, 0, 0, ANY, ANY},        {"n4Ring", 4, 0, 0, ANY, ANY},
    {"n5Ring", 5, 0, 0, ANY, ANY},        {"n6Ring", 6, 0, 0, ANY, ANY},
    {"n7Ring", 7, 0, 0, ANY, ANY},        {"nG12Ring", 12, 1, 0, ANY, ANY},
    {"n3HRing", 3, 0, 0, ANY, YES},       {"n4HRing", 4, 0, 0, ANY, YES},
    {"n5HRing", 5, 0, 0, ANY, YES},       {"n6HRing", 6, 0, 0, ANY, YES},
    {"n7HRing", 7, 0, 0, ANY, YES},       {"naRing", -1, 0, 0, YES, ANY},
    {"n5aRing", 5, 0, 0, YES, ANY},       {"n6aRing", 6, 0, 0, YES, ANY},
    {"naHRing", -1, 0, 0, YES, YES},      {"n6aHRing", 6, 0, 0, YES, YES},
    {"nARing", -1, 0, 0, NO, ANY},        {"n5ARing", 5, 0, 0, NO, ANY},
    {"n6ARing", 6, 0, 0, NO, ANY},        {"nAHRing", -1, 0, 0, NO, YES},
    {"n5AHRing", 5, 0, 0, NO, YES},
    {"n6AHRing", 6, 0, 0, NO, YES},       {"nFRing", -1, 0, 1, ANY, ANY},
    {"n7FRing", 7, 0, 1, ANY, ANY},       {"n8FRing", 8, 0, 1, ANY, ANY},
    {"n9FRing", 9, 0, 1, ANY, ANY},       {"n10FRing", 10, 0, 1, ANY, ANY},
    {"n11FRing", 11, 0, 1, ANY, ANY},     {"n12FRing", 12, 0, 1, ANY, ANY},
    {"nG12FRing", 12, 1, 1, ANY, ANY},    {"nFHRing", -1, 0, 1, ANY, YES},
    {"n9FHRing", 9, 0, 1, ANY, YES},      {"n10FHRing", 10, 0, 1, ANY, YES},
    {"nG12FHRing", 12, 1, 1, ANY, YES},   {"nFaRing", -1, 0, 1, YES, ANY},
    {"n8FaRing", 8, 0, 1, YES, ANY},      {"n9FaRing", 9, 0, 1, YES, ANY},
    {"n10FaRing", 10, 0, 1, YES, ANY},    {"nG12FaRing", 12, 1, 1, YES, ANY},
    {"nFaHRing", -1, 0, 1, YES, YES},     {"n10FaHRing", 10, 0, 1, YES, YES},
    {"nG12FaHRing", 12, 1, 1, YES, YES},
    {"nFARing", -1, 0, 1, NO, ANY},       {"n8FARing", 8, 0, 1, NO, ANY},
    {"n9FARing", 9, 0, 1, NO, ANY},
    {"n10FARing", 10, 0, 1, NO, ANY},     {"nG12FARing", 12, 1, 1, NO, ANY},
    {"nFAHRing", -1, 0, 1, NO, YES},      {"n8FAHRing", 8, 0, 1, NO, YES},
    {"n9FAHRing", 9, 0, 1, NO, YES},      {"n10FAHRing", 10, 0, 1, NO, YES},
    {"nG12FAHRing", 12, 1, 1, NO, YES},
};

struct Mol {
  int n = 0;
  std::vector<int32_t> z;         // GetAtomicNum()
  std::vector<uint8_t> arom;      // GetIsAromatic()
  std::vector<int32_t> ring_off;  // n_rings + 1 entries
  std::vector<int32_t> ring_at;   // concatenated ring atom indices

  int n_rings() const { return ring_off.empty() ? 0 : (int)ring_off.size() - 1; }

  void alloc(int natoms) {
    n = natoms;
    z.assign(n, 0);
    arom.assign(n, 0);
    ring_off.assign(1, 0);
    ring_at.clear();
  }
  void add_ring(const int32_t *atoms, int k) {
    for (int q = 0; q < k; ++q) ring_at.push_back(atoms[q]);
    ring_off.push_back((int32_t)ring_at.size());
  }
};

// Reused across molecules so the hot loop makes no allocation.
struct Scratch {
  std::vector<int32_t> stamp;   // per atom: which ring last touched it
  std::vector<int32_t> parent;  // union-find over rings
  std::vector<int32_t> comp;    // component id per ring, -1 if unfused
  std::vector<uint8_t> seen;    // per atom, for the fused-system union
  // per-object (ring or fused system) properties: size, all-aromatic, has-hetero
  std::vector<int32_t> sz;
  std::vector<uint8_t> ar, het;
  std::vector<int32_t> members;
  std::vector<uint8_t> touched;
  std::vector<int32_t> fsz;
  std::vector<uint8_t> farom, fhet;
};

static inline int uf_find(std::vector<int32_t> &p, int x) {
  while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
  return x;
}

// One ring's three properties. `size` is the number of DISTINCT atoms, because mordred measures
// len(frozenset(ring)) and not len(ring); `stamp` is the distinctness test and is left dirty on
// purpose (the next caller stamps with its own tag).
static inline void ring_props(const Mol &m, Scratch &S, const int32_t *at, int k, int tag,
                              int32_t &size, uint8_t &all_arom, uint8_t &has_het) {
  int cnt = 0;
  uint8_t a = 1, h = 0;
  for (int q = 0; q < k; ++q) {
    const int i = at[q];
    if (S.stamp[i] == tag) continue;
    S.stamp[i] = tag;
    ++cnt;
    if (!m.arom[i]) a = 0;
    if (m.z[i] != 6) h = 1;
  }
  size = cnt;
  all_arom = a;
  has_het = h;
}

static inline bool passes(const Spec &s, int32_t size, uint8_t all_arom, uint8_t has_het) {
  if (s.order >= 0) {
    if (s.greater) { if (size < s.order) return false; }
    else if (size != s.order) return false;
  }
  if (s.arom != ANY && (int)(all_arom != 0) != s.arom) return false;
  if (s.hetero != ANY && (int)(has_het != 0) != s.hetero) return false;
  return true;
}

// out must have room for N_COLS doubles. Values are integer counts; they are emitted as double
// so the block shares one row type with everything else.
inline void compute(const Mol &m, double *out, Scratch &S) {
  for (int c = 0; c < N_COLS; ++c) out[c] = 0.0;
  const int R = m.n_rings();
  if (R == 0) return;

  if ((int)S.stamp.size() < m.n) S.stamp.assign(m.n, -1);
  else std::fill(S.stamp.begin(), S.stamp.begin() + m.n, -1);
  if ((int)S.seen.size() < m.n) S.seen.assign(m.n, 0);
  else std::fill(S.seen.begin(), S.seen.begin() + m.n, 0);

  // ---- plain rings -------------------------------------------------------------------------
  S.sz.assign(R, 0); S.ar.assign(R, 0); S.het.assign(R, 0);
  for (int r = 0; r < R; ++r)
    ring_props(m, S, &m.ring_at[m.ring_off[r]], m.ring_off[r + 1] - m.ring_off[r], r, S.sz[r],
               S.ar[r], S.het[r]);
  for (int c = 0; c < N_COLS; ++c) {
    if (COLS[c].fused) continue;
    int k = 0;
    for (int r = 0; r < R; ++r) k += passes(COLS[c], S.sz[r], S.ar[r], S.het[r]);
    out[c] = (double)k;
  }

  // ---- fused systems -----------------------------------------------------------------------
  // mordred short-circuits at fewer than two rings; so do we, and the union-find below would
  // give the same answer anyway (an isolated ring joins no component and is skipped).
  if (R < 2) return;

  S.parent.resize(R);
  for (int r = 0; r < R; ++r) S.parent[r] = r;
  S.touched.assign(R, 0);
  std::vector<uint8_t> &touched = S.touched;
  for (int i = 0; i < R; ++i) {
    const int bi = m.ring_off[i], ei = m.ring_off[i + 1];
    // tag i's atoms with a value no ring index can collide with
    const int tag = R + i;
    for (int q = bi; q < ei; ++q) S.stamp[m.ring_at[q]] = tag;
    for (int j = i + 1; j < R; ++j) {
      int shared = 0;
      // count DISTINCT shared atoms: |Ri & Rj| on frozensets, so a repeated atom in Rj must not
      // be counted twice. Re-stamping to a third value makes the count set-valued.
      const int bj = m.ring_off[j], ej = m.ring_off[j + 1];
      for (int q = bj; q < ej && shared < 2; ++q) {
        const int a = m.ring_at[q];
        if (S.stamp[a] == tag) { S.stamp[a] = tag + R; ++shared; }
      }
      for (int q = bj; q < ej; ++q) {  // restore i's tag for the next j
        const int a = m.ring_at[q];
        if (S.stamp[a] == tag + R) S.stamp[a] = tag;
      }
      if (shared >= 2) {
        const int ra = uf_find(S.parent, i), rb = uf_find(S.parent, j);
        if (ra != rb) S.parent[ra] = rb;
        touched[i] = touched[j] = 1;
      }
    }
  }

  // Collect each component's atom union. Only rings that acquired at least one fusion edge take
  // part -- networkx.Graph() never learns about a ring that add_edge was not called for.
  S.comp.assign(R, -1);
  int ncomp = 0;
  for (int r = 0; r < R; ++r) {
    if (!touched[r]) continue;
    const int root = uf_find(S.parent, r);
    if (S.comp[root] < 0) S.comp[root] = ncomp++;
    S.comp[r] = S.comp[root];
  }
  if (ncomp == 0) return;

  S.fsz.assign(ncomp, 0);
  S.farom.assign(ncomp, 1);
  S.fhet.assign(ncomp, 0);
  std::vector<int32_t> &fsz = S.fsz;
  std::vector<uint8_t> &farom = S.farom, &fhet = S.fhet;
  for (int c = 0; c < ncomp; ++c) {
    // clear only the atoms this component touched, on the way out
    S.members.clear();
    for (int r = 0; r < R; ++r) {
      if (S.comp[r] != c) continue;
      for (int q = m.ring_off[r]; q < m.ring_off[r + 1]; ++q) {
        const int a = m.ring_at[q];
        if (S.seen[a]) continue;
        S.seen[a] = 1;
        S.members.push_back(a);
        ++fsz[c];
        if (!m.arom[a]) farom[c] = 0;
        if (m.z[a] != 6) fhet[c] = 1;
      }
    }
    for (int a : S.members) S.seen[a] = 0;
  }
  for (int col = 0; col < N_COLS; ++col) {
    if (!COLS[col].fused) continue;
    int k = 0;
    for (int c = 0; c < ncomp; ++c) k += passes(COLS[col], fsz[c], farom[c], fhet[c]);
    out[col] = (double)k;
  }
}

// -----------------------------------------------------------------------------------------
// DRIFT GUARD. The only thing here derived from an upstream library is the 49-row COLS table:
// which parameter tuples mordred's RingCount.preset() yields, in what order, and what
// RingCount.__str__ names them. Nothing else in this file depends on an external table or
// constant -- the arithmetic is three predicates over a graph.
//
// So the guard checks exactly that, and it checks it in the direction a transcription error
// would break: it regenerates the FULL 138-entry preset from mordred's own nested loops, names
// every entry with mordred's own __str__ rules, and requires COLS to be a SUBSEQUENCE of that
// list with matching parameters at every position. A wrong `order`, a swapped `arom`/`hetero`,
// a fused flag on the wrong row, or two rows out of preset order all fail here rather than on
// some molecule nobody tried. cpp/verify_topo3.py then closes the loop from the other side by
// asserting these 49 names and tuples against the live mordred objects.
// -----------------------------------------------------------------------------------------
inline std::string preset_name(int order, int greater, int fused, int arom, int hetero) {
  std::string a;
  if (greater) a += "G";
  if (order >= 0) a += std::to_string(order);
  if (fused) a += "F";
  if (arom == YES) a += "a"; else if (arom == NO) a += "A";
  if (hetero == YES) a += "H"; else if (hetero == NO) a += "C";
  return "n" + a + "Ring";
}

inline void selfCheck() {
  std::vector<Spec> pre;
  std::vector<std::string> names;
  static std::vector<std::string> hold;   // owns the generated name strings
  hold.clear();
  const int aroms[3] = {ANY, YES, NO};
  const int hets[2] = {ANY, YES};
  for (int fused = 0; fused <= 1; ++fused)
    for (int ai = 0; ai < 3; ++ai)
      for (int hi = 0; hi < 2; ++hi) {
        const int a = aroms[ai], h = hets[hi];
        pre.push_back({nullptr, -1, 0, fused, a, h});
        for (int nn = (fused ? 4 : 3); nn < 13; ++nn) pre.push_back({nullptr, nn, 0, fused, a, h});
        pre.push_back({nullptr, 12, 1, fused, a, h});
      }
  if (pre.size() != 138)
    throw std::runtime_error("ringcount::selfCheck: preset size " + std::to_string(pre.size()) +
                             " != mordred's 138");
  for (auto &s : pre) hold.push_back(preset_name(s.order, s.greater, s.fused, s.arom, s.hetero));

  size_t p = 0;
  for (int c = 0; c < N_COLS; ++c) {
    const std::string want(COLS[c].name);
    size_t q = p;
    while (q < pre.size() && hold[q] != want) ++q;
    if (q == pre.size())
      throw std::runtime_error("ringcount::selfCheck: column '" + want +
                               "' is not in mordred's preset at or after preset index " +
                               std::to_string(p) + " (wrong name, or COLS out of preset order)");
    const Spec &g = pre[q];
    if (g.order != COLS[c].order || g.greater != COLS[c].greater || g.fused != COLS[c].fused ||
        g.arom != COLS[c].arom || g.hetero != COLS[c].hetero)
      throw std::runtime_error(
          "ringcount::selfCheck: '" + want + "' parameters disagree with mordred's preset: table "
          "(order=" + std::to_string(COLS[c].order) + ",greater=" + std::to_string(COLS[c].greater) +
          ",fused=" + std::to_string(COLS[c].fused) + ",arom=" + std::to_string(COLS[c].arom) +
          ",hetero=" + std::to_string(COLS[c].hetero) + ") vs preset (order=" +
          std::to_string(g.order) + ",greater=" + std::to_string(g.greater) + ",fused=" +
          std::to_string(g.fused) + ",arom=" + std::to_string(g.arom) + ",hetero=" +
          std::to_string(g.hetero) + ")");
    p = q + 1;
  }
}

}  // namespace ringcount

#endif  // HUME_RINGCOUNT_H
