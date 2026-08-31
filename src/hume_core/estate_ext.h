// The 22 columns of group D_estate: mordred's per-atom-type E-state EXTREMA, its `SZ`
// constitutional sum, and five self-returning walk counts.
//
//     MAX<t> / MIN<t>   16   sCH3 ssCH2 aaCH sssCH dssC aasC ssNH dO, max and min
//     SZ                 1   sum of atomic number / Z_carbon, over the HYDROGEN-ADDED atom list
//     SRW04/06/08/09/10  5   log(trace(A^k) + 1)
//
// SPECIFICATIONS: mordred/EState.py, mordred/Constitutional.py, mordred/WalkCount.py and
// mordred/_graph_matrix.py at 1.2.0. Read, not remembered.
//
// ------------------------------------------------------------------------------------------
// THIS FILE ADDS ALMOST NO ARITHMETIC. THAT IS THE POINT.
// ------------------------------------------------------------------------------------------
// Every one of the 22 is a REDUCTION over a vector this repository already builds and has
// already verified, so all three families are wired to the existing code rather than restated:
//
//   * the E-state TYPE tuple per atom            esttyper::typeAtom() (estate_typer.h)
//   * the E-state INDEX per atom                 hume_blocks.h's estate_from(), i.e. BlockWork::ES
//   * trace(A^k) for k = 1..10                   topomisc::detail::walkTraces(), the function
//                                                that already produces SRW05 and SRW07
//   * numpy's pairwise summation                 topomisc::npPairwiseSum()
//
// `MAX<t>` is `S<t>` with `+=` replaced by a running maximum over the SAME atoms the SAME typer
// selected, and `SRW04` is a different subscript of the SAME `tr[]` array `SRW05` is read from.
// If a future edit finds itself re-deriving an E-state type or re-forming a matrix power here,
// the wiring is wrong, not the reduction.
//
// THE INPUT IS `esttyper::Mol` AND NOTHING ELSE. That struct already carries `z`, `nH`
// (= GetTotalNumHs(False), the boundary's A_NH column) and the CSR adjacency of the
// HYDROGEN-SUPPRESSED graph, which is every input all three families need -- so wiring this in
// costs bindings.cpp one call next to the `esttyper::aggregate()` it already makes, with no new
// marshalling and no second graph. The hydrogen-added atom COUNT that `SZ` needs is derived from
// `nH` the same way topomisc.h derives it, for the same reason: Constitutional needs no bonds.
//
// ------------------------------------------------------------------------------------------
// THE ONE QUIRK: MAX/MIN OF AN ABSENT ATOM TYPE IS NaN, NOT ZERO. ALL SIXTEEN.
// ------------------------------------------------------------------------------------------
// `AtomTypeEState.calculate` builds `indices` as a lazy filter over the typed atoms and then
// calls the BUILTIN `max`/`min` on it inside `with self.rethrow_na(ValueError)`. A molecule with
// no `sCH3` gives an empty iterator, the builtin raises `ValueError: max() arg is an empty
// sequence`, `rethrow_na` turns that into `self.fail(e)` -- a MissingValueException -- and the
// column comes out NaN. The docstring says so out loud: "returns NaN when type in [min, max] and
// N_X = 0". This is NOT the `count`/`sum` path, which starts from a real zero and returns 0.
//
// It is a large fraction of the corpus, not a corner: 5,150 of the 20,000 molecules of
// data/dedupe2 have no `sCH3` and 9,009 have no `ssNH`, so a "0 when absent" convention would
// mis-fill 45% of `MIN/MAXssNH`. `N<t> == 0 <=> MAX<t> is NaN` is checked column by column over
// all 20,000 in verify_estate.py rather than asserted here.
//
// THE COMPARISON IS CPython's, NOT std::max_element's. `max(it)` keeps the first element and
// replaces it only on a STRICT `>`; `min` only on a strict `<`. The loops below are written that
// way so ties, signed zeros and any future NaN in the index resolve identically. No summation
// happens at all, so there is no associativity question in these sixteen columns.
//
// ------------------------------------------------------------------------------------------
// SZ IS THE HYDROGEN-ADDED SUM, AND IT IS numpy's PAIRWISE SUM
// ------------------------------------------------------------------------------------------
// `Constitutional` is the one mordred family that never sets `explicit_hydrogens`, so it
// inherits the base class's `True` and sees `Chem.AddHs(mol)`. topomisc.h's note 3 documents
// this at length for `MZ`/`Mv`/`Mp` and the array built here is byte-identical to the one it
// builds for `MZ`: heavy atoms in their own order, then `sum(GetTotalNumHs())` hydrogens, since
// AddHs appends. `SZ` is that array's `np.sum` and `MZ` is the same sum divided by its length --
// they are the same number over the same association, which is exactly why the summation is
// `topomisc::npPairwiseSum` and not a fresh loop.
//
// ------------------------------------------------------------------------------------------
// SRW04/06/08/09/10 ARE FIVE MORE SUBSCRIPTS OF AN ARRAY THAT ALREADY EXISTS
// ------------------------------------------------------------------------------------------
// `WalkCount(order=k, self_returning=True)` is `np.log(trace(A^k) + 1)` on the integer
// adjacency matrix of the hydrogen-suppressed graph. topomisc::detail::walkTraces already
// produces `tr[1..10]` exactly -- via `trace(A^2p) = ||A^p||_F^2` and
// `trace(A^(2p+1)) = <A^p, A^(p+1)>_F`, five sparse-times-dense products instead of nine matrix
// powers -- because TSRW10 needs all ten. SRW05 and SRW07 are `log(tr[5]+1)` and `log(tr[7]+1)`
// and are verified exact; these five are `tr[4]`, `tr[6]`, `tr[8]`, `tr[9]` and `tr[10]` from
// the identical call. Calling that function rather than copying its loop is the whole reason
// this header includes topomisc.h.
//
// SRW09 IS ZERO FOR 6,460 OF THE 20,000 MOLECULES AND THAT IS CORRECT, NOT A GAP. An odd-length
// closed walk requires an odd cycle, so `trace(A^9) = 0` -- and `log(0+1) = 0` -- for every
// bipartite molecular graph, which is most of them. `np.log` of an exact integer + 1 is the only
// floating-point operation in these five columns; both sides agree on the integer bit for bit.
//
// HOW TO WIRE IT (bindings.cpp, not edited here per the contract -- see NOTES_estate.md):
//
//     estate_ext::Scratch  ex;                    // one per AllWork, like topomisc::Scratch
//     ...
//     esttyper::aggregate(W.em, W.bw.ES.data(), W.ecount.data(), W.esum.data());
//     estate_ext::compute(W.em, W.bw.ES.data(), out + OFF_ESTATE_EXT, W.ex);
//
// `W.em` is already filled and `W.bw.ES` is already the per-atom E-state index, so the call
// needs nothing that is not already in hand. It MUST sit inside the same guard as the F_ESTATE
// block: like the `S<t>` columns, sixteen of these read `BlockWork::ES` and would otherwise
// weight by a stale molecule's index.
#ifndef HUME_ESTATE_EXT_H
#define HUME_ESTATE_EXT_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "estate_typer.h"
#include "topomisc.h"

namespace estate_ext {

static constexpr int N_COLS = 22;

// The eight E-state atom types that carry a MAX/MIN column in this group, in column order.
static constexpr int N_EXT = 8;
static const char *const EXT_TYPES[N_EXT] = {
    "sCH3", "ssCH2", "aaCH", "sssCH", "dssC", "aasC", "ssNH", "dO",
};

static const char *const COLS[N_COLS] = {
    "SZ",
    "MAXsCH3", "MAXssCH2", "MAXaaCH", "MAXsssCH", "MAXdssC", "MAXaasC", "MAXssNH", "MAXdO",
    "MINsCH3", "MINssCH2", "MINaaCH", "MINsssCH", "MINdssC", "MINaasC", "MINssNH", "MINdO",
    "SRW04", "SRW06", "SRW08", "SRW09", "SRW10",
};

inline const char *col_name(int i) {
  if (i < 0 || i >= N_COLS)
    throw std::runtime_error("estate_ext::col_name: column index " + std::to_string(i) +
                             " out of range 0.." + std::to_string(N_COLS - 1));
  return COLS[i];
}

// Column offsets, so the reductions below never spell an index twice.
enum : int { C_SZ = 0, C_MAX0 = 1, C_MIN0 = 9, C_SRW0 = 17 };

// The five walk orders of this group, in column order. `tr[]` is 1-based on the walk length.
static constexpr int SRW_ORDER[5] = {4, 6, 8, 9, 10};

//! Reused across molecules so the timed loop does not allocate. `wm`/`ws` are the topomisc
//! structures walkTraces() reads and writes; `prop` is the Constitutional property array.
struct Scratch {
  topomisc::Mol wm;
  topomisc::Scratch ws;
  std::vector<double> prop;
};

// Row index in estate_tbl::ROWS for each of EXT_TYPES, resolved ONCE by name.
//
// Resolving by name rather than hard-coding eight integers is deliberate: estate_tables.h is
// generated from RDKit's own `_rawD` and its row ORDER is upstream's, so a literal index would
// silently point at a different pattern if a row were ever inserted. A name that is not in the
// table is a hard error at first call, naming the column, rather than a column of NaN.
struct TypeSlots {
  int row[N_EXT];                      // estate_tbl row index of each of the eight
  int8_t slot[estate_tbl::N_ROWS];     // inverse: row index -> 0..7, or -1

  TypeSlots() {
    for (int r = 0; r < estate_tbl::N_ROWS; ++r) slot[r] = -1;
    for (int t = 0; t < N_EXT; ++t) {
      row[t] = -1;
      for (int r = 0; r < estate_tbl::N_ROWS; ++r)
        if (std::string(estate_tbl::ROWS[r].name) == EXT_TYPES[t]) { row[t] = r; break; }
      if (row[t] < 0)
        throw std::runtime_error(
            std::string("estate_ext: E-state type '") + EXT_TYPES[t] +
            "' (columns MAX" + EXT_TYPES[t] + " / MIN" + EXT_TYPES[t] +
            ") is not present in cpp/estate_tables.h -- RDKit's AtomTypes.py _rawD changed "
            "upstream and the table needs regenerating with cpp/verify_estate.py tables");
      slot[row[t]] = (int8_t)t;
    }
  }
};

// Function-local static: C++11 guarantees this is built exactly once, thread-safely.
inline const TypeSlots &typeSlots() {
  static const TypeSlots T;
  return T;
}

//! All 22 columns for one molecule. `em` is the HYDROGEN-SUPPRESSED molecule already built for
//! esttyper::aggregate(); `estate` is the per-atom E-state index (hume_blocks.h estate_from(),
//! i.e. BlockWork::ES), which the sixteen MAX/MIN columns reduce over. NaN where mordred is NaN.
//!
//! `tr_in`, if given, is `trace(A^k)` for k = 0..10 -- exactly topomisc's `tr[]`. Passing it
//! SKIPS the walkTraces call entirely, which is 70% of this group's cost (4.36 us of 6.25 us
//! per molecule, measured; see NOTES_estate.md) and is work topomisc.h already does for TSRW10.
//! It is optional rather than required because topomisc::compute() keeps `tr[]` local today,
//! and this header may not edit it.
inline void compute(const esttyper::Mol &em, const double *estate, double *out, Scratch &S,
                    const int64_t *tr_in = nullptr) {
  const double NANV = std::numeric_limits<double>::quiet_NaN();
  const int n = em.n;

  if (estate == nullptr && n > 0)
    throw std::runtime_error(
        "estate_ext::compute: the per-atom E-state index is null, but all sixteen MAX*/MIN* "
        "columns are reductions over it. Pass BlockWork::ES (hume_blocks.h estate_from()); "
        "there is no meaningful value to substitute.");

  // ---- SZ ------------------------------------------------------------------------------
  // Chem.AddHs appends, so the array is the given heavy atoms in their own order followed by
  // sum(GetTotalNumHs()) hydrogens -- the same array topomisc.h builds for MZ, summed the same
  // way. numpy's pairwise association is not decoration here: the terms are p/6 for integer p,
  // which is not dyadic for p not a multiple of 3.
  {
    int64_t nh_total = 0;
    for (int i = 0; i < n; ++i) nh_total += (int64_t)em.nH[i];
    const int64_t nh = (int64_t)n + nh_total;
    S.prop.resize(nh > 0 ? (size_t)nh : 1);
    for (int i = 0; i < n; ++i) S.prop[i] = (double)em.z[i] / chiwalk_tables::CARBON_Z;
    for (int64_t i = n; i < nh; ++i) S.prop[(size_t)i] = 1.0 / chiwalk_tables::CARBON_Z;
    out[C_SZ] = topomisc::npPairwiseSum(S.prop.data(), nh);
  }

  // ---- MAX<t> / MIN<t> -----------------------------------------------------------------
  // One pass over the atoms with the SAME typer aggregate() uses. `seen` is what makes the
  // empty case NaN rather than +/-inf or 0: it is mordred's `N_X = 0` test, read off the same
  // traversal instead of from a second one.
  {
    // Row index -> slot in the eight, so the inner loop over an atom's type tuple is a lookup
    // and not eight comparisons. -1 means "this type has no MAX/MIN column in this group".
    const int8_t *slotOf = typeSlots().slot;

    double mx[N_EXT], mn[N_EXT];
    bool seen[N_EXT];
    for (int t = 0; t < N_EXT; ++t) { seen[t] = false; mx[t] = 0.0; mn[t] = 0.0; }

    uint8_t hit[esttyper::MAX_TYPES];
    for (int i = 0; i < n; ++i) {
      const int k = esttyper::typeAtom(em, i, hit);
      if (k == 0) continue;
      const double v = estate[i];
      for (int q = 0; q < k; ++q) {
        const int t = slotOf[hit[q]];
        if (t < 0) continue;
        if (!seen[t]) { seen[t] = true; mx[t] = v; mn[t] = v; continue; }
        // CPython's max()/min(): replace only on a STRICT comparison.
        if (v > mx[t]) mx[t] = v;
        if (v < mn[t]) mn[t] = v;
      }
    }
    for (int t = 0; t < N_EXT; ++t) {
      out[C_MAX0 + t] = seen[t] ? mx[t] : NANV;
      out[C_MIN0 + t] = seen[t] ? mn[t] : NANV;
    }
  }

  // ---- SRW04 / SRW06 / SRW08 / SRW09 / SRW10 -------------------------------------------
  // topomisc::detail::walkTraces() is called, not copied. It wants a topomisc::Mol, which is
  // the same graph em already holds: `bu`/`bv` are the half-edges taken once each (u < v),
  // `start`/`nbr` the CSR verbatim. `z` is unread by walkTraces but is filled anyway so the
  // struct is never half-initialised.
  {
    int64_t tr[11] = {0}, sums[11] = {0};
    if (tr_in != nullptr) {
      for (int k = 0; k <= 10; ++k) tr[k] = tr_in[k];
    } else if (n > 0) {
      topomisc::Mol &g = S.wm;
      g.n = n;
      g.nh_total = 0;
      g.z.resize(n);
      for (int i = 0; i < n; ++i) g.z[i] = (int32_t)em.z[i];
      g.start.assign(em.start.begin(), em.start.end());
      g.nbr.assign(em.nbr.begin(), em.nbr.end());
      g.bu.clear();
      g.bv.clear();
      g.bu.reserve(em.nbr.size() / 2);
      g.bv.reserve(em.nbr.size() / 2);
      for (int i = 0; i < n; ++i)
        for (int e = em.start[i]; e < em.start[i + 1]; ++e)
          if (em.nbr[e] > i) { g.bu.push_back(i); g.bv.push_back(em.nbr[e]); }
      topomisc::detail::walkTraces(g, S.ws, tr, sums);
    }
    for (int s = 0; s < 5; ++s)
      out[C_SRW0 + s] = std::log((double)(tr[SRW_ORDER[s]] + 1));
  }
}

}  // namespace estate_ext

#endif  // HUME_ESTATE_EXT_H
