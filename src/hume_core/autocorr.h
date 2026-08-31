// Mordred's Autocorrelation block as a header the extension can call: 6 variants x 9 lags x
// 10 weights = 540 columns, on the HYDROGEN-ADDED graph.
//
// WHAT THIS IS. cpp/ac.cpp was a standalone program -- a text loader, this computation, and a
// main(). Its arithmetic is verified against cpp/ac_reference.py, which was written first and
// confirmed against Mordred to 2.2e-16 over 3,402 cells. This header is its computation half,
// lifted out of the I/O exactly the way hume_blocks.h was lifted out of cpp/hume.cpp's text
// loader. cpp/ac.cpp now INCLUDES this file rather than carrying a copy: it keeps its main(),
// its `mols_h.txt` loader and its verify/bench modes, so the evidence is intact and there is
// only one copy of the accumulation to keep correct. A verbatim copy would have needed a drift
// guard; a shared header cannot drift.
//
// THE GRAPH IS NOT THE BOUNDARY'S GRAPH, and that is the whole difficulty. mordred sets
// `explicit_hydrogens = True` for this family, so aspirin is 21 atoms here and 13 in every other
// block. src/hume/_extract.py therefore serialises a SECOND molecule per input -- `Chem.AddHs(m)`
// with its own Gasteiger charges -- and bindings.cpp parses that blob with the same
// molpickle.h reader. See the note on `Pickles.h_blobs` for why the charges cannot simply be
// derived from the heavy-atom molecule's.
//
// ONE PASS OVER PAIRS, NOT NINE LAGS x TEN WEIGHTS OF MATRIX WORK. The obvious implementation
// builds B_k = (D == k) for each lag and evaluates w^T B_k w per weight: 90 matrix products of
// an n x n matrix. Instead every unordered pair is visited once, its distance read, and the ten
// weights accumulated into that lag's bucket -- O(n^2 * 10) rather than O(n^2 * 90), and the
// distance matrix is traversed a single time.
//
// The conventions that are not guessable, all inherited from cpp/ac_reference.py:
//   * lag 0 is NOT halved; every other lag is. Delta_0 = A, not A/2.
//   * GATS divides by 4*gsum, because the gmat sum double-counts each pair.
//   * GATS normalises by (A - 1); MATS normalises by A. Two different denominators in one family.
//
// A WEIGHT WITH A NON-FINITE ATOM IS NaN FOR ITS 54 COLUMNS, NOT A DROPPED MOLECULE. mordred
// fails one AtomicProperty at a time, so a selenium molecule loses the 54 `se` columns and keeps
// the other 486. `Z` is the one weight that can never take this path -- GetAtomicNum() is never
// NaN, so its 54 columns survive every molecule; see ac_weights.h.
//
// ALL TWELVE WEIGHTS ARE HERE. This header computed nine for a while and said so; the tenth, `Z`,
// closes the 52 members of the 865 that were the last Autocorrelation gap
// ({ATS,AATS,ATSC,AATSC} x lags 0-8 + {MATS,GATS} x lags 1-8, all suffixed `Z`; mordred defines
// no MATS0/GATS0, so it is 52 of the 54 emitted). Adding it re-shaped cpp/values_ac.txt from 486
// columns to 540, which is why it was deferred rather than difficult: the artifact whose md5
// proved the header lift changed nothing had to be regenerated.
//
// WHAT IS PROVEN ABOUT THE 486, EXACTLY. Projecting the new 540-column values_ac.txt back onto
// its 486 non-`Z` columns -- same %.12g text, drop every tenth field -- reproduces the old md5
// 7f08884f8700c23fd41e2a5315870a2e BYTE FOR BYTE over all 98,905 molecules. Adding the tenth
// weight moved no cell of the other nine. That is a statement about this file's arithmetic and
// it stands on its own; the mordred grade of the 54 new columns is cpp/verify_ac.py's to make
// and belongs in PORT_STATUS.md next to its version banner, not asserted here.
//
// `Z` is APPENDED as weight index 9 rather than inserted in mordred's getter order, so no
// pre-existing column changed its name.
#ifndef HUME_AUTOCORR_H
#define HUME_AUTOCORR_H

#include <cmath>
#include <cstdio>
#include <vector>

// The ten weight vectors and the element tables they read, generated from mordred itself by
// cpp/gen_ac_tables.py. Included from cpp/ rather than copied so there is exactly one of each in
// the repository -- the same arrangement crippen_typer.h has with cpp/crippen_tables.h.
#include "../../cpp/ac_weights.h"

namespace autocorr {

inline constexpr int NL = 9;    // lags 0..8
inline constexpr int NVAR = 6;  // ATS, AATS, ATSC, AATSC, MATS, GATS -- this order is fixed
inline constexpr int BIG = 1 << 20;
inline constexpr int N_COLS = NVAR * NL * NW;   // 648 = 6 x 9 x 12

// out is laid out [variant][lag][weight]. This is the order cpp/values_ac.txt is written in and
// the order verify_ac.py reads, so it is part of the format rather than an implementation
// detail: col_name() below is the only place that has to know it.
inline const char *col_name(int i) {
  static const char *VAR[NVAR] = {"ATS", "AATS", "ATSC", "AATSC", "MATS", "GATS"};
  static const char *WT[NW] = {"c", "d", "dv", "i", "p", "v", "se", "pe", "are", "Z",
                               "m", "s"};
  static char buf[16][24];
  static int slot = 0;
  const int q = i % NW, k = (i / NW) % NL, v = i / (NL * NW);
  char *b = buf[slot = (slot + 1) % 16];
  std::snprintf(b, 24, "%s%d%s", VAR[v], k, WT[q]);
  return b;
}

//! The hydrogen-added molecule, in the shape ac_weights.h wants. `at` is AtomRec, defined there.
struct Mol {
  int n = 0, nb = 0;
  std::vector<AtomRec> at;
  std::vector<double> w;                       // n * NW, atom-major
  std::vector<int> bu, bv;
  std::vector<std::vector<int>> adj;
};

//! Reused across molecules so the timed loop does not allocate.
struct Work {
  std::vector<int> D;
  std::vector<double> mean, csq, ats, atsc, gea, cnt;
};

inline void distances(const Mol &m, std::vector<int> &D) {
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

//! All N_COLS values for one molecule. NaN where Mordred returns NaN.
inline void row(Mol &m, Work &W, double *out) {
  const int n = m.n;
  distances(m, W.D);
  ac_weights(m.at, m.adj, m.w);
  std::vector<int> &D = W.D;

  // Per-weight availability, mordred's AtomicProperty.calculate() fail path. One non-finite
  // atom kills that weight's 54 columns and nothing else.
  bool okw[NW];
  for (int j = 0; j < NW; j++) okw[j] = true;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < NW; j++)
      if (!std::isfinite(m.w[(size_t)i * NW + j])) okw[j] = false;
  // Substitute zero so a dead weight cannot make the arithmetic below NaN-bound; the columns
  // are independent per weight, and every cell of a dead one is overwritten with NaN at the end.
  for (int j = 0; j < NW; j++)
    if (!okw[j])
      for (int i = 0; i < n; i++) m.w[(size_t)i * NW + j] = 0.0;

  W.mean.assign(NW, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < NW; j++) W.mean[j] += m.w[(size_t)i * NW + j];
  for (int j = 0; j < NW; j++) W.mean[j] /= n;
  std::vector<double> &mean = W.mean;

  // sum of centred squares, per weight -- the denominator both MATS and GATS are built on
  W.csq.assign(NW, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < NW; j++) {
      double c = m.w[(size_t)i * NW + j] - mean[j];
      W.csq[j] += c * c;
    }
  std::vector<double> &csq = W.csq;

  W.ats.assign((size_t)NL * NW, 0.0);
  W.atsc.assign((size_t)NL * NW, 0.0);
  W.gea.assign((size_t)NL * NW, 0.0);
  W.cnt.assign(NL, 0.0);
  std::vector<double> &ats = W.ats, &atsc = W.atsc, &gea = W.gea, &cnt = W.cnt;

  // lag 0: the self-pairs. Not halved, and Delta_0 = A.
  cnt[0] = n;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < NW; j++) {
      double v = m.w[(size_t)i * NW + j], c = v - mean[j];
      ats[(size_t)0 * NW + j] += v * v;
      atsc[(size_t)0 * NW + j] += c * c;
    }

  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      int d = D[(size_t)i * n + j];
      if (d <= 0 || d >= NL) continue;
      cnt[d] += 1.0;
      const double *wi = &m.w[(size_t)i * NW], *wj = &m.w[(size_t)j * NW];
      double *a = &ats[(size_t)d * NW], *ac = &atsc[(size_t)d * NW], *g = &gea[(size_t)d * NW];
      for (int q = 0; q < NW; q++) {
        double vi = wi[q], vj = wj[q];
        a[q] += vi * vj;
        ac[q] += (vi - mean[q]) * (vj - mean[q]);
        double df = vi - vj;
        g[q] += df * df;
      }
    }

  const double NaN = std::nan("");
  for (int k = 0; k < NL; k++)
    for (int q = 0; q < NW; q++) {
      double gsum = cnt[k];                       // already the UNORDERED pair count
      double A = ats[(size_t)k * NW + q], C = atsc[(size_t)k * NW + q];
      double den_m = csq[q] / n;
      double den_g = (n > 1) ? csq[q] / (n - 1) : 0.0;
      double aatsc = gsum > 0 ? C / gsum : NaN;
      int base = (k * NW + q);
      if (!okw[q]) {
        for (int v = 0; v < NVAR; v++) out[v * NL * NW + base] = NaN;
        continue;
      }
      out[0 * NL * NW + base] = A;
      out[1 * NL * NW + base] = gsum > 0 ? A / gsum : NaN;
      out[2 * NL * NW + base] = C;
      out[3 * NL * NW + base] = aatsc;
      out[4 * NL * NW + base] = (den_m != 0.0) ? aatsc / den_m : NaN;
      // GATS numerator: sum over ORDERED pairs / (4 gsum) == sum over unordered / (2 gsum)
      out[5 * NL * NW + base] =
          (n > 1 && gsum > 0 && den_g != 0.0)
              ? (gea[(size_t)k * NW + q] / (2.0 * gsum)) / den_g
              : NaN;
    }
}

}  // namespace autocorr

#endif
