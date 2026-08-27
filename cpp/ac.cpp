// Mordred's Autocorrelation block in C++: 6 variants x 9 lags x 9 weights = 486 columns,
// 419 of which are in HUME's CORE set -- 71% of that block's Mordred half.
//
//   ./ac verify [mols_h.txt]   -> values_ac.txt
//   ./ac bench  [mols_h.txt]
//
// The specification is cpp/ac_reference.py, which was written first and confirmed against
// Mordred to 2.2e-16 over 3,402 cells. This file is a transliteration of it, so a disagreement
// here is a porting bug rather than a misreading of the descriptor.
//
// ONE PASS OVER PAIRS, NOT NINE LAGS x NINE WEIGHTS OF MATRIX WORK. The obvious implementation
// builds B_k = (D == k) for each lag and evaluates w^T B_k w per weight: 81 matrix products of
// an n x n matrix. Instead every unordered pair is visited once, its distance read, and the
// nine weights accumulated into that lag's bucket -- O(n^2 * 9) rather than O(n^2 * 81), and
// the distance matrix is traversed a single time.
//
// The conventions that are not guessable, all inherited from ac_reference.py:
//   * lag 0 is NOT halved; every other lag is. Delta_0 = A, not A/2.
//   * GATS divides by 4*gsum, because the gmat sum double-counts each pair.
//   * GATS normalises by (A - 1); MATS normalises by A. Two different denominators in one
//     family.
//
// THE NINE WEIGHT VECTORS ARE NOW BUILT HERE, by ac_weights.h, not handed over by Python. The
// export used to carry finished numbers because calling mordred's own getters was "cheap"; it
// was 473.9 us/mol, twenty times the cost of the O(n^2) accumulation this file exists to do.
// The export now carries the raw graph and only the Gasteiger charge survives from RDKit.
//
// A WEIGHT WITH A NON-FINITE ATOM IS NaN FOR ITS 54 COLUMNS, NOT A DROPPED MOLECULE. mordred
// fails one AtomicProperty at a time, so a selenium molecule loses the 54 `se` columns and keeps
// the other 432. The exporter used to skip such molecules outright, which hid every rare element
// in the corpus from the verifier; they are carried now and gated here.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "ac_weights.h"       // the nine getters; also defines NW = 9 and struct AtomRec

static const int NL = 9;      // lags 0..8
static const int BIG = 1 << 20;

struct MolH {
  int n = 0, nb = 0;
  std::vector<AtomRec> at;
  std::vector<double> w;                       // n * NW, atom-major
  std::vector<int> bu, bv;
  std::vector<std::vector<int>> adj;
};

static std::vector<MolH> load(const char *path) {
  std::ifstream f(path);
  int nm;
  f >> nm;
  std::vector<MolH> ms(nm);
  for (int k = 0; k < nm; k++) {
    MolH &m = ms[k];
    f >> m.n >> m.nb;
    m.at.resize(m.n);
    for (int i = 0; i < m.n; i++) f >> m.at[i].z >> m.at[i].fc >> m.at[i].nh >> m.at[i].c;
    m.adj.assign(m.n, {});
    m.bu.resize(m.nb);
    m.bv.resize(m.nb);
    for (int b = 0; b < m.nb; b++) {
      f >> m.bu[b] >> m.bv[b];
      m.adj[m.bu[b]].push_back(m.bv[b]);
      m.adj[m.bv[b]].push_back(m.bu[b]);
    }
    // The nan/inf export desync cost a silently wrong 19.9-atom mean on a 30.6-atom corpus
    // once. Four tokens an atom, checked every molecule, so a bad field stops the run here
    // instead of shifting every number after it.
    if (!f) { fprintf(stderr, "PARSE FAILED at molecule %d -- reader desynced\n", k); exit(1); }
  }
  return ms;
}

static void distances(const MolH &m, std::vector<int> &D) {
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

// out is laid out [variant][lag][weight], variants in the order
// ATS, AATS, ATSC, AATSC, MATS, GATS. NaN where Mordred returns NaN.
static const int NVAR = 6;
static void autocorr(MolH &m, std::vector<int> &D, std::vector<double> &mean, double *out) {
  const int n = m.n;
  distances(m, D);
  ac_weights(m.at, m.adj, m.w);

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

  mean.assign(NW, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < NW; j++) mean[j] += m.w[(size_t)i * NW + j];
  for (int j = 0; j < NW; j++) mean[j] /= n;

  // sum of centred squares, per weight -- the denominator both MATS and GATS are built on
  std::vector<double> csq(NW, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < NW; j++) {
      double c = m.w[(size_t)i * NW + j] - mean[j];
      csq[j] += c * c;
    }

  std::vector<double> ats((size_t)NL * NW, 0.0), atsc((size_t)NL * NW, 0.0),
      gea((size_t)NL * NW, 0.0);
  std::vector<double> cnt(NL, 0.0);

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

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "bench";
  auto ms = load(argc > 2 ? argv[2] : "mols_h.txt");
  double na = 0;
  for (auto &m : ms) na += m.n;
  fprintf(stderr, "%zu molecules, mean %.1f atoms (with H)\n", ms.size(), na / ms.size());

  std::vector<int> D;
  std::vector<double> mean;
  std::vector<double> out((size_t)NVAR * NL * NW);

  if (mode == "verify") {
    FILE *f = fopen("values_ac.txt", "w");
    for (auto &m : ms) {
      autocorr(m, D, mean, out.data());
      for (size_t i = 0; i < out.size(); i++)
        fprintf(f, i ? " %.12g" : "%.12g", out[i]);
      fputc('\n', f);
    }
    fclose(f);
    fprintf(stderr, "wrote values_ac.txt (%zu cols)\n", out.size());
    return 0;
  }

  volatile double sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  const int REPS = 20;
  for (int r = 0; r < REPS; r++)
    for (auto &m : ms) { autocorr(m, D, mean, out.data()); sink += out[0]; }
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (REPS * ms.size());
  printf("  %-46s %8.2f us/mol\n", "Autocorrelation, 486 cols (6x9x9)", us);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
