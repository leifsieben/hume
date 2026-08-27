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

// THE COMPUTATION NOW LIVES IN A HEADER, so the package and this harness run the SAME code
// rather than two copies of it. This file keeps its main(), its mols_h.txt loader and its
// verify/bench modes -- it is the evidence, and `./ac verify` still writes values_ac.txt
// from exactly the arithmetic src/hume/__init__.py now calls. ac_weights.h (the nine
// getters, NW and struct AtomRec) comes in through it.
#include "../src/hume_core/autocorr.h"

using autocorr::NL;
using autocorr::NVAR;
using MolH = autocorr::Mol;

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

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "bench";
  auto ms = load(argc > 2 ? argv[2] : "mols_h.txt");
  double na = 0;
  for (auto &m : ms) na += m.n;
  fprintf(stderr, "%zu molecules, mean %.1f atoms (with H)\n", ms.size(), na / ms.size());

  autocorr::Work W;
  std::vector<double> out(autocorr::N_COLS);

  if (mode == "verify") {
    FILE *f = fopen("values_ac.txt", "w");
    for (auto &m : ms) {
      autocorr::row(m, W, out.data());
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
    for (auto &m : ms) { autocorr::row(m, W, out.data()); sink += out[0]; }
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (REPS * ms.size());
  printf("  %-46s %8.2f us/mol\n", "Autocorrelation, 486 cols (6x9x9)", us);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
