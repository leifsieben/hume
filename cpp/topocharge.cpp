// Standalone harness for src/hume_core/topocharge.h.
//
//   ./cpp/topocharge dump  IN OUT   21 columns per molecule, %.17g
//   ./cpp/topocharge bench IN       contended timing
//   ./cpp/topocharge names          GGI1..GGI10 JGI1..JGI10 JGT10
//
// No selfCheck(): this family depends on no upstream table or constant. See the header.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/hume_core/topocharge.h"
#include "topo_io.h"

int main(int argc, char **argv) {
  const char *cmd = argc > 1 ? argv[1] : "names";
  if (!std::strcmp(cmd, "names")) {
    for (int c = 0; c < topocharge::N_COLS; ++c)
      std::printf("%s%s", topocharge::col_name(c), c + 1 == topocharge::N_COLS ? "\n" : " ");
    return 0;
  }
  const char *path = argc > 2 ? argv[2] : "cpp/topo3_mols.txt";
  std::vector<topo_io::Rec> recs = topo_io::load(path);
  topocharge::Mol m;
  topocharge::Scratch S;
  double out[topocharge::N_COLS];

  if (!std::strcmp(cmd, "dump")) {
    FILE *f = std::fopen(argc > 3 ? argv[3] : "cpp/topo3_topocharge.txt", "w");
    if (!f) { std::fprintf(stderr, "cannot write output\n"); return 2; }
    for (const auto &r : recs) {
      topocharge::build(m, r.n, r.nb, r.brows.data(), topo_io::Rec::BSTRIDE, 0, 1);
      topocharge::compute(m, out, S);
      for (int c = 0; c < topocharge::N_COLS; ++c)
        std::fprintf(f, c ? " %.17g" : "%.17g", out[c]);
      std::fputc('\n', f);
    }
    std::fclose(f);
    std::printf("topocharge: wrote %zu molecules x %d columns\n", recs.size(), topocharge::N_COLS);
    return 0;
  }

  if (!std::strcmp(cmd, "bench")) {
    // CONTENDED; see the note in cpp/ringcount.cpp. The CSR build is inside the loop because it
    // is per-molecule work; the file parse is not.
    std::vector<double> reps;
    double sink = 0;
    long long nat = 0;
    for (const auto &r : recs) nat += r.n;
    for (int rep = 0; rep < 11; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      for (const auto &r : recs) {
        topocharge::build(m, r.n, r.nb, r.brows.data(), topo_io::Rec::BSTRIDE, 0, 1);
        topocharge::compute(m, out, S);
        for (int c = 0; c < topocharge::N_COLS; ++c) sink += out[c];
      }
      auto t1 = std::chrono::steady_clock::now();
      reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                     (double)recs.size());
    }
    std::sort(reps.begin(), reps.end());
    std::printf("topocharge %2d cols  %zu mols  median %.3f us/mol  min %.3f  max %.3f  CONTENDED "
                "(mean %.1f atoms)\n",
                topocharge::N_COLS, recs.size(), reps[reps.size() / 2], reps.front(), reps.back(),
                (double)nat / (double)recs.size());
    if (sink == 12345.6789) std::printf("");
    return 0;
  }
  std::fprintf(stderr, "usage: topocharge [dump IN OUT | bench IN | names]\n");
  return 1;
}
