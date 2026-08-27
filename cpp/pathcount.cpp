// Standalone harness for src/hume_core/pathcount.h.
//
//   ./cpp/pathcount dump  IN OUT   11 columns per molecule, %.17g
//   ./cpp/pathcount bench IN       contended timing
//   ./cpp/pathcount names          MPC4 MPC6 MPC9 piPC1..piPC6 piPC8 piPC10
//
// No selfCheck(): this family depends on no upstream table. The bond orders are RDKit's own
// GetBondTypeAsDouble() arriving in the exchange file, not a transcribed table. See the header.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/hume_core/pathcount.h"
#include "topo_io.h"

int main(int argc, char **argv) {
  const char *cmd = argc > 1 ? argv[1] : "names";
  if (!std::strcmp(cmd, "names")) {
    for (int c = 0; c < pathcount::N_COLS; ++c)
      std::printf("%s%s", pathcount::COLS[c].name, c + 1 == pathcount::N_COLS ? "\n" : " ");
    return 0;
  }
  const char *path = argc > 2 ? argv[2] : "cpp/topo3_mols.txt";
  std::vector<topo_io::Rec> recs = topo_io::load(path);
  pathcount::Mol m;
  pathcount::Scratch S;
  double out[pathcount::N_COLS];

  if (!std::strcmp(cmd, "dump")) {
    FILE *f = std::fopen(argc > 3 ? argv[3] : "cpp/topo3_pathcount.txt", "w");
    if (!f) { std::fprintf(stderr, "cannot write output\n"); return 2; }
    for (const auto &r : recs) {
      pathcount::build_from_rows(m, r.n, r.nb, r.brows.data(), topo_io::Rec::BSTRIDE, 0, 1,
                                 r.bo.data(), r.arows.data(), topo_io::Rec::ASTRIDE, 0);
      pathcount::compute(m, out, S);
      for (int c = 0; c < pathcount::N_COLS; ++c)
        std::fprintf(f, c ? " %.17g" : "%.17g", out[c]);
      std::fputc('\n', f);
    }
    std::fclose(f);
    std::printf("pathcount: wrote %zu molecules x %d columns\n", recs.size(), pathcount::N_COLS);
    return 0;
  }

  if (!std::strcmp(cmd, "bench")) {
    // CONTENDED; see the note in cpp/ringcount.cpp. build() is inside the loop -- it is where
    // the useHs=False rule is applied, so it is part of the descriptor, not part of the I/O.
    std::vector<double> reps;
    double sink = 0;
    for (int rep = 0; rep < 11; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      for (const auto &r : recs) {
        pathcount::build_from_rows(m, r.n, r.nb, r.brows.data(), topo_io::Rec::BSTRIDE, 0, 1,
                                 r.bo.data(), r.arows.data(), topo_io::Rec::ASTRIDE, 0);
        pathcount::compute(m, out, S);
        for (int c = 0; c < pathcount::N_COLS; ++c) sink += out[c];
      }
      auto t1 = std::chrono::steady_clock::now();
      reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                     (double)recs.size());
    }
    std::sort(reps.begin(), reps.end());
    std::printf("pathcount  %2d cols  %zu mols  median %.3f us/mol  min %.3f  max %.3f  CONTENDED\n",
                pathcount::N_COLS, recs.size(), reps[reps.size() / 2], reps.front(), reps.back());
    if (sink == 12345.6789) std::printf("");
    return 0;
  }
  std::fprintf(stderr, "usage: pathcount [dump IN OUT | bench IN | names]\n");
  return 1;
}
