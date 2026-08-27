// Standalone harness for src/hume_core/ringcount.h.
//
//   python cpp/verify_topo3.py          builds cpp/topo3_mols.txt, runs this, compares
//   ./cpp/ringcount dump  IN OUT        49 columns per molecule, %.17g
//   ./cpp/ringcount bench IN            contended timing
//   ./cpp/ringcount names               the 49 column names, in emit order
//
// selfCheck() runs first, always, and throws before any molecule is read. See the drift-guard
// note at the bottom of ringcount.h for what it actually proves.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/hume_core/ringcount.h"
#include "topo_io.h"

static void fill(ringcount::Mol &m, const topo_io::Rec &r) {
  m.n = r.n;
  m.z.assign(r.z.begin(), r.z.end());
  m.arom.assign(r.n, 0);
  for (int i = 0; i < r.n; ++i) m.arom[i] = (uint8_t)r.arom[i];
  m.ring_off = r.ring_off;
  m.ring_at = r.ring_at;
}

int main(int argc, char **argv) {
  try {
    ringcount::selfCheck();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "DRIFT GUARD FAILED: %s\n", e.what());
    return 3;
  }
  const char *cmd = argc > 1 ? argv[1] : "names";
  if (!std::strcmp(cmd, "names")) {
    for (int c = 0; c < ringcount::N_COLS; ++c)
      std::printf("%s%s", ringcount::COLS[c].name, c + 1 == ringcount::N_COLS ? "\n" : " ");
    return 0;
  }
  const char *path = argc > 2 ? argv[2] : "cpp/topo3_mols.txt";
  std::vector<topo_io::Rec> recs = topo_io::load(path);
  ringcount::Mol m;
  ringcount::Scratch S;
  double out[ringcount::N_COLS];

  if (!std::strcmp(cmd, "dump")) {
    FILE *f = std::fopen(argc > 3 ? argv[3] : "cpp/topo3_ringcount.txt", "w");
    if (!f) { std::fprintf(stderr, "cannot write output\n"); return 2; }
    for (const auto &r : recs) {
      fill(m, r);
      ringcount::compute(m, out, S);
      for (int c = 0; c < ringcount::N_COLS; ++c)
        std::fprintf(f, c ? " %.17g" : "%.17g", out[c]);
      std::fputc('\n', f);
    }
    std::fclose(f);
    std::printf("ringcount: wrote %zu molecules x %d columns\n", recs.size(), ringcount::N_COLS);
    return 0;
  }

  if (!std::strcmp(cmd, "bench")) {
    // CONTENDED. Several jobs share this machine, so the number is an upper bound on a quiet
    // one. Every column is consumed into `sink` because -O3 will otherwise delete the half of
    // the work nobody reads. Mol construction is INSIDE the timed loop -- it is per-molecule
    // work the caller would also pay -- but the file parse is not.
    std::vector<double> reps;
    double sink = 0;
    for (int rep = 0; rep < 11; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      for (const auto &r : recs) {
        fill(m, r);
        ringcount::compute(m, out, S);
        for (int c = 0; c < ringcount::N_COLS; ++c) sink += out[c];
      }
      auto t1 = std::chrono::steady_clock::now();
      reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                     (double)recs.size());
    }
    std::sort(reps.begin(), reps.end());
    std::printf("ringcount  %2d cols  %zu mols  median %.3f us/mol  min %.3f  max %.3f  CONTENDED\n",
                ringcount::N_COLS, recs.size(), reps[reps.size() / 2], reps.front(), reps.back());
    if (sink == 12345.6789) std::printf("");
    return 0;
  }
  std::fprintf(stderr, "usage: ringcount [dump IN OUT | bench IN | names]\n");
  return 1;
}
