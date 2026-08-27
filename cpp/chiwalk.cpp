// Standalone harness for src/hume_core/chi.h (40 columns) + src/hume_core/topomisc.h (15).
//
//   python cpp/verify_chiwalk.py all       builds cpp/chiwalk_mols.txt, runs this, compares
//   ./cpp/chiwalk dump  IN OUT             55 columns per molecule, %.17g
//   ./cpp/chiwalk bench IN                 contended timing, chi and topomisc separately
//   ./cpp/chiwalk names                    the 55 column names, in emit order
//
// THE EXCHANGE FILE (cpp/chiwalk_mols.txt) IS NOT cpp/topo3_mols.txt AND CANNOT BE. topo_io.h
// carries neither the formal charge nor the hydrogen count that mordred's `dv` needs, and -- the
// part that is easy to miss -- it does not promise that a bond's `u` is RDKit's BEGIN atom.
// chi.h's product order is first-appearance over (begin, end) pairs, so an orientation-agnostic
// exchange format would silently give a different answer in the last bits for every molecule
// containing a non-period-2 element. This file's `u v` are GetBeginAtomIdx()/GetEndAtomIdx() in
// bond index order, exactly as `bond_i` delivers them.
//
// No selfCheck(): the only tables involved are cpp/chiwalk_tables.h, which is GENERATED from the
// pinned RDKit and mordred rather than transcribed, and verify_chiwalk.py::check_spec() asserts
// the 55 column names against live mordred objects built from the same parameter tuples.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/hume_core/chi.h"
#include "../src/hume_core/topomisc.h"

// The boundary's strides, so the harness exercises the very builders bindings.cpp will call.
static constexpr int ASTRIDE = 10;   // atom_i: Z deg nH fchg hyb arom ring cip nring tval
static constexpr int BSTRIDE = 5;    // bond_i: u v conj ring code

struct Rec {
  int n = 0, nb = 0;
  std::vector<int32_t> arows, brows;
};

static std::vector<Rec> load(const char *path) {
  FILE *f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s (run: python cpp/verify_chiwalk.py dump)\n", path);
    std::exit(1);
  }
  int nm = 0;
  if (std::fscanf(f, "%d", &nm) != 1) { std::fprintf(stderr, "%s: bad header\n", path); std::exit(1); }
  std::vector<Rec> out(nm);
  for (int k = 0; k < nm; ++k) {
    Rec &r = out[k];
    if (std::fscanf(f, "%d %d", &r.n, &r.nb) != 2) {
      std::fprintf(stderr, "%s: truncated at molecule %d\n", path, k); std::exit(1);
    }
    r.arows.assign((size_t)r.n * ASTRIDE, 0);
    for (int i = 0; i < r.n; ++i) {
      int z = 0, fchg = 0, nh = 0;
      if (std::fscanf(f, "%d %d %d", &z, &fchg, &nh) != 3) {
        std::fprintf(stderr, "%s: bad atom %d of molecule %d\n", path, i, k); std::exit(1);
      }
      r.arows[(size_t)i * ASTRIDE + 0] = z;      // Z
      r.arows[(size_t)i * ASTRIDE + 2] = nh;     // nH   = GetTotalNumHs()
      r.arows[(size_t)i * ASTRIDE + 3] = fchg;   // fchg = GetFormalCharge()
    }
    r.brows.assign((size_t)r.nb * BSTRIDE, 0);
    for (int b = 0; b < r.nb; ++b) {
      int u = 0, v = 0;
      if (std::fscanf(f, "%d %d", &u, &v) != 2) {
        std::fprintf(stderr, "%s: bad bond %d of molecule %d\n", path, b, k); std::exit(1);
      }
      r.brows[(size_t)b * BSTRIDE + 0] = u;
      r.brows[(size_t)b * BSTRIDE + 1] = v;
    }
  }
  std::fclose(f);
  return out;
}

static const int N_ALL = chisub::N_COLS + topomisc::N_COLS;

int main(int argc, char **argv) {
  const char *cmd = argc > 1 ? argv[1] : "names";
  if (!std::strcmp(cmd, "names")) {
    for (int c = 0; c < chisub::N_COLS; ++c) std::printf("%s ", chisub::COLS[c].name);
    for (int c = 0; c < topomisc::N_COLS; ++c)
      std::printf("%s%s", topomisc::COLS[c], c + 1 == topomisc::N_COLS ? "\n" : " ");
    return 0;
  }
  // Two transliterations in topomisc.h are checked against the library they copy rather than
  // against a molecule, because a numpy summation order and a libm log are not properties of any
  // molecule and would otherwise be verified only by luck. verify_chiwalk.py::selftest() drives
  // both. If either of these reports a mismatch, nothing below is worth reading.
  if (!std::strcmp(cmd, "pwtest")) {
    FILE *f = std::fopen(argv[2], "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    int nc = 0;
    if (std::fscanf(f, "%d", &nc) != 1) return 2;
    std::vector<double> a;
    for (int c = 0; c < nc; ++c) {
      int L = 0;
      if (std::fscanf(f, "%d", &L) != 1) return 2;
      a.resize(L);
      for (int i = 0; i < L; ++i)
        if (std::fscanf(f, "%lf", &a[i]) != 1) return 2;
      std::printf("%.17g\n", topomisc::npPairwiseSum(a.data(), L));
    }
    std::fclose(f);
    return 0;
  }
  if (!std::strcmp(cmd, "logtest")) {
    FILE *f = std::fopen(argv[2], "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    int nc = 0;
    if (std::fscanf(f, "%d", &nc) != 1) return 2;
    for (int c = 0; c < nc; ++c) {
      long long v = 0;
      if (std::fscanf(f, "%lld", &v) != 1) return 2;
      std::printf("%.17g\n", std::log((double)v));
    }
    std::fclose(f);
    return 0;
  }

  const char *path = argc > 2 ? argv[2] : "cpp/chiwalk_mols.txt";
  std::vector<Rec> recs = load(path);
  chisub::Mol cm; chisub::Scratch cs;
  topomisc::Mol tm; topomisc::Scratch ts;
  std::vector<double> out(N_ALL);

  if (!std::strcmp(cmd, "dump")) {
    FILE *f = std::fopen(argc > 3 ? argv[3] : "cpp/chiwalk_cpp.txt", "w");
    if (!f) { std::fprintf(stderr, "cannot write output\n"); return 2; }
    for (const auto &r : recs) {
      chisub::build_from_rows(cm, r.n, r.nb, r.arows.data(), ASTRIDE, r.brows.data(), BSTRIDE);
      chisub::compute(cm, out.data(), cs);
      topomisc::build_from_rows(tm, r.n, r.nb, r.arows.data(), ASTRIDE, r.brows.data(), BSTRIDE);
      topomisc::compute(tm, out.data() + chisub::N_COLS, ts);
      for (int c = 0; c < N_ALL; ++c) std::fprintf(f, c ? " %.17g" : "%.17g", out[c]);
      std::fputc('\n', f);
    }
    std::fclose(f);
    std::printf("chiwalk: wrote %zu molecules x %d columns\n", recs.size(), N_ALL);
    return 0;
  }

  if (!std::strcmp(cmd, "bench")) {
    // CONTENDED; see the note in cpp/ringcount.cpp. build() is inside the timed loop because it
    // is per-molecule work the caller would also pay; the file parse is not.
    const char *tag[2] = {"chi     ", "topomisc"};
    const int ncol[2] = {chisub::N_COLS, topomisc::N_COLS};
    for (int which = 0; which < 2; ++which) {
      std::vector<double> reps;
      double sink = 0;
      for (int rep = 0; rep < 11; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (const auto &r : recs) {
          if (which == 0) {
            chisub::build_from_rows(cm, r.n, r.nb, r.arows.data(), ASTRIDE, r.brows.data(), BSTRIDE);
            chisub::compute(cm, out.data(), cs);
          } else {
            topomisc::build_from_rows(tm, r.n, r.nb, r.arows.data(), ASTRIDE, r.brows.data(),
                                      BSTRIDE);
            topomisc::compute(tm, out.data(), ts);
          }
          for (int c = 0; c < ncol[which]; ++c) sink += out[c];
        }
        auto t1 = std::chrono::steady_clock::now();
        reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                       (double)recs.size());
      }
      std::sort(reps.begin(), reps.end());
      std::printf("%s  %2d cols  %zu mols  median %.3f us/mol  min %.3f  max %.3f  CONTENDED\n",
                  tag[which], ncol[which], recs.size(), reps[reps.size() / 2], reps.front(),
                  reps.back());
      if (sink == 12345.6789) std::printf("");
    }
    return 0;
  }
  std::fprintf(stderr, "usage: chiwalk [dump IN OUT | bench IN | names]\n");
  return 1;
}
