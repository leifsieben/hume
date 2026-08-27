// Standalone harness for src/hume_core/constit.h.
//
//     c++ -O2 -std=c++17 -o cpp/constit cpp/constit.cpp
//     python cpp/verify_constit.py --dump 100000      (in the PINNED oracle env)
//     ./cpp/constit cpp/constit_in.txt cpp/constit_cpp.txt
//     python cpp/verify_constit.py --compare
//
// Reads cpp/constit_in.txt, which cpp/verify_constit.py writes in the BOUNDARY's strided-row
// layout, so this harness exercises `constit::Mol::build_from_rows` -- the very function
// bindings.cpp will call -- rather than a second loader written only for the test.  Every double
// in the file is written %.17g, which round-trips a float64 exactly, so the exchange format has
// no precision of its own and a difference in the comparison is a difference in the arithmetic.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>

#include "../src/hume_core/constit.h"

namespace {

struct Rec {
  int n = 0, nb = 0, nr = 0, nhadd = 0;
  std::vector<int32_t> arows, brows, rptr, rat, stAtom, stBond;
  std::vector<double> adbl, bdbl, hchg;
  bool haveChg = false;
  double molLogP = 0, molMR = 0, tpsa = 0, naRing = 0, nARing = 0;
  int nHBDon = 0, nHBAcc = 0, nRot = 0, qedAlerts = -1;
};

void die(const char* msg, int k) {
  std::fprintf(stderr, "constit harness: %s (molecule %d)\n", msg, k);
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: constit <in.txt> <out.txt>\n"
                         "       constit <in.txt> --bench <reps>\n");
    return 2;
  }
  const bool bench = std::strcmp(argv[2], "--bench") == 0;
  const int reps = bench ? (argc > 3 ? std::atoi(argv[3]) : 7) : 1;
  // The drift guard runs BEFORE any number is produced, so a hand-edited table cannot reach the
  // comparison and be argued about afterwards.
  constit::checkSpec();

  FILE* f = std::fopen(argv[1], "r");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  int nm = 0;
  if (std::fscanf(f, "%d", &nm) != 1) { std::fprintf(stderr, "bad header\n"); return 1; }

  FILE* o = 0;
  if (!bench) {
    o = std::fopen(argv[2], "w");
    if (!o) { std::fprintf(stderr, "cannot open %s for writing\n", argv[2]); return 1; }
    // Column names first, so the comparison cannot line up the wrong pair of columns.
    for (int c = 0; c < constit::N_COLS; ++c)
      std::fprintf(o, "%s%s", constit::col_name(c), c + 1 == constit::N_COLS ? "\n" : " ");
  }

  std::vector<double> out(constit::N_COLS);
  constit::Mol m;
  int nthrew = 0;
  std::vector<Rec> all;
  if (bench) all.reserve(nm);
  for (int k = 0; k < nm; ++k) {
    Rec r;
    int hasChg = 0;
    if (std::fscanf(f, "%d %d %d %d %d", &r.n, &r.nb, &r.nr, &r.nhadd, &hasChg) != 5)
      die("truncated record header", k);
    r.arows.assign((size_t)r.n * 10, 0);
    r.adbl.assign((size_t)r.n * 2, 0.0);
    for (int i = 0; i < r.n; ++i) {
      int32_t* a = &r.arows[(size_t)i * 10];
      double mass = 0;
      if (std::fscanf(f, "%d %d %d %d %d %d %d %d %d %d %lf",
                      &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7], &a[8], &a[9],
                      &mass) != 11)
        die("bad atom row", k);
      r.adbl[(size_t)i * 2 + 0] = mass;
    }
    r.brows.assign((size_t)r.nb * 5, 0);
    r.bdbl.assign((size_t)r.nb, 0.0);
    for (int e = 0; e < r.nb; ++e) {
      int32_t* b = &r.brows[(size_t)e * 5];
      if (std::fscanf(f, "%d %d %d %d %d %lf", &b[0], &b[1], &b[2], &b[3], &b[4], &r.bdbl[e]) != 6)
        die("bad bond row", k);
    }
    r.rptr.assign(1, 0);
    for (int q = 0; q < r.nr; ++q) {
      int len = 0;
      if (std::fscanf(f, "%d", &len) != 1) die("bad ring length", k);
      for (int t = 0; t < len; ++t) {
        int a = 0;
        if (std::fscanf(f, "%d", &a) != 1) die("bad ring atom", k);
        r.rat.push_back(a);
      }
      r.rptr.push_back((int32_t)r.rat.size());
    }
    if (hasChg) {
      r.haveChg = true;
      r.hchg.resize((size_t)r.n + r.nhadd);
      for (size_t i = 0; i < r.hchg.size(); ++i)
        if (std::fscanf(f, "%lf", &r.hchg[i]) != 1) die("bad hchg", k);
    }
    r.stAtom.resize(r.n);
    for (int i = 0; i < r.n; ++i)
      if (std::fscanf(f, "%d", &r.stAtom[i]) != 1) die("bad stereoAtom", k);
    // One slot minimum so `.data()` is never null on a bondless molecule -- a null there would
    // silently turn SPS into NaN instead of into a number, which is the wrong kind of failure.
    r.stBond.assign(r.nb ? r.nb : 1, 0);
    for (int e = 0; e < r.nb; ++e)
      if (std::fscanf(f, "%d", &r.stBond[e]) != 1) die("bad stereoBond", k);
    if (std::fscanf(f, "%lf %lf %lf %lf %lf %d %d %d %d",
                    &r.molLogP, &r.molMR, &r.tpsa, &r.naRing, &r.nARing,
                    &r.nHBDon, &r.nHBAcc, &r.nRot, &r.qedAlerts) != 9)
      die("bad inputs line", k);

    m.build_from_rows(r.n, r.arows.data(), 10, r.adbl.data(), 2,
                      r.nb, r.brows.data(), 5, r.bdbl.data(),
                      r.nr, r.rptr.data(), r.rat.data());
    constit::Inputs in;
    in.molLogP = r.molLogP; in.molMR = r.molMR;
    in.nHBDon = r.nHBDon; in.nHBAcc = r.nHBAcc; in.nRot = r.nRot;
    in.naRing = r.naRing; in.nARing = r.nARing;
    if (r.haveChg) { in.hchg = r.hchg.data(); in.nhchg = (int)r.hchg.size(); }
    in.qedAlerts = r.qedAlerts;
    in.stereoAtom = r.stAtom.empty() ? 0 : r.stAtom.data();
    in.stereoBond = r.stBond.data();

    if (bench) { all.push_back(r); continue; }

    // constit.h THROWS rather than guessing when its nBondsKD kekule reconstruction does not
    // hold.  That is the right behaviour in the extension; here it is caught per molecule so a
    // single bad record names itself in the comparison instead of aborting a 100,000-molecule
    // run and leaving nothing to look at.
    try {
      constit::compute(m, in, out.data(), r.tpsa);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "constit: molecule %d threw: %s\n", k, e.what());
      ++nthrew;
      for (int c = 0; c < constit::N_COLS; ++c) out[c] = std::nan("");
    }
    for (int c = 0; c < constit::N_COLS; ++c)
      std::fprintf(o, "%.17g%s", out[c], c + 1 == constit::N_COLS ? "\n" : " ");
  }
  std::fclose(f);
  if (nthrew) std::fprintf(stderr, "constit: %d molecules threw\n", nthrew);

  if (bench) {
    // TIMED: build_from_rows + compute, i.e. everything constit.h does once the boundary arrays
    // exist.  The file parse above is NOT in the loop; it is the harness's, not the port's.  The
    // reps are reported with their spread because a contended machine is the usual case here and
    // a single number hides it.
    std::vector<double> us;
    double sink = 0.0;
    for (int rep = 0; rep < reps; ++rep) {
      const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
      for (size_t k = 0; k < all.size(); ++k) {
        Rec& r = all[k];
        m.build_from_rows(r.n, r.arows.data(), 10, r.adbl.data(), 2,
                          r.nb, r.brows.data(), 5, r.bdbl.data(),
                          r.nr, r.rptr.data(), r.rat.data());
        constit::Inputs in;
        in.molLogP = r.molLogP; in.molMR = r.molMR;
        in.nHBDon = r.nHBDon; in.nHBAcc = r.nHBAcc; in.nRot = r.nRot;
        in.naRing = r.naRing; in.nARing = r.nARing;
        if (r.haveChg) { in.hchg = r.hchg.data(); in.nhchg = (int)r.hchg.size(); }
        in.qedAlerts = r.qedAlerts;
        in.stereoAtom = r.stAtom.data();
        in.stereoBond = r.stBond.data();
        constit::compute(m, in, out.data(), r.tpsa);
        sink += out[0] + out[constit::N_COLS - 1];
      }
      const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / all.size());
    }
    double mean = 0.0;
    for (size_t i = 0; i < us.size(); ++i) mean += us[i];
    mean /= us.size();
    double sd = 0.0;
    for (size_t i = 0; i < us.size(); ++i) sd += (us[i] - mean) * (us[i] - mean);
    sd = us.size() > 1 ? std::sqrt(sd / (us.size() - 1)) : 0.0;
    std::printf("constit.h  %zu molecules x %d reps: %.3f +/- %.3f us/mol  "
                "(43 columns; sink %.3g)\n", all.size(), reps, mean, sd, sink);
    return 0;
  }

  std::fclose(o);
  std::fprintf(stderr, "constit: %d molecules -> %s\n", nm, argv[2]);
  return 0;
}
