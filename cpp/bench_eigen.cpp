// bench_eigen.cpp -- is a self-contained eigensolver good enough to drop HUME's LAPACK dependency?
//
// THE QUESTION. bcut2d calls dsytd2 + dsterf four times per molecule. Those symbols resolve to
// Accelerate on macOS and to OpenBLAS (or MKL) on Linux, and the same HUME source measures
// 138.09 us/mol against 218.93 us/mol depending only on which one it found. BCUT2D is ~57% of
// the C++ descriptor block, so the host's BLAS decides HUME's headline number. This program
// asks whether cpp/eigen_small.h -- Householder + implicit-shift QL/QR, no BLAS at all -- is
// close enough in speed to make that dependency deletable.
//
//   ./bench_eigen verify [mols.txt]   max abs/rel deviation from Accelerate, per size bucket
//   ./bench_eigen bench  [mols.txt]   three-way timing, bucketed, alternating-order pairs
//   ./bench_eigen solo   [mols.txt]   eigen_small only -- the -march=native comparison
//   ./bench_eigen share  [mols.txt]   does the shared off-diagonal structure survive step 1?
//
// The matrices are the REAL Burden matrices, built exactly as bcut_one builds them (heavy atoms
// only, 0.001 off the sparsity pattern, 1/sqrt(bond order) on bonds, property on the diagonal),
// so this is not a synthetic-matrix benchmark that agrees with the shipping one by luck.
//
// MEASURED UNDER CONTENTION. This machine is shared. Every comparison here is in-process and
// order-alternating (forward M,A,O then reverse O,A,M within each cycle, medians across cycles)
// because two separate process invocations cannot resolve differences at the few-percent level
// on this box -- hume.cpp records the unchanged dsyevd path reading 122.7 to 136.0 us/mol with
// no code change at all.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>

#include "eigen_small.h"

// Accelerate's, by direct linkage: the two stages bcut_one actually calls.
extern "C" {
void dsytd2_(char *, int *, double *, int *, double *, double *, double *, int *);
void dsterf_(int *, double *, double *, int *);
}

// OpenBLAS's, by dlopen -- BOTH LAPACKs in ONE process. Two builds in two runs cannot answer a
// few-percent question on a shared machine; the same loop alternating between two function
// pointers can. The handle keeps OpenBLAS's internal symbols in its own namespace, so
// Accelerate's dsterf is not silently interposed on OpenBLAS's dsytd2 or the reverse -- checked
// by confirming the two produce slightly DIFFERENT last digits, which identical code could not.
typedef void (*sytd2_fn)(char *, int *, double *, int *, double *, double *, double *, int *);
typedef void (*sterf_fn)(int *, double *, double *, int *);
static sytd2_fn ob_sytd2 = nullptr;
static sterf_fn ob_sterf = nullptr;

static bool load_openblas() {
  const char *cands[] = {"/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
                         "/usr/local/opt/openblas/lib/libopenblas.dylib",
                         "libopenblas.so.0", "libopenblas.so"};
  for (const char *p : cands) {
    void *h = dlopen(p, RTLD_LAZY | RTLD_LOCAL);
    if (!h) continue;
    ob_sytd2 = (sytd2_fn)dlsym(h, "dsytd2_");
    ob_sterf = (sterf_fn)dlsym(h, "dsterf_");
    if (ob_sytd2 && ob_sterf) { printf("  openblas: %s\n", p); return true; }
  }
  return false;
}

// ------------------------------------------------------------------------------- corpus

struct Mol {
  int n = 0, nb = 0;
  std::vector<int> Z, bu, bv;
  std::vector<double> mass, gast, clogp, cmr, bord;
};

// Same on-disk format export_predict.py writes and hume.cpp's load() reads; the fields this
// program does not need are read and dropped rather than skipped by position, so a format
// change desyncs loudly instead of quietly shifting a column.
static std::vector<Mol> load(const char *path) {
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  int nm = 0;
  f >> nm;
  std::vector<Mol> ms(nm);
  for (int k = 0; k < nm; k++) {
    Mol &m = ms[k];
    int chg_ok;
    f >> m.n >> m.nb >> chg_ok;
    m.Z.resize(m.n); m.mass.resize(m.n); m.gast.resize(m.n);
    m.clogp.resize(m.n); m.cmr.resize(m.n);
    for (int i = 0; i < m.n; i++) {
      int deg, nH, fchg, hyb, arom, ring, cip;
      f >> m.Z[i] >> deg >> nH >> fchg >> hyb >> arom >> ring >> m.mass[i] >> m.gast[i]
        >> m.clogp[i] >> m.cmr[i] >> cip;
    }
    m.bu.resize(m.nb); m.bv.resize(m.nb); m.bord.resize(m.nb);
    for (int b = 0; b < m.nb; b++) {
      int bconj, bring, bstereo;
      f >> m.bu[b] >> m.bv[b] >> m.bord[b] >> bconj >> bring >> bstereo;
    }
  }
  return ms;
}

struct Mat {
  int n;
  std::vector<double> a;                 // n x n, symmetric, full (row-major == col-major)
};

// bcut_one's construction, verbatim: heavy atoms only (explicit H are isotope labels and are
// not in RDKit's Burden matrix), 0.001 everywhere, property on the diagonal, 1/sqrt(bond order)
// on bonds. Deviating here would benchmark a different matrix than the one that ships.
static bool burden(const Mol &m, const std::vector<double> &prop, Mat &out) {
  std::vector<int> hv, pos(m.n, -1);
  for (int i = 0; i < m.n; i++)
    if (m.Z[i] != 1) { pos[i] = (int)hv.size(); hv.push_back(i); }
  const int n = (int)hv.size();
  if (n < 2) return false;
  out.n = n;
  out.a.assign((size_t)n * n, 0.001);
  for (int i = 0; i < n; i++) out.a[(size_t)i * n + i] = prop[hv[i]];
  for (int b = 0; b < m.nb; b++) {
    int i = pos[m.bu[b]], j = pos[m.bv[b]];
    if (i < 0 || j < 0) continue;
    const double v = 1.0 / std::sqrt(m.bord[b]);
    out.a[(size_t)i * n + j] = v;
    out.a[(size_t)j * n + i] = v;
  }
  return true;
}

// ------------------------------------------------------------------------------- solvers

struct LapWork { std::vector<double> a, d, e, tau; };

static void lap_extremal(const Mat &M, LapWork &W, sytd2_fn f_sytd2, sterf_fn f_sterf,
                         double *lo, double *hi) {
  const int n = M.n;
  if ((int)W.a.size() < (size_t)n * n) W.a.resize((size_t)n * n);
  if ((int)W.d.size() < n) { W.d.resize(n); W.e.resize(n); W.tau.resize(n); }
  std::memcpy(W.a.data(), M.a.data(), sizeof(double) * (size_t)n * n);
  int nn = n, lda = n, info = 0;
  char uplo = 'U';
  f_sytd2(&uplo, &nn, W.a.data(), &lda, W.d.data(), W.e.data(), W.tau.data(), &info);
  if (info == 0) f_sterf(&nn, W.d.data(), W.e.data(), &info);
  if (info != 0) { *lo = *hi = 0.0; return; }
  *lo = W.d[0];
  *hi = W.d[n - 1];               // dsterf sorts ascending
}

// ------------------------------------------------------------------------------- buckets

static const int NB = 6;
static const int BLO[NB] = {0, 16, 23, 31, 46, 71};
static const int BHI[NB] = {15, 22, 30, 45, 70, 1 << 30};
static const char *BNAME[NB] = {"n<=15", "16-22", "23-30", "31-45", "46-70", "71+"};
static int bucket_of(int n) {
  for (int b = 0; b < NB; b++)
    if (n >= BLO[b] && n <= BHI[b]) return b;
  return NB - 1;
}

static double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t k = v.size();
  return k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
}

// ------------------------------------------------------------------------------- main

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "verify";
  const char *path = argc > 2 ? argv[2] : "mols.txt";
  const int cap = argc > 3 ? atoi(argv[3]) : 0;      // per-bucket matrix cap for bench modes

  printf("loading %s ...\n", path);
  std::vector<Mol> ms = load(path);
  printf("  %zu molecules\n", ms.size());

  const bool have_ob = (mode == "bench") ? load_openblas() : false;
  if (mode == "bench" && !have_ob) printf("  openblas: NOT FOUND -- two-way only\n");

  // Build every Burden matrix once, bucketed. Four per molecule (mass, charge, logP, MR),
  // which is exactly what bcut2d solves.
  std::vector<std::vector<Mat>> buck(NB);
  std::vector<size_t> want(NB, cap ? (size_t)cap : (size_t)-1);
  // POP is the TRUE corpus population of each bucket, counted before the sampling cap. The cap
  // exists so every bucket gets enough matrices to time (71+ is 2.9% of molecules but 1500x the
  // cost of the smallest bucket, so an uncapped run is 20 minutes of mostly one bucket). The
  // corpus-weighted line at the bottom therefore uses POP, not the sampled sizes -- reporting a
  // flat-sampled average as a corpus number is exactly the "single average hides it" failure.
  std::vector<size_t> pop(NB, 0);
  {
    Mat tmp;
    for (const Mol &m : ms) {
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      for (int p = 0; p < 4; p++) {
        if (!burden(m, *props[p], tmp)) continue;
        const int b = bucket_of(tmp.n);
        pop[b]++;
        if (buck[b].size() >= want[b]) continue;
        buck[b].push_back(tmp);
      }
    }
  }
  size_t tot = 0, totpop = 0;
  for (int b = 0; b < NB; b++) { tot += buck[b].size(); totpop += pop[b]; }
  printf("  %zu Burden matrices in the corpus, %zu sampled for timing\n\n", totpop, tot);

  hume_eig::Work EW;
  LapWork LW;

  // ------------------------------------------------------------------ correctness
  if (mode == "verify") {
    // THE GATE IS NOT A BARE RELATIVE TOLERANCE. verify_hume.py tests
    //     |a - b| <= atol + rtol*|b|,  atol = 1e-12, rtol = 1e-9
    // so a huge RELATIVE deviation on a value of 1.4e-05 is not a failure and reporting it as
    // one would be theatre. `gate` below is |a-b| divided by the budget that test allows: 1.0
    // means exactly on the line. Max relative deviation is reported too, because it is the
    // number a reader will want, but it is not the criterion.
    const double ATOL = 1e-12, RTOL = 1e-9;
    printf("%-8s %8s  %12s %12s %12s %12s   %s\n", "bucket", "mats", "max|dlo|", "max|dhi|",
           "max rel", "max gate", "value at max rel");
    double gabs = 0, grel = 0, ggate = 0;
    long long fails = 0;
    for (int b = 0; b < NB; b++) {
      double mabs_lo = 0, mabs_hi = 0, mrel = 0, mgate = 0, worst = 0;
      for (const Mat &M : buck[b]) {
        double alo, ahi, mlo, mhi;
        lap_extremal(M, LW, dsytd2_, dsterf_, &alo, &ahi);
        if (!hume_eig::extremal(M.a.data(), M.n, &mlo, &mhi, EW)) { fails++; continue; }
        const double dlo = std::fabs(mlo - alo), dhi = std::fabs(mhi - ahi);
        mabs_lo = std::max(mabs_lo, dlo);
        mabs_hi = std::max(mabs_hi, dhi);
        mgate = std::max(mgate, dlo / (ATOL + RTOL * std::fabs(alo)));
        mgate = std::max(mgate, dhi / (ATOL + RTOL * std::fabs(ahi)));
        const double rl = alo != 0 ? dlo / std::fabs(alo) : dlo;
        const double rh = ahi != 0 ? dhi / std::fabs(ahi) : dhi;
        if (std::max(rl, rh) > mrel) { mrel = std::max(rl, rh); worst = rl > rh ? alo : ahi; }
      }
      gabs = std::max(gabs, std::max(mabs_lo, mabs_hi));
      grel = std::max(grel, mrel);
      ggate = std::max(ggate, mgate);
      printf("%-8s %8zu  %12.3e %12.3e %12.3e %12.3e   %g\n", BNAME[b], buck[b].size(),
             mabs_lo, mabs_hi, mrel, mgate, worst);
    }
    printf("\nOVERALL  max abs %.3e   max rel %.3e   non-convergences %lld\n", gabs, grel, fails);
    printf("gate usage (1.0 = exactly at verify_hume.py's limit): %.4f  -> %s\n", ggate,
           ggate < 1.0 ? "PASS" : "FAIL");
    printf("max abs deviation %.3e is %s the atol floor of %.0e alone, so the gate cannot be\n"
           "broken by this solver at ANY value magnitude.\n", gabs,
           gabs < ATOL ? "BELOW" : "ABOVE", ATOL);
    return 0;
  }

  // ------------------------------------------------------------------ shared-structure probe
  //
  // The four Burden matrices of one molecule share EVERY off-diagonal and differ only on the
  // diagonal. The tempting conclusion is that the Householder vectors are shared too and one
  // reduction could serve four solves. This measures rather than assumes it: reduce all four,
  // and compare the reflectors step by step.
  if (mode == "share") {
    Mat A[4];
    LapWork w[4];
    std::vector<double> scratch;
    int checked = 0;
    // Bucket the reflector disagreement by how many steps into the reduction we are. Step 0 is
    // the FIRST one performed (the dsytd2 loop runs from the bottom-right corner upward).
    std::vector<double> step_max(8, 0.0);
    std::vector<long long> step_cnt(8, 0);
    for (const Mol &m : ms) {
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      bool ok = true;
      for (int p = 0; p < 4 && ok; p++) ok = burden(m, *props[p], A[p]);
      if (!ok || A[0].n < 8) continue;
      const int n = A[0].n;
      if ((int)scratch.size() < n) scratch.resize(n);
      for (int p = 0; p < 4; p++) {
        if ((int)w[p].a.size() < n * n) w[p].a.resize((size_t)n * n);
        if ((int)w[p].d.size() < n) { w[p].d.resize(n); w[p].e.resize(n); w[p].tau.resize(n); }
        std::memcpy(w[p].a.data(), A[p].a.data(), sizeof(double) * (size_t)n * n);
        hume_eig::sytd2_upper(w[p].a.data(), n, n, w[p].d.data(), w[p].e.data(),
                              w[p].tau.data(), scratch.data());
      }
      // The reflector generated at step ii is left in column ii+1, rows 0..ii-1. Later steps
      // only touch columns 0..ii, so it survives the rest of the reduction unmodified.
      for (int s = 0; s < 8; s++) {
        const int ii = n - 2 - s;
        if (ii < 1) break;
        double dmax = 0;
        for (int k = 0; k < ii; k++)
          for (int p = 1; p < 4; p++)
            dmax = std::max(dmax, std::fabs(w[p].a[(size_t)(ii + 1) * n + k] -
                                            w[0].a[(size_t)(ii + 1) * n + k]));
        // tau too: a reflector is (v, tau), and equal v with different tau is still not shared.
        for (int p = 1; p < 4; p++)
          dmax = std::max(dmax, std::fabs(w[p].tau[ii] - w[0].tau[ii]));
        step_max[s] = std::max(step_max[s], dmax);
        step_cnt[s]++;
      }
      if (++checked >= 5000) break;
    }
    printf("shared-structure probe over %d molecules (n >= 8), %s\n", checked, path);
    printf("The four Burden matrices of one molecule share EVERY off-diagonal and differ only\n"
           "on the diagonal. Do they therefore share Householder reflectors?\n\n");
    printf("  %-6s %10s   %s\n", "step", "mols", "max |(v,tau)_p - (v,tau)_mass| over p=1..3");
    for (int s = 0; s < 8; s++) {
      if (!step_cnt[s]) break;
      printf("  %-6d %10lld   %.3e%s\n", s, step_cnt[s], step_max[s],
             s == 0 ? "   <- first step performed" : "");
    }
    return 0;
  }

  // ------------------------------------------------------------------ timing
  const int cycles = argc > 4 ? atoi(argv[4]) : 9;
  auto run_mine = [&](const std::vector<Mat> &v) {
    double s = 0, lo, hi;
    for (const Mat &M : v) { hume_eig::extremal(M.a.data(), M.n, &lo, &hi, EW); s += lo + hi; }
    return s;
  };
  auto run_lap = [&](const std::vector<Mat> &v, sytd2_fn f1, sterf_fn f2) {
    double s = 0, lo, hi;
    for (const Mat &M : v) { lap_extremal(M, LW, f1, f2, &lo, &hi); s += lo + hi; }
    return s;
  };
  volatile double sink = 0;

  if (mode == "solo") {
    printf("%-8s %8s %12s %12s\n", "bucket", "mats", "us/mat", "spread%");
    for (int b = 0; b < NB; b++) {
      if (buck[b].empty()) continue;
      std::vector<double> t;
      for (int c = 0; c < cycles; c++) {
        const double t0 = now_us();
        sink += run_mine(buck[b]);
        t.push_back((now_us() - t0) / buck[b].size());
      }
      std::sort(t.begin(), t.end());
      printf("%-8s %8zu %12.3f %12.1f\n", BNAME[b], buck[b].size(), median(t),
             100.0 * (t.back() - t.front()) / median(t));
    }
    return 0;
  }

  // Three-way. Within one cycle the order is M,A,O then O,A,M -- a palindrome, so no
  // implementation can be flattered by always running second on a warm cache or by a thermal
  // ramp inside the cycle. The per-cycle number is the mean of the two positions; the reported
  // number is the MEDIAN over cycles and the spread is min..max of those cycle medians.
  printf("%-8s %8s %8s | %11s %10s %10s | %9s %11s | %s\n", "bucket", "mats", "corpus%",
         "eigen_small", "Accel", "OpenBLAS", "vs Accel", "vs OpenBLAS", "cycle spread");
  double wm = 0, wa = 0, wo = 0;
  for (int b = 0; b < NB; b++) {
    if (buck[b].empty()) continue;
    std::vector<double> tm, ta, to;
    for (int c = 0; c < cycles; c++) {
      double t0, m1, a1, o1 = 0, m2, a2, o2 = 0;
      t0 = now_us(); sink += run_mine(buck[b]);                    m1 = now_us() - t0;
      t0 = now_us(); sink += run_lap(buck[b], dsytd2_, dsterf_);   a1 = now_us() - t0;
      if (have_ob) { t0 = now_us(); sink += run_lap(buck[b], ob_sytd2, ob_sterf); o1 = now_us() - t0; }
      if (have_ob) { t0 = now_us(); sink += run_lap(buck[b], ob_sytd2, ob_sterf); o2 = now_us() - t0; }
      t0 = now_us(); sink += run_lap(buck[b], dsytd2_, dsterf_);   a2 = now_us() - t0;
      t0 = now_us(); sink += run_mine(buck[b]);                    m2 = now_us() - t0;
      const double d = 2.0 * buck[b].size();
      tm.push_back((m1 + m2) / d);
      ta.push_back((a1 + a2) / d);
      if (have_ob) to.push_back((o1 + o2) / d);
    }
    const double M = median(tm), A = median(ta), O = have_ob ? median(to) : 0.0;
    wm += M * pop[b]; wa += A * pop[b]; wo += O * pop[b];
    std::sort(tm.begin(), tm.end());
    printf("%-8s %8zu %7.1f%% | %11.3f %10.3f %10.3f | %8.2fx %10.2fx | %+.0f%%/%+.0f%%\n",
           BNAME[b], buck[b].size(), 100.0 * pop[b] / totpop, M, A, O, A / M,
           have_ob ? O / M : 0.0, 100.0 * (tm.front() - M) / M, 100.0 * (tm.back() - M) / M);
  }
  printf("\nus per MATRIX; four matrices per molecule. Ratios > 1 mean eigen_small wins.\n");
  printf("cycle spread is min/max of the %d per-cycle medians against the median -- this box is\n"
         "SHARED, so treat anything inside that band as a tie.\n\n", cycles);
  printf("CORPUS-WEIGHTED (bucket medians x true corpus bucket populations):\n");
  printf("  per matrix:   eigen_small %7.3f   Accel %7.3f   OpenBLAS %7.3f us\n",
         wm / totpop, wa / totpop, wo / totpop);
  printf("  per molecule: eigen_small %7.2f   Accel %7.2f   OpenBLAS %7.2f us   "
         "(x4 matrices; this is the BCUT2D solve cost, excluding matrix fill)\n",
         4.0 * wm / totpop, 4.0 * wa / totpop, 4.0 * wo / totpop);
  return 0;
}
