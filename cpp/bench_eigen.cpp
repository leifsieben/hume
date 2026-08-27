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
//   ./bench_eigen march  [mols.txt]   -O3 vs -mcpu=apple-m1 vs -march=native, in one process
//   ./bench_eigen solo   [mols.txt]   eigen_small only
//   ./bench_eigen adv    [adv.txt]    per-molecule deviation on the adversarial set, named
//   ./bench_eigen share  [mols.txt]   does the shared off-diagonal structure survive step 1?
//
// BUILD (the three variant objects exist only for `march` mode and are always linked):
//   clang++ -O3               -std=c++17 -DVNS=v_plain  -DVFN=eig_plain  -c eigen_variant.cpp -o v_plain.o
//   clang++ -O3 -mcpu=apple-m1 -std=c++17 -DVNS=v_m1     -DVFN=eig_m1     -c eigen_variant.cpp -o v_m1.o
//   clang++ -O3 -march=native -std=c++17 -DVNS=v_native -DVFN=eig_native -c eigen_variant.cpp -o v_native.o
//   clang++ -O3 -march=native -std=c++17 bench_eigen.cpp v_plain.o v_m1.o v_native.o \
//           -o bench_eigen -framework Accelerate
//
// Usage: ./bench_eigen <mode> <mols.txt> [per-bucket sample cap] [timing cycles]
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

// The same header compiled three ways (see eigen_variant.cpp), for `march` mode. These are
// always linked in -- a weak reference was tried first and Mach-O does not support a weak
// UNDEFINED symbol from a static object, only from a dylib, so the "build without the variants
// and skip that mode" convenience is not available. Three extra objects is the cheaper answer.
extern "C" {
bool eig_plain(const double *, int, double *, double *);
bool eig_m1(const double *, int, double *, double *);
bool eig_native(const double *, int, double *, double *);
}

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
    // Two passes. The first counts, the second takes every k-th matrix of each bucket so the
    // sample spans the whole corpus rather than being a prefix of it -- the 71+ bucket runs
    // from n = 71 to n = 245 and costs ~n^3, so a prefix sample can be 40% off on the one
    // bucket that carries 46% of the time.
    Mat tmp;
    for (const Mol &m : ms) {
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      for (int p = 0; p < 4; p++) {
        if (!burden(m, *props[p], tmp)) continue;
        pop[bucket_of(tmp.n)]++;
      }
    }
    std::vector<size_t> stride(NB, 1), seen(NB, 0);
    for (int b = 0; b < NB; b++)
      if (want[b] != (size_t)-1 && pop[b] > want[b]) stride[b] = pop[b] / want[b];
    for (const Mol &m : ms) {
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      for (int p = 0; p < 4; p++) {
        if (!burden(m, *props[p], tmp)) continue;
        const int b = bucket_of(tmp.n);
        const size_t k = seen[b]++;
        if (buck[b].size() >= want[b] || k % stride[b]) continue;
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

  // ------------------------------------------------------------------ adversarial, per molecule
  //
  // Aggregate maxima hide WHICH molecule is worst, and for this question that is the whole
  // point: degenerate spectra from high-symmetry cages and near-degenerate frontier pairs from
  // polyacenes and long polyenes are exactly where a QL sweep is supposed to struggle. Names
  // come from the .smi written alongside the .txt by export_predict.py, so a row can be read
  // rather than merely counted.
  if (mode == "adv") {
    std::string smipath(path);
    const size_t dot = smipath.rfind('.');
    if (dot != std::string::npos) smipath = smipath.substr(0, dot) + ".smi";
    std::vector<std::string> smi;
    { std::ifstream sf(smipath); std::string ln;
      while (std::getline(sf, ln)) if (!ln.empty()) smi.push_back(ln); }
    const double ATOL = 1e-12, RTOL = 1e-9;
    printf("%-4s %5s %6s | %11s %11s %9s | %s\n", "#", "nheavy", "mats", "max abs", "max rel",
           "max gate", "SMILES");
    Mat tmp;
    double gworst = 0;
    for (size_t k = 0; k < ms.size(); k++) {
      const Mol &m = ms[k];
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      double mabs = 0, mrel = 0, mgate = 0;
      int nh = 0, nmat = 0;
      for (int p = 0; p < 4; p++) {
        if (!burden(m, *props[p], tmp)) continue;
        nh = tmp.n; nmat++;
        double alo, ahi, mlo, mhi;
        lap_extremal(tmp, LW, dsytd2_, dsterf_, &alo, &ahi);
        if (!hume_eig::extremal(tmp.a.data(), tmp.n, &mlo, &mhi, EW)) {
          printf("%-4zu %5d %6d | NON-CONVERGENCE\n", k, tmp.n, nmat); continue;
        }
        const double dlo = std::fabs(mlo - alo), dhi = std::fabs(mhi - ahi);
        mabs = std::max(mabs, std::max(dlo, dhi));
        if (alo != 0) mrel = std::max(mrel, dlo / std::fabs(alo));
        if (ahi != 0) mrel = std::max(mrel, dhi / std::fabs(ahi));
        mgate = std::max(mgate, dlo / (ATOL + RTOL * std::fabs(alo)));
        mgate = std::max(mgate, dhi / (ATOL + RTOL * std::fabs(ahi)));
      }
      gworst = std::max(gworst, mgate);
      std::string lbl = k < smi.size() ? smi[k] : std::string("(no smi)");
      if (lbl.size() > 58) lbl = lbl.substr(0, 55) + "...";
      printf("%-4zu %5d %6d | %11.3e %11.3e %9.3e | %s\n", k, nh, nmat, mabs, mrel, mgate,
             lbl.c_str());
    }
    printf("\nworst gate usage on the adversarial set: %.3e (1.0 = at verify_hume.py's limit)\n",
           gworst);
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

  if (mode == "march") {
    typedef bool (*eig_fn)(const double *, int, double *, double *);
    eig_fn fns[3] = {eig_plain, eig_m1, eig_native};
    const char *fnm[3] = {"-O3", "-O3 -mcpu=apple-m1", "-O3 -march=native"};
    auto run_v = [&](const std::vector<Mat> &v, eig_fn f) {
      double s = 0, lo, hi;
      for (const Mat &M : v) { f(M.a.data(), M.n, &lo, &hi); s += lo + hi; }
      return s;
    };
    // Correctness first: three code generations of the same source must agree BIT FOR BIT, or
    // -march=native is not a free speed knob, it is a second numerical path.
    {
      long long diff = 0, tot2 = 0;
      double lo0, hi0, lo1, hi1, lo2, hi2;
      for (int b = 0; b < NB; b++)
        for (const Mat &M : buck[b]) {
          eig_plain(M.a.data(), M.n, &lo0, &hi0);
          eig_m1(M.a.data(), M.n, &lo1, &hi1);
          eig_native(M.a.data(), M.n, &lo2, &hi2);
          tot2++;
          if (lo0 != lo1 || hi0 != hi1 || lo0 != lo2 || hi0 != hi2) diff++;
        }
      printf("bit-identical across the three builds: %lld of %lld matrices differ\n\n",
             diff, tot2);
    }
    printf("%-8s %6s %6s | %10s %10s %10s | %11s %11s\n", "bucket", "mats", "mean n",
           fnm[0], "apple-m1", "native", "m1/plain", "native/plain");
    for (int b = 0; b < NB; b++) {
      if (buck[b].empty()) continue;
      double mean_n = 0;
      for (const Mat &M : buck[b]) mean_n += M.n;
      mean_n /= buck[b].size();
      std::vector<double> t[3], r1, r2;
      for (int c = 0; c < cycles; c++) {
        double f[3], g[3], t0;
        for (int i = 0; i < 3; i++) { t0 = now_us(); sink += run_v(buck[b], fns[i]); f[i] = now_us() - t0; }
        for (int i = 2; i >= 0; i--) { t0 = now_us(); sink += run_v(buck[b], fns[i]); g[i] = now_us() - t0; }
        const double d = 2.0 * buck[b].size();
        for (int i = 0; i < 3; i++) t[i].push_back((f[i] + g[i]) / d);
        r1.push_back(t[1].back() / t[0].back());
        r2.push_back(t[2].back() / t[0].back());
      }
      std::sort(r1.begin(), r1.end());
      std::sort(r2.begin(), r2.end());
      printf("%-8s %6zu %6.1f | %10.3f %10.3f %10.3f | %5.2fx[%.2f-%.2f] %5.2fx[%.2f-%.2f]\n",
             BNAME[b], buck[b].size(), mean_n, median(t[0]), median(t[1]), median(t[2]),
             median(r1), r1.front(), r1.back(), median(r2), r2.front(), r2.back());
    }
    printf("\nRatios BELOW 1.00 mean the tuned build is FASTER than plain -O3. The bracket is\n"
           "min-max over %d per-cycle ratios on a machine at load average ~28.\n\n", cycles);
    printf("READ THIS AS A NULL EXPERIMENT. On arm64 macOS all three flag sets resolve to the\n"
           "SAME -target-cpu (apple-m1 -- clang does not know the M4 Pro and falls back) and the\n"
           "SAME 27 target features, and compiling eigen_variant.cpp three ways with identical\n"
           "symbol names produces BYTE-IDENTICAL object files. So the three columns above are\n"
           "three copies of one binary, and every difference in them is measurement noise.\n"
           "That makes this table the harness's NOISE FLOOR: whatever spread appears here is\n"
           "what `bench` mode cannot distinguish from zero. Treat any bench ratio inside this\n"
           "band as a tie.\n"
           "It also answers the portability question for this platform outright -- -march=native\n"
           "buys nothing here and can be dropped. It says NOTHING about x86-64, where native\n"
           "unlocks AVX2/AVX-512 and the answer may well differ; that has not been measured.\n");
    return 0;
  }

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

  // Three-way. Within one cycle the order is M,A,O,O,A,M -- a palindrome, so no implementation
  // can be flattered by always running second on a warm cache or by a thermal ramp inside the
  // cycle.
  //
  // THE REPORTED NUMBER IS A MEDIAN OF PER-CYCLE RATIOS, NOT A RATIO OF MEDIANS. That is not
  // pedantry on this box: load average during these runs was 28 on 12 cores, other people's
  // jobs, and absolute microsecond figures move by 3-5x between minutes. A ratio formed from
  // two timings taken MILLISECONDS APART survives that; a ratio formed from two medians taken
  // across a run that spanned a load change does not. Absolute us/matrix is printed anyway
  // because the shape across n is informative, but it is the ratio column that is evidence.
  // TWO ESTIMATORS, because on a box at load average 31 neither is sufficient alone.
  //   MIN   -- the fastest cycle each implementation ever achieved. Under contention the
  //            distribution is a clean cost plus a one-sided delay, so the minimum estimates
  //            the UNCONTENDED cost and is the standard estimator for a noisy machine. It is
  //            not paired, so it can be fooled by one implementation happening to get the only
  //            quiet window in the run.
  //   PAIR  -- median of per-cycle ratios, each formed from two timings milliseconds apart.
  //            Immune to slow drift, but every sample carries whatever noise hit that cycle.
  // They are independent failure modes. Where the two agree the answer is real; where they
  // disagree, nothing here resolves it and the report should say so rather than pick one.
  printf("%-8s %6s %6s %7s | %9s %8s %8s | %6s %6s | %13s %13s\n", "bucket", "mats", "mean n",
         "corpus%", "mine", "Accel", "OpenBL", "A/m", "O/m", "paired A/m", "paired O/m");
  double wm = 0, wa = 0, wo = 0;
  for (int b = 0; b < NB; b++) {
    if (buck[b].empty()) continue;
    double mean_n = 0;
    for (const Mat &M : buck[b]) mean_n += M.n;
    mean_n /= buck[b].size();
    std::vector<double> tm, ta, to, ra, ro;
    for (int c = 0; c < cycles; c++) {
      double t0, m1, a1, o1 = 0, m2, a2, o2 = 0;
      t0 = now_us(); sink += run_mine(buck[b]);                    m1 = now_us() - t0;
      t0 = now_us(); sink += run_lap(buck[b], dsytd2_, dsterf_);   a1 = now_us() - t0;
      if (have_ob) { t0 = now_us(); sink += run_lap(buck[b], ob_sytd2, ob_sterf); o1 = now_us() - t0; }
      if (have_ob) { t0 = now_us(); sink += run_lap(buck[b], ob_sytd2, ob_sterf); o2 = now_us() - t0; }
      t0 = now_us(); sink += run_lap(buck[b], dsytd2_, dsterf_);   a2 = now_us() - t0;
      t0 = now_us(); sink += run_mine(buck[b]);                    m2 = now_us() - t0;
      const double d = 2.0 * buck[b].size();
      const double M = (m1 + m2) / d, A = (a1 + a2) / d, O = (o1 + o2) / d;
      tm.push_back(M); ta.push_back(A); ra.push_back(A / M);
      if (have_ob) { to.push_back(O); ro.push_back(O / M); }
    }
    const double M = *std::min_element(tm.begin(), tm.end());
    const double A = *std::min_element(ta.begin(), ta.end());
    const double O = have_ob ? *std::min_element(to.begin(), to.end()) : 0.0;
    wm += M * pop[b]; wa += A * pop[b]; wo += O * pop[b];
    std::sort(ra.begin(), ra.end());
    if (have_ob) std::sort(ro.begin(), ro.end());
    printf("%-8s %6zu %6.1f %6.1f%% | %9.3f %8.3f %8.3f | %5.2fx %5.2fx | %5.2fx[%.2f-%.2f] "
           "%5.2fx[%.2f-%.2f]\n",
           BNAME[b], buck[b].size(), mean_n, 100.0 * pop[b] / totpop, M, A, O, A / M,
           have_ob ? O / M : 0.0, median(ra), ra.front(), ra.back(),
           have_ob ? median(ro) : 0.0, have_ob ? ro.front() : 0.0, have_ob ? ro.back() : 0.0);
  }
  printf("\nus per MATRIX (four per molecule), BEST of %d cycles. Ratio > 1 means eigen_small\n"
         "WINS. A/m and O/m are ratios of those best times; `paired` is the median of the %d\n"
         "per-cycle ratios with its min-max bracket. Absolute us are inflated ~4x by whatever\n"
         "else this shared box was running -- hume's own tridiagscale read 526.74 us/mol during\n"
         "these runs against the 128.65 recorded in its source. Read ratios, not microseconds.\n\n",
         cycles, cycles);
  printf("CORPUS-WEIGHTED (per-bucket best x true corpus bucket populations):\n");
  printf("  per matrix:   eigen_small %7.3f   Accel %7.3f   OpenBLAS %7.3f us\n",
         wm / totpop, wa / totpop, wo / totpop);
  printf("  per molecule: eigen_small %7.2f   Accel %7.2f   OpenBLAS %7.2f us   "
         "(BCUT2D solve only, no matrix fill)\n",
         4.0 * wm / totpop, 4.0 * wa / totpop, 4.0 * wo / totpop);
  return 0;
}
