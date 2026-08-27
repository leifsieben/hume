// HUME's featuriser in C++: every block, one binary, one exactness harness.
//
//   ./hume verify [mols.txt]   -> values.txt, one line per molecule
//   ./hume bench  [mols.txt]   -> us/mol per block
//
// Consolidates what was split across bench.cpp (timing only, never checked for correctness) and
// predict.cpp (checked). Two different references are being matched and the distinction matters:
//
//   * chi, BalabanJ, Phi, EState, Kappa, BCUT2D  -> must equal RDKIT
//   * cycles, resistance, conjugation, stereo    -> must equal OUR OWN Python modules, which are
//                                                   themselves already verified on the 1M corpus
//
// Nothing here recomputes an atom property RDKit already produces cheaply in C++ (Crippen 0.85
// us, Gasteiger 9.41 us, Labute ASA 4.27 us, TPSA 0.26 us). Those arrive through the exporter.
// What is reimplemented is what is SLOW in RDKit/Mordred for reasons of implementation rather
// than of arithmetic -- Chi at 273 us for six columns being the clearest case.

#include "../src/hume_core/hume_blocks.h"

// THE COMPUTATION NOW LIVES IN src/hume_core/hume_blocks.h, shared with the pybind11 extension.
// What stays here is exactly what a library must not contain: a text loader, a CLI and the
// timing harness. `./hume verify mols.txt` followed by `python cpp/verify_hume.py` still
// reports ALL EXACT on 182 descriptors over 98,905 molecules, and values_hume.txt is
// byte-identical to what it produced before the split (sha256 0832c5aa...b4ab7f) -- checked,
// not assumed.
//
//   c++ -O3 -std=c++17 -o hume hume.cpp -framework Accelerate
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

static std::vector<Mol> load(const char *path) {
  std::ifstream f(path);
  int nm;
  f >> nm;
  std::vector<Mol> ms(nm);
  for (int k = 0; k < nm; k++) {
    Mol &m = ms[k];
    f >> m.n >> m.nb >> m.chg_ok;
    m.Z.resize(m.n); m.deg.resize(m.n); m.nH.resize(m.n); m.fchg.resize(m.n);
    m.hyb.resize(m.n); m.arom.resize(m.n); m.ring.resize(m.n); m.cip.resize(m.n);
    m.mass.resize(m.n); m.gast.resize(m.n); m.clogp.resize(m.n); m.cmr.resize(m.n);
    for (int i = 0; i < m.n; i++)
      f >> m.Z[i] >> m.deg[i] >> m.nH[i] >> m.fchg[i] >> m.hyb[i] >> m.arom[i] >> m.ring[i]
        >> m.mass[i] >> m.gast[i] >> m.clogp[i] >> m.cmr[i] >> m.cip[i];
    m.adj.assign(m.n, {});
    m.bu.resize(m.nb); m.bv.resize(m.nb); m.bord.resize(m.nb);
    m.bconj.resize(m.nb); m.bring.resize(m.nb); m.bstereo.resize(m.nb);
    for (int b = 0; b < m.nb; b++) {
      f >> m.bu[b] >> m.bv[b] >> m.bord[b] >> m.bconj[b] >> m.bring[b] >> m.bstereo[b];
      m.adj[m.bu[b]].push_back(m.bv[b]);
      m.adj[m.bv[b]].push_back(m.bu[b]);
    }
    m.inc.assign(m.n, {});
    for (int b = 0; b < m.nb; b++) {
      m.inc[m.bu[b]].push_back({m.bv[b], b});
      m.inc[m.bv[b]].push_back({m.bu[b], b});
    }
  }
  return ms;
}

// ---------------------------------------------------------------------------------------

static volatile double sink_g = 0;             // keeps timed work from being optimised away

template <typename F>
static double time_it(const std::vector<Mol> &ms, int reps, F &&f) {
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)ms.size());
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "bench";
  auto ms = load(argc > 2 ? argv[2] : "mols.txt");
  double na = 0;
  for (auto &m : ms) na += m.n;
  fprintf(stderr, "%zu molecules, mean %.1f heavy atoms\n", ms.size(), na / ms.size());

  std::vector<int> D;
  std::vector<double> WD;
  std::vector<double> dn, dv;
  std::vector<double> ES;
  BcutWork BW;
  BW.z.resize(1);
  double kp[4], bc[8];
  double cn[CHI_MAX + 1], cv[CHI_MAX + 1], cc[CHI_MAX + 1];
  double cy[31], rs[60], cj[24], st[23];

  // PAIRED, ALTERNATING SOLVER A/B. The naive in-process A/B -- time all of A, then all of B --
  // is not good enough: each half takes minutes, and the machine drifts (thermal, turbo, other
  // load) by more than the effect being measured. Across three builds the SAME dsyevd code read
  // 134.02, 135.99, 130.97 and 122.45 us/mol, and the A-then-B form reported dsytd2+dsterf as
  // 2.8% faster in one build and 0.9% SLOWER in the next.
  //
  // The fix is pairing at a fine grain: alternate A and B one corpus pass at a time and compare
  // WITHIN each cycle. Drift is common-mode across a pair separated by seconds rather than
  // minutes, so the per-cycle ratio is stable even when the absolutes are not. Reported as the
  // median of the per-cycle ratios plus the best-case (minimum) time for each, since the minimum
  // is the least contaminated estimate of the true cost.
  //
  // NOW A LAPACK-ONLY MODE, for the same reason certcheck is: A is the tuned library and B is
  // ours, so without a library there is no A. Built with -DHUME_WITH_LAPACK it times
  // eigen_small against dsyevd; without it, `./hume solverab` says so and points at
  // cpp/bench_eigen.cpp, which does the same comparison against BOTH Accelerate and OpenBLAS
  // and is where the shipped 166/161 vs 216/198 us numbers come from.
#ifndef HUME_WITH_LAPACK
  if (mode == "solverab") {
    fprintf(stderr,
            "solverab needs a reference LAPACK to compare against, and this binary links none.\n"
            "  rebuild:  c++ -O3 -std=c++17 -DHUME_WITH_LAPACK -o hume hume.cpp "
            "-framework Accelerate\n"
            "  or use:   cpp/bench_eigen.cpp, which times eigen_small against Accelerate AND "
            "OpenBLAS\n");
    return 2;
  }
#endif
  if (mode == "solverab") {
    const int cycles = 15;
    std::vector<double> ta, tb, ratio;
    // ORDER IS ALTERNATED between cycles. Running A-then-B every time would hand B a systematic
    // second-position advantage (warm caches, settled clocks), and a 2.7% effect is exactly the
    // size such a bias could manufacture. Odd cycles run A first, even cycles run B first, so
    // any position effect cancels in the median instead of being attributed to the solver.
    for (int r = 0; r < cycles; r++) {
      const bool a_first = (r % 2) == 0;
      double a = 0, b = 0;
      auto run_a = [&] { BCUT_SOLVER = 0; a = time_it(ms, 1, [&] {
        for (auto &m : ms) { bcut2d(m, BW, bc); sink_g += bc[0]; } }); };
      auto run_b = [&] { BCUT_SOLVER = 1; b = time_it(ms, 1, [&] {
        for (auto &m : ms) { bcut2d(m, BW, bc); sink_g += bc[0]; } }); };
      if (a_first) { run_a(); run_b(); } else { run_b(); run_a(); }
      ta.push_back(a); tb.push_back(b); ratio.push_back(b / a);
      fprintf(stderr, "  cycle %2d [%s]  dsyevd %7.2f  eigen_small %7.2f  ratio %.4f\n",
              r + 1, a_first ? "A,B" : "B,A", a, b, b / a);
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    printf("  %d paired cycles over %zu molecules\n", cycles, ms.size());
    printf("  dsyevd          median %7.2f  min %7.2f us/mol\n", med(ta),
           *std::min_element(ta.begin(), ta.end()));
    printf("  eigen_small     median %7.2f  min %7.2f us/mol\n", med(tb),
           *std::min_element(tb.begin(), tb.end()));
    printf("  PAIRED ratio (eigen_small / dsyevd): median %.4f  -> %+.2f%%\n",
           med(ratio), 100.0 * (med(ratio) - 1.0));
    return 0;
  }

  // DOES IDEA #12 (4-wide SIMD across the four tridiagonalisations) HAVE A CEILING WORTH THE
  // BUILD? That depends entirely on whether dsytd2 at n ~ 27 is OVERHEAD-bound or FLOP-bound.
  // Batching four reductions into one vectorised pass amortises per-call and per-column
  // overhead and improves ILP; it does NOT reduce the flop count. So:
  //   cost / n^3 roughly constant  -> FLOP-bound, batching buys ILP only, ceiling modest
  //   cost roughly flat in n       -> overhead-bound, batching is the right lever, ceiling high
  // Measured by bucketing the corpus on heavy-atom count and timing tridiagonalisation only.
  if (mode == "tridiagscale") {
    struct Bucket { int lo, hi; std::vector<const Mol *> ms; };
    std::vector<Bucket> bk = {{3, 15, {}}, {16, 22, {}}, {23, 30, {}},
                              {31, 45, {}}, {46, 70, {}}, {71, 1000, {}}};
    for (auto &m : ms) {
      int hv = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv++;
      for (auto &b : bk) if (hv >= b.lo && hv <= b.hi) { b.ms.push_back(&m); break; }
    }
    auto timed = [&](std::vector<const Mol *> &v, int solver) {
      BCUT_SOLVER = solver;
      const int reps = 5;
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; r++)
        for (auto *m : v) { bcut2d(*m, BW, bc); sink_g += bc[0]; }
      auto t1 = std::chrono::steady_clock::now();
      BCUT_SOLVER = 1;
      return std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)v.size());
    };
    std::vector<double> full(bk.size(), 0), tri(bk.size(), 0), mean_n(bk.size(), 0);
    double total = 0;
    for (size_t i = 0; i < bk.size(); i++) {
      if (bk[i].ms.empty()) continue;
      double mn = 0;
      for (auto *m : bk[i].ms) { int hv = 0; for (int j = 0; j < m->n; j++) if (m->Z[j] != 1) hv++;
                                 mn += hv; }
      mean_n[i] = mn / bk[i].ms.size();
      full[i] = timed(bk[i].ms, 1);
      tri[i] = timed(bk[i].ms, 2);
      total += full[i] * bk[i].ms.size();
    }
    printf("  %-10s %8s %7s %6s %11s %11s %10s %9s\n", "n range", "count", "%mols", "mean n",
           "BCUT us/mol", "tridiag us", "us/n^3 e6", "%of time");
    for (size_t i = 0; i < bk.size(); i++) {
      if (bk[i].ms.empty()) continue;
      printf("  %3d-%-6d %8zu %6.1f%% %6.1f %11.2f %11.2f %10.1f %8.1f%%\n",
             bk[i].lo, bk[i].hi, bk[i].ms.size(), 100.0 * bk[i].ms.size() / ms.size(), mean_n[i],
             full[i], tri[i], 1e6 * tri[i] / (mean_n[i] * mean_n[i] * mean_n[i]),
             100.0 * full[i] * bk[i].ms.size() / total);
    }
    printf("  corpus-weighted BCUT2D: %.2f us/mol\n", total / ms.size());
    return 0;
  }

  // MEASURE BEFORE BUILDING. Three probes decide whether a structured-matvec Lanczos can beat
  // dsytd2+dsterf on the large-molecule tail that carries 46% of BCUT2D's time.
  if (mode == "lanczosprobe") {
    std::vector<const Mol *> tail, mid;
    for (auto &m : ms) {
      int hv = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv++;
      if (hv >= 71) tail.push_back(&m);
      else if (hv >= 31 && hv <= 45) mid.push_back(&m);
    }
    printf("tail (n>=71): %zu molecules   mid (31-45): %zu\n\n", tail.size(), mid.size());

    // ---- PROBE 1: structured vs dense MATVEC ONLY -------------------------------------
    // The operator is built OUTSIDE the timed loop. Lanczos builds once and then does m
    // matvecs, so folding construction into the per-matvec cost would flatter the dense side
    // (whose construction is an n^2 fill) and answer a question nobody asked.
    printf("PROBE 1  matvec cost with the operator already built, structured vs dense\n");
    printf("  %8s %7s %7s %12s %12s %8s\n", "n range", "mols", "mean n", "dense us", "sparse us",
           "speedup");
    struct B2 { int lo, hi; };
    std::vector<B2> rng = {{3, 15}, {16, 22}, {23, 30}, {31, 45}, {46, 70}, {71, 100000}};
    for (auto &r : rng) {
      std::vector<const Mol *> g;
      double mn = 0;
      for (auto &m : ms) {
        int hv2 = 0;
        for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv2++;
        if (hv2 >= r.lo && hv2 <= r.hi) { g.push_back(&m); mn += hv2; }
      }
      if (g.size() < 50) continue;
      mn /= g.size();
      if (g.size() > 1500) g.resize(1500);
      std::vector<BurdenOp> ops(g.size());
      std::vector<std::vector<double>> dense(g.size());
      std::vector<int> hv;
      for (size_t t = 0; t < g.size(); t++) {
        burden_build(*g[t], ops[t], hv);
        burden_diag(g[t]->mass, hv, ops[t]);
        int n2 = ops[t].n;
        dense[t].assign((size_t)n2 * n2, 0.001);
        for (int i = 0; i < n2; i++) dense[t][(size_t)i * n2 + i] = ops[t].dg[i];
        for (size_t e2 = 0; e2 < ops[t].bi.size(); e2++) {
          double v = ops[t].bw[e2] + 0.001;
          dense[t][(size_t)ops[t].bi[e2] * n2 + ops[t].bj[e2]] = v;
          dense[t][(size_t)ops[t].bj[e2] * n2 + ops[t].bi[e2]] = v;
        }
      }
      std::vector<double> x, y;
      const int reps = 300;
      auto t0 = std::chrono::steady_clock::now();
      for (int r2 = 0; r2 < reps; r2++)
        for (size_t t = 0; t < g.size(); t++) {
          x.assign(ops[t].n, 1.0); y.assign(ops[t].n, 0.0);
          burden_mv(ops[t], x.data(), y.data());
          sink_g += y[0];
        }
      auto t1 = std::chrono::steady_clock::now();
      double ts = std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)g.size());
      t0 = std::chrono::steady_clock::now();
      for (int r2 = 0; r2 < reps; r2++)
        for (size_t t = 0; t < g.size(); t++) {
          int n2 = ops[t].n;
          x.assign(n2, 1.0); y.assign(n2, 0.0);
          for (int i = 0; i < n2; i++) {
            double s2 = 0.0;
            const double *row = &dense[t][(size_t)i * n2];
            for (int j = 0; j < n2; j++) s2 += row[j] * x[j];
            y[i] = s2;
          }
          sink_g += y[0];
        }
      t1 = std::chrono::steady_clock::now();
      double td = std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * (double)g.size());
      printf("  %3d-%-4d %7zu %7.1f %12.4f %12.4f %7.2fx\n", r.lo, r.hi, g.size(), mn, td, ts,
             td / ts);
    }

    // ---- PROBE 2: iterations to converge on the tail ----------------------------------
    printf("\nPROBE 2  Lanczos iterations for lambda_min AND lambda_max to 1e-12 relative\n");
    printf("         (ground truth = dense eigen_small; over the n>=71 bucket)\n");
    const int REORTH[3] = {0, 8, -1};
    const char *RNAME[3] = {"none", "window-8", "full"};
    std::vector<const Mol *> samp = tail;
    if (samp.size() > 300) samp.resize(300);
    std::vector<std::array<int, 4>> need(samp.size(), {0, 0, 0, 0});
    for (int rm = 0; rm < 3; rm++) {
      std::vector<int> iters;
      int failed = 0;
      double worst_rel = 0.0;
      LanczosWork L, S;
      for (size_t t = 0; t < samp.size(); t++) {
        const Mol *m = samp[t];
        BurdenOp B;
        std::vector<int> hv;
        burden_build(*m, B, hv);
        const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
        for (int p = 0; p < 4; p++) {
          burden_diag(*props[p], hv, B);
          double dlo, dhi;
          BCUT_SOLVER = 1;
          bcut_one(*m, *props[p], BW, &dhi, &dlo);
          lanczos_run(B, std::min(B.n, 400), REORTH[rm], L);
          int used = -1;
          for (int k = 2; k <= (int)L.alpha.size(); k++) {
            double rlo, rhi;
            ritz_extremes(L, k, S, &rlo, &rhi);
            if (std::fabs(rlo - dlo) <= 1e-12 * std::max(std::fabs(dlo), 1e-30) &&
                std::fabs(rhi - dhi) <= 1e-12 * std::max(std::fabs(dhi), 1e-30)) { used = k; break; }
          }
          if (used < 0) {
            failed++;
            double rlo, rhi;
            ritz_extremes(L, (int)L.alpha.size(), S, &rlo, &rhi);
            worst_rel = std::max(worst_rel,
                std::max(std::fabs(rlo - dlo) / std::max(std::fabs(dlo), 1e-30),
                         std::fabs(rhi - dhi) / std::max(std::fabs(dhi), 1e-30)));
            used = (int)L.alpha.size();
          } else {
            iters.push_back(used);
          }
          if (rm == 2) need[t][p] = used;               // full reorth drives probe 3
        }
      }
      std::sort(iters.begin(), iters.end());
      auto pct = [&](double f) {
        return iters.empty() ? -1 : iters[std::min(iters.size() - 1, (size_t)(f * iters.size()))];
      };
      printf("  reorth=%-9s converged %5zu/%5zu  iters p50 %4d  p90 %4d  p99 %4d  max %4d",
             RNAME[rm], iters.size(), samp.size() * 4, pct(0.50), pct(0.90), pct(0.99),
             iters.empty() ? -1 : iters.back());
      if (failed) printf("   FAILED %d (worst rel %.2e)", failed, worst_rel);
      printf("\n");
    }

    // ---- PROBE 3: end-to-end, the number that actually decides -------------------------
    // Lanczos is run with FULL reorthogonalisation (the only variant that reached 1e-12 on
    // every case) for EXACTLY the iteration count convergence needed, taken from probe 2.
    // That is a LOWER BOUND on the real cost: a shipping version cannot know the count in
    // advance and must pay for convergence testing on top. If the lower bound loses, the
    // idea is dead without writing a stopping rule.
    printf("\nPROBE 3  end-to-end on the n>=71 tail: Lanczos(full reorth, oracle iters) vs dense\n");
    {
      LanczosWork L, S;
      std::vector<int> hv;
      double t_lan = 0, t_dense = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (size_t t = 0; t < samp.size(); t++) {
        const Mol *m = samp[t];
        BurdenOp B;
        burden_build(*m, B, hv);
        const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
        for (int p = 0; p < 4; p++) {
          burden_diag(*props[p], hv, B);
          lanczos_run(B, need[t][p], -1, L);
          double rlo, rhi;
          ritz_extremes(L, (int)L.alpha.size(), S, &rlo, &rhi);
          sink_g += rlo + rhi;
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      t_lan = std::chrono::duration<double, std::micro>(t1 - t0).count() / samp.size();
      BCUT_SOLVER = 1;
      t0 = std::chrono::steady_clock::now();
      for (size_t t = 0; t < samp.size(); t++) { bcut2d(*samp[t], BW, bc); sink_g += bc[0]; }
      t1 = std::chrono::steady_clock::now();
      t_dense = std::chrono::duration<double, std::micro>(t1 - t0).count() / samp.size();
      printf("  dense eigen_small, 4 spectra   : %9.2f us/mol\n", t_dense);
      printf("  Lanczos ORACLE iters, 4 spectra: %9.2f us/mol  (%.2fx %s)\n", t_lan,
             t_dense > t_lan ? t_dense / t_lan : t_lan / t_dense,
             t_dense > t_lan ? "FASTER" : "SLOWER");
      // The shipping version cannot know the iteration count in advance; it must certify.
      // How many MORE iterations does certifying cost than converging?
      extern long long LANCZOS_ITERS, LANCZOS_CALLS;
      LANCZOS_ITERS = LANCZOS_CALLS = 0;
      long long oracle_sum = 0, oracle_n = 0;
      for (size_t t = 0; t < samp.size(); t++)
        for (int p = 0; p < 4; p++) { oracle_sum += need[t][p]; oracle_n++; }
      t0 = std::chrono::steady_clock::now();
      int nconv = 0;
      for (size_t t = 0; t < samp.size(); t++) {
        const Mol *m = samp[t];
        BurdenOp B;
        burden_build(*m, B, hv);
        const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
        for (int p = 0; p < 4; p++) {
          burden_diag(*props[p], hv, B);
          double rlo, rhi;
          if (lanczos_extremal(B, LANCZOS_TOL, &rlo, &rhi, L)) { nconv++; sink_g += rlo + rhi; }
        }
      }
      t1 = std::chrono::steady_clock::now();
      double t_cert = std::chrono::duration<double, std::micro>(t1 - t0).count() / samp.size();
      printf("  Lanczos CERTIFIED,   4 spectra: %9.2f us/mol  (%.2fx %s)\n", t_cert,
             t_dense > t_cert ? t_dense / t_cert : t_cert / t_dense,
             t_dense > t_cert ? "FASTER" : "SLOWER");
      printf("    mean iters: oracle %.1f   certified %.1f   ratio %.2fx   certified %d/%lld\n",
             (double)oracle_sum / oracle_n,
             LANCZOS_CALLS ? (double)LANCZOS_ITERS / LANCZOS_CALLS : 0.0,
             (LANCZOS_CALLS && oracle_n) ?
               ((double)LANCZOS_ITERS / LANCZOS_CALLS) / ((double)oracle_sum / oracle_n) : 0.0,
             nconv, oracle_n);
    }
    return 0;
  }

  if (mode == "certsweep") {
    std::vector<const Mol *> tail;
    for (auto &m : ms) {
      int hv2 = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv2++;
      if (hv2 >= 71) tail.push_back(&m);
    }
    if (tail.size() > 200) tail.resize(200);
    printf("tail sample: %zu molecules x 4 properties\n\n", tail.size());
    const double TOLS[3] = {1e-10, 1e-11, 1e-12};
    const int STEP = 2;
    // [tol][0]=hi only, [1]=lo only, [2]=both
    double sum_it[3][3] = {{0}};
    long long cnt_it[3][3] = {{0}};
    long long nfail[3][3] = {{0}};
    double sum_oracle = 0; long long cnt_oracle = 0;
    double worst_err[3] = {0, 0, 0};
    LanczosWork L, S;
    std::vector<int> hv;
    for (auto *m : tail) {
      BurdenOp B;
      burden_build(*m, B, hv);
      const std::vector<double> *props[4] = {&m->mass, &m->gast, &m->clogp, &m->cmr};
      for (int p = 0; p < 4; p++) {
        burden_diag(*props[p], hv, B);
        double dlo, dhi;
        BCUT_SOLVER = 1;
        bcut_one(*m, *props[p], BW, &dhi, &dlo);
        const int maxit = std::min(B.n, 260);
        lanczos_run(B, maxit, -1, L);
        const int K = (int)L.alpha.size();
        int first_or = -1;
        int first[3][3];
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) first[a][b] = -1;
        for (int kk = 8; kk <= K; kk += STEP) {
          S.d.assign(L.alpha.begin(), L.alpha.begin() + kk);
          S.e.assign(kk > 1 ? kk - 1 : 1, 0.0);
          for (int i = 0; i + 1 < kk; i++) S.e[i] = L.beta[i];
          double sd_lo = 0.0, sd_hi = 0.0;
          if (!hume_eig::sterf_min_max(kk, S.d.data(), S.e.data(), &sd_lo, &sd_hi)) continue;
          S.d[0] = sd_lo;
          S.d[kk - 1] = sd_hi;
          const double bn = L.beta[kk - 1];
          double scale = std::max(std::fabs(S.d[0]), std::fabs(S.d[kk - 1]));
          if (scale < 1.0) scale = 1.0;
          const double r_lo = bn * tri_last_comp(L.alpha.data(), L.beta.data(), kk, S.d[0]);
          const double r_hi = bn * tri_last_comp(L.alpha.data(), L.beta.data(), kk, S.d[kk - 1]);
          if (first_or < 0 &&
              std::fabs(S.d[0] - dlo) <= 1e-12 * std::max(std::fabs(dlo), 1e-30) &&
              std::fabs(S.d[kk - 1] - dhi) <= 1e-12 * std::max(std::fabs(dhi), 1e-30))
            first_or = kk;
          for (int a = 0; a < 3; a++) {
            const double t = TOLS[a] * scale;
            if (first[a][0] < 0 && r_hi <= t) {
              first[a][0] = kk;
              worst_err[a] = std::max(worst_err[a],
                  std::fabs(S.d[kk - 1] - dhi) / std::max(std::fabs(dhi), 1e-30));
            }
            if (first[a][1] < 0 && r_lo <= t) {
              first[a][1] = kk;
              worst_err[a] = std::max(worst_err[a],
                  std::fabs(S.d[0] - dlo) / std::max(std::fabs(dlo), 1e-30));
            }
            if (first[a][2] < 0 && r_lo <= t && r_hi <= t) first[a][2] = kk;
          }
        }
        if (first_or > 0) { sum_oracle += first_or; cnt_oracle++; }
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) {
            if (first[a][b] > 0) { sum_it[a][b] += first[a][b]; cnt_it[a][b]++; }
            else nfail[a][b]++;
          }
      }
    }
    printf("  oracle (true err <= 1e-12, both ends): mean %.1f iters\n\n",
           cnt_oracle ? sum_oracle / cnt_oracle : 0.0);
    printf("  %-8s %14s %14s %14s %16s %12s\n", "tol", "cert hi only", "cert lo only",
           "cert BOTH", "both/oracle", "never cert");
    for (int a = 0; a < 3; a++) {
      double mo = cnt_oracle ? sum_oracle / cnt_oracle : 1.0;
      auto mean = [&](int b) { return cnt_it[a][b] ? sum_it[a][b] / cnt_it[a][b] : -1.0; };
      printf("  %-8.0e %14.1f %14.1f %14.1f %15.2fx %12lld\n", TOLS[a], mean(0), mean(1),
             mean(2), mean(2) / mo, nfail[a][2]);
    }
    printf("\n  worst TRUE relative error at first certification: 1e-10 %.2e  1e-11 %.2e  "
           "1e-12 %.2e\n", worst_err[0], worst_err[1], worst_err[2]);
    printf("  reorth cost scales as m^2, so predicted work ratio vs oracle is (both/oracle)^2\n");
    return 0;
  }

  // IS THE CERTIFICATE ITSELF BROKEN? tri_last_comp uses the FORWARD three-term recurrence,
  // which is the textbook-unstable direction when the eigenvector decays away from y_1:
  // rounding excites the growing solution and |y_k| comes out far too LARGE, which inflates
  // the residual and stops the certificate from ever firing. Compare it against dsteqr, which
  // is expensive but correct, at a fixed k well past true convergence.
  //
  // LAPACK-ONLY MODE. dsteqr is the whole point of this check -- it is the independent, correct
  // answer the recurrence is being graded against, and grading the recurrence against
  // eigen_small's own tridiagonal code would be marking its own homework. So this mode is kept
  // exactly as it was and compiled only with -DHUME_WITH_LAPACK; the shipped binary needs no
  // BLAS at all. Its verdict is already recorded: the backward recurrence matches dsteqr's last
  // component to within a factor ~1 at k = 40/60/80 on the tail, and the forward one does not.
#ifdef HUME_WITH_LAPACK
  if (mode == "certcheck") {
    std::vector<const Mol *> tail;
    for (auto &m : ms) {
      int hv2 = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) hv2++;
      if (hv2 >= 71) tail.push_back(&m);
    }
    if (tail.size() > 40) tail.resize(40);
    LanczosWork L;
    std::vector<int> hv;
    std::vector<double> d, e, z, wk;
    printf("  %6s %6s %14s %14s %14s %12s\n", "n", "k", "recur s_hi", "dsteqr s_hi",
           "ratio", "true relerr");
    int shown = 0;
    for (auto *m : tail) {
      if (shown >= 8) break;
      BurdenOp B;
      burden_build(*m, B, hv);
      burden_diag(m->mass, hv, B);
      double dlo, dhi;
      BCUT_SOLVER = 1;
      bcut_one(*m, m->mass, BW, &dhi, &dlo);
      lanczos_run(B, std::min(B.n, 260), -1, L);
      for (int kk : {40, 60, 80}) {
        if (kk > (int)L.alpha.size()) continue;
        d.assign(L.alpha.begin(), L.alpha.begin() + kk);
        e.assign(kk - 1, 0.0);
        for (int i = 0; i + 1 < kk; i++) e[i] = L.beta[i];
        z.assign((size_t)kk * kk, 0.0);
        wk.assign(2 * kk - 2 > 0 ? 2 * kk - 2 : 1, 0.0);
        char compz = 'I';
        int nn = kk, info = 0;
        dsteqr_(&compz, &nn, d.data(), e.data(), z.data(), &nn, wk.data(), &info);
        if (info != 0) { printf("  dsteqr info=%d\n", info); continue; }
        double exact_hi = std::fabs(z[(size_t)(kk - 1) * kk + (kk - 1)]);
        double rec_hi = tri_last_comp(L.alpha.data(), L.beta.data(), kk, d[kk - 1]);
        double relerr = std::fabs(d[kk - 1] - dhi) / std::max(std::fabs(dhi), 1e-30);
        printf("  %6d %6d %14.3e %14.3e %14.3e %12.2e\n", B.n, kk, rec_hi, exact_hi,
               exact_hi > 0 ? rec_hi / exact_hi : -1.0, relerr);
      }
      shown++;
    }
    return 0;
  }
#endif  // HUME_WITH_LAPACK

  // DOES THE SECOND NUMERICAL PATH SURVIVE INTO THE SHIPPED ARTEFACT AT ALL?
  //
  // The objection to a Krylov path is that its values are within ~1e-12 of the dense ones but
  // not bit-identical to them. That objection has a premise: that the difference is OBSERVABLE.
  // The descriptor matrix this project ships is FLOAT32 -- every Python module returns float32,
  // which is why verify_hume.py holds them to rtol 3e-6 -- and float32 carries ~1.2e-7 relative
  // precision. A 1e-12 relative difference is five orders BELOW the representation it lands in,
  // so the two paths can only diverge in the shipped file when a value happens to straddle a
  // float32 rounding boundary. Expected rate ~ 1e-12 / 1.2e-7 ~ 1e-5.
  //
  // Both paths are computed for every molecule, cast to float32, and compared BIT FOR BIT.
  if (mode == "f32cmp") {
    const int thr = argc > 3 ? atoi(argv[3]) : 71;
    long long n_lan_mol = 0, n_fallback = 0, n_vals = 0, n_diff = 0;
    double worst64 = 0.0, worst32 = 0.0;
    LanczosWork LW;
    std::vector<int> hv;
    for (auto &m : ms) {
      double dense8[8], lan8[8];
      // FORCE the dense path for the reference. bcut2d honours BCUT_LANCZOS_MIN_N, so with the
      // hook enabled this would otherwise compare Lanczos against itself and report a
      // beautifully clean zero that means nothing.
      const int keep_thr = BCUT_LANCZOS_MIN_N;
      BCUT_LANCZOS_MIN_N = 0;
      bcut2d(m, BW, dense8);
      BCUT_LANCZOS_MIN_N = keep_thr;
      int nheavy = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
      if (nheavy < thr) continue;
      BurdenOp B;
      burden_build(m, B, hv);
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      bool ok = true;
      for (int k = 0; k < 4 && ok; k++) {
        burden_diag(*props[k], hv, B);
        ok = lanczos_extremal(B, LANCZOS_TOL, &lan8[2 * k + 1], &lan8[2 * k], LW);
      }
      if (!ok) { n_fallback++; continue; }
      n_lan_mol++;
      for (int i = 0; i < 8; i++) {
        n_vals++;
        const double rel64 = std::fabs(dense8[i] - lan8[i]) /
                             std::max(std::fabs(dense8[i]), 1e-30);
        if (rel64 > worst64) worst64 = rel64;
        const float fa = (float)dense8[i], fb = (float)lan8[i];
        unsigned ia, ib;
        std::memcpy(&ia, &fa, 4);
        std::memcpy(&ib, &fb, 4);
        if (ia != ib) {
          n_diff++;
          const double r32 = std::fabs((double)fa - (double)fb) /
                             std::max(std::fabs((double)fa), 1e-30);
          if (r32 > worst32) worst32 = r32;
        }
      }
    }
    printf("  threshold n >= %d\n", thr);
    printf("  molecules via Lanczos      : %lld   (fell back to dense: %lld)\n",
           n_lan_mol, n_fallback);
    printf("  BCUT2D values compared     : %lld   (whole corpus would be %zu)\n",
           n_vals, ms.size() * 8);
    printf("  float64 worst relative diff: %.3e\n", worst64);
    printf("  FLOAT32 BIT DIFFERENCES    : %lld of %lld  (%.3e of compared)\n",
           n_diff, n_vals, n_vals ? (double)n_diff / n_vals : 0.0);
    if (n_diff) printf("  worst float32 relative diff: %.3e  (one ulp is ~1.2e-7)\n", worst32);
    return 0;
  }

  // ADVERSARIAL AUDIT. Forces the Krylov attempt on EVERY molecule regardless of size, and
  // compares against dense in float64 and in the float32 the artefact actually ships in.
  // Reports fallbacks, so a class that refuses to certify shows up as a refusal rather than
  // as a wrong number.
  if (mode == "lanczosaudit") {
    long long tried = 0, fell = 0, nvals = 0, ndiff32 = 0;
    double worst64 = 0.0, worst32 = 0.0;
    std::string worst_smi;
    LanczosWork LW;
    std::vector<int> hv;
    size_t idx = 0;
    for (auto &m : ms) {
      idx++;
      double dense8[8], lan8[8];
      const int keep = BCUT_LANCZOS_MIN_N;
      BCUT_LANCZOS_MIN_N = 0;
      bcut2d(m, BW, dense8);
      BCUT_LANCZOS_MIN_N = keep;
      int nheavy = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
      if (nheavy < 3) continue;
      BurdenOp B;
      burden_build(m, B, hv);
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      tried++;
      bool ok = true;
      for (int k = 0; k < 4 && ok; k++) {
        burden_diag(*props[k], hv, B);
        ok = lanczos_extremal(B, LANCZOS_TOL, &lan8[2 * k + 1], &lan8[2 * k], LW);
      }
      if (!ok) {
        fell++;
        const char *why = LZ_LAST_REASON == 3 ? "MISSED-EXTREME"
                        : LZ_LAST_REASON == 2 ? "true-residual" : "no-convergence";
        printf("    row %3zu  n=%4d  FALLBACK (%s)\n", idx, nheavy, why);
        continue;
      }
      double mrel = 0.0;
      for (int i = 0; i < 8; i++) {
        nvals++;
        const double rel = std::fabs(dense8[i] - lan8[i]) /
                           std::max(std::fabs(dense8[i]), 1e-30);
        mrel = std::max(mrel, rel);
        if (rel > worst64) { worst64 = rel; worst_smi = std::to_string(idx); }
        const float fa = (float)dense8[i], fb = (float)lan8[i];
        unsigned ia, ib;
        std::memcpy(&ia, &fa, 4); std::memcpy(&ib, &fb, 4);
        if (ia != ib) {
          ndiff32++;
          const double r32 = std::fabs((double)fa - (double)fb) /
                             std::max(std::fabs((double)fa), 1e-30);
          if (r32 > worst32) worst32 = r32;
        }
      }
      printf("    row %3zu  n=%4d  certified, max rel %.2e\n", idx, nheavy, mrel);
    }
    printf("  molecules attempted    : %lld\n", tried);
    printf("  fell back to dense     : %lld  (%.2f%%)\n", fell,
           tried ? 100.0 * fell / tried : 0.0);
    printf("  values compared        : %lld\n", nvals);
    printf("  worst |Lanczos-dense|  : %.3e relative (float64), worst row #%s\n", worst64,
           worst_smi.c_str());
    printf("  FLOAT32 bit differences: %lld of %lld\n", ndiff32, nvals);
    if (ndiff32) printf("  worst float32 rel diff : %.3e (one ulp ~1.2e-7)\n", worst32);
    return 0;
  }

  // DOES THE MISSED-EXTREME GUARD ACTUALLY FIRE? The audit never tripped it, which proves
  // nothing on its own. So construct the pathology directly: find the true lambda_max
  // eigenvector by power iteration, project it out of the Lanczos start, and run. Because A is
  // symmetric and v_max is an eigenvector, the ENTIRE Krylov space then stays orthogonal to
  // v_max -- lambda_max is genuinely invisible to Lanczos, theta_max converges tightly onto
  // lambda_2, and both residual tests pass while the answer is wrong. Only an independent
  // probe can catch that. Anything reported ACCEPTED-WRONG here is a correctness bug.
  if (mode == "ghosttest") {
    extern const double *LZ_FORCE_START;
    LanczosWork LW;
    std::vector<int> hv;
    std::vector<double> v(1), w(1), st(1);
    int nacc_wrong = 0, nrefused = 0, nacc_ok = 0, ntested = 0;
    for (auto &m : ms) {
      int nheavy = 0;
      for (int i = 0; i < m.n; i++) if (m.Z[i] != 1) nheavy++;
      if (nheavy < 20) continue;
      BurdenOp B;
      burden_build(m, B, hv);
      const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
      for (int p = 0; p < 4; p++) {
        burden_diag(*props[p], hv, B);
        double dlo, dhi;
        BCUT_SOLVER = 1;
        bcut_one(m, *props[p], BW, &dhi, &dlo);
        const int n = B.n;
        // power-iterate to the true lambda_max eigenvector
        v.assign(n, 0.0); w.assign(n, 0.0);
        for (int i = 0; i < n; i++) v[i] = std::sin(0.31 * i + 1.7) + 0.5;
        for (int it = 0; it < 3000; it++) {
          burden_mv(B, v.data(), w.data());
          double nn = 0.0;
          for (int i = 0; i < n; i++) nn += w[i] * w[i];
          nn = std::sqrt(nn);
          if (!(nn > 1e-300)) break;
          for (int i = 0; i < n; i++) v[i] = w[i] / nn;
        }
        double rq = 0.0;
        burden_mv(B, v.data(), w.data());
        for (int i = 0; i < n; i++) rq += v[i] * w[i];
        if (std::fabs(rq - dhi) > 1e-6 * std::max(std::fabs(dhi), 1.0)) continue;  // not converged
        // start := generic, with v_max projected out
        st.assign(n, 0.0);
        double dp = 0.0;
        for (int i = 0; i < n; i++) { st[i] = std::sin(1.2345 * i + 0.678 + 0.31 * B.dg[i]);
                                      dp += st[i] * v[i]; }
        for (int i = 0; i < n; i++) st[i] -= dp * v[i];
        LZ_FORCE_START = st.data();
        double llo, lhi;
        const bool ok = lanczos_extremal(B, LANCZOS_TOL, &llo, &lhi, LW);
        LZ_FORCE_START = nullptr;
        ntested++;
        if (!ok) { nrefused++; continue; }
        const double err = std::fabs(lhi - dhi) / std::max(std::fabs(dhi), 1e-30);
        if (err > 1e-9) {
          nacc_wrong++;
          printf("    ACCEPTED-WRONG n=%d prop=%d  lanczos %.10g  dense %.10g  rel %.2e\n",
                 n, p, lhi, dhi, err);
        } else {
          nacc_ok++;   // recovered lambda_max anyway (rounding reintroduced the direction)
        }
      }
    }
    printf("  adversarial starts tested : %d\n", ntested);
    printf("  REFUSED (guard fired)     : %d\n", nrefused);
    printf("  accepted, still correct   : %d\n", nacc_ok);
    printf("  ACCEPTED-WRONG            : %d   <-- must be zero\n", nacc_wrong);
    return 0;
  }

  if (mode == "verify") {
    // The 182 columns and their order are defined by blocks_row() in hume_blocks.h, which the
    // Python extension calls too. This mode is now purely the text serialisation of that row:
    // "%.12g", space-separated, one line per molecule -- unchanged, and byte-for-byte so.
    FILE *out = fopen("values_hume.txt", "w");
    BlockWork BWk;
    double row[HUME_NBLOCK_COLS];
    for (auto &m : ms) {
      blocks_row(m, BWk, row);
      fprintf(out, "%.12g", row[0]);
      for (int i = 1; i < HUME_NBLOCK_COLS; i++) fprintf(out, " %.12g", row[i]);
      fputc('\n', out);
    }
    fclose(out);
    fprintf(stderr, "wrote values_hume.txt\n");
    {
      extern long long BCUT_LANCZOS_TRIED, BCUT_LANCZOS_FELL_BACK, LZ_FAIL[4];
      extern double LZ_MAX_TRUE_OVER_PROXY, LZ_MAX_TRUE_OVER_TOL;
      fprintf(stderr,
              "  Krylov: tried %lld, fell back %lld (%.3f%%)  [no-conv %lld, true-resid %lld, "
              "missed-extreme %lld]\n  true residual / tol: max %.2e   true/proxy ratio "
              "(both > 1e-30): max %.2e\n",
              BCUT_LANCZOS_TRIED, BCUT_LANCZOS_FELL_BACK,
              BCUT_LANCZOS_TRIED ? 100.0 * BCUT_LANCZOS_FELL_BACK / BCUT_LANCZOS_TRIED : 0.0,
              LZ_FAIL[1], LZ_FAIL[2], LZ_FAIL[3], LZ_MAX_TRUE_OVER_TOL,
              LZ_MAX_TRUE_OVER_PROXY);
    }
    return 0;
  }

  // THE HONEST TOTAL: exactly what verify does -- one unweighted BFS, one weighted BFS, every
  // block once. The per-block numbers below each pay for their own distance matrix, so adding
  // them up counts that traversal five times and overstates the pipeline.
  volatile double sink = 0;
  {
    double t_all = time_it(ms, 10, [&] {
      for (auto &m : ms) {
        distances(m, D);
        weighted_distances(m, WD);
        chi(m, dn, dv, cn, cv, cc);
        cycle_counts(m, cy);
        resistance(m, D, rs);
        conjugation(m, D, cj);
        stereo(m, D, st);
        estate_from(m, D, ES);
        kappa(m, kp);
        bcut2d(m, BW, bc);
        sink += balaban(m, WD) + cn[2] + cy[0] + rs[0] + cj[0] + st[0] + ES[0] + kp[0] + bc[0];
      }
    });
    printf("  %-44s %8.2f us/mol   <== the real pipeline cost\n",
           "ALL BLOCKS, shared distance matrices", t_all);
  }
  double t_d = time_it(ms, 20, [&] { for (auto &m : ms) { distances(m, D); sink += D[0]; } });
  double t_c = time_it(ms, 20, [&] {
    for (auto &m : ms) { chi(m, dn, dv, cn, cv, cc); sink += cn[2]; } });
  double t_y = time_it(ms, 20, [&] {
    for (auto &m : ms) { cycle_counts(m, cy); sink += cy[0]; } });
  double t_r = time_it(ms, 10, [&] {
    for (auto &m : ms) { distances(m, D); resistance(m, D, rs); sink += rs[0]; } });
  double t_j = time_it(ms, 20, [&] {
    for (auto &m : ms) { weighted_distances(m, WD); sink += balaban(m, WD); } });
  double t_g = time_it(ms, 20, [&] {
    for (auto &m : ms) { distances(m, D); conjugation(m, D, cj); sink += cj[0]; } });
  double t_s = time_it(ms, 20, [&] {
    for (auto &m : ms) { distances(m, D); stereo(m, D, st); sink += st[0]; } });
  printf("  %-44s %8.2f us/mol\n", "distance matrix (shared by four blocks)", t_d);
  printf("  %-44s %8.2f us/mol\n", "chi k<=7, n and v variants (16 cols)", t_c);
  printf("  %-44s %8.2f us/mol\n", "cycle counts C3-C5", t_y);
  printf("  %-44s %8.2f us/mol\n", "resistance (incl. distances)", t_r);
  printf("  %-44s %8.2f us/mol\n", "BalabanJ (incl. distances)", t_j);
  printf("  %-44s %8.2f us/mol\n", "conjugation (incl. distances)", t_g);
  printf("  %-44s %8.2f us/mol\n", "stereo (incl. distances)", t_s);
  double t_e = time_it(ms, 20, [&] {
    for (auto &m : ms) { distances(m, D); estate_from(m, D, ES); sink += ES[0]; } });
  double t_k = time_it(ms, 20, [&] { for (auto &m : ms) { kappa(m, kp); sink += kp[0]; } });
  double t_b = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  // STAGE SPLIT: how much of our BCUT2D is matrix assembly and how much is the eigensolve?
  // RDKit's own cost is dominated by removeAllHs + Gasteiger + Crippen + Eigen computing
  // eigenVECTORS it discards; ours excludes all of that (properties arrive precomputed, and we
  // pass jobz='N'), so the split here is a different question with a different answer.
  extern bool BCUT_SKIP_SOLVE;
  BCUT_SKIP_SOLVE = true;
  double t_asm = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  BCUT_SKIP_SOLVE = false;
  // SOLVER A/B IN ONE PROCESS. t_b above was timed with whatever BCUT_SOLVER is set to; time
  // the other one right here, back to back, so the comparison cannot be contaminated by machine
  // load drifting between two separate bench runs. With no LAPACK linked there IS no other
  // solver -- BCUT_SOLVER 0 falls through to the same eigen_small path -- so the line is
  // suppressed rather than printed as a meaningless 0%.
  extern int BCUT_SOLVER;
  int keep_solver = BCUT_SOLVER;
#ifdef HUME_WITH_LAPACK
  BCUT_SOLVER = keep_solver ? 0 : 1;
  double t_b2 = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  BCUT_SOLVER = keep_solver;
#endif
  printf("  %-44s %8.2f us/mol\n", "EState (incl. distances)", t_e);
  printf("  %-44s %8.2f us/mol\n", "Kappa1-3 + HallKierAlpha", t_k);
  printf("  %-44s %8.2f us/mol\n",
         keep_solver ? "BCUT2D (eigen_small, 4 spectra)" : "BCUT2D (dsyevd, 4 dense spectra)",
         t_b);
  printf("  %-44s %8.2f us/mol  (%.0f%% of BCUT2D)\n", "  ^ of which: matrix assembly only",
         t_asm, 100.0 * t_asm / t_b);
  printf("  %-44s %8.2f us/mol  (%.0f%%)\n", "  ^ of which: 4 eigensolves", t_b - t_asm,
         100.0 * (t_b - t_asm) / t_b);
#ifdef HUME_WITH_LAPACK
  printf("  %-44s %8.2f us/mol  (%+.1f%% vs above)\n",
         keep_solver ? "  ALT solver: dsyevd" : "  ALT solver: eigen_small",
         t_b2, 100.0 * (t_b2 - t_b) / t_b);
#endif
  // Stage probe: where inside the eigensolve does the time actually go? This decides whether
  // vectorising the four Householder reductions is worth building at all.
  BCUT_SOLVER = 2;
  double t_tri = time_it(ms, 10, [&] { for (auto &m : ms) { bcut2d(m, BW, bc); sink += bc[0]; } });
  BCUT_SOLVER = keep_solver;
  double solve_only = t_b - t_asm;
  printf("  %-44s %8.2f us/mol  (%.0f%% of the eigensolve)\n",
         "  ^ tridiagonalisation only", t_tri - t_asm,
         100.0 * (t_tri - t_asm) / solve_only);
  printf("  %-44s %8.2f us/mol  (%.0f%%)\n", "  ^ QR sweep (PWK) only",
         t_b - t_tri, 100.0 * (t_b - t_tri) / solve_only);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
