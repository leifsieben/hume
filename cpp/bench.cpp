// Real C++ timings for the three primitives whose cost was in question.
//
// Everything reported in the markdown so far was a projection from Python timings divided by
// a guessed interpreter factor. This measures the actual thing on actual molecular graphs
// exported from the benchmark set by export_graphs.py.
//
//   build:  clang++ -O3 -march=native -std=c++17 bench.cpp -framework Accelerate -o bench
//   run:    ./bench graphs.txt
//
// LAPACK comes from Accelerate on macOS; on Linux link -llapack instead. Only two routines
// are used: dposv for the Laplacian solve and dsyevd for the spectrum.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

extern "C" {
void dposv_(char *, int *, int *, double *, int *, double *, int *, int *);
void dsyevd_(char *, char *, int *, double *, int *, double *, double *, int *, int *, int *,
             int *);
}

struct Mol {
  int n = 0;
  std::vector<int> deg, degv;              // Chi weights: sigma and valence connectivity
  std::vector<std::vector<int>> adj;
  std::vector<int> bu, bv;
};

// ---------------------------------------------------------------------------------------
// Bounded simple-path enumeration with Chi accumulation.
//
// Chi_k = sum over simple paths of length k of prod(delta_i)^-1/2. Paths cannot revisit an
// atom, so there is no matrix-power shortcut -- but molecular graphs are sparse and nearly
// tree-like, so the search is far smaller than the worst case suggests.
// ---------------------------------------------------------------------------------------
static constexpr int PMAX = 7;

struct PathState {
  const Mol *m;
  std::vector<char> on;
  double chi[PMAX + 1];
  double chiv[PMAX + 1];

  void dfs(int u, int depth, double prod, double prodv) {
    for (int v : m->adj[u]) {
      if (on[v]) continue;
      double p = prod / std::sqrt((double)m->deg[v]);
      double pv = prodv / std::sqrt((double)m->degv[v]);
      chi[depth] += p;
      chiv[depth] += pv;
      if (depth < PMAX) {
        on[v] = 1;
        dfs(v, depth + 1, p, pv);
        on[v] = 0;
      }
    }
  }
};

static void chi_paths(const Mol &m, double *out) {
  PathState s;
  s.m = &m;
  s.on.assign(m.n, 0);
  for (int k = 0; k <= PMAX; k++) s.chi[k] = s.chiv[k] = 0.0;
  for (int a = 0; a < m.n; a++) {
    s.on[a] = 1;
    s.dfs(a, 1, 1.0 / std::sqrt((double)m.deg[a]), 1.0 / std::sqrt((double)m.degv[a]));
    s.on[a] = 0;
  }
  // Each path is walked from both ends.
  for (int k = 1; k <= PMAX; k++) {
    out[k - 1] = s.chi[k] * 0.5;
    out[PMAX + k - 1] = s.chiv[k] * 0.5;
  }
}

// ---------------------------------------------------------------------------------------
// Bounded simple-cycle enumeration. Each cycle is found from its lowest-indexed member, in
// both directions, hence the halving.
// ---------------------------------------------------------------------------------------
static constexpr int CMAX = 8;

struct CycState {
  const Mol *m;
  std::vector<char> on;
  std::vector<int> path;
  double count[CMAX + 1];
  std::vector<double> per;

  void dfs(int u, int start, int depth) {
    for (int v : m->adj[u]) {
      if (v == start) {
        if (depth >= 3) {
          count[depth] += 1.0;
          for (int x : path) per[x] += 1.0;
        }
      } else if (v > start && !on[v] && depth < CMAX) {
        on[v] = 1;
        path.push_back(v);
        dfs(v, start, depth + 1);
        path.pop_back();
        on[v] = 0;
      }
    }
  }
};

static void cycles(const Mol &m, double *out) {
  CycState s;
  s.m = &m;
  s.on.assign(m.n, 0);
  s.per.assign(m.n, 0.0);
  for (int k = 0; k <= CMAX; k++) s.count[k] = 0.0;
  s.path.reserve(CMAX + 1);
  for (int a = 0; a < m.n; a++) {
    s.on[a] = 1;
    s.path.push_back(a);
    s.dfs(a, a, 1);
    s.path.pop_back();
    s.on[a] = 0;
  }
  for (int k = 3; k <= CMAX; k++) out[k - 3] = s.count[k] * 0.5;
}

// ---------------------------------------------------------------------------------------
// Resistance distance. L is singular (rows sum to zero), so L+ = (L + J/n)^-1 - J/n. The
// shifted matrix is symmetric positive definite, so dposv (Cholesky) applies -- no
// eigendecomposition needed for Omega.
// ---------------------------------------------------------------------------------------
static double resistance(const Mol &m, std::vector<double> &work) {
  int n = m.n;
  work.assign((size_t)n * n, 0.0);
  double inv = 1.0 / n;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) work[(size_t)i * n + j] = inv;
  for (size_t b = 0; b < m.bu.size(); b++) {
    int u = m.bu[b], v = m.bv[b];
    work[(size_t)u * n + u] += 1.0;
    work[(size_t)v * n + v] += 1.0;
    work[(size_t)u * n + v] -= 1.0;
    work[(size_t)v * n + u] -= 1.0;
  }
  std::vector<double> rhs((size_t)n * n, 0.0);
  for (int i = 0; i < n; i++) rhs[(size_t)i * n + i] = 1.0;
  char uplo = 'L';
  int nn = n, nrhs = n, info = 0;
  dposv_(&uplo, &nn, &nrhs, work.data(), &nn, rhs.data(), &nn, &info);
  if (info != 0) return 0.0;
  double kf = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      kf += rhs[(size_t)i * n + i] + rhs[(size_t)j * n + j] - 2.0 * rhs[(size_t)i * n + j];
  return kf;
}

static double spectrum(const Mol &m, std::vector<double> &A, std::vector<double> &w) {
  int n = m.n;
  A.assign((size_t)n * n, 0.0);
  std::vector<double> dinv(n);
  for (int i = 0; i < n; i++) dinv[i] = m.deg[i] > 0 ? 1.0 / std::sqrt((double)m.deg[i]) : 0.0;
  for (int i = 0; i < n; i++) A[(size_t)i * n + i] = 1.0;
  for (size_t b = 0; b < m.bu.size(); b++) {
    int u = m.bu[b], v = m.bv[b];
    A[(size_t)u * n + v] -= dinv[u] * dinv[v];
    A[(size_t)v * n + u] -= dinv[u] * dinv[v];
  }
  w.assign(n, 0.0);
  char jobz = 'N', uplo = 'L';
  int nn = n, info = 0, lwork = -1, liwork = -1;
  double wq;
  int iwq;
  dsyevd_(&jobz, &uplo, &nn, A.data(), &nn, w.data(), &wq, &lwork, &iwq, &liwork, &info);
  lwork = (int)wq;
  liwork = iwq;
  std::vector<double> wk(std::max(lwork, 1));
  std::vector<int> iwk(std::max(liwork, 1));
  dsyevd_(&jobz, &uplo, &nn, A.data(), &nn, w.data(), wk.data(), &lwork, iwk.data(), &liwork,
          &info);
  return info == 0 ? w[1] : 0.0;   // Fiedler value
}

// ---------------------------------------------------------------------------------------

static std::vector<Mol> load(const char *path) {
  std::ifstream f(path);
  int nm;
  f >> nm;
  std::vector<Mol> ms(nm);
  for (int k = 0; k < nm; k++) {
    Mol &m = ms[k];
    int nb;
    f >> m.n >> nb;
    m.deg.resize(m.n);
    m.degv.resize(m.n);
    for (int i = 0; i < m.n; i++) f >> m.deg[i] >> m.degv[i];
    m.adj.assign(m.n, {});
    m.bu.resize(nb);
    m.bv.resize(nb);
    for (int b = 0; b < nb; b++) {
      f >> m.bu[b] >> m.bv[b];
      m.adj[m.bu[b]].push_back(m.bv[b]);
      m.adj[m.bv[b]].push_back(m.bu[b]);
    }
  }
  return ms;
}

template <typename F>
static double time_it(const std::vector<Mol> &ms, int reps, F &&f) {
  f();  // warm
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) f();
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  return us / (reps * (double)ms.size());
}

int main(int argc, char **argv) {
  auto ms = load(argc > 1 ? argv[1] : "graphs.txt");
  double na = 0;
  for (auto &m : ms) na += m.n;
  printf("%zu molecules, mean %.1f heavy atoms\n\n", ms.size(), na / ms.size());

  volatile double sink = 0.0;
  double buf[2 * PMAX];

  double t_chi = time_it(ms, 200, [&] {
    for (auto &m : ms) {
      chi_paths(m, buf);
      sink += buf[0];
    }
  });
  double t_cyc = time_it(ms, 200, [&] {
    for (auto &m : ms) {
      double o[CMAX - 2] = {0};
      cycles(m, o);
      sink += o[0];
    }
  });
  std::vector<double> work, w;
  double t_res = time_it(ms, 50, [&] {
    for (auto &m : ms) sink += resistance(m, work);
  });
  double t_spec = time_it(ms, 50, [&] {
    for (auto &m : ms) sink += spectrum(m, work, w);
  });

  printf("  %-42s %8.2f us/mol\n", "Chi paths k<=7 (14 descriptors)", t_chi);
  printf("  %-42s %8.2f us/mol\n", "cycle enumeration k<=8", t_cyc);
  printf("  %-42s %8.2f us/mol\n", "resistance L+ (dposv, whole molecule)", t_res);
  printf("  %-42s %8.2f us/mol\n", "normalised Laplacian spectrum (dsyevd)", t_spec);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
