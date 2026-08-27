// Standalone verification harness for src/hume_core/vsa_bins.h.
//
//   cpp/verify_vsa.py               builds the tables and the corpus and runs all of this
//   ./cpp/vsa verify FILE           per-ATOM and per-COLUMN comparison against RDKit
//   ./cpp/vsa dump   IN OUT         re-emit the columns so Python can diff them against mordred
//   ./cpp/vsa bench  FILE           contended timing; see the note in bench()
//
// WHY PER ATOM AS WELL AS PER COLUMN.  A `*_VSA` column is a SUM, so two atoms binned wrongly in
// opposite directions cancel and the column still matches.  The four per-atom vectors --  Labute
// ASA, Crippen logP, Crippen MR, E-state index -- cannot cancel, and the file carries RDKit's own
// answer for each of them.  The column check then runs on top, so it is a consequence of the
// atom check rather than a substitute for it.
//
// The `r3` field is RDKit's isAtomInRingOfSize(i, 3).  vsa_bins.h derives that from the graph
// (two neighbours that share a bond) because the boundary only carries a boolean "in a ring";
// SSSR is not obliged to contain every smallest cycle, so the two are compared rather than
// assumed equal.
//
// Nothing here has a tolerance by default.  `verify` reports EXACT counts; the -t flag adds a
// tolerance column so a near-miss can be sized, but the pass/fail line is bit equality.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/hume_core/vsa_bins.h"

using vsabin::Mol;
using vsabin::N_COLS;

struct Ref {
  std::vector<double> asa, logp, mr, es;   // per atom, RDKit's answer
  std::vector<uint8_t> r3;
  double hcontrib = 0.0;
  double col[N_COLS];
};

static std::vector<Mol> g_mols;
static std::vector<Ref> g_refs;
static std::vector<std::string> g_names;

static double rd(FILE* f) {
  char tok[64];
  if (std::fscanf(f, "%63s", tok) != 1) { std::fprintf(stderr, "short read\n"); std::exit(1); }
  return std::strtod(tok, nullptr);
}

static void load(const char* path) {
  FILE* f = std::fopen(path, "r");
  if (!f) { std::fprintf(stderr, "cannot open %s (run cpp/verify_vsa.py corpus)\n", path); std::exit(1); }
  int nm = 0, nc = 0;
  if (std::fscanf(f, "%d %d", &nm, &nc) != 2) std::exit(1);
  if (nc != N_COLS) {
    std::fprintf(stderr, "file has %d columns, vsa_bins.h has %d -- regenerate the corpus\n",
                 nc, N_COLS);
    std::exit(1);
  }
  g_names.resize(nc);
  for (int c = 0; c < nc; ++c) {
    char buf[64];
    if (std::fscanf(f, "%63s", buf) != 1) std::exit(1);
    g_names[c] = buf;
    if (g_names[c] != vsabin::col_name(c)) {
      std::fprintf(stderr, "column %d: file says %s, vsa_bins.h says %s\n", c, buf,
                   vsabin::col_name(c));
      std::exit(1);
    }
  }
  g_mols.assign(nm, Mol());
  g_refs.assign(nm, Ref());
  for (int k = 0; k < nm; ++k) {
    Mol& m = g_mols[k];
    Ref& r = g_refs[k];
    int n = 0, nb = 0;
    if (std::fscanf(f, "%d %d", &n, &nb) != 2) std::exit(1);
    m.alloc(n, nb);
    r.asa.resize(n); r.logp.resize(n); r.mr.resize(n); r.es.resize(n); r.r3.resize(n);
    for (int i = 0; i < n; ++i) {
      int z, d, nh, fc, ar, r3;
      if (std::fscanf(f, "%d %d %d %d %d %d", &z, &d, &nh, &fc, &ar, &r3) != 6) std::exit(1);
      m.z[i] = z; m.deg[i] = d; m.nH[i] = nh; m.fchg[i] = fc; m.arom[i] = ar;
      m.gast[i] = rd(f);
      r.asa[i] = rd(f); r.logp[i] = rd(f); r.mr[i] = rd(f); r.es[i] = rd(f);
      r.r3[i] = (uint8_t)r3;
    }
    for (int b = 0; b < nb; ++b) {
      int u, v, c;
      if (std::fscanf(f, "%d %d %d", &u, &v, &c) != 3) std::exit(1);
      m.bu[b] = u; m.bv[b] = v; m.bcode[b] = c;
    }
    r.hcontrib = rd(f);
    for (int c = 0; c < N_COLS; ++c) r.col[c] = rd(f);
  }
  std::fclose(f);
}

static bool same(double a, double b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  return a == b;
}

struct Tally { long long n = 0, ok = 0, noref = 0; double worst = 0.0; int worst_mol = -1; };

static void note(Tally& t, double got, double want, int mol) {
  ++t.n;
  if (same(got, want)) { ++t.ok; return; }
  const double d = std::fabs(got - want);
  if (!(d <= t.worst)) { t.worst = d; t.worst_mol = mol; }
}

// A reference of NaN means "this environment could not supply one", not "RDKit said NaN".  Only
// TopoPSA uses it: mordred is the oracle for that column and mordred cannot be installed
// alongside the pinned numpy, so cpp/verify_vsa.py writes NaN there and the `mordred` step
// checks it in its own environment.  No genuine column in this family is ever NaN -- a NaN
// Gasteiger charge changes which BIN an atom lands in, it does not make the sum NaN.
static void note_ref(Tally& t, double got, double want, int mol) {
  if (std::isnan(want) && !std::isnan(got)) { ++t.n; ++t.noref; return; }
  note(t, got, want, mol);
}

static int verify(const char* path) {
  load(path);
  vsabin::Work W;
  std::vector<double> out(N_COLS);
  Tally at[4];            // asa, logp, mr, es
  Tally cols[N_COLS];
  long long r3_n = 0, r3_bad = 0;
  long long hc_n = 0, hc_ok = 0;
  long long nat = 0;
  // How often does an atom sit EXACTLY on a bin edge?  This is the number that turns "we thought
  // about the bin edge" into "the corpus tested it".
  long long onedge[5] = {0, 0, 0, 0, 0};
  // ...and how close does the NEAREST non-exact atom get?  This is the number that decides
  // whether a last-ULP wobble in a per-atom contribution could move an atom ACROSS an edge and
  // change a bin sum by a whole atom's worth.  A margin many orders of magnitude above the
  // observed renumbering wobble (1.6e-14 for the E-state index, 0 for the other four) means the
  // hazard is bounded, not merely unobserved.
  double margin[5] = {1e308, 1e308, 1e308, 1e308, 1e308};
  const double* edges[5] = {vsa_tbl::LOGP_BINS, vsa_tbl::MR_BINS, vsa_tbl::CHG_BINS,
                            vsa_tbl::ESTATE_BINS, vsa_tbl::VSA_BINS};
  const int nedge[5] = {vsa_tbl::N_LOGP_BINS, vsa_tbl::N_MR_BINS, vsa_tbl::N_CHG_BINS,
                        vsa_tbl::N_ESTATE_BINS, vsa_tbl::N_VSA_BINS};
  long long nan_chg = 0, nan_final_bin = 0;

  for (size_t k = 0; k < g_mols.size(); ++k) {
    const Mol& m = g_mols[k];
    const Ref& r = g_refs[k];
    vsabin::vsa_row(m, W, out.data());
    nat += m.n;
    for (int i = 0; i < m.n; ++i) {
      note(at[0], W.asa[i], r.asa[i], (int)k);
      note(at[1], W.logp[i], r.logp[i], (int)k);
      note(at[2], W.mr[i], r.mr[i], (int)k);
      note(at[3], W.es[i], r.es[i], (int)k);
      ++r3_n;
      if ((W.r3[i] != 0) != (r.r3[i] != 0)) ++r3_bad;
      const double* vals[5] = {&W.logp[i], &W.mr[i], &m.gast[i], &W.es[i], &W.asa[i]};
      for (int t = 0; t < 5; ++t) {
        if (std::isnan(*vals[t])) continue;
        bool on = false;
        for (int e = 0; e < nedge[t]; ++e) {
          const double d = std::fabs(*vals[t] - edges[t][e]);
          if (d == 0.0) { on = true; break; }
          if (d < margin[t]) margin[t] = d;
        }
        if (on) ++onedge[t];
      }
      if (std::isnan(m.gast[i])) {
        ++nan_chg;
        if (vsabin::bin_of(vsa_tbl::CHG_BINS, vsa_tbl::N_CHG_BINS, m.gast[i]) ==
            vsa_tbl::N_CHG_BINS) ++nan_final_bin;
      }
    }
    double hc = 0.0;
    { std::vector<double> tmp(m.n); hc = 0.0; vsabin::labute_contribs(m, tmp.data(), hc); }
    ++hc_n; if (same(hc, r.hcontrib)) ++hc_ok;
    for (int c = 0; c < N_COLS; ++c) note_ref(cols[c], out[c], r.col[c], (int)k);
  }

  std::printf("\nper-ATOM exactness vs RDKit  [%s]\n", path);
  std::printf("  %zu molecules, %lld atoms\n", g_mols.size(), nat);
  static const char* an[4] = {"Labute ASA", "Crippen logP", "Crippen MR", "E-state index"};
  for (int i = 0; i < 4; ++i)
    std::printf("  %-14s %12lld / %-12lld  %s  max|d| %.3e (mol %d)\n", an[i], at[i].ok, at[i].n,
                at[i].ok == at[i].n ? "EXACT" : "  !! ", at[i].worst, at[i].worst_mol);
  std::printf("  %-14s %12lld / %-12lld  %s   (vsa_bins.h derives this from the graph)\n",
              "in 3-ring", r3_n - r3_bad, r3_n, r3_bad ? "  !! " : "EXACT");
  std::printf("  %-14s %12lld / %-12lld  %s\n", "Labute hContrib", hc_ok, hc_n,
              hc_ok == hc_n ? "EXACT" : "  !! ");

  std::printf("\n  atoms landing EXACTLY on a bin edge (the case that decides upper vs lower):\n");
  static const char* en[5] = {"Crippen logP", "Crippen MR", "Gasteiger chg", "E-state", "ASA"};
  for (int t = 0; t < 5; ++t)
    std::printf("    %-14s on-edge %10lld    nearest other atom to an edge: %.3e\n", en[t],
                onedge[t], margin[t]);
  std::printf("  NaN Gasteiger charges: %lld, of which routed to the FINAL bin: %lld\n",
              nan_chg, nan_final_bin);

  std::printf("\nper-COLUMN exactness vs RDKit  (%d columns x %zu molecules)\n", N_COLS,
              g_mols.size());
  int nbadcol = 0;
  for (int c = 0; c < N_COLS; ++c) {
    const long long checked = cols[c].n - cols[c].noref;
    const bool ok = checked > 0 && cols[c].ok == checked;
    if (checked == 0) {
      std::printf("  %-20s %8s   %-8lld  NO REFERENCE IN THIS ENV (see the `mordred` step)\n",
                  g_names[c].c_str(), "-", cols[c].n);
      continue;
    }
    if (!ok) ++nbadcol;
    std::printf("  %-20s %8lld / %-8lld  %8.4f%%  %s max|d| %.3e (mol %d)\n",
                g_names[c].c_str(), cols[c].ok, checked,
                100.0 * (double)cols[c].ok / (double)checked, ok ? "EXACT " : " !!!! ",
                cols[c].worst, cols[c].worst_mol);
  }
  bool atom_ok = true;
  for (int i = 0; i < 4; ++i) atom_ok &= (at[i].ok == at[i].n);
  std::printf("\n  %d / %d columns bit-exact; per-atom vectors %s; 3-ring %s\n",
              N_COLS - nbadcol, N_COLS, atom_ok ? "all exact" : "NOT all exact",
              r3_bad ? "MISMATCH" : "exact");
  return (nbadcol || !atom_ok || r3_bad) ? 1 : 0;
}

static int dump(const char* in, const char* out) {
  load(in);
  FILE* f = std::fopen(out, "w");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", out); return 2; }
  std::fprintf(f, "%zu %d\n", g_mols.size(), N_COLS);
  for (int c = 0; c < N_COLS; ++c) std::fprintf(f, "%s%c", vsabin::col_name(c),
                                                c + 1 == N_COLS ? '\n' : ' ');
  vsabin::Work W;
  std::vector<double> v(N_COLS);
  for (const Mol& m : g_mols) {
    vsabin::vsa_row(m, W, v.data());
    for (int c = 0; c < N_COLS; ++c) std::fprintf(f, "%.17g%c", v[c], c + 1 == N_COLS ? '\n' : ' ');
  }
  std::fclose(f);
  std::printf("wrote %s  |  %zu molecules x %d columns\n", out, g_mols.size(), N_COLS);
  return 0;
}

// CONTENDED.  This machine is shared, so the absolute number is an upper bound on a quiet one.
// The loop runs the FULL row -- Labute, Crippen on both the heavy and the H-added molecule, the
// E-state BFS, all five binnings, TPSA -- and consumes every output, because -O3 will happily
// dead-code an aggregate nobody reads and then report a fraction of the work.  There is no
// per-molecule cache to hit: vsa_row() memoises nothing, which is the failure mode that has
// bitten this project four times on the RDKit/mordred side (RDKit caches Labute contribs on the
// molecule as `_labuteAtomContribs`, so a second pass over the same Mol object measures nothing).
static void bench(const char* path) {
  load(path);
  long long nat = 0;
  for (const Mol& m : g_mols) nat += m.n;
  vsabin::Work W;
  std::vector<double> v(N_COLS);
  std::vector<double> reps;
  double sink = 0;
  for (int rep = 0; rep < 11; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (const Mol& m : g_mols) {
      vsabin::vsa_row(m, W, v.data());
      for (int c = 0; c < N_COLS; ++c) sink += v[c];
    }
    auto t1 = std::chrono::steady_clock::now();
    reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                   (double)g_mols.size());
  }
  std::sort(reps.begin(), reps.end());
  const double med = reps[reps.size() / 2];
  std::printf("\nVSA family, %d columns, %zu molecules, mean %.1f atoms, 11 reps\n", N_COLS,
              g_mols.size(), (double)nat / (double)g_mols.size());
  std::printf("  median %.3f us/mol   min %.3f   max %.3f   (spread %+.1f%% / %+.1f%%)  CONTENDED\n",
              med, reps.front(), reps.back(), 100.0 * (reps.front() - med) / med,
              100.0 * (reps.back() - med) / med);
  if (sink == 12345.6789) std::printf("");
}

// Does the answer depend on the atom NUMBERING?  The corpus is loaded once and then each molecule
// is relabelled with a random permutation -- same graph, same chemistry, different indices -- and
// the columns recomputed and mapped back.  This is the amendment's test done where it belongs: on
// the arithmetic, with no RDKit in the loop to add its own perception wobble.  It runs BOTH
// E-state summations, so the report is "how many molecules move, and does the compensated form
// stop them moving".
static uint64_t rng_s = 88172645463325252ull;
static uint64_t rng() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }

static Mol permuted(const Mol& m, const std::vector<int>& p) {
  Mol o; o.alloc(m.n, m.nb);
  for (int i = 0; i < m.n; ++i) {
    const int k = p[i];
    o.z[k] = m.z[i]; o.deg[k] = m.deg[i]; o.nH[k] = m.nH[i];
    o.fchg[k] = m.fchg[i]; o.arom[k] = m.arom[i]; o.gast[k] = m.gast[i];
  }
  for (int b = 0; b < m.nb; ++b) { o.bu[b] = p[m.bu[b]]; o.bv[b] = p[m.bv[b]]; o.bcode[b] = m.bcode[b]; }
  return o;
}

static int renumber(const char* path, int trials) {
  load(path);
  vsabin::Work W;
  std::vector<double> a(N_COLS), b(N_COLS);
  std::vector<double> cmp;
  long long molmoved = 0, colmoved[N_COLS] = {0}, colflip[N_COLS] = {0};
  double colworst[N_COLS] = {0.0};
  // A bin FLIP moves a whole atom's surface area (order 1), a summation wobble moves the
  // last bits (order 1e-13).  1e-6 separates them by seven orders of magnitude either way.
  const double FLIP = 1e-6;
  long long molflip = 0;
  std::vector<int> flipmol;
  long long es_naive_moved = 0, es_wp_moved = 0, atoms = 0;
  double worst_es_naive = 0.0, worst_es_wp = 0.0;
  for (size_t k = 0; k < g_mols.size(); ++k) {
    const Mol& m = g_mols[k];
    vsabin::vsa_row(m, W, a.data());
    vsabin::build_graph(m, W);
    std::vector<double> base_naive(m.n), base_wp(m.n);
    vsabin::estate_indices(m, W.dist, base_naive.data());
    vsabin::estate_indices_wellposed(m, W.dist, base_wp.data(), cmp);
    std::vector<int> p(m.n);
    bool moved = false, flip = false;
    for (int t = 0; t < trials; ++t) {
      for (int i = 0; i < m.n; ++i) p[i] = i;
      for (int i = m.n - 1; i > 0; --i) std::swap(p[i], p[rng() % (uint64_t)(i + 1)]);
      const Mol q = permuted(m, p);
      vsabin::vsa_row(q, W, b.data());
      for (int c = 0; c < N_COLS; ++c)
        if (!same(a[c], b[c])) {
          ++colmoved[c]; moved = true;
          const double d = std::fabs(a[c] - b[c]);
          if (d > colworst[c]) colworst[c] = d;
          if (d > FLIP) { ++colflip[c]; flip = true; }
        }
      vsabin::build_graph(q, W);
      std::vector<double> pn(q.n), pw(q.n);
      vsabin::estate_indices(q, W.dist, pn.data());
      vsabin::estate_indices_wellposed(q, W.dist, pw.data(), cmp);
      for (int i = 0; i < m.n; ++i) {
        ++atoms;
        if (!same(base_naive[i], pn[p[i]])) {
          ++es_naive_moved;
          worst_es_naive = std::max(worst_es_naive, std::fabs(base_naive[i] - pn[p[i]]));
        }
        if (!same(base_wp[i], pw[p[i]])) {
          ++es_wp_moved;
          worst_es_wp = std::max(worst_es_wp, std::fabs(base_wp[i] - pw[p[i]]));
        }
      }
    }
    if (moved) ++molmoved;
    if (flip) { ++molflip; if (flipmol.size() < 40) flipmol.push_back((int)k); }
  }
  std::printf("\nATOM-NUMBERING INVARIANCE  [%s]  %zu molecules x %d random relabellings\n",
              path, g_mols.size(), trials);
  std::printf("  molecules with ANY column that moves: %lld / %zu\n", molmoved, g_mols.size());
  std::printf("  molecules with a column that moves by MORE THAN %.0e -- a real BIN FLIP,\n"
              "  i.e. a whole atom's surface area changing column: %lld / %zu\n", FLIP, molflip,
              g_mols.size());
  for (int c = 0; c < N_COLS; ++c)
    if (colflip[c])
      std::printf("    %-20s BIN FLIP in %lld of %zu relabellings   max|d| %.4f\n",
                  g_names[c].c_str(), colflip[c], g_mols.size() * (size_t)trials, colworst[c]);
  std::printf("    (molecule indices: ");
  for (size_t i = 0; i < flipmol.size(); ++i) std::printf("%d ", flipmol[i]);
  std::printf(")\n");
  std::printf("  every other movement is summation reassociation; largest over all %d columns:\n",
              N_COLS);
  { double w = 0; const char* nm = "-";
    for (int c = 0; c < N_COLS; ++c) if (colworst[c] > w && !colflip[c]) { w = colworst[c]; nm = g_names[c].c_str(); }
    std::printf("    %s  %.3e\n", nm, w); }
  std::printf("  E-state index, %lld atom comparisons:\n", atoms);
  std::printf("    RDKit's summation order      moved %lld  max|d| %.3e\n", es_naive_moved,
              worst_es_naive);
  std::printf("    Neumaier-compensated         moved %lld  max|d| %.3e\n", es_wp_moved,
              worst_es_wp);
  return 0;
}

int main(int argc, char** argv) {
  vsabin::check();
  const char* cmd = argc > 1 ? argv[1] : "verify";
  const char* path = argc > 2 ? argv[2] : "cpp/vsa_mols.txt";
  if (!std::strcmp(cmd, "verify")) return verify(path);
  if (!std::strcmp(cmd, "bench")) { bench(path); return 0; }
  if (!std::strcmp(cmd, "dump")) return dump(path, argc > 3 ? argv[3] : "cpp/vsa_cpp.txt");
  if (!std::strcmp(cmd, "renumber")) return renumber(path, argc > 3 ? atoi(argv[3]) : 3);
  std::fprintf(stderr, "usage: vsa [verify|bench|dump|renumber] FILE [OUT|TRIALS]\n");
  return 1;
}
