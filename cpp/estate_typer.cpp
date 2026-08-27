// Standalone verification harness for src/hume_core/estate_typer.h.
//
//   cpp/verify_estate.py            builds the tables and both corpora and runs all of this
//   ./cpp/estate_typer verify FILE  per-ATOM comparison against RDKit's own TypeAtoms
//   ./cpp/estate_typer dump  IN OUT re-emit the types so Python can build the 50 columns
//   ./cpp/estate_typer bench FILE   contended timing; see the note in bench()
//
// WHY PER ATOM AND NOT PER COLUMN. `NsCH3` is a count, so two atoms mistyped in opposite
// directions cancel and the column still matches. An atom's TYPE TUPLE cannot cancel: the
// reference file carries RDKit's own answer as pattern INDICES, in pattern order, and this
// compares the whole tuple. The 50-column check in verify_estate.py then runs on top of the
// types this binary emits, so it is a consequence of the atom check rather than a substitute.
//
// The file this reads is written by cpp/verify_estate.py; the format is documented there in
// dump_mols(). Nothing here has a tolerance -- the answer is a table lookup and a graph
// matching, so it is identical or it is a bug.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/hume_core/estate_typer.h"

using esttyper::Mol;
using estate_tbl::N_ROWS;
using estate_tbl::ROWS;

struct Ref { std::vector<uint8_t> k; std::vector<uint8_t> t; std::vector<int32_t> off; };

static std::vector<Mol> g_mols;
static std::vector<Ref> g_refs;

static void load(const char* path) {
  FILE* f = std::fopen(path, "r");
  if (!f) { std::fprintf(stderr, "cannot open %s (run cpp/verify_estate.py)\n", path); std::exit(1); }
  int nm = 0;
  if (std::fscanf(f, "%d", &nm) != 1) std::exit(1);
  g_mols.assign(nm, Mol());
  g_refs.assign(nm, Ref());
  for (int k = 0; k < nm; ++k) {
    Mol& m = g_mols[k];
    int nb = 0, n = 0;
    if (std::fscanf(f, "%d %d", &n, &nb) != 2) std::exit(1);
    m.alloc(n, 2 * nb);
    std::vector<int> deg(n, 0);
    for (int i = 0; i < n; ++i) {
      int z, d, nh, ar;
      if (std::fscanf(f, "%d %d %d %d", &z, &d, &nh, &ar) != 4) std::exit(1);
      m.z[i] = (uint8_t)(z > 255 ? 255 : z);
      m.nH[i] = (uint8_t)nh;
      m.arom[i] = (uint8_t)ar;
      deg[i] = d;
    }
    std::vector<int> eu(nb), ev(nb), ec(nb), cnt(n + 1, 0);
    for (int e = 0; e < nb; ++e) {
      if (std::fscanf(f, "%d %d %d", &eu[e], &ev[e], &ec[e]) != 3) std::exit(1);
      cnt[eu[e]]++; cnt[ev[e]]++;
    }
    for (int i = 0; i < n; ++i) m.start[i + 1] = m.start[i] + cnt[i];
    std::vector<int> cur(m.start.begin(), m.start.end() - 1);
    for (int e = 0; e < nb; ++e) {
      m.nbr[cur[eu[e]]] = ev[e]; m.btype[cur[eu[e]]++] = (uint8_t)ec[e];
      m.nbr[cur[ev[e]]] = eu[e]; m.btype[cur[ev[e]]++] = (uint8_t)ec[e];
    }
    m.finish();
    // The dumped degree is RDKit's GetDegree(); the CSR must reproduce it exactly or every
    // `D` primitive is being asked a different question than RDKit was.
    for (int i = 0; i < n; ++i)
      if (m.deg(i) != deg[i]) { std::fprintf(stderr, "degree mismatch mol %d atom %d\n", k, i); std::exit(1); }
    Ref& r = g_refs[k];
    r.k.resize(n); r.off.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
      int c = 0;
      if (std::fscanf(f, "%d", &c) != 1) std::exit(1);
      r.k[i] = (uint8_t)c;
      for (int q = 0; q < c; ++q) {
        int t = 0;
        if (std::fscanf(f, "%d", &t) != 1) std::exit(1);
        r.t.push_back((uint8_t)t);
      }
      r.off[i + 1] = (int32_t)r.t.size();
    }
  }
  std::fclose(f);
}

static void show(const Mol& m, int i, char* buf, size_t nbuf) {
  int p = std::snprintf(buf, nbuf, "Z=%d arom=%d D=%d H=%d bonds[", (int)m.z[i], (int)m.arom[i],
                        m.deg(i), (int)m.sh[i]);
  for (int e = m.start[i]; e < m.start[i + 1] && p < (int)nbuf - 8; ++e)
    p += std::snprintf(buf + p, nbuf - p, "%s%d>Z%d", e == m.start[i] ? "" : " ",
                       (int)m.btype[e], (int)m.z[m.nbr[e]]);
  std::snprintf(buf + p, nbuf - p, "]");
}

static int verify(const char* path) {
  load(path);
  long long nat = 0, nbad = 0, nmolbad = 0, nempty = 0, nmulti = 0;
  std::vector<long long> hit(N_ROWS, 0), bad(N_ROWS, 0);
  uint8_t got[esttyper::MAX_TYPES];
  int shown = 0;
  for (size_t k = 0; k < g_mols.size(); ++k) {
    const Mol& m = g_mols[k];
    const Ref& r = g_refs[k];
    bool molbad = false;
    for (int i = 0; i < m.n; ++i) {
      ++nat;
      const int ng = esttyper::typeAtom(m, i, got);
      const int nr = r.k[i];
      const uint8_t* want = &r.t[r.off[i]];
      if (nr == 0) ++nempty;
      if (nr > 1) ++nmulti;
      for (int q = 0; q < nr; ++q) ++hit[want[q]];
      bool same = (ng == nr);
      for (int q = 0; q < nr && same; ++q) same = (got[q] == want[q]);
      if (!same) {
        ++nbad; molbad = true;
        for (int q = 0; q < nr; ++q) ++bad[want[q]];
        for (int q = 0; q < ng; ++q) {
          bool inref = false;
          for (int w = 0; w < nr; ++w) inref |= (want[w] == got[q]);
          if (!inref) ++bad[got[q]];
        }
        if (shown < 25) {
          ++shown;
          char b[512]; show(m, i, b, sizeof b);
          std::fprintf(stderr, "  mol %zu atom %d %s\n    got ", k, i, b);
          for (int q = 0; q < ng; ++q) std::fprintf(stderr, "%s ", ROWS[got[q]].name);
          std::fprintf(stderr, "\n    want ");
          for (int q = 0; q < nr; ++q) std::fprintf(stderr, "%s ", ROWS[want[q]].name);
          std::fprintf(stderr, "\n");
        }
      }
    }
    if (molbad) ++nmolbad;
  }
  std::printf("\nper-atom exactness vs rdkit.Chem.EState.AtomTypes.TypeAtoms  [%s]\n", path);
  std::printf("  %zu molecules, %lld atoms\n", g_mols.size(), nat);
  std::printf("  identical type tuple : %lld / %lld  (%.6f%%)\n", nat - nbad, nat,
              100.0 * (double)(nat - nbad) / (double)nat);
  std::printf("  molecules exact      : %lld / %lld\n", (long long)(g_mols.size() - nmolbad),
              (long long)g_mols.size());
  std::printf("  atoms matching NO pattern (empty tuple, contributes to nothing): %lld\n", nempty);
  std::printf("  atoms matching MORE THAN ONE pattern: %lld\n", nmulti);

  std::printf("\n  %-9s %12s %10s %9s\n", "pattern", "atoms", "exact", "pct");
  int cold = 0;
  for (int r = 0; r < N_ROWS; ++r) {
    if (hit[r] == 0) { ++cold; continue; }
    std::printf("  %-9s %12lld %10lld %8.4f%%%s\n", ROWS[r].name, hit[r], hit[r] - bad[r],
                100.0 * (double)(hit[r] - bad[r]) / (double)hit[r], bad[r] ? "   <-- MISMATCH" : "");
  }
  std::printf("\n  patterns never exercised by this corpus: %d / %d\n", cold, N_ROWS);
  for (int r = 0; r < N_ROWS; ++r)
    if (hit[r] == 0) std::printf("    %-9s %s\n", ROWS[r].name, ROWS[r].smarts);
  return nbad ? 1 : 0;
}

// Re-emit what the C++ decided, so verify_estate.py can build mordred's 50 columns from it and
// compare against mordred itself. One line per molecule: nat, then per atom `k i0 i1 ...`.
static int dump(const char* in, const char* out) {
  load(in);
  FILE* f = std::fopen(out, "w");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", out); return 2; }
  uint8_t got[esttyper::MAX_TYPES];
  for (const Mol& m : g_mols) {
    std::fprintf(f, "%d", m.n);
    for (int i = 0; i < m.n; ++i) {
      const int k = esttyper::typeAtom(m, i, got);
      std::fprintf(f, " %d", k);
      for (int q = 0; q < k; ++q) std::fprintf(f, " %d", (int)got[q]);
    }
    std::fprintf(f, "\n");
  }
  std::fclose(f);
  std::printf("wrote %s  |  %zu molecules\n", out, g_mols.size());
  return 0;
}

// CONTENDED. This machine is shared, so the absolute number is an upper bound on a quiet one.
// The loop runs the FULL aggregation (types + both accumulators) over every molecule and
// consumes all 79 outputs, because -O3 will happily dead-code an aggregate nobody reads and
// then report a fraction of the work. There is no per-molecule cache to hit: the Mol structs are
// plain arrays and typeAtom() memoises nothing, which is the failure mode that has bitten this
// project on the RDKit/mordred side four times.
static void bench(const char* path) {
  load(path);
  long long nat = 0;
  for (const Mol& m : g_mols) nat += m.n;
  std::vector<double> es;
  int32_t N[N_ROWS]; double S[N_ROWS];
  std::vector<double> reps;
  double sink = 0;
  for (int rep = 0; rep < 11; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (const Mol& m : g_mols) {
      es.assign(m.n, 1.0);                       // stand-in index; estate_from() supplies the real one
      esttyper::aggregate(m, es.data(), N, S);
      for (int t = 0; t < N_ROWS; ++t) sink += N[t] + S[t];
    }
    auto t1 = std::chrono::steady_clock::now();
    reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)g_mols.size());
  }
  std::sort(reps.begin(), reps.end());
  const double med = reps[reps.size() / 2];
  std::printf("\nestate typer + 79-column aggregation, %zu molecules, mean %.1f heavy atoms, 11 reps\n",
              g_mols.size(), (double)nat / (double)g_mols.size());
  std::printf("  median %.3f us/mol   min %.3f   max %.3f   (spread %+.1f%% / %+.1f%%)  CONTENDED\n",
              med, reps.front(), reps.back(), 100.0 * (reps.front() - med) / med,
              100.0 * (reps.back() - med) / med);
  std::printf("  %.4f us per atom\n", med / ((double)nat / (double)g_mols.size()));
  if (sink == 12345.6789) std::printf("");
}

int main(int argc, char** argv) {
  esttyper::selfCheck();
  const char* cmd = argc > 1 ? argv[1] : "verify";
  const char* path = argc > 2 ? argv[2] : "cpp/estate_mols.txt";
  if (!std::strcmp(cmd, "verify")) return verify(path);
  if (!std::strcmp(cmd, "bench")) { bench(path); return 0; }
  if (!std::strcmp(cmd, "dump")) return dump(path, argc > 3 ? argv[3] : "cpp/estate_cpp_types.txt");
  std::fprintf(stderr, "usage: estate_typer [verify|bench|dump] FILE [OUT]\n");
  return 1;
}
