// Standalone verification harness for src/hume_core/infocontent.h.
//
//   cpp/verify_ic.py                    builds the tables and every corpus and drives all of this
//   ./cpp/infocontent values IN OUT     45 columns per molecule, for Python to compare
//   ./cpp/infocontent bench  IN         contended timing; see the note in bench()
//   ./cpp/infocontent flip              the worked example from the header comment, in full
//
// WHY THE COMPARISON IS DONE IN PYTHON AND NOT HERE. Unlike the E-state port there is no
// per-atom answer to check: an information content is a property of the whole equivalence-class
// histogram, so the only thing worth comparing is the column value. And the primary claim is
// DETERMINISM, which is a comparison of this binary's output against ITSELF on permuted inputs
// -- so the harness's job is to be a pure function from a dumped graph to 45 doubles, and
// nothing here may look at anything except the graph it was handed.
//
// THE VALUES ARE WRITTEN WITH %.17g. Determinism is checked BIT-IDENTICALLY on the Python side,
// so the exchange format has to round trip a double exactly; %.17g does and %.15g does not.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/hume_core/infocontent.h"

using infoic::Mol;
using infoic::Row;

static std::vector<Mol> g_mols;

// The dump format is written by cpp/verify_ic.py's dump_mols(); it is the boundary and nothing
// else. Anything this loader has to INFER rather than read would be a place the verification
// could drift away from what bindings.cpp will actually pass.
static void load(const char *path) {
  FILE *f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s (run cpp/verify_ic.py dump)\n", path);
    std::exit(1);
  }
  int nm = 0;
  if (std::fscanf(f, "%d", &nm) != 1) std::exit(1);
  g_mols.assign(nm, Mol());
  for (int k = 0; k < nm; k++) {
    Mol &m = g_mols[k];
    int n = 0, nb = 0;
    if (std::fscanf(f, "%d %d", &n, &nb) != 2) std::exit(1);
    m.alloc(n, nb);
    std::vector<int> deg(n, 0), seen(n, 0);
    for (int i = 0; i < n; i++) {
      int z, d, nh, chg, ar;
      if (std::fscanf(f, "%d %d %d %d %d", &z, &d, &nh, &chg, &ar) != 5) std::exit(1);
      m.z[i] = (uint8_t)(z > 255 ? 255 : z);
      m.nh[i] = (uint8_t)nh;
      m.chg[i] = (int8_t)chg;
      m.arom[i] = (uint8_t)ar;
      deg[i] = d;
    }
    for (int b = 0; b < nb; b++) {
      int u, v, code;
      double ord;
      if (std::fscanf(f, "%d %d %d %lf", &u, &v, &code, &ord) != 4) std::exit(1);
      m.bu[b] = u; m.bv[b] = v; m.bcode[b] = (uint8_t)code; m.bord[b] = ord;
      seen[u]++; seen[v]++;
    }
    // RDKit's own GetDegree() must fall out of the edge list, or every `(Z, degree)` token in
    // every code is answering a different question than mordred was asked.
    for (int i = 0; i < n; i++)
      if (seen[i] != deg[i]) {
        std::fprintf(stderr, "degree mismatch mol %d atom %d: edges %d, rdkit %d\n", k, i,
                     seen[i], deg[i]);
        std::exit(1);
      }
  }
  std::fclose(f);
}

static int values(const char *in, const char *out) {
  load(in);
  FILE *f = std::fopen(out, "w");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", out); return 2; }
  infoic::CodeBuilder cb;
  Row r;
  long long overflow = 0;
  int widest = 0;
  for (const Mol &m : g_mols) {
    infoic::compute(m, r, &cb);
    overflow += r.ipcOverflow;
    for (int c = 0; c < infoic::N_COLS; c++)
      std::fprintf(f, c ? " %.17g" : "%.17g", r.v[c]);
    // Trailing DIAGNOSTIC column, not a descriptor: the bit length of the largest EXACT
    // characteristic-polynomial coefficient. It is what says where RDKit's own double
    // arithmetic stopped being exact, and it is checked for determinism like everything else.
    std::fprintf(f, " %d\n", r.ipcMaxCoeffBits);
    if (r.ipcMaxCoeffBits > widest) widest = r.ipcMaxCoeffBits;
  }
  std::fclose(f);
  std::printf("wrote %s  |  %zu molecules  |  Ipc saturated at DBL_MAX on %lld  |  widest exact "
              "coefficient %d bits\n", out, g_mols.size(), overflow, widest);
  return 0;
}

// The worked example, computed rather than quoted, so the header comment cannot rot. mordred's
// two values are quoted from the mordred run (they need mordred); ours are computed here, under
// every permutation of the five heavy atoms, and must all be one value.
static int flip() {
  Mol base;
  infoic::buildWorkedExample(base);           // ON=Cc1ccccn1, 9 heavy atoms
  int p[9];
  for (int i = 0; i < 9; i++) p[i] = i;
  long long nperm = 0;
  Row ref;
  bool have = false;
  do {
    Mol m;
    m.alloc(base.n, base.nb);
    for (int i = 0; i < base.n; i++) {
      m.z[p[i]] = base.z[i]; m.nh[p[i]] = base.nh[i];
      m.arom[p[i]] = base.arom[i]; m.chg[p[i]] = base.chg[i];
    }
    for (int e = 0; e < base.nb; e++) {
      m.bu[e] = p[base.bu[e]]; m.bv[e] = p[base.bv[e]];
      m.bcode[e] = base.bcode[e]; m.bord[e] = base.bord[e];
    }
    Row r;
    infoic::compute(m, r);
    if (!have) { ref = r; have = true; }
    for (int c = 0; c < infoic::N_COLS; c++)
      if (std::memcmp(&ref.v[c], &r.v[c], sizeof(double)) != 0) {
        std::fprintf(stderr, "NOT DETERMINISTIC: %s moved under a permutation of ON=Cc1ccccn1\n",
                     infoic::columnNames()[c]);
        return 1;
      }
    ++nperm;
  } while (std::next_permutation(p, p + 9));
  std::printf("\nworked example  ON=Cc1ccccn1  (pyridine-2-carbaldehyde oxime, 9 heavy atoms)\n");
  std::printf("  all %lld permutations of the atoms give ONE value for all %d columns\n", nperm,
              infoic::N_COLS);
  std::printf("  ours     IC1 = %.17g   IC2 = %.17g\n",
              ref.v[infoic::F_IC * infoic::N_ORDERS + 1],
              ref.v[infoic::F_IC * infoic::N_ORDERS + 2]);
  std::printf("  mordred  IC1 = 2.682588730501833  or  2.8159220638351665   depending on\n"
              "           IC2 = 3.4565647621309532 or  3.5898980954642865   the numbering\n");
  std::printf("  (the two numberings differ by the single transposition (0 5); see the header\n"
              "   comment of src/hume_core/infocontent.h)\n");
  return 0;
}

// CONTENDED. This machine is shared, so the absolute number is an upper bound on a quiet one.
// The loop consumes every one of the 45 outputs, because -O3 will happily dead-code an
// aggregate nobody reads and then report a fraction of the work; and there is no per-molecule
// cache to hit, because compute() memoises nothing across molecules -- the CodeBuilder holds
// only scratch arrays sized to the current graph.
static void bench(const char *path) {
  load(path);
  long long nat = 0;
  for (const Mol &m : g_mols) nat += m.n;
  infoic::CodeBuilder cb;
  Row r;
  std::vector<double> reps;
  double sink = 0;
  for (int rep = 0; rep < 7; rep++) {
    auto t0 = std::chrono::steady_clock::now();
    for (const Mol &m : g_mols) {
      infoic::compute(m, r, &cb);
      for (int c = 0; c < infoic::N_COLS; c++) sink += r.v[c] == r.v[c] ? r.v[c] : 0.0;
    }
    auto t1 = std::chrono::steady_clock::now();
    reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                   (double)g_mols.size());
  }
  std::sort(reps.begin(), reps.end());
  const double med = reps[reps.size() / 2];
  std::printf("\nInformationContent (42 cols) + Ipc/AvgIpc/Log2Ipc, %zu molecules, "
              "mean %.1f heavy atoms, 7 reps\n",
              g_mols.size(), (double)nat / (double)g_mols.size());
  std::printf("  median %.3f us/mol   min %.3f   max %.3f   (spread %+.1f%% / %+.1f%%)  "
              "CONTENDED\n",
              med, reps.front(), reps.back(), 100.0 * (reps.front() - med) / med,
              100.0 * (reps.back() - med) / med);
  if (sink == 12345.6789) std::printf("");
}

int main(int argc, char **argv) {
  infoic::selfCheck();
  const char *cmd = argc > 1 ? argv[1] : "flip";
  if (!std::strcmp(cmd, "flip")) return flip();
  if (!std::strcmp(cmd, "values"))
    return values(argc > 2 ? argv[2] : "cpp/ic_in0.txt", argc > 3 ? argv[3] : "cpp/ic_out0.txt");
  if (!std::strcmp(cmd, "bench")) { bench(argc > 2 ? argv[2] : "cpp/ic_in0.txt"); return 0; }
  std::fprintf(stderr, "usage: infocontent [values IN OUT | bench IN | flip]\n");
  return 1;
}
