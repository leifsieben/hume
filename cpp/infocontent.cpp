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

// WHERE THE TIME GOES, per ORDER and per PHASE. Asked for before optimising, because "make the
// 400 us smaller" is a different job depending on whether it is one order or all six, and on
// whether it is building the codes or sorting them.
//
// The instrumentation is ~18 steady_clock reads per molecule. That is well under 1% of the
// number being measured, but it is not zero, so the TOTAL printed here runs slightly above what
// `bench` reports and the breakdown is the point rather than the absolute.
static void profile(const char *path) {
  load(path);
  infoic::CodeBuilder cb;
  infoic::Profile p;
  Row r;
  // CPU time, not wall: see the CpuClock note in infocontent.h. On this box a wall-clock
  // profile of the same binary on the same input reported one phase at 74 us and at 601 us on
  // consecutive runs; the CPU-time breakdown is stable to a few percent under the same load.
  const double t0 = infoic::CpuClock::nowUs();
  for (const Mol &m : g_mols) infoic::compute(m, r, &cb, &p);
  const double wall = infoic::CpuClock::nowUs() - t0;
  const double nm = (double)g_mols.size();
  double codes = 0, group = 0, ent = 0;
  for (int o = 0; o < infoic::N_ORDERS; o++) { codes += p.codes[o]; group += p.group[o]; ent += p.entropy[o]; }
  long long heavy = 0;
  for (const Mol &m : g_mols) heavy += m.n;
  std::printf("\nWHERE THE TIME GOES  |  %zu molecules, mean %.1f heavy atoms, %.1f with H added"
              "  |  CONTENDED MACHINE\n", g_mols.size(), heavy / nm, p.atoms / nm);
  std::printf("  CPU %.1f us/mol (instrumented; `bench` is the uninstrumented wall figure)\n\n", wall / nm);
  std::printf("  %-22s %10s %8s\n", "phase", "us/mol", "%");
  std::printf("  %-22s %10.2f %7.1f%%\n", "H-graph build + B", p.build / nm, 100 * p.build / wall);
  std::printf("  %-22s %10.2f %7.1f%%\n", "path DFS (all orders)", p.dfs / nm, 100 * p.dfs / wall);
  std::printf("  %-22s %10.2f %7.1f%%\n", "code construction", codes / nm, 100 * codes / wall);
  std::printf("  %-22s %10.2f %7.1f%%\n", "sort + class grouping", group / nm, 100 * group / wall);
  std::printf("  %-22s %10.2f %7.1f%%\n", "entropies (7 x 6 cols)", ent / nm, 100 * ent / wall);
  std::printf("  %-22s %10.2f %7.1f%%\n", "Ipc exact char poly", p.ipc / nm, 100 * p.ipc / wall);
  std::printf("\n  %-7s %9s %9s %9s %9s %8s %9s %9s\n", "order", "codes", "group", "entropy",
              "total", "%", "paths/mol", "classes");
  for (int o = 0; o < infoic::N_ORDERS; o++) {
    const double tot = p.codes[o] + p.group[o] + p.entropy[o];
    std::printf("  %-7d %9.2f %9.2f %9.2f %9.2f %7.1f%% %9.1f %9.1f\n", o, p.codes[o] / nm,
                p.group[o] / nm, p.entropy[o] / nm, tot / nm, 100 * tot / wall,
                (double)p.paths[o] / nm, (double)p.classes[o] / nm);
  }
  std::printf("\n  paths/mol is the number of root-to-leaf paths the layered tree enumerates --\n"
              "  the quantity that decides both the DFS cost and the bytes the sort moves.\n");
  std::printf("  Since 2026-08-28 ONE depth-5 DFS emits every path of every order, so the\n"
              "  enumeration is charged to `path DFS` and `code construction` holds only the\n"
              "  order-0 atomic-number count; `sort + class grouping` is per order as before.\n");
}

// ============================================================================================
// keycheck -- THE COLLISION EVIDENCE.
//
// infocontent.h replaced a 24-byte path key, ordered by memcmp, with a 128-bit packed PKey.
// The claim is that the replacement is an INJECTION whose numeric order is the memcmp order of
// the bytes -- not a hash with a small collision probability. This checks the claim the only way
// that is worth anything: it rebuilds the ORIGINAL byte keys here, from the git-HEAD algorithm
// transcribed below, and requires that the partition they induce is IDENTICAL to the one PKey
// induces -- same number of classes, same class SIZES, same class ORDER, and the same atom in
// each -- for every molecule at every order.
//
// It also reports, per molecule per order, DISTINCT BYTE CODES vs DISTINCT PKey CODES. Those two
// counts differing by one anywhere is exactly what a collision would look like, and it is the
// instrumentation the brief asked for. Because the map is injective they cannot differ, and the
// run over cpp/hard.smi is what turns that from an argument into a measurement.
// ============================================================================================
namespace refkey {

enum { KEY_BYTES = 24 };
struct Key { uint8_t b[KEY_BYTES]; };
inline bool keyLess(const Key &x, const Key &y) { return std::memcmp(x.b, y.b, KEY_BYTES) < 0; }

// Transcribed from src/hume_core/infocontent.h at git HEAD (CodeBuilder::codeFor / walk), the
// implementation PKey replaces. Deliberately unoptimised: it is the reference, not the product.
struct Ref {
  const infoic::HGraph *g = nullptr;
  std::vector<int32_t> dist, stamp;
  std::vector<int32_t> bfs;
  int32_t epoch = 0;

  void reset(const infoic::HGraph &gg) {
    g = &gg;
    dist.assign(gg.N, -1);
    stamp.assign(gg.N, 0);
    epoch = 0;
  }

  void codeFor(int root, int order, std::vector<Key> &out) {
    out.clear();
    ++epoch;
    bfs.clear();
    bfs.push_back(root);
    stamp[root] = epoch;
    dist[root] = 0;
    for (size_t h = 0; h < bfs.size(); h++) {
      const int u = bfs[h];
      if (dist[u] >= order) continue;
      for (int e = g->start[u]; e < g->start[u + 1]; e++) {
        const int v = g->nbr[e];
        if (stamp[v] != epoch) { stamp[v] = epoch; dist[v] = dist[u] + 1; bfs.push_back(v); }
      }
    }
    Key k;
    std::memset(k.b, 0, KEY_BYTES);
    k.b[0] = g->z[root];
    k.b[1] = g->deg[root];
    walk(root, 0, order, 2, k, out);
    std::sort(out.begin(), out.end(), keyLess);
  }

  void walk(int u, int d, int order, int pos, Key &k, std::vector<Key> &out) {
    bool leaf = true;
    if (d < order) {
      for (int e = g->start[u]; e < g->start[u + 1]; e++) {
        const int v = g->nbr[e];
        if (stamp[v] != epoch || dist[v] != d + 1) continue;
        leaf = false;
        const uint8_t s0 = k.b[pos], s1 = k.b[pos + 1], s2 = k.b[pos + 2];
        k.b[pos] = g->sym[e];
        k.b[pos + 1] = g->z[v];
        k.b[pos + 2] = g->deg[v];
        walk(v, d + 1, order, pos + 3, k, out);
        k.b[pos] = s0; k.b[pos + 1] = s1; k.b[pos + 2] = s2;
      }
    }
    if (leaf) {
      Key t = k;
      t.b[pos] = 0xFF;
      for (int q = pos + 1; q < KEY_BYTES; q++) t.b[q] = 0;
      out.push_back(t);
    }
  }
};

}  // namespace refkey

static int keycheck(const char *path) {
  load(path);
  infoic::CodeBuilder cb;
  refkey::Ref ref;
  std::vector<refkey::Key> paths;
  long long nmol = 0, ncmp = 0, badpart = 0, badcount = 0;
  long long distinctBytes = 0, distinctPk = 0;
  for (const Mol &m : g_mols) {
    infoic::HGraph g;
    g.build(m);
    const int A = g.N;
    cb.reset(g);
    cb.buildAll();
    ref.reset(g);
    for (int order = 1; order <= infoic::MAX_ORDER; order++) {
      // reference: byte blobs, one per atom
      std::vector<uint8_t> arena;
      std::vector<int32_t> off(A + 1, 0);
      for (int i = 0; i < A; i++) {
        ref.codeFor(i, order, paths);
        for (const refkey::Key &k : paths) arena.insert(arena.end(), k.b, k.b + refkey::KEY_BYTES);
        off[i + 1] = (int32_t)arena.size();
      }
      const uint8_t *ar = arena.data();
      const int32_t *of = off.data();
      std::vector<int32_t> ri(A), pi(A);
      for (int i = 0; i < A; i++) { ri[i] = i; pi[i] = i; }
      std::sort(ri.begin(), ri.end(), [ar, of](int32_t x, int32_t y) {
        const int32_t lx = of[x + 1] - of[x], ly = of[y + 1] - of[y];
        const int c = std::memcmp(ar + of[x], ar + of[y], (size_t)std::min(lx, ly));
        return c != 0 ? c < 0 : lx < ly;
      });
      const infoic::PKey *pa = cb.arena[order].data();
      const int32_t *po = cb.off[order].data();
      std::sort(pi.begin(), pi.end(), [pa, po](int32_t x, int32_t y) {
        const int32_t lx = po[x + 1] - po[x], ly = po[y + 1] - po[y];
        const int32_t n = lx < ly ? lx : ly;
        const infoic::PKey *px = pa + po[x], *py = pa + po[y];
        for (int32_t q = 0; q < n; q++) {
          if (px[q].hi != py[q].hi) return px[q].hi < py[q].hi;
          if (px[q].lo != py[q].lo) return px[q].lo < py[q].lo;
        }
        return lx < ly;
      });
      // class boundaries under each encoding
      std::vector<int> rc, pc;
      for (int a = 0; a < A;) {
        int b = a + 1;
        const int32_t la = of[ri[a] + 1] - of[ri[a]];
        while (b < A) {
          const int32_t lb = of[ri[b] + 1] - of[ri[b]];
          if (lb != la || std::memcmp(ar + of[ri[a]], ar + of[ri[b]], (size_t)la) != 0) break;
          b++;
        }
        rc.push_back(b - a);
        a = b;
      }
      for (int a = 0; a < A;) {
        int b = a + 1;
        const int32_t la = po[pi[a] + 1] - po[pi[a]];
        while (b < A) {
          const int32_t lb = po[pi[b] + 1] - po[pi[b]];
          if (lb != la) break;
          bool same = true;
          for (int32_t q = 0; q < la; q++)
            if (!(pa[po[pi[a]] + q] == pa[po[pi[b]] + q])) { same = false; break; }
          if (!same) break;
          b++;
        }
        pc.push_back(b - a);
        a = b;
      }
      distinctBytes += (long long)rc.size();
      distinctPk += (long long)pc.size();
      if (rc.size() != pc.size()) badcount++;
      if (rc != pc) badpart++;
      // and the atoms themselves, in the same order
      if (ri != pi) {
        // A tie inside a class is not a difference: only the CLASS boundaries are load bearing.
        // Compare the sorted atom set within each class instead.
        std::vector<int32_t> a1 = ri, a2 = pi;
        int at = 0;
        for (size_t c = 0; c < rc.size() && c < pc.size(); c++) {
          std::sort(a1.begin() + at, a1.begin() + at + rc[c]);
          std::sort(a2.begin() + at, a2.begin() + at + pc[c]);
          at += rc[c];
        }
        if (a1 != a2) badpart++;
      }
      ncmp++;
    }
    nmol++;
  }
  std::printf("\nkeycheck  |  %lld molecules x %d orders = %lld partitions compared\n", nmol,
              infoic::MAX_ORDER, ncmp);
  std::printf("  distinct codes, ORIGINAL 24-byte keys : %lld\n", distinctBytes);
  std::printf("  distinct codes, 128-bit PKey          : %lld\n", distinctPk);
  std::printf("  partitions where the two DISAGREE     : %lld   (class count %lld)\n", badpart,
              badcount);
  std::printf("  %s\n", (badpart || badcount || distinctBytes != distinctPk)
                            ? "*** FAIL: the packed key is not faithful ***"
                            : "OK -- same classes, same sizes, same order, zero collisions");
  return (badpart || badcount || distinctBytes != distinctPk) ? 1 : 0;
}

int main(int argc, char **argv) {
  infoic::selfCheck();
  const char *cmd = argc > 1 ? argv[1] : "flip";
  if (!std::strcmp(cmd, "flip")) return flip();
  if (!std::strcmp(cmd, "values"))
    return values(argc > 2 ? argv[2] : "cpp/ic_in0.txt", argc > 3 ? argv[3] : "cpp/ic_out0.txt");
  if (!std::strcmp(cmd, "bench")) { bench(argc > 2 ? argv[2] : "cpp/ic_in0.txt"); return 0; }
  if (!std::strcmp(cmd, "profile")) { profile(argc > 2 ? argv[2] : "cpp/ic_in0.txt"); return 0; }
  if (!std::strcmp(cmd, "keycheck")) return keycheck(argc > 2 ? argv[2] : "cpp/ic_in0.txt");
  std::fprintf(stderr,
               "usage: infocontent [values IN OUT | bench IN | profile IN | keycheck IN | flip]\n");
  return 1;
}
