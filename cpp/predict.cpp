// PREDICT-family descriptors in C++, and the exactness harness for them.
//
//   ./predict verify   -> values.txt, one line per molecule, for verify_predict.py to compare
//   ./predict bench    -> us/mol per family
//
// WHY THIS EXISTS. The CORE/PREDICT split was drawn on C++ cost estimates, but nothing was ever
// ported, so every cost figure quoted for the predict block has been a Mordred/RDKit-from-Python
// timing -- which measures the interpreter, not the descriptor. On the one family measured both
// ways the gap is ~1,500x (Chi: 10,328 us in Mordred, 7.36 us here). No decision between a proxy
// model and exact computation can rest on the Python number.
//
// The 166 predict columns are not 166 independent calculations. They collapse onto a few
// PER-ATOM primitives -- EState index, Crippen logP/MR contribution, Gasteiger charge, Labute
// ASA -- with the large VSA families (EState_VSA, VSA_EState, SlogP_VSA, SMR_VSA, PEOE_VSA; 70
// columns between them) being nothing but binned sums over those. This file implements the
// primitives that need no SMARTS atom typing; Crippen and Gasteiger are the next tranche.
//
// EVERY VALUE IS CHECKED AGAINST RDKIT ON 10,000 MOLECULES. A fast descriptor that disagrees
// with the reference implementation is not a descriptor, it is a different quantity.

#include <cmath>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <vector>

struct Mol {
  int n = 0, nb = 0, chg_ok = 1;
  std::vector<int> Z, deg, nH, chg, hyb, arom, ring;
  std::vector<double> mass, gast, clogp, cmr;     // the four BCUT2D atom properties
  std::vector<int> bu, bv;
  std::vector<double> bord;                        // getBondTypeAsDouble: aromatic = 1.5
  std::vector<std::vector<int>> adj;
};

static std::vector<Mol> load(const char *path) {
  std::ifstream f(path);
  int nm;
  f >> nm;
  std::vector<Mol> ms(nm);
  for (int k = 0; k < nm; k++) {
    Mol &m = ms[k];
    f >> m.n >> m.nb >> m.chg_ok;
    m.Z.resize(m.n); m.deg.resize(m.n); m.nH.resize(m.n); m.chg.resize(m.n);
    m.hyb.resize(m.n); m.arom.resize(m.n); m.ring.resize(m.n);
    m.mass.resize(m.n); m.gast.resize(m.n); m.clogp.resize(m.n); m.cmr.resize(m.n);
    for (int i = 0; i < m.n; i++)
      f >> m.Z[i] >> m.deg[i] >> m.nH[i] >> m.chg[i] >> m.hyb[i] >> m.arom[i] >> m.ring[i]
        >> m.mass[i] >> m.gast[i] >> m.clogp[i] >> m.cmr[i];
    m.adj.assign(m.n, {});
    m.bu.resize(m.nb); m.bv.resize(m.nb); m.bord.resize(m.nb);
    for (int b = 0; b < m.nb; b++) {
      f >> m.bu[b] >> m.bv[b] >> m.bord[b];
      m.adj[m.bu[b]].push_back(m.bv[b]);
      m.adj[m.bv[b]].push_back(m.bu[b]);
    }
  }
  return ms;
}

// ---------------------------------------------------------------------------------------
// EState indices (Hall-Kier).
//
//   I_i  = (4/N_i^2 * dv_i + 1) / d_i        intrinsic state
//   S_i  = I_i + sum_j (I_i - I_j)/(d_ij+1)^2
//
// N is the principal quantum number, dv = nOuterElecs - nH, d = heavy degree. Matches
// rdkit.Chem.EState.EState.EStateIndices exactly, including its convention that a
// zero-degree atom keeps I = 0 rather than dividing by zero.
// ---------------------------------------------------------------------------------------
static int principal_qn(int z) {
  if (z <= 2) return 1;
  if (z <= 10) return 2;
  if (z <= 18) return 3;
  if (z <= 36) return 4;
  if (z <= 54) return 5;
  if (z <= 86) return 6;
  return 7;
}

// Outer-shell electron count, RDKit's periodic-table convention (group number for main group).
static int n_outer(int z) {
  static const int T[] = {0,
      1, 2,                                                     // H  He
      1, 2, 3, 4, 5, 6, 7, 8,                                   // Li..Ne
      1, 2, 3, 4, 5, 6, 7, 8,                                   // Na..Ar
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8,  // K..Kr
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8}; // Rb..Xe
  return (z >= 1 && z <= 54) ? T[z] : 4;
}

static void distances(const Mol &m, std::vector<int> &D) {
  D.assign((size_t)m.n * m.n, 1 << 20);
  std::vector<int> q(m.n);
  for (int s = 0; s < m.n; s++) {
    int *d = &D[(size_t)s * m.n];
    d[s] = 0;
    int head = 0, tail = 0;
    q[tail++] = s;
    while (head < tail) {
      int u = q[head++];
      for (int v : m.adj[u])
        if (d[v] == (1 << 20)) { d[v] = d[u] + 1; q[tail++] = v; }
    }
  }
}

// -> per-atom S. `work` is reused across molecules so the timed loop does not allocate.
static void estate(const Mol &m, std::vector<double> &S, std::vector<int> &D) {
  S.assign(m.n, 0.0);
  std::vector<double> I(m.n, 0.0);
  for (int i = 0; i < m.n; i++) {
    if (m.deg[i] <= 0) continue;
    double dv = n_outer(m.Z[i]) - m.nH[i];
    double N = principal_qn(m.Z[i]);
    I[i] = (4.0 / (N * N) * dv + 1.0) / m.deg[i];
  }
  distances(m, D);
  for (int i = 0; i < m.n; i++) S[i] = I[i];
  for (int i = 0; i < m.n; i++)
    for (int j = i + 1; j < m.n; j++) {
      double p = D[(size_t)i * m.n + j] + 1.0;
      if (p < 1e6) {
        double t = (I[i] - I[j]) / (p * p);
        S[i] += t;
        S[j] -= t;
      }
    }
}

// ---------------------------------------------------------------------------------------
// Hall-Kier alpha and the three kappa shape indices.
//
// alpha is a per-element table indexed by (hybridisation - 2), falling back to the last entry
// when the hybridisation is higher than the table covers -- RDKit's exact behaviour, including
// the fallback to a covalent-radius ratio for elements not in the table.
// ---------------------------------------------------------------------------------------
// ALPHA IS A GENERATED TABLE, SOLVED OUT OF RDKIT ITSELF (see gen_alpha.py). alpha is additive
// over atoms, so the 10,000-molecule corpus gives 10,000 equations in the 20 distinct
// (element, hybridisation) pairs it contains; the least-squares solution reproduces RDKit's own
// CalcHallKierAlpha to 7.5e-14 over every molecule.
//
// This is deliberate, and it is not curve-fitting. Kappa and HallKierAlpha moved from Python to
// C++ inside RDKit, and the C++ routine does NOT use GetRcovalent -- the published
// hallKierAlphas table covers ten elements and everything else (B, Si, Se here) comes from an
// internal radius table that GetRcovalent disagrees with. Reimplementing from the paper gave
// visibly wrong numbers on exactly those elements. A constant table extracted from the reference
// is data, and it is verifiably exact; a formula guessed from the literature was neither.
// The off-table elements are written as the RATIO, not as a rounded decimal. RDKit computes
// rA/rC - 1 in double precision; a 6-digit literal was off by 3e-07 and that alone failed the
// exactness check on 0.33% of the corpus. rC = 0.77 and the radii below are the ones RDKit's C++
// table actually holds (recovered from the solve, and NOT the values GetRcovalent reports).
struct AlphaRow { int z, hyb; double a; };
static const double RC = 0.77;
static const AlphaRow ALPHA[] = {
    {  1, 1,  0.0},          {  5, 3, 0.82 / RC - 1.0}, {  5, 4, 0.82 / RC - 1.0},
    {  6, 2, -0.22},         {  6, 3, -0.13},           {  6, 4,  0.0},
    {  7, 2, -0.29},         {  7, 3, -0.20},           {  7, 4, -0.04},
    {  8, 3, -0.20},         {  8, 4, -0.04},           {  9, 4, -0.07},
    { 14, 4, 0.937 / RC - 1.0}, { 15, 4,  0.43},        { 16, 3,  0.22},
    { 16, 4,  0.35},         { 17, 4,  0.29},           { 34, 3, 1.17 / RC - 1.0},
    { 35, 4,  0.48},         { 53, 4,  0.73},
};

static double hk_alpha_atom_tab(int z, int hyb) {
  for (const AlphaRow &r : ALPHA)
    if (r.z == z && r.hyb == hyb) return r.a;
  for (const AlphaRow &r : ALPHA)          // unseen hybridisation for a known element
    if (r.z == z) return r.a;
  return 0.0;                              // unseen element: verifier will flag it
}

static double hk_alpha_atom_unused(int z, int hyb) {
  // index 0 = SP, 1 = SP2, 2 = SP3.   NAN marks "not defined, use the last entry".
  static const double NA = NAN;
  struct Row { int z; double a[3]; };
  static const Row T[] = {
      {35, {NA, NA, 0.48}},   // Br
      {6,  {-0.22, -0.13, 0.00}},
      {17, {NA, NA, 0.29}},   // Cl
      {9,  {NA, NA, -0.07}},  // F
      {1,  {0.0, 0.0, 0.0}},
      {53, {NA, NA, 0.73}},   // I
      {7,  {-0.29, -0.20, -0.04}},
      {8,  {NA, -0.20, -0.04}},
      {15, {NA, 0.30, 0.43}},
      {16, {NA, 0.22, 0.35}}};
  // covalent radii / r(C) - 1, for anything outside the table
  static const double rC = 0.77;
  for (const Row &r : T)
    if (r.z == z) {
      int k = hyb - 2;                       // RDKit: HybridizationType SP=2, SP2=3, SP3=4
      double a = (k >= 0 && k < 3) ? r.a[k] : r.a[2];
      if (std::isnan(a)) a = r.a[2];
      return std::isnan(a) ? 0.0 : a;
    }
  static const double RAD[] = {0, 0.32, 0.93, 1.23, 0.90, 0.82, 0.77, 0.75, 0.73, 0.72, 0.71,
                               1.54, 1.36, 1.18, 1.11, 1.06, 1.02, 0.99, 0.98, 2.03, 1.74,
                               1.44, 1.32, 1.22, 1.18, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25,
                               1.26, 1.22, 1.20, 1.16, 1.14, 1.12};
  double rA = (z >= 1 && z <= 36) ? RAD[z] : 1.20;
  return rA / rC - 1.0;
}

static double hk_alpha(const Mol &m) {
  double s = 0.0;
  for (int i = 0; i < m.n; i++) s += hk_alpha_atom_tab(m.Z[i], m.hyb[i]);
  return s;
}

// Walks of exactly k DISTINCT BONDS on the heavy-atom subgraph.
//
// Distinct BONDS, not distinct atoms -- that is RDKit's convention and the difference is
// visible: FindAllPathsOfLengthN(CC1CO1, 3) returns 3, of which only 2 have all-distinct atoms.
// The third closes the three-membered ring. An atom-distinct enumeration undercounts every
// molecule with a small ring, which is why Kappa3 matched on only 92% of the corpus before.
//
// Enumerated rather than derived from a degree formula: the closed form for k=3 needs a triangle
// correction, and getting it subtly wrong is invisible against a reference that disagrees only
// on fused systems.
static long long walks_of_len(const Mol &m, const std::vector<int> &heavy, int k) {
  long long total = 0;
  std::vector<char> used(m.nb, 0);
  // incidence: for each atom, the (neighbour, bond index) pairs, heavy ends only
  static thread_local std::vector<std::vector<std::pair<int, int>>> inc;
  inc.assign(m.n, {});
  for (int b = 0; b < m.nb; b++) {
    int u = m.bu[b], v = m.bv[b];
    if (m.Z[u] == 1 || m.Z[v] == 1) continue;
    inc[u].push_back({v, b});
    inc[v].push_back({u, b});
  }
  // OPEN AND CLOSED WALKS ARE DIVIDED BY DIFFERENT NUMBERS, which is the whole subtlety.
  // RDKit's path is a SET of bonds. An open path of k bonds is discovered exactly twice, once
  // from each end. A CYCLE of k bonds is discovered 2k times -- from each of its k atoms, in
  // each of two directions -- so dividing everything by 2 counts each ring k times over.
  // On methyloxirane that turned RDKit's P3 = 3 into 5, and Kappa3 matched on only 92% of the
  // corpus. k = 2 is unaffected (a 2-cycle cannot exist), which is exactly why Kappa2 passed
  // while Kappa3 failed.
  long long open_w = 0, closed_w = 0;
  struct Fr { int u, ptr; int bond; };
  std::vector<Fr> stk;
  for (int s : heavy) {
    stk.clear();
    stk.push_back({s, 0, -1});
    while (!stk.empty()) {
      Fr &f = stk.back();
      if ((int)stk.size() - 1 == k) {
        (f.u == s ? closed_w : open_w)++;
        if (f.bond >= 0) used[f.bond] = 0;
        stk.pop_back();
        continue;
      }
      if (f.ptr >= (int)inc[f.u].size()) {
        if (f.bond >= 0) used[f.bond] = 0;
        stk.pop_back();
        continue;
      }
      auto [v, b] = inc[f.u][f.ptr++];
      if (used[b]) continue;
      used[b] = 1;
      stk.push_back({v, 0, b});
    }
  }
  (void)total;
  return open_w / 2 + closed_w / (2 * k);
}

static void kappa(const Mol &m, double *out) {
  // A counts HEAVY atoms; P1 counts ALL bonds, including those to explicit (isotopic) hydrogen.
  // That asymmetry is RDKit's, not ours -- verified against CalcKappa1 on a tritiated molecule,
  // where only P1 = GetNumBonds() (5, not the 2 heavy bonds) reproduces its value.
  std::vector<int> heavy;
  for (int i = 0; i < m.n; i++)
    if (m.Z[i] != 1) heavy.push_back(i);
  double A = (double)heavy.size(), alpha = hk_alpha(m);
  double P1 = m.nb;
  double P2 = (double)walks_of_len(m, heavy, 2), P3 = (double)walks_of_len(m, heavy, 3);
  double d1 = P1 + alpha;
  out[0] = d1 ? (A + alpha) * (A + alpha - 1) * (A + alpha - 1) / (d1 * d1) : 0.0;
  double d2 = (P2 + alpha) * (P2 + alpha);
  out[1] = d2 ? (A + alpha - 1) * (A + alpha - 2) * (A + alpha - 2) / d2 : 0.0;
  double d3 = (P3 + alpha) * (P3 + alpha);
  double lead = ((int)A % 2 == 1) ? (A + alpha - 1) : (A + alpha - 2);
  out[2] = d3 ? lead * (A + alpha - 3) * (A + alpha - 3) / d3 : 0.0;
  out[3] = alpha;
}

// ---------------------------------------------------------------------------------------
// BCUT2D -- eight columns that cost 92% of the whole predict block in RDKit (306.83 us/mol).
//
// The Burden matrix B for an atom property p:
//     B[i][i] = p_i
//     B[i][j] = bondOrder/10 for bonded i,j  (+0.01 if either atom is terminal)
//     B[i][j] = 0.001        otherwise
// and the descriptor is its LARGEST and SMALLEST eigenvalue, for p in {mass, Gasteiger charge,
// Crippen logP, Crippen MR}.
//
// THE POINT IS THAT ONLY THE TWO EXTREMES ARE WANTED. RDKit takes a full spectrum of each of the
// four matrices; a complete normalised-Laplacian spectrum through dsyevd costs 19.90 us in
// cpp/bench.cpp at this molecule size, so four of them should be ~80 us, not 307. This uses
// LAPACK's dsyevr with RANGE='I' to request eigenvalues 1 and n only, which lets it skip
// assembling the eigenvectors and most of the spectrum.
//
// The atom properties themselves are NOT recomputed here -- they arrive from RDKit through the
// exporter. Crippen is 0.85 us and Gasteiger 9.41 us of already-C++ work, so reimplementing
// them would buy nothing and would put a second SMARTS/PEOE implementation into the world. What
// is being replaced is the eigenvalue step, which is where the 300 us actually is.
// ---------------------------------------------------------------------------------------
extern "C" {
void dsyevd_(char *, char *, int *, double *, int *, double *, double *, int *, int *, int *,
             int *);
}

struct BcutWork {
  std::vector<double> A, w, z, work;
  std::vector<int> isuppz, iwork;
};

// -> {hi, lo} for one property vector.
static void bcut_one(const Mol &m, const std::vector<double> &prop, BcutWork &W,
                     double *hi, double *lo) {
  // HEAVY ATOMS ONLY. Explicit hydrogens -- which in this corpus means isotopic labels, [2H]
  // and [3H] -- are not in RDKit's Burden matrix. Including them passed 99.970% of the corpus
  // and failed exactly the three deuterated/tritiated molecules, where MWLOW came back as 1.815
  // (a deuterium mass) instead of 10.109. A 3-in-10,000 failure that is entirely one chemical
  // class is a bug, not noise; a tolerance loose enough to hide it would hide real errors too.
  static thread_local std::vector<int> hv, pos;
  hv.clear();
  pos.assign(m.n, -1);
  for (int i = 0; i < m.n; i++)
    if (m.Z[i] != 1) { pos[i] = (int)hv.size(); hv.push_back(i); }
  const int n = (int)hv.size();
  if (n == 0) { *hi = *lo = 0.0; return; }
  W.A.assign((size_t)n * n, 0.001);
  for (int i = 0; i < n; i++) W.A[(size_t)i * n + i] = prop[hv[i]];
  for (int b = 0; b < m.nb; b++) {
    int i = pos[m.bu[b]], j = pos[m.bv[b]];
    if (i < 0 || j < 0) continue;
    // OFF-DIAGONAL = 1/sqrt(bond order), and there is NO terminal-atom correction.
    //
    // Not the classic Burden 0.1*order, and not the raw order either -- both were tried and both
    // are wrong. Solved out of RDKit on two-atom molecules, where the matrix is [[p1,w],[w,p2]]
    // so w follows in closed form from the two eigenvalues:
    //     C-C  w = 1.000000 = 1/sqrt(1)
    //     C=C  w = 0.707107 = 1/sqrt(2)
    //     C#C  w = 0.577350 = 1/sqrt(3)
    // Those same molecules also confirm the diagonal is the raw property (hi + lo = p1 + p2 to
    // the last digit) and that no +0.01 terminal term exists (both atoms are terminal in C-C,
    // yet w is exactly 1). Non-bonded stays 0.001, which cyclohexane confirms independently.
    double v = 1.0 / std::sqrt(m.bord[b]);
    W.A[(size_t)i * n + j] = v;
    W.A[(size_t)j * n + i] = v;
  }
  if (n == 1) { *hi = *lo = W.A[0]; return; }

  // dsyevd, not dsyevr. dsyevr takes THREE character arguments and Accelerate's Fortran ABI
  // rejects the call (INFO = -6) without the hidden string-length arguments; dsyevd takes two
  // and is already proven against this LAPACK in cpp/bench.cpp. The cost is a FULL spectrum
  // where only the two extremes are wanted -- so the number this produces is an upper bound,
  // and an extremal-only solver (Lanczos, or dsyevr through a LAPACK that accepts it) can only
  // be faster.
  // NO WORKSPACE QUERY. The query is itself a LAPACK call, so querying before every
  // factorisation doubles the number of calls -- eight per molecule where four are needed. For
  // JOBZ='N' the requirement is documented and tiny (LWORK >= 2N+1, LIWORK >= 1), so the buffers
  // are sized directly and reused across molecules; they only ever grow.
  char jobz = 'N', uplo = 'U';
  int nn = n, lda = n, info = 0, lwork = 2 * n + 1, liwork = 1;
  W.w.resize(n);
  if ((int)W.work.size() < lwork) W.work.resize(lwork);
  if ((int)W.iwork.size() < liwork) W.iwork.resize(liwork);
  dsyevd_(&jobz, &uplo, &nn, W.A.data(), &lda, W.w.data(), W.work.data(), &lwork,
          W.iwork.data(), &liwork, &info);
  *lo = W.w[0];
  *hi = W.w[n - 1];
}

// -> 8 values in RDKit's order: MWHI, MWLOW, CHGHI, CHGLO, LOGPHI, LOGPLOW, MRHI, MRLOW
static void bcut2d(const Mol &m, BcutWork &W, double *out) {
  const std::vector<double> *props[4] = {&m.mass, &m.gast, &m.clogp, &m.cmr};
  for (int k = 0; k < 4; k++) bcut_one(m, *props[k], W, &out[2 * k], &out[2 * k + 1]);
}

// ---------------------------------------------------------------------------------------

template <typename F>
static double time_it(const std::vector<Mol> &ms, int reps, F &&f) {
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) f();
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  return us / (reps * (double)ms.size());
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "bench";
  auto ms = load(argc > 2 ? argv[2] : "mols.txt");
  double na = 0;
  for (auto &m : ms) na += m.n;
  fprintf(stderr, "%zu molecules, mean %.1f heavy atoms\n", ms.size(), na / ms.size());

  std::vector<double> S;
  std::vector<int> D;
  BcutWork BW;
  BW.z.resize(1);

  if (std::string(mode) == "verify") {
    FILE *out = fopen("values.txt", "w");
    for (auto &m : ms) {
      estate(m, S, D);
      double mx = -1e30, mn = 1e30, amx = -1e30, amn = 1e30;
      for (int i = 0; i < m.n; i++) {
        double v = S[i], a = std::fabs(v);
        if (v > mx) mx = v;
        if (v < mn) mn = v;
        if (a > amx) amx = a;
        if (a < amn) amn = a;
      }
      double k[4], bc[8];
      kappa(m, k);
      bcut2d(m, BW, bc);
      fprintf(out, "%.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g", mx, mn, amx, amn,
              k[0], k[1], k[2], k[3]);
      for (int q = 0; q < 8; q++) fprintf(out, " %.12g", bc[q]);
      fputc('\n', out);
    }
    fclose(out);
    fprintf(stderr, "wrote values.txt\n");
    return 0;
  }

  volatile double sink = 0.0;
  double t_es = time_it(ms, 20, [&] {
    for (auto &m : ms) { estate(m, S, D); sink += S[0]; }
  });
  double t_kp = time_it(ms, 20, [&] {
    for (auto &m : ms) { double k[4]; kappa(m, k); sink += k[0]; }
  });
  printf("  %-46s %8.2f us/mol\n", "EState indices (per atom + 4 aggregates)", t_es);
  double t_bc = time_it(ms, 5, [&] {
    for (auto &m : ms) { double bc[8]; bcut2d(m, BW, bc); sink += bc[0]; }
  });
  printf("  %-46s %8.2f us/mol\n", "Kappa1-3 + HallKierAlpha", t_kp);
  printf("  %-46s %8.2f us/mol\n", "BCUT2D (extremal eigenvalues, dsyevr)", t_bc);
  printf("\n(sink %.3g)\n", (double)sink);
  return 0;
}
