// pybind11 surface for the verified blocks. Arrays in, one 2-D array out.
//
// This file is the ONLY thing between `hume.featurize_blocks` and the arithmetic that was
// verified ALL EXACT on 98,905 molecules. It contains no descriptor code of its own: it fills
// the same `Mol` struct that cpp/hume.cpp's text loader fills, and calls the same blocks_row().
// If a value ever disagrees with cpp/values_hume.txt, the disagreement is here or in
// src/hume/_extract.py, and nowhere else -- which is the property milestone 1 was chosen for.
//
// BATCHED, ONE CALL PER N MOLECULES. Crossing the boundary was measured at 0.106 us, so per
// molecule would be affordable; per-molecule *Python attribute access on RDKit objects* is not,
// and batching is what amortises the extraction, not the call.
//
// THE ONE PIECE OF CHEMISTRY THAT HAPPENS HERE is Crippen atom typing, and it is here rather
// than in hume_blocks.h on purpose: hume_blocks.h is the file cpp/hume.cpp shares, and the text
// path feeds it Crippen contributions RDKit already computed. This path does not call RDKit for
// them at all -- rdMolDescriptors._CalcCrippenContribs cost 92 us/mol COLD, the largest single
// item in the pipeline, and this answers the same question in 1.4. So `atom_d` arrives
// with two columns instead of four and the logP/MR pair is filled in below, from the same
// integers the rest of the row already needs. See src/hume_core/crippen_typer.h.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "autocorr.h"
#include "crippen_typer.h"
#include "estate_typer.h"
#include "hume_blocks.h"
#include "infocontent.h"
#include "molpickle.h"
#include "pathcount.h"
#include "ringcount.h"
#include "topocharge.h"
#include "vsa_bins.h"

namespace py = pybind11;

using ArrI = py::array_t<int, py::array::c_style | py::array::forcecast>;
using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void need(bool cond, const char *what) {
  if (!cond) throw std::invalid_argument(std::string("hume._core: ") + what);
}

// Column counts in the flat per-atom / per-bond blocks. Mirrored in src/hume/_extract.py; the
// checks below turn a mismatch into an exception at the boundary instead of into silently
// transposed descriptors.
static constexpr int N_ATOM_INT = 9;   // Z, deg, nH, fchg, hyb, arom, ring, cip, nring
static constexpr int N_ATOM_DBL = 2;   // mass, gasteiger      (Crippen is computed here)
static constexpr int N_BOND_INT = 5;   // u, v, conjugated, in-ring, SMARTS bond code

// Column offsets inside a row of atom_i / bond_i, so the two loops below and the Crippen fill
// cannot drift apart on what column 4 means.
// A_NRING is the RING COUNT, A_RING the boolean. Both are carried because SMARTS asks both
// questions independently: `[R]` is membership, `[R1]`/`[R2]` are counts, and the count cannot
// be recovered from the boolean. It comes across the boundary from the single ring perception
// RDKit has already done rather than being recomputed C++-side -- ring perception is
// numbering-dependent for 24 molecules in the 100k corpus, so a second perception is a second
// chance to disagree.
enum { A_Z = 0, A_DEG = 1, A_NH = 2, A_FCHG = 3, A_HYB = 4, A_AROM = 5, A_RING = 6, A_CIP = 7,
       A_NRING = 8 };
enum { B_U = 0, B_V = 1, B_CONJ = 2, B_RING = 3, B_CODE = 4 };

// --------------------------------------------------------------------------------------------
// Crippen: the same five quantities cpp/export_crippen.py writes, read out of the arrays
// _extract.py already sends.
//
//   SMARTS X == getTotalDegree() == degree + GetTotalNumHs(False)      -> tx
//   SMARTS H == GetTotalNumHs(False) + neighbouring H ATOMS            -> sh
//
// The H-neighbour term is the second loop: it needs the adjacency, so it cannot be folded into
// the first. On a heavy-atom graph from SMILES it adds nothing, but [2H]C([2H])([2H])O is the
// molecule where H and X part company and the typer was verified on it.
// --------------------------------------------------------------------------------------------
static void crippen_fill(criptyper::Mol &c, std::vector<int32_t> &cur, const int *AI,
                         const int *BI, int n, int nb, int a0, int b0, double *logp, double *mr) {
  c.alloc(n, 2 * nb);
  for (int i = 0; i < n; i++) {
    const int *r = AI + (ssize_t)(a0 + i) * N_ATOM_INT;
    const int z = r[A_Z];
    c.z[i] = (uint8_t)(z > 255 ? 255 : z);
    c.arom[i] = (uint8_t)r[A_AROM];
    c.chg[i] = (int8_t)r[A_FCHG];
    c.tx[i] = (uint8_t)(r[A_DEG] + r[A_NH]);
    c.sh[i] = (uint8_t)r[A_NH];
  }
  // CSR adjacency, counting sort. Bond order within an atom's list does not matter to any
  // predicate (they all quantify existentially over neighbours), unlike hume_blocks.h's `adj`.
  for (int b = 0; b < nb; b++) {
    const int *r = BI + (ssize_t)(b0 + b) * N_BOND_INT;
    c.start[r[B_U] + 1]++;
    c.start[r[B_V] + 1]++;
  }
  for (int i = 0; i < n; i++) c.start[i + 1] += c.start[i];
  cur.assign(c.start.begin(), c.start.end() - 1);   // caller-owned scratch; not a per-mol malloc
  for (int b = 0; b < nb; b++) {
    const int *r = BI + (ssize_t)(b0 + b) * N_BOND_INT;
    const uint8_t code = (uint8_t)r[B_CODE];
    c.nbr[cur[r[B_U]]] = r[B_V];
    c.bcode[cur[r[B_U]]++] = code;
    c.nbr[cur[r[B_V]]] = r[B_U];
    c.bcode[cur[r[B_V]]++] = code;
  }
  for (int i = 0; i < n; i++) {
    int extra = 0;
    for (int e = c.start[i]; e < c.start[i + 1]; e++)
      if (c.z[c.nbr[e]] == 1) extra++;
    c.sh[i] = (uint8_t)(c.sh[i] + extra);
  }
  criptyper::contribs(c, logp, mr);
}

// One molecule's hume_blocks::Mol, filled out of the flat arrays. Lifted out of blocks()
// unchanged so the pickle path and the all-families path below call the SAME code rather than
// growing two more copies of it. Nothing here touches a Python object; the caller holds the
// buffers alive and has released the GIL.
static void fill_hume_mol(Mol &m, criptyper::Mol &c, std::vector<int32_t> &cur, const int *AI,
                          const double *AD, const int *BI, const int *BS, const double *BD,
                          int a0, int b0, int n, int nb, int chg_ok) {
    m.n = n;
    m.nb = nb;
    m.chg_ok = chg_ok;

    m.Z.resize(m.n); m.deg.resize(m.n); m.nH.resize(m.n); m.fchg.resize(m.n);
    m.hyb.resize(m.n); m.arom.resize(m.n); m.ring.resize(m.n); m.cip.resize(m.n);
    m.mass.resize(m.n); m.gast.resize(m.n); m.clogp.resize(m.n); m.cmr.resize(m.n);
    for (int i = 0; i < m.n; i++) {
      const int *r = AI + (ssize_t)(a0 + i) * N_ATOM_INT;
      m.Z[i] = r[0]; m.deg[i] = r[1]; m.nH[i] = r[2]; m.fchg[i] = r[3];
      m.hyb[i] = r[4]; m.arom[i] = r[5]; m.ring[i] = r[6]; m.cip[i] = r[7];
      const double *d = AD + (ssize_t)(a0 + i) * N_ATOM_DBL;
      m.mass[i] = d[0]; m.gast[i] = d[1];
    }
    crippen_fill(c, cur, AI, BI, m.n, m.nb, a0, b0, m.clogp.data(), m.cmr.data());

    // Built in exactly the order cpp/hume.cpp's load() builds them: adjacency first over all
    // bonds, then the (neighbour, bond index) incidence list. Neighbour ORDER is part of the
    // answer -- chi's DFS and the cycle enumeration walk these lists -- so this loop must not
    // be fused or reordered.
    m.bu.resize(m.nb); m.bv.resize(m.nb); m.bord.resize(m.nb);
    m.bconj.resize(m.nb); m.bring.resize(m.nb); m.bstereo.resize(m.nb);
    m.adj.assign(m.n, {});
    for (int b = 0; b < m.nb; b++) {
      const int *r = BI + (ssize_t)(b0 + b) * N_BOND_INT;
      m.bu[b] = r[0]; m.bv[b] = r[1]; m.bconj[b] = r[2]; m.bring[b] = r[3];
      m.bstereo[b] = BS[b0 + b];
      m.bord[b] = BD[b0 + b];
      m.adj[m.bu[b]].push_back(m.bv[b]);
      m.adj[m.bv[b]].push_back(m.bu[b]);
    }
    m.inc.assign(m.n, {});
    for (int b = 0; b < m.nb; b++) {
      m.inc[m.bu[b]].push_back({m.bv[b], b});
      m.inc[m.bv[b]].push_back({m.bu[b], b});
    }
}

static void run_blocks(ssize_t nm, const int *AO, const int *BO, const int *OKf, const int *AI,
                       const double *AD, const int *BI, const int *BS, const double *BD,
                       double *O) {
  BlockWork W;
  Mol m;
  criptyper::Mol c;
  std::vector<int32_t> cur;
  for (ssize_t k = 0; k < nm; k++) {
    fill_hume_mol(m, c, cur, AI, AD, BI, BS, BD, AO[k], BO[k], AO[k + 1] - AO[k],
                  BO[k + 1] - BO[k], OKf[k]);
    blocks_row(m, W, O + (ssize_t)k * HUME_NBLOCK_COLS);
  }
}

static py::array_t<double> blocks(ArrI atom_off, ArrI bond_off, ArrI chg_ok, ArrI atom_i,
                                  ArrD atom_d, ArrI bond_i, ArrI bond_s, ArrD bond_d) {
  need(atom_off.ndim() == 1 && bond_off.ndim() == 1 && chg_ok.ndim() == 1, "offsets must be 1-D");
  const ssize_t nm = chg_ok.shape(0);
  need(atom_off.shape(0) == nm + 1 && bond_off.shape(0) == nm + 1,
       "offset arrays must have n_mol + 1 entries");
  need(atom_i.ndim() == 2 && atom_i.shape(1) == N_ATOM_INT, "atom_i must be (n_atoms, 8)");
  need(atom_d.ndim() == 2 && atom_d.shape(1) == N_ATOM_DBL, "atom_d must be (n_atoms, 2)");
  need(bond_i.ndim() == 2 && bond_i.shape(1) == N_BOND_INT, "bond_i must be (n_bonds, 5)");
  need(atom_i.shape(0) == atom_d.shape(0), "atom_i and atom_d disagree on n_atoms");
  need(bond_i.shape(0) == bond_d.shape(0) && bond_i.shape(0) == bond_s.shape(0),
       "bond arrays disagree on n_bonds");

  const int *AO = atom_off.data(), *BO = bond_off.data(), *OKf = chg_ok.data();
  const int *AI = atom_i.data(), *BI = bond_i.data(), *BS = bond_s.data();
  const double *AD = atom_d.data(), *BD = bond_d.data();
  const ssize_t n_atoms = atom_i.shape(0), n_bonds = bond_i.shape(0);
  need(nm == 0 || (AO[nm] == n_atoms && BO[nm] == n_bonds),
       "last offset does not match the flat array length");

  auto out = py::array_t<double>({(ssize_t)nm, (ssize_t)HUME_NBLOCK_COLS});
  double *O = out.mutable_data();

  // The GIL goes back only after the loop. Nothing below touches a Python object: the input
  // buffers are borrowed and kept alive by the argument objects, and `out` is written through a
  // raw pointer obtained above.
  {
    py::gil_scoped_release nogil;
    run_blocks(nm, AO, BO, OKf, AI, AD, BI, BS, BD, O);
  }
  return out;
}

// --------------------------------------------------------------------------------------------
// THE PICKLE PATH. Same arithmetic, same run_blocks(), different way of getting the numbers in.
//
// `extract()` builds the arrays above by asking an RDKit molecule ~300 questions from Python.
// This asks it ONE: `m.ToBinary(...)`. The blob is parsed by src/hume_core/molpickle.h, which
// fills byte-for-byte the same arrays -- verified as such on both corpora by
// cpp/verify_molpickle.py, which is the only reason this path is allowed to exist.
//
// The existing path is NOT going anywhere: it reads the molecule through RDKit's supported
// Python API and is therefore the oracle, the fallback for molecules this reader refuses
// (queries, substance groups), and the thing that still works the day RDKit moves its format.
// --------------------------------------------------------------------------------------------

//! The flat arrays for a batch, owned by std::vector so the fast path allocates no numpy.
struct Flat {
  std::vector<int> atom_off, bond_off, chg_ok, atom_i, bond_i, bond_s;
  std::vector<double> atom_d, bond_d;
  // Only filled when the caller asks for it -- the Autocorrelation `c` weight, which is a
  // different quantity from atom_d's charge column. See molpickle::Sink::ac_charge.
  std::vector<double> ac_charge;
};

//! Borrowed views of the caller's bytes objects. Gathered under the GIL, used without it; the
//! Python list keeps every buffer alive for the duration of the call.
struct Blobs {
  std::vector<const std::uint8_t *> ptr;
  std::vector<std::size_t> len;
};

static Blobs borrow(const py::sequence &pickles) {
  Blobs b;
  const ssize_t nm = py::len(pickles);
  b.ptr.resize(nm);
  b.len.resize(nm);
  for (ssize_t k = 0; k < nm; k++) {
    py::object o = pickles[k];
    char *data = nullptr;
    ssize_t n = 0;
    if (PyBytes_AsStringAndSize(o.ptr(), &data, &n) != 0) {
      throw py::error_already_set();
    }
    b.ptr[k] = (const std::uint8_t *)data;
    b.len[k] = (std::size_t)n;
  }
  return b;
}

//! Two passes: peek every header for its atom/bond counts to build the offsets, then parse.
//! The peek is 28 bytes per molecule and is what lets the flat arrays be allocated exactly once.
static void fill_from_pickles(const Blobs &b, Flat &f, bool want_ac_charge = false) {
  const ssize_t nm = (ssize_t)b.ptr.size();
  f.atom_off.resize(nm + 1);
  f.bond_off.resize(nm + 1);
  f.chg_ok.resize(nm);
  f.atom_off[0] = f.bond_off[0] = 0;
  for (ssize_t k = 0; k < nm; k++) {
    int na = 0, nb = 0;
    molpickle::peek_sizes(b.ptr[k], b.len[k], na, nb);
    f.atom_off[k + 1] = f.atom_off[k] + na;
    f.bond_off[k + 1] = f.bond_off[k] + nb;
  }
  const int n_atoms = f.atom_off[nm], n_bonds = f.bond_off[nm];
  f.atom_i.resize((std::size_t)n_atoms * N_ATOM_INT);
  f.atom_d.resize((std::size_t)n_atoms * N_ATOM_DBL);
  f.bond_i.resize((std::size_t)n_bonds * N_BOND_INT);
  f.bond_s.resize(n_bonds);
  f.bond_d.resize(n_bonds);
  if (want_ac_charge) f.ac_charge.resize(n_atoms);

  molpickle::Work w;
  for (ssize_t k = 0; k < nm; k++) {
    const int a0 = f.atom_off[k], b0 = f.bond_off[k];
    molpickle::Sink s{f.atom_i.data() + (std::size_t)a0 * N_ATOM_INT,
                      f.atom_d.data() + (std::size_t)a0 * N_ATOM_DBL,
                      f.bond_i.data() + (std::size_t)b0 * N_BOND_INT,
                      f.bond_s.data() + b0, f.bond_d.data() + b0, nullptr, nullptr,
                      want_ac_charge ? f.ac_charge.data() + a0 : nullptr};
    f.chg_ok[k] = molpickle::parse(b.ptr[k], b.len[k], f.atom_off[k + 1] - a0,
                                   f.bond_off[k + 1] - b0, s, w);
  }
}

static py::array_t<double> blocks_from_pickles(py::sequence pickles) {
  Blobs b = borrow(pickles);
  const ssize_t nm = (ssize_t)b.ptr.size();
  auto out = py::array_t<double>({(ssize_t)nm, (ssize_t)HUME_NBLOCK_COLS});
  double *O = out.mutable_data();
  {
    py::gil_scoped_release nogil;
    Flat f;
    fill_from_pickles(b, f);
    run_blocks(nm, f.atom_off.data(), f.bond_off.data(), f.chg_ok.data(), f.atom_i.data(),
               f.atom_d.data(), f.bond_i.data(), f.bond_s.data(), f.bond_d.data(), O);
  }
  return out;
}

//! The parsed boundary as arrays, for cpp/verify_molpickle.py to hold against extract(). The
//! fast path never builds these -- a paired comparison needs them, a descriptor does not.
static py::tuple pickle_extract(py::sequence pickles) {
  Blobs b = borrow(pickles);
  Flat f;
  {
    py::gil_scoped_release nogil;
    fill_from_pickles(b, f);
  }
  const ssize_t nm = (ssize_t)b.ptr.size();
  const ssize_t na = f.atom_off[nm], nb = f.bond_off[nm];
  auto vec_i = [](const std::vector<int> &v, ssize_t rows, ssize_t cols) {
    auto a = cols == 1 ? py::array_t<int>({rows}) : py::array_t<int>({rows, cols});
    std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(int));
    return a;
  };
  auto vec_d = [](const std::vector<double> &v, ssize_t rows, ssize_t cols) {
    auto a = cols == 1 ? py::array_t<double>({rows}) : py::array_t<double>({rows, cols});
    std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
  };
  return py::make_tuple(vec_i(f.atom_off, nm + 1, 1), vec_i(f.bond_off, nm + 1, 1),
                        vec_i(f.chg_ok, nm, 1), vec_i(f.atom_i, na, N_ATOM_INT),
                        vec_d(f.atom_d, na, N_ATOM_DBL), vec_i(f.bond_i, nb, N_BOND_INT),
                        vec_i(f.bond_s, nb, 1), vec_d(f.bond_d, nb, 1));
}

//! The import-time drift guard. src/hume/_extract.py hands this one probe pickle at import, so
//! an RDKit whose MolPickler version differs from the pinned one is an ImportError naming both,
//! rather than 182 quietly wrong columns.
static void pickle_check(py::buffer probe) {
  py::buffer_info info = probe.request();
  molpickle::check_version((const std::uint8_t *)info.ptr, (std::size_t)info.size * info.itemsize);
}

// ============================================================================================
// EVERY FAMILY THAT HAS C++, IN ONE PASS
//
// Six verified headers now exist and each was being reached only by its own standalone harness
// in cpp/. This is the single entry point that runs all of them over one parse of one pickle,
// so a caller pays the boundary once rather than six times and the families cannot disagree
// about which molecule they were given.
//
// WHY ONE PASS IS NOT MERELY TIDIER. Two of the families need something another has already
// computed, and recomputing it would be both slower and a second chance to differ:
//
//   * the 21 `S*` EState columns are the typer's hits weighted by the per-atom E-state INDEX,
//     which hume_blocks.h's blocks_row() has just written into BlockWork::ES. cpp/verify_estate.py
//     names that function as the index source, so taking W.ES is taking the verified one.
//   * RingCount wants the ring SET, and gets it as a boundary array from src/hume/_rings.py --
//     NOT from the pickle, even though molpickle.h can parse the pickle's ring section and the
//     `Sink::ring_at` hooks for doing so are still there. The blob carries RDKit's RAW
//     `GetSymmSSSR`, which is not a function of the graph: 32 molecules in 100,000 get a
//     different ring set depending on the order they are presented in, and `rings_for()`
//     repairs exactly those. Reading rings from the blob here would leave the two boundaries
//     sourcing them from genuinely different places, agreeing by argument rather than by
//     construction -- and a divergence there is 49 quietly wrong columns, not a loud failure.
//
// WHAT IS NOT HERE, and it is the largest single item: Autocorrelation, 419 of the 865 columns.
// It is verified, but it lives in cpp/ac.cpp as a standalone `main()` over a text file, and it
// descriptors the HYDROGEN-ADDED graph with charges summed as `_GasteigerCharge +
// _GasteigerHCharge`. Wiring it is a header refactor plus an H-added graph build, not a call --
// a separate piece of work, and this file says so rather than letting a column count imply
// otherwise.
// ============================================================================================

// The per-atom EState index that the S* columns weight by. Nothing to compute here: blocks_row()
// fills BlockWork::ES on every molecule, unconditionally, as the comment above its call in
// hume_blocks.h explains. This is a load-bearing dependency on that fact, so it is asserted.
static constexpr int N_ESTATE_TYPES = estate_tbl::N_ROWS;   // 79 patterns -> 79 N + 79 S columns

enum {
  OFF_BLOCKS = 0,
  OFF_VSA    = OFF_BLOCKS + HUME_NBLOCK_COLS,
  OFF_ESTATE = OFF_VSA + vsabin::N_COLS,
  OFF_RING   = OFF_ESTATE + 2 * N_ESTATE_TYPES,
  OFF_PATH   = OFF_RING + ringcount::N_COLS,
  OFF_TOPO   = OFF_PATH + pathcount::N_COLS,
  OFF_IC     = OFF_TOPO + topocharge::N_COLS,
  // infoic emits 45; the last three (Ipc, AvgIpc, Log2Ipc) are NOT wired. They are
  // numbering-dependent on 2.8% of the corpus -- an open, diagnosed bug recorded at the top of
  // src/hume_core/infocontent.h -- and an ill-posed column is worse than a missing one because
  // it looks like a value. The 42 IC/TIC/SIC/BIC/CIC/MIC/ZMIC columns below it are bit-identical
  // under renumbering and are wired.
  OFF_AC     = OFF_IC + infoic::N_IC,
  N_ALL_COLS = OFF_AC + autocorr::N_COLS,
};

//! Scratch for every family, allocated once per batch rather than once per molecule.
struct AllWork {
  BlockWork bw;
  Mol m;
  criptyper::Mol c;
  std::vector<int32_t> cur;
  vsabin::Mol vm;
  vsabin::Work vw;
  esttyper::Mol em;
  std::vector<int32_t> ecount;
  std::vector<double> esum;
  ringcount::Mol rm;
  ringcount::Scratch rs;
  pathcount::Mol pm;
  pathcount::Scratch ps;
  topocharge::Mol tm;
  topocharge::Scratch ts;
  infoic::Mol im;
  infoic::Row irow;
  autocorr::Mol am;
  autocorr::Work aw;
  AllWork() : ecount(N_ESTATE_TYPES), esum(N_ESTATE_TYPES) {}
};

// A family selector, so a caller can time one family or compute a subset. Not an optimisation
// hook: `all_from_pickles` defaults to everything and the whole point of the entry point is one
// pass. It exists because bench_e2e.py cannot otherwise say WHICH family the C++ time is in,
// and "the compute is 623 us" is not an actionable number.
//
// F_BLOCKS is not optional. The 182 blocks are what fill BlockWork::ES, and the EState `S*`
// columns weight by it -- asking for EState without BLOCKS would silently weight by a stale
// molecule's index, so the flag is forced on rather than left as a trap.
enum : unsigned {
  F_BLOCKS = 1u, F_VSA = 2u, F_ESTATE = 4u, F_RING = 8u, F_PATH = 16u, F_TOPO = 32u, F_IC = 64u,
  F_AC = 128u,
  F_ALL = 255u,
};

static void all_row(AllWork &W, const int *AI, const double *AD, const int *BI, const int *BS,
                    const double *BD, int a0, int b0, int n, int nb, int chg_ok,
                    const int *ring_ptr, const int *ring_at, int n_rings,
                    const int *HAI, const int *HBI, const double *HAC, int ha0, int hb0,
                    int hn, int hnb, unsigned fams, double *out) {
  const int *ai = AI + (ssize_t)a0 * N_ATOM_INT;
  const int *bi = BI + (ssize_t)b0 * N_BOND_INT;
  const double *bd = BD + b0;
  for (int c = 0; c < N_ALL_COLS; c++) out[c] = 0.0;

  // ---- the 182 blocks, and the EState index the S* columns need ----
  fill_hume_mol(W.m, W.c, W.cur, AI, AD, BI, BS, BD, a0, b0, n, nb, chg_ok);
  blocks_row(W.m, W.bw, out + OFF_BLOCKS);
  if ((int)W.bw.ES.size() != n)
    throw std::runtime_error("hume._core: blocks_row did not fill BlockWork::ES; the EState "
                             "sum columns have lost their index source");

  // ---- VSA binning: 66 columns ----
  if (fams & F_VSA) {
  W.vm.alloc(n, nb);
  for (int i = 0; i < n; i++) {
    const int *r = ai + (ssize_t)i * N_ATOM_INT;
    W.vm.z[i] = r[A_Z]; W.vm.deg[i] = r[A_DEG]; W.vm.nH[i] = r[A_NH];
    W.vm.fchg[i] = r[A_FCHG]; W.vm.arom[i] = r[A_AROM];
    W.vm.gast[i] = AD[(ssize_t)(a0 + i) * N_ATOM_DBL + 1];
  }
  for (int b = 0; b < nb; b++) {
    const int *r = bi + (ssize_t)b * N_BOND_INT;
    W.vm.bu[b] = r[B_U]; W.vm.bv[b] = r[B_V]; W.vm.bcode[b] = r[B_CODE];
  }
  vsabin::vsa_row(W.vm, W.vw, out + OFF_VSA);
  }

  // ---- EState typer: 79 counts + 79 sums ----
  if (fams & F_ESTATE) {
  // btype is the bond TYPE alone, not the boundary's mask: esttyper::btypeFromBcode() collapses
  // "SINGLE carrying the aromatic flag" to SINGLE, which is the question RDKit's own EState
  // SMARTS ask. Doing it here rather than in the caller is why that helper exists.
  W.em.alloc(n, 2 * nb);
  for (int i = 0; i < n; i++) {
    const int *r = ai + (ssize_t)i * N_ATOM_INT;
    W.em.z[i] = (uint8_t)(r[A_Z] > 255 ? 255 : r[A_Z]);
    W.em.nH[i] = (uint8_t)r[A_NH];
    W.em.arom[i] = (uint8_t)r[A_AROM];
  }
  for (int b = 0; b < nb; b++) {
    const int *r = bi + (ssize_t)b * N_BOND_INT;
    W.em.start[r[B_U] + 1]++;
    W.em.start[r[B_V] + 1]++;
  }
  for (int i = 0; i < n; i++) W.em.start[i + 1] += W.em.start[i];
  W.cur.assign(W.em.start.begin(), W.em.start.end() - 1);
  for (int b = 0; b < nb; b++) {
    const int *r = bi + (ssize_t)b * N_BOND_INT;
    const uint8_t bt = esttyper::btypeFromBcode((uint8_t)r[B_CODE]);
    W.em.nbr[W.cur[r[B_U]]] = r[B_V]; W.em.btype[W.cur[r[B_U]]++] = bt;
    W.em.nbr[W.cur[r[B_V]]] = r[B_U]; W.em.btype[W.cur[r[B_V]]++] = bt;
  }
  W.em.finish();
  esttyper::aggregate(W.em, W.bw.ES.data(), W.ecount.data(), W.esum.data());
  for (int t = 0; t < N_ESTATE_TYPES; t++) {
    out[OFF_ESTATE + t] = (double)W.ecount[t];
    out[OFF_ESTATE + N_ESTATE_TYPES + t] = W.esum[t];
  }
  }

  // ---- RingCount: 49 columns, off the pickle's own SSSR ----
  if (fams & F_RING) {
  W.rm.alloc(n);
  for (int i = 0; i < n; i++) {
    const int *r = ai + (ssize_t)i * N_ATOM_INT;
    W.rm.z[i] = r[A_Z];
    W.rm.arom[i] = (uint8_t)r[A_AROM];
  }
  // ring_ptr is the molecule's slice of the batch CSR; ring_at is indexed by it directly, so
  // the atom indices arriving here are already LOCAL to this molecule (see _extract.Rings).
  for (int q = 0; q < n_rings; q++)
    W.rm.add_ring(ring_at + ring_ptr[q], ring_ptr[q + 1] - ring_ptr[q]);
  ringcount::compute(W.rm, out + OFF_RING, W.rs);
  }

  // ---- PathCount 11 and TopologicalCharge 21, both straight off the strided rows ----
  if (fams & F_PATH) {
    pathcount::build_from_rows(W.pm, n, nb, bi, N_BOND_INT, B_U, B_V, bd, ai, N_ATOM_INT, A_Z);
    pathcount::compute(W.pm, out + OFF_PATH, W.ps);
  }
  if (fams & F_TOPO) {
    topocharge::build(W.tm, n, nb, bi, N_BOND_INT, B_U, B_V);
    topocharge::compute(W.tm, out + OFF_TOPO, W.ts);
  }

  // ---- InformationContent: 42 columns (Ipc / AvgIpc / Log2Ipc deliberately not wired) ----
  if (fams & F_IC) {
  W.im.alloc(n, nb);
  for (int i = 0; i < n; i++) {
    const int *r = ai + (ssize_t)i * N_ATOM_INT;
    W.im.z[i] = (uint8_t)(r[A_Z] > 255 ? 255 : r[A_Z]);
    W.im.nh[i] = (uint8_t)r[A_NH];
    W.im.arom[i] = (uint8_t)r[A_AROM];
    W.im.chg[i] = (int8_t)r[A_FCHG];
  }
  for (int b = 0; b < nb; b++) {
    const int *r = bi + (ssize_t)b * N_BOND_INT;
    W.im.bu[b] = r[B_U]; W.im.bv[b] = r[B_V];
    W.im.bcode[b] = (uint8_t)r[B_CODE];
    W.im.bord[b] = bd[b];
  }
  infoic::compute(W.im, W.irow);
  for (int c = 0; c < infoic::N_IC; c++) out[OFF_IC + c] = W.irow.v[c];
  }

  // ---- Autocorrelation: 486 columns, on the HYDROGEN-ADDED graph ----
  // Every field here comes from the second blob, parsed by the same molpickle.h reader into the
  // same boundary layout -- `nh` is `GetTotalNumHs()` AFTER AddHs (normally 0, but `dv` adds it
  // to the explicit-H neighbour count and mordred reads it, so it is carried rather than
  // assumed), and `HAC` is mordred's `c` getter with its conditional already applied.
  if (fams & F_AC) {
    const int *hai = HAI + (std::size_t)ha0 * N_ATOM_INT;
    const int *hbi = HBI + (std::size_t)hb0 * N_BOND_INT;
    W.am.n = hn;
    W.am.nb = hnb;
    W.am.at.resize(hn);
    for (int i = 0; i < hn; i++) {
      const int *r = hai + (std::size_t)i * N_ATOM_INT;
      AtomRec &a = W.am.at[i];
      a.z = r[A_Z];
      a.fc = r[A_FCHG];
      a.nh = r[A_NH];
      a.c = HAC[ha0 + i];
    }
    W.am.bu.resize(hnb);
    W.am.bv.resize(hnb);
    W.am.adj.assign(hn, {});
    for (int b = 0; b < hnb; b++) {
      const int *r = hbi + (std::size_t)b * N_BOND_INT;
      W.am.bu[b] = r[B_U];
      W.am.bv[b] = r[B_V];
      W.am.adj[r[B_U]].push_back(r[B_V]);
      W.am.adj[r[B_V]].push_back(r[B_U]);
    }
    autocorr::row(W.am, W.aw, out + OFF_AC);
  }
}

static unsigned family_mask(const py::object &families) {
  if (families.is_none()) return F_ALL;
  static const std::pair<const char *, unsigned> NAMED[] = {
      {"blocks", F_BLOCKS}, {"vsa", F_VSA}, {"estate", F_ESTATE}, {"ringcount", F_RING},
      {"pathcount", F_PATH}, {"topocharge", F_TOPO}, {"infocontent", F_IC},
      {"autocorr", F_AC}};
  unsigned mask = F_BLOCKS;   // never optional; see the note on the enum
  for (auto h : families) {
    const std::string want = py::cast<std::string>(h);
    unsigned bit = 0;
    for (const auto &kv : NAMED)
      if (want == kv.first) bit = kv.second;
    if (!bit) throw std::invalid_argument("hume._core: unknown family '" + want + "'");
    mask |= bit;
  }
  return mask;
}

static py::array_t<double> all_from_pickles(py::sequence pickles, ArrI ring_moff, ArrI ring_ptr,
                                            ArrI ring_at, py::sequence h_pickles,
                                            py::object families) {
  Blobs b = borrow(pickles);
  Blobs hb = borrow(h_pickles);
  need((ssize_t)hb.ptr.size() == (ssize_t)b.ptr.size(),
       "h_pickles must have one hydrogen-added blob per molecule");
  const unsigned fams = family_mask(families);
  const ssize_t nm = (ssize_t)b.ptr.size();
  need(ring_moff.ndim() == 1 && ring_ptr.ndim() == 1 && ring_at.ndim() == 1,
       "the ring arrays must be 1-D");
  need(ring_moff.shape(0) == nm + 1, "ring_moff must have n_mol + 1 entries");
  const int *RM = ring_moff.data(), *RP = ring_ptr.data(), *RA = ring_at.data();
  need(nm == 0 || (RM[nm] + 1 == ring_ptr.shape(0) && RP[RM[nm]] == ring_at.shape(0)),
       "the ring CSR does not close: ring_moff / ring_ptr / ring_at disagree on their lengths");

  auto out = py::array_t<double>({(ssize_t)nm, (ssize_t)N_ALL_COLS});
  double *O = out.mutable_data();
  {
    py::gil_scoped_release nogil;
    Flat f;
    // The pickle's own ring section is NOT read here -- molpickle::Sink's ring hooks are left
    // null on purpose; see the note at the top of this section for why the rings come across as
    // a boundary array instead.
    fill_from_pickles(b, f);
    Flat h;
    if (fams & F_AC) fill_from_pickles(hb, h, /*want_ac_charge=*/true);
    AllWork W;
    for (ssize_t k = 0; k < nm; k++) {
      const int r0 = RM[k], nr = RM[k + 1] - r0;
      const int ha0 = (fams & F_AC) ? h.atom_off[k] : 0;
      const int hb0 = (fams & F_AC) ? h.bond_off[k] : 0;
      all_row(W, f.atom_i.data(), f.atom_d.data(), f.bond_i.data(), f.bond_s.data(),
              f.bond_d.data(), f.atom_off[k], f.bond_off[k], f.atom_off[k + 1] - f.atom_off[k],
              f.bond_off[k + 1] - f.bond_off[k], f.chg_ok[k], RP + r0, RA, nr,
              h.atom_i.data(), h.bond_i.data(), h.ac_charge.data(), ha0, hb0,
              (fams & F_AC) ? h.atom_off[k + 1] - ha0 : 0,
              (fams & F_AC) ? h.bond_off[k + 1] - hb0 : 0, fams,
              O + (ssize_t)k * N_ALL_COLS);
    }
  }
  return out;
}

//! Column names, from the C++ that decides the order, so Python cannot disagree about which
//! number is which. The first HUME_NBLOCK_COLS are NOT here -- they are named in
//! src/hume/_columns.py, which is generated from the same modules cpp/verify_hume.py checks --
//! so this returns the tail and src/hume/__init__.py concatenates.
static py::list all_column_names_tail() {
  py::list out;
  for (int i = 0; i < vsabin::N_COLS; i++) out.append(py::str(vsabin::col_name(i)));
  for (int t = 0; t < N_ESTATE_TYPES; t++)
    out.append(py::str(std::string("N") + estate_tbl::ROWS[t].name));
  for (int t = 0; t < N_ESTATE_TYPES; t++)
    out.append(py::str(std::string("S") + estate_tbl::ROWS[t].name));
  for (int c = 0; c < ringcount::N_COLS; c++) out.append(py::str(ringcount::COLS[c].name));
  for (int c = 0; c < pathcount::N_COLS; c++) out.append(py::str(pathcount::COLS[c].name));
  for (int c = 0; c < topocharge::N_COLS; c++) out.append(py::str(topocharge::col_name(c)));
  for (int c = 0; c < infoic::N_IC; c++) out.append(py::str(infoic::columnNames()[c]));
  for (int c = 0; c < autocorr::N_COLS; c++) out.append(py::str(autocorr::col_name(c)));
  return out;
}

// The Crippen typer on its own, for src/hume/_verify_crippen.py. blocks() consumes the pair
// internally, so without this there is no way to compare it against RDKit's per-atom answer --
// and a per-atom comparison is strictly stronger than watching four BCUT2D columns agree.
static py::array_t<double> crippen_pairs(ArrI atom_off, ArrI bond_off, ArrI atom_i, ArrI bond_i) {
  need(atom_i.ndim() == 2 && atom_i.shape(1) == N_ATOM_INT, "atom_i must be (n_atoms, 8)");
  need(bond_i.ndim() == 2 && bond_i.shape(1) == N_BOND_INT, "bond_i must be (n_bonds, 5)");
  need(atom_off.ndim() == 1 && bond_off.ndim() == 1 && atom_off.shape(0) >= 1 &&
       atom_off.shape(0) == bond_off.shape(0), "offsets must be 1-D and the same length");
  const ssize_t nm = atom_off.shape(0) - 1;
  need(nm == 0 || (atom_off.data()[nm] == atom_i.shape(0) &&
                   bond_off.data()[nm] == bond_i.shape(0)),
       "last offset does not match the flat array length");
  const int *AO = atom_off.data(), *BO = bond_off.data();
  const int *AI = atom_i.data(), *BI = bond_i.data();
  auto out = py::array_t<double>({atom_i.shape(0), (ssize_t)2});
  double *O = out.mutable_data();
  py::gil_scoped_release nogil;
  criptyper::Mol c;
  std::vector<int32_t> cur;
  std::vector<double> lp, mr;
  for (ssize_t k = 0; k < nm; k++) {
    const int a0 = AO[k], n = AO[k + 1] - a0, b0 = BO[k], nb = BO[k + 1] - b0;
    lp.resize(n); mr.resize(n);
    crippen_fill(c, cur, AI, BI, n, nb, a0, b0, lp.data(), mr.data());
    for (int i = 0; i < n; i++) { O[2 * (a0 + i)] = lp[i]; O[2 * (a0 + i) + 1] = mr[i]; }
  }
  return out;
}

PYBIND11_MODULE(_core, mod) {
  // Crippen's drift guard, raised here so that a predicate that no longer matches the row of
  // cpp/crippen_tables.h it claims to implement is an ImportError with the two strings in it,
  // not a wrong number in column 178.
  criptyper::check();
  // The other families' guards, in the same place and for the same reason. Each hashes or
  // re-derives its own spec (VSA's bin edges, the 79 EState SMARTS, RingCount's presets) and
  // throws; pybind11 turns that into an ImportError naming what moved.
  vsabin::check();
  esttyper::selfCheck();
  ringcount::selfCheck();
  infoic::selfCheck();
  mod.doc() = "HUME's verified descriptor blocks. Arrays in, (n_mol, 182) out.";
  mod.attr("N_COLS") = (int)HUME_NBLOCK_COLS;
  mod.def("blocks", &blocks, py::arg("atom_off"), py::arg("bond_off"), py::arg("chg_ok"),
          py::arg("atom_i"), py::arg("atom_d"), py::arg("bond_i"), py::arg("bond_s"),
          py::arg("bond_d"),
          "Compute the 182 verified block columns for a batch. See src/hume/_extract.py "
          "for the array layout.");
  mod.def("crippen", &crippen_pairs, py::arg("atom_off"), py::arg("bond_off"), py::arg("atom_i"),
          py::arg("bond_i"),
          "Per-atom Wildman-Crippen (logP, MR) as (n_atoms, 2). For verification against "
          "rdMolDescriptors._CalcCrippenContribs; blocks() computes this internally.");

  // The pickle path. See src/hume_core/molpickle.h for the format pin and what it costs.
  mod.attr("PICKLE_VERSION") = py::make_tuple(molpickle::PIN_MAJOR, molpickle::PIN_MINOR,
                                              molpickle::PIN_PATCH);
  mod.attr("PICKLE_TABLES_SPEC") = std::string(pickletab::SPEC_SHA256);
  mod.attr("PICKLE_TABLES_RDKIT") = std::string(pickletab::SOURCE_RDKIT);
  mod.def("blocks_from_pickles", &blocks_from_pickles, py::arg("pickles"),
          "Compute the 182 verified block columns from a sequence of RDKit ToBinary() blobs. "
          "The whole boundary is parsed in C++; see src/hume/_extract.py's extract_pickles().");
  mod.def("pickle_extract", &pickle_extract, py::arg("pickles"),
          "The boundary arrays parsed out of ToBinary() blobs, in Batch order. For "
          "cpp/verify_molpickle.py's paired comparison against extract(); the fast path does "
          "not build these.");
  mod.def("pickle_check", &pickle_check, py::arg("probe"),
          "Assert that a pickle's MolPickler format version is the one this reader was written "
          "against. Raises RuntimeError naming both versions if not.");

  // Every family that has C++, in one pass over one parse. See all_row() for what is in it and,
  // more importantly, for what is not: Autocorrelation's 419 columns still live in cpp/ac.cpp.
  mod.attr("N_ALL_COLS") = (int)N_ALL_COLS;
  mod.attr("ALL_OFFSETS") = py::dict(
      py::arg("blocks") = (int)OFF_BLOCKS, py::arg("vsa") = (int)OFF_VSA,
      py::arg("estate") = (int)OFF_ESTATE, py::arg("ringcount") = (int)OFF_RING,
      py::arg("pathcount") = (int)OFF_PATH, py::arg("topocharge") = (int)OFF_TOPO,
      py::arg("infocontent") = (int)OFF_IC, py::arg("autocorr") = (int)OFF_AC,
      py::arg("end") = (int)N_ALL_COLS);
  mod.def("all_from_pickles", &all_from_pickles, py::arg("pickles"), py::arg("ring_moff"),
          py::arg("ring_ptr"), py::arg("ring_at"), py::arg("h_pickles"),
          py::arg("families") = py::none(),
          "Every natively computed column for a batch of ToBinary() blobs, as "
          "(n_mol, N_ALL_COLS). Pickle-only: RingCount needs the SSSR ring lists, which are in "
          "the pickle and not in the extract() boundary arrays. `families` restricts which "
          "families are computed -- for cpp/bench_e2e.py's per-family breakdown, NOT for "
          "production use; the columns of a family left out are zero, not missing.");
  mod.def("all_column_names_tail", &all_column_names_tail,
          "Column names for everything after the first 182; src/hume/_columns.py names those.");
}
