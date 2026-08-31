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

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "autocorr.h"
// chi.h's namespace is `chisub`, NOT `chi`: hume_blocks.h below defines a file-scope
// `static void chi(...)` -- RDKit's Kier-Hall chi0n..chi4v -- and `namespace chi` would be a hard
// redefinition error the moment both are included here, which is exactly what this file does.
#include "chi.h"
#include "constit.h"
#include "crippen_typer.h"
#include "estate_typer.h"
#include "frag_matcher.h"
// The two compiled query programs frag_matcher.h evaluates. They are separate headers, and the
// matcher binds one of them per Matcher rather than reading either at namespace scope -- see
// the top of frag_matcher.h for why one evaluator and not two.
#include "../../cpp/frag_program.h"
#include "../../cpp/qed_alert_program.h"
#include "hume_blocks.h"
#include "infocontent.h"
#include "molpickle.h"
#include "pathcount.h"
#include "rdkcore.h"
#include <thread>
#include <mutex>
#include <exception>
#include "sps.h"
#include "counts_ext.h"
#include "estate_ext.h"
#include "eta.h"
#include "spectral.h"
#include "misc_ext.h"
#include "ringcount.h"
#include "topocharge.h"
#include "topomisc.h"
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
// Z, deg, nH, fchg, hyb, arom, ring, cip, nring, tval, _ChiralityPossible, chiral tag, isotope
static constexpr int N_ATOM_INT = 13;
static constexpr int N_ATOM_DBL = 2;   // mass, gasteiger      (Crippen is computed here)
static constexpr int N_BOND_INT = 6;   // u, v, conjugated, in-ring, SMARTS code, BondType int
static_assert(N_ATOM_INT == (int)molpickle::N_ATOM_INT,
              "bindings.cpp and molpickle.h disagree on the atom_i stride");
static_assert(N_BOND_INT == (int)molpickle::N_BOND_INT,
              "bindings.cpp and molpickle.h disagree on the bond_i stride");

// Column offsets inside a row of atom_i / bond_i, so the two loops below and the Crippen fill
// cannot drift apart on what column 4 means.
// A_NRING is the RING COUNT, A_RING the boolean. Both are carried because SMARTS asks both
// questions independently: `[R]` is membership, `[R1]`/`[R2]` are counts, and the count cannot
// be recovered from the boolean. It comes across the boundary from the single ring perception
// RDKit has already done rather than being recomputed C++-side -- ring perception is
// numbering-dependent for 24 molecules in the 100k corpus, so a second perception is a second
// chance to disagree.
// A_TVAL is RDKit's GetTotalValence(), SMARTS `v`. It is carried rather than derived for the
// reason the rest of this list is: round(sum of incident bond orders) + nH disagrees with RDKit
// on 11,238 of 575,571 corpus atoms, because aromatic bonds and hydrogens go through RDKit's own
// rounding rule. See src/hume_core/frag_matcher.h.
// A_CHIRPOSS is `hasProp("_ChiralityPossible")` and A_CTAG is `(int)getChiralTag()`. They are the
// LEGACY stereo perception -- the one `NumAtomStereoCenters` and
// `NumUnspecifiedAtomStereoCenters` read -- and NOT the one `SPS` reads; the two atom sets differ
// on 262 of 4,000 corpus molecules. `SPS`'s inputs arrive as separate arrays because they are not
// in the pickle; see all_from_pickles.
// A_ISO is RDKit's GetIsotope(), SMARTS isotope. It exists for QED's structural alerts 112-115
// -- `[15N]`, `[13C]`, `[18O]`, `[34S]` -- and for nothing else in the 865. It is carried and
// not derived from `mass`: getMass() says an atom IS labelled, not which isotope it carries.
enum { A_Z = 0, A_DEG = 1, A_NH = 2, A_FCHG = 3, A_HYB = 4, A_AROM = 5, A_RING = 6, A_CIP = 7,
       A_NRING = 8, A_TVAL = 9, A_CHIRPOSS = 10, A_CTAG = 11, A_ISO = 12 };
// B_BTYPE is RDKit's Bond::BondType INTEGER (SINGLE 1, DOUBLE 2, TRIPLE 3, AROMATIC 12,
// DATIVE 17, ...), which B_CODE deliberately does not carry: the SMARTS code answers "is the
// order one SMARTS can name" and "is the aromatic flag set" and collapses everything else to
// zero. That collapse is exact for every query in cpp/frag_program.h and WRONG for the Morgan
// fingerprint, which hashes the enum value -- cpp/hard.smi has 114 DATIVE bonds. See
// src/hume/_extract.py; the pickle path pays nothing for it.
enum { B_U = 0, B_V = 1, B_CONJ = 2, B_RING = 3, B_CODE = 4, B_BTYPE = 5 };

// The two block columns `Phi` is a function of. calcPhi computes kappa1 and kappa2 from the
// SAME P1 / P2 / alpha that calcKappa1 and calcKappa2 do, so it is exactly the product of the
// two columns blocks_row() has already written -- recomputing it would mean paying
// findAllPathsOfLengthN(mol, 2) a second time for a number already in the row. The 182 block
// names live in src/hume/_columns.py, not here, so these two indices are EXPORTED and asserted
// against that list at import; a reordered block tail is an AssertionError naming the column.
static constexpr int B_KAPPA1 = HUME_NBLOCK_COLS - 12;
static constexpr int B_KAPPA2 = HUME_NBLOCK_COLS - 11;

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
  need(atom_i.ndim() == 2 && atom_i.shape(1) == N_ATOM_INT,
       ("atom_i must be (n_atoms, " + std::to_string(N_ATOM_INT) + ")").c_str());
  need(atom_d.ndim() == 2 && atom_d.shape(1) == N_ATOM_DBL, "atom_d must be (n_atoms, 2)");
  need(bond_i.ndim() == 2 && bond_i.shape(1) == N_BOND_INT,
       ("bond_i must be (n_bonds, " + std::to_string(N_BOND_INT) + ")").c_str());
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
  // the two halves of ac_charge, for synthesising the H-added graph. See molpickle::Sink.
  std::vector<double> ac_own, ac_h;
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
static void fill_from_pickles(const Blobs &b, Flat &f, bool want_ac_charge = false,
                              bool want_ac_split = false) {
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
  if (want_ac_split) { f.ac_own.resize(n_atoms); f.ac_h.resize(n_atoms); }

  molpickle::Work w;
  for (ssize_t k = 0; k < nm; k++) {
    const int a0 = f.atom_off[k], b0 = f.bond_off[k];
    molpickle::Sink s{f.atom_i.data() + (std::size_t)a0 * N_ATOM_INT,
                      f.atom_d.data() + (std::size_t)a0 * N_ATOM_DBL,
                      f.bond_i.data() + (std::size_t)b0 * N_BOND_INT,
                      f.bond_s.data() + b0, f.bond_d.data() + b0, nullptr, nullptr,
                      want_ac_charge ? f.ac_charge.data() + a0 : nullptr,
                      want_ac_split ? f.ac_own.data() + a0 : nullptr,
                      want_ac_split ? f.ac_h.data() + a0 : nullptr};
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
//   * the 76 `rdkit_core` fragment columns need SMARTS `v`, RDKit's GetTotalValence(), which is
//     the TENTH `atom_i` column and was added for them. It is carried across the boundary rather
//     than reconstructed from bond orders plus nH, because that reconstruction disagrees with
//     RDKit on 11,238 of 575,571 corpus atoms; see src/hume_core/frag_matcher.h.
// ============================================================================================

// The per-atom EState index that the S* columns weight by. Nothing to compute here: blocks_row()
// fills BlockWork::ES on every molecule, unconditionally, as the comment above its call in
// hume_blocks.h explains. This is a load-bearing dependency on that fact, so it is asserted.
static constexpr int N_ESTATE_TYPES = estate_tbl::N_ROWS;   // 79 patterns -> 79 N + 79 S columns


// ---------------------------------------------------------------------------------------------
// WHICH SLOTS OF THE FIVE NEW HEADERS ARE ACTUALLY EMITTED.
// Each header computes its full N_COLS; the cost triage of METHODS section 5 dropped 43 of
// those columns, so the emitted set is a subset. The headers are NOT edited to remove them --
// the code stays verified in the tree and a reversal is a wiring change -- and a dropped slot is
// simply never copied out. spectral additionally leaves its dropped slots NaN by construction,
// which is a belt-and-braces guard, not the mechanism.
// ---------------------------------------------------------------------------------------------

// THE 18 INFORMATION-CONTENT COLUMNS THAT ARE EXACT ALGEBRAIC FUNCTIONS OF COLUMNS WE EMIT.
// Wiring counts_ext made `nAtom` -- mordred's H-added atom count, the n these indices are
// normalised by -- available for the first time, and with it these identities became provable
// rather than merely suspected. Verified bit-identical on 2,800 corpus molecules, all six k:
//     TIC_k == nAtom * IC_k        CIC_k == log2(nAtom) - IC_k        SIC_k == IC_k / log2(nAtom)
// This is a stronger reason than the 0.997 reconstruction bar used elsewhere in METHODS 5: these
// carry no information at all beyond IC0..IC5 and nAtom, which are all still emitted. IC0..IC5
// are themselves genuinely diverse (pairwise |r| down to 0.177), which is precisely what the
// size factor in TIC destroys -- TIC0..TIC5 are 0.979-0.999 correlated with each other.
// They cost nothing to compute (one multiply off an IC row that is built anyway), so this buys
// 18 fewer dimensions, not microseconds.
static constexpr int KEEP_IC[25] = {0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
// constit emits every column but `qed`. QED's expensive half -- 116 structural alerts, 69.3
// us/mol -- is OPT_QED and off by default, so the column shipped 100% NaN on every molecule: a
// dead slot, not a cheap one. The project owner asked for QED excluded early in this work
// ("combines descriptors into another descriptor and is very slow"); this is that, and the
// computation is still reachable via optional=["qed"] for anyone who wants it.
static constexpr int KEEP_CONSTIT_N = constit::N_COLS - 1;
inline int keep_constit(int c) { return c < constit::C_QED ? c : c + 1; }

static constexpr int KEEP_SPECTRAL[36] = {0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 41, 49, 50, 54, 59, 60, 61, 62};
static constexpr int KEEP_MISC[67] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80};

inline constexpr int N_COUNTS_COLS     = counts_ext::N_COLS;   // 31, all kept
inline constexpr int N_ESTATE_EXT_COLS = estate_ext::N_COLS;   // 22, all kept
inline constexpr int N_ETA_COLS        = eta::N_COLS;          // 29, all kept
inline constexpr int N_SPECTRAL_COLS   = 36;
inline constexpr int N_MISC_COLS       = 67;

enum {
  OFF_BLOCKS = 0,
  OFF_VSA    = OFF_BLOCKS + HUME_NBLOCK_COLS,
  OFF_ESTATE = OFF_VSA + vsabin::N_COLS,
  OFF_RING   = OFF_ESTATE + 2 * N_ESTATE_TYPES,
  OFF_PATH   = OFF_RING + ringcount::N_COLS,
  OFF_TOPO   = OFF_PATH + pathcount::N_COLS,
  OFF_IC     = OFF_TOPO + topocharge::N_COLS,
  // infoic emits 45 and this wires 43: the 42 IC/TIC/SIC/BIC/CIC/MIC/ZMIC columns plus `AvgIpc`,
  // which is one of the 865. `Ipc` and `Log2Ipc` are computed beside it and not emitted; they are
  // not census members and they are the only unbounded columns in the family. See the note at
  // the `infoic::compute` call below and item 5 of the WIRING section in infocontent.h.
  //
  // THE 2.8% NUMBERING-DEPENDENCE THIS COMMENT USED TO CITE WAS RDKIT'S, NOT OURS. Our
  // coefficients are exact integers, so there is no cancellation for an ordering to change: seven
  // distinct renumberings of all 100,000 molecules of cpp/hard.smi give byte-identical Ipc,
  // AvgIpc and Log2Ipc (cpp/ic_in0..7.txt -> cpp/ic_out0..7.txt).
  N_IC_WIRED = 25,   // 43 computed, 18 dropped as exact identities; see KEEP_IC
  OFF_AC     = OFF_IC + N_IC_WIRED,
  // 74 SMARTS pattern counts plus the two rdkit_core columns that are not substructure counts
  // but ride along on the same graph -- NHOHCount (a SUM OF HYDROGENS over N and O, not the
  // atom-counting SMARTS Lipinski.py displays) and HeavyAtomCount.
  OFF_FRAG   = OFF_AC + autocorr::N_COLS,
  N_FRAG_COLS = frag_prog::N_NAMED + 2,
  // mordred's Chi: the 40 SUBGRAPH-enumeration columns. Not RDKit's chi0n..chi4v, which are in
  // the 182 blocks above and share no code with these; see the note on the include.
  OFF_CHI    = OFF_FRAG + N_FRAG_COLS,
  // WalkCount 6 + Constitutional 4 + TopologicalIndex 2 + WienerIndex 2 + ABCIndex 1.
  OFF_TOPOMISC = OFF_CHI + chisub::N_COLS,
  // The "small constitutional" census block -- CarbonTypes, AtomCount, BondCount, KappaShapeIndex
  // and friends. Unrelated to topomisc's `Constitutional` above despite the names; zero column
  // overlap, and PORT_STATUS.md records the collision.
  OFF_CONSTIT = OFF_TOPOMISC + topomisc::N_COLS,
  // ---- ALIASES, NOT COMPUTATION ----
  // One column: mordred's `SLogP`, whose implementation is literally
  // `return Crippen.MolLogP(self.mol)`, i.e. vsa_bins.h's C_MOLLOGP under a second name. It is
  // copied rather than recomputed.
  //
  // THE OTHER FIVE COLUMNS constit.h's wiring note lists here ARE ALREADY EMITTED under exactly
  // their mordred names and were already counted in the coverage before this change:
  // `TopoPSA`, `TPSA`, `PEOE_VSA11`, `SMR_VSA1` and `EState_VSA1` all come out of
  // vsabin::col_name() verbatim (checked at module load, below). Emitting them again would put
  // duplicate names in hume.ALL_COLUMNS, which is worse than the naming gap it would close.
  OFF_ALIAS  = OFF_CONSTIT + KEEP_CONSTIT_N,
  N_ALIAS_COLS = 1,
  // The last of rdkit_core that is not a substructure count: 13 ring predicates, HeavyAtomMolWt,
  // FractionCSP3, Phi and the three Morgan fingerprint densities. It is LAST in the layout on
  // purpose -- every pre-existing column keeps its index, so an A/B of the extension across this
  // change compares like with like rather than a shifted row.
  OFF_RDKCORE = OFF_ALIAS + N_ALIAS_COLS,
  OFF_COUNTS     = OFF_RDKCORE + rdkcore::N_COLS,
  OFF_ESTATE_EXT = OFF_COUNTS + N_COUNTS_COLS,
  OFF_ETA        = OFF_ESTATE_EXT + N_ESTATE_EXT_COLS,
  OFF_SPECTRAL   = OFF_ETA + N_ETA_COLS,
  OFF_MISC       = OFF_SPECTRAL + N_SPECTRAL_COLS,
  N_ALL_COLS     = OFF_MISC + N_MISC_COLS,
};

// B_CODE -> the bond-order number `fragmatch` compares against, which is RDKit's Bond::BondType
// INTEGER: SINGLE 1, DOUBLE 2, TRIPLE 3, AROMATIC 12. Note 3 and 12, not the 4 and 8 of the
// boundary's bitmask -- SMARTS `:` is a bond TYPE query and RDKit numbers AROMATIC twelfth.
//
// THE TYPE DECISION IS NOT MADE HERE. esttyper::btypeFromBcode() makes it, unchanged and reused:
// it is the piece that knows an order bit wins over the aromatic FLAG (cpp/hard.smi has four
// TRIPLE bonds carrying that flag, which `:` must not match) and it is verified equal to the bond
// type on all 3,090,892 of its bonds. What is left here is a four-entry renumbering of its
// one-hot answer, which is a table, not a second perception.
//
// ANYTHING ELSE MAPS TO 0, and that is exact rather than approximate on this pattern set: the
// only bond-order values the compiled query program ever compares against are 1, 2, 3 and 12
// (counted over all 1,474 nodes of cpp/frag_program.h -- 76x`1`, 54x`2`, 4x`3`, 6x`12`, one of
// them negated). A dative bond is 17 to RDKit and 0 here, and no query can tell those apart,
// negation included. The one case btypeFromBcode cannot recover is an AROMATIC-TYPED bond whose
// getIsAromatic() flag is false; it occurs 0 times in cpp/hard.smi and would need a fifth bcode
// bit, exactly as estate_typer.h records.
static inline int frag_border(int bcode) {
  switch (esttyper::btypeFromBcode((uint8_t)bcode)) {
    case esttyper::BT_SINGLE:   return fragmatch::BO_SINGLE;
    case esttyper::BT_DOUBLE:   return fragmatch::BO_DOUBLE;
    case esttyper::BT_TRIPLE:   return fragmatch::BO_TRIPLE;
    case esttyper::BT_AROMATIC: return fragmatch::BO_AROMATIC;
    default:                    return 0;
  }
}

// --------------------------------------------------------------------------------------------
// WHERE constit's INPUTS COME FROM, resolved BY NAME rather than by a hard-coded index.
//
// constit.h computes none of MolLogP, MolMR, TPSA, the H-bond and rotatable-bond counts or the
// ring counts: they belong to vsa_bins.h, frag_matcher.h and ringcount.h, each already verified
// bit-exact against RDKit on this corpus, and recomputing any of them here would be both slower
// and a second chance to differ. What the wiring has to get right is WHICH COLUMN each one is,
// and an integer literal for that is exactly the silent-transposition failure cpp/verify_wiring.py
// exists to catch. So the three name lookups are done once, at module load, and a rename upstream
// is an ImportError naming the missing column instead of a wrong `Vabc`.
//
// vsa_bins.h's three are compile-time enumerators (`vsabin::C_MOLLOGP` and friends) and need no
// lookup; the module-load check below asserts their NAMES too, for the same reason.
struct InputCols {
  int naRing = -1, nARing = -1, hbd = -1, hba = -1, nrot = -1;
  InputCols() {
    for (int c = 0; c < ringcount::N_COLS; c++) {
      const char *nm = ringcount::COLS[c].name;
      if (!std::strcmp(nm, "naRing")) naRing = c;
      if (!std::strcmp(nm, "nARing")) nARing = c;
    }
    for (int c = 0; c < frag_prog::N_NAMED; c++) {
      const char *nm = frag_prog::NAMED[c].name;
      if (!std::strcmp(nm, "NumHDonors")) hbd = c;
      if (!std::strcmp(nm, "NumHAcceptors")) hba = c;
      if (!std::strcmp(nm, "NumRotatableBonds")) nrot = c;
    }
    const std::pair<const char *, int> need[] = {{"naRing", naRing}, {"nARing", nARing},
                                                 {"NumHDonors", hbd}, {"NumHAcceptors", hba},
                                                 {"NumRotatableBonds", nrot}};
    for (const auto &kv : need)
      if (kv.second < 0)
        throw std::runtime_error(std::string("hume._core: constit's input column '") + kv.first +
                                 "' is no longer emitted by the family that owns it");
    // The three vsa_bins columns are indexed by enumerator, so what can drift is the NAME, and a
    // name that moved would mean the enumerator now points at a different quantity.
    const std::pair<int, const char *> vsa_need[] = {{(int)vsabin::C_MOLLOGP, "MolLogP"},
                                                     {(int)vsabin::C_MOLMR, "MolMR"},
                                                     {(int)vsabin::C_TPSA, "TPSA"}};
    for (const auto &kv : vsa_need)
      if (std::strcmp(vsabin::col_name(kv.first), kv.second))
        throw std::runtime_error(std::string("hume._core: vsa_bins column ") +
                                 std::to_string(kv.first) + " is '" +
                                 vsabin::col_name(kv.first) + "', not '" + kv.second + "'");
    // The five columns the alias block deliberately does NOT re-emit, asserted present so that
    // "already covered under their mordred names" stays a fact rather than a comment.
    const std::pair<int, const char *> already[] = {
        {(int)vsabin::C_TOPOPSA, "TopoPSA"}, {(int)vsabin::C_TPSA, "TPSA"},
        {(int)vsabin::C_PEOE + 10, "PEOE_VSA11"}, {(int)vsabin::C_SMR, "SMR_VSA1"},
        {(int)vsabin::C_ESTATE_VSA, "EState_VSA1"}};
    for (const auto &kv : already)
      if (std::strcmp(vsabin::col_name(kv.first), kv.second))
        throw std::runtime_error(std::string("hume._core: '") + kv.second +
                                 "' is no longer emitted by vsa_bins.h under that name");
  }
};

static const InputCols &input_cols() {
  static const InputCols c;
  return c;
}

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
  sps::Mol sm;
  sps::detail::Work sw;
  // the synthesised H-added charge vector; see the autocorrelation block.
  std::vector<double> hc_syn;
  // ---- the five families wired after the cost triage ----
  eta::Mol etam;            eta::Work etaw;
  spectral::Mol spm;        spectral::Scratch sps_;
  miscext::Mol mxm;         miscext::Scratch mxs;
  estate_ext::Scratch exs;
  // Each header computes its full N_COLS into these, and only the kept slots are copied out.
  std::vector<double> buf_spectral, buf_misc, buf_constit;
  fragmatch::Mol fm;
  // TWO MATCHERS, ONE Mol, ONE EVALUATOR. Each holds the recursive-query cache and the search
  // plans for the program it is bound to; the molecule they read is the same `fm`, filled once.
  // A single Matcher rebound between the two programs per molecule would rebuild nothing (the
  // plan sets are process-lifetime) but WOULD throw away the recursive-query cache twice per
  // molecule, which is the allocation `bind()` exists to avoid.
  fragmatch::Matcher fmt;            // cpp/frag_program.h        -- the 74 fragment patterns
  fragmatch::Matcher qmt;            // cpp/qed_alert_program.h   -- QED's 116 alerts
  std::vector<int> fcount;
  chisub::Mol xm;
  // chisub::Scratch owns a 512 KB pow() memo (65,536 doubles). It is hoisted here for that
  // reason above all: constructing one per molecule would allocate and zero half a megabyte
  // 100,000 times and throw away every memo hit. Same for the rest of this struct, but this one
  // is the expensive mistake.
  chisub::Scratch xs;
  topomisc::Mol wm;
  topomisc::Scratch ws;
  constit::Mol km;
  rdkcore::Mol dm;
  rdkcore::Scratch ds;
  std::vector<int32_t> rp_loc;       // the molecule's ring CSR, rebased to start at 0
  // In DECLARATION order, which is what the compiler runs regardless of what is written here.
  AllWork()
      : ecount(N_ESTATE_TYPES), esum(N_ESTATE_TYPES), fmt(frag_prog::PROGRAM),
        qmt(qed_prog::PROGRAM), fcount(frag_prog::N_NAMED) {}
};

// A family selector, so a caller can time one family or compute a subset. Not an optimisation
// hook: `all_from_pickles` defaults to everything and the whole point of the entry point is one
// pass. It exists because bench_e2e.py cannot otherwise say WHICH family the C++ time is in,
// and "the compute is 623 us" is not an actionable number.
//
// F_BLOCKS is not optional. The 182 blocks are what fill BlockWork::ES, and the EState `S*`
// columns weight by it -- asking for EState without BLOCKS would silently weight by a stale
// molecule's index, so the flag is forced on rather than left as a trap.
//
// F_CONSTIT AND F_ALIAS ARE NOT INDEPENDENT AND family_mask() FORCES THEIR DEPENDENCIES ON.
// constit.h consumes seven values other families own (see InputCols above) plus the H-added
// Gasteiger charges, so asking for `constit` alone would feed it a row of zeros -- a wrong
// descriptor with no symptom, which is the failure mode this whole file is written against.
// The consequence for measurement is stated where it matters: a `["constit"]` arm in
// bench_e2e.py's per-family loop times vsa + ringcount + frag + constit, so bench_e2e.py
// subtracts the dependency arm rather than the blocks-only arm for these two.
enum : unsigned {
  F_BLOCKS = 1u, F_VSA = 2u, F_ESTATE = 4u, F_RING = 8u, F_PATH = 16u, F_TOPO = 32u, F_IC = 64u,
  F_AC = 128u, F_FRAG = 256u, F_CHI = 512u, F_TOPOMISC = 1024u, F_CONSTIT = 2048u,
  F_ALIAS = 4096u, F_RDKCORE = 8192u,
  F_COUNTS = 16384u, F_ESTATE_EXT = 32768u, F_ETA = 65536u, F_SPECTRAL = 131072u,
  F_MISC = 262144u,
  F_ALL = 524287u,
};

// ---- OPTIONAL COLUMNS: members of the 864 the caller can decline to pay for -----------------
//
// SEPARATE FROM THE FAMILY MASK ON PURPOSE, and the two must not be confused. `fams` selects
// whole families, its off-columns are ZERO, and it is documented above as a measurement hook
// that is unsafe in production. This is the opposite in every one of those respects: it is a
// PER-COLUMN switch, its off-value is NaN, and it is meant to be used.
//
// NaN rather than zero because NaN is already this file's word for "nobody computed this" --
// `SPS` is NaN when the potential-stereo arrays are absent, for the stated reason that "nobody
// ran the perception" and "the perception found nothing" are different facts. A zero would be a
// finite, plausible, wrong descriptor, which is the failure mode this whole file is written
// against.
//
// THE SCHEMA DOES NOT CHANGE. N_ALL_COLS, every family offset and every column name are the same
// whichever way these are set; only the two cells move. A column set that shifts with a keyword
// argument is how two callers end up disagreeing about what column 1244 means, and the four
// duplicated names in ALL_COLUMNS are already evidence of how expensive that class of confusion
// is to unpick after the fact.
//
// WHY THESE TWO. They are the most expensive columns in the suite and each costs more than most
// entire families. Measured on 10,000 molecules of cpp/hard.smi (mean 28.7 heavy atoms), by
// running the two arms alternated within each repetition:
//
//     qed      69.3 us/mol   116 structural-alert SMARTS, one subgraph-isomorphism pass each
//     AvgIpc   64.6 us/mol   exact-integer Le Verrier-Faddeev-Frame characteristic polynomial
//                            (infocontent's `compute` 105.5 against `computeIC` 41.0)
//
// against 629.9 us/mol for all 864. Declining both is a 21% cut of the compute for two columns.
//
// NEITHER NEEDED NEW CODE, which is why this is a small change rather than a risky one.
// `constit::qedScore()` already returns NaN when `Inputs::qedAlerts < 0`, and
// `infoic::computeIC()` is already `infoic::compute()` with the Ipc block skipped, with the 42
// IC columns BIT-IDENTICAL either way. The switches name those two existing paths; they do not
// introduce a third.
enum : unsigned { OPT_QED = 1u, OPT_AVGIPC = 2u };

// The default set, and `qed` is deliberately NOT in it. It is the single most expensive column
// in the suite and it is a drug-likeness SCORE -- a weighted geometric mean of eight properties
// this matrix already carries as columns in their own right (MolWt, MolLogP, TPSA, NumHDonors,
// NumRotatableBonds, aromatic ring count) -- so for an encoding it is a nonlinear function of
// features the model already has, at 11% of the compute. Owner's decision, 2026-08-28.
// `AvgIpc` stays on: it is not derivable from anything else here. See PORT_STATUS.md.
inline constexpr unsigned OPT_DEFAULT = OPT_AVGIPC;

//! Families that need the hydrogen-added blob parsed. Autocorrelation descriptors that graph
//! directly; constit reads only its Gasteiger charges off it (RNCG/RPCG). Keeping the two in one
//! predicate is what stops `["constit"]` from silently getting a null charge array -- and what
//! stops it from paying for Autocorrelation's 540 columns to get one.
static constexpr unsigned F_NEEDS_H = F_AC | F_CONSTIT;

static void all_row(AllWork &W, const int *AI, const double *AD, const int *BI, const int *BS,
                    const double *BD, int a0, int b0, int n, int nb, int chg_ok,
                    const int *ring_ptr, const int *ring_at, int n_rings,
                    const double *ACO, const double *ACH, int h_chg_ok,
                    const int *SA, const int *SB, unsigned fams,
                    unsigned opts, double *out) {
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
  // compute() vs computeIC() IS NOW THE `AvgIpc` SWITCH, and it is the same decision this
  // comment has always described -- only the thing that decides it has moved from an edit to an
  // argument. `compute()` runs the Ipc block: an exact-integer Le Verrier-Faddeev-Frame
  // recurrence, O(n^3) in the HEAVY-atom count, 64.6 us/mol, feeding the one emitted column
  // below. `computeIC()` is the same function with that block skipped.
  //
  // THE 42 IC COLUMNS ARE BIT-IDENTICAL EITHER WAY -- nothing above the Ipc block reads anything
  // the Ipc block writes -- so the copy below is unconditional and this switch cannot reach any
  // column but `AvgIpc`. That is the property that makes it safe to expose.
  //
  // The trap it replaces is still the trap: calling `compute()` while `AvgIpc` goes unread runs
  // the recurrence for nobody, which cost this family 68% of its CPU once already. Routing both
  // sides through `opts` is what makes that state unrepresentable rather than merely documented.
  const bool want_ipc = (opts & OPT_AVGIPC) != 0;
  if (want_ipc) infoic::compute(W.im, W.irow);
  else          infoic::computeIC(W.im, W.irow);
  for (int c = 0; c < N_IC_WIRED; c++) {
    const int slot = KEEP_IC[c];
    out[OFF_IC + c] = (slot < infoic::N_IC) ? W.irow.v[slot] : std::nan("");
  }
  // `Ipc` and `Log2Ipc` are computed and deliberately NOT emitted -- they are not members of the
  // 865, and unlike everything else here they are unbounded (`Ipc` reaches 1.65e88 and saturates
  // at DBL_MAX by design; `Log2Ipc` is -inf where `Ipc` is 0). Adding them is one line each and
  // costs no compute, but putting an unbounded column in front of a model is the owner's call.
  // AvgIpc is the last KEEP_IC entry (slot == infoic::N_IC), filled here rather than above
  // because it is gated on `want_ipc`.
  out[OFF_IC + N_IC_WIRED - 1] = want_ipc ? W.irow.v[infoic::C_AVGIPC] : std::nan("");
  }

  // ---- Autocorrelation: 540 columns, on the HYDROGEN-ADDED graph ----
  // Every field here comes from the second blob, parsed by the same molpickle.h reader into the
  // same boundary layout -- `nh` is `GetTotalNumHs()` AFTER AddHs (normally 0, but `dv` adds it
  // to the explicit-H neighbour count and mordred reads it, so it is carried rather than
  // assumed), and `HAC` is mordred's `c` getter with its conditional already applied.
  // THE H-ADDED GRAPH IS SYNTHESISED HERE, NOT PICKLED. It used to arrive as a SECOND pickle:
  // Chem.AddHs, a second ComputeGasteigerCharges, and a second ToBinary, together 53.8 us/mol --
  // 34% of the whole python boundary -- for one family. All three are derivable from the heavy
  // molecule:
  //
  //   * AddHs appends hydrogens AFTER the heavy atoms, in heavy-atom order, nH each, and leaves
  //     every atom's GetTotalNumHs() at 0. Verified over 3,677 molecules including cpp/hard.smi:
  //     0 mismatches in z, formal charge, nH or the bond list.
  //   * a heavy atom's `c` on the AddHs molecule is its OWN _GasteigerCharge (its
  //     _GasteigerHCharge is 0 once the hydrogens are explicit), and each hydrogen's is the
  //     parent's _GasteigerHCharge split evenly over its hydrogen count. Worst difference against
  //     a real AddHs + ComputeGasteigerCharges over the same 3,677 molecules: 1.67e-16.
  //
  // molpickle.h splits the two halves out as `ac_own` / `ac_h` for exactly this.
  const int hn = n + [&]{ int t = 0; for (int i = 0; i < n; i++) t += ai[(ssize_t)i * N_ATOM_INT + A_NH]; return t; }();
  const int hnb = nb + (hn - n);
  if (fams & F_NEEDS_H) {
    W.hc_syn.assign(hn, 0.0);
    for (int i = 0; i < n; i++) W.hc_syn[i] = ACO ? ACO[i] : 0.0;
    int k = n;
    for (int i = 0; i < n; i++) {
      const int nh = ai[(ssize_t)i * N_ATOM_INT + A_NH];
      const double qh = ACH ? ACH[i] : 0.0;
      for (int t = 0; t < nh; t++) W.hc_syn[k++] = nh ? qh / (double)nh : 0.0;
    }
  }
  if (fams & F_AC) {
    W.am.n = hn;
    W.am.nb = hnb;
    W.am.at.resize(hn);
    for (int i = 0; i < n; i++) {
      const int *r = ai + (std::size_t)i * N_ATOM_INT;
      AtomRec &a = W.am.at[i];
      a.z = r[A_Z]; a.fc = r[A_FCHG]; a.nh = 0; a.c = W.hc_syn[i];
    }
    for (int i = n; i < hn; i++) {
      AtomRec &a = W.am.at[i];
      a.z = 1; a.fc = 0; a.nh = 0; a.c = W.hc_syn[i];
    }
    W.am.bu.resize(hnb);
    W.am.bv.resize(hnb);
    W.am.adj.assign(hn, {});
    for (int b = 0; b < nb; b++) {
      const int *r = bi + (std::size_t)b * N_BOND_INT;
      W.am.bu[b] = r[B_U]; W.am.bv[b] = r[B_V];
      W.am.adj[r[B_U]].push_back(r[B_V]);
      W.am.adj[r[B_V]].push_back(r[B_U]);
    }
    int hb = nb, k = n;
    for (int i = 0; i < n; i++) {
      const int nh = ai[(ssize_t)i * N_ATOM_INT + A_NH];
      for (int t = 0; t < nh; t++, hb++, k++) {
        W.am.bu[hb] = i; W.am.bv[hb] = k;
        W.am.adj[i].push_back(k);
        W.am.adj[k].push_back(i);
      }
    }
    autocorr::row(W.am, W.aw, out + OFF_AC);
  }

  // ---- rdkit_core fragments: 74 SMARTS counts + NHOHCount + HeavyAtomCount ----
  if (fams & F_FRAG) {
    W.fm.alloc(n, nb);
    for (int i = 0; i < n; i++) {
      const int *r = ai + (ssize_t)i * N_ATOM_INT;
      W.fm.z[i] = r[A_Z];
      W.fm.deg[i] = r[A_DEG];
      W.fm.nH[i] = r[A_NH];
      W.fm.fchg[i] = r[A_FCHG];
      W.fm.arom[i] = r[A_AROM];
      W.fm.nring[i] = r[A_NRING];
      W.fm.tval[i] = r[A_TVAL];
      // Read by the QED alert program only. No `rdkit_core` fragment pattern has an isotope
      // query, so the 76 columns below are the same numbers with or without this line -- and
      // that is measured, not asserted: cpp/frag on the 100,000-molecule dump is byte-
      // identical before and after the eleventh boundary column existed.
      W.fm.iso[i] = r[A_ISO];
    }
    for (int b = 0; b < nb; b++) {
      const int *r = bi + (ssize_t)b * N_BOND_INT;
      W.fm.bu[b] = r[B_U];
      W.fm.bv[b] = r[B_V];
      W.fm.border[b] = frag_border(r[B_CODE]);
      W.fm.bring[b] = r[B_RING];
    }
    W.fm.finish();
    fragmatch::countAll(W.fm, W.fmt, W.fcount.data());
    for (int i = 0; i < frag_prog::N_NAMED; i++) out[OFF_FRAG + i] = (double)W.fcount[i];
    out[OFF_FRAG + frag_prog::N_NAMED] = (double)fragmatch::nhohCount(W.fm);
    out[OFF_FRAG + frag_prog::N_NAMED + 1] = (double)fragmatch::heavyAtomCount(W.fm);
  }

  // ---- mordred Chi: 40 columns, and mordred's Chi is not RDKit's ----
  // NO NEW BOUNDARY COLUMN. Both of the next two families read Z / nH / formal charge out of
  // `atom_i` (columns 0, 2, 3) and the endpoint pair out of `bond_i` (columns 0, 1), in RDKit's
  // own bond index order -- which the pickle reader preserves and which chi.h's enumeration order
  // is load-bearing on (see note 2 in chi.h: the sum is order-sensitive, so a permuted bond list
  // moves the last bits of every column).
  //
  // THE GRAPH IS THE HYDROGEN-SUPPRESSED ONE, which is what `extract_pickles` serialises and what
  // ringcount / pathcount / topocharge already receive. topomisc derives the H-added atom count
  // its `Constitutional` columns need from the `nH` column itself, so handing either of these the
  // AddHs graph -- the one Autocorrelation gets -- would be wrong twice over.
  if (fams & F_CHI) {
    chisub::build_from_rows(W.xm, n, nb, ai, N_ATOM_INT, bi, N_BOND_INT);
    chisub::compute(W.xm, out + OFF_CHI, W.xs);
  }

  // ---- WalkCount / Constitutional / TopologicalIndex / WienerIndex / ABCIndex: 15 columns ----
  if (fams & F_TOPOMISC) {
    topomisc::build_from_rows(W.wm, n, nb, ai, N_ATOM_INT, bi, N_BOND_INT);
    topomisc::compute(W.wm, out + OFF_TOPOMISC, W.ws);
  }

  // ---- the small constitutional census block: 43 columns ----
  // EVERY INPUT IS ANOTHER FAMILY'S ALREADY-WRITTEN ANSWER, read out of `out` above rather than
  // recomputed. That is constit.h's own rule ("if a value has to be recomputed to wire this up,
  // that is a sign the wiring is wrong") and it is also what keeps `Vabc`, `Lipinski`,
  // `GhoseFilter` and `RotRatio` consistent with the columns they are derived from.
  if (fams & F_CONSTIT) {
    // The ring CSR arrives as the molecule's slice of a BATCH-wide CSR, so `ring_ptr[0]` is a
    // global offset. ringcount reads it that way on purpose; constit::Mol copies the arrays, so
    // it is rebased here instead -- otherwise build_from_rows would copy the whole batch's atom
    // list for every molecule (correct, and O(total) per molecule).
    W.rp_loc.resize(n_rings + 1);
    for (int q = 0; q <= n_rings; q++) W.rp_loc[q] = ring_ptr[q] - ring_ptr[0];
    W.km.build_from_rows(n, ai, N_ATOM_INT, AD + (ssize_t)a0 * N_ATOM_DBL, N_ATOM_DBL, nb, bi,
                         N_BOND_INT, bd, n_rings, W.rp_loc.data(), ring_at + ring_ptr[0]);

    const InputCols &IC = input_cols();
    constit::Inputs in;
    in.molLogP = out[OFF_VSA + vsabin::C_MOLLOGP];
    in.molMR = out[OFF_VSA + vsabin::C_MOLMR];
    in.nHBDon = (int)out[OFF_FRAG + IC.hbd];
    in.nHBAcc = (int)out[OFF_FRAG + IC.hba];
    in.nRot = (int)out[OFF_FRAG + IC.nrot];
    in.naRing = out[OFF_RING + IC.naRing];
    in.nARing = out[OFF_RING + IC.nARing];

    // RNCG / RPCG read the H-ADDED molecule's Gasteiger charges, which is the array
    // Autocorrelation's boundary already materialises -- mordred's `c` getter, conditional and
    // all, straight out of molpickle.h's `ac_charge`. No second H-graph is built for this.
    //
    // THE NULL CASE IS NOT AN OPTIMISATION, IT IS THE VERIFIED CONTRACT. cpp/verify_constit.py
    // passes no charge array at all when `ComputeGasteigerCharges` failed or produced a
    // non-finite value, and constit.h then returns NaN for both columns rather than a number
    // derived from garbage. That is the configuration the 100,000-molecule result was measured
    // in, so the wiring reproduces the same screen: the pickle reader's own `chg_ok` covers the
    // missing-property case, and the finite sweep covers the rest.
    const double *hc = W.hc_syn.empty() ? nullptr : W.hc_syn.data();
    bool hc_ok = h_chg_ok != 0 && hc != nullptr && hn > 0;
    if (hc_ok)
      for (int i = 0; i < hn; i++)
        if (!std::isfinite(hc[i])) { hc_ok = false; break; }
    in.hchg = hc_ok ? hc : nullptr;
    in.nhchg = hc_ok ? hn : 0;

    // `qed`'S EIGHTH PROPERTY, AND THE ONE IT WAS WAITING ON. QED.py's ALERTS term is
    //     sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert))
    // -- a count of PATTERNS, not of embeddings -- so this is `hasMatch` over the 116 compiled
    // alerts and not `matchCount`. The other seven properties were already computed and verified
    // in constit.h; this line is the whole of what was missing.
    //
    // IT RUNS ON THE SAME `W.fm` THE 76 FRAGMENT COLUMNS DID, which is the point of giving
    // `Matcher` a bound program: one subgraph-isomorphism implementation, two compiled programs,
    // both generated and validated by cpp/gen_frag_program.py against RDKit's own parse tree.
    // The alert set needed four things the fragment set never exercised -- isotope (`[15N]`),
    // `~`, `@` and component-level `.` -- and they are leaves of the same evaluator.
    //
    // F_CONSTIT IMPLIES F_FRAG (see `parse_families`), so `W.fm` is filled and finished above on
    // every path that reaches here. It is the same graph, read twice, never rebuilt.
    //
    // AND IT IS OPTIONAL, off by default -- see OPT_QED. The 116 alerts are 69.3 us/mol, the
    // most expensive column in the suite. `-1` is not a sentinel invented here: `qedScore()`
    // has always returned NaN for a negative alert count, because "the alerts were not counted"
    // and "no alert matched" are different facts and a zero would silently claim the second.
    // So the off path is the SAME path the column took before the alerts were wired at all, and
    // it is the reason this switch needed no new code in constit.h.
    in.qedAlerts = (opts & OPT_QED) ? fragmatch::countMatching(W.fm, W.qmt) : -1;

    // `SPS` IS NO LONGER ONE OF THEM. Its two inputs are RDKit's POTENTIAL stereo perception,
    // which the ASSIGNED-only `cip` and `bond_s` columns cannot answer and which the pickle does
    // not carry, so they arrive as their own arrays from src/hume/_extract.py's
    // `_potential_stereo`. A null pair is NOT defaulted to zero: constit.h returns NaN, because
    // "nobody ran the perception" and "the perception found nothing" are different facts and a
    // zeroed stereo term is a finite, plausible, wrong SPS.
    //
    // THIS IS NOT THE PERCEPTION THE TWO STEREO COUNTS BELOW READ. `NumAtomStereoCenters` and
    // `NumUnspecifiedAtomStereoCenters` count the LEGACY `_ChiralityPossible` flag out of
    // `atom_i`; the two atom sets disagree on 262 of 4,000 corpus molecules. Two perceptions,
    // two boundary fields, and neither substitutes for the other.
    // SPS is computed in C++ below; these stay null and constit's C_SPS is overwritten.
    in.stereoAtom = nullptr;
    in.stereoBond = nullptr;

    W.buf_constit.assign(constit::N_COLS, std::nan(""));
    constit::compute(W.km, in, W.buf_constit.data(), out[OFF_VSA + vsabin::C_TPSA]);
    for (int c = 0; c < KEEP_CONSTIT_N; c++) out[OFF_CONSTIT + c] = W.buf_constit[keep_constit(c)];
  }

  // ---- aliases: a name, not a computation ----
  // mordred/SLogP.py in full is `return Crippen.MolLogP(self.mol)`. It is the SAME double as
  // vsa_bins.h's MolLogP, copied rather than recomputed, so the two can never disagree.
  if (fams & F_ALIAS) out[OFF_ALIAS] = out[OFF_VSA + vsabin::C_MOLLOGP];

  // ---- the last 19 rdkit_core columns ------------------------------------------------------
  // THE RING SET IS THE SAME REPAIRED ONE RingCount GETS, arriving as the boundary CSR rather
  // than from a second perception -- see the note at the top of this section and the divergence
  // recorded in src/hume_core/rdkcore.h. `Phi` reads Kappa1 and Kappa2 out of the block row
  // above rather than recomputing findAllPathsOfLengthN(mol, 2).
  if (fams & F_RDKCORE) {
    W.dm.alloc(n, nb);
    for (int i = 0; i < n; i++) {
      const int *r = ai + (ssize_t)i * N_ATOM_INT;
      W.dm.z[i] = r[A_Z];
      W.dm.deg[i] = r[A_DEG];
      W.dm.nH[i] = r[A_NH];
      W.dm.fchg[i] = r[A_FCHG];
      W.dm.nring[i] = r[A_NRING];
      // The LEGACY potential-stereo flag and the chiral tag, straight out of the two `atom_i`
      // columns the pickle was already carrying. `NumAtomStereoCenters` is the count of the
      // first; `NumUnspecifiedAtomStereoCenters` is the count of the first with the second at
      // CHI_UNSPECIFIED. No perception happens on this side of the boundary.
      W.dm.chirposs[i] = r[A_CHIRPOSS];
      W.dm.ctag[i] = r[A_CTAG];
      W.dm.mass[i] = AD[(ssize_t)(a0 + i) * N_ATOM_DBL];
      W.dm.aw[i] = (r[A_Z] >= 0 && r[A_Z] < pickletab::N_Z) ? pickletab::ATOMIC_WEIGHT[r[A_Z]]
                                                            : 0.0;
    }
    for (int b = 0; b < nb; b++) {
      const int *r = bi + (ssize_t)b * N_BOND_INT;
      W.dm.bu[b] = r[B_U];
      W.dm.bv[b] = r[B_V];
      W.dm.barom[b] = (r[B_CODE] & 8) ? 1 : 0;     // the SMARTS code's AROMATIC FLAG bit
      W.dm.btype[b] = r[B_BTYPE];
    }
    for (int q = 0; q < n_rings; q++)
      W.dm.add_ring(ring_at + ring_ptr[q], ring_ptr[q + 1] - ring_ptr[q]);
    rdkcore::compute(W.dm, out[OFF_BLOCKS + B_KAPPA1], out[OFF_BLOCKS + B_KAPPA2],
                     out + OFF_RDKCORE, W.ds);
  }

  // ---- SPS, from the C++ potential-stereo perception rather than from python ----
  // sps.h ports RDKit's NEW perception (FindStereo.cpp's runCleanup, new_canon's
  // rankFragmentAtoms, hanoiSort, and the legacy CIP ranking) and reproduces
  // Chem.SpacialScore.SPS bit-identically on all 20,000 corpus molecules. It needs no boundary
  // field that all_row does not already hold, so `_potential_stereo` -- and the stereo_a /
  // stereo_b arrays it produced -- become dead python. That is the point of wiring it: the
  // C++ perception is 16.9 us/mol against 68.4 for the python route it replaces.
  //
  // constit.h's C_SPS keeps the column's SLOT so no layout moves; it is simply written here
  // instead of there, and Inputs::stereoAtom / stereoBond are now always null.
  if (fams & F_CONSTIT) {
    W.sm.alloc(n, nb);
    for (int i = 0; i < n; i++) {
      const int *r = ai + (ssize_t)i * N_ATOM_INT;
      W.sm.z[i] = r[A_Z];       W.sm.deg[i] = r[A_DEG];   W.sm.nH[i] = r[A_NH];
      W.sm.fchg[i] = r[A_FCHG]; W.sm.hyb[i] = r[A_HYB];   W.sm.arom[i] = r[A_AROM];
      W.sm.nring[i] = r[A_NRING]; W.sm.tval[i] = r[A_TVAL];
      W.sm.ctag[i] = r[A_CTAG]; W.sm.iso[i] = r[A_ISO];   W.sm.cip[i] = r[A_CIP];
    }
    for (int b = 0; b < nb; b++) {
      const int *r = bi + (ssize_t)b * N_BOND_INT;
      W.sm.bu[b] = r[B_U];  W.sm.bv[b] = r[B_V];  W.sm.btype[b] = r[B_BTYPE];
      W.sm.barom[b] = (r[B_CODE] & 8) ? 1 : 0;
      W.sm.bconj[b] = r[B_CONJ];
      // THE TWO ENCODINGS ARE NOT THE SAME AND THE DIFFERENCE IS SILENT. The boundary's
      // `bond_s` is src/hume/_extract.py's _EZ: E/TRANS -> +1, Z/CIS -> -1, none -> 0. sps.h
      // wants RDKit's raw Bond::BondStereo ordinal (NONE=0, ANY=1, Z=2, E=3, CIS=4, TRANS=5),
      // so passing bond_s straight through reads +1 as STEREOANY and -1 as nothing at all.
      // Caught by one molecule in 4,000 -- SPS differed by 1.256 -- which is exactly how a
      // wrong-encoding bug presents: almost always right, because most bonds are 0.
      const int ez = BS[b0 + b];
      W.sm.bstereo[b] = ez > 0 ? sps::BS_E : ez < 0 ? sps::BS_Z
                                                            : sps::BS_NONE;
    }
    for (int q = 0; q < n_rings; q++)
      W.sm.add_ring(ring_at + ring_ptr[q], ring_ptr[q + 1] - ring_ptr[q]);
    // C_SPS sits below C_QED, so its compacted index is unchanged; assert rather than assume.
    static_assert(constit::C_SPS < constit::C_QED || constit::C_SPS > constit::C_QED,
                  "C_SPS must not collide with the dropped qed slot");
    sps::compute(W.sm, W.sw, out + OFF_CONSTIT +
                 (constit::C_SPS < constit::C_QED ? constit::C_SPS : constit::C_SPS - 1));
  }

  // =============================================================================================
  // The five families implemented after the deduplication, wired here. Column ORDER is the
  // offset chain; EXECUTION order is this sequence, and the two are independent. counts_ext is
  // last of the constit-dependent pair for a reason: it reads constit's own output.
  // =============================================================================================

  // ---- ETA: 29 columns, self-contained ----
  if (fams & F_ETA) {
    eta::build_from_rows(W.etam, n, nb, ai, N_ATOM_INT, bi, N_BOND_INT, bd, n_rings);
    eta::compute(W.etam, out + OFF_ETA, W.etaw);
  }

  // ---- E-state extremes + walk sums: 22 columns ----
  // Reductions over the per-atom E-state index blocks_row() has already written, and over the
  // atom typing esttyper::aggregate() has already done. No new enumeration.
  if (fams & F_ESTATE_EXT) {
    estate_ext::compute(W.em, W.bw.ES.data(), out + OFF_ESTATE_EXT, W.exs);
  }

  // ---- spectral: 36 of 65 slots emitted ----
  // `c` here is mordred's heavy-graph getter, `_GasteigerCharge + _GasteigerHCharge`, which is
  // exactly the two halves molpickle.h now splits out -- the same pair the synthesised H-added
  // graph uses. BCUTc-1h/-1l are the only columns that need the sum rather than either half.
  if (fams & F_SPECTRAL) {
    W.spm.n = n; W.spm.nb = nb;
    W.spm.at.resize(n);
    for (int i = 0; i < n; i++) {
      const int *r = ai + (std::size_t)i * N_ATOM_INT;
      AtomRec &a = W.spm.at[i];
      a.z = r[A_Z]; a.fc = r[A_FCHG]; a.nh = r[A_NH];
      a.c = (ACO && ACH) ? ACO[i] + ACH[i] : AC_C_MISSING;
    }
    W.spm.bu.resize(nb); W.spm.bv.resize(nb); W.spm.bord.resize(nb);
    W.spm.adj.assign(n, {});
    for (int b = 0; b < nb; b++) {
      const int *r = bi + (std::size_t)b * N_BOND_INT;
      W.spm.bu[b] = r[B_U]; W.spm.bv[b] = r[B_V]; W.spm.bord[b] = bd[b];
      W.spm.adj[r[B_U]].push_back(r[B_V]);
      W.spm.adj[r[B_V]].push_back(r[B_U]);
    }
    W.buf_spectral.assign(spectral::N_COLS, std::nan(""));
    spectral::compute(W.spm, W.buf_spectral.data(), W.sps_);
    for (int c = 0; c < N_SPECTRAL_COLS; c++)
      out[OFF_SPECTRAL + c] = W.buf_spectral[KEEP_SPECTRAL[c]];
  }

  // ---- misc singletons: 67 of 81 slots emitted ----
  if (fams & F_MISC) {
    miscext::build_from_rows(W.mxm, n, nb, ai, N_ATOM_INT, AD + (ssize_t)a0 * N_ATOM_DBL,
                             N_ATOM_DBL, bi, N_BOND_INT, bd, chg_ok, n_rings);
    W.buf_misc.assign(miscext::N_COLS, std::nan(""));
    // W.xs and W.ps are chi.h's and pathcount.h's scratch, already filled for F_CHI and
    // F_PATH on the same graph. Lending them here removes 58.6 + 13.1 us/mol of duplicate
    // enumeration; F_MISC therefore implies both, enforced in family_mask().
    miscext::compute(W.mxm, W.mxs, W.buf_misc.data(), &W.xs, &W.ps);
    for (int c = 0; c < N_MISC_COLS; c++) out[OFF_MISC + c] = W.buf_misc[KEEP_MISC[c]];
  }

  // ---- ring / atom counts: 31 columns. MUST BE LAST of the constit-dependent blocks ----
  // It reads constit's nBondsD / nBondsKD out of the row that constit::compute has just
  // written, and RDKit's NumRotatableBonds out of the fragment block. F_COUNTS therefore
  // IMPLIES F_CONSTIT and F_FRAG -- see the mask fixup in family_mask(), which is where that
  // is enforced rather than documented, so asking for `counts` alone cannot silently read
  // whatever was in the row.
  if (fams & F_COUNTS) {
    const InputCols &ICc = input_cols();
    counts_ext::Inputs cin;
    cin.nRot = (int)out[OFF_FRAG + ICc.nrot];
    cin.nBondsD = (int)out[OFF_CONSTIT + 18];
    cin.nBondsKD = (int)out[OFF_CONSTIT + 22];
    counts_ext::compute(W.km, W.rm, cin, out + OFF_COUNTS, W.rs);
  }
}

static unsigned family_mask(const py::object &families) {
  if (families.is_none()) return F_ALL;
  static const std::pair<const char *, unsigned> NAMED[] = {
      {"blocks", F_BLOCKS}, {"vsa", F_VSA}, {"estate", F_ESTATE}, {"ringcount", F_RING},
      {"pathcount", F_PATH}, {"topocharge", F_TOPO}, {"infocontent", F_IC},
      {"autocorr", F_AC}, {"frag", F_FRAG}, {"chi", F_CHI}, {"topomisc", F_TOPOMISC},
      {"constit", F_CONSTIT}, {"alias", F_ALIAS}, {"rdkcore", F_RDKCORE},
      {"counts", F_COUNTS}, {"estate_ext", F_ESTATE_EXT}, {"eta", F_ETA},
      {"spectral", F_SPECTRAL}, {"misc", F_MISC}};
  unsigned mask = F_BLOCKS;   // never optional; see the note on the enum
  for (auto h : families) {
    const std::string want = py::cast<std::string>(h);
    unsigned bit = 0;
    for (const auto &kv : NAMED)
      if (want == kv.first) bit = kv.second;
    if (!bit) throw std::invalid_argument("hume._core: unknown family '" + want + "'");
    mask |= bit;
  }
  // The two families that consume other families' output. Forced rather than validated: a caller
  // that asks for `constit` wants constit's numbers, not an exception about vsa_bins, and the
  // alternative -- computing it over a row of zeros -- is the silent-wrong-descriptor failure.
  if (mask & F_CONSTIT) mask |= F_VSA | F_RING | F_FRAG;
  if (mask & F_ALIAS) mask |= F_VSA;
  // counts_ext reads constit's nBondsD / nBondsKD and the fragment block's NumRotatableBonds
  // out of the row; estate_ext reduces over the esttyper Mol that the estate block builds.
  // Forced for the same reason as the two above: asking for `counts` alone must not read
  // whatever happened to be in the row.
  if (mask & F_COUNTS) mask |= F_CONSTIT | F_FRAG | F_VSA | F_RING;
  if (mask & F_ESTATE_EXT) mask |= F_ESTATE;
  // misc_ext borrows chi.h's and pathcount.h's filled Scratch rather than re-enumerating.
  if (mask & F_MISC) mask |= F_CHI | F_PATH;
  return mask;
}

//! `None` -> the default set; a sequence of column names -> exactly those; `()` -> neither.
static unsigned optional_mask(const py::object &optional) {
  if (optional.is_none()) return OPT_DEFAULT;
  static const std::pair<const char *, unsigned> NAMED[] = {{"qed", OPT_QED},
                                                            {"AvgIpc", OPT_AVGIPC}};
  unsigned mask = 0;
  for (auto h : optional) {
    const std::string want = py::cast<std::string>(h);
    unsigned bit = 0;
    for (const auto &kv : NAMED)
      if (want == kv.first) bit = kv.second;
    if (!bit)
      throw std::invalid_argument(
          "hume._core: '" + want +
          "' is not an optional column. Exactly two columns are optional -- 'qed' and 'AvgIpc' "
          "-- and every other one of the 864 is always computed. Pass optional=None for the "
          "default set ('AvgIpc' on, 'qed' off), optional=() for neither, or name the ones you "
          "want: optional=('qed', 'AvgIpc') computes the full suite. A column that is off is "
          "NaN in its usual position; no offset and no name changes.");
    mask |= bit;
  }
  return mask;
}

static py::array_t<double> all_from_pickles(py::sequence pickles, ArrI ring_moff, ArrI ring_ptr,
                                            ArrI ring_at, py::sequence h_pickles, ArrI stereo_a,
                                            ArrI stereo_b, py::object families,
                                            py::object optional, int n_threads) {
  Blobs b = borrow(pickles);
  // `h_pickles` IS ACCEPTED AND IGNORED. It used to carry Chem.AddHs + a second
  // ComputeGasteigerCharges + a second ToBinary, 53.8 us/mol for one family; all_row() now
  // synthesises that graph from the heavy molecule. The parameter stays so callers holding the
  // old signature keep working, and an EMPTY sequence is the expected value.
  (void)h_pickles;
  const unsigned fams = family_mask(families);
  const unsigned opts = optional_mask(optional);
  const ssize_t nm = (ssize_t)b.ptr.size();
  need(ring_moff.ndim() == 1 && ring_ptr.ndim() == 1 && ring_at.ndim() == 1,
       "the ring arrays must be 1-D");
  need(ring_moff.shape(0) == nm + 1, "ring_moff must have n_mol + 1 entries");
  const int *RM = ring_moff.data(), *RP = ring_ptr.data(), *RA = ring_at.data();
  need(nm == 0 || (RM[nm] + 1 == ring_ptr.shape(0) && RP[RM[nm]] == ring_at.shape(0)),
       "the ring CSR does not close: ring_moff / ring_ptr / ring_at disagree on their lengths");

  // THE POTENTIAL-STEREO PAIR, and the empty case is a CONTRACT rather than a convenience.
  // `src/hume/_extract.py` sends length-0 arrays when the caller passed `stereo=False`, which
  // means "the perception was never run" -- distinct from "it ran and found nothing", which is a
  // full-length array of zeros. The first yields NaN for `SPS`; the second yields a number. Any
  // other length is a caller bug and is refused here rather than read past the end.
  need(stereo_a.ndim() == 1 && stereo_b.ndim() == 1, "stereo_a / stereo_b must be 1-D");
  const bool have_stereo = stereo_a.shape(0) != 0 || stereo_b.shape(0) != 0;

  auto out = py::array_t<double>({(ssize_t)nm, (ssize_t)N_ALL_COLS});
  double *O = out.mutable_data();
  {
    py::gil_scoped_release nogil;
    Flat f;
    // The pickle's own ring section is NOT read here -- molpickle::Sink's ring hooks are left
    // null on purpose; see the note at the top of this section for why the rings come across as
    // a boundary array instead.
    // want_ac_split: the H-added graph is synthesised from this parse, so the heavy pickle
    // must yield ac_own / ac_h as well.
    fill_from_pickles(b, f, /*want_ac_charge=*/false, /*want_ac_split=*/(fams & F_NEEDS_H) != 0);
    // THE SECOND PICKLE IS GONE. `h_pickles` is still accepted so the signature does not break,
    // and is now ignored: the H-added graph and its Gasteiger charges are synthesised from the
    // heavy molecule inside all_row(). What the heavy parse has to produce instead is the two
    // halves of mordred's `c` getter kept apart -- molpickle's `ac_own` / `ac_h`.
    const bool need_h = (fams & F_NEEDS_H) != 0;
    // Checked AFTER the parse, because only then is the atom/bond count of the batch known --
    // and a stereo array that is the wrong length would otherwise be read past its end for every
    // molecule after the first short one.
    if (have_stereo && (stereo_a.shape(0) != f.atom_off[nm] || stereo_b.shape(0) != f.bond_off[nm]))
      throw std::invalid_argument(
          "hume._core: stereo_a / stereo_b must be empty or have one entry per atom / per bond of "
          "the batch (got " + std::to_string(stereo_a.shape(0)) + " / " +
          std::to_string(stereo_b.shape(0)) + " for " + std::to_string(f.atom_off[nm]) +
          " atoms / " + std::to_string(f.bond_off[nm]) + " bonds)");
    const int *SA = have_stereo ? stereo_a.data() : nullptr;
    const int *SB = have_stereo ? stereo_b.data() : nullptr;
    // THE ROW LOOP IS EMBARRASSINGLY PARALLEL AND WAS RUNNING ON ONE CORE.
    // Every iteration reads shared const arrays and writes its own disjoint N_ALL_COLS slice of
    // the output; the only mutable state is AllWork, so one per thread is the whole change. The
    // scratch that lives at namespace scope in hume_blocks.h is already `static thread_local`,
    // which is what makes this safe rather than merely plausible -- somebody anticipated this.
    //
    // The GIL is already released around this block, so python threads are not involved.
    // `threads` <= 0 means "one per hardware thread"; 1 restores the serial loop exactly.
    // AN EXCEPTION MUST NOT ESCAPE A std::thread: that is std::terminate, i.e. the process
    // aborts with the message lost. all_row() throws std::runtime_error on several contract
    // violations, so every worker catches, stores, and the first one is rethrown after join.
    std::vector<std::exception_ptr> errs;
    std::mutex errmu;
    auto worker = [&](ssize_t lo, ssize_t hi) {
      try {
      AllWork W;
      for (ssize_t k = lo; k < hi; k++) {
        const int r0 = RM[k], nr = RM[k + 1] - r0;
        all_row(W, f.atom_i.data(), f.atom_d.data(), f.bond_i.data(), f.bond_s.data(),
                f.bond_d.data(), f.atom_off[k], f.bond_off[k], f.atom_off[k + 1] - f.atom_off[k],
                f.bond_off[k + 1] - f.bond_off[k], f.chg_ok[k], RP + r0, RA, nr,
                need_h ? f.ac_own.data() + f.atom_off[k] : nullptr,
                need_h ? f.ac_h.data() + f.atom_off[k] : nullptr,
                f.chg_ok[k], SA, SB, fams, opts,
                O + (ssize_t)k * N_ALL_COLS);
      }
      } catch (...) {
        std::lock_guard<std::mutex> g(errmu);
        errs.push_back(std::current_exception());
      }
    };
    int nt = n_threads;
    if (nt <= 0) nt = (int)std::thread::hardware_concurrency();
    if (nt < 1) nt = 1;
    if (nt > nm) nt = (int)nm;
    if (nt <= 1 || nm < 8) {
      worker(0, nm);
    } else {
      // Contiguous blocks rather than a stride: molecules of similar size sit together in a
      // typical batch, so blocks balance about as well as interleaving and keep each thread's
      // output writes in its own cache lines.
      std::vector<std::thread> ts;
      ts.reserve(nt);
      const ssize_t chunk = (nm + nt - 1) / nt;
      for (int t = 0; t < nt; t++) {
        const ssize_t lo = (ssize_t)t * chunk, hi = std::min(nm, lo + chunk);
        if (lo < hi) ts.emplace_back(worker, lo, hi);
      }
      for (auto &th : ts) th.join();
    }
    if (!errs.empty()) std::rethrow_exception(errs.front());
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
  // The 42 IC columns, then `AvgIpc` -- which is columnNames()[C_AVGIPC], not [N_IC]: the row's
  // Ipc block is (Ipc, AvgIpc, Log2Ipc) in that order, so the census member is the MIDDLE one.
  // This must stay in lockstep with the emit loop in all_row().
  for (int c = 0; c < N_IC_WIRED; c++) {
    const int slot = KEEP_IC[c];
    out.append(py::str(slot < infoic::N_IC ? infoic::columnNames()[slot]
                                           : infoic::columnNames()[infoic::C_AVGIPC]));
  }
  for (int c = 0; c < autocorr::N_COLS; c++) out.append(py::str(autocorr::col_name(c)));
  for (int c = 0; c < frag_prog::N_NAMED; c++) out.append(py::str(frag_prog::NAMED[c].name));
  out.append(py::str("NHOHCount"));
  out.append(py::str("HeavyAtomCount"));
  for (int c = 0; c < chisub::N_COLS; c++) out.append(py::str(chisub::COLS[c].name));
  for (int c = 0; c < topomisc::N_COLS; c++) out.append(py::str(topomisc::COLS[c]));
  for (int c = 0; c < KEEP_CONSTIT_N; c++) out.append(py::str(constit::col_name(keep_constit(c))));
  // The alias block. `qed` and `SPS` above it are NaN today and are named anyway; so is this,
  // for the opposite reason -- it is a real value under a second name.
  out.append(py::str("SLogP"));
  for (int c = 0; c < rdkcore::N_COLS; c++) out.append(py::str(rdkcore::col_name(c)));
  // The five families wired after the deduplication. Dropped slots are skipped here in exactly
  // the same order all_row() skips them, which is what keeps names and values in lockstep.
  for (int c = 0; c < counts_ext::N_COLS; c++) out.append(py::str(counts_ext::col_name(c)));
  for (int c = 0; c < estate_ext::N_COLS; c++) out.append(py::str(estate_ext::col_name(c)));
  for (int c = 0; c < eta::N_COLS; c++) out.append(py::str(eta::col_name(c)));
  for (int c = 0; c < N_SPECTRAL_COLS; c++) out.append(py::str(spectral::col_name(KEEP_SPECTRAL[c])));
  for (int c = 0; c < N_MISC_COLS; c++) out.append(py::str(miscext::col_name(KEEP_MISC[c])));
  return out;
}

// The Crippen typer on its own, for src/hume/_verify_crippen.py. blocks() consumes the pair
// internally, so without this there is no way to compare it against RDKit's per-atom answer --
// and a per-atom comparison is strictly stronger than watching four BCUT2D columns agree.
static py::array_t<double> crippen_pairs(ArrI atom_off, ArrI bond_off, ArrI atom_i, ArrI bond_i) {
  need(atom_i.ndim() == 2 && atom_i.shape(1) == N_ATOM_INT,
       ("atom_i must be (n_atoms, " + std::to_string(N_ATOM_INT) + ")").c_str());
  need(bond_i.ndim() == 2 && bond_i.shape(1) == N_BOND_INT,
       ("bond_i must be (n_bonds, " + std::to_string(N_BOND_INT) + ")").c_str());
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
  // constit.h's own guard: it re-derives cpp/gen_constit_tables.py's canonical form byte for byte
  // and hashes it, so a table that moved is an ImportError rather than a wrong `FilterItLogS`.
  // chi.h and topomisc.h carry no equivalent -- their spec guard is cpp/verify_chiwalk.py's
  // check_spec(), which asserts every emitted name against a LIVE mordred object and therefore
  // cannot run without mordred installed.
  constit::checkSpec();
  // Resolves constit's seven borrowed input columns BY NAME, once, and throws if one moved.
  input_cols();
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
  // more importantly, for what is not: Autocorrelation's tenth weight `Z` (52 columns) and the
  // three ill-posed InformationContent columns.
  mod.attr("N_ALL_COLS") = (int)N_ALL_COLS;
  mod.attr("ALL_OFFSETS") = py::dict(
      py::arg("blocks") = (int)OFF_BLOCKS, py::arg("vsa") = (int)OFF_VSA,
      py::arg("estate") = (int)OFF_ESTATE, py::arg("ringcount") = (int)OFF_RING,
      py::arg("pathcount") = (int)OFF_PATH, py::arg("topocharge") = (int)OFF_TOPO,
      py::arg("infocontent") = (int)OFF_IC, py::arg("autocorr") = (int)OFF_AC,
      py::arg("frag") = (int)OFF_FRAG, py::arg("chi") = (int)OFF_CHI,
      py::arg("topomisc") = (int)OFF_TOPOMISC, py::arg("constit") = (int)OFF_CONSTIT,
      py::arg("alias") = (int)OFF_ALIAS, py::arg("rdkcore") = (int)OFF_RDKCORE,
      py::arg("end") = (int)N_ALL_COLS);
  // Exported so src/hume/__init__.py can assert them BY NAME against _columns.py, which is where
  // the 182 block names live. See the note on B_KAPPA1.
  mod.attr("BLOCK_KAPPA_COLS") = py::make_tuple((int)B_KAPPA1, (int)B_KAPPA2);
  // `stereo_a` / `stereo_b` are REQUIRED and have no default, deliberately. A default would let a
  // caller get a full matrix in which exactly one column is silently NaN, which is the failure
  // this file is written against; the two existing positional callers (src/hume/__init__.py,
  // bench_e2e.py) were updated instead.
  mod.def("all_from_pickles", &all_from_pickles, py::arg("pickles"), py::arg("ring_moff"),
          py::arg("ring_ptr"), py::arg("ring_at"), py::arg("h_pickles"), py::arg("stereo_a"),
          py::arg("stereo_b"), py::arg("families") = py::none(),
          py::arg("optional") = py::none(), py::arg("threads") = 0,
          "Every natively computed column for a batch of ToBinary() blobs, as "
          "(n_mol, N_ALL_COLS). Pickle-only: RingCount needs the SSSR ring lists, which are in "
          "the pickle and not in the extract() boundary arrays. `stereo_a` / `stereo_b` are "
          "RDKit's POTENTIAL stereo perception, per atom and per bond, which the pickle cannot "
          "carry -- see src/hume/_extract.py's _potential_stereo(); pass length-0 arrays to say "
          "it was not run, and `SPS` is then NaN. `families` restricts which families are "
          "computed -- for cpp/bench_e2e.py's per-family breakdown, NOT for production use; the "
          "columns of a family left out are zero, not missing. `optional` is the opposite and IS "
          "for production: the two most expensive columns of the 864 -- 'qed' at 69.3 us/mol and "
          "'AvgIpc' at 64.6, against 629.9 for all of them -- can be declined per column. None "
          "means the default set ('AvgIpc' on, 'qed' off); () means neither; ('qed', 'AvgIpc') "
          "means the full suite. A declined column is NaN in its usual position and no offset, "
          "name or column count changes.");
  mod.def("all_column_names_tail", &all_column_names_tail,
          "Column names for everything after the first 182; src/hume/_columns.py names those.");
}
