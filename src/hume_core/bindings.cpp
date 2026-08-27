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

#include "crippen_typer.h"
#include "hume_blocks.h"

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
    BlockWork W;
    Mol m;
    criptyper::Mol c;
    std::vector<int32_t> cur;
    for (ssize_t k = 0; k < nm; k++) {
      const int a0 = AO[k], a1 = AO[k + 1], b0 = BO[k], b1 = BO[k + 1];
      m.n = a1 - a0;
      m.nb = b1 - b0;
      m.chg_ok = OKf[k];

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

      blocks_row(m, W, O + (ssize_t)k * HUME_NBLOCK_COLS);
    }
  }
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
}
