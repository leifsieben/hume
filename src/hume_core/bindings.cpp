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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

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
static constexpr int N_ATOM_INT = 8;   // Z, deg, nH, fchg, hyb, arom, ring, cip
static constexpr int N_ATOM_DBL = 4;   // mass, gasteiger, crippen logP, crippen MR
static constexpr int N_BOND_INT = 4;   // u, v, conjugated, in-ring

static py::array_t<double> blocks(ArrI atom_off, ArrI bond_off, ArrI chg_ok, ArrI atom_i,
                                  ArrD atom_d, ArrI bond_i, ArrI bond_s, ArrD bond_d) {
  need(atom_off.ndim() == 1 && bond_off.ndim() == 1 && chg_ok.ndim() == 1, "offsets must be 1-D");
  const ssize_t nm = chg_ok.shape(0);
  need(atom_off.shape(0) == nm + 1 && bond_off.shape(0) == nm + 1,
       "offset arrays must have n_mol + 1 entries");
  need(atom_i.ndim() == 2 && atom_i.shape(1) == N_ATOM_INT, "atom_i must be (n_atoms, 8)");
  need(atom_d.ndim() == 2 && atom_d.shape(1) == N_ATOM_DBL, "atom_d must be (n_atoms, 4)");
  need(bond_i.ndim() == 2 && bond_i.shape(1) == N_BOND_INT, "bond_i must be (n_bonds, 4)");
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
        m.mass[i] = d[0]; m.gast[i] = d[1]; m.clogp[i] = d[2]; m.cmr[i] = d[3];
      }

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

PYBIND11_MODULE(_core, mod) {
  mod.doc() = "HUME's verified descriptor blocks. Arrays in, (n_mol, 182) out.";
  mod.attr("N_COLS") = (int)HUME_NBLOCK_COLS;
  mod.def("blocks", &blocks, py::arg("atom_off"), py::arg("bond_off"), py::arg("chg_ok"),
          py::arg("atom_i"), py::arg("atom_d"), py::arg("bond_i"), py::arg("bond_s"),
          py::arg("bond_d"),
          "Compute the 182 verified block columns for a batch. See src/hume/_extract.py "
          "for the array layout.");
}
