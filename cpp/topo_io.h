// The text exchange format shared by the three pure-topology harnesses
// (cpp/ringcount.cpp, cpp/topocharge.cpp, cpp/pathcount.cpp).
//
// Written by cpp/verify_topo3.py's dump_mols(); one format for all three because all three read
// the SAME molecule -- mordred's `Chem.RemoveHs(mol)` -- and differ only in which fields they
// look at. Keeping one file means the three binaries cannot silently be verified against three
// different graphs, which is the failure mode that would make an "ALL EXACT" line meaningless.
//
//   NMOL
//   repeated NMOL times:
//     n nb nr
//     n lines:   z arom                       GetAtomicNum(), GetIsAromatic()
//     nb lines:  u v order                    bond ends, GetBondTypeAsDouble() as %.17g
//     nr lines:  k a1 a2 ... ak               one ring of Chem.GetSymmSSSR(mol)
//
// The bond order is the ONE float in the file and it is written %.17g, which round-trips a
// float64 exactly. Everything else is an integer, so this file has no precision of its own.
#ifndef HUME_TOPO_IO_H
#define HUME_TOPO_IO_H

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace topo_io {

struct Rec {
  int n = 0, nb = 0, nr = 0;
  std::vector<int32_t> z, arom;
  std::vector<int32_t> bu, bv;
  std::vector<double> bo;
  std::vector<int32_t> ring_off, ring_at;
  // The same atoms and bonds in the BOUNDARY's strided-row layout, so the harnesses exercise the
  // very builders bindings.cpp will call -- topocharge::build() and pathcount::build_from_rows()
  // -- instead of a second CSR written only for the harness. The atom rows are the FIRST NINE
  // columns of src/hume/_extract.py's (n_atoms, 10) atom_i -- Z deg nH fchg hyb arom ring cip
  // nring, with `tval` left off because neither builder reads it and both take the stride as an
  // argument; bond_i (n_bonds, 5) = u v conj
  // ring code. Only the columns those two builders read are filled; the rest are left at zero,
  // which is exactly what makes a wrong column index show up as a failure rather than as a value.
  static constexpr int ASTRIDE = 9, BSTRIDE = 5;
  std::vector<int32_t> arows, brows;
};

inline std::vector<Rec> load(const char *path) {
  FILE *f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s (run: python cpp/verify_topo3.py --dump)\n", path);
    std::exit(1);
  }
  int nm = 0;
  if (std::fscanf(f, "%d", &nm) != 1) { std::fprintf(stderr, "%s: bad header\n", path); std::exit(1); }
  std::vector<Rec> out(nm);
  for (int k = 0; k < nm; ++k) {
    Rec &r = out[k];
    if (std::fscanf(f, "%d %d %d", &r.n, &r.nb, &r.nr) != 3) {
      std::fprintf(stderr, "%s: truncated at molecule %d\n", path, k); std::exit(1);
    }
    r.z.resize(r.n); r.arom.resize(r.n);
    for (int i = 0; i < r.n; ++i)
      if (std::fscanf(f, "%d %d", &r.z[i], &r.arom[i]) != 2) {
        std::fprintf(stderr, "%s: bad atom %d of molecule %d\n", path, i, k); std::exit(1);
      }
    r.bu.resize(r.nb); r.bv.resize(r.nb); r.bo.resize(r.nb);
    for (int b = 0; b < r.nb; ++b)
      if (std::fscanf(f, "%d %d %lf", &r.bu[b], &r.bv[b], &r.bo[b]) != 3) {
        std::fprintf(stderr, "%s: bad bond %d of molecule %d\n", path, b, k); std::exit(1);
      }
    r.ring_off.assign(1, 0);
    for (int q = 0; q < r.nr; ++q) {
      int len = 0;
      if (std::fscanf(f, "%d", &len) != 1) {
        std::fprintf(stderr, "%s: bad ring %d of molecule %d\n", path, q, k); std::exit(1);
      }
      for (int t = 0; t < len; ++t) {
        int a = 0;
        if (std::fscanf(f, "%d", &a) != 1) {
          std::fprintf(stderr, "%s: bad ring atom, molecule %d\n", path, k); std::exit(1);
        }
        r.ring_at.push_back(a);
      }
      r.ring_off.push_back((int32_t)r.ring_at.size());
    }
    r.arows.assign((size_t)r.n * Rec::ASTRIDE, 0);
    for (int i = 0; i < r.n; ++i) r.arows[(size_t)i * Rec::ASTRIDE + 0] = r.z[i];
    r.brows.assign((size_t)r.nb * Rec::BSTRIDE, 0);
    for (int b = 0; b < r.nb; ++b) {
      r.brows[(size_t)b * Rec::BSTRIDE + 0] = r.bu[b];
      r.brows[(size_t)b * Rec::BSTRIDE + 1] = r.bv[b];
    }
  }
  std::fclose(f);
  return out;
}

}  // namespace topo_io

#endif  // HUME_TOPO_IO_H
