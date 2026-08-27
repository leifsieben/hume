// The ten Autocorrelation weight vectors, in C++. Transliterated from mordred 1.2.0
// `_atomic_property.py`; the element tables come from cpp/ac_tables.h, which is generated from
// mordred itself by cpp/gen_ac_tables.py rather than retyped.
//
// This is the file that took Python out of the loop. `export_ac.py` used to call
// `ap.getters[w](atom)` once per weight per atom -- 473.9 us/mol, the single largest item in HUME's
// pipeline, larger than every other block combined -- and handed the C++ a text file of finished
// numbers. Now the export carries the RAW GRAPH (atomic number, formal charge, attached-H count,
// Gasteiger charge, bond list) and the numbers are made here.
//
// THE TEN, exactly as mordred defines them:
//
//   c    Gasteiger charge. The ONLY one still coming from RDKit, and it arrives precomputed
//        because ComputeGasteigerCharges is already C++. Note mordred's getter is
//            (_GasteigerCharge + _GasteigerHCharge) if HasProp("_GasteigerHCharge") else 0.0
//        -- the sum is inside the conditional, so an atom missing the H-charge prop contributes
//        0.0 rather than its own charge. export_ac.py reproduces that.
//   d    sigma electrons  = number of neighbours whose atomic number is NOT 1.
//   dv   valence electrons. NOT "the valence":
//            Z == 1                      -> 0
//            Zv = NOuterElecs(Z) - fc,  Zn = Z - fc,  h = GetTotalNumHs() + (# H neighbours)
//            -> (Zv - h) / (Zn - Zv - 1)
//        The formal charge cancels out of the denominator (Zn - Zv == Z - NOuterElecs), so it is
//        the core-electron count minus one and is never zero for Z > 1.
//   i    IONIZATION POTENTIAL, a table lookup. It is *not* the intrinsic state -- mordred's
//        intrinsic-state getter is short `s`, and `s` is not one of these ten.
//   p    polarizability (1994 table)          v   vdW volume, 4/3 pi r^3 from the vdW radius
//   se   Sanderson EN    pe  Pauling EN       are Allred-Rochow EN
//   Z    THE BARE ATOMIC NUMBER, and it is the odd one out in two ways worth stating rather than
//        rediscovering. mordred's getter is the whole of `get_atomic_number(a)`:
//            return a.GetAtomicNum()
//        -- no table, so it is the ONLY weight that reads no row of ac_tables.h, and the ONLY one
//        that can never be NaN. `AtomicProperty.calculate()` fails a weight when any atom's value
//        is NaN; `Z` therefore never fails, not even on the selenium and dummy-atom molecules that
//        void `se`/`p`/`i`. A dummy atom `*` is Z = 0, which is a legitimate weight of zero here
//        and not a missing value -- mordred agrees, because 0.0 is not NaN. Do not route it
//        through ac_look(): that would map Z outside [0,118] to NaN, which mordred does not do
//        (GetAtomicNum() cannot leave that range anyway, but the two would be saying different
//        things and only one of them is the spec).
//
// MISSING ELEMENTS. mordred's PeriodicTable returns NaN below Z=1 and past the end of the file,
// and a "-" cell in the file is NaN too. AtomicProperty.calculate() then sees the NaN and calls
// self.fail(), so the WHOLE WEIGHT VECTOR is missing and every descriptor built on it is missing
// -- but only that weight. A selenium-containing molecule still gets its 54 `c` columns; it is
// the 54 `se` columns that vanish. The tables are padded to Z=118 with NaN so the lookup is a
// bare index, and ac.cpp gates per weight, not per molecule.

#pragma once

#include <cmath>
#include <vector>

#include "ac_tables.h"

// c d dv i p v se pe are Z -- this order is fixed by verify_ac.py and by autocorr.h's col_name().
// `Z` is APPENDED, not inserted in mordred's own getter order, so that every one of the 486
// pre-existing columns keeps the name it already had; the 54 new ones interleave into the row as
// weight index 9 of each (variant, lag) group rather than landing in a block at the end.
static const int NW = 10;

// A Gasteiger charge RDKit could not produce. It travels as a SENTINEL NUMBER, not as the token
// "nan", because libc++'s `istream >> double` refuses "nan": it sets failbit and leaves 0.0
// behind, which is the old nan/inf export desync wearing a quieter disguise. Real Gasteiger
// charges live in about [-1, 1]. Keep in sync with C_MISSING in export_ac.py.
static const double AC_C_MISSING = -1e30;

// One atom as it comes off the wire. Isotope is deliberately absent: all ten getters read
// GetAtomicNum(), never GetMass(), so [2H] and [3H] weigh exactly what [H] does here.
struct AtomRec {
  int z = 0;      // atomic number (0 for a dummy atom `*`)
  int fc = 0;     // formal charge
  int nh = 0;     // GetTotalNumHs(), i.e. Hs still implicit AFTER AddHs -- normally 0
  double c = 0.0; // Gasteiger charge, from RDKit
};

// Z outside [0, 118] is off the end of every table, which is mordred's NaN case, not a crash.
static inline double ac_look(const double *tbl, int z) {
  return (z < 0 || z > 118) ? NAN : tbl[z];
}

// Fill w[i*NW + q] for every atom. adj must already be built from the bond list.
static void ac_weights(const std::vector<AtomRec> &at,
                       const std::vector<std::vector<int>> &adj, std::vector<double> &w) {
  const int n = (int)at.size();
  w.resize((size_t)n * NW);
  for (int i = 0; i < n; i++) {
    const AtomRec &a = at[i];
    int heavy = 0, hnb = 0;
    for (int j : adj[i]) (at[j].z == 1 ? hnb : heavy)++;

    double dv;
    if (a.z == 1) {
      dv = 0.0;
    } else {
      double zv = ac_look(AC_NOUTER, a.z) - a.fc, zn = (double)a.z - a.fc;
      dv = (zv - (a.nh + hnb)) / (zn - zv - 1.0);
    }

    double *o = &w[(size_t)i * NW];
    o[0] = (a.c <= AC_C_MISSING / 2.0) ? NAN : a.c;
    o[1] = (double)heavy;
    o[2] = dv;
    o[3] = ac_look(AC_IP, a.z);
    o[4] = ac_look(AC_POL, a.z);
    o[5] = ac_look(AC_VDWVOL, a.z);
    o[6] = ac_look(AC_SE, a.z);
    o[7] = ac_look(AC_PE, a.z);
    o[8] = ac_look(AC_ARE, a.z);
    o[9] = (double)a.z;  // mordred's `Z`: GetAtomicNum(), no table, never NaN. See the header.
  }
}
