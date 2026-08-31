// A reader for RDKit's MolPickler format, filling HUME's boundary arrays with no Python in the
// path at all.
//
// WHAT THIS REPLACES AND WHY. src/hume/_extract.py asks an RDKit molecule ~300 questions per
// molecule from Python -- `a.GetAtomicNum()` is a Boost.Python round trip, not arithmetic. That
// file has already been optimised as far as the approach goes (231 -> 91 us/mol, column-at-a-time
// unbound `map`), and what is left decomposes into 15 us of building atom and bond WRAPPER
// OBJECTS THAT READ NOTHING plus per-column passes within a small factor of that floor. The only
// way past it is not to touch an RDKit object from Python at all:
//
//     m.ToBinary(PrivateProps | AtomProps | ComputedProps | NoConformers)
//
// serialises the graph, aromaticity, hybridisation, ring info, CIP codes and Gasteiger charges in
// one C++ call, and this header turns that blob straight into the same arrays.
//
// THE PRICE, STATED PLAINLY: this is a hand-written parser for a format RDKit documents only in
// its own source, and whose header says "if you add to this list, be sure to put new entries AT
// THE BOTTOM, otherwise you will break old pickles" -- a promise about READING old pickles, not
// about the bytes staying put. A misparse is a wrong descriptor with no symptom, which is the
// worst failure available here, so the format version is PINNED and asserted:
//
//   * check_version() throws naming both versions. src/hume/_extract.py calls it at import with
//     a probe pickle, so a version bump is an ImportError before any molecule is read.
//   * parse() re-checks the header of EVERY blob (three int compares, unmeasurable), so a pickle
//     made by a different RDKit than the one that greeted us cannot slip in later.
//
// WHAT IS IN THE PICKLE AND WHAT IS NOT, per boundary field. Verified field by field against
// extract() on 98,905 + 100,000 molecules by cpp/verify_molpickle.py:
//
//   IN THE PICKLE   Z, formal charge, CHIRAL TAG, hybridisation, aromatic flag, isotope,
//                   explicit H count, EXPLICIT valence, IMPLICIT valence, no-implicit flag, bond
//                   endpoints, bond type, bond aromatic flag, bond conjugation flag, bond stereo,
//                   the ring atom lists, and -- in the per-atom property section -- `_CIPCode`,
//                   `_GasteigerCharge` and the `_ChiralityPossible` bit.
//   NOT IN IT, AND NOT MAKEABLE TO BE: RDKit's NEW potential-stereo perception
//                   (`FindPotentialStereo` / `FindPotentialStereoBonds`), which `SPS` reads.
//                   Running it on the molecule before ToBinary would set STEREOANY on bonds that
//                   have no stereo, i.e. would overwrite `bond_s`. It crosses the boundary as two
//                   separate arrays; see src/hume/_extract.py's `_potential_stereo`.
//   DERIVED HERE    degree (count the bonds), total H count (explicit + implicit valence, unless
//                   noImplicit), TOTAL VALENCE (explicit + implicit valence, same guard), atom
//                   ring membership and RING COUNT (count the pickled rings an atom appears in),
//                   bond ring membership (the pickled rings name atoms; the bonds are the
//                   consecutive pairs, which is exactly how RDKit's own depickler rebuilds them),
//                   mass (Z + isotope through cpp/pickle_tables.h), bond order (BondType through
//                   the same header), and the SMARTS bond code.
//   NOT IN IT       nothing that the boundary needs. There is no field left on the Python side.
//
// TOTAL VALENCE COST THIS FILE NOTHING, and that is a measurement rather than a hope. SMARTS `v`
// is the one fragment-pattern primitive the (n_atoms, 9) boundary could not answer, and the fix
// on the reference path is one more Python call per atom -- but here the two halves were ALREADY
// IN THE BLOB and were being skipped: property-flag bit 5 is `getExplicitValence()` and bit 6 is
// `getImplicitValence()`, and RDKit's `getTotalValence()` is exactly their sum (0 of 575,571
// atoms of cpp/hard.smi disagree). So the reader stopped throwing bit 5 away instead of asking
// _extract.py to serialise a tenth field; cpp/verify_molpickle.py compares the result against
// `Atom.GetTotalValence()` column-wise on both corpora.
//
// ONE SURPRISE WORTH THE LINE. `_GasteigerCharge` is a COMPUTED property, not a private one, so
// `PrivateProps | AtomProps` does NOT contain it -- that flag pair pickles CIP codes and stops.
// ComputedProps is required, and it costs: the blob goes from 452 to 5034 bytes/mol, because
// every atom then also carries `_GasteigerHCharge`, `_CIPRank` and a `__computedProps` vector of
// their names. That is the price of the charges and it is paid knowingly; see cpp/bench_molpickle.py.
#ifndef HUME_MOLPICKLE_H
#define HUME_MOLPICKLE_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../cpp/pickle_tables.h"

// MolPickler writes little-endian regardless of host (StreamOps.h swaps on write), and this
// reader memcpy's straight out of the buffer. Every platform the wheels target is little-endian;
// failing to compile beats reading every integer backwards.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "src/hume_core/molpickle.h assumes a little-endian host"
#endif

namespace molpickle {

// --------------------------------------------------------------------------------------------
// THE PIN. RDKit 2025.09.2 writes 16.2.0 (Code/GraphMol/MolPickler.cpp, versionMajor/Minor/Patch).
// Bumping these is a decision, not a merge conflict to resolve: it means re-reading the pickler
// and re-running cpp/verify_molpickle.py on both corpora.
// --------------------------------------------------------------------------------------------
inline constexpr std::int32_t PIN_MAJOR = 16;
inline constexpr std::int32_t PIN_MINOR = 2;
inline constexpr std::int32_t PIN_PATCH = 0;

inline constexpr std::uint32_t ENDIAN_ID = 0xDEADBEEFu;

// MolPickler::Tags, as of the pinned version. Only the ones that can appear in a blob written
// with our flag set are named; anything else is an error rather than a skip.
enum Tag : std::uint8_t {
  T_VERSION = 0,
  T_BEGINATOM = 1,
  T_ENDMOL = 22,
  T_BEGINCONFS = 23,
  T_ATOM_MAPNUMBER = 24,
  T_ATOM_DUMMYLABEL = 44,
  T_BEGINBOND = 11,
  T_BEGINPROPS = 18,
  T_ENDPROPS = 19,
  T_BEGINSSSR = 20,
  T_BEGINATOMPROPS = 58,
  T_BEGINBONDPROPS = 59,
  T_BEGINSGROUP = 61,
  T_BEGINSTEREOGROUP = 62,
  T_BEGINCONFS_DOUBLE = 64,
  T_BEGINSYMMSSSR = 66,
  T_BEGINFASTFIND = 67,
  T_BEGINFINDOTHERORUNKNOWN = 68,
};

// RDGeneral/StreamOps.h DTags -- the type tag in front of each pickled property value.
enum DTag : std::uint8_t {
  D_STRING = 0, D_INT = 1, D_UINT = 2, D_BOOL = 3, D_FLOAT = 4, D_DOUBLE = 5,
  D_VECSTRING = 6, D_VECINT = 7, D_VECUINT = 8, D_VECBOOL = 9, D_VECFLOAT = 10,
  D_VECDOUBLE = 11, D_CUSTOM = 0xFE, D_END = 0xFF,
};

// Bond::BondStereo -> the boundary's +/-1, matching src/hume/_extract.py's _EZ exactly
// (TRANS is E is +1, CIS is Z is -1; STEREONONE and STEREOANY are 0).
enum : std::uint8_t { BS_NONE = 0, BS_ANY = 1, BS_Z = 2, BS_E = 3, BS_CIS = 4, BS_TRANS = 5 };

// The SMARTS bond code _extract.py sends: a bit for the ORDER when SMARTS can name it, and a
// SEPARATE bit for the aromatic FLAG. Not recoverable from the order alone -- a dative bond has
// order 1.0 without being SINGLE, and cpp/mols.smi has TRIPLE bonds carrying the aromatic flag.
enum : int { BIT_SINGLE = 1, BIT_DOUBLE = 2, BIT_TRIPLE = 4, BIT_AROM = 8 };
enum : std::uint8_t { BT_SINGLE = 1, BT_DOUBLE = 2, BT_TRIPLE = 3 };

// Atom::HybridizationType::SP3. The pickler omits hybridisation when it equals this.
inline constexpr int HYB_SP3 = 4;

// Columns per row of the boundary's `atom_i`. Mirrored in src/hume/_extract.py's N_ATOM_INT and
// in bindings.cpp; every stride in this file goes through it so a tenth column cannot be added
// in one place and forgotten in another.
//
// THE ELEVENTH AND TWELFTH COST THIS FILE NOTHING EITHER, and that is the same finding `tval`
// produced, twice over. `NumAtomStereoCenters` counts atoms carrying `_ChiralityPossible` and
// `NumUnspecifiedAtomStereoCenters` counts those of them whose chiral tag is CHI_UNSPECIFIED
// (Code/GraphMol/Descriptors/Lipinski.cpp at rdkit 2025.09.2). Both were ALREADY IN THE BLOB and
// were being stepped over:
//   * the chiral tag is atom property-flag bit 2, which this reader used to `r.skip(1)`;
//   * `_ChiralityPossible` is bit 0x8 of the per-atom `pickleExplicitProperties` bitmask, whose
//     payload this reader used to `r.skip(2)`.
// So the reader reads two bytes it was already walking past. The one real change was on the
// PYTHON side: `extract_pickles` called `AssignStereochemistry(cleanIt, force)` before ToBinary,
// which CLEARS the flag on 911 of 2,000 molecules unless `flagPossibleStereoCenters=True` is
// also passed. See src/hume/_extract.py.
//
// THE THIRTEENTH IS `getIsotope()`, and it is the third field in a row that was already being
// decoded and then dropped. The loop below has always read atom property-flag bit 8 into `iso`
// because `Atom::getMass()` needs it; it now also writes it out. QED's structural alerts 112-115
// are `[15N]`, `[13C]`, `[18O]` and `[34S]`, and nothing else at the boundary can answer them:
// `mass` says an atom IS isotope-labelled -- which is exactly the test constit.h's `exactMolWt`
// makes -- but not labelled with WHAT, and recovering the mass number by searching ISO_MASS
// backwards would put an injectivity argument where a value RDKit already has would do.
inline constexpr std::size_t N_ATOM_INT = 13;

// Columns per row of `bond_i`, for the same reason. The sixth is RDKit's Bond::BondType INTEGER,
// which is NOT recoverable from the five that were already there: the SMARTS code collapses every
// type SMARTS cannot name to the same zero, and cpp/hard.smi carries 114 DATIVE bonds (type 17)
// that a SINGLE-with-no-order-bit is indistinguishable from. It costs no bytes on this path --
// the pickler already writes the type byte and the loop below already reads it into `bt`; it is
// only being kept rather than thrown away, exactly as `tval` was.
inline constexpr std::size_t N_BOND_INT = 6;

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &m) : std::runtime_error("hume molpickle: " + m) {}
};

// --------------------------------------------------------------------------------------------
// byte reader. Bounds-checked on every read: this parses attacker-shaped input in the sense that
// matters here -- one wrong offset and every field after it is silently shifted, which is the
// exact failure mode (a `nan` in a text export shifting every subsequent column) that cost this
// project a day and that the array boundary was chosen to make impossible.
// --------------------------------------------------------------------------------------------
class Reader {
 public:
  Reader(const std::uint8_t *p, std::size_t n) : p_(p), n_(n), i_(0) {}

  std::size_t pos() const { return i_; }
  std::size_t size() const { return n_; }
  bool at_end() const { return i_ == n_; }

  void need(std::size_t k) const {
    if (i_ + k > n_) throw Error("pickle truncated");
  }
  void skip(std::size_t k) {
    need(k);
    i_ += k;
  }
  std::uint8_t u8() {
    need(1);
    return p_[i_++];
  }
  std::int8_t i8() { return (std::int8_t)u8(); }
  std::uint16_t u16() { return (std::uint16_t)raw<std::uint16_t>(); }
  std::int32_t i32() { return raw<std::int32_t>(); }
  std::uint32_t u32() { return (std::uint32_t)raw<std::int32_t>(); }
  std::uint64_t u64() { return (std::uint64_t)raw<std::int64_t>(); }
  double f64() { return raw<double>(); }

  // A pickled std::string: uint32 length then the bytes. Returned as a borrowed view.
  const char *str(std::uint32_t &len) {
    len = u32();
    need(len);
    const char *s = (const char *)(p_ + i_);
    i_ += len;
    return s;
  }

 private:
  template <typename T>
  T raw() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_ + i_, sizeof(T));   // little-endian host; see the note in parse()
    i_ += sizeof(T);
    return v;
  }
  const std::uint8_t *p_;
  std::size_t n_, i_;
};

// --------------------------------------------------------------------------------------------
// the version guard
// --------------------------------------------------------------------------------------------
inline void version_error(std::int32_t maj, std::int32_t min, std::int32_t pat) {
  throw Error("MolPickler format version " + std::to_string(maj) + "." + std::to_string(min) +
              "." + std::to_string(pat) + " but this reader is written against " +
              std::to_string(PIN_MAJOR) + "." + std::to_string(PIN_MINOR) + "." +
              std::to_string(PIN_PATCH) +
              ". The pickle layout is not a stable API; re-read Code/GraphMol/MolPickler.cpp, "
              "re-run cpp/verify_molpickle.py on both corpora, then move the pin in "
              "src/hume_core/molpickle.h. Until then use hume.featurize_blocks(), which reads "
              "the molecule through RDKit's Python API and is unaffected.");
}

//! Reads the 20-byte header, throwing unless it is exactly the pinned version.
inline void check_header(Reader &r) {
  if (r.u32() != ENDIAN_ID) throw Error("bad endian id -- not an RDKit pickle");
  if (r.i32() != (std::int32_t)T_VERSION) throw Error("no version tag");
  const std::int32_t maj = r.i32(), min = r.i32(), pat = r.i32();
  if (maj != PIN_MAJOR || min != PIN_MINOR || pat != PIN_PATCH) version_error(maj, min, pat);
}

//! The import-time guard. Hand it any pickle; it validates the header and nothing else.
inline void check_version(const std::uint8_t *buf, std::size_t len) {
  Reader r(buf, len);
  check_header(r);
}

// --------------------------------------------------------------------------------------------
// tables
// --------------------------------------------------------------------------------------------
inline double mass_of(int z, unsigned iso) {
  if (iso == 0) {
    if (z < 0 || z >= pickletab::N_Z) throw Error("atomic number out of range");
    return pickletab::ATOMIC_WEIGHT[z];
  }
  // Atom::getMass(): the isotope mass, or -- when RDKit has no mass for that isotope and the
  // atom is not a dummy -- the mass number itself.
  const std::uint32_t key = (std::uint32_t)z * 1024u + iso;
  int lo = 0, hi = pickletab::N_ISO;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (pickletab::ISO_KEY[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < pickletab::N_ISO && pickletab::ISO_KEY[lo] == key) return pickletab::ISO_MASS[lo];
  return z ? (double)iso : 0.0;
}

inline double order_of(std::uint8_t bt) {
  if (bt >= pickletab::N_BONDTYPE) throw Error("bond type out of range");
  const double o = pickletab::BOND_ORDER[bt];
  // RDKit's getBondTypeAsDouble() throws for THREECENTER / DATIVEL / DATIVER / OTHER. So does
  // _extract.py, by calling it. Throwing here keeps the two paths failing on the same molecules.
  if (std::isnan(o)) throw Error("bond type " + std::to_string((int)bt) +
                                 " has no numeric order (RDKit refuses it too)");
  return o;
}

// --------------------------------------------------------------------------------------------
// the parse
// --------------------------------------------------------------------------------------------

//! Scratch reused across molecules, so a batch does no per-molecule allocation.
struct Work {
  std::vector<int> adj_start, adj_nbr, adj_bond, cur, ring;
};

//! Where one molecule's fields land. Column meanings are bindings.cpp's A_* / B_* enums.
//
// THE RING LISTS ARE OPTIONAL AND THEY ARE NOT IN THE BOUNDARY ARRAYS AT ALL. `atom_i` carries
// a per-atom ring COUNT and `bond_i` a per-bond in-ring flag, which is everything the 182 blocks
// and the Crippen typer ask. RingCount asks a different question -- it needs the rings
// themselves, `Chem.GetSymmSSSR(mol)`, because `nG12FAHRing` is about a fused system's size and
// heteroatom content and no per-atom count can answer that. Those lists are already in the
// pickle (RDKit writes RingInfo's atom rings), so the pickle path can hand them over for free
// where src/hume/_extract.py would need a whole new pair of arrays and another Python pass.
// Pass nullptr for both and nothing changes for the callers that do not want them.
struct Sink {
  int *atom_i;      // (n, 12) Z, deg, nH, fchg, hyb, arom, ring, cip, nring, tval,
                    //         _ChiralityPossible, chiral tag
  double *atom_d;   // (n, 2)  mass, gasteiger
  int *bond_i;      // (nb, 6) u, v, conjugated, in-ring, SMARTS bond code, BondType int
  int *bond_s;      // (nb,)   E/Z as +/-1
  double *bond_d;   // (nb,)   bond order
  // Appended to, not written by index. The molecule's ring count is the growth of `ring_len`
  // across the parse() call; its rings are the corresponding runs of LOCAL atom indices in
  // `ring_at`. Both must be given together or not at all.
  std::vector<int32_t> *ring_at = nullptr;
  std::vector<int32_t> *ring_len = nullptr;
  // OPTIONAL, and a DIFFERENT QUANTITY from atom_d's charge column: mordred's Autocorrelation
  // `c` getter, which is
  //     (_GasteigerCharge + _GasteigerHCharge) if HasProp("_GasteigerHCharge") else 0.0
  // -- the sum is INSIDE the conditional, so an atom with no H-charge property contributes 0.0
  // rather than its own charge. Reproducing that here rather than in the caller is what makes it
  // a matter of construction instead of an argument about what RDKit "always" sets.
  //
  // It is also RAW: the non-finite contract that zeroes atom_d's charge column and clears
  // chg_ok is NOT applied. mordred maps a non-finite weight to a dead 54-column block, which is
  // what autocorr.h's isfinite screen does, and zeroing here would hide it.
  double *ac_charge = nullptr;
  // OPTIONAL, and the two halves of `ac_charge` kept apart, because SYNTHESISING THE H-ADDED
  // GRAPH needs them separately. On the AddHs molecule a heavy atom's `c` is its own
  // `_GasteigerCharge` (its `_GasteigerHCharge` is 0 there), and each explicit hydrogen's `c` is
  // the parent's `_GasteigerHCharge` divided by its hydrogen count. Both are recoverable from
  // the HEAVY pickle alone -- verified against a real AddHs + ComputeGasteigerCharges over 3,677
  // molecules including cpp/hard.smi: 0 mismatches in z, formal charge, nH and the bond list,
  // worst charge difference 1.67e-16. That is what lets the second pickle go away entirely.
  // Both carry mordred's conditional, exactly as `ac_charge` does: 0.0 when there is no
  // `_GasteigerHCharge` property at all.
  double *ac_own = nullptr;    // the atom's own charge
  double *ac_h = nullptr;      // the summed charge of its implicit hydrogens
};

inline void peek_sizes(const std::uint8_t *buf, std::size_t len, int &n_atoms, int &n_bonds) {
  Reader r(buf, len);
  check_header(r);
  n_atoms = r.i32();
  n_bonds = r.i32();
  if (n_atoms < 0 || n_bonds < 0) throw Error("negative atom or bond count");
}

//! Skip one pickled property value, given its DTag. Used for every atom property that is not
//! `_GasteigerCharge` or `_CIPCode` -- with ComputedProps on there are three of those per atom.
inline void skip_value(Reader &r, std::uint8_t tag) {
  std::uint32_t len;
  switch (tag) {
    case D_STRING: r.str(len); return;
    case D_INT: case D_UINT: case D_FLOAT: r.skip(4); return;
    case D_BOOL: r.skip(1); return;
    case D_DOUBLE: r.skip(8); return;
    case D_VECSTRING: {
      const std::uint64_t k = r.u64();
      for (std::uint64_t j = 0; j < k; j++) r.str(len);
      return;
    }
    case D_VECINT: case D_VECUINT: case D_VECFLOAT: r.skip(4 * (std::size_t)r.u64()); return;
    case D_VECDOUBLE: r.skip(8 * (std::size_t)r.u64()); return;
    default:
      // D_CUSTOM needs a registered handler to know its length, so there is no way to skip it.
      throw Error("unhandled property type tag " + std::to_string((int)tag));
  }
}

//! Parse one blob into `out`. Returns 1 if every atom carried a finite Gasteiger charge.
//!
//! THE NON-FINITE GASTEIGER CONTRACT, reproduced exactly as src/hume/_extract.py has it: RDKit
//! hands back inf or nan for elements PEOE has no parameters for (selenium is the common case).
//! Those get 0.0 in the charge column and the MOLECULE gets chg_ok = 0. The 0.0 is load-bearing
//! -- a nan on BCUT2D's Burden diagonal propagates through the eigensolver and returns nan for
//! all eight columns rather than the two that depend on charge.
inline int parse(const std::uint8_t *buf, std::size_t len, int n_atoms, int n_bonds,
                 const Sink &out, Work &w) {
  Reader r(buf, len);
  check_header(r);
  if (r.i32() != n_atoms || r.i32() != n_bonds) throw Error("atom/bond count moved");
  if (r.u8() != 0x80) throw Error("unexpected molecule flag byte");

  const bool wide = n_atoms > 255;   // MolPickler picks the index width off the atom count
  auto idx = [&r, wide]() -> int { return wide ? r.i32() : (int)r.u8(); };

  int *AI = out.atom_i;
  double *AD = out.atom_d;
  for (int i = 0; i < n_atoms; i++) {
    int *row = AI + (std::size_t)i * N_ATOM_INT;
    row[1] = 0;   // degree, accumulated over the bond loop
    row[6] = 0;   // in-ring boolean, from the ring section
    row[7] = 0;   // CIP, from the atom property section
    row[8] = 0;   // ring count, from the ring section
    row[10] = 0;  // _ChiralityPossible, from the atom property section's explicit-property byte
    AD[(std::size_t)i * 2 + 1] = 0.0;   // Gasteiger, from the atom property section
  }

  // ---- atoms ----
  if (r.u8() != T_BEGINATOM) throw Error("expected BEGINATOM");
  for (int i = 0; i < n_atoms; i++) {
    int *row = AI + (std::size_t)i * N_ATOM_INT;
    const int z = r.u8();
    const std::uint8_t flags = r.u8();
    const std::int32_t pf = r.i32();

    int fchg = 0, hyb = HYB_SP3, n_expl_h = 0, expl_val = 0, impl_val = 0, ctag = 0;
    unsigned iso = 0;
    if (pf & (1 << 1)) fchg = r.i8();
    // Atom::ChiralType, verbatim: CHI_UNSPECIFIED 0, CHI_TETRAHEDRAL_CW 1, CHI_TETRAHEDRAL_CCW 2,
    // CHI_OTHER 3, then the square-planar / trigonal-bipyramidal / octahedral values. The pickler
    // omits the byte when the tag is CHI_UNSPECIFIED, which is the 0 above -- so this is RDKit's
    // own enum on every atom. `NumUnspecifiedAtomStereoCenters` tests it against 0.
    if (pf & (1 << 2)) ctag = r.u8();
    if (pf & (1 << 3)) hyb = r.u8();
    if (pf & (1 << 4)) n_expl_h = r.u8();
    if (pf & (1 << 5)) expl_val = r.u8();    // getExplicitValence(); only written when > 0
    if (pf & (1 << 6)) impl_val = r.u8();
    if (pf & (1 << 7)) r.skip(1);            // radical electrons
    if (pf & (1 << 8)) iso = r.u32();
    // Bit 0 is the legacy float-isotope slot; the pinned pickler never writes it (the code that
    // did is commented out in _pickleAtomData) and this reader has no branch for it, so seeing
    // it means the format moved. Bits above 8 likewise.
    if (pf & ~0x1FE) throw Error("unknown atom property flag bit");

    if (flags & 0x10) throw Error("query atom -- hume.featurize_blocks() handles these");
    if (flags & 0x08) {                      // atom map number
      if (r.u8() != T_ATOM_MAPNUMBER) throw Error("expected ATOM_MAPNUMBER");
      if (r.i8() == -1) r.skip(4);
    }
    if (flags & 0x04) {                      // dummy label
      if (r.u8() != T_ATOM_DUMMYLABEL) throw Error("expected ATOM_DUMMYLABEL");
      std::uint32_t l;
      r.str(l);
    }
    if (flags & 0x02) throw Error("atom monomer info is not supported by this reader");

    row[0] = z;
    // GetTotalNumHs(False) == getNumExplicitHs() + getNumImplicitHs(), and getNumImplicitHs()
    // is 0 when the atom is flagged noImplicit and d_implicitValence otherwise.
    row[2] = (flags & 0x20) ? n_expl_h : n_expl_h + impl_val;
    row[3] = fchg;
    row[4] = hyb;
    row[5] = (flags & 0x40) ? 1 : 0;
    // SMARTS `v`. getTotalValence() == getExplicitValence() + getImplicitValence(), and the
    // second of those is 0 when the atom is flagged noImplicit -- the same guard row[2] applies
    // to the H count, and applied here for the same reason: it is RDKit's accessor, not an
    // observation about what d_implicitValence happens to hold.
    row[9] = expl_val + ((flags & 0x20) ? 0 : impl_val);
    row[11] = ctag;
    // SMARTS isotope, for QED's structural alerts. `iso` is already in hand -- `mass_of` on the
    // next line needs it -- so this costs the fast path a store and not a byte.
    row[12] = (int)iso;
    AD[(std::size_t)i * 2] = mass_of(z, iso);
  }

  // ---- bonds ----
  if (r.u8() != T_BEGINBOND) throw Error("expected BEGINBOND");
  int *BI = out.bond_i;
  for (int b = 0; b < n_bonds; b++) {
    const int u = idx(), v = idx();
    if (u < 0 || u >= n_atoms || v < 0 || v >= n_atoms) throw Error("bond index out of range");
    const std::uint8_t flags = r.u8();
    const std::uint8_t bt = (flags & 0x08) ? r.u8() : BT_SINGLE;
    if (flags & 0x04) r.skip(1);             // bond direction
    int stereo = 0;
    if (flags & 0x02) {
      const std::uint8_t st = r.u8();
      const int k = r.u8();
      for (int j = 0; j < k; j++) idx();      // the stereo reference atoms
      stereo = (st == BS_E || st == BS_TRANS) ? 1 : (st == BS_Z || st == BS_CIS) ? -1 : 0;
    }
    if (flags & 0x10) throw Error("query bond -- hume.featurize_blocks() handles these");
    if (flags & 0x01) {                       // _MolFileBondEndPts / _MolFileBondAttach
      std::uint32_t l;
      r.str(l);
      r.str(l);
    }

    int *row = BI + (std::size_t)b * N_BOND_INT;
    row[0] = u;
    row[1] = v;
    row[2] = (flags & 0x20) ? 1 : 0;          // conjugated
    row[3] = 0;                               // in-ring, from the ring section
    row[4] = (bt == BT_SINGLE ? BIT_SINGLE : bt == BT_DOUBLE ? BIT_DOUBLE
              : bt == BT_TRIPLE ? BIT_TRIPLE : 0) | ((flags & 0x40) ? BIT_AROM : 0);
    // Bond::BondType, verbatim. The pickler omits the byte only when the type is SINGLE, which is
    // the default `bt` above, so this is RDKit's own integer on every bond.
    row[5] = bt;
    out.bond_s[b] = stereo;
    out.bond_d[b] = order_of(bt);
    AI[(std::size_t)u * N_ATOM_INT + 1]++;
    AI[(std::size_t)v * N_ATOM_INT + 1]++;
  }

  // (neighbour -> bond index), so the ring section can name a bond by its two atoms the way
  // RDKit's own _addRingInfoFromPickle does. CSR, counting-sorted, from the caller's scratch.
  w.adj_start.assign(n_atoms + 1, 0);
  for (int b = 0; b < n_bonds; b++) {
    w.adj_start[BI[(std::size_t)b * N_BOND_INT] + 1]++;
    w.adj_start[BI[(std::size_t)b * N_BOND_INT + 1] + 1]++;
  }
  for (int i = 0; i < n_atoms; i++) w.adj_start[i + 1] += w.adj_start[i];
  w.adj_nbr.resize(2 * (std::size_t)n_bonds);
  w.adj_bond.resize(2 * (std::size_t)n_bonds);
  w.cur.assign(w.adj_start.begin(), w.adj_start.end() - 1);
  for (int b = 0; b < n_bonds; b++) {
    const int u = BI[(std::size_t)b * N_BOND_INT], v = BI[(std::size_t)b * N_BOND_INT + 1];
    w.adj_nbr[w.cur[u]] = v;
    w.adj_bond[w.cur[u]++] = b;
    w.adj_nbr[w.cur[v]] = u;
    w.adj_bond[w.cur[v]++] = b;
  }

  // ---- the tagged sections ----
  bool have_rings = false, have_atomprops = false, chg_missing = false;
  int chg_ok = 1;
  for (;;) {
    const std::uint8_t tag = r.u8();
    if (tag == T_ENDMOL) break;

    switch (tag) {
      case T_BEGINSSSR:
      case T_BEGINSYMMSSSR:
      case T_BEGINFASTFIND:
      case T_BEGINFINDOTHERORUNKNOWN: {
        have_rings = true;
        const std::uint32_t nrings = r.u32();
        for (std::uint32_t k = 0; k < nrings; k++) {
          const int sz = idx();
          if (sz <= 0) throw Error("non-positive ring size");
          w.ring.resize(sz);
          for (int j = 0; j < sz; j++) {
            const int a = idx();
            if (a < 0 || a >= n_atoms) throw Error("ring atom index out of range");
            w.ring[j] = a;
            AI[(std::size_t)a * N_ATOM_INT + 8]++;             // ring count
            AI[(std::size_t)a * N_ATOM_INT + 6] = 1;           // in-ring boolean
          }
          for (int j = 0; j < sz; j++) {
            const int a = w.ring[j], b2 = w.ring[(j + 1) % sz];
            bool found = false;
            for (int e = w.adj_start[a]; e < w.adj_start[a + 1]; e++) {
              if (w.adj_nbr[e] == b2) {
                BI[(std::size_t)w.adj_bond[e] * N_BOND_INT + 3] = 1;
                found = true;
                break;
              }
            }
            if (!found) throw Error("ring names two atoms with no bond between them");
          }
          if (out.ring_at) {
            out.ring_at->insert(out.ring_at->end(), w.ring.begin(), w.ring.end());
            out.ring_len->push_back(sz);
          }
        }
        break;
      }
      case T_BEGINATOMPROPS: {
        have_atomprops = true;
        const std::int32_t blk = r.i32();
        if (blk < 0) throw Error("negative atom-property block length");
        const std::size_t end = r.pos() + (std::size_t)blk;
        for (int i = 0; i < n_atoms; i++) {
          bool got_charge = false, got_hcharge = false;
          double q_raw = 0.0, q_h = 0.0;
          const std::uint16_t count = r.u16();
          for (std::uint16_t k = 0; k < count; k++) {
            std::uint32_t klen;
            const char *key = r.str(klen);
            const std::uint8_t dt = r.u8();
            if (klen == 16 && std::memcmp(key, "_GasteigerCharge", 16) == 0) {
              if (dt != D_DOUBLE) throw Error("_GasteigerCharge is not a double");
              const double q = r.f64();
              q_raw = q;
              if (std::isfinite(q)) {
                AD[(std::size_t)i * 2 + 1] = q;
              } else {
                AD[(std::size_t)i * 2 + 1] = 0.0;
                chg_ok = 0;
              }
              got_charge = true;
            } else if (klen == 17 && std::memcmp(key, "_GasteigerHCharge", 17) == 0) {
              if (dt != D_DOUBLE) throw Error("_GasteigerHCharge is not a double");
              q_h = r.f64();
              got_hcharge = true;
            } else if (klen == 8 && std::memcmp(key, "_CIPCode", 8) == 0) {
              if (dt != D_STRING) throw Error("_CIPCode is not a string");
              std::uint32_t vlen;
              const char *v = r.str(vlen);
              AI[(std::size_t)i * N_ATOM_INT + 7] = (vlen == 1 && v[0] == 'R') ? 1 : -1;
            } else {
              skip_value(r, dt);
            }
          }
          // pickleExplicitProperties: a bitmask byte, then one int16 per set bit, in BIT ORDER.
          // Four are defined (molStereoCare, molParity, molInversionFlag, _ChiralityPossible);
          // a fifth would be a format change, so an unknown bit is an error rather than a bad
          // offset.
          //
          // BIT 3 IS `_ChiralityPossible` AND IT IS THE WHOLE OF TWO COLUMNS. What is carried is
          // the PRESENCE of the property, not its value -- Lipinski.cpp asks `hasProp` and never
          // reads the number, and MolOps::assignStereochemistry only ever writes 1. So the bit
          // is the answer and the two payload bytes are still stepped over.
          const std::uint8_t bp = r.u8();
          if (bp & 0xF0) throw Error("unknown explicit atom-property bit");
          if (bp & 0x8) AI[(std::size_t)i * N_ATOM_INT + 10] = 1;
          for (int k = 0; k < 4; k++) {
            if (bp & (1 << k)) r.skip(2);
          }
          if (!got_charge) chg_missing = true;
          // mordred's getter, conditional and all. `got_hcharge` is HasProp; the sum only
          // happens when it is true.
          if (out.ac_charge) out.ac_charge[i] = got_hcharge ? q_raw + q_h : 0.0;
          if (out.ac_own) out.ac_own[i] = got_hcharge ? q_raw : 0.0;
          if (out.ac_h) out.ac_h[i] = got_hcharge ? q_h : 0.0;
        }
        if (r.pos() != end) throw Error("atom-property block length disagrees with its content");
        if (r.u8() != T_ENDPROPS) throw Error("expected ENDPROPS after the atom properties");
        break;
      }
      case T_BEGINPROPS:
      case T_BEGINBONDPROPS: {
        const std::int32_t blk = r.i32();
        if (blk < 0) throw Error("negative property block length");
        r.skip((std::size_t)blk);
        if (r.u8() != T_ENDPROPS) throw Error("expected ENDPROPS");
        break;
      }
      case T_BEGINCONFS:
      case T_BEGINCONFS_DOUBLE: {
        const std::int32_t blk = r.i32();
        if (blk < 0) throw Error("negative conformer block length");
        r.skip((std::size_t)blk);
        break;
      }
      case T_BEGINSTEREOGROUP: {
        // Enhanced stereo. Nothing in it reaches the boundary, so it is stepped over rather
        // than refused -- but it is stepped over by parsing it, not by a length field, because
        // _pickleStereo does not write one.
        const int ngroups = idx();
        for (int g = 0; g < ngroups; g++) {
          const int type = idx();
          if (type != 0) idx();               // write id, absent for STEREO_ABSOLUTE
          const int na = idx();
          for (int j = 0; j < na; j++) idx();
          const int nb2 = idx();
          for (int j = 0; j < nb2; j++) idx();
        }
        break;
      }
      case T_BEGINSGROUP:
        throw Error("substance groups are not supported by this reader "
                    "(they do not occur in SMILES-derived molecules)");
      default:
        throw Error("unexpected tag " + std::to_string((int)tag) + " at offset " +
                    std::to_string(r.pos() - 1));
    }
  }
  if (!r.at_end()) throw Error("trailing bytes after ENDMOL");
  if (!have_rings) {
    // RDKit only writes the ring section when RingInfo is initialised. Without it the boundary's
    // `ring` and `nring` columns would silently read zero for every atom -- a wrong descriptor
    // with no symptom, which is the one thing this reader must never produce.
    throw Error("no ring section in the pickle: the molecule was not sanitised");
  }
  // _extract.py's except branch: a molecule RDKit could not charge at all gets 0.0 for EVERY
  // atom, not just the ones that failed, and chg_ok = 0. The section is missing entirely when
  // NO atom carries any property -- an uncharged molecule with no stereocentres -- and that case
  // has to land here too, or the boundary would report chg_ok = 1 over a column of zeros.
  if (n_atoms && (!have_atomprops || chg_missing)) {
    for (int i = 0; i < n_atoms; i++) AD[(std::size_t)i * 2 + 1] = 0.0;
    chg_ok = 0;
  }
  return chg_ok;
}

}  // namespace molpickle

#endif
