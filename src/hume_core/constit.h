// The small constitutional families: 43 of the 865 deduplicated columns that are each a short
// walk over the graph plus a table, and that share no machinery big enough to deserve a file of
// its own.
//
//   CarbonTypes 9   AtomCount 8   BondCount 6   KappaShapeIndex 3   MolecularDistanceEdge 3
//   CPSA 2 (RNCG/RPCG)   Lipinski 2   AcidBase 2   rdkit_composite 2 (qed, SPS)
//   VdwVolumeABC 1   RotatableBond 1   Polarizability 1   LogS 1   Framework 1
//   FragmentComplexity 1
//
// SIX MORE COLUMNS OF THE SAME CENSUS BLOCK ARE NOT HERE, because they are already computed and
// already verified elsewhere, and a second implementation of a verified column is a second answer
// that can drift:
//
//   TopoPSA, TPSA, SLogP, PEOE_VSA11, SMR_VSA1, EState_VSA1   -> src/hume_core/vsa_bins.h
//
// `SLogP` is mordred's five-line wrapper over `Crippen.MolLogP`, so it is the SAME NUMBER as
// vsa_bins.h's `MolLogP` and is emitted by aliasing, not by computing.  The three `MoeType`
// columns resolve through `getattr(rdkit.Chem.MolSurf | EState_VSA, name)` to the same functions
// and the same bin edges as the `rdkit_*` VSA columns.  See the wiring note at the bottom.
//
// ===========================================================================================
// HOUSE RULE 1, RUN FIRST AND RUN WITH BONDS SHUFFLED: NONE OF THESE 49 COLUMNS IS ILL-POSED.
// ===========================================================================================
//
// cpp/screen_constit.py, 2,000 molecules of cpp/hard.smi, six perturbations each -- two atom
// renumberings, three atom-renumbering-AND-bond-list-shuffles (the molecule rebuilt from scratch
// with its bonds added in a random order), and a Kekule round trip (re-parsed from
// `MolToSmiles(kekuleSmiles=True)`).  All 6,000 rebuilds and all 2,000 Kekule round trips
// reproduced the molecule, so every axis has full coverage; the run is in the file's docstring.
// Result:
//
//     ILL-POSED COLUMNS: 0 of 49
//
// with ONE MEASURED EXCEPTION FOUND LATER AND AT 100,000 SCALE, recorded here rather than left
// to the screen's sample size: `Vabc` reads the RING COUNT, and rdkit's `symmetrizeSSSR` is the
// object PORT_STATUS documents as unstable on ~0.03% of the corpus.  On 34 of 100,000 molecules
// mordred's own Vabc gives two or three different answers under atom+bond shuffling.  None of
// those 34 is in the screen's first 2,000, which is why the screen shows a clean 0 and is not
// wrong to.  See the divergence note on `vabc()` below.
//
// The largest movement any column showed on any perturbation is 1.4e-13 RELATIVE, on SLogP, and
// every column that moved at all is a SUM OVER ATOMS IN ATOM ORDER -- TPSA, SLogP, TopoPSA, bpol,
// Vabc, the VSA columns, RNCG/RPCG, MDEC.  That is float64 addition not being associative, not
// the descriptor failing to be a function of the molecule.  Every integer-valued column showed
// literally zero movement, including the two that read the KEKULE structure and were the ones
// worth watching: `nBondsKD` counts double bonds after `Chem.Kekulize`, and `CarbonTypes` sets
// `kekulize = True`.  A Kekule structure's DOUBLE-BOND COUNT is fixed by the atom set even when
// the matching is not, and that is now measured rather than argued.
//
// (`BertzCT` is named in the same census block and is the other column that would have been worth
// watching -- it does not appear here because it does not survive data/dedupe.json.  The block's
// `rdkit_composite` pair is `qed` and `SPS`.)
//
// TWO BUGS IN THE SCREEN ITSELF WERE FOUND BEFORE THE SCREEN COULD BE BELIEVED, and both would
// have produced a confident false positive:
//
//   * the first version compared values with `==`, and reported 15 ill-posed columns.  Thirteen
//     of them were summation order.  A screen for ill-posedness must measure the SIZE of the
//     movement; the two populations here are ten orders of magnitude apart.
//   * the guard on the atom+bond rebuild was ISOMERIC for every column, so any rebuild that lost
//     E/Z was rejected and FELL BACK TO ATOM-ONLY renumbering -- 882 of 6,000, concentrated
//     exactly on the stereo-rich molecules.  Atom-only is the axis PORT_STATUS says is too weak,
//     so those 882 were being screened with the weak probe under an "atom+bond" label.  It had
//     consequences: three molecules whose mordred `Vabc` looked STABLE over 400 perturbations
//     under that guard give three different answers each once the stereo-blind guard lets the
//     bond shuffle actually happen.  Only `SPS` reads stereo, so the guard is now per column and
//     the other 48 keep every rebuild.
//   * the Kekule round trip wrote `MolToSmiles(..., isomericSmiles=False)` and then compared two
//     isomeric-False smiles, so a round trip that DROPPED THE ISOTOPE LABELS passed its own
//     guard.  It reported FilterItLogS, qed and SPS as ill-posed on ~25 molecules each and every
//     discriminating example contained a `[13C]` or a `[125I]`: the screen had replaced the
//     molecule and then blamed the descriptor.  Likewise `verify_ic.rebuilt`'s equality check is
//     deliberately stereo-blind (no InformationContent column reads stereo) -- SPS DOES read
//     stereo, so this screen tightens that guard to isomeric smiles rather than reusing it as is.
//
// ===========================================================================================
// THE SPECIFICATION IS THE SOURCE CODE, and in this block the documentation is actively wrong.
// ===========================================================================================
//
// Every column below was read off mordred 1.2.0's and rdkit 2025.09.2's actual code path, never
// off a docstring.  The traps that were live here:
//
//   * `rdkit/Chem/Lipinski.py` builds `HDonorSmarts` / `NHOHSmarts` / `RotatableBondSmarts` at
//     import and THEN NEVER USES THEM -- `NumHDonors`, `NumHAcceptors`, `NOCount`, `NHOHCount`
//     and `NumRotatableBonds` are all one-line lambdas onto `rdMolDescriptors.Calc*`.  This file
//     needs `CalcNumHBD` / `CalcNumHBA` / `CalcNumRotatableBonds` and takes them from
//     src/hume_core/frag_matcher.h, which is built from RDKit's C++ parse trees.
//   * mordred computes each descriptor on ITS OWN molecule variant, and a descriptor and its
//     DEPENDENCY can disagree about which.  `Framework` has no `explicit_hydrogens` override, so
//     it inherits the base class's `True` and runs on `Chem.AddHs(mol)` -- while its `Rings()`
//     dependency is a `RingCountBase` with `explicit_hydrogens = False` and runs on
//     `Chem.RemoveHs(mol)`.  So fMF's denominator counts hydrogens and its ring sets do not.
//     Reading `explicit_hydrogens` off the wrong class puts fMF out by 40-60% on every molecule.
//     `GhoseFilter` is the same trap: its `self.mol.GetNumAtoms()` is the H-ADDED count.
//   * mordred's `KappaShapeIndex` is NOT rdkit's `Kappa1-3`.  It has no HallKierAlpha correction
//     at all, and its `P` is `len(ChiCache(order).path)`.  src/hume_core/hume_blocks.h's
//     `Kappa1-3` are the rdkit ones and are a different four columns of the census.
//   * mordred's `CPSA` pair that survives dedupe is `RNCG`/`RPCG` only.  Neither touches surface
//     area -- they are Qmax/sum(Q) over Gasteiger charges.  The CPSA members that DO need a SASA
//     (PNSA, PPSA, ...) did not survive, so no solvent-accessible surface is computed here.
//
// ===========================================================================================
// WHAT THE CALLER MUST SUPPLY
// ===========================================================================================
//
// THE MOLECULE IS THE HYDROGEN-SUPPRESSED ONE, `Chem.RemoveHs(mol)`, exactly as for ringcount /
// topocharge / pathcount -- src/hume/_extract.py's contract.  Columns that mordred computes on
// `Chem.AddHs(mol)` are reconstructed from it; see `HDerived` below for the four reconstructions
// and the evidence for each.
//
// Fields read from the boundary, all present today:
//   atom_i (n_atoms, 10)  Z, deg, nH, fchg, hyb, arom, ring, cip, nring, tval
//   atom_d (n_atoms, 2)   mass, gasteiger        -- only `mass` is read here
//   bond_i (n_bonds, 5)   u, v, conjugated, in-ring, SMARTS bond code
//   bond_d (n_bonds,)     GetBondTypeAsDouble()
//   rings                 the ring SET as ring_ptr / ring_at
//
// Values computed by OTHER verified headers and passed in rather than recomputed (`Inputs`):
//   molLogP, molMR        vsa_bins.h    -- SLogP/SMR, for Lipinski and GhoseFilter
//   nHBDon, nHBAcc, nRot  frag_matcher.h -- CalcNumHBD / CalcNumHBA / CalcNumRotatableBonds
//   naRing, nARing        ringcount.h   -- the aromatic and aliphatic ring counts Vabc subtracts
//   hchg                  the H-ADDED molecule's Gasteiger charges, which autocorr.h's boundary
//                         already carries as a second pickled molecule (see PORT_STATUS: the
//                         H-graph charges are NOT derivable from the heavy-atom pickle -- 5,221
//                         of 42,359 heavy atoms get a different `_GasteigerCharge` from
//                         `AddHs(m)` than from `m`).  mordred's `c` atomic property is
//                         `_GasteigerCharge + _GasteigerHCharge`; on the H-ADDED molecule the
//                         second term is 0.0 on every atom of 500 molecules, so `hchg` is the
//                         plain charge array in AddHs atom order.
//
// TWO INPUTS THAT ARE NOT AT THE BOUNDARY YET, and the two columns that wait on them:
//   qedAlerts             the count of rdkit QED's 116 structural-alert SMARTS that match.  The
//                         other seven QED properties are computed here exactly; see `qedScore`.
//   stereoAtom / stereoBond   rdkit's POTENTIAL stereo perception, which `SPS` reads and the
//                         boundary's assigned-only `cip` / `bond_s` columns cannot answer.
// Both are described in the wiring note at the bottom of this file.  Everything downstream of
// them is implemented and verified here, so each is one boundary column away, not a port.
#ifndef HUME_CONSTIT_H
#define HUME_CONSTIT_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../cpp/constit_tables.h"
#include "topocharge.h"   // for topocharge::pairwise_sum -- numpy's summation ORDER, which
                          // RNCG/RPCG need because mordred reduces their charges with np.sum.
                          // Reusing the existing transcription rather than making a third copy
                          // (infocontent.h has the other one).

namespace constit {

static constexpr int N_COLS = 43;

// Column names in `out[]` order.  cpp/verify_constit.py reads these so the two sides cannot
// disagree about which column is which.
inline const char* col_name(int i) {
  static const char* N[N_COLS] = {
      "C1SP1", "C2SP1", "C1SP2", "C2SP2", "C3SP2", "C1SP3", "C2SP3", "C3SP3", "C4SP3",
      "nH", "nB", "nC", "nN", "nO", "nS", "nCl", "nBr",
      "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM", "nBondsKD",
      "Kier1", "Kier2", "Kier3",
      "MDEC-22", "MDEC-23", "MDEC-33",
      "RNCG", "RPCG",
      "Lipinski", "GhoseFilter",
      "nAcid", "nBase",
      "Vabc", "RotRatio", "bpol", "FilterItLogS", "fMF", "fragCpx",
      "qed", "SPS"};
  return N[i];
}

enum : int {
  C_C1SP1 = 0, C_NH = 9, C_NBONDSS = 17, C_KIER1 = 23, C_MDEC22 = 26, C_RNCG = 29,
  C_LIPINSKI = 31, C_NACID = 33, C_VABC = 35, C_ROTRATIO = 36, C_BPOL = 37, C_LOGS = 38,
  C_FMF = 39, C_FRAGCPX = 40, C_QED = 41, C_SPS = 42
};

// RDKit's Atom::HybridizationType enum, as it crosses the boundary in `atom_i`'s `hyb` column.
// hume_blocks.h's HallKierAlpha indexes a per-element table by (hyb - 2) with SP at index 0, so
// these are the same numbers that file already depends on.  cpp/verify_constit.py asserts them
// against the running rdkit rather than trusting this comment.
enum : int { HYB_UNSPEC = 0, HYB_S = 1, HYB_SP = 2, HYB_SP2 = 3, HYB_SP3 = 4,
             HYB_SP2D = 5, HYB_SP3D = 6, HYB_SP3D2 = 7, HYB_OTHER = 8 };

// The SMARTS bond-code bits src/hume/_extract.py packs into bond_i's fifth column.  A bit for the
// bond ORDER when it is one of the three SMARTS can name, and a SEPARATE bit for the aromatic
// FLAG -- the two are independent questions.
enum : int { BC_SINGLE = 1, BC_DOUBLE = 2, BC_TRIPLE = 4, BC_AROM = 8 };

// "THE BOND TYPE IS AROMATIC" IS NOT "THE BOND CARRIES THE AROMATIC FLAG", AND cpp/hard.smi
// CONTAINS BOTH.  The boundary's code packs a bit for the ORDER when it is SINGLE, DOUBLE or
// TRIPLE and a separate bit for the FLAG, so a bond of type AROMATIC is the one with the flag set
// and NO order bit.  `Cc1c#cc#cc(C)cccc(=O)c#cc(=O)ccc1` (row 52519 of cpp/hard.smi) has TRIPLE
// bonds carrying the aromatic flag -- src/hume/_extract.py's docstring records the same thing for
// cpp/mols.smi -- and reading the flag as the type there made the nBondsKD reconstruction below
// count a triple bond as contributing 1 to its atoms' valence instead of 3, which threw.  Two of
// the 100,000 molecules discriminate the two readings; a 20,000-molecule sample showed neither.
//
// Measured over all 3,090,892 bonds of cpp/hard.smi: `flag set && type != AROMATIC` on 4 bonds,
// every one of them TRIPLE, in those 2 molecules.  The converse, `type == AROMATIC && flag
// clear`, is 0 -- which is what makes "flag set and no order bit" an exact test for the type,
// since an AROMATIC-type bond with the flag clear would be indistinguishable from DATIVE here.
static inline bool isAromType(int bcode) {
  return (bcode & BC_AROM) != 0 && (bcode & (BC_SINGLE | BC_DOUBLE | BC_TRIPLE)) == 0;
}

// THE BOND WRITTEN BETWEEN TWO SMARTS ATOMS WHEN NOTHING IS WRITTEN is rdkit's
// `SingleOrAromaticBond` -- a THIRD query, distinct from `-` (bond order SINGLE) and from `:`
// (bond type AROMATIC).  Every pattern in this file that has an unadorned bond uses this, and
// every pattern that writes `-` or `=` uses the order bits directly.  frag_matcher.h and
// estate_tables.h record the same trap for their own pattern sets; getting it wrong changes
// QED's AROM term on 62 of 3,000 molecules of cpp/hard.smi.
static inline bool singleOrArom(int bcode) {
  return (bcode & BC_SINGLE) != 0 || isAromType(bcode);
}

// RDKit fills the distance matrix with this for a pair of atoms in different fragments.  It is
// not a sentinel that mordred checks for: MolecularDistanceEdge multiplies it into a product,
// which is exactly how a multi-fragment molecule overflows a float64 to +inf and returns 0.0.
static const double DISCONNECTED = 1e8;

// FLOATING-POINT CONTRACTION IS THE REASON SEVERAL EXPRESSIONS BELOW ARE WRITTEN IN PIECES.
// The repo builds with plain -O3, and clang's default `-ffp-contract=on` fuses a multiply and an
// adjacent add WITHIN ONE EXPRESSION into an FMA -- which is more accurate and therefore a
// DIFFERENT number from the one python computed with two rounded operations.  Measured: `Vabc`
// then differs from mordred on 43 of 300 molecules and `FilterItLogS` on 74, both by one or two
// ulps, and `qed` on 240 of 300.  Every `a + b * c` in this file is therefore split across
// statements, which clang's `on` (as opposed to `fast`) will not contract across.  Do not
// "simplify" them back into one line, and do not compile this file with -ffp-contract=fast.
// hume_blocks.h documents the opposite case, where -ffp-contract=off makes things worse; the
// rule is to match the reference, not to maximise accuracy.

static inline double qnan() { return std::numeric_limits<double>::quiet_NaN(); }

// ---------------------------------------------------------------------------------------------
// The molecule, in the boundary's layout.
// ---------------------------------------------------------------------------------------------
struct Mol {
  int n = 0, nb = 0, nr = 0;
  std::vector<int32_t> z, deg, nh, fchg, hyb, arom, ring, nring, tval;
  std::vector<double> mass;
  std::vector<int32_t> bu, bv, bcode;
  std::vector<double> bord;
  std::vector<int32_t> ring_ptr, ring_at;      // ring r is ring_at[ring_ptr[r] .. ring_ptr[r+1])
  std::vector<int32_t> start, nbr, nbond;      // CSR adjacency + parallel bond index
  std::vector<int32_t> hcount, tdeg;           // SMARTS H and X, derived exactly as in
                                               // frag_matcher.h (verified there on 575,571 atoms)

  void alloc(int na, int nbonds) {
    n = na; nb = nbonds;
    z.assign(na, 0); deg.assign(na, 0); nh.assign(na, 0); fchg.assign(na, 0);
    hyb.assign(na, 0); arom.assign(na, 0); ring.assign(na, 0); nring.assign(na, 0);
    tval.assign(na, 0); mass.assign(na, 0.0);
    bu.assign(nbonds, 0); bv.assign(nbonds, 0); bcode.assign(nbonds, 0);
    bord.assign(nbonds, 0.0);
    ring_ptr.assign(1, 0); ring_at.clear(); nr = 0;
  }

  void finish() {
    std::vector<int32_t> cnt(n + 1, 0);
    for (int e = 0; e < nb; ++e) { cnt[bu[e]]++; cnt[bv[e]]++; }
    start.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) start[i + 1] = start[i] + cnt[i];
    nbr.assign(start[n], 0); nbond.assign(start[n], 0);
    std::vector<int32_t> fill(start.begin(), start.end() - 1);
    for (int e = 0; e < nb; ++e) {
      nbr[fill[bu[e]]] = bv[e]; nbond[fill[bu[e]]++] = e;
      nbr[fill[bv[e]]] = bu[e]; nbond[fill[bv[e]]++] = e;
    }
    hcount.assign(n, 0); tdeg.assign(n, 0);
    for (int i = 0; i < n; ++i) {
      int h = nh[i];
      for (int k = start[i]; k < start[i + 1]; ++k) if (z[nbr[k]] == 1) ++h;
      hcount[i] = h;                 // GetTotalNumHs(true)   -- SMARTS `H`
      tdeg[i] = deg[i] + nh[i];      // GetTotalDegree()      -- SMARTS `X`
    }
    nr = (int)ring_ptr.size() - 1;
  }

  // Fill from the boundary's strided rows.  `astride`/`bstride` are arguments so that a boundary
  // that grows a column does not silently shift every field by one.
  void build_from_rows(int na, const int32_t* ai, int astride, const double* ad, int adstride,
                       int nbonds, const int32_t* bi, int bstride, const double* bd,
                       int nrings, const int32_t* rptr, const int32_t* rat) {
    alloc(na, nbonds);
    for (int i = 0; i < na; ++i) {
      const int32_t* r = ai + (size_t)i * astride;
      z[i] = r[0]; deg[i] = r[1]; nh[i] = r[2]; fchg[i] = r[3]; hyb[i] = r[4];
      arom[i] = r[5]; ring[i] = r[6]; nring[i] = r[8]; tval[i] = r[9];
      mass[i] = ad[(size_t)i * adstride + 0];
    }
    for (int e = 0; e < nbonds; ++e) {
      const int32_t* r = bi + (size_t)e * bstride;
      bu[e] = r[0]; bv[e] = r[1]; bcode[e] = r[4];
      bord[e] = bd[e];
    }
    ring_ptr.assign(rptr, rptr + nrings + 1);
    ring_at.assign(rat, rat + rptr[nrings]);
    finish();
  }

  int bondBetween(int a, int b) const {
    for (int k = start[a]; k < start[a + 1]; ++k) if (nbr[k] == b) return nbond[k];
    return -1;
  }
};

// ---------------------------------------------------------------------------------------------
// Values other verified headers own.  Passed in, never recomputed here.
// ---------------------------------------------------------------------------------------------
struct Inputs {
  double molLogP = 0.0, molMR = 0.0;        // vsa_bins.h
  int nHBDon = 0, nHBAcc = 0, nRot = 0;     // frag_matcher.h
  double naRing = 0.0, nARing = 0.0;        // ringcount.h

  // The H-ADDED molecule's Gasteiger charges, in Chem.AddHs atom order, length n + sum(nh).
  // Null means "no charges" -- RNCG/RPCG then follow mordred, which returns 0.0 when the mask
  // selects nothing rather than raising.
  const double* hchg = 0;
  int nhchg = 0;

  // NOT AT THE BOUNDARY YET.  See the wiring note at the bottom of this file.
  int qedAlerts = -1;                       // <0 -> qed is NaN
  const int32_t* stereoAtom = 0;            // per heavy atom, rdkit FindMolChiralCenters
                                            //   (includeUnassigned=True, legacy=False)
  const int32_t* stereoBond = 0;            // per bond, GetStereo() != STEREONONE after
                                            //   rdmolops.FindPotentialStereoBonds
};

// ---------------------------------------------------------------------------------------------
// The four quantities of the H-ADDED molecule that ARE exact functions of the heavy-atom
// boundary, with the measurement behind each.  Everything mordred computes with
// `explicit_hydrogens = True` is built from these rather than from a second molecule.
//
//   nHadd   the hydrogens Chem.AddHs would create   = sum of GetTotalNumHs(false)
//   nHtot   mordred's `nH` column                   = nHadd + #{atoms with Z == 1}
//           The second term is not pedantry: `Chem.RemoveHs` KEEPS isotopic hydrogen, so
//           cpp/hard.smi's `[2H]` atoms are in the heavy graph AND are hydrogens in the AddHs
//           graph.  Verified 0 mismatches on 4,000 molecules against
//           `sum(a.GetSymbol()=="H" for a in Chem.AddHs(m).GetAtoms())`.
//   nAtomsH  GetNumAtoms() of the AddHs molecule    = n + nHadd
//   nBondsH  GetNumBonds() of the AddHs molecule    = nb + nHadd  (every added H is terminal)
// ---------------------------------------------------------------------------------------------
struct HDerived {
  int nHadd = 0, nHtot = 0, nAtomsH = 0, nBondsH = 0;
  explicit HDerived(const Mol& m) {
    int nh1 = 0;
    for (int i = 0; i < m.n; ++i) { nHadd += m.nh[i]; if (m.z[i] == 1) ++nh1; }
    nHtot = nHadd + nh1;
    nAtomsH = m.n + nHadd;
    nBondsH = m.nb + nHadd;
  }
};

// ---------------------------------------------------------------------------------------------
// The drift guard.  Reproduces cpp/gen_constit_tables.py:canonical() byte for byte and hashes it,
// so a hand-edited number in the generated header is caught here with neither rdkit nor mordred
// in the process.  Hashing the SPEC and not the file, per house rule 6.
// ---------------------------------------------------------------------------------------------
namespace detail {

// sha256, just enough of it to hash a few kilobytes.  Self-contained on purpose: a drift guard
// that needs a library to run is a drift guard that gets switched off.
struct Sha256 {
  uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t nbuf;
  Sha256() : len(0), nbuf(0) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::memcpy(h, iv, sizeof(h));
  }
  static uint32_t ror(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,
      0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,
      0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,
      0x4a7484aau,0x5cb0a9dcu,0x76f988dau,0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
      0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,
      0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
      0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,0x19a4c116u,
      0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,
      0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
             ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
      uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  void update(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data; len += n;
    while (n) {
      size_t k = 64 - nbuf; if (k > n) k = n;
      std::memcpy(buf + nbuf, p, k); nbuf += k; p += k; n -= k;
      if (nbuf == 64) { block(buf); nbuf = 0; }
    }
  }
  std::string hex() {
    uint64_t bits = len * 8;
    uint8_t pad = 0x80; update(&pad, 1);
    uint8_t zero = 0;
    while (nbuf != 56) update(&zero, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; ++i) be[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(be, 8);
    static const char* HX = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 8; ++i)
      for (int j = 3; j >= 0; --j) {
        uint8_t byte = (uint8_t)(h[i] >> (8 * j));
        s += HX[byte >> 4]; s += HX[byte & 15];
      }
    return s;
  }
};

inline void appendBits(std::string& s, const char* name, int idx, double v) {
  uint64_t u; std::memcpy(&u, &v, 8);
  char b[128];
  std::snprintf(b, sizeof(b), "%s[%d]=0x%016llx\n", name, idx, (unsigned long long)u);
  s += b;
}

// The exact byte string cpp/gen_constit_tables.py hashes.  The tables and their ORDER are part of
// it; reordering the list is a spec change, not a refactor.
inline std::string specString() {
  using namespace constit_tbl;
  std::string s;
  for (int i = 0; i <= MAX_Z; ++i) appendBits(s, "POL94", i, POL94[i]);
  for (int i = 0; i <= MAX_Z; ++i) appendBits(s, "BONDI", i, BONDI[i]);
  for (int i = 0; i <= MAX_Z; ++i) appendBits(s, "BONDI_OK", i, (double)BONDI_OK[i]);
  for (int i = 0; i <= MAX_Z; ++i) appendBits(s, "MONOISO", i, MONOISO[i]);
  for (int i = 0; i <= MAX_Z; ++i) appendBits(s, "AWEIGHT", i, AWEIGHT[i]);
  appendBits(s, "ELECTRON_MASS", 0, ELECTRON_MASS);
  for (int i = 0; i < N_LOGS; ++i) appendBits(s, "LOGS_COEF", i, LOGS_COEF[i]);
  appendBits(s, "LOGS_CONST", 0, LOGS_A);
  appendBits(s, "LOGS_CONST", 1, LOGS_B);
  return s;
}

}  // namespace detail

// Throws unless cpp/constit_tables.h still holds the numbers it was generated with.  Cheap enough
// to call once per process; `compute()` does not call it, so a batch loop pays nothing.
inline void checkSpec() {
  detail::Sha256 sh;
  const std::string s = detail::specString();
  sh.update(s.data(), s.size());
  const std::string got = sh.hex();
  if (got != std::string(constit_tbl::SPEC_SHA256))
    throw std::runtime_error("constit: cpp/constit_tables.h spec hash is " + got +
                             ", generated as " + constit_tbl::SPEC_SHA256 +
                             " -- regenerate with cpp/gen_constit_tables.py under the pin");
}

// ---------------------------------------------------------------------------------------------
// CarbonTypes.  mordred sets kekulize = True, which changes bond TYPES and leaves hybridisation
// and the neighbour sets alone, so nothing here depends on which Kekule structure was chosen --
// confirmed by the screen, which moved this family on none of 12,000 perturbed molecules.
// mordred's `_hybridization` dict maps SP->1, SP2->2, and SP3/SP3D/SP3D2 all ->3; anything else
// (UNSPECIFIED, S, SP2D, OTHER) misses the dict, lands in a `None` bucket and is emitted by no
// column.  `carbon` is the number of neighbours that are carbon, in the HEAVY graph.
// ---------------------------------------------------------------------------------------------
inline void carbonTypes(const Mol& m, double* out) {
  int r[4][8];
  std::memset(r, 0, sizeof(r));
  for (int i = 0; i < m.n; ++i) {
    if (m.z[i] != 6) continue;
    int sp;
    switch (m.hyb[i]) {
      case HYB_SP:  sp = 1; break;
      case HYB_SP2: sp = 2; break;
      case HYB_SP3: case HYB_SP3D: case HYB_SP3D2: sp = 3; break;
      default: continue;                       // mordred's None bucket
    }
    int c = 0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) if (m.z[m.nbr[k]] == 6) ++c;
    if (c < 8) r[sp][c]++;
  }
  // mordred's preset order: (nCarbon, SP) = (1,1)(2,1)(1,2)(2,2)(3,2)(1,3)(2,3)(3,3)(4,3)
  static const int NC[9] = {1, 2, 1, 2, 3, 1, 2, 3, 4};
  static const int SP[9] = {1, 1, 2, 2, 2, 3, 3, 3, 3};
  for (int c = 0; c < 9; ++c) out[c] = (double)r[SP[c]][NC[c]];
}

// ---------------------------------------------------------------------------------------------
// AtomCount.  `nH` is the only member with explicit_hydrogens = True; the rest count symbols in
// the heavy graph.  mordred compares `a.GetSymbol() == type`, which is a function of the atomic
// number alone -- so a `[2H]` counts as an H and NOT as a heavy atom of some other element.
// ---------------------------------------------------------------------------------------------
inline void atomCount(const Mol& m, const HDerived& H, double* out) {
  static const int ZS[7] = {5, 6, 7, 8, 16, 17, 35};       // B C N O S Cl Br
  int c[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < m.n; ++i)
    for (int k = 0; k < 7; ++k) if (m.z[i] == ZS[k]) { ++c[k]; break; }
  out[0] = (double)H.nHtot;
  for (int k = 0; k < 7; ++k) out[1 + k] = (double)c[k];
}

// ---------------------------------------------------------------------------------------------
// BondCount.  Six survivors, and they do not all read the same molecule:
//
//   nBondsS   SINGLE, non-kekulized, EXPLICIT HYDROGENS (mordred sets explicit_hydrogens = True
//             for `any` and `single` only).  Every hydrogen Chem.AddHs creates brings exactly one
//             single bond, so this is the heavy count plus nHadd.
//   nBondsD   DOUBLE, non-kekulized, heavy.
//   nBondsT   TRIPLE, non-kekulized, heavy.
//   nBondsA   `b.GetIsAromatic() or b.GetBondType() == AROMATIC`, heavy.
//   nBondsM   `b.GetIsAromatic() or b.GetBondType() != SINGLE`, heavy.
//   nBondsKD  DOUBLE after `Chem.Kekulize(m)`, heavy.
//
// nBondsA IS AN `or` OVER TWO GENUINELY DIFFERENT QUESTIONS -- `GetIsAromatic()` and
// `getBondType() == AROMATIC` -- and cpp/hard.smi contains 4 bonds where they disagree.  Because
// it is an `or`, and because the disagreeing bonds all have the FLAG set, the flag alone answers
// it; the `or` is written out below anyway so that the day a corpus contains the other direction
// it is caught rather than quietly averaged over.  nBondsKD is where the distinction bites, and
// there it is not cosmetic: see `isAromType`.
//
// nBondsKD IS THE ONE COLUMN THAT NEEDS THE KEKULE STRUCTURE, AND THE BOUNDARY DOES NOT CARRY IT.
// It is reconstructed, not re-kekulized.  Kekulization rewrites only AROMATIC-type bonds, and it
// preserves every atom's valence, so for an atom carrying aromatic bonds:
//
//     (bond orders after kekulization) = (non-aromatic valence contributions) + (aromatic bonds)
//                                        + (1 if this atom takes a ring double bond)
//
// which rearranges to a per-atom flag readable straight off the boundary:
//
//     takesDouble(i) = tval[i] - nh[i] - round(nonAromaticValenceContrib(i)) - nAromaticBonds(i)
//
// and nBondsKD = (DOUBLE bonds already) + sum(takesDouble)/2.  Verified against
// `Chem.Kekulize(m)` on 4,000 molecules: 0 mismatches, 0 odd sums, and takesDouble was never
// outside {0,1}.  THE VALENCE CONTRIBUTION IS NOT `GetBondTypeAsDouble()`: a DATIVE bond has
// order 1.0 but contributes 0.0 to the DONOR and 1.0 to the ACCEPTOR, and the one molecule that
// discriminates the two readings on 4,000 is `CCN1CCCc2c(OCCCN3CCCCC3)ccc[c]21->[SnH3]`, which
// comes out 2 instead of 3 if the dative bond is counted for its donor.  `Bond::getValenceContrib`
// is reproduced by `valenceContrib()` below, whose table was read out of the running rdkit.
// ---------------------------------------------------------------------------------------------

// Bond::getValenceContrib(atom), measured over every bond of 4,000 molecules of cpp/hard.smi:
//   SINGLE 1.0/1.0   DOUBLE 2.0/2.0   TRIPLE 3.0/3.0   AROMATIC 1.5/1.5   DATIVE 0.0(begin)/1.0
// The boundary's bond code has no bit for DATIVE -- a dative bond is simply "none of SINGLE,
// DOUBLE, TRIPLE, aromatic-flagged", i.e. code == 0 -- which is exactly the test used here.
static inline double valenceContrib(const Mol& m, int e, int atom) {
  const int c = m.bcode[e];
  if (c == 0) return (m.bu[e] == atom) ? 0.0 : m.bord[e];   // dative: donor gets nothing
  return m.bord[e];
}

inline void bondCount(const Mol& m, const HDerived& H, double* out) {
  int nS = 0, nD = 0, nT = 0, nA = 0, nM = 0;
  for (int e = 0; e < m.nb; ++e) {
    const int c = m.bcode[e];
    const bool isSingle = (c & BC_SINGLE) != 0;
    const bool flagArom = (c & BC_AROM) != 0;
    const bool typeArom = isAromType(c);
    if (isSingle) ++nS;
    if (c & BC_DOUBLE) ++nD;
    if (c & BC_TRIPLE) ++nT;
    if (flagArom || typeArom) ++nA;
    if (flagArom || !isSingle) ++nM;
  }
  int takes = 0;
  for (int i = 0; i < m.n; ++i) {
    int narom = 0; double nonarom = 0.0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
      const int e = m.nbond[k];
      if (isAromType(m.bcode[e])) ++narom;         // TYPE aromatic: this is what Kekulize rewrites
      else nonarom += valenceContrib(m, e, i);
    }
    if (!narom) continue;
    const int g = m.tval[i] - m.nh[i] - (int)std::floor(nonarom + 0.5) - narom;
    if (g < 0 || g > 1)
      throw std::runtime_error("constit: nBondsKD takesDouble out of {0,1} -- the kekule "
                               "reconstruction does not hold for this molecule");
    takes += g;
  }
  if (takes % 2)
    throw std::runtime_error("constit: nBondsKD takesDouble sum is odd -- no perfect matching");
  out[0] = (double)(nS + H.nHadd);      // nBondsS, explicit hydrogens
  out[1] = (double)nD;
  out[2] = (double)nT;
  out[3] = (double)nA;
  out[4] = (double)nM;
  out[5] = (double)(nD + takes / 2);    // nBondsKD
}

// ---------------------------------------------------------------------------------------------
// KappaShapeIndex.  NOT rdkit's Kappa: no HallKierAlpha, and P is the number of `path`-type
// connected edge subgraphs of `order` bonds, from mordred's ChiCache.  A ChiCache subgraph is
// typed `path` exactly when its DFS finds no cycle and every degree is 1 or 2 -- i.e. when it is
// a SIMPLE PATH.  So P_k is the count of simple paths of k bonds, and mordred's own PathCount
// (MPC_k) counts the same objects: verified equal on 4,500 (molecule, order) pairs.
//
// TWO ASYMMETRIES OF MORDRED'S THAT ARE REPRODUCED, NOT FIXED.  `Chem.FindAllSubgraphsOfLengthN`
// takes `useHs=False` by default and mordred never passes it, so the isotopic hydrogens that
// `Chem.RemoveHs` leaves behind are INVISIBLE to P -- while `A = self.mol.GetNumAtoms()` two
// lines later COUNTS them.  This is the same trap src/hume_core/pathcount.h documents, and the
// same molecules discriminate it.
//
// The arithmetic is written in mordred's association order (`2 * Pmax * Pmin / (P * P)`) because
// the result is a float and reassociating it changes the last bits.
// ---------------------------------------------------------------------------------------------

// Simple paths of exactly k bonds, k in 1..3, hydrogen atoms excluded from the graph.  Written as
// an explicit walk rather than a closed form: the closed form for k=3 needs a triangle count and
// a correction term, and this is called three times per molecule.
inline int simplePaths(const Mol& m, int k) {
  int total = 0;
  for (int a = 0; a < m.n; ++a) {
    if (m.z[a] == 1) continue;
    if (k == 1) {
      for (int p = m.start[a]; p < m.start[a + 1]; ++p)
        if (m.z[m.nbr[p]] != 1 && m.nbr[p] > a) ++total;
      continue;
    }
    for (int p = m.start[a]; p < m.start[a + 1]; ++p) {
      const int b = m.nbr[p];
      if (m.z[b] == 1) continue;
      if (k == 2) {
        for (int q = m.start[b]; q < m.start[b + 1]; ++q) {
          const int c = m.nbr[q];
          if (m.z[c] == 1 || c == a) continue;
          if (c > a) ++total;                       // each path once, by its endpoints
        }
        continue;
      }
      for (int q = m.start[b]; q < m.start[b + 1]; ++q) {
        const int c = m.nbr[q];
        if (m.z[c] == 1 || c == a) continue;
        for (int r = m.start[c]; r < m.start[c + 1]; ++r) {
          const int d = m.nbr[r];
          if (m.z[d] == 1 || d == b || d == a) continue;
          if (d > a) ++total;
        }
      }
    }
  }
  return total;
}

inline void kappaShape(const Mol& m, double* out) {
  const int A = m.n;                       // mordred's self.mol.GetNumAtoms(), isotopic H and all
  for (int k = 1; k <= 3; ++k) {
    const int P = simplePaths(m, k);
    if (P == 0) { out[k - 1] = qnan(); continue; }
    const double Pmin = (double)(A - k);
    double v;
    if (k == 1) {
      const double Pmax = 0.5 * (double)A * (double)(A - 1);
      v = 2 * Pmax * Pmin / (double)(P * P);
    } else if (k == 2) {
      const double Pmax = 0.5 * (double)(A - 1) * (double)(A - 2);
      v = 2 * Pmax * Pmin / (double)(P * P);
    } else {
      const double Pmax = (A % 2 == 0) ? 0.25 * (double)((A - 2) * (A - 2))
                                       : 0.25 * (double)(A - 1) * (double)(A - 3);
      v = 4 * Pmax * Pmin / (double)(P * P);
    }
    out[k - 1] = v;
  }
}

// ---------------------------------------------------------------------------------------------
// MolecularDistanceEdge.  MDEC-22, MDEC-23, MDEC-33: over pairs of CARBONS whose HEAVY DEGREES
// are (2,2), (2,3) and (3,3),
//
//     dx = product(topological distances) ** (1 / (2n));   MDE = n / dx**2
//
// mordred's `V` is `AdjacencyMatrix(...).sum(axis=0)`, i.e. the plain heavy-atom degree -- not a
// valence, despite the parameter's name.
//
// THE PRODUCT IS TAKEN IN FLOAT64 AND IT OVERFLOWS, AND THAT IS NOT A BUG TO BE FIXED.  For a
// multi-fragment molecule rdkit's distance matrix holds 1e8 between fragments, so a few dozen
// qualifying pairs push `numpy.product` straight to +inf; then dx is inf and `n / dx**2` is 0.0.
// mordred emits that 0.0 and numpy prints `RuntimeWarning: overflow encountered in reduce` while
// it does.  Computing this in log space would give a "better" answer and a different column, so
// the accumulation below is a plain left-to-right float64 product in mordred's own pair order
// (i ascending, then j > i) and is allowed to reach infinity.
//
// n == 0 makes mordred evaluate `1.0 ** (1.0 / 0.0)`, a ZeroDivisionError caught by
// `rethrow_zerodiv` and reported as a missing value -> NaN here.
// ---------------------------------------------------------------------------------------------
inline void distanceMatrix(const Mol& m, std::vector<double>& D) {
  const int n = m.n;
  D.assign((size_t)n * n, DISCONNECTED);
  std::vector<int> q(n);
  for (int s = 0; s < n; ++s) {
    double* row = &D[(size_t)s * n];
    row[s] = 0.0;
    int head = 0, tail = 0;
    q[tail++] = s;
    while (head < tail) {
      const int u = q[head++];
      for (int k = m.start[u]; k < m.start[u + 1]; ++k) {
        const int v = m.nbr[k];
        if (row[v] == DISCONNECTED && v != s) { row[v] = row[u] + 1.0; q[tail++] = v; }
      }
    }
  }
}

// `dx ** 2` IS NOT `dx * dx`, AND WRITING `std::pow(dx, 2.0)` IS NOT ENOUGH TO SAY SO.
//
// mordred returns `n / dx ** 2`.  Python's `**` on a float calls libm's correctly-rounded pow(),
// which rounds once from the exact product; `dx * dx` rounds the same exact product under a
// different rule and the two differ in the last bit.  On dx = 2.097773301070409 the results are
// 4.4006528226838411 and 4.4006528226838419, and that one ulp reaches the column: 53, 77 and 72
// of 100,000 molecules for MDEC-22, MDEC-23 and MDEC-33.
//
// AND THEN CLANG UNDOES THE FIX.  LLVM's libcall simplifier rewrites `pow(x, 2.0)` to `x * x` at
// -O2 unconditionally, so the obvious spelling compiles back into the wrong answer -- which is
// exactly what happened here, and the column stayed out by one ulp with `std::pow(dx, 2.0)`
// sitting in the source.  The `volatile` exponent below is what makes the call survive.  It costs
// three real pow() calls per molecule and it is the whole reason these three columns are exact.
static inline double squareViaPow(double x) {
  volatile double two = 2.0;
  return std::pow(x, two);
}

inline void molecularDistanceEdge(const Mol& m, const std::vector<double>& D, double* out) {
  static const int V1[3] = {2, 2, 3};
  static const int V2[3] = {2, 3, 3};
  for (int c = 0; c < 3; ++c) {
    const int v1 = V1[c], v2 = V2[c];
    double prod = 1.0;
    long long n = 0;
    for (int i = 0; i < m.n; ++i) {
      if (m.z[i] != 6) continue;
      for (int j = i + 1; j < m.n; ++j) {
        if (m.z[j] != 6) continue;
        const int di = m.deg[i], dj = m.deg[j];
        if (!((di == v1 && dj == v2) || (dj == v1 && di == v2))) continue;
        prod *= D[(size_t)i * m.n + j];
        ++n;
      }
    }
    if (n == 0) { out[c] = qnan(); continue; }
    const double dx = std::pow(prod, 1.0 / (2.0 * (double)n));
    out[c] = (double)n / squareViaPow(dx);
  }
}

// ---------------------------------------------------------------------------------------------
// CPSA: RNCG and RPCG.  Charges are the H-ADDED molecule's Gasteiger charges (mordred's CPSABase
// leaves explicit_hydrogens at the base class's True).  Both are
//
//     Qmax / sum(Q)   over the negative (RNCG) or positive (RPCG) charges,
//
// with Qmax the charge of largest ABSOLUTE value -- `np.argmax(np.abs(charges))`, which returns
// the FIRST maximum, so ties go to the lower atom index.  An empty mask returns 0.0, not NaN;
// that is mordred's `if len(charges) == 0: return 0.0` and not a repair.
//
// THE SUM IS numpy's, NOT A LOOP.  `np.sum` on a float64 array is PAIRWISE, and the compacted
// charge array is usually 10-60 long, which is numpy's 8-accumulator unrolled regime rather than
// its naive one.  topocharge::pairwise_sum is the existing transcription of
// numpy/core/src/umath/loops_utils.h.src.
// ---------------------------------------------------------------------------------------------
inline void chargeRatios(const Inputs& in, int nAtomsH, double* out) {
  if (!in.hchg || in.nhchg != nAtomsH) { out[0] = qnan(); out[1] = qnan(); return; }
  std::vector<double> sel;
  for (int sign = 0; sign < 2; ++sign) {
    sel.clear();
    for (int i = 0; i < in.nhchg; ++i) {
      const double q = in.hchg[i];
      if (sign == 0 ? (q < 0.0) : (q > 0.0)) sel.push_back(q);
    }
    if (sel.empty()) { out[sign] = 0.0; continue; }
    size_t best = 0;
    double bv = std::fabs(sel[0]);
    for (size_t i = 1; i < sel.size(); ++i) {
      const double a = std::fabs(sel[i]);
      if (a > bv) { bv = a; best = i; }        // strict >, so argmax keeps the first
    }
    out[sign] = sel[best] / topocharge::pairwise_sum(sel.data(), sel.size());
  }
}

// ---------------------------------------------------------------------------------------------
// Molecular weight, both flavours, in rdkit's own accumulation order.
//
//   MolWt        `_CalcMolWt`: sum of Atom::getMass() over the H-ADDED molecule.  getMass()
//                already returns the ISOTOPE's mass when one is set, so the boundary's `mass`
//                column is the whole answer for the heavy atoms and each added hydrogen adds
//                AWEIGHT[1] once.  Bit-exact on 4,000/4,000.
//   ExactMolWt   `CalcExactMolWt`: the monoisotopic mass per element, the isotope's own mass
//                when one is set, and ONE ELECTRON MASS SUBTRACTED PER UNIT OF FORMAL CHARGE
//                INSIDE the atom loop.  Getting any of those three details wrong leaves 78-3,083
//                of 4,000 molecules out by up to 1.6e-3; with all three it is bit-exact on
//                4,000/4,000.  See cpp/gen_constit_tables.py for how ELECTRON_MASS was recovered.
//
// An atom is isotope-labelled iff its mass differs from the element's standard atomic weight.
// There is no isotope column at the boundary and this test does not need one: rdkit's getMass()
// returns AWEIGHT[z] exactly when the isotope is unset, so the comparison is exact, not a
// tolerance.
// ---------------------------------------------------------------------------------------------
inline double molWt(const Mol& m, const HDerived& H) {
  double acc = 0.0;
  for (int i = 0; i < m.n; ++i) acc += m.mass[i];
  for (int k = 0; k < H.nHadd; ++k) acc += constit_tbl::AWEIGHT[1];
  return acc;
}

// `_CalcMolWt` OF THE HEAVY MOLECULE IS A DIFFERENT NUMBER FROM `_CalcMolWt` OF THE H-ADDED ONE,
// and both are wanted here.  rdkit's calcAMW walks the atoms it is given and adds, PER ATOM, the
// atom's mass and then `getTotalNumHs() * atomicWeight(1)`.  On the H-added molecule every atom
// contributes only its own mass, so the hydrogens arrive one at a time at the END; on the heavy
// molecule they arrive INTERLEAVED, in groups, as a multiply.  The two accumulations differ in
// the last bits on 3,924 of 4,000 molecules of cpp/hard.smi -- they are the same real number and
// not the same float64.
//
//   mordred's `Weight(exact=False)`, which FilterItLogS depends on, sets explicit_hydrogens=True
//   -> `molWt()` above, the H-ADDED accumulation.
//   rdkit's QED.properties() does `mol = Chem.RemoveHs(mol)` first
//   -> `heavyMolWt()` here, the interleaved one.
//
// Using either for both leaves `qed` out by up to 5e-15 relative on three quarters of the corpus,
// which is small, systematic, and would have been reported as "a tolerance" instead of as the
// wrong function.
inline double heavyMolWt(const Mol& m) {
  double acc = 0.0;
  for (int i = 0; i < m.n; ++i) {
    acc += m.mass[i];
    const double h = (double)m.nh[i] * constit_tbl::AWEIGHT[1];   // split: no FMA
    acc += h;
  }
  return acc;
}

inline double exactMolWt(const Mol& m, const HDerived& H) {
  double acc = 0.0;
  for (int i = 0; i < m.n; ++i) {
    const int z = m.z[i];
    const bool labelled = (m.mass[i] != constit_tbl::AWEIGHT[z]);
    acc += labelled ? m.mass[i] : constit_tbl::MONOISO[z];
    if (m.fchg[i]) {
      const double e = constit_tbl::ELECTRON_MASS * (double)m.fchg[i];   // split: no FMA
      acc -= e;
    }
  }
  for (int k = 0; k < H.nHadd; ++k) acc += constit_tbl::MONOISO[1];
  return acc;
}

// ---------------------------------------------------------------------------------------------
// AcidBase.  mordred fuses each family's alternatives into ONE single-atom query,
// `[$(a),$(b),...]`, so the count is the number of ATOMS matching at least one alternative --
// there is no embedding to uniquify and no search order to depend on.
//
//   nAcid  [O;H1]-[C,S,P]=O  |  [*;-;!$(*~[*;+])]  |  [NH](S(=O)=O)C(F)(F)F  |  n1nnnc1
//   nBase  [NH2]-[CX4]  |  [NH](-[CX4])-[CX4]  |  N(-[CX4])(-[CX4])-[CX4]
//          |  [*;+;!$(*~[*;-])]  |  N=C-N  |  N-C=N
//
// `-` and `+` in a bracket mean charge EXACTLY -1 and EXACTLY +1, not "any negative".
// Uppercase `N`/`C` are ALIPHATIC; `n` and `c` are aromatic.  `X4` is total degree.  `H1`/`H2` is
// the total hydrogen count including neighbours.
//
// MORDRED MATCHES THESE ON THE H-ADDED MOLECULE (SmartsCountBase leaves explicit_hydrogens at the
// base class's True) AND THIS FILE MATCHES ON THE HEAVY ONE.  That is a deliberate, measured
// equivalence and not an oversight: SMARTS `H` and `X` are already total counts, so the added
// hydrogens change no predicate, and the only construct that could see them is the bare `*` in
// the two charge alternatives -- which would need a charged hydrogen ATOM to differ.  Measured on
// 4,000 molecules of cpp/hard.smi: nAcid differs on 0, nBase differs on 0.
// ---------------------------------------------------------------------------------------------
inline void acidBase(const Mol& m, double* out) {
  int nAcid = 0, nBase = 0;
  for (int i = 0; i < m.n; ++i) {
    bool acid = false, base = false;

    // [O;H1]-[C,S,P]=O   -- aliphatic O with one H, single-bonded to C/S/P which doubles to an O
    if (!acid && m.z[i] == 8 && !m.arom[i] && m.hcount[i] == 1) {
      for (int k = m.start[i]; k < m.start[i + 1] && !acid; ++k) {
        const int e = m.nbond[k], j = m.nbr[k];
        if (!(m.bcode[e] & BC_SINGLE)) continue;
        if (m.arom[j] || (m.z[j] != 6 && m.z[j] != 16 && m.z[j] != 15)) continue;
        for (int q = m.start[j]; q < m.start[j + 1]; ++q) {
          const int e2 = m.nbond[q], o = m.nbr[q];
          if (o == i) continue;
          if ((m.bcode[e2] & BC_DOUBLE) && m.z[o] == 8 && !m.arom[o]) { acid = true; break; }
        }
      }
    }
    // [*;-;!$(*~[*;+])] / [*;+;!$(*~[*;-])]
    if (m.fchg[i] == -1 || m.fchg[i] == 1) {
      const int opp = -m.fchg[i];
      bool touching = false;
      for (int k = m.start[i]; k < m.start[i + 1]; ++k)
        if (m.fchg[m.nbr[k]] == opp) { touching = true; break; }
      if (!touching) { if (m.fchg[i] == -1) acid = true; else base = true; }
    }
    // [NH](S(=O)=O)C(F)(F)F
    if (!acid && m.z[i] == 7 && !m.arom[i] && m.hcount[i] == 1) {
      bool hasS = false, hasCF3 = false;
      for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
        const int j = m.nbr[k];
        if (!singleOrArom(m.bcode[m.nbond[k]])) continue;     // N-S and N-C are default bonds
        if (m.z[j] == 16 && !m.arom[j]) {
          int nO = 0;
          for (int q = m.start[j]; q < m.start[j + 1]; ++q)
            if ((m.bcode[m.nbond[q]] & BC_DOUBLE) && m.z[m.nbr[q]] == 8 && !m.arom[m.nbr[q]]) ++nO;
          if (nO >= 2) hasS = true;
        } else if (m.z[j] == 6 && !m.arom[j]) {
          int nF = 0;
          for (int q = m.start[j]; q < m.start[j + 1]; ++q)
            if (singleOrArom(m.bcode[m.nbond[q]]) && m.z[m.nbr[q]] == 9 && !m.arom[m.nbr[q]]) ++nF;
          if (nF >= 3) hasCF3 = true;
        }
      }
      if (hasS && hasCF3) acid = true;
    }
    // n1nnnc1 -- the root aromatic nitrogen of a tetrazole ring, walked outward n-n-n-c back to i
    if (!acid && m.z[i] == 7 && m.arom[i]) {
      for (int k1 = m.start[i]; k1 < m.start[i + 1] && !acid; ++k1) {
        const int b = m.nbr[k1];
        if (m.z[b] != 7 || !m.arom[b] || !singleOrArom(m.bcode[m.nbond[k1]])) continue;
        for (int k2 = m.start[b]; k2 < m.start[b + 1] && !acid; ++k2) {
          const int c = m.nbr[k2];
          if (c == i || m.z[c] != 7 || !m.arom[c]) continue;
          if (!singleOrArom(m.bcode[m.nbond[k2]])) continue;
          for (int k3 = m.start[c]; k3 < m.start[c + 1] && !acid; ++k3) {
            const int d = m.nbr[k3];
            if (d == b || d == i || m.z[d] != 7 || !m.arom[d]) continue;
            if (!singleOrArom(m.bcode[m.nbond[k3]])) continue;
            for (int k4 = m.start[d]; k4 < m.start[d + 1] && !acid; ++k4) {
              const int e5 = m.nbr[k4];
              if (e5 == c || e5 == b || e5 == i) continue;
              if (m.z[e5] != 6 || !m.arom[e5]) continue;
              if (!singleOrArom(m.bcode[m.nbond[k4]])) continue;
              const int closing = m.bondBetween(e5, i);        // the `1` ring closure, also default
              if (closing >= 0 && singleOrArom(m.bcode[closing])) acid = true;
            }
          }
        }
      }
    }
    // [NH2]-[CX4] / [NH](-[CX4])-[CX4] / N(-[CX4])(-[CX4])-[CX4]
    if (!base && m.z[i] == 7 && !m.arom[i]) {
      int nsp3c = 0;
      for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
        const int e = m.nbond[k], j = m.nbr[k];
        if (!(m.bcode[e] & BC_SINGLE)) continue;
        if (m.z[j] == 6 && !m.arom[j] && m.tdeg[j] == 4) ++nsp3c;
      }
      const int h = m.hcount[i];
      if ((h == 2 && nsp3c >= 1) || (h == 1 && nsp3c >= 2) || nsp3c >= 3) base = true;
    }
    // N=C-N and N-C=N: the root is the FIRST nitrogen of the three-atom pattern
    if (!base && m.z[i] == 7 && !m.arom[i]) {
      for (int k = m.start[i]; k < m.start[i + 1] && !base; ++k) {
        const int e = m.nbond[k], c = m.nbr[k];
        if (m.z[c] != 6 || m.arom[c]) continue;
        const bool first_double = (m.bcode[e] & BC_DOUBLE) != 0;
        const bool first_single = (m.bcode[e] & BC_SINGLE) != 0;
        if (!first_double && !first_single) continue;
        for (int q = m.start[c]; q < m.start[c + 1]; ++q) {
          const int e2 = m.nbond[q], n2 = m.nbr[q];
          if (n2 == i || m.z[n2] != 7 || m.arom[n2]) continue;
          if (first_double && (m.bcode[e2] & BC_SINGLE)) { base = true; break; }
          if (first_single && (m.bcode[e2] & BC_DOUBLE)) { base = true; break; }
        }
      }
    }
    if (acid) ++nAcid;
    if (base) ++nBase;
  }
  out[0] = (double)nAcid;
  out[1] = (double)nBase;
}

// ---------------------------------------------------------------------------------------------
// VdwVolumeABC.  `ac - 5.92*Nb - 14.7*NRa - 3.8*NRA`, left to right as mordred writes it, where
// `ac` sums the Bondi sphere volumes over the H-ADDED molecule, Nb is its BOND count, NRa is the
// AROMATIC ring count and NRA the ALIPHATIC one (mordred's argument names are the other way round
// from the way they read: `RingCount(None,False,False,True,None)` is naRing and is bound to NRa).
// An element outside mordred's thirteen-entry `bondi_radii` is a KeyError there and a NaN here.
// ---------------------------------------------------------------------------------------------
// THE ONE COLUMN OF THE 43 THAT IS NOT BIT-EXACT AGAINST MORDRED, AND WHY THAT IS THE RIGHT
// ANSWER RATHER THAN A DEFECT.
//
// 99,970 of 100,000 exact; 30 differ, and all 30 are molecules where the boundary's REPAIRED ring
// set (src/hume/_rings.py) differs from `Chem.GetSymmSSSR`.  Vabc reads rings only through naRing
// and nARing, so it inherits RingCount's documented divergence exactly and adds nothing of its
// own.  The disagreement is always a multiple of 3.8 or 14.7 -- one ring.
//
// Every one of the 30 was checked individually: mordred's Vabc was recomputed on each under 24
// (and for the residual cases 400) atom-renumbering-AND-bond-shuffle perturbations, and on every
// one of them MORDRED GIVES TWO OR THREE DIFFERENT ANSWERS.  Examples, with mordred's own set of
// answers for a single molecule:
//
//     row 14189   790.657003707, 794.457003707, 798.257003707
//     row 1582    511.743135141, 515.543135141
//     row 30733   (naRing, nARing) takes (16,12), (16,15) and (16,17) over 400 perturbations
//
// so there is no value on those molecules to be exact against.  The claim for this column is
// therefore "exact on 99,970, and DETERMINISTIC where mordred is not", which is the same claim
// RingCount already carries and is strictly stronger than matching a coin flip.
//
// A NEAR MISS WORTH KEEPING.  Rows 30733, 76531 and 98739 first came back STABLE over 400
// perturbations, which would have made them three real disagreements.  They were not: all three
// are stereo-rich, the screen's isomeric guard rejected every bond-shuffled rebuild for them, and
// it had been silently substituting atom-only renumbering -- the weak axis.  With the guard
// relaxed for a column that does not read stereo, all 400 rebuilds succeed and all three move.
// The wrong conclusion here was one guard away, in the direction of claiming a defect that is
// not there.
//
// fMF READS THE SAME RING SETS AND IS EXACT ON ALL 100,000, which is not a contradiction: what
// symmetrizeSSSR adds or drops is a symmetry-equivalent ring of a size already present, whose
// atoms are already ring atoms.  fMF counts the UNION of ring atoms, so it cannot see the extra
// ring; Vabc counts the RINGS, so it can.
inline double vabc(const Mol& m, const HDerived& H, const Inputs& in) {
  double ac = 0.0;
  for (int i = 0; i < m.n; ++i) {
    const int z = m.z[i];
    if (z < 0 || z > constit_tbl::MAX_Z || !constit_tbl::BONDI_OK[z]) return qnan();
    ac += constit_tbl::BONDI[z];
  }
  for (int k = 0; k < H.nHadd; ++k) ac += constit_tbl::BONDI[1];
  const double t1 = 5.92 * (double)H.nBondsH;      // split into statements: no FMA, see the
  const double t2 = 14.7 * in.naRing;              // contraction note at the top of this file
  const double t3 = 3.8 * in.nARing;
  double r = ac - t1;
  r -= t2;
  r -= t3;
  return r;
}

// ---------------------------------------------------------------------------------------------
// Polarizability, `bpol`: sum over the H-ADDED molecule's bonds of |pol[za] - pol[zb]|, with
// mordred's 1994 table.  The summation ORDER is Chem.AddHs's bond order -- the heavy bonds first,
// unchanged, then one bond per added hydrogen grouped by the heavy atom it hangs off -- because
// this is a float sum and the screen measured it moving by 1.3e-15 under renumbering.
// ---------------------------------------------------------------------------------------------
inline double bpol(const Mol& m) {
  const double pH = constit_tbl::POL94[1];
  double acc = 0.0;
  for (int e = 0; e < m.nb; ++e)
    acc += std::fabs(constit_tbl::POL94[m.z[m.bu[e]]] - constit_tbl::POL94[m.z[m.bv[e]]]);
  for (int i = 0; i < m.n; ++i) {
    const double d = std::fabs(constit_tbl::POL94[m.z[i]] - pH);
    for (int k = 0; k < m.nh[i]; ++k) acc += d;
  }
  return acc;
}

// ---------------------------------------------------------------------------------------------
// LogS (Filter-it).  `0.89823 - 0.10369*sqrt(MW)` plus one `count * coefficient` term per SMARTS,
// IN mordred's DICT INSERTION ORDER, on the HEAVY, NON-kekulized molecule.  MW is the AVERAGE
// molecular weight of the H-added molecule (`Weight(exact=False)`).
//
// All sixteen patterns are single-atom bracket queries, so each is a predicate over the boundary
// columns and there is no matcher here.  cpp/verify_constit.py checks every one of them, pattern
// by pattern and atom by atom, against rdkit's own SMARTS matcher, so the translation below is
// evidence-backed rather than a reading.
// ---------------------------------------------------------------------------------------------
// SMARTS `H` AND SMARTS `h` ARE DIFFERENT PRIMITIVES AND THIS FAMILY USES BOTH.  Fourteen of the
// sixteen patterns are written with an UPPERCASE `H`, which is the TOTAL hydrogen count including
// hydrogens present as neighbouring atoms -- `hcount` here, rdkit's `GetTotalNumHs(true)`.  Two of
// them, `[ch0]` and `[ch1]`, are written with a LOWERCASE `h`, which is a different query, and the
// boundary column that answers it is `nh` -- rdkit's `GetTotalNumHs(false)`, measured equal to the
// `h` query on every atom of 20,000 molecules with 0 mismatches, where `GetNumImplicitHs()` is
// wrong on 62 and the total-H reading is wrong on 8.
//
// The molecules that discriminate them all carry a `[2H]` on an aromatic carbon -- e.g.
// `COP(=S)(OC)SCn1nnc2ccccc2c1=O.[2H]c1cc(C(=O)OCC)nc(C(=O)OCC)c1`, where the deuterated carbon is
// `[ch0]` to rdkit and would be `[ch1]` under the total-H reading.  98 aromatic carbons of 20,000
// molecules carry one.  Reading mordred's dict quickly and seeing "H" in every key is exactly how
// this gets shipped wrong, which is why the verifier checks all sixteen patterns individually
// against rdkit's own matcher rather than checking the column.
inline bool logsMatch(const Mol& m, int i, int p) {
  const int z = m.z[i], h = m.hcount[i], X = m.tdeg[i], v = m.tval[i];
  const int himp = m.nh[i];                 // SMARTS lowercase `h`
  const bool ar = m.arom[i] != 0, R = m.nring[i] != 0;
  switch (p) {
    case 0:  return z == 7 && !ar && h == 0 && X == 3 && v == 3;   // [NH0;X3;v3]
    case 1:  return z == 7 && !ar && h == 2 && X == 3 && v == 3;   // [NH2;X3;v3]
    case 2:  return z == 7 &&  ar && h == 0 && X == 3;             // [nH0;X3]
    case 3:  return z == 8 && !ar && h == 0 && X == 2 && v == 2;   // [OH0;X2;v2]
    case 4:  return z == 8 && !ar && h == 0 && X == 1 && v == 2;   // [OH0;X1;v2]
    case 5:  return z == 8 && !ar && h == 1 && X == 2 && v == 2;   // [OH1;X2;v2]
    case 6:  return z == 6 && !ar && h == 2 && !R;                 // [CH2;!R]
    case 7:  return z == 6 && !ar && h == 3 && !R;                 // [CH3;!R]
    case 8:  return z == 6 && !ar && h == 0 &&  R;                 // [CH0;R]
    case 9:  return z == 6 && !ar && h == 2 &&  R;                 // [CH2;R]
    case 10: return z == 6 &&  ar && himp == 0;                    // [ch0]  -- lowercase h
    case 11: return z == 6 &&  ar && himp == 1;                    // [ch1]  -- lowercase h
    case 12: return z == 9  && !ar;                                // F
    case 13: return z == 17 && !ar;                                // Cl
    case 14: return z == 35 && !ar;                                // Br
    case 15: return z == 53 && !ar;                                // I
  }
  return false;
}

inline double filterItLogS(const Mol& m, double mw) {
  const double b = constit_tbl::LOGS_B * std::sqrt(mw);      // split: no FMA
  double logS = constit_tbl::LOGS_A + b;
  for (int p = 0; p < constit_tbl::N_LOGS; ++p) {
    int c = 0;
    for (int i = 0; i < m.n; ++i) if (logsMatch(m, i, p)) ++c;
    const double term = (double)c * constit_tbl::LOGS_COEF[p];
    logS += term;
  }
  return logS;
}

// ---------------------------------------------------------------------------------------------
// Framework, `fMF`.  The molecule is contracted: every ring becomes one node, every non-ring atom
// stays its own node, and `linkers` is the set of non-ring atoms lying on a shortest path between
// two ring nodes.  fMF = (|linkers| + |ring atoms|) / N.
//
// THE TWO MOLECULES.  `FrameworkCache` inherits explicit_hydrogens = True and so walks the BONDS
// OF THE H-ADDED MOLECULE, while its `Rings()` dependency is a RingCountBase with
// explicit_hydrogens = False and returns rings of the HEAVY one -- and `Framework.calculate`'s
// `N = self.mol.GetNumAtoms()` is again the H-ADDED count.  Chem.AddHs appends, so the heavy
// indices are valid in both and the ring sets transfer unchanged.  The hydrogens themselves can
// never be linkers (a terminal atom lies on no shortest path between two other nodes), so the
// only thing they change is N -- which is the whole difference between fMF and 1.5x fMF.
//
// `Rd` MAPS AN ATOM TO THE LAST RING CONTAINING IT, because mordred builds it as a dict
// comprehension over `enumerate(Rs)` and a later ring silently overwrites an earlier one.  A ring
// all of whose atoms are claimed by later rings therefore contributes NO NODE at all, which is
// why the node list is built from the values actually present rather than from range(len(Rs)).
//
// THE RING SETS ARE THE BOUNDARY'S, which is src/hume/_rings.py's REPAIRED answer and not
// `Chem.GetSymmSSSR`'s raw one.  The two differ on 1 of 3,000 molecules of cpp/hard.smi, and the
// repo has already taken the position that the repaired set is the well-posed object (see
// PORT_STATUS on RingCount: 22 of 100,000 molecules move under renumbering before the repair and
// 0 after).  Sourcing rings from a second place here to match mordred's raw answer would put two
// different ring perceptions in one process, which is the failure the repair exists to prevent.
// cpp/verify_constit.py reports the fMF disagreements separately for exactly this reason.
// ---------------------------------------------------------------------------------------------
inline double framework(const Mol& m, const HDerived& H) {
  if (H.nAtomsH == 0) return qnan();
  const int NR = m.nr;
  std::vector<int> owner((size_t)m.n, -1);
  for (int r = 0; r < NR; ++r)
    for (int k = m.ring_ptr[r]; k < m.ring_ptr[r + 1]; ++k) owner[m.ring_at[k]] = r;

  // Ring atoms, counted once each -- the union over ALL rings, including rings that own no atom.
  std::vector<char> isRingAtom((size_t)m.n, 0);
  for (int r = 0; r < NR; ++r)
    for (int k = m.ring_ptr[r]; k < m.ring_ptr[r + 1]; ++k) isRingAtom[m.ring_at[k]] = 1;
  int nRingAtoms = 0;
  for (int i = 0; i < m.n; ++i) nRingAtoms += isRingAtom[i];

  // Node ids: 0..m.n-1 are atoms, m.n + r is ring r.  Only rings that own an atom are reachable,
  // which reproduces `R = list(set(Rd.values()))`.
  std::vector<int> present;
  {
    std::vector<char> seen((size_t)NR, 0);
    for (int i = 0; i < m.n; ++i) if (owner[i] >= 0) seen[owner[i]] = 1;
    for (int r = 0; r < NR; ++r) if (seen[r]) present.push_back(r);
  }
  if (present.size() < 2) return (double)nRingAtoms / (double)H.nAtomsH;

  const int NN = m.n + NR;
  std::vector<std::vector<int> > adj((size_t)NN);
  // Hydrogens are not added as nodes: they are terminal, so they cannot lie on any shortest path
  // between two ring nodes, and the only thing their presence would change is running time.
  for (int e = 0; e < m.nb; ++e) {
    const int a = owner[m.bu[e]] >= 0 ? m.n + owner[m.bu[e]] : m.bu[e];
    const int b = owner[m.bv[e]] >= 0 ? m.n + owner[m.bv[e]] : m.bv[e];
    if (a == b) continue;                       // networkx keeps the self-loop; it changes nothing
    adj[a].push_back(b);
    adj[b].push_back(a);
  }

  std::vector<char> linker((size_t)m.n, 0);
  std::vector<int> dist((size_t)NN), par((size_t)NN), q((size_t)NN);
  for (size_t ia = 0; ia + 1 < present.size(); ++ia) {
    const int src = m.n + present[ia];
    std::fill(dist.begin(), dist.end(), -1);
    int head = 0, tail = 0;
    dist[src] = 0; par[src] = -1; q[tail++] = src;
    while (head < tail) {
      const int u = q[head++];
      for (size_t k = 0; k < adj[u].size(); ++k) {
        const int v = adj[u][k];
        if (dist[v] < 0) { dist[v] = dist[u] + 1; par[v] = u; q[tail++] = v; }
      }
    }
    for (size_t ib = ia + 1; ib < present.size(); ++ib) {
      int t = m.n + present[ib];
      if (dist[t] < 0) continue;                // nx.NetworkXNoPath, which mordred swallows
      while (t != src) { if (t < m.n) linker[t] = 1; t = par[t]; }
    }
  }
  int nLink = 0;
  for (int i = 0; i < m.n; ++i) nLink += linker[i];
  return (double)(nLink + nRingAtoms) / (double)H.nAtomsH;
}

// ---------------------------------------------------------------------------------------------
// FragmentComplexity: |B^2 - A^2 + A| + H/100 over the HEAVY molecule, H being the non-carbon
// count.  Integer arithmetic until the final divide, as mordred has it.
// ---------------------------------------------------------------------------------------------
inline double fragCpx(const Mol& m) {
  long long A = m.n, B = m.nb, het = 0;
  for (int i = 0; i < m.n; ++i) if (m.z[i] != 6) ++het;
  long long v = B * B - A * A + A;
  if (v < 0) v = -v;
  return (double)v + (double)het / 100.0;
}

// ---------------------------------------------------------------------------------------------
// rdkit `SPS` (normalised spacial score).  Per atom, hybridisation x stereo x ring x degree^2,
// summed and divided by the heavy-atom count.
//
//     hyb     SP 1, SP2 2, SP3 3, ANYTHING ELSE 4 (rdkit uses a defaultdict(lambda: 4))
//     stereo  2 if the atom is a (pseudo)stereocentre INCLUDING UNASSIGNED ones, or is an end of
//             a double bond whose stereo is set after `FindPotentialStereoBonds`; else 1
//     ring    1 if aromatic, 2 if in a ring, else 1
//     bond    GetDegree()^2
//
// THE STEREO TERM IS THE ONLY THING NOT AT THE BOUNDARY.  `cip` carries an ASSIGNED R/S and
// `bond_s` an ASSIGNED E/Z, and SPS asks a different question -- `FindMolChiralCenters(
// includeUnassigned=True, useLegacyImplementation=False)` and `rdmolops.FindPotentialStereoBonds`
// are rdkit's POTENTIAL stereo perception, the same subsystem `NumAtomStereoCenters` and
// `NumUnspecifiedAtomStereoCenters` are still waiting on.  So the two flags are taken as inputs;
// everything else is computed here and is verified exact when they are supplied.
// ---------------------------------------------------------------------------------------------
inline double sps(const Mol& m, const Inputs& in) {
  if (!in.stereoAtom || !in.stereoBond) return qnan();
  int heavy = 0;
  for (int i = 0; i < m.n; ++i) if (m.z[i] != 1) ++heavy;
  if (!heavy) return qnan();
  std::vector<char> onStereoBond((size_t)m.n, 0);
  for (int e = 0; e < m.nb; ++e)
    if (in.stereoBond[e]) { onStereoBond[m.bu[e]] = 1; onStereoBond[m.bv[e]] = 1; }
  long long score = 0;
  for (int i = 0; i < m.n; ++i) {
    int hy;
    switch (m.hyb[i]) {
      case HYB_SP: hy = 1; break;
      case HYB_SP2: hy = 2; break;
      case HYB_SP3: hy = 3; break;
      default: hy = 4; break;
    }
    const int st = (in.stereoAtom[i] || onStereoBond[i]) ? 2 : 1;
    const int rg = m.arom[i] ? 1 : (m.nring[i] ? 2 : 1);
    const long long bd = (long long)m.deg[i] * m.deg[i];
    score += (long long)hy * st * rg * bd;
  }
  return (double)score / (double)heavy;
}

// ---------------------------------------------------------------------------------------------
// rdkit `qed`.  Eight properties through an ADS desirability function, geometric-mean weighted.
// Seven of the eight are computed here exactly:
//
//   MW     _CalcMolWt of the HEAVY molecule (QED.properties() calls Chem.RemoveHs first)
//   ALOGP  Crippen.MolLogP           -- vsa_bins.h
//   HBA    the eleven acceptor SMARTS, all single-atom, summed as MATCH COUNTS
//   HBD    CalcNumHBD               -- frag_matcher.h
//   PSA    MolSurf.TPSA             -- vsa_bins.h
//   ROTB   CalcNumRotatableBonds(Strict)  -- frag_matcher.h
//   AROM   len(GetSSSR(DeleteSubstructs(mol, '[$([A;R][!a])]')))
//   ALERTS how many of 116 structural-alert SMARTS match          <- NOT COMPUTED HERE
//
// AROM WITHOUT EDITING A MOLECULE.  rdkit deletes every atom matching `[$([A;R][!a])]` -- an
// ALIPHATIC RING atom having at least one NON-AROMATIC neighbour -- and counts the SSSR of what
// is left.  |SSSR| is the cyclomatic number E - V + C of any graph, so the deletion and the ring
// perception are both replaced by one pass over the surviving subgraph.  (rdkit's own comment
// says this "tends to count more rings" than NumAromaticRings and names three molecules where
// they differ; the cyclomatic count reproduces rdkit, not the comment.)
//
// ALERTS IS 116 SMARTS THIS FILE CANNOT MATCH.  They use recursive queries, ring closures,
// component-level `.` grouping, `~` and `@` bonds and isotope queries -- a general matcher, which
// src/hume_core/frag_matcher.h already is, but bound to `frag_prog`'s tables at namespace scope.
// Writing a second matcher here would put two subgraph-isomorphism implementations in the repo.
// So `qedAlerts` is an input; see the wiring note.
// ---------------------------------------------------------------------------------------------
inline int qedHBA(const Mol& m) {
  int n = 0;
  for (int i = 0; i < m.n; ++i) {
    const int z = m.z[i], h = m.hcount[i], X = m.tdeg[i], v = m.tval[i], c = m.fchg[i];
    const bool ar = m.arom[i] != 0;
    if (z == 8 &&  ar && h == 0 && X == 2) { ++n; continue; }      // [oH0;X2]
    if (z == 8 && !ar && h == 1 && X == 2 && v == 2) { ++n; continue; }   // [OH1;X2;v2]
    if (z == 8 && !ar && h == 0 && X == 2 && v == 2) { ++n; continue; }   // [OH0;X2;v2]
    if (z == 8 && !ar && h == 0 && X == 1 && v == 2) { ++n; continue; }   // [OH0;X1;v2]
    if (z == 8 && !ar && c == -1 && X == 1) { ++n; continue; }            // [O-;X1]
    if (z == 16 && !ar && h == 0 && X == 2 && v == 2) { ++n; continue; }  // [SH0;X2;v2]
    if (z == 16 && !ar && h == 0 && X == 1 && v == 2) { ++n; continue; }  // [SH0;X1;v2]
    if (z == 16 && !ar && c == -1 && X == 1) { ++n; continue; }           // [S-;X1]
    if (z == 7 &&  ar && h == 0 && X == 2) { ++n; continue; }             // [nH0;X2]
    if (z == 7 && !ar && h == 0 && X == 1 && v == 3) { ++n; continue; }   // [NH0;X1;v3]
    // [$([N;+0;X3;v3]);!$(N[C,S]=O)] -- aliphatic-or-aromatic N is `N` here, which is ALIPHATIC;
    // the outer `[$(...)]` wrapper changes nothing but the parse tree.
    if (z == 7 && !ar && c == 0 && X == 3 && v == 3) {
      bool amide = false;
      for (int k = m.start[i]; k < m.start[i + 1] && !amide; ++k) {
        const int j = m.nbr[k];
        if (m.arom[j] || (m.z[j] != 6 && m.z[j] != 16)) continue;
        if (!singleOrArom(m.bcode[m.nbond[k]])) continue;     // N[C,S] is a default bond
        for (int q = m.start[j]; q < m.start[j + 1]; ++q)
          if ((m.bcode[m.nbond[q]] & BC_DOUBLE) && m.z[m.nbr[q]] == 8 && !m.arom[m.nbr[q]]) {
            amide = true; break;
          }
      }
      if (!amide) { ++n; continue; }
    }
  }
  return n;
}

// len(GetSSSR(DeleteSubstructs(mol, AliphaticRings))) as a cyclomatic number; see above.
//
// THE BOND IN `[$([A;R][!a])]` IS WRITTEN, NOT ABSENT.  Two SMARTS atoms with nothing between
// them are joined by rdkit's DEFAULT bond query, `SingleOrAromatic` -- which is a third thing
// again, not `~` and not `-`.  So an aliphatic ring atom joined to its non-aromatic neighbour by
// a DOUBLE or TRIPLE bond does NOT match and is NOT deleted.  Reading the pattern as "any bond"
// gives a different atom set on 62 of 3,000 molecules of cpp/hard.smi and the single-or-aromatic
// reading gives 0, which is the same trap src/hume_core/estate_tables.h and frag_matcher.h both
// record for their own pattern sets.
inline int qedArom(const Mol& m) {
  std::vector<char> keep((size_t)m.n, 1);
  for (int i = 0; i < m.n; ++i) {
    if (m.arom[i] || !m.nring[i]) continue;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
      if (!singleOrArom(m.bcode[m.nbond[k]])) continue;        // default = SingleOrAromatic
      if (!m.arom[m.nbr[k]]) { keep[i] = 0; break; }
    }
  }
  int V = 0, E = 0;
  std::vector<int> id((size_t)m.n, -1);
  for (int i = 0; i < m.n; ++i) if (keep[i]) id[i] = V++;
  if (!V) return 0;
  std::vector<int> uf((size_t)V);
  for (int i = 0; i < V; ++i) uf[i] = i;
  struct F { static int find(std::vector<int>& p, int x) {
      while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; } };
  for (int e = 0; e < m.nb; ++e) {
    if (!keep[m.bu[e]] || !keep[m.bv[e]]) continue;
    ++E;
    const int a = F::find(uf, id[m.bu[e]]), b = F::find(uf, id[m.bv[e]]);
    if (a != b) uf[a] = b;
  }
  int C = 0;
  for (int i = 0; i < V; ++i) if (F::find(uf, i) == i) ++C;
  return E - V + C;
}

// rdkit/Chem/QED.py `ads()` and its eight parameter rows, and WEIGHT_MEAN.  These are ordinary
// literals in a .py file with no rdkit-side table behind them, so they live here beside the
// arithmetic that uses them rather than in the generated tables header; cpp/verify_constit.py
// reads them back out of QED.py and asserts they still agree.
struct AdsP { double A, B, C, D, E, F, DMAX; };
static const AdsP QED_ADS[8] = {
  {2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561},
  {3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604},
  {2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046},
  {1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616},
  {1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167},
  {0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403},
  {3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610},
  {0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140},
};
static const double QED_WEIGHT_MEAN[8] = {0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95};

inline double qedAds(double x, const AdsP& p) {
  const double exp1 = 1 + std::exp(-1 * (x - p.C + p.D / 2) / p.E);
  const double exp2 = 1 + std::exp(-1 * (x - p.C - p.D / 2) / p.F);
  const double q = p.B / exp1 * (1 - 1 / exp2);              // split: no FMA
  const double dx = p.A + q;
  return dx / p.DMAX;
}

inline double qedScore(const Mol& m, const Inputs& in, double heavyMolWt, double tpsa) {
  if (in.qedAlerts < 0) return qnan();
  const double props[8] = {heavyMolWt, in.molLogP, (double)qedHBA(m), (double)in.nHBDon,
                           tpsa, (double)in.nRot, (double)qedArom(m), (double)in.qedAlerts};
  // qed() = exp(sum(w_i * log(d_i)) / sum(w_i)), accumulated in QEDproperties field order.
  double t = 0.0, s = 0.0;
  for (int i = 0; i < 8; ++i) {
    const double term = QED_WEIGHT_MEAN[i] * std::log(qedAds(props[i], QED_ADS[i]));
    t += term;                                                // split: no FMA
    s += QED_WEIGHT_MEAN[i];
  }
  return std::exp(t / s);
}

// ---------------------------------------------------------------------------------------------
// Everything, into `out[N_COLS]`.
//
// `heavyMolWt()` is _CalcMolWt of the HEAVY molecule, which qed wants; `molWt()` is the H-ADDED
// one, which LogS wants.  They are the same real number and different float64s; see the note on
// `heavyMolWt`.
// ---------------------------------------------------------------------------------------------
inline void compute(const Mol& m, const Inputs& in, double* out, double tpsa) {
  const HDerived H(m);
  carbonTypes(m, out + 0);
  atomCount(m, H, out + C_NH);
  bondCount(m, H, out + C_NBONDSS);
  kappaShape(m, out + C_KIER1);
  {
    std::vector<double> D;
    distanceMatrix(m, D);
    molecularDistanceEdge(m, D, out + C_MDEC22);
  }
  chargeRatios(in, H.nAtomsH, out + C_RNCG);

  const double mwExact = exactMolWt(m, H);
  const double mwAvg = molWt(m, H);
  out[C_LIPINSKI] = (in.nHBDon <= 5 && in.nHBAcc <= 10 && mwExact <= 500 && in.molLogP <= 5)
                    ? 1.0 : 0.0;
  out[C_LIPINSKI + 1] = (160 <= mwExact && mwExact <= 480 &&
                         20 <= H.nAtomsH && H.nAtomsH <= 70 &&
                         -0.4 <= in.molLogP && in.molLogP <= 5.6 &&
                         40 <= in.molMR && in.molMR <= 130) ? 1.0 : 0.0;

  acidBase(m, out + C_NACID);
  out[C_VABC] = vabc(m, H, in);
  out[C_ROTRATIO] = m.nb ? (double)in.nRot / (double)m.nb : qnan();
  out[C_BPOL] = bpol(m);
  out[C_LOGS] = filterItLogS(m, mwAvg);
  out[C_FMF] = framework(m, H);
  out[C_FRAGCPX] = fragCpx(m);
  // qed's MW is the HEAVY molecule's _CalcMolWt, which is NOT the same float as the H-added one.
  out[C_QED] = qedScore(m, in, heavyMolWt(m), tpsa);
  out[C_SPS] = sps(m, in);
}

}  // namespace constit

// =============================================================================================
// WIRING NOTES for src/hume_core/bindings.cpp -- NOT applied here, that file is another agent's.
// =============================================================================================
//
// 1. SIX COLUMNS OF THIS CENSUS BLOCK ARE ALREADY COMPUTED and need only to be EMITTED, not
//    implemented.  vsa_bins.h's `out[]` already carries them; give them their mordred names:
//
//        TopoPSA      = vsabin C_TOPOPSA      (already named "TopoPSA")
//        TPSA         = vsabin C_TPSA         (already named "TPSA")
//        SLogP        = vsabin C_MOLLOGP      -- ALIAS, do not recompute: mordred/SLogP.py is
//                                                `return Crippen.MolLogP(self.mol)`
//        PEOE_VSA11   = vsabin PEOE_VSA11     -- MoeType resolves by getattr to the same function
//        SMR_VSA1     = vsabin SMR_VSA1
//        EState_VSA1  = vsabin EState_VSA1
//
// 2. constit::compute() needs, per molecule:
//        Mol::build_from_rows(atom_i, 10, atom_d, 2, bond_i, 5, bond_d, ring_ptr, ring_at)
//        Inputs{ molLogP, molMR   from vsa_bins,
//                nHBDon, nHBAcc, nRot   from frag_matcher's countAll() slots
//                    NumHDonors / NumHAcceptors / NumRotatableBonds,
//                naRing, nARing   from ringcount's out[] at the "naRing"/"nARing" columns,
//                hchg, nhchg      the H-ADDED molecule's Gasteiger charges -- autocorr's boundary
//                                 ALREADY MATERIALISES THIS GRAPH; pass the same array }
//        tpsa                     from vsa_bins, for qed's PSA term
//    Call `constit::checkSpec()` once at module load, next to the other selfChecks.
//
// 3. TWO COLUMNS NEED ONE NEW BOUNDARY FIELD EACH, and only that:
//
//    * `SPS` needs rdkit's POTENTIAL stereo, which the assigned-only `cip` and `bond_s` columns
//      cannot answer.  Two arrays, both one Python call per molecule in src/hume/_extract.py:
//          stereoAtom[i] = i in {idx for idx,_ in Chem.FindMolChiralCenters(
//                              m, includeUnassigned=True, includeCIP=False,
//                              useLegacyImplementation=False)}
//          stereoBond[e] = after `rdmolops.FindPotentialStereoBonds(Chem.Mol(m))`,
//                          bond e is DOUBLE and its GetStereo() != STEREONONE
//      Do NOT try to derive it: it is a perception, not a graph query.
//
//      IT IS NOT, HOWEVER, THE SAME PERCEPTION `NumAtomStereoCenters` AND
//      `NumUnspecifiedAtomStereoCenters` WANT, and this note used to claim it was ("one boundary
//      addition unblocks three columns").  Measured at rdkit 2025.09.2:
//      Code/GraphMol/Descriptors/Lipinski.cpp counts atoms carrying `_ChiralityPossible`, set by
//      the LEGACY `MolOps::assignStereochemistry(cleanIt, force, flagPossible)` -- not by
//      FindPotentialStereo.  The two atom sets differ on 262 of 4,000 cpp/hard.smi molecules.  So
//      those two columns cost nothing (the flag is already in the pickle, explicit-property bit
//      0x8, and the chiral tag is already in atom-property flag bit 2) while `SPS` still needs a
//      real boundary addition of its own.
//
//    * `qed` needs `qedAlerts`, the count of rdkit QED's 116 structural-alert SMARTS that match.
//      The cheapest correct route is to make src/hume_core/frag_matcher.h's `Matcher` take its
//      program tables (NODES / AROOTS / QBONDS / PATTERNS / N_PATTERNS) as a bound reference
//      instead of reading `frag_prog`'s at namespace scope, then generate a second program from
//      QED.StructuralAlertSmarts with cpp/gen_frag_program.py and match it with the SAME
//      matcher.  That keeps ONE subgraph-isomorphism implementation in the repo.  The alert set
//      needs opcodes the current program does not exercise -- isotope (`[15N]`), any-bond (`~`),
//      ring-bond (`@`) and component-level `.` grouping -- so the generator will need those four
//      added; they are all leaf predicates except the last.
//      Everything downstream of the count is implemented and exact here.
//
// 4. NOTHING IN THIS FILE RECOMPUTES ANOTHER HEADER'S ANSWER.  Crippen, TPSA, the H-bond and
//    rotatable-bond counts, the ring counts and the numpy summation order all come from the
//    header that already owns them.  If a value has to be recomputed to wire this up, that is a
//    sign the wiring is wrong, not that this file needs a copy.
#endif  // HUME_CONSTIT_H
