// F_misc: the 81 columns of results/dedupe2/agent_groups.json["F_misc"].
//
// THIS GROUP IS NOT A FAMILY.  It is eleven unrelated sub-families that happened to survive the
// same dedupe, and the only thing they share is this file.  NOTES_misc.md carries the
// sub-classification and the per-cluster grade; the clusters, in the order they are computed
// below, are:
//
//   S1  scalars        11  Chi0 ExactMolWt NumValenceElectrons VAdjMat VMcGowan mZagreb1
//                          mZagreb2 ECIndex Radius HybRatio Sv
//   S2  Chi subgraphs  13  Xch-3d Xch-3dv Xch-4dv Xc-6dv Xpc-6d Xp-1d Xp-3d Xp-7d AXp-0d
//                          Xp-2dv Xp-3dv Xp-4dv Xp-7dv         -- src/hume_core/chi.h
//   S3  PathCount       8  MPC5 MPC7 MPC8 MPC10 TMPC10 piPC7 piPC9 TpiPC10
//                                                          -- src/hume_core/pathcount.h
//   S4  WalkCount       5  MWC02 MWC04 MWC07 MWC09 TMWC10  -- src/hume_core/topomisc.h
//   S5  MolDistEdge     5  MDEC-11 MDEC-12 MDEC-13 MDEO-11 MDEN-22
//   S6  partial charge  4  MinPartialCharge MaxPartialCharge MinAbsPartialCharge
//                          MaxAbsPartialCharge
//   S7  fr_* SMARTS     7  fr_lactam fr_benzodiazepine fr_barbitur fr_azo fr_nitro_arom
//                          fr_phenol_noOrthoHbond fr_phos_ester  -- src/hume_core/frag_matcher.h
//   S8  ETA averaged   14  AETA_alpha AETA_beta{,_s,_ns,_ns_d} AETA_eta{,_L,_R,_RL,_F,_FL,_B,_BR}
//                          AETA_dBeta
//   S9  MolecularId    12  MID AMID MID_h AMID_h MID_C AMID_C MID_N AMID_N MID_O AMID_O
//                          MID_X AMID_X
//   S10 BertzCT         1
//   S11 LogEE_A         1
//
// FOUR OF THE ELEVEN ARE NOT NEW MACHINERY, AND THAT IS DELIBERATE.  S2/S3/S4 are sibling
// columns of families this repository already computes exactly: chi.h accumulates every
// (shape, order, property) bucket and emits 40 of them, pathcount.h accumulates every path
// order and emits 11, topomisc.h walks every adjacency power and emits 6.  This file calls
// those headers and reads the buckets they already fill, rather than carrying a second
// enumeration that could drift from the verified one.  S5 is the same argument one step weaker:
// constit.h's `molecularDistanceEdge` computes three of the fifteen MDE columns and this file
// generalises its loop to five more (a different element and different valence pairs), because
// its signature is fixed to the three it emits.
//
// SPECIFICATIONS, all read rather than recalled (AGENT_CONTRACT house rule 2):
//   mordred 1.2.0    Chi.py PathCount.py WalkCount.py Constitutional.py TopologicalIndex.py
//                    ZagrebIndex.py VertexAdjacencyInformation.py EccentricConnectivityIndex.py
//                    McGowanVolume.py CarbonTypes.py MolecularDistanceEdge.py MolecularId.py
//                    ExtendedTopochemicalAtom.py BertzCT.py AdjacencyMatrix.py
//                    _atomic_property.py _graph_matrix.py _matrix_attributes.py
//                    _base/calculator.py _base/descriptor.py
//   rdkit 2025.09.2  Chem/Descriptors.py Chem/GraphDescriptors.py Chem/Fragments.py,
//                    Code/GraphMol/MolProps.cpp (getExactMolWt),
//                    Code/ML/InfoTheory/InfoGainFuncs.h (InfoEntropy)
//
// ------------------------------------------------------------------------------------------
// SIX THINGS THAT ARE NOT GUESSABLE AND EACH OF WHICH BREAKS EXACTNESS ALONE
// ------------------------------------------------------------------------------------------
//
// 1. RDKit's `InfoEntropy` IS COMPILED WITH FMA CONTRACTION, and BertzCT is a function of it.
//    Code/ML/InfoTheory/InfoGainFuncs.h reads `accum += -d * log(d)` in one statement, which
//    clang contracts to `fma(-d, log(d), accum)`.  The uncontracted form disagrees with the
//    running rdkit on 799 of 3,000 random count vectors (27%); `std::fma` disagrees on 0.
//    The same applies to `res += nHsToCount * mass_H` at the end of getExactMolWt: without the
//    fma, ExactMolWt is wrong on 20 of 4,000 corpus molecules.  Both are written as explicit
//    `std::fma` here so the result does not depend on this file's own -ffp-contract setting.
//
// 2. RDKit's ELECTRON MASS IS 0.00054857991 (Code/GraphMol/atomic_data.cpp), which is NOT the
//    CODATA value 0.000548579909065.  Using CODATA moves ExactMolWt by 5.7e-13 per unit of
//    formal charge -- 20 ulps at 148 Da, far outside any tolerance.
//
// 3. A GASTEIGER FAILURE IS PER CONNECTED COMPONENT, AND THE BOUNDARY DESTROYS WHICH ONE.
//    RDKit's PEOE iteration propagates a nan along BONDS, so an unparameterised atom nans its
//    whole fragment and leaves the others alone: `[Se]C` is two nans, but
//    `O=[Sn]([O-])[O-].[Ca+2]` is four nans and a 2.0, and RDKit's own MinPartialCharge for it
//    is 2.0.  `src/hume/_extract.py` replaces every non-finite charge with 0.0 and records
//    `chg_ok = 0` for the molecule, which cannot distinguish "this atom was nan" from "this atom
//    really is 0.0".  So `chg_ok == 0` is taken here as "all four columns NaN", which is right
//    for a molecule whose charges are entirely nan and WRONG for a partial failure.  Measured on
//    the 20,000-molecule corpus: 4 molecules have `chg_ok == 0`, and exactly ONE of them
//    (`O=[Sn]([O-])[O-].[Ca+2]`) disagrees.  The fold below reads the charge array itself, so it
//    is already correct for a boundary that stops zeroing; NOTES_misc.md carries the one-line
//    `_extract.py` change that would close it.
//
// 4. `min(chg, minChg)` IN PYTHON IS NOT A NaN-SKIPPING MIN.  `Descriptors._ChargeDescriptors`
//    folds with the two-argument builtin, whose first argument seeds the accumulator: a nan
//    charge REPLACES the running minimum and is then itself replaced by the next atom.  It never
//    matters on this corpus (note 3), but the fold below is written in that order anyway.
//    The empty molecule keeps upstream's sentinels: MinPartialCharge 500, MaxPartialCharge -500.
//
// 5. mordred's ETA FAMILY IS KEKULIZED AND THE BOUNDARY IS NOT.  `EtaBase.kekulize = True`, so
//    `get_eta_nonsigma_contribute` sees a former aromatic bond as SINGLE (contributing 0) or
//    DOUBLE (contributing 2.0, because `GetIsAromatic()` survives kekulization).  The Kekule
//    structure is NOT at the boundary; it is reconstructed with the per-atom identity
//    constit.h's `nBondsKD` already uses and proves --
//        takesDouble(i) = tval(i) - nH(i) - round(non-aromatic valence contributions) - nArom(i)
//    -- which says whether atom i carries the double end of a ring bond.  That is exactly what
//    beta_ns_i needs.  See `kekuleTakesDouble` below.
//
// 6. `require_connected` IS A NaN, NOT AN ERROR, and it covers 26 of these 81 columns (all of
//    S8, all of S9, and LogEE_A).  mordred's Calculator returns a MissingValue when
//    `n_frags != 1`, which lands in the matrix as NaN.  `n_frags` is `len(Chem.GetMolFrags(mol))`
//    -- the connected components of the graph as given -- and is computed here rather than
//    carried, since it is a BFS over the adjacency this file already builds.
//
// WHAT THE CALLER MUST SUPPLY: the boundary's own arrays for the HYDROGEN-SUPPRESSED molecule --
// `atom_i` (13 columns), `atom_d` (2), `bond_i` (6), `bond_d`, `chg_ok`, and the ring count from
// the ring CSR.  `build_from_rows()` below takes them in exactly that layout.
#ifndef HUME_MISC_EXT_H
#define HUME_MISC_EXT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../cpp/chiwalk_tables.h"
#include "../../cpp/constit_tables.h"
#include "../../cpp/eigen_small.h"
#include "../../cpp/frag_prog_types.h"
#include "chi.h"
#include "frag_matcher.h"
#include "pathcount.h"
#include "topomisc.h"

namespace miscext {

static constexpr int N_COLS = 81;

// The column order IS results/dedupe2/agent_groups.json["F_misc"], element for element, so the
// wiring in bindings.cpp can be diffed against that list rather than trusted.
enum : int {
  C_MinPartialCharge = 0, C_MinAbsPartialCharge, C_MaxAbsPartialCharge, C_ExactMolWt,
  C_fr_lactam, C_fr_benzodiazepine, C_fr_barbitur, C_fr_azo, C_fr_nitro_arom,
  C_fr_phenol_noOrthoHbond, C_fr_phos_ester,
  C_Chi0, C_LogEE_A, C_BertzCT, C_HybRatio,
  C_Xch3d, C_Xch3dv, C_Xch4dv, C_Xc6dv, C_Xpc6d, C_Xp1d, C_Xp3d, C_Xp7d, C_AXp0d,
  C_Xp2dv, C_Xp3dv, C_Xp4dv, C_Xp7dv,
  C_Sv, C_ECIndex,
  C_AETA_alpha, C_AETA_beta, C_AETA_beta_s, C_AETA_beta_ns, C_AETA_beta_ns_d,
  C_AETA_eta, C_AETA_eta_L, C_AETA_eta_R, C_AETA_eta_RL, C_AETA_eta_F, C_AETA_eta_FL,
  C_AETA_eta_B, C_AETA_eta_BR, C_AETA_dBeta,
  C_VMcGowan, C_MDEC11, C_MDEC12, C_MDEC13, C_MDEO11, C_MDEN22,
  C_MID, C_AMID, C_MID_h, C_AMID_h, C_MID_C, C_AMID_C, C_MID_N, C_AMID_N,
  C_MID_O, C_AMID_O, C_MID_X, C_AMID_X,
  C_MPC5, C_MPC7, C_MPC8, C_MPC10, C_TMPC10, C_piPC7, C_piPC9, C_TpiPC10,
  C_Radius, C_VAdjMat, C_MWC02, C_MWC04, C_MWC07, C_MWC09, C_TMWC10,
  C_mZagreb1, C_mZagreb2, C_NumValenceElectrons, C_MaxPartialCharge
};

inline const char *col_name(int c) {
  static const char *N[N_COLS] = {
      "MinPartialCharge", "MinAbsPartialCharge", "MaxAbsPartialCharge", "ExactMolWt",
      "fr_lactam", "fr_benzodiazepine", "fr_barbitur", "fr_azo", "fr_nitro_arom",
      "fr_phenol_noOrthoHbond", "fr_phos_ester",
      "Chi0", "LogEE_A", "BertzCT", "HybRatio",
      "Xch-3d", "Xch-3dv", "Xch-4dv", "Xc-6dv", "Xpc-6d", "Xp-1d", "Xp-3d", "Xp-7d", "AXp-0d",
      "Xp-2dv", "Xp-3dv", "Xp-4dv", "Xp-7dv",
      "Sv", "ECIndex",
      "AETA_alpha", "AETA_beta", "AETA_beta_s", "AETA_beta_ns", "AETA_beta_ns_d",
      "AETA_eta", "AETA_eta_L", "AETA_eta_R", "AETA_eta_RL", "AETA_eta_F", "AETA_eta_FL",
      "AETA_eta_B", "AETA_eta_BR", "AETA_dBeta",
      "VMcGowan", "MDEC-11", "MDEC-12", "MDEC-13", "MDEO-11", "MDEN-22",
      "MID", "AMID", "MID_h", "AMID_h", "MID_C", "AMID_C", "MID_N", "AMID_N",
      "MID_O", "AMID_O", "MID_X", "AMID_X",
      "MPC5", "MPC7", "MPC8", "MPC10", "TMPC10", "piPC7", "piPC9", "TpiPC10",
      "Radius", "VAdjMat", "MWC02", "MWC04", "MWC07", "MWC09", "TMWC10",
      "mZagreb1", "mZagreb2", "NumValenceElectrons", "MaxPartialCharge"};
  if (c < 0 || c >= N_COLS) throw std::out_of_range("miscext::col_name");
  return N[c];
}

// ---------------------------------------------------------------------------------------------
// Element tables that are NOT already in cpp/.  Both are mordred's own, emitted from the pinned
// mordred 1.2.0 process (`_atomic_property.mc_gowan_volume` / `.period`) rather than typed.
// NaN is mordred's PeriodicTable.__getitem__ for an element the file marks `-` or does not
// reach; a NaN entry makes VMcGowan NaN for the whole molecule, which is upstream's behaviour.
// ---------------------------------------------------------------------------------------------
#define NANV (std::numeric_limits<double>::quiet_NaN())
static const double MCGOWAN_VOL[119] = {
    NANV, 8.71, 6.75, 22.23, 20.27, 18.31,
    16.35, 14.39, 12.43, 10.47, 8.51, 32.71,
    30.75, 28.79, 26.83, 24.87, 22.91, 20.95,
    18.99, 51.89, 50.28, 48.68, 47.07, 45.47,
    43.86, 42.26, 40.65, 39.05, 37.44, 35.84,
    34.23, 32.63, 31.02, 29.42, 27.81, 26.21,
    24.6, 60.22, 58.61, 57.01, 55.4, 53.8,
    52.19, 50.59, 48.98, 47.38, 45.77, 44.17,
    42.56, 40.96, 39.35, 37.75, 36.14, 34.54,
    32.93, 77.25, 76.0, 74.75, 73.49, 72.24,
    70.99, 69.74, 68.49, 67.23, 65.98, 64.73,
    63.48, 62.23, 60.97, 59.72, 58.47, 57.22,
    55.97, 54.71, 53.46, 52.21, 50.96, 49.71,
    48.45, 47.2, 45.95, 44.7, 43.45, 42.19,
    40.94, 39.69, 38.44, 75.59, 74.34, 73.09,
    71.83, 70.58, 69.33, 68.08, 66.83, 65.57,
    64.32, 63.07, 61.82, 60.57, 59.31, 58.06,
    56.81, 55.56, NANV, NANV, NANV, NANV,
    NANV, NANV, NANV, NANV, NANV, NANV,
    NANV, NANV, NANV, NANV, NANV,
};
// mordred's `period`: ([1]*2)+([2]*8)+([3]*8)+([4]*18)+([5]*18)+([6]*32)+([7]*32), 1-indexed.
static const double PERIOD[119] = {
    NANV, 1.0, 1.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    4.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
    6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
    6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
    6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
    6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
    6.0, 6.0, 6.0, 7.0, 7.0, 7.0,
    7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    7.0, 7.0, 7.0, 7.0, 7.0,
};
#undef NANV

// ---------------------------------------------------------------------------------------------
// The seven fr_* SMARTS, compiled by cpp/gen_frag_program.py's own compiler (the one that
// produced cpp/frag_program.h and cpp/qed_alert_program.h) and evaluated by the one matcher in
// src/hume_core/frag_matcher.h.  NOT a second matcher and NOT hand-typed SMARTS: the seven
// patterns are rows of $RDDATA/FragmentDescriptors.csv, and the compiler's `validate` re-renders
// every node in RDKit's own DescribeQuery() format -- 7 top-level patterns + 13 recursive
// sub-queries, 0 mismatches.  The program lives here rather than in cpp/ only because this agent
// may not add files outside its own three; NOTES_misc.md asks for it to be moved to
// cpp/frag_misc_program.h and regenerated by `gen_frag_program.py program misc` when the group
// is wired in, so that the existing `check` drift guard covers it too.
// ---------------------------------------------------------------------------------------------
namespace fr_prog {
using frag_prog_types::Node;
using frag_prog_types::QBond;
using frag_prog_types::Pattern;
using frag_prog_types::Named;
using frag_prog_types::Spec;
constexpr const char SPEC_SHA256[] = "334cb9027881feec3312ddff56cd9f02f2b1164327692b6d04052c49b0d6b75c";
constexpr int N_NODES = 260;
constexpr Node NODES[N_NODES] = {
    {3,0,7,-1,-1},
    {3,0,6,-1,-1},
    {3,0,8,-1,-1},
    {3,0,6,-1,-1},
    {3,0,6,-1,-1},
    {19,0,1,-1,-1},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {3,0,1006,-1,-1},
    {9,0,2,-1,-1},
    {0,0,0,10,11},
    {3,0,1006,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,13,14},
    {3,0,1006,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,16,17},
    {3,0,1006,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,19,20},
    {3,0,1006,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,22,23},
    {3,0,1006,-1,-1},
    {9,0,2,-1,-1},
    {0,0,0,25,26},
    {3,0,7,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,28,29},
    {3,0,6,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,31,32},
    {3,0,6,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,34,35},
    {3,0,7,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,37,38},
    {3,0,6,-1,-1},
    {9,0,1,-1,-1},
    {0,0,0,40,41},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {3,0,6,-1,-1},
    {3,0,6,-1,-1},
    {3,0,8,-1,-1},
    {3,0,7,-1,-1},
    {3,0,6,-1,-1},
    {3,0,8,-1,-1},
    {3,0,7,-1,-1},
    {3,0,6,-1,-1},
    {3,0,8,-1,-1},
    {19,0,1,-1,-1},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {17,0,2,-1,-1},
    {4,0,6,-1,-1},
    {3,0,7,-1,-1},
    {3,0,7,-1,-1},
    {4,0,6,-1,-1},
    {17,0,1,-1,-1},
    {17,0,2,-1,-1},
    {17,0,1,-1,-1},
    {13,0,5,-1,-1},
    {3,0,1006,-1,-1},
    {13,0,6,-1,-1},
    {13,0,7,-1,-1},
    {1,0,0,82,83},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {17,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {3,0,7,-1,-1},
    {6,0,3,-1,-1},
    {0,0,0,97,98},
    {3,0,8,-1,-1},
    {3,0,8,-1,-1},
    {17,0,2,-1,-1},
    {17,0,2,-1,-1},
    {3,0,7,-1,-1},
    {6,0,3,-1,-1},
    {0,0,0,104,105},
    {8,0,1,-1,-1},
    {0,0,0,106,107},
    {3,0,8,-1,-1},
    {3,0,8,-1,-1},
    {8,0,-1,-1,-1},
    {0,0,0,110,111},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {13,0,9,-1,-1},
    {13,1,10,-1,-1},
    {0,0,0,115,116},
    {13,1,11,-1,-1},
    {0,0,0,117,118},
    {13,1,12,-1,-1},
    {0,0,0,119,120},
    {3,0,1006,-1,-1},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,123,124},
    {7,0,1,-1,-1},
    {0,0,0,125,126},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {17,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,6,-1,-1},
    {7,0,2,-1,-1},
    {0,0,0,142,143},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,145,146},
    {7,0,1,-1,-1},
    {0,0,0,147,148},
    {19,0,1,-1,-1},
    {17,0,1,-1,-1},
    {17,1,12,-1,-1},
    {14,0,0,151,152},
    {17,0,1,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,6,-1,-1},
    {3,0,8,-1,-1},
    {3,0,8,-1,-1},
    {7,0,1,-1,-1},
    {8,0,-1,-1,-1},
    {1,0,0,160,161},
    {0,0,0,159,162},
    {19,0,1,-1,-1},
    {17,0,1,-1,-1},
    {17,1,12,-1,-1},
    {14,0,0,165,166},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,1006,-1,-1},
    {3,0,6,-1,-1},
    {3,0,8,-1,-1},
    {3,0,7,-1,-1},
    {7,0,2,-1,-1},
    {0,0,0,174,175},
    {19,0,1,-1,-1},
    {17,0,1,-1,-1},
    {17,1,12,-1,-1},
    {14,0,0,178,179},
    {17,0,2,-1,-1},
    {17,0,1,-1,-1},
    {13,0,14,-1,-1},
    {13,0,19,-1,-1},
    {1,0,0,183,184},
    {3,0,15,-1,-1},
    {3,0,8,-1,-1},
    {6,0,1,-1,-1},
    {0,0,0,187,188},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,190,191},
    {4,0,6,-1,-1},
    {13,0,15,-1,-1},
    {13,0,16,-1,-1},
    {1,0,0,194,195},
    {13,0,17,-1,-1},
    {1,0,0,196,197},
    {13,0,15,-1,-1},
    {13,0,16,-1,-1},
    {1,0,0,199,200},
    {13,0,17,-1,-1},
    {1,0,0,201,202},
    {13,0,18,-1,-1},
    {1,0,0,203,204},
    {17,0,2,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,211,212},
    {7,0,1,-1,-1},
    {0,0,0,213,214},
    {3,0,8,-1,-1},
    {6,0,1,-1,-1},
    {0,0,0,216,217},
    {8,0,-1,-1,-1},
    {0,0,0,218,219},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,221,222},
    {4,0,6,-1,-1},
    {19,0,1,-1,-1},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,226,227},
    {3,0,15,-1,-1},
    {19,0,1,-1,-1},
    {3,0,15,-1,-1},
    {8,0,1,-1,-1},
    {0,0,0,231,232},
    {3,0,8,-1,-1},
    {6,0,1,-1,-1},
    {0,0,0,234,235},
    {8,0,-1,-1,-1},
    {0,0,0,236,237},
    {3,0,8,-1,-1},
    {6,0,2,-1,-1},
    {0,0,0,239,240},
    {4,0,6,-1,-1},
    {13,0,15,-1,-1},
    {13,0,16,-1,-1},
    {1,0,0,243,244},
    {13,0,17,-1,-1},
    {1,0,0,245,246},
    {13,0,15,-1,-1},
    {13,0,16,-1,-1},
    {1,0,0,248,249},
    {13,0,17,-1,-1},
    {1,0,0,250,251},
    {13,0,18,-1,-1},
    {1,0,0,252,253},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
    {19,0,1,-1,-1},
};
constexpr int N_AROOTS = 84;
constexpr int32_t AROOTS[N_AROOTS] = {0,1,2,3,4,12,15,18,21,24,27,30,33,36,39,42,55,56,57,58,59,60,61,62,63,73,74,75,76,80,81,84,85,86,87,88,89,99,100,101,108,109,112,121,122,127,128,129,130,131,132,140,141,144,149,155,156,157,158,163,170,171,172,173,176,185,186,189,192,193,198,205,215,220,223,224,228,229,233,238,241,242,247,254};
constexpr int N_QBONDS = 70;
constexpr QBond QBONDS[N_QBONDS] = {
    {0,1,5},
    {1,2,6},
    {1,3,7},
    {3,4,8},
    {0,4,9},
    {0,1,43},
    {1,2,44},
    {2,3,45},
    {3,4,46},
    {4,5,47},
    {0,5,48},
    {5,6,49},
    {6,7,50},
    {7,8,51},
    {8,9,52},
    {9,10,53},
    {0,10,54},
    {0,1,64},
    {1,2,65},
    {1,3,66},
    {3,4,67},
    {4,5,68},
    {4,6,69},
    {6,7,70},
    {0,7,71},
    {7,8,72},
    {0,1,77},
    {1,2,78},
    {2,3,79},
    {0,1,90},
    {0,2,91},
    {2,3,92},
    {3,4,93},
    {4,5,94},
    {5,6,95},
    {0,6,96},
    {0,1,102},
    {0,2,103},
    {0,1,113},
    {0,2,114},
    {0,1,133},
    {0,2,134},
    {2,3,135},
    {3,4,136},
    {4,5,137},
    {5,6,138},
    {0,6,139},
    {0,1,150},
    {1,2,153},
    {2,3,154},
    {0,1,164},
    {1,2,167},
    {2,3,168},
    {2,4,169},
    {0,1,177},
    {1,2,180},
    {2,3,181},
    {2,4,182},
    {0,1,206},
    {0,2,207},
    {2,3,208},
    {0,4,209},
    {0,5,210},
    {0,1,225},
    {0,1,230},
    {0,1,255},
    {0,2,256},
    {2,3,257},
    {0,4,258},
    {0,5,259},
};
constexpr int N_PATTERNS = 20;
constexpr Pattern PATTERNS[N_PATTERNS] = {
    {"fr_lactam",0,5,0,5},
    {"fr_benzodiazepine",5,11,5,12},
    {"fr_barbitur",16,9,17,9},
    {"fr_azo",25,4,26,3},
    {"fr_nitro_arom",29,1,29,0},
    {"fr_nitro_arom$",30,7,29,7},
    {"fr_nitro_arom$$",37,3,36,2},
    {"fr_nitro_arom$$",40,3,38,2},
    {"fr_phenol_noOrthoHbond",43,1,40,0},
    {"fr_phenol_noOrthoHbond$",44,7,40,7},
    {"fr_phenol_noOrthoHbond$",51,4,47,3},
    {"fr_phenol_noOrthoHbond$",55,5,50,4},
    {"fr_phenol_noOrthoHbond$",60,5,54,4},
    {"fr_phos_ester",65,1,58,0},
    {"fr_phos_ester$",66,6,58,5},
    {"fr_phos_ester$$",72,1,63,0},
    {"fr_phos_ester$$",73,1,63,0},
    {"fr_phos_ester$$",74,2,63,1},
    {"fr_phos_ester$$",76,2,64,1},
    {"fr_phos_ester$",78,6,65,5},
};
constexpr int N_NAMED = 7;
constexpr Named NAMED[N_NAMED] = {
    {"fr_lactam",0},
    {"fr_benzodiazepine",1},
    {"fr_barbitur",2},
    {"fr_azo",3},
    {"fr_nitro_arom",4},
    {"fr_phenol_noOrthoHbond",8},
    {"fr_phos_ester",13},
};
constexpr Spec SPEC[N_NAMED] = {
    { "fr_lactam", "N1C(=O)CC1" },
    { "fr_benzodiazepine", "[c&R2]12[c&R1][c&R1][c&R1][c&R1][c&R2]1[N&R1][C&R1][C&R1][N&R1]=[C&R1]2" },
    { "fr_barbitur", "C1C(=O)NC(=O)NC1=O" },
    { "fr_azo", "[#6]-N=N-[#6]" },
    { "fr_nitro_arom", "[$(c1(-[$([NX3](=O)=O),$([NX3+](=O)[O-])])ccccc1)]" },
    { "fr_phenol_noOrthoHbond", "[$(c1(-[OX2H])ccccc1);!$(cc-!:[CH2]-[OX2H]);!$(cc-!:C(=O)[O;H1,-]);!$(cc-!:C(=O)-[NH2])]" },
    { "fr_phos_ester", 
      "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6"
      "]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$(["
      "OX1-]),$([OX2][#6]),$([OX2]P)])]" },
};
constexpr frag_prog_types::Program PROGRAM = {
    "fr_misc", SPEC_SHA256, NODES, AROOTS, QBONDS, PATTERNS, N_PATTERNS, NAMED, N_NAMED};
}  // namespace fr_prog

// ---------------------------------------------------------------------------------------------
// The molecule, in the boundary's own quantities.  Nothing here is perceived by this file.
// ---------------------------------------------------------------------------------------------
struct Mol {
  int n = 0, nb = 0;
  int chg_ok = 1;      // 0 = RDKit could not charge this molecule; charges are 0.0
  int n_rings = 0;     // rings in the boundary's (repaired) ring set -- ETA_eta_BR only
  std::vector<int32_t> z, deg, nH, fchg, hyb, arom, nring, tval, iso;
  std::vector<double> mass, chg;
  std::vector<int32_t> bu, bv, bcode, btype, bring;
  std::vector<double> bord;                 // GetBondTypeAsDouble()
  std::vector<int32_t> start, nbr, nbond;   // CSR over every atom, in bond index order

  void alloc(int na, int nbonds) {
    n = na; nb = nbonds;
    z.assign(na, 0); deg.assign(na, 0); nH.assign(na, 0); fchg.assign(na, 0);
    hyb.assign(na, 0); arom.assign(na, 0); nring.assign(na, 0); tval.assign(na, 0);
    iso.assign(na, 0); mass.assign(na, 0.0); chg.assign(na, 0.0);
    bu.assign(nbonds, 0); bv.assign(nbonds, 0); bcode.assign(nbonds, 0);
    btype.assign(nbonds, 0); bring.assign(nbonds, 0); bord.assign(nbonds, 0.0);
  }

  // CSR in ASCENDING BOND INDEX per atom, which MolecularId depends on: networkx's adjacency is
  // insertion-ordered and mordred inserts edges in `mol.GetBonds()` order, so an atom's
  // neighbours are visited in the order their bonds appear.  A counting sort over bonds gives
  // exactly that.
  void finish() {
    start.assign(n + 1, 0);
    for (int e = 0; e < nb; ++e) { start[bu[e] + 1]++; start[bv[e] + 1]++; }
    for (int i = 0; i < n; ++i) start[i + 1] += start[i];
    nbr.assign(start[n], 0);
    nbond.assign(start[n], 0);
    std::vector<int32_t> fill(start.begin(), start.end() - 1);
    for (int e = 0; e < nb; ++e) {
      nbr[fill[bu[e]]] = bv[e]; nbond[fill[bu[e]]++] = e;
      nbr[fill[bv[e]]] = bu[e]; nbond[fill[bv[e]]++] = e;
    }
  }
};

// The boundary's layout, so no caller has to remember which column is which.  Column indices are
// bindings.cpp's A_* / B_* enums; they are asserted against the strides rather than assumed.
inline void build_from_rows(Mol &m, int n, int nb, const int32_t *arows, int astride,
                            const double *adbl, int adstride, const int32_t *brows, int bstride,
                            const double *bord, int chg_ok, int n_rings) {
  if (astride < 13 || adstride < 2 || bstride < 6)
    throw std::invalid_argument("miscext::build_from_rows: boundary stride too small");
  m.alloc(n, nb);
  m.chg_ok = chg_ok;
  m.n_rings = n_rings;
  for (int i = 0; i < n; ++i) {
    const int32_t *r = arows + (ptrdiff_t)i * astride;
    m.z[i] = r[0]; m.deg[i] = r[1]; m.nH[i] = r[2]; m.fchg[i] = r[3]; m.hyb[i] = r[4];
    m.arom[i] = r[5]; m.nring[i] = r[8]; m.tval[i] = r[9]; m.iso[i] = r[12];
    m.mass[i] = adbl[(ptrdiff_t)i * adstride + 0];
    m.chg[i] = adbl[(ptrdiff_t)i * adstride + 1];
  }
  for (int e = 0; e < nb; ++e) {
    const int32_t *r = brows + (ptrdiff_t)e * bstride;
    m.bu[e] = r[0]; m.bv[e] = r[1]; m.bring[e] = r[3]; m.bcode[e] = r[4]; m.btype[e] = r[5];
    m.bord[e] = bord[e];
  }
  m.finish();
}

// ---------------------------------------------------------------------------------------------
// Scratch: everything reused across molecules so the timed loop does not allocate.
// ---------------------------------------------------------------------------------------------
struct Scratch {
  chisub::Mol chim;
  chisub::Scratch chis;
  pathcount::Mol pcm;
  pathcount::Scratch pcs;
  topomisc::Mol tpm;
  topomisc::Scratch tps;
  fragmatch::Mol fm;
  fragmatch::Matcher fmt;
  std::vector<int32_t> arows, brows;
  std::vector<double> prop;
  std::vector<int32_t> dist;      // n x n graph distance, LOCAL_INF for unreachable
  std::vector<int32_t> bfs, comp;
  std::vector<double> gamma, eps, alpha;
  std::vector<uint8_t> visited;
  std::vector<double> bal;        // n x n bond-order-weighted distance (BertzCT)
  std::vector<int32_t> symcls;
  std::vector<std::string> keys;
  std::unordered_map<std::string, int> keymap;      // key -> class, for `keysSeen.index`
  std::unordered_map<uint64_t, int> ckmap;         // packed connection key -> slot
  std::unordered_map<uint64_t, int> valmap;        // distance bits -> "%.4f" id
  std::unordered_map<std::string, int> strmap;     // "%.4f" text -> id
  std::vector<double> eigA, eigW; // adjacency matrix and its eigenvalues (LogEE_A)
  std::vector<double> chibuf, pcbuf, tpbuf;
  std::vector<int> fcount;
  Scratch() : fmt(fr_prog::PROGRAM), chibuf(chisub::N_COLS), pcbuf(pathcount::N_COLS),
              tpbuf(topomisc::N_COLS), fcount(fr_prog::N_NAMED) {}
};

namespace detail {

inline double qnan() { return std::numeric_limits<double>::quiet_NaN(); }

// RDKit's Code/GraphMol/MolOps.cpp fills unreachable pairs of getDistanceMat with 1e8, and
// mordred passes them straight through -- see topomisc.h note 4.  Same constant, same reason.
static constexpr int32_t LOCAL_INF = 100000000;

inline void distances(const Mol &m, Scratch &S) {
  const int n = m.n;
  S.dist.assign((size_t)n * n, LOCAL_INF);
  S.bfs.resize(n);
  for (int s = 0; s < n; ++s) {
    int32_t *d = &S.dist[(size_t)s * n];
    d[s] = 0;
    int head = 0, tail = 0;
    S.bfs[tail++] = s;
    while (head < tail) {
      const int u = S.bfs[head++];
      for (int k = m.start[u]; k < m.start[u + 1]; ++k) {
        const int v = m.nbr[k];
        if (d[v] == LOCAL_INF) { d[v] = d[u] + 1; S.bfs[tail++] = v; }
      }
    }
  }
}

//! len(Chem.GetMolFrags(mol)) -- the connected components of the graph as given.
inline int nFrags(const Mol &m, Scratch &S) {
  const int n = m.n;
  if (n == 0) return 0;
  S.comp.assign(n, -1);
  S.bfs.resize(n);
  int nc = 0;
  for (int s = 0; s < n; ++s) {
    if (S.comp[s] >= 0) continue;
    int head = 0, tail = 0;
    S.bfs[tail++] = s;
    S.comp[s] = nc;
    while (head < tail) {
      const int u = S.bfs[head++];
      for (int k = m.start[u]; k < m.start[u + 1]; ++k) {
        const int v = m.nbr[k];
        if (S.comp[v] < 0) { S.comp[v] = nc; S.bfs[tail++] = v; }
      }
    }
    ++nc;
  }
  return nc;
}

// `dx ** 2` in Python is float.__pow__, i.e. a libm pow() call and not a multiply.  Same device
// constit.h uses for the same expression; the volatile stops the compiler folding it.
inline double squareViaPow(double x) {
  volatile double two = 2.0;
  return std::pow(x, two);
}

// A MULTIPLY THAT CANNOT BE CONTRACTED INTO THE ADD NEXT TO IT.  clang's default
// -ffp-contract=on fuses `a + b * c` within one expression, and Python never does: without this
// VMcGowan and AETA_eta_BR were each off by one ulp on 2 of the first 200 corpus molecules --
// visible, tiny, and entirely a build-flag artefact.  Every place upstream writes a product and
// then adds it goes through here, so the header's answer does not depend on how it is compiled.
inline double mulNoFma(double a, double b) {
  volatile double t = a * b;
  return t;
}

// ---- S1 -------------------------------------------------------------------------------------

// rdkit GraphDescriptors.Chi0: `sum(numpy.sqrt(1./deltas))` over degrees with the ZEROS REMOVED.
// The outer `sum` is the BUILTIN, so the accumulation is left to right in atom order and starts
// from the integer 0 -- not numpy's pairwise sum.
inline double chi0(const Mol &m) {
  double acc = 0.0;
  for (int i = 0; i < m.n; ++i) {
    if (m.deg[i] == 0) continue;
    acc += std::sqrt(1.0 / (double)m.deg[i]);
  }
  return acc;
}

// rdkit Code/GraphMol/MolProps.cpp getExactMolWt(mol, onlyHeavy=false), transcribed including
// the two places the compiler contracts a multiply-add (notes 1 and 2 at the top).
inline double exactMolWt(const Mol &m) {
  double res = 0.0;
  int64_t nHs = 0;
  for (int i = 0; i < m.n; ++i) {
    const int z = m.z[i];
    if (z < 0 || z > constit_tbl::MAX_Z) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "miscext [ExactMolWt]: atom %d has atomic number %d, outside the "
                    "monoisotopic-mass table (0..%d)", i, z, (int)constit_tbl::MAX_Z);
      throw std::out_of_range(msg);
    }
    res += m.iso[i] ? m.mass[i] : constit_tbl::MONOISO[z];
    res = std::fma(-constit_tbl::ELECTRON_MASS, (double)m.fchg[i], res);
    nHs += m.nH[i];
  }
  return std::fma((double)nHs, constit_tbl::MONOISO[1], res);
}

// rdkit Descriptors.NumValenceElectrons: sum(GetNOuterElecs(Z) - charge + GetTotalNumHs()).
inline double numValenceElectrons(const Mol &m) {
  int64_t s = 0;
  for (int i = 0; i < m.n; ++i)
    s += chiwalk_tables::nOuterElecs(m.z[i]) - m.fchg[i] + m.nH[i];
  return (double)s;
}

// mordred VertexAdjacencyInformation: 1 + log2(#heavy-heavy bonds); NaN at zero, because
// `rethrow_zerodiv` runs np.log2 under errstate(divide="raise").
inline double vAdjMat(const Mol &m) {
  int64_t k = 0;
  for (int e = 0; e < m.nb; ++e) if (m.z[m.bu[e]] != 1 && m.z[m.bv[e]] != 1) ++k;
  if (k == 0) return qnan();
  return 1.0 + std::log2((double)k);
}

// mordred McGowanVolume: sum over the H-ADDED molecule (the base class leaves
// explicit_hydrogens = True) minus 6.56 per bond of that molecule.  `sum(generator)` is the
// builtin, so the order is heavy atoms as given, then the appended hydrogens.
inline double vMcGowan(const Mol &m) {
  double a = 0.0;
  int64_t nh = 0;
  for (int i = 0; i < m.n; ++i) {
    const int z = m.z[i];
    a += (z >= 1 && z <= 118) ? MCGOWAN_VOL[z] : qnan();
    nh += m.nH[i];
  }
  for (int64_t k = 0; k < nh; ++k) a += MCGOWAN_VOL[1];
  return a - mulNoFma((double)(m.nb + nh), 6.56);
}

// mordred ZagrebIndex(version, variable=-1).  V is the adjacency column sum, i.e. the degree.
//   mZagreb1 = np.sum(V ** -2)          -- numpy PAIRWISE sum, and 0 ** -2 raises -> NaN
//   mZagreb2 = sum((Vi*Vj) ** -1)       -- BUILTIN sum over bonds, in bond index order
inline void mZagreb(const Mol &m, Scratch &S, double *m1, double *m2) {
  S.prop.resize(m.n > 0 ? m.n : 1);
  bool bad = false;
  for (int i = 0; i < m.n; ++i) {
    if (m.deg[i] == 0) bad = true;
    S.prop[i] = std::pow((double)m.deg[i], -2.0);
  }
  *m1 = bad ? qnan() : topomisc::npPairwiseSum(S.prop.data(), m.n);
  double acc = 0.0;
  for (int e = 0; e < m.nb; ++e)
    acc += std::pow((double)m.deg[m.bu[e]] * (double)m.deg[m.bv[e]], -1.0);
  *m2 = acc;
}

// mordred EccentricConnectivityIndex and TopologicalIndex.Radius, both off the same
// eccentricity vector E = D.max(axis=0).  UNREACHABLE PAIRS ARE 1e8 AND ARE NOT SPECIAL-CASED:
// a salt really does get Radius 100000000.
inline void eccentricity(const Mol &m, Scratch &S, double *ecidx, double *radius) {
  const int n = m.n;
  int64_t s = 0;
  int32_t rmin = LOCAL_INF;
  for (int j = 0; j < n; ++j) {
    int32_t e = 0;
    for (int i = 0; i < n; ++i) { const int32_t d = S.dist[(size_t)i * n + j]; if (d > e) e = d; }
    s += (int64_t)e * (int64_t)m.deg[j];
    if (e < rmin) rmin = e;
  }
  *ecidx = (double)s;
  *radius = (double)(n == 0 ? 0 : rmin);
}

// mordred CarbonTypes.HybridizationRatio, over the same buckets constit.h's carbonTypes fills
// (but not capped at 8 carbon neighbours, since this one sums the whole row).
inline double hybRatio(const Mol &m) {
  int64_t n2 = 0, n3 = 0;
  for (int i = 0; i < m.n; ++i) {
    if (m.z[i] != 6) continue;
    switch (m.hyb[i]) {
      case 3: ++n2; break;                       // SP2
      case 4: case 6: case 7: ++n3; break;       // SP3, SP3D, SP3D2
      default: break;                            // SP and mordred's None bucket
    }
  }
  if (n2 == 0 && n3 == 0) return qnan();
  return (double)n3 / (double)(n2 + n3);
}

// mordred ConstitutionalSum('v'): np.sum(P / P_carbon) over the H-ADDED molecule, numpy's
// PAIRWISE summation.  Identical in shape to topomisc.h's Sp/Mv; only the table differs.
inline double sumVdw(const Mol &m, Scratch &S) {
  int64_t nh = 0;
  for (int i = 0; i < m.n; ++i) nh += m.nH[i];
  const int64_t tot = (int64_t)m.n + nh;
  S.prop.resize(tot > 0 ? (size_t)tot : 1);
  for (int i = 0; i < m.n; ++i)
    S.prop[i] = chiwalk_tables::vdwVol(m.z[i]) / chiwalk_tables::CARBON_VDW_VOL;
  for (int64_t k = m.n; k < tot; ++k)
    S.prop[(size_t)k] = chiwalk_tables::VDW_VOL[1] / chiwalk_tables::CARBON_VDW_VOL;
  return topomisc::npPairwiseSum(S.prop.data(), tot);
}

// ---- S5 -------------------------------------------------------------------------------------

// mordred MolecularDistanceEdge, generalised over (element, valence1, valence2).  Same shape as
// constit.h's molecularDistanceEdge, which is fixed to MDEC-22/23/33.
//   Dv = [D[i,j] for i<j if {V[i],V[j]} == {v1,v2} and Z[i] == Z[j] == element]
//   n / (product(Dv) ** (1/(2n))) ** 2,  NaN when n == 0 (1.0/(2.0*0) raises ZeroDivisionError)
inline double mde(const Mol &m, const Scratch &S, int elem, int v1, int v2) {
  const int n = m.n;
  double prod = 1.0;
  int64_t cnt = 0;
  for (int i = 0; i < n; ++i) {
    if (m.z[i] != elem) continue;
    for (int j = i + 1; j < n; ++j) {
      if (m.z[j] != elem) continue;
      const int di = m.deg[i], dj = m.deg[j];
      if (!((di == v1 && dj == v2) || (dj == v1 && di == v2))) continue;
      prod *= (double)S.dist[(size_t)i * n + j];
      ++cnt;
    }
  }
  if (cnt == 0) return qnan();
  const double dx = std::pow(prod, 1.0 / (2.0 * (double)cnt));
  return (double)cnt / squareViaPow(dx);
}

// ---- S6 -------------------------------------------------------------------------------------

// rdkit Descriptors._ChargeDescriptors, folded in atom order with the builtin two-argument
// min/max (note 4), and short-circuited to NaN when the boundary says the molecule could not be
// charged (note 3).
inline void partialCharges(const Mol &m, double *out) {
  if (!m.chg_ok) {
    out[C_MinPartialCharge] = out[C_MaxPartialCharge] = qnan();
    out[C_MinAbsPartialCharge] = out[C_MaxAbsPartialCharge] = qnan();
    return;
  }
  double mn = 500.0, mx = -500.0;
  for (int i = 0; i < m.n; ++i) {
    const double c = m.chg[i];
    // Python's two-argument min/max SEED WITH THE FIRST ARGUMENT and keep the second only on a
    // true comparison, so a NaN charge replaces the accumulator and is itself replaced by the
    // next atom.  Written this way rather than with std::min so that a boundary which stops
    // zeroing non-finite charges gets the upstream answer without a further change here.
    mn = (mn < c) ? mn : c;      // Python min(c, mn)
    mx = (mx > c) ? mx : c;      // Python max(c, mx)
  }
  out[C_MinPartialCharge] = mn;
  out[C_MaxPartialCharge] = mx;
  const double a = std::fabs(mn), b = std::fabs(mx);
  out[C_MinAbsPartialCharge] = a < b ? a : b;
  out[C_MaxAbsPartialCharge] = a > b ? a : b;
}

// ---- S8 -------------------------------------------------------------------------------------

// mordred _atomic_property.get_core_count: (Z - Zv) / (Zv * (PN - 1)), and 0.0 for hydrogen.
inline double coreCount(int z) {
  if (z == 1) return 0.0;
  const double zv = (double)chiwalk_tables::nOuterElecs(z);
  const double pn = (z >= 1 && z <= 118) ? PERIOD[z] : qnan();
  return ((double)z - zv) / (zv * (pn - 1.0));
}

inline double etaEpsilon(int z) {
  return 0.3 * (double)chiwalk_tables::nOuterElecs(z) - coreCount(z);
}

// How many former-aromatic bonds at atom i are DOUBLE after Chem.Kekulize -- the identity
// constit.h's nBondsKD uses and verified against RDKit on 4,000 molecules (0 mismatches, and the
// flag never left {0,1}).  Kekulization rewrites only TYPE-aromatic bonds and preserves every
// atom's valence, so
//     takesDouble(i) = tval - nH - round(non-aromatic valence contributions) - nAromaticBonds.
// A DATIVE bond contributes 0.0 to its donor and its order to its acceptor, which is why this
// cannot be a plain sum of GetBondTypeAsDouble().
//
// WHICH aromatic bond becomes the double one IS A KEKULE CHOICE AND IS NOT NEEDED.  Every
// former-aromatic bond contributes 2.0 to beta_ns if it is double and 0.0 if it is single, so an
// atom's total is 2.0 * takesDouble(i) whatever matching Kekulize picks.  That is why this file
// reconstructs a per-ATOM flag and never a per-BOND assignment -- the per-bond one would be
// ill-posed and the per-atom one is not.
inline void kekuleTakesDouble(const Mol &m, std::vector<uint8_t> &takes) {
  takes.assign(m.n, 0);
  for (int i = 0; i < m.n; ++i) {
    int narom = 0;
    double nonarom = 0.0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
      const int e = m.nbond[k];
      if (m.btype[e] == 12) { ++narom; continue; }             // Bond::AROMATIC
      nonarom += (m.bcode[e] == 0 && m.bu[e] == i) ? 0.0 : m.bord[e];   // dative donor gets none
    }
    if (!narom) continue;
    const int g = m.tval[i] - m.nH[i] - (int)std::floor(nonarom + 0.5) - narom;
    if (g < 0 || g > 1) {
      char msg[320];
      std::snprintf(msg, sizeof(msg),
                    "miscext [AETA_beta*/AETA_eta*]: the Kekule reconstruction constit.h's "
                    "nBondsKD relies on does not hold at atom %d (Z=%d): expected takesDouble "
                    "in {0,1}, got %d from tval=%d nH=%d nonAromaticValence=%.3f nAromatic=%d. "
                    "The molecule cannot be kekulized from the boundary's columns alone.",
                    i, (int)m.z[i], g, (int)m.tval[i], (int)m.nH[i], nonarom, narom);
      throw std::runtime_error(msg);
    }
    takes[i] = (uint8_t)g;
  }
}

// beta_s, beta_ns and beta_ns_d per atom, on the KEKULIZED molecule (note 5).
//   sigma_i   = sum over heavy NEIGHBOURS of (0.5 if |eps_i - eps_j| <= 0.3 else 0.75)
//   nonsigma  = sum over BONDS to heavy atoms of y * f, where a SINGLE bond contributes nothing,
//               f = 2.0 iff GetBondTypeAsDouble() == 3, and y = 2.0 for a bond flagged aromatic,
//               else 1.5 or 1.0 on |eps_i - eps_j| > 0.3
//   delta_i   = 0.5 iff i is not aromatic, not in a ring, has spare outer electrons, and has an
//               aromatic neighbour
// Every term is a multiple of 0.25 and the per-atom total is far below 2^53, so the order of
// accumulation is not observable here -- which is what lets the aromatic term be folded in at
// the end rather than in bond order.
inline void etaBeta(const Mol &m, const std::vector<double> &eps,
                    const std::vector<uint8_t> &takes, std::vector<double> &bs,
                    std::vector<double> &bns, std::vector<double> &bnsd) {
  const int n = m.n;
  bs.assign(n, 0.0); bns.assign(n, 0.0); bnsd.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double sigma = 0.0, nonsigma = 0.0;
    for (int k = m.start[i]; k < m.start[i + 1]; ++k) {
      const int j = m.nbr[k], e = m.nbond[k];
      if (m.z[j] == 1) continue;
      sigma += (std::fabs(eps[j] - eps[i]) <= 0.3) ? 0.5 : 0.75;
      if (m.btype[e] == 12) continue;               // kekulized below, 2.0 * takes[i]
      if (m.btype[e] == 1) continue;                // SINGLE contributes nothing
      const double f = (m.bord[e] == 3.0) ? 2.0 : 1.0;
      const double y = (m.bcode[e] & 8) ? 2.0
                                        : ((std::fabs(eps[i] - eps[j]) > 0.3) ? 1.5 : 1.0);
      nonsigma += y * f;
    }
    nonsigma += 2.0 * (double)takes[i];
    bs[i] = sigma / 2.0;
    double d = 0.0;
    if (!m.arom[i] && m.nring[i] == 0 &&
        chiwalk_tables::nOuterElecs(m.z[i]) - m.tval[i] > 0) {
      for (int k = m.start[i]; k < m.start[i + 1]; ++k)
        if (m.arom[m.nbr[k]]) { d = 0.5; break; }
    }
    bnsd[i] = d;
    bns[i] = nonsigma / 2.0 + d;
  }
}

// mordred EtaCompositeIndex: sum_{i<j, checker(r)} sqrt(gamma_i gamma_j / r^2), accumulated in
// the same nesting Python's two generators produce (inner over j, outer over i).
inline double etaComposite(int n, const std::vector<double> &gamma, const int32_t *D, bool local) {
  double outer = 0.0;
  for (int i = 0; i < n; ++i) {
    double inner = 0.0;
    for (int j = i + 1; j < n; ++j) {
      const int32_t r = D[(size_t)i * n + j];
      if (local ? (r != 1) : (r == 0)) continue;
      const double rr = (double)r;
      inner += std::sqrt(gamma[i] * gamma[j] / (rr * rr));
    }
    outer += inner;
  }
  return outer;
}

// ---- S9 -------------------------------------------------------------------------------------

// mordred MolecularId.AtomicId._search, transliterated.  `lim` is `int(1.0 / eps**2)` for
// eps = 1e-10 EVALUATED IN FLOATING POINT, which is 99999999999999983616 and not 1e20; the
// comparison `w < lim` is an exact integer one, so the difference is real.  The running product
// can reach lim * 81 > 2^63, hence __int128.
struct MidWalk {
  const Mol *m;
  const int32_t *w;               // per half-edge: deg(u) * deg(v)
  std::vector<uint8_t> *visited;
  double id;
  // int(1.0 / 1e-10 ** 2) EVALUATED IN FLOATING POINT is 99999999999999983616, not 10^20, and
  // `w < lim` is an exact integer comparison -- so the difference is observable.  Written as
  // 5 * 2^64 + 7766279631452225536 because the value does not fit a 64-bit literal; the two
  // static_asserts below pin its decimal digits.
  static constexpr unsigned __int128 LIM =
      ((unsigned __int128)5 << 64) + (unsigned __int128)7766279631452225536ULL;
  static_assert((uint64_t)(LIM / 10000000000ULL) == 9999999999ULL, "MidWalk::LIM high digits");
  static_assert((uint64_t)(LIM % 10000000000ULL) == 9999983616ULL, "MidWalk::LIM low digits");

  void search(int u, unsigned __int128 acc) {
    for (int k = m->start[u]; k < m->start[u + 1]; ++k) {
      const int v = m->nbr[k];
      if ((*visited)[v]) continue;
      (*visited)[v] = 1;
      const unsigned __int128 nw = acc * (unsigned __int128)w[k];
      id += 1.0 / std::sqrt((double)nw);
      if (nw < LIM) search(v, nw);
      (*visited)[v] = 0;
    }
  }
};

// ---- S10 ------------------------------------------------------------------------------------

// rdkit Code/ML/InfoTheory/InfoGainFuncs.h::InfoEntropy, WITH THE FMA THE COMPILER EMITS.  See
// note 1: the uncontracted form disagrees with the running rdkit on 27% of random count vectors.
inline double infoEntropy(const double *v, int n) {
  double nInstances = 0.0;
  for (int i = 0; i < n; ++i) nInstances += v[i];
  double accum = 0.0;
  if (nInstances != 0.0) {
    for (int i = 0; i < n; ++i) {
      const double d = v[i] / nInstances;
      if (d != 0.0) accum = std::fma(-d, std::log(d), accum);
    }
  }
  return accum / std::log(2.0);
}

// rdkit Code/GraphMol/Matrices.cpp::getDistanceMat(useBO=true) -- the "Balaban" matrix
// _AssignSymmetryClasses reads.  An AROMATIC-FLAGGED bond contributes 2/3, NOT 1/1.5; every
// other bond contributes 1/GetBondTypeAsDouble().  The relaxation is upstream's own
// double-buffered Floyd-Warshall, reproduced rather than replaced by a Dijkstra: the value of a
// path is assembled from sub-path sums in the k-order, and a different association would move
// the last bits of a number that is then formatted to 4 decimals and compared as a STRING.
inline void balabanDistances(const Mol &m, std::vector<double> &D) {
  const int n = m.n;
  const double INF = (double)LOCAL_INF;
  D.assign((size_t)n * n, INF);
  for (int i = 0; i < n; ++i) D[(size_t)i * n + i] = 0.0;
  for (int e = 0; e < m.nb; ++e) {
    const double contrib = (m.bcode[e] & 8) ? (2.0 / 3.0) : (1.0 / m.bord[e]);
    D[(size_t)m.bu[e] * n + m.bv[e]] = contrib;
    D[(size_t)m.bv[e] * n + m.bu[e]] = contrib;
  }
  // UPSTREAM DOUBLE-BUFFERS (lastD / currD) AND THIS RELAXES IN PLACE.  That substitution is
  // exact, not an approximation: at step k, row k is unchanged, because
  // cur[k][j] = min(last[k][j], last[k][k] + last[k][j]) and last[k][k] is 0.  Every read an
  // in-place sweep makes of row k therefore sees the same numbers the buffered sweep would, so
  // the two produce bit-identical matrices -- while the in-place one does not write n^2 doubles
  // into a second buffer and swap it n times.  Verified empirically as well: BertzCT stays
  // 20,000/20,000 bit-identical against rdkit after the change.
  //
  // The two skips are exact for the same kind of reason.  Every entry is <= LOCAL_INF by
  // construction, so when d[i][k] == LOCAL_INF the candidate v2 = INF + d[k][j] >= INF >= v1 and
  // the row cannot change; and for i == k, v2 = 0 + v1 = v1 and `v1 <= v2` keeps v1.
  for (int k = 0; k < n; ++k) {
    const double *dk = &D[(size_t)k * n];
    for (int i = 0; i < n; ++i) {
      if (i == k) continue;
      double *di = &D[(size_t)i * n];
      const double ik = di[k];
      if (ik >= INF) continue;
      for (int j = 0; j < n; ++j) {
        const double v2 = ik + dk[j];
        if (v2 < di[j]) di[j] = v2;
      }
    }
  }
}

// rdkit GraphDescriptors.BertzCT, at cutoff = 100 (mordred passes only `dMat`, which
// forceDMat = 1 then discards).
inline double bertzCT(const Mol &m, Scratch &S) {
  const int n = m.n;
  if (n < 2) return 0.0;                       // upstream returns the integer 0

  // --- symmetry classes: the sorted Balaban row, formatted "%.4f", first 100 entries ----------
  balabanDistances(m, S.bal);
  S.symcls.assign(n, 0);
  S.keys.clear();
  S.keymap.clear();
  // TWO EXACT SUBSTITUTIONS, both pure speed, and neither changes which atoms share a class.
  //
  //  (a) `'%.4f' % x` IS FORMATTED ONCE PER DISTINCT DOUBLE, not once per matrix entry.  The
  //      key is a tuple of formatted strings compared for equality, so interning each distinct
  //      double to the id of its formatted string gives an identical equivalence relation --
  //      including the case that two DIFFERENT doubles format alike, which is why the id is
  //      interned through the STRING and not through the bit pattern.  A distance matrix has a
  //      handful of distinct values and n^2 entries, so this replaces ~n*100 snprintf calls per
  //      molecule with a few dozen.
  //  (b) only the first `cutoff` = 100 sorted values are read, so for n > 100 a partial_sort of
  //      the first 100 is the same tuple as a full sort.
  std::vector<double> row(n);
  std::vector<int32_t> ids;
  std::string key;
  char buf[64];
  S.valmap.clear();
  S.strmap.clear();
  const int lim = n < 100 ? n : 100;
  ids.resize(lim);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) row[j] = S.bal[(size_t)i * n + j];
    if (lim < n) std::partial_sort(row.begin(), row.begin() + lim, row.end());
    else std::sort(row.begin(), row.end());
    for (int j = 0; j < lim; ++j) {
      uint64_t bits;
      std::memcpy(&bits, &row[j], 8);
      std::unordered_map<uint64_t, int>::const_iterator vt = S.valmap.find(bits);
      int id;
      if (vt != S.valmap.end()) {
        id = vt->second;
      } else {
        std::snprintf(buf, sizeof(buf), "%.4f", row[j]);
        std::string sv(buf);
        std::unordered_map<std::string, int>::const_iterator st = S.strmap.find(sv);
        if (st != S.strmap.end()) id = st->second;
        else { id = (int)S.strmap.size(); S.strmap.insert(std::make_pair(sv, id)); }
        S.valmap.insert(std::make_pair(bits, id));
      }
      ids[j] = id;
    }
    key.assign(reinterpret_cast<const char *>(ids.data()), (size_t)lim * sizeof(int32_t));
    // `keysSeen.index(theKey)` is a linear scan upstream; a hash lookup answers the same
    // question -- FIRST APPEARANCE ORDER is preserved by `keys`, which is what the class number
    // is.
    std::unordered_map<std::string, int>::const_iterator it = S.keymap.find(key);
    int idx;
    if (it == S.keymap.end()) {
      idx = (int)S.keys.size();
      S.keys.push_back(key);
      S.keymap.insert(std::make_pair(key, idx));
    } else {
      idx = it->second;
    }
    S.symcls[i] = idx + 1;
  }

  // --- the two dictionaries, in Python's INSERTION ORDER --------------------------------------
  // _LookUpBondOrder: 1.5 when the bond is aromatic (flag or type), else float(BondType enum) --
  // so a DATIVE bond has "order" 17.0.  That is upstream's, and it is reproduced.
  std::vector<double> bo(m.nb);
  for (int e = 0; e < m.nb; ++e)
    bo[e] = ((m.bcode[e] & 8) || m.btype[e] == 12) ? 1.5 : (double)m.btype[e];

  std::vector<int32_t> atypeZ;
  std::vector<int64_t> atypeN;
  // connectionDict, as Python has it: a dict keyed by a 2-tuple OR a 3-tuple (they never
  // collide, and _CalculateEntropies reads `.values()`, i.e. INSERTION ORDER).  `cval` is that
  // value list in insertion order; `ckmap` only accelerates the lookup.  Symmetry classes are
  // 1..n, so 20 bits each is ample and the pack cannot alias a 2-tuple onto a 3-tuple.
  S.ckmap.clear();
  std::vector<double> cval;
  auto bump = [&](int len, int a, int b, int c, double add) {
    if (a >= (1 << 20) || b >= (1 << 20) || c >= (1 << 20)) {
      char msg[240];
      std::snprintf(msg, sizeof(msg),
                    "miscext [BertzCT]: symmetry classes (%d,%d,%d) exceed the 20-bit "
                    "connection-key pack; the molecule has more than 2^20 symmetry classes, "
                    "which needs a wider key here.", a, b, c);
      throw std::runtime_error(msg);
    }
    const uint64_t k = ((uint64_t)len << 60) | ((uint64_t)a << 40) | ((uint64_t)b << 20) |
                       (uint64_t)c;
    std::unordered_map<uint64_t, int>::const_iterator it = S.ckmap.find(k);
    if (it == S.ckmap.end()) {
      S.ckmap.insert(std::make_pair(k, (int)cval.size()));
      cval.push_back(add);
    } else {
      cval[it->second] += add;
    }
  };

  for (int i = 0; i < n; ++i) {
    const int z = m.z[i];
    bool seen = false;
    for (size_t k = 0; k < atypeZ.size(); ++k)
      if (atypeZ[k] == z) { atypeN[k] += 1; seen = true; break; }
    if (!seen) { atypeZ.push_back(z); atypeN.push_back(1); }

    const int hinge = S.symcls[i];
    // `nList[i]` is built by appending each new neighbour and then SORTED ascending, so the
    // (i, j) double loop below runs over neighbours in ascending atom index.
    std::vector<std::pair<int, int> > nb;        // (neighbour atom, bond index)
    for (int k = m.start[i]; k < m.start[i + 1]; ++k)
      nb.push_back(std::make_pair((int)m.nbr[k], (int)m.nbond[k]));
    std::sort(nb.begin(), nb.end());
    const int deg = (int)nb.size();
    for (int a = 0; a < deg; ++a) {
      const int ni = nb[a].first, nic = S.symcls[ni];
      const double boi = bo[nb[a].second];
      if (boi > 1.0 && ni > i) {
        const double numConn = boi * (boi - 1.0) / 2.0;
        bump(2, hinge < nic ? hinge : nic, hinge < nic ? nic : hinge, 0, numConn);
      }
      for (int b = a + 1; b < deg; ++b) {
        const int njc = S.symcls[nb[b].first];
        const double numConn = boi * bo[nb[b].second];
        bump(3, nic < njc ? nic : njc, hinge, nic < njc ? njc : nic, numConn);
      }
    }
  }
  if (cval.empty()) cval.push_back(1.0);        // upstream's {'a': 1}

  double tot = 0.0;
  for (size_t k = 0; k < cval.size(); ++k) tot += cval[k];
  const double connectionIE = tot * (infoEntropy(cval.data(), (int)cval.size()) +
                                     std::log(tot) / std::log(2.0));
  std::vector<double> at(atypeN.size());
  for (size_t k = 0; k < atypeN.size(); ++k) at[k] = (double)atypeN[k];
  const double atomTypeIE = (double)n * infoEntropy(at.data(), (int)at.size());
  return atomTypeIE + connectionIE;
}

// ---- S11 ------------------------------------------------------------------------------------

// mordred AdjacencyMatrix('LogEE') -- the Estrada-like index of the adjacency spectrum, computed
// in the log-sum-exp form _matrix_attributes.LogEE uses:
//     a = max(largest eigenvalue, 0);  sx = sum(exp(val - a)) + exp(-a);  LogEE = a + log(sx)
// The eigenvalues come from cpp/eigen_small.h -- LAPACK's dsytd2 + dsterf, written out -- and
// NOT from LAPACK's dsyevd, which is what numpy's eigh calls.  THIS COLUMN IS THEREFORE NOT
// BIT-EXACT and cannot be made so without shipping a divide-and-conquer eigensolver; see
// NOTES_misc.md for the measured agreement.  The sum is numpy's PAIRWISE one over the ASCENDING
// spectrum, which is the order eigh returns.
inline double logEE_A(const Mol &m, Scratch &S) {
  const int n = m.n;
  if (n <= 0) return qnan();
  S.eigA.assign((size_t)n * n, 0.0);
  for (int e = 0; e < m.nb; ++e) {
    S.eigA[(size_t)m.bu[e] * n + m.bv[e]] += 1.0;
    S.eigA[(size_t)m.bv[e] * n + m.bu[e]] += 1.0;
  }
  S.eigW.assign(n, 0.0);
  if (n == 1) {
    S.eigW[0] = S.eigA[0];
  } else if (n == 2) {
    double r1, r2;
    hume_eig::lae2(S.eigA[0], S.eigA[1], S.eigA[3], &r1, &r2);
    S.eigW[0] = r1 < r2 ? r1 : r2;
    S.eigW[1] = r1 < r2 ? r2 : r1;
  } else {
    static thread_local hume_eig::Work W;
    W.ensure(n);
    double *M = W.a.data();
    for (int j = 0; j < n; ++j)
      for (int k = 0; k <= j; ++k) M[(size_t)j * n + k] = S.eigA[(size_t)j * n + k];
    hume_eig::sytd2_upper(M, n, n, W.d.data(), W.e.data(), W.tau.data(), W.wk.data());
    double lo, hi;
    if (!hume_eig::sterf_min_max(n, W.d.data(), W.e.data(), &lo, &hi)) return qnan();
    for (int i = 0; i < n; ++i) S.eigW[i] = W.d[i];
    std::sort(S.eigW.begin(), S.eigW.end());     // dsterf finishes with dlasrt; eigh is ascending
  }
  double a = S.eigW[n - 1];
  if (a < 0.0) a = 0.0;                          // np.maximum(val[argmax], 0)
  S.prop.resize(n);
  for (int i = 0; i < n; ++i) S.prop[i] = std::exp(S.eigW[i] - a);
  const double sx = topomisc::npPairwiseSum(S.prop.data(), n) + std::exp(-a);
  return a + std::log(sx);
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------
// All 81 columns for one molecule.
// ---------------------------------------------------------------------------------------------
inline void compute(const Mol &m, Scratch &S, double *out) {
  using namespace detail;
  const int n = m.n;
  const double NAN_ = qnan();
  for (int c = 0; c < N_COLS; ++c) out[c] = NAN_;

  const int nfrag = nFrags(m, S);
  const bool connected = (nfrag == 1);
  distances(m, S);

  // ---- S6 partial charges --------------------------------------------------------------------
  partialCharges(m, out);

  // ---- S1 scalars ----------------------------------------------------------------------------
  out[C_Chi0] = chi0(m);
  out[C_ExactMolWt] = exactMolWt(m);
  out[C_NumValenceElectrons] = numValenceElectrons(m);
  out[C_VAdjMat] = vAdjMat(m);
  out[C_VMcGowan] = vMcGowan(m);
  mZagreb(m, S, &out[C_mZagreb1], &out[C_mZagreb2]);
  eccentricity(m, S, &out[C_ECIndex], &out[C_Radius]);
  out[C_HybRatio] = hybRatio(m);
  out[C_Sv] = sumVdw(m, S);

  // ---- S5 molecular distance edge ------------------------------------------------------------
  out[C_MDEC11] = mde(m, S, 6, 1, 1);
  out[C_MDEC12] = mde(m, S, 6, 1, 2);
  out[C_MDEC13] = mde(m, S, 6, 1, 3);
  out[C_MDEO11] = mde(m, S, 8, 1, 1);
  out[C_MDEN22] = mde(m, S, 7, 2, 2);

  // ---- S2 Chi --------------------------------------------------------------------------------
  // chi.h fills sum[prop][shape][order] / cnt / bad for EVERY bucket and then emits its own 40;
  // these 13 are other buckets of the same accumulation, read straight out of its Scratch.
  {
    S.arows.resize((size_t)n * 4);
    for (int i = 0; i < n; ++i) {
      int32_t *r = &S.arows[(size_t)i * 4];
      r[0] = m.z[i]; r[1] = m.deg[i]; r[2] = m.nH[i]; r[3] = m.fchg[i];
    }
    S.brows.resize((size_t)m.nb * 2);
    for (int e = 0; e < m.nb; ++e) {
      S.brows[(size_t)e * 2] = m.bu[e];
      S.brows[(size_t)e * 2 + 1] = m.bv[e];
    }
    chisub::build_from_rows(S.chim, n, m.nb, S.arows.data(), 4, S.brows.data(), 2);
    chisub::compute(S.chim, S.chibuf.data(), S.chis);
    struct CS { int col, shape, order, prop, avg; };
    static const CS CC[13] = {
        {C_Xch3d, chisub::CHAIN, 3, chisub::D, 0},
        {C_Xch3dv, chisub::CHAIN, 3, chisub::DV, 0},
        {C_Xch4dv, chisub::CHAIN, 4, chisub::DV, 0},
        {C_Xc6dv, chisub::CLUSTER, 6, chisub::DV, 0},
        {C_Xpc6d, chisub::PATH_CLUSTER, 6, chisub::D, 0},
        {C_Xp1d, chisub::PATH, 1, chisub::D, 0},
        {C_Xp3d, chisub::PATH, 3, chisub::D, 0},
        {C_Xp7d, chisub::PATH, 7, chisub::D, 0},
        {C_AXp0d, chisub::PATH, 0, chisub::D, 1},
        {C_Xp2dv, chisub::PATH, 2, chisub::DV, 0},
        {C_Xp3dv, chisub::PATH, 3, chisub::DV, 0},
        {C_Xp4dv, chisub::PATH, 4, chisub::DV, 0},
        {C_Xp7dv, chisub::PATH, 7, chisub::DV, 0},
    };
    for (int k = 0; k < 13; ++k) {
      const CS &c = CC[k];
      if (S.chis.bad[c.prop][c.shape][c.order]) { out[c.col] = NAN_; continue; }
      const double x = S.chis.sum[c.prop][c.shape][c.order];
      if (!c.avg) { out[c.col] = x; continue; }
      const int64_t k2 = S.chis.cnt[c.shape][c.order];
      out[c.col] = k2 == 0 ? NAN_ : x / (double)k2;
    }
  }

  // ---- S3 PathCount --------------------------------------------------------------------------
  {
    S.brows.resize((size_t)m.nb * 2);
    for (int e = 0; e < m.nb; ++e) {
      S.brows[(size_t)e * 2] = m.bu[e];
      S.brows[(size_t)e * 2 + 1] = m.bv[e];
    }
    pathcount::build_from_rows(S.pcm, n, m.nb, S.brows.data(), 2, 0, 1, m.bord.data(),
                               m.z.data(), 1, 0);
    pathcount::compute(S.pcm, S.pcbuf.data(), S.pcs);
    out[C_MPC5] = (double)(S.pcs.cnt[5] / 2);
    out[C_MPC7] = (double)(S.pcs.cnt[7] / 2);
    out[C_MPC8] = (double)(S.pcs.cnt[8] / 2);
    out[C_MPC10] = (double)(S.pcs.cnt[10] / 2);
    out[C_piPC7] = std::log(S.pcs.w[7] * 0.5 + 1.0);
    out[C_piPC9] = std::log(S.pcs.w[9] * 0.5 + 1.0);
    // The `total` variants recurse acc_k = acc_{k-1} + PC_k from order 0 upwards, so the
    // accumulation order is part of the definition.  acc_0 is int(A) for MPC and float(A) for
    // piPC, and only the LAST term of TpiPC10 takes the log.
    int64_t tm = n;
    for (int k = 1; k <= 10; ++k) tm += S.pcs.cnt[k] / 2;
    out[C_TMPC10] = (double)tm;
    double tp = (double)n;
    for (int k = 1; k <= 10; ++k) tp += S.pcs.w[k] * 0.5;
    out[C_TpiPC10] = std::log(tp + 1.0);
  }

  // ---- S4 WalkCount --------------------------------------------------------------------------
  {
    S.arows.resize((size_t)n * 3);
    for (int i = 0; i < n; ++i) {
      int32_t *r = &S.arows[(size_t)i * 3];
      r[0] = m.z[i]; r[1] = 0; r[2] = m.nH[i];
    }
    S.brows.resize((size_t)m.nb * 2);
    for (int e = 0; e < m.nb; ++e) {
      S.brows[(size_t)e * 2] = m.bu[e];
      S.brows[(size_t)e * 2 + 1] = m.bv[e];
    }
    topomisc::build_from_rows(S.tpm, n, m.nb, S.arows.data(), 3, S.brows.data(), 2);
    int64_t tr[11] = {0}, sums[11] = {0};
    if (n > 0) topomisc::detail::walkTraces(S.tpm, S.tps, tr, sums);
    out[C_MWC02] = std::log((double)(sums[2] + 1));
    out[C_MWC04] = std::log((double)(sums[4] + 1));
    out[C_MWC07] = std::log((double)(sums[7] + 1));
    out[C_MWC09] = std::log((double)(sums[9] + 1));
    // TMWC10 = A + MWC01 + sum_{k=2..10} MWC0k, accumulated upwards.  MWC01 is 0.5 * A.sum(),
    // NOT a log -- the log starts at order 2.
    double t = (double)n + mulNoFma(0.5, (double)sums[1]);
    for (int k = 2; k <= 10; ++k) t = t + std::log((double)(sums[k] + 1));
    out[C_TMWC10] = t;
  }

  // ---- S7 fr_* -------------------------------------------------------------------------------
  {
    S.fm.alloc(n, m.nb);
    for (int i = 0; i < n; ++i) {
      S.fm.z[i] = m.z[i]; S.fm.deg[i] = m.deg[i]; S.fm.nH[i] = m.nH[i];
      S.fm.fchg[i] = m.fchg[i]; S.fm.arom[i] = m.arom[i]; S.fm.nring[i] = m.nring[i];
      S.fm.tval[i] = m.tval[i]; S.fm.iso[i] = m.iso[i];
    }
    for (int e = 0; e < m.nb; ++e) {
      S.fm.bu[e] = m.bu[e]; S.fm.bv[e] = m.bv[e];
      S.fm.border[e] = m.btype[e]; S.fm.bring[e] = m.bring[e];
    }
    S.fm.finish();
    fragmatch::countAll(S.fm, S.fmt, S.fcount.data());
    // The seven counts are written as a contiguous run, so the ONE way this can go wrong
    // silently is the program's NAMED order drifting from the column order.  Checked once per
    // process against col_name() rather than assumed -- a transposed fr_* block is exactly the
    // defect that does not announce itself.
    static const bool order_ok = []() {
      for (int i = 0; i < fr_prog::N_NAMED; ++i)
        if (std::strcmp(fr_prog::NAMED[i].name, col_name(C_fr_lactam + i)) != 0)
          throw std::runtime_error(
              std::string("miscext: fr_* program order drifted -- program has '") +
              fr_prog::NAMED[i].name + "' where the columns have '" +
              col_name(C_fr_lactam + i) + "'");
      return true;
    }();
    (void)order_ok;
    for (int i = 0; i < fr_prog::N_NAMED; ++i)
      out[C_fr_lactam + i] = (double)S.fcount[i];
  }

  // ---- S8 ETA --------------------------------------------------------------------------------
  if (connected && n > 0) {
    std::vector<uint8_t> takes;
    kekuleTakesDouble(m, takes);
    S.eps.resize(n);
    S.alpha.resize(n);
    for (int i = 0; i < n; ++i) { S.eps[i] = etaEpsilon(m.z[i]); S.alpha[i] = coreCount(m.z[i]); }
    std::vector<double> bs, bns, bnsd;
    etaBeta(m, S.eps, takes, bs, bns, bnsd);

    double alpha = 0.0, beta = 0.0, beta_s = 0.0, beta_ns = 0.0, beta_nsd = 0.0;
    for (int i = 0; i < n; ++i) {
      alpha += S.alpha[i];
      beta += bs[i] + bns[i];
      beta_s += bs[i];
      beta_ns += bns[i];
      beta_nsd += bnsd[i];
    }
    const double A = (double)n;
    out[C_AETA_alpha] = alpha / A;
    out[C_AETA_beta] = beta / A;
    out[C_AETA_beta_s] = beta_s / A;
    out[C_AETA_beta_ns] = beta_ns / A;
    out[C_AETA_beta_ns_d] = beta_nsd / A;
    out[C_AETA_dBeta] = (beta_ns - beta_s) / A;

    // get_eta_gamma sums the three UNHALVED contributions: sigma + nonsigma + delta.  bs/bns
    // above are already halved for the beta columns, so they are doubled back here rather than
    // recomputed.
    S.gamma.resize(n);
    for (int i = 0; i < n; ++i) {
      const double b = bs[i] * 2.0 + (bns[i] - bnsd[i]) * 2.0 + bnsd[i];
      S.gamma[i] = (b == 0.0) ? NAN_ : S.alpha[i] / b;
    }
    const double eta = etaComposite(n, S.gamma, S.dist.data(), false);
    const double eta_L = etaComposite(n, S.gamma, S.dist.data(), true);
    out[C_AETA_eta] = eta / A;
    out[C_AETA_eta_L] = eta_L / A;

    // The reference alkane: the same graph with every heavy atom a carbon and every bond single,
    // hydrogens DROPPED.  mordred fails the whole descriptor when any bond endpoint has degree
    // > 4, and it counts the averaged reference eta over the REFERENCE molecule's atoms.
    bool refok = true;
    for (int e = 0; e < m.nb; ++e)
      if (m.deg[m.bu[e]] > 4 || m.deg[m.bv[e]] > 4) { refok = false; break; }
    if (refok) {
      // map heavy atoms to reference indices
      std::vector<int32_t> idx(n, -1);
      int rn = 0;
      for (int i = 0; i < n; ++i) if (m.z[i] != 1) idx[i] = rn++;
      std::vector<int32_t> rdeg(rn, 0), ru, rv;
      for (int e = 0; e < m.nb; ++e) {
        const int i = idx[m.bu[e]], j = idx[m.bv[e]];
        if (i < 0 || j < 0) continue;
        ru.push_back(i); rv.push_back(j);
        rdeg[i]++; rdeg[j]++;
      }
      // A carbon in the reference alkane has alpha = 0.5, beta = 0.5 * degree, no non-sigma and
      // no delta term (4 outer electrons - a total valence of 4 is not > 0), so gamma = 1/degree.
      std::vector<double> rg(rn);
      for (int i = 0; i < rn; ++i) rg[i] = rdeg[i] == 0 ? NAN_ : 0.5 / (0.5 * (double)rdeg[i]);
      // distances on the reference graph
      std::vector<int32_t> rd((size_t)rn * rn, LOCAL_INF);
      {
        std::vector<int32_t> rs(rn + 1, 0), rnbr;
        for (size_t e = 0; e < ru.size(); ++e) { rs[ru[e] + 1]++; rs[rv[e] + 1]++; }
        for (int i = 0; i < rn; ++i) rs[i + 1] += rs[i];
        rnbr.assign(rs[rn], 0);
        std::vector<int32_t> fill(rs.begin(), rs.end() - 1);
        for (size_t e = 0; e < ru.size(); ++e) {
          rnbr[fill[ru[e]]++] = rv[e];
          rnbr[fill[rv[e]]++] = ru[e];
        }
        std::vector<int32_t> q(rn);
        for (int s = 0; s < rn; ++s) {
          int32_t *d = &rd[(size_t)s * rn];
          d[s] = 0;
          int head = 0, tail = 0;
          q[tail++] = s;
          while (head < tail) {
            const int u = q[head++];
            for (int k = rs[u]; k < rs[u + 1]; ++k)
              if (d[rnbr[k]] == LOCAL_INF) { d[rnbr[k]] = d[u] + 1; q[tail++] = rnbr[k]; }
          }
        }
      }
      const double eta_R = etaComposite(rn, rg, rd.data(), false);
      const double eta_RL = etaComposite(rn, rg, rd.data(), true);
      out[C_AETA_eta_R] = rn ? eta_R / (double)rn : NAN_;
      out[C_AETA_eta_RL] = rn ? eta_RL / (double)rn : NAN_;
      out[C_AETA_eta_F] = (eta_R - eta) / A;
      out[C_AETA_eta_FL] = (eta_RL - eta_L) / A;
      if (n > 1) {
        const double eta_NL =
            (n == 2) ? 1.0 : (std::sqrt(2.0) + mulNoFma(0.5, (double)n - 3.0));
        out[C_AETA_eta_B] = (eta_NL - eta_RL + mulNoFma(0.086, 0.0)) / A;
        out[C_AETA_eta_BR] =
            (eta_NL - eta_RL + mulNoFma(0.086, (double)m.n_rings)) / A;
      }
    }
  }

  // ---- S9 MolecularId ------------------------------------------------------------------------
  // DROPPED IN THE COST TRIAGE (METHODS 5.2) AND THEREFORE NOT COMPUTED. 55.6 us/mol measured by
  // difference, and the reason is the exhaustive weighted-path walk below, whose worst case on
  // this corpus is 16 ms on a single 76-atom bridged cage. -DMISC_WANT_DROPPED restores it.
#ifdef MISC_WANT_DROPPED
  if (connected && n > 0) {
    std::vector<int32_t> w(m.start[n]);
    for (int i = 0; i < n; ++i)
      for (int k = m.start[i]; k < m.start[i + 1]; ++k)
        w[k] = m.deg[i] * m.deg[m.nbr[k]];
    S.visited.assign(n, 0);
    std::vector<double> aid(n);
    MidWalk W;
    W.m = &m; W.w = w.data(); W.visited = &S.visited;
    for (int s = 0; s < n; ++s) {
      std::fill(S.visited.begin(), S.visited.end(), (uint8_t)0);
      S.visited[s] = 1;
      W.id = 0.0;
      W.search(s, (unsigned __int128)1);
      aid[s] = 1.0 + W.id / 2.0;
    }
    // any, hetero, C, N, O, X -- summed with the builtin sum in atom order, then averaged.
    static const int COLS[6][2] = {{C_MID, C_AMID},   {C_MID_h, C_AMID_h}, {C_MID_C, C_AMID_C},
                                   {C_MID_N, C_AMID_N}, {C_MID_O, C_AMID_O}, {C_MID_X, C_AMID_X}};
    for (int t = 0; t < 6; ++t) {
      double v = 0.0;
      for (int i = 0; i < n; ++i) {
        const int z = m.z[i];
        bool take = false;
        switch (t) {
          case 0: take = true; break;
          case 1: take = (z != 1 && z != 6); break;
          case 2: take = (z == 6); break;
          case 3: take = (z == 7); break;
          case 4: take = (z == 8); break;
          default:
            take = (z == 9 || z == 17 || z == 35 || z == 53 || z == 85 || z == 117);
            break;
        }
        if (take) v += aid[i];
      }
      out[COLS[t][0]] = v;
      out[COLS[t][1]] = v / (double)n;
    }
  }

#endif
  // ---- S10 BertzCT ---------------------------------------------------------------------------
  // DROPPED IN THE COST TRIAGE (METHODS 5.2) AND THEREFORE NOT COMPUTED. Skipping the copy-out
  // in bindings.cpp was not enough: this call is 55.5 us/mol on its own, the highest per-column
  // cost anywhere in the package, and it was still being paid for a column nothing emits.
  // -DMISC_WANT_DROPPED restores all three clusters for verification against mordred.
#ifdef MISC_WANT_DROPPED
  out[C_BertzCT] = bertzCT(m, S);
  out[C_LogEE_A] = connected ? logEE_A(m, S) : NAN_;
#else
  out[C_BertzCT] = NAN_;
  out[C_LogEE_A] = NAN_;
#endif
}

}  // namespace miscext

#endif  // HUME_MISC_EXT_H
