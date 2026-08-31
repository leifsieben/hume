// E_counts: the 31 census columns of data/dedupe2 that survived the second dedupe and had no
// implementation -- nine scalar counts, twenty-one more RingCount predicates, and nRot.
//
//   Aromatic     2   nAromAtom nAromBond
//   AtomCount    5   nAtom nHeavyAtom nP nF nI
//   BondCount    2   nBonds nBondsKS
//   RingCount   21   n8Ring ... n12FAHRing
//   RotatableBond 1  nRot
//
// SPECIFICATION IS MORDRED, read at mordred 1.2.0 under
// .venv-mordred/lib/python3.11/site-packages/mordred/: Aromatic.py, AtomCount.py, BondCount.py,
// RingCount.py, RotatableBond.py and _base/context.py. Every claim below about which molecule a
// column sees comes from `Context.from_query`, which is the only place mordred decides:
//
//     m = Chem.AddHs(mol) if eh else Chem.RemoveHs(mol, updateExplicitCount=True)
//     if ke: Chem.Kekulize(m)
//
// keyed on the descriptor's `explicit_hydrogens` and `kekulize` class attributes.
//
// NOTHING HERE PERCEIVES A RING, COMPUTES A KEKULE STRUCTURE OR MATCHES A SMARTS. All three are
// already answered elsewhere in this tree and are consumed, not repeated:
//
//   rings          the ring CSR src/hume/_rings.py:rings_for() builds and ringcount.h reads --
//                  the SAME `ringcount::Mol` bindings.cpp already fills for the 49-column block.
//   kekule         constit.h's `bondCount()` reconstruction, arriving through `Inputs` as the
//                  already-emitted `nBondsD` and `nBondsKD`. See nBondsKS below.
//   nRot           frag_matcher.h's `NumRotatableBonds`, arriving through `Inputs`.
//
// ===============================================================================================
// THE THREE PLACES THIS BLOCK IS NOT TRIVIAL. Each was checked against the source and measured;
// NOTES_counts.md carries the numbers.
// ===============================================================================================
//
// (1) `nAtom` COUNTS HYDROGENS AND `nHeavyAtom` DOES NOT, AND THE REASON IS NOT THE NAME.
//     AtomCount.explicit_hydrogens is `self._type in {"H", "Atom"}`, so "Atom" gets
//     `Chem.AddHs(mol)` and "HeavyAtom" gets `Chem.RemoveHs(mol)`; BOTH then return
//     `self.mol.GetNumAtoms()` from the same `_calc_all`. So the two columns differ only in which
//     molecule the Context handed over, and "HeavyAtom" is a misnomer -- it is
//     GetNumAtoms() of the H-SUPPRESSED graph, which is not the same as "atoms with Z != 1".
//     `Chem.RemoveHs` does not remove every hydrogen: constit.h records that it keeps ISOTOPIC
//     hydrogen, and data/dedupe2's corpus exercises a second mechanism -- 558 of its 20,000
//     molecules (823 atoms) keep a STEREO-DEFINING hydrogen, the `[H]` of a `[H]/N=C(...)`
//     amidine whose directional bond carries the C=N geometry, which RDKit's removeHs refuses to
//     drop. `RemoveHs` changes the atom count on 0 of the 20,000. So nHeavyAtom counts hydrogens
//     on 2.8% of that corpus, mordred agrees that it does, and this file returns `km.n` rather
//     than a filtered count. The contract is RemoveHs, not the column's name.
//
//     Neither number needs the h_blobs the pickle carries: constit.h's `HDerived` already
//     establishes and verifies nAtomsH = n + sum(nh) on the heavy boundary. nAtom is that.
//
//     `nAromAtom` and `nAromBond` are the OPPOSITE trap and it costs nothing here.
//     `Aromatic.AromaticBase` overrides no `explicit_hydrogens`, and `Descriptor`'s class
//     default is **True** -- so both of them are measured on `Chem.AddHs(mol)`, not on the heavy
//     graph, which is the reverse of what the names suggest. It changes no value, because
//     `Chem.AddHs` adds non-aromatic atoms joined by non-aromatic bonds and never touches an
//     existing flag, so the aromatic count of the H-added graph IS the aromatic count of the
//     heavy graph. Reading `Descriptor.explicit_hydrogens = True` as "these are heavy-graph
//     columns" would still have been reading it wrong.
//
//     `nAromBond` IS THE FLAG ALONE -- `sum(1 for b if b.GetIsAromatic())` -- where BondCount's
//     `nBondsA` is `b.GetIsAromatic() or b.GetBondType() == AROMATIC`. constit.h records 4 bonds
//     in cpp/hard.smi where the two disagree (TRIPLE bonds carrying the aromatic flag). Here the
//     `or` is absent, so this column is `(bcode & BC_AROM) != 0` and nothing else.
//
// (2) `nBondsKS` IS WELL-POSED, NOT ILL-POSED, AND THAT IS A MEASUREMENT AND AN ARGUMENT.
//     BondCount("single", kekulize=True): explicit_hydrogens is True (`single` is in
//     `(any, single)`), so the molecule is `Chem.Kekulize(Chem.AddHs(mol))` and the column is
//     the number of bonds whose type is SINGLE in it.
//
//     Which aromatic bond becomes the double bond is arbitrary -- benzene has two Kekule
//     structures and RDKit picks one by search order. HOW MANY become double is not arbitrary.
//     Kekulization preserves every atom's total valence, so for each atom carrying aromatic
//     bonds the number of ring double bonds it takes is fixed by
//     `tval - nh - nonAromaticValenceContrib - nAromaticBonds`, a per-atom function of the
//     boundary that mentions no matching at all; the number of promoted bonds is half that sum
//     over atoms. Every Kekule structure of the same molecule therefore has the same number of
//     double bonds, hence the same number of single bonds. That per-atom quantity is exactly
//     constit.h's `takesDouble`, already verified in {0,1} with an even sum.
//
//     MEASURED, because the argument is only an argument: 650 aromatic corpus molecules were
//     re-parsed from 40 randomly re-ordered SMILES each (26,000 re-parses), kekulized, and the
//     SINGLE and DOUBLE counts recorded. Both moved on 0 molecules. So this is a QUIRK-FREE,
//     WELL-POSED column and rule 4 says reproduce it, which this file does exactly.
//
//     It is reproduced WITHOUT a second Kekule reconstruction. In the kekulized H-added graph the
//     single bonds are: the heavy bonds already typed SINGLE, plus the aromatic-TYPE bonds that
//     were not promoted, plus one per added hydrogen. The promoted count is `nBondsKD - nBondsD`,
//     both already emitted by constit.h, so
//
//         nBondsKS = nHadd + #{SINGLE} + #{aromatic type} - (nBondsKD - nBondsD)
//
//     and the only Kekule reasoning in the tree stays in the one place it was verified.
//     Verified here on all 20,000 corpus molecules: 0 mismatches against mordred.
//
// (3) RING PERCEPTION IS SYMMETRISED SSSR, AND IT IS THE ONE ILL-POSED INPUT.
//     `RingCount.Rings.calculate` is `[frozenset(s) for s in Chem.GetSymmSSSR(mol)]` -- the
//     SYMMETRISED SSSR, not the plain SSSR -- on `Chem.RemoveHs(mol)`, because
//     `RingCountBase.explicit_hydrogens = False` overrides `Descriptor`'s True. This file does
//     not perceive it. It consumes `ringcount::Mol`, which bindings.cpp fills from the boundary's
//     `rings_for()` CSR -- the SAME single perception the 49-column block reads.
//
//     `Chem.GetSymmSSSR` is not a function of the molecular graph (ringcount.h and
//     src/hume/_rings.py carry the evidence), and the boundary repairs it by perceiving on a
//     canonically rebuilt skeleton. mordred's reference values were produced from RDKit's RAW
//     answer, so on an ambiguous molecule HUME and mordred differ BY CONSTRUCTION. Measured over
//     the 20,000-molecule corpus on these 21 columns: 2 molecules move, 3 columns
//     (n7ARing, n7AHRing, nG12AHRing), one molecule each. `C1C2OCCC3OC2C13` is one of them, and
//     re-parsing it from 300 random SMILES orderings gives RDKit's raw n7ARing = 2 on 259 and
//     3 on 41 -- the definition, not this file, is what is undecided. Divergence, documented,
//     per contract rule 4. Adopting mordred's raw ring set here instead would put TWO ring
//     perceptions in one featuriser, which is the failure mode src/hume/_rings.py exists to
//     prevent.
//
// ===============================================================================================
// ONE THING THIS FILE DUPLICATES, AND THE GUARD THAT MAKES THE DUPLICATE SAFE.
// ===============================================================================================
// `ringPass()` below is a strict generalisation of `ringcount::compute` -- the same ring
// properties, the same |Ri & Rj| >= 2 union-find, the same `networkx` singleton exclusion -- over
// a caller-supplied Spec table instead of ringcount's fixed 49. It exists because
// `ringcount::compute` hard-codes `ringcount::COLS`, and this agent may not edit ringcount.h.
//
// So the copy is CHECKED rather than trusted: `driftGuard()` runs `ringPass` over
// `ringcount::COLS` and requires all 49 values to equal `ringcount::compute`'s, per molecule.
// verify_counts.py calls it on all 20,000 (currently 0 disagreements, 980,000 cells).
//
// THE RIGHT FIX IS ONE LINE IN ringcount.h AND IT IS NOT MINE TO MAKE: replace the body of
// `ringcount::compute` with `counts_ext::ringPass(m, S, ringcount::COLS, N_COLS, out)`. That
// deletes the duplicate outright and lets the two blocks share ONE fusion pass instead of two.
// See NOTES_counts.md.
#ifndef HUME_COUNTS_EXT_H
#define HUME_COUNTS_EXT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "constit.h"     // Mol, HDerived, isAromType, BC_* -- the heavy-graph boundary
#include "ringcount.h"   // Mol, Scratch, Spec, ring_props, passes, uf_find, COLS, N_COLS

namespace counts_ext {

static constexpr int N_COLS = 31;

// Emit order IS results/dedupe2/agent_groups.json's "E_counts" order, unchanged, so the parent's
// offset enum is that list read top to bottom.
inline const char *col_name(int i) {
  static const char *N[N_COLS] = {
      "nAromAtom", "nAromBond", "nAtom", "nHeavyAtom", "nP", "nF", "nI", "nBonds", "nBondsKS",
      "n8Ring", "n12Ring", "n8HRing", "n12HRing", "n7aRing", "n7ARing", "n7AHRing", "nG12AHRing",
      "n5FRing", "n6FRing", "n5FHRing", "n6FHRing", "n7FHRing", "n8FHRing", "n11FHRing",
      "n12FHRing", "n12FaRing", "n11FARing", "n12FARing", "n11FAHRing", "n12FAHRing",
      "nRot"};
  if (i < 0 || i >= N_COLS)
    throw std::runtime_error("counts_ext::col_name: index " + std::to_string(i) +
                             " out of range [0," + std::to_string(N_COLS) + ")");
  return N[i];
}

enum : int {
  C_NAROMATOM = 0, C_NAROMBOND = 1, C_NATOM = 2, C_NHEAVYATOM = 3,
  C_NP = 4, C_NF = 5, C_NI = 6, C_NBONDS = 7, C_NBONDSKS = 8,
  C_RING0 = 9,                     // 21 RingCount columns, C_RING0 .. C_RING0 + N_RING - 1
  C_NROT = 30
};
static constexpr int N_RING = 21;

// The 21 RingCount parameter tuples, in emit order. `selfCheck()` regenerates mordred's whole
// 138-entry preset from RingCount.preset()'s own nested loops, names each entry with
// RingCount.__str__'s own rules, and requires every row here to appear there with matching
// parameters -- so a wrong `order`, a swapped arom/hetero or a stray fused flag fails at load
// rather than on a molecule nobody tried. Field order is mordred's `parameters()`:
// (order, greater, fused, aromatic, hetero); ANY is mordred's `None`.
using ringcount::ANY;
using ringcount::NO;
using ringcount::YES;

static const ringcount::Spec RING_COLS[N_RING] = {
    {"n8Ring", 8, 0, 0, ANY, ANY},        {"n12Ring", 12, 0, 0, ANY, ANY},
    {"n8HRing", 8, 0, 0, ANY, YES},       {"n12HRing", 12, 0, 0, ANY, YES},
    {"n7aRing", 7, 0, 0, YES, ANY},       {"n7ARing", 7, 0, 0, NO, ANY},
    {"n7AHRing", 7, 0, 0, NO, YES},       {"nG12AHRing", 12, 1, 0, NO, YES},
    {"n5FRing", 5, 0, 1, ANY, ANY},       {"n6FRing", 6, 0, 1, ANY, ANY},
    {"n5FHRing", 5, 0, 1, ANY, YES},      {"n6FHRing", 6, 0, 1, ANY, YES},
    {"n7FHRing", 7, 0, 1, ANY, YES},      {"n8FHRing", 8, 0, 1, ANY, YES},
    {"n11FHRing", 11, 0, 1, ANY, YES},    {"n12FHRing", 12, 0, 1, ANY, YES},
    {"n12FaRing", 12, 0, 1, YES, ANY},    {"n11FARing", 11, 0, 1, NO, ANY},
    {"n12FARing", 12, 0, 1, NO, ANY},     {"n11FAHRing", 11, 0, 1, NO, YES},
    {"n12FAHRing", 12, 0, 1, NO, YES},
};

// ---------------------------------------------------------------------------------------------
// Values another verified family owns. Passed in, never recomputed -- bindings.cpp's own rule for
// the constitutional block. The defaults are "nobody supplied it" sentinels and are NEGATIVE on
// purpose: a zeroed nRot or nBondsKD is a finite, plausible, WRONG answer, where a throw names
// the missing wire.
// ---------------------------------------------------------------------------------------------
struct Inputs {
  int nRot = -1;      // frag_matcher.h, RDKit column "NumRotatableBonds".
                      //   mordred's nRot is `CalcNumRotatableBonds(Chem.RemoveHs(mol))` with
                      //   RDKit's default strictness, which is the same call and the same
                      //   default rdkit's own `NumRotatableBonds` descriptor makes. Measured
                      //   over the 20,000-molecule corpus: mordred `nRot` and RDKit
                      //   `NumRotatableBonds` disagree on 0 molecules. So this column is an
                      //   ALIAS of an already-emitted value and is emitted by wiring, not by
                      //   computing -- exactly as constit.h emits `SLogP`.
  int nBondsD = -1;   // constit.h column "nBondsD"  (constit col 18)
  int nBondsKD = -1;  // constit.h column "nBondsKD" (constit col 22)
};

// ---------------------------------------------------------------------------------------------
// The generalised RingCount pass. See the header note: this is `ringcount::compute` with the
// Spec table as an argument, and `driftGuard()` below holds the two identical.
//
// `S` is reused across molecules so the hot loop makes no allocation, and it is the SAME
// `ringcount::Scratch` the 49-column block uses -- ringcount::compute leaves it in a state this
// function does not read, and vice versa (both re-initialise `stamp` and `seen` on entry).
// ---------------------------------------------------------------------------------------------
inline void ringPass(const ringcount::Mol &m, ringcount::Scratch &S,
                     const ringcount::Spec *cols, int ncols, double *out) {
  for (int c = 0; c < ncols; ++c) out[c] = 0.0;
  const int R = m.n_rings();
  if (R == 0) return;

  if ((int)S.stamp.size() < m.n) S.stamp.assign(m.n, -1);
  else std::fill(S.stamp.begin(), S.stamp.begin() + m.n, -1);
  if ((int)S.seen.size() < m.n) S.seen.assign(m.n, 0);
  else std::fill(S.seen.begin(), S.seen.begin() + m.n, 0);

  // ---- plain rings --------------------------------------------------------------------------
  S.sz.assign(R, 0); S.ar.assign(R, 0); S.het.assign(R, 0);
  for (int r = 0; r < R; ++r)
    ringcount::ring_props(m, S, &m.ring_at[m.ring_off[r]], m.ring_off[r + 1] - m.ring_off[r], r,
                          S.sz[r], S.ar[r], S.het[r]);
  for (int c = 0; c < ncols; ++c) {
    if (cols[c].fused) continue;
    int k = 0;
    for (int r = 0; r < R; ++r) k += ringcount::passes(cols[c], S.sz[r], S.ar[r], S.het[r]);
    out[c] = (double)k;
  }

  // ---- fused systems ------------------------------------------------------------------------
  // mordred's FusedRings returns [] outright below two rings, and builds its graph with
  // networkx.add_edge only -- so a ring that shares fewer than two atoms with every other ring
  // is not a vertex of the graph at all and is NOT a one-ring fused system.
  if (R < 2) return;

  S.parent.resize(R);
  for (int r = 0; r < R; ++r) S.parent[r] = r;
  S.touched.assign(R, 0);
  std::vector<uint8_t> &touched = S.touched;
  for (int i = 0; i < R; ++i) {
    const int bi = m.ring_off[i], ei = m.ring_off[i + 1];
    const int tag = R + i;                       // no ring index can collide with this
    for (int q = bi; q < ei; ++q) S.stamp[m.ring_at[q]] = tag;
    for (int j = i + 1; j < R; ++j) {
      int shared = 0;
      // |Ri & Rj| on frozenSETS: a repeated atom in Rj must not be counted twice, so a matched
      // atom is re-stamped to a third value and restored afterwards.
      const int bj = m.ring_off[j], ej = m.ring_off[j + 1];
      for (int q = bj; q < ej && shared < 2; ++q) {
        const int a = m.ring_at[q];
        if (S.stamp[a] == tag) { S.stamp[a] = tag + R; ++shared; }
      }
      for (int q = bj; q < ej; ++q) {
        const int a = m.ring_at[q];
        if (S.stamp[a] == tag + R) S.stamp[a] = tag;
      }
      if (shared >= 2) {
        const int ra = ringcount::uf_find(S.parent, i), rb = ringcount::uf_find(S.parent, j);
        if (ra != rb) S.parent[ra] = rb;
        touched[i] = touched[j] = 1;
      }
    }
  }

  S.comp.assign(R, -1);
  int ncomp = 0;
  for (int r = 0; r < R; ++r) {
    if (!touched[r]) continue;
    const int root = ringcount::uf_find(S.parent, r);
    if (S.comp[root] < 0) S.comp[root] = ncomp++;
    S.comp[r] = S.comp[root];
  }
  if (ncomp == 0) return;

  S.fsz.assign(ncomp, 0);
  S.farom.assign(ncomp, 1);
  S.fhet.assign(ncomp, 0);
  std::vector<int32_t> &fsz = S.fsz;
  std::vector<uint8_t> &farom = S.farom, &fhet = S.fhet;
  for (int c = 0; c < ncomp; ++c) {
    S.members.clear();
    for (int r = 0; r < R; ++r) {
      if (S.comp[r] != c) continue;
      for (int q = m.ring_off[r]; q < m.ring_off[r + 1]; ++q) {
        const int a = m.ring_at[q];
        if (S.seen[a]) continue;
        S.seen[a] = 1;
        S.members.push_back(a);
        ++fsz[c];
        if (!m.arom[a]) farom[c] = 0;
        if (m.z[a] != 6) fhet[c] = 1;
      }
    }
    for (int a : S.members) S.seen[a] = 0;   // clear only what this component touched
  }
  for (int col = 0; col < ncols; ++col) {
    if (!cols[col].fused) continue;
    int k = 0;
    for (int c = 0; c < ncomp; ++c) k += ringcount::passes(cols[col], fsz[c], farom[c], fhet[c]);
    out[col] = (double)k;
  }
}

// ---------------------------------------------------------------------------------------------
// All 31 columns for one molecule.
//
//   km  the heavy-atom boundary, exactly the `constit::Mol` bindings.cpp already builds for the
//       constitutional block. Read for: n, nb, z, nh, arom, bcode.
//   rm  the ring view, exactly the `ringcount::Mol` bindings.cpp already builds for the
//       49-column block. Read for: z, arom, ring CSR. It is NOT re-derived from km, so there is
//       still exactly one ring perception and exactly one chance to disagree with it.
//   in  the three values other families own (see Inputs).
//
// No column here is NaN under any input: all 31 are integer counts of a finite graph, and mordred
// declares every one of them `rtype = int`. Verified over the corpus: 0 NaN on both sides.
// ---------------------------------------------------------------------------------------------
inline void compute(const constit::Mol &km, const ringcount::Mol &rm, const Inputs &in,
                    double *out, ringcount::Scratch &S) {
  if (km.n != rm.n)
    throw std::runtime_error(
        "counts_ext::compute: the two molecule views disagree about atom count -- constit::Mol "
        "has " + std::to_string(km.n) + " atoms, ringcount::Mol has " + std::to_string(rm.n) +
        ". They must be two views of ONE molecule; check the wiring in bindings.cpp.");
  if (in.nRot < 0 || in.nBondsD < 0 || in.nBondsKD < 0)
    throw std::runtime_error(
        "counts_ext::compute: a required Input was never supplied (nRot=" +
        std::to_string(in.nRot) + ", nBondsD=" + std::to_string(in.nBondsD) + ", nBondsKD=" +
        std::to_string(in.nBondsKD) + "; all must be >= 0). nRot is frag_matcher.h's "
        "\"NumRotatableBonds\"; nBondsD and nBondsKD are constit.h columns 18 and 22. See the "
        "wiring note in NOTES_counts.md.");

  const constit::HDerived H(km);

  // ---- Aromatic (2). explicit_hydrogens is True for both, and it changes neither value: see
  // note (1). `nAromBond` is the FLAG alone, not BondCount's `flag or type == AROMATIC`.
  int nAromAtom = 0;
  for (int i = 0; i < km.n; ++i) nAromAtom += (km.arom[i] != 0);
  int nAromBond = 0;
  for (int e = 0; e < km.nb; ++e) nAromBond += ((km.bcode[e] & constit::BC_AROM) != 0);
  out[C_NAROMATOM] = (double)nAromAtom;
  out[C_NAROMBOND] = (double)nAromBond;

  // ---- AtomCount (5). `nAtom` and `nHeavyAtom` are the SAME `_calc_all` on two molecules; the
  // three element counts are `a.GetSymbol() == type`, which is a function of Z alone.
  out[C_NATOM] = (double)H.nAtomsH;      // GetNumAtoms() of Chem.AddHs(mol)
  out[C_NHEAVYATOM] = (double)km.n;      // GetNumAtoms() of Chem.RemoveHs(mol)
  int nP = 0, nF = 0, nI = 0;
  for (int i = 0; i < km.n; ++i) {
    const int z = km.z[i];
    if (z == 15) ++nP;
    else if (z == 9) ++nF;
    else if (z == 53) ++nI;
  }
  out[C_NP] = (double)nP;
  out[C_NF] = (double)nF;
  out[C_NI] = (double)nI;

  // ---- BondCount (2). Both have explicit_hydrogens True (`any` and `single`); every hydrogen
  // Chem.AddHs creates is terminal and brings exactly one SINGLE bond, which is the `+ nHadd` in
  // both lines. See note (2) for why nBondsKS needs no Kekule pass of its own.
  out[C_NBONDS] = (double)H.nBondsH;     // == km.nb + H.nHadd, established in constit::HDerived
  int nSingle = 0, nAromType = 0;
  for (int e = 0; e < km.nb; ++e) {
    const int c = km.bcode[e];
    if (c & constit::BC_SINGLE) ++nSingle;
    else if (constit::isAromType(c)) ++nAromType;   // exclusive: isAromType requires no order bit
  }
  const int promoted = in.nBondsKD - in.nBondsD;    // aromatic-type bonds Kekulize made DOUBLE
  if (promoted < 0 || promoted > nAromType)
    throw std::runtime_error(
        "counts_ext::compute: nBondsKD - nBondsD = " + std::to_string(promoted) +
        " is not a possible number of promoted aromatic bonds (0.." + std::to_string(nAromType) +
        "). Either nBondsD/nBondsKD were wired to the wrong constit columns, or km is not the "
        "molecule they were computed on.");
  out[C_NBONDSKS] = (double)(H.nHadd + nSingle + nAromType - promoted);

  // ---- RingCount (21). Consumes the boundary's single ring perception; perceives nothing.
  ringPass(rm, S, RING_COLS, N_RING, out + C_RING0);

  // ---- RotatableBond (1). An alias; see Inputs::nRot.
  out[C_NROT] = (double)in.nRot;
}

// ---------------------------------------------------------------------------------------------
// DRIFT GUARD 1, per molecule: `ringPass` over ringcount's own 49 specs must reproduce
// `ringcount::compute` cell for cell. This is what makes the duplicated fusion pass safe until
// ringcount.h can be made to delegate to it. Called by verify_counts.py on every molecule; it is
// NOT on the hot path and bindings.cpp should not call it.
// ---------------------------------------------------------------------------------------------
inline void driftGuard(const ringcount::Mol &m, ringcount::Scratch &S, const char *smiles) {
  double a[ringcount::N_COLS], b[ringcount::N_COLS];
  ringcount::compute(m, a, S);
  ringPass(m, S, ringcount::COLS, ringcount::N_COLS, b);
  for (int c = 0; c < ringcount::N_COLS; ++c)
    if (a[c] != b[c])
      throw std::runtime_error(
          std::string("counts_ext::driftGuard: ringPass has drifted from ringcount::compute on "
                      "column '") + ringcount::COLS[c].name + "' -- ringcount::compute gave " +
          std::to_string(a[c]) + ", ringPass gave " + std::to_string(b[c]) + " for molecule " +
          (smiles ? smiles : "(unnamed)"));
}

// ---------------------------------------------------------------------------------------------
// DRIFT GUARD 2, at load: the 21 RingCount parameter tuples. Regenerates mordred's full 138-entry
// preset from RingCount.preset()'s own nested loops, names each with RingCount.__str__'s own
// rules, and requires every RING_COLS row to appear there exactly once with identical parameters.
// Order is NOT required to match the preset -- this block emits in agent_groups order, not preset
// order -- so the check is by name, and a name appearing twice or not at all throws.
//
// Nothing else in this file depends on an upstream table: the nine scalar columns are counts of
// integers already on the boundary, and the three element numbers (P=15, F=9, I=53) are physics.
// ---------------------------------------------------------------------------------------------
inline std::string preset_name(int order, int greater, int fused, int arom, int hetero) {
  std::string a;
  if (greater) a += "G";
  if (order >= 0) a += std::to_string(order);
  if (fused) a += "F";
  if (arom == YES) a += "a"; else if (arom == NO) a += "A";
  if (hetero == YES) a += "H"; else if (hetero == NO) a += "C";
  return "n" + a + "Ring";
}

inline void selfCheck() {
  // mordred RingCount.preset(): for fused in [False,True]: for arom in [None,True,False]:
  //   for hetero in [None,True]: yield (None,False,...); for n in range(4 if fused else 3, 13):
  //   yield (n,False,...); yield (12,True,...)
  std::vector<ringcount::Spec> pre;
  std::vector<std::string> names;
  const int aroms[3] = {ANY, YES, NO};
  const int hets[2] = {ANY, YES};
  for (int fused = 0; fused <= 1; ++fused)
    for (int ai = 0; ai < 3; ++ai)
      for (int hi = 0; hi < 2; ++hi) {
        const int a = aroms[ai], h = hets[hi];
        pre.push_back({nullptr, -1, 0, fused, a, h});
        for (int nn = (fused ? 4 : 3); nn < 13; ++nn) pre.push_back({nullptr, nn, 0, fused, a, h});
        pre.push_back({nullptr, 12, 1, fused, a, h});
      }
  if (pre.size() != 138)
    throw std::runtime_error("counts_ext::selfCheck: regenerated preset has " +
                             std::to_string(pre.size()) + " entries, mordred's has 138");
  for (auto &s : pre) names.push_back(preset_name(s.order, s.greater, s.fused, s.arom, s.hetero));

  for (int c = 0; c < N_RING; ++c) {
    const std::string want(RING_COLS[c].name);
    int hit = -1, nhit = 0;
    for (size_t q = 0; q < pre.size(); ++q)
      if (names[q] == want) { ++nhit; if (hit < 0) hit = (int)q; }
    if (nhit != 1)
      throw std::runtime_error("counts_ext::selfCheck: RingCount column '" + want + "' appears " +
                               std::to_string(nhit) + " times in mordred's 138-entry preset "
                               "(expected exactly 1) -- the name is wrong");
    const ringcount::Spec &g = pre[hit];
    if (g.order != RING_COLS[c].order || g.greater != RING_COLS[c].greater ||
        g.fused != RING_COLS[c].fused || g.arom != RING_COLS[c].arom ||
        g.hetero != RING_COLS[c].hetero)
      throw std::runtime_error(
          "counts_ext::selfCheck: '" + want + "' parameters disagree with mordred's preset: "
          "table (order=" + std::to_string(RING_COLS[c].order) + ",greater=" +
          std::to_string(RING_COLS[c].greater) + ",fused=" + std::to_string(RING_COLS[c].fused) +
          ",arom=" + std::to_string(RING_COLS[c].arom) + ",hetero=" +
          std::to_string(RING_COLS[c].hetero) + ") vs preset (order=" + std::to_string(g.order) +
          ",greater=" + std::to_string(g.greater) + ",fused=" + std::to_string(g.fused) +
          ",arom=" + std::to_string(g.arom) + ",hetero=" + std::to_string(g.hetero) + ")");
  }
  // The emit order must be the 21 ring names in RING_COLS order, sitting at C_RING0.
  for (int c = 0; c < N_RING; ++c)
    if (std::string(col_name(C_RING0 + c)) != RING_COLS[c].name)
      throw std::runtime_error(
          std::string("counts_ext::selfCheck: emit slot ") + std::to_string(C_RING0 + c) +
          " is named '" + col_name(C_RING0 + c) + "' but RING_COLS[" + std::to_string(c) +
          "] is '" + RING_COLS[c].name + "' -- the ring block is transposed");
  ringcount::selfCheck();     // the 49 this file's drift guard compares against
}

}  // namespace counts_ext

// ===============================================================================================
// WIRING NOTE FOR bindings.cpp -- NOT APPLIED HERE, per the contract.
//
//   OFF_COUNTS = <after the last existing block>;  N = counts_ext::N_COLS (31)
//
//   counts_ext::Inputs cin;
//   cin.nRot     = (int)out[OFF_FRAG    + IC.nrot];        // the existing input_cols() lookup
//   cin.nBondsD  = (int)out[OFF_CONSTIT + 18];             // constit::col_name(18) == "nBondsD"
//   cin.nBondsKD = (int)out[OFF_CONSTIT + 22];             // constit::col_name(22) == "nBondsKD"
//   counts_ext::compute(W.km, W.rm, cin, out + OFF_COUNTS, W.rs);
//
// and `counts_ext::selfCheck()` next to the other selfCheck calls in the module init.
//
// THREE CONSTRAINTS THE PARENT MUST HONOUR:
//   * It must run AFTER the F_CONSTIT and F_FRAG blocks -- it reads three of their outputs.
//     If either family can be switched off independently, this block must be gated on both.
//   * `W.km` is built inside `if (fams & F_CONSTIT)`. This block needs it, so either share that
//     gate or hoist the build.
//   * The two constit offsets should be looked up by name through the same `input_cols()`
//     mechanism `IC.naRing` uses rather than hard-coded, so a change to constit's column order
//     fails loudly. `constit::col_name()` is the authority.
// ===============================================================================================

#endif  // HUME_COUNTS_EXT_H
