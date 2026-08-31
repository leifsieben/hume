# NOTES — group `A_sps`

One column: **`SPS`**. `NumAtomStereoCenters` and `NumUnspecifiedAtomStereoCenters` were checked
against `hume.ALL_COLUMNS` first and are already present and already computed (`rdkcore.h`, from
the pickle's legacy `_ChiralityPossible` flag), so they were out of scope.

Deliverables: `src/hume_core/sps.h`, `verify_sps.py`, this file. `src/hume_core/bindings.cpp` was
not touched; the wiring it needs is described at the end.

**Result: exact.** 20,000 / 20,000 corpus molecules bit-identical in double precision against
`rdkit.Chem.SpacialScore.SPS`, and 20,000 / 20,000 with an identical potential-stereocentre atom
set against `Chem.FindPotentialStereo`. Independently: 100,000 / 100,000 on `cpp/hard.smi`, both
measures. **16.90 µs/mol** against **68.43 µs/mol** for the Python route it replaces, measured in
the same run on the same machine — a 4.05× speedup.

---

## 1. The measurement that justifies the work

`SPS` needs `Chem.FindMolChiralCenters(useLegacyImplementation=False, includeUnassigned=True)`,
which is RDKit's **new** perception (`Chirality::findPotentialStereo`). The pickle carries the
**legacy** `_ChiralityPossible` flag. Over all 20,000 corpus molecules, parsed exactly as
`src/hume/_extract.py` parses them (`MolFromSmiles` under the default legacy setting, then
`AssignStereochemistry(cleanIt=True, force=True, flagPossibleStereoCenters=True)` on a copy for
the flag, and `FindPotentialStereoBonds` + `FindPotentialStereo` on a copy with
`SetUseLegacyStereoPerception(False)` for the new one):

    the two atom sets DIFFER ON 662 OF 20,000 MOLECULES  (3.31%)

The task brief quoted 262 of 4,000 (6.5%); the 3.31% here is the same fact on the full 20,000
rather than a 4,000 subset. The disagreement is systematic, not marginal: the legacy flag misses
ring and bridgehead centres that the new perception's canonical-rank refinement finds.

    CC12C3C4C3C1C(O)C24            legacy {3,4,5,8}      new {1,2,3,4,5,6,8}
    C1OC2COC3CC12C3                legacy {2}            new {2,5,7}
    CC(C)C12CC(C1)O2               legacy {}             new {3,5}
    OCC1CC(O)(CO)C1                legacy {}             new {2,4}

A first attempt at this number gave **1 of 20,000**, and it was wrong for a reason worth
recording: it set `SetUseLegacyStereoPerception(False)` *before* `MolFromSmiles`, so both sides
were the new perception. The setting has to be flipped only around the `FindPotentialStereo`
call, after parsing — which is what `_extract.py` does, and what `verify_sps.py` now does.

---

## 2. What was ported, and from where

Everything is read from the RDKit 2025.09.2 sources (`.venv` has 2025.09.2), fetched from the
tagged release and followed call for call. Nothing here is from memory.

| RDKit source | what came across |
| --- | --- |
| `Code/GraphMol/FindStereo.cpp` | `findPotentialStereo` / `runCleanup`, `initAtomInfo`, `initBondInfo`, `flagRingStereo`, `updateAtoms`, `updateBonds`, `areStereobondControllingAtomsDupes`, `getStereoInfo(atom)`, `getStereoInfo(bond)`, `isAtomPotentialTetrahedralCenter`, `isAtomPotentialNontetrahedralCenter`, `isBondPotentialStereoBond` |
| `Code/GraphMol/new_canon.{h,cpp}` | `Canon::rankFragmentAtoms`, `AtomCompareFunctor`, `SpecialSymmetryAtomCompareFunctor`, `bondholder::compare`, `RefinePartitions`, `ActivatePartitions`, `CreateSinglePartition`, `compareRingAtomsConcerningNumNeighbors` |
| `Code/RDGeneral/hanoiSort.h` | `hanoi` / `hanoisort` (verbatim) |
| `Code/GraphMol/Chirality.cpp` | `MolOps::findPotentialStereoBonds`, `buildCIPInvariants`, `iterateCIPRanks`, `assignAtomCIPRanks`, `rerankAtoms`, `findAtomNeighborsHelper`, `translateEZLabelToCisTrans`, `is_regular_h`, `bondAffectsAtomChirality`, `getAtomNonzeroDegree` |
| `Code/GraphMol/QueryOps.cpp` | `queryIsAtomBridgehead` |
| `Code/GraphMol/Bond.cpp` | `getTwiceBondType` |
| `Code/RDGeneral/utils.h` | `countSwapsToInterconvert` |
| `rdkit/Chem/SpacialScore.py` | the score itself |

Two things fell out of reading rather than guessing, and both are load-bearing:

* **The flags this call site passes collapse the comparator.** `FindStereo.cpp` calls
  `rankFragmentAtoms` with `includeChirality=false, includeIsotopes=false, includeAtomMaps=false,
  includeChiralPresence=false, useRingStereo=false, breakTies=false` and with **both** symbol
  vectors supplied. `AtomCompareFunctor::basecomp` then reduces to three tests — current class,
  degree, and the atom symbol string — because the `p_symbol` branch `return`s even when the
  strings are equal, short-circuiting atomic number, isotope, **H count** and charge. The symbol
  carries isotope/element/charge, so the only thing genuinely dropped is the hydrogen count, and
  it is dropped in RDKit too. Similarly every `bondholder` has `bondStereo == 0`, so
  `compareStereo` is unreachable.

* **`SPS` runs on a copy with `FindPotentialStereoBonds` already applied**, and that changes both
  of its answers. The bond term reads `bond.GetStereo() != STEREONONE` on that copy, so STEREOANY
  marks count; and those same marks change the bond symbols the canonical ranking sees, so they
  can move the *atom* answer as well. The header does the same thing in the same order, into a
  local copy of the bond-stereo vector.

---

## 3. The one thing that was nearly wrong, and how it showed up

The first full-corpus run was **19,998 / 20,000**. The two failures were both 3-substituted
tropane/granatane bicyclics:

    [H]c1c([H])c([H])c([C@@]([H])(ON=C2C([H])([H])[C@@]3([H])N(C)…    ref 25.92  ours 24.16
    [H]c1c([H])c([H])c(C(=C2C([H])([H])[C@@]3([H])N(…              ref 24.06  ours 22.56

Both differences are exactly one missing STEREOANY double bond (44 and 54 score units
respectively). The cause: `MolOps::findPotentialStereoBonds` **does not recompute CIP ranks when
the atoms already carry `_CIPRank`** — and after `MolFromSmiles` they do. The stored rank is not
the constitution-only `assignAtomCIPRanks` answer; the legacy stereo perception runs
`rerankAtoms`, which supplements the ranks with the assigned R/S and E/Z labels. In both
molecules the two bridge arms are constitutionally equivalent and are separated only by the
bridgehead stereocentres, so the plain ranking ties them and no STEREOANY is set.

Confirmed independently, without reference to the C++ at all: clearing `_CIPRank` on a copy
before calling `FindPotentialStereoBonds` changes RDKit's own answer on **exactly those two
molecules of the 20,000**.

The fix is `rerankAtoms` in `sps.h`, fed from the boundary's existing `cip` column (`_CIPCode` as
+1 / −1 / 0) and the bond-stereo column. It is applied once, guarded on there being a label to
fold in — which is RDKit's own `keepGoing` condition, and the common case is one rerank (round
one assigns every label, round two assigns nothing new and the loop exits).

The stereo atoms of a bond that arrives already E/Z are needed too (`getStereoInfo(bond)` matches
them against the controlling atoms to decide cis vs trans) and the boundary does not carry them.
They are recovered as "the highest-legacy-CIP-rank neighbour at each end", which was verified
against RDKit on **all 2,481 E/Z bonds in the corpus: 0 disagreements**.

---

## 4. Divergences and quirks

**Nothing diverges on the corpus.** The three items below are the places where a divergence is
*possible* and where it was measured to be zero, plus one genuine behavioural difference.

1. **The ring set is the repaired one, not RDKit's raw `GetSymmSSSR`.** `findPotentialStereo`
   reads `mol.getRingInfo()`; the boundary carries `src/hume/_rings.py`'s canonicalised set,
   because `GetSymmSSSR` is not a function of the graph (see that module). Both were run:
   `verify_sps.py --rings repaired` and `--rings rdkit` each give **20,000 / 20,000**. The
   repaired set is the one to ship — it is what RingCount already uses, and one ring perception
   per molecule is the point.

2. **Ring atom order is recovered, not carried.** The boundary CSR's within-ring atom order is
   whatever the repaired perception produced, and `flagRingStereo` indexes rings as *sequences*
   ("the atom half-way round", "walk the shared edge"). `RingSet::build` walks the induced cycle
   in the bond graph to get a cyclic order. Every use is invariant under rotation; the
   common-edge walk is direction-dependent in principle, but it is run from every candidate atom
   of the ring, so a fused edge is found from at least one end either way. Measured: 0 of 20,000
   differ, and 0 of 100,000 on `cpp/hard.smi`.

3. **A molecule with zero heavy atoms is NaN here and a `ZeroDivisionError` in RDKit.**
   `SpacialScore.py` divides by `GetNumHeavyAtoms()` unguarded, so `[H][H]` raises. This returns
   NaN, which is what `constit.h` already did and what the rest of the suite means by "no value".
   Not reachable from the corpus (0 such molecules); reachable from a user's SMILES.

4. **Heavy-atom count is `atomicNum > 1`, not `!= 1`.** `constit.h`'s existing `sps()` used
   `!= 1`, which counts dummy atoms (`*`, Z = 0) as heavy; `ROMol::getNumHeavyAtoms` does not.
   No corpus molecule has a dummy atom so nothing moves, but the header matches RDKit.

**Quirks reproduced deliberately** (deterministic RDKit oddities, kept):

* `iterateCIPRanks` stops when the rank count stops *growing*, or after `numAtoms/2 + 1`
  iterations — it is not run to a fixed point. A faster stable-partition refinement would be a
  different function, so the termination rule is copied.
* `computeBondFeatures` sorts `numNeighbors + 1` entries, one past the filled ones, including a
  default `{0, 0}` pair. Its count is zero so it inserts nothing; the extra element is not
  reproduced because it provably cannot affect the entry.
* `computeBondFeatures` also increments `numNeighbors[nbrIdx]` rather than `numNeighbors[atomIdx]`
  — over the full double loop each atom still ends at its own degree, so the result is the same.
* The `AtomCompareFunctor` atom-map block runs even with `includeAtomMaps=false`, because
  `df_useAtomMapsOnDummies` defaults true. It reads a map number only on dummy atoms. SMILES with
  mapped dummy atoms would need that field at the boundary; nothing else would.
* Bond symbols *accumulate* a second `_cis` / `_trans` / `_unk` suffix in `updateBonds` on top of
  the one `initBondInfo` already appended (`"=_cis"` becomes `"=_cis_cis"`). It happens exactly
  once per bond, because the same pass sets `fixedBonds`; reproduced as written.

**Input assumptions**, stated because they are the one real limitation. Three things RDKit
consults are not in the boundary and are treated as absent, which is provably correct for
SMILES-sourced molecules and **would be wrong for mol-file-sourced ones**:

* `Bond::getBondDir() == UNKNOWN` (a squiggle bond) and the `_UnknownStereo` atom/bond property —
  these make an atom's or bond's stereo `Unknown` rather than `Unspecified`.
* `Bond::BondDir::EITHERDOUBLE` (a crossed double bond).
* atom-map numbers on dummy atoms (above).

If HUME ever ingests SDF, these three become new boundary fields. Everything else the perception
needs is already carried.

---

## 5. Verification

`verify_sps.py` (test harness only — it computes no descriptor value; it calls RDKit for ground
truth, serialises the boundary quantities, and diffs). It compiles a standalone harness into
`build_sps/` that `#include`s `src/hume_core/sps.h` unmodified, rather than going through
`hume._core`, because `bindings.cpp` is off limits while five agents share this checkout.

    .venv/bin/python verify_sps.py

| | |
| --- | --- |
| stage 1 — legacy flag vs new perception | differ on **662 / 20,000** |
| stage 2 — this header's perception vs `Chem.FindPotentialStereo` | **20,000 / 20,000 identical atom sets** |
| stage 3 — `SPS` vs `Chem.SpacialScore.SPS`, double | **20,000 exact (bit-identical)**, 0 at 1e-9, 0 at 1e-6, 0 mismatched, 0 NaN-on-one-side |
| stage 3 — `SPS` vs `data/dedupe2/matrix.npz` | 3,000 exact, 11 within 1e-9, 16,989 within 1e-6, **0 mismatched** |

The `matrix.npz` row needs one word of explanation: `RD` is stored as **float32**, so it cannot
settle a double-precision question — the same 3,000 molecules are bit-identical between the store
and RDKit's own double, and the other 17,000 differ from RDKit itself at the float32 rounding.
That is why the script reports both, and treats the double-precision comparison as the bar.

Independent corpora:

* `cpp/hard.smi`, 100,000 molecules: **100,000 / 100,000** identical perception **and**
  100,000 / 100,000 bit-identical `SPS`.
* 328 adversarial molecules (`cpp/adv.smi` cages, `cpp/crip_stress.smi`, plus a hand-built set:
  phosphines, arsines, sulfoxides, `[S+]`, `SF6`, square-planar Pt, `[13C]`/`[2H]` isotopes,
  charged and multi-fragment species, macrocyclic and 8-ring double bonds, single atoms, salts):
  **0 mismatches**.

---

## 6. Timing

Measured by `verify_sps.py` — minimum over 9 repetitions per molecule, both arms in the same run
on the same machine. **The machine was under load average ~100–130 from four other agents**, so
the absolute numbers are pessimistic; the ratio is the trustworthy part, and the C++ arm is
timed the same way as the Python arm.

| stratum | n | C++ `sps::compute` | SD | Python `_potential_stereo` | SD | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 0–15 | 4,000 | **2.40 µs/mol** | 1.09 | 17.08 | 3.86 | 7.1× |
| 15–25 | 4,000 | **7.18** | 3.26 | 35.46 | 8.50 | 4.9× |
| 25–35 | 4,000 | **12.36** | 5.07 | 54.74 | 12.71 | 4.4× |
| 35–55 | 4,000 | **19.39** | 7.46 | 79.83 | 18.38 | 4.1× |
| 55+ | 4,000 | **43.16** | 18.17 | 155.03 | 37.30 | 3.6× |
| **all** | 20,000 | **16.90 µs/mol** | 17.00 | **68.43** | 52.00 | **4.05×** |

Against the 54.4 µs/mol quoted in the brief, and against HUME's ~830 µs/mol total budget: this
group adds **~17 µs/mol, about 2% of the budget**, and *removes* the Python route's cost from the
extraction side. The earlier "amortise it into one call" attempt that measured 219.9 µs/mol is
not what this is — the whole perception happens C++-side, per molecule, with no RDKit call.

Where the time goes (min-of-9, 55+ stratum, the worst case):

    graph + rings + alloc     1.2 µs
    legacy CIP ranks          5.7 µs
    findPotentialStereoBonds  0.1 µs
    the new perception       35.6 µs      (3.5 canonicalisation rounds per molecule)

Four optimisations were applied on top of the straight port, each verified to move **0 molecules
of 20,000**:

1. **The CIP ranking is gated.** Only two things read it — E/Z stereo-atom recovery, and
   `findPotentialStereoBonds` when a qualifying double bond has two neighbours on at least one
   side. Most molecules qualify for neither. 10.2 → 1.8 µs/mol.
2. **Symbol strings are interned per round.** RDKit compares `std::string`s in the innermost loop
   of the refinement; here each distinct symbol gets a dense integer code by sorting the distinct
   strings once per round, so `code(a) < code(b)` iff `a < b` and the sign of every comparison the
   algorithm makes is unchanged.
3. **The interning sort uses an 8-byte big-endian key** built from the first 8 characters
   (zero-padded — byte 0 sorts before every character that can appear, so for symbols of ≤ 8
   characters the integer order *is* the string order; longer ones tie and fall through to a real
   string compare). This alone was 52 → 37 µs/mol on the 55+ stratum.
4. **`updateAtomNeighborIndex` is memoised on a stamp** that is bumped exactly where a class
   index is written. RDKit calls it on both operands of every comparison, and no index can move
   during a `hanoisort` pass, so the memo is exact rather than approximate. And
   `RefinePartitions`' O(n) "touched partitions" scan per processed partition became a collected,
   sorted list — same partitions, same ascending order, same `activeset` sequence.

One optimisation was applied and immediately caught a bug worth recording: interning initially
indexed the bond symbols by graph edge slot, but the `bondholder` slices are **sorted in place**,
so after round one `bonds[e]` is no longer edge `e`. It cost 69 molecules of 20,000 and was
found by the verify script, not by reading. The holder now carries its own `bondIdx`, which is
what RDKit's `bondholder` does and for the same reason.

---

## 7. What `bindings.cpp` needs (NOT made — described, per the contract)

`sps.h` needs no new boundary field. It takes what `all_row` already has in hand:

* per atom: `A_Z`, `A_DEG`, `A_NH`, `A_FCHG`, `A_HYB`, `A_AROM`, `A_NRING`, `A_TVAL`, `A_CTAG`,
  `A_ISO`, `A_CIP`
* per bond: `B_U`, `B_V`, `B_BTYPE`, the aromatic bit of `B_CODE` (`r[B_CODE] & 8`), `B_CONJ`,
  and `bond_s` (the assigned `Bond::getStereo()`, the `BS` array)
* the ring CSR already passed to `RingCount` / `rdkcore`

so the wiring is the same shape as the `rdkcore` block: fill `sps::Mol` from `ai` / `bi` / `BS`,
`add_ring` over the CSR, and call `sps::compute(m, W.sps, out + OFF_SPS)` with a `sps::detail::Work`
added to `AllWork` so the scratch is reused across the batch.

Three consequences for the surrounding code, for whoever does the wiring:

1. **`constit.h`'s `C_SPS` becomes dead** and `Inputs::stereoAtom` / `Inputs::stereoBond` become
   unused. `sps.h` owns the column now. Whether `SPS` keeps its slot inside the `constit` block
   or moves is a layout decision for the parent session; `sps.h` exposes `N_COLS`/`col_name`
   either way.
2. **`src/hume/_extract.py`'s `_potential_stereo` becomes dead**, and with it the `stereo_a` /
   `stereo_b` boundary arrays, the `stereo=` keyword, and the `have_stereo` contract in
   `all_from_pickles`. That is the 68 µs/mol of Python this removes. `SPS` stops being NaN when
   the caller did not ask for stereo — it is always available.
3. **`SPS` becomes a candidate for the `optional` set.** At ~17 µs/mol of an ~830 µs/mol budget it
   is not in `qed`/`AvgIpc` territory, but it is the third most expensive single column, and it is
   now a self-contained switch rather than a boundary contract.
