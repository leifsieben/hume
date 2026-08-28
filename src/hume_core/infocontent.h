// InformationContent (33 surviving mordred columns) and Ipc / AvgIpc, as a header the
// extension can call.
//
// ============================================================================================
// READ THIS FIRST: THIS FILE DELIBERATELY DOES NOT REPRODUCE MORDRED, AND THAT IS THE POINT.
// ============================================================================================
//
// PORT_STATUS.md house rule 1 says: reproduce a QUIRK, diverge from an ILL-POSED DEFINITION,
// and the test is whether the upstream descriptor is a function of the molecule. mordred's
// InformationContent is NOT a function of the molecule. It has two independent defects, both
// established by measurement on cpp/hard.smi rather than by reading:
//
//   1. IT KEKULIZES BEFORE BUILDING THE ATOM-EQUIVALENCE CODES. `InformationContentBase` sets
//      `kekulize = True`, so mordred/_base/context.py hands `Ag.calculate` a mol on which
//      `Chem.Kekulize` has run, and `BFSTree.__init__` records `b.GetBondType()` from THAT mol.
//      An aromatic bond therefore enters the code as SINGLE or DOUBLE depending on which Kekule
//      structure the perceiver happened to pick, and the equivalence classes move with it.
//
//   2. ITS BFS TREE MUTATES A VISITED SET WHILE ITERATING OVER IT. `BFSTree._expand` is
//
//          for src, dst in list(tree.items()):
//              self.visited.add(src)
//              if dst is ():
//                  tree[src] = {n.GetIdx(): () for n in self.atoms[src][2]
//                               if n.GetIdx() not in self.visited}
//
//      Two SIBLINGS at the same depth that happen to be adjacent are both in `tree` but neither
//      is in `visited` when the loop starts. Whichever the dict yields FIRST is added to
//      `visited` and then claims the other as its CHILD; the second one, now looking at a
//      `visited` that contains the first, does not claim it back. Dict order here is insertion
//      order, which is `GetNeighbors()` order, which is atom numbering. So the tree -- and every
//      code built from it -- depends on how the atoms happen to be numbered.
//
// MEASURED CONSEQUENCE: on the first 2,000 molecules of cpp/hard.smi, 32.3% change at least one
// InformationContent column under a single perturbation of the input ORDER -- 15.6% under
// `Chem.RenumberAtoms` alone, and 28.0% once the BOND LIST is shuffled as well, with 16.8% of
// the corpus visible ONLY to the bond-order screen. (Full-corpus numbers are in the
// cpp/verify_ic.py report; the shape does not change.) There is nothing there to be bit-exact
// against, and "we reproduce mordred" would be a claim about a coin flip.
//
// AND NOTE WHICH SCREEN FINDS WHAT. `Chem.RenumberAtoms` permutes atoms and LEAVES THE BOND
// LIST ALONE, so an atom-only screen under-samples anything that reads bonds in order -- which
// includes RDKit's ring perception. Half of mordred's instability here is invisible to it.
//
// ------------------------------------------------------------------------------------------
// THE WORKED EXAMPLE. This is the evidence that what follows is a repair and not a discrepancy.
// ------------------------------------------------------------------------------------------
//
//   SMILES   ON=Cc1ccccn1     pyridine-2-carbaldehyde oxime. 9 heavy atoms, 15 with H added.
//                             It is the smallest molecule in cpp/hard.smi that flips.
//
//   numbering A -- exactly as parsed:  0=O 1=N 2=C 3=c 4=c 5=c 6=c 7=c 8=n
//       mordred  IC1 = 2.682588730501833     IC2 = 3.4565647621309532
//   numbering B -- ONE TRANSPOSITION, Chem.RenumberAtoms(mol, [5,1,2,3,4,0,6,7,8]), i.e. swap
//                  the oxime oxygen with one ring CH, then SanitizeMol
//       mordred  IC1 = 2.8159220638351665    IC2 = 3.5898980954642865
//
//   Same molecule, same SMILES, same mordred, same RDKit, one swapped label: two answers, 5%
//   apart. TIC1/SIC1/BIC1/CIC1/MIC1/ZMIC1 and the order-2 row all move with them.
//
//   NOTE WHICH DEFECT THIS IS. `IC1` moves, and defect 2 CANNOT reach order 1 -- at order 1 the
//   tree is the root and its neighbours and no sibling has been expanded yet. So this is
//   defect 1: pyridine has two Kekule structures, RDKit's choice between them depends on atom
//   order, and the chosen bond orders go straight into the codes. (Defect 2 needs a cycle to
//   put two adjacent atoms at the same depth, so it starts at order 2.)
//
//   THIS FILE returns IC1 = 2.8159220638351665 and IC2 = 3.5898980954642865 for BOTH numberings,
//   for all 36 transpositions, and for the canonical-SMILES round trip, because the aromatic
//   bonds never become single or double. Which of mordred's two values we land on is not the
//   point and is not always either of them; the point is that there is exactly one.
//
//   AND THE CONVERSE CONTROL, which is what turns "we differ" into a characterisation instead
//   of a count. An ACYCLIC molecule has no aromatic bond (R1 cannot fire) and cannot have two
//   adjacent atoms at equal distance from any root -- that needs a cycle -- so R2 cannot fire
//   either. On acyclic molecules this file must therefore agree with mordred EXACTLY at every
//   order, MIC on isotope-labelled molecules excepted, since R3 is a third mechanism and is
//   independent of topology. And a molecule with a ring but no aromatic bond must agree at
//   ORDER 1 and may differ from order 2 up, because that is exactly where R2 starts. Both are
//   checked over the whole corpus by cpp/verify_ic.py rather than argued. (The MIC exception is
//   not a licence taken in advance: the first run of the acyclic control FAILED on 4 molecules,
//   every one of them MIC and every one carrying a [13C] or [15N].)
//
// ------------------------------------------------------------------------------------------
// THE RESOLUTION IMPLEMENTED HERE -- decided by the project owner, recorded in PORT_STATUS.md
// ------------------------------------------------------------------------------------------
//
//   R1. AN AROMATIC BOND KEEPS ITS OWN BOND-TYPE SYMBOL. `SYM_AROMATIC` is a fifth symbol
//       alongside single / double / triple / other, rather than being kekulized into one of
//       them. Nothing about a Kekule choice can reach the codes.
//
//   R2. THE TREE IS LAYERED BY GRAPH DISTANCE. The children of a node at depth d are its
//       neighbours at distance d+1 from the ROOT -- the BFS DAG of shortest paths -- so a
//       sibling is never a child of a sibling and atom numbering cannot reach the codes either.
//       Leaves are nodes at depth `order` and dead ends short of it, exactly as mordred's
//       `()`-vs-`{}` termination gives.
//
//   R3 (found during this port, not previously recorded). MIC's PER-CLASS WEIGHT WAS ALSO
//       ILL-POSED. mordred builds `ad = {a: i for i, a in enumerate(atoms)}`, a dict keyed by
//       CODE, so `ad[k]` is the LAST atom index carrying code k; `ModifiedIC` then weights the
//       class by `GetMass()` of THAT atom. Every atom in a class shares an atomic number, so
//       this is invisible until isotopes appear -- and 5.0% of cpp/hard.smi carries an isotope
//       label, where the class weight becomes "the mass of whichever labelled atom was numbered
//       last". Resolution: the weight is the STANDARD ATOMIC WEIGHT OF THE ELEMENT, which is
//       bit-identical to `GetMass()` for every unlabelled atom and is therefore a change only
//       where mordred had no defensible value. ZMIC is not affected -- its weight is the atomic
//       NUMBER, which is constant within a class by construction.
//
// CONSEQUENCES, EXPECTED AND CORRECT:
//   * ORDER 0 IS UNCHANGED. It never builds a tree -- `Ag.calculate` short-circuits to
//     `[a.GetAtomicNum() for a in mol.GetAtoms()]` -- so R1 and R2 cannot touch it. It is the
//     CONTROL: if IC0/TIC0/SIC0/CIC0/MIC0/ZMIC0 do not match mordred, this file has a bug.
//   * ORDERS 1-5 DIFFER FROM MORDRED BY DESIGN. cpp/verify_ic.py quantifies the divergence and
//     splits it into "mordred was unstable here anyway" and "mordred was stable and we still
//     differ"; see its docstring for the numbers and the characterisation of the second set.
//   * WHAT IS CLAIMED INSTEAD OF EXACTNESS IS DETERMINISM, and it is demonstrated rather than
//     asserted: bit-identical output for every column on every molecule of cpp/hard.smi under
//     three random atom renumberings, three permutations that ALSO SHUFFLE THE BOND LIST, and a
//     canonical-SMILES round trip. The bond-order half is not optional -- `Chem.RenumberAtoms`
//     leaves the bond list alone, so an atom-only screen cannot see anything that reads bonds in
//     order, and a determinism claim made against it is provisional.
//
// ============================================================================================
// WHAT IS *NOT* CHANGED, AND THE ARITHMETIC THAT IS REPRODUCED EXACTLY
// ============================================================================================
//
// THE MOLECULE IS THE HYDROGEN-ADDED, and for B the KEKULIZED, ONE. `Descriptor` defaults to
// `explicit_hydrogens = True` and `InformationContentBase` sets `kekulize = True`, so
// context.py hands these descriptors `Chem.Kekulize(Chem.AddHs(mol))`. Every count below is
// over that graph: A is `GetNumAtoms()` WITH hydrogens, the atom code is
// `(GetAtomicNum(), GetDegree())` with hydrogens as real neighbours AND real nodes, and B is
// `sum(b.GetBondTypeAsDouble())` over the KEKULIZED bonds. Missing the AddHs makes every one of
// the 33 columns wrong; this was checked by transcribing mordred into Python and reproducing
// all 42 columns bit-for-bit before a line of C++ was written.
//
// B IS RECOVERED WITHOUT KEKULIZING, AND THE RECOVERY IS VERIFIED, NOT ASSUMED. We do not
// kekulize (R1), but BIC needs the kekulized bond-order sum, which for pyrrole (7) differs from
// the aromatic-form sum (7.5). `kekuleBondOrderSum()` below rebuilds it from valences:
//
//     B = sum(order of every non-aromatic bond) + n_aromatic_bonds + k
//     k = (number of aromatic atoms that still need a ring double bond) / 2
//
// where an aromatic atom needs one iff its accumulated valence is short of the default valence
// for its element and charge -- RDKit's own Kekulize candidate test. A DATIVE bond contributes
// 0 to its DONOR and its order to its ACCEPTOR (RDKit's `Bond::getValenceContrib`), which is
// the only place the direction of a boundary bond row matters and the one case that a
// symmetric accumulation gets wrong. VERIFIED: 100,000 / 100,000 molecules of cpp/hard.smi
// reproduce `sum(GetBondTypeAsDouble())` over the AddHs+Kekulize mol exactly, with the number
// of "needs a double bond" atoms even on every single one.
//
// THE SUMMATION ORDER IS NUMPY'S. mordred's `shannon_entropy` is `-np.sum(w * (a/N)*log2(a/N))`
// and numpy's `np.sum` is PAIRWISE, not sequential. `pairwiseSum()` below is a transcription of
// numpy's `pairwise_sum_DOUBLE` (sequential under 8, 8-way unrolled to 128, recursive split
// above). Without it the order-0 control is a 1-ulp story instead of a bit-exact one.
//
// ============================================================================================
// Ipc / AvgIpc -- rdkit/Chem/GraphDescriptors.py, a DIFFERENT graph and a DIFFERENT problem
// ============================================================================================
//
//   adjMat = (GetDistanceMatrix(mol, 0) == 1)      <- HYDROGEN-SUPPRESSED, unlike everything
//   cPoly  = abs(CharacteristicPolynomial(mol, adjMat))       above
//   Ipc    = sum(cPoly) * InfoEntropy(cPoly)
//   AvgIpc =                InfoEntropy(cPoly)
//
// rdkit/Chem/Graphs.py's CharacteristicPolynomial is Le Verrier-Faddeev-Frame in FLOATING POINT
// and returns n+1 coefficients, leading 1 first. InfoEntropy is rdInfoTheory's C++ one:
// normalise by the sum, SKIP ZERO ENTRIES, natural log divided by log 2.
//
// --------------------------------------------------------------------------------------------
// Ipc IS A THIRD ILL-POSED DESCRIPTOR, AND THIS ONE IS RDKIT'S. Measured, on cpp/hard.smi.
// --------------------------------------------------------------------------------------------
//
// THE COEFFICIENTS ARE INTEGERS. det(xI - A) for a 0/1 adjacency matrix is monic with integer
// coefficients, and every tr(A M_k)/k in the recurrence is an exact integer. A double holds
// integers exactly up to 2^53 and not one bit further, so the whole question is how many bits
// the coefficients need. Measured exactly, by running the recurrence in Python integers:
//
//     heavy atoms      20    30    40    50    55    60    65    70
//     max |c_k| bits   13    20    27    33    37    39    44    48
//
// -- about 0.7 bits per atom. It crosses 53 at roughly 75 atoms, and cpp/hard.smi runs to 245.
//
// PAST THAT POINT RDKIT'S OWN ANSWER MOVES WITH THE ATOM NUMBERING, because the coefficients
// come out of a trace after catastrophic cancellation and the cancellation pattern is a
// function of the numbering. Six random renumberings of one molecule, RDKit's AvgIpc:
//
//      n =  65    one value      (coefficients still exact; agrees with exact integers)
//      n =  70    6 values, agreeing to 9 significant figures
//      n = 100    6 values, 1.5217 .. 1.5548
//      n = 199    6 values, 0.6905 .. 1.5129        <- a factor of 2.2 apart
//      n = 245    6 values, 1.6720 .. 1.6831
//
// So RDKIT'S `AvgIpc` fails PORT_STATUS.md house rule 1's test -- is it a function of the
// molecule? -- for every molecule above about 70 heavy atoms, which is 2.9% of cpp/hard.smi.
// Reproducing RDKit bit-for-bit there would again be reproducing a coin flip.
//
// READ THAT AS A STATEMENT ABOUT RDKIT AND NOT ABOUT THE COLUMN THIS FILE SHIPS. `AvgIpc` IS
// wired, and it is well posed: see the determinism evidence under "WHAT THAT BUYS" below. The
// name is the same and the two values differ on 2.9% of the corpus, which is the whole point.
//
// AND OVERFLOW IS NOT THE PROBLEM, contrary to what this file said before it was measured.
// Over all 100,000 molecules RDKit's Ipc is FINITE EVERY TIME: 100000 finite, 0 inf, 0 nan,
// largest 1.65e88 at n = 245, against a double's 1.8e308. The failure is PRECISION and it
// starts an order of magnitude sooner than overflow would. Anyone reaching for `AvgIpc`
// "because Ipc overflows" has the right conclusion for the wrong reason.
//
// --------------------------------------------------------------------------------------------
// THE RESOLUTION: COMPUTE THE COEFFICIENTS EXACTLY. They are integers, so this is available.
// --------------------------------------------------------------------------------------------
//
// The Faddeev recurrence needs only THREE operations on the big values -- add, subtract, and
// exact division by a small integer k <= n. There is no big multiplication anywhere, because A
// is the adjacency matrix and `A M` is one ROW ADDITION per bond. So `bigCharPoly()` below runs
// the recurrence in fixed-width two's-complement multiword integers, with the word count chosen
// ADAPTIVELY: it starts at one 64-bit word, watches every addition for signed overflow, and
// restarts with twice the width if any addition overflows. Small molecules therefore pay
// single-word arithmetic -- the common case is not slowed down to pay for the tail.
//
// WHAT THAT BUYS, and each of these is checked on the corpus by cpp/verify_ic.py:
//   * DETERMINISTIC everywhere. Integer arithmetic has no cancellation error to depend on an
//     ordering, so the coefficients are the same whatever order the atoms or the bonds arrive in.
//
//     THIS IS NOW MEASURED, AND IT IS WHAT CLOSED THE `Ipc` BUG. An earlier version of this file
//     said `Ipc` was numbering-dependent on 2.8% of the corpus and left all three of its columns
//     unwired. That was RDKIT'S instability, described above, and the determinism evidence for
//     OUR value had a hole in it: six byte-identical outputs existed with no surviving record of
//     what had been fed in, and six identical outputs prove nothing without distinct inputs.
//     The hole is closed. cpp/ic_in0..7.txt and cpp/ic_out0..7.txt are on disk together:
//     SEVEN DISTINCT INPUTS (in0 and in4 coincide, because the canonical-SMILES round trip is a
//     control and is vacuous on an already-canonical corpus) producing EIGHT BYTE-IDENTICAL
//     OUTPUTS, md5 6b5ddecc3a5dd574fe06ea626f032a93, 100,000 molecules x 45 columns. The inputs
//     were re-checked to be the same 100,000 GRAPHS under a renumbering-invariant fingerprint and
//     to differ in the atom order on ~98,700 of them and in the bond list on ~99,300 -- so they
//     are perturbations, not copies. Per column, Ipc / AvgIpc / Log2Ipc each moved on 0 molecules
//     across all seven, with 63,132 distinct values apiece: deterministic, and not trivially so.
//   * BIT-IDENTICAL TO RDKIT wherever RDKit is right. When the largest |c_k| fits in 60 bits the
//     scale factor below is 1, the exact integers convert to double exactly, and the entropy is
//     then computed with RDKit's own formula in RDKit's own order. Measured over all 100,000:
//     of the 96,244 molecules whose largest coefficient needs 40 bits or fewer, 96,221 agree
//     with RDKit TO THE LAST BIT (max relative deviation on the rest, 1.08e-15).
//   * WHERE THEY DIVERGE IT IS RDKIT'S ERROR, not ours -- checked against exact integer
//     arithmetic rather than waved at. In the 41-53 bit band only 638 of 1,271 still match, and
//     in every case examined ours is within ~1e-15 of the exact answer while RDKit is out by up
//     to 1e-2. COEFFICIENT width is not the right predictor: RDKit's Faddeev ITERATE MATRIX
//     crosses 2^53 well before the final coefficients do, so RDKit stops being exact earlier
//     than "max |c_k| <= 2^53" suggests. Hence a measurement here rather than a theorem.
//   * CORRECT past that, where RDKit is not.
//
// THE SCALE FACTOR. Entropy is invariant under a common scaling of its arguments, so the
// coefficients are converted to double divided by 2^E with E = max(0, maxbits - 60). That keeps
// every double in range no matter how large the integers get, costs nothing when E = 0 (which is
// what preserves the bit-identity above), and is exact because it is a power of two.
//
// IPC ITSELF is sum(cPoly) * AvgIpc, which is 2^E * (sum of the scaled doubles) * AvgIpc. It is
// returned exactly when it is representable; if it ever were not, it SATURATES AT DBL_MAX and
// the row's `ipcOverflow` flag is set -- never a silent inf -- and `Log2Ipc` is emitted as a
// third column, finite for every molecule, so nothing is lost at the boundary. On cpp/hard.smi
// the saturation never fires; it exists because "it did not happen on this corpus" is not the
// same claim as "it cannot happen".
//
// THE ONE TRAP, since it produces plausible 24-digit coefficients and a confidently wrong
// oracle: the recurrence is M_0 = 0, c_0 = 1, M_k = A M_{k-1} + c_{k-1} I, c_k = -tr(A M_k)/k.
// Seeding M_1 = A with c_1 = -tr(A) is the SAME recurrence shifted by one step and is fine;
// taking tr(M) instead of tr(A M) at the matching step is not.
//
// ============================================================================================
// WHAT THE CALLER MUST SUPPLY -- all of it already exists at bindings.cpp's boundary
// ============================================================================================
//
// Boundary as of 2026-08-27: atom_i is (n_atoms, 10) -- Z, deg, nH, fchg, hyb, arom, ring, cip,
// nring, tval. This file reads five of the ten and needs nothing that is not already there; the
// per-atom ring COUNT in column 8 and the total valence in column 9 are not used here.
//
//   per atom   z      GetAtomicNum()               atom_i[:, 0]
//              deg    GetDegree()   (heavy)        atom_i[:, 1]   (cross-check only)
//              nh     GetTotalNumHs(False)         atom_i[:, 2]
//              chg    GetFormalCharge()            atom_i[:, 3]
//              arom   GetIsAromatic()              atom_i[:, 5]
//   per bond   u, v   begin/end atom index         bond_i[:, 0], bond_i[:, 1]
//              code   SMARTS bond code             bond_i[:, 4]
//              order  GetBondTypeAsDouble()        bond_d
//
// `u` MUST BE THE BEGIN ATOM and `v` THE END ATOM, because that is what makes a dative bond's
// donor and acceptor distinguishable; _extract.py already fills them that way.
//
// The bond code is _extract.py's: bit 0 SINGLE, bit 1 DOUBLE, bit 2 TRIPLE, bit 3 the aromatic
// FLAG, and 0 for everything RDKit will not name (DATIVE and friends). A bond whose TYPE is
// AROMATIC is exactly `(code & 7) == 0 && (code & 8) != 0`, and that is the only bond that gets
// SYM_AROMATIC -- a TRIPLE bond carrying the aromatic flag, which cpp/mols.smi contains, is
// still a triple bond to mordred's kekulized `GetBondType()` and is still a triple bond here.
//
// Hydrogens are not in the boundary and do not need to be: `nh` per heavy atom is exactly what
// `Chem.AddHs` would add, and an isotopic [2H] that is already an atom in its own right arrives
// as an ordinary Z=1 heavy row and is not double counted. That correspondence is checked
// per molecule by cpp/verify_ic.py against the real `Chem.AddHs` graph.
//
// --------------------------------------------------------------------------------------------
// WIRING. DONE -- bindings.cpp emits 43 columns from this file: the 42 IC columns and `AvgIpc`.
// What follows is the record of what the wiring must keep true, not a to-do list; every item is
// something that would break a value silently rather than loudly if it were changed.
//
//   1. bindings.cpp, next to `crippen_fill`: fill an `infoic::Mol` from the SAME `AI` / `BI`
//      pointers that loop already walks. Per atom take columns A_Z, A_NH, A_FCHG, A_AROM; per
//      bond take B_U, B_V, B_CODE and `BD[b0 + b]`. Nothing else is needed, and nothing new has
//      to cross the boundary -- note in particular that `hume_blocks.h`'s own `Mol` does NOT
//      carry the SMARTS bond code, so read it from `BI` directly as crippen_fill does, rather
//      than adding a field there.
//
//   2. `u` MUST STAY THE BEGIN ATOM and `v` the end atom. That is the only thing that tells a
//      dative bond's donor from its acceptor, and the donor is the one whose valence the bond
//      does not count towards. _extract.py already fills them that way; a loop that normalises
//      to (min, max) would silently change B for every molecule with a dative bond.
//
//   3. Call `infoic::selfCheck()` once at module load, beside `criptyper::selfCheck()` and
//      `esttyper::selfCheck()`. It checks the generated tables, the numpy summation order, and
//      -- the one that matters -- that the worked example is still invariant under all 36
//      transpositions of its atoms.
//
//   4. Hoist one `infoic::CodeBuilder` outside the per-molecule loop and pass it to
//      `compute(m, row, &cb)`. It is scratch only; it memoises nothing across molecules, so this
//      changes no value. NO LONGER REQUIRED FOR SPEED: as of 2026-08-28 the fallback inside
//      `compute()` is a function-local `thread_local` rather than a stack object, so a caller
//      that passes nothing gets the same reuse. Passing one explicitly is still tidier.
//
//   5. `compute()` VS `computeIC()` -- WHICH ONE THE EXTENSION CALLS IS A COST DECISION, and it
//      has been taken twice in opposite directions, so read the reason and not just the line.
//
//      `computeIC()` is `compute()` with the Ipc block skipped. The 42 IC columns are
//      BIT-IDENTICAL either way -- nothing above the Ipc block reads anything the Ipc block
//      writes -- so the only difference is that `compute()` also runs the exact-integer Le
//      Verrier-Faddeev-Frame recurrence, which is O(n^3) in the HEAVY-atom count.
//
//      It was switched to `computeIC()` when Ipc's three columns were unwired, because the
//      recurrence was 68% of this file's CPU on the 5,000-molecule sample it was measured on --
//      81.9% on the whole corpus, which is the honest number -- and the result went on the
//      floor. It is back to
//      `compute()` now that `AvgIpc` -- which IS one of the 865 -- is wired and the determinism
//      question above is closed. What made that affordable is recorded on `faddeevMultiword`
//      below: the multiword path was 90% of the Ipc block's cost and is 3.07x faster than it
//      was, so the block costs 67.0 us/mol over all 100,000 of cpp/hard.smi instead of 179.3.
//
//      MEASURE IT WITH `./cpp/infocontent ipcbench`, which runs `compute` and `computeIC` back
//      to back inside one repetition and reports the DIFFERENCE, so the number is the block and
//      not two unpaired runs subtracted. Whole corpus, quiet box at load1 1.9, 5 reps:
//
//                                  us/mol      spread     the Ipc block
//          HEAD                    218.93        1.46         179.30
//          + CSR restructure       159.98        0.41         120.12
//          + templated W           107.71        1.28          67.02
//
//      The `computeIC` arm is 39.66 / 39.84 / 40.72 across those three, i.e. flat: none of this
//      touched the 42 IC columns, which is the control that says so.
//
//      IF THE THREE Ipc COLUMNS ARE EVER UNWIRED AGAIN, switch back to `computeIC()` in the same
//      edit. Leaving `compute()` in place would silently reinstate the whole recurrence for
//      nothing, which is exactly the bug that was found here before.
//
//   6. `AvgIpc` IS `row.v[C_AVGIPC]`, NOT `row.v[N_IC]`. The Ipc block is three columns in the
//      order (Ipc, AvgIpc, Log2Ipc), so the census member is the MIDDLE one and `v[N_IC]` is
//      `Ipc` -- a number 80-odd orders of magnitude larger that would look like a value and pass
//      every shape check. The emit loop and `all_column_names_tail()` must move together, or the
//      column ships under the wrong name, which no test in this repo would catch.
//
//   Column names come from `infoic::columnNames()`. 33 of the 42 InformationContent columns and
//   `AvgIpc` are the ones that survive data/dedupe.json. `Ipc` and `Log2Ipc` are COMPUTED but not
//   emitted: they are not members of the 865, and the two of them are the only columns in this
//   family that are not O(1)-bounded -- `Ipc` reaches 1.65e88 and saturates by design, `Log2Ipc`
//   is -inf where `Ipc` is 0 -- so putting them in front of a downstream model is a decision for
//   the project owner rather than a free consequence of having computed them. Emitting them is
//   two lines in bindings.cpp and costs nothing if the dedupe is ever rerun at another threshold
//   and asks for them.
//
//   #include "infocontent.h"
//   infoic::selfCheck();                       // once, at module load
//   infoic::Row r; infoic::compute(mol, r);    // all 45; the wiring emits 42 + AvgIpc
//   infoic::Row r; infoic::computeIC(mol, r);  // ... or the 42 alone, if Ipc is not wanted
//
#ifndef HUME_INFOCONTENT_H
#define HUME_INFOCONTENT_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ic_tables.h"

namespace infoic {

// --------------------------------------------------------------------------------------------
// Columns. 7 families x 6 orders, then the three Ipc outputs. The 33 that survive data/
// dedupe.json are a SUBSET of the 42; computing the other 9 is free (they share every Ag) and
// dropping them would violate house rule 7 the moment the dedupe is rerun at another threshold.
// --------------------------------------------------------------------------------------------
enum { MAX_ORDER = 5, N_ORDERS = 6, N_FAM = 7, N_IC = N_FAM * N_ORDERS, N_COLS = N_IC + 3 };
enum { F_IC = 0, F_TIC, F_SIC, F_BIC, F_CIC, F_MIC, F_ZMIC };
enum { C_IPC = N_IC, C_AVGIPC, C_LOG2IPC };

inline const char *const *columnNames() {
  static const char *n[N_COLS] = {
      "IC0",   "IC1",   "IC2",   "IC3",   "IC4",   "IC5",
      "TIC0",  "TIC1",  "TIC2",  "TIC3",  "TIC4",  "TIC5",
      "SIC0",  "SIC1",  "SIC2",  "SIC3",  "SIC4",  "SIC5",
      "BIC0",  "BIC1",  "BIC2",  "BIC3",  "BIC4",  "BIC5",
      "CIC0",  "CIC1",  "CIC2",  "CIC3",  "CIC4",  "CIC5",
      "MIC0",  "MIC1",  "MIC2",  "MIC3",  "MIC4",  "MIC5",
      "ZMIC0", "ZMIC1", "ZMIC2", "ZMIC3", "ZMIC4", "ZMIC5",
      "Ipc",   "AvgIpc", "Log2Ipc"};
  return n;
}

// Optional instrumentation. `nullptr` in production and the pointer is only tested once per
// section, so the normal path pays nothing measurable; the profiling path pays ~18 chrono reads
// per molecule, which is under 1% of the figure being measured. Used by `infocontent profile`.
// PROFILING CLOCK: THREAD CPU TIME, not wall. On a contended box a steady_clock read charges
// every phase for whatever descheduling happened to land inside it, and this machine has run at
// load 130 on 12 cores -- a wall-clock profile there reported the same phase at 74 us and at 601
// us on consecutive runs of the same binary on the same input. CLOCK_THREAD_CPUTIME_ID does not
// tick while the thread is off-CPU, so the breakdown is stable to a few percent under exactly
// the same load. Instrumentation only; `prof` is nullptr in production.
struct CpuClock {
  static double nowUs() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
#else
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
  }
};

struct Profile {
  double build = 0, dfs = 0, ipc = 0;   // `dfs` is the ONE traversal that serves all six orders
  double codes[N_ORDERS] = {}, group[N_ORDERS] = {}, entropy[N_ORDERS] = {};
  long long paths[N_ORDERS] = {}, classes[N_ORDERS] = {}, mols = 0, atoms = 0;
};

struct Row {
  double v[N_COLS];
  bool ipcOverflow = false;   // Ipc saturated at DBL_MAX; Log2Ipc is still exact
  int ipcMaxCoeffBits = 0;    // bit length of the largest EXACT char-poly coefficient. Above 53
                              // RDKit's own double arithmetic has stopped being exact; the
                              // harness reports the distribution rather than assuming a cutoff.
};

// --------------------------------------------------------------------------------------------
// Bond symbols. The VALUES are arbitrary -- a code is a sorted multiset of paths and only
// equality of codes ever matters -- but they are kept distinct and stable so a dumped code is
// readable. SYM_AROMATIC is repair R1: it exists precisely so that no Kekule choice can reach
// the codes.
// --------------------------------------------------------------------------------------------
enum : uint8_t { SYM_OTHER = 0, SYM_SINGLE = 1, SYM_DOUBLE = 2, SYM_TRIPLE = 3,
                 SYM_AROMATIC = 4, SYM_NONE = 15 };

inline uint8_t symbolFromCode(int code) {
  if (code & 1) return SYM_SINGLE;
  if (code & 2) return SYM_DOUBLE;
  if (code & 4) return SYM_TRIPLE;
  if (code & 8) return SYM_AROMATIC;   // GetBondType() == AROMATIC; see the header note
  return SYM_OTHER;                    // DATIVE and anything else RDKit will not name
}

// --------------------------------------------------------------------------------------------
// Input: the HEAVY-ATOM graph, in exactly the shape bindings.cpp already has. Hydrogens are
// materialised inside compute(); see build().
// --------------------------------------------------------------------------------------------
struct Mol {
  int n = 0, nb = 0;
  std::vector<uint8_t> z, nh, arom;
  std::vector<int8_t> chg;
  std::vector<int32_t> bu, bv;
  std::vector<uint8_t> bcode;
  std::vector<double> bord;

  void alloc(int na, int nbonds) {
    n = na; nb = nbonds;
    z.assign(na, 0); nh.assign(na, 0); arom.assign(na, 0); chg.assign(na, 0);
    bu.assign(nbonds, 0); bv.assign(nbonds, 0); bcode.assign(nbonds, 0);
    bord.assign(nbonds, 0.0);
  }
};

// --------------------------------------------------------------------------------------------
// The hydrogen-added graph mordred actually descriptors. Heavy atoms keep their indices; the
// hydrogens `Chem.AddHs` would append are appended here in the same order (heavy atom 0's
// hydrogens first), which is not load bearing -- nothing downstream reads an index -- but makes
// a dump comparable with RDKit's atom-by-atom.
// --------------------------------------------------------------------------------------------
struct HGraph {
  int N = 0;
  std::vector<uint8_t> z, deg;
  std::vector<int32_t> start, nbr;
  std::vector<uint8_t> sym;
  std::vector<int32_t> cnt, cur;      // scratch, members so a reused HGraph stops allocating

  void build(const Mol &m) {
    int nhtot = 0;
    for (int i = 0; i < m.n; i++) nhtot += m.nh[i];
    N = m.n + nhtot;
    z.assign(N, 1); deg.assign(N, 1);
    for (int i = 0; i < m.n; i++) z[i] = m.z[i];
    cnt.assign(N, 0);
    for (int b = 0; b < m.nb; b++) { cnt[m.bu[b]]++; cnt[m.bv[b]]++; }
    for (int i = 0; i < m.n; i++) cnt[i] += m.nh[i];       // the C-H bonds
    for (int q = m.n; q < N; q++) cnt[q] = 1;              // every hydrogen is terminal
    start.assign(N + 1, 0);
    for (int i = 0; i < N; i++) start[i + 1] = start[i] + cnt[i];
    nbr.assign(start[N], 0); sym.assign(start[N], SYM_SINGLE);
    cur.assign(start.begin(), start.end() - 1);
    for (int b = 0; b < m.nb; b++) {
      const uint8_t s = symbolFromCode(m.bcode[b]);
      const int u = m.bu[b], v = m.bv[b];
      nbr[cur[u]] = v; sym[cur[u]++] = s;
      nbr[cur[v]] = u; sym[cur[v]++] = s;
    }
    int h = m.n;
    for (int i = 0; i < m.n; i++)
      for (int q = 0; q < m.nh[i]; q++, h++) {
        nbr[cur[i]] = h; sym[cur[i]++] = SYM_SINGLE;
        nbr[cur[h]] = i; sym[cur[h]++] = SYM_SINGLE;
      }
    for (int i = 0; i < N; i++) deg[i] = (uint8_t)std::min(start[i + 1] - start[i], 255);
  }
};

// --------------------------------------------------------------------------------------------
// numpy's pairwise summation, transcribed from numpy/core/src/umath/loops_utils.h.src
// (`pairwise_sum_DOUBLE`). mordred's entropies go through `np.sum`, so reproducing the ORDER of
// the additions is what makes the order-0 control bit-exact rather than 1-ulp-ish. Any change
// here is a change to the reference, not an optimisation.
// --------------------------------------------------------------------------------------------
inline double pairwiseSum(const double *a, size_t n) {
  const size_t PW_BLOCKSIZE = 128;
  if (n < 8) {
    double res = 0.0;
    for (size_t i = 0; i < n; i++) res += a[i];
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    double r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
}

// mordred: shannon_entropy(a, w) = -np.sum(w * (a/N) * log2(a/N)), N = np.sum(a).
// `w` is scalar 1 for IC and a per-class vector for MIC / ZMIC. The temporaries are built in the
// same order numpy would build them.
inline double shannonEntropy(const double *a, const double *w, size_t k, double *scratch) {
  const double N = pairwiseSum(a, k);
  for (size_t j = 0; j < k; j++) {
    const double p = a[j] / N;
    scratch[j] = (w ? w[j] : 1.0) * (p * std::log2(p));
  }
  return -pairwiseSum(scratch, k);
}

// --------------------------------------------------------------------------------------------
// B: the KEKULIZED bond-order sum over the H-added mol, rebuilt without kekulizing. See the
// header note. Verified 100,000/100,000 against mordred's own value on cpp/hard.smi.
// --------------------------------------------------------------------------------------------
inline double kekuleBondOrderSum(const Mol &m, std::vector<double> &used) {
  double tot = 0.0;
  int narom = 0;
  used.assign(m.n, 0.0);
  for (int i = 0; i < m.n; i++) { tot += m.nh[i]; used[i] += m.nh[i]; }  // the C-H bonds
  for (int b = 0; b < m.nb; b++) {
    const int u = m.bu[b], v = m.bv[b], code = m.bcode[b];
    const bool aromatic = ((code & 7) == 0) && (code & 8);
    if (aromatic) { narom++; used[u] += 1.0; used[v] += 1.0; continue; }
    tot += m.bord[b];
    if (code == 0) {
      used[v] += m.bord[b];         // DATIVE: 0 to the donor (u), the order to the acceptor (v)
    } else {
      used[u] += m.bord[b];
      used[v] += m.bord[b];
    }
  }
  int need = 0;
  for (int i = 0; i < m.n; i++) {
    if (!m.arom[i]) continue;
    const int dv = ic_tbl::DEFAULT_VALENCE[m.z[i]];
    if (dv < 0) continue;                       // no default valence: cannot be a candidate
    const int ev = m.chg[i] == 0
                       ? dv
                       : (ic_tbl::N_OUTER_ELECS[m.z[i]] >= 5 ? dv + m.chg[i]
                                                             : dv - std::abs((int)m.chg[i]));
    if (used[i] < (double)ev) need++;
  }
  return tot + (double)narom + 0.5 * (double)need;
}

// --------------------------------------------------------------------------------------------
// The atom-equivalence codes, repaired. A code is the sorted multiset of the root-to-leaf paths
// of the DISTANCE-LAYERED tree (repair R2), each path a run of (Z, degree) nodes joined by bond
// SYMBOLS (repair R1), 0xFF-terminated and zero-padded so that concatenation is injective.
//
// ============================================================================================
// PKey: THAT KEY PACKED INTO 128 BITS. IT IS AN INJECTION, NOT A HASH. Read this before
// trusting it, because the obvious version of this optimisation is not safe.
// ============================================================================================
//
// The key used to be 24 BYTES -- (Z, degree) for the root, then one (symbol, Z, degree) triple
// per edge, 0xFF-terminated and zero-padded -- built with a memset per path, copied into a byte
// arena, and ordered with memcmp. Nothing downstream ever reads a byte of it: the only things a
// code is used for are EQUALITY (which atoms share a class) and ORDER (which class comes first,
// and that matters only because numpy's pairwise summation is not associative). So the bytes
// were never the point, and moving 24 of them per path was most of the cost.
//
// THE TEMPTING VERSION IS A 64-BIT HASH OF THE PATH LIST, AND IT IS NOT SAFE HERE. A collision
// merges two distinct equivalence classes, lowers the entropy, and leaves no symptom -- and no
// amount of corpus testing turns "we saw none" into "there are none". So this is not a hash.
// Every bit of the key that can vary is carried, the map is injective, and the NUMERIC order of
// the packed value is EXACTLY the memcmp order of the bytes it replaces. There is no collision
// to have a probability of.
//
//   field   bits  what it replaces
//   R        16   the root, Z << 8 | degree                      key bytes 0 and 1
//   L1..L5   19   one per level, most significant first          key bytes 2.. in triples
//
//   a level that CONTINUES  : 1 + (symbol << 16 | Z << 8 | degree)   in [1, 327680]
//   a level that TERMINATES : 0x7FFFF = 524287                        (the 0xFF byte)
//   a level PAST the end    : 0                                       (the zero padding)
//
// The three ranges are disjoint and ordered 0 < continue < terminate, which is precisely the
// byte order 0x00 < {symbol, at most 4} < 0xFF at the position the symbol byte occupied; and
// inside a continuing level the (symbol, Z, degree) packing is big-endian, so it orders like the
// three bytes it stands for. 16 + 5*19 = 111 bits: (R, L1, L2) left-aligned in `hi` and
// (L3, L4, L5) left-aligned in `lo`, so comparing (hi, lo) lexicographically compares the fields
// in order.
//
// A path can be at most MAX_ORDER edges long, so five levels is not a bound that can be hit --
// the static_assert below ties the two together. `selfCheck()` re-derives the whole partition
// from the ORIGINAL 24-byte byte keys on the worked example, and `./cpp/infocontent keycheck`
// does the same over the corpus: same number of distinct codes, same class sizes, same class
// ORDER, per molecule per order. That is the collision instrumentation the brief asked for,
// applied to an encoding that cannot collide.
// --------------------------------------------------------------------------------------------
enum { MAX_PATHS = 65536 };
static_assert(MAX_ORDER == 5, "PKey packs exactly MAX_ORDER = 5 levels of 19 bits");
enum : uint32_t { PK_TERM = 0x7FFFFu, PK_MAX_CONT = 1u + (4u << 16 | 255u << 8 | 255u) };
static_assert(PK_MAX_CONT < PK_TERM, "a continuing level must sort before a terminating one");

struct PKey {
  uint64_t hi, lo;
  bool operator<(const PKey &o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
  bool operator==(const PKey &o) const { return hi == o.hi && lo == o.lo; }
};

// Level `lv` in 1..MAX_ORDER. R occupies hi[48..63]; L1 hi[29..47], L2 hi[10..28],
// L3 lo[45..63], L4 lo[26..44], L5 lo[7..25]. No field crosses a word and none overlaps.
inline void pkSetField(uint64_t &hi, uint64_t &lo, int lv, uint32_t f) {
  switch (lv) {
    case 1: hi |= (uint64_t)f << 29; break;
    case 2: hi |= (uint64_t)f << 10; break;
    case 3: lo |= (uint64_t)f << 45; break;
    case 4: lo |= (uint64_t)f << 26; break;
    default: lo |= (uint64_t)f << 7; break;
  }
}

// --------------------------------------------------------------------------------------------
// ONE traversal for ALL SIX ORDERS. The old builder ran a BFS and a DFS per (atom, order), six
// times over. It did not need to: a key TERMINATES at its own depth and is zero-padded from
// there, so the key a dead end at depth 3 produces is the SAME key at order 3, 4 and 5 -- and a
// node at depth d is a leaf of the depth-k tree exactly when d == k, or when it has no children
// at all and d <= k. So one depth-MAX_ORDER DFS emits every path of every order:
//
//     emit the key at order max(d, 1) .. (has layered children ? max(d, 1) : MAX_ORDER)
//
// with the d == 0 case falling out correctly -- a root WITH children is a leaf only at order 0,
// which is handled separately, and an isolated atom emits its one path at every order.
//
// WHAT THAT IS WORTH. Measured over 5,000 molecules of cpp/hard.smi, the old scheme walked
// 5*N1 + 4*N2 + 3*N3 + 2*N4 + N5 tree nodes per molecule where the new one walks
// N1 + .. + N5 -- with the measured layer sizes that is 4,478 against 2,006, a factor of 2.2,
// plus five BFS layerings per atom collapsing to one.
// --------------------------------------------------------------------------------------------
class CodeBuilder {
 public:
  // Scratch, reused across molecules. It memoises NOTHING -- reset() re-sizes and re-zeros
  // everything from the graph in hand -- so hoisting it changes no value, only the number of
  // allocations.
  std::vector<PKey> arena[N_ORDERS];
  std::vector<int32_t> off[N_ORDERS];       // off[k][i] .. off[k][i+1], for k >= 1
  std::vector<int32_t> ordIdx;
  std::vector<double> cnt, wmass, wz, scratch, used;
  HGraph hg;
  std::vector<double> cpoly;

  void reset(const HGraph &g) {
    g_ = &g;
    dist_.assign(g.N, -1);
    stamp_.assign(g.N, 0);
    epoch_ = 0;
    bfs_.clear();
    bfs_.reserve(g.N);
    ordIdx.resize(g.N);
    for (int k = 1; k <= MAX_ORDER; k++) {
      arena[k].clear();
      off[k].assign(g.N + 1, 0);
    }
  }

  // Fills arena[1..MAX_ORDER] with every atom's sorted path multiset.
  void buildAll() {
    const HGraph &g = *g_;
    for (int root = 0; root < g.N; root++) {
      root_ = root;
      layer(root);
      const uint64_t rhi = ((uint64_t)(((uint32_t)g.z[root] << 8) | g.deg[root])) << 48;
      walk(root, 0, rhi, 0);
      for (int k = 1; k <= MAX_ORDER; k++) {
        std::sort(arena[k].begin() + off[k][root], arena[k].end());
        off[k][root + 1] = (int32_t)arena[k].size();
      }
    }
  }

  long long pathsAt(int k) const { return (long long)arena[k].size(); }

 private:
  // BFS to depth MAX_ORDER. `dist_` is the true graph distance for everything it reaches, so
  // "child" -- a neighbour at distance d+1 -- means the same thing at every order.
  void layer(int root) {
    const HGraph &g = *g_;
    ++epoch_;
    bfs_.clear();
    bfs_.push_back(root);
    stamp_[root] = epoch_;
    dist_[root] = 0;
    for (size_t h = 0; h < bfs_.size(); h++) {
      const int u = bfs_[h];
      if (dist_[u] >= MAX_ORDER) continue;
      for (int e = g.start[u]; e < g.start[u + 1]; e++) {
        const int v = g.nbr[e];
        if (stamp_[v] != epoch_) {
          stamp_[v] = epoch_;
          dist_[v] = dist_[u] + 1;
          bfs_.push_back(v);
        }
      }
    }
  }

  void walk(int u, int d, uint64_t hi, uint64_t lo) {
    const HGraph &g = *g_;
    bool kids = false;
    if (d < MAX_ORDER) {
      for (int e = g.start[u]; e < g.start[u + 1]; e++) {
        const int v = g.nbr[e];
        if (stamp_[v] != epoch_ || dist_[v] != d + 1) continue;   // layered: children only
        kids = true;
        const uint32_t f =
            1u + (((uint32_t)g.sym[e] << 16) | ((uint32_t)g.z[v] << 8) | (uint32_t)g.deg[v]);
        uint64_t nh = hi, nl = lo;
        pkSetField(nh, nl, d + 1, f);
        walk(v, d + 1, nh, nl);
      }
    }
    const int klo = d < 1 ? 1 : d;
    const int khi = kids ? d : MAX_ORDER;
    if (klo > khi) return;                     // the root of a tree that has children: not a leaf
    uint64_t th = hi, tl = lo;
    if (d < MAX_ORDER) pkSetField(th, tl, d + 1, PK_TERM);
    const PKey key{th, tl};
    for (int k = klo; k <= khi; k++) {
      if (arena[k].size() - (size_t)off[k][root_] >= MAX_PATHS)
        throw std::runtime_error("infoic: path explosion");
      arena[k].push_back(key);
    }
  }

  const HGraph *g_ = nullptr;
  std::vector<int32_t> dist_;
  std::vector<int32_t> stamp_;
  std::vector<int32_t> bfs_;
  int32_t epoch_ = 0;
  int root_ = 0;
};

// --------------------------------------------------------------------------------------------
// Ipc: EXACT INTEGER Le Verrier-Faddeev-Frame on the HYDROGEN-SUPPRESSED graph. See the header
// comment for why exact and not floating point: the coefficients are integers that outgrow a
// double at around 75 heavy atoms, and RDKit's own answer moves with the atom numbering above
// that. Zero coefficients are skipped by the entropy, which is what rdInfoTheory's
// `if (tPtr[i])` does.
// --------------------------------------------------------------------------------------------
// Fixed-width two's-complement multiword integers. Only what the recurrence needs: add with
// signed-overflow detection, negate, and exact division by a small positive integer. No
// multiplication -- `A M` is a row addition per bond, which is the whole reason exact arithmetic
// is affordable here. Limbs are 64-bit, little endian, and the carry is done in plain C++ rather
// than __int128 so this stays portable C++17.
namespace big {

// ADD, WITH W A COMPILE-TIME CONSTANT. This used to take W as a runtime argument, and that is
// what made it expensive: it is called once per matrix ELEMENT inside the Faddeev recurrence --
// 2 * n_bonds * n times per step -- and a two- or four-limb carry chain behind an unknown trip
// count cannot be unrolled, so every limb pays a loop-back branch and a dependent compare that a
// fixed-length sequence does not.
//
// ONLY THE LOOP BOUND CHANGED. The body is the runtime version's body character for character:
// same limb order, same carry rule, same signed-overflow test. It is nevertheless checked by
// measurement rather than left as an argument -- see the bit-identity run over all eight
// ic_in*.txt dumps recorded in PORT_STATUS.md.
//
// There is deliberately no runtime-W twin. The O(n) and O(W) callers (the trace, the diagonal
// update) are inside the same templated recurrence and get W for free, and a second copy of a
// carry chain is exactly the kind of thing that gets fixed in one place only.
template <int W>
inline bool addT(uint64_t *d, const uint64_t *s) {
  const uint64_t ds = d[W - 1] >> 63, ss = s[W - 1] >> 63;
  uint64_t carry = 0;
  for (int i = 0; i < W; i++) {
    const uint64_t a = d[i], b = s[i];
    const uint64_t t = a + b;
    const uint64_t c1 = (uint64_t)(t < a);
    const uint64_t t2 = t + carry;
    carry = c1 | (uint64_t)(t2 < t);
    d[i] = t2;
  }
  return (ds == ss) && ((d[W - 1] >> 63) != ds);      // signed overflow
}

inline bool isNeg(const uint64_t *x, int W) { return (x[W - 1] >> 63) != 0; }

// Two's complement negate. Returns true on the one input it cannot represent, the most negative
// value, which is a width overflow like any other and must widen rather than wrap.
inline bool negate(uint64_t *x, int W) {
  const uint64_t was = x[W - 1] >> 63;
  bool zero = true;
  for (int i = 0; i < W; i++) if (x[i]) zero = false;
  uint64_t carry = 1;
  for (int i = 0; i < W; i++) {
    const uint64_t t = ~x[i] + carry;
    carry = (carry && t == 0) ? 1 : 0;
    x[i] = t;
  }
  return !zero && ((x[W - 1] >> 63) == was);
}

// x /= k, exact, k in [1, 2^32). Sign-magnitude round trip; the caller guarantees divisibility.
inline void divSmall(uint64_t *x, uint64_t k, int W) {
  const bool neg = isNeg(x, W);
  if (neg) negate(x, W);
  uint64_t r = 0;
  for (int i = W - 1; i >= 0; i--) {
    const uint64_t hi = x[i] >> 32, lo = x[i] & 0xffffffffULL;
    uint64_t cur = (r << 32) | hi;
    const uint64_t qh = cur / k;
    r = cur % k;
    cur = (r << 32) | lo;
    const uint64_t ql = cur / k;
    r = cur % k;
    x[i] = (qh << 32) | ql;
  }
  if (neg) negate(x, W);
}

// |x| -> a correctly-rounded double times 2^(-scaleExp), and the bit length of |x|.
// The 64-bit window plus a sticky bit is what makes the double conversion correctly rounded
// instead of merely close.
inline double toScaledDouble(const uint64_t *x, int W, int scaleExp, int *bitlen) {
  uint64_t mag[64];
  for (int i = 0; i < W; i++) mag[i] = x[i];
  if (isNeg(x, W)) negate(mag, W);
  int top = -1;
  for (int i = W - 1; i >= 0; i--)
    if (mag[i]) { top = i; break; }
  if (top < 0) { if (bitlen) *bitlen = 0; return 0.0; }
  int hb = 63;
  while (hb >= 0 && !((mag[top] >> hb) & 1ULL)) hb--;
  const int bits = top * 64 + hb + 1;
  if (bitlen) *bitlen = bits;
  // Assemble the top 64 bits of the magnitude, with a STICKY bit for everything below, so the
  // conversion to double is correctly rounded rather than merely close. In both branches
  // |x| == win * 2^shift exactly (up to the sticky bit).
  const int shift = bits - 64;
  uint64_t win;
  bool sticky = false;
  if (shift <= 0) {
    win = mag[0] << (-shift);                  // bits <= 64, so limb 0 holds all of it
  } else {
    const int wi = shift / 64, bo = shift % 64;
    win = mag[wi] >> bo;
    if (bo && wi + 1 <= top) win |= mag[wi + 1] << (64 - bo);
    for (int i = 0; i < wi; i++) if (mag[i]) sticky = true;
    if (bo && (mag[wi] & ((1ULL << bo) - 1ULL))) sticky = true;
  }
  if (sticky) win |= 1ULL;
  return std::ldexp((double)win, shift - scaleExp);
}

}  // namespace big

// --------------------------------------------------------------------------------------------
// The Faddeev recurrence in multiword integers, at a FIXED word count W. Fills `C` with the n+1
// coefficients (W limbs each, little endian) and returns true; returns false if any addition
// overflowed W words, in which case the caller widens and starts the molecule again.
//
// THIS IS THE 90% CASE. Only 2.08% of cpp/hard.smi reaches it -- 2,076 molecules of 100,000,
// the ones whose characteristic-polynomial coefficients pass 62 bits, which is essentially the
// ones with the most atoms -- but it is an O(n^3) recurrence and those are the largest n, so it
// carried ~90% of the entire Ipc block's CPU. On that subset alone, 5 reps, paired:
//
//         HEAD 7790.67 us/mol  ->  CSR 5017.69 (1.55x)  ->  + templated W 2534.22 (3.07x)
//
// The 42 IC columns on the same 2,076 molecules cost 192.87 us/mol, so before this rewrite the
// Ipc block was FORTY TIMES the rest of the descriptor there. It is still thirteen times.
//
// TWO THINGS MAKE IT QUICK, and each is bit-identity preserving for a reason that is worth
// stating rather than trusting:
//
//   * `T = A M` IS DRIVEN BY THE SAME CSR THE FAST PATH USES, and each row is SEEDED by copying
//     its first neighbour's row instead of zeroing all of T and adding every neighbour into it.
//     The zero pass was n^2 * W words per step -- as much memory traffic as the additions it was
//     preparing for, and pure waste, since every row that is not isolated is overwritten anyway.
//
//     THE PER-ROW ACCUMULATION ORDER IS UNCHANGED, which is what makes overflow detection
//     identical rather than merely similar. The old bond loop gave row u its neighbours in
//     increasing bond index; the CSR is built by walking the bond list in that same order, so it
//     hands back the same sequence. The only partial sum that disappears is `0 + M[v0]`, and
//     adding to zero can never change a sign, so it could never have been the addition that
//     overflowed. The set of W for which the recurrence overflows is therefore the same set, and
//     the widening decision is the same decision.
//
//   * W IS A TEMPLATE PARAMETER, so `big::addT<W>` unrolls into a fixed carry chain. The old code
//     called the carry loop once per matrix ELEMENT with W unknown at compile time.
//
// The early `!overflow` bail is per ROW here and was per BOND before. That changes only how much
// work is thrown away after an overflow is seen, never whether one is seen: both forms are a lazy
// evaluation of "does any addition in the full sequence overflow", and both truncate only after
// the answer is already true.
// --------------------------------------------------------------------------------------------
template <int W>
inline bool faddeevMultiword(const Mol &m, const int32_t *astart, const int32_t *anbr,
                             std::vector<uint64_t> &C) {
  const int n = m.n;
  const size_t rowlen = (size_t)n * (size_t)W;
  std::vector<uint64_t> M(rowlen * (size_t)n, 0), T(rowlen * (size_t)n, 0);
  C.assign((size_t)(n + 1) * (size_t)W, 0);
  bool overflow = false;
  const uint64_t ONE = 1;
  for (int b = 0; b < m.nb; b++) {                     // M_1 = A
    M[(size_t)m.bu[b] * rowlen + (size_t)m.bv[b] * W] = ONE;
    M[(size_t)m.bv[b] * rowlen + (size_t)m.bu[b] * W] = ONE;
  }
  C[0] = ONE;                                          // c_0 = 1
  uint64_t cprev[W], acc[W];
  for (int q = 0; q < W; q++) { cprev[q] = 0; acc[q] = 0; }
  // c_1 = -tr(M_1) = -tr(A) = 0, but computed rather than assumed.
  for (int i = 0; i < n && !overflow; i++)
    overflow |= big::addT<W>(acc, &M[(size_t)i * rowlen + (size_t)i * W]);
  overflow |= big::negate(acc, W);
  for (int q = 0; q < W; q++) { C[(size_t)W + q] = acc[q]; cprev[q] = acc[q]; }

  for (int k = 2; k <= n && !overflow; k++) {
    for (int i = 0; i < n && !overflow; i++)           // M += c_{k-1} I
      overflow |= big::addT<W>(&M[(size_t)i * rowlen + (size_t)i * W], cprev);
    for (int u = 0; u < n && !overflow; u++) {         // T = A M, one row per atom
      uint64_t *tu = &T[(size_t)u * rowlen];
      const int e0 = astart[u], e1 = astart[u + 1];
      if (e0 == e1) { std::memset(tu, 0, rowlen * sizeof(uint64_t)); continue; }
      std::memcpy(tu, &M[(size_t)anbr[e0] * rowlen], rowlen * sizeof(uint64_t));
      for (int e = e0 + 1; e < e1; e++) {
        const uint64_t *mv = &M[(size_t)anbr[e] * rowlen];
        for (int j = 0; j < n; j++)
          overflow |= big::addT<W>(tu + (size_t)j * W, mv + (size_t)j * W);
      }
    }
    M.swap(T);
    for (int q = 0; q < W; q++) acc[q] = 0;
    for (int i = 0; i < n && !overflow; i++)
      overflow |= big::addT<W>(acc, &M[(size_t)i * rowlen + (size_t)i * W]);
    overflow |= big::negate(acc, W);
    if (overflow) break;
    big::divSmall(acc, (uint64_t)k, W);                // exact: tr(A M_k) is divisible by k
    for (int q = 0; q < W; q++) { C[(size_t)k * W + q] = acc[q]; cprev[q] = acc[q]; }
  }
  return !overflow;
}

// Exact characteristic-polynomial coefficients |c_0| .. |c_n| of the hydrogen-suppressed
// adjacency matrix, returned as doubles scaled by 2^-E with E chosen so that E == 0 whenever the
// integers fit in 60 bits. `maxbits` reports the true bit length of the largest coefficient, so
// the harness can say where RDKit's own double arithmetic stops being exact.
inline void charPolyScaled(const Mol &m, std::vector<double> &out, int &E, int &maxbits) {
  const int n = m.n;
  out.assign(n + 1, 0.0);
  E = 0;
  maxbits = 0;
  if (n == 0) { out[0] = 1.0; return; }

  // ------------------------------------------------------------------------------------------
  // FAST PATH: every value still fits in one int64, which is the case for 99%+ of a drug-like
  // corpus. This is the SAME arithmetic as the multiword path below and produces the same exact
  // integers -- it is not an approximation and it does not weaken determinism. What it avoids is
  // calling a carry-propagating routine once per matrix ELEMENT, which is what made the general
  // path cost 305 us/mol: at W = 1 that call does the work of a single `+=` and defeats
  // vectorisation of the row addition completely.
  //
  // Three things make it quick, in order of what they were worth:
  //   * plain `int64_t` row additions, which the compiler vectorises;
  //   * T = A M written as COPY the first neighbour's row then ADD the rest, instead of zeroing
  //     T and adding all of them -- the zero pass was a whole extra O(n^2) per step;
  //   * a CSR adjacency built once instead of walking the bond list per step.
  //
  // OVERFLOW IS RULED OUT BEFORE IT CAN HAPPEN, not detected after. `mx` is the exact maximum
  // magnitude in M, measured each step; the next step cannot produce anything larger than
  // maxdeg * (mx + |c|), so if that bound reaches 2^62 we abandon the fast path and redo the
  // whole molecule in multiword. Bailing is conservative and rare, and the two paths are checked
  // against each other and against exact Python integers by cpp/verify_ic.py.
  // ------------------------------------------------------------------------------------------
  std::vector<int32_t> astart(n + 1, 0), anbr(2 * (size_t)m.nb);
  for (int b = 0; b < m.nb; b++) { astart[m.bu[b] + 1]++; astart[m.bv[b] + 1]++; }
  for (int i = 0; i < n; i++) astart[i + 1] += astart[i];
  {
    std::vector<int32_t> cur(astart.begin(), astart.end() - 1);
    for (int b = 0; b < m.nb; b++) { anbr[cur[m.bu[b]]++] = m.bv[b]; anbr[cur[m.bv[b]]++] = m.bu[b]; }
  }
  int maxdeg = 1;
  for (int i = 0; i < n; i++) maxdeg = std::max(maxdeg, astart[i + 1] - astart[i]);

  {
    const int64_t GUARD = (int64_t)1 << 62;
    std::vector<int64_t> M((size_t)n * n, 0), T((size_t)n * n, 0), C(n + 1, 0);
    for (int b = 0; b < m.nb; b++) {
      M[(size_t)m.bu[b] * n + m.bv[b]] = 1;
      M[(size_t)m.bv[b] * n + m.bu[b]] = 1;
    }
    C[0] = 1;
    int64_t cprev = 0;
    for (int i = 0; i < n; i++) cprev -= M[(size_t)i * n + i];
    C[1] = cprev;
    int64_t mx = m.nb ? 1 : 0;
    bool ok = true;
    for (int k = 2; k <= n && ok; k++) {
      const int64_t room = mx + (cprev < 0 ? -cprev : cprev);
      if (room >= GUARD / maxdeg) { ok = false; break; }        // cannot overflow if we proceed
      for (int i = 0; i < n; i++) M[(size_t)i * n + i] += cprev;
      int64_t nmx = 0;
      for (int u = 0; u < n; u++) {
        int64_t *tu = &T[(size_t)u * n];
        const int e0 = astart[u], e1 = astart[u + 1];
        // The running maximum is folded into the LAST neighbour's pass rather than taken in a
        // pass of its own: a separate |max| sweep is another full O(n^2) per step, which on a
        // matrix that no longer fits in L1 costs about as much as the additions it is guarding.
        if (e0 == e1) {
          for (int j = 0; j < n; j++) tu[j] = 0;
        } else if (e1 - e0 == 1) {
          const int64_t *m0 = &M[(size_t)anbr[e0] * n];
          for (int j = 0; j < n; j++) {
            const int64_t x = m0[j];
            tu[j] = x;
            const int64_t a = x < 0 ? -x : x;
            if (a > nmx) nmx = a;
          }
        } else {
          const int64_t *m0 = &M[(size_t)anbr[e0] * n];
          for (int j = 0; j < n; j++) tu[j] = m0[j];
          for (int e = e0 + 1; e < e1 - 1; e++) {
            const int64_t *mv = &M[(size_t)anbr[e] * n];
            for (int j = 0; j < n; j++) tu[j] += mv[j];
          }
          const int64_t *ml = &M[(size_t)anbr[e1 - 1] * n];
          for (int j = 0; j < n; j++) {
            const int64_t x = tu[j] + ml[j];
            tu[j] = x;
            const int64_t a = x < 0 ? -x : x;
            if (a > nmx) nmx = a;
          }
        }
      }
      M.swap(T);
      mx = nmx;
      int64_t tr = 0;
      for (int i = 0; i < n; i++) tr += M[(size_t)i * n + i];
      cprev = -tr / (int64_t)k;                                 // exact: k divides tr(A M_k)
      C[k] = cprev;
    }
    if (ok) {
      maxbits = 0;
      for (int k = 0; k <= n; k++) {
        const uint64_t a = (uint64_t)(C[k] < 0 ? -C[k] : C[k]);
        int b = 0;
        for (uint64_t t = a; t; t >>= 1) b++;
        maxbits = std::max(maxbits, b);
      }
      E = std::max(0, maxbits - 60);
      for (int k = 0; k <= n; k++)
        out[k] = std::ldexp((double)(C[k] < 0 ? -C[k] : C[k]), -E);
      return;
    }
  }

  // The word count is still chosen ADAPTIVELY -- start narrow, widen on overflow -- but the
  // widths are now an explicit ladder of template instantiations rather than a runtime `W *= 2`,
  // so each one compiles to its own unrolled carry chain. The ladder is the same 2, 4, 8, 16, 32
  // the loop walked, and the same 2048-bit ceiling terminates it.
  {
    const int32_t *as = astart.data(), *an = anbr.data();
    std::vector<uint64_t> C;
    int W = 0;
    if      (faddeevMultiword<2>(m, as, an, C))  W = 2;
    else if (faddeevMultiword<4>(m, as, an, C))  W = 4;
    else if (faddeevMultiword<8>(m, as, an, C))  W = 8;
    else if (faddeevMultiword<16>(m, as, an, C)) W = 16;
    else if (faddeevMultiword<32>(m, as, an, C)) W = 32;
    else throw std::runtime_error("infoic: characteristic polynomial needs more than 2048 bits");

    maxbits = 0;
    for (int k = 0; k <= n; k++) {
      int b = 0;
      big::toScaledDouble(&C[(size_t)k * (size_t)W], W, 0, &b);
      maxbits = std::max(maxbits, b);
    }
    E = std::max(0, maxbits - 60);
    for (int k = 0; k <= n; k++)
      out[k] = big::toScaledDouble(&C[(size_t)k * (size_t)W], W, E, nullptr);
  }
}

// rdkit/ML/InfoTheory's InfoEntropy, in its own order: one sequential sum for the total, skip
// zero entries, accumulate -t*ln(t), divide by ln 2 at the end. Reproduced rather than improved
// so that the values agree with RDKit to the last bit wherever RDKit's own input was exact.
inline double infoEntropy(const std::vector<double> &v, double &total) {
  total = 0.0;
  for (double x : v) total += x;
  if (total == 0.0) return 0.0;
  double acc = 0.0;
  for (double x : v) {
    if (x == 0.0) continue;
    const double t = x / total;
    acc += -t * std::log(t);
  }
  return acc / std::log(2.0);
}

// --------------------------------------------------------------------------------------------
// The whole row.
// --------------------------------------------------------------------------------------------
inline void compute(const Mol &m, Row &row, CodeBuilder *cb = nullptr, Profile *prof = nullptr,
                    bool wantIpc = true) {
  auto tick = [](double &t) {
    const double n = CpuClock::nowUs();
    const double us = n - t;
    t = n;
    return us;
  };
  double tp = 0.0;
  if (prof) { tp = CpuClock::nowUs(); prof->mols++; }

  for (int c = 0; c < N_COLS; c++) row.v[c] = std::numeric_limits<double>::quiet_NaN();
  row.ipcOverflow = false;

  // The fallback builder is a FUNCTION-LOCAL THREAD_LOCAL, not a stack object. It is pure
  // scratch -- reset() re-sizes and re-zeros every array from the graph in hand and nothing is
  // carried across molecules -- so this changes no value; what it removes is the ~14 vector
  // allocations a caller that does not hoist its own CodeBuilder was paying per molecule.
  // bindings.cpp is such a caller today (`infoic::compute(W.im, W.irow)`), and this way it gets
  // the hoist without an edit to a file another agent owns.
  static thread_local CodeBuilder local;
  CodeBuilder &bld = cb ? *cb : local;

  HGraph &g = bld.hg;
  g.build(m);
  const int A = g.N;
  const double B = kekuleBondOrderSum(m, bld.used);
  const double log2A = std::log2((double)A);
  const double log2B = std::log2(B);

  bld.reset(g);

  std::vector<int32_t> &ordIdx = bld.ordIdx;
  std::vector<double> &cnt = bld.cnt, &wmass = bld.wmass, &wz = bld.wz, &scratch = bld.scratch;
  if (prof) { prof->build += tick(tp); prof->atoms += A; }

  bld.buildAll();                    // ONE traversal, all six orders
  if (prof) {
    prof->dfs += tick(tp);
    for (int o = 1; o <= MAX_ORDER; o++) prof->paths[o] += bld.pathsAt(o);
  }

  for (int order = 0; order <= MAX_ORDER; order++) {
    cnt.clear(); wmass.clear(); wz.clear();
    if (order == 0) {
      // mordred short-circuits: the code IS the atomic number. Nothing here can be order
      // dependent, which is exactly why order 0 is the control. Sorting one-byte codes put the
      // classes in ASCENDING Z; a 256-bucket count reproduces that order exactly and is the one
      // place where the class ORDER is worth thinking about, since numpy's pairwise summation
      // is not associative and the order-0 control is bit-exact against mordred.
      int zc[256];
      std::memset(zc, 0, sizeof zc);
      for (int i = 0; i < A; i++) zc[g.z[i]]++;
      if (prof) prof->codes[0] += tick(tp);
      for (int z = 0; z < 256; z++) {
        if (!zc[z]) continue;
        const double c = (double)zc[z];
        cnt.push_back(c);
        wmass.push_back(ic_tbl::ATOMIC_WEIGHT[z]);
        wz.push_back(c * (double)z);
      }
    } else {
      if (prof) prof->codes[order] += tick(tp);
      for (int i = 0; i < A; i++) ordIdx[i] = i;
      const PKey *ar = bld.arena[order].data();
      const int32_t *of = bld.off[order].data();
      // Identical to the memcmp over the old byte blobs: the records are fixed width and PKey's
      // numeric order IS their memcmp order, so lexicographic-on-records then shorter-first is
      // the same total order it was, and therefore the same class ORDER.
      std::sort(ordIdx.begin(), ordIdx.end(), [ar, of](int32_t x, int32_t y) {
        const int32_t lx = of[x + 1] - of[x], ly = of[y + 1] - of[y];
        const int32_t n = lx < ly ? lx : ly;
        const PKey *px = ar + of[x], *py = ar + of[y];
        for (int32_t q = 0; q < n; q++) {
          if (px[q].hi != py[q].hi) return px[q].hi < py[q].hi;
          if (px[q].lo != py[q].lo) return px[q].lo < py[q].lo;
        }
        return lx < ly;
      });

      for (int a = 0; a < A;) {
        int b = a + 1;
        const int32_t la = of[ordIdx[a] + 1] - of[ordIdx[a]];
        const PKey *pa = ar + of[ordIdx[a]];
        while (b < A) {
          const int32_t lb = of[ordIdx[b] + 1] - of[ordIdx[b]];
          if (lb != la) break;
          const PKey *pb = ar + of[ordIdx[b]];
          bool same = true;
          for (int32_t q = 0; q < la; q++)
            if (!(pa[q] == pb[q])) { same = false; break; }
          if (!same) break;
          b++;
        }
        const double c = (double)(b - a);
        // Every atom in a class shares the code's root field, which carries the root's atomic
        // number, so the class HAS an element -- that is what makes repair R3 well defined.
        const int z = g.z[ordIdx[a]];
        cnt.push_back(c);
        wmass.push_back(ic_tbl::ATOMIC_WEIGHT[z]);
        wz.push_back(c * (double)z);
        a = b;
      }
    }
    const size_t k = cnt.size();
    if (prof) { prof->group[order] += tick(tp); prof->classes[order] += (long long)k; }
    scratch.assign(k, 0.0);
    const double ic = shannonEntropy(cnt.data(), nullptr, k, scratch.data());
    row.v[F_IC * N_ORDERS + order] = ic;
    row.v[F_TIC * N_ORDERS + order] = (double)A * ic;
    // SIC and BIC are NaN when their denominator is zero, and this is a QUIRK REPRODUCED, not a
    // choice. mordred wraps both divisions in `rethrow_zerodiv`, which is
    // `np.errstate(divide="raise", invalid="raise")` -- so where IEEE would hand back an
    // infinity, numpy RAISES and mordred records a missing value instead. Plain `ic / log2B`
    // gives +inf and disagrees; `Cl[Se]` is the one molecule in cpp/hard.smi that discriminates
    // them (A = 2, B = 1, so log2 B = 0, IC0 = 1: ours was +inf, mordred's is missing). The
    // 0/0 cases already agree because IEEE also calls those NaN -- it is only x/0 that differs.
    row.v[F_SIC * N_ORDERS + order] = log2A == 0.0 || !std::isfinite(log2A)
                                          ? std::numeric_limits<double>::quiet_NaN()
                                          : ic / log2A;
    row.v[F_BIC * N_ORDERS + order] = log2B == 0.0 || !std::isfinite(log2B)
                                          ? std::numeric_limits<double>::quiet_NaN()
                                          : ic / log2B;
    row.v[F_CIC * N_ORDERS + order] = log2A - ic;
    row.v[F_MIC * N_ORDERS + order] = shannonEntropy(cnt.data(), wmass.data(), k, scratch.data());
    row.v[F_ZMIC * N_ORDERS + order] = shannonEntropy(cnt.data(), wz.data(), k, scratch.data());
    if (prof) prof->entropy[order] += tick(tp);
  }

  // ---- Ipc: exact integer coefficients, then RDKit's own entropy formula ----------------
  // STILL THE MAJORITY OF THIS DESCRIPTOR even after the multiword rewrite, and by a wide margin
  // the most concentrated cost in it: 2.1% of cpp/hard.smi takes the multiword path and carries
  // ~80% of the block. `wantIpc == false` leaves Ipc/AvgIpc/Log2Ipc as the NaN they were
  // initialised to and touches nothing else, so the 42 IC columns are bit-identical either way.
  // The shipped wiring emits `AvgIpc` and passes `wantIpc` through as true; see `computeIC()`
  // for when the other entry point is the right one.
  if (!wantIpc) {
    if (prof) prof->ipc += tick(tp);
    return;
  }
  std::vector<double> &cpoly = bld.cpoly;
  int E = 0, maxbits = 0;
  charPolyScaled(m, cpoly, E, maxbits);
  double total = 0.0;
  const double h = infoEntropy(cpoly, total);
  row.v[C_AVGIPC] = h;
  row.ipcMaxCoeffBits = maxbits;
  if (h == 0.0 || total == 0.0) {
    row.v[C_IPC] = 0.0;
    row.v[C_LOG2IPC] = -std::numeric_limits<double>::infinity();
  } else {
    row.v[C_LOG2IPC] = std::log2(total) + (double)E + std::log2(h);
    const double ipc = std::ldexp(total, E) * h;          // E == 0 on all of cpp/hard.smi
    if (std::isfinite(ipc)) {
      row.v[C_IPC] = ipc;
    } else {
      row.v[C_IPC] = std::numeric_limits<double>::max();   // SATURATES, and says so:
      row.ipcOverflow = true;                              // never a silent inf
    }
  }
  if (prof) prof->ipc += tick(tp);
}

// THE 42 IC COLUMNS ONLY -- the Ipc block skipped entirely.
//
// NOT WHAT THE EXTENSION CALLS TODAY. bindings.cpp calls `compute()` and emits the 42 plus
// `AvgIpc`, which is one of the 865; see item 5 of the WIRING note at the top of this file for
// why that flipped back and what it costs.
//
// This entry point remains because the choice is a real one and may be taken again. It skips an
// exact-integer Le Verrier-Faddeev-Frame recurrence that is O(n^3) in the HEAVY-ATOM count and
// is still the majority of this descriptor's CPU. The 42 columns it returns are bit-identical to
// the same 42 out of `compute()`, because nothing above the Ipc block reads anything the Ipc
// block writes -- so switching between the two is a cost decision and never a value decision.
inline void computeIC(const Mol &m, Row &row, CodeBuilder *cb = nullptr, Profile *prof = nullptr) {
  compute(m, row, cb, prof, false);
}

// --------------------------------------------------------------------------------------------
// selfCheck -- the guard against silent drift, run once at module load.
//
// It checks the two things that are cheap to check and expensive to discover on a corpus: the
// generated tables are the ones this file was written against, and the layered code builder
// gives the SAME code for a molecule under an atom permutation that mordred's tree does not.
// The permutation case is the worked example from the header comment, in graph form, so a
// regression that reintroduces order dependence fails at load.
// --------------------------------------------------------------------------------------------
// ON=Cc1ccccn1 -- pyridine-2-carbaldehyde oxime, 9 heavy atoms, as parsed by RDKit 2025.09.2.
// This is the header comment's worked example, in graph form.
inline void buildWorkedExample(Mol &m) {
  const uint8_t Z[9] = {8, 7, 6, 6, 6, 6, 6, 6, 7};
  const uint8_t NH[9] = {1, 0, 1, 0, 1, 1, 1, 1, 0};
  const uint8_t AR[9] = {0, 0, 0, 1, 1, 1, 1, 1, 1};
  const int E[9][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 3}};
  const int CODE[9] = {1, 2, 1, 8, 8, 8, 8, 8, 8};      // 1 SINGLE, 2 DOUBLE, 8 AROMATIC flag
  const double ORD[9] = {1.0, 2.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
  m.alloc(9, 9);
  for (int i = 0; i < 9; i++) { m.z[i] = Z[i]; m.nh[i] = NH[i]; m.arom[i] = AR[i]; }
  for (int e = 0; e < 9; e++) {
    m.bu[e] = E[e][0]; m.bv[e] = E[e][1]; m.bcode[e] = (uint8_t)CODE[e]; m.bord[e] = ORD[e];
  }
}

inline void selfCheck() {
  if (ic_tbl::ATOMIC_WEIGHT[6] <= 12.0 || ic_tbl::ATOMIC_WEIGHT[6] >= 12.02)
    throw std::runtime_error("infoic: ic_tables.h atomic weights look wrong");
  if (ic_tbl::DEFAULT_VALENCE[6] != 4 || ic_tbl::DEFAULT_VALENCE[7] != 3 ||
      ic_tbl::DEFAULT_VALENCE[8] != 2)
    throw std::runtime_error("infoic: ic_tables.h default valences look wrong");
  if (ic_tbl::N_OUTER_ELECS[7] != 5 || ic_tbl::N_OUTER_ELECS[6] != 4)
    throw std::runtime_error("infoic: ic_tables.h outer-electron counts look wrong");

  // numpy pairwise summation. Three shapes, one per branch of numpy's function, each written
  // out longhand here so that a "simplification" of pairwiseSum that changes the ORDER of the
  // additions fails at load rather than as a 1-ulp mystery in the order-0 control.
  {
    double a[300];
    for (int i = 0; i < 300; i++) a[i] = 1.0 / (double)(i + 3);
    double s7 = 0.0;
    for (int i = 0; i < 7; i++) s7 += a[i];          // n < 8: plain sequential
    if (pairwiseSum(a, 7) != s7) throw std::runtime_error("infoic: pairwiseSum n<8");
    {                                                // 8 <= n <= 128: 8 accumulators, tail last
      double r[8];
      for (int j = 0; j < 8; j++) r[j] = a[j];
      double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      res += a[8];
      if (pairwiseSum(a, 9) != res) throw std::runtime_error("infoic: pairwiseSum 8<=n<=128");
    }
    {                                                // n > 128: split at n/2 rounded down to 8
      const size_t n = 300, n2 = (n / 2) - ((n / 2) % 8);
      if (pairwiseSum(a, n) != pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2))
        throw std::runtime_error("infoic: pairwiseSum n>128");
    }
  }

  // PKey IS AN INJECTION AND ITS NUMERIC ORDER IS THE BYTE ORDER IT REPLACED. That is the whole
  // safety argument for the 128-bit packing, so it is checked rather than asserted: build the
  // ORIGINAL 24-byte key and the PKey for a spread of synthetic paths -- every length from 0 to
  // MAX_ORDER edges, both ends of the symbol range, both ends of the Z and degree ranges -- and
  // require that memcmp and (hi, lo) agree on every ORDERED PAIR, equality included. A hash
  // would fail this at once; a packing with an overlapping or mis-shifted field fails it too.
  {
    struct Path { int len; uint8_t sym[MAX_ORDER], z[MAX_ORDER], dg[MAX_ORDER]; uint8_t rz, rd; };
    std::vector<Path> ps;
    const uint8_t ZS[4] = {1, 6, 118, 255}, DS[4] = {0, 1, 4, 255}, SS[3] = {SYM_OTHER,
                                                                            SYM_SINGLE,
                                                                            SYM_AROMATIC};
    for (int len = 0; len <= MAX_ORDER; len++)
      for (int a = 0; a < 4; a++)
        for (int s = 0; s < 3; s++) {
          Path p{};
          p.len = len; p.rz = ZS[a]; p.rd = DS[a];
          for (int q = 0; q < len; q++) {
            p.sym[q] = SS[(s + q) % 3];
            p.z[q] = ZS[(a + q) % 4];
            p.dg[q] = DS[(a + q + 1) % 4];
          }
          ps.push_back(p);
        }
    const size_t NP = ps.size();
    std::vector<std::array<uint8_t, 24> > bytes(NP);
    std::vector<PKey> pk(NP);
    for (size_t i = 0; i < NP; i++) {
      const Path &p = ps[i];
      std::array<uint8_t, 24> b{};
      b.fill(0);
      b[0] = p.rz; b[1] = p.rd;
      uint64_t hi = ((uint64_t)(((uint32_t)p.rz << 8) | p.rd)) << 48, lo = 0;
      int pos = 2;
      for (int q = 0; q < p.len; q++) {
        b[pos] = p.sym[q]; b[pos + 1] = p.z[q]; b[pos + 2] = p.dg[q];
        pos += 3;
        pkSetField(hi, lo, q + 1,
                   1u + (((uint32_t)p.sym[q] << 16) | ((uint32_t)p.z[q] << 8) | p.dg[q]));
      }
      b[pos] = 0xFF;                              // the terminator, exactly where walk() put it
      if (p.len < MAX_ORDER) pkSetField(hi, lo, p.len + 1, PK_TERM);
      bytes[i] = b;
      pk[i] = PKey{hi, lo};
    }
    for (size_t i = 0; i < NP; i++)
      for (size_t j = 0; j < NP; j++) {
        const int c = std::memcmp(bytes[i].data(), bytes[j].data(), 24);
        const bool blt = c < 0, beq = c == 0;
        const bool plt = pk[i] < pk[j], peq = pk[i] == pk[j];
        if (blt != plt || beq != peq)
          throw std::runtime_error("infoic: PKey packing is not order-preserving -- the 128-bit "
                                   "code no longer reproduces the 24-byte key it replaced");
      }
  }

  // The worked example, as a graph, so the check needs no RDKit and cannot go stale against a
  // comment. ON=Cc1ccccn1 under the identity and under every one of its 36 transpositions:
  // mordred moves on the single swap (0 5), this must not move on any of them.
  //
  // BOTH AXES ARE PERTURBED, not just the atoms. A caller's bond list arrives in whatever order
  // it arrives in, and `Chem.RenumberAtoms` -- the obvious way to write this test in Python --
  // leaves that order ALONE, which is how an order dependence in bond-list-reading code hides
  // from an atom-only screen. Each transposition here also ROTATES the bond list, so a
  // regression that reintroduces a dependence on either axis fails at module load.
  Mol base;
  buildWorkedExample(base);
  double ref[N_COLS];
  int perm[9];
  for (int a = -1; a < 9; a++)
    for (int b = a + 1; b < 9; b++) {
      if (a < 0 && b > 0) break;                    // a == -1 is the identity, done once
      for (int i = 0; i < 9; i++) perm[i] = i;
      if (a >= 0) { perm[a] = b; perm[b] = a; }
      Mol m;
      m.alloc(base.n, base.nb);
      for (int i = 0; i < base.n; i++) {
        m.z[perm[i]] = base.z[i]; m.nh[perm[i]] = base.nh[i]; m.arom[perm[i]] = base.arom[i];
        m.chg[perm[i]] = base.chg[i];
      }
      const int rot = a < 0 ? 0 : (a + 1);          // a different bond order for each case
      for (int q = 0; q < base.nb; q++) {
        const int e = (q + rot) % base.nb;
        m.bu[q] = perm[base.bu[e]]; m.bv[q] = perm[base.bv[e]];
        m.bcode[q] = base.bcode[e]; m.bord[q] = base.bord[e];
      }
      Row r;
      compute(m, r);
      if (a < 0) {
        std::memcpy(ref, r.v, sizeof ref);
        // The value itself, so a change of DEFINITION is caught here and not only a change of
        // determinism. mordred's two answers are 2.682588730501833 and 2.8159220638351665.
        if (std::fabs(r.v[F_IC * N_ORDERS + 1] - 2.8159220638351665) > 1e-12)
          throw std::runtime_error("infoic: selfCheck IC1 for ON=Cc1ccccn1 moved");
      } else {
        for (int c = 0; c < N_IC; c++)
          if (std::memcmp(&ref[c], &r.v[c], sizeof(double)) != 0)
            throw std::runtime_error(std::string("infoic: order dependence reintroduced in ") +
                                     columnNames()[c]);
      }
    }
}

}  // namespace infoic

#endif  // HUME_INFOCONTENT_H
