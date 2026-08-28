# The port: all 865 columns in C++, no Python in the compute path

**Goal.** Every one of the 865 deduplicated Mordred ∪ RDKit columns computed in C++, and the
end-to-end SMILES → ECFP + descriptors time measured per step with a standard deviation, against
an unoptimised RDKit + Mordred baseline.

Regenerate this census with the snippet at the bottom. Do not hand-edit the counts.

## Where it stands

**TWO NUMBERS, AND THEY ARE NOT THE SAME NUMBER.** Conflating them overstates the position by
2.7×, and this file did exactly that for a while:

| | columns |
|---|---|
| **producing a value from the package** (`hume.featurize_all`) | **859** |
| named by the package but NaN (`PENDING_COLUMNS`) | 2 (`qed`, `SPS`) |
| verified C++ exists, but NOT wired into the extension | 0 |
| not named at all | 3 (`NumAtomStereoCenters`, `NumUnspecifiedAtomStereoCenters`, `AvgIpc`) |

"Verified C++ exists" is a claim about `cpp/*.cpp`; "callable" is a claim about the wheel; and
"produces a value" is a third claim, narrower than "has a name". Only the last one is what a user
gets, and only the last belongs in a speed comparison. Both live numbers come from
`bench_e2e._survivors_covered` — on `ALL_COLUMNS` for the named count, on
`ALL_COLUMNS - PENDING_COLUMNS` for this one.

Note also the dedupe set has **864 unique names**, not 865: one name is defined by both RDKit and
Mordred and survives under both sources.

**A warning about the number 182.** `hume.featurize_blocks` returns 182 columns and they are
`ALL EXACT`, but they are **mostly HUME-specific descriptors** — `SATS*`, `RATSC*`, `RW*`,
`sysbin*`, `conj_*`, `pa*_max`, `C3`–`C8` — and only about 22 of them are members of the 865.
Reading "182 verified" as "182 of the 865 done" overstates the position by roughly eightfold.
The bulk of the real coverage is elsewhere: `src/hume_core/autocorr.h` computes **419**
Autocorrelation columns and `src/hume_core/estate_typer.h` computes **50** E-state columns.

## Ported and verified

| family | n | where | evidence |
|---|---|---|---|
| Autocorrelation | 419 | `src/hume_core/autocorr.h`, `cpp/ac_weights.h`, `cpp/ac_tables.h`, `cpp/ac.cpp` | ATS/AATS/ATSC/AATSC/MATS/GATS × **10 weights** = 540 emitted, covering all 419. The ten weight vectors are computed in C++, not handed in — that removed 473.9 µs/mol, the single largest item in the pipeline. `values_ac.txt` md5 `1fdb9ca4d92ce808cba2a3a466677fea`, 98,905 × 540. Two things proven, and they are different claims: (a) the 486 nine-weight columns are byte-for-byte unchanged by the `Z` addition (projection md5, whole corpus, see the handoff); (b) **through the shipped wiring, 540/540 columns agree with mordred 1.2.0 in-process on 2,000 molecules — 486/486 pre-existing, 54/54 new `Z`.** NOTE THIS FAMILY IS NOT BITWISE, unlike every other row in this table: the check is rel 1e-8 with an absolute floor at 1e-8 of each column's own scale. That is not slack, it is necessary — `ATSC`/`AATSC` centre by subtracting a mean, so agreement would require exact cancellation in two different summation orders, which IEEE arithmetic does not provide. Max deviation on the new weight is `Z` 3.648e-07, mid-range among the nine already verified (`d` 6.412e-07, `se` 3.630e-05, `are` 4.476e-05). The 98,905-molecule artifact grade is a third claim and was **started and abandoned** — I killed it at 3h25m after it exceeded its predicted cost while blocking the end-to-end measurement. It is claimed nowhere, and `cpp/verify_ac.py` now checkpoints its reference so a re-run resumes rather than restarts. |
| EState | 50 | `src/hume_core/estate_typer.h`, `cpp/estate_tables.h` | 2,868,290 / 2,868,290 atoms exact on `cpp/hard.smi`; 100,000/100,000 column values vs mordred 1.2.0. 0.834 µs/mol vs 636. |
| VSA binning | 59 | `src/hume_core/vsa_bins.h`, `cpp/vsa_tables.h` | **66/66 columns bit-exact vs RDKit** over 100,000 molecules, **5/5 vs mordred**, and all four per-atom vectors exact on 2,868,290 atoms. Labute ASA was the real work. |
| RingCount + TopologicalCharge + PathCount | 81 | `src/hume_core/{ringcount,topocharge,pathcount}.h` | RingCount 49/49 and PathCount 11/11 bit-exact on 100,000. TopologicalCharge 12/21 bit-exact, the other 9 within 6.661e-16 relative — mordred disagrees with *itself* there on 21–70% of the corpus. **20.2 µs/mol against mordred's 11,602 (~575×).** |
| rdkit_core fragments | 76 | `src/hume_core/frag_matcher.h`, `cpp/frag_program.h` | 74 SMARTS pattern counts + `NHOHCount` + `HeavyAtomCount`. **76/76 bit-exact vs RDKit's own `Descriptors` through the shipped wiring** on 5,000 molecules, every column exercised. Needed the tenth `atom_i` column, `tval`. 119.5 ± 7.80 µs/mol. |
| InformationContent | 33 | `src/hume_core/infocontent.h`, `cpp/ic_tables.h` | Not exact-vs-mordred — mordred is ill-posed here. **42 columns bit-identical under renumbering**, order-0 control passes. `Ipc` has an open bug; see the header. |

Plus, inside the 182 blocks: `BCUT2D_*` (8), `Kappa1-3` + `HallKierAlpha` (4), RDKit `Chi*` (9),
`BalabanJ` (1), and the four `*EStateIndex` reductions.

## The two things standing between here and the goal

**1. DONE — Autocorrelation is in the extension, all 419.** The computation moved into
`src/hume_core/autocorr.h`, `cpp/ac.cpp` includes it, and `_extract.py` serialises the
**hydrogen-added** molecule alongside the heavy-atom one so the charges are
`_GasteigerCharge + _GasteigerHCharge` computed on that graph. The tenth weight `Z` closed the
last 52 columns; see the 2026-08-28 handoff at the bottom.

**2. ~~`infocontent` is now the pipeline~~ — FIXED. 280.8 → 62.4 µs/mol.** Two independent
causes, and the larger one was invisible until the profiler was fixed (it had been wall-clock,
reporting the same phase at 74 µs and 601 µs on consecutive runs):
  * **68% of the cost was the `Ipc` block, whose three columns are not wired.** `bindings.cpp`
    called `compute()` for all 45 and copied only 42, so every molecule paid an O(n³)
    exact-integer Faddeev recurrence and the result went on the floor. One line: `computeIC()`.
  * The equivalence codes are now a **128-bit packed injection** (`PKey`), not a hash — the
    packed value's numeric order IS the byte key's memcmp order, so class ordering is preserved
    bit-for-bit and there are no collisions to bound. 2.20× on top of the gating. One depth-5
    DFS now serves all six orders: 4,480 → 2,007 tree nodes per molecule.
It still has the open `Ipc` bug, which is untouched and out of scope of that work.

## In flight

| family | n | agent |
|---|---|---|
| `rdkit_core` | 99 | 76 wired (`frag_matcher.h`); the rest is stereo perception + FpDensity |

## Still to port

Grouped by the machinery they share, which is how they are being scheduled — not by size.

**A. VSA binning, 59.** `rdkit_EState` 23, `rdkit_Crippen` 20, `rdkit_Gasteiger` 13, `MoeType` 3,
`rdkit_TPSA` 1, `TopoPSA` 1, `SLogP` 1. One mechanism: a per-atom contribution vector, binned by
fixed edges, summed. The contributions are already computed natively (Crippen from
`crippen_typer.h`, Gasteiger from the boundary, E-state from `estate_from`). What is missing is
Labute ASA and the bin machinery. `MoeType` resolves via `getattr` to the *same* code path and the
*same* edges as the `rdkit_*` VSA columns, so it is not a separate implementation.

**B. `rdkit_core`, 99 — 76 of them are DONE.** Fragment counts (`fr_*`), H-bond
donors/acceptors, ring and heteroatom counts, all wired via `src/hume_core/frag_matcher.h`; see
"2. DONE" in the handoff. What is left is `NumAtomStereoCenters` +
`NumUnspecifiedAtomStereoCenters` (RDKit's `FindPotentialStereo`, a real subsystem) and
`FpDensityMorgan1/2/3`; the other ~18 are computed elsewhere or trivial from the boundary.

**C. Mordred Chi + walks, 55.** `Chi` 40 (`AXp-*`/`Xp-*` path, `Xc-*` cluster, `Xpc-*`
path-cluster, `Xch-*` chain), `WalkCount` 6, `Constitutional` 4, `WienerIndex` 2,
`TopologicalIndex` 2, `ABCIndex` 1. **Mordred's Chi is not RDKit's Chi** — the C++ already has
RDKit-style `chi0n`–`chi7v`, which does not cover the cluster / path-cluster / chain variants.
Those need `FindAllSubgraphsOfLengthN`.

**D. `InformationContent` 33 + `rdkit_Ipc` 1.** The hard one. Order-dependent traversal, and Ipc is
the information content of the characteristic polynomial's coefficients — numerically delicate at
large *n*.

**E. Small constitutional, 43.** `CarbonTypes` 9, `AtomCount` 8, `BondCount` 6,
`KappaShapeIndex` 3, `MolecularDistanceEdge` 3, `CPSA` 2, `Lipinski` 2, `AcidBase` 2,
`rdkit_composite` 2 (`qed`, `SPS`, `BertzCT`), `VdwVolumeABC` 1, `RotatableBond` 1,
`Polarizability` 1, `LogS` 1, `Framework` 1, `FragmentComplexity` 1.

## House rules for every port

1. **The specification is the source code**, not the documentation. Reproduce upstream bugs and
   say in a comment that you did. Two have already been found and kept: five E-state SMARTS rows
   whose missing semicolon voids an aromaticity constraint, and `[SeD2H0]` decoding as an
   element-number query with no aromaticity constraint at all.

   **The exception, and it is not a small one: reproduce a QUIRK, diverge from an ILL-POSED
   DEFINITION.** The two are distinguished by a single test — *is the upstream descriptor a
   function of the molecule?* A quirk gives the same wrong answer every time, so we match it
   bit-for-bit and comment why. An ill-posed descriptor gives different answers for the same
   molecule depending on atom numbering or on which Kekulé structure the perceiver happened to
   pick. There is nothing there to be exact against, and "we reproduce Mordred" would be a claim
   about a coin flip.

   **How to tell, mechanically: perturb the input ordering and recompute.** Any column that moves
   is ill-posed. Do this for every family you port, before you start optimising anything — it is
   cheap and it changes what the target even is.

   **THE SCREEN MUST SHUFFLE BONDS, NOT ONLY ATOMS. An atom-only screen is too weak and this
   repo shipped it for a while.** `Chem.RenumberAtoms` permutes atoms and **leaves the bond list
   order alone**, and RDKit's ring perception reads the bond list — so an atom-only screen
   under-samples the very axis the answer depends on. Demonstrated:
   `O=C1c2cc(ccc2-n2nccn2)CCCCc2ccc3cc(ccc3c2)N2CCCN1CC2` is stable across **201** atom
   renumberings and yields two different ring sets the moment bond order is shuffled too. Of the
   32 molecules the RingCount repair changes, **every one** gives 2–4 distinct RDKit answers
   under atom+bond shuffling and appears perfectly stable under atom shuffling alone.

   A canonical-SMILES round trip is a **control that should show zero**, not a second probe: it
   reproduces the canonical numbering, so it is not a perturbation. (It has its own use — it is
   how RDKit re-perception shows up, and it moves aromaticity on 19 corpus molecules.)

   Any determinism claim made against the atom-only screen is provisional and must be re-run.

   When a column is ill-posed: pick the well-posed definition that the upstream code was evidently
   reaching for, implement THAT, make it deterministic, and document the divergence loudly with
   the specific molecules that discriminate the two. Report it to me rather than deciding alone.
   Known cases, with the resolution already taken:

   * **`InformationContent`** — Mordred kekulizes before building its atom-equivalence codes, so
     the equivalence classes depend on which Kekulé structure is chosen; and its BFS tree mutates
     a visited set while iterating over it, so they depend on atom numbering too. **~20% of
     molecules had order-dependent IC.** Resolution: an aromatic bond keeps its own bond-type
     symbol rather than being kekulized away, and the tree is layered by graph distance. Orders
     1–5 therefore **differ from Mordred by design**; order 0 is unchanged.
   * **`ExtendedTopochemicalAtom`** — the π contribution was gated on the *kekulized* bond order,
     so an aromatic bond that happened to come out "single" contributed nothing. A pyrrole
     nitrogen scored 0 while a pyridine nitrogen scored 2, and which one you got could flip with
     atom numbering. Resolution: every aromatic atom contributes one aromatic π unit. Mordred's
     own source has an `if bond.GetIsAromatic(): y = 2.0` branch that its `kekulize = True`
     setting largely defeats — the intent was there, the wiring wasn't. (This family contributes
     no surviving columns under the current dedupe, so it is recorded for the principle.)

   **Consequence for the paper, stated here so nobody discovers it in review:** "exact against
   RDKit and Mordred" is true for the well-posed columns and must carry a named exception for the
   ill-posed ones. The right claim for those is *deterministic and documented*, which is strictly
   stronger than what Mordred offers, and it should be presented that way rather than buried.

   * **`RingCount` (25 of 100,000 molecules)** — `nARing`, `nG12Ring`, `n6Ring`, `n7Ring`,
     `n6ARing`. The SSSR *basis* is stable; what flips is whether `symmetrizeSSSR` finds one
     symmetry-equivalent extra ring of a size already present. `C1=CC2C3C(C=C1)C23` gives ring
     sizes (3,3,7) on 33 of 60 numberings and (3,3,7,7) on the other 27. Brute force over every
     simple cycle confirms the larger answer is the **relevant-cycle set** — the object
     `symmetrizeSSSR` is reaching for and reaches only sometimes.

     **My first prescribed repair for this was WRONG, and the record should say so.** I specified
     "canonical atom ranks, rings compared by (size, sorted canonical-rank vector)". Implemented
     exactly, it left **3 of 100,000 still moving and made those 3 worse than doing nothing** —
     because canonical ranks fix the atom numbering and not the bond order, which is the axis
     that actually decides. `Chem.CanonicalRankAtoms(breakTies=True)` is also not a graph
     invariant on symmetric molecules (it varies by an automorphism on e.g. 1,4-disubstituted
     cyclohexanes).

     **The repair that works:** perceive rings on a **skeleton rebuilt from scratch** — *n*
     carbons in canonical-rank order, bonds added in sorted `(rank_u, rank_v)` order. Ring
     perception reads only the graph, so the skeleton asks exactly the right question and puts
     bond order under canonical control too. 100,000 × 49 columns × 5 numberings: 22 molecules
     move before, **0 after**; it changes RDKit's answer on 32 molecules, all 32 independently
     confirmed unstable.

1b. **Aromaticity perception, two repairs** — relevant to any port that does its own ring or
   aromaticity reasoning rather than inheriting the `arom` flag from the boundary:

   * A ring sulfur carrying an exocyclic double bond is a sulfoxide, which is pyramidal and
     therefore cannot be aromatic.
   * The rule "a bond in an all-aromatic ring is aromatic" must run **after** perception, not
     during it: a fused system can contain ring bonds that belong to no tested subset.
2. **A cycle is not its vertex set.** Fixed twice in this repo. In K4, three distinct 4-cycles
   share one vertex set. Canonicalise by `path[1] < path.back()` for `depth >= 3`.
3. **The oracle is pinned**, and asking for it wrongly fails *silently* — see `constraints.txt`.
   Print the resolved versions from the process that produced the numbers. A verify log without
   its RDKit version on it is not evidence.
4. **Never `uv pip install` into `.venv`.** Always `uv pip install -e . -c constraints.txt`.
5. **Exactness on 100,000 molecules** (`cpp/hard.smi`), reported per column. A tolerance is
   allowed only with the max observed deviation and a floating-point reason.
6. **Drift guards hash the spec, not the file.** `sha256(AtomTypes.py)` differs across RDKit
   versions and the whole diff is a deleted `# $Id$` line; a file hash would cry wolf on a
   copyright edit and mask a real change.
7. **No column may be dropped.** All 865 are wanted.

## Regenerating the census

```bash
uv run --isolated --python 3.11 --with "mordred==1.2.0" --with "rdkit==2025.9.2" \
       --with "numpy==1.26.4" python -c "
import json
from mordred import Calculator, descriptors as mdesc
full = Calculator(mdesc, ignore_3D=True)
json.dump({str(x): type(x).__module__.split('.')[-1] for x in full.descriptors}, open('fam.json','w'))"
```

then `blocks.split(fam)` and count by family. Note mordred 1.2.0 needs **Python 3.11**
(`distutils`, removed in 3.12) and **numpy 1.x**.

---

# HANDOFF — 2026-08-27 evening

State at the pause. Everything below is committed; nothing important lives only in a scratch
directory.

## What is true right now

    named by the package    (hume.ALL_COLUMNS)                861 of 864
    PRODUCING A VALUE       (ALL_COLUMNS - PENDING_COLUMNS)   859 of 864
    verified C++ NOT yet wired into the extension               0
    not named at all                                            3

THE LAST FIVE, and none is a porting gap of the ordinary kind:

  qed                              named, NaN. Needs QED's 116 structural-alert SMARTS. The
                                   cheapest correct route is to give frag_matcher.h's Matcher a
                                   bound program table and generate a second program, keeping ONE
                                   subgraph-isomorphism implementation; it needs four new leaf
                                   opcodes (isotope, `~`, `@`, component-level `.`).
  SPS                              named, NaN. Needs the NEW FindPotentialStereo, and it is NOT
                                   in the pickle -- extract_pickles would have to run
                                   FindPotentialStereo + FindPotentialStereoBonds per molecule
                                   and ship two extra arrays.
  NumAtomStereoCenters             unnamed. Needs the LEGACY _ChiralityPossible flag, which IS in
  NumUnspecifiedAtomStereoCenters  the pickle in bits molpickle.h currently skips (0x8 of
                                   pickleExplicitProperties; chiral tag at atom flag bit 2).
                                   ~a field, not a subsystem. One real change: extract_pickles
                                   calls AssignStereochemistry(cleanIt, force) BEFORE ToBinary,
                                   which WIPES the flag on 906 of 2,000 molecules --
                                   flagPossibleStereoCenters=True restores it.
  AvgIpc                           unnamed. Blocked on the open Ipc question, see infocontent.h.

THE LEGACY AND NEW STEREO PERCEPTIONS ARE NOT THE SAME THING. An earlier note in this file said
one boundary addition would unblock all three stereo-flavoured columns. It will not: the legacy
`_ChiralityPossible` atom set and the `FindPotentialStereo` atom set differ on 262 of 4,000
corpus molecules (6.6%). Two additions, two perceptions.

Both numbers come from `bench_e2e._survivors_covered`, run on the two different inputs; they
differ by exactly `PENDING_COLUMNS` = (`qed`, `SPS`), which are named and NaN. Do not quote one
as the other.

Autocorrelation IS now wired, and **all 419** of it: the tenth weight `Z` has been added, so the
52 columns that were held back are in. See "Autocorrelation is complete" below for the evidence
and the new artifact checksum.

The 76 `rdkit_core` fragment columns ARE now wired (+76). See "2. DONE" below.

`hume.featurize_all(smiles) -> (fp, X, ALL_COLUMNS)` works today: SMILES -> ECFP (2048, r=3,
chirality) + 1,244 emitted columns, through ONE pickle parse, one boundary fill.
The 840 is not the 1,244: `bench_e2e._survivors_covered` counts the members of the 865 and prints
it next to the timing, so the two can never be read apart.

## The three things to do next, in priority order

**1. DONE — Autocorrelation is wired, all ten weights.** `src/hume_core/autocorr.h` holds the
computation and `cpp/ac.cpp` now includes it rather than carrying a copy, so there is one copy of
the arithmetic. Proof the header lift changed nothing: `./ac verify mols_h.txt` over 98,905
molecules × 486 columns produced `values_ac.txt` with md5 `7f08884f8700c23fd41e2a5315870a2e`,
**identical before and after**.

  **THAT CHECKSUM IS NOW HISTORICAL.** The `Z` weight took the artifact to 540 columns; the
  current one is **md5 `1fdb9ca4d92ce808cba2a3a466677fea`**, 98,905 × 540. The old md5 is not
  dead evidence, though — it is how the change was proved harmless. Projecting the 540-column
  file back onto its 486 non-`Z` columns (`awk`, drop every tenth field, same `%.12g` text)
  reproduces `7f08884f8700c23fd41e2a5315870a2e` **byte for byte over the whole corpus**. Adding
  the tenth weight moved no cell of the other nine, on 48 million cells, exactly.

  Wiring checked against a *second, independent* H-graph construction in the harness.

  Two findings worth keeping: the H-graph charges are **not** derivable from the heavy-atom
  pickle (5,221 of 42,359 heavy atoms get a different `_GasteigerCharge` from `AddHs(m)` than
  from `m`), so `extract_pickles` serialises a real second molecule rather than putting 419
  columns permanently on a tolerance. And a "lean" AC pickle silently drops the charges:
  `AtomProps|ComputedProps` without `PrivateProps` yields 277 bytes with no `_GasteigerCharge`
  at all — it is private *and* computed.

  **Remaining AC work: none.** `Z` is in — `cpp/ac_weights.h` now emits ten weights, `NW = 10`,
  and `autocorr::N_COLS` is 540. All 419 Autocorrelation members of the 865 are covered.

**2. DONE — the 76 fragment columns are wired.** `atom_i` is now **(n_atoms, 10)**; the tenth
column is `tval`, SMARTS `v`.

  **`tval` did NOT need a new serialised field, and that is the finding worth keeping.** The
  pickle was already carrying it and `molpickle.h` was throwing it away: atom property-flag
  **bit 5 is `getExplicitValence()`** (previously `r.skip(1)`) and **bit 6 is
  `getImplicitValence()`**, and `getTotalValence()` is exactly their sum — 0 of 575,571 atoms
  of `hard.smi` disagree, measured through RDKit's own accessors. So the fast path reads two
  bytes it was already stepping over, and only `extract()` pays a Python call. Both paths are
  compared field-by-field by `cpp/verify_molpickle.py`, whose `FIELDS` now carries
  `atom total valence`: **EXACT on 2,866,100 + 2,868,290 atoms**, both corpora.

  Cost of the wider boundary, A/B in one process on the same 2,000 molecules (the "before" arm
  is the same module source with the two `tval` lines removed): `extract()` 130.3 → 130.7
  µs/mol, i.e. **+0.4 µs/mol**. `extract_pickles()` is unchanged by construction — the bytes
  were already in the blob.

  `border` was NOT free, and the earlier note here was wrong: `esttyper::btypeFromBcode()`
  returns the one-hot **bitmask** (SINGLE 1, DOUBLE 2, TRIPLE **4**, AROMATIC **8**), while
  `fragmatch` compares against RDKit's `BondType` **integer** (TRIPLE **3**, AROMATIC **12**).
  `bindings.cpp`'s `frag_border()` reuses `btypeFromBcode` for the type DECISION — the part that
  knows an order bit beats the aromatic flag — and adds a four-entry renumbering on top. That is
  a table, not a second converter. Anything else maps to 0, which is exact rather than
  approximate here: the only bond-order values in all 1,474 nodes of `cpp/frag_program.h` are
  1, 2, 3 and 12, so a dative bond (17 to RDKit, 0 here) is indistinguishable to every query,
  negation included.

  Verified THROUGH THE WIRING, not through `cpp/frag`: `cpp/verify_wiring.py` now grades the 76
  against RDKit's own `Descriptors` in-process. **76/76 EXACT, bitwise, on 5,000 molecules**, and
  every column is exercised (the thinnest is `fr_azide`, nonzero on 7). Every pre-existing family
  is unchanged in the same run, and `featurize_all[:, :182]` is still bit-identical to
  `featurize_blocks` — through both readers.

  **It is not cheap: 119.5 ± 7.80 µs/mol for 76 columns**, the second-largest family after
  `infocontent`. Nothing has been optimised; `matchCount()` still allocates a vector of match
  vectors per pattern per molecule (74 patterns), which is the obvious first lever.

**3. ~~`infocontent` IS THE PIPELINE~~ — FIXED, 280.8 → 62.4 µs/mol. Was 63% of all compute,
2× the entire 182-column block, ~6× everything else wired combined, for 33 columns of the 865.
Profile per order before optimising. Its `Ipc`/`AvgIpc`/`Log2Ipc` are deliberately NOT wired —
open bug, see the header.

## Open bugs and debts

* **`Ipc` — the direction of this bug REVERSED, and it needs one more check before it is closed.**
  The accuracy side now looks settled and in our favour: over all 100,000, of the 96,244
  molecules whose largest coefficient needs ≤40 bits, 96,221 match RDKit to the last bit; in the
  41–53 bit band only 638 of 1,271 match, and there **RDKit is the inaccurate one** (ours within
  ~1e-15 of exact integer arithmetic, RDKit out by up to 1e-2). Coefficient width is not the
  right predictor — RDKit's Faddeev *iterate matrix* crosses 2^53 well before the final
  coefficients do.

  **But the determinism evidence has a hole.** `cpp/ic_out0..5.txt` are six byte-identical
  full-corpus outputs, which would settle it — except the matching `cpp/ic_in*.txt` dumps are no
  longer on disk, so it cannot be confirmed those runs were fed *different numberings* rather
  than the same input six times. Six identical outputs prove nothing without distinct inputs.
  **Re-run the determinism table (dump the perturbed inputs, keep them, compare) before quoting
  any of it.** `Ipc`/`AvgIpc`/`Log2Ipc` stay unwired until then.
* **FIXED — the `gate()` predicate.** Rewritten O(#rings) from `RingInfo` alone: 25.7 → 10.2
  ± 0.15 µs/mol, of which 4.7 is the `AtomRings()` call `rings_for` needs anyway, so the
  predicate itself went 21.1 → 5.5. Identical on all 100,000 (0 disagreements with the old gate,
  same 21.3% firing rate, hexaprismane still fires) and `gatecheck` still reports
  `gated != unconditional on 0 / 100000`. It was deliberately NOT moved to C++: at 5.5 µs it is
  no longer the lever — `canon_rings` on the gated 20% is 27.7 of the remaining 39.4.
* **`bench_e2e.py`'s `baseline.json` is a stale 100-molecule run.** Regenerate at the same size
  as the hume arm. `report` correctly refuses a headline ratio while the arms disagree on a
  shared step or the machine is contended — do not defeat that, it has already caught a bad
  comparison.
* **The end-to-end table needs a QUIET machine.** Everything measured so far says CONTENDED and
  says so in the JSON.
* Remaining rdkit_core: `NumAtomStereoCenters` + `NumUnspecifiedAtomStereoCenters` (RDKit's
  `FindPotentialStereo`, a real subsystem) and `FpDensityMorgan1/2/3`. The other ~18 are already
  computed elsewhere or trivial from the boundary.

## Two things that will waste your time if you forget them

* **The pin is enforced** in `pyproject.toml` via `[tool.uv] constraint-dependencies`. A bare
  `uv pip install -e .` no longer moves rdkit. Isolated runs need `--no-project` or the project
  constraints block a deliberately different rdkit.
* **A version banner is not evidence** — `cpp/verify_hume.py` carries a numeric canary for this,
  checked at both ends of the run. A process can print `rdkit 2025.09.2` and compute 2026.3.5's
  numbers out of unlinked-but-still-mapped dylibs.

## Measured at the pause (CONTENDED — ordering only, not publishable)

**STALE AS OF THE `Z` WEIGHT, and deliberately not re-measured here.** The `autocorr` line below
is 486 columns; the block is now 540, and `cpp_all_columns` predates three families besides. No
replacement number was taken because the box was at load ~129 on 12 cores when the `Z` work
landed, and a µs/mol figure from that would look like a measurement without being one. Re-run the
whole table on a quiet machine rather than patching one row of it.

`bench_e2e.py hume 2000 7`, CPU time, cold molecules, load1 10.92:

    cpp_all_columns            646.3 ± 89.94    1015 emitted, 615 of the 865
    extract_pickles_boundary   167.1 ±  0.99
    smiles_parse                59.7 ±  0.61
    ecfp_r3_2048                30.3 ±  0.98

Per family, paired and differenced *within* each repetition:

    infocontent   291.3 ± 13.23    42 cols     <- SUPERSEDED: now 62.4 (see the Ipc-gating note)
    blocks_182    201.6 ±  4.97   182 cols
    autocorr       22.5 ±  5.33   486 cols     <- cheapest per column by a wide margin
    topocharge     14.8 ± 11.14    21 cols
    pathcount      13.8 ±  5.95    11 cols
    vsa            13.5 ±  3.98    66 cols
    estate          2.6 ±  4.44   158 cols
    ringcount       1.7 ±  4.70    49 cols

**Hold these loosely.** The ±89.94 on `cpp_all_columns` is 14% — the box got busy mid-run. The
same arm at load1 5.05 gave 635.3 ± 3.12, so the SD is the machine, not the code. AC's 22.5 µs is
COMPUTE ONLY; its boundary cost is the +67 µs in `extract_pickles_boundary` (100.3 → 167.1), the
H-added pickle.

## The baseline arm, and why it is not done

`results/e2e/baseline.json` is still a 100-molecule / 3-rep run. Not an invocation problem — at
mordred's ~390 ms/mol it is roughly **an hour** at 2000×3. The docstring's command is also
incomplete: `calc.pandas()` needs pandas, which mordred does not declare. Working invocation:

```
uv run --isolated --python 3.11 --with "mordred==1.2.0" --with "rdkit==2025.9.2" \
       --with "numpy==1.26.4" --with "pandas<2.2" python bench_e2e.py baseline 2000 3
```

## A `pgrep` self-match that has now stalled two agents

`until ! pgrep -f "verify_something" ; do sleep 30; done` **can never exit**, because `pgrep -f`
matches full command lines and the waiting shell's own command line contains the string it is
grepping for. It sees itself, forever. This has now stalled two separate agents for 8+ minutes
each with the work long finished.

Match on a bracketed pattern that cannot match itself — `pgrep -f "[v]erify_something"` — or wait
on a PID.

## A git hazard that has now fired twice

`4eec23a` ("intermediate commit and push") swept up an agent's mid-flight Autocorrelation work.
It happened to catch a consistent state and was verified after the fact — but earlier in this
session the same pattern captured a *slower* eigensolver that then had to be superseded. **Prefer
staging named paths over `git add -A` while agents are running**, or check `ListAgents` first.

## Non-finite values are CORRECT and expected

`featurize_all` returns NaN in some columns — 144 of 1015 for ethanol. These are `AATS<k>*` at a
lag longer than the molecule's diameter: 0/0. Confirmed against mordred, which returns an error
object for exactly those and a real value for `AATS1c`. `cpp/ac_weights.h` states the contract:
NaN where mordred returns NaN. Do not "fix" this, and do not let a downstream model see it
without an explicit decision.

## An audit finding about the 865 themselves

**`ABCGG` is one of the 865, and the pinned oracle cannot compute it.** `mordred/ABCIndex.py`
ends `return np.float(...)`; numpy removed `np.float` in 1.24, and mordred 1.2.0 requires numpy
1.x. Under the pinned env (mordred 1.2.0 / numpy 1.26.4) **both `ABCIndex` and `ABCGG` raise
`AttributeError` on every molecule** — verified directly, not inferred.

Two consequences, and the second is the one that matters:

1. Our port compares against the **restored** function (`np.float` re-aliased to builtin `float`,
   which is all it ever was), so the 55-column bit-exact result stands. The shim is recorded in
   `cpp/verify_chiwalk.py`, the same pattern as `verify_topo3.py`'s `np.product`.
2. **`data/dedupe.json` cannot have been produced in the pinned environment.** A column that
   raises on every molecule cannot have a correlation computed for it, yet `ABCGG` survived the
   r > 0.99 dedupe. So the set of 865 was defined under some numpy < 1.24, and its provenance is
   not the environment every exactness claim is pinned to.

Nothing downstream is known to be wrong — but "the 865" is a load-bearing number for this project
and the environment that produced it should be recorded and, ideally, the dedupe re-run under the
pin. Until then, cite the 865 as inherited rather than as reproducible.

## Two families whose names collide and are unrelated

* `src/hume_core/topomisc.h` contains mordred's **`Constitutional`** family (`Sp`, `MZ`, `Mv`,
  `Mp`, …), computed on the **hydrogen-added** molecule.
* `src/hume_core/constit.h` is the **"small constitutional" census block** — `CarbonTypes`,
  `AtomCount`, `BondCount`, `KappaShapeIndex` and friends.

Zero column overlap, confirmed. The names read as a collision to anyone skimming.

## Two traps that will bite the next person, both found by falling into them

**1. `AssignStereochemistry(cleanIt=True, force=True)` CLEARS `_ChiralityPossible`** unless you
also pass `flagPossibleStereoCenters=True`. `Descriptors.NumAtomStereoCenters` counts that flag,
so without it the descriptor silently returns 0 — and an ill-posedness screen built on that call
reported a *well-posed* column as unstable on 4,125 of 9,000 shuffles. **`verify_chiwalk.py`,
`verify_topo3.py` and `src/hume/_extract.py` all omit the flag.** Any future stereo work run
through them will measure an artifact.

**2. `hume.ALL_COLUMNS` contains four DUPLICATED names** — `MaxEStateIndex`, `MinEStateIndex`,
`MaxAbsEStateIndex`, `MinAbsEStateIndex` — emitted once by the 182-column block and again by the
VSA family, as *independent computations that differ in the last bit*. So
`{n: i for i, n in enumerate(ALL_COLUMNS)}` silently resolves to the second copy, and a
whole-matrix A/B against a stored dump "finds" thousands of moved molecules that never moved.
`hume.DUPLICATE_COLUMNS` exposes them. Not silently deduplicated: dropping either copy shifts
every index above it, which is a schema change and the owner's call.

## And one about git, which has now cost a broken HEAD

`git add -u` stages modifications and **not** untracked files. Committing a change that adds a
new header with `git add -u` produces exactly what happened at `ddc2959`: `bindings.cpp` went in
carrying `#include "rdkcore.h"` while the header stayed untracked, so HEAD did not compile.
Stage named paths, and check `git status --short` for `??` before committing a change that
introduces a file.

## Cost is heavy-tailed, and that is the finding that matters

**RETRACTED, and the retraction is the useful part.** This section began as "a prefix of the
corpus is not a sample of it". A paired measurement — same process, same clock, three arms
interleaved — killed it:

    prefix [0:60]      mean 4669.4 ms   median 4.8 ms   max 36621.7 ms
    random 60          mean 1148.8 ms   median 4.1 ms   max 35183.8 ms
    prefix [60:120]    mean 4038.0 ms   median 6.0 ms   max 35741.5 ms

All three **medians agree within 25%**. The means differ by 4x, in the *opposite* direction from
the original claim. Prefix versus random had nothing to do with it.

**THE MEAN IS ~1000x THE MEDIAN.** A handful of molecules cost ~35 SECONDS against a ~5 ms
median, so any small sample is an outlier lottery and its *mean* estimates almost nothing. Sample
size, not sample position, was the whole effect.

### What this demands of every timing number in this project

A mean over N molecules from a distribution with max/median ~7000x is not a stable estimator
unless N is large enough to sample the tail representatively. `bench_e2e.py` uses N = 2,000. Its
SD across repetitions is small (mordred 30,030.7 +/- 196.83) — **but that SD measures repetition
noise on a FIXED sample, not sampling error in the mean**, so it cannot detect this and must not
be read as if it could.

Before the end-to-end ratio is published: re-run at several N (2,000 / 10,000 / 25,000) and show
the mean has stopped moving, or report a median-based figure alongside and say which is which.
This is not hypothetical for HUME either — `bench_e2e.py`'s own docstring already records that
2.9% of molecules carry 46% of BCUT2D time.

## Prefix sampling: measured anyway, and the answer is "fine, but not by design"

Seven harnesses take a **prefix** of the corpus rather than a random sample —
`bench_crippen.py`, `verify_estate.py`, `verify_vsa.py`, `verify_constit.py`, `verify_ic.py`,
`verify_topo3.py`. Only `bench_e2e.py` and `cpp/verify_wiring.py` use `rng.choice`.

That is a latent hazard, so it was measured rather than argued:

    cpp/hard.smi     first 2,000    mean 28.75 heavy atoms, median 26.0
                     random 2,000   mean 28.59 / 29.07,     median 26.0    <- NOT biased
    cpp/mols_h.smi   first 200      mean 32.11,             median 28.0
                     random 200     mean 29.61,             median 28.0    <- biased ~+8%

**`hard.smi` is well mixed, and every subset SCREEN in this repo runs on `hard.smi`** — the
InformationContent ill-posedness percentages, constit's screen, the topo3 screens. None of them
is compromised. The full-corpus runs are unaffected by construction, since a prefix of 100,000
from 100,000 is the whole file.

But this holds by luck rather than by design: nothing enforces that `hard.smi` stays shuffled,
and a future corpus built by concatenating sources would break every one of those seven
harnesses silently. **Prefer `rng.choice` in new harnesses.**

Two cautions worth carrying:
* a prefix can be unrepresentative in ways a mean does not show. `mols_h.smi`'s prefix is ~8%
  larger by mean atom count, which for an O(n²) block predicts ~17% more cost — yet the
  measured cost difference was **14×**. The mean is not the mechanism; a small number of very
  large molecules early in the file is the likely one. Cost lives in the tail.
* an estimate taken from a prefix is a scheduling number and nothing more. It must not reach a
  document, and no correctness claim in this repo rests on one.
