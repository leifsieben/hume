# The port: all 865 columns in C++, no Python in the compute path

**Goal.** Every one of the 865 deduplicated Mordred ∪ RDKit columns computed in C++, and the
end-to-end SMILES → ECFP + descriptors time measured per step with a standard deviation, against
an unoptimised RDKit + Mordred baseline.

Regenerate this census with the snippet at the bottom. Do not hand-edit the counts.

## Where it stands

| | columns |
|---|---|
| ported and verified in C++ | **561** |
| in flight | **~180** |
| still Python | **~124** |
| (already native, counted under an RDKit-family label) | ~22 |

**A warning about the number 182.** `hume.featurize_blocks` returns 182 columns and they are
`ALL EXACT`, but they are **mostly HUME-specific descriptors** — `SATS*`, `RATSC*`, `RW*`,
`sysbin*`, `conj_*`, `pa*_max`, `C3`–`C8` — and only about 22 of them are members of the 865.
Reading "182 verified" as "182 of the 865 done" overstates the position by roughly eightfold.
The bulk of the real coverage is elsewhere: `cpp/ac.cpp` computes **419** Autocorrelation columns
and `src/hume_core/estate_typer.h` computes **50** E-state columns.

## Ported and verified

| family | n | where | evidence |
|---|---|---|---|
| Autocorrelation | 419 | `cpp/ac.cpp`, `cpp/ac_weights.h`, `cpp/ac_tables.h` | ATS/AATS/ATSC/AATSC/MATS/GATS × 9 weights. The nine weight vectors are computed in C++, not handed in — that removed 473.9 µs/mol, the single largest item in the pipeline. |
| EState | 50 | `src/hume_core/estate_typer.h`, `cpp/estate_tables.h` | 2,868,290 / 2,868,290 atoms exact on `cpp/hard.smi`; 100,000/100,000 column values vs mordred 1.2.0. 0.834 µs/mol vs 636. |
| VSA binning | 59 | `src/hume_core/vsa_bins.h`, `cpp/vsa_tables.h` | **66/66 columns bit-exact vs RDKit** over 100,000 molecules, **5/5 vs mordred**, and all four per-atom vectors exact on 2,868,290 atoms. Labute ASA was the real work. |
| InformationContent | 33 | `src/hume_core/infocontent.h`, `cpp/ic_tables.h` | Not exact-vs-mordred — mordred is ill-posed here. **42 columns bit-identical under renumbering**, order-0 control passes. `Ipc` has an open bug; see the header. |

Plus, inside the 182 blocks: `BCUT2D_*` (8), `Kappa1-3` + `HallKierAlpha` (4), RDKit `Chi*` (9),
`BalabanJ` (1), and the four `*EStateIndex` reductions.

## In flight

| family | n | agent |
|---|---|---|
| RingCount | 49 | topology port |
| TopologicalCharge | 21 | topology port |
| PathCount | 11 | topology port |

## Still to port

Grouped by the machinery they share, which is how they are being scheduled — not by size.

**A. VSA binning, 59.** `rdkit_EState` 23, `rdkit_Crippen` 20, `rdkit_Gasteiger` 13, `MoeType` 3,
`rdkit_TPSA` 1, `TopoPSA` 1, `SLogP` 1. One mechanism: a per-atom contribution vector, binned by
fixed edges, summed. The contributions are already computed natively (Crippen from
`crippen_typer.h`, Gasteiger from the boundary, E-state from `estate_from`). What is missing is
Labute ASA and the bin machinery. `MoeType` resolves via `getattr` to the *same* code path and the
*same* edges as the `rdkit_*` VSA columns, so it is not a separate implementation.

**B. `rdkit_core`, 99.** Fragment counts (`fr_*`), H-bond donors/acceptors, ring and heteroatom
counts. Mechanically the largest block and mostly SMARTS counting.

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

   **How to tell, mechanically: permute the atom numbering and recompute.** Renumber with a
   random permutation (`Chem.RenumberAtoms`), or round-trip through canonical SMILES, and compare.
   Any column that moves is ill-posed. Do this for every family you port, before you start
   optimising anything — it is cheap and it changes what the target even is.

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
