# E_counts — 31 columns, `src/hume_core/counts_ext.h`

    .venv/bin/python verify_counts.py            # grade 20,000 x 31, the ring measurement, timing
    .venv/bin/python verify_counts.py --kekule   # + the Kekule stability experiment of §3

Deliverables: `src/hume_core/counts_ext.h`, `verify_counts.py`, this file.
`src/hume_core/bindings.cpp` is untouched; the wiring the parent needs is at the bottom of
`counts_ext.h` and repeated in §6 below.

---

## 1. Result

All 31 columns, all 20,000 corpus molecules, against `data/dedupe2/matrix.npz`:

| | |
|---|---|
| columns exact on 20,000 / 20,000 | **28 of 31** |
| columns exact on 19,999 / 20,000 | 3 (`n7ARing`, `n7AHRing`, `nG12AHRing`) |
| mismatched cells | **3 of 620,000** (0.00048%) |
| cells inside 1e-9 but not bit-identical | 0 |
| cells inside 1e-6 but not 1e-9 | 0 |
| NaN on either side, any column | 0 |

The three mismatched cells are all one thing — the ill-posed ring set — and are §4.
Every other column is bit-identical on every molecule. There is no column I could not make
exact for a reason other than the reference being undefined.

Two guards ride along in `verify_counts.py` and both pass:

* `counts_ext::selfCheck()` — regenerates mordred's whole 138-entry `RingCount.preset()` from its
  own nested loops, names each entry with `RingCount.__str__`'s own rules, and requires all 21
  ring rows to appear there exactly once with identical `(order, greater, fused, arom, hetero)`.
  Also asserts the emit slots `C_RING0..C_RING0+20` carry the RING_COLS names in order, so a
  transposed ring block fails at load rather than on a molecule.
* `counts_ext::driftGuard()` — per molecule, **980,000 cells, 0 disagreements** (see §5).

---

## 2. What each column actually reads — and the H-graph question

`Context.from_query` in `mordred/_base/context.py` is the only place mordred decides which
molecule a descriptor sees:

    m = Chem.AddHs(mol) if eh else Chem.RemoveHs(mol, updateExplicitCount=True)
    if ke: Chem.Kekulize(m)

keyed on the descriptor class's `explicit_hydrogens` / `kekulize`. Read, not assumed — and it
does not line up with the column names:

| column | `explicit_hydrogens` | molecule it counts on |
|---|---|---|
| `nAtom` | **True** (`self._type in {"H","Atom"}`) | `AddHs(mol)` |
| `nHeavyAtom` | False | `RemoveHs(mol)` |
| `nP` `nF` `nI` | False | `RemoveHs(mol)` |
| `nAromAtom` `nAromBond` | **True** — `Descriptor`'s class default, which `AromaticBase` never overrides | `AddHs(mol)` |
| `nBonds` (`any`) | **True** | `AddHs(mol)` |
| `nBondsKS` (`single`, kekulize) | **True** | `Kekulize(AddHs(mol))` |
| the 21 RingCount | False (`RingCountBase.explicit_hydrogens = False`) | `RemoveHs(mol)` |
| `nRot` | False (`RotatableBondsBase`) | `RemoveHs(mol)` |

Three consequences, all reproduced:

**`nAtom` and `nHeavyAtom` are the same function on two molecules.** Both dispatch to
`_calc_all`, which is `self.mol.GetNumAtoms()`. So "HeavyAtom" is a misnomer: it is not
"atoms with Z != 1", it is the atom count of the H-suppressed graph — and
`Chem.RemoveHs` does **not** remove every hydrogen. **558 of the 20,000 corpus molecules
(823 atoms) carry an explicit hydrogen vertex that survives both `MolFromSmiles` and
`RemoveHs`**, and on **0 of 20,000** does `RemoveHs` change the atom count at all. So
`nHeavyAtom` counts hydrogens on 2.8% of this corpus, and mordred agrees with us that it does —
we reproduce it as `km.n`, not as a filtered count.

`constit.h` records the isotope mechanism (`RemoveHs` keeps `[2H]`) from `cpp/hard.smi`; this
corpus exercises a *different* one, and there are 0 isotopic hydrogens in it. Every one of the
823 is a **stereo-defining** hydrogen: the `[H]` of a `[H]/N=C(...)` amidine, whose directional
bond carries the C=N geometry, which RDKit's `removeHs` refuses to drop. Confirmed by deleting
the `/` and `\` from one such SMILES: 15 atoms with the directional bonds, 14 without, and
`RemoveHs` is the identity in both cases. Two independent reasons a hydrogen ends up inside a
column named "HeavyAtom"; the boundary contract is `RemoveHs`, not "Z != 1", and this file
honours the contract rather than the name.

**No `h_blobs` are needed.** `constit::HDerived` already establishes and verifies
`nAtomsH = n + sum(GetTotalNumHs())` and `nBondsH = nb + nHadd` on the heavy boundary.
Re-checked directly here: `Chem.AddHs(m).GetNumAtoms() != n + sum(GetTotalNumHs())` on
**0 of 20,000** molecules. `nAtom` is `HDerived::nAtomsH`; `nBonds` is `HDerived::nBondsH`.
The pickle's second (H-added) molecule is untouched by this block.

**`nAromAtom`/`nAromBond` are H-added columns whose values are heavy-graph values.** Reading
`Descriptor.explicit_hydrogens = True` as "these are heavy-graph columns" would have been reading
it wrong; it costs nothing only because `Chem.AddHs` adds non-aromatic atoms joined by
non-aromatic bonds and never touches an existing flag. Checked: 0 of 20,000 corpus molecules have
an aromatic hydrogen vertex.

One separate trap: **`nAromBond` is the flag alone**, `sum(1 for b if b.GetIsAromatic())`, where
`BondCount`'s `nBondsA` is `b.GetIsAromatic() or b.GetBondType() == AROMATIC`. `constit.h` records
4 bonds in `cpp/hard.smi` where those two disagree (TRIPLE bonds carrying the aromatic flag), so
the distinction is real even though it does not fire on this corpus. `nAromBond` is written as
`(bcode & BC_AROM) != 0` and nothing else.

---

## 3. `nBondsKS` — asked whether it is ill-posed; it is not. **Quirk-free, well-posed, reproduced.**

`BondCount("single", kekulize=True)` counts bonds whose type is `SINGLE` in
`Chem.Kekulize(Chem.AddHs(mol))`. The contract asked whether the arbitrary Kekule assignment
makes this column ill-posed. **It does not**, and there are two independent reasons to say so.

**The argument.** Kekulization rewrites only AROMATIC-type bonds and preserves every atom's total
valence. So for an atom carrying aromatic bonds, the number of ring double bonds it ends up with
is fixed by

    takesDouble(i) = tval[i] - nh[i] - round(nonAromaticValenceContrib(i)) - nAromaticBonds(i)

— a per-atom function of the boundary that mentions no matching at all. The number of promoted
bonds is half that sum. Every Kekule structure of the same molecule therefore has the same number
of doubles, hence the same number of singles. *Which* bonds move is arbitrary; *how many* is a
valence invariant. That per-atom quantity is exactly `constit.h`'s already-verified `takesDouble`
(in {0,1}, even sum).

**The measurement**, because an argument is only an argument (`verify_counts.py --kekule`).
650 aromatic corpus molecules, sampled from the 15,943 that have an aromatic bond, were each
re-parsed from **40 randomly re-ordered SMILES** (26,000 re-parses total: renumber, write
non-canonical SMILES, re-parse, `AddHs`, `Kekulize`, count RDKit's own bond types):

    nBondsKS moved on 0 of 650 molecules
    nBondsKD moved on 0 of 650 molecules

Contrast with the ring set in §4, where the same style of experiment moves the answer on 41 of
300 numberings of a single molecule. So rule 4 says **reproduce**, and the header does — exactly,
on 20,000 of 20,000.

**It is reproduced without a second Kekule pass.** In the kekulized H-added graph the single bonds
are: the heavy bonds already typed SINGLE, plus the aromatic-TYPE bonds that were not promoted,
plus one per added hydrogen. The promoted count is `nBondsKD - nBondsD`, both already emitted by
`constit.h`, so

    nBondsKS = nHadd + #{SINGLE} + #{aromatic type} - (nBondsKD - nBondsD)

and the only Kekule reasoning in the tree stays in the one place it was verified. `#{aromatic
type}` uses `constit::isAromType` (flag set **and** no order bit), not the flag, which is what
makes the count exclusive with `#{SINGLE}` and correct on the flag/type-disagreeing bonds.
`compute()` throws with the molecule's numbers if `nBondsKD - nBondsD` is outside
`0 .. #{aromatic type}`, which is the shape a mis-wired constit offset would take.

---

## 4. The ring set — the one divergence. **3 cells, 2 molecules, documented.**

`RingCount.Rings.calculate` is `[frozenset(s) for s in Chem.GetSymmSSSR(mol)]` — the
**symmetrised** SSSR, not the plain SSSR and not all cycles. This header perceives nothing: it
consumes the `ringcount::Mol` `bindings.cpp` already fills from the boundary's `rings_for()` CSR,
which is the same single perception the existing 49-column block reads.

`Chem.GetSymmSSSR` is **not a function of the molecular graph** (`ringcount.h` and
`src/hume/_rings.py` carry the standing evidence). The boundary repairs it by perceiving on a
canonically rebuilt skeleton; mordred's reference values in `matrix.npz` come from RDKit's **raw**
answer. On an ambiguous molecule the two differ by construction.

`verify_counts.py` measures how far apart, by **running the same C++ driver twice** over two boundary dumps
that differ in nothing but the ring CSR, so the ring set is isolated as the sole cause and no
descriptor logic exists on the Python side:

| ring set fed to `counts_ext::compute` | columns exact on 20,000 | mismatched cells |
|---|---|---|
| RDKit raw `RingInfo().AtomRings()` — what mordred saw | **31 of 31** | **0** |
| the boundary's repaired `rings_for()` — what ships | 28 of 31 | 3 (2 molecules) |

The first row is the strong statement: **with mordred's own ring set this header reproduces
mordred exactly on every one of the 620,000 cells.** Everything separating us from 31/31 is the
boundary's deliberate repair, and nothing else in the group is even slightly approximate.

The three cells, in full:

    n7ARing      ref 2  ours 3    C1C2OCCC3OC2C13
    n7AHRing     ref 2  ours 3    C1C2OCCC3OC2C13
    nG12AHRing   ref 5  ours 4    [H]/N=C(/c1c([H])c([H])c(C([H])([H])N([H])C(=O)[C@@]2([H])... (a macrocyclic peptidomimetic)

That the *definition* is what is undecided, not this file: re-parsing `C1C2OCCC3OC2C13` from
**300 random SMILES orderings** and reading RDKit's raw ring set each time gives

    ring sizes (4,4,7,7)   -> raw n7ARing = 2   on 259 of 300 numberings
    ring sizes (4,4,7,7,7) -> raw n7ARing = 3   on  41 of 300 numberings

mordred's 2 is one of RDKit's two answers, reached because of the order the corpus SMILES happens
to present the molecule in. Ours is the other one, reached deterministically. **Divergence from an
ill-posed definition, per contract rule 4** — the same divergence `ringcount.h`'s 49 columns
already carry, on the same repaired ring set, for the same reason.

Taking mordred's raw ring set here instead would score 31/31 and put **two ring perceptions in
one featuriser**, which is the exact failure mode `src/hume/_rings.py` exists to prevent. I did
not do it, and I do not recommend it.

`verify_counts.py` re-measures this every run (the "RING SET" block), so if a future corpus makes
the divergence wider it is a printed number and not a surprise. It also prints "columns where the
RAW ring set reproduces mordred exactly", which is the line that would drop below 31 if anything
*other* than the ring set ever went wrong in the ring block.

---

## 5. The one duplication, and the guard that makes it safe

`counts_ext::ringPass()` is a strict generalisation of `ringcount::compute` — same ring
properties, same `|Ri ∩ Rj| >= 2` union-find, same networkx-singleton exclusion — over a
caller-supplied `Spec` table instead of ringcount's fixed 49. It exists **only** because
`ringcount::compute` hard-codes `ringcount::COLS` and this agent may not edit `ringcount.h`.
Everything reusable was reused: `ringcount::Mol`, `ringcount::Scratch`, `ringcount::Spec`,
`ringcount::ring_props`, `ringcount::passes`, `ringcount::uf_find`.

So the copy is **checked, not trusted**. `counts_ext::driftGuard()` runs `ringPass` over
`ringcount::COLS` and requires all 49 values to equal `ringcount::compute`'s, per molecule.
`verify_counts.py` calls it on every molecule: **980,000 cells, 0 disagreements.**

**Recommended follow-up for the parent — one line, and it deletes the duplicate outright:**
replace the body of `ringcount::compute` with

    counts_ext::ringPass(m, S, ringcount::COLS, ringcount::N_COLS, out);

(or, better, move `ringPass` into `ringcount.h` and have `counts_ext.h` call it there, which
inverts the include and needs no `counts_ext` dependency in `ringcount.h`). Either way
`driftGuard` then becomes a tautology and can be dropped. A further win is available on top: the
49 + 21 = 70 specs could share **one** fusion pass instead of two, which is most of §7's cost.

---

## 6. Wiring for `bindings.cpp` (not applied — contract §"What you deliver")

    OFF_COUNTS = <after the last existing block>;   counts_ext::N_COLS == 31

    counts_ext::Inputs cin;
    cin.nRot     = (int)out[OFF_FRAG    + IC.nrot];   // existing input_cols() lookup
    cin.nBondsD  = (int)out[OFF_CONSTIT + 18];        // constit::col_name(18) == "nBondsD"
    cin.nBondsKD = (int)out[OFF_CONSTIT + 22];        // constit::col_name(22) == "nBondsKD"
    counts_ext::compute(W.km, W.rm, cin, out + OFF_COUNTS, W.rs);

and `counts_ext::selfCheck()` beside the other selfCheck calls in the module init.

Emit order is `results/dedupe2/agent_groups.json`'s `E_counts` list read top to bottom, unchanged;
`verify_counts.py` asserts `counts_ext::col_name` against that file on every run, so a
transposition between the header and the group list cannot survive.

Three constraints the parent must honour:

1. **This block must run after `F_CONSTIT` and `F_FRAG`** — it reads three of their outputs.
   If either family can be switched off independently, gate this block on both.
2. **`W.km` is built inside `if (fams & F_CONSTIT)`.** This block needs it, so either share that
   gate or hoist the build.
3. **Look the two constit offsets up by name** through the same `input_cols()` mechanism
   `IC.naRing` uses, rather than hard-coding 18 and 22. `constit::col_name()` is the authority.
   The offsets above are stated so the wiring is reviewable, not so they get pasted.

No change to `bindings.cpp` is *required* beyond adding the block; nothing existing needs editing.

**`nRot` is emitted by wiring, not by computing.** mordred's `nRot` is
`CalcNumRotatableBonds(Chem.RemoveHs(mol))` at RDKit's default strictness — the same call and the
same default as RDKit's own `NumRotatableBonds` descriptor, which `frag_matcher.h` already emits
and which `constit.h` already consumes as `Inputs::nRot`. Measured over the corpus: mordred `nRot`
and RDKit `NumRotatableBonds` disagree on **0 of 20,000** molecules. Computing it a second time
would be a second answer that can drift, so this column is an alias — the arrangement `constit.h`
uses for `SLogP`.

---

## 7. Timing

`build_counts/counts_driver bench`, 11 reps per mode, **6 independent runs**, reporting the
**minimum of the per-stratum medians** across those runs. The machine is heavily contended (load
average ~35 — five agents build in this checkout concurrently), so per-run means and SDs are
dominated by scheduler noise and are not reported: they are non-monotonic in stratum and even
non-monotonic across modes, which is a measurement of the machine and not of this header. The
per-stratum *medians* are stable to about ±15% across runs and monotonic in molecule size, so the
minimum of them is the honest least-contended estimate. Molecule construction is excluded from
modes 0 and 1 because `W.km` and `W.rm` already exist in `bindings.cpp`'s `Work`.

| stratum (heavy atoms) | n | mode 0 — **marginal cost** | mode 1 (+ recomputing the constit bonds) | mode 2 (+ building both views) |
|---|---|---|---|---|
| 0–15 | 4,000 | 0.119 µs/mol | 0.177 | 0.432 |
| 15–25 | 4,000 | 0.209 | 0.323 | 0.989 |
| 25–35 | 4,000 | 0.303 | 0.461 | 0.952 |
| 35–55 | 4,000 | 0.413 | 0.645 | 1.996 |
| 55+ | 4,000 | 0.593 | 1.204 | 3.555 |
| **corpus (equal strata)** | 20,000 | **0.33 µs/mol** | 0.56 | 1.59 |

**Mode 0 is what this group adds: ~0.33 µs/mol, 0.04% of HUME's ~830 µs/mol budget** — and 0.59
µs/mol on the largest stratum, so there is no size at which it matters. Mode 1 is what it would
cost if `nBondsD`/`nBondsKD` were recomputed instead of wired (70% more, the price of ignoring
§6); mode 2 is the standalone harness's own upper bound and is not a cost the extension pays.

The nine scalar columns are two linear sweeps over atoms and bonds and are flat in molecule size;
the 5× spread across strata is the fusion pass, which is O(R²) in the ring count. Folding the
49 + 21 specs into one pass (§5) would leave the two RingCount blocks running **one** fusion pass
between them instead of two, which is where the remaining saving is.

---

## 8. Things I could not make exact, and things I deliberately did not do

* **3 cells of 620,000**, §4. Not fixable without a second ring perception; the definition, not
  the implementation, is what is undecided. Everything else is bit-identical.
* **I did not touch `bindings.cpp`.** §6 is the wiring, as a note.
* **I did not touch `ringcount.h`**, though §5 says where the one-line improvement is.
* **No descriptor value is computed in Python anywhere in the harness** (house rule 1). The two
  diagnostics that could have been tempted into it are not: the ring measurement runs the same
  C++ driver twice over two ring sets, and the Kekule experiment reads RDKit's own kekulized bond
  types, which is calling the reference for ground truth.
* **`verify_counts.py` writes and compiles its own driver into `build_counts/`.** counts_ext.h is
  not reachable from the extension until the parent wires it, so the harness compiles the header
  directly with the same `c++ -O3 -std=c++17` the shipped extension is built with. `build_counts/`
  holds only build artefacts; the only repository source this group adds is the header. A full run
  leaves ~78 MB of regenerable corpus dumps in it, which I have deleted (only `counts_driver` and
  its generated `.cpp` are left); `build_counts/` is not in `.gitignore`, so **do not commit it** —
  add it there alongside the other agents' build dirs.
  Compiled clean under `-Wall -Wextra` (0 warnings from `counts_ext.h`), and
  `cmake -S . -B <dir> -DCMAKE_BUILD_TYPE=Release && cmake --build` still builds the extension
  unchanged (the header is not yet included by it). Note `cmake` is not on `PATH` in this
  environment — it is `.venv/bin/cmake`, and pybind11 needs
  `-Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir)`.
