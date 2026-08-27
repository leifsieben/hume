# HUME vs mordred-x

A read of [yzimmermann/mordred-x](https://github.com/yzimmermann/mordred-x) against this project,
done at the owner's request: what to adopt, what not to, and where the two disagree about what
"correct" means.

Their repo was cloned read-only into a scratch directory and not modified. Nothing has been
copied from it — see **Licensing**, which is currently a hard blocker on doing so.

---

## Licensing — read this before copying anything

**mordred-x has no LICENSE file.** `git ls-files` shows none, and the README's "Provenance and
licensing" section documents only what mordred-x *inherited* (mordred and RDKit, both
BSD-3-Clause). It grants nothing for mordred-x's own code — and the README is explicit that
almost everything is its own: "the SMILES and molblock parsers, graph and ring perception, the
SMARTS engine, the eigensolver, every descriptor kernel, the runtime — is original."

Absent a licence, that code is all-rights-reserved by default. So **the 3D descriptors cannot be
copied into HUME as things stand**, and neither can anything else of theirs. This is trivially
fixable — the author adding a BSD-3 or MIT file would settle it, and the provenance section
suggests that is the intent — but it must be settled *before* the code lands in a repository
headed for PyPI, not after.

What is *not* blocked: ideas, and facts. An algorithm is not copyrightable expression, and
neither is "Bondi's 1964 vdW radius for iodine is 1.98 Å". The adoption list below is restricted
to those.

---

## What each project is

| | HUME | mordred-x |
|---|---|---|
| scope | 865 deduplicated columns, mordred ∪ RDKit | 1,613 mordred columns, + 213 3D = 1,826 |
| stack | RDKit parses and perceives; C++ computes | **entirely its own** — parser, ring perception, SMARTS engine, eigensolver |
| 3D | none | CPSA/SASA, MoRSE, GeometricalIndex, GravitationalIndex, MomentOfInertia, PBF |
| dedupe | yes, r > 0.99 — the 865 are what survives | no, full mordred schema |

The architectural difference is the important one and it explains most of the performance gap:
**mordred-x never touches RDKit at runtime, so it pays no interop cost.** HUME pays 58 µs/mol to
parse plus 165 µs/mol at the boundary — 223 of our 779 µs/mol, **29% of end-to-end**, on work
mordred-x simply does not do. Their README claims their parser alone is ~8× faster than RDKit's.

---

## Exactness — the two projects mean different things by it

This is the material difference and it should not be smoothed over.

**mordred-x** (`VALIDATION.txt`, 3,000 molecules × 1,613 columns): 98.64% value agreement with
mordred; **1,335 of 1,613 columns exact on every molecule**; 864 of 3,000 molecules (28.8%) exact
on every column. So 278 columns disagree somewhere.

**HUME**: bit-exactness *per column* over 100,000 molecules, family by family — VSA 66/66,
RingCount 49/49, PathCount 11/11, the fragment matcher 76/76, Chi + walks 55/55, E-state
2,868,290/2,868,290 atoms. Where a definition is ill-posed we diverge deliberately and say so,
and the claim becomes *deterministic and documented* instead.

For Autocorrelation the claim in force is narrower and worth stating precisely, because a
stronger version of it was written into this file before its evidence existed: the 486 columns
that predate the `Z` weight are proven **unchanged** by it — projecting the new 540-column
artifact back onto its non-`Z` columns reproduces the previous file's md5 `7f08884f…`
byte-for-byte across all 98,905 molecules. The full 540-column grade against mordred 1.2.0 is
**running, not finished**, and the 52 new `*Z` columns are unverified until it lands.

Neither standard is wrong; they answer different questions. But a reader comparing "98.6%
agreement" with "bit-exact on 100,000" is not comparing like with like, and our paper must not
imply otherwise.

Two places their looser standard shows against ours: they report `PathCount` at 99.91% with 11
columns differing, where we are 11/11 bit-exact on 100,000; and `TopologicalIndex`, `BalabanJ`,
`WienerIndex`, `VdwVolumeABC` all differ for them, partly by the deliberate NaN choice below.

---

## Where they are ahead

1. **Coverage.** 1,826 columns including 3D against our 615 callable. Not close.
2. **Per-column speed.** Their full 1,613-column schedule is ~1,258 mol/s single-threaded
   (~795 µs/mol) *including their own parse*. Ours is 779 µs/mol for 615 columns plus RDKit
   interop. Per column they are roughly 2.5× ahead.
3. **`InformationContent`: 46.7 µs/mol for 42 columns, against our 279.9.** Six times faster, on
   the exact block that is our bottleneck. See the adoption list.
4. **No interop cost at all**, per the architecture note above.
5. **The Kekulé axis in their invariance screen.** `tools/invariance.py` compares canonical
   aromatic SMILES, *Kekulé* SMILES, and randomly renumbered non-canonical SMILES. We test atom
   and bond order but not the aromatic/Kekulé axis — and that axis is exactly what made
   InformationContent and ETA ill-posed.

## Where we are ahead

1. **Depth of verification.** 100,000-molecule corpora and per-column bit-exactness, against
   their 3,000 (2D) and 300 (3D).
2. **Statistical power on rare defects.** Their `INVARIANCE.txt` reports 0 unstable columns over
   1,500 molecules. Our RingCount ill-posedness occurs on **25 of 100,000 — 0.025%**, giving an
   expected count of 0.4 in 1,500. A screen that size cannot see it: their "100% invariant" is
   under-powered for rare phenomena, not contradicted. (They may also sidestep it entirely,
   since their ring perception is their own and need not inherit RDKit's `symmetrizeSSSR`
   non-uniqueness.)
3. **Drift guards that fire.** Spec-hash guards on every generated table, proven to fire by
   deliberate corruption, plus a numeric canary in `verify_hume.py` because a version banner is
   not evidence — a process can print `rdkit 2025.09.2` while executing 2026.3.5 out of
   unlinked-but-mapped dylibs.
4. **RDKit-side exactness.** We verify against RDKit *and* mordred; they target mordred's schema.
   For the 180 RDKit-sourced columns of our 865 that matters.

---

## Adopt — ideas only, all licence-clean

**1. Hash the IC codes instead of materialising them. [highest value]**
Their `infocontent.hpp` derives atom equivalence classes from "64-bit rolling hashes of the
sorted root-to-leaf path list" rather than building explicit code structures and sorting them.
That is very likely most of the 46.7-vs-279.9 µs/mol gap, and it is an algorithm, not expression.

**Adopt with a condition this project's standards demand and theirs does not discuss: a hash
collision silently merges two distinct equivalence classes and changes the descriptor with no
symptom.** Before shipping it, verify zero collisions across the 100,000-molecule corpus — or use
a wider digest and show the collision probability is below the point of caring. A 6× speedup on
our worst block is worth real effort here.

**2. Add the Kekulé axis to our ill-posedness screen.**
Free, and it directly targets the defect class that has already cost us twice. Our screen
perturbs atom order and bond order; add `Chem.Kekulize(clearAromaticFlags=True)` as a third form.

**3. Bondi radii for iodine and tellurium in `VdwVolumeABC`.**
Mordred's radius table stops at selenium, so *every iodine-containing molecule* returns a missing
value — they measure 8.1% of a typical library. The values are published (Bondi 1964: I 1.98 Å,
Te 2.06 Å) and are facts, not expression. This is a divergence from mordred and must be recorded
as one, but a descriptor that is absent on 8% of a drug library is barely a descriptor.

**4. Exact-integer threshold comparison in ETA.**
`beta_sigma` tests `|eps_i − eps_j| <= 0.3` and some element pairs sit *exactly* on it, so the
answer was decided by one ulp — their `ETA_beta` changed between `-O0` and `-O3` on 3 molecules
in 8,000. Since eps is rational, compare in integers. **ETA contributes no surviving columns
under our dedupe, so this is not actionable today** — but it is the same bug class as our
resistance-block bin edges, and the pattern (a threshold test on a value that can land exactly
on the threshold) is worth grepping our code for.

## Do not adopt without an explicit decision from the owner

**The 1e8 sentinel → NaN change.** RDKit's `GetDistanceMatrix` stores `1e8` for atom pairs in
different fragments, and mordred feeds it straight into the arithmetic: `WPath` = 700000004 for
`CCO.[Na+].[Cl-]`, `BalabanJ` = −2.0e-08, `ECIndex` = 4e8. mordred-x returns NaN, arguing a
topological distance between unconnected atoms does not exist.

**They are right that the values are garbage, and it is still not our rule.** Our house rule
distinguishes a *quirk* (deterministic, so we reproduce it bit-for-bit and comment why) from an
*ill-posed definition* (not a function of the molecule, so we diverge). The sentinel is
deterministic. Under our own stated standard we reproduce it — and our TopologicalCharge port
already does, deliberately: the 1e-16 terms it produces enter `A·D2` and we are bit-exact
including them.

So this is a genuine fork in the road and it is the owner's call, not mine:
* **Reproduce** (current behaviour): "bit-exact against mordred" stays literally true; a library
  with 20% salts carries Wiener indices in the 10⁹ range into whatever model consumes them.
* **Diverge to NaN**: better numbers, but the headline exactness claim needs a second named
  exception alongside the ill-posed ones.

A third option, and the one I would suggest if asked: reproduce by default, and expose the NaN
behaviour under the same flag that will eventually control 3D — an explicit, documented choice
rather than a silent one.

## Blocked on licensing

**The 3D descriptors** — CPSA/Shrake-Rupley SASA (43), MoRSE (160), GeometricalIndex (4),
GravitationalIndex (4), MomentOfInertia (3), PBF (1). The owner wants these, off by default and
computed on request. Their `VALIDATION_3D.txt` reports 1,663 of 1,826 columns exact on 300
molecules, and `PROFILE.txt` puts the full 3D schedule at 470 mol/s with CPSA alone 61% of it
(1,298 µs/mol, 5,112 mesh points per atom).

Two independent obstacles, and the second does not go away if the first does:
1. **No licence** (above).
2. **They need conformers.** Every 3D descriptor is a function of coordinates, so the cost is
   dominated by conformer generation, which is not in that 1,298 µs/mol — HUME's pipeline today
   is 2D-only and has no conformer stage at all. Bolting on 3D means an ETKDG + optimise step
   whose cost exceeds every descriptor in this document combined.

Recommended shape when unblocked: a `dim="2d"` / `"3d"` flag defaulting to `2d`, with 3D
requiring the caller to supply conformers explicitly rather than generating them implicitly —
consistent with `API.md`'s existing refusal to give the input policy a default, for the same
reason: the user should have to think about it.
