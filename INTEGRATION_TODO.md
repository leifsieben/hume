# Wiring queue -- parent session only, after all six agents land

Agents may not edit shared files. These are the changes they identified and correctly
refused to make. Each is applied here, serially, re-verifying after each.

## 1. estate_from() accumulation order  (hume_blocks.h)  -- AFFECTS 79 SHIPPING COLUMNS

RDKit (Chem/EState/EState.py, verified by reading it):
    accum = zeros(n); ...deltas...; res = accum + Is
Ours: S[i] = I[i] first, then deltas accumulate into S.

Mathematically identical, bitwise not. Independently measured by the parent over
39,592 atoms: 80.3% of atoms differ, worst |delta| 3.55e-14. Agent D measured
541,049/670,280 (80.7%), worst 2.13e-14, and that reordering makes it 0/670,280.

This is floating-point associativity, NOT a chemical error -- nothing downstream moves.
It matters only because our stated bar is bit-exactness: it takes the 16 MAX*/MIN*
columns from 6-30% exact to 100%.

Fix: accumulate into a zero array, add I at the end. 3 lines + one n-length array.
AFTER APPLYING: re-verify the 79 existing S<t> columns move by <= 1 ulp and no more.

## 2. ringcount::compute delegation  (ringcount.h)

Agent E could not edit ringcount.h, so counts_ext.h carries ringPass(), a generalised
copy. It guarded the copy rather than trusting it: driftGuard() over 980,000 cells,
0 disagreements. Make ringcount::compute delegate to ringPass -- one line -- which also
lets the two blocks share ONE fusion pass instead of two.

## 3. topomisc walkTraces sharing  (topomisc.h)

estate_ext::compute() takes an optional `const int64_t *tr_in`. topomisc::compute()
already computes the same trace(A^k) k=1..10 on the same graph for TSRW10 but keeps
tr[] local. Expose it and pass it: agent D's group drops 6.22 -> ~1.9 us/mol
(0.75% -> 0.23% of budget). Pure win, no numerical change.

## 4. Ring perception divergence -- DECIDED, NO ACTION

Mordred's reference used RDKit's RAW SSSR; our boundary hands over the REPAIRED set.
Costs 3 cells / 620,000 on 2 molecules. Adopting raw would score 31/31 but put two ring
perceptions in one featuriser. C1C2OCCC3OC2C13 gives raw n7ARing=2 on 259/300 random
numberings and 3 on 41 -- an undecided definition, not a bug in our code. Keep repaired,
document as a divergence in METHODS.md sec 4.

---

# Wiring recipe (read off bindings.cpp, so it is fast when the fleet lands)

Each group needs four edits, all in bindings.cpp:

1. **offset**, in the `OFF_*` chain (line ~422-471). Sets COLUMN order only.
       OFF_<G> = OFF_<prev> + <prev>::N_COLS,
   and move `N_ALL_COLS` to the end of the chain.
2. **family flag**, line ~628. Currently 14 flags, F_ALL = 16383 = 2^14-1.
   Six more takes it to 2^20-1 = 1048575. Still fits `unsigned`.
   F_SPS=16384, F_SPECTRAL=32768, F_ETA=65536, F_ESTATE_EXT=131072,
   F_COUNTS=262144, F_MISC=524288.
3. **compute call**, gated on the flag, writing to `out + OFF_<G>`.
4. **names**, appended in the same order as the offsets.

## EXECUTION order is NOT column order -- and one group depends on it

`counts_ext::compute` needs `nBondsD`/`nBondsKD` from constit columns 18/22 and reads
`W.km`, which is currently BUILT INSIDE the `F_CONSTIT` gate. So:
  - its CALL must come after `constit::compute` (line ~982) and after F_FRAG,
  - and `W.km` must be hoisted out of the F_CONSTIT gate, or F_COUNTS must imply F_CONSTIT.
Prefer hoisting: a caller asking for F_COUNTS alone must not silently get garbage.
Add to the existing `F_NEEDS_H`-style implication machinery (line ~682) rather than
inventing a second mechanism.

## Verify after EACH wiring, not at the end

The whole reason bindings.cpp is single-writer is that a transposed offset is silent.
After each group: rebuild, then assert every PREVIOUSLY wired column is unchanged
bit-for-bit as well as the new group being correct. A wrong offset shows up as the
old columns moving, not as the new ones failing.
