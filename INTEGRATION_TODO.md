# Wiring queue -- parent session only, after all six agents land

Agents may not edit shared files. These are the changes they identified and correctly
refused to make. Each is applied here, serially, re-verifying after each.

## 1. estate_from() accumulation order -- WITHDRAWN, DO NOT APPLY

Agent D reported that estate_from() in hume_blocks.h uses the wrong accumulation order
(RDKit does `res = accum + Is`; ours seeds `S[i] = I[i]`) and that 79 shipping S<t>
columns sit on that vector. The ORDER claim is correct -- the parent verified it
independently over 39,592 atoms, 80.3% of atoms differ, worst 3.55e-14.

THE IMPACT CLAIM IS WRONG, and the fix was applied, tested and reverted.

Measured decisively: injecting `S[i] *= 2.0` at the end of estate_from and rebuilding
changes **0 of 1374 columns** over 400 corpus molecules. estate_from is dead code with
respect to every shipping column.

The reason is already written down in src/hume_core/vsa_bins.h:369-390, which the parent
found only after applying the fix. That file carries a SECOND copy of the E-state index
in RDKit's exact association order, precisely because of this, and says so with its own
measurements over 86,654 atoms:
    seed with Is, then accumulate   (estate_from's order)   22,482 / 86,654 bit-exact
    accumulate into zero, add Is last (RDKit's order)       86,654 / 86,654 bit-exact
and states outright: "It is also the reason MaxEStateIndex and friends are computed here
rather than read from the blocks", and "Left alone deliberately: it is another agent's
file and its own callers are verified against it."

So: do not touch estate_from. Its callers (cpp/hume.cpp) are verified against its current
behaviour.

CONSEQUENCE FOR WIRING estate_ext.h: its compute() asks for "BlockWork::ES (hume_blocks.h
estate_from())". Pass vsa_bins.h's RDKit-ordered `estate_indices()` output INSTEAD. The 16
MAX*/MIN* columns should then be bit-exact with no change to any shared file -- agent D
measured them at 6-30% exact against estate_from's order and 100% against RDKit's, and
vsa_bins already computes RDKit's. VERIFY THIS AT WIRING TIME rather than assuming it.

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
