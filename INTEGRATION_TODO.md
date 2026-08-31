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
