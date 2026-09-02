# What the 1,269 columns actually measure

*A provenance map, built by reading the implementations rather than by correlating them. The
companion to `docs/MINIMAL_SPEC.md`, which selects columns; this one says what they are.*

Correlation tells you two columns move together on the molecules you looked at. It cannot tell
you they are **the same measurement in different units**, and it cannot tell you two columns
that agree on drug-like molecules diverge on salts. Only reading tells you that, and the answer
does not depend on which corpus you happened to have.

---

## 1. Every column is a (property, operator, parameter) triple

**The property** is the per-atom or per-bond quantity being measured. **The operator** is how it
is aggregated over the graph. **The parameter** is the knob — a lag, a bin index, a subgraph
order.

| operator | columns | what it does |
| --- | ---: | --- |
| autocorrelation | **519** | correlate a property with itself at topological distance *k* |
| *(unclassified: constitutional, misc, ETA, topocharge)* | 348 | see §4 |
| ring perception | 76 | count rings by size, aromaticity, fusion |
| substructure match | 75 | does this SMARTS pattern occur, and how often |
| E-state atom typing | 60 | sum/extremes of the E-state index per atom type |
| VSA binning | 55 | histogram of a property over accessible surface area |
| subgraph count (chi) | 53 | connectivity index over paths, clusters, chains |
| information content | 22 | Shannon entropy of the orbit partition |
| matrix spectrum | 11 | eigenvalue functionals of a weighted graph matrix |
| Burden eigenvalue | 8 | extreme eigenvalues of the Burden matrix |

**41% of the library is one operator.** Autocorrelation is 12 properties x 6 normalizations x 9
lags, and that factorization is the single most useful fact in this document.

## 2. The property vocabulary, and where it double-counts

The twelve autocorrelation weights (`cpp/ac_weights.h`) are not twelve independent quantities.
Grouped by what they physically measure:

| construct | weights | columns emitted | kept by minimal-v1 | pairwise r |
| --- | --- | ---: | ---: | --- |
| **electronegativity** | `se` Sanderson, `pe` Pauling, `are` Allred-Rochow | 52 + 42 + 35 = **129** | 33 + 1 + 12 = **46** | — |
| **size / dispersion** | `v` vdW volume, `p` polarizability | 45 + 49 = **94** | 22 + 34 = **56** | **0.995** |
| **nuclear identity** | `Z` atomic number, `m` atomic mass | 52 + 12 = **64** | 39 + 5 = **44** | **0.999** |
| charge | `c` Gasteiger | 34 | | |
| connectivity | `d` sigma electrons, `dv` valence electrons | 52 + 50 | | |
| ionization | `i` | 44 | | |
| composite | `s` intrinsic state (electronegativity x degree) | 52 | | |

**146 columns for three physical quantities.** Three electronegativity scales are one
measurement in three units; polarizability is proportional to volume; mass is proportional to Z.

 **This is why a statistical criterion cannot fix it.** `ATS3v` and `ATS3p` correlate at 0.995,
so 0.5% of their variance is independent — and a criterion optimizing *orthogonality*, as
pivoted QR does, will happily spend a column on that 0.5% when it has 800 slots to fill. QR
collapsed Pauling (42 -> 1) but kept 33 Sanderson **and** 12 Allred-Rochow. There is no
correlation cutoff that separates "0.995, same construct" from "0.99, genuinely different"
without knowing which is which, and knowing that requires reading.

## 3. Redundancy has two kinds, and they need different tools

**Parametric** — within one (property, operator) cell. Do you need 9 lags or 3? 14 charge bins
or 5? Empirical, cheap, and a statistical criterion answers it well.

**Constructual** — across properties within one operator. `ATS3se`, `ATS3pe`, `ATS3are` are one
measurement written three ways. **Only provenance answers this**, and no amount of held-out data
will.

The current spec conflates them, which is why it spends 146 columns on three quantities while
correctly collapsing many genuinely redundant lags.

## 4. What is in "unclassified"

Not a family — the residue of several. `misc_ext.h` says so itself: *"THIS GROUP IS NOT A
FAMILY. It is eleven unrelated sub-families that happened to survive the same dedupe."*
Constitutional counts, ETA (29), topological charge (21), Kappa shape, CPSA, Lipinski,
molecular-distance-edge, and our own additions. Any MECE scheme has to break this up; keeping it
as a bucket is what let the notation defects in §5 hide.

## 5. Notation stability, measured (experiment B)

3,000 molecules, each rewritten 4 times with randomized atom ordering, re-parsed and
re-featurized. A descriptor is a function of the molecule, so every column must return the same
value.

- **828 of 1,269 columns are not bit-identical.** Almost all of that is floating-point
  accumulation order changing with atom ordering: harmless, and 816 of them move by less than 1%
  of the column's own standard deviation.
- **12 columns move by more than 1% of their own SD. Nine of the twelve are descriptors we
  introduced.**

| column | worst / SD | molecules affected | origin |
| --- | ---: | ---: | --- |
| `XATS4`, `XATS2` | 29.3 | 0.03% | ours |
| `T_sum` | 9.4 | 0.03% | ours |
| `het_frac_max` | 4.2 | 1.3% | ours |
| `het_in_max` | 2.8 | 1.3% | ours |
| `extra_arom_max` | 2.7 | 1.1% | ours |
| `linearity` | 1.9 | 1.5% | ours |
| `sys_max_rings` | 1.7 | 1.1% | ours |
| `diam_max` | 0.9 | 1.5% | ours |
| `ETA_dEpsilon_B` | 0.36 | 15.9% | mordred, documented |
| `ETA_epsilon_4`, `ETA_dEpsilon_C` | 0.24 | 33.1% | mordred, documented |

**The six ring-system columns share one root cause, and it is instructive.** `conjugation.py`
already identified the tie-break and pinned the sort to `kind="stable"`, with a long comment
explaining that an unstable sort picked different pi systems depending on RDKit's atom
numbering. That fix made the value **reproducible** — the same SMILES always gives the same
answer — but not **canonical**: stable sort keeps the last maximal system *in atom order*, and
rewriting the molecule changes atom order. Reproducibility and notation-invariance are different
properties, and the first was mistaken for the second.

The fix is to break ties on a chemical invariant (ring-system composition, then diameter) rather
than on position in the atom list. Until then these six carry roughly 1% label noise that no
amount of training data removes, because the same molecule from a different source file gets a
different value.

## 6. What this map is for

A budget allocated per (construct, operator) cell is roughly **40 decisions**, each interpretable
and defensible in a methods section, instead of a 1,269-way ranking that no reader can check.
And it separates the question a statistical criterion is good at — *how many lags?* — from the
one it is structurally blind to — *is this the same quantity in different units?*

 **One property of the current criterion must survive any successor.** Pivoted QR ranks by
residual orthogonality, not by variance, and this protects rare features for free: columns
firing on <=2% of molecules have median rank **71** of 1,267, against **704** for columns firing
on >50%, and 64 of 70 are kept. A family-representative scheme that picks each family's
representative by typicality or variance would delete exactly the long tail that matters for
rare-substructure activity. Whatever replaces QR has to be checked against this.
