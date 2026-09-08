# How default, small and minimal were derived

*Plain English. Every number here is measured; the record with the statistics is
`docs/selection/HUME_N_PREREGISTRATION.md` and `docs/selection/HUME_Minimal_definition.md`.*

The library computes 1,269 descriptors. Three smaller sets ship alongside that, and they were
arrived at by three different methods, in that order, because each one exhausted the arguments
available to it and the next had to find a new lever.

## default, 622 columns: removed for reasons you can read off the definitions

The first cut needed no experiment. Going through the 1,269 definitions one at a time, a column
was removed only when one of three things was true of it:

1. **It was the same physical quantity in different units.** Three electronegativity scales.
   Atomic mass against atomic number. Polarizability against volume. These are not correlated
   columns that happen to move together; they are the same measurement written twice, and which
   one you keep is arbitrary.
2. **It was already carried by the fingerprint** shipped alongside the descriptors. The
   substructure flags are the clear case: a fingerprint that records which fragments are present
   already contains the answer to "is this fragment present".
3. **It was exact arithmetic on columns that remain.** Ring counts and atom counts that are sums
   of other ring counts and atom counts.

No threshold was involved and no benchmark was consulted, which is the point. A correlation
cutoff cannot tell "0.995, and it is the same construct" from "0.99, and they are genuinely
different"; reading the definitions can. That matters because a set chosen by fitting to
benchmarks inherits whatever those benchmarks happen to reward.

Held out afterwards on 33 tasks that took no part in choosing it, **default is not merely as good
as the full 1,269 -- it is better**, by 1.17% (p = 0.001, worse on only 9 of the 33). The 647
columns it drops were a mild liability rather than dead weight, which is what you would expect
from feeding a tree several copies of the same number.

An earlier attempt at this cut, since withdrawn, ranked columns by whether the survivors could
reconstruct them *linearly*. That failed on contact with the actual consumer: a column
reconstructible at linear R² 0.997 was not reconstructible by a depth-6 tree, which cannot split
on a linear combination. The lesson shaped everything after it -- test on the model that will
read the features, not on a criterion that is merely convenient.

## small, 408 columns: the resolution of the parameter sweeps

The three arguments above were now spent, so this cut needed something else. What remained in the
622 was dominated by **parameter sweeps**: one idea evaluated at many settings. Connectivity
indices at path orders 0 through 7. Information content at six orders crossed with four
normalisations. Charge autocorrelation at ten topological lags. Random-walk statistics at seven
window sizes. In each case the number of settings was chosen by whoever wrote the original
library, not by chemistry.

So the question became: is the resolution higher than it needs to be? Each sweep was halved and
the result measured against the full set on the downstream tasks. Every one of them was free.
Two further things had to be checked before that meant anything:

- **Do the cuts compose?** Each halving was measured with the other sweeps still present to
  absorb the loss, so "each is individually free" does not imply "all of them together are free".
  Applying all eight at once removes 172 columns, and that combination was measured directly
  rather than inferred. It held.
- **Does it matter which half you keep?** A random subset of the same size costs 1.4% and is
  worse on 28 of 33 tasks, while the principled one costs nothing (p < 0.001 head to head). So
  the selection is doing real work. Interestingly, *which* half of a sweep you keep barely
  matters -- low orders are no better than high orders -- but keeping some of every family
  matters a great deal. A random subset can strip a small family bare by accident, and the small
  families are where the non-redundant information lives.

Also dropped here: 32 of the 75 substructure flags, the ones no downstream task made use of. The
other 43 stayed, because removing all of them costs a real 1.25%. That was a surprise worth
recording -- those flags are in principle redundant with a good fingerprint, and in practice the
tree uses them anyway.

**small is free**: +0.30% against the full set on held-out tasks (p = 0.711), and it is the
cheapest of the four to compute.

## minimal, 256 columns: budget allocated by information, not by width

The last cut is the only one that costs something, and it is offered as a trade rather than as a
free reduction.

The obvious way to shrink further is to take a proportional slice of every family. That performs
badly, and the reason is instructive: families differ enormously in how redundant they are. One
family has 121 columns but only about 38 independent directions in it; another has 43 columns and
29 independent directions. Proportional allocation hands the first family three times the budget
of the second while the second is the one carrying distinct information.

So the budget is allocated in proportion to each family's **effective rank** -- how many
genuinely independent directions it contains -- rather than to its column count. That single
change moved roughly 30 slots from the most redundant family to the least redundant ones and
turned a measurable penalty into a much smaller one.

**minimal costs about 2.5% on classification tasks** (its overall figure, +1.47%, has a
confidence interval that includes zero, so there is no demonstrated overall cost; the
classification penalty is the real one). Two things are worth knowing before choosing it:

- It is **not cheaper to compute** than small. 122.4 microseconds per molecule against 116.7:
  fewer columns, slightly more time. Featurization is organised by family, and 256 columns still
  touch every family that 408 do, so dropping columns inside a family saves nothing. The benefit
  of minimal is fewer columns downstream, not less work upstream.
- Its 256 columns are much better than 256 arbitrary ones -- the selection beats a random subset
  of the same size by a wide margin -- but they are not as good as 408.

## How the sizes were chosen, and one that was rejected

The sizes were not picked in advance. A stopping rule was written down and committed *before any
result existed*: take the smallest set whose cost cannot be distinguished from the full set, by a
test that has to demonstrate the cost is small rather than merely fail to prove it is large.
Candidate sizes of 128, 200, 256, 300, 350 and 408 were then measured.

That rule selected **200**, and an entirely separate label-free calculation -- summing the
independent directions in each family -- independently suggested 210. Both were rejected by the
held-out data, where their confidence intervals excluded zero: a 200-column set costs between
0.1% and 2.6% overall and 4.5% on classification. The selection tasks could not see this because
they were individually noisier than the held-out ones, so a real penalty of about 1.3% sat inside
their resolution.

That rejection is why small is 408 rather than 200, and it is the main reason the held-out set
was kept sealed from the first line. Two independent criteria agreed on a number, and the data
that had no say in choosing it disagreed with both.
