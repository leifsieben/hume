# How the three reduced sets were derived

*Plain-English companion to Table~\ref{tab:column-sets}. Every number here is measured; the
method record is `docs/selection/HUME_N_PREREGISTRATION.md`.*

The library computes 1,269 descriptors. Three smaller sets are shipped alongside it, each nested
inside the one above: `minimal` is a subset of `small`, which is a subset of `default`. They were
built in that order, and each step used a different kind of argument.

## `default` (622 columns): remove what is provably redundant

The first cut asks nothing about performance. A column was removed only if one of three things
was true of it by construction:

- **It is the same physical quantity in different units.** Three electronegativity scales say the
  same thing three ways; so do atomic mass against atomic number, and polarisability against
  volume. Which of these is dropped is a reading of the definitions, not a statistical decision,
  because no correlation cutoff cleanly separates "0.995, same construct" from "0.99, genuinely
  different quantities".
- **It is already carried by the fingerprint that ships alongside.** The descriptor block is
  emitted with a 2,048-bit ECFP, and a substructure flag that the fingerprint already encodes is
  a duplicate of information the model receives anyway.
- **It is an exact arithmetic identity of columns that remain.** Several ring and constitutional
  counts are sums of other counts. These were verified on two chemical spaces rather than assumed.

This is why `default` is the recommended set despite not being the smallest. It was never fitted
to a benchmark, so there is no dataset it could have been overfitted to. Measured afterwards on
33 scaffold-split tasks, it is not merely as good as the full 1,269 but slightly better: a median
1.17% lower error, better on 24 of 33 tasks (p = 0.001). The 647 columns it removes were a mild
liability rather than a neutral cost.

## `small` (408 columns): remove what is over-resolved

The mechanical arguments above are exhausted at 622. Going further needs evidence, and the
evidence is that much of what remains is a *parameter sweep* rather than a set of distinct
quantities: one construct evaluated at many orders, lags, bins or window sizes, where the
resolution was chosen by a library author and not by chemistry. Connectivity indices are computed
at eight path orders, information content at six orders times four normalisations, charge
autocorrelation at ten lags, surface area in five binning schemes.

Two reductions were made and each was tested by removing it and re-fitting:

1. **32 substructure flags were dropped** because they were never used by any of the 33 tasks.
   The remaining 40 were kept: dropping *all* of them costs a real 1.25% (p = 0.001), so the
   block as a whole is load-bearing even though most of its members individually are not.
2. **Eight parameter sweeps were halved.** Each halving was measured on its own, with the other
   seven sweeps still present, and none cost anything detectable.

An important negative result sits inside step 2. We expected low orders and short lags to carry
the signal, and tested it by also measuring the *opposite* half of every sweep. They performed
the same. So the justification for the cut is not "keep the informative end" — that is
unsupported — but the weaker and more honest "these sweeps are over-resolved; half of them
suffices, and it does not much matter which half".

What does matter is *coverage*. A randomly chosen 408 columns costs 1.44% (p < 0.001), while the
408 chosen to keep some of every descriptor family costs nothing (+0.30%, p = 0.711). Random
subsets occasionally strip a small family bare, and the small families are where the
non-redundant information lives.

`small` is also the cheapest set to compute, at 116.7 microseconds per molecule against 144.5 for
the full block, because whole families of descriptors are skipped rather than computed and
discarded.

## `minimal` (256 columns): spend a fixed budget as well as possible

The last set answers a different question: given a hard limit on how many columns a downstream
model can take, which 256 are best? The budget is divided between descriptor families in
proportion to each family's *effective rank* — how many independent directions its columns
actually span — rather than in proportion to how many columns it happens to have. That
distinction matters: one family has 121 columns but only 38 independent directions, so allocating
by column count would hand it a fifth of the budget to say very little.

**`minimal` is not a free cut, and the name does not say so.** Against the full block on held-out
data it is 1.47% worse overall, which is inside noise, but 2.49% worse on classification tasks
specifically, which is not. It is also *not cheaper to compute* than `small` — 122.4 against 116.7
microseconds per molecule — because cost is determined by which descriptor families must be
evaluated, and 256 columns still touch every family that 408 does. Its only advantage is fewer
columns downstream. Use it when that is worth about 2.5% on classification, and use `small`
otherwise.

## How the sets were validated

The reductions were chosen on 81 tasks and tested once on 33 others that took no part in choosing
them. The stopping rule was written down and committed before any result existed, which turned
out to matter: on the 81 selection tasks a 200-column set passed every test, and on the held-out
33 it was demonstrably worse, with a confidence interval that excluded zero. The selection tasks
were individually noisier, and a real penalty of roughly 1.3% sat inside their resolution. Without
the held-out gate, the shipped default would have been a 200-column set that costs several percent
on classification.
