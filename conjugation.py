"""Conjugated-system topology: verified absent from both descriptor libraries.

Zero of RDKit's 217 and zero of Mordred's 1,613 descriptor names match /conjug/. The nearest
things we have are `NumAromaticRings`, `NumAromaticCarbocycles` and friends (in CORE), which
count *aromatic rings* -- a strictly narrower object than a conjugated system.

Three things a ring count cannot express, all of which this block targets:

1. **Non-aromatic conjugation.** Enones, dienes, polyenes, acrylamides, vinyl sulfones. An
   alpha,beta-unsaturated ketone is the reactive warhead of a large fraction of covalent
   inhibitors and contributes nothing to any aromatic ring count.
2. **Merging across boundaries.** A conjugated system runs through the ring-ring bond of a
   biphenyl and out of a ring into a pendant carbonyl. Ring counts see two rings and a
   substituent; the pi system is one object of 12-14 atoms.
3. **Shape of the pi system.** Anthracene and a 14-carbon linear polyene are both one
   conjugated component of 14 atoms. Their *diameters* are ~7 and 13. Nothing in the union
   distinguishes them on this axis, and it drives every optical and redox property there is.

Why ECFP cannot assemble it: connected-component extent is a global question, and ECFP is a
bag of radius-2 environments. Every interior atom of a decapentaene and of a hexatriene has
the identical radius-2 environment; only the two ends differ. The bag can tell you a polyene
is present, not how long it is.

Cost: union-find over bonds is O(n) and measured at 20.6 us/mol in pure Python, so ~2 us in
C++. The diameter comes from the distance matrix CORE already builds, so it is free.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"

_SBINS = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 16), (17, 10_000)]

NAMES = (["n_sys", "conj_atoms", "conj_frac", "sys_max", "sys_max_frac", "sys_2nd", "sys_mean"]
         + [f"sysbin{i}" for i in range(len(_SBINS))]
         + ["diam_max", "linearity", "diam_sum",
            "branch_pts", "branch_frac",
            "extra_arom", "extra_arom_frac", "extra_arom_max",
            "het_in_max", "het_frac_max", "sys_max_rings"])
NDIM = len(NAMES)


def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def featurize(mol) -> np.ndarray:
    """-> (NDIM,) float32. All-zero for a molecule with no conjugated bonds at all (an alkane),
    which is a legitimate value."""
    if mol is None or mol.GetNumAtoms() < 2:
        return np.full(NDIM, np.nan, np.float32)

    n = mol.GetNumAtoms()
    parent = list(range(n))
    ncbonds = np.zeros(n, np.int64)         # conjugated bonds per atom -> cross-conjugation
    any_conj = False
    for b in mol.GetBonds():
        if not b.GetIsConjugated():
            continue
        any_conj = True
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ncbonds[i] += 1
        ncbonds[j] += 1
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            parent[ri] = rj

    if not any_conj:
        return np.zeros(NDIM, np.float32)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        if ncbonds[i]:
            groups.setdefault(_find(parent, i), []).append(i)

    D = rdmolops.GetDistanceMatrix(mol)
    arom = np.fromiter((a.GetIsAromatic() for a in mol.GetAtoms()), bool, n)
    het = np.fromiter((a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms()), bool, n)

    sizes, diams, extra, hets, rings = [], [], [], [], []
    for members in groups.values():
        idx = np.asarray(members)
        sizes.append(idx.size)
        sub = D[np.ix_(idx, idx)]
        sub = np.where(np.isfinite(sub) & (sub < 1e6), sub, 0.0)
        diams.append(float(sub.max()) if idx.size > 1 else 0.0)
        extra.append(int((~arom[idx]).sum()))
        hets.append(int(het[idx].sum()))
        rings.append(int(arom[idx].sum()))

    # ---- picking THE largest system, by a key that is a function of the system's contents ----
    #
    # Six emitted columns -- diam_max, linearity, het_in_max, het_frac_max, extra_arom_max and
    # sys_max_rings -- are read off ONE chosen pi system, so the rule that chooses it is part of
    # the feature definition. It has to be a property of the molecule. It was not.
    #
    # This used to be `np.argsort(sizes, kind="stable")[::-1]`, with a long comment pinning
    # kind="stable" because argsort's default introsort is not stable past 16 elements. That
    # comment was right about what it fixed and wrong about it being enough. Stable sorting made
    # the answer REPRODUCIBLE -- the same SMILES always gave the same number -- but not
    # CANONICAL: a reversed stable ascending sort takes the LAST maximal system in list order,
    # `groups` is filled by scanning atoms in RDKit's atom order, so "last" means "last in the
    # atom numbering". Rewrite the same molecule with the atoms permuted -- a randomized SMILES,
    # a different SD file, another toolkit's ordering -- and the tie goes to a different system
    # and the column moves. tools/notation_stability.py over 3,000 molecules x 4 random
    # rewritings: het_frac_max moved by 4.25 of its own SD on 151 of 12,000 cells, het_in_max by
    # 2.85, extra_arom_max by 2.72, linearity by 1.88, sys_max_rings by 1.73, diam_max by 0.88.
    # Which of two tied systems wins was being decided by atom numbering, which is not a
    # property of the molecule.
    #
    # The tie is now broken on invariants of the system itself. Each key is a count or an
    # extremum over the system's own atoms and bond-graph distances, so permuting the atom
    # numbering permutes the members of a group and leaves every key untouched. Largest wins on
    # each in turn:
    #
    #   1. size             -- the definition of "largest"; the rest is only a tie-break.
    #   2. diameter         -- decides diam_max, and through it linearity. First among the
    #                          tie-breaks because shape is the axis this block exists to expose.
    #   3. heteroatom count -- decides het_in_max, and with size decides het_frac_max.
    #   4. aromatic count   -- decides sys_max_rings, and with size decides extra_arom_max
    #                          (extra = size - aromatic), so once keys 1-4 are settled no further
    #                          key can change any of the six columns.
    #   5. atomic numbers, sorted descending, compared lexicographically -- the multiset of
    #                          elements. Redundant for the six columns and included anyway, so
    #                          the order is total on chemically distinct systems instead of
    #                          falling back to position in the atom list.
    #
    # Two systems that tie on all five have the same size, the same diameter, and the same bag of
    # elements with the same aromatic/heteroatom split; all six columns are then identical
    # whichever we pick, so the residual tie is immaterial. The old rule could not say that.
    #
    # This must stay byte-identical in behavior to conjugation() in src/hume_core/hume_blocks.h,
    # which is the code that actually ships. cpp/find_mismatch.py compares the two on linearity.
    members = list(groups.values())
    key = [(sizes[g], diams[g], hets[g], rings[g],
            sorted((int(mol.GetAtomWithIdx(int(a)).GetAtomicNum()) for a in members[g]),
                   reverse=True))
           for g in range(len(members))]
    gmax = max(range(len(members)), key=lambda g: key[g])
    sizes = np.asarray(sizes, np.float64)
    smax = sizes[gmax]
    # sys_2nd is the second largest size WITH MULTIPLICITY, so two systems tied at the maximum
    # make it equal to sys_max. Dropping only the ONE group the tie-break chose and taking the
    # max of the rest reproduces that, and unlike gmax it does not depend on which of a tied
    # pair won.
    s2nd = max([sizes[g] for g in range(len(members)) if g != gmax], default=0.0) \
        if sizes.size > 1 else 0.0
    tot = float(sizes.sum())
    dmax = diams[gmax]

    hist = [float(sum(1 for s in sizes if lo <= s <= hi)) for lo, hi in _SBINS]

    feats = [float(len(sizes)), tot, tot / n, smax, smax / n, s2nd, float(sizes.mean())]
    feats += hist
    # Linearity: diameter over size. A polyene approaches 1, a fused aromatic sits near 0.5,
    # a big cross-conjugated dendritic system goes lower still.
    feats += [dmax, dmax / max(smax - 1.0, 1.0), float(sum(diams))]
    nbranch = int((ncbonds >= 3).sum())
    feats += [float(nbranch), nbranch / max(tot, 1.0)]
    feats += [float(sum(extra)), sum(extra) / max(tot, 1.0), float(extra[gmax])]
    feats += [float(hets[gmax]), hets[gmax] / max(smax, 1.0), float(rings[gmax])]
    return np.asarray(feats, np.float32)


def featurize_smiles(s: str) -> np.ndarray:
    return featurize(Chem.MolFromSmiles(s))


def _selftest() -> None:
    f = featurize_smiles
    i_n, i_max, i_lin = NAMES.index("n_sys"), NAMES.index("sys_max"), NAMES.index("linearity")
    i_ex = NAMES.index("extra_arom")

    assert not f("CCCCCCCC").any(), "alkane must be identically zero"

    # The headline case: same conjugated size, different shape.
    ant, pol = f("c1ccc2cc3ccccc3cc2c1"), f("C=CC=CC=CC=CC=CC=CC=C")
    assert ant[i_max] == pol[i_max] == 14, f"sizes {ant[i_max]} {pol[i_max]}"
    assert pol[i_lin] > ant[i_lin] * 1.5, f"linearity failed: {pol[i_lin]} vs {ant[i_lin]}"

    # Non-aromatic conjugation is invisible to aromatic ring counts but not here.
    enone = f("CC(=O)C=CC")
    assert enone[i_n] == 1 and enone[i_ex] > 0, "enone not captured"

    # Biphenyl merges into one system across the ring-ring bond; two separate rings do not.
    bip, two = f("c1ccccc1-c1ccccc1"), f("c1ccccc1.c1ccccc1")
    assert bip[i_n] == 1 and two[i_n] == 2, f"merge failed: {bip[i_n]} {two[i_n]}"

    print(f"selftest ok | {NDIM} features | anthracene linearity {ant[i_lin]:.2f} vs "
          f"polyene {pol[i_lin]:.2f} (both size {ant[i_max]:.0f}) | biphenyl {bip[i_n]:.0f} system")


def main() -> None:
    _selftest()
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    smiles = list(d["smiles"])
    print(f"featurising {len(smiles):,} benchmark molecules")
    t0 = time.time()
    R = np.stack([featurize_smiles(s) for s in smiles])
    dt = time.time() - t0
    print(f"  {R.shape} in {dt:.0f}s = {1e6 * dt / len(smiles):.0f} us/mol")
    np.savez_compressed(OUT / "bench_conjugation.npz", R=R, names=np.array(NAMES))
    print(f"wrote {OUT / 'bench_conjugation.npz'}")


if __name__ == "__main__":
    main()
