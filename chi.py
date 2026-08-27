"""Chi connectivity indices and path counts, computed rather than predicted.

These were in PREDICT on a cost assumption that turned out to be wrong. Mordred's Chi family
costs 10,328 us, so the split put all 54 Chi/Kappa columns and 11 PathCount columns on the
surrogate. Measured in C++ (`cpp/bench.cpp`, 3,000 benchmark molecules), bounded path
enumeration with Chi accumulation to k<=7 costs **6.82 us/mol**.

Mordred's Chi is not expensive because paths are expensive. It is expensive because Mordred is
Python. RDKit demonstrates the same thing from the inside: `Chi0n`, `Chi1n` and `Chi0v` are
implemented in C++ and cost 0.17-0.23 us, while `Chi2n`, `Chi3n` and `Chi4n` fall back to the
Python code in `GraphDescriptors.py` and cost 35, 56 and 87 us.

Why the enumeration is affordable at all: a *walk* may revisit atoms, so walk counts are matrix
powers. A *path* may not, so there is no matrix shortcut -- but molecular graphs are sparse and
nearly tree-like, and the path count barely grows with length. Measured on the benchmark:
60.6 paths of length 3, 102.8 of length 5, 144.5 of length 7 per molecule. The combinatorial
explosion that makes path enumeration frightening in general graph theory does not happen in
drug-like chemistry.

This matters more than any other block here because **Chi is the highest-value predicted family
we have measured, at +0.126 downstream** -- the finding that first established the
degree-reduction mechanism. Moving it out of the surrogate removes the largest single source of
surrogate error from the default path.

Two delta conventions, both verified against RDKit rather than assumed:

    Chi_n:  delta = nOuterElecs - nH                       (no row correction)
    Chi_v:  delta = nOuterElecs - nH                       for Z <= 10
            delta = (nOuterElecs - nH) / (Z - nOuterElecs - 1)   for Z > 10

RDKit's own comment on the difference: "This makes a big difference after we get out of the
first row." Getting this wrong is silent -- it matches perfectly on hydrocarbons and diverges
only on heteroatoms, which is exactly how it survived my first two attempts.
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
PMAX = 7
_PT = Chem.GetPeriodicTable()

NAMES = ([f"chi{k}n" for k in range(PMAX + 1)] + [f"chi{k}v" for k in range(PMAX + 1)]
         + [f"path{k}" for k in range(1, PMAX + 1)]
         + ["path_total", "path_ratio", "chi_nv_ratio"])
NDIM = len(NAMES)


def deltas(mol, kind: str) -> np.ndarray:
    """Kier-Hall connectivity deltas. `kind` is 'n' (nVal) or 'v' (valence, row-corrected)."""
    n = mol.GetNumAtoms()
    d = np.empty(n)
    for i, a in enumerate(mol.GetAtoms()):
        z = a.GetAtomicNum()
        nv = _PT.GetNOuterElecs(z) - a.GetTotalNumHs()
        d[i] = nv / float(z - _PT.GetNOuterElecs(z) - 1) if (kind == "v" and z > 10) else nv
    return d


def _ring_closures(adj, n, lo: int = 3, hi: int = PMAX + 1):
    """Simple cycles with lo <= length <= hi, each yielded once.

    Same enumerator as `cycles.py`: start from the lowest-indexed member and restrict the
    walk to higher indices, so a cycle is found twice (once per direction) and every second
    discovery is dropped.

    A CYCLE IS NOT ITS VERTEX SET, and de-duplicating on `tuple(sorted(path))` -- which this
    did until 2026-08-26 -- is therefore wrong. In K4, the complete graph on four vertices
    (tetrahedrane), the three distinct 4-cycles a-b-c-d, a-b-d-c and a-c-b-d all have vertex
    set {a,b,c,d}, so two of the three were silently dropped. RDKit's
    FindAllPathsOfLengthN(K4, 4, useBonds=True) returns 15 four-bond paths (3 cycles + 12
    lollipops); the old key returned 13, and Chi4n on bare tetrahedrane came out 0.880911
    against RDKit's 1.103134.

    The walk is already pinned to the cycle's lowest-indexed member, so each cycle arrives
    exactly twice -- as [s, a, ..., b] and [s, b, ..., a]. Keeping `path[1] < path[-1]` keeps
    exactly one of the two in O(1), and drops the per-discovery sort and set insert with it.
    """
    on = bytearray(n)
    path: list[int] = []

    def dfs(u, start, depth):
        for v in adj[u]:
            if v == start:
                # depth >= lo guarantees path[1] and path[-1] are two DISTINCT neighbours
                # of start within the cycle, so the comparison is always decisive.
                if depth >= lo and path[1] < path[-1]:
                    yield path.copy()
            elif v > start and not on[v] and depth < hi:
                on[v] = 1
                path.append(v)
                yield from dfs(v, start, depth + 1)
                path.pop()
                on[v] = 0

    for s in range(n):
        on[s] = 1
        path.append(s)
        yield from dfs(s, s, 1)
        path.pop()
        on[s] = 0


def _walk(adj, dv, n):
    """-> (chi[0..PMAX], counts[0..PMAX]). Each path is walked from both ends, hence halving.

    Zero or negative deltas are pushed to +inf so the atom contributes 0 rather than a
    divide-by-zero; RDKit drops such atoms outright in Chi0, and they are vanishingly rare in
    drug-like input.
    """
    w = np.where(dv > 0, dv, np.inf)
    inv = 1.0 / np.sqrt(w)
    chi = [0.0] * (PMAX + 1)
    cnt = [0.0] * (PMAX + 1)
    closures = [0.0] * (PMAX + 2)
    ccnt = [0.0] * (PMAX + 2)
    on = bytearray(n)

    def dfs(u, depth, p):
        for v in adj[u]:
            if on[v]:
                continue
            q = p * inv[v]
            chi[depth] += q
            cnt[depth] += 1.0
            if depth < PMAX:
                on[v] = 1
                dfs(v, depth + 1, q)
                on[v] = 0

    for a in range(n):
        on[a] = 1
        dfs(a, 1, inv[a])
        on[a] = 0
    chi = [c * 0.5 for c in chi]     # open walks are found from both ends
    cnt = [c * 0.5 for c in cnt]

    # Ring closures count as paths, which RDKit's own docstring flags as deliberate: "the
    # current path finding code does, by design, detect rings as paths". We match RDKit rather
    # than the textbook -- the point of computing Chi ourselves is to replace the reference
    # with an identical-but-faster implementation, and a "corrected" definition would be
    # unverifiable and incomparable with the existing targets.
    #
    # The emitted shape is a *lollipop*: an optional simple tail feeding into a cycle, written
    # as a walk whose last atom repeats the attachment point. Cyclopropane gives (0,1,2,0);
    # methylcyclopropane gives (0,1,2,3,1), tail atom 0 attached at atom 1. A cycle of L atoms
    # with a tail of t atoms lands at order L+t, with the product over its L+t *distinct*
    # atoms -- verified against cyclopropane Chi3n = 0.7071^3 and cyclobutane Chi4n = 0.7071^4.
    # Only one tail is possible, because a walk is a sequence and cannot branch.
    for cyc in _ring_closures(adj, n):
        L = len(cyc)
        if L > PMAX:
            continue
        base = 1.0
        for x in cyc:
            base *= inv[x]
        chi[L] += base
        cnt[L] += 1.0
        if L >= PMAX:
            continue
        seen = bytearray(n)
        for x in cyc:
            seen[x] = 1

        def tail(u, depth, p):
            for v in adj[u]:
                if seen[v]:
                    continue
                q = p * inv[v]
                chi[depth + 1] += q
                cnt[depth + 1] += 1.0
                if depth + 1 < PMAX:
                    seen[v] = 1
                    tail(v, depth + 1, q)
                    seen[v] = 0

        # The attachment atom is counted TWICE in a lollipop but once in a bare cycle. Derived
        # from RDKit's values, not assumed: CC1CC1 Chi4n = 0.16667 needs the attachment
        # doubled, while C1CCC1 Chi4n = 0.25 is the plain 4-atom ring product. Cross-checked on
        # CCC1CC1, where ring*inv[tail]*inv[attach] + two open 5-atom paths = 0.11785 + 0.40825
        # = 0.52610, matching RDKit exactly.
        for x in cyc:
            tail(x, L, base * inv[x])
    return chi, cnt


# EXPLICIT HYDROGEN: RDKIT'S CONVENTION, INCLUDING ITS INCONSISTENCY.
#
# This module used to call RemoveHs(removeIsotopes=True) here and featurise the stripped
# molecule, so that [2H]C(C)O came out as plain ethanol. The docstring justified it by claiming
# stripping "makes our values agree with every RDKit variant simultaneously". THAT CLAIM WAS
# FALSE, and it is worth recording why, because the reasoning is seductive.
#
# RemoveHs does not merely delete the hydrogen. It ALSO increments the neighbour's hydrogen
# count -- GetTotalNumHs on the carbon of [2H]C(C)O goes 1 -> 2 -- which changes the Kier-Hall
# delta and therefore changes every chi value that atom participates in. So stripping made this
# module internally consistent and simultaneously moved it AWAY from RDKit: measured, it
# disagreed with GD.Chi2n on 468 of 468 explicit-H molecules, not on a subset.
#
# Chi is somebody else's descriptor and our number has no standing on it, so we now reproduce
# RDKit exactly, inconsistency and all. Measured against RDKit on eight explicit-H molecules:
#
#   Chi0n, Chi1n  ALL atoms / ALL bonds. An explicit H is a vertex of delta = 1, contributing
#                 1/sqrt(1) = 1 to Chi0n and 1/sqrt(delta_nbr) to Chi1n.
#   Chi0v, Chi1v  HEAVY atoms / heavy-heavy bonds only (these route through _hkDeltas, skipHs=1).
#   k >= 2        HEAVY only in BOTH variants (FindAllPathsOfLengthN, useHs=False).
#
# Hence Chi0n - Chi0v is exactly the explicit-hydrogen count, and the variants agree from k = 2
# up. CC[13CH3] gives 2.707107 for both, so the trigger is an explicit H ATOM, not an isotope
# label -- isotope information was never in chi and still is not.
#
# The path COUNTS stay on the heavy graph at every k including k = 1. They are ours, they have
# no RDKit counterpart, and one graph for one family is the consistent choice; only chi0n/chi1n
# straddle the H boundary, because only those are RDKit's to define.


def featurize(mol) -> np.ndarray:
    if mol is None or mol.GetNumAtoms() < 2:
        return np.full(NDIM, np.nan, np.float32)
    n = mol.GetNumAtoms()
    A = rdmolops.GetAdjacencyMatrix(mol)
    heavy = np.fromiter((a.GetAtomicNum() != 1 for a in mol.GetAtoms()), bool, n)
    # The walk runs on the HEAVY subgraph. Explicit H are left in the numbering as isolated
    # vertices rather than renumbered away, so an index is an index everywhere in this function
    # -- and an isolated vertex contributes nothing at k >= 1, which is exactly what is wanted.
    adj = [list(np.flatnonzero(A[i] * heavy)) if heavy[i] else [] for i in range(n)]

    dn, dv = deltas(mol, "n"), deltas(mol, "v")
    chin, cnt = _walk(adj, dn, n)
    chiv, _ = _walk(adj, dv, n)
    invn = 1.0 / np.sqrt(np.where(dn > 0, dn, np.inf))
    invv = 1.0 / np.sqrt(np.where(dv > 0, dv, np.inf))
    # k = 0 is an atom sum, not a path sum: n over every atom, v over heavy atoms only.
    chin[0] = float(invn.sum())
    chiv[0] = float(invv[heavy].sum())
    # k = 1 is a bond sum. The walk above already gave the heavy-heavy part, which is Chi1v;
    # Chi1n additionally takes every bond with a hydrogen at one end.
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not (heavy[i] and heavy[j]):
            chin[1] += float(invn[i] * invn[j])

    paths = cnt[1:]
    tot = float(sum(paths))
    feats = chin + chiv + paths
    feats += [tot, paths[-1] / max(paths[0], 1.0),
              chin[1] / chiv[1] if chiv[1] else 0.0]
    return np.asarray(feats, np.float32)


def featurize_smiles(s: str) -> np.ndarray:
    return featurize(Chem.MolFromSmiles(s))


def verify(smiles, verbose: bool = True) -> dict:
    """Exact-match check against RDKit's own Chi implementations.

    This is the gate that matters: a fast wrong descriptor is worth less than a slow right one.
    Returns per-descriptor exact-match counts and the worst absolute deviation.
    """
    from rdkit.Chem import Descriptors
    lut = dict(Descriptors._descList)
    ref = [f"Chi{k}{s}" for s in ("n", "v") for k in range(5)]
    ok = dict.fromkeys(ref, 0)
    worst = dict.fromkeys(ref, 0.0)
    n = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n += 1
        v = featurize(m)
        for name in ref:
            k, kind = int(name[3]), name[4]
            got = v[NAMES.index(f"chi{k}{kind}")]
            try:
                exp = lut[name](m)
            except Exception:
                continue
            e = abs(float(got) - float(exp))
            worst[name] = max(worst[name], e)
            ok[name] += e <= 1e-5
    if verbose:
        print(f"{n:,} molecules")
        for name in ref:
            frac = 100.0 * ok[name] / n if n else 0.0
            flag = "" if frac > 99.99 else "   <-- MISMATCH"
            print(f"  {name:8s} exact {ok[name]:7,}/{n:,} ({frac:6.2f}%)  "
                  f"worst {worst[name]:.2e}{flag}")
    return {"n": n, "exact": ok, "worst": worst}


def main() -> None:
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    smiles = list(d["smiles"])
    verify(smiles[:2000])
    print(f"\nfeaturising {len(smiles):,} benchmark molecules")
    t0 = time.time()
    R = np.stack([featurize_smiles(s) for s in smiles])
    dt = time.time() - t0
    print(f"  {R.shape} in {dt:.0f}s = {1e6 * dt / len(smiles):.0f} us/mol (C++ measured: 6.82)")
    np.savez_compressed(OUT / "bench_chi.npz", R=R, names=np.array(NAMES))
    print(f"wrote {OUT / 'bench_chi.npz'}")


if __name__ == "__main__":
    main()
