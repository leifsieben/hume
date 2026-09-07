"""Exact cycle counts: the part of graph structure that colour refinement provably cannot see.

WL-1 distinguishes exactly the graphs that homomorphism counts *from trees* distinguish
(Dell/Grohe/Rattan). ECFP is WL-1 refinement, so every pattern containing a cycle is outside
what it can express. That makes cycle counts the principled place to look, not merely a
plausible one.

This is not RingCount. `RingCount` and its 49 CORE columns report the SSSR -- a *basis* of
n_bonds - n_atoms + n_components rings. The number of actual cycles is larger and carries
different information: naphthalene has two 6-cycles in its SSSR but three cycles in total
(the two rings and the 10-membered perimeter); a cubane skeleton has 5 SSSR rings and far
more real cycles. Cycle *redundancy* -- total cycles over SSSR size -- is a direct measure of
cage-likeness and fusion that no ring basis reports.

Two routes, chosen per k by which is exact and cheaper:

* **k = 3, 4, 5 by inclusion-exclusion on traces of A^k**, which we already build for
  WalkCount. Exact, closed-form, essentially free:

      C3 = tr(A^3) / 6
      C4 = [tr(A^4) - 2m - 2*sum_i d_i(d_i-1)] / 8
      C5 = [tr(A^5) - 5*tr(A^3) - 5*sum_i (d_i-2)*(A^3)_ii] / 10

  All three are verified against hand-checkable graphs in `_selftest`.

* **k = 6, 7, 8 by bounded DFS enumeration.** The closed forms past k=5 are long and
  error-prone, and molecular graphs are tiny and sparse (n <= 60, max degree 4), so direct
  enumeration is both exact and affordable. Each cycle is enumerated from its lowest-indexed
  vertex in two directions, hence the halving.

Cut at k=8 on chemical grounds rather than computational ones. Rings of size 3-8 cover
essentially all drug-like chemistry; the longer cycles in a fused system are perimeters
(naphthalene's 10-ring) which are not chemically ring-like, and macrocycles are better served
by RingCount. Raising `LMAX` is a one-line change if a macrocycle programme needs it.

The per-atom participation counts are the strictly-beyond-WL part and come free from the
enumeration -- how many k-cycles each atom sits in, pooled rather than summed, for the same
reason RWSE beats trace(A^k).
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

LMAX = 8
_KS = tuple(range(3, LMAX + 1))

NAMES = ([f"C{k}" for k in _KS]
         + ["C_total", "C_sssr", "C_redundancy", "C_maxlen", "C_minlen"]
         + [f"pa{k}_max" for k in _KS] + [f"pa{k}_mean" for k in _KS]
         + ["pa_all_max", "pa_all_mean", "pa_all_std", "frac_in_cycle", "n_multi_cycle"]
         + ["C5_het", "C6_het", "C5_arom", "C6_arom", "C6_carbo"])
NDIM = len(NAMES)


def _closed_form(A: np.ndarray) -> tuple[float, float, float]:
    """C3, C4, C5 from traces of A^k. Exact for any simple graph."""
    d = A.sum(1)
    m = d.sum() / 2.0
    A2 = A @ A
    A3 = A2 @ A
    t3 = np.trace(A3)
    t4 = float((A2 * A2).sum())                 # tr(A^4) = sum of squares of A^2
    t5 = float((A2 * A3).sum())                 # tr(A^5) = <A^2, A^3>
    c3 = t3 / 6.0
    c4 = (t4 - 2.0 * m - 2.0 * float((d * (d - 1.0)).sum())) / 8.0
    c5 = (t5 - 5.0 * t3 - 5.0 * float(((d - 2.0) * np.diag(A3)).sum())) / 10.0
    return c3, c4, c5


def _enumerate(adj: list[list[int]], n: int, lo: int, hi: int):
    """All simple cycles with lo <= length <= hi.

    Yields vertex lists. Each cycle is found from its lowest-indexed member in both
    directions, so callers must halve. Restricting the walk to vertices above `start` is what
    makes that invariant hold and keeps the search from revisiting the same cycle n times.
    """
    onpath = bytearray(n)
    path: list[int] = []

    def dfs(u: int, start: int, depth: int):
        for v in adj[u]:
            if v == start:
                if depth >= lo:
                    yield path.copy()
            elif v > start and not onpath[v] and depth < hi:
                onpath[v] = 1
                path.append(v)
                yield from dfs(v, start, depth + 1)
                path.pop()
                onpath[v] = 0

    for s in range(n):
        onpath[s] = 1
        path.append(s)
        yield from dfs(s, s, 1)
        path.pop()
        onpath[s] = 0


def featurize(mol) -> np.ndarray:
    """-> (NDIM,) float32. All-zero for an acyclic molecule, a legitimate value."""
    if mol is None or mol.GetNumAtoms() < 3:
        return np.full(NDIM, np.nan, np.float32)

    n = mol.GetNumAtoms()
    A = rdmolops.GetAdjacencyMatrix(mol).astype(np.float64)
    adj = [list(np.flatnonzero(A[i])) for i in range(n)]

    c3, c4, c5 = _closed_form(A)
    counts = {3: c3, 4: c4, 5: c5}

    het = np.fromiter((a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms()), bool, n)
    arom = np.fromiter((a.GetIsAromatic() for a in mol.GetAtoms()), bool, n)

    # Per-atom participation for every k, and typing for the 5- and 6-rings that dominate
    # medicinal chemistry. k=3,4,5 are re-enumerated here: the closed forms give totals only,
    # and the per-atom distribution is the part WL cannot reach.
    per = {k: np.zeros(n) for k in _KS}
    typed = dict.fromkeys(("C5_het", "C6_het", "C5_arom", "C6_arom", "C6_carbo"), 0.0)
    long_counts = {k: 0.0 for k in _KS if k > 5}
    for cyc in _enumerate(adj, n, 3, LMAX):
        k = len(cyc)
        idx = np.asarray(cyc)
        per[k][idx] += 1
        if k > 5:
            long_counts[k] += 1
        if k in (5, 6):
            h, a = het[idx].any(), arom[idx].all()
            if h:
                typed[f"C{k}_het"] += 1
            if a:
                typed[f"C{k}_arom"] += 1
            if k == 6 and not h:
                typed["C6_carbo"] += 1
    for k in per:
        per[k] /= 2.0
    for k in long_counts:
        counts[k] = long_counts[k] / 2.0
    for t in typed:
        typed[t] /= 2.0

    total = sum(counts.values())
    nz = [k for k in _KS if counts[k] > 0.5]
    sssr = float(len(Chem.GetSymmSSSR(mol)))
    pa_all = sum(per.values())

    feats = [counts[k] for k in _KS]
    feats += [total, sssr, total / sssr if sssr else 0.0,
              float(max(nz)) if nz else 0.0, float(min(nz)) if nz else 0.0]
    feats += [per[k].max() for k in _KS] + [per[k].mean() for k in _KS]
    feats += [pa_all.max(), pa_all.mean(), pa_all.std(),
              float((pa_all > 0).sum()) / n, float((pa_all > 1).sum())]
    feats += [typed["C5_het"], typed["C6_het"], typed["C5_arom"],
              typed["C6_arom"], typed["C6_carbo"]]
    return np.asarray(feats, np.float32)


def featurize_smiles(s: str) -> np.ndarray:
    return featurize(Chem.MolFromSmiles(s))


def _selftest() -> None:
    f = featurize_smiles
    i = {k: NAMES.index(f"C{k}") for k in _KS}
    i_tot, i_red = NAMES.index("C_total"), NAMES.index("C_redundancy")

    assert not f("CCCCCCCC").any(), "alkane must be identically zero"

    # Closed forms against hand-checkable graphs.
    A_k4 = np.ones((4, 4)) - np.eye(4)
    c3, c4, c5 = _closed_form(A_k4)
    assert (round(c3), round(c4), round(c5)) == (4, 3, 0), f"K4 gave {c3},{c4},{c5}"
    A_c5 = np.zeros((5, 5))
    for a in range(5):
        A_c5[a, (a + 1) % 5] = A_c5[(a + 1) % 5, a] = 1
    assert round(_closed_form(A_c5)[2]) == 1, "C5 ring miscounted"

    assert round(f("c1ccccc1")[i[6]]) == 1, "benzene"
    assert round(f("C1CC1")[i[3]]) == 1, "cyclopropane"
    assert round(f("C1CCC1")[i[4]]) == 1, "cyclobutane"

    # Naphthalene: two 6-rings in the SSSR, and the 10-membered perimeter is correctly
    # outside LMAX=8, so C_total is 2 and redundancy is exactly 1.
    nap = f("c1ccc2ccccc2c1")
    assert round(nap[i[6]]) == 2 and round(nap[i_tot]) == 2, f"naphthalene {nap[i[6]]}"

    # Cubane: 5 SSSR rings but many real cycles -> redundancy well above 1. This is the case
    # RingCount cannot express.
    cub = f("C12C3C4C1C1C4C3C21")
    assert cub[i_red] > 2.0, f"cubane redundancy {cub[i_red]}"
    assert round(cub[i[4]]) == 6, f"cubane should have 6 four-cycles, got {cub[i[4]]}"

    # Bicyclo[2.2.2]octane: three 6-cycles sharing two bridgeheads.
    bo = f("C1CC2CCC1CC2")
    print(f"selftest ok | {NDIM} features | naphthalene C6={nap[i[6]]:.0f} redundancy "
          f"{nap[i_red]:.2f} | cubane C4={cub[i[4]]:.0f} C6={cub[i[6]]:.0f} redundancy "
          f"{cub[i_red]:.2f} | bicyclooctane C6={bo[i[6]]:.0f}")


def main() -> None:
    _selftest()
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    smiles = list(d["smiles"])
    print(f"featurising {len(smiles):,} benchmark molecules")
    t0 = time.time()
    R = np.stack([featurize_smiles(s) for s in smiles])
    dt = time.time() - t0
    print(f"  {R.shape} in {dt:.0f}s = {1e6 * dt / len(smiles):.0f} us/mol")
    np.savez_compressed(OUT / "bench_cycles.npz", R=R, names=np.array(NAMES))
    print(f"wrote {OUT / 'bench_cycles.npz'}")


if __name__ == "__main__":
    main()
