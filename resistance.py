"""Resistance-distance block: the path-multiplicity axis ECFP and Mordred both miss.

MiniMol v1 conditions on the graph Laplacian (`laplacian_eigvec` / `laplacian_eigval`, 8 each)
and the random-walk matrix (`rw_return_probs`, k=16). Mordred has neither: zero of its 1,613
descriptors mention Laplacian, Kirchhoff, resistance or commute time.

But the obvious way to use a new matrix is already known to fail here. All 110 spectral
*scalars* Mordred does compute -- SpMax/SpDiam/SpAD/SpMAD/VE1-3/VR1-3 over the adjacency,
Barysz, distance and detour matrices -- were killed by the |rho|>=0.99 dedupe, zero survivors.
The cover runs in ascending cost order, so every one of them had a *cheaper* non-spectral
correlate. Collapsing n x n to one number discards whatever was new about the matrix.

So this block is built around the one quantity that is orthogonal by construction:

    Delta_ij = d_ij - Omega_ij        d = shortest path, Omega = resistance distance

On a tree Omega == d exactly, so Delta is *identically zero for every acyclic molecule*. It
cannot proxy for size, weight or atom counts. It measures path multiplicity and nothing else:
how much closer two atoms become when parallel routes exist. That is ring-fusion topology,
which ECFP at radius 2 cannot see beyond a 5-atom ball, and which distance-binned
autocorrelation (419 of our 639 CORE columns) ignores because it bins on d alone.

Falsifiable consequence: this block must help on fused / polycyclic datasets and do nothing on
acyclic or monocyclic ones. If it helps uniformly it is leaking size information and should be
rejected, not celebrated.

**Cost, measured in C++ rather than projected** (`cpp/bench.cpp`, 3,000 benchmark molecules):

    resistance L+ via Cholesky (dposv), whole molecule     5.80 us/mol
    normalised Laplacian spectrum (dsyevd)                20.79 us/mol   <- CUT

Two consequences. The biconnected-component optimisation is **unnecessary** -- 5.80 us on the
whole molecule is already cheap, so the planned "invert only the ring systems" work is
dropped. And the eigendecomposition is **removed**: it cost 3.6x the resistance solve and paid
for the least defensible features in the block (Fiedler value, spectral density), in a project
where all 110 of Mordred's spectral scalars were already killed by the |rho|>=0.99 dedupe.
Blocks 77 -> 65 features. Later 65 -> 60, when the five RATSC*_c columns were removed: the
unity weight is a constant, and a constant centres to exactly zero, so those five were
identically 0.0 by algebra. RPAIR{b} already carries the uncentred version of the same thing.

**RATSC*/RPAIR* VALUES PUBLISHED BEFORE THE BIN-EDGE SNAP ARE SUPERSEDED.** Delta lands
*exactly* on a bin edge for real molecules -- confirmed in exact rational arithmetic -- so
before the snap the bin was decided by the host BLAS's last-ulp rounding rather than by the
molecule. Accelerate's own two LAPACKs disagreed with EACH OTHER on 9.00% of the 98,905-molecule
corpus, more than reference LU disagreed with either. 13,029 molecules (13.17%) have different
bin values as a result of the fix, and all three solvers now agree with each other and with
exact arithmetic. See `_snap_to_edges` below for the full argument and the tolerance.
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

# Autocorrelation weights, restated here so the block is self-contained. Pauling
# electronegativity and atomic polarizability (A^3) for the organic subset; anything else
# falls back to the carbon value rather than NaN, since a single exotic atom should not void
# a whole molecule's descriptor.
_EN = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19,
       16: 2.58, 17: 3.16, 34: 2.55, 35: 2.96, 53: 2.66}
_POL = {1: 0.667, 5: 3.03, 6: 1.76, 7: 1.10, 8: 0.802, 9: 0.557, 14: 5.38, 15: 3.63,
        16: 2.90, 17: 2.18, 34: 3.77, 35: 3.05, 53: 5.35}

# Delta bins. Benzene gives Delta = 1/6 for an adjacent pair and 3/2 for a para pair, so the
# interesting range is decades wide and the lowest bin has to be tight to separate "barely
# cyclic" from "genuinely fused".
_DBINS = [(1e-6, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]
# mass, electronegativity, polarizability, vdW volume.
#
# THE UNITY WEIGHT "c" USED TO BE HERE AND WAS REMOVED, on algebra rather than on measurement.
# The autocorrelation is CENTRED (Pc = P - P.mean(0)), and a constant column centres to exactly
# zero, so RATSC{0..4}_c was identically 0.0 for every molecule that can exist -- not merely for
# every molecule in our corpus. Five columns carrying no information.
#
# Nothing is lost by removing it, because RPAIR{b} already IS the uncentred unity
# autocorrelation: with a constant weight every pair product is 1 * 1 = 1, so the weighted sum
# degenerates to the pair count, which is exactly what np.bincount(bk, minlength=_NB) computes.
#
# NAMING HAZARD, since the letter invites it: Mordred's ATSC0c means Gasteiger CHARGE, not
# unity. Same letter, unrelated quantity. Do not "restore" a _c column by analogy with Mordred;
# if a charge-weighted resistance autocorrelation is ever wanted it is a new property, and it
# would need a real per-atom charge, not a column of ones.
_PROPS = ("m", "e", "p", "v")
_KSTEPS = (2, 3, 4, 6, 8, 12, 16)

_PT = Chem.GetPeriodicTable()
_EDGES = np.array([lo for lo, _ in _DBINS] + [np.inf])
_NB = len(_DBINS)

# ---------------------------------------------------------------- the bin-edge snap
#
# WITHOUT THIS, RATSC*/RPAIR* ARE NOT A PROPERTY OF THE MOLECULE. They are a property of whichever
# LAPACK happens to be linked, and that is not a worry, it is a measurement:
#
#   * Omega_ij is a RATIONAL function of the graph, so delta = d - Omega has an EXACT value.
#     Computed in exact rational arithmetic on 22 molecules where two solvers disagreed, every
#     single one had atom pairs whose delta equals a bin edge EXACTLY -- up to 66 such pairs on
#     one molecule -- and the pairs that flipped were always a subset of those. Highly symmetric
#     molecules put many pairs on the same edge at once, so one ulp moves them all together.
#   * np.digitize(..., right=False) puts an on-edge value in the HIGHER bin, so a solver that
#     rounds Omega up by a single ulp drops the pair into the LOWER one.
#   * Accelerate's OWN TWO LAPACKs -- same vendor, same machine -- disagreed with each other on
#     9.00% of the 98,905-molecule corpus (worst case 242 pair-counts on one molecule). That is
#     MORE than reference LU disagreed with either of them (8.77% vs the modern kernel, 5.77% vs
#     the legacy one). Against the exact rational answer all three were wrong on some molecule,
#     and on at least one molecule ALL THREE were wrong.
#
# So the pre-snap RATSC*/RPAIR* numbers were an artifact of one vendor's summation order. The
# same resistance.py on a Linux box with OpenBLAS would have produced different integers.
#
# TOLERANCE 1e-9, and the number follows from two measured facts rather than from taste. The
# disagreement between solvers on these matrices is ~1e-12 (max 9.5e-12 absolute over every
# entry of every inverse in the corpus), so 1e-9 is roughly a THOUSAND TIMES the numerical noise
# -- comfortably wide enough to catch every tie. And the edges it must not blur are 0.1 apart at
# the closest (0.1 / 0.5 / 1.0 / 2.0), so 1e-9 is EIGHT ORDERS below the narrowest real gap: no
# delta that is genuinely inside a bin can be dragged out of it. Anything between about 1e-11 and
# 1e-6 would work identically; 1e-9 sits in the middle of that plateau on a log scale.
#
# SNAP APPLIES TO BINNING ONLY. Kf, Cyclicity and the RW* columns are continuous, already agree
# across solvers to 1e-12, and have no edge to fall off -- snapping them would change a definition
# to solve a problem they do not have. DeltaMax and DeltaMean likewise keep the raw delta.
_SNAP_TOL = 1e-9
_SNAP_EDGES = _EDGES[np.isfinite(_EDGES)]


def _snap_to_edges(delta: np.ndarray) -> np.ndarray:
    """delta, with any value within _SNAP_TOL of a bin edge moved exactly ONTO that edge.

    The windows are disjoint (the closest pair of edges is 1e-6 and 0.1), so the order in which
    they are applied cannot matter. Returns a copy; the caller's delta is left alone because
    DeltaMax/DeltaMean are computed from the unsnapped value.
    """
    out = delta.copy()
    for e in _SNAP_EDGES:
        out = np.where(np.abs(out - e) <= _SNAP_TOL, e, out)
    return out

# Property lookup tables indexed by atomic number. Building these once turns the per-atom
# work into a single fancy-index instead of four Python calls per atom -- the sort of thing
# the C++ core does everywhere, previewed here because the glue dominated the maths.
_ZMAX = 119
_T = np.zeros((len(_PROPS), _ZMAX), np.float64)
for _z in range(1, _ZMAX):
    _T[0, _z] = _PT.GetAtomicWeight(_z)
    _T[1, _z] = _EN.get(_z, _EN[6])
    _T[2, _z] = _POL.get(_z, _POL[6])
    _T[3, _z] = 4.0 / 3.0 * np.pi * _PT.GetRvdw(_z) ** 3


def _atom_props(mol) -> np.ndarray:
    """(n, 4) atom property matrix, columns ordered as _PROPS."""
    z = np.fromiter((a.GetAtomicNum() for a in mol.GetAtoms()), np.int64, mol.GetNumAtoms())
    return _T[:, np.clip(z, 0, _ZMAX - 1)].T.copy()


def _omega(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resistance distance and a within-component mask, computed per connected component.

    For a connected component, L+ = (L + J/n)^-1 - J/n exactly -- one dense solve rather than
    an eigendecomposition. Pairs in different components get Omega = 0 and mask = False; the
    shortest path between them is undefined anyway (RDKit reports 1e8), so Delta is not
    meaningful there and must not be silently binned as a large value.
    """
    n = A.shape[0]
    Om = np.zeros((n, n), np.float64)
    mask = np.zeros((n, n), bool)
    _, lbl = _components(A)
    for c in np.unique(lbl):
        idx = np.flatnonzero(lbl == c)
        k = idx.size
        if k < 2:
            continue
        Ac = A[np.ix_(idx, idx)]
        L = np.diag(Ac.sum(1)) - Ac
        Lp = np.linalg.inv(L + 1.0 / k) - 1.0 / k
        d = np.diag(Lp)
        Om[np.ix_(idx, idx)] = d[:, None] + d[None, :] - 2.0 * Lp
        mask[np.ix_(idx, idx)] = True
    return Om, mask


def _components(A: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components by BFS. n <= ~60, so an explicit loop beats pulling in scipy."""
    n = A.shape[0]
    lbl = np.full(n, -1, np.int64)
    c = 0
    for s in range(n):
        if lbl[s] >= 0:
            continue
        stack, lbl[s] = [s], c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if lbl[v] < 0:
                    lbl[v] = c
                    stack.append(v)
        c += 1
    return c, lbl


def _names() -> list[str]:
    out = []
    for b in range(len(_DBINS)):
        out += [f"RATSC{b}_{p}" for p in _PROPS] + [f"RPAIR{b}"]
    out += ["Kf", "Kf_n", "Kf_norm", "Cyclicity", "Cyclicity_n", "DeltaMax", "DeltaMean"]
    out += [f"RW{k}_{s}" for k in _KSTEPS for s in ("mean", "std", "max", "q90")]
    return out


NAMES = _names()
NDIM = len(NAMES)


def featurize(mol) -> np.ndarray:
    """-> (NDIM,) float32. All-NaN if the molecule is unusable; all-zero blocks are legitimate
    (an acyclic molecule genuinely has no path multiplicity)."""
    if mol is None or mol.GetNumAtoms() < 2:
        return np.full(NDIM, np.nan, np.float32)

    n = mol.GetNumAtoms()
    A = rdmolops.GetAdjacencyMatrix(mol).astype(np.float64)
    deg = A.sum(1)
    if not deg.any():
        return np.full(NDIM, np.nan, np.float32)
    D = rdmolops.GetDistanceMatrix(mol)
    Om, mask = _omega(A)

    iu = np.triu_indices(n, 1)
    pm = mask[iu]
    delta = np.zeros(iu[0].size)
    delta[pm] = D[iu][pm] - Om[iu][pm]
    delta = np.clip(delta, 0.0, None)       # numerical noise only; Omega <= d is a theorem

    # --- resistance-binned autocorrelation ------------------------------------------------
    # Centred properties, so a bin does not simply restate "how many atom pairs are in rings"
    # -- that count is emitted separately as RPAIR.
    P = _atom_props(mol)
    Pc = P - P.mean(0)
    prod = Pc[iu[0]] * Pc[iu[1]]            # (n_pairs, 5)

    # One digitize + bincount per property beats one masked pass per bin: the pair list is
    # scanned once instead of _NB times.
    # SNAP BEFORE DIGITISING, so the value np.digitize sees is exactly the edge and its
    # `right=False` rule (on-edge -> HIGHER bin) resolves the tie the same way for every solver
    # on every platform. See _snap_to_edges. delta itself is untouched: DeltaMax and DeltaMean
    # below still use the raw value.
    b = np.digitize(_snap_to_edges(delta), _EDGES) - 1
    keep = pm & (b >= 0) & (b < _NB)
    bk, pk = b[keep], prod[keep]
    feats = []
    acc = np.empty((_NB, len(_PROPS)))
    for j in range(len(_PROPS)):
        acc[:, j] = np.bincount(bk, weights=pk[:, j], minlength=_NB)
    cnt = np.bincount(bk, minlength=_NB).astype(np.float64)
    for i in range(_NB):
        feats.extend(acc[i])
        feats.append(cnt[i])

    # --- global resistance scalars --------------------------------------------------------
    om_p, d_p = Om[iu][pm], D[iu][pm]
    kf = float(om_p.sum())
    cyc = float((d_p - om_p).sum())         # Wiener - Kirchhoff: total path multiplicity
    feats += [kf, kf / n, kf / max(n * (n - 1) / 2, 1),
              cyc, cyc / n, float(delta.max(initial=0.0)),
              float(delta[pm].mean()) if pm.any() else 0.0]

    # --- random-walk return probabilities -------------------------------------------------
    # diag((D^-1 A)^k) == diag(S^k) for S = D^-1/2 A D^-1/2, and S is symmetric, so use S.
    # Mordred keeps only the trace of these (SRW05, SRW07, TSRW10); the per-atom distribution
    # is what MiniMol feeds its network and what a sum destroys.
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    S = A * dinv[:, None] * dinv[None, :]
    Pk = np.eye(n)
    prev = 0
    for k in _KSTEPS:
        for _ in range(k - prev):
            Pk = Pk @ S
        prev = k
        d_k = np.sort(np.diag(Pk))          # sort once; np.quantile on a 20-element array
        feats += [d_k.mean(), d_k.std(), d_k[-1],   # costs more than the matmul that made it
                  float(d_k[min(int(0.9 * n), n - 1)])]

    # The normalised Laplacian spectrum (Fiedler value, spectral density) used to live here
    # and was cut. See the module docstring: 20.79 us of measured C++ for the least defensible
    # features in the block.

    return np.asarray(feats, np.float32)


def featurize_smiles(smiles: str) -> np.ndarray:
    return featurize(Chem.MolFromSmiles(smiles))


def _selftest() -> None:
    """Delta == 0 on every acyclic molecule is the property the whole block rests on."""
    acyclic = ["CCCC", "CC(C)CC(=O)O", "CCOCCN", "CC(C)(C)CCCCO"]
    cyclic = ["c1ccccc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CC2CCC1CC2"]
    b0 = NAMES.index("Cyclicity")
    for s in acyclic:
        v = featurize_smiles(s)
        assert abs(v[b0]) < 1e-4, f"acyclic {s} has cyclicity {v[b0]}"
    for s in cyclic:
        v = featurize_smiles(s)
        assert v[b0] > 1e-3, f"cyclic {s} has cyclicity {v[b0]}"
    # Naphthalene (fused) must carry more multiplicity than two separate benzenes.
    fused = featurize_smiles("c1ccc2ccccc2c1")[b0]
    sep = featurize_smiles("c1ccccc1.c1ccccc1")[b0]
    assert fused > sep, f"fused {fused} <= separate {sep}"
    print(f"selftest ok | {NDIM} features | naphthalene cyc {fused:.2f} vs 2x benzene {sep:.2f}")


def main() -> None:
    _selftest()
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    smiles = list(d["smiles"])
    print(f"featurising {len(smiles):,} benchmark molecules")
    t0 = time.time()
    R = np.stack([featurize_smiles(s) for s in smiles])
    dt = time.time() - t0
    bad = ~np.isfinite(R).all(1)
    acyc = (R[:, NAMES.index("Cyclicity")] == 0).sum()
    print(f"  {R.shape} in {dt:.0f}s = {1e6 * dt / len(smiles):.0f} us/mol")
    print(f"  unusable {bad.sum()} | acyclic (block identically zero) {acyc} "
          f"({100 * acyc / len(smiles):.1f}%)")
    np.savez_compressed(OUT / "bench_resistance.npz", R=R, names=np.array(NAMES))
    print(f"wrote {OUT / 'bench_resistance.npz'}")


if __name__ == "__main__":
    main()
