"""Stereochemistry block: the axis the descriptor union is totally blind to.

Measured, not assumed: across four enantiomer pairs (butane-2,3-diol, alanine, ibuprofen,
thalidomide), **zero** of RDKit's 217 and **zero** of Mordred's 1,613 descriptors change.
Every one of them is a 2D graph invariant. ECFP with `includeChirality=True` does see it --
4 to 10 bits move -- so this is not missing information, it is thin information: a handful of
bits inside a ~60-on-bit sparse vector, with no ordered axis for a tree to split on. Same
class of argument as Chi/PathCount (+0.126), and it should be judged by the same standard.

Two parity variables, which behave differently under reflection:

    s_i in {-1, 0, +1}   atom CIP parity     R = +1, S = -1, none/unspecified = 0
    t_b in {-1, 0, +1}   bond geometry       E = +1, Z = -1, none/unspecified = 0

Mirroring flips every s_i and leaves every t_b alone (a cis double bond reflects to a cis
double bond). So the features split cleanly by order:

    odd  in s   Sum s_i           flips sign -> separates enantiomers (absolute configuration)
    even in s   Sum_{d=k} s_i s_j invariant  -> separates diastereomers (relative configuration)
    any  in t   Sum t_b, Sum t t  invariant  -> achiral, safe to mix with either

Each sub-block is identically zero when its stereo type is absent, so an achiral dataset
cannot be helped or hurt by it -- the same orthogonality-by-construction that `resistance.py`
gets from Delta = 0 on trees.

**Known weakness, stated up front.** CIP R/S is a priority convention, not a physical
quantity: L-cysteine is R and L-serine is S despite identical spatial arrangement, because a
sulfur outranks an oxygen. So `Sum s_i` is a categorical separator with a convention-dependent
sign, not a smooth chemical axis. The even-order terms are on firmer ground -- a product
s_i*s_j at fixed topological distance reads as "same or opposite relative configuration" --
and within a congeneric series, which is what MoleculeACE is, priorities are usually
consistent across the series so the convention is at least locally coherent. Do not expect
this block to transfer across chemotypes the way `Kf` does.

Coverage in the benchmark: 30.9% of the 56,197 molecules carry >=1 specified stereocenter,
ranging 0.0% (CHEMBL4203_Ki, ESOL) to 91.8% (CHEMBL4616_EC50); 21/34 datasets exceed 20%.
Defined E/Z is only 4.5% overall and never above 11.6% in any one dataset, so the t-features
are along for the ride rather than independently powered.

Falsifiable prediction: gain must track the per-dataset stereo fraction. Flat gain across
high- and zero-stereo datasets means the block is proxying something else and should be cut.
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

_SK = (1, 2, 3, 4, 5, 6)        # topological lags for atom-atom parity autocorrelation
_TK = (1, 2, 3, 4)              # lags for bond-bond and atom-bond terms
_E = {Chem.BondStereo.STEREOE: 1.0, Chem.BondStereo.STEREOTRANS: 1.0,
      Chem.BondStereo.STEREOZ: -1.0, Chem.BondStereo.STEREOCIS: -1.0}


def _names() -> list[str]:
    out = ["S_sum", "S_absum", "S_sum_norm", "S_central", "S_mass"]
    out += [f"SATS{k}" for k in _SK] + ["SATS_far"]
    out += ["T_sum", "T_absum", "n_EZ_any"]
    out += [f"TATS{k}" for k in _TK]
    out += [f"XATS{k}" for k in _TK]
    return out


NAMES = _names()
NDIM = len(NAMES)
_ZERO = np.zeros(NDIM, np.float32)


def featurize(mol) -> np.ndarray:
    """-> (NDIM,) float32. All-zero for a molecule with no defined stereochemistry, which is
    a legitimate value and not a missing one."""
    if mol is None or mol.GetNumAtoms() < 2:
        return np.full(NDIM, np.nan, np.float32)

    # MolFromSmiles already assigns CIP codes, so this is a 2.3 us safety net for molecules
    # built some other way, not a cost centre -- verified identical output either way. The
    # legacy assigner is used rather than rdCIPLabeler; swap that in if the block earns its
    # place and the exotic-priority cases turn out to matter.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    n = mol.GetNumAtoms()
    s = np.zeros(n)
    for a in mol.GetAtoms():
        if a.HasProp("_CIPCode"):
            s[a.GetIdx()] = 1.0 if a.GetProp("_CIPCode") == "R" else -1.0

    tb, t = [], []
    for b in mol.GetBonds():
        v = _E.get(b.GetStereo())
        if v is not None:
            tb.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            t.append(v)
    t = np.asarray(t)

    if not s.any() and not t.size:
        return _ZERO.copy()

    D = rdmolops.GetDistanceMatrix(mol)
    feats = []

    # --- odd order in s: absolute configuration ------------------------------------------
    idx = np.flatnonzero(s)
    ns = idx.size
    if ns:
        # Centrality weight: an R centre buried in the core is not the same object as one on
        # a terminal substituent, and Sum s_i alone cannot tell them apart.
        Dm = np.where(np.isfinite(D) & (D < 1e6), D, 0.0)
        cen = 1.0 / (1.0 + Dm.mean(1))
        mass = np.fromiter((a.GetMass() for a in mol.GetAtoms()), np.float64, n)
        feats += [s.sum(), np.abs(s).sum(), s.sum() / ns,
                  float(s @ cen), float(s @ (mass / mass.mean()))]
    else:
        feats += [0.0] * 5

    # --- even order in s: relative configuration -----------------------------------------
    # Sum_{d(i,j)=k} s_i s_j over stereocentre pairs only; there are rarely more than a
    # handful, so the explicit pair loop is cheaper than masking the full n x n matrix.
    sats = dict.fromkeys(_SK, 0.0)
    far = 0.0
    for ai in range(ns):
        for aj in range(ai + 1, ns):
            i, j = idx[ai], idx[aj]
            d = D[i, j]
            if not np.isfinite(d) or d > 1e6:
                continue
            p = s[i] * s[j]
            if d in sats:
                sats[d] += p
            elif d > max(_SK):
                far += p
    feats += [sats[k] for k in _SK] + [far]

    # --- bond geometry: mirror-invariant, so it mixes safely with either block above ------
    feats += [float(t.sum()) if t.size else 0.0, float(np.abs(t).sum()) if t.size else 0.0,
              float(sum(1 for b in mol.GetBonds()
                        if b.GetStereo() != Chem.BondStereo.STEREONONE))]

    tats = dict.fromkeys(_TK, 0.0)
    for bi in range(len(tb)):
        for bj in range(bi + 1, len(tb)):
            d = _bdist(D, tb[bi], tb[bj])
            if d in tats:
                tats[d] += t[bi] * t[bj]
    feats += [tats[k] for k in _TK]

    # --- cross terms: stereocentre next to a defined double bond -------------------------
    xats = dict.fromkeys(_TK, 0.0)
    for i in idx:
        for bj in range(len(tb)):
            d = min(D[i, tb[bj][0]], D[i, tb[bj][1]])
            if d in xats:
                xats[d] += s[i] * t[bj]
    feats += [xats[k] for k in _TK]

    return np.asarray(feats, np.float32)


def _bdist(D, b1, b2) -> float:
    """Topological distance between two bonds = min distance over their endpoints."""
    return min(D[b1[0], b2[0]], D[b1[0], b2[1]], D[b1[1], b2[0]], D[b1[1], b2[1]])


def featurize_smiles(s: str) -> np.ndarray:
    return featurize(Chem.MolFromSmiles(s))


def _selftest() -> None:
    f = featurize_smiles
    i_sum, i_s1 = NAMES.index("S_sum"), NAMES.index("SATS1")
    i_t = NAMES.index("T_sum")

    # Achiral -> identically zero.
    for s in ("CCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"):
        assert not f(s).any(), f"achiral {s} is nonzero"

    # Enantiomers -> odd order flips sign, even order agrees.
    rr, ss = f("C[C@@H](O)[C@H](O)C"), f("C[C@H](O)[C@@H](O)C")
    assert np.isclose(rr[i_sum], -ss[i_sum]) and rr[i_sum] != 0, "odd order does not flip"
    assert np.isclose(rr[i_s1], ss[i_s1]), "even order is not mirror invariant"
    assert not np.allclose(rr, ss), "enantiomers must be separable somewhere"

    # Diastereomers -> even order differs. (R,R) and the meso (R,S) form.
    meso = f("C[C@H](O)[C@H](O)C")
    assert not np.isclose(rr[i_s1], meso[i_s1]), "even order cannot see diastereomers"

    # E/Z -> mirror invariant, opposite sign to each other.
    e, z = f("C/C=C/C(=O)O"), f(r"C/C=C\C(=O)O")
    assert np.isclose(e[i_t], -z[i_t]) and e[i_t] != 0, "E/Z sign broken"

    print(f"selftest ok | {NDIM} features | S_sum R,R={rr[i_sum]:+.0f} S,S={ss[i_sum]:+.0f} "
          f"meso={meso[i_sum]:+.0f} | SATS1 R,R={rr[i_s1]:+.0f} meso={meso[i_s1]:+.0f}")


def main() -> None:
    _selftest()
    d = np.load(OUT / "bench.npz", allow_pickle=True)
    smiles = list(d["smiles"])
    print(f"featurising {len(smiles):,} benchmark molecules")
    t0 = time.time()
    R = np.stack([featurize_smiles(s) for s in smiles])
    dt = time.time() - t0
    nz = (np.abs(R).sum(1) > 0).sum()
    print(f"  {R.shape} in {dt:.0f}s = {1e6 * dt / len(smiles):.0f} us/mol")
    print(f"  nonzero (has some stereo) {nz:,} ({100 * nz / len(smiles):.1f}%)")
    np.savez_compressed(OUT / "bench_stereo.npz", R=R, names=np.array(NAMES))
    print(f"wrote {OUT / 'bench_stereo.npz'}")


if __name__ == "__main__":
    main()
