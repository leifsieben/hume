"""Figure A without a model: does an edit move every molecule the SAME WAY?

    .venv/bin/python research/figa_consistency.py  -> results/figures/figA/consistency.json

THE METRIC, and why it needs no background corpus and no threshold.

For one edit, take each pair's difference vector d_i = v(A'_i) - v(A_i) and normalise it to a unit
vector u_i. Hold pair i out, average the other n-1 directions, and ask whether u_i projects
POSITIVELY onto that average. Report the fraction of pairs that do.

    1.0  every pair moves the same way; a linear head trained on the others transfers to this one
    0.5  directions are unrelated; nothing about pair i is predictable from the rest

0.5 IS ANALYTIC, NOT MEASURED. For isotropic random directions the leave-one-out projection is
positive exactly half the time, in any dimension -- verified at d = 384 and 2048, n = 100 and
1000, all four within 0.03 of 0.5. So the axis is calibrated without a background corpus, which
matters because no background exists that works for every arm: re-written SMILES is the right
noise floor for a language model and EXACTLY ZERO for a fingerprint, a GNN or descriptors, while
random different molecules are near-orthogonal for a continuous embedding, so every real edit
looks like nothing beside them. There is no quantity that is simultaneously non-zero for a
fingerprint and non-saturating for a transformer. Asking about DIRECTION removes the need for one.

WHY NOT A PER-DIMENSION THRESHOLD ("resolved if some dimension moves beyond k sigma"). It is a
maximum over d independent chances, so it is biased by width before chemistry enters: on pure
noise at k=3, a 2048-d arm is called resolving 99.6% of the time and a 384-d one 64%. Correcting
for d reintroduces the free parameter the whole design is trying to avoid. Fingerprint bits are
also Bernoulli with p ~ 0.001-0.05, where one flip of a rare bit is a 30-sigma event.

WHY DIRECTION RATHER THAN DISTANCE. Scale is not what makes an edit learnable. A tiny but
consistent move is amplified by any linear head; a large inconsistent one cannot be. Simulated:
a move of median norm 10.0 in random directions scores 0.478, and one of norm 0.072 in a
consistent direction scores 1.000. Median cosine distance is reported alongside as the magnitude
column, because the two together say more than either alone -- but the ranking axis is direction.

WHAT THIS GIVES UP against the XGBoost AUC in figa_resolution.py, stated so the two plates are
read correctly: this tests for ONE CONSISTENT LINEAR DIRECTION. An edit resolvable by a tree
through two different valid directions -- F->Cl against Br->I inside `halogen_swap` -- scores low
here and high there. It is stricter and model-free, not uniformly better. Both files are kept.

STANDARDISATION IS A PRECONDITION, NOT A REFINEMENT. HUME's block mixes a 2,048-bit fingerprint
with unstandardised descriptors whose columns reach 1.3e5; measured, the three largest columns
carry a median 16.4% of each row's squared norm, so a raw cosine on HUME is a cosine on two or
three descriptors. Every arm is therefore z-scored per dimension over all 31,704 molecules before
anything is computed. The raw-cosine variant is computed too and written alongside, so the choice
is visible rather than buried.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIGA = ROOT / "results" / "figures" / "figA"
OUT = FIGA / "consistency.json"

#: `notation` is dropped (Leif 2026-10-27): the character n-gram control existed to prove an edit
#: was not separable from the SMILES string alone, which was a statement about the AUC harness.
#: `r3cfp` is dropped with the move to r=4 as HUME's default -- the paper compares ECFP4 against
#: ECFP8 and no longer carries the intermediate radius.
DROP = {"notation", "r3cfp"}
N_BOOT = 400
#: A cell needs this fraction of its edit's pairs present, or it is not written at all.
MIN_COVERAGE = 0.95
SEED = 0


def load():
    """-> (smiles -> row, pairs). Rows run [order | background | extra], in that order.

    `extra` holds molecules added after the first embedding run. Appending rather than
    reindexing is what lets an arm that has not been re-embedded keep its existing matrix: its
    rows still mean what they meant, it is simply short. Every scorer below checks each arm's
    height and skips the edits whose molecules are past it, rather than indexing out of bounds or
    silently scoring a different molecule.
    """
    idx = json.loads((FIGA / "smiles_index.json").read_text())
    order = list(idx["order"]) + list(idx["background"]) + list(idx.get("extra", []))
    row = {s: i for i, s in enumerate(order)}
    pj = json.loads((FIGA / "pairs.json").read_text())
    return row, pj["pairs"]


#: A pair counts as having MOVED only if its difference is this large relative to the vectors
#: themselves. Not a tuning knob -- a float32 precision floor, and the gap it sits in is five
#: orders of magnitude wide.
#:
#: WITHOUT IT THE METRIC READS ROUNDING NOISE AS CHEMISTRY. Normalising a difference to a unit
#: vector throws away its size, so a wobble in the last bits points somewhere just as firmly as a
#: real edit does. Measured on the null panels, where the molecule is provably unchanged -- all
#: 1,000 Kekule pairs re-parse to an identical canonical SMILES -- CheMeleon, Chemprop and
#: MiniMol differ by a median RELATIVE 1e-7 against 1e-1 for adding a methyl. A D-MPNN pools over
#: atoms, floating-point addition is not associative, and a re-written SMILES hands it the same
#: atoms in a different order. Counting only exact zeros as degenerate let 255 of those pairs
#: through and drew CheMeleon at 0.663 on a panel whose correct answer is 0.5.
NOISE_FLOOR = 1e-5


def unit(D, ref):
    """-> (unit rows, mask of rows that moved by more than float32 noise).

    `ref` is the pair's own scale, so the floor is relative rather than absolute: descriptor
    blocks and 384-d embeddings do not share a unit.
    """
    n = np.linalg.norm(D, axis=1)
    ok = np.isfinite(n) & (n > NOISE_FLOOR * np.maximum(ref, 1e-300))
    U = np.zeros_like(D)
    U[ok] = D[ok] / n[ok, None]
    return U, ok


def consistency(U):
    """Leave-one-out directional agreement. Null 0.5 in any dimension."""
    S = U.sum(0)
    proj = U @ S - 1.0            # u_i . (S - u_i), since u_i . u_i == 1
    return float(np.mean(proj > 0)), proj


def boot_ci(proj, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap of the MEAN of the per-pair leave-one-out indicators.

    RESAMPLING THE VECTORS AND RE-DERIVING THE MEAN DIRECTION IS BIASED, and visibly so: a
    bootstrap draw contains duplicates of pair i, those duplicates stay in the "everything else"
    it is projected onto, and a vector agrees perfectly with its own copy. ECFP on stereo_flip
    came out at 0.726 with an interval of [0.783, 0.838] -- an interval that does not contain its
    own point estimate, which is how the bias announced itself.

    The leave-one-out projection is computed ONCE on the full set, which is the statistic; the
    bootstrap then resamples those per-pair indicators, which is an ordinary bootstrap of a mean.
    """
    n = len(proj)
    if n < 3:
        return (float("nan"), float("nan"))
    ind = (proj > 0).astype(float)
    draws = ind[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))


def cos_dist(A, B):
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    ok = (na > 0) & (nb > 0) & np.isfinite(na) & np.isfinite(nb)
    d = np.full(len(A), np.nan)
    d[ok] = 1.0 - np.sum(A[ok] * B[ok], axis=1) / (na[ok] * nb[ok])
    return d


def main() -> None:
    row, pairs = load()
    arms = sorted(p.stem for p in (FIGA / "embeddings").glob("*.npz") if p.stem not in DROP)
    edits = sorted({p["edit"] for p in pairs})
    print(f"  {len(arms)} arms, {len(edits)} edits, {len(pairs)} pairs"
          f"   (dropped: {', '.join(sorted(DROP))})\n")

    by_edit = {e: [(row[p["a"]], row[p["b"]]) for p in pairs
                   if p["edit"] == e and p["a"] in row and p["b"] in row] for e in edits}
    out, skipped = {}, []
    for arm in arms:
        X = np.load(FIGA / "embeddings" / f"{arm}.npz")["X"].astype(np.float64)
        n_rows = X.shape[0]
        raw = np.where(np.isfinite(X), X, 0.0)
        mu, sd = raw.mean(0), raw.std(0)
        keep = sd > 0
        Z = (raw[:, keep] - mu[keep]) / sd[keep]
        rng = np.random.default_rng(SEED)
        out[arm] = {}
        for e in edits:
            ij = [(i, j) for i, j in by_edit[e] if i < n_rows and j < n_rows]
            # COVERAGE, NOT MERELY NON-EMPTINESS. An arm that has not been re-embedded still has
            # rows for the molecules it already knew, and for a protonation edit the NEUTRAL
            # member is an unmodified pool molecule -- so 85 of 1,000 pairs survived the bounds
            # check and were scored as though they were the panel. For ChemBERTa 82 of those 85
            # were degenerate, which put a bar reading 0.333 on the plate off THREE pairs. A cell
            # built from a tenth of its edit is not a measurement of that edit.
            avail = len(ij) / len(by_edit[e]) if by_edit[e] else 0.0
            if avail < MIN_COVERAGE:
                skipped.append((arm, e, len(by_edit[e]) - len(ij)))
                continue
            if not ij:
                continue
            a = np.fromiter((i for i, _ in ij), int)
            b = np.fromiter((j for _, j in ij), int)
            ref = np.maximum(np.linalg.norm(Z[a], axis=1), np.linalg.norm(Z[b], axis=1))
            U, ok = unit(Z[b] - Z[a], ref)
            n_deg = int((~ok).sum())
            if ok.sum() < 3:
                out[arm][e] = {"consistency": 0.5, "ci_lo": float("nan"), "ci_hi": float("nan"),
                               "median_cos_z": 0.0, "median_cos_raw": 0.0,
                               "n_pairs": len(ij), "degenerate_pairs": n_deg,
                               "degenerate_frac": n_deg / len(ij),
                               "note": "all pairs degenerate; 0.5 is forced, not measured"}
                continue
            Uo = U[ok]
            c_moved, proj = consistency(Uo)
            # DILUTE BY THE PAIRS THAT DID NOT MOVE AT ALL. A direction is undefined for a zero
            # vector, so degenerate pairs cannot enter the mean -- but dropping them silently
            # rescales the panel to whatever minority did move, and on a null panel that minority
            # is the whole artefact. RDKit+Mordred move on 158 of 1,000 re-written SMILES and
            # agree on 67% of those; reporting 0.671 says the descriptors follow notation two
            # times in three, when five molecules in six do not respond at all.
            #
            # A pair that does not move carries no information about direction, which is exactly
            # what 0.5 means here, so it contributes 0.5. This also restores the behaviour of the
            # AUC plate, where a degenerate pair is two identical points and forces 0.5 by
            # construction. ChemBERTa on stereo_flip moves 13 pairs in 1,000: 0.692 on those
            # becomes 0.502 overall, which is the true statement -- it does not see stereochemistry.
            n_all = len(ij)
            frac_moved = len(Uo) / n_all
            c = 0.5 * (1.0 - frac_moved) + c_moved * frac_moved
            lo_m, hi_m = boot_ci(proj, rng)
            lo = 0.5 * (1.0 - frac_moved) + lo_m * frac_moved
            hi = 0.5 * (1.0 - frac_moved) + hi_m * frac_moved
            out[arm][e] = {
                "consistency": c, "ci_lo": lo, "ci_hi": hi,
                "consistency_among_moved": c_moved, "frac_moved": frac_moved,
                "median_cos_z": float(np.nanmedian(cos_dist(Z[a], Z[b]))),
                "median_cos_raw": float(np.nanmedian(cos_dist(raw[a], raw[b]))),
                "n_pairs": len(ij), "degenerate_pairs": n_deg,
                "degenerate_frac": n_deg / len(ij)}
        done = [out[arm][e]["consistency"] for e in out[arm]]
        print(f"  {arm:<16s} mean consistency {np.mean(done):.3f} over {len(done)} edits")
    if skipped:
        agg = {}
        for arm, e, n in skipped:
            agg.setdefault(arm, set()).add(e)
        print()
        for arm, es in sorted(agg.items()):
            print(f"  ! {arm}: not re-embedded, so {sorted(es)} could not be scored")
    OUT.write_text(json.dumps(
        {"meta": {"metric": "leave-one-out directional consistency of the pair difference vector",
                  "null": 0.5, "standardised": "z-scored per dimension over all molecules",
                  "n_boot": N_BOOT, "dropped_arms": sorted(DROP)},
         "arms": out}, indent=1))
    print(f"\n  -> {OUT.relative_to(ROOT)}   (resolution.json untouched)")


if __name__ == "__main__":
    main()
