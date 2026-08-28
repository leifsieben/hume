"""Figure A's axis: can a tree tell the two members of a matched pair apart?

    .venv/bin/python figa_resolution.py [arm ...]   -> results/figures/figA/resolution.json

THE METRIC. For one edit, label every A molecule 0 and its edited partner A' 1, fit XGBoost, and
report held-out ROC-AUC. 0.5 = cannot tell them apart; 1.0 = separates every held-out pair.

WHY THIS RATHER THAN A MAGNITUDE. A tree splits on single dimensions, so resolution means "is
there a dimension that separates them", not "how far did the vector move". A magnitude axis
systematically understates a representation where one coordinate is decisive -- one flipped bit
in 2048 is a negligible cosine change however much it carries. Counting dimensions instead does
not work either: for a continuous embedding the count moves three orders of magnitude with the
threshold (CheMeleon's stereo count is 27% of dims at a median-of-reference threshold, 0.05% at
p95 of background pairs, 0% at p99.9), and the natural calibration is degenerate because for a
continuous embedding "a completely different molecule" IS a random background pair. AUC has no
free parameter and is scale-free, so a 2048-bit fingerprint and a 384-d transformer share one
axis with no normalisation at all.

WHAT 0.5 MEANS, and it is not "the vectors are identical". Two different situations both give
0.5: identical vectors (a fingerprint cannot see a re-written SMILES), and vectors that differ
in an INCONSISTENT DIRECTION (ChemBERTa on stereo -- the vectors do move, but a different way per
molecule, so there is no rule to learn and none to transfer). The second is invisible to a
magnitude axis and is exactly what matters downstream: a model does not need the vectors to
differ, it needs the difference to mean the same thing on a molecule it has not seen.

---------------------------------------------------------------------------------------------
THREE DEFECTS THIS HARNESS FIXES, all found by the CLIMB figures session running the same metric
on their pair set and asserting what the pilot merely intended.
---------------------------------------------------------------------------------------------

1. SPLITTING BY PAIR IS NOT ENOUGH. It is sufficient only if no molecule appears in more than one
   pair. Measured on our own set, 32 molecules do (null_kekulize 20, ez_flip 4, null_enumerate 4,
   n_methylation 2, regioisomer 2). With any reuse, pair 3 can go to train and pair 47 to test
   while both contain molecule M -- M is then on both sides and the model can memorise it instead
   of learning the edit. THE LEAK RAISES AUC, so the failure mode looks like a better result.
   The fix is to group pairs into connected components under "shares a molecule" and assign whole
   COMPONENTS; that is the only split under which "the model has never seen this molecule" is
   true. It is asserted below rather than assumed.

2. DEGENERATE PAIRS ARE A FACT ABOUT THE PAIR SET, NOT A SCORE. Where A and A' have the same
   vector the two classes are literally one point and 0.5 is forced. 24 of our pairs are
   identical as written (20 null_kekulize -- molecules already Kekule -- and 4 null_enumerate
   where the random SMILES came back canonical). Those are counted and reported, never averaged
   in as though an arm had failed to resolve something resolvable.

3. ONE SPLIT IS NOT AN ESTIMATE. Every representation here is deterministic, so the split is the
   only stochastic element and its spread IS the uncertainty of the number. CLIMB measured
   0.833 +/- 0.020 on one of theirs; a single split would have been worth about +/-0.04, which is
   the same size as differences we would otherwise have called results. Five seeds, mean and SD.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
FIGA = ROOT / "results" / "figures" / "figA"
EMB = FIGA / "embeddings"
N_SEEDS = 5
TEST_FRAC = 0.20


def components(pair_rows):
    """Group pairs into connected components under "shares a molecule" (union-find)."""
    parent = list(range(len(pair_rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    seen: dict[int, int] = {}
    for i, (a, b) in enumerate(pair_rows):
        for m in (a, b):
            if m in seen:
                union(i, seen[m])
            else:
                seen[m] = i
    out: dict[int, list[int]] = {}
    for i in range(len(pair_rows)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def split_components(pair_rows, rng, test_frac=TEST_FRAC):
    """-> (train pair indices, test pair indices). Whole components, never split."""
    comps = components(pair_rows)
    rng.shuffle(comps)
    target = int(test_frac * len(pair_rows))
    te: list[int] = []
    for c in comps:
        if len(te) >= target:
            break
        te.extend(c)
    te_set = set(te)
    tr = [i for i in range(len(pair_rows)) if i not in te_set]
    return tr, te


def auc_for(X, pair_rows, seed):
    rng = np.random.default_rng(seed)
    tr, te = split_components(pair_rows, rng)
    if len(te) < 20 or len(tr) < 50:
        return np.nan
    rows = lambda idx: ([pair_rows[i][0] for i in idx], [pair_rows[i][1] for i in idx])
    atr, btr = rows(tr)
    ate, bte = rows(te)
    # THE ASSERTION THAT FOUND THE BUG. Intending a clean split is not the same as having one.
    leak = (set(atr) | set(btr)) & (set(ate) | set(bte))
    assert not leak, (
        f"pair split leaked {len(leak)} molecule(s) across the boundary -- a molecule is in both "
        f"train and test, so the model can memorise it instead of learning the edit, and the "
        f"leak RAISES AUC. Component assignment is broken.")
    Xtr = np.vstack([X[atr], X[btr]])
    ytr = np.r_[np.zeros(len(atr)), np.ones(len(btr))]
    Xte = np.vstack([X[ate], X[bte]])
    yte = np.r_[np.zeros(len(ate)), np.ones(len(bte))]
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.15, subsample=0.9,
                      colsample_bytree=0.5, n_jobs=8, tree_method="hist",
                      eval_metric="logloss", random_state=seed)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def main(arms):
    pairs = json.load(open(FIGA / "pairs.json"))["pairs"]
    idx = json.load(open(FIGA / "smiles_index.json"))
    pos = {s: i for i, s in enumerate(idx["order"])}
    edits = sorted({p["edit"] for p in pairs})
    out = {}
    for arm in arms:
        f = EMB / f"{arm}.npz"
        if not f.exists():
            print(f"  {arm}: no embedding yet, skipping", flush=True)
            continue
        X = np.nan_to_num(np.load(f)["X"].astype(np.float32))
        out[arm] = {}
        for e in edits:
            rows = [(pos[p["a"]], pos[p["b"]]) for p in pairs
                    if p["edit"] == e and p["a"] in pos and p["b"] in pos]
            if len(rows) < 100:
                continue
            # Degenerate pairs: identical VECTORS, which is stronger than identical SMILES --
            # a fingerprint is invariant to a re-written string, so its whole null column is
            # degenerate by construction and that is the correct reading, not a failure.
            degen = int(sum(1 for a, b in rows if np.array_equal(X[a], X[b])))
            v = [auc_for(X, rows, s) for s in range(N_SEEDS)]
            v = [x for x in v if np.isfinite(x)]
            out[arm][e] = {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                           "min": float(np.min(v)), "max": float(np.max(v)),
                           "n_pairs": len(rows), "n_seeds": len(v),
                           "degenerate_pairs": degen,
                           "degenerate_frac": degen / len(rows)}
            d = out[arm][e]
            print(f"  {arm:<15s} {e:<16s} {d['mean']:.3f} +/- {d['sd']:.3f}"
                  f"   [{d['min']:.3f},{d['max']:.3f}]"
                  f"{'   degenerate ' + str(degen) if degen else ''}", flush=True)
        FIGA.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(FIGA / "resolution.json", "w"), indent=1)
    print(f"\n  -> {FIGA / 'resolution.json'}")


if __name__ == "__main__":
    main(sys.argv[1:] or [p.stem for p in sorted(EMB.glob("*.npz"))])
