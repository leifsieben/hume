"""The six arms of the LSD AmpC screen, and the one head they all share.

EVERY FEATURIZER HERE IS IMPORTED, NOT REIMPLEMENTED. `bench/bench_downstream.py` holds the
definitions that produced Figures C and D, and this module calls them. That is not tidiness: a
local re-score of the HUME arms once reimplemented the estimator without `feature_weights` and
was compared against an anchor that had tuned it -- same numbers, different head, and nothing in
the records said so (bench_downstream.py:367). A screen whose `hume` arm was a lookalike of the
plotted `hume` arm would be incomparable with every figure in the paper, and silently.

THE RADIUS IS PASSED EXPLICITLY, never defaulted. The screen runs HUME at r=2 and r=4 in one
comparison, so a run whose radius came from whatever the installed package happens to default to
is a run whose radius is not in its own record. `_hume_block(..., fp_radius=N)` takes it.

ARM NAMES MATCH THE FIGURES. `chemberta_mtr`, not `chemberta_mlm`: Figure C's results.json
records `chemberta_mtr`, and the two are different weights. Picking MLM here because a boot
script exports CHEMBERTA_PATH at the MLM directory would have produced an arm that cannot be
plotted beside the existing ChemBERTa point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# bench_downstream.py and embed_pairs.py live in different places on a box and in the checkout:
# the dsbundle flattens everything to /opt/ds (with PYTHONPATH=/opt/ds), while the repo keeps
# them in bench/ and verification/. Resolving both here means a caller never has to know which
# layout it is on, and a missing module raises where it is imported rather than as a confusing
# ModuleNotFoundError three frames deep inside a featurizer.
_HERE = Path(__file__).resolve().parent
for _c in (_HERE.parent / "bench", _HERE.parent / "verification", Path("/opt/ds")):
    if _c.is_dir() and str(_c) not in sys.path:
        sys.path.append(str(_c))

W_FIXED = 10.0
#: The `feature_weights` multiplier, FIXED A PRIORI -- no inner CV, no per-arm search.
#:
#: Hall et al. tune nothing ("We used the default parameters, featurizers, and settings"), so the
#: paper-faithful choice is a constant. It is 10 and not 1 because docs/API.md section 7 measures
#: w=1 over 34 datasets at this exact colsample_bynode and finds it DOMINATED on both suites
#: (0.8080 vs 0.7882 activity, 0.8901 vs 0.8712 physchem): "uniform column sampling is not a
#: neutral default here; it is a bad one". w=1 would be a measured-bad choice that merely happens
#: to be XGBoost's default.
#:
#: Tuning w per arm was rejected for FAIRNESS, not cost. feature_weights is a no-op for an arm
#: with no fingerprint/descriptor boundary, so an inner-CV search spends selection variance only
#: on the descriptor-carrying arms -- bench_downstream.py:404 records this as "a systematic bias
#: in favour of the dense arms across the whole grid". Dropping the tuner removes a penalty on
#: HUME rather than granting it a privilege. The same constant applies to the ECFP4+RDKit
#: baseline, which carries the same boundary.
#:
#: NOTE: Figures C and D DID tune w. This differs from them deliberately; say so in the methods.

#: arm -> (featurizer, fp_block, which interpreter it needs)
#:
#: fp_block is (where, width) or None, in the same convention as bench_downstream.FP_BLOCK: HUME
#: stacks descriptors FIRST so its bits are the LAST 2048 columns, while ecfp_rdkit_desc stacks
#: the fingerprint first. Getting that backwards weights the descriptors by w and the bits by 1,
#: the exact inverse of the intended prior, and completely silently.
SPEC = {
    "hume_r4":          dict(fp_block=("tail", 2048), env="main", radius=4),
    "hume_r2":          dict(fp_block=("tail", 2048), env="main", radius=2),
    "ecfp_rdkit_desc":  dict(fp_block=("head", 2048), env="main", radius=2),
    "chemberta_mtr":    dict(fp_block=None,           env="main", radius=None),
    "chemeleon":        dict(fp_block=None,           env="main", radius=None),
    "minimol":          dict(fp_block=None,           env="mini", radius=None),
}
ALL_ARMS = tuple(SPEC)


def quiet_rdkit() -> None:
    """Silence RDKit's per-molecule parse complaints. At 100M molecules a 0.001% failure rate is
    a thousand five-line tracebacks interleaved with the progress log."""
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def parse_mask(smis: list[str]) -> np.ndarray:
    """-> bool array, True where RDKit can parse the SMILES.

    ONE PARSE-FAILURE SET FOR EVERY ARM, and it cannot be inferred from the feature matrices.
    The obvious detector -- "the row is all NaN" -- is wrong twice over. `f_ecfp` initialises its
    output to ZEROS and leaves a failed molecule as an all-zero fingerprint, so `ecfp_rdkit_desc`
    returns a row that is half zeros and half NaN rather than all NaN; and `f_rdkit_desc`
    legitimately leaves NaN for individual descriptors that raise on a perfectly valid molecule.
    So a NaN-based rule would both miss failures and invent them, differently per arm.

    RDKit parsing is deterministic, so every box computes the identical mask without coordinating.
    Phase 3 asserts the arms agree rather than assuming it.
    """
    from rdkit import Chem
    out = np.empty(len(smis), dtype=bool)
    for i, s in enumerate(smis):
        out[i] = Chem.MolFromSmiles(s) is not None
    return out


def featurize(arm: str, smis: list[str]) -> np.ndarray:
    """-> (len(smis), n_cols) float32. Rows RDKit cannot parse come back all-NaN.

    A row per input, never dropped, is the house convention and it is load-bearing here: the arms
    stay index-aligned with each other and with the frozen split without any arm having to know
    what another failed on. Which rows are failures comes from `parse_mask`, NOT from looking for
    NaN in the result -- see that function for why the NaN rule is wrong.
    """
    import bench_downstream as BD           # /opt/ds on a box, bench/ locally
    if arm not in SPEC:
        raise KeyError(f"unknown arm {arm!r}; the screen has {ALL_ARMS}")
    if arm in ("hume_r4", "hume_r2"):
        return BD._hume_block(smis, "full", fp_radius=SPEC[arm]["radius"])
    if arm == "ecfp_rdkit_desc":
        # Fingerprint FIRST, then the 217 RDKit descriptors -- the ("head", 2048) convention.
        return np.hstack([BD.f_ecfp(smis, radius=2), BD.f_rdkit_desc(smis)]).astype(np.float32)
    if arm == "minimol":
        return BD.f_minimol(smis).astype(np.float32)
    import embed_pairs as EP
    return np.asarray(EP.ARMS[arm](smis), dtype=np.float32)


def feature_weights(arm: str, n_cols: int) -> np.ndarray | None:
    """-> the feature_weights vector at the fixed W_FIXED, or None if the arm has no boundary."""
    blk = SPEC[arm]["fp_block"]
    if blk is None:
        return None
    where, width = blk
    v = np.ones(n_cols, dtype=np.float32)
    if where == "head":
        v[:width] = W_FIXED
    elif where == "tail":
        v[-width:] = W_FIXED
    else:
        raise ValueError(f"fp_block {blk!r} for arm {arm!r}")
    return v


def make_head(arm: str, n_cols: int, seed: int):
    """The one head, identical for every arm bar the fixed feature_weights vector.

    Built on bench_downstream.make_head so the estimator is the SAME OBJECT the figures used,
    with two deliberate differences, both recorded here rather than discovered later:

      1. `feature_weights` is the fixed W_FIXED vector, not a tuned one (see W_FIXED).
      2. Early stopping is enabled against the frozen scaffold-disjoint validation set. Hall et
         al. use "early stopping based on validation RMSE loss, with a patience of 5 epochs";
         n_estimators=300 becomes a cap rather than a fixed count. The figures had no validation
         set to stop on -- they were cross-validated -- so this is the screen reading across to
         the paper's protocol, applied identically to every arm.
    """
    import bench_downstream as BD
    m = BD.make_head("regression", seed, feature_weights(arm, n_cols))
    m.set_params(early_stopping_rounds=5, eval_metric="rmse")
    return m
