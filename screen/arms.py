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

N_TREES = 1000
#: The early-stopping CAP, raised from bench_downstream's 300 on 2026-10-29.
#:
#: WHY, measured: at 300 the cap bound on EVERY arm and EVERY replicate -- all 30 models came
#: back with best_iteration == 299, so the number of trees was decided by the ceiling and not by
#: the validation curve. Early stopping was configured, reported, and never once fired, which
#: means the scaffold-disjoint validation set was doing nothing for the purpose it was built for.
#:
#: The validation curve for hume_r4 replicate 0, over all 185,024 validation molecules:
#:     25 trees 7.5435 | 100 6.6978 | 200 6.3842 | 275 6.2936 | 300 6.2683
#: Decelerating hard -- the last 25 trees are worth +0.025 where the first 25 were worth +0.47 --
#: but still falling, so 300 was leaving real fit on the table.
#:
#: 300 WAS DEFENSIBLE AND IS NOT WHAT WE CHOSE. Every arm shared the cap, so the comparison was
#: fair, and Hall et al. say "default parameters", where XGBoost's own default is 100 -- their
#: models would have been cap-limited too. The call (2026-10-29) was to let early stopping
#: actually decide per arm, which is what the paper's stated protocol describes, and to pay a
#: phase 1 re-run for it. If 1000 also binds on every arm, that is a finding to report rather
#: than a number to raise again.
#:
#: Prediction cost scales with trees: 2.5 us/mol measured at 300 becomes ~8 us/mol at 1000,
#: still a small fraction of featurization for every arm, so phase 2's budget is unaffected.

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


def column_name(arm: str, j: int) -> str:
    """-> a human name for column `j` of `arm`, for error messages that say WHICH column.

    Only `ecfp_rdkit_desc` has named columns: its tail is RDKit's descriptor list in order, so a
    column index can be resolved to the descriptor that produced it. The learned embeddings have
    no names and say so rather than inventing one.
    """
    if arm != "ecfp_rdkit_desc":
        return f"column {j}"
    if j < 2048:
        return f"ECFP4 bit {j}"
    from rdkit.Chem import Descriptors
    names = [n for n, _f in Descriptors._descList]
    k = j - 2048
    return f"RDKit descriptor {names[k]!r}" if k < len(names) else f"column {j} (past the block)"


def sanitize(arm: str, X: np.ndarray) -> np.ndarray:
    """Map every +-inf to NaN, in place, and say which column produced it.

    WHY THIS EXISTS, measured 2026-10-29: `ecfp_rdkit_desc` trained for 98.6 minutes and then
    died in XGBoost's QuantileDMatrix with "Input data contains `inf` or a value too large,
    while `missing` is not set to `inf`".

    THE CULPRIT IS `MaxPartialCharge` AND `MaxAbsPartialCharge`, and this sanitizer is what
    identified them -- it names the column, which is the only reason the cause is known rather
    than guessed. RDKit returns inf from them directly; nothing overflows a cast. Both read
    Gasteiger partial charges, and charge assignment has no parameters for the phosphorus in
    `C(=O)[PH](C)(C)O`, so it yields inf for some such molecules and NaN for others. Over 36,706
    real training molecules: 2 inf and 6 NaN, every one carrying that motif. 0.0055% is about 54
    molecules in a 1,000,000-row training set, which is ample to kill QuantileDMatrix.

    (The first guess here was `Ipc` overflowing float32 -- an information-content index that does
    grow exponentially with molecular size, reaching 8.9e6 where every other descriptor sits in
    the hundreds. Plausible, and wrong: it never came close to 3.40e38 on this library. The
    column-naming above is what settled it.)

    NaN IS THE RIGHT TARGET, not a clamp to the float32 maximum. XGBoost reads NaN as missing and
    learns a default direction for it at every split, which is the honest encoding of "this
    descriptor could not be computed for this molecule". It also makes the two failure modes of
    one broken calculation agree: Gasteiger already returns NaN for six molecules where it
    returns inf for two, and there is no reason those should reach the model differently.
    Clamping would instead assert that the molecule sits exactly at the largest finite float, a
    number no descriptor ever means.

    THIS LIVES IN `featurize`, WHICH IS WHY IT IS CORRECT. train.py and score_test.py both reach
    the featurizers only through that one function, so the matrix a model is trained on and the
    matrix it later predicts from get the identical treatment. A sanitizer bolted onto the
    training path alone would have trained on NaN and scored on inf, and the 100M scoring run
    would have been the thing that discovered it.

    Applied to EVERY arm, not just the one that overflowed. It is a no-op where there is nothing
    to fix -- the five other arms trained without raising -- and a uniform rule means no arm gets
    a numerical treatment the others did not.
    """
    bad = np.isinf(X)
    if not bad.any():
        return X
    n = int(bad.sum())
    cols = np.flatnonzero(bad.any(axis=0))
    named = ", ".join(column_name(arm, int(j)) for j in cols[:4])
    more = f" (+{cols.size - 4} more)" if cols.size > 4 else ""
    print(f"    sanitize {arm}: {n:,} inf values in {cols.size} column(s) -> NaN; {named}{more}",
          file=sys.stderr, flush=True)
    X[bad] = np.nan
    return X


def featurize(arm: str, smis: list[str]) -> np.ndarray:
    """-> (len(smis), n_cols) float32. Rows RDKit cannot parse come back all-NaN.

    A row per input, never dropped, is the house convention and it is load-bearing here: the arms
    stay index-aligned with each other and with the frozen split without any arm having to know
    what another failed on. Which rows are failures comes from `parse_mask`, NOT from looking for
    NaN in the result -- see that function for why the NaN rule is wrong.

    Every arm's block passes through `sanitize` on the way out; see it for why inf becomes NaN
    and why doing it here rather than in the callers is the whole point.
    """
    import bench_downstream as BD           # /opt/ds on a box, bench/ locally
    if arm not in SPEC:
        raise KeyError(f"unknown arm {arm!r}; the screen has {ALL_ARMS}")
    if arm in ("hume_r4", "hume_r2"):
        return sanitize(arm, BD._hume_block(smis, "full", fp_radius=SPEC[arm]["radius"]))
    if arm == "ecfp_rdkit_desc":
        # Fingerprint FIRST, then the 217 RDKit descriptors -- the ("head", 2048) convention.
        X = np.hstack([BD.f_ecfp(smis, radius=2), BD.f_rdkit_desc(smis)]).astype(np.float32)
        return sanitize(arm, X)
    if arm == "minimol":
        return sanitize(arm, BD.f_minimol(smis).astype(np.float32))
    import embed_pairs as EP
    return sanitize(arm, np.asarray(EP.ARMS[arm](smis), dtype=np.float32))


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
         n_estimators becomes a cap rather than a fixed count. The figures had no validation
         set to stop on -- they were cross-validated -- so this is the screen reading across to
         the paper's protocol, applied identically to every arm.
      3. The cap is N_TREES, raised from the 300 bench_downstream uses. See N_TREES.
    """
    import bench_downstream as BD
    m = BD.make_head("regression", seed, feature_weights(arm, n_cols))
    m.set_params(n_estimators=N_TREES, early_stopping_rounds=5, eval_metric="rmse")
    return m


# ---------------------------------------------------------------------------------------------
# PARALLEL FEATURIZATION
#
# WHY, measured 2026-10-29 on the phase 1 fleet: every arm featurizes in ONE process while the
# box has sixteen vCPUs. `f_rdkit_desc` and `f_ecfp` measure cpu/wall 0.96 -- one core of
# sixteen -- and the HUME block 2.92. The cost table in scripts/launch_screen2.sh came from
# bench/bench_aws.py, which runs each arm in a sixteen-worker ProcessPoolExecutor, so its
# numbers are AGGREGATE THROUGHPUTS and the serial production path cannot reach them:
#
#   arm               planning table   phase 1 actual   ratio
#   hume_r4                    144.5            582     4.0x
#   hume_r2                    144.5            492     3.4x
#   chemberta_mtr              340.4            534     1.6x
#   chemeleon                 2892.5           5304     1.8x
#   ecfp_rdkit_desc            464.0           5956    12.8x
#
# The ratios differ because each library threads differently, and ECFP+RDKit -- the one pure
# Python loop -- loses the most. Left serial, phase 2 is ~529 box-hours instead of ~154.
#
# EVERY WORKER IS SINGLE-THREADED, and that is a choice, not an oversight. Mixing process
# parallelism with each library's own threads oversubscribes the box and can make a threaded arm
# slower than serial. The initializer pins OMP/MKL/OpenBLAS/rayon to one thread BEFORE the arm's
# featurizer is imported -- which works only because every featurizer here imports lazily inside
# `featurize`, so the environment is already set when torch and molhume first read it.
#
# OUTPUT IS BIT-IDENTICAL TO SERIAL. A molecule's features depend on that molecule alone, slabs
# are contiguous and written back at their own indices, and results are consumed in submission
# order. That is what makes this safe to apply to arms whose models were already trained by the
# serial path: the matrix does not change, only how long it takes to build.
# ---------------------------------------------------------------------------------------------

#: Workers per arm. Default is every vCPU; the cap exists where an arm genuinely cannot use them.
#: minimol is 4 after bench_aws.py's MAX_WORKERS -- above that it dies with BrokenProcessPool,
#: an error that names no memory and reads like an unrelated crash.
WORKERS_CAP = {"minimol": 4}

#: Molecules per task. Sized to AMORTIZE EACH ARM'S PER-CALL SETUP, which differs by orders of
#: magnitude: `_chemprop_embed` rebuilds the CheMeleon model on every call, `f_minimol` spawns a
#: whole separate interpreter that reloads its weights, while `f_ecfp`/`f_rdkit_desc` have no
#: setup at all and want small slabs for even load balancing. Slabs too small for a model-loading
#: arm would spend the parallelism on reloading weights.
SLAB = {"ecfp_rdkit_desc": 2_000, "hume_r4": 5_000, "hume_r2": 5_000,
        "chemberta_mtr": 25_000, "chemeleon": 25_000, "minimol": 25_000}

_POOL: dict = {}


def n_workers(arm: str) -> int:
    """-> workers for `arm`. SCREEN_WORKERS overrides, so a bad guess needs no code change."""
    import os
    env = os.environ.get("SCREEN_WORKERS", "")
    if env:
        return max(1, int(env))
    return max(1, min(os.cpu_count() or 1, WORKERS_CAP.get(arm, 1 << 30)))


def slab_size(arm: str) -> int:
    """-> molecules per task for `arm`. SCREEN_SLAB overrides."""
    import os
    env = os.environ.get("SCREEN_SLAB", "")
    return max(1, int(env)) if env else SLAB.get(arm, 5_000)


def _worker_init() -> None:
    """Pin the worker to one thread, before any featurizer library is imported."""
    import os
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    quiet_rdkit()


def _worker_featurize(arm: str, smis: list[str]) -> np.ndarray:
    return featurize(arm, smis)


def _worker_parse_mask(smis: list[str]) -> np.ndarray:
    return parse_mask(smis)


def _pool(arm: str, nw: int):
    """-> a CACHED pool for (arm, nw). Cached because every worker pays the import cost of its
    arm's featurizer stack once; a pool per call would pay it per call."""
    import atexit
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    key = (arm, nw)
    if key not in _POOL:
        ex = ProcessPoolExecutor(max_workers=nw, initializer=_worker_init,
                                 mp_context=mp.get_context("spawn"))
        _POOL[key] = ex
        atexit.register(lambda: ex.shutdown(wait=False, cancel_futures=True))
    return _POOL[key]


def _parallel_into(fn, arm: str, smis: list[str], out: np.ndarray, label: str, log) -> np.ndarray:
    """Run `fn(arm-or-nothing, slab)` over contiguous slabs into `out`, in order."""
    import time
    n = len(smis)
    nw, slab = n_workers(arm), slab_size(arm)
    if nw <= 1 or n <= slab:
        out[...] = fn(smis) if fn is _worker_parse_mask else fn(arm, smis)
        return out
    ex = _pool(arm, nw)
    bounds = [(i, min(i + slab, n)) for i in range(0, n, slab)]
    t0 = time.perf_counter()
    for g in range(0, len(bounds), nw):
        group = bounds[g:g + nw]
        futs = [ex.submit(fn, smis[lo:hi]) if fn is _worker_parse_mask
                else ex.submit(fn, arm, smis[lo:hi]) for lo, hi in group]
        for (lo, hi), f in zip(group, futs):
            out[lo:hi] = f.result()
        done = group[-1][1]
        if label and (g // nw) % 4 == 0:
            el = time.perf_counter() - t0
            log(f"    {label} {done:>9,}/{n:,}  {el/60:5.1f} min  "
                f"{(n-done)*el/done/60:5.1f} min left  {el/done*1e6:6.1f} us/mol  {nw}w")
    return out


def featurize_many(arm: str, smis: list[str], out: np.ndarray, label: str = "", log=print):
    """Featurize `smis` into preallocated `out` using every core. Identical to serial output."""
    return _parallel_into(_worker_featurize, arm, smis, out, label, log)


def parse_mask_many(arm: str, smis: list[str], out: np.ndarray, label: str = "", log=print):
    """`parse_mask` across workers. Serial it is another single-core RDKit pass over every row."""
    return _parallel_into(_worker_parse_mask, arm, smis, out, label, log)
