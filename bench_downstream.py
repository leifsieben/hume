"""Figures B and C: does each representation help downstream, and at what cost?

    CHEMPFN_DATA_ROOT=~/chempfn-data python bench_downstream.py --arms ecfp hume --out out.json

WHAT THIS PRODUCES. One record per (dataset, fold, arm) in the same schema as the existing
results/figures/figB/grid_all.json, so new arms are DIRECTLY COMPARABLE with the base arms
already measured rather than forming a fresh, incomparable grid. Same 28 DEV datasets, same
5-fold scaffold split, same seed.

THE DATA COMES FROM THE CHEMPFN LAKE, read-only. `chemtfm` is a discontinued repo and its
vendored copy here resolves only 8 of the 28 datasets; the chempfn lake resolves all 28, every
one role DEV, so nothing LOCKED is touched and nothing needs downloading. The lake is 46 GB but
these 28 sets are 209 MB.

SCAFFOLD FOLDS, NOT RANDOM. A random split puts near-duplicates of a test molecule in train and
every representation looks good; the whole question here is whether a representation carries
information that transfers to an unseen scaffold. chempfn.eval.splits.scaffold_folds is used
rather than a local reimplementation so these numbers sit beside chempfn's own.

METRICS ARE SCALE-FREE AND ARE NOT POOLED ACROSS DATASETS IN NATIVE UNITS -- Spearman for
regression, ROC-AUC for binary, per chempfn.eval.protocol's rule. A mean of RMSE over datasets
whose units differ by two orders of magnitude is arithmetic on incommensurables.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import json
import numpy as np
from pathlib import Path

CHEMPFN = os.environ.get("CHEMPFN_SRC", "/Users/lsieben/VSCode/ChemPFN")
sys.path.insert(0, CHEMPFN)


def _quiet():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


# ------------------------------------------------------------------ featurisers
def f_ecfp(smis, radius=2, size=2048):
    """ECFP4 -- Morgan RADIUS 2, which is what `ecfp` means everywhere else in this project.

    This defaulted to radius=3 for one commit and the reproduction check caught it: `ecfp` on
    esol came out 1.089 against grid_all's 1.207, and on bioavail 0.696 against 0.675. Every
    new arm would have been silently incomparable with the base arms already in the grid, and
    the only symptom would have been a figure where the new arms all looked slightly different
    for no stated reason. arms.py labels this arm "ECFP4"; radius 2 is what ECFP4 is.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as fg
    g = fg.GetMorganGenerator(radius=radius, fpSize=size, includeChirality=True)
    out = np.zeros((len(smis), size), np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = g.GetCountFingerprintAsNumPy(m)
    return out


def f_r3cfp(smis):
    return f_ecfp(smis, radius=3)


def f_r4cfp(smis):
    return f_ecfp(smis, radius=4)


def f_rdkit_desc(smis):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fns = Descriptors._descList
    out = np.full((len(smis), len(fns)), np.nan, np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for j, (_n, f) in enumerate(fns):
            try:
                out[i, j] = f(m)
            except Exception:
                pass
    return out


MORDRED_PY = os.environ.get("MORDRED_PY", "")
MINIMOL_PY = os.environ.get("MINIMOL_PY", "")


def f_mordred_desc(smis):
    """Mordred, computed in a SEPARATE INTERPRETER, because it cannot share this one.

    mordred 1.2.0 requires numpy 1.x (it calls `np.float`, removed in numpy 1.24) while hume and
    torch here run on numpy 2.4.6. The two cannot coexist in one environment, so `MORDRED_PY`
    points at a second venv and the features cross as a .npy file. Without this the three
    Mordred-carrying arms would have died on ImportError -- after the box had already paid for
    every other arm.

    Set MORDRED_PY to run it; unset, this raises rather than silently returning a column of NaN
    that would look like a legitimately missing descriptor.
    """
    if not MORDRED_PY:
        raise RuntimeError(
            "MORDRED_PY is not set. Mordred needs its own interpreter (numpy 1.x) and this "
            "process is on numpy 2.x; point MORDRED_PY at a python that has mordred installed. "
            "Refusing to return NaN, which would be indistinguishable from a real missing value.")
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        smi_f, out_f = f"{td}/in.txt", f"{td}/out.npy"
        open(smi_f, "w").write("\n".join(smis))
        code = (
            "import sys, numpy as np\n"
            "from mordred import Calculator, descriptors as md\n"
            "from rdkit import Chem, RDLogger\n"
            "RDLogger.DisableLog('rdApp.*')\n"
            "smis = open(sys.argv[1]).read().split('\\n')\n"
            "calc = Calculator(md, ignore_3D=True)\n"
            "mols = [Chem.MolFromSmiles(s) for s in smis]\n"
            "ok = [m for m in mols if m is not None]\n"
            "rows = [[float(v) if isinstance(v,(int,float)) else np.nan for v in r]\n"
            "        for r in calc.map(ok, nproc=1, quiet=True)]\n"
            "vals = np.array(rows, np.float32)\n"
            "out = np.full((len(smis), vals.shape[1]), np.nan, np.float32)\n"
            "out[[i for i,m in enumerate(mols) if m is not None]] = vals\n"
            "np.save(sys.argv[2], out)\n")
        r = subprocess.run([MORDRED_PY, "-c", code, smi_f, out_f], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"mordred subprocess failed: {r.stderr.decode()[-2000:]}")
        return np.load(out_f)


#: The 185 columns wired after the deduplication -- counts_ext, estate_ext, eta, spectral and
#: misc_ext, minus the 43 the cost triage dropped. `hume_no_new` masks exactly these, so the pair
#: (hume_no_new, hume_all_desc) isolates what they are worth downstream.
_NEW_COLS = None


def _hume_block(smis, drop_new: bool):
    """ECFP + HUME's descriptors. `drop_new` masks the 185 post-dedup columns.

    `optional=` NO LONGER ASKS FOR qed. The column was dropped in the cost triage because it
    shipped 100% NaN -- its 116 structural alerts are OPT_QED and off by default -- so requesting
    it now buys 69.3 us/mol of alert matching for a column that is not emitted. AvgIpc is still
    requested: it is emitted, and it is not reconstructible (GBM R2 0.889 from the cheap basis).
    """
    global _NEW_COLS
    import molhume as hume
    from rdkit import Chem
    mols, keep = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m); keep.append(i)
    fp, X, cols = hume.featurize_all_from_mols(mols, optional=("AvgIpc",))
    if drop_new:
        if _NEW_COLS is None:
            _NEW_COLS = set(json.loads(
                Path("results/dedupe2/new_columns.json").read_text()))
        mask = np.array([c not in _NEW_COLS for c in cols], dtype=bool)
        if mask.sum() == len(cols):
            raise RuntimeError(
                "hume_no_new: results/dedupe2/new_columns.json masked 0 of "
                f"{len(cols)} columns; the ablation would be identical to hume_all_desc")
        X = X[:, mask]
    out = np.full((len(smis), X.shape[1] + fp.shape[1]), np.nan, np.float32)
    out[keep] = np.hstack([X, fp]).astype(np.float32)
    return out


def f_hume(smis):
    """ECFP + every descriptor HUME computes, including our own. One arm, no variants."""
    return _hume_block(smis, drop_new=False)


def f_hume_no_new(smis):
    """HUME WITHOUT the 185 columns wired after the deduplication."""
    return _hume_block(smis, drop_new=True)


#: Per-dataset memo for feature BLOCKS. Cleared at the top of every dataset, so it never holds
#: one dataset's features while another is being scored.
_BLOCK_CACHE: dict = {}


def _block(name, fn):
    """Compute a feature block at most once per dataset, however many arms ask for it."""
    def go(smis):
        if name not in _BLOCK_CACHE:
            _BLOCK_CACHE[name] = fn(smis)
        return _BLOCK_CACHE[name]
    go.__name__ = f"block:{name}"
    return go


def _cat(*fs):
    return lambda smis: np.hstack([f(smis) for f in fs])


def f_minimol(smis):
    """MiniMol, in a SEPARATE INTERPRETER, for the same reason Mordred needs one.

    minimol depends on graphium, whose pins move torch underneath everything else in the
    environment. Installed alongside transformers it left `is_torch_available()` False, so
    `AutoModel.from_pretrained` on the ChemBERTa weights died with
    `ModuleNotFoundError: Could not import module 'RobertaModel'` -- an error that names Roberta
    and says nothing about minimol, on a box where minimol was the only thing that had changed.
    That killed a whole four-arm downstream run at preflight.

    The local setup already had this right (there is a `.venv-minimol` beside `.venv-mordred`
    for exactly this reason) and the cloud image did not, which is the kind of drift that only
    shows up on the box.
    """
    if not MINIMOL_PY:
        raise RuntimeError(
            "MINIMOL_PY is not set. minimol pulls graphium, which moves torch and breaks "
            "transformers in the same environment; point MINIMOL_PY at a python that has "
            "minimol installed. Refusing to return NaN, which would be indistinguishable from "
            "a real missing value.")
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        smi_f, out_f = f"{td}/in.txt", f"{td}/out.npy"
        open(smi_f, "w").write("\n".join(smis))
        # RETRY SINGLY ON A BATCH FAILURE. torch_geometric's `collate` raises for the WHOLE
        # batch when one molecule produces a graph it cannot stack, so a single bad SMILES took
        # 256 molecules -- and then the whole arm -- down with it. Measured on aqsoldb, where all
        # three minimol arms failed and nothing else did. Molecules that still fail alone are
        # left as NaN rows, which is what "this featuriser has no value here" means and what
        # XGBoost already treats as missing; the count is printed so it is never silent.
        code = (
            "import sys, numpy as np, torch\n"
            "from minimol import Minimol\n"
            "torch.set_grad_enabled(False)\n"
            "smis = open(sys.argv[1]).read().split('\\n')\n"
            "m = Minimol()\n"
            "rows = [None] * len(smis)\n"
            "D = None\n"
            "def run(chunk, base):\n"
            "    global D\n"
            "    for j, v in enumerate(m(chunk)):\n"
            "        a = np.asarray(v, np.float32); D = a.shape[0]; rows[base + j] = a\n"
            "B = 256\n"
            "for i in range(0, len(smis), B):\n"
            "    chunk = smis[i:i + B]\n"
            "    try:\n"
            "        run(chunk, i)\n"
            "    except Exception:\n"
            "        for j, one in enumerate(chunk):\n"
            "            try: run([one], i + j)\n"
            "            except Exception: pass\n"
            "if D is None:\n"
            "    raise RuntimeError('minimol embedded no molecule at all')\n"
            "bad = sum(1 for a in rows if a is None)\n"
            "if bad: print(f'  minimol: {bad}/{len(smis)} molecules unfeaturisable -> NaN',\n"
            "              file=sys.stderr)\n"
            "out = np.full((len(smis), D), np.nan, np.float32)\n"
            "for k, a in enumerate(rows):\n"
            "    if a is not None: out[k] = a\n"
            "np.save(sys.argv[2], out)\n")
        r = subprocess.run([MINIMOL_PY, "-c", code, smi_f, out_f], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"minimol subprocess failed: {r.stderr.decode()[-2000:]}")
        return np.load(out_f)


def f_learned(kind):
    """ChemBERTa-2 / CheMeleon / MolFormer, whichever is asked for. Imported lazily: none of
    them shares a dependency set with the others and a missing one must not kill the run.

    MiniMol is NOT routed through here -- see f_minimol.
    """
    assert kind != "minimol", "minimol needs its own interpreter; use f_minimol"
    def go(smis):
        import embed_pairs as EP
        return EP.ARMS[kind](smis)
    return go


#: Which columns of each arm are FINGERPRINT BITS, as (offset, width) or None.
#:
#: `feature_weights` needs this and nothing else does. It only has an effect on an arm that MIXES
#: sparse bits with dense descriptors -- for a pure fingerprint or a pure descriptor block every
#: weight is equal and the parameter is a no-op, so those arms skip the tuning entirely and cost
#: one fit instead of five.
#:
#: `hume` is the odd one: featurize_all returns (fp, X) and f_hume stacks them DESCRIPTORS FIRST,
#: so its bits are the LAST 2048 columns, not the first. Getting that backwards would weight the
#: descriptors by w and the bits by 1 -- the exact inverse of the intended prior, and completely
#: silent.
FP_BLOCK = {
    "ecfp": "all", "r3cfp": "all", "r4cfp": "all",
    "desc": None, "chemberta_mtr": None, "chemberta_mlm": None,
    "minimol": None, "chemeleon": None, "molformer": None,
    "ecfp_rdkit_desc": ("head", 2048), "ecfp_mordred_desc": ("head", 2048),
    "ecfp_all_desc": ("head", 2048), "hume": ("tail", 2048),
}

#: The `w` grid from API.md section 7. w=1 is XGBoost's uniform default and is dominated on BOTH
#: suites there (0.8080 vs 0.7882 activity, 0.8901 vs 0.8712 physchem), so it is kept in the grid
#: as a control rather than as a plausible answer.
W_GRID = (1.0, 5.0, 10.0, 100.0)

#: Inner-CV settings for the `w` search. K folds rather than one split, and a floor on the
#: inner-validation size below which we use API.md section 7's documented default instead of
#: tuning -- a tuner given 90 molecules returns noise, and noise costs only the arms that tune.
#: Harness protocol version, STAMPED INTO EVERY RECORD. Runs of different vintages coexist in
#: the same S3 prefix and a collector cannot otherwise tell them apart -- "newest file wins" is
#: wrong the moment a long-running box keeps re-uploading its partial, which is exactly what
#: happened: dsA had been running for 20 hours, so its old-protocol records carried a fresher
#: timestamp than a re-run that superseded them and silently won every merge.
#: 1 = single 80/20 inner split for w.  2 = 3-fold inner CV with a MIN_INNER_VAL floor.
PROTO = 2

W_INNER_K = 3
MIN_INNER_VAL = 200
W_UNTUNED = 10.0


def fp_weights(arm, n_cols, w):
    """-> the feature_weights vector, or None when the arm has no fingerprint/descriptor split."""
    blk = FP_BLOCK.get(arm)
    if blk is None or blk == "all" or w == 1.0:
        return None
    where, width = blk
    v = np.ones(n_cols, dtype=np.float32)
    if where == "head":
        v[:width] = w
    else:
        v[-width:] = w
    return v


B_ECFP = _block("ecfp", f_ecfp)
B_RDKIT = _block("rdkit_desc", f_rdkit_desc)
B_MORD = _block("mordred_desc", f_mordred_desc)
B_LEARNED = {k: _block(k, f_minimol if k == "minimol" else f_learned(k))
             for k in ("chemeleon", "minimol", "chemberta_mtr", "chemberta_mlm", "molformer")}

ARMS = {
    "ecfp":            B_ECFP,
    "r3cfp":           f_r3cfp,
    "r4cfp":           f_r4cfp,
    "ecfp_rdkit_desc": _cat(B_ECFP, B_RDKIT),
    "ecfp_mordred_desc": _cat(B_ECFP, B_MORD),
    "ecfp_all_desc":   _cat(B_ECFP, B_RDKIT, B_MORD),
    "desc":            _cat(B_RDKIT, B_MORD),
    "hume":            f_hume,
    "hume_all_desc":   f_hume,          # same block; named for the ablation pair
    "hume_no_new":     f_hume_no_new,
    "chemberta_mtr":   B_LEARNED["chemberta_mtr"],
    "chemberta_mlm":   B_LEARNED["chemberta_mlm"],
    "minimol":         B_LEARNED["minimol"],
    "chemeleon":       B_LEARNED["chemeleon"],
    "molformer":       B_LEARNED["molformer"],
}

# ---- FIGURE B PANEL 2: base (x) add ---------------------------------------------------------
#
# Figure B's whole claim is a CONCATENATION test -- if a learned embedding carries chemical
# signal a classical block lacks, gluing it on must beat the block alone -- so the plate needs
# every base(x)add cell, not just the bases and the adds separately. Three bases against four
# embeddings is twelve arms, and they are generated rather than typed out because a hand-written
# list is where `ecfp_chemeleon` quietly becomes `desc_chemeleon`.
#
# THE FINGERPRINT BLOCK STAYS AT THE HEAD in every generated arm, because `_cat` concatenates in
# the order given and the base is always first. `desc` has no bits at all, so those four arms
# have no fingerprint/descriptor split and correctly skip the `w` tuning.
FIGB_BASES = {"ecfp": (B_ECFP, "all"),
              "desc": (_cat(B_RDKIT, B_MORD), None),
              "ecfp_all_desc": (_cat(B_ECFP, B_RDKIT, B_MORD), ("head", 2048))}
FIGB_ADDS = ["chemeleon", "minimol", "chemberta_mtr", "molformer"]
for _b, (_fn, _blk) in FIGB_BASES.items():
    for _a in FIGB_ADDS:
        _k = f"{_b}__{_a}"
        ARMS[_k] = _cat(_fn, B_LEARNED[_a])
        # The added block is dense, so the base's own bit layout carries over unchanged: a
        # "head" block stays at the head and an all-bits base stops being all-bits.
        FP_BLOCK[_k] = ("head", 2048) if _blk in ("all", ("head", 2048)) else None

DATASETS = ["ames", "aqsoldb", "bioavail", "cycpept_pampa", "cyp2d6_inh", "esol", "fartdb",
            "hia", "ld50_zhu", "lipophilicity", "pb_ames", "pb_bbb", "pb_cyp2c9", "pb_cyp2d6",
            "pb_cyp3a4", "pb_hum_mic_cl", "pb_logd", "pb_mou_mic_cl", "pb_ppb",
            "pb_rat_mic_cl", "pb_water_sol", "photoswitch", "qm8", "qm9", "qm9_gap",
            "qmugs_gap", "rascore", "vdss_lombardo",
    # SEVEN MORE CLASSIFICATION SETS. The grid had six, two of them under 700 molecules, and the
    # one place a learned representation showed a consistent edge over both HUME and the
    # descriptor union was classification -- CheMeleon beat both on bioavail, hia and pb_ames and
    # nowhere else. Six datasets is a thin basis for that conclusion in either direction.
    #
    # All seven were checked for MOLECULE OVERLAP against what is already in the grid before
    # being added; the largest is herg at 13.4% and the rest are under 2%. bbbp was rejected at
    # 78.2% overlap with pb_bbb, clintox at 93.6% positives, cbs at 0.4% positives, and litpcba
    # at 2.8M molecules.
    "bace",            #  1,513, 45.7% positive -- BACE-1 inhibition (local loader)
    "hiv",             # 41,127,  3.5% -- HIV replication (local loader)
    "herg",            #    655, 68.9% -- hERG cardiotoxicity
    "wong_hepg2",      # 39,101,  8.5% -- human liver cytotoxicity
    "wong_imr90",      # 39,074,  8.7% -- human fibroblast
    "wong_hskmc",      # 39,115,  3.8% -- skeletal muscle
    "wong_saureus",    # 39,121,  1.3% -- antibacterial
]


#: DATASETS THAT LIVE IN THE LAKE BUT HAVE NO SPEC IN IT.
#:
#: `eval/dev/moleculenet_legacy/` holds the MoleculeNet classification benchmarks as plain CSVs,
#: and chempfn's spec table does not register bace or hiv at all -- the four it does register
#: there (tox21, sider, toxcast, muv) point `relpath` at a FILE while its loader globs
#: `relpath/*.csv`, so they resolve to zero files. That is a bug in a repo that is not ours and
#: we do not touch it; we read the two single-label files we want, ourselves, here.
#:
#: bbbp is deliberately ABSENT: 78.2% of its molecules are already in pb_bbb, so adding it would
#: double-count blood-brain barrier and quietly reweight the classification panel. clintox is
#: absent too, at 93.6% positive -- about 19 negatives per test fold, where AUROC is mostly noise.
LOCAL_DATASETS = {
    "bace": dict(relpath="eval/dev/moleculenet_legacy/bace.csv",
                 smiles_col="mol", label_col="Class", task="binary"),
    "hiv":  dict(relpath="eval/dev/moleculenet_legacy/hiv.csv",
                 smiles_col="smiles", label_col="HIV_active", task="binary"),
}


class _LocalSpec:
    """Enough of chempfn's DatasetSpec for the two fields the runner reads off it."""

    def __init__(self, name, task):
        self.name, self.task, self.role = name, task, "LOCAL"


def load_ds(name):
    """Read a lake dataset with an EXPLICIT encoding, rather than the ambient locale's.

    chempfn's own `load_dataset` opens CSVs with a bare `open(newline="")`, which decodes using
    whatever `locale.getpreferredencoding()` returns. That is UTF-8 on this laptop and was
    something else on a cloud-init EC2 box, where the first dsA run died on
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3` reading a file that is
    byte-identical and valid UTF-8 here. Rather than depend on a box's locale -- or patch a repo
    that is not ours -- this reads the same files the same way everywhere, and replaces an
    undecodable byte instead of losing a whole dataset to one of them.

    Column names, task and file layout still come from chempfn's spec, so the dataset definition
    stays theirs and only the byte-level read is ours.
    """
    import csv as _csv
    if name in LOCAL_DATASETS:
        from chempfn.data.lake import lake_root
        import os as _os
        cfg = LOCAL_DATASETS[name]
        path = _os.path.join(str(lake_root()), cfg["relpath"])
        smis, ys = [], []
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in _csv.DictReader(fh):
                sm, lab = row.get(cfg["smiles_col"]), row.get(cfg["label_col"])
                if not sm or lab in (None, "", "None"):
                    continue
                try:
                    v = float(lab)
                except ValueError:
                    continue
                smis.append(sm); ys.append(v)
        if not smis:
            raise RuntimeError(
                f"{name}: no usable rows from {path} using smiles_col="
                f"{cfg['smiles_col']!r} label_col={cfg['label_col']!r}; the file's header is "
                f"the thing to check")
        return {"name": name, "smiles": smis, "y": ys,
                "task": LOCAL_DATASETS[name]["task"]}
    from chempfn.data.lake import spec, _csv_files
    sp = spec(name)
    smis, ys = [], []
    for f in _csv_files(sp):
        with open(f, newline="", encoding="utf-8", errors="replace") as fh:
            for row in _csv.DictReader(fh):
                s_, y_ = row.get(sp.smiles_col), row.get(sp.label_col)
                if not s_ or y_ in (None, ""):
                    continue
                try:
                    ys.append(float(y_))
                except ValueError:
                    continue
                smis.append(s_)
    if not smis:
        raise RuntimeError(f"{name}: no usable rows from {[str(x) for x in _csv_files(sp)]}")
    return {"name": name, "smiles": smis, "y": ys, "task": sp.task}


#: DATASETS WE USE DESPITE A NON-DEV ROLE, NAMED ONE BY ONE.
#:
#: The DEV/LOCKED split is CHEMPFN'S held-out discipline, not ours -- it exists to stop that
#: project's model selection from touching its own test set. HUME is a featurizer benchmark: it
#: fits a fresh XGBoost per fold per dataset and selects nothing across them, so a dataset being
#: LOCKED over there says nothing about whether measuring a representation on it here is sound.
#: (Leif, asked directly: "locked doesn't matter for us, that's for the chempfn project".)
#:
#: The guard stays for everything NOT on this list. A blanket `role != DEV` bypass would also
#: silently pull in TEST and RETIRED datasets -- `krishnan` is retired with a data-quality note
#: attached -- and the point of an allowlist is that each entry was looked at.
ALLOW_NON_DEV = {
    "herg",            #    655, 68.9% positive -- hERG cardiotoxicity, standard endpoint
    "wong_hepg2",      # 39,101,  8.5% -- human liver cytotoxicity
    "wong_imr90",      # 39,074,  8.7% -- human fibroblast
    "wong_hskmc",      # 39,115,  3.8% -- skeletal muscle
    "wong_saureus",    # 39,121,  1.3% -- antibacterial
}


def run(arm_names, datasets, out_path, folds_k=5, seed=0):
    _quiet()
    from chempfn.data.lake import spec, lake_root, _csv_files
    from chempfn.eval.splits import scaffold_folds, train_test
    from sklearn.metrics import roc_auc_score, accuracy_score
    import xgboost as xgb

    def make(task, fw=None):
        """Untuned XGBoost, per METHODS -- with the one documented exception.

        Everything about the head is left at a fixed default so a difference between arms is a
        difference between REPRESENTATIONS. The exception is `feature_weights`, which API.md
        section 7 recommends tuning: it sets how strongly column sampling favours fingerprint
        bits over descriptors, and the best value differs between potency and physicochemical
        endpoints.

        `colsample_bynode=0.3` IS PART OF THE MECHANISM, NOT A TUNING CHOICE. feature_weights is
        inert unless some colsample_by* < 1, so at XGBoost's default of 1.0 the whole parameter
        does nothing and would have been silently ignored. Per-node rather than per-tree because
        weights apply at each sampling event, which gives the reweighting far more chances to
        act.
        """
        cls = xgb.XGBClassifier if task in ("binary", "multiclass") else xgb.XGBRegressor
        kw = dict(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1, verbosity=0,
                  colsample_bynode=0.3, tree_method="hist")
        if fw is not None:
            kw["feature_weights"] = fw
        return cls(**kw)

    def score(task, yte, p):
        if task == "binary":
            return roc_auc_score(yte, p)
        if task == "multiclass":
            return accuracy_score(yte, np.rint(p).astype(int))
        return float(np.sqrt(np.mean((yte - p) ** 2)))

    def predict(m, task, X):
        return m.predict_proba(X)[:, 1] if task == "binary" else m.predict(X)

    records, t0 = [], time.time()
    for ds in datasets:
        sp = (_LocalSpec(ds, LOCAL_DATASETS[ds]["task"]) if ds in LOCAL_DATASETS
              else spec(ds))
        if sp.role not in ("DEV", "LOCAL") and ds not in ALLOW_NON_DEV:
            print(f"  SKIP {ds}: role {sp.role}, not DEV and not in ALLOW_NON_DEV", flush=True)
            continue
        d = load_ds(ds)
        smis, y = list(d["smiles"]), np.asarray(d["y"], dtype=np.float64)
        # CAP AT 50,000, matching grid_all.json exactly. Four datasets exceed it (qm9 133,885;
        # qm9_gap 133,885; qmugs_gap 665,911; rascore 199,348) and every existing record for
        # them carries n = 50,000. A new arm measured on the full set would not be comparable
        # with the base arms already in the grid -- and Mordred on 665,911 molecules alone is
        # ~60 core-hours, so this is a correctness requirement that happens to also be cheap.
        if len(smis) > 50_000:
            take = np.random.default_rng(seed).choice(len(smis), 50_000, replace=False)
            take.sort()
            smis = [smis[i] for i in take]; y = y[take]
        folds = scaffold_folds(smis, k=folds_k, seed=seed)
        task = d["task"]
        _BLOCK_CACHE.clear()
        feats = {}
        for a in arm_names:
            try:
                X = np.asarray(ARMS[a](smis), dtype=np.float32)
                # +/-inf -> NaN, WHICH XGBOOST TREATS AS MISSING AND +inf IT REFUSES OUTRIGHT.
                #
                # RDKit emits inf on real molecules -- Ipc overflows on larger graphs and the
                # partial-charge descriptors do it on a few odd valences -- and XGBoost then
                # raises `Check failed: valid: Input data contains 'inf' or a value too large`.
                # The per-fold except below caught it, so ALL FIVE FOLDS of that arm silently
                # vanished for that dataset and the run still looked complete.
                #
                # THAT HIT THE ANCHOR. `ecfp_all_desc` is the denominator of every ratio in
                # Figures B and C, so a dataset losing it drops out of the task average
                # ENTIRELY -- and only for the arms that happened to carry RDKit descriptors, so
                # different arms were being averaged over different dataset sets. Measured on the
                # first run: aqsoldb, fartdb, pb_bbb and pb_logd all lost desc, ecfp_rdkit_desc
                # and ecfp_all_desc, and aqsoldb/fartdb/cyp2d6_inh also lost hume.
                #
                # NaN rather than a clip: NaN is what "this descriptor has no value here" means,
                # and XGBoost routes it natively. Clipping would invent a number.
                n_bad = int((~np.isfinite(X)).sum() - np.isnan(X).sum())
                if n_bad:
                    print(f"  {ds}/{a}: {n_bad} non-finite cells -> NaN", flush=True)
                    X[~np.isfinite(X)] = np.nan
                feats[a] = X
            except Exception as e:
                print(f"  {ds}/{a}: FEATURISATION FAILED {type(e).__name__}: {e}", flush=True)
        for a, X in feats.items():
            for i in range(len(folds)):
                tr, te = train_test(folds, i)
                if task in ("binary", "multiclass") and len(np.unique(y[tr])) < 2:
                    continue
                if task == "regression" and float(np.std(y[tr])) == 0.0:
                    continue
                try:
                    # TUNE w ON TRAINING FOLDS ONLY. API.md: "Tuning it on the test fold leaks,
                    # and the effect is large enough to leak meaningfully." An inner 80/20 of the
                    # outer training set picks w; the final model is refit on the whole training
                    # set with the winner. Arms with no fingerprint/descriptor split skip this.
                    lower_better = task == "regression"
                    if FP_BLOCK.get(a) in (None, "all"):
                        best_w = 1.0
                    else:
                        # K-FOLD INNER CV, NOT ONE 80/20 SPLIT.
                        #
                        # The single inner split was fitting noise, and it was doing so at every
                        # dataset size. Counted over the first grid: the number of DISTINCT w
                        # values chosen across the five outer folds was 3 or 4 for the tuned arms
                        # on almost every dataset -- including ames at n=7,278 and pb_ames at
                        # n=9,139 -- where a tuner finding real signal would pick the same value
                        # each time.
                        #
                        # THE COST OF THAT FELL ON ONE SIDE OF THE COMPARISON. `feature_weights`
                        # is a no-op for an arm with no fingerprint/descriptor boundary, so every
                        # dense embedding (CheMeleon, MiniMol, ChemBERTa, MoLFormer) skipped the
                        # loop entirely and paid no variance for it, while every descriptor-
                        # carrying arm did. That is a systematic bias in favour of the dense arms
                        # across the whole grid, not a property of any representation.
                        #
                        # Averaging over K inner folds cuts that variance by ~K, and below a
                        # usable inner-validation size we do not guess at all: API.md section 7's
                        # documented default is used instead.
                        idx = np.asarray(tr)
                        if len(idx) // W_INNER_K < MIN_INNER_VAL:
                            best_w = W_UNTUNED
                        else:
                            best_w, best_s = W_UNTUNED, None
                            for w in W_GRID:
                                ss = []
                                for j in range(W_INNER_K):
                                    iva = idx[j::W_INNER_K]
                                    itr = np.concatenate([idx[q::W_INNER_K]
                                                          for q in range(W_INNER_K) if q != j])
                                    if len(iva) < 20 or (task in ("binary", "multiclass")
                                                         and len(np.unique(y[itr])) < 2):
                                        continue
                                    mi = make(task, fp_weights(a, X.shape[1], w))
                                    mi.fit(X[itr], y[itr])
                                    ss.append(score(task, y[iva], predict(mi, task, X[iva])))
                                if not ss:
                                    continue
                                si = float(np.mean(ss))
                                if best_s is None or (si < best_s if lower_better else si > best_s):
                                    best_w, best_s = w, si
                    m = make(task, fp_weights(a, X.shape[1], best_w))
                    m.fit(X[tr], y[tr])
                    v = score(task, y[te], predict(m, task, X[te]))
                except Exception as e:
                    print(f"  {ds}/{a}/fold{i}: {type(e).__name__}: {e}", flush=True)
                    continue
                records.append({"proto": PROTO, "dataset": ds, "fold": i, "arm": a,
                                "metric": {"binary": "auroc", "multiclass": "acc"}.get(
                                    task, "rmse"),
                                "value": float(v), "task": task, "n": len(smis),
                                "w": float(best_w)})
            got = [r["value"] for r in records if r["dataset"] == ds and r["arm"] == a]
            print(f"  {ds:<16s} {a:<20s} {np.mean(got):6.3f} over {len(got)} folds "
                  f"({time.time()-t0:6.0f}s)", flush=True)
            with open(out_path, "w") as fh:     # after every ARM, not every dataset
                json.dump(records, fh)
        with open(out_path, "w") as fh:
            json.dump(records, fh)
    print(f"  -> {out_path}  {len(records)} records in {time.time()-t0:.0f}s", flush=True)
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad = [x for x in a.arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known: {sorted(ARMS)}")
    run(a.arms, a.datasets, a.out)
