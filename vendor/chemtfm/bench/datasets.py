"""Datasets: a uniform container, a CSV loader, downsampling, and a smoke-test fixture.

The real M1 deliverable runs on OpenADMET / MoleculeNet / MoleculeACE / LIT-PCBA. Those are
downloaded CSVs (SMILES column + label column), so the loader here is a dependency-free CSV
reader — no pandas (masterplan: minimal deps). Point it at a file, name the two columns, declare
the task, and it produces a ``Dataset``.

``downsample`` implements the roadmap's data-size study: rerun the whole benchmark on shrunk
copies to see whether dataset size changes which feature combination wins (it usually does — the
low-data regime is where the PFN is meant to pay off).

``demo_dataset`` is a **plumbing fixture**, not science: ~50 real drug-like SMILES with a label
synthesised deterministically from structure, so the smoke test can exercise the full harness
(parse → featurise → scaffold-CV → XGBoost → metrics) offline with real learnable signal. It is
clearly *not* a benchmark and must never be reported as one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from chemtfm.feat.parse import parse_smiles

# Task types. Regression is primary for the cliff thesis; binary is the deployment-realism
# check. (Ranking is a *metric view* on a regression target, not a stored task type — see
# metrics.ranking_metrics — so it is not listed here.)
REGRESSION = "regression"
BINARY = "binary"
TASKS = (REGRESSION, BINARY)


@dataclass(frozen=True)
class Dataset:
    """A benchmark dataset: aligned SMILES and labels, plus the task type.

    ``smiles[i]`` and ``y[i]`` are the same molecule. ``y`` is float64; for a binary task it is
    0.0/1.0. SMILES are stored exactly as given (no standardisation — roadmap §3).
    """

    name: str
    smiles: list[str]
    y: np.ndarray
    task: str

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {TASKS}")
        if len(self.smiles) != len(self.y):
            raise ValueError(f"smiles/y length mismatch: {len(self.smiles)} vs {len(self.y)}")

    def __len__(self) -> int:
        return len(self.smiles)


def from_csv(
    path: str,
    name: str,
    smiles_col: str,
    label_col: str,
    task: str,
    *,
    validate: bool = True,
) -> Dataset:
    """Load a dataset from a CSV with a SMILES column and a label column.

    ``validate`` (default on) drops rows whose SMILES cannot be parsed *at all* — but only after
    reporting how many, so a silently-corrupt file is visible. Note this is a stricter drop than
    the featurisation ladder (which keeps failures as NaN rows); for a *labelled benchmark* it is
    cleaner to exclude a molecule we cannot represent than to score the head on an all-NaN row.
    """
    smiles: list[str] = []
    labels: list[float] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or smiles_col not in reader.fieldnames:
            raise KeyError(f"column {smiles_col!r} not in {reader.fieldnames}")
        if label_col not in reader.fieldnames:
            raise KeyError(f"column {label_col!r} not in {reader.fieldnames}")
        for row in reader:
            smi = (row[smiles_col] or "").strip()
            raw = (row[label_col] or "").strip()
            if not smi or not raw:
                continue
            try:
                labels.append(float(raw))
            except ValueError:
                continue
            smiles.append(smi)

    if validate:
        keep_smiles: list[str] = []
        keep_labels: list[float] = []
        n_dropped = 0
        for smi, lab in zip(smiles, labels):
            if parse_smiles(smi, warn=False).mol is not None:
                keep_smiles.append(smi)
                keep_labels.append(lab)
            else:
                n_dropped += 1
        if n_dropped:
            print(f"[{name}] dropped {n_dropped} unparseable SMILES of {len(smiles)}")
        smiles, labels = keep_smiles, keep_labels

    return Dataset(name, smiles, np.asarray(labels, dtype=np.float64), task)


def downsample(dataset: Dataset, n: int, seed: int = 0) -> Dataset:
    """Return a random ``n``-molecule subset (the data-size study).

    Plain uniform subsample with a fixed seed. If ``n >= len(dataset)`` the dataset is returned
    unchanged. Downsampling is uniform, not scaffold-stratified, on purpose: the scaffold
    structure of the *subset* is what the subsequent scaffold-CV split should see, and
    stratifying here would leak split structure into the sampling.
    """
    if n >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    return Dataset(
        f"{dataset.name}@{n}",
        [dataset.smiles[i] for i in idx],
        dataset.y[idx],
        dataset.task,
    )


def binarize(dataset: Dataset, quantile: float = 0.5) -> Dataset:
    """Turn a regression dataset into a binary one by thresholding at a quantile.

    The roadmap wants the *same* dataset evaluated under regression and binary labels (binary
    makes everything a step function; the τ threshold is a knob to sweep). Actives = values at
    or above the ``quantile`` of ``y``. Always report which quantile was used, or the binary
    number is uninterpretable (roadmap §6).
    """
    if dataset.task != REGRESSION:
        raise ValueError("binarize expects a regression dataset")
    thresh = float(np.quantile(dataset.y, quantile))
    yb = (dataset.y >= thresh).astype(np.float64)
    return Dataset(f"{dataset.name}[bin@q{quantile}]", list(dataset.smiles), yb, BINARY)


# --------------------------------------------------------------------------- smoke fixture
# ~50 real drug-like SMILES spanning scaffolds (so scaffold-CV actually has >1 group per fold).
# Plumbing fixture only.
_DEMO_SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "OC(=O)c1ccccc1O",  # salicylic acid
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",  # naproxen
    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",  # amitriptyline
    "Clc1ccccc1C1=NCC(=O)Nc2ccc(cc12)[N+](=O)[O-]",  # clonazepam-like
    "CC(C)NCC(O)COc1cccc2ccccc12",  # propranolol
    "CCOC(=O)c1ccccc1",  # ethyl benzoate
    "c1ccc(cc1)c1ccccc1",  # biphenyl
    "c1ccc2ccccc2c1",  # naphthalene
    "c1ccc2c(c1)cccc2",  # naphthalene (alt)
    "O=C(O)c1ccc(O)cc1",  # 4-hydroxybenzoic acid
    "O=C(O)c1ccc(N)cc1",  # 4-aminobenzoic acid
    "Nc1ccc(cc1)S(=O)(=O)N",  # sulfanilamide
    "CC(=O)Nc1ccc(cc1)S(=O)(=O)N",  # acetylsulfanilamide
    "OCC1OC(O)C(O)C(O)C1O",  # glucose-like
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",  # galactose-like
    "c1ccncc1",  # pyridine
    "c1ccc(cc1)O",  # phenol
    "c1ccc(cc1)N",  # aniline
    "c1ccc(cc1)Cl",  # chlorobenzene
    "c1ccc(cc1)F",  # fluorobenzene
    "Cc1ccccc1",  # toluene
    "Cc1ccc(C)cc1",  # p-xylene
    "COc1ccccc1",  # anisole
    "CCc1ccccc1",  # ethylbenzene
    "c1ccc(cc1)C(=O)O",  # benzoic acid
    "c1ccc(cc1)C=O",  # benzaldehyde
    "c1ccc(cc1)C(=O)C",  # acetophenone
    "O=C1CCCCC1",  # cyclohexanone
    "OC1CCCCC1",  # cyclohexanol
    "C1CCCCC1",  # cyclohexane
    "c1ccc2[nH]ccc2c1",  # indole-like
    "c1ccc2c(c1)[nH]cn2",  # benzimidazole
    "c1ccc2c(c1)nccn2",  # quinoxaline
    "c1ccc2ncccc2c1",  # quinoline
    "c1ccc2occc2c1",  # benzofuran
    "c1ccc2sccc2c1",  # benzothiophene
    "O=C1c2ccccc2C(=O)c2ccccc21",  # anthraquinone
    "Nc1ncnc2[nH]cnc12",  # adenine
    "Cc1cc(=O)[nH]c(=O)[nH]1",  # thymine
    "O=c1cc[nH]c(=O)[nH]1",  # uracil
    "Nc1cc[nH]c(=O)n1",  # cytosine-like
    "C(=O)(O)CCCCCCC",  # octanoic acid
    "CCCCCCCCO",  # octanol
    "CCCCCCCC",  # octane
    "OC(=O)CCc1ccccc1",  # hydrocinnamic acid
)


def demo_dataset(task: str = REGRESSION) -> Dataset:
    """A tiny offline fixture for smoke-testing the harness end-to-end.

    The label is synthesised **deterministically from structure** — a smooth function of a few
    RDKit properties, so XGBoost has genuine signal to fit and the metrics come out meaningful —
    but it is not a measured property and this dataset is not a benchmark. Purpose: prove the
    pipeline runs and produces sane numbers without any network access.
    """
    from rdkit.Chem import Descriptors  # local import; keeps module import light

    smiles = list(_DEMO_SMILES)
    y = np.empty(len(smiles), dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = parse_smiles(smi, warn=False).mol
        # Smooth structural function: MW and aromatic-ring count. Deterministic, no noise.
        y[i] = 0.01 * Descriptors.MolWt(mol) + 0.5 * Descriptors.NumAromaticRings(mol)
    ds = Dataset("demo", smiles, y, REGRESSION)
    return ds if task == REGRESSION else binarize(ds, 0.5)
