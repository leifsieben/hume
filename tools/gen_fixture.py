"""Regenerate the committed test fixture.

    .venv/bin/python tools/gen_fixture.py

Draws 200 molecules from the 42k exactness corpus, stratified by heavy-atom count so the
fixture spans the same size range the exactness numbers were measured over, and writes the
full 200 x 1269 descriptor matrix next to them.

The fixture is a REGRESSION net, not an oracle: it records what this build produces, so a
later change that moves a value has to be a deliberate act. Correctness against RDKit and
Mordred is what `verify_*.py` measure, and those need the corpus and a second environment.
Regenerate only when a value change is intended, and say so in CHANGELOG.md.
"""
import hashlib
import json
import sys

import platform

import numpy as np
from rdkit import Chem, RDLogger

import molhume

RDLogger.DisableLog("rdApp.*")

N = 200
STRATA = [(1, 10), (11, 17), (18, 24), (25, 34), (35, 10_000)]

corpus = json.load(open("data/exactness_corpus.json"))["smiles"]
rng = np.random.default_rng(20260831)

picked: list[str] = []
for lo, hi in STRATA:
    pool = []
    for s in corpus:
        m = Chem.MolFromSmiles(s)
        if m is not None and lo <= m.GetNumHeavyAtoms() <= hi:
            pool.append(s)
        if len(pool) >= 4000:
            break
    if len(pool) < N // len(STRATA):
        sys.exit(f"stratum {lo}-{hi} heavy atoms yielded only {len(pool)} molecules")
    take = rng.choice(len(pool), size=N // len(STRATA), replace=False)
    picked += [pool[i] for i in sorted(take)]

# Canonicalize so the committed SMILES are stable text, then featurize them as given.
picked = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in picked]
X = molhume.featurize(picked, standardize="none")
names = molhume.feature_names()
assert names == molhume.ALL_COLUMNS, "fixture must cover every emitted column"

with open("tests/data/fixture_smiles.txt", "w") as fh:
    fh.write("\n".join(picked) + "\n")
np.savez_compressed(
    "tests/data/fixture_expected.npz",
    X=X,
    names=np.array(names),
    rdkit_version=np.array(Chem.rdBase.rdkitVersion),
    # WHICH MACHINE THESE NUMBERS ARE FROM. This library reproduces upstream floating-point
    # behavior, so the architecture and the libm are part of the specification: the same source
    # on x86-64/gcc moves the last two or three digits of several hundred columns. The
    # regression test compares exactly here and within a measured tolerance elsewhere, and it
    # can only tell the two apart if the fixture says where it came from.
    platform=np.array(f"{platform.system()} {platform.machine()}"),
    n_heavy=np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in picked]),
)
digest = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16]
heavy = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in picked]
print(f"  {len(picked)} molecules, {X.shape[1]} columns, rdkit {Chem.rdBase.rdkitVersion}")
print(f"  heavy atoms {min(heavy)}-{max(heavy)}, non-finite {(~np.isfinite(X)).mean():.4%}")
print(f"  sha256[:16] {digest}")
