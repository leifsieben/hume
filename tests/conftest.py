"""Fixtures shared by the fast suite.

The whole of `tests/` runs in seconds against the committed fixture, with no corpus and no
mordred. The exactness verifications that need those live in the repo root as `verify_*.py`
and are run separately -- see tests/README.md.
"""
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def smiles():
    return DATA.joinpath("fixture_smiles.txt").read_text().split()


@pytest.fixture(scope="session")
def expected():
    with np.load(DATA / "fixture_expected.npz", allow_pickle=False) as z:
        return {"X": z["X"], "names": tuple(str(n) for n in z["names"]),
                "rdkit_version": str(z["rdkit_version"]), "n_heavy": z["n_heavy"],
                "platform": str(z["platform"]) if "platform" in z else "(not recorded)"}
