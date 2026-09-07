# Verification

The Python reference implementations, and the scripts that compare the C++ against them.

`verification/blocks.py`, `verification/chi.py`, `verification/conjugation.py`, `verification/cycles.py`, `verification/resistance.py`, `verification/stereo.py` are ports of
the descriptor definitions written to be obviously correct rather than fast. They are the oracle
every exactness claim in `METHODS.md` is measured against; nothing in the shipped package imports
them.

`verify_*.py` here and in `cpp/` run the comparison. They need this directory on the path and the
corpus, which is not in the repository:

```bash
PYTHONPATH=verification .venv/bin/python verification/verify_counts.py
PYTHONPATH=verification .venv/bin/python cpp/verify_hume.py
```

The fast tests that run on every commit are `tests/`, against a committed fixture. These are the
slow, complete checks and need a second environment for Mordred; see `docs/MAINTENANCE.md`.
