# Tests

Two tiers, split by what they need to run.

## Fast — `tests/`, no corpus, no mordred, seconds

```bash
.venv/bin/python -m pytest tests/
```

- `test_api.py` — every flag, what it does, and what it says when given nonsense.
- `test_regression.py` — all 1,269 columns for 200 molecules against a committed fixture.
- `test_invariance.py` — properties that must hold across thread count, batch size and row
  order. These are the screens for the failure modes threading actually produces: a shared
  scratch buffer, a cache without a lock, an offset that assumes one molecule per batch.

The fixture is a **regression net, not an oracle**: it records what this build produces, so a
change to a value has to be a deliberate act. It is 200 molecules drawn from the 42k exactness
corpus, stratified across five heavy-atom bands so it spans 1 to 64 heavy atoms. Regenerate with
`.venv/bin/python tools/gen_fixture.py` and record the reason in `CHANGELOG.md`.

`test_regression.py` **skips rather than fails** when the environment's RDKit differs from the
one the fixture was generated against, since perceived atom and bond properties drift across
releases and the difference would not be a regression.

## Slow — root-level `verify_*.py`, needs the corpus and a second environment

`verify_counts.py`, `verify_estate.py`, `verify_eta.py`, `verify_misc.py`, `verify_spectral.py`,
`verify_sps.py` and `verify.py` are the exactness checks against RDKit and Mordred. They read the
42k corpus from `data/`, and the Mordred comparison needs a Python 3.11 environment (mordred
1.2.0 imports `distutils`). They are what the exactness numbers in `README.md` and `METHODS.md`
come from, and they are not part of the fast suite.
