# Releasing mol-hume

The build and upload are automated. Two steps are not, and cannot be: they need your PyPI
account.

---

## One-time setup (you, not CI)

Publishing uses **Trusted Publishing (OIDC)**, so no API token is stored in this repository or
in Actions secrets. PyPI is told to trust a specific repo + workflow + environment, and mints a
short-lived credential per run. Nothing to rotate, nothing to leak.

**1. Register the publisher on PyPI.** At <https://pypi.org/manage/account/publishing/>, add a
pending publisher:

| field | value |
| --- | --- |
| PyPI project name | `mol-hume` |
| Owner | `leifsieben` |
| Repository name | `hume` |
| Workflow name | `wheels.yml` |
| Environment name | `pypi` |

**2. Do the same on TestPyPI**, at <https://test.pypi.org/manage/account/publishing/>, with
environment name `testpypi`.

**3. Create the two environments on GitHub**, at
<https://github.com/leifsieben/hume/settings/environments>: one named `pypi`, one named
`testpypi`. They need no secrets — the name is the thing PyPI matches against. Adding a required
reviewer to `pypi` is worth it: it turns an accidental tag into a prompt rather than a release.

The name `mol-hume` is claimed by whoever uploads first. If you want to hold it before the code
is ready, that is what step 1's "pending publisher" does.

---

## Every release

1. **Check RDKit.** If a release has appeared since the last one:
   ```bash
   .venv/bin/python tools/check_rdkit_release.py 2026.09.1
   ```
   It reports whether the cap in `pyproject.toml` can move. See `MAINTENANCE.md` section 1.

2. **Bump the version** in `pyproject.toml` and write the `CHANGELOG.md` entry. Say which RDKit
   the release was verified against — that number is what makes the exactness claim mean
   anything.

3. **Dry run the whole matrix**, without publishing:
   Actions → wheels → Run workflow → `publish_to: none`. Every wheel is tested on the machine
   that built it, and the sdist job builds the sdist back into a wheel and runs the suite
   against it.

4. **TestPyPI.** Actions → wheels → Run workflow → `publish_to: testpypi`. Then install it
   somewhere clean and check it actually works:
   ```bash
   uv venv /tmp/tp && uv pip install --python /tmp/tp/bin/python \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ mol-hume
   /tmp/tp/bin/python -c "import molhume; print(molhume.featurize(['CCO'], standardize='none').shape)"
   ```
   The extra index is needed because RDKit is not on TestPyPI.

5. **Release.**
   ```bash
   git tag v0.1.0 && git push --tags
   ```
   The tag runs the matrix and publishes to PyPI.

---

## Things that are cheap now and expensive later

- **PyPI filenames are immutable.** A bad `0.1.0` can be yanked but never replaced; the version
  number is burned. This is the entire reason step 4 exists.
- **The version cap on RDKit is load-bearing.** Without it `pip install mol-hume` resolves the
  newest RDKit and the package may not import at all. Do not widen it without running
  `tools/check_rdkit_release.py`.
- **CI minutes are billed** — this repository is private, macOS bills at 10x and Windows at 2x.
  A full matrix run is on the order of 500 billed minutes against a 2,000/month allowance, which
  is why `ci.yml` runs one platform on push and the matrix is manual. Budget roughly three full
  release runs per month, or make the repository public, where all of it is free.
- **`linux-aarch64` is emulated** under QEMU, since a private repo has no free arm runner. It is
  by far the slowest leg. If a release is urgent it can be dropped from the matrix without
  touching the other four.
