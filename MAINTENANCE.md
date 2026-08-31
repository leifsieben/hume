# Maintaining mol-hume

What breaks this package over time, how you find out, and what it costs to fix. Ordered by how
much attention each actually needs, not by how alarming it sounds.

---

## 1. RDKit, which is the one that matters

**There are two separate exposures here, and only one of them is gradual.**

### 1a. The pickle format: a hard wall, not a drift

`molpickle.h` reads RDKit's `MolPickler` blob **directly** — that is where a large part of the
speed comes from — and the pickle layout is explicitly not a stable API. The reader is pinned to
format **16.2.0** and refuses to import against anything else, because a silently misparsed
pickle is a wrong descriptor with no symptom. Measured, one probe per release:

| rdkit | pickle format | |
| --- | --- | --- |
| 2023.09.6 | 15.0.0 | no |
| 2024.03.6 | 16.1.0 | no |
| **2024.09.1** | 16.2.0 | **lower bound** |
| 2025.03.6 | 16.2.0 | yes |
| 2025.09.2 | 16.2.0 | yes — the exactness numbers are measured here |
| 2026.03.5 | 16.3.0 | yes — added after measuring, below |
| 2026.09.x | ? | **upper bound, unmeasured** |

`pyproject.toml` declares `rdkit>=2024.09.1,<2026.09`. Without a cap, `pip install mol-hume` on
a clean machine resolves whatever RDKit is newest and the package may not import at all. **This
is the single most likely reason a user cannot install it**, and it is how CI found the problem
in the first place: CI pins nothing, so the first three-platform run installed RDKit 2026.3.5
and failed identically on all three.

**A new format is not automatically a "no".** 16.3.0 was added on evidence, not on a reading of
the diff. The only wire-format change from 16.2.0 is `AtomMonomerInfo` — PDB residue fields moved
to the base class — and a SMILES-derived molecule never carries it, while one that does (a `Mol`
read from a PDB file and handed to `featurize`) is rejected by the reader on *both* formats. So
the changed region is one the reader errors out of before it could misread it. Measured: 4,000
corpus molecules pickled under both releases differ **only in the version triple**, 0 elsewhere;
and end to end, all 1,269 columns over 8,000 molecules are **bit-identical**.

**Run the check, do not repeat the investigation.** `tools/check_rdkit_release.py` performs all
three steps in throwaway venvs and tells you which of four situations you are in:

```bash
.venv/bin/python tools/check_rdkit_release.py 2026.09.1
```

If it reports the blobs differ beyond the version triple, the reader genuinely has to be updated:
re-read `Code/GraphMol/MolPickler.cpp`, update the tag table in `src/hume_core/molpickle.h`, add
the triple to `SUPPORTED`, and re-run `cpp/verify_molpickle.py` on both corpora plus the full
`verify_*.py` suite before widening the cap.

### 1b. Perceived properties: the gradual one

**The exposure.** `mol-hume` computes its descriptors itself, but it does not perceive the
molecule itself. RDKit parses the SMILES and supplies every input: element, degree, formal
charge, aromaticity, ring membership, bond order, hybridization, chirality. Those perceptions
are RDKit's model of chemistry, and **RDKit changes them across releases** — aromaticity of
unusual rings, ring-perception tie-breaks, new or corrected Crippen parameters, changes to what
counts as a hydrogen-bond donor. When one changes, this library's output changes with it, from
the same input SMILES, with no code change here.

So the honest statement is not "these values are correct" but **"these values are what RDKit
2025.9.2 implies"**. That is why `pyproject.toml` declares `rdkit>=2023.03` and not a pin: a
hard pin in a library fights the user's environment and would make `mol-hume` uninstallable
alongside anything else that has an opinion about RDKit. The version relativity belongs in the
documentation, and it is stated in the README.

**How you find out.** Two independent alarms, and you need both:

- `tests/test_regression.py` compares all 1,269 columns against a fixture and **skips rather
  than fails** when the RDKit differs from the one that generated it. That skip is the signal —
  a skipped regression test in CI means "nobody has checked this RDKit".
- The root-level `verify_*.py` are the real oracle. They need the 42k corpus and, for Mordred,
  a Python 3.11 environment.

**What to do on a new RDKit release** (roughly twice a year):

1. Install it in a scratch environment, run `pytest tests/` and read the skip.
2. Run `tools/gen_fixture.py` against it and diff the matrix against the committed one. That
   tells you *which columns moved and by how much*, in minutes, without the corpus.
3. If columns moved: run the relevant `verify_*.py` on the corpus to find out whether the new
   RDKit disagrees with *itself*-as-oracle (fine, we track it) or whether we have a bug.
4. Record the outcome in `CHANGELOG.md` — which RDKit was checked, which columns moved, by how
   much. **A version nobody checked and a version that was checked and was clean look identical
   in a changelog that only records failures.**

**The realistic failure.** Not a dramatic break. A handful of columns move in the last few ulps,
a couple of ring-count columns change on a few hundred exotic molecules, and nothing errors.
This is only detectable by comparison, never by exception. Hence the fixture.

---

## 2. Everything else you depend on, and why it is quiet

Worth knowing precisely, because the list is shorter than it looks.

| dependency | when | exposure |
| --- | --- | --- |
| **rdkit** | runtime | §1. The whole risk surface. |
| **numpy** | runtime | Almost none. `pybind11/numpy.h` resolves NumPy through the Python C API at runtime, so the extension carries **no NumPy ABI**: `otool -L` / `ldd` on `_core` shows only the C++ runtime. One wheel spans NumPy 1.x and 2.x, and the 2.0 ABI break that forced most compiled packages to rebuild did not apply here. |
| **pybind11** | build only | Never shipped. It matters when a new CPython appears (§3). |
| **scikit-build-core, CMake** | build only | Never shipped. |
| **cibuildwheel, manylinux image** | CI only | Bump when the manylinux baseline moves; keep it matched to what RDKit ships (§4). |
| BLAS / LAPACK | — | **None.** Removed deliberately; see the history in `CMakeLists.txt`. This is what makes the package buildable off macOS at all. |

---

## 3. New CPython versions

CPython releases every October and the ecosystem expects wheels within weeks. Adding one is
three edits — `build` in `[tool.cibuildwheel]`, the classifier list, and `requires-python` if a
floor moves — but it is gated on two things outside your control: **pybind11 supporting the new
C API**, and **RDKit publishing a wheel for it**. Until RDKit does, a `mol-hume` wheel for that
version installs and then cannot import anything useful, so there is no point shipping first.

Dropping an old version is the same edit. `requires-python = ">=3.11"` today.

---

## 4. Platforms, and why the matrix is what it is

The wheel matrix is pinned to what RDKit itself ships (README has the table). The rule is
simple: **build for a platform if and only if RDKit has a wheel there**, because a `mol-hume`
wheel without a matching RDKit is unusable. That excludes musllinux (Alpine), 32-bit, and PyPy.
When RDKit adds or drops a platform, follow it.

macOS x86_64 is the live exception: RDKit dropped it after 2025.9.2, and the matrix still builds
it, so an Intel Mac user pinning that RDKit does not also have to compile this from source.
Drop it when that stops being worth the CI minutes.

---

## 5. The thing that will actually surprise someone: compiler-dependent values

This library reproduces upstream floating-point *behavior*, not merely upstream *mathematics*,
and that makes the compiler part of the specification.

- `constit.h` splits every `a + b * c` across two statements **on purpose**. Clang contracts
  within an expression and not across statements; contracting produces a *more accurate* number
  and therefore a *different* one from the Python being reproduced. Measured: `Vabc` differs
  from Mordred on 43 of 300 molecules, `FilterItLogS` on 74, `qed` on 240.
- GCC defaults to `-ffp-contract=fast`, which contracts **across** statements and walks straight
  through that defense. `CMakeLists.txt` therefore says `-ffp-contract=on` explicitly. Without
  it, Linux wheels are quietly not the same library as macOS wheels.
- MSVC's `/fp:precise` does not contract at all, which is the `off` case that `hume_blocks.h`
  documents as making the resistance disagreement fifty times *worse*. `/fp:contract` restores
  expression-level contraction.

**Do not treat those flags as tuning.** If you change them, values move.

`tests/test_regression.py` is the guard: it compares exactly, so running it on each platform in
CI *is* the measurement of whether the three toolchains agree. Keep it exact. If a platform
genuinely cannot match, record the tolerance and the reason next to it rather than loosening the
comparison everywhere.

Related: `src/hume_core/u128.h` exists because MSVC has no `unsigned __int128`, and
`tests/cpp/test_u128.cpp` differentially tests the fallback against the native type in one
translation unit. That test is how a developer on macOS finds out they broke Windows.

---

## 6. Releasing

`.github/workflows/wheels.yml` has the procedure at the top. Publishing is **Trusted Publishing
(OIDC)**: no API token is stored in the repository or in Actions secrets, PyPI trusts the
repo + workflow + environment triple and mints a short-lived token per run. Rotating nothing is
the point.

Two rules that are cheap now and expensive later:

- **TestPyPI first, every time.** PyPI filenames are immutable — a bad `0.1.0` can be yanked but
  never replaced, and the version number is burned.
- **Version relativity in the changelog.** Every release should say which RDKit it was verified
  against, because that is the number that makes the exactness claim mean anything.

---

## 7. The fixture

`tests/data/fixture_expected.npz` records what the current build produces for 200 molecules
spanning 1–64 heavy atoms. It is a **regression net, not an oracle**: it does not know what is
correct, only what changed. Regenerate with `tools/gen_fixture.py` **only** when a value change
is intended, and say in `CHANGELOG.md` which columns moved and why. A value that moves without a
changelog entry is a bug that got committed.
