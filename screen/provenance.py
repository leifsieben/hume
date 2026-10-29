"""What produced a screen record. Written into every output file.

WHY THIS EXISTS RATHER THAN A VERSION STRING. The r=4 default ships in the working tree before
it ships on PyPI, so `importlib.metadata.version("mol-hume")` is the wrong thing to stamp: an
editable install reports whatever was in `pyproject.toml` when it was installed, which during
this work read `1.0.1` -- a version that exists on PyPI and means radius 3. A record stamped
`mol-hume 1.0.1` whose features are radius 4 is worse than one with no version at all, because
it looks checkable and is wrong. So the authority here is the git commit plus the EFFECTIVE
default read back out of the installed module, and `require_fp_radius` fails the run rather than
let a disagreement through.

`pyproject.toml` carries `1.1.0.dev0` for the same reason: while the radius change is unreleased,
nothing this repo installs can claim to be a release that behaves differently.
"""
from __future__ import annotations

import importlib.metadata as _md
import inspect, json, platform, subprocess, sys
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def effective_fp_radius() -> int:
    """The default radius `molhume.featurize` will actually use, read from the live signature."""
    import molhume
    return int(inspect.signature(molhume.featurize).parameters["fp_radius"].default)


def require_fp_radius(expected: int) -> int:
    """Fail the run if the installed module would not use `expected`."""
    got = effective_fp_radius()
    if got != expected:
        raise RuntimeError(
            f"this run is defined at ECFP radius {expected} but the installed molhume defaults "
            f"to {got}. Featurizing now would label radius-{got} features as radius-{expected}. "
            f"Install the intended tree ("
            f"uv pip install -e . --python .venv/bin/python -c constraints.txt) or pass the "
            f"radius explicitly, then re-run.")
    return got


def _ver(mod: str) -> str | None:
    """Version of an installed module, or None if it is not installed here.

    Phase 0 of the screen reads parquet and writes SMILES: it needs neither molhume, xgboost nor
    RDKit, and making those a hard import would force a twenty-minute featurization stack onto a
    box that only moves bytes. Absence is recorded as null rather than guessed or crashed on --
    but see `expected_fp_radius`, which is still strict wherever features are actually computed.
    """
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _bundle() -> dict:
    """Provenance baked in at bundle-staging time, for boxes that have no git checkout.

    A box unpacks a tarball, so `git rev-parse` there returns nothing and the commit -- the one
    field that says WHICH CODE RAN -- came out null. scripts/stage_screen.sh writes this file
    next to the modules it bundles, so the record survives the trip. On a laptop with a real
    checkout the live git answer wins; this is the fallback, and it is labelled as such.
    """
    f = Path(__file__).resolve().parent / "BUNDLE_PROVENANCE.json"
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def collect(expected_fp_radius: int | None = None) -> dict:
    """Everything needed to say what produced a record. Cheap; call it per output file."""
    sha, dirty = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    b = _bundle()
    staged = None
    if not sha and b:
        sha, staged = b.get("git_commit"), b.get("staged_at")
        dirty = "\n".join(b.get("git_dirty_files", []))
    if expected_fp_radius is not None:
        radius = require_fp_radius(expected_fp_radius)
    else:
        try:
            radius = effective_fp_radius()
        except Exception:
            radius = None          # molhume not installed; nothing here computes a fingerprint
    try:
        mh_version = _md.version("mol-hume")
    except Exception:
        mh_version = None
    return {
        "git_commit": sha,
        # Set when the commit came from the bundle rather than a live checkout, so a reader can
        # tell "this is the code that was staged" from "this is the code that is checked out".
        # The TIME the bundle was staged, not a commit -- it was called git_commit_from_bundle
        # and read as a commit field holding a timestamp. The commit itself is in git_commit
        # above, which takes the bundle's value when there is no live checkout.
        "bundle_staged_at": staged,
        "git_dirty": bool(dirty),
        "git_dirty_files": sorted(l.split(maxsplit=1)[-1] for l in dirty.splitlines()) if dirty else [],
        # The installed dist version is recorded but is NOT the authority -- see the module note.
        "mol_hume_dist_version_untrusted": mh_version,
        "fp_radius_effective": radius,
        "rdkit": _ver("rdkit"),
        "numpy": _ver("numpy"),
        "pyarrow": _ver("pyarrow"),
        "xgboost": _ver("xgboost"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=1))
