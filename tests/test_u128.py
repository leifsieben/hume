"""The two-limb 128-bit fallback, against `unsigned __int128` itself.

MSVC has no `__int128`, so `MolecularId` needs a fallback -- and a fallback nobody can run is a
fallback nobody has checked. This compiles both paths in one translation unit and compares them
on the operations MolecularId performs, which is the only way a developer on macOS or Linux ever
exercises the code that Windows users will actually run.

Skipped where there is no C++ compiler on PATH, and where there is no native `__int128` to
compare against -- on MSVC itself there is nothing to differential-test.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent / "cpp" / "test_u128.cpp"


@pytest.mark.skipif(sys.platform == "win32", reason="no native __int128 to compare against")
def test_fallback_matches_native_int128(tmp_path):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    exe = tmp_path / "u128"
    build = subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe), str(SRC)],
                           capture_output=True, text=True)
    if build.returncode != 0:
        pytest.fail(f"{cxx} could not build the fallback test:\n{build.stderr[-2000:]}")
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, (
        "the two-limb fallback disagrees with unsigned __int128, so a Windows wheel would "
        f"produce different MolecularId values:\n{run.stdout}")
