"""Make the vendored third-party packages under ``vendor/`` importable.

Import this module for its side effect, immediately before any ``chemtfm`` import:

    import _vendor  # noqa: F401  — puts vendor/chemtfm on sys.path
    from chemtfm.bench import metrics as M

``chemtfm`` used to be imported from a sibling checkout (``/Users/lsieben/VSCode/ChemTFM_OLD``)
via ``PYTHONPATH`` or an inline ``sys.path.insert``. It is now vendored into this repo under
``vendor/chemtfm``; see ``vendor/README.md``. The package keeps its original name so no call
site's import statement had to change.

Idempotent by module caching: the ``sys.path`` edit happens once per interpreter no matter how
many times this is imported, which matters because most call sites import it inside a helper
that runs once per CV fold.

Requires the repo root on ``sys.path``, which is automatic when a script here is run as
``python <script>.py`` (``sys.path[0]`` is the script's directory). ``multiprocessing`` under
the macOS ``spawn`` start method copies the parent's ``sys.path`` into the child, so worker
processes resolve this too.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")

if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)
