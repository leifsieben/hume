"""Alias for :mod:`molhume`, because the distribution is called ``mol-hume``.

``pip install mol-hume`` followed by ``import mol-hume`` is a **SyntaxError**: Python reads the
hyphen as subtraction. A distribution name and an import name are different things and only the
latter has to be a Python identifier, so no packaging mechanism can make the hyphen importable.

``mol_hume`` with an underscore is a valid identifier, and that is what this exists for. It
re-exports :mod:`molhume` unchanged, so ``import mol_hume as mh`` and ``import molhume as mh``
are interchangeable.

The canonical name is ``molhume``. This alias holds no state and is a thin re-export by design,
so there is no version of it that can drift from the package it points at.
"""

import molhume as _molhume
from molhume import *          # noqa: F401,F403
from molhume import __all__    # noqa: F401

__version__ = getattr(_molhume, "__version__", None)
