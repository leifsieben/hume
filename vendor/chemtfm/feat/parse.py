"""SMILES → RDKit Mol, perception only (roadmap §3).

This is the *entire* input pipeline for M1. It embodies the masterplan's sharpest data
stance: **perception yes, transformation no**. We read the molecule as written and never
change which molecule it is — no salt stripping, no neutralisation, no canonical tautomer,
no parent extraction. A different counter-ion is a different molecule; standardisation is
the user's job, not ours. (Contrast the retired ``chem/standardize.py``, which did all of
those transformations and is now superseded by this file.)

WHAT ``sanitize=True`` IS AND ISN'T
-----------------------------------
``Chem.MolFromSmiles`` sanitises by default, and we keep it on. Sanitisation is *perception*
— it computes ring membership, aromaticity, implicit valence and hybridisation from the graph
as given. It does not rewrite the molecule. Turning it off does not give us a "rawer" molecule;
it gives us a molecule whose aromatic atoms are mis-typed and whose ring bits are missing, so
every downstream fingerprint is silently wrong. A silent wrong answer is worse than an error,
so the floor is: sanitise, or fail loudly to NaN.

We deliberately do **not** kekulise — Morgan, atom-pair, ErG and the descriptors all read the
aromatic form directly, so kekulisation would be work that changes nothing we consume.

THE FAILURE LADDER (roadmap §3)
-------------------------------
One bad SMILES must never crash a billion-molecule virtual screen, so failures degrade to a
NaN feature vector that propagates harmlessly. The ladder:

1. ``MolFromSmiles(smi)`` with full sanitisation. Handles ~99% of real input.
2. On ``None``: **salvage** — reparse with ``sanitize=False`` and re-run sanitisation while
   skipping *only* the single operation that failed. This recovers the common odd-valence
   cases (hypervalent halogens like ClF3, over-substituted carbons) where the graph is
   perfectly usable but RDKit's default valence model rejects it. We skip the minimum
   necessary, never the whole sanitisation.
3. On failure: warn once and return a molecule of ``None`` → the caller emits a NaN row.
4. ``strict=True`` disables step 2 — parse fails go straight to NaN, no salvage. This exists
   for callers who would rather drop weird input than featurise a partially-perceived graph.

The salvage step reparses from the original SMILES rather than mutating the half-sanitised mol
in place, because ``SanitizeMol`` with ``catchErrors`` can leave the molecule in a partially
updated state, and we want the retry to start from a clean slate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from rdkit import Chem, RDLogger

# RDKit logs every rejected SMILES to stderr at C++ level. In a large screen that is millions
# of lines of noise; we surface failures through the returned status instead, and (once) a
# Python warning. Callers who want the RDKit chatter back can re-enable it.
RDLogger.DisableLog("rdApp.*")


# Parse outcomes. Kept as plain strings (not an enum) so they serialise straight into a
# results manifest and read clearly in a log without an import.
OK = "ok"  # clean parse, full sanitisation succeeded
SALVAGED = "salvaged"  # parsed only after skipping the one failing sanitisation op
FAILED = "failed"  # unparseable or unsalvageable → NaN feature vector downstream


@dataclass(frozen=True)
class ParseResult:
    """The outcome of parsing one SMILES.

    ``mol`` is ``None`` exactly when ``status == FAILED``. ``detail`` carries the failing
    sanitisation operation for salvaged/failed cases, for diagnostics — it is never used to
    make decisions, only to report.
    """

    smiles: str
    mol: Chem.Mol | None
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True if we have a usable molecule (clean *or* salvaged)."""
        return self.mol is not None


def parse_smiles(smi: str, strict: bool = False, warn: bool = True) -> ParseResult:
    """Parse one SMILES to a perceived molecule, following the failure ladder (§3).

    Parameters
    ----------
    smi
        The SMILES string, exactly as the user supplied it. Not stripped, not canonicalised.
    strict
        If True, skip the salvage step (ladder rung 2): a molecule that fails full
        sanitisation goes straight to ``FAILED``.
    warn
        Emit a Python ``warning`` on salvage and on failure. Default on; callers featurising
        millions of molecules will usually pass ``warn=False`` and inspect statuses in bulk.
    """
    # Rung 1: strict parse with full sanitisation (perception). The common path.
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return ParseResult(smi, mol, OK)

    if strict:
        if warn:
            warnings.warn(f"unparseable SMILES (strict): {smi!r}", stacklevel=2)
        return ParseResult(smi, None, FAILED, "strict: no salvage")

    # Rung 2: salvage. Reparse without sanitisation, then sanitise skipping only the op that
    # fails. ``UpdatePropertyCache(strict=False)`` computes valences without raising on the
    # odd ones, which is what lets the subsequent partial sanitisation succeed.
    raw = Chem.MolFromSmiles(smi, sanitize=False)
    if raw is None:
        # Not a valence problem — the SMILES itself is malformed. Nothing to salvage.
        if warn:
            warnings.warn(f"unparseable SMILES: {smi!r}", stacklevel=2)
        return ParseResult(smi, None, FAILED, "unparseable")

    raw.UpdatePropertyCache(strict=False)
    # ``catchErrors=True`` makes SanitizeMol return the flag of the first failing operation
    # instead of raising. SANITIZE_NONE (0) means everything passed.
    failed_op = Chem.SanitizeMol(raw, catchErrors=True)
    if failed_op == Chem.SanitizeFlags.SANITIZE_NONE:
        # Full sanitisation actually succeeded on the reparse. (Rare, but it means the
        # molecule is clean; treat it as salvaged so the caller can see it took the slow path.)
        return ParseResult(smi, raw, SALVAGED, "clean on reparse")

    # Reparse fresh and sanitise with every op EXCEPT the one that failed. Skipping the
    # minimum keeps as much perception as possible.
    mol2 = Chem.MolFromSmiles(smi, sanitize=False)
    mol2.UpdatePropertyCache(strict=False)
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ failed_op
    residual = Chem.SanitizeMol(mol2, sanitizeOps=ops, catchErrors=True)
    if residual == Chem.SanitizeFlags.SANITIZE_NONE:
        if warn:
            warnings.warn(f"salvaged SMILES (skipped {failed_op!r}): {smi!r}", stacklevel=2)
        return ParseResult(smi, mol2, SALVAGED, f"skipped {failed_op!r}")

    # A second operation also failed — skipping one wasn't enough. We stop here rather than
    # peeling off operations until something parses; past one skip we no longer trust the
    # perception, and NaN is the honest answer.
    if warn:
        warnings.warn(f"unsalvageable SMILES (failed {residual!r}): {smi!r}", stacklevel=2)
    return ParseResult(smi, None, FAILED, f"unsalvageable: {residual!r}")
