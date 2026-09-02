"""THE NUMBERS IN THE DOCS ARE CHECKED, NOT TRUSTED.

Every count in the prose is a claim that rots the moment the column set moves, and it has moved
four times: 1,266 -> 1,536 -> 1,269 emitted, and 550 -> 612 -> 622 for `minimal`. The rot is
silent -- a stale "1,266 columns, 14 families" reads exactly like a current one -- and it was
found by a human asking "are the docs up to date", which is not a mechanism.

This checks the live package against the docs that make current-tense claims. It does NOT check
CHANGELOG.md, API.md's superseded-specification table, docs/MINIMAL_SPEC.md or PACKAGING.md's
captured transcripts: those record what WAS true, and a test that forced them to say what is
true now would destroy the record it is meant to protect.
"""
import pathlib
import re

import pytest

import molhume

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Docs that describe the package AS IT IS. A stale number in one of these is a bug.
CURRENT = ["README.md", "MAINTENANCE.md", "METHODS.md", "docs/HUME_DESCRIPTORS.md",
           "docs/DESCRIPTOR_MAP.md", "HANDOVER_minimal_v2.md"]

#: Counts that have been wrong in the docs before, with what they should be now.
STALE = {
    "1,266": 'the emitted set before it grew; it is now 1,269',
    "1266": 'the emitted set before it grew; it is now 1,269',
    "1,536": 'the emitted set before the dedup filter landed; it is now 1,269',
}


@pytest.mark.parametrize("doc", CURRENT)
def test_no_superseded_column_counts(doc):
    p = ROOT / doc
    if not p.exists():
        pytest.skip(f"{doc} is gone")
    # PARAGRAPH-SCOPED, NOT LINE-SCOPED. A note that says "this used to read 1,266" is the fix
    # and not the bug, and the disclaimer is rarely on the same line as the number -- the first
    # version of this test flagged its own correction notice for exactly that reason.
    lines = p.read_text().splitlines()
    paras, start = [], 0
    for i, ln in enumerate(lines):
        if not ln.strip():
            if i > start:
                paras.append((start + 1, lines[start:i]))
            start = i + 1
    if start < len(lines):
        paras.append((start + 1, lines[start:]))
    bad = []
    for first, block in paras:
        low = "\n".join(block).lower()
        if any(w in low for w in ("used to", "superseded", "no longer", "since gone",
                                  "earlier", "historic", "withdrawn")):
            continue
        for j, line in enumerate(block):
            for n, why in STALE.items():
                if n in line:
                    bad.append(f"{doc}:{first + j}: {n!r} -- {why}\n      {line.strip()[:110]}")
    assert not bad, "superseded column counts stated as current:\n    " + "\n    ".join(bad)


def test_the_live_counts_are_what_the_docs_claim():
    """The numbers the prose is allowed to use, read off the package."""
    assert len(molhume.column_set("full")) == 1269
    assert len(molhume.column_set("full_no_new")) == 1109
    assert len(molhume.column_set("minimal")) == 622
    assert len(molhume.ALL_COLUMNS) == 1270
    fams = [f for f, (a, b) in molhume.FAMILY_OFFSETS.items()
            if b > a and f not in molhume.OPTIONAL_COLUMNS]
    assert len(fams) == 19, (
        f"{len(fams)} descriptor families, but README and docs/HUME_DESCRIPTORS.md both say "
        "nineteen. Update the prose or explain the new one.")


@pytest.mark.parametrize("doc", CURRENT)
def test_no_pre_rename_package_name(doc):
    """`hume` has not been an importable package since the rename to `molhume`."""
    p = ROOT / doc
    if not p.exists():
        pytest.skip(f"{doc} is gone")
    bad = [f"{doc}:{i}: {ln.strip()[:110]}"
           for i, ln in enumerate(p.read_text().splitlines(), 1)
           if re.search(r"(?<![-\w.])hume\.(featurize|ALL_COLUMNS|feature_names|column_set)", ln)
           or re.search(r"^\s*import hume\s*$", ln)]
    assert not bad, ("the pre-rename package name is used as if it still worked:\n    "
                     + "\n    ".join(bad))


def test_minimal_spec_v1_is_marked_superseded():
    """It documents a withdrawn spec and several files still cite its sections."""
    head = (ROOT / "docs" / "MINIMAL_SPEC.md").read_text()[:1500]
    assert "SUPERSEDED" in head and "minimal-v1" in head, (
        "docs/MINIMAL_SPEC.md describes the withdrawn v1 ordering. Without a banner it reads as "
        "the method behind the shipped set, which is what it says in its own first line.")
