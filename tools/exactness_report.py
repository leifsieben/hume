"""One coherent exactness report over the SHIPPED column set.

    .venv/bin/python tools/exactness_report.py

WHY THIS FILE EXISTS. METHODS.md section 7 quotes exactness figures that no script in this repo
reproduces -- the analysis was done ad hoc and only its conclusions were written down. That makes
the central correctness claim of the package unauditable, and it also meant the numbers were
computed from `data/exactness_hume.npz`, the 1,540-column PRE-EMIT-FILTER row, rather than from
the 1,269 columns `molhume.featurize` actually returns.

This recomputes them from `data/exactness_hume3.npz` (the shipped layout) against the same stored
RDKit and mordred references, and additionally audits COVERAGE -- which selected columns are
emitted at all -- because "the values are right" and "the right columns are there" are different
claims and this project has already conflated them once (docs/HUME_DESCRIPTOR_GAP.md).

MATCHING IS CASE-INSENSITIVE AGAINST RDKIT AND CASE-SENSITIVE AGAINST MORDRED. RDKit's chi family
is registered lowercase here (`chi0n` against RDKit's `Chi0n`) and a case-sensitive diff once
produced a phantom nine-column gap. Mordred's ring counts use case to MEAN something -- `naRing`
is aromatic and `nARing` is aliphatic -- so folding case there would silently pair the wrong
columns. The two rules are different on purpose.

A cell counts as agreeing if it is bit-identical, or both sides are NaN, or the relative
difference is within RTOL. A NaN on one side only is a disagreement and is counted separately,
because it is a structural difference rather than a rounding one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "exactness_report.json"
RTOL = 1e-9
HUME_NPZ = "exactness_hume3.npz"


def load(name):
    z = np.load(DATA / name, allow_pickle=True)
    return {k: z[k] for k in z}


def compare(H, hnames, R, rnames, fold_case: bool):
    """Per-column agreement between two (n_mol, n_col) float matrices."""
    key = (lambda s: s.lower()) if fold_case else (lambda s: s)
    rindex = {}
    for j, n in enumerate(rnames):
        rindex.setdefault(key(str(n)), j)

    rows = []
    for i, n in enumerate(hnames):
        j = rindex.get(key(str(n)))
        if j is None:
            continue
        a, b = H[:, i].astype(np.float64), R[:, j].astype(np.float64)
        na, nb = np.isnan(a), np.isnan(b)
        both_nan = na & nb
        one_nan = na ^ nb
        finite = ~(na | nb)
        bit = np.zeros(len(a), bool)
        bit[finite] = a[finite] == b[finite]
        close = bit.copy()
        with np.errstate(all="ignore"):
            denom = np.maximum(np.abs(a), np.abs(b))
            rel = np.where(denom > 0, np.abs(a - b) / denom, 0.0)
        close[finite] |= rel[finite] <= RTOL
        n_tot = len(a)
        rows.append({
            "name": str(n),
            "ref": str(rnames[j]),
            "n": n_tot,
            "bit": int(bit.sum() + both_nan.sum()),
            "close": int(close.sum() + both_nan.sum()),
            "nan_only_one": int(one_nan.sum()),
            "max_rel": float(np.nanmax(rel[finite])) if finite.any() else 0.0,
        })
    return rows


def summarize(rows, label):
    if not rows:
        return {"label": label, "columns": 0}
    cells = sum(r["n"] for r in rows)
    bit = sum(r["bit"] for r in rows)
    close = sum(r["close"] for r in rows)
    full_bit = sum(1 for r in rows if r["bit"] == r["n"])
    close_999 = sum(1 for r in rows if r["close"] >= 0.999 * r["n"])
    nan_cols = sum(1 for r in rows if r["nan_only_one"])
    return {
        "label": label,
        "columns": len(rows),
        "cells": cells,
        "bit_identical_frac": bit / cells,
        "within_rtol_frac": close / cells,
        "columns_100pct_bit": full_bit,
        "columns_999pct_within_rtol": close_999,
        "columns_with_nan_mismatch": nan_cols,
    }


def coverage():
    """Which dedupe2 survivors does the package actually emit?"""
    import molhume as mh
    sel = json.load(open(ROOT / "results" / "dedupe2" / "dedupe2.json"))["kept"]
    dropped = set(json.load(open(ROOT / "results" / "dedupe2" / "dropped_columns.json")))
    emitted = list(mh.feature_names(fingerprint=False))
    elo = {c.lower() for c in emitted}

    def names(k):
        return [k["name"], *(k.get("aliases") or [])]

    missing = [k["name"] for k in sel if not any(n.lower() in elo for n in names(k))]
    unexplained = sorted(set(missing) - dropped)
    return {
        "selected": len(sel),
        "emitted": len(emitted),
        "selected_not_emitted": len(missing),
        "cost_dropped_as_expected": len(set(missing) & dropped),
        "unexplained_absent": unexplained,
        "emitted_not_selected": [c for c in emitted
                                 if c.lower() not in {n.lower() for k in sel for n in names(k)}],
    }


def main():
    hume = load(HUME_NPZ)
    H, hnames = hume["X"], hume["names"]
    # `keep` is an INDEX array into the 42,000-molecule corpus, not a mask: the 8 molecules
    # RDKit could not parse are absent from it, so it is length 41,992 with values up to 41,999.
    keep = hume["keep"].astype(np.int64)

    rd = load("exactness_rdkit.npz")
    md = load("exactness_mordred.npz")
    # RDKit's matrix is already restricted to the molecules that parsed; mordred's is not.
    MD = md["X"][keep] if md["X"].shape[0] != H.shape[0] else md["X"]

    rd_rows = compare(H, hnames, rd["X"], rd["names"], fold_case=True)
    md_rows = compare(H, hnames, MD, md["names"], fold_case=False)

    report = {
        "corpus_molecules": int(H.shape[0]),
        "emitted_columns": int(H.shape[1]),
        "rtol": RTOL,
        "source_matrix": HUME_NPZ,
        "vs_rdkit": summarize(rd_rows, "rdkit"),
        "vs_mordred": summarize(md_rows, "mordred"),
        "coverage": coverage(),
        "worst_rdkit": sorted(rd_rows, key=lambda r: r["close"] / r["n"])[:15],
        "worst_mordred": sorted(md_rows, key=lambda r: r["close"] / r["n"])[:25],
        "per_column": {"rdkit": rd_rows, "mordred": md_rows},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1)

    for k in ("vs_rdkit", "vs_mordred"):
        s = report[k]
        print(f"\n=== {s['label']} ===")
        print(f"  columns compared        {s['columns']}")
        print(f"  cells compared          {s['cells']:,}")
        print(f"  bit-identical           {100 * s['bit_identical_frac']:.2f}%")
        print(f"  within {RTOL:g} relative    {100 * s['within_rtol_frac']:.2f}%")
        print(f"  columns 100% bit        {s['columns_100pct_bit']} / {s['columns']}")
        print(f"  columns >=99.9% close   {s['columns_999pct_within_rtol']} / {s['columns']}")
        print(f"  columns w/ NaN mismatch {s['columns_with_nan_mismatch']}")
    c = report["coverage"]
    print(f"\n=== coverage ===")
    print(f"  dedupe2 survivors       {c['selected']}")
    print(f"  emitted                 {c['emitted']}")
    print(f"  selected, not emitted   {c['selected_not_emitted']} "
          f"({c['cost_dropped_as_expected']} cost-dropped as expected)")
    print(f"  UNEXPLAINED absent      {len(c['unexplained_absent'])}: "
          f"{', '.join(c['unexplained_absent'])}")
    print(f"  emitted, not selected   {len(c['emitted_not_selected'])}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
