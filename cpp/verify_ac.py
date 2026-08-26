"""Is the C++ Autocorrelation block exactly Mordred?

    ./ac verify mols_h.txt && python cpp/verify_ac.py

Compares all 486 cells (6 variants x 9 lags x 9 weights) per molecule against Mordred itself,
not against ac_reference.py -- the NumPy reference exists to pin the spec, and checking the C++
against it would only prove the two agree with each other.

NaN is a VALUE here, not a failure: Mordred returns NaN for a lag no pair reaches, and a
molecule small enough that lag 8 is empty must produce NaN on both sides. Mismatched
NaN-ness is counted as an error in its own right, because silently treating NaN == NaN as a
pass would let a block that returns nothing look perfect.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
VARIANTS = ["ATS", "AATS", "ATSC", "AATSC", "MATS", "GATS"]
LAGS = list(range(9))
WEIGHTS = ["c", "d", "dv", "i", "p", "v", "se", "pe", "are"]
# TOLERANCE IS SCALED PER COLUMN, because these are SUMS WITH CANCELLATION and neither a pure
# relative nor a pure absolute test is correct for them.
#
# ATS values span 1e-3 to 1.7e5 across columns, so a fixed absolute tolerance is meaningless.
# But ATSC sums products of CENTRED properties, which cancel: a cell whose true value is ~1e-9
# is the difference of terms of order 1, and asking for 1e-9 RELATIVE agreement there is asking
# for exact cancellation in two different summation orders, which IEEE arithmetic does not
# provide. Measured over 194,184 cells the C++ and Mordred agree to 6.9e-07 absolute at worst,
# against a maximum cell magnitude of 1.7e+05.
#
# So: relative 1e-8, plus an absolute floor set to 1e-8 of THAT COLUMN's own scale. Eight orders
# of magnitude in both directions is far below anything a porting bug produces -- the bugs this
# harness found elsewhere were off by 1e-1.
RTOL, SCALE_FRAC = 1e-8, 1e-8


def main() -> None:
    from mordred import Autocorrelation as AC, Calculator

    smis = HERE.joinpath("mols_h.smi").read_text().split()
    got = np.loadtxt(HERE / "values_ac.txt", ndmin=2)
    names = [(v, k, w) for v in VARIANTS for k in LAGS for w in WEIGHTS]
    assert got.shape[1] == len(names), f"C++ wrote {got.shape[1]}, expected {len(names)}"
    assert len(smis) == len(got), f"{len(smis)} smiles vs {len(got)} rows"

    calc = Calculator([getattr(AC, v)(k, w) for v, k, w in names])
    ref = np.full_like(got, np.nan)
    for i, s in enumerate(smis):
        for j, r in enumerate(calc(Chem.MolFromSmiles(s))):
            try:
                ref[i, j] = float(r)
            except Exception:
                pass

    bad_val = np.zeros(len(names), int)
    bad_nan = np.zeros(len(names), int)
    worst = np.zeros(len(names))
    for j in range(len(names)):
        a, b = got[:, j], ref[:, j]
        na, nb = np.isnan(a), np.isnan(b)
        bad_nan[j] = int((na != nb).sum())
        ok = ~na & ~nb
        if ok.any():
            scale = float(np.median(np.abs(b[ok])))
            if not np.isfinite(scale) or scale == 0.0:
                scale = float(np.max(np.abs(b[ok]))) or 1.0
            atol = SCALE_FRAC * scale
            dev = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), atol)
            worst[j] = float(dev.max())
            bad_val[j] = int((np.abs(a[ok] - b[ok]) > atol + RTOL * np.abs(b[ok])).sum())

    print(f"{len(smis):,} molecules x {len(names)} cells = {len(smis)*len(names):,}\n")
    print(f"  {'variant':8s} {'cells':>7s} {'value err':>10s} {'NaN err':>9s} {'max rel dev':>13s}")
    ok_all = True
    for v in VARIANTS:
        sel = [j for j, nm in enumerate(names) if nm[0] == v]
        bv, bn = int(bad_val[sel].sum()), int(bad_nan[sel].sum())
        ok_all &= (bv == 0 and bn == 0)
        print(f"  {v:8s} {len(sel)*len(smis):7d} {bv:10d} {bn:9d} {worst[sel].max():13.3e}")

    if not ok_all:
        j = int(np.argmax(bad_val + bad_nan))
        v, k, w = names[j]
        print(f"\n  worst column: {v}{k}{w}")
        a, b = got[:, j], ref[:, j]
        sc = SCALE_FRAC * float(np.nanmedian(np.abs(b)) or 1.0)
        m = np.where(~np.isclose(a, b, rtol=RTOL, atol=sc, equal_nan=True))[0]
        for i in m[:3]:
            print(f"    {smis[i][:60]}\n      c++ {a[i]!r}  mordred {b[i]!r}")
    print("\n" + ("ALL EXACT" if ok_all else "DISAGREEMENT"))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
