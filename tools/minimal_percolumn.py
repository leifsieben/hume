"""Per-column reconstruction R^2 at the shipped operating point.

    .venv/bin/python tools/minimal_percolumn.py

`minimal_curve()` reports only the worst case and a count below 0.99 per k, which answers "is
this k safe overall" but not "is the column I care about safe". A user whose endpoint leans on
one specific descriptor needs the second question answered, and worst-case-over-467-columns
cannot answer it.

Held-out, not in-sample, for the reason in MINIMAL_SPEC.md section 6: the kept set is
numerically singular, so an in-sample fit reports a number that does not survive a second
sample. The map is fitted on repA + adv (the derivation samples) and scored on repB, which is
disjoint from both.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import qr

sys.path.insert(0, "tools")
from minimal_select import RIDGE, prepare, select_columns  # noqa: E402

OUT = Path("results/minimal")
z = np.load(OUT / "matrices.npz", allow_pickle=False)
names = np.array([str(n) for n in z["names"]])

idx = select_columns(z["repA"], False)
ZA, _ = prepare(z["repA"], idx)
ZB, _ = prepare(z["repB"], idx)
ZV, _ = prepare(z["adv"], idx)
_, _, piv = qr(np.vstack([ZA, ZV]), mode="economic", pivoting=True)
order = np.asarray(piv)

sel = json.load(open(OUT / "selection_pooled.json"))
assert order.tolist() == sel["order"], "ordering drifted from the shipped selection"

N = 800
keep, drop = order[:N], order[N:]
Zfit = np.vstack([ZA, ZV])
A, B = Zfit[:, keep], Zfit[:, drop]
coef = np.linalg.solve(A.T @ A + RIDGE * len(A) * np.eye(len(keep)), A.T @ B)
resid = ZB[:, drop] - ZB[:, keep] @ coef
sse = (resid ** 2).sum(axis=0)
sst = (ZB[:, drop] ** 2).sum(axis=0)
r2 = 1.0 - sse / np.where(sst == 0, 1, sst)

col_names = names[idx]
out = {str(col_names[c]): round(float(v), 5) for c, v in zip(drop, r2)}
print(f"  {len(out)} dropped columns at n={N}; held-out R^2 fitted on repA+adv, scored on repB")
worst = sorted(out.items(), key=lambda kv: kv[1])[:12]
print("  hardest to reconstruct:")
for k, v in worst:
    print(f"     {k:26s} {v:8.4f}")
print(f"  median {np.median(list(out.values())):.4f}, "
      f"{sum(1 for v in out.values() if v < 0.99)} below 0.99, "
      f"{sum(1 for v in out.values() if v < 0.9)} below 0.90")
json.dump(out, open(OUT / "per_column_r2.json", "w"), indent=1, sort_keys=True)
print(f"  -> {OUT / 'per_column_r2.json'}")
