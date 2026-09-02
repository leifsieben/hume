"""Read head_sweep.json and answer the question it was run to answer.

    .venv/bin/python tools/head_sweep_report.py

The number that matters is not any head's absolute RMSE -- heads are not comparable to each
other, since only some of them see imputed features. It is the GAP between `hume` and
`hume_minimal` WITHIN a head, which is comparable across heads because both arms get identical
treatment inside one.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("results/reanalysis")
rows = json.load(open(OUT / "head_sweep.json"))
by = {(r["dataset"], r["head"], r["arm"]): r for r in rows}
datasets = sorted({r["dataset"] for r in rows})
heads = list(dict.fromkeys(r["head"] for r in rows))

print("  Cost of the 800-column spec on physicochemical endpoints, by prediction head")
print("  (RMSE, so positive = minimal is worse; the AWS grid used xgb_d6)\n")
hdr = f"  {'dataset':15s}" + "".join(f"{h:>19s}" for h in heads)
print(hdr)
table = {}
for d in datasets:
    line = f"  {d:15s}"
    for h in heads:
        a, b = by.get((d, h, "hume")), by.get((d, h, "hume_minimal"))
        if not a or not b:
            line += f"{'-':>19s}"; continue
        rel = (b["rmse"] - a["rmse"]) / max(abs(a["rmse"]), 1e-12)
        table.setdefault(h, []).append(rel)
        line += f"{rel*100:+18.2f}%"
    print(line)

print()
print(f"  {'head':19s} {'mean':>8s} {'median':>8s} {'worst':>8s} {'n worse':>8s} {'sign p':>8s}")
summary = {}
for h in heads:
    v = np.array(table.get(h, []))
    if not len(v):
        continue
    pos = int((v > 0).sum())
    p = stats.binomtest(pos, len(v), 0.5).pvalue
    summary[h] = dict(mean=float(v.mean()), median=float(np.median(v)),
                      worst=float(v.max()), n_worse=pos, n=len(v), sign_p=float(p))
    print(f"  {h:19s} {v.mean()*100:+7.2f}% {np.median(v)*100:+7.2f}% "
          f"{v.max()*100:+7.2f}% {pos:5d}/{len(v)} {p:8.3f}")

base = summary.get("xgb_d6", {}).get("mean")
print()
if base is not None:
    print(f"  Baseline xgb_d6 reproduces the grid at {base*100:+.2f}% "
          "(the AWS run measured +3.83% over the same six datasets).")
    for h, s in summary.items():
        if h == "xgb_d6":
            continue
        rec = 1 - s["mean"] / base if base else float("nan")
        print(f"    {h:19s} recovers {rec*100:5.1f}% of the gap  "
              f"(mean {s['mean']*100:+.2f}%)")
json.dump(summary, open(OUT / "head_sweep_summary.json", "w"), indent=1)
print(f"\n  -> {OUT / 'head_sweep_summary.json'}")
