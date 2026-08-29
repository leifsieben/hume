#!/bin/bash
# ONE COMMAND to fold whatever has landed into Figures B, C and D and re-render them.
#
#     ./refresh_figures.sh
#
# Safe to run at any time and as often as you like: it reads only what is in S3, merges
# newest-protocol-wins, and re-renders. Figure A does not appear here -- it is complete and its
# inputs are local.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== cost axis (results/scale) ==="
$PY collect_scale.py 2>&1 | grep -E "arms:|priced" || true

echo
echo "=== downstream grid ==="
$PY collect_downstream.py 2>&1 | grep -vE "^  i-" || true

echo
echo "=== what is still missing ==="
$PY - <<'PY'
import json
from collections import defaultdict
import sys; sys.path.insert(0, ".")
from collect_downstream import TASKS, FIGB_ADDS, FIGB_BASES
r = json.load(open("results/figures/downstream_raw.json"))
by = defaultdict(set)
for x in r:
    by[x["dataset"]].add(x["arm"])
concat = {f"{b}__{a}" for b in FIGB_BASES for a in FIGB_ADDS}
alld = [d for _l, ds in TASKS.values() for d in ds]
miss = [d for d in alld if len(by[d] & concat) < 12]
print(f"  concat arms complete on {len(alld)-len(miss)}/{len(alld)} plotted datasets")
if miss:
    print(f"    still missing: {' '.join(miss)}")
p2 = {x["dataset"] for x in r if x.get("proto", 1) >= 2}
cls = TASKS["classif"][1]
print(f"  protocol-2 (3-fold inner CV for w) on {len(set(cls) & p2)}/{len(cls)} classification sets")
todo = [d for d in cls if d not in p2]
if todo:
    print(f"    still on protocol 1: {' '.join(todo)}")
PY

echo
echo "=== render ==="
for f in fig_b fig_c fig_d; do $PY figures/src/$f.py 2>&1 | grep -E "saved|WARNING|SKIP" || true; done
