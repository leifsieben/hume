#!/bin/sh
# Queued behind the complementarity run to avoid core contention.
set -x
CHEM=/Users/lsieben/VSCode/ChemTFM_OLD
export PYTHONPATH=$CHEM
# wait for complementarity to finish (bounded)
i=0
while pgrep -f "complementarity.py" >/dev/null 2>&1; do
    i=$((i+1)); [ $i -gt 720 ] && { echo "TIMEOUT waiting for complementarity"; break; }
    sleep 10
done
$CHEM/.venv/bin/python noise_threshold.py
