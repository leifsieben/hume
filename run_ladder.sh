#!/bin/sh
set -x
CHEM=/Users/lsieben/VSCode/ChemTFM_OLD
export PYTHONPATH=$CHEM:/Users/lsieben/VSCode/universal-encoder
PY=$CHEM/.venv/bin/python
i=0
while [ ! -f data/surrogate/bench.npz ]; do
    i=$((i+1)); [ $i -gt 180 ] && { echo "TIMEOUT waiting for assemble"; exit 1; }
    sleep 10
done
sleep 5
$PY models.py --models ridge,linquad,pinet,mlp,gnn --epochs 25 --gnn-epochs 12 || exit 1
$PY downstream.py || exit 1
echo "LADDER COMPLETE"
