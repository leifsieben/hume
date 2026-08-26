#!/bin/sh
# Follow-on queue: train + evaluate once the data stages land.
#
# Runs alongside the UMA embed (6 workers) but capped at 4 torch threads so it does not
# starve them — UMA aggregate throughput collapses past ~6 workers on this machine.
#
# Waits are bounded. A stage that never produces its data must fail the script rather than
# hang silently until morning.
set -x
CHEM=/Users/lsieben/VSCode/ChemTFM_OLD
PY=$CHEM/.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$CHEM

wait_for() {  # wait_for <dir> <glob> <count> <max_minutes> <label>
    i=0
    while [ "$(ls "$1"/$2 2>/dev/null | wc -l)" -lt "$3" ]; do
        i=$((i + 1))
        [ "$i" -gt "$(($4 * 6))" ] && { echo "TIMEOUT waiting for $5"; return 1; }
        sleep 10
    done
    echo "$5 ready"
}

# --- descriptor surrogate: the primary result, and it does not need UMA -----------------
wait_for data/uma100k/targets 'tgt_*.npz' 20 40 "descriptor targets" || exit 1
$PY train_surrogate.py --epochs 30           || exit 1
$PY eval_surrogate.py                        || exit 1

# --- UMA surrogate: needs the overnight embed to finish ---------------------------------
wait_for data/uma100k/embeddings 'emb_*.npz' 50 240 "UMA embeddings" || exit 1
$PY train_surrogate.py --epochs 30 --with-uma || exit 1
echo "ALL STAGES COMPLETE"
