#!/usr/bin/env bash
# Pull the GNN checkpoint to local disk every few minutes.
#
# Spot reclaimed the previous run and DeleteOnTermination took its per-epoch checkpoints with
# it. Keeping a local copy means an interruption costs one pull interval, and the next instance
# resumes from it instead of starting over.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PEM="$HOME/.ssh/hume-gnn-key.pem"
mkdir -p data/surrogate/gnn_ckpt
while true; do
  IP=$(cat gpu/.instance_ip 2>/dev/null) || { sleep 60; continue; }
  if scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$PEM" \
       ubuntu@"$IP":job/ckpt_gnn.pt data/surrogate/gnn_ckpt/ckpt_gnn.pt 2>/dev/null; then
    ep=$(/Users/lsieben/VSCode/ChemTFM_OLD/.venv/bin/python -c "
import torch;print(torch.load('data/surrogate/gnn_ckpt/ckpt_gnn.pt',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
    echo "$(date +%H:%M:%S) pulled checkpoint, epoch ${ep:-?}"
  fi
  sleep 180
done
