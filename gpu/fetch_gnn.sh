#!/usr/bin/env bash
# Retrieve GNN results and terminate. Safe to run repeatedly; only terminates once results land.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IID=$(cat gpu/.instance_id)
PEM="$HOME/.ssh/hume-gnn-key.pem"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$PEM")
# Wait for the run to finish rather than racing it.
for i in $(seq 1 240); do
  "${SSH[@]}" ubuntu@"$IP" 'test -f job/DONE' 2>/dev/null && break
  sleep 30
done
scp -o StrictHostKeyChecking=no -i "$PEM" \
  ubuntu@"$IP":job/{pred_bench_gnn.npz,pred_val_gnn.npz,gnn_report.json,ckpt_gnn.pt,train.log} data/surrogate/ \
  && echo "fetched" && cat data/surrogate/gnn_report.json \
  && aws ec2 terminate-instances --instance-ids "$IID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
