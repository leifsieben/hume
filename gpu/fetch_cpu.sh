#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IID=$(cat gpu/.cpu_instance_id); PEM="$HOME/.ssh/hume-gnn-key.pem"
IP=$(cat gpu/.cpu_instance_ip)
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$PEM")
for i in $(seq 1 480); do "${SSH[@]}" ubuntu@"$IP" 'test -f grid/DONE' 2>/dev/null && break; sleep 30; done
mkdir -p results/figures/figB
scp -o StrictHostKeyChecking=no -i "$PEM" -r ubuntu@"$IP":grid/grid_out/. results/figures/figB/
echo "fetched $(ls results/figures/figB | wc -l) files"
aws ec2 terminate-instances --instance-ids "$IID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
