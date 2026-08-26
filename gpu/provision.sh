#!/usr/bin/env bash
# Upload the job to an already-launched instance and start training.
#
# The first attempt died in two minutes because the command was `python run_remote.py; sudo
# shutdown -h now` and the Deep Learning AMI has no bare `python` on PATH -- so the failure
# chained straight into the shutdown and terminated the box before anything could be inspected.
# Two structural fixes, not just a renamed binary:
#   * resolve the interpreter by TESTING it (torch + cuda + rdkit all importable), from a list
#     of candidates, and abort the whole script if none works
#   * the shutdown only arms after a successful run; a crash leaves the instance up so its log
#     can be read. The boot-time `shutdown -h +720` dead-man's switch still bounds the cost.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IID=$(cat gpu/.instance_id)
PEM="$HOME/.ssh/hume-gnn-key.pem"
EPOCHS="${1:-12}"
H="${2:-128}"
DEPTH="${3:-4}"
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$PEM")

aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "instance $IID at $IP -- waiting for sshd"
for i in $(seq 1 60); do "${SSH[@]}" ubuntu@"$IP" true 2>/dev/null && break; sleep 10; done
echo "$IP" > gpu/.instance_ip

echo "== resolve interpreter =="
PY=$("${SSH[@]}" ubuntu@"$IP" 'for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python \
      /opt/conda/bin/python "$HOME/anaconda3/bin/python" python3; do
        if "$c" -c "import torch" >/dev/null 2>&1; then echo "$c"; exit 0; fi
      done; exit 1') || { echo "FATAL: no interpreter with torch on $IP; instance left UP for inspection"; exit 1; }
echo "  using $PY"

echo "== upload (307 MB) =="
scp -o StrictHostKeyChecking=no -i "$PEM" gpu/gnn_job.tar.gz gpu/run_remote.py ubuntu@"$IP":~/
"${SSH[@]}" ubuntu@"$IP" "mkdir -p job && tar xzf gnn_job.tar.gz -C job && mv -f run_remote.py job/ && \
   $PY -m pip install -q rdkit 2>&1 | tail -1; true"

echo "== verify environment BEFORE arming shutdown =="
"${SSH[@]}" ubuntu@"$IP" "$PY -c \"
import torch, rdkit, numpy
assert torch.cuda.is_available(), 'no CUDA'
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| rdkit', rdkit.__version__)
\"" || { echo "FATAL: environment check failed; instance left UP for inspection"; exit 1; }
"${SSH[@]}" ubuntu@"$IP" "cd job && $PY -c \"
import sys; sys.path.insert(0,'.')
import models, numpy as np
z=np.load('job.npz',allow_pickle=True)
print('payload ok: train',len(z['smi_tr']),'val',len(z['smi_va']),'bench',len(z['smi_bench']),'targets',z['Ytr'].shape[1])
print('graph_of ok:', [a.shape for a in models.graph_of(str(z['smi_tr'][0]))])
\"" || { echo "FATAL: payload/import check failed; instance left UP"; exit 1; }

# NO auto-shutdown. The first GNN run finished, shut itself down on success, and took its
# results with it -- DeleteOnTermination meant the volume went too. The box now stays up until
# `fetch_gnn.sh` has the results on local disk and terminates it explicitly. The boot-time
# `shutdown -h +720` dead-man's switch still bounds the cost if nobody ever fetches.
echo "== start training ($EPOCHS epochs; box stays UP until fetched) =="
"${SSH[@]}" ubuntu@"$IP" \
  "cd job && nohup sh -c '$PY run_remote.py $EPOCHS $H $DEPTH > train.log 2>&1; touch DONE' >/dev/null 2>&1 & echo started"
echo
echo "watch : ssh -i $PEM ubuntu@$IP 'tail -f job/train.log'"
echo "fetch : bash gpu/fetch_gnn.sh"
echo "kill  : aws ec2 terminate-instances --instance-ids $IID"
