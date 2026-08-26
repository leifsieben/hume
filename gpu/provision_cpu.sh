#!/usr/bin/env bash
# Ship the DEV grid to the CPU box and run it, 14 datasets at a time on 32 vCPU.
#
# Parallel over DATASETS rather than inside XGBoost: the grid is embarrassingly parallel across
# (dataset, fold, arm) and per-dataset processes keep memory bounded and make each unit
# independently resumable -- dev_grid.py skips any dataset whose json already exists.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IID=$(cat gpu/.cpu_instance_id)
PEM="$HOME/.ssh/hume-gnn-key.pem"
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$PEM")

aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "cpu box $IID at $IP"
for i in $(seq 1 60); do "${SSH[@]}" ubuntu@"$IP" true 2>/dev/null && break; sleep 10; done
echo "$IP" > gpu/.cpu_instance_ip

echo "== deps =="
"${SSH[@]}" ubuntu@"$IP" 'sudo apt-get -qq update >/dev/null 2>&1; \
  sudo apt-get -qq install -y python3-pip >/dev/null 2>&1; \
  pip3 install -q --upgrade pip 2>&1 | tail -1; \
  pip3 install -q rdkit mordred xgboost scikit-learn networkx "numpy<2" \
    torch --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3; \
  python3 -c "import rdkit,mordred,xgboost,sklearn,torch,numpy;print(\"deps ok\", numpy.__version__)"'

echo "== upload =="
scp -o StrictHostKeyChecking=no -i "$PEM" gpu/grid_job.tar.gz ubuntu@"$IP":~/
"${SSH[@]}" ubuntu@"$IP" 'rm -rf grid && mkdir -p grid && tar xzf grid_job.tar.gz -C grid && ls grid | tr "\n" " "'

echo "== verify one dataset before committing the fleet =="
"${SSH[@]}" ubuntu@"$IP" 'cd grid && OMP_NUM_THREADS=2 python3 dev_grid.py photoswitch 2>&1 | tail -3' \
  || { echo "FATAL: grid failed on the smoke dataset; box left UP"; exit 1; }

echo "== launch all 28, 14-way parallel =="
"${SSH[@]}" ubuntu@"$IP" 'cd grid && ls devsets/*.npz | xargs -n1 basename | sed "s/.npz//" > all.txt && \
  nohup sh -c "cat all.txt | xargs -P 14 -I{} env OMP_NUM_THREADS=2 python3 dev_grid.py {} > grid.log 2>&1; \
               python3 dev_grid.py > final.log 2>&1; touch DONE" >/dev/null 2>&1 & echo launched'
echo
echo "watch : ssh -i $PEM ubuntu@$IP 'ls grid/grid_out | wc -l; tail -3 grid/grid.log'"
echo "fetch : bash gpu/fetch_cpu.sh"
