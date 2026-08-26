#!/usr/bin/env bash
# Launch a DEDICATED g5.2xlarge for the HUME GNN arm, run it, retrieve results, terminate.
#
# Touches nothing that already exists on the account. Every resource it creates carries
# Project=HUME so it is distinguishable from the chempfn-* and climb-* instances, and it filters
# on that tag for every query it makes.
#
# THREE independent stops, because an instance that outlives its job is the expensive failure:
#   1. the remote script shuts the box down when training finishes
#   2. instance-initiated-shutdown-behavior=terminate turns that shutdown into a termination
#   3. a `shutdown -h +MAXMIN` armed at boot fires even if training hangs or crashes
#
#   gpu/launch_gnn.sh [EPOCHS] [MAXMIN]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EPOCHS="${1:-12}"
MAXMIN="${2:-720}"           # hard kill after 12 h no matter what
TYPE=g5.2xlarge
AMI=ami-012ba162b9cd2729c    # Deep Learning OSS Nvidia PyTorch 2.7, Ubuntu 22.04, x86_64
KEY=hume-gnn-key
SG=hume-gnn-sg
TAG=hume-gnn
PEM="$HOME/.ssh/${KEY}.pem"

echo "== key pair =="
if ! aws ec2 describe-key-pairs --key-names "$KEY" >/dev/null 2>&1; then
  aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial --output text > "$PEM"
  chmod 600 "$PEM"; echo "created $PEM"
else
  [ -f "$PEM" ] || { echo "key pair $KEY exists in AWS but $PEM is missing locally; delete the"\
                          "key pair or restore the file"; exit 1; }
  echo "reusing $PEM"
fi

echo "== security group =="
MYIP=$(curl -s https://checkip.amazonaws.com)
if ! aws ec2 describe-security-groups --group-names "$SG" >/dev/null 2>&1; then
  VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  GID=$(aws ec2 create-security-group --group-name "$SG" --description "HUME GNN ssh" \
        --vpc-id "$VPC" --query GroupId --output text)
else
  GID=$(aws ec2 describe-security-groups --group-names "$SG" --query 'SecurityGroups[0].GroupId' --output text)
fi
aws ec2 authorize-security-group-ingress --group-id "$GID" --protocol tcp --port 22 \
  --cidr "${MYIP}/32" >/dev/null 2>&1 || true
echo "sg $GID open to ${MYIP}/32"

echo "== payload =="
"${PY:-../ChemTFM_OLD/.venv/bin/python}" gpu/pack_job.py
ls -la gpu/gnn_job.tar.gz

echo "== launch $TYPE =="
# SPOT, not on-demand. The on-demand G/VT quota is fully consumed by the chempfn and climb
# instances (48 + 16 = 64 of 64) and the P quota is 0, so on-demand cannot launch at all.
# Spot is a separate quota with 64 vCPUs free and zero usage, and is ~25% cheaper. The cost is
# interruption risk: there is no resume path, so an interrupted run is relaunched from scratch.
IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$GID" --instance-initiated-shutdown-behavior terminate \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data "#!/bin/bash
shutdown -h +${MAXMIN}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=HUME}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "instance $IID (dead-man's switch: shutdown -h +${MAXMIN} min)"
echo "$IID" > gpu/.instance_id

aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "ip $IP -- waiting for sshd"
for i in $(seq 1 40); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$PEM" ubuntu@"$IP" true 2>/dev/null && break
  sleep 15
done

echo "== upload =="
scp -o StrictHostKeyChecking=no -i "$PEM" gpu/gnn_job.tar.gz gpu/run_remote.py ubuntu@"$IP":~/
ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$IP" \
  'mkdir -p job && tar xzf gnn_job.tar.gz -C job && mv run_remote.py job/ && \
   pip install -q rdkit 2>&1 | tail -2 && python -c "import torch,rdkit;print(torch.__version__, torch.cuda.is_available())"'

echo "== train (nohup; safe to disconnect) =="
ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$IP" \
  "cd job && nohup sh -c 'python run_remote.py $EPOCHS > train.log 2>&1; sudo shutdown -h now' > /dev/null 2>&1 & echo started"
echo
echo "instance : $IID  ($IP)"
echo "watch    : ssh -i $PEM ubuntu@$IP 'tail -f job/train.log'"
echo "fetch    : gpu/fetch_gnn.sh"
echo "kill now : aws ec2 terminate-instances --instance-ids $IID"
