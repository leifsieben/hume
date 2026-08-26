#!/usr/bin/env bash
# Dedicated CPU box for the Figure B / C grid: XGBoost CV over (dataset, fold, arm).
# CPU-bound and embarrassingly parallel, which is exactly what a wide box collapses.
#
# c7i is a STANDARD instance -- a different quota from the exhausted G/VT one. Both the
# on-demand and spot Standard quotas are 32 vCPU, so c7i.8xlarge is the largest that fits.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GID=sg-082c5d5ebeb465414
KEY=hume-gnn-key
AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --query 'Parameters[0].Value' --output text)
echo "ubuntu AMI $AMI"
SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text)
for T in c7i.8xlarge c6i.8xlarge m7i.8xlarge c7i.4xlarge; do
  for S in $SUBNETS; do
    OUT=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$KEY" \
      --security-group-ids "$GID" --subnet-id "$S" --associate-public-ip-address \
      --instance-initiated-shutdown-behavior terminate \
      --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --user-data '#!/bin/bash
shutdown -h +720' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hume-cpu},{Key=Project,Value=HUME}]' \
      --query 'Instances[0].InstanceId' --output text 2>&1)
    case "$OUT" in
      i-*) echo "LAUNCHED $OUT  type=$T  subnet=$S"; echo "$OUT" > gpu/.cpu_instance_id; exit 0 ;;
    esac
  done
  echo "  no capacity: $T"
done
echo "NO CPU CAPACITY"; exit 1
