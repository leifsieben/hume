#!/usr/bin/env bash
# Spot capacity is per (instance-type, AZ). Sweep both rather than failing on the default.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GID=sg-082c5d5ebeb465414
KEY=hume-gnn-key
AMI=ami-012ba162b9cd2729c
SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text)
for T in g4dn.xlarge g5.xlarge g6.xlarge g4dn.2xlarge g5.2xlarge; do
  for S in $SUBNETS; do
    OUT=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$KEY" \
      --security-group-ids "$GID" --subnet-id "$S" --associate-public-ip-address \
      --instance-initiated-shutdown-behavior terminate \
      --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --user-data '#!/bin/bash
shutdown -h +720' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hume-gnn},{Key=Project,Value=HUME}]' \
      --query 'Instances[0].InstanceId' --output text 2>&1)
    case "$OUT" in
      i-*) echo "LAUNCHED $OUT  type=$T  subnet=$S"
           echo "$OUT" > gpu/.instance_id; echo "$T" > gpu/.instance_type; exit 0 ;;
    esac
  done
  echo "  no spot capacity: $T"
done
echo "NO SPOT CAPACITY IN ANY (type, AZ) COMBINATION"
exit 1
