#!/usr/bin/env bash
# Phase 1 of the LSD AmpC screen: one box per arm, each training five models.
#
#     ARMS="hume_r4" ./scripts/launch_screen1.sh                 # first box: also builds trainval
#     ARMS="hume_r2 ecfp_rdkit_desc chemberta_mtr chemeleon minimol" ./scripts/launch_screen1.sh
#
# LAUNCH ONE ARM FIRST. The first box to run also builds trainval.npz and uploads it; every
# later box finds it in S3 and skips straight to training. Launching all six at once would have
# six boxes each spend ~25 minutes building the identical file, and -- worse -- race to upload
# it, so arms could end up trained against different bytes with nothing in the record saying so.
#
# 64 GB, NOT 32. A HUME training matrix is 1,000,000 x 3,317 float32 = 13.3 GB and XGBoost then
# builds its own quantized copy; a 32 GB box dies partway through `fit`, after an hour of
# featurizing. The memory is the reason for the instance type here, not the core count.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUCKET=hume-bench-use1-075120018132
PROFILE=HumeBenchEC2
GID=sg-082c5d5ebeb465414
ARMS="${ARMS:?set ARMS to one or more of: hume_r4 hume_r2 ecfp_rdkit_desc chemberta_mtr chemeleon minimol}"
TYPES="${TYPES:-m7i.4xlarge r7i.4xlarge m6i.4xlarge}"

AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
      --query 'Parameters[0].Value' --output text) || { echo "FATAL: cannot resolve AMI"; exit 1; }
SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
          --query 'Subnets[].SubnetId' --output text) || { echo "FATAL: cannot list subnets"; exit 1; }

LOCAL_SHA=$(shasum -a 256 dist/screenbundle.tar.gz 2>/dev/null | cut -d' ' -f1)
[ -n "$LOCAL_SHA" ] || { echo "FATAL: no dist/screenbundle.tar.gz; run scripts/stage_screen.sh" >&2; exit 1; }
REMOTE_SHA=$(aws s3api head-object --bucket "$BUCKET" --key screenbundle.tar.gz \
             --query 'Metadata.sha256' --output text 2>/dev/null || echo none)
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
  echo "FATAL: s3://$BUCKET/screenbundle.tar.gz does not match the local bundle.
  local  $LOCAL_SHA
  remote $REMOTE_SHA
Run scripts/stage_screen.sh before launching." >&2
  exit 1
fi
aws s3api head-object --bucket "$BUCKET" --key dsbundle.tar.gz >/dev/null 2>&1 \
  || { echo "FATAL: s3://$BUCKET/dsbundle.tar.gz is missing; the arms' model weights live in it." >&2; exit 1; }
echo "AMI $AMI   bundle $LOCAL_SHA   arms: $ARMS"

for ARM in $ARMS; do
  UD=$(mktemp -t s1ud).gz
  { printf '#!/bin/bash\nexport PHASE=train\nexport ARM=%s\n' "$ARM"
    tail -n +2 boot/boot_screen.sh; } | gzip -9 > "$UD"
  B=$(( ( $(wc -c < "$UD" | tr -d ' ') + 2) / 3 * 4 ))
  # The 16,384-byte limit applies to the BASE64 form, not the raw bytes -- checking the raw size
  # is what made RunInstances refuse an earlier launch in this project.
  [ "$B" -le 16384 ] || { echo "FATAL: user-data is $B base64 bytes, over 16384. Fetch the boot
script from S3 instead, and install the AWS CLI in user-data FIRST (the stock image has none)." >&2; exit 1; }
  LAUNCHED=""
  for T in $TYPES; do
    for S in $SUBNETS; do
      for MARKET in spot ondemand; do
        EXTRA=()
        [ "$MARKET" = spot ] && EXTRA=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
        OUT=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" \
          --security-group-ids "$GID" --subnet-id "$S" --associate-public-ip-address \
          --iam-instance-profile "Name=$PROFILE" \
          --instance-initiated-shutdown-behavior terminate \
          --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
          --user-data "fileb://$UD" ${EXTRA[@]+"${EXTRA[@]}"} \
          --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hume-screen1},{Key=Project,Value=HUME},{Key=Phase,Value=screen1},{Key=Arm,Value=$ARM}]" \
          --query 'Instances[0].InstanceId' --output text 2>&1)
        case "$OUT" in
          i-*) printf '  LAUNCHED %s  arm %-17s %-14s %-18s %s\n' "$OUT" "$ARM" "$T" "$S" "$MARKET"
               echo "$OUT $ARM" >> .screen1_instances; LAUNCHED=$OUT ;;
          *InsufficientInstanceCapacity*|*Unsupported*) ;;
          *) echo "  $T/$S/$MARKET refused: $OUT" ;;
        esac
        [ -n "$LAUNCHED" ] && break
      done
      [ -n "$LAUNCHED" ] && break
    done
    [ -n "$LAUNCHED" ] && break
  done
  rm -f "$UD"
  [ -n "$LAUNCHED" ] || { echo "FATAL: no capacity for arm $ARM in [$TYPES], any AZ, spot or on-demand." >&2; exit 1; }
done
echo
echo "ids in .screen1_instances"
echo "  watch  : ./scripts/watch_screen.sh screen1"
echo "  models : aws s3 ls s3://$BUCKET/screen/models/"
