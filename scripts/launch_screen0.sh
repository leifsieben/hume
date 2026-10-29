#!/usr/bin/env bash
# Phase 0 of the LSD AmpC screen, fanned out over NBOX boxes. Each box takes a contiguous slice
# of the 246 shards, reads their SMILES column from HuggingFace once, keeps the ~414k molecules
# per shard that the frozen split selects, ships them to S3 and terminates itself.
#
#     NBOX=8 ./scripts/launch_screen0.sh
#     NBOX=1 LO=0 HI=2 ./scripts/launch_screen0.sh      # one box, two shards, as a live test
#
# SPOT FIRST, AND TYPE FALLBACK IS ALLOWED HERE. Unlike launch_scale.sh -- where the instance
# type IS the measurement, because us/mol is not portable across hardware -- phase 0 only moves
# bytes. Nothing it produces depends on which CPU did the moving, so this takes whatever capacity
# exists. Work is per-shard idempotent (a shard is re-done unless its stored row set matches the
# split exactly), so a spot reclaim costs at most one shard.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUCKET=hume-bench-use1-075120018132
PROFILE=HumeBenchEC2
GID=sg-082c5d5ebeb465414
N_SHARDS=246
NBOX="${NBOX:-8}"
TYPES="${TYPES:-c7i.2xlarge c7i.4xlarge m7i.2xlarge c6i.2xlarge}"

AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
      --query 'Parameters[0].Value' --output text) || { echo "FATAL: cannot resolve AMI"; exit 1; }
SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
          --query 'Subnets[].SubnetId' --output text) || { echo "FATAL: cannot list subnets"; exit 1; }
echo "ubuntu AMI $AMI   $NBOX box(es)   types: $TYPES"

# The bundle must already be in S3, and it must be the CURRENT one: a box that unpacks last
# week's screen/ would verify the split against last week's spec and look perfectly healthy.
LOCAL_SHA=$(shasum -a 256 dist/screenbundle.tar.gz 2>/dev/null | cut -d' ' -f1)
[ -n "$LOCAL_SHA" ] || { echo "FATAL: dist/screenbundle.tar.gz does not exist. Run scripts/stage_screen.sh"; exit 1; }
REMOTE_SHA=$(aws s3api head-object --bucket "$BUCKET" --key screenbundle.tar.gz \
             --query 'Metadata.sha256' --output text 2>/dev/null || echo none)
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
  echo "FATAL: s3://$BUCKET/screenbundle.tar.gz does not match the local bundle.
  local  $LOCAL_SHA
  remote $REMOTE_SHA
Run scripts/stage_screen.sh to upload the current tree before launching." >&2
  exit 1
fi
echo "bundle verified: $LOCAL_SHA"

launch_one () {           # $1=lo  $2=hi
  local LO=$1 HI=$2 UD_FILE UD_BYTES UD_B64 OUT
  UD_FILE=$(mktemp -t screenud).gz
  { printf '#!/bin/bash\nexport LO=%s\nexport HI=%s\n' "$LO" "$HI"
    tail -n +2 boot/boot_screen0.sh; } | gzip -9 > "$UD_FILE"
  UD_BYTES=$(wc -c < "$UD_FILE" | tr -d ' ')
  # EC2 applies the 16,384-byte user-data limit to the BASE64 form, not the raw bytes. Checking
  # the raw size is exactly the mistake that made RunInstances refuse an earlier launch here.
  UD_B64=$(( (UD_BYTES + 2) / 3 * 4 ))
  if [ "$UD_B64" -gt 16384 ]; then
    echo "FATAL: boot_screen0.sh is $UD_B64 base64 bytes, over the 16384 user-data limit.
Fetch it from S3 instead -- and install the AWS CLI in user-data FIRST, because the stock
Ubuntu image has none and that is how a box once sat running and silent." >&2
    return 1
  fi
  for T in $TYPES; do
    for S in $SUBNETS; do
      for MARKET in spot ondemand; do
        EXTRA=()
        [ "$MARKET" = spot ] && EXTRA=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
        OUT=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" \
          --security-group-ids "$GID" --subnet-id "$S" --associate-public-ip-address \
          --iam-instance-profile "Name=$PROFILE" \
          --instance-initiated-shutdown-behavior terminate \
          --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
          --user-data "fileb://$UD_FILE" ${EXTRA[@]+"${EXTRA[@]}"} \
          --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hume-screen0},{Key=Project,Value=HUME},{Key=Phase,Value=screen0},{Key=Shards,Value=${LO}-${HI}}]" \
          --query 'Instances[0].InstanceId' --output text 2>&1)
        case "$OUT" in
          i-*) printf '  LAUNCHED %s  shards [%3d,%3d)  %-13s %-18s %s\n' \
                 "$OUT" "$LO" "$HI" "$T" "$S" "$MARKET"
               echo "$OUT $LO $HI" >> .screen0_instances
               rm -f "$UD_FILE"; return 0 ;;
          *InsufficientInstanceCapacity*) ;;
          *Unsupported*) ;;
          # PRINT THE WHOLE ERROR. An empty branch here is how a permissions failure reads as a
          # capacity failure; see the skill note on judging IAM by the error code, never by the
          # exit status.
          *) echo "  $T/$S/$MARKET refused: $OUT" ;;
        esac
      done
    done
  done
  echo "FATAL: no capacity for any of [$TYPES] in any AZ, spot or on-demand, for shards [$LO,$HI)." >&2
  rm -f "$UD_FILE"; return 1
}

# B0/B1 select which of the NBOX slices to launch, so the fleet can go up in two steps: one
# box first to prove the boot path, then the rest. The slice boundaries are a function of NBOX
# alone, so box 0 launched on its own covers exactly the range it would have in the full fleet
# and its work is not repeated.
B0="${B0:-0}"
B1="${B1:-$NBOX}"
[ "$B0" = 0 ] && rm -f .screen0_instances
if [ -n "${LO:-}" ] && [ -n "${HI:-}" ]; then
  launch_one "$LO" "$HI" || exit 1
else
  # Contiguous slices. Shards are near-identical in SIZE, so an even split of the count is an
  # even split of the bytes; the manifest's per-shard row counts vary by under 0.1%. They are
  # NOT even in content -- the release is score-sorted, so the whole top 1% sits in the last
  # shards -- but phase 0 only moves bytes, so that does not unbalance it.
  for ((b = B0; b < B1; b++)); do
    L=$(( b * N_SHARDS / NBOX ))
    H=$(( (b + 1) * N_SHARDS / NBOX ))
    [ "$L" -lt "$H" ] || continue
    launch_one "$L" "$H" || exit 1
  done
fi

echo
echo "launched $(wc -l < .screen0_instances | tr -d ' ') box(es); ids in .screen0_instances"
echo "  watch    : ./scripts/watch_screen0.sh"
echo "  logs     : aws s3 ls s3://$BUCKET/logs/"
echo "  output   : aws s3 ls s3://$BUCKET/screen/smiles/"
