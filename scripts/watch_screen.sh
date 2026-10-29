#!/usr/bin/env bash
# What is a screen phase doing right now, read from AWS alone -- no laptop state, no session
# memory. Safe from a fresh session: boxes are found by tag.
#
#     ./scripts/watch_screen.sh screen0|screen1|screen2
set -uo pipefail
BUCKET=hume-bench-use1-075120018132
REGION=us-east-1
PHASE="${1:-screen1}"

echo "=== instances tagged Phase=$PHASE ==="
aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Phase,Values=$PHASE Name=instance-state-name,Values=pending,running,shutting-down \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Arm`]|[0].Value,Tags[?Key==`Shards`]|[0].Value,LaunchTime]' \
  --output text | sort -k4 || true
RUNNING=$(aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Phase,Values=$PHASE Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$RUNNING" ] && echo "  (none running)"

echo
echo "=== log freshness -- a running box with a stale log IS the silent failure ==="
NOW=$(date -u +%s)
for ID in $RUNNING; do
  MOD=$(aws s3api head-object --bucket "$BUCKET" --key "logs/$ID.log" \
        --query 'LastModified' --output text 2>/dev/null || echo "")
  if [ -z "$MOD" ]; then echo "  $ID  NO LOG YET (still in apt/uv install)"; continue; fi
  T=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "${MOD%%+*}" +%s 2>/dev/null \
      || date -u -d "$MOD" +%s 2>/dev/null || echo 0)
  AGE=$(( (NOW - T) / 60 ))
  STAGE=$(aws s3 cp "s3://$BUCKET/logs/$ID.log" - 2>/dev/null | grep -oE "^=== [a-z0-9 :]+" | tail -1)
  printf '  %s  log %3d min old  %-28s%s\n' "$ID" "$AGE" "${STAGE:-...}" \
    "$([ "$AGE" -gt 15 ] && echo '  <-- STALE' || true)"
done

echo
echo "=== artifacts ==="
printf '  phase 0 smiles : %s/246 shards\n' "$(aws s3 ls s3://$BUCKET/screen/smiles/ 2>/dev/null | grep -c 'shard_')"
printf '  trainval.npz   : %s\n' "$(aws s3 ls s3://$BUCKET/screen/trainval.npz 2>/dev/null | awk '{print $3" bytes"}' || echo 'not yet')"
printf '  models         : %s/30\n' "$(aws s3 ls s3://$BUCKET/screen/models/ 2>/dev/null | grep -c '\.ubj')"
printf '  predictions    : %s files\n' "$(aws s3 ls s3://$BUCKET/screen/preds/ 2>/dev/null | grep -c 'shard_')"

echo
echo "=== status markers, this phase only ==="
for ID in $(aws ec2 describe-instances --region $REGION --filters Name=tag:Phase,Values=$PHASE \
            --query 'Reservations[].Instances[].InstanceId' --output text); do
  for K in DONE FAILED; do
    aws s3api head-object --bucket "$BUCKET" --key "status/$ID.$K.json" >/dev/null 2>&1 \
      && echo "  $ID  $K"
  done
done

echo
echo "tail a log: aws s3 cp s3://$BUCKET/logs/<instance-id>.log - | tail -40"
