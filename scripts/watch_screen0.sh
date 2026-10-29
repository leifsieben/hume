#!/usr/bin/env bash
# What is phase 0 doing right now, read from AWS alone -- no laptop state, no session memory.
# Safe to run from a fresh session: it finds the boxes by tag.
set -uo pipefail
BUCKET=hume-bench-use1-075120018132
REGION=us-east-1
echo "=== instances tagged Phase=screen0 ==="
aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Phase,Values=screen0 Name=instance-state-name,Values=pending,running,shutting-down \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,Tags[?Key==`Shards`]|[0].Value]' \
  --output text | sort -k5 || echo "  (none running)"

echo
echo "=== status markers ==="
aws s3 ls "s3://$BUCKET/status/" --region $REGION 2>/dev/null | grep -E 'DONE|FAILED' || echo "  (none yet)"

echo
echo "=== shard files produced: $(aws s3 ls s3://$BUCKET/screen/smiles/ --region $REGION 2>/dev/null | grep -c 'shard_') of 246 ==="

echo
echo "=== log freshness (a running box with a stale log is the silent failure) ==="
NOW=$(date -u +%s)
for ID in $(aws ec2 describe-instances --region $REGION \
     --filters Name=tag:Phase,Values=screen0 Name=instance-state-name,Values=running \
     --query 'Reservations[].Instances[].InstanceId' --output text); do
  MOD=$(aws s3api head-object --bucket "$BUCKET" --key "logs/$ID.log" \
        --query 'LastModified' --output text 2>/dev/null || echo "")
  if [ -z "$MOD" ]; then echo "  $ID  NO LOG YET"; continue; fi
  T=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "${MOD%%+*}" +%s 2>/dev/null \
      || date -u -d "$MOD" +%s 2>/dev/null || echo 0)
  AGE=$(( (NOW - T) / 60 ))
  printf '  %s  log %3d min old%s\n' "$ID" "$AGE" \
    "$([ "$AGE" -gt 10 ] && echo '   <-- STALE, check it' || true)"
done

echo
echo "tail a log: aws s3 cp s3://$BUCKET/logs/<instance-id>.log - | tail -40"
