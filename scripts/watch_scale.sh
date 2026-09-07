#!/usr/bin/env bash
# Log-freshness watch for the scale box. Uptime says a box is billing; only the log says it is
# working, and a `running` instance whose log has not moved in twenty minutes is the silent
# failure this whole harness is written against.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BUCKET=hume-bench-use1-075120018132
IID="${1:-$(cat .scale_instance_id)}"
LAST=""; STALE=0
while true; do
  ST=$(aws ec2 describe-instances --instance-ids "$IID" \
       --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
  SZ=$(aws s3 ls "s3://$BUCKET/logs/$IID.log" 2>/dev/null | awk '{print $3}')
  NRES=$(aws s3 ls "s3://$BUCKET/results/$IID/" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date -u +%H:%M:%S) state=${ST:-?} log=${SZ:-none} results=$NRES"
  # A FAILED AWS CALL IS NOT A TERMINAL STATE. An earlier monitor treated an empty reply as
  # "instance gone", declared a spot reclaim that had not happened, and exited.
  case "$ST" in
    terminated|shutting-down) echo "TERMINAL: $ST"; break ;;
  esac
  if [ -n "$SZ" ] && [ "$SZ" = "$LAST" ]; then
    STALE=$((STALE+1))
    [ $STALE -ge 40 ] && echo "STALL: log has not grown in ~20 minutes"
  else
    STALE=0
  fi
  LAST="$SZ"
  sleep 30
done
aws s3 cp "s3://$BUCKET/logs/$IID.log" - 2>/dev/null | tail -40
