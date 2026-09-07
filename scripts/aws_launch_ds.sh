#!/bin/bash
# launch_ds.sh <name> <hours> <arms...> [-- datasets...]
# One downstream box. Wall-clock cap is an ARGUMENT because it must be sized to the work:
# the first dsA carried a hard-coded `shutdown -h +330` against a 14h grid and would have been
# killed 40% through.
set -euo pipefail
B=hume-bench-use1-075120018132
NAME=$1; HOURS=$2; shift 2
ARMS=""; DSETS=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; DSETS="$*"; break; fi
  ARMS="$ARMS $1"; shift
done
ARMS=$(echo $ARMS); MIN=$(( HOURS * 60 ))
UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
shutdown -h +$MIN
export DS_ARMS="$ARMS"
export DS_DATASETS="$DSETS"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq; apt-get install -y -qq curl unzip >/dev/null 2>&1
if ! command -v aws >/dev/null; then
  curl -sf "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/a.zip
  unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install >/dev/null
fi
aws s3 cp s3://$B/boot/${BOOT:-boot_ds.sh} /root/boot_ds.sh
chmod +x /root/boot_ds.sh
DS_ARMS="$ARMS" DS_DATASETS="$DSETS" /root/boot_ds.sh
EOF
# SPOT IS A DIFFERENT QUOTA AND A DIFFERENT CAPACITY POOL. On-demand Standard and Spot Standard
# are 32 vCPU each, so spot doubles the fleet today rather than waiting on a pending increase.
# It is also the right risk trade here: every box ships its partial grid to S3 within 30s of
# finishing a dataset, so a reclaimed instance costs at most the dataset it was in the middle of
# -- which is exactly the checkpointing that makes spot a saving rather than a lottery.
# INSTANCE TYPE IS AN OVERRIDE because spot capacity is per (type, AZ) pool, not global. Four
# spot boxes were reclaimed within 90 minutes on 2026-09-01 with
# `instance-terminated-no-capacity` on c7i.4xlarge, while on-demand of the same type kept
# running -- so the answer to a capacity reclaim is a DIFFERENT POOL, not a retry of the same
# request. ITYPE=m7i.4xlarge or c6i.4xlarge are the same 16 vCPU at a similar price.
MARKET=""
if [ "${SPOT:-0}" = "1" ]; then
  MARKET='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
fi
IID=$(aws ec2 run-instances $MARKET \
  --image-id ami-0d7f022123f8ff19d --instance-type "${ITYPE:-c7i.4xlarge}" \
  --iam-instance-profile Arn=arn:aws:iam::075120018132:instance-profile/HumeBenchEC2 \
  --subnet-id subnet-0ee6327e8f5b315df --security-group-ids sg-0c90752120ffd64df \
  --instance-initiated-shutdown-behavior terminate \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=hume-bench},{Key=Role,Value=$NAME},{Key=Name,Value=hume-$NAME}]" \
  --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"

# VALIDATE WHAT WE CAPTURED, and sweep for boxes that carry no Name tag.
#
# Both halves of this come from a failure the CLIMB session hit on 2026-08-29 and diagnosed for
# me. They put a comment line inside a backslash continuation, so bash ended the command at the
# '#' and `aws ec2 run-instances` ran TRUNCATED -- valid enough for AWS to launch a real box, but
# with no --tag-specifications, no --user-data and no --query. The box came up untagged with
# nothing to do, their `case "$id" in i-*)` did not match the default JSON that came back, and
# the script reported "nothing launched" while a g5.4xlarge sat idle.
#
# My scripts have no comment-in-continuation (checked, all four), but I had the same missing
# guard: nothing here asserted that $IID was an instance id, so any malformed call would have
# been recorded as a launch and the queue's circuit breaker would have skipped it.
case "$IID" in
  i-[0-9a-f]*) ;;
  *) echo "FATAL: run-instances did not return an instance id. Got: ${IID:-<empty>}" >&2
     echo "  A truncated but VALID request still launches a box, so check the console for an" >&2
     echo "  untagged instance before retrying." >&2
     exit 1 ;;
esac

# An untagged box is invisible to every `--filters tag:Project` sweep, which is the only way a
# later session reconstructs this fleet from AWS alone. Report it rather than assume none exist:
# an empty result is also what a wrong query returns, so the count of TAGGED boxes is printed as
# evidence that the query discriminates at all.
orphans=$(aws ec2 describe-instances --filters "Name=instance-state-name,Values=running,pending" \
  --query "Reservations[].Instances[?IamInstanceProfile.Arn=='arn:aws:iam::075120018132:instance-profile/HumeBenchEC2' && !not_null(Tags[?Key=='Name'].Value | [0])].InstanceId" \
  --output text 2>/dev/null)
tagged=$(aws ec2 describe-instances --filters "Name=instance-state-name,Values=running,pending" \
  --query "Reservations[].Instances[?IamInstanceProfile.Arn=='arn:aws:iam::075120018132:instance-profile/HumeBenchEC2' && not_null(Tags[?Key=='Name'].Value | [0])].InstanceId" \
  --output text 2>/dev/null | wc -w)
if [ -n "$orphans" ]; then
  echo "WARNING: untagged instance(s) on HumeBenchEC2, invisible to a Project sweep: $orphans" >&2
fi

echo "$NAME $IID  cap=${HOURS}h  ($tagged tagged, ${orphans:-0 untagged}) arms:$ARMS  datasets:${DSETS:-ALL}"
