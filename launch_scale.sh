#!/usr/bin/env bash
# One self-contained CPU box for the Figure C / D cost axis. No SSH, no key pair, no laptop in
# the loop: it boots, downloads boot/boot.sh and bundle.tar.gz from S3, runs bench_aws.py for the
# arms named in ONLY_ARMS, ships every result and the log to S3 on a timer, and terminates
# itself. Closing the laptop is a non-event; `./resume_scale.sh` reads the state back from AWS.
#
#     ONLY_ARMS="hume hume_minimal hume_no_new" ./launch_scale.sh
#
# THE INSTANCE TYPE IS NOT A PREFERENCE, IT IS THE COMPARISON. Every cost number already on
# Figure C's x-axis was measured on c7i.4xlarge, and us/mol is not portable across instance
# types -- a run on a c6i would produce numbers that cannot be plotted next to the existing ones
# without silently mixing two machines on one axis. So this refuses to fall back to another
# type; it falls back across AVAILABILITY ZONES only, and if c7i.4xlarge has no capacity
# anywhere it fails loudly rather than measuring the wrong thing cheaply.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BUCKET=hume-bench-use1-075120018132
TYPE=c7i.4xlarge
ROLE="${ROLE:-cpu}"
ONLY_ARMS="${ONLY_ARMS:-}"
PROFILE=HumeBenchEC2
GID=sg-082c5d5ebeb465414

AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
      --query 'Parameters[0].Value' --output text) || { echo "FATAL: cannot resolve AMI"; exit 1; }
echo "ubuntu AMI $AMI   type $TYPE   role $ROLE   arms: ${ONLY_ARMS:-<all>}"

# BOOT_SCALE.SH IS INLINED AND GZIPPED, AND BOTH HALVES OF THAT ARE FIXES FOR REAL FAILURES.
#
# 1. It used to be FETCHED: user-data ran `aws s3 cp s3://.../boot.sh`. The stock Ubuntu image
#    has no AWS CLI -- boot_scale.sh is where the CLI gets installed -- so the very first command
#    failed, nothing else ran, and the box sat `running` and silent, billing, with an empty
#    console and no log in S3 because the log shipper lives inside the script that never started.
#    Ten minutes of on-demand c7i for nothing, and no diagnostic on the box to say why.
#
# 2. Inlining it plainly does not fit. EC2's 16,384-byte user-data limit is applied to the
#    BASE64 form, not the raw bytes: 14.2 KB of script is 19.0 KB encoded, and RunInstances
#    refuses it. cloud-init decompresses gzipped user-data on its own, so the script goes in
#    compressed -- 14.2 KB becomes about 4.5 KB, which is not close to the limit.
UD_FILE=$(mktemp -t humeud).gz
{ printf '#!/bin/bash\nexport ROLE=%s\nexport ONLY_ARMS="%s"\n' "$ROLE" "$ONLY_ARMS"
  tail -n +2 boot_scale.sh; } | gzip -9 > "$UD_FILE"
UD_BYTES=$(wc -c < "$UD_FILE" | tr -d ' ')
# The refusal threshold is on the base64 form, so check THAT and not the raw size -- checking the
# raw size is exactly the mistake that produced failure 2.
UD_B64=$(( (UD_BYTES + 2) / 3 * 4 ))
echo "user-data: $UD_BYTES bytes gzipped, $UD_B64 base64 of 16384"
if [ "$UD_B64" -gt 16384 ]; then
  echo "FATAL: boot_scale.sh no longer fits in user-data even compressed. Go back to fetching it
from S3 -- and have user-data install the AWS CLI FIRST, because the stock image has none." >&2
  exit 1
fi

SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
          --query 'Subnets[].SubnetId' --output text)
for S in $SUBNETS; do
  for MARKET in spot ondemand; do
    # `${EXTRA[@]+"${EXTRA[@]}"}` RATHER THAN `"${EXTRA[@]}"`. Under `set -u`, bash 3.2 (which is
    # what macOS ships) treats an EMPTY array expansion as an unbound variable and aborts -- so
    # the on-demand path, whose EXTRA is empty by construction, killed the loop on its first try.
    EXTRA=()
    [ "$MARKET" = spot ] && EXTRA=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
    OUT=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" \
      --security-group-ids "$GID" --subnet-id "$S" --associate-public-ip-address \
      --iam-instance-profile "Name=$PROFILE" \
      --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --user-data "fileb://$UD_FILE" ${EXTRA[@]+"${EXTRA[@]}"} \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hume-scale},{Key=Project,Value=HUME},{Key=Arms,Value=${ONLY_ARMS// /-}}]" \
      --query 'Instances[0].InstanceId' --output text 2>&1)
    case "$OUT" in
      i-*) echo "LAUNCHED $OUT  type=$TYPE  subnet=$S  market=$MARKET"
           echo "$OUT" > .scale_instance_id
           echo "  log      : aws s3 cp s3://$BUCKET/logs/$OUT.log -"
           echo "  results  : aws s3 ls s3://$BUCKET/results/$OUT/"
           exit 0 ;;
      *InsufficientInstanceCapacity*) echo "  no $MARKET capacity in $S" ;;
      *Unsupported*) echo "  $TYPE not offered in this AZ ($S)" ;;
      # PRINT THE WHOLE ERROR, ALWAYS. An empty branch here is how a permissions failure reads
      # as a capacity failure -- see the skill note on testing IAM by the error code and never
      # by the exit status.
      *) echo "  $S/$MARKET refused: ${OUT}" ;;
    esac
  done
done
echo "NO c7i.4xlarge CAPACITY IN ANY AZ, spot or on-demand. Not falling back to another instance
type: us/mol is not portable across hardware and Figure C's existing points are all c7i.4xlarge.
Retry later, or accept a different type and re-measure EVERY arm on it." >&2
exit 1
