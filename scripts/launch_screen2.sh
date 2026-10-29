#!/usr/bin/env bash
# Phase 2 of the LSD AmpC screen: score the 100M test set, fanned out over (arm, shard range).
#
#     ARMS="hume_r4 hume_r2" ./scripts/launch_screen2.sh
#     ARMS="chemeleon" BOXES=16 ./scripts/launch_screen2.sh
#     DRY=1 ARMS="all" ./scripts/launch_screen2.sh      # print the plan, launch nothing
#
# BOX COUNTS COME FROM MEASURED THROUGHPUT, not from guessing. The per-arm us/mol below are this
# project's own Figure C numbers at N=1,000,000 on c7i.4xlarge, so the count for each arm is
# whatever makes its share of the work land in roughly TARGET_H hours of wall clock. Prediction
# with five models adds 2.5 us/mol -- measured, about 1% -- and is ignored here.
#
# ALL CPU, DELIBERATELY. CheMeleon is 2892 us/mol on CPU against 1047 on a g6.xlarge, and
# ChemBERTa 340 against 112, so a GPU wave would cost about $5 less. It would also add a CUDA
# torch path to thirty boxes for that $5. Wall clock is set by how many boxes run in parallel,
# not by per-box speed, so the cheap and boring choice wins. The GPU numbers are in
# results/scale/g6.xlarge_gpu.json if a cheaper re-run is ever wanted.
#
# SPOT ACROSS EVERY AZ BEFORE ANY ON-DEMAND. The phase 1 launcher tried both markets within each
# AZ, so it took an on-demand box in the first AZ while spot sat available in the second -- about
# 4x the price. With thirty boxes that ordering is the difference between ~$36 and ~$110.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUCKET=hume-bench-use1-075120018132
PROFILE=HumeBenchEC2
GID=sg-082c5d5ebeb465414
N_SHARDS=246
TARGET_H="${TARGET_H:-5}"
TYPES="${TYPES:-c7i.4xlarge c6i.4xlarge m7i.4xlarge m6i.4xlarge}"
DRY="${DRY:-}"

# arm:us_per_mol -- measured, N=1M, c7i.4xlarge. hume_r2 is hume_r4 less the cheaper fingerprint.
RATES="hume_r4:144.5 hume_r2:141.4 ecfp_rdkit_desc:464.1 chemberta_mtr:340.4 chemeleon:2892.5 minimol:1544.7"
ARMS="${ARMS:?set ARMS to a space-separated list, or \"all\"}"
[ "$ARMS" = "all" ] && ARMS="hume_r4 hume_r2 ecfp_rdkit_desc chemberta_mtr chemeleon minimol"

rate_of () { for kv in $RATES; do case "$kv" in "$1":*) echo "${kv#*:}"; return;; esac; done; echo ""; }

if [ -z "$DRY" ]; then
  LOCAL_SHA=$(shasum -a 256 dist/screenbundle.tar.gz 2>/dev/null | cut -d' ' -f1)
  REMOTE_SHA=$(aws s3api head-object --bucket "$BUCKET" --key screenbundle.tar.gz \
               --query 'Metadata.sha256' --output text 2>/dev/null || echo none)
  [ -n "$LOCAL_SHA" ] && [ "$LOCAL_SHA" = "$REMOTE_SHA" ] || {
    echo "FATAL: dist/screenbundle.tar.gz missing or does not match S3.
  local  ${LOCAL_SHA:-none}
  remote $REMOTE_SHA
Run scripts/stage_screen.sh before launching." >&2; exit 1; }
  AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
        --query 'Parameters[0].Value' --output text) || { echo "FATAL: cannot resolve AMI" >&2; exit 1; }
  SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
            --query 'Subnets[].SubnetId' --output text) || { echo "FATAL: no subnets" >&2; exit 1; }
  echo "AMI $AMI   bundle $LOCAL_SHA"
fi

# --- the plan, printed before anything is spent -----------------------------------------------
TOTAL_BOX_H=0; TOTAL_BOXES=0
echo
printf '%-18s %9s %10s %7s %9s\n' arm us/mol box-hours boxes h/box
for ARM in $ARMS; do
  R=$(rate_of "$ARM")
  [ -n "$R" ] || { echo "FATAL: no measured rate for arm '$ARM'" >&2; exit 1; }
  BH=$(awk -v r="$R" 'BEGIN{printf "%.2f", r*100000000/1e6/3600}')
  NB="${BOXES:-$(awk -v b="$BH" -v t="$TARGET_H" 'BEGIN{n=int(b/t); if(b/t>n)n++; if(n<1)n=1; if(n>40)n=40; print n}')}"
  HPB=$(awk -v b="$BH" -v n="$NB" 'BEGIN{printf "%.1f", b/n}')
  printf '%-18s %9s %10s %7s %9s\n' "$ARM" "$R" "$BH" "$NB" "$HPB"
  TOTAL_BOX_H=$(awk -v a="$TOTAL_BOX_H" -v b="$BH" 'BEGIN{printf "%.2f", a+b}')
  TOTAL_BOXES=$((TOTAL_BOXES + NB))
done
echo
printf 'total: %s box-hours over %s boxes; ~$%s at spot $0.2337/h, ~$%s on-demand $0.714/h\n' \
  "$TOTAL_BOX_H" "$TOTAL_BOXES" \
  "$(awk -v b="$TOTAL_BOX_H" 'BEGIN{printf "%.0f", b*0.2337}')" \
  "$(awk -v b="$TOTAL_BOX_H" 'BEGIN{printf "%.0f", b*0.714}')"
[ -n "$DRY" ] && { echo; echo "DRY=1, nothing launched."; exit 0; }

launch_one () {        # $1=arm $2=lo $3=hi
  local ARM=$1 LO=$2 HI=$3 UD B OUT T S
  UD=$(mktemp -t s2ud).gz
  { printf '#!/bin/bash\nexport PHASE=score\nexport ARM=%s\nexport LO=%s\nexport HI=%s\n' "$ARM" "$LO" "$HI"
    tail -n +2 boot/boot_screen.sh; } | gzip -9 > "$UD"
  B=$(( ( $(wc -c < "$UD" | tr -d ' ') + 2) / 3 * 4 ))
  # The limit is on the BASE64 form, not the raw bytes -- checking raw size is what made
  # RunInstances refuse an earlier launch in this project.
  [ "$B" -le 16384 ] || { echo "FATAL: user-data $B base64 bytes > 16384" >&2; rm -f "$UD"; return 1; }
  # MARKET OUTERMOST: every AZ is tried on spot before any on-demand is considered.
  for MARKET in spot ondemand; do
    for T in $TYPES; do
      for S in $SUBNETS; do
        EXTRA=()
        [ "$MARKET" = spot ] && EXTRA=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
        OUT=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" \
          --security-group-ids "$GID" --subnet-id "$S" --associate-public-ip-address \
          --iam-instance-profile "Name=$PROFILE" \
          --instance-initiated-shutdown-behavior terminate \
          --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
          --user-data "fileb://$UD" ${EXTRA[@]+"${EXTRA[@]}"} \
          --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hume-screen2},{Key=Project,Value=HUME},{Key=Phase,Value=screen2},{Key=Arm,Value=$ARM},{Key=Shards,Value=${LO}-${HI}}]" \
          --query 'Instances[0].InstanceId' --output text 2>&1)
        case "$OUT" in
          i-*) printf '  %s  %-17s shards [%3d,%3d)  %-14s %s\n' "$OUT" "$ARM" "$LO" "$HI" "$T" "$MARKET"
               echo "$OUT $ARM $LO $HI $MARKET" >> .screen2_instances
               rm -f "$UD"; return 0 ;;
          *InsufficientInstanceCapacity*|*Unsupported*) ;;
          *) echo "  $ARM $T/$S/$MARKET refused: $OUT" ;;
        esac
      done
    done
  done
  echo "FATAL: no capacity for $ARM [$LO,$HI) in [$TYPES], any AZ, spot or on-demand." >&2
  rm -f "$UD"; return 1
}

echo
for ARM in $ARMS; do
  R=$(rate_of "$ARM")
  BH=$(awk -v r="$R" 'BEGIN{printf "%.4f", r*100000000/1e6/3600}')
  NB="${BOXES:-$(awk -v b="$BH" -v t="$TARGET_H" 'BEGIN{n=int(b/t); if(b/t>n)n++; if(n<1)n=1; if(n>40)n=40; print n}')}"
  for ((b = 0; b < NB; b++)); do
    L=$(( b * N_SHARDS / NB )); H=$(( (b + 1) * N_SHARDS / NB ))
    [ "$L" -lt "$H" ] || continue
    launch_one "$ARM" "$L" "$H" || exit 1
  done
done
echo
echo "launched $(wc -l < .screen2_instances | tr -d ' ') box(es); ids in .screen2_instances"
echo "  watch : ./scripts/watch_screen.sh screen2"
