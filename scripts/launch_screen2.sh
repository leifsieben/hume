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

# arm:us_per_mol -- PROJECTED PARALLEL RATE, built from two measurements rather than one.
#
# The previous table read these straight out of results/scale/c7i.4xlarge_cpu.json, which was
# WRONG BY 1.6x TO 12.8x: bench/bench_aws.py runs every arm in a sixteen-worker pool, so those
# are aggregate throughputs, and the production path featurized in one process. Phase 1's own
# logs give the serial truth, and the laptop gives the speedup the parallel path recovers:
#
#   arm               old table   phase 1 serial   parallel speedup   this table
#   hume_r4               144.5            582      2.23x measured         261
#   hume_r2               144.5            492      2.23x (as r4)          221
#   ecfp_rdkit_desc       464.1           5956      8.29x measured         718
#   chemberta_mtr         340.4            534      4.53x measured         118
#   chemeleon            2892.5           5304      UNMEASURED            5304
#   minimol              1544.7         not run     UNMEASURED            1545
#
# CHEMELEON IS DELIBERATELY NOT DISCOUNTED. chemprop is not installed on the laptop, so its
# speedup was never measured, and it is the single largest line in the budget. Carrying its
# serial rate means the plan is safe if it does not parallelize and simply finishes early if it
# does -- the opposite choice would under-provision the biggest arm on an assumption.
# minimol has no production rate at all (it never got past dependency install), so it keeps the
# four-worker benchmark figure, which is the configuration it will actually run in.
#
# Speedups measured on 12 cores; the boxes have 16, so these are conservative.
RATES="hume_r4:261 hume_r2:221 ecfp_rdkit_desc:718 chemberta_mtr:118 chemeleon:5304 minimol:1545"
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
# SHARD RANGE OVERRIDE, resolved here rather than at the launch loop so that DRY=1 describes the
# job that would actually run. It defaults to the whole test set, which is what a first launch
# wants. It exists because a relaunch usually does not: on 2026-10-30 the fleet hit the 256-vCPU
# quota partway through chemeleon, leaving shards [188,246) unscored, and with no range the only
# way to finish them was to re-score the 188 already in S3. Box counts scale with the span, so a
# partial range still lands near TARGET_H per box instead of inheriting the full run's count.
LO_SHARD="${LO_SHARD:-0}"
HI_SHARD="${HI_SHARD:-$N_SHARDS}"
SPAN=$(( HI_SHARD - LO_SHARD ))
[ "$SPAN" -gt 0 ] || { echo "FATAL: empty shard range [$LO_SHARD,$HI_SHARD)" >&2; exit 1; }
[ "$HI_SHARD" -le "$N_SHARDS" ] || { echo "FATAL: HI_SHARD $HI_SHARD > $N_SHARDS shards" >&2; exit 1; }
[ "$LO_SHARD" -ge 0 ] || { echo "FATAL: negative LO_SHARD $LO_SHARD" >&2; exit 1; }
[ "$SPAN" -eq "$N_SHARDS" ] || \
  echo "shard range [$LO_SHARD,$HI_SHARD) -- $SPAN of $N_SHARDS shards"
TOTAL_BOX_H=0; TOTAL_BOXES=0
echo
printf '%-18s %9s %10s %7s %9s\n' arm us/mol box-hours boxes h/box
for ARM in $ARMS; do
  R=$(rate_of "$ARM")
  [ -n "$R" ] || { echo "FATAL: no measured rate for arm '$ARM'" >&2; exit 1; }
  BH=$(awk -v r="$R" -v f="$SPAN" -v n="$N_SHARDS" \
       'BEGIN{printf "%.2f", r*100000000/1e6/3600*f/n}')
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
  #
  # MARKETS OVERRIDES THAT ORDER, AND EXISTS BECAUSE SPOT CAN LOSE A RACE. On 2026-10-30 this
  # region reclaimed c7i/m7i 4xlarge spot faster than the fleet could be relaunched -- 20
  # reclaims in under two hours, with minimol stuck at 16 of 246 shards across three launches.
  # Spot is right for the bulk of a run and wrong for its tail, when what is left is small, the
  # deadline is close, and another reclaim costs more than the instance does. MARKETS="ondemand"
  # skips spot entirely; the default is unchanged.
  for MARKET in ${MARKETS:-spot ondemand}; do
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
  BH=$(awk -v r="$R" -v f="$SPAN" -v n="$N_SHARDS" \
       'BEGIN{printf "%.4f", r*100000000/1e6/3600*f/n}')
  NB="${BOXES:-$(awk -v b="$BH" -v t="$TARGET_H" 'BEGIN{n=int(b/t); if(b/t>n)n++; if(n<1)n=1; if(n>40)n=40; print n}')}"
  for ((b = 0; b < NB; b++)); do
    L=$(( LO_SHARD + b * SPAN / NB )); H=$(( LO_SHARD + (b + 1) * SPAN / NB ))
    [ "$L" -lt "$H" ] || continue
    launch_one "$ARM" "$L" "$H" || exit 1
  done
done
echo
echo "launched $(wc -l < .screen2_instances | tr -d ' ') box(es); ids in .screen2_instances"
echo "  watch : ./scripts/watch_screen.sh screen2"
