#!/bin/bash
# Phase 0 of the LSD AmpC screen: read the SMILES column for shards [LO, HI) and keep only the
# molecules the frozen split selects. Runs at boot from user-data; nothing depends on a laptop,
# an SSH channel or a session. LO and HI are exported by user-data before this runs.
#
# This box installs numpy, pyarrow and huggingface_hub and NOTHING ELSE -- no RDKit, no torch, no
# molhume. Phase 0 computes no features, so it comes up in about two minutes instead of twenty,
# and it cannot fail on a featurization dependency it never uses.
set -uo pipefail
BUCKET=hume-bench-use1-075120018132
PREFIX=screen/smiles
LOG=/var/log/screen0.log
exec > >(tee -a "$LOG") 2>&1
echo "=== boot $(date -u +%FT%TZ) shards [${LO}, ${HI}) ==="

# INSTANCE ID MUST BE NON-EMPTY: if this is blank every box writes the same log key and they
# overwrite each other, which is a documented failure of this fleet. Fatal here.
TOK=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600" || true)
IID=$(curl -sf -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id || true)
if [ -z "$IID" ]; then echo "FATAL: could not read instance-id from IMDS"; shutdown -h now; fi
echo "instance $IID  shards [${LO}, ${HI})"

ship() { aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID.log" --only-show-errors 2>/dev/null || true; }

FINISHED=0
on_exit() {
  [ "$FINISHED" = "1" ] && return
  echo "ABORTED: boot_screen0.sh exited without reaching completion"
  printf '{"instance_id":"%s","lo":%s,"hi":%s,"error":"aborted early"}\n' "$IID" "$LO" "$HI" > /tmp/FAILED.json
  aws s3 cp /tmp/FAILED.json "s3://$BUCKET/status/$IID.FAILED.json" --only-show-errors 2>/dev/null || true
  ship; shutdown -h now
}
trap on_exit EXIT

die() {
  echo "FAILED: $*"
  printf '{"instance_id":"%s","lo":%s,"hi":%s,"error":"%s"}\n' "$IID" "$LO" "$HI" "$*" > /tmp/FAILED.json
  aws s3 cp /tmp/FAILED.json "s3://$BUCKET/status/$IID.FAILED.json" --only-show-errors || true
  ship
  # Terminate rather than idle. The log ships every 30 s and every finished shard is already in
  # S3, so a running box holds nothing that cannot be read from S3 -- it is a bill, not a
  # diagnostic. These boxes have no key pair, so "left up for inspection" is not inspectable.
  shutdown -h now
  exit 1
}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || die "apt update"
apt-get install -y -qq curl unzip git >/dev/null || die "apt install"
if ! command -v aws >/dev/null; then
  curl -sf "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/a.zip || die "awscli dl"
  unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install >/dev/null || die "awscli install"
fi
# The log shipper starts only now: before the AWS CLI exists `aws s3 cp` is a command-not-found
# loop, which is how an earlier version of this fleet sat running and silent with an empty log.
( while true; do ship; sleep 30; done ) &

curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null || die "uv install"
export PATH="/root/.local/bin:$PATH"
uv python install 3.12 >/dev/null 2>&1 || die "uv python install"

mkdir -p /opt/screen && cd /opt/screen || die "cd"
aws s3 cp "s3://$BUCKET/screenbundle.tar.gz" . --only-show-errors || die "bundle download"
tar xzf screenbundle.tar.gz || die "bundle extract"

uv venv --python 3.12 --python-preference only-managed /opt/screen/env >/dev/null 2>&1 || die "venv"
P=/opt/screen/env/bin/python
# --no-config: /opt/screen carries no pyproject, but keep the flag so a bundle that later does
# cannot silently impose this project's rdkit/numpy constraints on a box that needs neither.
uv pip install -q --no-config --python $P "numpy==2.4.6" "pyarrow==26.0.0" \
   "huggingface_hub[hf_transfer]==1.2.1" || die "python deps"
export HF_HUB_ENABLE_HF_TRANSFER=1
export AWS_REGION=us-east-1
for V in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS; do export $V=4; done

# --- PREFLIGHT: the box must be able to do the job before it starts billing for it ----------
echo "=== preflight ==="
$P -c "
import numpy, pyarrow, pyarrow.parquet
print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
pf = pq.ParquetFile(HfFileSystem().open(
  'datasets/IrwinLab/LSD_AmpC_Liu_2025/docking_results/train-00000-of-00246.parquet','rb'))
assert [f.name for f in pf.schema_arrow] == ['smiles','zincid','score'], pf.schema_arrow
print('hf reachable; shard 0 has', format(pf.metadata.num_rows, ','), 'rows')
" || die "preflight: cannot read the dataset from HuggingFace"
# Disk: the kept SMILES for 31 shards are about 250 MB, but the parquet read buffers a whole
# shard (~150 MB) at a time. Assert headroom rather than discovering it at shard 20.
# `tr -dcs '0-9'` was wrong here: -d and -s together require two operand sets, so tr errored,
# AVAIL came out empty, and the box killed itself on a disk check that had measured nothing.
AVAIL=$(df -BG --output=avail /opt | tail -1 | tr -d 'G ')
case "$AVAIL" in ''|*[!0-9]*) die "could not read free space on /opt (df gave '$AVAIL')";; esac
[ "$AVAIL" -ge 20 ] || die "only ${AVAIL}G free on /opt; need 20G"
$P -c "
import sys; sys.path.insert(0,'/opt/screen/screen')
import split
spec = split.load_spec()
sizes, starts, n = split.geometry()
assert n == spec['n_molecules'], (n, spec['n_molecules'])
print('split spec loaded:', format(n,','), 'molecules,', len(spec['replicates']), 'replicates')
" || die "preflight: split spec or score files missing from the bundle"
free -g | head -2

echo "=== extract ==="
cd /opt/screen || die cd
$P screen/extract_smiles.py --lo "$LO" --hi "$HI" \
   --out /opt/screen/out --s3 "s3://$BUCKET/$PREFIX" || die "extract shards [$LO,$HI)"

# --- COMPLETION: extract_smiles.py already asserted that the molecules kept equal the distinct
# split indices in this range, and exited non-zero otherwise. Re-read the manifest it wrote so
# the marker in S3 carries the achieved count, not just the fact that the script returned 0.
MAN=$(printf '/opt/screen/out/manifest_%05d_%05d.json' "$LO" "$HI")
[ -s "$MAN" ] || die "no manifest at $MAN"
$P -c "
import json, sys
m = json.load(open('$MAN'))
n = len(m['shards']); kept = m['n_kept']
assert n == $HI - $LO, f'manifest covers {n} shards, expected {$HI - $LO}'
assert kept > 0
json.dump({'instance_id':'$IID','lo':$LO,'hi':$HI,'shards':n,'n_kept':kept,
           'complete':True,'provenance':m['provenance']}, open('/tmp/DONE.json','w'), indent=1)
print('complete:', n, 'shards,', format(kept,','), 'molecules')
" || die "completion check"
aws s3 cp /tmp/DONE.json "s3://$BUCKET/status/$IID.DONE.json" --only-show-errors || die "DONE upload"
cat /tmp/DONE.json
FINISHED=1
echo "=== COMPLETE $(date -u +%FT%TZ) ==="
ship
sleep 5
shutdown -h now
