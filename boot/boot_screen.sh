#!/bin/bash
# Phases 1 and 2 of the LSD AmpC screen, in ONE script selected by PHASE:
#   PHASE=train  -- assemble the train/validation sets if nobody has yet, then train one arm's
#                   five XGBoost models.
#   PHASE=score  -- load that arm's five models and score shards [LO, HI) of the 100M test set.
# ARM, PHASE and (for score) LO/HI are exported by user-data before this runs.
#
# ONE SCRIPT FOR BOTH PHASES BECAUSE THE ENVIRONMENT IS THE RISKY PART. Training and scoring need
# the identical interpreter, pins and model weights -- if they diverged, an arm could be scored by
# a stack subtly different from the one that trained it, and nothing in the output would say so.
# Two scripts would make that divergence a single careless edit away.
#
# THE ENVIRONMENT BUILD IS LIFTED FROM boot/boot_ds.sh, DELIBERATELY AND ALMOST VERBATIM. That
# script's pins and its separate minimol interpreter are the product of several failed boots
# (graphium moves torch underneath transformers, so `AutoModel.from_pretrained` on the ChemBERTa
# weights dies with a ModuleNotFoundError naming Roberta on a box where minimol was the only
# thing that changed). Re-deriving an environment for these arms would re-earn those failures.
# The screen adds its own modules on top and changes nothing underneath.
set -uo pipefail
BUCKET=hume-bench-use1-075120018132
LOG=/var/log/screen.log
exec > >(tee -a "$LOG") 2>&1
echo "=== boot $(date -u +%FT%TZ) phase=${PHASE} arm=${ARM} ${LO:+shards [$LO,$HI)} ==="

TOK=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600" || true)
IID=$(curl -sf -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id || true)
if [ -z "$IID" ]; then echo "FATAL: no instance-id from IMDS"; shutdown -h now; fi
export HOME=/root
echo "instance $IID  phase $PHASE  arm $ARM"

ship() { aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID.log" --only-show-errors 2>/dev/null || true; }

# SNS IS A PUSH NOTIFICATION, NOT THE RECORD. The DONE/FAILED markers in S3 remain the only
# source of truth about what a box achieved; this just means a failure at 3am reaches a phone
# instead of waiting to be noticed. So a publish failure must never kill a run that is otherwise
# fine -- but it must also not pass silently, because "alerting configured and delivering
# nothing" is the exact trap this is meant to close. The instance role was granted sns:Publish
# scoped to this one topic; if that is ever missing, this prints loudly into the shipped log.
TOPIC="arn:aws:sns:us-east-1:075120018132:hume-screen-alerts"
notify() {   # $1=subject  $2=body
  if ! aws sns publish --topic-arn "$TOPIC" --subject "$1" --message "$2" \
        --query MessageId --output text >/dev/null 2>/tmp/sns.err; then
    echo "WARNING: SNS publish failed ($(tr -d '\n' < /tmp/sns.err | tail -c 200)).
The run is unaffected -- S3 status markers are the record -- but no push alert was sent."
  fi
}

FINISHED=0
on_exit() {
  [ "$FINISHED" = "1" ] && return
  echo "ABORTED: boot_screen.sh exited without reaching completion"
  notify "HUME screen ABORTED: $PHASE/$ARM" "Instance $IID aborted without completing.
phase=$PHASE arm=$ARM ${LO:+shards [$LO,$HI)}
log: s3://$BUCKET/logs/$IID.log"
  printf '{"instance_id":"%s","phase":"screen-%s","arm":"%s","error":"aborted early"}\n' "$IID" "$PHASE" "$ARM" > /tmp/F.json
  aws s3 cp /tmp/F.json "s3://$BUCKET/status/$IID.FAILED.json" --only-show-errors 2>/dev/null || true
  ship; shutdown -h now
}
trap on_exit EXIT
die() {
  echo "FAILED: $*"
  notify "HUME screen FAILED: $PHASE/$ARM" "$*
instance=$IID phase=$PHASE arm=$ARM ${LO:+shards [$LO,$HI)}
log: s3://$BUCKET/logs/$IID.log"
  printf '{"instance_id":"%s","phase":"screen-%s","arm":"%s","error":"%s"}\n' "$IID" "$PHASE" "$ARM" "$*" > /tmp/F.json
  aws s3 cp /tmp/F.json "s3://$BUCKET/status/$IID.FAILED.json" --only-show-errors || true
  ship
  # Terminate, not idle: the log ships every 30 s and every model is uploaded as it is trained,
  # so a failed box holds nothing S3 does not already have. It has no key pair either, so "left
  # up for inspection" is not inspectable.
  shutdown -h now
  exit 1
}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || die "apt update"
# libX* are NOT for drawing: chemprop imports cuik_molmaker, whose native extension links
# libXrender/libXext/libSM. A headless image ships none and `import chemprop` then fails with an
# error naming a graphics library.
apt-get install -y -qq build-essential cmake curl unzip git python3-dev \
  libxrender1 libxext6 libsm6 libx11-6 libglib2.0-0 >/dev/null || die "apt install"
command -v aws >/dev/null || { curl -sf "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/a.zip && unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install >/dev/null; }
( while true; do ship; sleep 30; done ) &
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null || die "uv install"
export PATH="/root/.local/bin:$PATH"
uv python install 3.12 >/dev/null 2>&1 || die "uv python"

mkdir -p /opt/ds && cd /opt/ds || die cd
aws s3 cp "s3://$BUCKET/dsbundle.tar.gz" . --only-show-errors || die "dsbundle download"
tar xzf dsbundle.tar.gz || die "dsbundle extract"
# The screen's own modules and the frozen split go on top of the dsbundle, not instead of it.
aws s3 cp "s3://$BUCKET/screenbundle.tar.gz" . --only-show-errors || die "screenbundle download"
tar xzf screenbundle.tar.gz || die "screenbundle extract"
[ -s /opt/ds/screen/arms.py ] || die "screenbundle did not contain screen/arms.py"
# The screenbundle ships bench_downstream.py ON TOP of the dsbundle's older copy, because the
# screen's HUME arms pass fp_radius explicitly and the dsbundle (staged 2026-10-27) predates that
# parameter. Assert it here, by signature, rather than discovering it in preflight: the failure
# there reads as "arm hume_r4 cannot featurize", which points at the arm and not at a stale file.
grep -q "fp_radius=None" /opt/ds/bench_downstream.py \
  || die "/opt/ds/bench_downstream.py has no fp_radius parameter -- the screenbundle did not
land on top of the dsbundle's older copy. The HUME arms pass the radius explicitly and cannot
run without it. Re-run scripts/stage_screen.sh and relaunch."
mkdir -p /root/.chemprop && cp chemprop_w/chemeleon_mp.pt /root/.chemprop/ 2>/dev/null || true

uv venv --python 3.12 --python-preference only-managed /opt/ds/env >/dev/null 2>&1 || die "venv"
P=/opt/ds/env/bin/python
uv pip install -q --python $P "numpy==2.4.6" "rdkit==2025.9.2" "scikit-build-core>=0.10" \
  "pybind11>=2.12" xgboost scikit-learn scipy pandas || die "base deps"
if [ "$ARM" = "chemberta_mtr" ] || [ "$ARM" = "chemeleon" ]; then
  uv pip install -q --python $P torch --index-url https://download.pytorch.org/whl/cpu || die "torch"
  uv pip install -q --python $P "transformers==5.16.1" "chemprop>=2.1" || die "nn deps"
fi
if [ "$ARM" = "minimol" ]; then
  # MINIMOL GETS ITS OWN INTERPRETER. See boot_ds.sh: graphium's pins move torch underneath the
  # environment. Versions are the ones the working laptop venv has, read off it rather than guessed.
  uv venv --python 3.12 --python-preference only-managed /opt/ds/envmini >/dev/null 2>&1 || die "venvmini"
  M=/opt/ds/envmini/bin/python
  uv pip install -q --no-config --python $M "torch==2.13.0" "torchvision==0.28.0" \
    --index-url https://download.pytorch.org/whl/cpu || die "minimol torch"
  uv pip install -q --no-config --python $M "numpy==2.2.6" "scipy==1.13.1" || die "minimol numpy"
  uv pip install -q --no-config --python $M --no-build-isolation minimol \
    "scipy==1.13.1" "numpy==2.2.6" "torch==2.13.0" || die "minimol"
  $M -c "
import numpy as np, torch
from minimol import Minimol
torch.set_grad_enabled(False)
v = np.asarray(Minimol()(['CCO','c1ccccc1'])[0]); assert v.shape == (512,), v.shape
print('  minimol venv ok', v.shape)" || die "minimol venv builds but cannot embed"
fi
uv pip install -q --python $P /opt/ds || die "molhume build"

export PYTHONPATH=/opt/ds
export CHEMBERTA_PATH=/opt/ds/models_hf/ChemBERTa-2-MTR
export MINIMOL_PY=/opt/ds/envmini/bin/python
export AWS_REGION=us-east-1
cd /opt/ds || die cd

# --- PREFLIGHT: exercise THIS ARM through the same code path the run will call ---------------
echo "=== preflight ==="
$P -c "
import sys; sys.path.insert(0,'/opt/ds'); sys.path.insert(0,'/opt/ds/screen')
import numpy as np, xgboost, molhume
sys.path.insert(0,'/opt/ds/screen')
import arms as A
A.quiet_rdkit()
print('molhume cols', len(molhume.ALL_COLUMNS), 'xgboost', xgboost.__version__)
arm = '$ARM'
X = A.featurize(arm, ['CC(=O)Oc1ccccc1C(=O)O','c1ccccc1O'])
assert X.shape[0] == 2 and X.shape[1] > 100, X.shape
assert np.isfinite(X).any(), 'featurizer produced no finite values'
fw = A.feature_weights(arm, X.shape[1])
print('  arm', arm, 'ok', X.shape, 'feature_weights', 'none' if fw is None else f'w={fw.max():.0f}')
m = A.make_head(arm, X.shape[1], 0)
print('  head', {k: m.get_params()[k] for k in ('n_estimators','max_depth','colsample_bynode','early_stopping_rounds')})
" || die "preflight: arm $ARM cannot featurize or build a head"
# A HUME arm must get the radius it ASKED for, not whatever the installed package defaults to.
if [ "$ARM" = "hume_r4" ] || [ "$ARM" = "hume_r2" ]; then
  $P -c "
import sys; sys.path.insert(0,'/opt/ds'); sys.path.insert(0,'/opt/ds/screen')
import numpy as np, arms as A
A.quiet_rdkit()
s = ['CC(=O)Oc1ccccc1C(=O)O','C1CCCCC1NC(=O)c1ccccc1']
r2, r4 = A.featurize('hume_r2', s), A.featurize('hume_r4', s)
assert not np.allclose(r2[:,-2048:], r4[:,-2048:], equal_nan=True), \
    'r2 and r4 produce the SAME fingerprint -- fp_radius is not reaching the featurizer, so both HUME arms would be the same arm'
assert np.allclose(r2[:,:-2048], r4[:,:-2048], equal_nan=True), \
    'r2 and r4 differ in the DESCRIPTOR block; the radius must change only the fingerprint'
print('  radius wired through: fingerprint differs, descriptors identical')
" || die "preflight: fp_radius is not reaching the HUME featurizer"
fi
free -g | head -2
df -h /opt | tail -1

mkdir -p /opt/ds/data/lsd_ampc /opt/ds/out/models /opt/ds/out/preds

if [ "$PHASE" = "train" ]; then
  # --- phase 1a: assemble train/validation, ONCE for all arms -------------------------------
  if aws s3 cp "s3://$BUCKET/screen/trainval.npz" /opt/ds/data/lsd_ampc/trainval.npz \
       --only-show-errors 2>/dev/null; then
    echo "=== trainval.npz already built by another box; reusing it ==="
  else
    echo "=== phase 1a: building trainval.npz ($(nproc) cores) ==="
    mkdir -p /opt/ds/data/lsd_ampc/smiles
    aws s3 sync "s3://$BUCKET/screen/smiles/" /opt/ds/data/lsd_ampc/smiles/ \
      --exclude "*" --include "shard_*.npz" --only-show-errors || die "smiles sync"
    N=$(ls /opt/ds/data/lsd_ampc/smiles/shard_*.npz 2>/dev/null | wc -l)
    [ "$N" = 246 ] || die "synced $N of 246 phase-0 shard files; trainval must be built over the whole library"
    $P screen/collect_trainval.py --smiles-dir /opt/ds/data/lsd_ampc/smiles \
       --out /opt/ds/data/lsd_ampc/trainval.npz --workers "$(nproc)" || die "collect_trainval"
    aws s3 cp /opt/ds/data/lsd_ampc/trainval.npz "s3://$BUCKET/screen/trainval.npz" \
      --only-show-errors || die "trainval upload"
    aws s3 cp /opt/ds/data/lsd_ampc/trainval.json "s3://$BUCKET/screen/trainval.json" \
      --only-show-errors || true
  fi

  # --- phase 1b: train this arm's five models -----------------------------------------------
  echo "=== phase 1b: training $ARM ==="
  export SCREEN_S3_MODELS="s3://$BUCKET/screen/models"
  $P screen/train.py --arm "$ARM" --trainval /opt/ds/data/lsd_ampc/trainval.npz \
     --out /opt/ds/out/models || die "train $ARM"

  # COMPLETION FROM ACHIEVED WORK: five models, each loadable, each with a recorded fit.
  $P - <<PY > /tmp/DONE.json || die "completion check"
import json, sys
sys.path.insert(0, "/opt/ds")
import xgboost as xgb
d = json.load(open("/opt/ds/out/models/${ARM}_train.json"))
reps = d["replicates"]
assert len(reps) == 5, f"{len(reps)} replicates recorded, expected 5"
for r in reps:
    m = xgb.XGBRegressor(); m.load_model(f"/opt/ds/out/models/{r['model']}")
    assert m.n_estimators > 0
    assert r["n_train"] > 900_000, f"replicate {r['replicate']} trained on only {r['n_train']} molecules"
json.dump({"instance_id": "$IID", "phase": "train", "arm": "$ARM", "complete": True,
           "models": [r["model"] for r in reps],
           "val_rmse": [r["val_rmse"] for r in reps],
           "best_iteration": [r["best_iteration"] for r in reps],
           "provenance": d["provenance"]}, open("/tmp/DONE.json", "w"), indent=1)
print("complete: 5 models, val rmse", [round(r["val_rmse"], 4) for r in reps])
PY

elif [ "$PHASE" = "score" ]; then
  # --- phase 2: score shards [LO, HI) with this arm's five models ----------------------------
  : "${LO:?PHASE=score needs LO}" "${HI:?PHASE=score needs HI}"
  echo "=== phase 2: scoring $ARM over shards [$LO, $HI) ==="
  aws s3 sync "s3://$BUCKET/screen/models/" /opt/ds/out/models/ \
    --exclude "*" --include "${ARM}_rep*.ubj" --only-show-errors || die "model sync"
  NM=$(ls /opt/ds/out/models/${ARM}_rep*.ubj 2>/dev/null | wc -l)
  [ "$NM" = 5 ] || die "found $NM of 5 models for $ARM in s3://$BUCKET/screen/models/.
Scoring with fewer would report a replicate spread over whichever models happened to exist.
Has phase 1 for this arm finished?"

  # ONLY THIS BOX'S SHARDS. Syncing all 246 would pull 2.6 GB to use a fraction of it; a box
  # doing 30 shards needs ~300 MB.
  mkdir -p /opt/ds/data/lsd_ampc/smiles
  for i in $(seq "$LO" $((HI - 1))); do
    F=$(printf 'shard_%05d.npz' "$i")
    aws s3 cp "s3://$BUCKET/screen/smiles/$F" "/opt/ds/data/lsd_ampc/smiles/$F" \
      --only-show-errors || die "download $F"
  done
  NS=$(ls /opt/ds/data/lsd_ampc/smiles/shard_*.npz | wc -l)
  [ "$NS" = "$((HI - LO))" ] || die "have $NS of $((HI - LO)) shard files"

  export SCREEN_S3_PREDS="s3://$BUCKET/screen/preds"
  $P screen/score_test.py --arm "$ARM" --lo "$LO" --hi "$HI" \
     --smiles-dir /opt/ds/data/lsd_ampc/smiles --models /opt/ds/out/models \
     --out /opt/ds/out/preds || die "score $ARM [$LO,$HI)"

  # COMPLETION FROM ACHIEVED WORK: every shard scored, every prediction array the right shape,
  # and the molecule count equal to the test rows those shards actually hold.
  $P - <<PY > /tmp/DONE.json || die "completion check"
import json, glob, sys
import numpy as np
man = json.load(open(f"/opt/ds/out/preds/${ARM}_preds_%05d_%05d.json" % ($LO, $HI)))
assert len(man["shards"]) == $HI - $LO, f"{len(man['shards'])} shards, expected {$HI - $LO}"
tot = 0
for i in range($LO, $HI):
    want = np.load(f"/opt/ds/data/lsd_ampc/smiles/shard_%05d.npz" % i)["test_pos"].size
    z = np.load(f"/opt/ds/out/preds/${ARM}_shard_%05d.npz" % i)
    assert z["pred"].shape == (want, 5), f"shard {i}: pred {z['pred'].shape}, expected ({want}, 5)"
    tot += want
json.dump({"instance_id": "$IID", "phase": "score", "arm": "$ARM", "lo": $LO, "hi": $HI,
           "shards": len(man["shards"]), "n_scored": tot, "complete": True,
           "provenance": man["provenance"]}, open("/tmp/DONE.json", "w"), indent=1)
print(f"complete: {$HI - $LO} shards, {tot:,} molecules scored")
PY

else
  die "PHASE must be 'train' or 'score', got '$PHASE'"
fi

aws s3 cp /tmp/DONE.json "s3://$BUCKET/status/$IID.DONE.json" --only-show-errors || die "DONE upload"
cat /tmp/DONE.json
FINISHED=1
echo "=== COMPLETE $(date -u +%FT%TZ) ==="
notify "HUME screen complete: $PHASE/$ARM" "$(cat /tmp/DONE.json)"
ship
sleep 5
shutdown -h now
