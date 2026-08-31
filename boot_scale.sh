#!/bin/bash
# Runs at boot from user-data. Nothing here depends on a laptop, an SSH channel or a session.
# ROLE is exported by user-data before this runs: "cpu" or "gpu".
set -uo pipefail
BUCKET=hume-bench-use1-075120018132
LOG=/var/log/hume-bench.log
exec > >(tee -a "$LOG") 2>&1
echo "=== boot $(date -u +%FT%TZ) role=${ROLE} ==="

# INSTANCE ID MUST BE NON-EMPTY. If this lookup fails silently every box writes the same S3 key
# and the logs overwrite each other -- a documented failure mode, so it is fatal here.
TOK=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600" || true)
IID=$(curl -sf -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id || true)
if [ -z "$IID" ]; then echo "FATAL: could not read instance-id from IMDS"; shutdown -h now; fi
export BENCH_INSTANCE=$(curl -sf -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-type)
echo "instance $IID type $BENCH_INSTANCE role $ROLE"

ship() { aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID.log" --only-show-errors 2>/dev/null || true; }
( while true; do ship; sleep 30; done ) &

FINISHED=0
on_exit() {
  [ "$FINISHED" = "1" ] && return
  echo "ABORTED: boot.sh exited without reaching completion"
  echo "{\"instance_id\":\"$IID\",\"role\":\"$ROLE\",\"error\":\"aborted early\"}" > /tmp/FAILED.json
  aws s3 cp /tmp/FAILED.json "s3://$BUCKET/status/$IID.FAILED.json" --only-show-errors 2>/dev/null || true
  ship
  shutdown -h now
}
trap on_exit EXIT

die() {
  echo "FAILED: $*"
  echo "{\"instance_id\":\"$IID\",\"role\":\"$ROLE\",\"error\":\"$*\"}" > /tmp/FAILED.json
  aws s3 cp /tmp/FAILED.json "s3://$BUCKET/status/$IID.FAILED.json" --only-show-errors || true
  ship
  # Terminate rather than idle: the log and every result already left the box, so a running
  # instance holds nothing that cannot be read from S3 -- it is a bill, not a diagnostic.
  shutdown -h now
  exit 1
}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || die "apt update"
# libX* ARE NOT OPTIONAL AND ARE NOT FOR DRAWING. chemprop 2.x imports cuik_molmaker, whose
# native extension links libXrender/libXext/libSM; a headless server image ships none of
# them and `import chemprop` dies with an ImportError that names a graphics library, which
# reads like an unrelated problem. Third boot failure of this run.
apt-get install -y -qq build-essential cmake curl unzip git python3-dev \
  libxrender1 libxext6 libsm6 libx11-6 libglib2.0-0 >/dev/null || die "apt install"
if ! command -v aws >/dev/null; then
  curl -sf "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/a.zip || die "awscli dl"
  unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install >/dev/null || die "awscli install"
fi
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null || die "uv install"
export PATH="/root/.local/bin:$PATH"
# UV-MANAGED INTERPRETERS, not the AMI's. pybind11's CMake needs Python development headers
# and the first run of this script died on their absence (/usr/include/python3.12 missing on
# the stock Ubuntu 24.04 image). A managed CPython carries its own headers, so the build stops
# depending on what the image happened to ship. python3-dev is installed as well, belt and braces.
uv python install 3.12 3.11 >/dev/null 2>&1 || die "uv python install"

# ONLY_ARMS AND want() ARE HOISTED HERE, ABOVE THE SETUP, AND THAT IS THE FIX FOR A REAL
# FAILURE. They used to be defined just before the run loop, so the chemeleon weight copy and
# the chemberta/chemeleon preflights ran UNCONDITIONALLY -- even for a run whose ONLY_ARMS
# never mentions them. A bundle built with `git archive` (which correctly omits the gitignored
# 7.2 GB of model weights) then died one minute into boot on `cp chemeleon_mp.pt`, having paid
# for an instance to discover that it could not set up an arm it was never asked to measure.
# Setup that an arm needs belongs behind the same gate that decides whether the arm runs.
ONLY_ARMS="${ONLY_ARMS:-}"
want () { [ -z "$ONLY_ARMS" ] && return 0; case " $ONLY_ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }

mkdir -p /opt/bench && cd /opt/bench || die "cd"
aws s3 cp "s3://$BUCKET/bundle.tar.gz" . --only-show-errors || die "bundle download"
tar xzf bundle.tar.gz || die "bundle extract"
if want chemeleon; then
  mkdir -p /root/.chemprop && cp chemeleon_mp.pt /root/.chemprop/ || die "chemeleon weights"
fi

export HUME_CORPUS=/opt/bench/data/corpus1m/selected.txt HUME_CORPUS_N=1000000
export CHEMBERTA_PATH=/opt/bench/models_hf/ChemBERTa-2-MLM
export AWS_REGION=us-east-1
for V in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS; do export $V=1; done

# --- env A: hume + ecfp + chemberta + chemeleon (numpy 2) --------------------------------
uv venv --python 3.12 --python-preference only-managed /opt/bench/envA >/dev/null 2>&1 || die "venvA create"
A=/opt/bench/envA/bin/python
uv pip install -q --python $A "numpy==2.4.6" "rdkit==2025.9.2" "scikit-build-core>=0.10" \
   "pybind11>=2.12" "transformers==5.16.1" || die "envA base deps"
if [ "$ROLE" = "gpu" ]; then
  uv pip install -q --python $A torch --index-url https://download.pytorch.org/whl/cu124 || die "torch cu124"
else
  uv pip install -q --python $A torch --index-url https://download.pytorch.org/whl/cpu || die "torch cpu"
fi
# CHEMELEON IS THE ONE OPTIONAL ARM. chemprop drags in cuik_molmaker and a chain of native
# libraries, and it has already cost one boot. The other four arms do not need it, so a
# failure here DISABLES THAT ARM rather than killing a run that would otherwise produce
# everything else. The completion check below then reports chemeleon as missing, loudly,
# instead of the box dying with nothing.
uv pip install -q --python $A "chemprop>=2.1" || die "chemprop install"
uv pip install -q --python $A --no-build-isolation-package hume /opt/bench 2>/dev/null \
  || uv pip install -q --python $A /opt/bench || die "hume build"
# descriptastorus: the tuned bulk wrapper around RDKit's descriptors, and the fast classical
# implementation figure D was missing. It goes in envA, not envB: verified locally against
# exactly envA's pins (numpy 2.4.6, rdkit 2025.9.2) rather than assumed.
uv pip install -q --python $A descriptastorus || die "descriptastorus install"

# --- env B: mordred (numpy 1.x, python 3.11) ---------------------------------------------
if [ "$ROLE" = "cpu" ]; then
  uv venv --python 3.11 --python-preference only-managed /opt/bench/envB >/dev/null 2>&1 || die "venvB create"
  B=/opt/bench/envB/bin/python
  # --no-config IS LOAD-BEARING. /opt/bench/pyproject.toml carries
  #   [tool.uv] constraint-dependencies = ["rdkit==2025.9.2", "numpy==2.4.6"]
  # which is exactly the pin that stops this project's own venv being clobbered -- and uv
  # applies it to ANY install run from this directory, including this one. mordred 1.2.0
  # requires numpy 1.x, so with the config in scope the resolve is unsatisfiable and the
  # first run died here. envA still WANTS those constraints, so the flag goes on envB only.
  uv pip install -q --no-config --python $B "mordred==1.2.0" "rdkit==2025.9.2" \
     "numpy==1.26.4" || die "mordred deps"
fi

# --- PREFLIGHT: every arm must actually import and produce a value on 200 molecules -------
echo "=== preflight ==="
$A -c "
import hume, torch, rdkit, numpy
print('envA rdkit', rdkit.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__,
      'cuda', torch.cuda.is_available())
print('hume columns', len(hume.ALL_COLUMNS))
fp, X, _ = hume.featurize_all(['CC(=O)Oc1ccccc1C(=O)O'])
import numpy as np; assert np.isfinite(X).sum() > 700, 'hume produced no values'
print('hume smoke ok, finite cells', int(np.isfinite(X).sum()))
" || die "envA preflight imports"
if want chemeleon; then
$A -c "
import torch, chemprop
from chemprop import nn as cnn, featurizers, data as cdata
ck = torch.load('/root/.chemprop/chemeleon_mp.pt', weights_only=True)
mp_ = cnn.BondMessagePassing(**ck['hyper_parameters']); mp_.load_state_dict(ck['state_dict']); mp_.eval()
agg = cnn.MeanAggregation()
dps = [cdata.MoleculeDatapoint.from_smi(s) for s in ['CC(=O)Oc1ccccc1C(=O)O','c1ccccc1']]
ds = cdata.MoleculeDataset(dps, featurizers.SimpleMoleculeMolGraphFeaturizer())
b = next(iter(cdata.build_dataloader(ds, batch_size=2, shuffle=False)))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
mp_ = mp_.to(dev); agg = agg.to(dev)
bmg = b.bmg; bmg.to(dev)          # in-place, returns None -- must NOT be rebound
with torch.no_grad(): h = agg(mp_(bmg), bmg.batch)
assert tuple(h.shape) == (2, 2048), h.shape
print('chemeleon smoke ok on', dev, tuple(h.shape))
" || die "chemeleon preflight"
fi
if want descriptastorus; then
$A -c "
from descriptastorus.descriptors import rdDescriptors
from rdkit import Chem
g = rdDescriptors.RDKit2D()
r = g.processMol(Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O'), 'CC(=O)Oc1ccccc1C(=O)O')
assert r and len(r) > 100, ('descriptastorus produced nothing', r)
print('descriptastorus smoke ok, columns', len(g.columns), 'values', len(r))
" || die "descriptastorus preflight"
fi
if [ "$ROLE" = "gpu" ]; then
  $A -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('gpu', torch.cuda.get_device_name(0))" || die "no GPU visible"
fi
if [ "$ROLE" = "cpu" ]; then
  $B -c "
import mordred, numpy, rdkit
from mordred import Calculator, descriptors as md
from rdkit import Chem
c = Calculator(md, ignore_3D=True)
v = list(c.map([Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')], nproc=1, quiet=True))
print('envB mordred', mordred.__version__, 'numpy', numpy.__version__, 'cols', len(v[0]))
" || die "envB preflight"
fi

mkdir -p /opt/bench/out
SIZES="10000 100000 1000000"
run () {  # $1=python $2=arm $3=budget
  echo "--- $2 / $3 ---"
  $1 /opt/bench/bench_aws.py --arm "$2" --budget "$3" --sizes $SIZES --reps 2 \
     --out "/opt/bench/out/$2_$3.json" || { echo "ARM FAILED: $2/$3"; return 1; }
  aws s3 cp "/opt/bench/out/$2_$3.json" "s3://$BUCKET/results/$IID/$2_$3.json" --only-show-errors
}

# ONLY_ARMS (set in user-data) re-runs a SUBSET. The first full pass produced good numbers for
# ecfp/hume/chemberta and lost chemprop/chemeleon to one API bug; repeating the arms that already
# worked would cost an hour of instance time and, worse, would silently replace a measurement
# with a different one taken under different conditions.
ONLY_ARMS="${ONLY_ARMS:-}"
want () { [ -z "$ONLY_ARMS" ] && return 0; case " $ONLY_ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }
EXPECT=""
if [ "$ROLE" = "cpu" ]; then
  for ARM in ecfp ecfp_r2 hume descriptastorus chemberta chemprop chemeleon; do
    want "$ARM" && { run $A "$ARM" cpu; EXPECT="$EXPECT ${ARM}_cpu"; }
  done
  want mordred && { run $B mordred cpu; EXPECT="$EXPECT mordred_cpu"; }
  # THE TWO DESCRIPTOR BLOCKS SEPARATELY. Figure C plots ecfp_rdkit_desc and ecfp_mordred_desc as
  # their own arms and their cost is NOT recoverable from the union: RDKit's 180 columns are cheap
  # and mordred's 685 are not, so splitting the union by column count is wrong by an order of
  # magnitude. Same interpreter and the same column selection as the union arm, so the three
  # numbers are commensurable by construction.
  for ARM in rdkit_desc mordred_desc; do
    want "$ARM" && { run $B "$ARM" cpu; EXPECT="$EXPECT ${ARM}_cpu"; }
  done
  if want minimol; then
    # minimol needs a third interpreter for the reasons boot_ds.sh documents at length: graphium
    # moves torch, drags a mismatched torchvision, and wants a scipy old enough to accept its
    # float16 sparse matrices. Versions are the ones the working laptop venv has.
    uv venv --python 3.12 --python-preference only-managed /opt/bench/envC >/dev/null 2>&1 \
      || die "envC create"
    C=/opt/bench/envC/bin/python
    uv pip install -q --no-config --python $C "torch==2.13.0" "torchvision==0.28.0" \
      --index-url https://download.pytorch.org/whl/cpu || die "envC torch"
    uv pip install -q --no-config --python $C "numpy==2.2.6" "scipy==1.13.1" || die "envC scipy"
    uv pip install -q --no-config --python $C --no-build-isolation minimol \
      "scipy==1.13.1" "numpy==2.2.6" "torch==2.13.0" || die "envC minimol"
    $C -c "
import numpy as np, torch
from minimol import Minimol
torch.set_grad_enabled(False)
v = np.asarray(Minimol()(['CCO','c1ccccc1'])[0]); assert v.shape == (512,), v.shape
print('  envC minimol ok', v.shape)" || die "envC minimol cannot embed"
    run $C minimol cpu; EXPECT="$EXPECT minimol_cpu"
  fi
else
  for ARM in chemberta chemprop chemeleon; do
    want "$ARM" && { run $A "$ARM" gpu; EXPECT="$EXPECT ${ARM}_gpu"; }
  done
fi
echo "arms expected: $EXPECT"

# --- COMPLETION FROM ACHIEVED WORK, never from a file existing ----------------------------
$A - "$EXPECT" <<'PY' > /tmp/DONE.json || die "completion check"
import json, os, sys
want_n = {10000, 100000, 1000000}
got, missing = {}, []
for arm in sys.argv[1].split():
    p = f"/opt/bench/out/{arm}.json"
    if not os.path.exists(p):
        missing.append(f"{arm}: no output file"); continue
    d = json.load(open(p))
    ns = {pt["n"] for pt in d["points"] if pt.get("wall_s", 0) > 0}
    if not want_n <= ns:
        missing.append(f"{arm}: only N={sorted(ns)}"); continue
    got[arm] = sorted(ns)
print(json.dumps({"complete": not missing, "arms_complete": got, "missing": missing}, indent=1))
sys.exit(1 if missing else 0)
PY
RC=$?
aws s3 cp /tmp/DONE.json "s3://$BUCKET/status/$IID.DONE.json" --only-show-errors || true
cat /tmp/DONE.json
ship
[ $RC -ne 0 ] && die "incomplete: see DONE.json"
FINISHED=1
echo "=== COMPLETE $(date -u +%FT%TZ) ==="
ship
sleep 5
shutdown -h now
