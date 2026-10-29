#!/usr/bin/env bash
# Build dist/screenbundle.tar.gz and upload it with its sha256 as S3 object metadata, so
# launch_screen0.sh can refuse to launch against a stale bundle.
#
# WHY THE DIGEST IS ON THE OBJECT, NOT IN IT. A box that unpacks a bundle from a previous
# session would regenerate the split from THAT session's split_spec.json, verify it against the
# digests in the same file, and pass -- then extract molecules for an experiment nobody is
# running. A spec verifying itself is not a check. The bundle's identity has to be pinned from
# outside, which is what the launcher compares before it spends anything. (It cannot live inside
# the tarball either: a digest cannot contain itself.)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUCKET=hume-bench-use1-075120018132
mkdir -p dist

# The score arrays are the row-order gate (see screen/extract_smiles.py) and must travel with
# the code: 3.9 MB for the whole library, against the 36.7 GB of SMILES they protect.
for f in data/lsd_ampc/scores_manifest.json data/lsd_ampc/score_summary.json \
         data/lsd_ampc/split_spec.json; do
  [ -s "$f" ] || { echo "FATAL: $f missing; the bundle would verify nothing." >&2; exit 1; }
done
N=$(ls data/lsd_ampc/scores/shard_*.npz 2>/dev/null | wc -l | tr -d ' ')
[ "$N" = 246 ] || { echo "FATAL: $N score shards present, expected 246." >&2; exit 1; }

# Bake the commit in: a box has no git checkout, so without this the one field that says WHICH
# CODE RAN is null in every record it writes. The dirty-file list travels too -- this screen is
# being built from a tree with uncommitted changes, and a record that hid that would be worse
# than useless.
python3 - "$(git rev-parse HEAD)" > screen/BUNDLE_PROVENANCE.json <<'PY'
import datetime, json, subprocess, sys
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
json.dump({"git_commit": sys.argv[1],
           "git_dirty": bool(dirty.strip()),
           "git_dirty_files": sorted(l.split(maxsplit=1)[-1] for l in dirty.splitlines()),
           "staged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
          sys.stdout, indent=1)
PY

# COPYFILE_DISABLE stops macOS tar embedding AppleDouble xattr headers, which make GNU tar on
# the box print one "Ignoring unknown extended header keyword" line per file -- 246 lines of
# noise through the middle of every boot log.
COPYFILE_DISABLE=1 tar czf dist/screenbundle.tar.gz \
  screen/split.py screen/provenance.py screen/extract_smiles.py \
  screen/arms.py screen/collect_trainval.py screen/train.py screen/score_test.py \
  screen/BUNDLE_PROVENANCE.json \
  data/lsd_ampc/scores_manifest.json data/lsd_ampc/score_summary.json \
  data/lsd_ampc/split_spec.json data/lsd_ampc/split_spec.sha256 \
  data/lsd_ampc/scores \
  -C bench bench_downstream.py
# bench_downstream.py RIDES ALONG, and it must. The dsbundle in S3 was staged 2026-10-27 and its
# copy predates the `fp_radius` parameter the screen's HUME arms pass explicitly; a box that
# extracted only the dsbundle died in preflight with "unexpected keyword argument 'fp_radius'".
# The boot script extracts the dsbundle FIRST and this bundle second, so this copy lands on top.
# Verified safe to overwrite: the last commit touching that file predates the dsbundle, so the
# repo's version is the dsbundle's plus that one patch -- not a different lineage.
SHA=$(shasum -a 256 dist/screenbundle.tar.gz | cut -d' ' -f1)
echo "dist/screenbundle.tar.gz  $(du -h dist/screenbundle.tar.gz | cut -f1)  sha256 $SHA"
echo "  commit $(git rev-parse --short HEAD)$(git status --porcelain | grep -q . && echo ' (tree dirty -- recorded in the bundle)')"

aws s3 cp dist/screenbundle.tar.gz "s3://$BUCKET/screenbundle.tar.gz" \
  --metadata "sha256=$SHA" --only-show-errors
GOT=$(aws s3api head-object --bucket "$BUCKET" --key screenbundle.tar.gz \
      --query 'Metadata.sha256' --output text)
[ "$GOT" = "$SHA" ] || { echo "FATAL: uploaded digest reads back as $GOT" >&2; exit 1; }
echo "uploaded to s3://$BUCKET/screenbundle.tar.gz, digest metadata verified"
