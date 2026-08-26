#!/usr/bin/env bash
# Snapshot figure inputs to a dated archive.
#
# The expensive artefacts are the embeddings -- GROVER and SMI-TED especially, which need
# environments we do not otherwise keep around. Losing them means re-standing-up a legacy
# torch/DGL stack to redraw one panel. Everything else is cheap to regenerate; these are not.
#
#   scripts/backup_figures.sh [dest]
#
# Writes only inside this repo unless an explicit destination is given.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/results/figures"
DEST="${1:-$ROOT/backups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$DEST/figures_$STAMP"

[ -d "$SRC" ] || { echo "nothing to back up: $SRC does not exist"; exit 0; }
mkdir -p "$OUT"

# Refresh the manifest before archiving, so the snapshot proves what it contains.
python - "$SRC" <<'PY'
import hashlib, json, subprocess, sys, platform
from pathlib import Path
src = Path(sys.argv[1])
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
files = {}
for p in sorted(src.rglob("*")):
    if p.is_file() and p.name not in ("MANIFEST.json", "ENVIRONMENT.json"):
        files[str(p.relative_to(src))] = {"sha256": sha(p), "bytes": p.stat().st_size}
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
except Exception:
    commit = None
json.dump({"git_commit": commit, "files": files},
          (src / "MANIFEST.json").open("w"), indent=2)
env = {"python": sys.version.split()[0], "platform": platform.platform(),
       "node": platform.node()}
for mod in ("rdkit", "numpy", "xgboost", "torch", "mordred"):
    try:
        m = __import__(mod)
        env[mod] = getattr(m, "__version__", "unknown")
    except Exception:
        env[mod] = None
json.dump(env, (src / "ENVIRONMENT.json").open("w"), indent=2)
print(f"manifest: {len(files)} files")
PY

cp -R "$SRC/." "$OUT/"
echo "backed up to $OUT"
du -sh "$OUT"
