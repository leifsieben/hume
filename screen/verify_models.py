"""Is phase 1 actually complete? Answered from the models in S3, not from a marker.

WHY THIS IS A SCRIPT AND NOT A LINE IN A BOOT SCRIPT. The box-side completion check asserted
`m.n_estimators > 0` on a freshly loaded XGBRegressor. In xgboost 3.4.1 that attribute is None
after `load_model`, so `None > 0` raised a TypeError and chemberta_mtr -- five models trained,
all five uploaded -- got a FAILED marker. The work was fine; the check was not. A gate that runs
in exactly one place, on a box that then terminates, is a gate nobody can re-run.

So the authority lives here, off the box, and reads the only durable evidence there is: the
model files themselves. It can be re-run at any time, against any subset of arms, long after
every instance is gone.

`n_estimators` was the wrong question anyway -- it is the configured cap, so it could only ever
echo the request back. `num_boosted_rounds()` counts the trees actually in the file.

    python screen/verify_models.py                     # every arm
    python screen/verify_models.py --arms hume_r4      # one
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A                                                        # noqa: E402

S3 = "s3://hume-bench-use1-075120018132/screen/models"
N_REPLICATES = 5
MIN_TRAIN = 900_000


def fetch(arm: str, dest: Path) -> bool:
    """-> True if this arm's manifest and models came down. Absent arm is not an error."""
    r = subprocess.run(["aws", "s3", "cp", f"{S3}/{arm}_train.json", str(dest), "--only-show-errors"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    subprocess.run(["aws", "s3", "sync", f"{S3}/", str(dest), "--exclude", "*",
                    "--include", f"{arm}_rep*.ubj", "--only-show-errors"], check=True)
    return True


def check(arm: str, d: Path) -> tuple[bool, list[str]]:
    import xgboost as xgb
    notes: list[str] = []
    man = json.loads((d / f"{arm}_train.json").read_text())
    reps = man["replicates"]
    if len(reps) != N_REPLICATES:
        return False, [f"{len(reps)} replicates recorded, expected {N_REPLICATES}"]
    n_val = {r["n_val"] for r in reps}
    if len(n_val) != 1:
        notes.append(f"validation sizes differ across replicates: {sorted(n_val)} -- the five "
                     f"early-stopping decisions were not made against equally sized sets")
    for r in reps:
        f = d / r["model"]
        if not f.exists():
            return False, [f"replicate {r['replicate']}: {r['model']} is not in {S3}"]
        m = xgb.XGBRegressor()
        m.load_model(str(f))
        nbr = m.get_booster().num_boosted_rounds()
        if nbr <= 0:
            return False, [f"replicate {r['replicate']}: model loads but holds no trees"]
        if nbr < r["best_iteration"] + 1:
            return False, [f"replicate {r['replicate']}: {nbr} trees in the file but "
                           f"best_iteration {r['best_iteration']} -- the file does not contain "
                           f"the iteration it claims"]
        if r["n_train"] <= MIN_TRAIN:
            return False, [f"replicate {r['replicate']} trained on {r['n_train']:,} molecules, "
                           f"under the {MIN_TRAIN:,} floor"]
        if not r.get("early_stopped", False):
            notes.append(f"rep{r['replicate']} never early-stopped: {nbr} trees is the whole "
                         f"budget, so the cap decided where it stopped, not the validation curve")
    return True, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=list(A.ALL_ARMS))
    a = ap.parse_args()
    done, missing, bad = [], [], []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for arm in a.arms:
            if not fetch(arm, d):
                missing.append(arm)
                print(f"  {arm:18s} NOT TRAINED YET (no {arm}_train.json in S3)")
                continue
            ok, notes = check(arm, d)
            man = json.loads((d / f"{arm}_train.json").read_text())
            rmse = [r["val_rmse"] for r in man["replicates"]]
            if ok:
                done.append(arm)
                print(f"  {arm:18s} COMPLETE  5 models  val rmse "
                      f"{min(rmse):.4f}-{max(rmse):.4f}  commit "
                      f"{str(man['provenance'].get('git_commit'))[:7]}")
            else:
                bad.append(arm)
                print(f"  {arm:18s} INCOMPLETE")
            for n in notes:
                print(f"      note: {n}")
    print(f"\n{len(done)}/{len(a.arms)} arms complete"
          + (f", {len(missing)} not started: {' '.join(missing)}" if missing else "")
          + (f", {len(bad)} BROKEN: {' '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
