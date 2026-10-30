"""What of phase 2 is neither scored nor currently being scored?

    .venv/bin/python scripts/screen2_gaps.py [--relaunch]

THE QUESTION A FLEET WATCHER ACTUALLY HAS. `watch_screen.sh` answers "what is running", which is
not the same thing: a spot reclaim leaves boxes gone and shards unscored, and a box that is
running tells you nothing about whether its range will finish. This asks the only question that
decides whether to spend money -- which shards have no prediction file in S3 AND no live box
assigned to them -- by taking the difference of two sets:

    scored    = {shard : s3://.../screen/preds/{arm}_shard_{i}.npz exists}
    assigned  = {shard : some pending/running box carries Arm={arm}, Shards=lo-hi covering i}
    gap       = all 246 - scored - assigned

A gap is work that will never happen unless something launches it. Everything else is either
done or in flight, and launching it again would pay twice for one result.

WHY THE SCORED SET IS READ FROM S3 AND NOT FROM A MANIFEST. The per-range manifest
({arm}_preds_{lo}_{hi}.json) is written at the END of a box's range, so a box reclaimed at 90%
has no manifest and nine tenths of its shards in S3. Counting manifests would call those shards
unscored and re-run them. The prediction files are the truth; they are uploaded per shard,
immediately, which is exactly what made the 2026-10-30 mass reclaim cost nothing.
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys

BUCKET = "hume-bench-use1-075120018132"
REGION = "us-east-1"
N_SHARDS = 246
ARMS = ("hume_r4", "hume_r2", "ecfp_rdkit_desc", "chemberta_mtr", "chemeleon", "minimol")


def scored() -> dict[str, set[int]]:
    out = subprocess.run(["aws", "s3", "ls", f"s3://{BUCKET}/screen/preds/"],
                         capture_output=True, text=True, check=True).stdout
    have: dict[str, set[int]] = {a: set() for a in ARMS}
    for line in out.splitlines():
        m = re.search(r"([a-z0-9_]+)_shard_(\d{5})\.npz$", line.split()[-1])
        if m and m.group(1) in have:
            have[m.group(1)].add(int(m.group(2)))
    return have


def assigned() -> dict[str, set[int]]:
    q = ("Reservations[].Instances[].[Tags[?Key=='Arm']|[0].Value,"
         "Tags[?Key=='Shards']|[0].Value]")
    out = subprocess.run(
        ["aws", "ec2", "describe-instances", "--region", REGION,
         "--filters", "Name=tag:Phase,Values=screen2",
         "Name=instance-state-name,Values=pending,running",
         "--query", q, "--output", "text"],
        capture_output=True, text=True, check=True).stdout
    live: dict[str, set[int]] = {a: set() for a in ARMS}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in live:
            continue
        nums = re.findall(r"\d+", parts[1])
        if len(nums) >= 2:
            live[parts[0]] |= set(range(int(nums[0]), int(nums[1])))
    return live


def runs(xs: set[int]) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) ranges covering `xs`."""
    s = sorted(xs)
    out, start = [], None
    for a, b in zip(s, s[1:] + [None]):
        if start is None:
            start = a
        if b != a + 1:
            out.append((start, a + 1)); start = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relaunch", action="store_true",
                    help="print the launch commands that would close every gap")
    a = ap.parse_args()

    have, live = scored(), assigned()
    gaps, done = {}, []
    for arm in ARMS:
        g = set(range(N_SHARDS)) - have[arm] - live[arm]
        n_live = len(live[arm] - have[arm])
        if not g and len(have[arm]) == N_SHARDS:
            done.append(arm)
            print(f"  {arm:<18} COMPLETE  {N_SHARDS}/{N_SHARDS}")
            continue
        gaps[arm] = g
        state = "no gap, in flight" if not g else f"GAP {len(g)} shards"
        print(f"  {arm:<18} {len(have[arm]):>3}/{N_SHARDS} scored, "
              f"{n_live:>3} in flight   {state}")
        if g:
            print(f"  {'':<18} {', '.join(f'[{x},{y})' for x, y in runs(g)[:8])}")
    total = sum(len(g) for g in gaps.values())
    print(f"\n{len(done)}/{len(ARMS)} arms complete; {total} shards need launching")
    if a.relaunch and total:
        print("\n# commands to close every gap:")
        for arm, g in gaps.items():
            for lo, hi in runs(g):
                # One box per ~10 shards keeps a reclaim cheap; the launcher caps at 40.
                nb = max(1, min(40, round((hi - lo) / 10) or 1))
                print(f'ARMS="{arm}" LO_SHARD={lo} HI_SHARD={hi} BOXES={nb} '
                      f'./scripts/launch_screen2.sh')
    return 0 if not total else 2


if __name__ == "__main__":
    sys.exit(main())
