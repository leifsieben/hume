"""Freeze the train / validation / test split for the LSD AmpC screen, from scores alone.

FOLLOWS Hall et al., J. Chem. Inf. Model. 2025, 65, 4458-4467 (doi 10.1021/acs.jcim.5c00394),
"Impact of Training Set Size and Sampling on Chemprop Performance":

  * test  = 100,000,000 molecules sampled at random from the data set, disjoint from training.
            DRAWN ONCE, not per replicate. The paper's five replicates vary the training draw;
            re-drawing the test set as well would multiply the only expensive step -- featurizing
            100M molecules -- by five, to measure a variance the replicates are not about. One
            test set is featurized once per arm and scored by all five models, which costs
            essentially nothing extra, so the replicate spread reported is training variance,
            which is what Hall et al. report min/max over.
  * train = 1,000,000 molecules, STRATIFIED: "80% of the training set was randomly sampled from
            the top ranking 1% of molecules and the remaining 20% was sampled from the rest".
  * val   = about 20% of the training size, Bemis-Murcko scaffold-disjoint from train. The
            scaffold split needs SMILES, so it is done on the featurisation pass from the
            `val_pool` indices frozen here, with the seed recorded.
  * 5 replicates, each a different seed.

NO INDEX FILE. 100M int32 indices is 400 MB per replicate and would have to be shipped to every
box. Instead this records the SEED and the ALGORITHM, and the sha256 of the resulting sorted
index array; each box regenerates the identical indices and asserts the digest before it
featurizes anything. The split is reproducible from 2 kB of JSON, and a box that would have
computed a different split fails instead of quietly scoring a different experiment.

The global index space is 0..N-1 over shards in ascending order, so a global index maps to
(shard, row) through the cumulative shard sizes recorded in the scores manifest.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

# ONE IMPLEMENTATION, IMPORTED -- NOT A SECOND COPY. This module used to carry its own
# unique_sample/digest and its own replicate draw, identical to screen/split.py's by inspection
# only. Two copies of a sampler that must agree exactly is the drift this arrangement exists to
# prevent: every box verifies its regenerated indices against the digests THIS file wrote, so if
# the two ever diverged the boxes would fail against a spec produced by the other copy and the
# cause would be invisible. The freezer and the boxes now run the same lines.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split import (DATA, N_TEST, N_TRAIN, TOP_FRAC, FRAC_FROM_TOP, VAL_MULT, VAL_POOL_MULT,
                   N_REPLICATES, SEED0, digest, draw_replicate, draw_test, geometry,
                   top_indices)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    a = ap.parse_args()

    sizes, _, N = geometry(DATA)
    shards = json.loads((DATA / "scores_manifest.json").read_text())["shards"]

    summ = json.loads((DATA / "score_summary.json").read_text())
    SCALE = int(summ["scale"])
    q_top = int(round(summ["thresholds"][str(TOP_FRAC)] * SCALE))
    print(f"{N:,} molecules over {len(shards)} shards")
    print(f"top {TOP_FRAC:.0%} cut: score <= {q_top/SCALE:.2f}\n")

    top_idx = top_indices(q_top, DATA)
    print(f"top-1% population: {top_idx.size:,} molecules "
          f"({top_idx.size/N:.4%} of the library)")

    n_top = int(round(N_TRAIN * FRAC_FROM_TOP))
    n_rest = N_TRAIN - n_top

    # The test set is drawn ONCE -- see the module docstring.
    test = draw_test(N, N_TEST, SEED0)
    test_sha = digest(test)
    print(f"test set (shared by every replicate): {test.size:,}  sha256 {test_sha[:16]}...")

    reps = []
    for r in range(a.replicates):
        seed = SEED0 + 1000 + r
        train, val_pool = draw_replicate(N, top_idx, test, seed)
        reps.append({
            "replicate": r, "seed": seed,
            "train": {"n": int(train.size), "n_from_top1pct": n_top, "n_from_rest": n_rest,
                      "sha256": digest(train)},
            "val_pool": {"n": int(val_pool.size), "sha256": digest(val_pool)},
        })
        print(f"  replicate {r} seed {seed}: train {train.size:,} "
              f"({n_top:,} top / {n_rest:,} rest)  val_pool {val_pool.size:,}")

    spec = {
        "dataset": "IrwinLab/LSD_AmpC_Liu_2025", "config": "docking_results",
        "n_molecules": N, "n_shards": len(shards),
        "shard_sizes": sizes.tolist(),
        "methodology": "Hall et al. JCIM 2025 65:4458-4467, doi 10.1021/acs.jcim.5c00394",
        "top_frac": TOP_FRAC, "top_cut_score": q_top / SCALE,
        "top_population": int(top_idx.size),
        "n_test": N_TEST, "n_train": N_TRAIN,
        "test": {"n": N_TEST, "seed": SEED0, "sha256": test_sha,
                 "shared_across_replicates": True},
        "train_frac_from_top": FRAC_FROM_TOP,
        "val_mult": VAL_MULT, "val_pool_mult": VAL_POOL_MULT,
        "sampler": "numpy default_rng; unique_sample = draw/sort/dedupe/top-up; see module docstring",
        "replicates": reps,
    }
    out = DATA / "split_spec.json"
    out.write_text(json.dumps(spec, indent=1))
    spec_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (DATA / "split_spec.sha256").write_text(spec_sha + "  split_spec.json\n")
    print(f"\n-> {out}\n   sha256 {spec_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
