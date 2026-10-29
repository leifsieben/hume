"""Phase 1a: assemble the training and validation sets, ONCE, for all six arms.

Reads the 246 per-shard files phase 0 wrote, pulls out the 5 x (1M train + 600k validation pool)
molecules, computes Bemis-Murcko scaffolds, and freezes the scaffold-disjoint validation
selection. Output is one `trainval.npz` that every arm box downloads.

WHY THE VALIDATION SELECTION IS FROZEN HERE AND NOT DONE PER ARM. Every arm early-stops on
validation RMSE, so the validation set is part of the training protocol, not an afterthought. If
each arm picked its own scaffold-disjoint subset -- even by the same rule, even deterministically
-- any later edit to the selection code would change some arms and not others, and the arms would
differ for a reason that has nothing to do with their representation. One selection, computed
once, recorded with a digest, used by all.

FOLLOWS Hall et al. section "Machine Learning": "we split molecules based on their Bemis-Murcko
scaffolds to assign them to the training and validation sets, ensuring no scaffold appeared in
both sets", with "validation sets about 20% as large".

ONE DELIBERATE DEPARTURE. The paper scaffold-splits a combined pool, which is why its training
sets are "approximately" 1,000,000. We draw train at exactly 1M (frozen in split_spec.json) and
then take the validation set from a separate 600k pool, keeping only scaffolds absent from train.
Same guarantee -- no scaffold in both -- but every arm trains on exactly the same 1,000,000
molecules, which is what makes the arms comparable to each other rather than merely to the paper.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance
import split as S

N_VAL_TARGET = 200_000   # "validation sets about 20% as large" as the 1M training set
N_VAL_FLOOR = 120_000    # below 12% of train, "about 20%" stops being an honest description
SCAF_BATCH = 50_000


def say(*a) -> None:
    print(*a, flush=True)


def _scaf_chunk(smis: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return _scaffold_hashes_serial(smis)


def scaffold_hashes(smis: list[str], workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Parallel over a process pool, order preserved. Murcko costs ~490 us/mol, so the 8M
    molecules across five replicates are over an hour single-threaded -- for a step whose result
    every arm waits on. Chunked map keeps the output in input order, which it must be: the
    hashes are indexed by position alongside the SMILES and the scores."""
    if workers in (0, 1) or len(smis) < 20_000:
        return _scaffold_hashes_serial(smis)
    import multiprocessing as mp
    n = max(1, len(smis) // (workers * 4))
    chunks = [smis[i:i + n] for i in range(0, len(smis), n)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_scaf_chunk, chunks)
    return (np.concatenate([h for h, _ in parts]),
            np.concatenate([o for _, o in parts]))


def _scaffold_hashes_serial(smis: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """-> (uint64 scaffold hash per molecule, bool parseable mask).

    Hashes rather than scaffold strings: 8 bytes against ~30, and the only operation needed is
    set membership between train and the validation pool. A molecule RDKit cannot parse gets
    hash 0 and parseable=False; a molecule with no ring system yields the empty scaffold, which
    hashes like any other string -- all such molecules therefore share one scaffold and land on
    one side of the split, which is the correct Bemis-Murcko behavior and not a bug to work round.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    h = np.zeros(len(smis), dtype=np.uint64)
    ok = np.zeros(len(smis), dtype=bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            continue
        ok[i] = True
        h[i] = int.from_bytes(hashlib.blake2b(sc.encode(), digest_size=8).digest(), "little")
    return h, ok


def read_shard(path: Path, keys: list[str]) -> dict:
    """-> {key: (list[str] smiles, int16 scores)} for the requested membership keys."""
    with np.load(path) as z:
        blob = z["smi_blob"].tobytes()
        off = z["smi_off"]
        q = z["q"]
        out = {}
        for k in keys:
            pos = z[f"{k}_pos"]
            smis = [blob[off[p]:off[p + 1]].decode() for p in pos]
            out[k] = (smis, q[pos])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles-dir", type=Path, default=Path("data/lsd_ampc/smiles"))
    ap.add_argument("--out", type=Path, default=Path("data/lsd_ampc/trainval.npz"))
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--shards", type=int, default=246)
    ap.add_argument("--workers", type=int, default=0,
                    help="processes for the Murcko scaffold pass; 0 = serial")
    a = ap.parse_args()

    prov = provenance.collect()
    say(json.dumps(prov, indent=1))
    spec = S.load_spec()
    keys = [f"train_{r}" for r in range(a.replicates)] + \
           [f"val_pool_{r}" for r in range(a.replicates)]

    say(f"\nreading {a.shards} shards from {a.smiles_dir}")
    acc = {k: ([], []) for k in keys}
    t0 = time.perf_counter()
    for i in range(a.shards):
        p = a.smiles_dir / f"shard_{i:05d}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing. The train/validation sets must be assembled over the WHOLE "
                f"library -- a missing shard silently removes its molecules from training and "
                f"nothing downstream would notice.")
        got = read_shard(p, keys)
        for k, (smis, q) in got.items():
            acc[k][0].extend(smis)
            acc[k][1].append(q)
        if (i + 1) % 40 == 0:
            say(f"  {i+1}/{a.shards} shards  {(time.perf_counter()-t0)/60:.1f} min")

    payload, report, scratch = {}, [], []
    N_TRAIN_NOMINAL = int(spec["n_train"])
    for r in range(a.replicates):
        tr_smis = acc[f"train_{r}"][0]
        tr_y = np.concatenate(acc[f"train_{r}"][1]).astype(np.float32) / 100.0
        vp_smis = acc[f"val_pool_{r}"][0]
        vp_y = np.concatenate(acc[f"val_pool_{r}"][1]).astype(np.float32) / 100.0

        exp_tr = int(spec["replicates"][r]["train"]["n"])
        exp_vp = int(spec["replicates"][r]["val_pool"]["n"])
        if len(tr_smis) != exp_tr or len(vp_smis) != exp_vp:
            raise RuntimeError(
                f"replicate {r}: collected {len(tr_smis):,} train and {len(vp_smis):,} val_pool "
                f"molecules but the frozen split says {exp_tr:,} and {exp_vp:,}. A shard's "
                f"membership arrays disagree with the spec; do not train on this.")

        say(f"\nreplicate {r}: scaffolds for {len(tr_smis):,} train + {len(vp_smis):,} pool")
        t1 = time.perf_counter()
        tr_h, tr_ok = scaffold_hashes(tr_smis, a.workers)
        vp_h, vp_ok = scaffold_hashes(vp_smis, a.workers)
        say(f"  {(time.perf_counter()-t1)/60:.1f} min; unparseable: "
            f"{int((~tr_ok).sum())} train, {int((~vp_ok).sum())} pool")

        # Validation = pool molecules that parse AND whose scaffold is absent from train.
        train_scafs = np.unique(tr_h[tr_ok])
        free = vp_ok & ~S.isin_sorted(vp_h, train_scafs)
        cand = np.flatnonzero(free)
        say(f"  scaffold-free pool: {cand.size:,} of {len(vp_smis):,} ({cand.size/len(vp_smis):.1%})")
        scratch.append(dict(r=r, tr_smis=tr_smis, tr_y=tr_y, tr_h=tr_h, tr_ok=tr_ok,
                            vp_smis=vp_smis, vp_y=vp_y, vp_h=vp_h, cand=cand,
                            train_scafs=train_scafs))

    # --- VALIDATION SIZE IS CHOSEN ONCE, ACROSS ALL REPLICATES, AND THAT IS THE POINT ---------
    # A 1,000,000-molecule training set drawn mostly from the top 1% of a make-on-demand library
    # covers so many Murcko scaffolds that only about a third of a 600,000 pool is scaffold-free
    # -- 186,197 on the first run, against a 200,000 target. Hall et al. say "validation sets
    # about 20% as large", not exactly 20%, and their own construction scaffold-splits a combined
    # pool, so their validation sizes vary too.
    #
    # Taking the MINIMUM across replicates rather than each replicate's own maximum is what keeps
    # the five comparable: early stopping is part of the training protocol, so a replicate that
    # stopped against 186k validation molecules and one that stopped against 193k would differ for
    # a reason that has nothing to do with the arm. One size, every replicate, recorded below.
    avail = min(d["cand"].size for d in scratch)
    n_val = min(N_VAL_TARGET, avail)
    if n_val < N_VAL_FLOOR:
        raise RuntimeError(
            f"the smallest scaffold-free validation pool across replicates is {avail:,}, under the "
            f"{N_VAL_FLOOR:,} floor ({N_VAL_FLOOR/1e6:.0%} of the training set). Calling that "
            f"'about 20% as large' would be false. Raise VAL_POOL_MULT in screen/split.py and "
            f"re-freeze the split (which also means re-running phase 0 for the new val_pool rows).")
    say(f"\nvalidation size {n_val:,} for every replicate "
        f"({n_val/N_TRAIN_NOMINAL:.1%} of training; smallest free pool was {avail:,}, "
        f"target was {N_VAL_TARGET:,})")

    for d in scratch:
        r = d["r"]
        tr_smis, tr_y, tr_h, tr_ok = d["tr_smis"], d["tr_y"], d["tr_h"], d["tr_ok"]
        vp_smis, vp_y, vp_h = d["vp_smis"], d["vp_y"], d["vp_h"]
        train_scafs = d["train_scafs"]
        take = d["cand"][:n_val]            # index order: deterministic, consumes no RNG

        keep_tr = np.flatnonzero(tr_ok)
        enc_tr = [tr_smis[i].encode() for i in keep_tr]
        enc_va = [vp_smis[i].encode() for i in take]
        for nm, enc, y in (("train", enc_tr, tr_y[keep_tr]), ("val", enc_va, vp_y[take])):
            payload[f"{nm}_{r}_blob"] = np.frombuffer(b"".join(enc), dtype=np.uint8)
            payload[f"{nm}_{r}_off"] = np.concatenate(
                [[0], np.cumsum([len(e) for e in enc])]).astype(np.int64)
            payload[f"{nm}_{r}_y"] = y.astype(np.float32)
        payload[f"train_{r}_scaf"] = tr_h[keep_tr]
        payload[f"val_{r}_scaf"] = vp_h[take]

        shared = np.intersect1d(np.unique(tr_h[keep_tr]), np.unique(vp_h[take]))
        if shared.size:
            raise RuntimeError(f"replicate {r}: {shared.size} scaffolds in BOTH train and "
                               f"validation. The disjointness filter did not hold.")
        report.append(dict(replicate=r, n_train=int(keep_tr.size), n_val=int(take.size),
                           n_train_unparseable=int((~tr_ok).sum()),
                           pool_scaffold_free=int(d["cand"].size),
                           train_scaffolds=int(train_scafs.size),
                           val_scaffolds=int(np.unique(vp_h[take]).size),
                           y_train_mean=float(tr_y[keep_tr].mean()),
                           y_val_mean=float(vp_y[take].mean())))
        say(f"  train {keep_tr.size:,} ({train_scafs.size:,} scaffolds)  "
            f"val {take.size:,} ({np.unique(vp_h[take]).size:,} scaffolds)  "
            f"scaffold-disjoint verified")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.stem + ".part.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(a.out)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    meta = a.out.with_suffix(".json")
    meta.write_text(json.dumps(
        {"n_val_target": N_VAL_TARGET, "n_val_floor": N_VAL_FLOOR, "n_val_used": n_val,
         "sha256": sha, "provenance": prov,
         "methodology": spec["methodology"], "replicates": report}, indent=1))
    say(f"\n-> {a.out}  {a.out.stat().st_size/1e6:.0f} MB  sha256 {sha[:16]}...")
    say(f"-> {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
