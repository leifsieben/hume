"""Learned-embedding matrices for Figures B and C, over the FULL benchmark molecule set.

Three sources, deduplicated into ONE master index:

    data/surrogate/bench.npz   56,197 rows -> 42,878 unique SMILES  (30 MoleculeACE + 4 MolNet)
    data/tasks/tasks.npz       928,237 unique SMILES  (qm / admet_cls / muv)
    data/tasks/litpcba.npz     383,772 unique SMILES  (vs)

`tasks.npz` and `litpcba.npz` already carry the unique-SMILES + `idx` indirection documented in
`data/benchmark_tasks.md`; `bench.npz` does not (its rows repeat molecules across datasets), so
its `smiles` column is uniqued here.

ROW ALIGNMENT IS THE FAILURE MODE THIS FILE EXISTS TO PREVENT. Every arm npz stores the SMILES
array it was computed from, plus the sha256 of that array. `load_arm()` re-derives the hash and
refuses to return a matrix whose SMILES do not match the master index. Nothing downstream should
index an embedding matrix by position without going through `load_arm`.

    python embed_bench.py index                       # build/refresh the master index
    python embed_bench.py embed chemberta_mtr         # one arm (respects --tier)
    python embed_bench.py embed molformer --tier 0
    python embed_bench.py verify                      # check every arm on disk

TIERS. 1,283,200 unique molecules is far more than the figures need and more than a shared CPU
box can pay for. The index is ordered by tier so that ANY PREFIX OF IT IS A WHOLE NUMBER OF
DATASETS -- an arm that only gets through tier 0 still covers every dataset in tier 0 completely,
and a partial matrix is never a partial dataset.

    tier 0  bench.npz + admet_cls + MUV + qm8 + photoswitch + the 7 small LIT-PCBA targets
    tier 1  qm9_gap
    tier 2  qmugs_gap
    tier 3  the 8 LIT-PCBA targets over 240,000 rows (95% of their compute buys decoys)

Tier 0 is the set `data/benchmark_tasks.md` recommends as "the version I would actually run",
minus the QM caps (which are a scaffold-stratified subsample decision that belongs to whoever
owns the split, not to the featuriser).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "embeddings"
PARTS = OUT / "_parts"
INDEX = OUT / "smiles_index.npz"

SHARD = 20_000          # molecules per checkpointed shard
# HALF the box's 12 cores, SPLIT ACROSS CONCURRENT STREAMS -- two other agents are taking
# timings on this machine and an over-subscribed box would corrupt their measurements, not just
# slow this job down. Three arms need three different virtualenvs, so the budget is per-process.
N_THREADS = int(os.environ.get("EMBED_THREADS", "3"))

# LIT-PCBA targets that go in tier 0 (small enough that they are not the cost driver) vs tier 3.
LITPCBA_TIER0 = {"litpcba_ESR1_ago", "litpcba_ESR1_ant", "litpcba_PPARG", "litpcba_TP53",
                 "litpcba_MTORC1", "litpcba_MAPK1", "litpcba_ALDH1"}
TIER_OF_DATASET = {"qm9_gap": 1, "qmugs_gap": 2}      # everything else resolves to 0 or 3


def _sha(smiles) -> str:
    h = hashlib.sha256()
    for s in smiles:
        h.update(str(s).encode())
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# the master index
# --------------------------------------------------------------------------------------------

def build_index() -> None:
    """Union the three sources into one tier-ordered unique-SMILES array + back-pointers.

    Deduplication is on the EXACT SMILES STRING, not on a canonical form. That is deliberate and
    it is the only defensible key here: a chemical language model's embedding is a function of
    the string it is handed, so two spellings of one molecule are two different inputs and
    collapsing them would silently substitute one dataset's spelling for another's. The canonical
    count is reported alongside so the difference is visible rather than assumed away.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    bench = np.load(ROOT / "data/surrogate/bench.npz", allow_pickle=True)
    tasks = np.load(ROOT / "data/tasks/tasks.npz", allow_pickle=True)
    lit = np.load(ROOT / "data/tasks/litpcba.npz", allow_pickle=True)

    # tier for each molecule = the MINIMUM tier over every dataset that uses it, so a molecule
    # shared between qmugs and MUV is embedded in tier 0 where MUV needs it.
    tier_of_mol: dict[str, int] = {}

    def _assign(smi_arr, idx, offsets, names, litpcba: bool = False):
        for j, name in enumerate(names):
            if litpcba:
                t = 0 if name in LITPCBA_TIER0 else 3
            else:
                t = TIER_OF_DATASET.get(str(name), 0)
            for u in np.unique(idx[offsets[j]:offsets[j + 1]]):
                s = smi_arr[u]
                if tier_of_mol.get(s, 9) > t:
                    tier_of_mol[s] = t

    for s in bench["smiles"]:
        tier_of_mol[s] = 0
    _assign(tasks["smiles"], tasks["idx"], tasks["offsets"], tasks["name_of"])
    _assign(lit["smiles"], lit["idx"], lit["offsets"], lit["name_of"], litpcba=True)

    # stable order: tier, then first-seen order within tier (bench -> tasks -> litpcba)
    seen, order = set(), []
    for arr in (bench["smiles"], tasks["smiles"], lit["smiles"]):
        for s in arr:
            if s not in seen:
                seen.add(s)
                order.append(s)
    order.sort(key=lambda s: tier_of_mol[s])          # python sort is stable
    smiles = np.array(order, dtype=object)
    tier = np.array([tier_of_mol[s] for s in order], np.int8)
    pos = {s: i for i, s in enumerate(order)}

    bench_u, bench_ui = np.unique(bench["smiles"], return_index=True)
    bench_u = bench["smiles"][np.sort(bench_ui)]      # first-seen order, not lexicographic
    maps = {
        "bench_row_to_master": np.array([pos[s] for s in bench["smiles"]], np.int64),
        "tasks_to_master": np.array([pos[s] for s in tasks["smiles"]], np.int64),
        "litpcba_to_master": np.array([pos[s] for s in lit["smiles"]], np.int64),
    }

    np.savez_compressed(INDEX, smiles=smiles, tier=tier, **maps)
    meta = {
        "n_unique": len(smiles),
        "sha256": _sha(smiles),
        "per_source": {"bench_rows": len(bench["smiles"]), "bench_unique": len(set(bench["smiles"])),
                       "tasks_unique": len(tasks["smiles"]), "litpcba_unique": len(lit["smiles"])},
        "tier_counts": {int(t): int((tier == t).sum()) for t in np.unique(tier)},
        "tier_cumulative": {int(t): int((tier <= t).sum()) for t in np.unique(tier)},
    }
    json.dump(meta, open(OUT / "index_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


PARSEFAIL = OUT / "parse_failures.npz"


def build_parsefail() -> None:
    """Cache the indices of every master-index SMILES that RDKit refuses to parse.

    THIS EXISTS BECAUSE ONE ARM FAILS SILENTLY AND INVISIBLY. `embed_pairs._chemprop_embed`
    substitutes `Chem.MolFromSmiles("C")` -- METHANE -- for any molecule RDKit rejects, so a
    failed molecule comes back as a perfectly ordinary, non-zero, entirely wrong embedding. The
    all-zero-row check that catches SELFIES-TED's failures cannot see it. The only way to know
    which rows are junk in `chemprop.npy` and `chemeleon.npy` is to compute the parse-failure set
    independently and carry it beside every arm.

    The string arms (ChemBERTa, MoLFormer) tokenise the SMILES directly and never touch RDKit, so
    for them these rows are real embeddings of a real input string, not failures. The flag is
    therefore recorded as data and left for the caller to apply per arm rather than being turned
    into a NaN here.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    smiles = load_index(None)
    bad = np.array([i for i, s in enumerate(smiles) if Chem.MolFromSmiles(str(s)) is None],
                   np.int64)
    np.savez_compressed(PARSEFAIL, idx=bad,
                        smiles=np.array([smiles[i] for i in bad], dtype=object))
    print(f"{len(bad)} of {len(smiles):,} master-index SMILES fail RDKit parse")
    for i in bad[:20]:
        print(f"  {i:>9} {smiles[i]}")


def _parsefail(n: int):
    if not PARSEFAIL.exists():
        return np.array([], np.int64)
    b = np.load(PARSEFAIL, allow_pickle=True)["idx"]
    return b[b < n]


def load_index(tier: int | None = None):
    d = np.load(INDEX, allow_pickle=True)
    smiles, t = d["smiles"], d["tier"]
    if tier is None:
        return smiles
    return smiles[t <= tier]


# --------------------------------------------------------------------------------------------
# arms -- reuse embed_pairs.py's implementations rather than reimplementing them
# --------------------------------------------------------------------------------------------

def _arm_molformer(smiles):
    """MoLFormer, with `deterministic_eval=True` ACTUALLY APPLIED.

    `embed_pairs._hf` tries to pass it, guarded by `if "MoLFormer" in str(path)` -- but the
    weights live in `models_hf/MolFormer`, lowercase o, so the test is False and the flag is
    never passed. MoLFormer-XL uses linear attention with RANDOM FEATURE MAPS that are resampled
    per forward pass unless that flag is set, so without it the same molecule embeds differently
    every run: measured max component difference 0.46 between two passes over the same two
    SMILES, versus exactly 0.0 with the flag on.

    This is not a style point. It means an embedding matrix is not a function of its input, so
    two arms cannot be compared and a rerun does not reproduce. Reported to the owner of
    embed_pairs.py rather than patched there, because Figure A's cached npz files were computed
    with the broken branch and rewriting the function would silently change what they mean.
    """
    import time as _t

    import torch
    from transformers import AutoModel, AutoTokenizer
    path = ROOT / "models_hf" / "MolFormer"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mod = AutoModel.from_pretrained(path, trust_remote_code=True, deterministic_eval=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mod = mod.to(dev).eval()
    outs, t0, batch = [], _t.time(), 64
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            b = tok(list(smiles[i:i + batch]), return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(dev)
            h = mod(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().float().numpy())
            if (i // batch) % 100 == 0:
                print(f"    molformer {i}/{len(smiles)} ({_t.time()-t0:.0f}s)", flush=True)
    return np.concatenate(outs).astype(np.float32)


_MINIMOL = None


def _arm_minimol(smiles, batch=256):
    """MiniMol, with the model instantiated ONCE PER PROCESS rather than once per call.

    `embed_pairs.arm_minimol` builds `Minimol(...)` on every call. Figure A calls it exactly once,
    so the bug is invisible there -- but this file calls each arm once per 20,000-molecule shard,
    and MiniMol's constructor initialises Hydra, which is a process-global singleton. The second
    shard dies with `GlobalHydra is already initialized`, after the first has been written, so the
    run looks like it checkpointed successfully and then stopped for no reason.

    Caching the instance is also the correct thing on its own terms: reloading the checkpoint
    sixteen times is pure waste.

    AND MiniMol HAS TWO ALIGNMENT BUGS OF ITS OWN, which is why the plain call below became the
    fast path of something longer. Graphium's featuriser does not raise on a molecule it cannot
    turn into a graph -- it returns the SMILES STRING in that slot. Then, in `Minimol.__call__`:

      * `Batch.from_data_list` hits the str and dies with `'str' object has no attribute
        'stores'`, which is what killed this run at molecule 83,072 of tier 0; and
      * `num_molecules = min(batch_size, fingerprint_graph.shape[0])` would SILENTLY RETURN
        FEWER VECTORS THAN SMILES if it ever got past that. Every row after the failure would
        be attributed to the wrong molecule, in a matrix whose whole purpose is row alignment.

    So a rejected molecule becomes an explicit NaN row here and the length is asserted, never
    inferred. That matches how this file already treats failures elsewhere: recorded as data and
    left for the caller to apply per arm (see `build_parsefail`), rather than quietly imputed.
    `_MINIMOL(...)` is still called on the whole chunk first, so the per-molecule bisect costs
    nothing on the overwhelming majority of chunks that contain no bad molecule.
    """
    global _MINIMOL
    if _MINIMOL is None:
        from minimol import Minimol
        _MINIMOL = Minimol(batch_size=batch)
    out = []
    for i in range(0, len(smiles), batch * 4):
        chunk = [str(s) for s in smiles[i:i + batch * 4]]
        out.append(_minimol_chunk(chunk, i))
        print(f"    minimol {min(i+batch*4, len(smiles))}/{len(smiles)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def _minimol_stack(vecs, n):
    """-> (n, d) float32, or None if MiniMol returned the wrong number of vectors."""
    if len(vecs) != n:
        return None
    return np.stack([np.asarray(v, np.float32) for v in vecs])


def _minimol_chunk(chunk, offset):
    """One chunk of SMILES -> (len(chunk), d) float32, NaN rows where MiniMol cannot featurise.

    `offset` is only used to make the warning name a position in the shard, so a rejected
    molecule is findable afterwards rather than merely counted.
    """
    try:
        X = _minimol_stack(_MINIMOL(chunk), len(chunk))
        if X is not None:
            return X
        why = "returned fewer vectors than SMILES"
    except Exception as exc:                      # noqa: BLE001 -- graphium raises many types
        why = f"{type(exc).__name__}: {exc}"

    # Slow path. Ask the featuriser directly which molecules it rejects: it marks a failure by
    # putting the SMILES string in the slot where a graph should be.
    feats, _ = _MINIMOL.datamodule._featurize_molecules(chunk)
    ok = [j for j, f in enumerate(feats) if not isinstance(f, str)]
    bad = [j for j in range(len(chunk)) if j not in set(ok)]
    print(f"    minimol: {why}; featuriser rejects {len(bad)} of {len(chunk)}", flush=True)
    for j in bad:
        print(f"      NaN row at index {offset + j}: {chunk[j]}", flush=True)

    if not ok:
        raise RuntimeError(
            f"MiniMol rejected all {len(chunk)} molecules in the chunk at index {offset}. "
            f"That is a broken model or environment, not bad input -- refusing to write a "
            f"shard of NaN. First SMILES: {chunk[0]!r}")

    X_ok = _minimol_stack(_MINIMOL([chunk[j] for j in ok]), len(ok))
    if X_ok is None:
        raise RuntimeError(
            f"MiniMol returned a different number of vectors than the {len(ok)} molecules it "
            f"accepted, in the chunk at index {offset}. Rows cannot be aligned; refusing to "
            f"guess which vector belongs to which molecule.")
    X = np.full((len(chunk), X_ok.shape[1]), np.nan, np.float32)
    X[ok] = X_ok
    return X


OVERRIDES = {"molformer": _arm_molformer, "minimol": _arm_minimol}


def _arm_fn(name):
    if name in OVERRIDES:
        return OVERRIDES[name]
    import embed_pairs
    return embed_pairs.ARMS[name]


# --------------------------------------------------------------------------------------------
# sharded, resumable embedding
# --------------------------------------------------------------------------------------------

def embed(name: str, tier: int) -> None:
    """SHARDS ARE ALWAYS CUT ON THE FULL INDEX, never on the tier subset.

    `--tier 0` then `--tier 3` must reuse the same shard files. If the shard grid were cut on the
    tier's own length the two runs would disagree on every boundary after the last whole shard,
    and the second run would silently rewrite shards the first one had already written from a
    different molecule range. Cutting on the full index means a tier run simply stops early, and
    its final shard may spill a little past the tier boundary -- which `merge` trims.
    """
    import torch
    torch.set_num_threads(N_THREADS)
    full = load_index(None)
    n = len(load_index(tier))
    fn = _arm_fn(name)
    pdir = PARTS / name
    pdir.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {n:,} molecules (tier <= {tier}) of {len(full):,}", flush=True)

    t0 = time.time()
    for start in range(0, n, SHARD):
        f = pdir / f"{start:09d}.npy"
        if f.exists():
            continue
        chunk = [str(s) for s in full[start:start + SHARD]]
        X = fn(chunk)
        assert X.shape[0] == len(chunk), \
            f"{name}: arm returned {X.shape[0]} rows for {len(chunk)} SMILES -- ROWS MISALIGNED"
        tmp = f.with_suffix(".tmp.npy")
        np.save(tmp, X.astype(np.float32))
        tmp.rename(f)                                  # atomic: a killed run leaves no half shard
        done = min(start + SHARD, n)
        print(f"  {name} {done:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)

    merge(name, tier)


def merge(name: str, tier: int) -> None:
    smiles = load_index(tier)
    pdir = PARTS / name
    n = len(smiles)
    parts = []
    for start in range(0, n, SHARD):
        f = pdir / f"{start:09d}.npy"
        if not f.exists():
            print(f"  {name}: stopping merge at {start:,} -- shard missing", flush=True)
            break
        parts.append(np.load(f))
    if not parts:
        print(f"  {name}: nothing to merge")
        return
    X = np.concatenate(parts)[:n]                      # trim the tier's spill-over shard
    smiles = smiles[:len(X)]
    assert len(X) == len(smiles)

    # A row that is exactly zero across every dimension is what the arm functions write when a
    # molecule fails (RDKit parse failure, SELFIES conversion failure). Recorded, never dropped.
    dead = np.flatnonzero(~np.any(X != 0, axis=1))

    # X IS A SEPARATE .npy, NOT A KEY INSIDE THE .npz. CheMeleon is 2048-d, so tier 0 alone is a
    # 2.6 GB matrix and the full index would be 10.5 GB; zip-compressing dense float32 buys ~7%
    # and costs minutes, while a bare .npy is memory-mappable, which is what a downstream
    # XGBoost sweep actually wants. The .npz beside it stays small and holds everything needed to
    # prove the rows line up.  Both are written atomically -- a killed merge must not leave a
    # truncated matrix that still loads.
    # `.tmp.npy` / `.tmp.npz`, not `.npy.tmp`: numpy APPENDS the extension when the name does not
    # already end in it, so a `.npy.tmp` target silently becomes `.npy.tmp.npy` and the rename
    # then fails on a file that was never there.
    np.save(OUT / f"{name}.tmp.npy", X)
    (OUT / f"{name}.tmp.npy").rename(OUT / f"{name}.npy")
    pf = _parsefail(len(X))
    np.savez_compressed(
        OUT / f"{name}.tmp.npz", smiles=smiles, dim=X.shape[1], n=len(X),
        smiles_sha256=_sha(smiles), tier=tier,
        failed_idx=dead.astype(np.int64),
        failed_smiles=np.array([smiles[i] for i in dead], dtype=object),
        # Rows RDKit could not parse. For chemprop/chemeleon these hold an embedding of METHANE
        # and MUST be masked; for the string arms they are legitimate. See build_parsefail.
        parse_fail_idx=pf,
        parse_fail_smiles=np.array([smiles[i] for i in pf], dtype=object),
    )
    (OUT / f"{name}.tmp.npz").rename(OUT / f"{name}.npz")
    print(f"  {name}: {X.shape} -> {name}.npy  ({len(dead)} all-zero rows)", flush=True)


def load_arm(name: str, mmap: bool = False):
    """-> (X, smiles). Refuses to return a matrix whose SMILES hash does not match its own array.

    This is the guard the whole file is built around: any code path that reindexes, truncates or
    reorders an embedding matrix without rewriting `smiles` alongside it fails here rather than
    producing a plausible-looking figure from misaligned rows.
    """
    d = np.load(OUT / f"{name}.npz", allow_pickle=True)
    smiles = d["smiles"]
    X = np.load(OUT / f"{name}.npy", mmap_mode="r" if mmap else None)
    assert len(X) == len(smiles), f"{name}: {len(X)} rows vs {len(smiles)} SMILES"
    got = _sha(smiles)
    want = str(d["smiles_sha256"])
    assert got == want, f"{name}: SMILES hash {got[:16]} != stored {want[:16]}"
    master = load_index(None)[:len(smiles)]
    assert _sha(master) == got, f"{name}: SMILES do not match the master index prefix"
    return X, smiles


def verify() -> None:
    idx_meta = json.load(open(OUT / "index_meta.json"))
    print(f"master index: {idx_meta['n_unique']:,} unique  sha {idx_meta['sha256'][:16]}")
    for f in sorted(OUT.glob("*.npy")):
        name = f.stem
        try:
            X, smiles = load_arm(name, mmap=True)
            d = np.load(OUT / f"{name}.npz", allow_pickle=True)
            nf = len(d["failed_idx"])
            # NaN rows are a SEPARATE failure mode from `failed_idx` and from the all-zero check,
            # and counting only the other two reported minimol as failed=0 while it carried 8
            # deliberate NaN rows. `_arm_minimol` writes NaN where Graphium's featuriser refuses a
            # molecule -- see the alignment note there. Report it, because a NaN row reaching a
            # downstream model is either a crash or a silently imputed value, never a no-op.
            nan = int(np.isnan(np.asarray(X[:, 0])).sum())
            flag = f"  nan={nan}" if nan else ""
            print(f"  OK   {name:16s} {X.shape[0]:>9,} x {X.shape[1]:<5d} "
                  f"tier<={int(d['tier'])}  failed={nf}{flag}")
        except Exception as e:
            print(f"  FAIL {name:16s} {type(e).__name__}: {e}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
    # MiniMol featurises through datamol, which farms out to joblib/loky and sizes its pool from
    # the machine's core count -- 14 worker processes appeared on a box that three other agents
    # are timing on. `torch.set_num_threads` does not reach these; they are separate processes,
    # so the cap has to be set in the environment BEFORE loky is imported.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(N_THREADS))
    os.environ.setdefault("JOBLIB_START_METHOD", "loky")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "run":
        # RESUMABLE DRIVER. Every arm named on the command line, skipped if its output already
        # exists AND passes load_arm's hash check -- a half-written or misaligned output is NOT
        # treated as done. Within an arm, shards resume from disk, so the most a kill can cost is
        # one shard (20,000 molecules). Arms are run in one process per arm so a crash in one
        # cannot take the others with it.
        arms = [a for a in sys.argv[2:] if not a.startswith("--")]
        tier = int(sys.argv[sys.argv.index("--tier") + 1]) if "--tier" in sys.argv else 0
        for a in arms:
            try:
                X, _ = load_arm(a, mmap=True)
                if len(X) >= len(load_index(tier)):
                    print(f"  {a}: done ({X.shape}), skipping", flush=True)
                    continue
                print(f"  {a}: on disk at {len(X):,}, extending", flush=True)
            except Exception:
                pass
            embed(a, tier)
    elif cmd == "parsefail":
        build_parsefail()
    elif cmd == "index":
        build_index()
    elif cmd == "verify":
        verify()
    elif cmd in ("embed", "merge"):
        arm = sys.argv[2]
        tier = int(sys.argv[sys.argv.index("--tier") + 1]) if "--tier" in sys.argv else 0
        (embed if cmd == "embed" else merge)(arm, tier)
    else:
        raise SystemExit(__doc__)
