"""Sustained end-to-end throughput at scale, measured across three decades of N.

WHY THIS FILE EXISTS. Every throughput number in this project before it was a per-molecule cost
from a 1k-2k molecule run multiplied by the target N. That is an extrapolation, not a
measurement, and it silently assumes the three things most likely to be false at scale:

  1. that per-molecule cost is FLAT in N (allocator behaviour, page faults, cache pressure and
     GC all say it need not be),
  2. that reading the input costs nothing,
  3. that moving results out of the workers costs nothing.

So this measures WALL CLOCK for the whole job, including reading the SMILES off disk, at
N = 10k / 100k / 1M. If mol/s is flat across two decades then extrapolating to 1e9 is a licensed
claim; if it bends, the extrapolation was wrong and we find out here rather than in review.

TWO HARDWARE BUDGETS, because they give different answers and the difference is the point:

  * `cpu`  -- all cores, no GPU. Every stage runs inside a worker process; only a checksum comes
             back, so this measures compute, not multiprocessing's pipe.
  * `gpu`  -- ONE core feeding one GPU. Input preparation (RDKit -> graph, or tokenise) is
             single-threaded on the main process; the forward pass is batched on the device. The
             two stages are timed SEPARATELY and both totals are reported: SUM is what you get
             with no overlap, MAX is what perfect pipelining would give. Reality is between them,
             and the gap says whether the GPU is being starved.

usage:  python bench_scale_e2e.py [cpu|gpu] [N ...]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RDKIT_NUM_THREADS"):
    os.environ[_v] = "1"

import time, json, statistics, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CORPUS = "/Users/lsieben/VSCode/universal-encoder/data/corpus1m/selected.txt"
MODELS = "/Users/lsieben/VSCode/universal-encoder/models_hf"
NCORE = os.cpu_count()


CORPUS_N = 1_000_000


def load(n):
    """Read the input, and COUNT IT. At 1M this is 51 MB and it is part of the job.

    STRIDE, NOT PREFIX, AND THIS IS NOT A DETAIL. `data/corpus1m/selected.txt` is ORDERED --
    mean SMILES length over the first 10k / 100k / 1M is 39.2 / 42.5 / 49.9 characters, against
    49.9 for a uniform sample. Taking the first n therefore hands the small-N arms systematically
    smaller molecules, and a "throughput bend" appears between 100k and 1M that is nothing but
    the corpus getting heavier. The first version of this harness did exactly that and reported
    HUME degrading 81.3 -> 124.7 us/mol; most of that was the corpus.

    Striding gives every N the same size distribution, which is the only way the three points
    answer the question they are being asked -- does cost per molecule depend on N?
    """
    step = max(1, CORPUS_N // n)
    out = []
    with open(CORPUS) as f:
        for i, line in enumerate(f):
            if i % step == 0:
                out.append(line.strip())
                if len(out) == n:
                    break
    return out


# ---------------------------------------------------------------- worker-side stages
def _init_hume():
    import molhume  # noqa: F401
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _hume(shard):
    import numpy as np, hume
    fp, X, _ = hume.featurize_all(shard)
    return float(np.nansum(X[:, ::197])) + float(fp[:, ::13].sum())


def _hume_stream(shard):
    """The same work, in chunks, never holding the whole matrix.

    `featurize_all` allocates its full (n, 1266) float64 output up front. At N = 1M over 12
    workers that is 844 MB of descriptors plus 171 MB of fingerprint PER WORKER -- 12.2 GB on a
    25.8 GB machine, before the parent and everything else on the box -- and the measured cost
    goes from 81.3 us/mol at N = 100k to 124.7 at N = 1M. That bend is memory pressure, not
    arithmetic. A real pipeline writes shards as it goes and never materialises the matrix, so
    this arm is the one that belongs in a scaling claim; the difference between them is a
    property of the CALLER, not of the descriptors.
    """
    import numpy as np, hume
    tot = 0.0
    for lo in range(0, len(shard), 8192):
        fp, X, _ = hume.featurize_all(shard[lo:lo + 8192])
        tot += float(np.nansum(X[:, ::197])) + float(fp[:, ::13].sum())
    return tot


def _init_gin():
    global _GIN
    import torch, torch.nn as nn
    from torch_geometric.nn import GINConv
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    torch.set_grad_enabled(False); torch.set_num_threads(1)
    H = 300
    class GIN(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Linear(9, H)
            s.convs = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(H, H), nn.ReLU(),
                                                           nn.Linear(H, H))) for _ in range(5)])
            s.bns = nn.ModuleList([nn.BatchNorm1d(H) for _ in range(5)])
        def forward(s, d):
            from torch_geometric.nn import global_mean_pool
            x = s.emb(d.x)
            for c, b in zip(s.convs, s.bns):
                x = torch.relu(b(c(x, d.edge_index)))
            return global_mean_pool(x, d.batch)
    _GIN = GIN().eval()


def to_graph(smi):
    import torch
    from torch_geometric.data import Data
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    x = torch.tensor([[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(),
                       int(a.GetHybridization()), int(a.GetIsAromatic()), int(a.IsInRing()),
                       int(a.GetChiralTag()), a.GetNumRadicalElectrons()] for a in m.GetAtoms()],
                     dtype=torch.float)
    ei = [[], []]
    for b in m.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei[0] += [u, v]; ei[1] += [v, u]
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long) if ei[0]
                else torch.zeros((2, 0), dtype=torch.long))


def _gin(shard):
    import torch
    from torch_geometric.data import Batch
    tot = 0.0
    for lo in range(0, len(shard), 256):
        gs = [g for g in (to_graph(s) for s in shard[lo:lo + 256]) if g is not None]
        if gs:
            tot += float(_GIN(Batch.from_data_list(gs)).sum())
    return tot


def _init_clm_mlm(): _init_clm(f"{MODELS}/ChemBERTa-2-MLM", False)
def _init_clm_mf():  _init_clm(f"{MODELS}/MolFormer", True)


def _init_clm(path, trust):
    global _TOK, _MDL
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_grad_enabled(False); torch.set_num_threads(1)
    _TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    _MDL = AutoModel.from_pretrained(path, trust_remote_code=trust).eval()


def _clm(shard):
    """Batch 1: on ONE core, padding a batch costs more than batching saves. Measured:
    ChemBERTa 1,198 us/mol at batch 1 against 2,240 at batch 256 length-sorted."""
    tot = 0.0
    for s in shard:
        e = _TOK([s], return_tensors="pt", truncation=True, max_length=512)
        tot += float(_MDL(**e).last_hidden_state.mean().item())
    return tot


ARMS = {"hume": (_init_hume, _hume), "hume_stream": (_init_hume, _hume_stream), "gin": (_init_gin, _gin),
        "chemberta": (_init_clm_mlm, _clm), "molformer": (_init_clm_mf, _clm)}


# ---------------------------------------------------------------- the two budgets
def run_cpu_paired(arms, sizes, reps):
    """Arms INTERLEAVED within each repetition, and the ratio reported alongside the absolutes.

    This machine is not quiet and cannot be made quiet on demand -- Spotlight indexing has held
    ~1 core through these runs, and a 1M non-streaming arm pushed it into 10 GB of swap. Absolute
    wall-clock numbers taken under that are worthless. A RATIO taken from arms that ran seconds
    apart under the same conditions is not: whatever slowed one arm slowed the other.

    So the deliverable here is `hume / gin`, and the absolutes are reported only so a later quiet
    re-run has something to be compared against.
    """
    print(f"\n=== BUDGET: {NCORE} CPU cores, no GPU. Arms INTERLEAVED, ratio is the claim. ===")
    res = {}
    for n in sizes:
        per = {a: [] for a in arms}
        for r in range(reps):
            for arm in (arms[r % len(arms):] + arms[:r % len(arms)]):
                init, fn = ARMS[arm]
                smis = load(n)
                shards = [smis[i::NCORE] for i in range(NCORE)]
                with ProcessPoolExecutor(max_workers=NCORE, initializer=init,
                                         mp_context=mp.get_context("spawn")) as ex:
                    list(ex.map(fn, [smis[:4]] * NCORE))
                    t0 = time.perf_counter()
                    list(ex.map(fn, shards))
                    per[arm].append(time.perf_counter() - t0)
        la = os.getloadavg()[0]
        line = f"  N={n:>9,}  load1 {la:6.1f}  "
        for arm in arms:
            w = min(per[arm])                       # min over repetitions: least-contaminated
            res[f"{arm}:{n}"] = {"wall_s": w, "us_per_mol": w / n * 1e6, "mol_s": n / w,
                                 "all_walls": per[arm], "load1": la}
            line += f"{arm} {w/n*1e6:7.1f} us/mol ({n/w:7.0f} mol/s)   "
        if len(arms) == 2:
            a, b = (min(per[x]) for x in arms)
            line += f"|  {arms[0]}/{arms[1]} = {a/b:.2f}x"
        print(line, flush=True)
    return res


def run_cpu(arms, sizes, reps):
    print(f"\n=== BUDGET: {NCORE} CPU cores, no GPU. Wall clock, input read from disk. ===")
    res = {}
    for arm in arms:
        init, fn = ARMS[arm]
        for n in sizes:
            xs = []
            for _ in range(reps if n <= 100_000 else 1):
                t0 = time.perf_counter()
                smis = load(n)
                shards = [smis[i::NCORE] for i in range(NCORE)]
                with ProcessPoolExecutor(max_workers=NCORE, initializer=init,
                                         mp_context=mp.get_context("spawn")) as ex:
                    list(ex.map(fn, [smis[:4]] * NCORE))       # warm workers, untimed
                    t0 = time.perf_counter()                    # <- clock starts after warmup
                    list(ex.map(fn, shards))
                xs.append(time.perf_counter() - t0)
            w = statistics.mean(xs)
            res[f"{arm}:{n}"] = {"wall_s": w, "us_per_mol": w / n * 1e6, "mol_s": n / w}
            print(f"  {arm:11s} N={n:>9,}  wall {w:8.2f}s   {w/n*1e6:9.1f} us/mol   "
                  f"{n/w:9.0f} mol/s", flush=True)
    return res


def run_gpu(arms, sizes):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n=== BUDGET: 1 CPU core + 1 GPU ({dev}). Stages timed separately. ===")
    res = {}
    if "gin" in arms:
        _init_gin()
        from torch_geometric.data import Batch
        for n in sizes:
            smis = load(n)
            t0 = time.perf_counter()
            gs = [g for g in (to_graph(s) for s in smis) if g is not None]
            t_prep = time.perf_counter() - t0
            m = _GIN.to(dev)
            t0 = time.perf_counter()
            for lo in range(0, len(gs), 256):
                m(Batch.from_data_list(gs[lo:lo + 256]).to(dev))
            if dev == "mps":
                torch.mps.synchronize()
            t_fwd = time.perf_counter() - t0
            _report("gin", n, t_prep, t_fwd, res)
            _GIN.to("cpu")
    for arm, init in (("chemberta", _init_clm_mlm), ("molformer", _init_clm_mf)):
        if arm not in arms:
            continue
        init()
        for n in sizes:
            smis = load(n)
            order = sorted(range(len(smis)), key=lambda i: len(smis[i]))
            srt = [smis[i] for i in order]
            t0 = time.perf_counter()
            encs = [_TOK(srt[lo:lo + 256], return_tensors="pt", padding=True, truncation=True,
                         max_length=512) for lo in range(0, n, 256)]
            t_prep = time.perf_counter() - t0
            m = _MDL.to(dev)
            t0 = time.perf_counter()
            for e in encs:
                e = {k: v.to(dev) for k, v in e.items()}
                o = m(**e).last_hidden_state
                msk = e["attention_mask"].unsqueeze(-1).float()
                (o * msk).sum(1) / msk.sum(1).clamp(min=1)
            if dev == "mps":
                torch.mps.synchronize()
            t_fwd = time.perf_counter() - t0
            _report(arm, n, t_prep, t_fwd, res)
            _MDL.to("cpu")
    return res


def _report(arm, n, t_prep, t_fwd, res):
    s, mx = t_prep + t_fwd, max(t_prep, t_fwd)
    res[f"{arm}:{n}"] = {"prep_s": t_prep, "fwd_s": t_fwd, "sum_s": s, "max_s": mx,
                         "us_per_mol_sum": s / n * 1e6, "mol_s_sum": n / s}
    print(f"  {arm:11s} N={n:>9,}  prep {t_prep:7.2f}s  fwd {t_fwd:7.2f}s  |  "
          f"no-overlap {s:7.2f}s ({n/s:8.0f} mol/s)  perfect-overlap {mx:7.2f}s "
          f"({n/mx:8.0f} mol/s)  bound: {'CPU' if t_prep > t_fwd else 'GPU'}", flush=True)


if __name__ == "__main__":
    budget = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    sizes = [int(x) for x in sys.argv[2:]] or [10_000, 100_000, 1_000_000]
    arms = os.environ.get("ARMS", "hume,gin").split(",")
    out = run_cpu_paired(arms, sizes, reps=3) if budget == "cpu" else run_gpu(arms, sizes)
    p = f"results/scale_{budget}.json"
    os.makedirs("results", exist_ok=True)
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(out); json.dump(old, open(p, "w"), indent=1)
    print(f"\n  -> {p}")
