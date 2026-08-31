"""Featurisation throughput at scale, on a named EC2 instance. Writes Figure D's data contract.

    python bench_aws.py --arm hume --budget cpu --sizes 10000 100000 1000000 --out o.json

WHAT THIS IS FOR. Every throughput number in this project until now was a per-molecule cost
measured on 1-2k molecules and multiplied out to the target N, on a contended laptop. That is an
extrapolation wearing a measurement's clothes. This runs the whole job, on a machine whose price
is public, at three values of N, and reports wall clock -- so panel A of Figure D can show
whether cost per molecule is actually flat in N before panels B and C multiply anything.

FIVE ARMS, and the set is deliberate rather than everything that would run:
    ecfp        the floor -- parse + Morgan. Nothing that produces features can be cheaper.
    chemprop    an ordinary task-specific D-MPNN (chemprop defaults, d_h=300, depth=3).
    hume        this project: 864 descriptors + ECFP.
    mordred     what hume replaces: RDKit's 180 + mordred's 685 survivors + ECFP.
    chemberta   a small chemical language model, as a reference scale.
    chemeleon   a D-MPNN foundation model (d_h=2048, depth=6), the graph reference scale.

EVERY ARM GETS ITS BEST CONFIGURATION. `--sweep` measures each arm's throughput across batch
sizes at a small N and pins the winner for the large runs, because a number obtained by
underbatching someone else's method is not a comparison, it is a strawman. The chosen batch is
recorded in the output next to the timing.

TWO BUDGETS.
    cpu   every vCPU, no GPU. Work is sharded round-robin across worker processes and only a
          checksum returns, so the pipe is not being benchmarked.
    gpu   input preparation on the box's few vCPU, forward pass on the device. The two stages
          are timed SEPARATELY: Figure D stacks them, and the finding it exists to show is that
          for the graph arms the RDKit half dominates.

SHARDS ARE ROUND-ROBIN, AND N IS A STRIDE OVER THE CORPUS, NOT A PREFIX. The corpus is ordered
by size -- mean SMILES length over its first 10k / 100k / 1M is 39.2 / 42.5 / 49.9 characters --
so a prefix hands the small-N runs systematically smaller molecules and manufactures a scaling
bend out of nothing. Both were real bugs in the laptop version of this harness.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, statistics, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

CORPUS = os.environ.get("HUME_CORPUS", "data/corpus1m/selected.txt")
CORPUS_N = int(os.environ.get("HUME_CORPUS_N", "1000000"))
NCPU = os.cpu_count() or 1


def load(n: int) -> list[str]:
    step = max(1, CORPUS_N // n)
    out = []
    with open(CORPUS) as f:
        for i, line in enumerate(f):
            if i % step == 0:
                s = line.strip()
                if s:
                    out.append(s)
                    if len(out) == n:
                        break
    if len(out) < n:
        raise SystemExit(f"corpus {CORPUS} yielded {len(out)} of {n} requested molecules")
    return out


# =============================================================================== worker stages
# Each `_init_*` runs once per worker; each `_run_*` takes a shard plus a batch size and returns
# a checksum. Returning a checksum rather than the matrix is what keeps this a measurement of
# compute instead of one of multiprocessing's pipe.

def _quiet():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _init_ecfp():
    _quiet()
    global _GEN
    from rdkit.Chem import rdFingerprintGenerator as rfg
    _GEN = rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)


def _run_ecfp(job):
    shard, _bs = job
    from rdkit import Chem
    tot = 0
    for s in shard:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            tot += int(_GEN.GetFingerprintAsNumPy(m)[::64].sum())
    return tot


def _init_hume():
    _quiet()
    import molhume  # noqa: F401


def _run_hume(job):
    """`threads=1` IS DELIBERATE AND IS NOT LEAVING PERFORMANCE ON THE TABLE.

    hume.featurize_all now threads its row loop, one worker per hardware thread by default.
    This harness ALREADY runs one process per vCPU, so the box is saturated before the
    featuriser sees it -- leaving threads=0 would give 16 processes x 12 threads and spend the
    run in the scheduler. The in-process threading is for a single-process caller; here the
    parallelism is at the process level and threads=1 is the honest configuration.
    """
    shard, bs = job
    import numpy as np, hume
    tot = 0.0
    for lo in range(0, len(shard), bs):
        fp, X, _ = hume.featurize_all(shard[lo:lo + bs], threads=1)
        tot += float(np.nansum(X[:, ::197])) + float(fp[:, ::13].sum())
    return tot


def _init_ecfp_r2():
    _quiet()
    global _GEN
    from rdkit.Chem import rdFingerprintGenerator as rfg
    _GEN = rfg.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def _init_mordred():
    """RDKit's surviving 180 + mordred's surviving 685. Restricted to the survivors on purpose:
    mordred's full 1,613 would flatter us."""
    _quiet()
    import warnings
    warnings.filterwarnings('ignore')
    global _CALC, _RDK
    sys.path.insert(0, os.getcwd())
    import blocks
    from mordred import Calculator, descriptors as mdesc
    from rdkit.Chem import Descriptors
    full = Calculator(mdesc, ignore_3D=True)
    fam = {str(x): type(x).__module__.split(".")[-1] for x in full.descriptors}
    sp = blocks.split(fam)
    rows = sp["core"] + sp["predict"]
    mord = {n for s, n, _ in rows if s == "mordred"}
    rdk = {n for s, n, _ in rows if s == "rdkit"}
    _CALC = Calculator([x for x in full.descriptors if str(x) in mord], ignore_3D=True)
    _RDK = [(n, f) for n, f in Descriptors._descList if n in rdk]
    if len(_RDK) != len(rdk):
        raise RuntimeError(f"only {len(_RDK)} of {len(rdk)} surviving RDKit columns resolved; "
                           "the baseline would silently compute fewer columns than HUME")


def _run_mordred(job):
    """CHUNKED, because the un-chunked version was OOM-killed at N = 1e6.

    Materialising the whole shard's RDKit molecules means 16 workers each holding 62,500 of them
    at once on a 32 GiB box; the kernel killed a worker and the pool surfaced it as
    `BrokenProcessPool: A process in the process pool was terminated abruptly` -- which names no
    memory and reads like an unrelated crash. Chunking caps each worker at `bs` molecules and
    changes no arithmetic: mordred's Calculator is stateless across calls and the timing is the
    same work in the same order.
    """
    shard, bs = job
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    gen = rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)
    tot = 0.0
    for lo in range(0, len(shard), bs):
        mols = [Chem.MolFromSmiles(s) for s in shard[lo:lo + bs]]
        mols = [m for m in mols if m is not None]
        for m in mols:
            for _n, f in _RDK:
                try:
                    tot += float(f(m) or 0.0)
                except Exception:
                    pass
            tot += float(gen.GetFingerprintAsNumPy(m)[::256].sum())
        for _row in _CALC.map(mols, nproc=1, quiet=True):
            pass
        del mols
    return tot


# FIGURE C NEEDS THE TWO DESCRIPTOR BLOCKS SEPARATELY, not only their union. `mordred` above is
# ECFP + RDKit-180 + mordred-685, i.e. the `ecfp_all_desc` arm; Figure C also plots
# `ecfp_rdkit_desc` and `ecfp_mordred_desc`, and their cost is NOT recoverable from the union --
# the two blocks are wildly unequal (RDKit's 180 are cheap, mordred's 685 are not), so splitting
# the union in proportion to column count would be wrong by an order of magnitude.
#
# Both reuse `_init_mordred`'s globals so the column sets are IDENTICAL to the union arm's by
# construction; a second selection path is where the two figures start disagreeing about what
# "RDKit descriptors" means.
def _run_desc_block(job, want_rdkit, want_mordred):
    shard, bs = job
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    gen = rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)
    tot = 0.0
    for lo in range(0, len(shard), bs):
        mols = [Chem.MolFromSmiles(s) for s in shard[lo:lo + bs]]
        mols = [m for m in mols if m is not None]
        for m in mols:
            if want_rdkit:
                for _n, f in _RDK:
                    try:
                        tot += float(f(m) or 0.0)
                    except Exception:
                        pass
            tot += float(gen.GetFingerprintAsNumPy(m)[::256].sum())
        if want_mordred:
            for _row in _CALC.map(mols, nproc=1, quiet=True):
                pass
        del mols
    return tot


def _run_rdkit_desc(job):
    return _run_desc_block(job, True, False)


_DSTORUS = None


def _init_descriptastorus():
    _quiet()
    global _DSTORUS
    from descriptastorus.descriptors import rdDescriptors
    _DSTORUS = rdDescriptors.RDKit2D()


def _run_descriptastorus(job):
    """descriptastorus RDKit2D -- 200 columns, the tuned bulk wrapper around RDKit's descriptors.

    `processMol` rather than `process`: the latter re-parses the SMILES it is handed, and this
    harness has the molecule already. Measured locally over 2,000 molecules, RDKit's own
    Descriptors._descList is 4938 us/mol, descriptastorus `process` 4322 and `processMol` 3147 --
    so the fair number for "a fast descriptor implementation" is the one that is not paying for
    a parse the caller already did.
    """
    shard, bs = job
    from rdkit import Chem
    tot = 0.0
    for lo in range(0, len(shard), bs):
        for s in shard[lo:lo + bs]:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            row = _DSTORUS.processMol(m, s)
            if row:
                tot += float(row[1] or 0.0)
    return tot


def _run_mordred_desc(job):
    return _run_desc_block(job, False, True)


def _init_minimol():
    _quiet()
    global _MINIMOL
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    from minimol import Minimol
    _MINIMOL = Minimol()


def _run_minimol(job):
    shard, bs = job
    tot = 0.0
    for lo in range(0, len(shard), bs):
        out = _MINIMOL(shard[lo:lo + bs])
        tot += float(sum(float(v[::37].sum()) for v in out))
    return tot


def _init_chemberta():
    _quiet()
    global _TOK, _MDL
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    p = os.environ.get("CHEMBERTA_PATH", "models_hf/ChemBERTa-2-MLM")
    _TOK = AutoTokenizer.from_pretrained(p)
    _MDL = AutoModel.from_pretrained(p).eval()


def _embed_chemberta(shard, bs, dev="cpu"):
    import torch
    m = _MDL.to(dev)
    order = sorted(range(len(shard)), key=lambda i: len(shard[i]))
    srt = [shard[i] for i in order]
    tot = 0.0
    for lo in range(0, len(srt), bs):
        e = _TOK(srt[lo:lo + bs], return_tensors="pt", padding=True, truncation=True,
                 max_length=512)
        e = {k: v.to(dev) for k, v in e.items()}
        o = m(**e).last_hidden_state
        msk = e["attention_mask"].unsqueeze(-1).float()
        tot += float(((o * msk).sum(1) / msk.sum(1).clamp(min=1)).sum())
    return tot


def _run_chemberta(job):
    shard, bs = job
    return _embed_chemberta(shard, bs)


def _init_dmpnn(pretrained: bool):
    """A chemprop D-MPNN. `pretrained` loads the CheMeleon foundation weights.

    THE TWO ARMS ARE THE SAME ARCHITECTURE AT DIFFERENT SCALES, and that is the point of running
    both. `chemprop` is the ordinary task-specific encoder people train per dataset -- chemprop's
    own defaults, d_h=300 over depth=3. `chemeleon` is the published foundation model, and its
    checkpoint says d_h=2048 over depth=6: 2048^2 per directed bond per step, six steps, which is
    ~56x the arithmetic. Reporting one as "a GNN" would misrepresent the other by that factor.

    The untrained arm is randomly initialised on purpose. Weights do not change the cost of a
    forward pass, and this benchmark measures cost -- so an untrained encoder at chemprop's
    default size is an honest stand-in for a trained one and needs no checkpoint to exist.
    """
    _quiet()
    global _MP, _AGG, _FEAT
    import torch
    from chemprop import nn as cnn, featurizers
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    if pretrained:
        ck = torch.load(os.environ.get("CHEMELEON_PATH", "/root/.chemprop/chemeleon_mp.pt"),
                        weights_only=True)
        _MP = cnn.BondMessagePassing(**ck["hyper_parameters"])
        _MP.load_state_dict(ck["state_dict"])
    else:
        _MP = cnn.BondMessagePassing(d_h=300, depth=3)
    _MP.eval()
    _AGG = cnn.MeanAggregation()
    _FEAT = featurizers.SimpleMoleculeMolGraphFeaturizer()


def _init_chemeleon():
    _init_dmpnn(True)


def _init_chemprop():
    _init_dmpnn(False)


def _embed_chemeleon(shard, bs, dev="cpu", split=False):
    """-> checksum, or (prep_seconds, fwd_seconds) when `split` is set."""
    import torch
    from chemprop import data as cdata
    mp_ = _MP.to(dev)
    agg = _AGG.to(dev)
    t_prep = t_fwd = 0.0
    tot = 0.0
    for lo in range(0, len(shard), bs):
        t0 = time.perf_counter()
        dps = [cdata.MoleculeDatapoint.from_smi(s) for s in shard[lo:lo + bs]]
        dset = cdata.MoleculeDataset(dps, _FEAT)
        batch = next(iter(cdata.build_dataloader(dset, batch_size=len(dps), shuffle=False)))
        t1 = time.perf_counter()
        # `BatchMolGraph.to()` MUTATES IN PLACE AND RETURNS None -- it is not torch's `.to()`.
        # Writing `bmg = batch.bmg.to(dev)` binds None and the failure surfaces much later as
        # "'NoneType' object has no attribute 'V'" from inside the message-passing forward,
        # which reads like a malformed graph rather than a rebinding mistake. The chemeleon
        # preflight missed it because it ran on CPU and never called `.to()` at all.
        bmg = batch.bmg
        bmg.to(dev)
        h = agg(mp_(bmg), bmg.batch)
        if dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_prep += t1 - t0
        t_fwd += t2 - t1
        tot += float(h.sum())
    return (t_prep, t_fwd) if split else tot


def _run_chemeleon(job):
    shard, bs = job
    return _embed_chemeleon(shard, bs)


_run_chemprop = _run_chemeleon      # identical code path; only the weights and dims differ


ARMS = {
    "ecfp":      (_init_ecfp,      _run_ecfp,      [4096]),
    # ECFP4 (r=2), matching the `ecfp` arm Figures A, B and C are built on. The `ecfp` arm above
    # is r=3, which is what HUME carries INTERNALLY -- two different baselines that were being
    # used interchangeably. Both are measured so each figure can cite the one it actually ran.
    "ecfp_r2":   (_init_ecfp_r2,   _run_ecfp,      [4096]),
    "hume":      (_init_hume,      _run_hume,      [1024, 4096, 16384]),
    "mordred":   (_init_mordred,   _run_mordred,   [4096]),
    "chemberta": (_init_chemberta, _run_chemberta, [1, 32, 128, 512]),
    "chemeleon": (_init_chemeleon, _run_chemeleon, [64, 256, 1024]),
    "chemprop":  (_init_chemprop,  _run_chemprop,  [256, 1024, 4096]),
    "rdkit_desc":   (_init_mordred, _run_rdkit_desc,   [4096]),
    "descriptastorus": (_init_descriptastorus, _run_descriptastorus, [4096]),
    "mordred_desc": (_init_mordred, _run_mordred_desc, [4096]),
    "minimol":      (_init_minimol, _run_minimol,      [128, 512, 2048]),
}
GPU_BATCHES = {"chemberta": [128, 512, 1024, 2048], "chemeleon": [256, 1024, 4096],
               "chemprop": [1024, 4096, 16384]}


# ==================================================================================== budgets
#: Arms that cannot have one worker per vCPU. minimol loads a full graphium model INTO EVERY
#: WORKER, so sixteen copies plus their featuriser state exhausted a 32 GiB box and the pool
#: surfaced it as `BrokenProcessPool: A process in the process pool was terminated abruptly` at
#: every batch size -- an error that names no memory and reads like an unrelated crash. This is
#: the same disguise the Mordred OOM wore, which is why it is a named table rather than a comment.
#:
#: FEWER WORKERS IS NOT A HANDICAP HERE. The number is reported alongside the throughput and the
#: arm still gets its own best batch size, so the figure compares each arm in the best
#: configuration it can actually be RUN in -- an arm that cannot use sixteen workers genuinely
#: cannot, and reporting a throughput it can only reach by being killed would be worse.
MAX_WORKERS = {"minimol": 4}


def run_cpu(arm: str, smis: list[str], bs: int) -> float:
    init, fn, _ = ARMS[arm]
    nw = min(NCPU, MAX_WORKERS.get(arm, NCPU))
    shards = [(smis[i::nw], bs) for i in range(nw)]
    with ProcessPoolExecutor(max_workers=nw, initializer=init,
                             mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(fn, [(smis[:4], bs)] * nw))         # warm every worker, untimed
        t0 = time.perf_counter()
        list(ex.map(fn, shards))
        return time.perf_counter() - t0


def run_gpu(arm: str, smis: list[str], bs: int) -> tuple[float, float]:
    """-> (prep_seconds, fwd_seconds). Single process: the GPU is the shared resource."""
    import torch
    init, _fn, _ = ARMS[arm]
    init()
    if arm in ("chemeleon", "chemprop"):
        return _embed_chemeleon(smis, bs, dev="cuda", split=True)
    if arm == "chemberta":
        order = sorted(range(len(smis)), key=lambda i: len(smis[i]))
        srt = [smis[i] for i in order]
        t0 = time.perf_counter()
        encs = [_TOK(srt[lo:lo + bs], return_tensors="pt", padding=True, truncation=True,
                     max_length=512) for lo in range(0, len(srt), bs)]
        t_prep = time.perf_counter() - t0
        m = _MDL.to("cuda")
        t0 = time.perf_counter()
        for e in encs:
            e = {k: v.to("cuda") for k, v in e.items()}
            o = m(**e).last_hidden_state
            msk = e["attention_mask"].unsqueeze(-1).float()
            (o * msk).sum(1) / msk.sum(1).clamp(min=1)
        torch.cuda.synchronize()
        return t_prep, time.perf_counter() - t0
    raise SystemExit(f"arm '{arm}' has no GPU path -- it is CPU-only by construction")


def sweep(arm: str, budget: str, n: int = 10000) -> int:
    """Pick the batch size that makes this arm fastest. Its best configuration, not ours."""
    cands = GPU_BATCHES.get(arm, ARMS[arm][2]) if budget == "gpu" else ARMS[arm][2]
    if len(cands) == 1:
        return cands[0]
    smis = load(n)
    best, best_t = cands[0], float("inf")
    for bs in cands:
        try:
            t = run_cpu(arm, smis, bs) if budget == "cpu" else sum(run_gpu(arm, smis, bs))
        except Exception as e:
            print(f"    sweep {arm} bs={bs}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        print(f"    sweep {arm} bs={bs:>5}: {t / n * 1e6:8.1f} us/mol", flush=True)
        if t < best_t:
            best, best_t = bs, t
    print(f"  -> {arm}/{budget} best batch = {best}", flush=True)
    return best


def meta(budget: str) -> dict:
    def imds(path):
        try:
            tok = subprocess.run(["curl", "-sf", "-X", "PUT",
                                  "http://169.254.169.254/latest/api/token", "-H",
                                  "X-aws-ec2-metadata-token-ttl-seconds: 60"],
                                 capture_output=True, timeout=4).stdout.decode()
            r = subprocess.run(["curl", "-sf", "-H", f"X-aws-ec2-metadata-token: {tok}",
                                f"http://169.254.169.254/latest/meta-data/{path}"],
                               capture_output=True, timeout=4)
            v = r.stdout.decode().strip()
            return v or None
        except Exception:
            return None
    # A FAILED LOOKUP MUST NOT BECOME AN EMPTY STRING. Every log key and every figure caption is
    # derived from the instance type; silently blank means every box writes the same S3 key.
    itype = imds("instance-type") or os.environ.get("BENCH_INSTANCE")
    if not itype:
        raise SystemExit("cannot determine instance type from IMDS and BENCH_INSTANCE is unset; "
                         "refusing to write a result that cannot be attributed to hardware")
    return {"instance": itype, "instance_id": imds("instance-id") or "local",
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "vcpu": NCPU, "budget": budget, "host": socket.gethostname(),
            "python": platform.python_version(),
            "usd_per_hour_ondemand": float(os.environ.get("USD_ONDEMAND", "0")),
            "usd_per_hour_spot": float(os.environ.get("USD_SPOT", "0") or 0),
            "priced_on": os.environ.get("PRICED_ON", ""),
            "corpus": CORPUS, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--budget", required=True, choices=["cpu", "gpu"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 100000, 1000000])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = meta(a.budget)
    print(f"== {a.arm} / {a.budget} on {m['instance']} ({NCPU} vCPU) ==", flush=True)
    bs = sweep(a.arm, a.budget)
    points = []
    for n in a.sizes:
        smis = load(n)
        reps = a.reps if n <= 100_000 else 1
        best = None
        for _ in range(reps):
            if a.budget == "cpu":
                w = run_cpu(a.arm, smis, bs)
                rec = {"wall_s": w, "prep_s": None, "fwd_s": None}
            else:
                p, f = run_gpu(a.arm, smis, bs)
                rec = {"wall_s": p + f, "prep_s": p, "fwd_s": f}
            if best is None or rec["wall_s"] < best["wall_s"]:
                best = rec
        points.append({"arm": a.arm, "n": n, "batch": bs, **best})
        print(f"  N={n:>9,}  {best['wall_s']:9.2f}s  "
              f"{best['wall_s'] / n * 1e6:9.1f} us/mol  {n / best['wall_s']:9.0f} mol/s",
              flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"meta": m, "points": points}, fh, indent=1)
    print(f"  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
