"""Build the 1M-molecule training corpus.

    python corpus.py scaffolds [--workers 6]     scan the pool, cache scaffold assignments
    python corpus.py select    [--n 1000000]     scaffold-stratified pick, benchmark excluded
    python corpus.py compute   [--workers 6]     descriptors, sharded and resumable
    python corpus.py pack                        slice shards into X / Y and report NaN rates

Reads the pool read-only from ChemTFM_OLD; **writes only under this repo's data/corpus1m/**.

Sized from measurements on this machine (12 cores), not estimates:

    Mordred, all 1,613 columns         80,088 us/mol   22.25 core-h per 1M
    Mordred, only the 685 we need      35,811 us/mol    9.95 core-h per 1M
    RDKit 217                           4,168 us/mol    1.16 core-h per 1M
    our five blocks (Python)              ~517 us/mol    0.14 core-h per 1M

Restricting Mordred is the single biggest saving available: Chi moved to CORE, and another
~928 columns were killed by the dedupe or go unused, so computing the full set would waste
12.3 core-hours per million molecules for nothing.

**Reference vs production path.** This script computes the *reference* values with Mordred and
RDKit. `chi.py`, `cycles.py`, `resistance.py`, `conjugation.py` and `stereo.py` are the fast
implementations that will actually run at inference. They are computed here too, side by side,
so `verify.py --corpus` can check one against the other over all 1M molecules. A descriptor is
only allowed into CORE once that check is exact.

**Benchmark exclusion.** 0.93% of the pool canonicalises into the benchmark's 42,390 unique
structures, so an unfiltered 1M draw would contain ~9,300 of them. The surrogate would then be
trained to reproduce descriptors for molecules it is later scored on, and the downstream
comparison would flatter it. Exclusion is a hash-set lookup and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "corpus1m"
POOL = Path("/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/pubchem_10M.smi")   # read-only
BENCH = ROOT / "data" / "surrogate" / "bench.npz"
SHARD = 50_000
DEPTH = 3          # molecules per scaffold; see PLAN.md for why not 1
MIN_ATOMS, MAX_ATOMS = 5, 60
ORGANIC = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}


# --- pass 1: scaffolds ---------------------------------------------------------------------

def _scan(args):
    lo, lines = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for k, line in enumerate(lines):
        s = line.split()[0] if line else ""
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n = m.GetNumAtoms()
        if n < MIN_ATOMS or n > MAX_ATOMS:
            continue
        if any(a.GetAtomicNum() not in ORGANIC for a in m.GetAtoms()):
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            continue
        # Canonical form is needed anyway for benchmark exclusion, and we already hold the mol.
        out.append((lo + k, scaf, Chem.MolToSmiles(m)))
    return out


def _chunks(workers):
    with POOL.open() as fh:
        buf, lo = [], 0
        for i, line in enumerate(fh):
            buf.append(line)
            if len(buf) >= 100_000:
                yield (lo, buf)
                buf, lo = [], i + 1
        if buf:
            yield (lo, buf)


def cmd_scaffolds(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    with Pool(a.workers) as p:
        for i, res in enumerate(p.imap_unordered(_scan, _chunks(a.workers), chunksize=1)):
            rows.extend(res)
            if (i + 1) % 10 == 0:
                print(f"  {(i + 1) * 100_000:,} scanned, {len(rows):,} kept "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"scan done: {len(rows):,} usable molecules in {time.time() - t0:.0f}s")

    idx = np.fromiter((r[0] for r in rows), np.int64, len(rows))
    scafs = [r[1] for r in rows]
    canon = [r[2] for r in rows]
    uniq = {}
    sid = np.empty(len(scafs), np.int64)
    for i, s in enumerate(scafs):
        sid[i] = uniq.setdefault(s, len(uniq))
    print(f"  {len(uniq):,} distinct scaffolds")
    np.savez_compressed(OUT / "scan.npz", idx=idx, sid=sid,
                        canon=np.array(canon, dtype=object))
    pickle.dump({v: k for k, v in uniq.items()}, open(OUT / "scaffold_names.pkl", "wb"))
    print(f"wrote {OUT / 'scan.npz'}")


# --- pass 2: selection ---------------------------------------------------------------------

def cmd_select(a) -> None:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    z = np.load(OUT / "scan.npz", allow_pickle=True)
    idx, sid, canon = z["idx"], z["sid"], list(z["canon"])

    b = np.load(BENCH, allow_pickle=True)
    bench = {Chem.MolToSmiles(m) for m in
             (Chem.MolFromSmiles(s) for s in b["smiles"]) if m is not None}
    keep = np.fromiter((c not in bench for c in canon), bool, len(canon))
    print(f"pool {len(canon):,} | benchmark structures {len(bench):,} | "
          f"excluded {int((~keep).sum()):,} ({100 * (~keep).mean():.2f}%)")

    idx, sid = idx[keep], sid[keep]
    order = np.argsort(sid, kind="stable")
    idx, sid = idx[order], sid[order]
    bounds = np.flatnonzero(np.diff(sid)) + 1
    groups = np.split(np.arange(len(sid)), bounds)
    sizes = np.fromiter((g.size for g in groups), np.int64, len(groups))
    print(f"  {len(groups):,} scaffolds | >={DEPTH} members: {int((sizes >= DEPTH).sum()):,}")

    rng = np.random.default_rng(0)

    # Acyclic molecules all share the *empty* Murcko scaffold, so plain depth-3 selection would
    # take three of them out of the entire pool. Measured: 15.9% of the pool is acyclic but only
    # 0.2% of the benchmark is, so heavy down-weighting is correct -- three molecules is not.
    # An explicit quota (default 2%) sits an order of magnitude above benchmark prevalence, so
    # the surrogate sees plenty of acyclic examples, while staying far below pool prevalence so
    # the corpus is not spent on the easy case. Several blocks are identically zero on acyclic
    # input, so there is genuinely less to learn there.
    empty_sid = next((i for i, s in pickle.load(
        open(OUT / "scaffold_names.pkl", "rb")).items() if s == ""), None)
    acyc_quota = int(a.acyclic_frac * a.n)
    picks = []
    rest = []
    for g in groups:
        if empty_sid is not None and sid[g[0]] == empty_sid:
            rng.shuffle(g)
            picks.extend(g[:acyc_quota])
            print(f"  acyclic: {g.size:,} available, taking {min(g.size, acyc_quota):,} "
                  f"({100 * a.acyclic_frac:.1f}% of corpus vs 0.2% of benchmark)")
        else:
            rest.append(g)

    rich = [g for g in rest if g.size >= DEPTH]
    poor = [g for g in rest if g.size < DEPTH]
    rng.shuffle(rich)
    rng.shuffle(poor)
    for g in rich:
        picks.extend(g[:DEPTH])
        if len(picks) >= a.n:
            break
    if len(picks) < a.n:
        print(f"  rich scaffolds gave {len(picks):,}; topping up from the singleton tail")
        for g in poor:
            picks.extend(g)
            if len(picks) >= a.n:
                break
    picks = np.array(picks[:a.n])
    chosen = idx[picks]
    per_scaf = len(picks) / len(set(sid[picks].tolist()))
    print(f"selected {len(chosen):,} molecules, {per_scaf:.2f} per scaffold")

    want = set(chosen.tolist())
    smiles = []
    with POOL.open() as fh:
        for i, line in enumerate(fh):
            if i in want:
                smiles.append(line.split()[0])
    (OUT / "selected.txt").write_text("\n".join(smiles) + "\n")
    print(f"wrote {OUT / 'selected.txt'} ({len(smiles):,} lines)")


# --- pass 3: descriptors -------------------------------------------------------------------

def _mordred_subset():
    import blocks
    from mordred import Calculator, descriptors as mdesc
    full = Calculator(mdesc, ignore_3D=True)
    fam = {str(x): type(x).__module__.split(".")[-1] for x in full.descriptors}
    sp = blocks.split(fam)
    need = {n for s, n, _ in sp["core"] if s == "mordred"}
    need |= {n for s, n, _ in sp["predict"] if s == "mordred"}
    sub = [x for x in full.descriptors if str(x) in need]
    return Calculator(sub, ignore_3D=True), [str(x) for x in sub]


def _shard(args):
    si, smiles, base = args
    f = Path(base) / "shards" / f"sh_{si:05d}.npz"
    if f.exists():
        return si, 0.0, True
    import importlib
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    calc, mnames = _mordred_subset()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rnames = [n for n, _ in Descriptors._descList]
    rlut = dict(Descriptors._descList)
    mods = {b: importlib.import_module(b)
            for b in ("resistance", "cycles", "conjugation", "stereo", "chi")}

    t0 = time.time()
    N = len(smiles)
    E = np.zeros((N, 2048), np.uint8)
    RD = np.full((N, len(rnames)), np.nan, np.float32)
    MD = np.full((N, len(mnames)), np.nan, np.float32)
    BK = {b: np.full((N, m.NDIM), np.nan, np.float32) for b, m in mods.items()}
    ok = np.zeros(N, bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        ok[i] = True
        # uint8 counts: 2.0 GB at 1M instead of 8.2 GB as float32. Counts above 255 do not
        # occur for radius-2 environments in <=60-atom molecules.
        E[i] = np.clip(gen.GetCountFingerprintAsNumPy(m), 0, 255).astype(np.uint8)
        for j, n in enumerate(rnames):
            try:
                RD[i, j] = rlut[n](m)
            except Exception:
                pass
        try:
            MD[i] = np.array([v if isinstance(v, (int, float)) else np.nan
                              for v in calc(m)], np.float32)
        except Exception:
            pass
        for b, mod in mods.items():
            try:
                BK[b][i] = mod.featurize(m)
            except Exception:
                pass
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, smiles=np.array(smiles, dtype=object), ok=ok, ecfp=E, rdkit=RD,
                        mordred=MD, **{f"blk_{b}": v for b, v in BK.items()})
    return si, time.time() - t0, False


def _bench_smiles():
    """Benchmark SMILES, so the eval matrices are built by the identical code path.

    Building train and bench with two different scripts is how column misalignment happens:
    the CORE/PREDICT split changed today, and any independently-maintained benchmark builder
    would now be silently one convention behind. One code path removes the possibility.
    """
    d = np.load(BENCH, allow_pickle=True)
    return list(d["smiles"])


def cmd_compute(a) -> None:
    base = OUT.parent / "bench1m" if getattr(a, "bench", False) else OUT
    base.mkdir(parents=True, exist_ok=True)
    smiles = _bench_smiles() if getattr(a, "bench", False) else \
        (OUT / "selected.txt").read_text().split()
    shards = [(i, smiles[i * SHARD:(i + 1) * SHARD], str(base))
              for i in range((len(smiles) + SHARD - 1) // SHARD)]
    todo = [s for s in shards if not (base / "shards" / f"sh_{s[0]:05d}.npz").exists()]
    print(f"{len(smiles):,} molecules | {len(shards)} shards | {len(todo)} to do | "
          f"{a.workers} workers")
    _, mnames = _mordred_subset()
    print(f"  mordred subset: {len(mnames)} columns -> {base}")
    json.dump({"mordred_names": mnames}, open(base / "colnames.json", "w"))

    t0, done, dts = time.time(), 0, []
    with Pool(a.workers) as p:
        for si, dt, cached in p.imap_unordered(_shard, todo, chunksize=1):
            done += 1
            el = time.time() - t0
            if dt:
                dts.append(dt)
            # ETA from per-shard wall time divided by worker count, NOT elapsed/done. With W
            # workers the first W shards all complete at roughly the same moment, so
            # elapsed/done overestimates by ~W at the first completion -- it read "eta 519m"
            # on a run that actually had 80 minutes left.
            per = (sum(dts) / len(dts)) if dts else (el / max(done, 1))
            eta = per * (len(todo) - done) / a.workers
            print(f"  shard {si:05d} {'cached' if cached else f'{dt:.0f}s'} | "
                  f"{done}/{len(todo)} | elapsed {el / 60:.0f}m eta {eta / 60:.0f}m", flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min")


def cmd_pack(a) -> None:
    base = OUT.parent / "bench1m" if getattr(a, "bench", False) else OUT
    """Slice raw shards into training matrices and report what is actually usable.

    Stays sharded: X at 1M x 2,897 float32 is 11.6 GB, which does not want to be one array in
    25 GB of RAM alongside a training process. Each packed shard is self-contained.

    The NaN report is not decoration. 159 CORE columns already carried NaN at 100k; a column
    that is 40% NaN at 1M is a column the surrogate cannot learn and should be dropped before
    training rather than discovered in the loss.
    """
    import importlib
    import blocks
    from mordred import Calculator, descriptors as mdesc
    from rdkit.Chem import Descriptors

    fam = {str(x): type(x).__module__.split(".")[-1]
           for x in Calculator(mdesc, ignore_3D=True).descriptors}
    sp = blocks.split(fam)
    mnames = json.load(open(base / "colnames.json"))["mordred_names"]
    rnames = [n for n, _ in Descriptors._descList]
    mpos = {n: i for i, n in enumerate(mnames)}
    rpos = {n: i for i, n in enumerate(rnames)}
    mods = {b: importlib.import_module(b)
            for b in ("resistance", "cycles", "conjugation", "stereo", "chi")}

    def cols(items):
        out = []
        for s, n, _ in items:
            if s == "mordred" and n in mpos:
                out.append(("m", mpos[n], f"mordred:{n}"))
            elif s == "rdkit" and n in rpos:
                out.append(("r", rpos[n], f"rdkit:{n}"))
        return out

    core_c, pred_c = cols(sp["core"]), cols(sp["predict"])
    xnames = ([f"ecfp:{i}" for i in range(2048)] + [c[2] for c in core_c]
              + [f"{b}:{n}" for b, m in mods.items() for n in m.NAMES])
    ynames = [c[2] for c in pred_c]
    print(f"X = 2048 ecfp + {len(core_c)} core + "
          f"{sum(m.NDIM for m in mods.values())} block = {len(xnames)}")
    print(f"Y = {len(ynames)} predict")

    shards = sorted((base / "shards").glob("sh_*.npz"))
    (base / "packed").mkdir(parents=True, exist_ok=True)
    ysum = np.zeros(len(ynames))
    xsum = np.zeros(len(xnames))
    n_tot = 0
    for f in shards:
        z = np.load(f, allow_pickle=True)
        ok = z["ok"]
        MD, RD = z["mordred"], z["rdkit"]
        take = lambda cs: np.stack([(MD if k == "m" else RD)[:, i] for k, i, _ in cs], 1) \
            if cs else np.zeros((len(ok), 0), np.float32)
        X = np.hstack([np.log1p(z["ecfp"].astype(np.float32)), take(core_c)]
                      + [z[f"blk_{b}"] for b in mods]).astype(np.float32)
        Y = take(pred_c).astype(np.float32)
        X, Y = X[ok], Y[ok]
        n_tot += int(ok.sum())
        xsum += (~np.isfinite(X)).sum(0)
        ysum += (~np.isfinite(Y)).sum(0)
        np.savez_compressed(base / "packed" / f.name.replace("sh_", "pk_"),
                            X=X, Y=Y, smiles=z["smiles"][ok])
        print(f"  {f.name} -> {X.shape} ({n_tot:,} total)", flush=True)

    bad_x = [(xnames[i], xsum[i] / n_tot) for i in np.argsort(-xsum)[:15] if xsum[i]]
    bad_y = [(ynames[i], ysum[i] / n_tot) for i in np.argsort(-ysum)[:15] if ysum[i]]
    print(f"\n{n_tot:,} usable molecules")
    print(f"X columns with any NaN: {int((xsum > 0).sum())}/{len(xnames)}")
    for n, f in bad_x[:8]:
        print(f"    {n:42s} {100 * f:6.2f}%")
    print(f"Y columns with any NaN: {int((ysum > 0).sum())}/{len(ynames)}")
    for n, f in bad_y[:8]:
        print(f"    {n:42s} {100 * f:6.2f}%")
    json.dump({"n": n_tot, "xnames": xnames, "ynames": ynames,
               "x_nan": (xsum / n_tot).tolist(), "y_nan": (ysum / n_tot).tolist()},
              open(base / "meta.json", "w"))
    print(f"\nwrote {base / 'packed'} and {base / 'meta.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scaffolds"); s.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=1_000_000)
    s.add_argument("--acyclic-frac", type=float, default=0.02,
                   help="corpus fraction reserved for acyclic molecules (benchmark is 0.2%%)")
    s = sub.add_parser("compute")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--bench", action="store_true", help="build the benchmark set instead")
    s = sub.add_parser("pack")
    s.add_argument("--bench", action="store_true", help="pack the benchmark set instead")
    a = ap.parse_args()
    {"scaffolds": cmd_scaffolds, "select": cmd_select, "compute": cmd_compute,
     "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
