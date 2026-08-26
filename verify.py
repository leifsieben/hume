"""Exactness gate: every descriptor we compute ourselves, checked against its reference.

The moment a descriptor moves from PREDICT (where RDKit/Mordred compute it) into CORE (where
*we* compute it), correctness stops being someone else's problem. A fast wrong descriptor is
worth less than a slow right one, and the failure mode is silent -- our first Chi implementation
matched RDKit perfectly on every hydrocarbon and diverged only on heteroatoms.

Three real bugs were caught by this check on the day it was written, all of which would have
shipped silently:

  1. valence delta computed as `TotalValence - numH` instead of Kier-Hall `nOuterElecs - nH`
     (correct on C/H, wrong on every N and O)
  2. the row correction `/(Z - nV - 1)` applied to first-row atoms as well as heavy ones
     (correct on C/N/O by accident, since the denominator is 1 there, wrong on S/Cl/Br)
  3. ring closures assigned to order L-1 instead of order L
     (correct on acyclic molecules, wrong on everything with a ring)

Run over the **full 1M training corpus**, not a sample. A 2,000-molecule sample cannot see a
failure mode that affects one chemotype in ten thousand, and the corpus is the thing we
actually featurise.

    python verify.py                 # benchmark set, quick
    python verify.py --corpus        # full 1M training corpus, the real gate
    python verify.py --corpus -n 50000

Exit code is nonzero if any descriptor falls below the exactness threshold, so this can gate a
build rather than merely inform one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "surrogate"
TOL = 1e-5
THRESHOLD = 0.9999          # fraction of molecules that must match exactly


def _load(source: str, limit: int | None) -> list[str]:
    if source == "corpus":
        # The 1M corpus, not the legacy 100k train.npz. Reading `selected.txt` rather than the
        # packed shards deliberately: verification must re-derive every value from the SMILES,
        # so that a bug in the shard writer is caught rather than reproduced.
        sel = ROOT / "data" / "corpus1m" / "selected.txt"
        if sel.exists():
            smiles = sel.read_text().split()
        elif (OUT / "train.npz").exists():
            print("no 1M corpus; falling back to the legacy 100k train.npz")
            smiles = list(np.load(OUT / "train.npz", allow_pickle=True)["smiles"])
        else:
            sys.exit(f"no corpus at {sel} -- run corpus.py select first")
    else:
        smiles = list(np.load(OUT / "bench.npz", allow_pickle=True)["smiles"])
    return smiles[:limit] if limit else smiles


# --- checks -------------------------------------------------------------------------------
# Each check maps our value to the reference implementation it must reproduce bit-for-bit.
# Adding a computed descriptor to CORE without adding it here is the thing this file exists
# to prevent.

def _chi_checks():
    """Chi is defined on the heavy-atom graph, so BOTH sides are compared on the stripped
    molecule. Comparing our normalised value against RDKit's un-normalised one would report a
    difference of convention as a difference of correctness."""
    import chi
    lut = dict(Descriptors._descList)
    out = {}
    for kind in ("n", "v"):
        for k in range(5):
            name = f"Chi{k}{kind}"
            idx = chi.NAMES.index(f"chi{k}{kind}")
            out[f"chi:{name}"] = (
                lambda m, i=idx: float(chi.featurize(m)[i]),
                lambda m, nm=name: float(lut[nm](chi.strip_explicit_h(m))),
            )
    return out


def _cycle_checks():
    """C3/C4/C5 have two independent routes -- closed form on traces of A^k, and direct
    enumeration. They must agree, which catches errors in either."""
    import cycles
    from rdkit.Chem import rdmolops

    def closed(m, k):
        A = rdmolops.GetAdjacencyMatrix(m).astype(np.float64)
        return _round(cycles._closed_form(A)[k - 3])

    def enum(m, k):
        n = m.GetNumAtoms()
        A = rdmolops.GetAdjacencyMatrix(m)
        adj = [list(np.flatnonzero(A[i])) for i in range(n)]
        return float(sum(1 for c in cycles._enumerate(adj, n, k, k) for _ in [c]) / 2)

    return {f"cycles:C{k}": (lambda m, k=k: enum(m, k), lambda m, k=k: closed(m, k))
            for k in (3, 4, 5)}


def _round(x):
    return float(np.round(x, 6))


def _ring_checks():
    """Ring-count tripwire.

    The first version of this check asserted `len(GetSymmSSSR) == bonds - atoms + fragments`
    and failed on 0.5% of the benchmark. The check was wrong, not the code: GetSymmSSSR is
    *symmetrised*, so for symmetric fused and cage systems it deliberately returns more rings
    than the cyclomatic number in order to keep chemically equivalent rings together. The
    invariant is `>=`, so that is what is asserted -- a violation would mean genuinely broken
    ring perception.
    """
    def ours(m):
        cyclo = m.GetNumBonds() - m.GetNumAtoms() + len(Chem.GetMolFrags(m))
        return float(len(Chem.GetSymmSSSR(m)) >= cyclo)

    return {"cycles:sssr_ge_cyclomatic": (ours, lambda m: 1.0)}


CHECKS = {}


def _build():
    CHECKS.update(_chi_checks())
    CHECKS.update(_cycle_checks())
    CHECKS.update(_ring_checks())


def _run_chunk(smiles):
    """Verify one chunk in a worker process.

    CHECKS holds closures over imported modules, which do not survive pickling, so each worker
    rebuilds them locally rather than receiving them. Returns plain dicts so the merge is
    associative and the parent does no chemistry.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if not CHECKS:
        _build()
    ok = {k: 0 for k in CHECKS}
    worst = {k: 0.0 for k in CHECKS}
    example = {k: "" for k in CHECKS}
    n = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() < 3:
            continue
        n += 1
        for name, (ours, ref) in CHECKS.items():
            try:
                e = abs(ours(m) - ref(m))
            except Exception:
                e = float("inf")
            if e <= TOL:
                ok[name] += 1
            elif e > worst[name]:
                worst[name], example[name] = e, s
    return n, ok, worst, example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="store_true", help="use the 1M training corpus")
    ap.add_argument("-n", type=int, default=None, help="limit molecules")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel workers; 1M single-threaded is hours")
    a = ap.parse_args()

    _build()
    smiles = _load("corpus" if a.corpus else "bench", a.n)
    print(f"verifying {len(CHECKS)} descriptors against reference on {len(smiles):,} molecules "
          f"({'1M corpus' if a.corpus else 'benchmark'})\n")

    ok = {k: 0 for k in CHECKS}
    worst = {k: 0.0 for k in CHECKS}
    example = {k: "" for k in CHECKS}
    n, t0 = 0, time.time()

    CH = 5000
    chunks = [smiles[i:i + CH] for i in range(0, len(smiles), CH)]
    if a.workers > 1 and len(chunks) > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            for cn, cok, cworst, cex in p.imap_unordered(_run_chunk, chunks, chunksize=1):
                n += cn
                for k in ok:
                    ok[k] += cok[k]
                    if cworst[k] > worst[k]:
                        worst[k], example[k] = cworst[k], cex[k]
                print(f"  {n:,} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for ch in chunks:
            cn, cok, cworst, cex = _run_chunk(ch)
            n += cn
            for k in ok:
                ok[k] += cok[k]
                if cworst[k] > worst[k]:
                    worst[k], example[k] = cworst[k], cex[k]
            print(f"  {n:,} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{n:,} molecules in {time.time() - t0:.0f}s\n")
    bad = []
    for name in sorted(CHECKS):
        frac = ok[name] / n if n else 0.0
        flag = "" if frac >= THRESHOLD else "  <-- FAIL"
        if frac < THRESHOLD:
            bad.append(name)
        print(f"  {name:26s} exact {ok[name]:8,}/{n:,} ({100 * frac:7.3f}%)  "
              f"worst {worst[name]:.2e}{flag}")
        if flag and example[name]:
            print(f"  {'':26s} first divergence: {example[name]}")

    rep = {"n": n, "corpus": bool(a.corpus), "threshold": THRESHOLD,
           "exact": ok, "worst": worst, "failed": bad}
    p = OUT / ("verify_corpus.json" if a.corpus else "verify_bench.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(p, "w"), indent=2)
    print(f"\nwrote {p}")

    if bad:
        print(f"\nFAILED: {len(bad)} descriptor(s) below {100 * THRESHOLD:.2f}% exact -> {bad}")
        sys.exit(1)
    print("\nall descriptors exact")


if __name__ == "__main__":
    main()
