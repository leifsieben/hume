"""Which descriptors fit inside an ECFP-sized time budget?

Rule of thumb under test: the whole descriptor suite should cost no more than generating the
fingerprint. Everything that fits gets COMPUTED; everything that does not is a candidate to
be PREDICTED (or dropped).

Two measurement subtleties this handles:

1. **Shared setup.** Many descriptors need the same intermediates (distance matrix, ring
   info, Gasteiger charges). Timing each alone over-states the marginal cost of adding one to
   a set that already computed them. So we report both the isolated cost and the *cumulative*
   cost of the cheapest-k set evaluated together, which is what you would actually pay.
2. **Parse is not free and may dominate.** `MolFromSmiles` is the real floor: you pay it
   before any fingerprint or descriptor. Reported separately so the budget is honest.

Also measures ECFP parallel scaling, since "as fast as ECFP" is only a meaningful budget if
we know what ECFP actually costs at scale.
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

CORPUS = "/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/chembl_150k.smi"
OUT = Path(__file__).resolve().parent / "data" / "budget_profile.json"
N = 300


def load(n):
    smis = []
    with open(CORPUS) as fh:
        for line in fh:
            s = line.split()[0] if line.strip() else ""
            if s:
                smis.append(s)
            if len(smis) >= n:
                break
    return smis


def _time(fn, mols, reps=1):
    fn(mols[0])
    t = time.time()
    for _ in range(reps):
        for m in mols:
            fn(m)
    return (time.time() - t) / (len(mols) * reps) * 1e6


def _ecfp_chunk(args):
    smis, radius, size = args
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size,
                                                    includeChirality=True)
    t = time.time()
    n = 0
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            gen.GetCountFingerprintAsNumPy(m)
            n += 1
    return n, time.time() - t


def main() -> None:
    smis = load(N)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    report = {}

    # ---- the floor: parsing -------------------------------------------------------------
    t = time.time()
    for s in smis:
        Chem.MolFromSmiles(s)
    parse = (time.time() - t) / len(smis) * 1e6
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    ecfp = _time(lambda m: gen.GetCountFingerprintAsNumPy(m), mols, reps=3)
    print(f"MolFromSmiles (parse)   {parse:8.1f} us/mol   <- paid before anything else")
    print(f"ECFP-2048 r2 counts     {ecfp:8.1f} us/mol   <- the budget")
    print(f"parse+ecfp total        {parse + ecfp:8.1f} us/mol  ({100 * parse / (parse + ecfp):.0f}% is parsing)\n")
    report["parse_us"] = parse
    report["ecfp_us"] = ecfp

    # ---- per-descriptor cost, isolated --------------------------------------------------
    rows = []
    for name, fn in Descriptors._descList:
        try:
            us = _time(lambda m, f=fn: f(m), mols)
        except Exception:
            continue
        rows.append((us, name))
    rows.sort()
    report["n_descriptors"] = len(rows)
    print(f"{len(rows)} RDKit descriptors timed individually")
    print("10 most expensive:")
    for us, name in rows[-10:][::-1]:
        print(f"   {name:28s} {us:9.1f} us")

    # ---- cumulative cost of the cheapest-k set, evaluated together ----------------------
    print("\ncheapest-k evaluated together (real cost, shared intermediates):")
    names = [n for _, n in rows]
    lut = dict(Descriptors._descList)
    cum = []
    for k in (25, 50, 75, 100, 125, 150, 175, len(rows)):
        fns = [lut[n] for n in names[:k]]

        def run(m, fns=fns):
            for f in fns:
                try:
                    f(m)
                except Exception:
                    pass
        us = _time(run, mols)
        cum.append({"k": k, "us": us})
        flag = "  <= within ECFP budget" if us <= ecfp else ""
        print(f"   cheapest {k:4d} descriptors  {us:9.1f} us/mol{flag}")
    report["cumulative"] = cum
    report["descriptors"] = [{"name": n, "us": u} for u, n in rows]

    # how many fit in the budget, measured cumulatively
    fit = max((c["k"] for c in cum if c["us"] <= ecfp), default=0)
    print(f"\n-> at most ~{fit} of {len(rows)} RDKit descriptors fit inside the {ecfp:.0f} us ECFP budget")

    # ---- ECFP parallel scaling ----------------------------------------------------------
    print("\nECFP+parse throughput scaling (process pool):")
    big = load(24000)
    scal = []
    for w in (1, 2, 4, 6, 8, 10, 12):
        chunks = [(big[i::w], 2, 2048) for i in range(w)]
        t0 = time.time()
        with Pool(w) as pool:
            res = pool.map(_ecfp_chunk, chunks)
        wall = time.time() - t0
        n = sum(r[0] for r in res)
        scal.append({"workers": w, "mol_per_s": n / wall, "us_per_mol": wall / n * 1e6})
        print(f"   {w:2d} workers  {n / wall:9.0f} mol/s  {wall / n * 1e6:7.1f} us/mol  "
              f"speedup {(n / wall) / scal[0]['mol_per_s']:4.1f}x")
    report["scaling"] = scal
    best = max(scal, key=lambda r: r["mol_per_s"])
    print(f"\n-> peak {best['mol_per_s']:.0f} mol/s; 1B molecules = "
          f"{1e9 / best['mol_per_s'] / 3600:.1f} core-hours wall at {best['workers']} workers")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
