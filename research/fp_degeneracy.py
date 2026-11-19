"""Degeneracy: how often do two DIFFERENT molecules get the SAME vector?

    .venv/bin/python research/fp_degeneracy.py [--limit N] [--reps a,b,c]

FOLLOWS Huber & Pollmann, J. Chem. Inf. Model-- no, Journal of Cheminformatics 2026, 18:83,
"Count your bits", section "Fingerprint duplicates" (their Figure 2). Their test, exactly:
take a large set of unique compounds, find every group of distinct molecules that map to an
IDENTICAL fingerprint, and report (a) how many compounds sit in such a group and (b) the
MAXIMUM MASS DIFFERENCE within each group, binned. The second half is the interesting half: a
duplicate spanning 0-1 Da is a stereoisomer or tautomer the representation was never going to
separate, while one spanning >400 Da means the representation called two completely different
molecules identical.

THE DATASET IS NOT THE ONE THEIR FIGURE 2 COUNTS, AND THE NUMBERS ARE THEREFORE NOT DIRECTLY
COMPARABLE. Figure 2 is over 718,067 compounds. Zenodo 18682051 publishes
`biostructures_with_classyfire_results.csv`, which is the ClassyFire-annotated SUBSET -- the
paper says 695,152 compounds were annotated, and the file holds 695,122 rows, the difference
being the 30 entries RDKit could not convert. So every count here is over ~96.8% of their set.
That is why this script RECOMPUTES reference fingerprints on its own input rather than quoting
their numbers: the comparison that matters is internal, same molecules, same duplicate-counting
code, and their published values are used only to check that this harness reproduces them.

WHAT "IDENTICAL" MEANS FOR A FLOAT VECTOR. Bitwise equality of the row's bytes. That is the
honest analogue of bit-for-bit fingerprint equality, it is deterministic for a deterministic
featurizer, and it handles NaN correctly where `==` would not (NaN != NaN, but NaN's bit pattern
equals its own). It also means this test CANNOT be gamed by floating-point noise in either
direction.

A 1,269-dimensional float vector has vastly more capacity than a 4,096-bit binary one, so HUME
having fewer duplicates than MACCS is not a finding -- it is arithmetic. Two things here are
findings: whether HUME has ANY duplicates at all, and what mass range they span. The paper's own
element-count baseline is the right comparison for that: 617,659 duplicated compounds, but
nearly all within 1 Da, i.e. a representation that collides constantly yet never confuses two
genuinely different molecules.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "fp_degeneracy" / "biostructures.csv"
#: The paper's own bin edges for Figure 2, in daltons.
BINS = [(0, 1), (1, 10), (10, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, np.inf)]
#: Figure 2, read off the published bars. Used ONLY as a sanity check on this harness, never as
#: a number this script reports. Their set is 718,067; ours is ~695,122, so expect ~96.8%.
PAPER = {"morgan3_count": 64899, "morgan3_binary": 84864, "morgan9_count": 21258,
         "maccs_binary": 309279, "rdkit_binary": 128578, "atompair_count": 7007,
         "map4_count": 223}


def say(*a) -> None:
    print(*a, flush=True)


def rows_hashes(mat: np.ndarray) -> list[bytes]:
    """-> one 16-byte digest per row, over the row's raw bytes.

    C-contiguity is forced because `tobytes()` on a non-contiguous view silently serialises in
    a different order, which would make two runs of the same data disagree.
    """
    m = np.ascontiguousarray(mat)
    return [hashlib.blake2b(m[i].tobytes(), digest_size=16).digest() for i in range(m.shape[0])]


def report(name: str, digests: list[bytes], masses: np.ndarray) -> dict:
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, h in enumerate(digests):
        groups[h].append(i)
    dup = [ix for ix in groups.values() if len(ix) > 1]
    n_dup = sum(len(ix) for ix in dup)
    hist = [0] * len(BINS)
    worst = 0.0
    for ix in dup:
        mm = masses[ix]
        d = float(mm.max() - mm.min())
        worst = max(worst, d)
        for b, (lo, hi) in enumerate(BINS):
            if lo <= d < hi:
                hist[b] += len(ix)
                break
    return {"name": name, "n_compounds": len(digests), "n_duplicated": n_dup,
            "n_groups": len(dup), "max_mass_diff": worst, "hist": hist,
            "frac": n_dup / max(len(digests), 1)}


# ---------------------------------------------------------------------------------------------
# REPRESENTATIONS. Each returns a matrix for a chunk of mols, hashed row-wise by the caller.
#
# HUME's descriptor block and its fingerprint are kept SEPARATE and tested separately, because
# bundling them answers the wrong question. `hume_full` contains a 2,048-bit radius-4 Morgan
# fingerprint, so its specificity is bounded below by that fingerprint's and a good score would
# say nothing about the descriptors. `hume_desc*` is the real test: a float descriptor vector
# competing against fingerprints on their own benchmark.
# ---------------------------------------------------------------------------------------------
#: One HUME featurization per chunk, shared by every variant. Without this each of the six
#: hume_* representations called featurize_all_from_mols on the same molecules, so the single
#: most expensive step in the script ran six times for one result -- which is what made the
#: first full run crawl. Keyed by id() of the chunk list and cleared by the caller, so a stale
#: entry cannot outlive the chunk it describes.
_HCACHE: dict = {}


def _hume(mols, columns):
    import molhume
    key = id(mols)
    if key not in _HCACHE:
        _HCACHE.clear()
        _HCACHE[key] = molhume.featurize_all_from_mols(mols, optional=("AvgIpc",))
    fp, X, cols = _HCACHE[key]
    want = set(molhume.column_set(columns))
    mask = np.array([c in want for c in cols], dtype=bool)
    if mask.sum() != len(want):
        raise RuntimeError(f"column_set({columns!r}) names {len(want)} columns but only "
                           f"{mask.sum()} are in this build's {len(cols)}-column output.")
    return fp, X[:, mask]


def make_reps(names):
    from rdkit.Chem import rdFingerprintGenerator as fg, MACCSkeys
    reps = {}
    if "hume_desc" in names:
        reps["hume_desc"] = lambda m: _hume(m, "full")[1]
    if "hume_desc_622" in names:
        reps["hume_desc_622"] = lambda m: _hume(m, "default")[1]
    if "hume_desc_408" in names:
        reps["hume_desc_408"] = lambda m: _hume(m, "small")[1]
    if "hume_desc_256" in names:
        reps["hume_desc_256"] = lambda m: _hume(m, "minimal")[1]
    if "hume_full" in names:
        def _full(m):
            fp, X = _hume(m, "full")
            return np.hstack([X, fp.astype(np.float64)])
        reps["hume_full"] = _full
    if "hume_fp_only" in names:
        reps["hume_fp_only"] = lambda m: _hume(m, "full")[0]
    # --- references, recomputed on this same input set -------------------------------------
    # 4,096 bits is the paper's default for every non-dictionary fingerprint.
    g3 = fg.GetMorganGenerator(radius=3, fpSize=4096)
    g9 = fg.GetMorganGenerator(radius=9, fpSize=4096)
    gr = fg.GetRDKitFPGenerator(fpSize=4096)
    ga = fg.GetAtomPairGenerator(fpSize=4096)
    ref = {
        "morgan3_count":  lambda m: np.array([g3.GetCountFingerprintAsNumPy(x) for x in m]),
        "morgan3_binary": lambda m: np.array([g3.GetFingerprintAsNumPy(x) for x in m]),
        "morgan9_count":  lambda m: np.array([g9.GetCountFingerprintAsNumPy(x) for x in m]),
        "rdkit_binary":   lambda m: np.array([gr.GetFingerprintAsNumPy(x) for x in m]),
        "atompair_count": lambda m: np.array([ga.GetCountFingerprintAsNumPy(x) for x in m]),
        "maccs_binary":   lambda m: np.array([np.array(MACCSkeys.GenMACCSKeys(x)) for x in m]),
    }
    for k, v in ref.items():
        if k in names:
            reps[k] = v
    missing = [n for n in names if n not in reps]
    if missing:
        sys.exit(f"unknown representation(s) {missing}. Available: hume_desc, hume_desc_622, "
                 f"hume_desc_408, hume_desc_256, hume_full, hume_fp_only, {', '.join(ref)}")
    return reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--limit", type=int, default=0, help="pilot on the first N compounds")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--reps", default="hume_desc,hume_full,morgan3_count,morgan3_binary,"
                                      "morgan9_count,rdkit_binary,atompair_count,maccs_binary")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "fp_degeneracy.json")
    a = ap.parse_args()
    if not a.csv.exists():
        sys.exit(f"{a.csv} is missing. Fetch it:\n"
                 f"  curl -L -o {a.csv} 'https://zenodo.org/records/18682051/files/"
                 f"biostructures_with_classyfire_results.csv?download=1'")

    import csv as _csv
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    # DEDUPLICATED BY INCHIKEY, WHICH IS THE DATASET'S OWN NOTION OF A COMPOUND. The file holds
    # 695,122 rows and 695,118 distinct keys: four compounds appear twice, written two ways
    # ([AsH3] / [AsH5], S / [SH4], OP=O / O[PH2]=O, and one more). They are the same compound
    # under InChI, so a shared vector between them is not a fingerprint duplicate -- it is the
    # representation being correct. Four rows is nothing against MACCS's hundreds of thousands,
    # but HUME's expected count here is near zero, so left in they could have BEEN the signal.
    smis, keys, seen, dropped = [], [], set(), 0
    with open(a.csv, newline="") as fh:
        for r in _csv.DictReader(fh):
            if r["inchikey"] in seen:
                dropped += 1
                continue
            seen.add(r["inchikey"])
            smis.append(r["SMILES"]); keys.append(r["inchikey"])
            if a.limit and len(smis) >= a.limit:
                break
    say(f"{len(smis):,} unique compounds from {a.csv.name}"
        + (f"  ({dropped} repeated inchikeys dropped)" if dropped else ""))

    names = [n.strip() for n in a.reps.split(",") if n.strip()]
    reps = make_reps(names)
    digests = {n: [] for n in names}
    masses, kept, skipped_light = [], 0, 0
    for s in range(0, len(smis), a.chunk):
        block = smis[s:s + a.chunk]
        mols = [Chem.MolFromSmiles(x) for x in block]
        # ZERO-HEAVY-ATOM "MOLECULES" ARE DROPPED FROM EVERY REPRESENTATION. The set contains
        # exactly three -- [H-], [HH], [H] -- and molhume refuses them by design, with the
        # message "eta: atom 0 is a hydrogen with no bonds; mordred's ETA_epsilon_5 indexes
        # GetNeighbors()[0]". RDKit's fingerprints accept them happily. Keeping them would
        # therefore score HUME on 695,115 molecules and the references on 695,118, and a
        # like-for-like duplicate count is the entire point of recomputing the references here.
        # Dropping three compounds out of 695,118 cannot move any number in this table; scoring
        # two representations on different sets silently could.
        ok = [m for m in mols if m is not None and m.GetNumHeavyAtoms() > 0]
        skipped_light += sum(1 for m in mols if m is not None and m.GetNumHeavyAtoms() == 0)
        _HCACHE.clear()
        kept += len(ok)
        masses.extend(Descriptors.MolWt(m) for m in ok)
        for n, fn in reps.items():
            try:
                digests[n].extend(rows_hashes(fn(ok)))
            except Exception as e:
                # FALL BACK TO ONE MOLECULE AT A TIME TO NAME THE CULPRIT. A representation that
                # raises on one input used to kill the run after a quarter of a million rows and
                # report only the exception, leaving the offending SMILES to be hunted by hand.
                for m in ok:
                    try:
                        fn([m])
                    except Exception:
                        sys.exit(f"{n} failed on {Chem.MolToSmiles(m)!r} "
                                 f"({m.GetNumHeavyAtoms()} heavy atoms): {e}")
                raise
        say(f"  {min(s + a.chunk, len(smis)):>8,} / {len(smis):,}")
    masses = np.array(masses)
    say(f"\nscored {kept:,} of {len(smis):,}: "
        f"{len(smis)-kept-skipped_light:,} RDKit failures, "
        f"{skipped_light:,} with no heavy atom\n")

    import json
    out = {"dataset": a.csv.name, "n_input": len(smis), "n_parsed": kept,
       "n_inchikey_dupes_dropped": dropped,
       "n_no_heavy_atom_dropped": skipped_light,
           "note": "counts are over the ClassyFire-annotated subset, NOT the 718,067 of the "
                   "paper's Fig 2; see this script's docstring",
           "bins_da": [[lo, None if hi == np.inf else hi] for lo, hi in BINS], "reps": {}}
    hdr = f"{'representation':<18}{'duplicated':>12}{'% of set':>10}{'groups':>9}{'max dDa':>10}  "
    say(hdr + "  ".join(f"{lo}-{'inf' if hi == np.inf else hi}" for lo, hi in BINS))
    say("-" * (len(hdr) + 46))
    for n in names:
        r = report(n, digests[n], masses)
        out["reps"][n] = r
        say(f"{n:<18}{r['n_duplicated']:>12,}{r['frac']*100:>9.2f}%{r['n_groups']:>9,}"
            f"{r['max_mass_diff']:>10.1f}  " + "  ".join(f"{h:>6,}" for h in r["hist"]))
        if n in PAPER:
            say(f"{'':<18}  paper Fig 2: {PAPER[n]:,} over 718,067 "
                f"({PAPER[n]/718067*100:.2f}%) -- this harness: {r['frac']*100:.2f}%")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    say(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
