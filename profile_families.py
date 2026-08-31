"""Per-family cost of the descriptor block, on the stratified 20,000-molecule corpus.

Timed by DIFFERENCE against a baseline that computes only `blocks` (which the C++ makes
mandatory), so each figure is the MARGINAL cost of adding that family to a run that has already
paid for pickle extraction and the shared primitives. That is the number a reader wants -- the
alternative, timing each family alone, charges every one of them the full setup and sums to far
more than the whole.

Reported with the SD over repetitions. RDKit pickle extraction and the Morgan fingerprint are
timed separately because they are not descriptor families and are paid regardless.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
import molhume as hume
from molhume._extract import extract_pickles
import molhume._core as core
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
RDLogger.DisableLog("rdApp.*")

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
smis = json.load(open("data/dedupe2/corpus.json"))["smiles"][:N]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
print(f"{len(mols):,} molecules, {len(hume.ALL_COLUMNS)} columns, {REPS} reps\n")

FAMS = ["vsa", "estate", "ringcount", "pathcount", "topocharge", "infocontent",
        "autocorr", "frag", "chi", "topomisc", "constit", "rdkcore"]
B = 2000

def run(fams, reps=REPS):
    out = []
    for _ in range(reps):
        t = 0.0
        for lo in range(0, len(mols), B):
            ch = mols[lo:lo + B]
            p = extract_pickles(ch)
            t0 = time.perf_counter()
            core.all_from_pickles(p.blobs, p.rings.ring_moff, p.rings.ring_ptr, p.rings.ring_at,
                                  p.h_blobs, p.stereo_a, p.stereo_b, families=fams)
            t += time.perf_counter() - t0
        out.append(t / len(mols) * 1e6)
    return np.array(out)

# non-descriptor overheads, paid regardless
ext = []
for _ in range(REPS):
    t0 = time.perf_counter()
    for lo in range(0, len(mols), B): extract_pickles(mols[lo:lo + B])
    ext.append((time.perf_counter() - t0) / len(mols) * 1e6)
gen = rfg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)
fpt = []
for _ in range(REPS):
    t0 = time.perf_counter()
    for m in mols: gen.GetFingerprintAsNumPy(m)
    fpt.append((time.perf_counter() - t0) / len(mols) * 1e6)

base = run([])                       # `blocks` only -- the C++ forces it
full = run(FAMS)
print(f"{'step':<26}{'us/mol':>10}{'SD':>8}   {'% of block':>11}")
print(f"{'RDKit pickle extraction':<26}{np.mean(ext):>10.1f}{np.std(ext):>8.1f}   {'(not a family)':>11}")
print(f"{'Morgan fingerprint':<26}{np.mean(fpt):>10.1f}{np.std(fpt):>8.1f}   {'(not a family)':>11}")
print(f"{'-'*56}")
print(f"{'blocks (mandatory)':<26}{base.mean():>10.1f}{base.std():>8.1f}   {100*base.mean()/full.mean():>10.1f}%")
rows = []
for f in FAMS:
    d = run(FAMS) - run([x for x in FAMS if x != f])   # leave-one-out, marginal
    rows.append((f, d.mean(), d.std()))
for f, m, s in sorted(rows, key=lambda r: -r[1]):
    print(f"{f:<26}{m:>10.1f}{s:>8.1f}   {100*m/full.mean():>10.1f}%")
print(f"{'-'*56}")
print(f"{'ALL FAMILIES':<26}{full.mean():>10.1f}{full.std():>8.1f}")
print(f"{'+ extraction + fingerprint':<26}{full.mean()+np.mean(ext)+np.mean(fpt):>10.1f}")
