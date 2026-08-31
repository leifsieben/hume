"""Do the Barysz VR* pairs separate on HETEROATOM-rich molecules?

The main run stratified by SIZE, but the Barysz matrix exists to inject heteroatom information
via a per-path Z-scaling. Size is the wrong axis to test it on, so this re-partitions the SAME
20,000-molecule matrix by heteroatom fraction and asks the question again. No recompute.

If |rho| stays >= 0.99 in the most heteroatom-rich decile -- where the Z-scaling has the most to
say -- the weighting genuinely contributes nothing and the pair is redundant. If it falls, the
main run absorbed a real distinction because it never looked along this axis.
"""
import json, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

z = np.load("data/dedupe2/matrix.npz", allow_pickle=True)
X = np.hstack([z["RD"], z["MD"], z["OW"]]).astype(np.float64)
names = list(z["rd_names"]) + list(z["md_names"]) + list(z["own_names"])
ix = {n: i for i, n in enumerate(names)}
smis = json.load(open("data/dedupe2/corpus.json"))["smiles"]
assert len(smis) == X.shape[0]

het, nheavy = np.zeros(len(smis)), np.zeros(len(smis))
for i, s in enumerate(smis):
    m = Chem.MolFromSmiles(s)
    if m is None: het[i] = np.nan; continue
    n = m.GetNumHeavyAtoms()
    nheavy[i] = n
    het[i] = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6)) / max(n, 1)

def srho(a, b, rows):
    x, y = X[rows, ix[a]], X[rows, ix[b]]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30: return np.nan
    xr = np.argsort(np.argsort(x[m])).astype(float)
    yr = np.argsort(np.argsort(y[m])).astype(float)
    if xr.std() == 0 or yr.std() == 0: return np.nan
    return abs(np.corrcoef(xr, yr)[0, 1])

PAIRS = [("VR1_D", "VR1_DzZ"), ("VR2_D", "VR2_DzZ"), ("VR3_D", "VR3_Dzv")]
ok = np.isfinite(het)
qs = np.nanquantile(het[ok], [0, .2, .4, .6, .8, 1.0])
print(f"heteroatom fraction over {int(ok.sum()):,} molecules: "
      f"min {qs[0]:.3f}  median {np.nanmedian(het[ok]):.3f}  max {qs[-1]:.3f}")
print(f"\n{'pair':<24}" + "".join(f"{f'Q{k+1}':>9}" for k in range(5)) + f"{'top decile':>12}{'MIN':>9}")
worst = {}
for a, b in PAIRS:
    row = f"{a+' / '+b:<24}"
    vals = []
    for k in range(5):
        rows = np.where(ok & (het >= qs[k]) & (het <= qs[k+1]))[0]
        v = srho(a, b, rows); vals.append(v); row += f"{v:>9.4f}"
    top = np.where(ok & (het >= np.nanquantile(het[ok], 0.9)))[0]
    vt = srho(a, b, top); vals.append(vt)
    row += f"{vt:>12.4f}{np.nanmin(vals):>9.4f}"
    worst[(a, b)] = float(np.nanmin(vals))
    print(row)
print()
for (a, b), v in worst.items():
    verdict = "REDUNDANT even where Barysz should matter" if v >= 0.99 else \
              "SEPARATES -- the size-stratified run absorbed a real distinction"
    print(f"  {a} / {b}: min |rho| = {v:.4f}  ->  {verdict}")
json.dump({f"{a}|{b}": v for (a, b), v in worst.items()},
          open("data/dedupe2/barysz_heteroatom_check.json", "w"), indent=1)
