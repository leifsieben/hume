"""Are HUME's OWN columns -- the 193 with no RDKit or Mordred name -- redundant?

The 864 columns the dedupe keeps are all present in HUME, so HUME can audit itself: compute the
full 1,266 on a representative corpus and ask, for each of the 193, its maximum |Spearman rho|
against (a) the 864 borrowed columns and (b) the other 192 own columns.

Same criterion the original selection used -- ranks, so monotone rather than merely linear
redundancy, and the same 0.99 threshold -- applied to the half of the block that has never been
put through it.
"""
import json, sys, os, random
import numpy as np, hume
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "."); sys.path.insert(0, "/Users/lsieben/VSCode/ChemPFN")
os.environ.setdefault("CHEMPFN_DATA_ROOT", "/Users/lsieben/chempfn-data")

CH = "/Users/lsieben/VSCode/ChemTFM_OLD/data/corpus/chembl_150k.smi"
random.seed(0)
smis = random.sample([l.split()[0] for l in open(CH) if l.strip()], 20000)
try:
    import bench_downstream as BD
    for ds in ["aqsoldb","esol","lipophilicity","photoswitch","cycpept_pampa","ld50_zhu",
               "ames","bioavail","hia","pb_bbb","qm8","qm9","qmugs_gap","rascore"]:
        try: v = BD.load_ds(ds)["smiles"]; smis += random.sample(v, min(1000, len(v)))
        except Exception: pass
except Exception as e:
    print("lake unavailable:", type(e).__name__)
print(f"corpus {len(smis):,} molecules")

# CHUNKED, and float32. The un-chunked call died with no traceback on 32,610 molecules --
# the signature of an OOM kill, and HUME's non-streaming path has allocated 12.2 GB on a large
# batch before. The fingerprint is discarded immediately; only the descriptor block is needed.
CACHE = "data/own_audit_matrix.npz"
if os.path.exists(CACHE):
    _z = np.load(CACHE, allow_pickle=True)
    X = _z["X"].astype(np.float64); smis = list(_z["smis"])
    print(f"loaded cached matrix {X.shape}")
else:
    # CHUNKED, and float32. The un-chunked call died with no traceback on 32,610 molecules --
    # the signature of an OOM kill, and HUME's non-streaming path has allocated 12.2 GB on a
    # large batch before. The fingerprint is discarded immediately.
    parts = []
    for lo in range(0, len(smis), 2000):
        _fp, xb, _ = hume.featurize_all(smis[lo:lo + 2000], optional=("qed", "AvgIpc"))
        parts.append(np.asarray(xb, np.float32)); del _fp, xb
        print(f"  featurised {min(lo + 2000, len(smis)):,}/{len(smis):,}", flush=True)
    X = np.vstack(parts).astype(np.float64); del parts
    np.savez_compressed(CACHE, X=X.astype(np.float32), smis=np.array(smis, dtype=object))
print(f"computed {X.shape}")

cols = list(hume.ALL_COLUMNS)
print(f"computed {X.shape}")

from rdkit.Chem import Descriptors
union_lc = {n.lower() for n,_ in Descriptors._descList} | \
           {n.lower() for n in open('/tmp/mordred_names.txt').read().split('\n') if n}
own_i = [i for i,c in enumerate(cols) if c.lower() not in union_lc]
bor_i = [i for i,c in enumerate(cols) if c.lower() in union_lc]
print(f"own {len(own_i)}   borrowed {len(bor_i)}")

# same preprocessing as dedupe.py: usable filter, median impute, rank transform
fin = np.isfinite(X).mean(0)
with np.errstate(all="ignore"): sd = np.nanstd(X, 0)
usable = (fin >= 0.95) & (sd > 0)
print(f"usable: {usable.sum()} of {X.shape[1]}  (unusable own: "
      f"{sum(1 for i in own_i if not usable[i])})")
def ranks(M):
    M = np.where(np.isfinite(M), M, np.nan)
    med = np.nanmedian(M, 0); M = np.where(np.isnan(M), med, M)
    R = np.argsort(np.argsort(M, 0), 0).astype(np.float64)
    R -= R.mean(0); R /= (np.linalg.norm(R, axis=0) + 1e-12); return R
own_u = [i for i in own_i if usable[i]]; bor_u = [i for i in bor_i if usable[i]]
Ro, Rb = ranks(X[:, own_u]), ranks(X[:, bor_u])
Cob = np.abs(Ro.T @ Rb)          # own vs borrowed
Coo = np.abs(Ro.T @ Ro); np.fill_diagonal(Coo, 0.0)

rows = []
for k, i in enumerate(own_u):
    jb = int(np.argmax(Cob[k])); jo = int(np.argmax(Coo[k]))
    rows.append((cols[i], float(Cob[k, jb]), cols[bor_u[jb]], float(Coo[k, jo]), cols[own_u[jo]]))
rows.sort(key=lambda r: -max(r[1], r[3]))
red = [r for r in rows if max(r[1], r[3]) >= 0.99]
print(f"\n*** {len(red)} of {len(own_u)} own columns are >=0.99 redundant ***")
print(f"{'own column':<20}{'max|r| vs borrowed':>20}  partner{'':<14}{'max|r| vs own':>15}  partner")
for n, rb, pb, ro, po in rows[:28]:
    print(f"  {n:<18}{rb:>18.4f}  {pb:<20}{ro:>13.4f}  {po}")
json.dump([{"col":n,"max_r_borrowed":rb,"partner_borrowed":pb,"max_r_own":ro,"partner_own":po}
           for n,rb,pb,ro,po in rows], open("results/figures/own_column_audit.json","w"), indent=1)
print("\n-> results/figures/own_column_audit.json")
