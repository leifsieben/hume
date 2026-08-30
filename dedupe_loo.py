"""WHICH dropped descriptors actually matter on bioavail / hia?

    python dedupe_loo.py

Leave-one-out over 966 dropped columns is 966 refits per dataset. ADD-ONE-BACK answers the same
question far cheaper and in the more useful direction: rank the dropped columns by the gain they
attract when they ARE present, then put the top few back one at a time and watch the deficit
close. If Leif's hypothesis is right -- one pair the 0.99 threshold should not have merged -- the
curve jumps at k=1 or k=2 and then flattens.

For each candidate it also reports the correlation with its nearest KEPT column ON THIS DATASET.
The filter measured |r| on the dedupe corpus; if a pair is 0.995 there and 0.90 here, that is
precisely the failure mode and the number says so.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MORDRED_PY", ".venv-mordred/bin/python")
import numpy as np, bench_downstream as BD
from chempfn.eval.splits import scaffold_folds, train_test
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator as rfg
import xgboost as xgb

RD=[n for n,_ in Descriptors._descList]
MD=[n for n in open("/tmp/mordred_names.txt").read().split("\n") if n]
NAMES=RD+MD
K=json.load(open("data/dedupe.json"))
KEPT={n for _s,n,_c in K["compute"]}|{n for _s,n,_c in K["predict"]}
seen=set(); keep=[]
for i,n in enumerate(NAMES):
    if n in KEPT and n not in seen: seen.add(n); keep.append(i)
keep=np.array(keep); drop=np.array([i for i in range(len(NAMES)) if i not in set(keep.tolist())])
print(f"kept {len(keep)}  dropped {len(drop)}")

def ecfp3(smis):
    g=rfg.GetMorganGenerator(radius=3,fpSize=2048,includeChirality=True)
    o=np.zeros((len(smis),2048),np.float32)
    for i,s in enumerate(smis):
        m=Chem.MolFromSmiles(s)
        if m is not None: o[i]=g.GetFingerprintAsNumPy(m)
    return o

def fw(nc,w):
    v=np.ones(nc,np.float32); v[:2048]=w; return v

def cv_auc(X,y,folds):
    out=[]
    for i in range(5):
        tr,te=train_test(folds,i)
        m=xgb.XGBClassifier(tree_method="hist",colsample_bynode=0.3,
                            feature_weights=fw(X.shape[1],10.0),n_jobs=8,random_state=0)
        m.fit(X[tr],y[tr]); out.append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
    return float(np.mean(out))

for ds in ["bioavail","hia"]:
    BD._BLOCK_CACHE.clear()
    d=BD.load_ds(ds); smis,y=d["smiles"],np.asarray(d["y"],float)
    D=np.hstack([BD.f_rdkit_desc(smis),BD.f_mordred_desc(smis)]).astype(np.float32)
    D[~np.isfinite(D)]=np.nan
    F=ecfp3(smis); folds=scaffold_folds(smis,k=5,seed=0)
    Xk=np.hstack([F,D[:,keep]]); Xf=np.hstack([F,D])
    a_k, a_f = cv_auc(Xk,y,folds), cv_auc(Xf,y,folds)
    print(f"\n=== {ds}  n={len(smis)}   deduped {a_k:.4f}   full {a_f:.4f}   gap {a_f-a_k:+.4f}")

    # rank dropped columns by the gain they attract in the FULL model, summed over folds
    gain={}
    for i in range(5):
        tr,_=train_test(folds,i)
        m=xgb.XGBClassifier(tree_method="hist",colsample_bynode=0.3,
                            feature_weights=fw(Xf.shape[1],10.0),n_jobs=8,random_state=0)
        m.fit(Xf[tr],y[tr])
        for k_,v in m.get_booster().get_score(importance_type="gain").items():
            j=int(k_[1:])
            if j>=2048 and (j-2048) not in set(keep.tolist()):
                gain[j-2048]=gain.get(j-2048,0.0)+v
    top=sorted(gain.items(), key=lambda x:-x[1])[:8]
    # correlation of each candidate with its nearest KEPT column, ON THIS DATASET
    Dk=D[:,keep]
    print(f"  {'dropped column':<22}{'gain':>9}{'max |r| with a kept col here':>31}")
    for j,g in top:
        col=D[:,j]; ok=np.isfinite(col)
        best=0.0; bestn=""
        if ok.sum()>10:
            c=Dk[ok]; v=col[ok]
            good=np.isfinite(c).all(0) & (np.nanstd(c,0)>0)
            if v.std()>0 and good.any():
                r=np.abs(np.corrcoef(np.vstack([v, c[:,good].T]))[0,1:])
                r=np.nan_to_num(r); b=int(np.argmax(r)); best=float(r[b])
                bestn=NAMES[keep[np.where(good)[0][b]]]
        print(f"  {NAMES[j]:<22}{g:>9.1f}{best:>16.4f}  ({bestn})")

    # add the top-k back, one at a time
    print(f"  {'add back':<22}{'AUROC':>9}{'recovered':>12}")
    for k in (1,2,3,5,8):
        idx=np.array([j for j,_ in top[:k]])
        X=np.hstack([F, D[:,keep], D[:,idx]])
        a=cv_auc(X,y,folds)
        rec = 100*(a-a_k)/(a_f-a_k) if abs(a_f-a_k)>1e-9 else float("nan")
        print(f"  top-{k:<18}{a:>9.4f}{rec:>11.0f}%")
