"""Acquire the three missing task families for Figure B, and audit them for leakage.

`data/surrogate/bench.npz` holds 34 datasets and every one of them is drug-like
potency/property **regression** (30 MoleculeACE congeneric series + 4 MoleculeNet
regressions). A headline claim about descriptors vs learned embeddings that rests on one
task family is a claim a referee will discount. This script adds the three families that
were missing:

    qm          QM HOMO-LUMO gap regression       (QM9, QMugs, QM8, photoswitch)
    admet_cls   ADMET binary classification       (TDC ADMET + Tox21)
    vs          virtual screening / hit retrieval (MUV, LIT-PCBA)

SOURCE
------
Everything is **copied** out of the local ChemPFN data lake at ``~/chempfn-data`` — nothing is
downloaded. That lake was ingested 2026-08-09 from the primary sources (MoleculeNet, TDC,
QMugs, LIT-PCBA), unit-checked against ``chempfn/eval/protocol.py``, and it is what the rest of
this project's benchmark grid already reads. A fresh download would produce a second, unverified
copy that could silently disagree with numbers already in flight.

The lake is treated as **read-only**: files are copied into ``data/tasks/raw/`` with their
sha256 recorded, and everything downstream reads the local copy.

⚠️ **DEV ONLY.** ChemPFN gates 16 LOCKED datasets behind a frozen-model hash and a one-shot
evaluation ledger, and this repo's own ``TASKS.md`` says "LOCKED IS OFF LIMITS UNTIL HUME IS
FROZEN". Nothing under ``eval/locked/`` is copied or scored. ``--audit-locked`` reads LOCKED
*structures* for the two audits that need them (the MoleculeACE curation comparison and the
LOCKED-candidate inventory) and writes them to a separate, clearly-marked file.

LEAKAGE IS THE POINT
--------------------
A Figure B test molecule that sits in the surrogate's training corpus hands the HUME-predicted
arms an advantage no other arm gets, and the comparison stops meaning anything. Every new
molecule is therefore checked against four reference sets: the existing benchmark, the 100k
UMA target set, the 1M pretraining corpus, and ``cpp/hard.smi``.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not featurise (that needs Mordred and belongs in ``assemble.py``) and it does not
evaluate. Acquisition, copying and validation only.

SCHEMA — see ``data/benchmark_tasks.md`` for the full contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10**9)

ROOT = Path(__file__).resolve().parent
LAKE = Path(os.environ.get("CHEMPFN_DATA", Path.home() / "chempfn-data"))
DEV = LAKE / "eval" / "dev"
LOCKED = LAKE / "eval" / "locked"
OUT = ROOT / "data" / "tasks"
RAW = OUT / "raw"
BENCH = ROOT / "data" / "surrogate" / "bench.npz"

HARTREE_EV = 27.211386245988

# Source files copied out of the lake. Local name -> lake-relative path.
SOURCES = {
    "qm9.csv": "eval/dev/qm9/qm9.csv",
    "qmugs_gap.csv": "eval/dev/qmugs_gap/qmugs_gap.csv",
    "qm8.csv": "eval/dev/qm8/qm8.csv",
    "photoswitch.csv": "eval/dev/photoswitch/photoswitch.csv",
    "ames.csv": "eval/dev/ames/ames.csv",
    "cyp2d6_inh.csv": "eval/dev/cyp2d6_inh/cyp2d6_inh.csv",
    "bioavail.csv": "eval/dev/bioavail/bioavail.csv",
    "hia.csv": "eval/dev/hia/hia.csv",
    "tox21.csv": "eval/dev/moleculenet_legacy/tox21.csv",
    "muv.csv": "eval/dev/moleculenet_legacy/muv.csv",
    "litpcba.csv": "eval/dev/litpcba/litpcba.csv",
    "pb_bbb_cls.csv": "eval/dev/pharmabench/bbb_cls.csv",
    "pb_ames_cls.csv": "eval/dev/pharmabench/ames_cls.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_sources() -> dict:
    """Copy every source file out of the read-only lake, recording sha256 on both sides.

    Hashing the source and the copy separately is the point: a matching pair proves the copy
    is faithful, and the source hash is what a future reader needs to tell whether the lake
    has since changed underneath us."""
    RAW.mkdir(parents=True, exist_ok=True)
    prov, stamp = {}, time.strftime("%Y-%m-%d %H:%M")
    for local, rel in SOURCES.items():
        src = LAKE / rel
        dst = RAW / local
        if not src.exists():
            print(f"  MISSING {rel}")
            continue
        src_hash = sha256(src)
        if not (dst.exists() and sha256(dst) == src_hash):
            shutil.copy2(src, dst)
        prov[local] = dict(lake_path=str(src), sha256=src_hash,
                           bytes=src.stat().st_size, copied=stamp)
        print(f"  {local:22s} {src.stat().st_size / 1e6:8.1f} MB  {src_hash[:16]}")
    return prov


# ---------------------------------------------------------------------------- CSV reading
def read_csv(path: Path):
    """Header + non-blank data rows. MoleculeNet's legacy dumps carry an interleaved blank
    line per record (MUV: 93,087 real rows, 186,175 physical), so blanks are dropped rather
    than allowed to become 93,088 unparseable molecules."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = [row for row in r if row and any(c != "" for c in row)]
    return hdr, rows


def col(hdr, name):
    return hdr.index(name)


# ------------------------------------------------------------------------------- loaders
def load_qm():
    """QM HOMO-LUMO gap. QM9/QMugs gaps are in Hartree at source; converted to **eV** so the
    two are on one scale and RMSE is readable. QM8 is an excitation energy, not a gap — kept
    as family breadth and labelled as such."""
    out = []

    hdr, rows = read_csv(RAW / "qm9.csv")
    s, g = col(hdr, "smiles"), col(hdr, "gap")
    out.append(dict(
        name="qm9_gap", suite="qm9", family="qm", task="regression",
        smiles=[r[s] for r in rows], y=[float(r[g]) * HARTREE_EV for r in rows],
        unit="eV", direction=1, metric="rmse",
        source="MoleculeNet QM9 (Ramakrishnan et al., Sci. Data 1:140022, 2014); "
               "B3LYP/6-31G(2df,p) HOMO-LUMO gap, Hartree -> eV",
    ))

    hdr, rows = read_csv(RAW / "qmugs_gap.csv")
    s, g = col(hdr, "Drug"), col(hdr, "Y")
    out.append(dict(
        name="qmugs_gap", suite="qmugs", family="qm", task="regression",
        smiles=[r[s] for r in rows], y=[float(r[g]) * HARTREE_EV for r in rows],
        unit="eV", direction=1, metric="rmse",
        source="QMugs (Isert et al., Sci. Data 9:273, 2022); HOMO-LUMO gap on drug-like "
               "ChEMBL molecules, Hartree -> eV",
    ))

    hdr, rows = read_csv(RAW / "qm8.csv")
    s, e = col(hdr, "smiles"), col(hdr, "E1-CC2")
    out.append(dict(
        name="qm8_e1cc2", suite="qm8", family="qm", task="regression",
        smiles=[r[s] for r in rows], y=[float(r[e]) * HARTREE_EV for r in rows],
        unit="eV", direction=1, metric="rmse",
        source="MoleculeNet QM8 (Ramakrishnan et al., J. Chem. Phys. 143:084111, 2015); "
               "first singlet excitation energy E1-CC2, Hartree -> eV. NOT a ground-state gap.",
    ))

    hdr, rows = read_csv(RAW / "photoswitch.csv")
    s, y = col(hdr, "Drug"), col(hdr, "Y")
    out.append(dict(
        name="photoswitch", suite="photoswitch", family="qm", task="regression",
        smiles=[r[s] for r in rows], y=[float(r[y]) for r in rows],
        unit="nm", direction=1, metric="rmse",
        source="Photoswitch dataset (Griffiths et al., Chem. Sci. 13:13541, 2022); "
               "experimental E-isomer pi-pi* transition wavelength. Experimental, not computed.",
    ))
    return out


def load_admet_cls():
    """Binary ADMET. TDC single-endpoint sets + the 12 Tox21 assays.

    Deliberately excluded, with reasons recorded in ``data/benchmark_tasks.md``:
      hia          burned as this project's harness smoke test
      pb_bbb_cls   76% of LOCKED `adme_bbb` sits inside it
      pb_ames_cls  duplicate endpoint against TDC `ames`
      adme_bbb     LOCKED
      herg         LOCKED
    Their files are still copied and inventoried so the choice can be revisited without
    going back to the lake.
    """
    out = []
    for name, fn, letter, src in [
        ("ames", "ames.csv", "T",
         "TDC `AMES` (Xu et al., J. Chem. Inf. Model. 52:2840, 2012); "
         "Ames mutagenicity, 1 = mutagenic"),
        ("cyp2d6_inh", "cyp2d6_inh.csv", "M",
         "TDC `CYP2D6_Veith` (Veith et al., Nat. Biotechnol. 27:1050, 2009); 1 = inhibitor"),
        ("bioavail", "bioavail.csv", "A",
         "TDC `Bioavailability_Ma` (Ma et al., J. Pharm. Sci. 97:2861, 2008); "
         "1 = oral bioavailability >= 20%"),
    ]:
        hdr, rows = read_csv(RAW / fn)
        s, y = col(hdr, "Drug"), col(hdr, "Y")
        out.append(dict(
            name=name, suite="tdc_admet", family="admet_cls", task="binary",
            smiles=[r[s] for r in rows], y=[float(r[y]) for r in rows],
            unit=f"binary ({letter} of ADMET)", direction=1, metric="auprc", source=src,
        ))

    hdr, rows = read_csv(RAW / "tox21.csv")
    si = col(hdr, "smiles")
    for a in [h for h in hdr if h not in ("mol_id", "smiles")]:
        ai = col(hdr, a)
        keep = [r for r in rows if r[ai] != ""]
        out.append(dict(
            name=f"tox21_{a}", suite="tox21", family="admet_cls", task="binary",
            smiles=[r[si] for r in keep], y=[float(r[ai]) for r in keep],
            unit="binary (T of ADMET)", direction=1, metric="auprc",
            source=f"MoleculeNet Tox21, assay {a}; sparse label matrix, "
                   f"missing cells dropped per assay",
        ))
    return out


def load_vs(skip_litpcba: bool = False):
    """Virtual screening. Two libraries, both scored by early recognition, never accuracy."""
    out = []

    hdr, rows = read_csv(RAW / "muv.csv")
    si = col(hdr, "smiles")
    for t in [h for h in hdr if h.startswith("MUV-")]:
        ti = col(hdr, t)
        keep = [r for r in rows if r[ti] != ""]
        out.append(dict(
            name=f"muv_{t}", suite="muv", family="vs", task="binary",
            smiles=[r[si] for r in keep], y=[float(r[ti]) for r in keep],
            unit="binary (PubChem confirmatory active)", direction=1, metric="bedroc",
            source=f"MoleculeNet MUV (Rohrer & Baumann, J. Chem. Inf. Model. 49:169, 2009), "
                   f"target {t}; decoys are simple-descriptor-matched to the actives, i.e. an "
                   f"*unbiased* screen by construction",
        ))

    if skip_litpcba:
        return out
    hdr, rows = read_csv(RAW / "litpcba.csv")
    ti, si, yi = col(hdr, "target"), col(hdr, "smiles"), col(hdr, "y")
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r[ti], []).append(r)
    for t in sorted(by):
        rs = by[t]
        out.append(dict(
            name=f"litpcba_{t}", suite="litpcba", family="vs", task="binary",
            smiles=[r[si] for r in rs], y=[float(r[yi]) for r in rs],
            unit="binary (confirmatory active)", direction=1, metric="bedroc",
            source=f"LIT-PCBA (Tran-Nguyen et al., J. Chem. Inf. Model. 60:4263, 2020), "
                   f"target {t}; PubChem dose-response HTS, inactives are experimentally "
                   f"verified rather than property-matched decoys",
        ))
    return out


# ------------------------------------------------------------------------------- metrics
def choose_metric(d, pos_rate: float | None):
    """Pick the primary metric from the **measured** base rate, never from the dataset's name.

    This is the trap ChemPFN's `expected_prevalence` guard exists for: a curated drug list at
    72% positive sat in a screening slot and every units check passed, because 0/1 labels are
    valid at any base rate. Enrichment metrics are uninterpretable at the wrong prevalence, and
    accuracy is uninterpretable at any low one — 99.8% accuracy on MUV is what you get by
    predicting "inactive" for everything.

        regression            rmse            (MAE reported alongside; QM9's literature is MAE)
        virtual screening     bedroc(a=20)    early recognition is the question being asked
        binary, <10% pos      auprc           the imbalanced regime; AUROC saturates here
        binary, >=10% pos     auroc           balanced enough for AUROC to mean something
    """
    if d["task"] == "regression":
        return "rmse", ["mae", "r2", "spearman"]
    if d["family"] == "vs":
        return "bedroc", ["nef1pct", "ef1pct", "auprc"]
    if pos_rate is not None and pos_rate < 0.10:
        return "auprc", ["auroc", "bedroc"]
    return "auroc", ["auprc"]


# ---------------------------------------------------------------------------- validation
def canon(smi):
    """(canonical_smiles, status). status: ok / salvaged / failed.

    Mirrors ``chemtfm.feat.parse``'s failure ladder, including the salvage retry, so the
    parse-failure count reported here is the count the harness would see."""
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return Chem.MolToSmiles(m), "ok"
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None, "failed"
    try:
        err = Chem.SanitizeMol(m, catchErrors=True)
        if err != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ err, catchErrors=True)
        return Chem.MolToSmiles(m), "salvaged"
    except Exception:
        return None, "failed"


def canon_set(smiles, label):
    t0, out = time.time(), set()
    for i, s in enumerate(smiles):
        c, _ = canon(s)
        if c:
            out.add(c)
        if (i + 1) % 250000 == 0:
            print(f"    [{label}] {i + 1:,} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {label:16s} {len(smiles):>9,} rows -> {len(out):>9,} unique canonical "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def audit_mol(smi):
    """Structural oddities the featurisers should be understood against."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return (len(Chem.GetMolFrags(m)), Chem.GetFormalCharge(m), m.GetNumHeavyAtoms(),
            {a.GetSymbol() for a in m.GetAtoms()} - {"C", "H", "N", "O", "S", "P",
                                                     "F", "Cl", "Br", "I", "B", "Si"},
            any(a.GetIsotope() for a in m.GetAtoms()))


def reference_corpora():
    """The four sets a new benchmark molecule must not silently belong to.

    `corpus1m` is the one that matters most: it is what every surrogate in `models.py` was
    fitted on, so a benchmark molecule inside it makes the `hume` (predicted-descriptor) arm
    a memorisation test rather than a generalisation one."""
    refs = {}
    refs["bench"] = list(np.load(BENCH, allow_pickle=True)["smiles"])
    for key, p in [("uma100k", ROOT / "data" / "uma100k" / "selected.txt"),
                   ("corpus1m", ROOT / "data" / "corpus1m" / "selected.txt"),
                   ("cpp_hard", ROOT / "cpp" / "hard.smi")]:
        if p.exists():
            refs[key] = [ln.split()[0] for ln in p.read_text().splitlines() if ln.strip()]
        else:
            print(f"  reference missing: {p}")
    return refs


def compare_moleculeace(cache_canon):
    """Two MoleculeACE curations are in play; confirm whether they are the same molecules.

    Ours lives inside `bench.npz` (30 datasets, inherited from the ChemTFM gate-1 cache); the
    lake holds `eval/locked/moleculeace/`. Structures only — no LOCKED label is scored, and
    nothing here goes through ChemPFN's evaluation ledger."""
    if not LOCKED.exists():
        return {"error": "locked directory absent"}
    d = np.load(BENCH, allow_pickle=True)
    names, suites, offs = list(d["name_of"]), list(d["suite_of"]), d["offsets"]
    smi = np.array(d["smiles"], dtype=object)
    ours = {}
    for j, nm in enumerate(names):
        if suites[j] == "moleculeace":
            ours[str(nm)] = list(smi[offs[j]:offs[j + 1]])

    theirs = {}
    for p in sorted(LOCKED.glob("moleculeace/*.csv")):
        hdr, rows = read_csv(p)
        s = col(hdr, "smiles")
        theirs[p.stem] = [r[s] for r in rows]

    def cs(lst):
        return {cache_canon.setdefault(x, canon(x))[0] for x in lst} - {None}

    per, shared = {}, sorted(set(ours) & set(theirs))
    for k in shared:
        a, b = cs(ours[k]), cs(theirs[k])
        per[k] = dict(ours_n=len(ours[k]), theirs_n=len(theirs[k]),
                      ours_unique=len(a), theirs_unique=len(b),
                      shared=len(a & b), only_ours=len(a - b), only_theirs=len(b - a),
                      jaccard=round(len(a & b) / max(1, len(a | b)), 4))
    return dict(datasets_ours=len(ours), datasets_theirs=len(theirs),
                name_matched=len(shared),
                only_in_ours=sorted(set(ours) - set(theirs)),
                only_in_theirs=sorted(set(theirs) - set(ours)),
                per_dataset=per)


def inventory_locked(cache_canon):
    """Structure-level inventory of the LOCKED binary sets the coordinator asked about.

    Counted, not acquired. Putting any of these into Figure B is a policy decision that has to
    be made explicitly — see `TASKS.md`, "LOCKED IS OFF LIMITS UNTIL HUME IS FROZEN"."""
    out = {}
    for name, rel, scol, ycol in [
        ("adme_bbb", "adme_bbb/adme_bbb.csv", "Drug", "Y"),
        ("herg", "herg/herg.csv", "Drug", "Y"),
        ("wong_saureus", "wong_saureus/wong_saureus.csv", "smiles", "y"),
    ]:
        p = LOCKED / rel
        if not p.exists():
            out[name] = {"error": f"not found at {p}"}
            continue
        hdr, rows = read_csv(p)
        sc = scol if scol in hdr else next((c for c in ("smiles", "Drug", "SMILES")
                                            if c in hdr), None)
        yc = ycol if ycol in hdr else next((c for c in ("y", "Y", "label") if c in hdr), None)
        if sc is None or yc is None:
            out[name] = {"error": f"columns {hdr[:8]}"}
            continue
        si, yi = col(hdr, sc), col(hdr, yc)
        y = np.array([float(r[yi]) for r in rows])
        out[name] = dict(file=str(p), n_rows=len(rows), columns=hdr,
                         n_positive=int((y > 0.5).sum()),
                         positive_rate=round(float((y > 0.5).mean()), 6),
                         status="LOCKED - inventoried only, not acquired")
    return out


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not DEV.exists():
        sys.exit(f"ChemPFN data lake not found at {LAKE}; set CHEMPFN_DATA")

    print(f"lake (read-only): {LAKE}")
    print("copying sources ...", flush=True)
    prov = copy_sources()

    print("loading ...", flush=True)
    # --no-litpcba is for iterating on the pipeline, never for producing the real report.
    dsets = load_qm() + load_admet_cls() + load_vs(skip_litpcba="--no-litpcba" in sys.argv)
    print(f"  {len(dsets)} datasets, {sum(len(d['smiles']) for d in dsets):,} rows "
          f"({time.time() - t0:.0f}s)", flush=True)

    # Metrics are assigned from the measured base rate, after loading and before anything
    # consumes them, so no dataset can carry a metric its prevalence does not support.
    for d in dsets:
        y = np.asarray(d["y"], dtype=np.float64)
        pr = float((y > 0.5).mean()) if d["task"] == "binary" else None
        d["metric"], d["secondary"] = choose_metric(d, pr)
        d["flag"] = ("majority-positive: enrichment metrics are meaningless here"
                     if pr is not None and pr > 0.5 else "")

    # --- reference corpora, canonicalised -------------------------------------------------
    print("canonicalising reference corpora ...", flush=True)
    refs = {k: canon_set(v, k) for k, v in reference_corpora().items()}
    ref_leak = {k: v for k, v in refs.items() if k != "bench"}

    # --- canonicalise every new molecule once ----------------------------------------------
    cache: dict[str, tuple] = {}
    n = 0
    for d in dsets:
        for s in d["smiles"]:
            if s not in cache:
                cache[s] = canon(s)
                n += 1
                if n % 250000 == 0:
                    print(f"    [new] {n:,} distinct ({time.time() - t0:.0f}s)", flush=True)
    print(f"  canonicalised {len(cache):,} distinct SMILES strings "
          f"({time.time() - t0:.0f}s)", flush=True)

    # --- per-dataset report ------------------------------------------------------------------
    report = {}
    for d in dsets:
        smi, y = d["smiles"], np.asarray(d["y"], dtype=np.float64)
        stats = [cache[s] for s in smi]
        cset = {c for c, st in stats if c}
        rec = dict(
            name=d["name"], suite=d["suite"], family=d["family"], task=d["task"],
            unit=d["unit"], metric=d["metric"], secondary_metrics=d["secondary"],
            flag=d["flag"], direction=d["direction"], source=d["source"],
            n_rows=len(smi), n_unique_canonical=len(cset),
            n_parse_failed=sum(1 for c, st in stats if st == "failed"),
            n_salvaged=sum(1 for c, st in stats if st == "salvaged"),
            overlap={k: len(cset & v) for k, v in refs.items()},
            overlap_frac={k: round(len(cset & v) / max(1, len(cset)), 5)
                          for k, v in refs.items()},
        )
        if d["task"] == "binary":
            npos = int((y > 0.5).sum())
            rec.update(n_positive=npos, positive_rate=round(npos / max(1, len(y)), 6))
        else:
            rec.update(y_mean=float(np.mean(y)), y_std=float(np.std(y)),
                       y_min=float(np.min(y)), y_max=float(np.max(y)))
        report[d["name"]] = rec
        leak = max(rec["overlap_frac"][k] for k in ref_leak) if ref_leak else 0.0
        print(f"  {d['name']:26s} n={len(smi):>9,}  " +
              (f"pos={rec.get('positive_rate', 0) * 100:6.3f}%  "
               if d["task"] == "binary" else " " * 15) +
              f"fail={rec['n_parse_failed']:<4} bench={rec['overlap']['bench']:<6} "
              f"train-leak={leak * 100:5.2f}%", flush=True)

    # --- structural audit on a bounded sample -------------------------------------------------
    print("structural audit ...", flush=True)
    rng = np.random.default_rng(0)
    for d in dsets:
        uniq = sorted({s for s in d["smiles"]})
        take = uniq if len(uniq) <= 5000 else [uniq[i] for i in
                                               rng.choice(len(uniq), 5000, replace=False)]
        multi = charged = iso = 0
        exotic: dict[str, int] = {}
        heavy = []
        for s in take:
            a = audit_mol(s)
            if a is None:
                continue
            f, q, h, ex, it = a
            multi += f > 1
            charged += q != 0
            iso += it
            heavy.append(h)
            for e in ex:
                exotic[e] = exotic.get(e, 0) + 1
        m = max(1, len(take))
        report[d["name"]]["audit"] = dict(
            sampled=len(take), frac_disconnected=round(multi / m, 4),
            frac_net_charged=round(charged / m, 4), frac_isotopic=round(iso / m, 4),
            exotic_elements=dict(sorted(exotic.items(), key=lambda kv: -kv[1])[:10]),
            heavy_atoms_p50=int(np.percentile(heavy, 50)) if heavy else 0,
            heavy_atoms_p99=int(np.percentile(heavy, 99)) if heavy else 0,
            heavy_atoms_max=int(max(heavy)) if heavy else 0,
        )

    # --- scaffold-split feasibility -------------------------------------------------------------
    print("scaffold split check ...", flush=True)
    import _vendor  # noqa: F401
    from chemtfm.bench.splits import check_no_leakage, scaffold_folds
    for d in dsets:
        smi = d["smiles"]
        if len(smi) > 60000:  # bounded probe; the algorithm is O(n) and does not change
            idx = rng.choice(len(smi), 20000, replace=False)
            probe, note = [smi[i] for i in idx], f"probe on 20,000 of {len(smi):,}"
        else:
            probe, note = list(smi), "full"
        folds = scaffold_folds(probe, k=5, seed=0)
        sizes = [len(f) for f in folds]
        rec = dict(note=note, clean=bool(check_no_leakage(probe, folds)), fold_sizes=sizes,
                   largest_fold_frac=round(max(sizes) / max(1, sum(sizes)), 4))
        if d["task"] == "binary" and note == "full":
            y = np.asarray(d["y"])
            rec["folds_with_zero_positives"] = int(sum(1 for f in folds
                                                       if (y[f] > 0.5).sum() == 0))
        report[d["name"]]["scaffold"] = rec

    # --- cross-family overlap inside the new sets -------------------------------------------------
    fams: dict[str, set] = {}
    for d in dsets:
        fams.setdefault(d["family"], set()).update(
            {cache[s][0] for s in d["smiles"] if cache[s][0]})
    cross = {}
    keys = sorted(fams)
    for i, a in enumerate(keys):
        cross[f"{a}|_unique_molecules"] = len(fams[a])
        for b in keys[i + 1:]:
            cross[f"{a}|{b}"] = len(fams[a] & fams[b])
        for rk, rv in refs.items():
            cross[f"{a}|REF_{rk}"] = len(fams[a] & rv)

    # --- LOCKED audits (structures only) -----------------------------------------------------------
    locked_audit = {}
    if "--audit-locked" in sys.argv:
        print("LOCKED audit (structures only, no labels scored) ...", flush=True)
        cache_c = {k: v for k, v in cache.items()}
        locked_audit = dict(moleculeace_curation=compare_moleculeace(cache_c),
                            candidate_inventory=inventory_locked(cache_c))
        json.dump(locked_audit, open(OUT / "locked_audit.json", "w"), indent=2)
        print(f"  wrote {OUT / 'locked_audit.json'}")

    # --- write -----------------------------------------------------------------------------------
    def pack(sel, path):
        """Row-per-(dataset, molecule) with an indirection into a shared unique-SMILES table,
        so a molecule appearing in several datasets is featurised once, not once per dataset.
        Tox21/MUV/LIT-PCBA share their molecule sets across assays; without the indirection the
        QM+ADMET+MUV file would demand 1.17M featurisations instead of 928k."""
        uniq, pos = [], {}
        idx, y, offs, meta = [], [], [0], []
        for d in sel:
            for s, v in zip(d["smiles"], d["y"]):
                if s not in pos:
                    pos[s] = len(uniq)
                    uniq.append(s)
                idx.append(pos[s])
                y.append(v)
            offs.append(len(idx))
            meta.append(d)
        np.savez_compressed(
            path,
            smiles=np.array(uniq, dtype=object),
            idx=np.array(idx, dtype=np.int64),
            y=np.array(y, dtype=np.float64),
            offsets=np.array(offs, dtype=np.int64),
            name_of=np.array([d["name"] for d in meta], dtype=object),
            suite_of=np.array([d["suite"] for d in meta], dtype=object),
            family_of=np.array([d["family"] for d in meta], dtype=object),
            task_of=np.array([d["task"] for d in meta], dtype=object),
            metric_of=np.array([d["metric"] for d in meta], dtype=object),
            secondary_metrics_of=np.array([",".join(d["secondary"]) for d in meta],
                                          dtype=object),
            flag_of=np.array([d["flag"] for d in meta], dtype=object),
            unit_of=np.array([d["unit"] for d in meta], dtype=object),
            direction_of=np.array([d["direction"] for d in meta], dtype=np.int8),
            source_of=np.array([d["source"] for d in meta], dtype=object),
        )
        print(f"  wrote {path.name}: {len(meta)} datasets, {len(idx):,} rows, "
              f"{len(uniq):,} unique molecules")
        return len(uniq), len(idx)

    main_sets = [d for d in dsets if d["suite"] != "litpcba"]
    big_sets = [d for d in dsets if d["suite"] == "litpcba"]
    nu_m, nr_m = pack(main_sets, OUT / "tasks.npz")
    nu_b, nr_b = pack(big_sets, OUT / "litpcba.npz") if big_sets else (0, 0)

    json.dump(dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        lake=str(LAKE),
        provenance=prov,
        reference_corpora={k: len(v) for k, v in refs.items()},
        files={"tasks.npz": dict(datasets=len(main_sets), rows=nr_m, unique_molecules=nu_m),
               "litpcba.npz": dict(datasets=len(big_sets), rows=nr_b, unique_molecules=nu_b)},
        cross_family_overlap=cross,
        per_dataset=report,
    ), open(OUT / "report.json", "w"), indent=2)
    print(f"\nwrote {OUT}/report.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
