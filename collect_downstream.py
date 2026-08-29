"""Pull the downstream grid off S3 and build the Figure B and Figure C contracts.

    python3 collect_downstream.py            # fetch, aggregate, write both results.json

WHAT IT READS. Every `downstream/<iid>.json` (a completed box) and `downstream/<iid>.partial.json`
(a box still running, or one that hit its wall-clock cap) in the bench bucket. Partials are used
DELIBERATELY: each box rewrites its grid after every dataset and ships it within 30 s, so a box
killed at hour 19 of 20 still contributes every dataset it finished. A completed file always wins
over a partial from the same instance.

HOW DATASETS COMBINE INTO A TASK. A task is a group of datasets sharing an endpoint type and a
metric. Their raw units are not comparable -- RMSE on logS is not RMSE on a microsomal clearance
-- so nothing is ever averaged in raw units. Each dataset is first divided by the ANCHOR arm's
score on that same dataset, and the per-dataset RATIOS are what get averaged. The anchor is then
1.000 by construction, which is exactly the unit both figures draw.

The spread is the SEM ACROSS DATASETS, not across folds. Fold-to-fold spread on one dataset
understates the thing a reader needs, which is whether the effect holds across endpoints; a
5-fold SEM on a single dataset would draw whiskers a third the size and imply a reproducibility
this grid has not demonstrated.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUCKET = "hume-bench-use1-075120018132"
OUT_B = ROOT / "results" / "figures" / "figB" / "results.json"
OUT_C = ROOT / "results" / "figures" / "figC" / "results.json"
CACHE = ROOT / "results" / "figures" / "downstream_raw.json"

#: Dataset -> task. Grouped by endpoint AND metric: a task whose members are scored in two
#: different units cannot share a panel, and the assertion below refuses to build one.
TASKS = {
    "physchem": ("Physicochemical", ["aqsoldb", "esol", "lipophilicity", "pb_logd",
                                     "pb_water_sol", "photoswitch"]),
    "adme":     ("ADME & tox", ["pb_hum_mic_cl", "pb_mou_mic_cl", "pb_rat_mic_cl", "pb_ppb",
                                "vdss_lombardo", "ld50_zhu", "cycpept_pampa",
                                "pb_cyp2c9", "pb_cyp2d6", "pb_cyp3a4"]),
    "classif":  ("Classification", ["ames", "pb_ames", "cyp2d6_inh", "bioavail", "hia",
                                    "pb_bbb"]),
    "quantum":  ("Quantum energy", ["qm8", "qm9", "qm9_gap", "qmugs_gap"]),
}
# GROUPED BY METRIC FIRST, endpoint second, and the assertion below enforces it. The obvious
# chemistry grouping does not survive contact with the lake: `cycpept_pampa` is a permeability
# REGRESSION and the three `pb_cyp*` panels are regressions while `cyp2d6_inh` is a
# classification, so a "bioactivity & tox" panel built on endpoint alone mixed AUROC with RMSE.
# A panel shows one unit; there is no honest way to average those.
#
# `fartdb` (accuracy) and `rascore` are the only datasets in no task: neither shares a metric AND
# an endpoint with a group, and a fifth panel holding two unrelated datasets communicates less
# than leaving them in the per-dataset CSV, which still carries them.

FIGB_BASES = ["ecfp", "desc", "ecfp_all_desc"]
FIGB_ANCHOR = "ecfp_all_desc"
FIGB_ADDS = ["chemeleon", "chemberta_mtr", "minimol", "molformer"]
FIGC_ARMS = ["ecfp", "ecfp_rdkit_desc", "ecfp_mordred_desc", "ecfp_all_desc", "hume",
             "minimol", "chemeleon", "chemberta_mtr"]
#: The three arms Figure C plots that are not measured end-to-end by bench_aws.py under the same
#: name. Cost is ADDITIVE here because the arm literally runs both blocks: `ecfp_rdkit_desc` is
#: the ECFP call plus the RDKit-180 call, one after the other, on the same molecule.
# NO SUMMING. `rdkit_desc` and `mordred_desc` in bench_aws ALREADY include the Morgan call --
# they are the concatenated arm, not the descriptor block alone -- so adding `ecfp` on top
# double-counted the fingerprint. Small (~4% on ecfp_rdkit_desc, under 1% on the others) but
# wrong, and the kind of wrong that survives review because the number still looks plausible.
COST_SUM = {"ecfp_rdkit_desc": ["rdkit_desc"],
            "ecfp_mordred_desc": ["mordred_desc"],
            "ecfp_all_desc": ["mordred"]}
#: THE ECFP BASELINE IS r=2 HERE, matching the `ecfp` arm the downstream grid actually ran.
#: bench_aws's `ecfp` arm is r=3, which is the radius HUME carries internally -- the two were
#: being paired as if they were the same arm. `ecfp_r2` is measured separately for this.
#:
#: The ECFP embedded inside the descriptor arms above is still r=3; it is under 1% of their
#: cost, and re-measuring a 1.7-hour Mordred sweep to change a number by 0.3% is not a good
#: trade. Stated rather than hidden.
COST_KEY = {"ecfp": "ecfp_r2", "hume": "hume", "chemeleon": "chemeleon",
            "chemberta_mtr": "chemberta", "minimol": "minimol"}


def fetch():
    """-> [record]. Completed files win over partials from the same instance."""
    ls = subprocess.run(["aws", "s3", "ls", f"s3://{BUCKET}/downstream/"],
                        capture_output=True, text=True)
    keys = [l.split()[-1] for l in ls.stdout.splitlines() if l.strip().endswith(".json")]
    done = {k.split(".")[0] for k in keys if not k.endswith(".partial.json")}
    use = [k for k in keys if not (k.endswith(".partial.json") and k.split(".")[0] in done)]
    # NEWEST WINS, per (dataset, arm, fold).
    #
    # A re-run does not replace the file it supersedes -- it lands under a NEW instance id -- so
    # concatenating everything would give a cell ten fold-values instead of five and quietly
    # average the old protocol with the new one. Files are read in S3 LastModified order and
    # later records overwrite earlier ones for the same cell. This also covers the shard overlap
    # that happens whenever a box is retired mid-dataset and its work relaunched elsewhere.
    order = {}
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".json"):
            order[parts[-1]] = parts[0] + " " + parts[1]
    # PROTOCOL BEATS TIMESTAMP. A record produced by a newer harness supersedes an older one
    # however recently the older file was written -- see PROTO in bench_downstream.py. Records
    # from before the stamp existed are treated as protocol 1.
    #
    # The six wfix instances ran the fixed harness before the stamp was added, so they are named
    # here explicitly rather than inferred; everything after this carries `proto` in the record.
    PROTO2_RUNS = {"i-084fac77c3e522334", "i-04496c0164e234040", "i-0617d9b2aeb8cf3dc",
                   "i-0fb9c9e172c079b65", "i-09645f62c2f264088", "i-04080faa787d8dd8f"}
    merged, seen = {}, []
    for k in sorted(use, key=lambda k: order.get(k, "")):
        r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/downstream/{k}", "-"],
                           capture_output=True)
        if r.returncode != 0:
            print(f"  ! could not read {k}")
            continue
        rows = json.loads(r.stdout)
        iid = k.split(".")[0]
        for x in rows:
            x.setdefault("proto", 2 if iid in PROTO2_RUNS else 1)
            key = (x["dataset"], x["arm"], x["fold"])
            prev = merged.get(key)
            if prev is None or x["proto"] >= prev["proto"]:
                merged[key] = x
        seen.append((k, len(rows), order.get(k, "?")))
    for k, n, when in seen:
        print(f"  {k:<44s} {n:6,d} records   {when}")
    dropped = sum(n for _k, n, _w in seen) - len(merged)
    if dropped:
        print(f"  superseded by a later run: {dropped:,} records")
    return list(merged.values())


def aggregate(recs):
    """-> {(task, arm): (mean_ratio, sem, n_folds)} plus the per-dataset table."""
    fold = defaultdict(list)                       # (dataset, arm) -> [value]
    metric = {}
    for r in recs:
        fold[(r["dataset"], r["arm"])].append(float(r["value"]))
        metric[r["dataset"]] = r["metric"]
    # EVERYTHING BECOMES AN ERROR, so that lower is better on every panel of every figure.
    #
    # The plates used to carry the field's own units and INVERT the axis for error metrics, so
    # "up" meant better everywhere while the y-label still said "rmse". That is two things at
    # once and it misled a reader of our own figure: HUME sits second-best on Quantum energy and
    # was read off the inverted axis as the worst point on the plate (Leif 2026-08-29).
    #
    # AUROC and accuracy are converted to 1 - x. RMSE is already an error. The ratio to the
    # anchor is then an error ratio: below 1.0 is better than ECFP4+descriptors, above is worse,
    # in one direction, with no inversion anywhere.
    # ABSOLUTE DIFFERENCE FROM THE ANCHOR, not a ratio (Leif 2026-08-29).
    #
    # The ratio was unsafe near ceiling and it decided a panel on one small dataset. `hia` has
    # n=578 and the anchor already scores AUROC 0.9718, so the anchor's error is 0.0282; dividing
    # by that turned a 2.3-POINT AUROC difference into a 2.5x error ratio, and that single
    # dataset moved CheMeleon's classification mean from 0.977 to 0.890 and HUME's from 1.026 to
    # 1.111. The models were tied on the three datasets with 7k-13k molecules.
    #
    # A difference has no denominator to explode. To make differences comparable ACROSS datasets
    # in a family, each regression error is first divided by the STANDARD DEVIATION OF THE TARGET
    # on its own dataset -- a property of the data, not of any model's skill -- so RMSE on log
    # solubility and RMSE on a clearance are on one scale. Classification is already unitless.
    #
    # The anchor is therefore 0.000 by construction, negative beats it, and lower is better.
    SD = json.loads((ROOT / "results" / "figures" / "target_sd.json").read_text())
    def _err(d, a, v):
        if metric[d] in ("auroc", "acc"):
            return 1.0 - v
        if d not in SD or not SD[d]:
            raise KeyError(f"{d}: no target standard deviation; regression errors cannot be put "
                           f"on a common scale without it. Re-run the sd extraction.")
        return v / SD[d]
    per_ds = {}
    for (d, a), v in fold.items():
        per_ds[(d, a)] = float(_err(d, a, float(np.mean(v))))

    out, table = {}, []
    for tkey, (_lab, dss) in TASKS.items():
        mets = {metric[d] for d in dss if d in metric}
        assert len(mets) <= 1, (
            f"task {tkey!r} mixes metrics {sorted(mets)}. A panel shows ONE unit; regroup the "
            f"datasets rather than averaging an AUROC with an RMSE.")
        arms = {a for (d, a) in per_ds if d in dss}
        for arm in arms:
            ratios = []
            for d in dss:
                v, ref = per_ds.get((d, arm)), per_ds.get((d, FIGB_ANCHOR))
                if v is None or ref is None:
                    continue
                ratios.append(v - ref)
                table.append({"task": tkey, "dataset": d, "arm": arm, "value": v,
                              "anchor": ref, "ratio": v - ref, "metric": metric[d]})
            if not ratios:
                continue
            out[(tkey, arm)] = (float(np.mean(ratios)),
                                float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
                                if len(ratios) > 1 else 0.0,
                                sum(len(fold[(d, arm)]) for d in dss if (d, arm) in fold))
    return out, table, metric


def task_specs(metric):
    specs = []
    for tkey, (lab, dss) in TASKS.items():
        m = next((metric[d] for d in dss if d in metric), None)
        if m is None:
            continue
        # Every task is now scored as an ERROR ratio, so lower is better without exception; the
        # label records what the error was derived from.
        specs.append({"key": tkey, "label": lab,
                      "metric": {"auroc": "1 - auroc", "acc": "1 - acc"}.get(m, "rmse / sd(y)"),
                      "lower_is_better": True})
    return specs


def costs():
    """us/mol per Figure C arm, from results/scale/. Missing arms are reported, not invented."""
    pts = defaultdict(dict)
    for f in (ROOT / "results" / "scale").glob("*_cpu.json"):
        d = json.loads(f.read_text())
        for p in d["points"]:
            pts[p["arm"]][p["n"]] = p["wall_s"] / p["n"] * 1e6
    # THE LARGEST N MEASURED, and the spread across decades is REPORTED. Figure C's x-axis is a
    # per-molecule cost, which presumes that cost does not depend on N -- true for every
    # descriptor and fingerprint arm here (all flat inside 3%) but NOT for minimol, which reads
    # 3243.9 us/mol at 1e4 and 1730.8 at 1e5 as its fixed model-loading cost amortises. Taking the
    # largest N is the right choice for a figure that extrapolates to a billion molecules, but an
    # arm whose cost is still moving is a caveat, not a number, and it says so out loud.
    best, drift = {}, {}
    for a, v in pts.items():
        if not v:
            continue
        best[a] = v[max(v)]
        lo, hi = min(v.values()), max(v.values())
        if lo > 0 and (hi - lo) / lo > 0.10:
            drift[a] = (sorted(v), [round(v[n], 1) for n in sorted(v)])
    for a, (ns, vs) in drift.items():
        print(f"  ! {a}: us/mol is NOT flat across N -- {vs} at N={ns}. The largest N is used; "
              f"the per-molecule axis is an asymptote for this arm, not a constant.")
    out, missing = {}, []
    for arm in FIGC_ARMS:
        if arm in COST_SUM:
            parts = COST_SUM[arm]
            if any(p not in best for p in parts):
                missing.append((arm, [p for p in parts if p not in best]))
                continue
            out[arm] = {"us_per_mol": sum(best[p] for p in parts),
                        "measured_on": "c7i.4xlarge, 16 vCPU",
                        "breakdown": {p: best[p] for p in parts}}
        else:
            k = COST_KEY.get(arm)
            if k not in best:
                missing.append((arm, [k]))
                continue
            out[arm] = {"us_per_mol": best[k], "measured_on": "c7i.4xlarge, 16 vCPU",
                        "breakdown": {k: best[k]}}
    for arm, parts in missing:
        print(f"  ! no measured cost for {arm} (needs {parts}) -- it cannot be placed on "
              f"figure C's x-axis and is dropped")
    return out


def main():
    recs = fetch()
    if not recs:
        raise SystemExit("no downstream results in S3 yet")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(recs))
    agg, table, metric = aggregate(recs)
    specs = task_specs(metric)
    have = sorted({a for _t, a in agg})
    print(f"\n  {len(recs):,} records | {len(specs)} tasks | {len(have)} arms: {have}")

    # ---- figure B ----------------------------------------------------------------------
    brecs = []
    for t in specs:
        for base in FIGB_BASES:
            for add in [None] + FIGB_ADDS:
                arm = base if add is None else f"{base}__{add}"
                cell = agg.get((t["key"], arm))
                if cell is None:
                    continue
                brecs.append({"task": t["key"], "base": base, "add": add,
                              "mean": cell[0], "sem": cell[1], "n_folds": cell[2]})
    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    OUT_B.write_text(json.dumps(
        {"meta": {"source": "bench_downstream.py on c7i.4xlarge", "n_records": len(recs),
                  "unit": "mean over datasets of (arm error - ecfp_all_desc error) on the same "
                          "dataset; regression error = rmse / sd(y), classification = 1 - auroc. "
                          "Anchor is 0 by construction; lower is better."},
         "tasks": specs, "bases": FIGB_BASES, "anchor": FIGB_ANCHOR, "adds": FIGB_ADDS,
         "records": brecs}, indent=1))
    print(f"  -> {OUT_B.relative_to(ROOT)}  {len(brecs)} cells")

    # ---- figure C ----------------------------------------------------------------------
    cost = costs()
    arms = [a for a in FIGC_ARMS if a in cost and any((t["key"], a) in agg for t in specs)]
    crecs = [{"task": t["key"], "arm": a, "head": "xgboost",
              "mean": agg[(t["key"], a)][0], "sem": agg[(t["key"], a)][1],
              "n_folds": agg[(t["key"], a)][2]}
             for t in specs for a in arms if (t["key"], a) in agg]
    OUT_C.parent.mkdir(parents=True, exist_ok=True)
    OUT_C.write_text(json.dumps(
        {"meta": {"source": "bench_downstream.py + results/scale", "n_records": len(recs),
                  "unit": "mean over datasets of (arm error - ecfp_all_desc error) on the same "
                          "dataset; regression error = rmse / sd(y), classification = 1 - auroc. "
                          "Anchor is 0 by construction; lower is better."},
         "tasks": specs, "arms": arms, "cost": cost, "records": crecs}, indent=1))
    print(f"  -> {OUT_C.relative_to(ROOT)}  {len(crecs)} cells, {len(arms)} arms")

    with open(ROOT / "figures" / "build" / "downstream_by_dataset.csv", "w") as fh:
        fh.write("task,dataset,arm,metric,value,anchor,ratio\n")
        for r in table:
            fh.write(f"{r['task']},{r['dataset']},{r['arm']},{r['metric']},"
                     f"{r['value']:.6f},{r['anchor']:.6f},{r['ratio']:.6f}\n")


if __name__ == "__main__":
    main()
