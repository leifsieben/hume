"""Pull the EC2 benchmark results out of S3 and attach today's prices. -> results/scale/*.json

WHY PRICES ARE ATTACHED HERE AND NOT ON THE INSTANCE. A price is a property of (instance type,
region, date), not of the run -- and the first fleet recorded `usd_per_hour_ondemand: 0.0`
because the env vars were never passed. Querying the AWS pricing API at collection time makes
the number authoritative rather than hand-typed, and stamps the date it was pulled so Figure D's
caption can say when. Re-running this later re-prices the same measurements, which is correct:
the timings do not change, the prices do.

Spot is recorded as the CURRENT median across availability zones. It is a lower band on the
figure, not a claim about what a run would have cost -- spot prices move by the hour.
"""
from __future__ import annotations
import json, statistics, subprocess, sys, time
from pathlib import Path

BUCKET = "hume-bench-use1-075120018132"
REGION = "us-east-1"
LOCATION = "US East (N. Virginia)"
OUT = Path("results/scale")


def sh(*a: str) -> str:
    return subprocess.run(a, capture_output=True, text=True).stdout


def ondemand(itype: str) -> float:
    raw = sh("aws", "pricing", "get-products", "--service-code", "AmazonEC2",
             "--region", "us-east-1",
             "--filters", f"Type=TERM_MATCH,Field=instanceType,Value={itype}",
             f"Type=TERM_MATCH,Field=location,Value={LOCATION}",
             "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
             "Type=TERM_MATCH,Field=tenancy,Value=Shared",
             "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
             "Type=TERM_MATCH,Field=capacitystatus,Value=Used",
             "--max-results", "1", "--query", "PriceList[0]", "--output", "text")
    d = json.loads(raw)
    term = list(d["terms"]["OnDemand"].values())[0]
    dim = list(term["priceDimensions"].values())[0]
    return float(dim["pricePerUnit"]["USD"])


def spot(itype: str) -> float | None:
    raw = sh("aws", "ec2", "describe-spot-price-history", "--instance-types", itype,
             "--product-descriptions", "Linux/UNIX", "--region", REGION, "--max-results", "12",
             "--query", "SpotPriceHistory[*].SpotPrice", "--output", "text")
    vals = [float(x) for x in raw.split()]
    return statistics.median(vals) if vals else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    listing = sh("aws", "s3", "ls", f"s3://{BUCKET}/results/", "--recursive")
    keys = [ln.split()[-1] for ln in listing.splitlines() if ln.strip().endswith(".json")]
    if not keys:
        sys.exit("no result objects under s3://%s/results/ -- has the fleet finished?" % BUCKET)
    today = time.strftime("%Y-%m-%d")
    by_instance: dict[str, dict] = {}
    price_cache: dict[str, tuple[float, float | None]] = {}
    for k in keys:
        raw = sh("aws", "s3", "cp", f"s3://{BUCKET}/{k}", "-")
        if not raw.strip():
            print(f"  WARNING unreadable: {k}")
            continue
        d = json.loads(raw)
        m, itype = d["meta"], d["meta"]["instance"]
        if itype not in price_cache:
            price_cache[itype] = (ondemand(itype), spot(itype))
            print(f"  priced {itype}: on-demand ${price_cache[itype][0]:.4f}/h  "
                  f"spot ${price_cache[itype][1]}/h")
        od, sp = price_cache[itype]
        m["usd_per_hour_ondemand"], m["usd_per_hour_spot"], m["priced_on"] = od, sp, today
        tag = f"{itype}_{m['budget']}"
        by_instance.setdefault(tag, {"meta": m, "points": []})["points"].extend(d["points"])
    for tag, blob in by_instance.items():
        blob["points"].sort(key=lambda p: (p["arm"], p["n"]))
        p = OUT / f"{tag}.json"
        p.write_text(json.dumps(blob, indent=1))
        arms = sorted({q["arm"] for q in blob["points"]})
        print(f"  -> {p}  {len(blob['points'])} points, arms: {', '.join(arms)}")


if __name__ == "__main__":
    main()
