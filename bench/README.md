# Benchmarks and collection

`bench_aws.py` measures featurization throughput on a named EC2 instance and writes Figure D's
data contract; `scripts/boot_scale.sh` runs it there. `bench_downstream.py` is the downstream
grid: XGBoost over (dataset, fold, arm). `bench_e2e.py` is the per-family timing breakdown.

`collect_scale.py` and `collect_downstream.py` pull results out of S3, merge newest-wins, and
write the JSON the plates in `figures/src/` read. `refresh_figures.sh` runs both and re-renders.
