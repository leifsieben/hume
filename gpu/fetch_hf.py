"""Download the CLM baselines. Pure I/O -- safe to run alongside the grid.

These are the schedule risk in the whole plan: they fail on environment and auth problems
rather than on compute, and that failure mode does not respect a deadline. Downloading them
early converts an unknown into either a working directory or a clear error.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "models_hf"
MODELS = [("ChemBERTa-2", "DeepChem/ChemBERTa-77M-MLM"),
          ("MolFormer", "ibm/MoLFormer-XL-both-10pct"),
          ("SMI-TED", "ibm/materials.smi-ted")]
def main():
    from huggingface_hub import snapshot_download
    for label, repo in MODELS:
        t0 = time.time()
        try:
            p = snapshot_download(repo_id=repo, local_dir=str(OUT / label),
                                  allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors",
                                                  "*.model", "*.py", "*.pt"])
            n = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
            print(f"  OK   {label:12s} {n/1e6:7.0f} MB  ({time.time()-t0:.0f}s)  {repo}", flush=True)
        except Exception as e:
            print(f"  FAIL {label:12s} {type(e).__name__}: {str(e)[:150]}", flush=True)
if __name__ == "__main__":
    main()
