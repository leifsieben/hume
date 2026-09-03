"""Featurize one shard, accumulate per-column statistics, keep no features.

    .venv/bin/python tools/fuzz/worker.py SHARD.txt OUT.json [THREADS]

Exit codes are the finding. 0 means the shard completed; anything else -- and especially a
negative status, i.e. a signal -- means the shard killed the interpreter, and the driver bisects
it to the molecule. That is why this is a separate process per shard rather than a loop.

FEATURES ARE NEVER RETAINED. A (25000, 1270) float64 block is 254 MB; the point of the run is
the statistics, so each batch is reduced and dropped.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
from rdkit import RDLogger                                    # noqa: E402

RDLogger.DisableLog("rdApp.*")
import molhume                                                # noqa: E402

BATCH = 2000


def main() -> None:
    shard, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    smis = shard.read_text(encoding="utf-8").split("\n")

    cols = list(molhume.column_set("full", extra=["qed"]))     # widest code coverage
    n_col = len(cols)
    acc = dict(
        n=0, rows_all_nan=0, warn_row=0, warn_col=0,
        nan=np.zeros(n_col, np.int64), posinf=np.zeros(n_col, np.int64),
        neginf=np.zeros(n_col, np.int64), zero=np.zeros(n_col, np.int64),
        lo=np.full(n_col, np.inf), hi=np.full(n_col, -np.inf),
        absmax=np.zeros(n_col, np.float64),
    )
    messages: dict[str, int] = {}

    for lo in range(0, len(smis), BATCH):
        chunk = smis[lo:lo + BATCH]
        if not chunk:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            X = molhume.featurize(chunk, columns=cols, standardize="none", fingerprint=False,
                                  on_error="nan", threads=threads)
        for w in caught:
            m = str(w.message)
            key = m.split(":")[-1].strip()[:150] if "could not" in m or "did not" in m else m[:150]
            messages[key] = messages.get(key, 0) + 1
            if "could not be computed" in m:
                acc["warn_col"] += 1
            elif "could not be featurized" in m or "did not parse" in m:
                acc["warn_row"] += 1

        finite = np.isfinite(X)
        acc["n"] += X.shape[0]
        acc["rows_all_nan"] += int(np.all(np.isnan(X), axis=1).sum())
        acc["nan"] += np.isnan(X).sum(0)
        acc["posinf"] += (np.isinf(X) & (X > 0)).sum(0)
        acc["neginf"] += (np.isinf(X) & (X < 0)).sum(0)
        acc["zero"] += ((X == 0) & finite).sum(0)
        Xf = np.where(finite, X, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                    # all-NaN column -> NaN, not a crash
            acc["lo"] = np.fmin(acc["lo"], np.nanmin(Xf, axis=0))
            acc["hi"] = np.fmax(acc["hi"], np.nanmax(Xf, axis=0))
            acc["absmax"] = np.fmax(acc["absmax"],
                                    np.nan_to_num(np.nanmax(np.abs(Xf), axis=0), nan=0.0))
        del X, Xf, finite

    out_path.write_text(json.dumps({
        "columns": cols,
        "n": acc["n"], "rows_all_nan": acc["rows_all_nan"],
        "warn_row": acc["warn_row"], "warn_col": acc["warn_col"],
        "nan": acc["nan"].tolist(), "posinf": acc["posinf"].tolist(),
        "neginf": acc["neginf"].tolist(), "zero": acc["zero"].tolist(),
        "lo": [None if not np.isfinite(v) else float(v) for v in acc["lo"]],
        "hi": [None if not np.isfinite(v) else float(v) for v in acc["hi"]],
        "absmax": [float(v) for v in acc["absmax"]],
        "messages": messages,
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
