"""Is mol-hume safe on a new RDKit release? Answer it in one command, with measurements.

    .venv/bin/python tools/check_rdkit_release.py 2026.09.1

RDKit releases roughly twice a year, and each one is a question this package cannot answer by
inspection: it reads the MolPickler blob directly, and it computes descriptors from RDKit's
perceived atom and bond properties. Both can change. This script runs the three checks that
decide whether the version cap in pyproject.toml can move, in the order that fails cheapest:

    1. PICKLE FORMAT. Probe the version triple the release writes. If it is already in
       molpickle.h's SUPPORTED set, done. If not, the answer is not automatically "no" -- go to
       check 2, which is what justified adding 16.3.0.

    2. BLOB EQUIVALENCE. Pickle N corpus molecules under both the reference release and the new
       one, with this package's exact flags, and compare the bytes. If they differ ONLY in the
       version triple, the new format is readable by the existing reader and the only change
       needed is adding the triple to SUPPORTED. If they differ elsewhere, the reader has to be
       updated and re-verified -- this script will not tell you it is safe.

    3. VALUE EQUIVALENCE, end to end. Compute all 1,269 columns under both releases and compare.
       This catches the OTHER kind of drift, which has nothing to do with pickles: RDKit
       changing an aromaticity model, a ring-perception tie-break, a Crippen parameter. Columns
       can move here even when the pickle is byte-identical.

Nothing is written and nothing is installed into the working environment: each release goes in
its own throwaway venv, and mol-hume is installed there with --no-deps from a wheel built here.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "2025.9.2"          # what the exactness numbers are quoted against
N_MOLS = 4000

PICKLE_PROBE = r"""
import struct
from rdkit import Chem
b = Chem.MolFromSmiles('C').ToBinary()
print('%d.%d.%d' % struct.unpack('<3i', b[8:20]))
"""

BLOB_SCRIPT = r"""
import json, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
F = (Chem.PropertyPickleOptions.PrivateProps | Chem.PropertyPickleOptions.AtomProps
     | Chem.PropertyPickleOptions.ComputedProps | Chem.PropertyPickleOptions.NoConformers)
smis = json.load(open(sys.argv[1]))['smiles'][:int(sys.argv[3])]
out = []
for s in smis:
    m = Chem.MolFromSmiles(s)
    if m is None:
        out.append(None); continue
    try: m.ClearComputedProps()
    except Exception: pass
    Chem.AssignStereochemistry(m, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    out.append(m.ToBinary(F).hex())
json.dump(out, open(sys.argv[2], 'w'))
"""

VALUE_SCRIPT = r"""
import json, sys, warnings
import numpy as np, molhume
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.simplefilter('ignore')
smis = json.load(open(sys.argv[1]))['smiles'][:int(sys.argv[3])]
np.save(sys.argv[2], molhume.featurize(smis, standardize='none'))
"""


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def make_env(tmp, version):
    """A throwaway venv with just this rdkit. Returns the interpreter path, or None."""
    d = tmp / f"env_{version}"
    if run(["uv", "venv", str(d), "-q", "--python", "3.12"], cwd=tmp).returncode != 0:
        return None
    py = d / "bin" / "python"
    r = run(["uv", "pip", "install", "--python", str(py), f"rdkit=={version}", "numpy", "-q"],
            cwd=tmp)                       # cwd=tmp so this repo's [tool.uv] pins do not apply
    if r.returncode != 0:
        print(f"    could not install rdkit=={version}:\n{r.stderr.strip()[-400:]}")
        return None
    return py


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="the rdkit release to check, e.g. 2026.09.1")
    ap.add_argument("--reference", default=REFERENCE, help=f"compare against (default {REFERENCE})")
    ap.add_argument("--n", type=int, default=N_MOLS, help=f"molecules (default {N_MOLS})")
    args = ap.parse_args()

    corpus = ROOT / "data" / "exactness_corpus.json"
    if not corpus.exists():
        sys.exit(f"need the corpus at {corpus}; it is gitignored, so this runs in the research "
                 "tree and not from an sdist.")

    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print(f"\n[1/3] pickle format written by rdkit {args.version}")
        py_new = make_env(tmp, args.version)
        if py_new is None:
            sys.exit("  cannot continue without that release")
        fmt = run([str(py_new), "-c", PICKLE_PROBE]).stdout.strip()
        print(f"      {fmt}")

        sys.path.insert(0, str(ROOT / "src"))
        try:
            from molhume import _core
            supported = {".".join(map(str, v)) for v in _core.PICKLE_VERSIONS}
        except Exception:
            supported = set()
        print(f"      reader supports: {', '.join(sorted(supported)) or '(could not import)'}")
        known = fmt in supported

        print(f"\n[2/3] blob equivalence vs rdkit {args.reference}, {args.n} molecules")
        py_ref = make_env(tmp, args.reference)
        if py_ref is None:
            sys.exit("  cannot continue without the reference release")
        for py, tag in ((py_ref, "ref"), (py_new, "new")):
            run([str(py), "-c", BLOB_SCRIPT, str(corpus), str(tmp / f"blob_{tag}.json"),
                 str(args.n)])
        A = json.load(open(tmp / "blob_ref.json"))
        B = json.load(open(tmp / "blob_new.json"))
        only_version = elsewhere = identical = 0
        for x, y in zip(A, B):
            if x is None or y is None:
                continue
            if x == y:
                identical += 1
            else:
                xb, yb = bytes.fromhex(x), bytes.fromhex(y)
                if len(xb) == len(yb) and xb[:8] == yb[:8] and xb[20:] == yb[20:]:
                    only_version += 1
                else:
                    elsewhere += 1
        print(f"      byte-identical            {identical}")
        print(f"      differ only in the version {only_version}")
        print(f"      differ ELSEWHERE           {elsewhere}")
        blob_ok = elsewhere == 0

        print(f"\n[3/3] all 1,269 columns vs rdkit {args.reference}, {args.n} molecules")
        wheel_dir = tmp / "wheel"
        if run(["uv", "build", "--wheel", "-o", str(wheel_dir)], cwd=ROOT).returncode != 0:
            sys.exit("      could not build a wheel to test with")
        whl = next(wheel_dir.glob("*.whl"))
        values_ok = None
        for py, tag in ((py_ref, "ref"), (py_new, "new")):
            run(["uv", "pip", "install", "--python", str(py), "--no-deps", str(whl), "-q"],
                cwd=tmp)
            r = run([str(py), "-c", VALUE_SCRIPT, str(corpus), str(tmp / f"X_{tag}.npy"),
                     str(args.n)], cwd=ROOT)
            if r.returncode != 0:
                print(f"      {tag}: mol-hume did not run:\n{r.stderr.strip()[-600:]}")
                values_ok = False
        if values_ok is None:
            a = np.load(tmp / "X_ref.npy")
            b = np.load(tmp / "X_new.npy")
            import molhume
            names = molhume.ALL_COLUMNS
            nan_d = np.isnan(a) != np.isnan(b)
            fin = np.isfinite(a) & np.isfinite(b)
            d = np.zeros_like(a)
            d[fin] = np.abs(a[fin] - b[fin])
            moved = sorted({int(c) for c in np.argwhere((d > 0) | nan_d)[:, 1]})
            print(f"      bit-identical columns  {a.shape[1] - len(moved)} / {a.shape[1]}")
            print(f"      max abs difference     {d.max():.3e}")
            for c in moved[:20]:
                n = int(((d[:, c] > 0) | nan_d[:, c]).sum())
                print(f"         {names[c]:26s} {n:5d} rows, max abs {d[:, c].max():.3e}")
            if len(moved) > 20:
                print(f"         ... and {len(moved) - 20} more")
            values_ok = not moved

    print("\n" + "=" * 78)
    if known and values_ok:
        print(f"  rdkit {args.version} is already supported and produces identical values.")
        print("  Widen the cap in pyproject.toml if it excludes this release.")
    elif blob_ok and values_ok:
        print(f"  rdkit {args.version} writes pickle {fmt}, which this reader does not list, but")
        print("  the blobs differ only in the version triple and every column is identical.")
        print(f"  SAFE to add {{{fmt.replace('.', ', ')}}} to SUPPORTED in molpickle.h and widen")
        print("  the cap. Re-run cpp/verify_molpickle.py on both corpora first.")
    elif not blob_ok:
        print(f"  rdkit {args.version} changes the blob in ways beyond the version triple.")
        print("  The reader must be updated: re-read Code/GraphMol/MolPickler.cpp, then re-run")
        print("  cpp/verify_molpickle.py and the full verify_*.py suite. NOT safe as-is.")
    else:
        print(f"  rdkit {args.version} is readable but MOVES VALUES. That is perception drift,")
        print("  not a pickle problem. Decide whether to follow it, then regenerate the fixture")
        print("  with tools/gen_fixture.py and record the moved columns in CHANGELOG.md.")
    print("=" * 78)
    return 0 if (values_ok and (known or blob_ok)) else 1


if __name__ == "__main__":
    sys.exit(main())
