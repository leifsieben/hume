"""Verification harness for `src/hume_core/sps.h` -- the `SPS` column and, underneath it,
RDKit's NEW potential-stereo perception ported to C++.

RULE 1 OF THE CONTRACT APPLIES: not one descriptor value is computed in this file. Python here
does three things and nothing else -- it calls RDKit for ground truth, it serialises the boundary
quantities the C++ header takes as input, and it diffs.

WHAT IS CHECKED, IN THE ORDER THE TASK STAGES IT

  stage 1  the LEGACY `_ChiralityPossible` flag set against the NEW perception's
           `Atom_Tetrahedral` set, over all 20,000 corpus molecules. This is the measurement that
           justifies the header existing at all: if the two agreed, `SPS` could read the flag the
           pickle already carries.

  stage 2  this header's perception against `Chem.FindPotentialStereo` ON ITS OWN, atom set by
           atom set, over all 20,000 molecules. Reported as: molecules whose full set of
           potential-stereocentre atom indices is identical.

  stage 3  the `SPS` column against `data/dedupe2/matrix.npz`, with the per-column tally the
           contract asks for (exact / 1e-9 / 1e-6 / mismatched / NaN agreement) and every
           mismatch printed with its SMILES.

  timing   microseconds per molecule with SD, per heavy-atom stratum, against the 54.4 us/mol the
           current Python route costs.

The C++ is exercised through a standalone harness compiled into `build_sps/` -- NOT through
`hume._core`, because `src/hume_core/bindings.cpp` is off limits to this agent (five of us are
editing this checkout in parallel and the column-offset enum in that file is the one place a
silent transposition could hide). The harness includes `src/hume_core/sps.h` unmodified and reads
a text dump of exactly the fields the header's `Mol` declares, which are exactly the fields
`bindings.cpp` already has in hand.

    .venv/bin/python verify_sps.py            # everything
    .venv/bin/python verify_sps.py --n 2000   # a subset, for iteration
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build_sps")
CORPUS = os.path.join(ROOT, "data", "dedupe2", "corpus.json")
MATRIX = os.path.join(ROOT, "data", "dedupe2", "matrix.npz")


# ---------------------------------------------------------------------------------------------
# The standalone harness. Two modes: `run` prints one line per molecule (the potential-stereo
# atom set, then SPS), `bench` re-runs the whole batch N times and prints per-molecule
# microseconds. Nothing in it computes anything -- it is a loader and a loop around sps::compute.
# ---------------------------------------------------------------------------------------------
HARNESS = r"""
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "sps.h"

static bool readMol(FILE *f, sps::Mol &m) {
  int n, nb, nr;
  if (fscanf(f, "%d %d %d", &n, &nb, &nr) != 3) return false;
  m.alloc(n, nb);
  for (int i = 0; i < n; i++) {
    int z, deg, nH, fchg, hyb, arom, nring, tval, ctag, iso, cip;
    if (fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d", &z, &deg, &nH, &fchg, &hyb, &arom, &nring,
               &tval, &ctag, &iso, &cip) != 11) { fprintf(stderr, "bad atom line\n"); exit(2); }
    m.z[i] = z; m.deg[i] = deg; m.nH[i] = nH; m.fchg[i] = fchg; m.hyb[i] = hyb;
    m.arom[i] = arom; m.nring[i] = nring; m.tval[i] = tval; m.ctag[i] = ctag; m.iso[i] = iso;
    m.cip[i] = cip;
  }
  for (int b = 0; b < nb; b++) {
    int u, v, bt, ba, bc, bs;
    if (fscanf(f, "%d %d %d %d %d %d", &u, &v, &bt, &ba, &bc, &bs) != 6) {
      fprintf(stderr, "bad bond line\n"); exit(2);
    }
    m.bu[b] = u; m.bv[b] = v; m.btype[b] = bt; m.barom[b] = ba; m.bconj[b] = bc;
    m.bstereo[b] = bs;
  }
  std::vector<int> ring;
  for (int r = 0; r < nr; r++) {
    int sz;
    if (fscanf(f, "%d", &sz) != 1) { fprintf(stderr, "bad ring line\n"); exit(2); }
    ring.resize(sz);
    for (int k = 0; k < sz; k++) {
      if (fscanf(f, "%d", &ring[k]) != 1) { fprintf(stderr, "bad ring atom\n"); exit(2); }
    }
    m.add_ring(ring.data(), sz);
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: harness <mode> <dump>\n"); return 2; }
  const std::string mode = argv[1];
  FILE *f = fopen(argv[2], "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
  std::vector<sps::Mol> mols;
  sps::Mol m;
  while (readMol(f, m)) mols.push_back(m);
  fclose(f);

  sps::detail::Work W;
  double out[sps::N_COLS];
  if (mode == "run") {
    for (auto &mm : mols) {
      sps::compute(mm, W, out);
      // the potential-stereo atom set is carried out of Work; it is what stage 2 diffs
      std::string s;
      for (int i = 0; i < mm.n; i++)
        if (W.isTet[i]) { if (!s.empty()) s += ","; s += std::to_string(i); }
      printf("%s\t%.17g\n", s.c_str(), out[sps::N_COLS - 1]);
    }
    return 0;
  }
  if (mode == "bench") {
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    std::vector<double> per(mols.size(), 0.0);
    for (int r = 0; r < reps; r++) {
      for (size_t i = 0; i < mols.size(); i++) {
        auto t0 = std::chrono::steady_clock::now();
        sps::compute(mols[i], W, out);
        auto t1 = std::chrono::steady_clock::now();
        const double us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
        if (r == 0 || us < per[i]) per[i] = us;   // min over reps: the cache-warm cost
      }
    }
    for (double u : per) printf("%.4f\n", u);
    return 0;
  }
  fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
"""


def build_harness() -> str:
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, "harness.cpp")
    exe = os.path.join(BUILD, "harness")
    with open(src, "w") as fh:
        fh.write(HARNESS)
    cmd = ["c++", "-std=c++17", "-O2", "-Wall", "-I", os.path.join(ROOT, "src", "hume_core"),
           src, "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("verify_sps: the harness did not compile.\n"
                         + " ".join(cmd) + "\n" + r.stdout + r.stderr)
    if r.stderr.strip():
        print("[compile warnings]\n" + r.stderr.strip())
    return exe


# ---------------------------------------------------------------------------------------------
# The dump. Every field is read straight off RDKit; nothing is derived here.
# ---------------------------------------------------------------------------------------------
def rings_of(mol, repaired: bool):
    """The ring set handed to the header.

    `repaired=True` is what `bindings.cpp` would pass -- `src/hume/_rings.py rings_for`, the
    canonicalised symmetrised SSSR that RingCount already uses. `repaired=False` is RDKit's own
    `RingInfo.AtomRings()`, which is what `findPotentialStereo` sees inside RDKit. The two differ
    on a handful of symmetric cages and the difference is measured rather than assumed.
    """
    if repaired:
        from hume._rings import rings_for
        return [list(r) for r in rings_for(mol)]
    return [list(r) for r in mol.GetRingInfo().AtomRings()]


def dump(mols, path: str, repaired: bool) -> None:
    out = []
    for mol in mols:
        rings = rings_of(mol, repaired)
        out.append("%d %d %d" % (mol.GetNumAtoms(), mol.GetNumBonds(), len(rings)))
        for a in mol.GetAtoms():
            cip = 0
            if a.HasProp("_CIPCode"):
                cip = 1 if a.GetProp("_CIPCode") == "R" else -1
            out.append("%d %d %d %d %d %d %d %d %d %d %d" % (
                a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs(), a.GetFormalCharge(),
                int(a.GetHybridization()), int(a.GetIsAromatic()),
                mol.GetRingInfo().NumAtomRings(a.GetIdx()), a.GetTotalValence(),
                int(a.GetChiralTag()), a.GetIsotope(), cip))
        for b in mol.GetBonds():
            out.append("%d %d %d %d %d %d" % (
                b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType()),
                int(b.GetIsAromatic()), int(b.GetIsConjugated()), int(b.GetStereo())))
        for r in rings:
            out.append("%d %s" % (len(r), " ".join(str(x) for x in r)))
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------------
_TET = Chem.StereoType.Atom_Tetrahedral


def reference_perception(mols):
    """`Chem.FindMolChiralCenters(useLegacyImplementation=False, includeUnassigned=True)`, in the
    cheap spelling `src/hume/_extract.py` already uses and validated as identical there."""
    old = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(False)
    res = []
    try:
        for m in mols:
            c = Chem.Mol(m)
            rdmolops.FindPotentialStereoBonds(c)
            itms = Chem.FindPotentialStereo(c)
            s = set()
            for k in range(len(itms)):
                si = itms[k]
                if si.type == _TET:
                    s.add(si.centeredOn)
            res.append(s)
    finally:
        Chem.SetUseLegacyStereoPerception(old)
    return res


def legacy_flag_sets(mols):
    """The LEGACY perception -- the `_ChiralityPossible` flag the pickle carries, produced exactly
    as `src/hume/_extract.py` produces it (`cleanIt=True, force=True,
    flagPossibleStereoCenters=True`, on a copy)."""
    res = []
    for m in mols:
        c = Chem.Mol(m)
        Chem.AssignStereochemistry(c, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        res.append({a.GetIdx() for a in c.GetAtoms() if a.HasProp("_ChiralityPossible")})
    return res


def python_route_timing(mols, reps: int):
    """Time `src/hume/_extract.py`'s `_potential_stereo` on the same molecules, spelled exactly as
    it is there. This is the cost the C++ header removes; measuring it in the same process on the
    same run is the only way the comparison means anything on a shared machine."""
    tet = Chem.StereoType.Atom_Tetrahedral
    bond_stereo = Chem.Bond.GetStereo
    bond_type = Chem.Bond.GetBondType
    stereonone = Chem.BondStereo.STEREONONE
    dbl = Chem.BondType.DOUBLE
    best = [float("inf")] * len(mols)
    old = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(False)
    try:
        for _ in range(reps):
            for k, m in enumerate(mols):
                t0 = time.perf_counter_ns()
                c = Chem.Mol(m)
                rdmolops.FindPotentialStereoBonds(c)
                row = [0] * c.GetNumAtoms()
                itms = Chem.FindPotentialStereo(c)
                for j in range(len(itms)):
                    si = itms[j]
                    if si.type == tet:
                        row[si.centeredOn] = 1
                nb = c.GetNumBonds()
                bonds = list(map(c.GetBondWithIdx, range(nb)))
                brow = [0] * nb
                for e, st in enumerate(map(bond_stereo, bonds)):
                    if st != stereonone and bond_type(bonds[e]) == dbl:
                        brow[e] = 1
                dt = (time.perf_counter_ns() - t0) / 1000.0
                if dt < best[k]:
                    best[k] = dt
    finally:
        Chem.SetUseLegacyStereoPerception(old)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="use only the first N corpus molecules")
    ap.add_argument("--rings", choices=("repaired", "rdkit"), default="repaired")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-report", type=int, default=15)
    args = ap.parse_args()

    corpus = json.load(open(CORPUS))
    smis = corpus["smiles"]
    strata = corpus["stratum"]
    if args.n:
        smis, strata = smis[: args.n], strata[: args.n]
    print("corpus: %d molecules, strata %s" % (len(smis), sorted(set(strata))))

    mols = [Chem.MolFromSmiles(s) for s in smis]
    bad = [i for i, m in enumerate(mols) if m is None]
    if bad:
        raise SystemExit("verify_sps: %d corpus SMILES did not parse, e.g. %r"
                         % (len(bad), smis[bad[0]]))

    # ---- stage 1: legacy vs new -------------------------------------------------------------
    new_sets = reference_perception(mols)
    leg_sets = legacy_flag_sets(mols)
    diff = [i for i in range(len(mols)) if leg_sets[i] != new_sets[i]]
    print("\n=== stage 1: the legacy flag is not the new perception ===")
    print("  molecules where _ChiralityPossible != FindPotentialStereo(Atom_Tetrahedral): "
          "%d / %d  (%.2f%%)" % (len(diff), len(mols), 100.0 * len(diff) / len(mols)))
    for i in diff[: args.max_report]:
        print("    %-60s legacy %s  new %s" % (smis[i], sorted(leg_sets[i]), sorted(new_sets[i])))

    # ---- build + run ------------------------------------------------------------------------
    exe = build_harness()
    path = os.path.join(BUILD, "dump.txt")
    dump(mols, path, repaired=(args.rings == "repaired"))
    r = subprocess.run([exe, "run", path], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("verify_sps: the harness failed.\n" + r.stdout + r.stderr)
    lines = r.stdout.rstrip("\n").split("\n")
    if len(lines) != len(mols):
        raise SystemExit("verify_sps: the harness returned %d rows for %d molecules"
                         % (len(lines), len(mols)))
    ours_sets, ours_sps = [], []
    for ln in lines:
        left, right = ln.split("\t")
        ours_sets.append(set(int(x) for x in left.split(",") if x != ""))
        ours_sps.append(float(right))

    # ---- stage 2: the perception on its own -------------------------------------------------
    print("\n=== stage 2: this header's perception vs Chem.FindPotentialStereo ===")
    mism = [i for i in range(len(mols)) if ours_sets[i] != new_sets[i]]
    print("  molecules with an IDENTICAL potential-stereocentre atom set: %d / %d  (%.4f%%)"
          % (len(mols) - len(mism), len(mols), 100.0 * (len(mols) - len(mism)) / len(mols)))
    print("  mismatched: %d   (rings = %s)" % (len(mism), args.rings))
    for i in mism[: args.max_report]:
        print("    %-70s\n      rdkit %s\n      ours  %s"
              % (smis[i], sorted(new_sets[i]), sorted(ours_sets[i])))

    # ---- stage 3: the SPS column vs the reference matrix ------------------------------------
    print("\n=== stage 3: SPS ===")
    npz = np.load(MATRIX, allow_pickle=True)
    rd_names = [str(x) for x in npz["rd_names"]]
    if "SPS" not in rd_names:
        raise SystemExit("verify_sps: matrix.npz has no SPS column in rd_names")
    col = rd_names.index("SPS")
    # matrix.npz stores float32, so it cannot settle a double-precision question on its own.
    # It is checked as the contract asks -- and the REAL bar, bit-identity in double against
    # rdkit's own SPS on the same molecules, is checked next to it.
    ref32 = np.asarray(npz["RD"])[: len(mols), col].astype(np.float64)
    from rdkit.Chem import SpacialScore
    ref64 = np.array([SpacialScore.SPS(m) for m in mols], dtype=np.float64)
    ours = np.asarray(ours_sps)

    def tally(ref, label):
        n_exact = n_9 = n_6 = n_bad = n_nan_both = n_nan_one = 0
        bad_rows = []
        for i in range(len(mols)):
            a, b = ref[i], ours[i]
            if np.isnan(a) and np.isnan(b):
                n_nan_both += 1
                continue
            if np.isnan(a) != np.isnan(b):
                n_nan_one += 1
                bad_rows.append(i)
                continue
            if a == b:
                n_exact += 1
                continue
            rel = abs(a - b) / max(abs(a), 1e-300)
            if rel <= 1e-9:
                n_9 += 1
            elif rel <= 1e-6:
                n_6 += 1
            else:
                n_bad += 1
                bad_rows.append(i)
        print("  --- %s ---" % label)
        print("    n exact (bit-identical) : %d" % n_exact)
        print("    n within 1e-9 relative  : %d" % n_9)
        print("    n within 1e-6 relative  : %d" % n_6)
        print("    n mismatched            : %d" % n_bad)
        print("    NaN on both sides       : %d" % n_nan_both)
        print("    NaN on one side only    : %d" % n_nan_one)
        for i in bad_rows[: args.max_report]:
            print("      %-68s ref %.17g  ours %.17g" % (smis[i], ref[i], ours[i]))
        return n_bad + n_nan_one

    n_bad64 = tally(ref64, "vs rdkit Chem.SpacialScore.SPS, double precision")
    tally(ref32, "vs data/dedupe2/matrix.npz (float32 store: 1e-6 is its own noise floor)")
    n_bad = n_bad64
    n_nan_one = 0
    # the float32 store cannot be tighter than ~1e-7 relative, so it is reported but not gating
    print("  agreement of the float32 store with rdkit's own double: %d of %d bit-identical"
          % (int(np.sum(ref32 == ref64)), len(mols)))

    # ---- timing -----------------------------------------------------------------------------
    print("\n=== timing (%d reps, min per molecule) ===" % args.reps)
    r = subprocess.run([exe, "bench", path, str(args.reps)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("verify_sps: the bench run failed.\n" + r.stdout + r.stderr)
    us = [float(x) for x in r.stdout.split()]
    by = {}
    for s, u in zip(strata, us):
        by.setdefault(s, []).append(u)
    for s in sorted(by):
        v = by[s]
        print("  stratum %-8s n=%5d  %7.2f us/mol   SD %6.2f" %
              (s, len(v), statistics.mean(v),
               statistics.pstdev(v) if len(v) > 1 else 0.0))
    print("  ALL      %-8s n=%5d  %7.2f us/mol   SD %6.2f" %
          ("", len(us), statistics.mean(us), statistics.pstdev(us)))
    # ...and the route it replaces, measured HERE, on the same machine under the same load, so
    # the comparison is not against a number from another day.
    py = python_route_timing(mols, args.reps)
    byp = {}
    for s, u in zip(strata, py):
        byp.setdefault(s, []).append(u)
    print("\n=== the Python route this replaces, same machine, same run ===")
    print("  (src/hume/_extract.py `_potential_stereo`: Chem.Mol copy, FindPotentialStereoBonds,"
          " FindPotentialStereo, the bond loop)")
    for s in sorted(byp):
        v = byp[s]
        print("  stratum %-8s n=%5d  %7.2f us/mol   SD %6.2f" %
              (s, len(v), statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0))
    print("  ALL      %-8s n=%5d  %7.2f us/mol   SD %6.2f" %
          ("", len(py), statistics.mean(py), statistics.pstdev(py)))
    print("  speedup: %.2fx overall" % (statistics.mean(py) / max(statistics.mean(us), 1e-9)))

    ok = (len(mism) == 0 and n_bad == 0 and n_nan_one == 0)
    print("\n%s" % ("ALL EXACT" if ok else "NOT EXACT -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print("(%.1f s)" % (time.time() - t0))
    raise SystemExit(code)
