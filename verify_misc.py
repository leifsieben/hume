#!/usr/bin/env python
"""Verification harness for src/hume_core/misc_ext.h -- the 81 columns of F_misc.

TEST HARNESS ONLY.  Nothing here computes a descriptor: the C++ header does, through a driver
this script generates and compiles into build_misc/.  Python's only jobs are (a) to drive the
BOUNDARY -- `src/hume._extract.extract`, the same arrays bindings.cpp would hand the header --
(b) to call the reference packages for ground truth, and (c) to grade.  AGENT_CONTRACT house
rule 1.

  .venv/bin/python verify_misc.py [--n N] [--rebuild] [--md-ref DIR]

GROUND TRUTH.
  * The 14 RDKit columns are recomputed live, in float64, from the same rdkit the repo pins.
  * The 67 mordred columns come from an mordred 1.2.0 run under .venv-mordred (float64), cached
    as .npz shards.  `data/dedupe2/matrix.npz` holds the same values but only as float32, which
    cannot support a bit-identical claim; it is used as a cross-check that the float64 reference
    and the shipped matrix are the same numbers.

REPORTED PER COLUMN, over all molecules: n bit-identical, n within 1e-9 relative, n within 1e-6,
n mismatched, NaN agreement, and the SMILES / reference / ours of every mismatch.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build_misc")
GROUP = "F_misc"

# ---------------------------------------------------------------------------------------------
# The driver.  Reads the boundary arrays as a binary blob (float64 / int32, no text round trip --
# see src/hume/_extract.py on why a text field can desync a whole file) and writes one
# little-endian float64 row of N_COLS per molecule.
# ---------------------------------------------------------------------------------------------
DRIVER = r"""
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "misc_ext.h"

// blob layout, all little-endian:
//   int32 n_mol
//   per molecule: int32 n, nb, chg_ok, n_rings
//                 int32 atom_i[n*13]
//                 double atom_d[n*2]
//                 int32 bond_i[nb*6]
//                 double bond_d[nb]
static bool rd(FILE *f, void *p, size_t nbytes) { return fread(p, 1, nbytes, f) == nbytes; }

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: driver in.bin out.bin [reps]\n"); return 2; }
  const int reps = argc > 3 ? atoi(argv[3]) : 1;
  FILE *f = fopen(argv[1], "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  int32_t nmol = 0;
  if (!rd(f, &nmol, 4)) return 2;
  std::vector<miscext::Mol> mols((size_t)nmol);
  for (int k = 0; k < nmol; ++k) {
    int32_t hdr[4];
    if (!rd(f, hdr, 16)) { fprintf(stderr, "short read at %d\n", k); return 2; }
    const int n = hdr[0], nb = hdr[1];
    std::vector<int32_t> ai((size_t)n * 13), bi((size_t)nb * 6);
    std::vector<double> ad((size_t)n * 2), bd((size_t)nb);
    if (n && !rd(f, ai.data(), ai.size() * 4)) return 2;
    if (n && !rd(f, ad.data(), ad.size() * 8)) return 2;
    if (nb && !rd(f, bi.data(), bi.size() * 4)) return 2;
    if (nb && !rd(f, bd.data(), bd.size() * 8)) return 2;
    miscext::build_from_rows(mols[k], n, nb, ai.data(), 13, ad.data(), 2, bi.data(), 6,
                             bd.data(), hdr[2], hdr[3]);
  }
  fclose(f);

  miscext::Scratch S;
  std::vector<double> out((size_t)nmol * miscext::N_COLS);
  std::vector<double> per(nmol, 1e300);          // best-of-reps microseconds per molecule
  typedef std::chrono::steady_clock clk;
  for (int r = 0; r < reps; ++r) {
    for (int k = 0; k < nmol; ++k) {
      const clk::time_point t0 = clk::now();
      try {
        miscext::compute(mols[k], S, &out[(size_t)k * miscext::N_COLS]);
      } catch (const std::exception &e) {
        fprintf(stderr, "molecule %d: %s\n", k, e.what());
        for (int c = 0; c < miscext::N_COLS; ++c)
          out[(size_t)k * miscext::N_COLS + c] = std::numeric_limits<double>::quiet_NaN();
      }
      const double us =
          std::chrono::duration<double, std::micro>(clk::now() - t0).count();
      if (us < per[k]) per[k] = us;
    }
  }
  {
    double tot = 0.0;
    for (int k = 0; k < nmol; ++k) tot += per[k];
    fprintf(stderr, "%.3f us/mol (mean, best of %d per molecule)\n", tot / nmol, reps);
    FILE *tf = fopen((std::string(argv[2]) + ".times").c_str(), "wb");
    fwrite(per.data(), 8, per.size(), tf);
    fclose(tf);
  }

  // ---- cost attribution ---------------------------------------------------------------------
  // The three families this header REUSES are already computed by bindings.cpp for other
  // columns, so their cost is only new if the wiring calls them twice.  Timed separately here
  // (on the same molecules, through the same public entry points) so NOTES_misc.md can quote
  // what F_misc actually adds rather than what it spends.
  {
    struct Part { const char *name; double us; };
    Part parts[6] = {{"chi.h enumeration", 0}, {"pathcount.h", 0}, {"topomisc walkTraces", 0},
                     {"fr_* matcher", 0}, {"BertzCT", 0}, {"LogEE_A", 0}};
    std::vector<double> buf(64);
    std::vector<int32_t> ar, br;
    std::vector<double> pp[6];
    for (int i = 0; i < 6; ++i) pp[i].assign(nmol, 1e300);
   for (int rep = 0; rep < reps; ++rep) {
    double acc[6] = {0, 0, 0, 0, 0, 0};
    for (int k = 0; k < nmol; ++k) {
      const miscext::Mol &m = mols[k];
      ar.resize((size_t)m.n * 4);
      for (int i = 0; i < m.n; ++i) {
        int32_t *r = &ar[(size_t)i * 4];
        r[0] = m.z[i]; r[1] = m.deg[i]; r[2] = m.nH[i]; r[3] = m.fchg[i];
      }
      br.resize((size_t)m.nb * 2);
      for (int e = 0; e < m.nb; ++e) { br[(size_t)e*2] = m.bu[e]; br[(size_t)e*2+1] = m.bv[e]; }
      clk::time_point t = clk::now();
      chisub::build_from_rows(S.chim, m.n, m.nb, ar.data(), 4, br.data(), 2);
      chisub::compute(S.chim, S.chibuf.data(), S.chis);
      { const double dt = std::chrono::duration<double, std::micro>(clk::now() - t).count();
        acc[0] += dt; if (dt < pp[0][k]) pp[0][k] = dt; }
      t = clk::now();
      pathcount::build_from_rows(S.pcm, m.n, m.nb, br.data(), 2, 0, 1, m.bord.data(),
                                 m.z.data(), 1, 0);
      pathcount::compute(S.pcm, S.pcbuf.data(), S.pcs);
      { const double dt = std::chrono::duration<double, std::micro>(clk::now() - t).count();
        acc[1] += dt; if (dt < pp[1][k]) pp[1][k] = dt; }
      ar.resize((size_t)m.n * 3);
      for (int i = 0; i < m.n; ++i) { ar[(size_t)i*3] = m.z[i]; ar[(size_t)i*3+2] = m.nH[i]; }
      t = clk::now();
      topomisc::build_from_rows(S.tpm, m.n, m.nb, ar.data(), 3, br.data(), 2);
      { int64_t tr[11], sums[11]; if (m.n) topomisc::detail::walkTraces(S.tpm, S.tps, tr, sums); }
      { const double dt = std::chrono::duration<double, std::micro>(clk::now() - t).count();
        acc[2] += dt; if (dt < pp[2][k]) pp[2][k] = dt; }
      t = clk::now();
      {
        S.fm.alloc(m.n, m.nb);
        for (int i = 0; i < m.n; ++i) {
          S.fm.z[i] = m.z[i]; S.fm.deg[i] = m.deg[i]; S.fm.nH[i] = m.nH[i];
          S.fm.fchg[i] = m.fchg[i]; S.fm.arom[i] = m.arom[i]; S.fm.nring[i] = m.nring[i];
          S.fm.tval[i] = m.tval[i]; S.fm.iso[i] = m.iso[i];
        }
        for (int e = 0; e < m.nb; ++e) {
          S.fm.bu[e] = m.bu[e]; S.fm.bv[e] = m.bv[e];
          S.fm.border[e] = m.btype[e]; S.fm.bring[e] = m.bring[e];
        }
        S.fm.finish();
        fragmatch::countAll(S.fm, S.fmt, S.fcount.data());
      }
      { const double dt = std::chrono::duration<double, std::micro>(clk::now() - t).count();
        acc[3] += dt; if (dt < pp[3][k]) pp[3][k] = dt; }
      t = clk::now();
      miscext::detail::bertzCT(m, S);
      { const double dt = std::chrono::duration<double, std::micro>(clk::now() - t).count();
        acc[4] += dt; if (dt < pp[4][k]) pp[4][k] = dt; }
      t = clk::now();
      miscext::detail::logEE_A(m, S);
      { const double dt = std::chrono::duration<double, std::micro>(clk::now() - t).count();
        acc[5] += dt; if (dt < pp[5][k]) pp[5][k] = dt; }
    }
    for (int i = 0; i < 6; ++i)
      if (rep == 0 || acc[i] < parts[i].us) parts[i].us = acc[i];
   }
    // Reported with the SAME statistic as the whole-group number above -- best of `reps` PER
    // MOLECULE -- so the two are comparable.  On a loaded machine the per-molecule minimum is
    // much the better estimator: a whole-pass total is dominated by whichever molecules were
    // descheduled during that pass.
    for (int i = 0; i < 6; ++i) {
      double s2 = 0.0;
      for (int k = 0; k < nmol; ++k) s2 += pp[i][k];
      fprintf(stderr, "  part %-22s %8.2f us/mol (best of %d per molecule; %8.2f best whole "
                      "pass)\n", parts[i].name, s2 / nmol, reps, parts[i].us / nmol);
    }
  }
  FILE *g = fopen(argv[2], "wb");
  fwrite(out.data(), 8, out.size(), g);
  fclose(g);
  // the column names, so the harness cannot mis-associate a column with a name
  FILE *h = fopen((std::string(argv[2]) + ".names").c_str(), "w");
  for (int c = 0; c < miscext::N_COLS; ++c) fprintf(h, "%s\n", miscext::col_name(c));
  fclose(h);
  return 0;
}
"""


def build(rebuild):
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, "driver_misc.cpp")
    exe = os.path.join(BUILD, "driver_misc")
    with open(src, "w") as fh:
        fh.write(DRIVER)
    if rebuild or not os.path.exists(exe) or os.path.getmtime(exe) < max(
            os.path.getmtime(src),
            os.path.getmtime(os.path.join(ROOT, "src/hume_core/misc_ext.h"))):
        cmd = ["c++", "-std=c++17", "-O2", "-Wall", "-Wno-unused-function",
               "-I", os.path.join(ROOT, "src/hume_core"), src, "-o", exe]
        print("$ " + " ".join(cmd))
        subprocess.check_call(cmd)
    return exe


def dump(mols, path):
    """The BOUNDARY's own arrays -- src/hume/_extract.extract -- written as a flat blob."""
    from hume import _extract

    b = _extract.extract(mols, stereo=False)
    moff = b.rings.ring_moff
    with open(path, "wb") as fh:
        fh.write(np.int32(len(mols)).tobytes())
        for k in range(len(mols)):
            a0, a1 = int(b.atom_off[k]), int(b.atom_off[k + 1])
            e0, e1 = int(b.bond_off[k]), int(b.bond_off[k + 1])
            nrings = int(moff[k + 1]) - int(moff[k])
            fh.write(np.array([a1 - a0, e1 - e0, int(b.chg_ok[k]), nrings],
                              dtype=np.int32).tobytes())
            fh.write(np.ascontiguousarray(b.atom_i[a0:a1], dtype=np.int32).tobytes())
            fh.write(np.ascontiguousarray(b.atom_d[a0:a1], dtype=np.float64).tobytes())
            fh.write(np.ascontiguousarray(b.bond_i[e0:e1], dtype=np.int32).tobytes())
            fh.write(np.ascontiguousarray(b.bond_d[e0:e1], dtype=np.float64).tobytes())


# ---------------------------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------------------------
RD_COLS = ["MinPartialCharge", "MinAbsPartialCharge", "MaxAbsPartialCharge", "ExactMolWt",
           "fr_lactam", "fr_benzodiazepine", "fr_barbitur", "fr_azo", "fr_nitro_arom",
           "fr_phenol_noOrthoHbond", "fr_phos_ester", "Chi0", "NumValenceElectrons",
           "MaxPartialCharge", "BertzCT"]


def rdkit_reference(mols):
    from rdkit.Chem import Descriptors, Fragments, GraphDescriptors

    fn = {
        "MinPartialCharge": Descriptors.MinPartialCharge,
        "MaxPartialCharge": Descriptors.MaxPartialCharge,
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
        "ExactMolWt": Descriptors.ExactMolWt,
        "NumValenceElectrons": Descriptors.NumValenceElectrons,
        "Chi0": GraphDescriptors.Chi0,
        "BertzCT": GraphDescriptors.BertzCT,
    }
    for nm in ("fr_lactam", "fr_benzodiazepine", "fr_barbitur", "fr_azo", "fr_nitro_arom",
               "fr_phenol_noOrthoHbond", "fr_phos_ester"):
        fn[nm] = getattr(Fragments, nm)
    X = np.full((len(mols), len(RD_COLS)), np.nan)
    for i, m in enumerate(mols):
        for j, nm in enumerate(RD_COLS):
            try:
                X[i, j] = float(fn[nm](m))
            except Exception:
                X[i, j] = np.nan
    return X


# ---------------------------------------------------------------------------------------------
# The mordred reference generator.  Written out and run under .venv-mordred (mordred 1.2.0 needs
# python 3.11 and numpy 1.x and cannot share the project venv), sharded because mordred's own
# `nproc` path needs a multiprocessing Manager that does not come up on this box.
# ---------------------------------------------------------------------------------------------
MD_REF_SRC = r'''
import json, os, sys
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred.error import MissingValueBase

ROOT, OUT = sys.argv[1], sys.argv[2]
LO, HI = int(sys.argv[3]), int(sys.argv[4])
WANT = json.load(open(os.path.join(ROOT, "results/dedupe2/agent_groups.json")))["F_misc"]
by_name = {}
for d in Calculator(descriptors, ignore_3D=True).descriptors:
    by_name.setdefault(str(d), d)
sel = [by_name[n] for n in WANT if n in by_name]
calc = Calculator(sel)
smiles = json.load(open(os.path.join(ROOT, "data/dedupe2/corpus.json")))["smiles"][LO:HI]
mols = [Chem.MolFromSmiles(s) for s in smiles]
assert all(m is not None for m in mols)
X = np.full((len(mols), len(sel)), np.nan)
for i, r in enumerate(calc.map(mols, nproc=1, quiet=True)):
    for j, v in enumerate(r):
        X[i, j] = np.nan if isinstance(v, MissingValueBase) else float(v)
np.savez(OUT, X=X, names=np.array([str(d) for d in sel], dtype=object), lo=LO)
print("wrote %s %s" % (OUT, X.shape))
'''


def gen_md_ref(nmol, shards=8):
    """Run mordred 1.2.0 under .venv-mordred and cache float64 references in build_misc/."""
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, "md_ref.py")
    with open(src, "w") as fh:
        fh.write(MD_REF_SRC)
    py = os.path.join(ROOT, ".venv-mordred/bin/python")
    if not os.path.exists(py):
        sys.exit("no .venv-mordred -- see AGENT_CONTRACT.md on the two interpreters")
    step = (nmol + shards - 1) // shards
    procs = []
    for i in range(shards):
        lo, hi = i * step, min((i + 1) * step, nmol)
        if lo >= hi:
            break
        out = os.path.join(BUILD, "md_ref_%d.npz" % i)
        procs.append(subprocess.Popen([py, src, ROOT, out, str(lo), str(hi)]))
    bad = sum(1 for p in procs if p.wait() != 0)
    if bad:
        sys.exit("%d mordred shard(s) failed" % bad)
    print("mordred reference: %d shards in %s" % (len(procs), BUILD))


def mordred_reference(mdref_dir, nmol):
    """Load the float64 mordred shards written by the generator under .venv-mordred."""
    shards = sorted(f for f in os.listdir(mdref_dir) if f.startswith("md_ref_")
                    and f.endswith(".npz") and "test" not in f)
    if not shards:
        return None, None
    parts, names = [], None
    for s in shards:
        z = np.load(os.path.join(mdref_dir, s), allow_pickle=True)
        parts.append((int(z["lo"]), z["X"]))
        names = [str(x) for x in z["names"]]
    parts.sort()
    X = np.concatenate([p[1] for p in parts], axis=0)
    return X[:nmol], names


# ---------------------------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------------------------
def grade(name, ref, got, smiles, show=25):
    nan_ref, nan_got = np.isnan(ref), np.isnan(got)
    both_nan = int(np.sum(nan_ref & nan_got))
    nan_disagree = np.where(nan_ref != nan_got)[0]
    fin = ~nan_ref & ~nan_got
    exact = int(np.sum(ref[fin] == got[fin]))
    d = np.abs(ref[fin] - got[fin])
    scale = np.maximum(np.abs(ref[fin]), np.abs(got[fin]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, d / scale, 0.0)
    e9 = int(np.sum(rel <= 1e-9))
    e6 = int(np.sum(rel <= 1e-6))
    bad_idx = np.where(fin)[0][rel > 1e-6]
    n_bad = len(bad_idx) + len(nan_disagree)
    worst = float(np.max(rel)) if len(rel) else 0.0
    status = "EXACT" if (exact == int(np.sum(fin)) and len(nan_disagree) == 0) else (
        "rel<=1e-9" if (e9 == int(np.sum(fin)) and len(nan_disagree) == 0) else "MISMATCH")
    print("  %-24s %-9s exact %6d/%6d  1e-9 %6d  1e-6 %6d  bad %5d  nan(both) %5d "
          "nan(disagree) %4d  worst-rel %.3g"
          % (name, status, exact, int(np.sum(fin)), e9, e6, n_bad, both_nan,
             len(nan_disagree), worst))
    for i in list(nan_disagree[:show]):
        print("      NaN disagreement  %s   ref %r  ours %r" % (smiles[i], ref[i], got[i]))
    if len(bad_idx) > show or len(nan_disagree) > show:
        print("      (listing the first %d of each; build_misc/grade.json has the counts)" % show)
    for i in list(bad_idx[:show]):
        print("      %s   ref %.17g  ours %.17g  rel %.3g"
              % (smiles[i], ref[i], got[i], abs(ref[i] - got[i]) /
                 max(abs(ref[i]), abs(got[i]), 1e-300)))
    return dict(name=name, status=status, exact=exact, n=int(np.sum(fin)), e9=e9, e6=e6,
                bad=n_bad, nan_both=both_nan, nan_bad=len(nan_disagree), worst=worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="first N molecules (0 = all)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--md-ref", default=os.environ.get("MD_REF_DIR", ""))
    ap.add_argument("--only", default="", help="comma-separated column names to report")
    ap.add_argument("--gen-md-ref", action="store_true",
                    help="(re)generate the float64 mordred reference under .venv-mordred first")
    args = ap.parse_args()

    from rdkit import Chem

    smiles = json.load(open(os.path.join(ROOT, "data/dedupe2/corpus.json")))["smiles"]
    if args.n:
        smiles = smiles[:args.n]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols), "corpus molecule failed to parse"
    print("corpus: %d molecules" % len(mols))

    if args.gen_md_ref:
        gen_md_ref(len(smiles))

    exe = build(args.rebuild)
    blob = os.path.join(BUILD, "mols.bin")
    outb = os.path.join(BUILD, "values.bin")
    t0 = time.time()
    dump(mols, blob)
    print("boundary extract + dump: %.1f s" % (time.time() - t0))
    subprocess.check_call([exe, blob, outb, str(args.reps)])
    names = [l.strip() for l in open(outb + ".names")]
    got = np.fromfile(outb, dtype=np.float64).reshape(len(mols), len(names))

    # per-stratum timing, from the driver's per-molecule clock
    times = np.fromfile(outb + ".times", dtype=np.float64)
    mz0 = np.load(os.path.join(ROOT, "data/dedupe2/matrix.npz"), allow_pickle=True)
    strata = [str(x) for x in mz0["stratum"]][:len(mols)]
    print("\nper-molecule cost of the whole 81-column group (best of %d):" % args.reps)
    for s in sorted(set(strata)):
        idx = [i for i, v in enumerate(strata) if v == s]
        t = times[idx]
        print("  stratum %-12s n=%5d   %8.2f +- %8.2f us/mol   median %7.2f  max %9.2f"
              % (s, len(idx), t.mean(), t.std(), np.median(t), t.max()))
    print("  ALL          %-4s n=%5d   %8.2f +- %8.2f us/mol"
          % ("", len(times), times.mean(), times.std()))

    want = json.load(open(os.path.join(ROOT, "results/dedupe2/agent_groups.json")))[GROUP]
    assert names == want, "column order drifted from agent_groups.json[%r]" % GROUP

    only = set(x for x in args.only.split(",") if x)
    rows = []

    print("\n--- RDKit-defined columns (reference recomputed live, float64) ---")
    t0 = time.time()
    cache = os.path.join(BUILD, "rd_ref_%d.npy" % len(mols))
    if os.path.exists(cache):
        R = np.load(cache)
        print("rdkit reference: cached (%s)" % cache)
    else:
        R = rdkit_reference(mols)
        np.save(cache, R)
        print("rdkit reference: %.1f s" % (time.time() - t0))
    for j, nm in enumerate(RD_COLS):
        if only and nm not in only:
            continue
        rows.append(grade(nm, R[:, j], got[:, names.index(nm)], smiles))

    mdref_dir = args.md_ref or os.path.join(ROOT, "build_misc")
    M, mnames = mordred_reference(mdref_dir, len(mols))
    if M is None:
        print("\n!! no mordred reference shards in %s -- mordred columns NOT graded" % mdref_dir)
    else:
        print("\n--- mordred-defined columns (mordred 1.2.0 under .venv-mordred, float64) ---")
        for j, nm in enumerate(mnames):
            if only and nm not in only:
                continue
            if nm not in names:
                continue
            rows.append(grade(nm, M[:, j], got[:, names.index(nm)], smiles))

    # The shipped float32 matrix, as a cross-check that the float64 reference IS the reference.
    mz = np.load(os.path.join(ROOT, "data/dedupe2/matrix.npz"), allow_pickle=True)
    md_names = [str(x) for x in mz["md_names"]]
    if M is not None:
        worst = 0.0
        for j, nm in enumerate(mnames):
            if nm not in md_names:
                continue
            a = mz["MD"][:len(mols), md_names.index(nm)].astype(np.float64)
            b = M[:, j]
            f = ~np.isnan(a) & ~np.isnan(b)
            sc = np.maximum(np.abs(a[f]), 1e-30)
            worst = max(worst, float(np.max(np.abs(a[f] - b[f]) / sc)) if f.any() else 0.0)
        print("\nfloat64 mordred reference vs data/dedupe2/matrix.npz (float32): "
              "worst relative %.3g  (float32 eps is 1.2e-7)" % worst)

    nex = sum(1 for r in rows if r["status"] == "EXACT")
    n9 = sum(1 for r in rows if r["status"] == "rel<=1e-9")
    print("\nSUMMARY  %d columns graded: %d EXACT, %d exact-modulo-fp(<=1e-9), %d MISMATCH"
          % (len(rows), nex, n9, len(rows) - nex - n9))
    for r in rows:
        if r["status"] == "MISMATCH":
            print("  MISMATCH %-24s bad %d / %d   worst rel %.3g"
                  % (r["name"], r["bad"], len(mols), r["worst"]))
    json.dump(rows, open(os.path.join(BUILD, "grade.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
