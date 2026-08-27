"""Build the corpus for, and verify, src/hume_core/vsa_bins.h -- the VSA-binning family.

FOUR STEPS, AND TWO OF THEM NEED DIFFERENT ENVIRONMENTS.  Getting this wrong does not error, it
silently changes the oracle; see constraints.txt.

    # 1. the numeric spec.  pinned rdkit, no mordred.
    uv run --isolated --python 3.12 --with "rdkit==2025.9.2" --with "numpy==2.4.6" \
           python cpp/gen_vsa_tables.py

    # 2. the corpus + RDKit's own answers.  pinned rdkit, no mordred.
    uv run --isolated --python 3.12 --with "rdkit==2025.9.2" --with "numpy==2.4.6" \
           python cpp/verify_vsa.py corpus cpp/hard.smi

    c++ -std=c++17 -O3 -o cpp/vsa cpp/vsa.cpp && ./cpp/vsa verify cpp/vsa_mols.txt

    # 3. mordred's five columns.  mordred 1.2.0 needs numpy 1.x AND python 3.11 (distutils).
    #    Asking for mordred with numpy 2 does NOT error -- uv resolves mordred DOWN to 0.6.0.
    uv run --isolated --python 3.11 --with "mordred==1.2.0" --with "rdkit==2025.9.2" \
           --with "numpy==1.26.4" python cpp/verify_vsa.py mordred

    # 4. timing, in alternating pairs on a contended machine.
    uv run --isolated --python 3.11 --with "mordred==1.2.0" --with "rdkit==2025.9.2" \
           --with "numpy==1.26.4" python cpp/verify_vsa.py time

WHY TopoPSA IS NOT IN STEP 2.  mordred's TopoPSA is `CalcTPSA(mol)` plus mordred's OWN phosphorus
and sulfur table, which is NOT rdkit's includeSandP=true path -- mordred matches an exact multiset
of ALL incident bonds (hydrogen bonds included) against a literal dict and zeroes any charged
atom, where rdkit counts heavy-bond orders against a degree.  So only mordred can be the oracle
for it, and mordred cannot live in the pinned-numpy environment.  Step 2 writes NaN for that
column, the C++ reports it as "no reference", and step 3 checks it.

THE CACHING TRAP.  RDKit memoises Labute contributions on the molecule as `_labuteAtomContribs`,
Crippen as `_crippenLogP`, and the Python E-state as `_eStateIndices`; mordred memoises per
molecule too.  Every reference value below is taken with force=True or from a FRESH molecule, and
the timing arms re-parse.  A second pass over the same Mol object measures a dict lookup.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "vsa_mols.txt"
CPPOUT = ROOT / "vsa_cpp.txt"
PINNED_RDKIT = "2025.9.2"

# Must match vsabin::col_name() in src/hume_core/vsa_bins.h, in order.  The C++ refuses to load a
# corpus whose header disagrees, so this cannot drift silently.
COLS = (
    [f"SlogP_VSA{i}" for i in range(1, 13)]
    + [f"SMR_VSA{i}" for i in range(1, 11)]
    + [f"PEOE_VSA{i}" for i in range(1, 15)]
    + [f"EState_VSA{i}" for i in range(1, 12)]
    + [f"VSA_EState{i}" for i in range(1, 11)]
    + ["MolLogP", "MolMR", "TPSA", "TopoPSA", "LabuteASA",
       "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex"]
)

# The 62 columns of the 865 that this family owns, as blocks.classify() assigns them.  The other
# 4 that vsa_bins.h computes (SlogP_VSA9, SMR_VSA8, PEOE_VSA11-as-rdkit, MaxAbsEStateIndex,
# LabuteASA) were dropped by the r>0.99 dedupe and are verified anyway because they cost nothing.
DELIVERABLE = (
    [f"SlogP_VSA{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]]
    + [f"SMR_VSA{i}" for i in [1, 2, 3, 4, 5, 6, 7, 9, 10]]
    + [f"PEOE_VSA{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    + [f"EState_VSA{i}" for i in range(1, 12)]
    + [f"VSA_EState{i}" for i in range(1, 11)]
    + ["MolMR", "TPSA", "TopoPSA", "MolLogP", "MaxEStateIndex", "MinEStateIndex",
       "MinAbsEStateIndex"]
)

_BIT = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 4}


def _versions() -> str:
    import numpy
    import rdkit
    out = [f"python {sys.version.split()[0]}", f"rdkit {rdkit.__version__}",
           f"numpy {numpy.__version__}"]
    try:
        import mordred
        out.append(f"mordred {mordred.__version__}")
    except Exception:
        out.append("mordred (absent)")
    return "RESOLVED  " + "   ".join(out) + f"   (rdkit pin: {PINNED_RDKIT})"


def _rd(x: float) -> str:
    """Shortest round-tripping text for a double.  repr() is exact for finite doubles and gives
    'nan' / 'inf', both of which C's strtod reads back."""
    return repr(float(x))


# ------------------------------------------------------------------------------------------
# step 2: the corpus, with RDKit's per-atom and per-column answers baked in
# ------------------------------------------------------------------------------------------
def corpus(smi_path: str, limit: int | None) -> int:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors, rdPartialCharges
    from rdkit.Chem.EState import EState, EState_VSA

    RDLogger.DisableLog("rdApp.*")
    print(_versions())

    smis = [ln.split()[0] for ln in open(smi_path) if ln.strip()]
    if limit:
        smis = smis[:limit]

    mols = []
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
    print(f"{len(mols)} / {len(smis)} molecules parsed from {smi_path}")

    with open(CORPUS, "w") as f:
        f.write(f"{len(mols)} {len(COLS)}\n")
        f.write(" ".join(COLS) + "\n")
        for m in mols:
            n, nb = m.GetNumAtoms(), m.GetNumBonds()
            f.write(f"{n} {nb}\n")

            ats, hcontrib = rdMolDescriptors._CalcLabuteASAContribs(m, True)
            asa = list(ats)
            crip = rdMolDescriptors._CalcCrippenContribs(m)
            rdPartialCharges.ComputeGasteigerCharges(m)
            es = list(EState.EStateIndices(m, force=True))
            ri = m.GetRingInfo()

            for i in range(n):
                a = m.GetAtomWithIdx(i)
                try:
                    q = float(a.GetProp("_GasteigerCharge"))
                except Exception:
                    q = float("nan")
                r3 = 1 if ri.IsAtomInRingOfSize(i, 3) else 0
                f.write(f"{a.GetAtomicNum()} {a.GetDegree()} {a.GetTotalNumHs()} "
                        f"{a.GetFormalCharge()} {int(a.GetIsAromatic())} {r3} "
                        f"{_rd(q)} {_rd(asa[i])} {_rd(crip[i][0])} {_rd(crip[i][1])} "
                        f"{_rd(es[i])}\n")
            for b in map(m.GetBondWithIdx, range(nb)):
                code = _BIT.get(str(b.GetBondType()), 0)
                if b.GetIsAromatic():
                    code |= 8
                f.write(f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} {code}\n")

            f.write(_rd(hcontrib) + "\n")
            f.write(" ".join(_rd(v) for v in _reference_row(m)) + "\n")

    print(f"wrote {CORPUS}  ({CORPUS.stat().st_size / 1e6:.1f} MB)")
    return 0


def _reference_row(m) -> list[float]:
    """RDKit's own answer for every column it can supply.  force=True everywhere: RDKit caches
    each of these on the molecule and the corpus writer has already touched it."""
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.EState import EState, EState_VSA

    row: list[float] = []
    row += list(rdMolDescriptors.SlogP_VSA_(m, force=True))
    row += list(rdMolDescriptors.SMR_VSA_(m, force=True))
    row += list(rdMolDescriptors.PEOE_VSA_(m, force=True))
    row += list(EState_VSA.EState_VSA_(m, force=True))
    row += list(EState_VSA.VSA_EState_(m, force=True))
    logp, mr = rdMolDescriptors.CalcCrippenDescriptors(m, force=True)
    row += [logp, mr, rdMolDescriptors.CalcTPSA(m, force=True)]
    row += [float("nan")]                                  # TopoPSA: mordred is the oracle
    row += [rdMolDescriptors.CalcLabuteASA(m, force=True)]
    es = list(EState.EStateIndices(m, force=True))
    row += [max(es), min(es), max(abs(x) for x in es), min(abs(x) for x in es)]
    assert len(row) == len(COLS), (len(row), len(COLS))
    return row


# ------------------------------------------------------------------------------------------
# step 3: mordred's five columns
# ------------------------------------------------------------------------------------------
def mordred_check(smi_path: str, limit: int | None) -> int:
    """Compare ./cpp/vsa dump against mordred 1.2.0 for the five columns mordred owns.

    mordred's MoeType descriptors resolve their function with
    `getattr(rdkit.Chem.MolSurf | rdkit.Chem.EState.EState_VSA, str(self))`, so PEOE_VSA11,
    SMR_VSA1 and EState_VSA1 are the SAME code path and the SAME bin edges as the rdkit_* VSA
    columns -- confirmed in mordred/MoeType.py, which is 120 lines and contains no arithmetic.
    SLogP is `Crippen.MolLogP(mol)` verbatim (mordred/SLogP.py).  Only TopoPSA is mordred's own.

    THE ONE PLACE MORDRED IS NOT ASKING THE SAME QUESTION: every one of these five descriptors
    declares `explicit_hydrogens = False`, so mordred hands the function `Chem.RemoveHs(mol)`,
    not the molecule as parsed.  This step measures how many molecules that changes, rather than
    assuming it changes none.
    """
    from mordred import Calculator
    from mordred import MoeType, SLogP, TopoPSA
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    print(_versions())

    if not CPPOUT.exists():
        print(f"missing {CPPOUT} -- run  ./cpp/vsa dump {CORPUS} {CPPOUT}")
        return 2
    lines = CPPOUT.read_text().splitlines()
    nmol, ncol = (int(x) for x in lines[0].split())
    names = lines[1].split()
    assert names == COLS, "cpp/vsa dump header does not match COLS"
    idx = {nm: i for i, nm in enumerate(names)}
    got = [[float(x) for x in ln.split()] for ln in lines[2:2 + nmol]]

    want_cols = ["PEOE_VSA11", "SMR_VSA1", "EState_VSA1", "MolLogP", "TopoPSA"]
    descs = [MoeType.PEOE_VSA(11), MoeType.SMR_VSA(1), MoeType.EState_VSA(1),
             SLogP.SLogP(), TopoPSA.TopoPSA(no_only=False)]
    calc = Calculator(descs)
    print("mordred descriptors: " + ", ".join(str(d) for d in descs))

    smis = [ln.split()[0] for ln in open(smi_path) if ln.strip()]
    if limit:
        smis = smis[:limit]

    ok = [0] * len(want_cols)
    tot = [0] * len(want_cols)
    worst = [0.0] * len(want_cols)
    worst_smi = [""] * len(want_cols)
    n_removehs = 0
    k = 0
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if Chem.RemoveHs(m).GetNumAtoms() != m.GetNumAtoms():
            n_removehs += 1
        vals = calc(Chem.MolFromSmiles(smi))          # fresh: mordred memoises per molecule
        for c, name in enumerate(want_cols):
            v = vals[c]
            try:
                v = float(v)
            except Exception:
                continue
            g = got[k][idx[name]]
            tot[c] += 1
            if g == v or (math.isnan(g) and math.isnan(v)):
                ok[c] += 1
            else:
                d = abs(g - v)
                if d > worst[c]:
                    worst[c], worst_smi[c] = d, smi
        k += 1

    print(f"\nvs mordred 1.2.0, {k} molecules")
    print(f"  molecules whose atom count CHANGES under mordred's Chem.RemoveHs: {n_removehs}")
    bad = 0
    for c, name in enumerate(want_cols):
        good = ok[c] == tot[c]
        bad += 0 if good else 1
        print(f"  {name:14s} {ok[c]:7d} / {tot[c]:<7d} {'EXACT ' if good else ' !!!! '}"
              f"max|d| {worst[c]:.3e}  {worst_smi[c][:60]}")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------
# step 4: timing
# ------------------------------------------------------------------------------------------
def timing(smi_path: str, limit: int | None) -> int:
    """ALTERNATING PAIRS on a CONTENDED machine.

    The arms are interleaved rep by rep rather than run back to back, so a load spike that lands
    during one arm lands during the other too.  Each arm re-parses the SMILES because RDKit and
    mordred both memoise on the molecule; timing a warm Mol measures a dict lookup.  The SMILES
    parse itself is measured as its own arm and is NOT subtracted -- it is charged to both.
    """
    from mordred import Calculator, MoeType, SLogP, TopoPSA
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.EState import EState_VSA

    RDLogger.DisableLog("rdApp.*")
    print(_versions())
    smis = [ln.split()[0] for ln in open(smi_path) if ln.strip()][:limit or 2000]
    mols_src = [s for s in smis if Chem.MolFromSmiles(s) is not None]

    def arm_parse():
        for s in mols_src:
            Chem.MolFromSmiles(s)

    def arm_rdkit():
        for s in mols_src:
            m = Chem.MolFromSmiles(s)
            rdMolDescriptors.SlogP_VSA_(m)
            rdMolDescriptors.SMR_VSA_(m)
            rdMolDescriptors.PEOE_VSA_(m)
            EState_VSA.EState_VSA_(m)
            EState_VSA.VSA_EState_(m)
            rdMolDescriptors.CalcCrippenDescriptors(m)
            rdMolDescriptors.CalcTPSA(m)

    calc = Calculator([MoeType.PEOE_VSA(11), MoeType.SMR_VSA(1), MoeType.EState_VSA(1),
                       SLogP.SLogP(), TopoPSA.TopoPSA(no_only=False)])

    def arm_mordred():
        for s in mols_src:
            calc(Chem.MolFromSmiles(s))

    arms = [("SMILES parse alone", arm_parse),
            ("rdkit, the 57 VSA columns + Crippen + TPSA", arm_rdkit),
            ("mordred, its 5 columns", arm_mordred)]
    reps = {name: [] for name, _ in arms}
    for _ in range(5):
        for name, fn in arms:
            t0 = time.perf_counter()
            fn()
            reps[name].append((time.perf_counter() - t0) * 1e6 / len(mols_src))
    print(f"\n{len(mols_src)} molecules, 5 alternating reps, CONTENDED machine")
    for name, _ in arms:
        r = sorted(reps[name])
        med = r[len(r) // 2]
        print(f"  {name:46s} median {med:9.2f} us/mol   min {r[0]:9.2f}  max {r[-1]:9.2f}")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "corpus"
    smi = argv[2] if len(argv) > 2 else str(ROOT / "hard.smi")
    lim = int(argv[3]) if len(argv) > 3 else None
    if cmd == "corpus":
        return corpus(smi, lim)
    if cmd == "mordred":
        return mordred_check(smi, lim)
    if cmd == "time":
        return timing(smi, lim)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
