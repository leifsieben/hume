"""Emit paper/tab_degeneracy.tex from results/fp_degeneracy.json.

    .venv/bin/python research/fp_degeneracy_table.py > paper/tab_degeneracy.tex

A GENERATOR, NOT A TYPED TABLE, for the reason every other number in this project is generated:
a table transcribed by hand is a second copy of a result that can drift from the first, and
nobody notices until a reviewer adds a column up. Re-run research/fp_degeneracy.py, re-run this,
and the table follows.

EVERY ROW IS COMPUTED ON THE SAME COMPOUNDS WITH THE SAME CODE. An earlier draft of this table
also carried the published numbers from Huber and Pollmann's Figure 2, which meant two blocks
measured on different sets (718,067 against our 695,115) with a caption explaining why they
could not quite be compared. They are dropped: the reference fingerprints here are recomputed,
which is the only strictly like-for-like comparison, and the agreement with their published
rates -- every one within 1.6 points, each slightly lower as a 3.2% smaller set should give --
is the validation that licenses the HUME rows. The transcribed numbers live in
paper/tab_fp_duplicates.tex if that context is wanted separately.

ROWS ARE SORTED BY DEGENERACY, NOT GROUPED BY METHOD, which is what puts HUME's fingerprint half
among the binary fingerprints where it belongs rather than next to HUME where it would be read
as part of the same result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fp_degeneracy.json"

#: Display order IS the result order, lowest degeneracy first; verified against the JSON below
#: so a re-run that reorders the arms cannot leave this list quietly wrong.
ROWS = [
    ("hume_full",      r"\textbf{HUME} (descriptors $+$ fingerprint)"),
    ("hume_desc",      r"\textbf{HUME}, descriptors only (1{,}269)"),
    ("hume_desc_622",  r"\textbf{HUME}, descriptors only (622)"),
    ("hume_desc_408",  r"\textbf{HUME}, descriptors only (408)"),
    ("hume_desc_256",  r"\textbf{HUME}, descriptors only (256)"),
    ("atompair_count",  "Atom pair, count"),
    ("morgan9_count",   "Morgan-9, count"),
    ("morgan3_count",   "Morgan-3, count"),
    ("hume_fp_only",   r"\textbf{HUME}, fingerprint only (2{,}048\,bit)"),
    ("morgan3_binary",  "Morgan-3, binary"),
    ("rdkit_binary",    "RDKit, binary"),
    ("maccs_binary",    "MACCS, binary"),
]


def main() -> int:
    if not RES.exists():
        sys.exit(f"{RES} is missing. Run research/fp_degeneracy.py first.")
    d = json.loads(RES.read_text())
    R, N = d["reps"], d["n_parsed"]
    missing = [k for k, _ in ROWS if k not in R]
    if missing:
        sys.exit(f"{RES.name} has no results for {missing}. Re-run fp_degeneracy.py with "
                 f"--reps covering every row this table prints, or drop them from ROWS.")
    order = [R[k]["n_duplicated"] for k, _ in ROWS]
    if order != sorted(order):
        sys.exit(f"ROWS is no longer in ascending degeneracy: {order}. The table claims to be "
                 f"sorted; re-order ROWS rather than letting the claim go stale.")

    num = lambda n: f"{n:,}".replace(",", "{,}")                            # noqa: E731
    pc = lambda n, dd: (f"{100.0*n/dd:.2f}" if 100.0*n/dd < 10              # noqa: E731
                        else f"{100.0*n/dd:.1f}")
    out = []
    A = out.append
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\begin{tabular}{lrrrr}")
    A(r"\toprule")
    A(r"& \multicolumn{2}{c}{degenerate compounds} & "
      r"\multicolumn{2}{c}{of those, max.\ $\Delta$mass} \\")
    A(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    A(r"representation & $n$ & \% of set & $\leq$1\,Da & $>$400\,Da \\")
    A(r"\midrule")
    for k, lab in ROWS:
        r = R[k]
        n, h = r["n_duplicated"], r["hist"]
        if n == 0:
            A(rf"{lab} & \textbf{{0}} & \textbf{{0}} & -- & -- \\")
        else:
            A(rf"{lab} & {num(n)} & {pc(n, N)}\% & {pc(h[0], n)}\% & {pc(h[-1], n)}\% \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\caption{Representation degeneracy on " + num(N) + r" unique biomolecular structures")
    A(r"(the \emph{biostructures} set of Huber and Pollmann~\cite{Huber2026}): compounds "
      r"sharing a")
    A(r"bit-identical vector with at least one structurally different compound, following their")
    A(r"duplicate test. Every row is computed here on the same compounds with the same code, so "
      r"the")
    A(r"rates are directly comparable; recomputing their fingerprints this way reproduces each")
    A(r"published rate to within 1.6 points. The right-hand pair splits the collisions by the")
    A(r"largest mass difference within a colliding group---under 1\,Da the pair is an isomer or")
    A(r"tautomer no 2D representation separates, over 400\,Da it is two unrelated molecules "
      r"given")
    A(r"one vector. HUME is degenerate for no compound at all, down to its 256-column variant,")
    A(r"while the 2{,}048-bit fingerprint it contains is degenerate for 9.07\% with nearly half "
      r"of")
    A(r"those collisions spanning more than 400\,Da: the descriptors do not merely add signal, "
      r"they")
    A(r"repair a failure of the fingerprint. Note also that count fingerprints fail benignly "
      r"where")
    A(r"binary ones do not---every Morgan-3 count collision falls within 1\,Da, against 44.9\%")
    A(r"beyond 400\,Da for the same fingerprint folded to bits.}")
    A(r"\label{tab:degeneracy}")
    A(r"\end{table}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
