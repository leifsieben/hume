"""RDKit molecules -> flat numpy arrays, batched. The Python half of the boundary.

This is `cpp/export_predict.py` with the text serialisation deleted. That file is the format the
verified C++ consumes, so this one reproduces its *content* field for field; what it does not
reproduce is the round-trip through `%.10g` and `std::ifstream`, which is the whole point.

WHY ARRAYS AND NOT A FILE. The text path writes 178 MB for 100k molecules and parses it back,
which costs more than the arithmetic it feeds. It also had a failure mode that arrays cannot
have: a `nan` written into the export made C++'s `istream` fail on the token, and every
subsequent field shifted by one -- the loader then reported a 19.9-atom mean for a 30.6-atom
corpus. In a strided float64 array there are no fields to shift, only offsets, so that class of
desync is structurally gone rather than defended against.

THE NON-FINITE GASTEIGER CONTRACT is kept anyway, unchanged, because it is doing a second job.
Molecules RDKit cannot charge (no PEOE parameters -- selenium is the common case) get 0.0 in the
charge column and `chg_ok = 0`. The 0.0 is no longer protecting a parser; it is keeping BCUT2D's
Burden matrix finite, since a nan on the diagonal propagates through the eigensolver and returns
nan for all eight BCUT2D columns rather than for the two that depend on charge. The flag records
that the molecule was uncharged instead of hiding it, which is what lets a verifier compare like
with like. Callers that want to know can read `chg_ok`.

LAYOUT. One batch is a set of flat arrays plus offsets, so N molecules cross the boundary in one
call. Per-atom integer properties share one (n_atoms, 8) C-contiguous block and per-atom doubles
one (n_atoms, 2) block, because the extension reads them by row and one allocation beats twelve.

A CONTRACT ON THE CALLER: THE MOLECULE MUST BE THE HYDROGEN-SUPPRESSED ONE. Molecules are taken
exactly as given here; nothing calls `Chem.RemoveHs`. mordred computes RingCount (49),
TopologicalCharge (21) and PathCount (11) on `Chem.RemoveHs(mol)`, because every one of those
descriptors has `explicit_hydrogens = False`. `Chem.MolFromSmiles` already suppresses hydrogens,
and on all 100,000 molecules of cpp/hard.smi `RemoveHs` is then the identity -- the only explicit
hydrogen there is isotopic, which plain `RemoveHs` keeps -- so nothing changes today. A molecule
that arrives from `Chem.AddHs` or out of an SDF is a different graph, and those 81 columns will
disagree with mordred for it. (PathCount alone is insulated: `pathcount::build_from_rows` applies
mordred's `useHs=False` itself, because `Chem.FindAllPathsOfLengthN` takes that by default and
mordred never passes it -- so hydrogens are invisible to PathCount while RingCount and
TopologicalCharge would see them. That asymmetry is mordred's, and it is reproduced, not fixed.)

    atom_off  int32   (n_mol + 1,)   atoms of molecule k are [atom_off[k] : atom_off[k+1]]
    bond_off  int32   (n_mol + 1,)
    chg_ok    int32   (n_mol,)       0 = Gasteiger unavailable, charges are 0.0
    atom_i    int32   (n_atoms, 13)  Z, degree, nH, formal charge, hyb, aromatic, in-ring, CIP,
                                     ring count, total valence, _ChiralityPossible, chiral tag,
                                     isotope
    atom_d    float64 (n_atoms, 2)   mass, Gasteiger charge
    bond_i    int32   (n_bonds, 6)   u, v, conjugated, in-ring, SMARTS bond code, BondType int
    bond_s    int32   (n_bonds,)     E/Z as +/-1, 0 for none
    bond_d    float64 (n_bonds,)     bond order
    rings                            the ring SET as a two-level CSR; see `Rings` below
    stereo_a  int32   (n_atoms,)     POTENTIAL tetrahedral stereo, the NEW perception; see
    stereo_b  int32   (n_bonds,)     `_potential_stereo` -- optional, empty when not asked for

THE LAST TWO ATOM COLUMNS ARE THE **LEGACY** STEREO PERCEPTION AND THE STEREO ARRAYS ARE THE
**NEW** ONE, and they are not the same question. `_ChiralityPossible` is set by
`MolOps::assignStereochemistry(cleanIt, force, flagPossible)`; `stereo_a` comes from
`FindPotentialStereo`. The two atom sets differ on 262 of 4,000 corpus molecules (6.6%), so a
port that computed one and used it for both would be wrong on one column in fifteen. RDKit's own
descriptors read them in exactly that split: `NumAtomStereoCenters` and
`NumUnspecifiedAtomStereoCenters` count `_ChiralityPossible`
(Code/GraphMol/Descriptors/Lipinski.cpp), while `SPS` (rdkit/Chem/SpacialScore.py) asks
`FindMolChiralCenters(useLegacyImplementation=False)`.

THE LEGACY PAIR COSTS THE PICKLE PATH NOTHING -- both were already in the blob and were being
skipped; see src/hume_core/molpickle.h. The NEW perception is not in the blob at all and is a
real RDKit call per molecule, priced in `_potential_stereo`'s docstring.

Hybridisation is passed through as RDKit's enum value rather than re-derived, for the reason
export_predict.py gives: HallKierAlpha indexes a per-element table by (hybridisation - 2), and
reimplementing RDKit's perception rules in C++ is the first place an "exact" claim would quietly
stop being true.

TOTAL VALENCE IS CARRIED FOR EXACTLY THAT REASON. SMARTS `v` is asked by three of the fragment
patterns -- `fr_Imine` (`[Nv3]`), `NumHDonors` (`v3`, `v4`) and `NumHAcceptors` (`v2`, `v3`) --
and it is NOT a function of the other nine columns. The obvious reconstruction, round(sum of
incident bond orders) + nH, is wrong on 11,238 of 575,571 atoms of cpp/hard.smi, because RDKit
folds aromatic-bond contributions and hydrogens together under its own rounding rule: pyrrole's
`[nH]` has two aromatic bonds summing to 3.0 and one H, and RDKit's total valence is 3, not 4.
So it is one more RDKit call per atom here, and on the pickle path it is not a new field at all
-- the blob already carries the explicit and implicit valences (see src/hume_core/molpickle.h).

THE BOND TYPE IS CARRIED FOR THE SAME REASON, AND IT IS NOT THE SMARTS CODE. The fifth `bond_i`
column answers SMARTS' two independent questions -- is the ORDER one SMARTS can name, and is the
AROMATIC FLAG set -- and deliberately collapses everything else to zero, which is exact for every
query in cpp/frag_program.h. RDKit's Morgan fingerprint hashes the `Bond::BondType` ENUM VALUE
instead, so that collapse is not exact for it: cpp/hard.smi contains 114 DATIVE bonds, which are
type 17 to RDKit and indistinguishable from an unnamed order here. The sixth column is that
integer, verbatim. It costs one `int()` on a Boost enum already in hand on this path and NOTHING
at all on the pickle path -- src/hume_core/molpickle.h was already reading the byte and throwing
it away, the same finding `tval` produced.

CRIPPEN IS NOT IN THE ARRAYS ANY MORE. It used to be two more `atom_d` columns filled by
`rdMolDescriptors._CalcCrippenContribs`, which cost 78 us/mol -- 42% of this module -- to run
110 SMARTS patterns over the whole molecule. src/hume_core/crippen_typer.h answers the same
question from the integers already in `atom_i` for 1.5 us/mol, bit-identically to RDKit on
2,869,048 atoms, so the call is gone and the extension fills the pair itself. What that costs
here is the fifth `bond_i` column: the typer needs SMARTS bond semantics, where the ORDER and
the AROMATIC FLAG are independent questions, and neither is recoverable from `bond_d` -- a
dative bond has order 1.0 without being SINGLE, and cpp/mols.smi contains TRIPLE bonds carrying
the aromatic flag. Both cases are real, and together they are one Python call per bond.

WHY THE LOOPS BELOW LOOK LIKE THAT. `a.GetAtomicNum()` is not arithmetic, it is a Boost.Python
round trip, and the old shape of this function made about eleven of them per atom and seven per
bond -- ~300 per molecule, measured at 88 us. Three things shrink that without changing a single
value, because every value is the same one RDKit would have handed back either way:

  * ONE PASS PER COLUMN, `list.extend(map(Atom.GetAtomicNum, ats))`, instead of one tuple per
    atom. The unbound method skips the per-call attribute lookup on the instance, which measured
    0.039 us/call against 0.11 for `a.GetAtomicNum()` -- and the whole pass over 29 atoms then
    costs about as much as ten individual calls did.
  * THE ATOM LIST IS BUILT ONCE, and BY INDEX. Wrapper construction alone -- building the list
    and reading nothing off it -- is a third of what is left, so the old code's three separate
    walks (charges, properties, CIP) were expensive before they read anything. Materialising
    once pays it once. And `map(m.GetAtomWithIdx, range(n))` is HALF the price of iterating the
    `m.GetAtoms()` sequence for the same objects in the same order -- 7.4 us/mol against 15.0,
    and 8.9 against 16.7 for bonds. RDKit's _ROAtomSeq iterator is simply dearer than indexing.
  * THE NON-FINITE CHARGE SCAN IS A BATCH numpy OP, not a per-atom `np.isfinite`. Calling a
    numpy ufunc on a Python float costs ~0.3 us, so the old scan cost more than the charges did.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, rdmolops

# E/Z as +/-1, matching stereo.py's _E exactly (TRANS is E, CIS is Z).
_EZ = {Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
       Chem.BondStereo.STEREOZ: -1, Chem.BondStereo.STEREOCIS: -1}

N_ATOM_INT, N_ATOM_DBL, N_BOND_INT = 13, 2, 6

# The bond half of the Crippen typer's input, byte-for-byte cpp/export_crippen.py's bond_code():
# a bit for the bond ORDER when it is one of the three SMARTS knows how to name, and a separate
# bit for the aromatic FLAG. `-` asks about the order, `:` about the flag, and the single bond
# joining biphenyl's two rings is the C20-vs-C19 distinction that needs them apart.
_BIT_SINGLE, _BIT_DOUBLE, _BIT_TRIPLE, _BIT_AROM = 1, 2, 4, 8
_TYPE_BIT = {Chem.BondType.SINGLE: _BIT_SINGLE,
             Chem.BondType.DOUBLE: _BIT_DOUBLE,
             Chem.BondType.TRIPLE: _BIT_TRIPLE}

# BondType -> GetBondTypeAsDouble(), memoised the first time each type is seen. The bond loop has
# to read `b.GetBondType()` anyway for the Crippen code, so caching the order against it replaces
# a second Python call per bond with a dict hit. It is filled from RDKit's own answer for a real
# bond of that type rather than transcribed, so there is no second copy of RDKit's table to go
# stale -- and the types RDKit refuses to give a number for (THREECENTER, DATIVEL, DATIVER,
# OTHER all raise) never enter the cache and raise from the call, exactly as before.
_ORDER: dict = {}

# Unbound accessors, bound once here. `map(_atomic_num, ats)` is a C-level loop over a C-level
# callable; `[a.GetAtomicNum() for a in ats]` re-resolves the attribute on every atom.
_atomic_num = Chem.Atom.GetAtomicNum
_degree = Chem.Atom.GetDegree
_total_num_hs = Chem.Atom.GetTotalNumHs
_formal_charge = Chem.Atom.GetFormalCharge
_hybridization = Chem.Atom.GetHybridization
_is_aromatic = Chem.Atom.GetIsAromatic
_atom_in_ring = Chem.Atom.IsInRing
_total_valence = Chem.Atom.GetTotalValence
_mass = Chem.Atom.GetMass
_has_prop = Chem.Atom.HasProp
_double_prop = Chem.Atom.GetDoubleProp

_chiral_tag = Chem.Atom.GetChiralTag
_isotope = Chem.Atom.GetIsotope
_bond_stereo = Chem.Bond.GetStereo
_bond_type = Chem.Bond.GetBondType

_CIP_CODE = "_CIPCode"
_CHIRALITY_POSSIBLE = "_ChiralityPossible"
_GASTEIGER = "_GasteigerCharge"

# The NEW stereo perception's three constants, bound once. `Atom_Tetrahedral` is the only
# StereoType `SPS` reads off `FindPotentialStereo`; the `Bond_Double` entries are NOT what its
# bond term asks (see `_potential_stereo`).
_TET = Chem.StereoType.Atom_Tetrahedral
_BT_DOUBLE = Chem.BondType.DOUBLE
_STEREONONE = Chem.BondStereo.STEREONONE

# "Not asked for", distinct from "asked for and empty". Shared, and never written to.
_Z32 = np.zeros(0, dtype=np.int32)


def _potential_stereo(mols):
    """RDKit's POTENTIAL stereo perception, per atom and per bond, for `SPS`.

    -> ``(stereo_a int32 (n_atoms,), stereo_b int32 (n_bonds,))``, flat across the batch in the
    same atom and bond order as `atom_i` / `bond_i`.

    THE SPECIFICATION IS rdkit/Chem/SpacialScore.py, and it is reproduced call for call:

        molCp = Chem.Mol(mol); rdmolops.FindPotentialStereoBonds(molCp)
        chiral_idxs = [i for i, _ in Chem.FindMolChiralCenters(
                           molCp, includeUnassigned=True, includeCIP=False,
                           useLegacyImplementation=False)]
        db_stereo   = {(u, v): bond.GetStereo() for DOUBLE bonds}   # on the same copy

    ON A COPY, AND THAT IS LOAD-BEARING RATHER THAN TIDY. `FindPotentialStereoBonds` sets
    STEREOANY on bonds the caller's molecule has no stereo on, and `FindPotentialStereo` sets its
    own `_ChiralityPossible` -- the NEW perception's flag, under the LEGACY perception's property
    name. Letting either touch the caller's molecule would corrupt `bond_s` and would silently
    overwrite the legacy flag that `NumAtomStereoCenters` counts, which is the one boundary field
    whose two producers must not be allowed to meet.

    WHY NOT `FindMolChiralCenters` ITSELF: IT IS 4.6x THE PRICE OF WHAT IT COMPUTES. Called
    verbatim it costs 243 us/mol on 2,000 corpus molecules against 52 us/mol here, and the
    difference is not perception -- `FindPotentialStereo` is 28 us/mol of it. It is that
    `for si in itms` over the returned `_vect...StereoInfo` costs **173 us per call regardless of
    how many items it holds**: 172.9 us for an EMPTY vector and 175.4 for a one-element one,
    measured. Boost.Python's default iterator ends by taking an out-of-range `__getitem__` and
    translating the C++ exception, and that unwind is the whole cost. Indexing the same vector by
    position is 0.3 us. So this loops `range(len(itms))` and indexes -- the same items, the same
    order, and 0 differing molecules out of 2,000 against the verbatim call.

    A CHEAPER ROUTE WAS TRIED AND IS WRONG: `FindPotentialStereo(c, cleanIt=True,
    flagPossible=True)` sets `_ChiralityPossible` on the copy, which would let this read the
    answer with an unbound `HasProp` map for 70 us/mol total. It disagrees with
    `FindMolChiralCenters` on 121 of 2,000 molecules -- the flag is set on a SUBSET of the
    `Atom_Tetrahedral` entries. `CC1Cc2ccccc2[N+]1=CC=C1N(C)c2ccccc2C1(C)C[SiH3]` has
    Atom_Tetrahedral at 1, 21 and 24 and gets the flag on 1 and 21 only.

    THE BOND TERM IS NOT THE `Bond_Double` STEREO INFOS. SpacialScore reads `bond.GetStereo()`
    off the molecule after `FindPotentialStereoBonds`, which is a different set: a bond can carry
    STEREOANY there and STEREONONE is the answer for most double bonds. Reproduced as written.
    """
    sa: list[int] = []
    sb: list[int] = []
    old = Chem.GetUseLegacyStereoPerception()
    # `FindMolChiralCenters(useLegacyImplementation=False)` sets this global for the duration of
    # its call and restores it. Hoisted out of the loop here, which is the same state seen by the
    # same perception -- and it is restored in a `finally`, because leaving it False would change
    # how every later `Chem.MolFromSmiles` in the process perceives stereo.
    Chem.SetUseLegacyStereoPerception(False)
    try:
        for m in mols:
            c = Chem.Mol(m)
            rdmolops.FindPotentialStereoBonds(c)

            row = [0] * c.GetNumAtoms()
            itms = Chem.FindPotentialStereo(c)
            for k in range(len(itms)):              # NOT `for si in itms`; see the docstring
                si = itms[k]
                if si.type == _TET:
                    row[si.centeredOn] = 1
            sa.extend(row)

            nb = c.GetNumBonds()
            bonds = list(map(c.GetBondWithIdx, range(nb)))
            brow = [0] * nb
            # One pass for the stereo, and `GetBondType` only on the few bonds that have any --
            # STEREONONE is the answer for the overwhelming majority, so the type call is rare.
            for e, st in enumerate(map(_bond_stereo, bonds)):
                if st != _STEREONONE and _bond_type(bonds[e]) == _BT_DOUBLE:
                    brow[e] = 1
            sb.extend(brow)
    finally:
        Chem.SetUseLegacyStereoPerception(old)
    return np.asarray(sa, dtype=np.int32), np.asarray(sb, dtype=np.int32)


@dataclass(frozen=True)
class Rings:
    """The ring SET, as a two-level CSR. Atom indices are LOCAL to their molecule.

        ring_moff  int32 (n_mol + 1,)     rings of molecule k are [ring_moff[k] : ring_moff[k+1])
        ring_ptr   int32 (n_rings + 1,)   atoms of ring r are ring_at[ring_ptr[r] : ring_ptr[r+1]]
        ring_at    int32 (n_ring_atoms,)

    WHY THIS IS NOT `atom_i`'s `nring` COLUMN. That column is a per-atom COUNT and answers SMARTS
    `[R2]`, which is all the 182 blocks and the Crippen typer ask. Every one of RingCount's 49
    columns is a predicate on a RING -- its size, whether ALL its atoms are aromatic, whether ANY
    is a heteroatom -- and its 28 fused columns need |Ri & Rj| for every ring pair. Benzene and
    cyclohexane have identical `nring` vectors and differ on 6 of the 49, so no count can stand
    in for the set. It is not a second perception either: both come from the one ring perception
    RDKit did at sanitisation. See src/hume/_rings.py.
    """
    ring_moff: np.ndarray
    ring_ptr: np.ndarray
    ring_at: np.ndarray


@dataclass(frozen=True)
class Batch:
    """Flat arrays for N molecules. See the module docstring for the layout."""
    atom_off: np.ndarray
    bond_off: np.ndarray
    chg_ok: np.ndarray
    atom_i: np.ndarray
    atom_d: np.ndarray
    bond_i: np.ndarray
    bond_s: np.ndarray
    bond_d: np.ndarray
    rings: Rings
    # Empty when `extract(..., stereo=False)`. NOT zero-filled: a zero array is a valid answer
    # (no potential stereo anywhere) and would make "nobody asked for this" indistinguishable
    # from "the perception found nothing", which is exactly how `SPS` would come out finite and
    # wrong. Length 0 means absent, and bindings.cpp refuses it rather than defaulting.
    stereo_a: np.ndarray
    stereo_b: np.ndarray

    @property
    def n_mol(self) -> int:
        return len(self.chg_ok)


def _rings_csr(mols) -> Rings:
    """The ring CSR for a batch, from src/hume/_rings.py -- the module that ships.

    ONE SOURCE OF RINGS FOR BOTH BOUNDARIES, and that is a decision rather than an accident. The
    pickle carries RDKit's RingInfo and src/hume_core/molpickle.h can already parse it, so the
    pickle path COULD take its rings from the blob for free. It does not, because the blob
    carries RDKit's RAW answer and `rings_for` returns a REPAIRED one: `Chem.GetSymmSSSR` is not
    a function of the graph, and 32 molecules in 100,000 get a different ring set depending on
    the order they are presented in. Two boundaries sourcing rings from genuinely different
    places, agreeing by argument rather than by construction, is how 49 columns go quietly wrong.
    The ~4 us/mol this costs the pickle path is under 2% of the end-to-end figure, and the
    `Sink::ring_at` hooks in molpickle.h stay in place, unused, if that ever stops being true.
    """
    from ._rings import rings_for

    moff, ptr, at = [0], [0], []
    for m in mols:
        rs = rings_for(m)
        for r in rs:
            at.extend(r)
            ptr.append(len(at))
        moff.append(len(ptr) - 1)
    return Rings(np.asarray(moff, dtype=np.int32), np.asarray(ptr, dtype=np.int32),
                 np.asarray(at, dtype=np.int32))


def extract(mols, stereo: bool = True) -> Batch:
    """Flatten an iterable of RDKit molecules into one Batch.

    Molecules are taken as given -- no filtering, no sanitisation beyond what the caller already
    did. `None` is rejected loudly rather than skipped, because a silently dropped molecule turns
    a row index into a lie, and every consumer of this array indexes by position.

    `stereo` runs the NEW potential-stereo perception (`_potential_stereo`, 52 us/mol) and fills
    `stereo_a` / `stereo_b`. It is a parameter rather than always-on because the 182-column block
    path reads neither, and paying 52 us/mol for two arrays nobody looks at is a third of that
    path's whole boundary cost. The LEGACY pair -- `atom_i`'s last two columns -- is not optional
    and costs two unbound `map`s.
    """
    mols = list(mols)
    atom_off, bond_off, chg_ok = [0], [0], []
    z: list[int] = []
    deg: list[int] = []
    nh: list[int] = []
    fchg: list[int] = []
    hyb: list[int] = []
    arom: list[int] = []
    ring: list[int] = []
    cip: list[int] = []
    nring: list[int] = []
    tval: list[int] = []
    chirposs: list[int] = []
    ctag: list[int] = []
    iso: list[int] = []
    mass: list[float] = []
    charge: list[float] = []
    bu: list[int] = []
    bv: list[int] = []
    bconj: list[int] = []
    bring: list[int] = []
    bcode: list[int] = []
    btype: list[int] = []
    bs: list[int] = []
    bd: list[float] = []
    na = nb = 0

    for k, m in enumerate(mols):
        if m is None:
            raise ValueError(f"molecule {k} is None; parse before calling extract()")

        n = m.GetNumAtoms()
        ats = list(map(m.GetAtomWithIdx, range(n)))

        # Gasteiger stays RDKit's. It is 9 us of iterative C++ for the charges themselves, and
        # PEOE is a fitted parameter set rather than a graph query -- a second implementation
        # would be a second set of parameters to keep in step, which is the argument that also
        # kept hybridisation and CIP on RDKit's side of the line. Crippen was the opposite case
        # and has moved; see the module docstring.
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
            charge.extend(map(_double_prop, ats, repeat(_GASTEIGER)))
            ok = 1
        except Exception:
            del charge[na:]                      # a partial molecule must not survive the throw
            charge.extend(repeat(0.0, n))
            ok = 0
        # CIP codes for the stereo block. MolFromSmiles assigns them already; the explicit call
        # is the safety net stereo.py also carries, and is verified to change nothing here.
        #
        # `flagPossibleStereoCenters=True` IS NOT OPTIONAL AND OMITTING IT IS A SILENT ZERO.
        # `cleanIt=True, force=True` CLEARS `_ChiralityPossible` unless it is passed -- on 911 of
        # 2,000 corpus molecules, measured. `Descriptors.NumAtomStereoCenters` counts exactly that
        # flag, so without this argument the column comes back 0 with no symptom, and an
        # ill-posedness screen built on the same call reported a well-posed column as unstable on
        # 4,125 of 9,000 shuffles. PORT_STATUS.md records it as trap #1.
        Chem.AssignStereochemistry(m, cleanIt=True, force=True, flagPossibleStereoCenters=True)

        z.extend(map(_atomic_num, ats))
        deg.extend(map(_degree, ats))
        nh.extend(map(_total_num_hs, ats))
        fchg.extend(map(_formal_charge, ats))
        hyb.extend(map(_hybridization, ats))
        arom.extend(map(_is_aromatic, ats))
        ring.extend(map(_atom_in_ring, ats))
        # SMARTS `v`. RDKit's own perception, not re-derived here; see the module docstring for
        # the 11,238-atom counterexample to the obvious reconstruction.
        tval.extend(map(_total_valence, ats))
        mass.extend(map(_mass, ats))
        # RING COUNT, not just ring membership. `[R1]` and `[R2]` are ordinary SMARTS primitives
        # in RDKit's fragment patterns, and the boolean above cannot answer them. Reconstructing
        # ring counts C++-side would mean perceiving rings a second time, on a graph whose ring
        # perception is already known to be numbering-dependent for 24 molecules in the 100k
        # corpus -- so it is carried across the boundary from the single perception RDKit has
        # already done, rather than recomputed and risked diverging.
        nring.extend(map(m.GetRingInfo().NumAtomRings, range(len(ats))))
        # THE LEGACY POTENTIAL-STEREO FLAG AND THE CHIRAL TAG, the two inputs
        # `NumAtomStereoCenters` and `NumUnspecifiedAtomStereoCenters` are exactly a function of:
        # Lipinski.cpp counts atoms carrying `_ChiralityPossible`, and the unspecified variant
        # additionally requires `getChiralTag() == CHI_UNSPECIFIED`. Reconstructing both from
        # these two properties reproduces RDKit on 4,000 of 4,000 corpus molecules. `HasProp`
        # returns 0/1, so the column is the map's result with no second pass.
        chirposs.extend(map(_has_prop, ats, repeat(_CHIRALITY_POSSIBLE)))
        # The RAW enum, not a `== CHI_UNSPECIFIED` boolean. CHI_UNSPECIFIED is 0 and the C++ only
        # tests for that, but carrying the tag itself means the boundary answers a question about
        # the molecule rather than a question about one descriptor -- and it is what the pickle
        # already holds (atom property-flag bit 2), so narrowing it here would throw information
        # away in exchange for nothing.
        ctag.extend(map(int, map(_chiral_tag, ats)))
        # SMARTS ISOTOPE, and it exists for exactly four queries: QED structural alerts 112-115
        # are `[15N]`, `[13C]`, `[18O]` and `[34S]`. It is NOT recoverable from the `mass` column
        # beside it -- `Atom.GetMass()` returns the isotope's mass, so mass says an atom IS
        # labelled (which is the test constit.h's `exactMolWt` makes) but not labelled with what,
        # and inverting RDKit's isotope-mass table would be an injectivity argument standing in
        # for a value RDKit hands over for free. 0 when unset, which is what `[0*]` asks about.
        iso.extend(map(_isotope, ats))

        # HasProp returns 0/1, so in the overwhelmingly common case of a molecule with no
        # assigned stereocentre the flag list IS the CIP column and no second pass happens.
        flags = list(map(_has_prop, ats, repeat(_CIP_CODE)))
        if any(flags):
            cip.extend([(1 if a.GetProp(_CIP_CODE) == "R" else -1) if f else 0
                        for f, a in zip(flags, ats)])
        else:
            cip.extend(flags)

        nbonds = m.GetNumBonds()
        for b in map(m.GetBondWithIdx, range(nbonds)):
            bt = b.GetBondType()
            code = _TYPE_BIT.get(bt, 0)
            if b.GetIsAromatic():
                code |= _BIT_AROM
            bu.append(b.GetBeginAtomIdx())
            bv.append(b.GetEndAtomIdx())
            bconj.append(b.GetIsConjugated())
            bring.append(b.IsInRing())
            bcode.append(code)
            # RDKit's Bond::BondType INTEGER, the sixth column. `bt` is already in hand for the
            # SMARTS code above, so this is an int() on a Boost enum and not a second RDKit call.
            btype.append(int(bt))
            bs.append(_EZ.get(b.GetStereo(), 0))
            order = _ORDER.get(bt)
            if order is None:
                order = _ORDER[bt] = b.GetBondTypeAsDouble()
            bd.append(order)

        na += n
        nb += nbonds
        atom_off.append(na)
        bond_off.append(nb)
        chg_ok.append(ok)

    if not chg_ok:
        return _empty()

    atom_off_a = np.asarray(atom_off, dtype=np.int32)
    chg_ok_a = np.asarray(chg_ok, dtype=np.int32)
    charge_a = np.asarray(charge, dtype=np.float64)

    # The non-finite contract, once per batch instead of once per atom. RDKit hands back inf or
    # nan for elements PEOE has no parameters for; those atoms get 0.0 and their molecule gets
    # chg_ok = 0. Doing it here rather than in the loop is what makes it free when nothing is
    # wrong, which is almost always -- and `searchsorted` on the offsets is exactly the "which
    # molecule owns atom i" question the flat layout was chosen to make cheap.
    bad = ~np.isfinite(charge_a)
    if bad.any():
        charge_a[bad] = 0.0
        owners = np.searchsorted(atom_off_a, np.flatnonzero(bad), side="right") - 1
        chg_ok_a[owners] = 0

    atom_i = np.empty((na, N_ATOM_INT), dtype=np.int32)
    for col, src in enumerate((z, deg, nh, fchg, hyb, arom, ring, cip, nring, tval,
                               chirposs, ctag, iso)):
        atom_i[:, col] = src
    atom_d = np.empty((na, N_ATOM_DBL), dtype=np.float64)
    atom_d[:, 0] = mass
    atom_d[:, 1] = charge_a
    bond_i = np.empty((nb, N_BOND_INT), dtype=np.int32)
    for col, src in enumerate((bu, bv, bconj, bring, bcode, btype)):
        bond_i[:, col] = src

    # The NEW perception, on the SAME shared helper the pickle path uses -- so the two boundaries
    # cannot disagree about it by construction, which is the only way this repo lets two paths
    # agree. cpp/verify_molpickle.py asserts it anyway.
    stereo_a, stereo_b = _potential_stereo(mols) if stereo else (_Z32, _Z32)

    return Batch(
        atom_off=atom_off_a,
        bond_off=np.asarray(bond_off, dtype=np.int32),
        chg_ok=chg_ok_a,
        atom_i=atom_i,
        atom_d=atom_d,
        bond_i=bond_i,
        bond_s=np.asarray(bs, dtype=np.int32),
        bond_d=np.asarray(bd, dtype=np.float64),
        rings=_rings_csr(mols),
        stereo_a=stereo_a,
        stereo_b=stereo_b,
    )


# ------------------------------------------------------------------------------------------
# THE PICKLE PATH -- the same boundary, with the ~300 Python calls per molecule deleted.
#
# extract() above is now the REFERENCE IMPLEMENTATION and stays that way. It reads the molecule
# through RDKit's supported Python API, so it is the oracle the pickle reader is graded against
# (cpp/verify_molpickle.py), the fallback for molecules the reader refuses -- query atoms,
# substance groups, anything unsanitised -- and the path that still works the day RDKit moves
# its pickle format. Nothing below replaces it; it sits alongside.
#
# WHAT MOVED. The 15 us/mol of atom- and bond-WRAPPER CONSTRUCTION that reads nothing, plus
# every per-column pass on top of it, are gone. What is left on this side is three RDKit calls
# per molecule -- charges, stereochemistry, serialise -- and none of them is a per-atom round
# trip. src/hume_core/molpickle.h fills the arrays from the blob.
#
# WHY ComputedProps IS IN THE FLAG SET, and it is not free. `_GasteigerCharge` is a COMPUTED
# property, not a private one: `PrivateProps | AtomProps` pickles `_CIPCode` and stops, and a
# reader trusting the obvious flag pair would silently see zero charges. Adding ComputedProps
# takes the blob from 452 to 5034 bytes/mol, because each atom then also carries
# `_GasteigerHCharge`, `_CIPRank` and a `__computedProps` vector naming them. Those are skipped
# by the reader; what they cost is ToBinary's time and they are why this is 26 us/mol rather
# than the 10 the smaller flag set takes. NoConformers drops a section we never had.
_PICKLE_FLAGS = (Chem.PropertyPickleOptions.PrivateProps
                 | Chem.PropertyPickleOptions.AtomProps
                 | Chem.PropertyPickleOptions.ComputedProps
                 | Chem.PropertyPickleOptions.NoConformers)

_to_binary = Chem.Mol.ToBinary


@dataclass(frozen=True)
class Pickles:
    """The pickle boundary: one blob per molecule, plus the ring CSR the blob does not carry.

    `rings` is here for the same reason it is on `Batch`, from the same `rings_for()`, and the
    two are trivially equal by construction -- which cpp/verify_molpickle.py asserts anyway.

    `h_blobs` IS A SECOND MOLECULE, NOT A SECOND VIEW. mordred sets `explicit_hydrogens = True`
    for Autocorrelation, so its 540 columns describe `Chem.AddHs(m)` -- 55 atoms where the rest
    of the pipeline sees 29. The TOPOLOGY of that graph is derivable in C++ from `nH` (which is
    exactly what infocontent.h's `HGraph` does, and it is not shared with this: infocontent needs
    the H graph and no charges, so deriving is free for it and would buy nothing here). What is
    NOT derivable is the CHARGE, because mordred's `c` weight wants Gasteiger run ON the H-added
    molecule, and PEOE is not invariant to making hydrogens explicit. Measured on 1,500 corpus
    molecules: 5,221 of 42,359 heavy atoms get a different `_GasteigerCharge` from `AddHs(m)`
    than from `m`, and 7,395 of 38,326 hydrogen charges differ from `_GasteigerHCharge / nH`
    (max 7.7e-16 relative -- last bit, but this repo grades autocorrelation bitwise). So the
    second molecule is charged and serialised for real. It costs 84.7 +/- 0.7 us/mol; the
    alternative that reads those charges per atom in Python costs 68.4 and puts an atom-wrapper
    pass back in the path, and the alternative that derives them costs nothing and puts 419
    columns permanently on a tolerance. See cpp/bench_molpickle.py.
    """
    blobs: list
    rings: Rings
    h_blobs: list
    # The NEW potential-stereo perception, which the blob does NOT carry and cannot be made to:
    # `FindPotentialStereoBonds` would have to run on the molecule being serialised, and it sets
    # STEREOANY on bonds that have no stereo, which is `bond_s`. So `SPS` gets two arrays
    # alongside the blobs. Empty when `extract_pickles(..., stereo=False)`; see `Batch`.
    stereo_a: np.ndarray
    stereo_b: np.ndarray

    def __len__(self) -> int:
        return len(self.blobs)


def extract_pickles(mols, stereo: bool = True) -> Pickles:
    """Serialise molecules for the C++ reader. The Python half of the pickle boundary.

    Same contract as `extract()`: `None` is rejected loudly rather than skipped, and the two
    RDKit computations that `extract()` performs happen here in the same order, because both are
    inputs the blob has to carry.

    `stereo` is the same switch `extract()` carries and costs the same 52 us/mol; the 182-column
    block path passes False because it reads neither array.
    """
    mols = list(mols)
    out, hout = [], []
    for k, m in enumerate(mols):
        if m is None:
            raise ValueError(f"molecule {k} is None; parse before calling extract_pickles()")
        # The Autocorrelation molecule, first, because AddHs copies and must not inherit the
        # heavy-atom charges we are about to compute -- it needs its own. `nIter` is left at
        # RDKit's default here and not 12, because cpp/export_ac.py calls it that way and that
        # is the call the 540 columns were verified against.
        mh = Chem.AddHs(m)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mh)
        except Exception:
            mh.ClearComputedProps()     # no `_GasteigerHCharge` -> mordred's getter yields 0.0
        hout.append(_to_binary(mh, _PICKLE_FLAGS))
        try:
            rdPartialCharges.ComputeGasteigerCharges(m, nIter=12)
        except Exception:
            # extract() zeroes the charge column for the WHOLE molecule when this throws, not
            # just for the atoms that failed. Clearing the computed props reproduces that on
            # this side: the reader then finds no `_GasteigerCharge` on any atom and writes 0.0
            # with chg_ok = 0. `_CIPCode` is private rather than computed and survives.
            m.ClearComputedProps()
        # THE ONE ARGUMENT THAT MADE TWO COLUMNS POSSIBLE, and see extract() for the measurement:
        # without `flagPossibleStereoCenters=True` this call CLEARS `_ChiralityPossible` on 911 of
        # 2,000 molecules, and the flag is what `NumAtomStereoCenters` counts. The blob already
        # carried the bit (pickleExplicitProperties 0x8) and molpickle.h was skipping it; this is
        # the only real change the two columns needed on this side.
        Chem.AssignStereochemistry(m, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        out.append(_to_binary(m, _PICKLE_FLAGS))
    sa, sb = _potential_stereo(mols) if stereo else (_Z32, _Z32)
    return Pickles(out, _rings_csr(mols), hout, sa, sb)


def _check_pickle_version() -> None:
    """Fail at import if RDKit's pickle format is not the one molpickle.h was written against.

    Same shape as the drift guards in src/hume_core/crippen_typer.h and cpp/estate_tables.h, and
    for the same reason: a silently misparsed pickle is a wrong descriptor with no symptom. The
    probe is one carbon atom, so this costs microseconds once per process.
    """
    from . import _core

    probe = Chem.MolFromSmiles("C")
    try:
        _core.pickle_check(_to_binary(probe, _PICKLE_FLAGS))
    except RuntimeError as exc:
        raise ImportError(str(exc)) from None


_check_pickle_version()


def _empty() -> Batch:
    z32, z64 = np.zeros(0, np.int32), np.zeros(0, np.float64)
    return Batch(np.zeros(1, np.int32), np.zeros(1, np.int32), z32,
                 z32.reshape(0, N_ATOM_INT), z64.reshape(0, N_ATOM_DBL),
                 z32.reshape(0, N_BOND_INT), z32, z64,
                 Rings(np.zeros(1, np.int32), np.zeros(1, np.int32), z32), z32, z32)
