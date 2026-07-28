"""Stereochemistry fingerprints for chirality-aware conformer selection (xyzgraph ``--stereo``).

A conformer's chirality is a *fingerprint* of every stereo element xyzgraph detects — point (R/S), E/Z,
axial (Rₐ/Sₐ), **planar (a metallocene's planar chirality)**, and helical. Comparing fingerprints lets us
keep / drop / invert conformers by handedness on any one element (or all), which is how we sample a metal
centre's geometry while holding a spectator ferrocene's planar chirality fixed.

Bonds are re-perceived from the **coordinates** (not the Mol graph), so a metallocene is recognised even on
a surrogate Mol whose metal-donor bonds were stripped — the only requirement is real element symbols.
"""

from __future__ import annotations

import os
import tempfile

from rxembed.rdkit_embed import io as _io

_KEYS = [("point", "atom"), ("ez", "bond"), ("axial", "atoms"), ("planar", "ring"), ("helical", "atoms")]


def _xyz_block(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    out = [str(mol.GetNumAtoms()), ""]
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        out.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(out) + "\n"


_INVERT = {"R": "S", "S": "R", "Rₐ": "Sₐ", "Sₐ": "Rₐ", "Rₚ": "Sₚ", "Sₚ": "Rₚ", "M": "P", "P": "M", "E": "Z", "Z": "E"}
# the chirality RDKit's embed cannot keep — what this filter is *for*. Point R/S and E/Z are the embed's
# own job (defined SMILES stereocentres) or labile (a protic-amine centre we do NOT want to lock), so
# 'preserve' leaves them free by default; opt in with {'point': 'preserve'}.
_NONGRAPH = {"planar", "axial", "helical"}


def _mode(kind, spec):
    """Resolve the per-kind chirality mode from ``spec``.

    A global string: ``'free'`` -> free all; ``'preserve'`` -> preserve only the non-graph kinds
    (planar/axial/helical), free point/ez; ``'all'`` -> preserve every kind; ``'invert'`` -> invert the
    non-graph kinds. A dict overrides per kind, falling back to its ``'default'`` or the same 'preserve' rule.
    """
    if isinstance(spec, str):
        if spec == "free":
            return "free"
        if spec == "all":
            return "preserve"
        if spec == "invert":
            return "invert" if kind in _NONGRAPH else "free"
        return "preserve" if kind in _NONGRAPH else "free"  # 'preserve' = non-graph only
    if kind in spec:
        return spec[kind]
    if "default" in spec:
        return spec["default"]
    return "preserve" if kind in _NONGRAPH else "free"


def signature(mol, conf_id=-1, charge=0):
    """Compute a conformer's chirality fingerprint: a multiset of handedness labels per element kind.

    ``{kind: Counter({label: count})}`` over ``point`` (R/S), ``ez``, ``axial`` (Rₐ/Sₐ), ``planar`` (a
    metallocene's Rₚ/Sₚ), ``helical`` (M/P) — via xyzgraph (bonds re-perceived from the geometry, so a
    metallocene is seen even on a bond-stripped surrogate). We key by *kind*, not atom indices, because
    xyzgraph picks different representative atoms per re-perception — the handedness label is the stable,
    physical quantity, the atom set is not.
    """
    from collections import Counter

    import xyzgraph
    from xyzgraph.stereo import annotate_stereo

    fd, path = tempfile.mkstemp(suffix=".xyz")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_xyz_block(mol, conf_id))
        summary = annotate_stereo(xyzgraph.build_graph(path, charge=charge, kekule=True))
    finally:
        os.unlink(path)
    sig = {}
    for kind, _key in _KEYS:
        labels = [e.get("label") for e in (summary.get(kind) or []) if e.get("label")]
        if labels:
            sig[kind] = Counter(labels)
    return sig


def satisfies_spec(sig, ref, spec="preserve"):
    """Return whether fingerprint ``sig`` satisfies the chirality ``spec`` against reference ``ref``.

    Judged on **handedness, not count** — xyzgraph's per-conformer perception is count-unstable (a
    metallocene may surface as ``{}``, ``{Sₚ:1}`` or ``{Sₚ:2}`` for the *same* configuration), but the
    *label* is stable. So per kind: ``'preserve'`` rejects only if the conformer carries the **flipped**
    label of a reference element (an actual inversion); ``'invert'`` rejects if it carries any reference
    label **unchanged**; ``'free'`` is unconstrained. `spec` is a global string or a dict
    ``{kind: mode, 'default': mode}``. A conformer passes only if every reference kind is satisfied.
    """
    for kind, refcount in ref.items():
        mode = _mode(kind, spec)
        if mode == "free":
            continue
        present = set(sig.get(kind, ()))  # distinct handedness labels in the conformer
        ref_labels = set(refcount)
        if mode == "preserve" and present & {_INVERT.get(lbl, lbl) for lbl in ref_labels}:
            return False  # a flipped element is present -> rejected
        if mode == "invert" and present & ref_labels:
            return False  # an un-flipped (original) element is present
    return True


def _stereo_label(mol, atom_centers, bond_centers, cap_to_metal=None):
    """Build a readable, index-keyed configuration tag over the enumerated centres, e.g. ``'1S'`` or ``'1R,3S,5=6:E'``.

    CIP R/S where RDKit assigns it (falls back to the raw CW/CCW tag for a centre it won't CIP-rank, e.g. some
    P), plus E/Z for each enumerated double bond. Keyed only on the *enumerated* atoms/bonds so distinct
    variants always get distinct, stable labels. ``cap_to_metal`` maps each donor's D-cap atom index to its
    metal's atomic number: the CIP is then computed with the METAL (highest priority) in the cap position, not
    the D (lowest) — so a metal-bound donor's R/S names the coordinated centre correctly (the D->M priority
    flip is NOT a fixed R<->S swap; it depends on the donor's other substituents, e.g. whether it carries an H).
    """
    from rdkit import Chem

    if cap_to_metal:  # temporarily give each D-cap the metal's atomic number for a coordinated-complex CIP
        rw = Chem.RWMol(mol)
        for d_idx, z in cap_to_metal.items():
            rw.GetAtomWithIdx(d_idx).SetAtomicNum(z)
            rw.GetAtomWithIdx(d_idx).SetIsotope(0)
            for nb in rw.GetAtomWithIdx(d_idx).GetNeighbors():  # neutralise the donor so metal+donor isn't hypervalent
                nb.SetFormalCharge(0)  # (an anionic carbanion C would be pentavalent with a real M bonded)
                nb.SetNoImplicit(True)
        mol = rw.GetMol()
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
        )
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    parts = []
    for idx in atom_centers:
        a = mol.GetAtomWithIdx(idx)
        code = a.GetPropsAsDict().get("_CIPCode") or {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: "CW",
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "CCW",
        }.get(a.GetChiralTag())
        if code:  # an UNRESOLVED centre (e.g. an allene axis RDKit can't set) is dropped — never a '?' tag
            parts.append(f"{idx}{code}")
    for bidx in bond_centers:
        b = mol.GetBondWithIdx(bidx)
        tag = {
            Chem.BondStereo.STEREOE: "E",
            Chem.BondStereo.STEREOZ: "Z",
            Chem.BondStereo.STEREOTRANS: "E",
            Chem.BondStereo.STEREOCIS: "Z",
        }.get(b.GetStereo())
        if tag:
            parts.append(f"{b.GetBeginAtomIdx()}={b.GetEndAtomIdx()}:{tag}")
    return ",".join(parts)


def _coordination_locked_double_bonds(mol, metals):
    """Double bonds whose E/Z is fixed by the coordination — endocyclic in a ring closed through the metal.

    Such a bond has one buildable geometry (decided by the coordination isomer, the polyhedron path's job), so
    enumerating both E and Z is a phantom: the wrong hand forces a bite the chelate can't span and the pipeline
    burns seeds relaxing it into broken bonds. An alpha-diimine (N=C-C=N chelate) is the type case — both C=N
    sit in the 5-membered metal ring and were enumerated 2x2.

    RDKit ignores dative M-donor bonds in ring perception, so the metal-closed ring is invisible natively;
    upgrade the datives to single to reveal it. A double bond still in a ring once the metal is removed is a
    genuine organic ring bond (RDKit already handles its E/Z) and left alone — only a bond cyclic because of the
    metal is locked here.
    """
    from rdkit import Chem

    metals = set(metals)
    if not metals:
        return set()
    up = Chem.RWMol(mol)  # dative -> single so RDKit sees the metal ring; FastFindRings avoids a valence sanitize
    for b in up.GetBonds():
        if b.GetBondType() == Chem.BondType.DATIVE:
            b.SetBondType(Chem.BondType.SINGLE)
    up = up.GetMol()
    Chem.FastFindRings(up)
    metal_rings = [set(r) for r in up.GetRingInfo().AtomRings() if metals & set(r)]
    free = Chem.RWMol(mol)  # the metal-free graph: which double bonds are still cyclic without the metal?
    for m in sorted(metals, reverse=True):
        for nb in [n.GetIdx() for n in free.GetAtomWithIdx(m).GetNeighbors()]:
            free.RemoveBond(m, nb)
    free = free.GetMol()
    Chem.FastFindRings(free)
    locked = set()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        in_metal_ring = any({a, c} <= r for r in metal_rings)
        fb = free.GetBondBetweenAtoms(a, c)
        if in_metal_ring and not (fb is not None and fb.IsInRing()):  # cyclic only because of the metal
            locked.add(frozenset((a, c)))
    return locked


def _lock_double_bond(work, fb):
    """Pin a coordination-locked double bond to an arbitrary definite stereo on ``work``; return True on success.

    Stops ``onlyUnassigned`` from enumerating it. The value is never grafted onto the full mol (``graft`` skips
    locked bonds), so the metal embed builds the one ring-feasible hand.

    ``SetStereoAtoms`` requires the two reference atoms in the bond's own begin/end order (each a neighbour of
    the corresponding end), so read the order off the bond, not off the unordered ``fb``.
    """
    from rdkit import Chem

    a, c = tuple(fb)
    wb = work.GetBondBetweenAtoms(a, c)
    if wb is None:
        return False
    bi, ei = wb.GetBeginAtomIdx(), wb.GetEndAtomIdx()
    nb_b = next((n.GetIdx() for n in work.GetAtomWithIdx(bi).GetNeighbors() if n.GetIdx() != ei), None)
    nb_e = next((n.GetIdx() for n in work.GetAtomWithIdx(ei).GetNeighbors() if n.GetIdx() != bi), None)
    if nb_b is None or nb_e is None:
        return False
    try:
        wb.SetStereoAtoms(nb_b, nb_e)
        wb.SetStereo(Chem.BondStereo.STEREOCIS)
    except (RuntimeError, ValueError):  # degrade to "not locked" rather than crash; the phantom just enumerates
        return False
    return True


def _build_enumeration_graph(mol, exclude):
    """Disconnect each metal and D-cap each freed sp3 donor so RDKit enumerates only ligand stereo.

    Returns ``(work, cap_to_metal)`` — the cap index → its metal's atomic number. With no `exclude` there is
    nothing to disconnect, so `mol` is returned unchanged.
    """
    from rdkit import Chem

    if not exclude:
        return mol, {}
    # Build the enumeration graph by DISCONNECTING each metal (bonds removed -> an isolated atom, never a false
    # stereocentre) and capping each metal-bound sp3 DONOR's freed valence with a DEUTERIUM. A dative metal bond
    # doesn't count toward a donor's valence, so RDKit sees a degree-3 phosphine and refuses the stereocentre; a
    # real single bond to D restores it. D (not a plain H, which would clobber a donor already carrying an H into
    # two identical substituents; not the metal, whose priority dominates) is distinct AND lowest-priority, so the
    # donor's R/S is the lone-pair convention. The D's are APPENDED, so every real atom index is preserved. The
    # embed carries the same cap (`metal._hold_donor_chirality`). A planar sp2 donor is left uncapped (non-stereo);
    # a backbone centre needs no cap — it is a normal stereocentre on the metal-free graph.
    cap_to_metal = {}  # D-cap atom index -> its metal's atomic number (for the coordinated-complex CIP label)
    work = Chem.RWMol(mol)
    for mi in exclude:
        z_metal = mol.GetAtomWithIdx(mi).GetAtomicNum()
        for nb in [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()]:
            work.RemoveBond(mi, nb)
            if mol.GetAtomWithIdx(nb).GetHybridization() == Chem.HybridizationType.SP3:
                d = work.AddAtom(Chem.Atom(1))
                work.GetAtomWithIdx(d).SetIsotope(2)  # deuterium
                work.AddBond(nb, d, Chem.BondType.SINGLE)
                work.GetAtomWithIdx(nb).SetNoImplicit(True)
                cap_to_metal[d] = z_metal
    work = work.GetMol()
    Chem.SanitizeMol(work, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    # The strip above can orphan a C=N whose stereo reference atom WAS the metal, and a flagged bond with no
    # references makes `FindPotentialStereo` below raise ("only can support 2 stereo neighbors"). The
    # tolerant sanitize happens to scrub most of them, but that is luck, not a contract — see `io`.
    _io.repair_bond_stereo(work)
    return work, cap_to_metal


def enumerate_unassigned(mol, cap=32, exclude=()):
    """Enumerate stereoisomers over ONLY the *unspecified* stereo elements (point R/S + double-bond E/Z).

    Returns ``(variants, n_unassigned, total, unresolved)``: ``variants`` a list of ``(variant_mol, label)``
    with the unlabeled centres expanded and every *defined* centre held fixed (`onlyUnassigned`), meso/duplicate
    configurations dropped (`unique`), truncated to ``cap`` of ``total`` possible; ``n_unassigned`` the count
    of unspecified elements (0, with ``variants == [(mol, '')]``, when the input is already fully defined);
    ``unresolved`` how many of those elements RDKit could NOT enumerate (an allene/cumulene axis or a flat
    biaryl atropisomer — `EnumerateStereoisomers` cannot encode axial chirality, so it stays one arbitrary hand
    and the caller must warn). Atom order is preserved, so index-based ``fix``/``constrain`` stay valid.

    A metal complex is safe: only genuine organic **point (tetrahedral) + double-bond** stereo is taken — the
    METAL atom (which `FindPotentialStereo` flags as `Atom_Octahedral`/`Atom_SquarePlanar`, or even a spurious
    `Atom_Tetrahedral`) is EXCLUDED, since its Λ/Δ is the coordination-isomer path's job (`constraints.polyhedron`)
    and RDKit's dative-metal stereo is not order-canonical. A *ligand* stereocentre — including a chiral-at-P or
    carbanion donor bonded to the metal (a genuine degree-4 → degree-3-after-strip tetrahedral centre) — IS
    enumerated. `exclude` is atom indices to never treat as a point centre (the metal indices).
    """
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        GetStereoisomerCount,
        StereoEnumerationOptions,
    )

    exclude = set(exclude)
    n_real = mol.GetNumAtoms()
    work, cap_to_metal = _build_enumeration_graph(mol, exclude)

    # A C=N / C=C whose E/Z is fixed by the coordination must not be enumerated. `FindPotentialStereo` runs on
    # `work` (metal disconnected), which opens any ring the metal closed, so a coordination-locked imine looks
    # like a free acyclic double bond and gets a phantom E and Z (an alpha-diimine chelate enumerates 2x2, most
    # of it unbuildable). A double bond endocyclic in a ring closed through the metal has one buildable geometry,
    # decided by the coordination isomer, not two hands.
    locked = _coordination_locked_double_bonds(mol, exclude)
    locked = {fb for fb in locked if _lock_double_bond(work, fb)}  # keep only the ones we could actually pin

    def enumerable(e):  # genuine organic point (R/S) + double-bond (E/Z); the isolated metal is never a centre
        if e.specified != Chem.StereoSpecified.Unspecified:
            return False
        if e.type == Chem.StereoType.Atom_Tetrahedral:
            return True
        if e.type == Chem.StereoType.Bond_Double:  # skip a double bond the coordination has already locked
            wb = work.GetBondWithIdx(e.centeredOn)
            return frozenset((wb.GetBeginAtomIdx(), wb.GetEndAtomIdx())) not in locked
        return False

    unassigned = [e for e in Chem.FindPotentialStereo(work) if enumerable(e)]
    if not unassigned:
        return [(mol, "")], 0, 1, 0
    atom_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Atom_Tetrahedral]
    bond_centers = [e.centeredOn for e in unassigned if e.type == Chem.StereoType.Bond_Double]
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=cap)
    total = GetStereoisomerCount(work, opts)
    work_isos = list(EnumerateStereoisomers(work, opts))

    def graft(wv):  # copy the enumerated ligand stereo (atom parity + E/Z) onto the FULL mol; skip the D caps
        full = Chem.Mol(mol)
        for a in wv.GetAtoms():
            if a.GetIdx() < n_real:  # a real atom (not an appended D)
                full.GetAtomWithIdx(a.GetIdx()).SetChiralTag(a.GetChiralTag())
        for b in wv.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            # skip a coordination-locked bond: its `work` stereo is the arbitrary lock value, not a real hand;
            # the full mol keeps it unspecified so the metal embed builds the ring-feasible geometry.
            if frozenset((i, j)) in locked:
                continue
            if b.GetStereo() != Chem.BondStereo.STEREONONE and i < n_real and j < n_real:
                fb = full.GetBondBetweenAtoms(i, j)
                if fb is not None:
                    fb.SetStereoAtoms(*b.GetStereoAtoms())
                    fb.SetStereo(b.GetStereo())
        return full

    variants = [(graft(wv), _stereo_label(wv, atom_centers, bond_centers, cap_to_metal)) for wv in work_isos]
    probe = work_isos[0] if work_isos else work  # centres still UNSPECIFIED after enumeration = axial (allene)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    unresolved = sum(
        1 for i in atom_centers if probe.GetAtomWithIdx(i).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    ) + sum(1 for b in bond_centers if probe.GetBondWithIdx(b).GetStereo() == Chem.BondStereo.STEREONONE)
    return variants, len(unassigned), total, unresolved
