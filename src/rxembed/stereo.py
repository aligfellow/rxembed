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


def passes(sig, ref, spec="preserve"):
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


def kinds(sig):
    """Return the chirality element kinds present in ``sig`` (``'planar'``, ``'point'``, ...).

    Lets a caller decide if chirality control is even relevant (no planar/axial/helical and SMILES-defined
    point -> the embed handles it).
    """
    return set(sig)


def relevant(ref):
    """Return whether ``ref`` carries chirality the embed would drop (planar/axial/helical).

    That is whether the auto-default should engage; a molecule with only point/EZ stereo needs no filtering
    (RDKit keeps it).
    """
    return bool(_NONGRAPH & set(ref))


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
    # Build the enumeration graph by DISCONNECTING each metal (its bonds removed -> an isolated atom, never a
    # false stereocentre) and capping each metal-bound DONOR stereocentre's freed valence with a DEUTERIUM. This
    # is the fix for OIN-SMILES's zone-A problem: a DATIVE metal bond doesn't count toward a donor's valence, so
    # RDKit sees a degree-3 phosphine and refuses the stereocentre; a real single bond (to D) restores it. D (not
    # a plain H, which would clobber a donor already carrying an H into two identical substituents; not the metal,
    # whose priority would dominate) is distinct AND lowest-priority — so the donor's R/S is the lone-pair
    # convention. The D's are APPENDED, so every real atom index is preserved.
    #
    # Every PYRAMIDAL (sp3) donor is capped; the embed carries the same D-cap (`metal._hold_donor_chirality`) so
    # a donor ETKDG would not otherwise hold as a degree-3 centre (a carbanion-C, an amine-N) still embeds its two
    # hands distinctly. A planar sp2 donor (a conjugated amidate/thiourea N) is left uncapped and stays non-stereo.
    # A backbone (non-donor) centre needs no cap — it is a normal stereocentre on the metal-free graph.
    cap_to_metal = {}  # D-cap atom index -> its metal's atomic number (for the coordinated-complex CIP label)
    if exclude:
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
        Chem.SanitizeMol(
            work, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
        )
    else:
        work = mol

    def enumerable(e):  # genuine organic point (R/S) + double-bond (E/Z); the isolated metal is never a centre
        return e.specified == Chem.StereoSpecified.Unspecified and e.type in (
            Chem.StereoType.Atom_Tetrahedral,
            Chem.StereoType.Bond_Double,
        )

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
