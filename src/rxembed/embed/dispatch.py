"""Embed dispatch: route a source + spec into embedded conformers (the machinery behind `pipeline.embed`).

Turns SMILES / .xyz / Mol / metal `Isomer` + a constraint spec into an `Ensemble` (or an `EnsembleSet` of
candidates). Owns the source normalisation, the metal carbon-surrogate context, the vdW encounter bounds, the
Kabsch frozen-core graft, and the isomer / template / auto-NCI routing. The user-facing surface (`embed`,
`Ensemble`, `EnsembleSet`, `wrap`) lives in `rxembed.pipeline`; this module constructs those objects.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from rxembed import refine as _refine
from rxembed import stereo as _stereo
from rxembed.constraints import add_distance, resolve_atom, resolve_core
from rxembed.constraints import metal as _metal
from rxembed.constraints import nci as _nci
from rxembed.log import logger
from rxembed.pipeline import Ensemble, EnsembleSet

from . import bounds as _embed

_PT = Chem.GetPeriodicTable()
_EPS = 1e-6  # near-zero norm floor for the graft axis
_AROMATIC_BO_TOL = 0.25  # |bond_order - 1.5| within this reads as aromatic
_BOND_ATOMS = 2  # a two-atom frozen core is a bond: fix its length, not an orientation
_XYZ_DIM = 3  # an (x, y, z) coordinate
_TEMPLATE_LEN = 2  # template= is (reference, mapping)
_MIN_FRAGS = 2  # below this there is no inter-fragment separation to enforce


def _xyz_to_mol(path, charge=0):
    """Read an ``.xyz`` into an RDKit Mol with **perceived bonds and a conformer** (robust for metals/TS).

    Bonds come from **xyzgraph** (transition-metal-aware perception + bond-order optimiser), not RDKit's
    organic-only ``rdDetermineBonds`` (which raises on a metal). The graph is converted to an RWMol and
    only *leniently* sanitised (ring perception, but no valence/property checks that choke on a metal),
    so a metal complex or a TS from xyz Just Works. Index addressing always works; SMARTS works too for the
    organic part. Pass `charge` for a charged species. Falls back to ``rdDetermineBonds`` only if xyzgraph
    is unavailable. (Do **not** pass ``quick=True`` to ``build_graph`` here — it skips bond-order/charge
    perception and would return all-single bonds.)
    """
    try:
        import xyzgraph
    except ImportError:
        from rdkit.Chem import rdDetermineBonds

        mol = Chem.MolFromXYZFile(path)
        if mol is None:
            raise ValueError(f"could not read {path} as an .xyz") from None
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        return mol
    from rdkit.Chem import BondType, Conformer
    from rdkit.Geometry import Point3D

    g = xyzgraph.build_graph(path, charge=charge, kekule=True)  # integer bond orders (no 1.5)
    order = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE}
    rw = Chem.RWMol()
    idx = {}
    for n, d in sorted(g.nodes(data=True)):
        a = Chem.Atom(int(d["atomic_number"]))
        a.SetFormalCharge(round(d.get("formal_charge", 0) or 0))
        a.SetNoImplicit(True)  # xyz is fully explicit (incl. H)
        idx[n] = rw.AddAtom(a)
    for u, v, d in g.edges(data=True):
        bo = d.get("bond_order", 1.0)
        if abs(bo - 1.5) < _AROMATIC_BO_TOL:  # aromatic (only if kekule fell through)
            b = rw.AddBond(idx[u], idx[v], BondType.AROMATIC) - 1
            rw.GetBondWithIdx(b).SetIsAromatic(True)
            rw.GetAtomWithIdx(idx[u]).SetIsAromatic(True)
            rw.GetAtomWithIdx(idx[v]).SetIsAromatic(True)
        else:
            rw.AddBond(idx[u], idx[v], order.get(round(bo), BondType.SINGLE))
    mol = rw.GetMol()
    conf = Conformer(mol.GetNumAtoms())
    for n, d in g.nodes(data=True):
        x, y, z = d["position"]
        conf.SetAtomPosition(idx[n], Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(
        mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
    )  # rings yes, valence checks no
    try:
        Chem.AssignStereochemistryFrom3D(mol)  # point R/S + E/Z from the geometry -> graph tags,
    except Exception:  # so the embed preserves them exactly as it would
        pass  # for a SMILES @/@@ (an .xyz behaves like SMILES)
    return mol


def parse_smiles(smi):
    """Parse a SMILES to a Mol, raising a clear error instead of returning ``None`` (which crashes downstream)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smi!r}")
    return mol


def _normalize(source, charge=0):
    """Normalise SMILES / an .xyz path / an RDKit Mol to ``(mol with Hs, has_geometry)``.

    `charge` is used for xyz bond perception (a charged metal complex / ion).
    """
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path, not just str
    if isinstance(source, Chem.Mol):
        mol = source
    elif isinstance(source, str) and source.lower().endswith(".xyz"):
        mol = _xyz_to_mol(source, charge)
        if mol is None:
            raise ValueError(f"could not read {source}")
    else:
        mol = parse_smiles(source)
    has_geom = mol.GetNumConformers() > 0
    return Chem.AddHs(mol, addCoords=has_geom), has_geom


def _encounter_bounds(mol, slack=1.5):
    """vdW-aware inter-fragment bounds so a multi-fragment SMILES does not embed on top of itself.

    For every pair of fragments, bound their closest heavy-atom pair to roughly van-der-Waals contact
    (sum of vdW radii .. + slack), giving the embedder a separated but touching encounter geometry.
    """
    tmp = Chem.Mol(mol)
    if rdDistGeom.EmbedMolecule(tmp, rdDistGeom.ETKDGv3()) != 0:
        return {}
    pt = Chem.GetPeriodicTable()
    pos = tmp.GetConformer().GetPositions()
    frags = Chem.GetMolFrags(mol)

    def heavy(f):
        return [i for i in f if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]

    bounds = {}
    for a in range(len(frags)):
        for b in range(a + 1, len(frags)):
            fa, fb = heavy(frags[a]), heavy(frags[b])
            if not fa or not fb:
                continue
            i, j = min(((i, j) for i in fa for j in fb), key=lambda p: np.linalg.norm(pos[p[0]] - pos[p[1]]))
            vdw = pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) + pt.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
            bounds[(i, j)] = (vdw, vdw + slack)
    return bounds


@dataclass
class _MetalCtx:
    mol: Chem.Mol
    metal: int
    real_z: int
    donors: list | None = None  # the metal's real donor atoms (for the donor-proton relax); None = skip it
    geometry: str | None = None  # the coordination polyhedron name (only named polyhedra get the proton relax)
    extra: list = field(default_factory=list)  # other surrogated metals (idx, real_z) in a multi-metal complex

    def restore(self):
        self.mol.GetAtomWithIdx(self.metal).SetAtomicNum(self.real_z)
        for mi, rz in self.extra:  # restore the rest of a bimetallic complex
            self.mol.GetAtomWithIdx(mi).SetAtomicNum(rz)

    def fix_donor_protons(self, cons, ids, distance_fc):
        """Stage-2 of the metal relax: settle the donor **hydrogens** with the coordination bonds restored.

        The main relax (stage-1) runs with the metal-donor bonds *removed* (the surrogate is a bondless
        anchor), which leaves a bonded donor under-coordinated, so UFF mis-hybridises it and its protons
        splay (M-D-H ~127° instead of ~109°). Here we re-add the metal-donor bonds, swap to a phosphorus
        surrogate (which UFF-types the *bonded* metal through CN6, unlike carbon), and relax with **every
        atom pinned except the donor hydrogens**. The donor regains correct hybridisation so the protons
        fall to ~109°, while the polyhedron and metal-donor distances from stage-1 are held exactly.

        No-op (nothing lost) when there are no donor H's, the geometry is not a named polyhedron (an
        arbitrary/retained or high-CN centre — ferrocene's η⁵ carbons have no donor protons), or the
        bonded metal still won't UFF-type.
        """
        if not self.donors or self.geometry not in _metal.ANGLES:
            return
        donor_h = [
            x.GetIdx() for d in self.donors for x in self.mol.GetAtomWithIdx(d).GetNeighbors() if x.GetAtomicNum() == 1
        ]
        if not donor_h:
            return
        em = Chem.RWMol(self.mol)
        for d in self.donors:
            if em.GetBondBetweenAtoms(d, self.metal) is None:
                em.AddBond(d, self.metal, Chem.BondType.SINGLE)
        a = em.GetAtomWithIdx(self.metal)
        a.SetAtomicNum(_metal.RELAX_SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
        relax = em.GetMol()
        Chem.SanitizeMol(
            relax, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True
        )
        relax.UpdatePropertyCache(strict=False)
        with rdBase.BlockLogs():  # the P surrogate trips a benign per-atom UFF
            if not rdForceFieldHelpers.UFFHasAllMoleculeParams(relax):  # "unrecognized charge state" log — mute it
                logger.debug(
                    "metal: bonded surrogate won't UFF-type (CN%d) — skipping donor-proton relax", len(self.donors)
                )
                return
            guided = copy.deepcopy(cons)  # guide each donor H toward a tetrahedral angle
            for d in self.donors:  # so a conformer whose stage-1 protons splayed
                for h in (x.GetIdx() for x in self.mol.GetAtomWithIdx(d).GetNeighbors() if x.GetAtomicNum() == 1):
                    guided.angles[(self.metal, d, h)] = (100.0, 118.0)  # away from the metal can't stick there.
            # NB the window is tuned for the dominant sp3 N/O donors (amine/water/alcohol/phosphine), where
            # ~109° is right; a rare sp2 or sulfur donor proton may land a few degrees off (an sp3-vs-sp2
            # split isn't worth the per-element complexity for how seldom those coordinate via an X-H).
            frozen = [i for i in range(relax.GetNumAtoms()) if i not in set(donor_h)]
            _refine.restrained_uff(relax, guided, distance_fc=distance_fc, extra_frozen=frozen)
        for c in ids:  # copy ONLY the donor-H positions back
            src, dst = relax.GetConformer(c), self.mol.GetConformer(c)
            for h in donor_h:
                dst.SetAtomPosition(h, src.GetAtomPosition(h))
        logger.debug("metal: settled %d donor proton(s) with coordination bonds restored", len(donor_h))


def _nci_windows(contacts):
    """Fold NCI ``contacts`` into ``(distances, angles)`` windows (soft, releasable by ``mc(explore=)``).

    `contacts` may be a single Contact, a list of them, a ``{label: Contact}`` dict (uses all), or a raw
    ``{(i,j):(lo,hi)}`` distance dict (indices, already windowed).
    """
    distances, angles = {}, {}
    if contacts is None:
        return distances, angles
    items: list[_nci.Contact] = []
    if isinstance(contacts, _nci.Contact):
        items = [contacts]
    elif isinstance(contacts, dict):
        vals = [v for v in contacts.values() if isinstance(v, _nci.Contact)]
        if vals and len(vals) != len(contacts):  # {label: Contact} must be ALL Contacts
            raise TypeError("contacts: dict mixes Contact and non-Contact values")
        if vals:
            items = vals
        else:
            distances.update(contacts)  # a raw {(i,j):(lo,hi)} distance dict
    else:
        items = [c for c in contacts if isinstance(c, _nci.Contact)]
        if len(items) != len(list(contacts)):
            raise TypeError("contacts: list must contain Contact objects (from rx.nci_candidates)")
    for ct in items:
        distances.update(ct.distances)
        angles.update(ct.angles)
    return distances, angles


def _add_soft(cons, distances, angles):
    """Add soft (releasable) distance/angle windows to `cons`, recording them in the ``contacts`` provenance.

    Used for NCI ``contacts=`` grips folded on top of the resolved ``fix``/``constrain`` core. ``mc(explore=)``
    releases exactly these (and any ``constrain=`` windows) while structural holds stay put. **Explicit wins**
    (DESIGN §5.2): a soft window that lands on a pair the user already pinned with a ``fix`` number (a
    structural distance/angle *not* already soft) is dropped — the rigid fix is neither overwritten nor made
    releasable.
    """
    dk, ak = set(cons.contacts[0]), set(cons.contacts[1])
    struct_d = {k for k in cons.distances if k not in dk}  # fix numbers / frozen-core shape — non-releasable
    struct_a = {k for k in cons.angles if k not in ak}
    for (i, j), (lo, hi) in distances.items():
        key = (min(i, j), max(i, j))
        if key in struct_d:  # user fix on this pair overrides a soft grip — leave it rigid
            continue
        add_distance(cons.distances, i, j, lo, hi)
        dk.add(key)
    for akey, val in angles.items():
        key = tuple(akey)
        if key in struct_a:
            continue
        cons.angles[key] = val
        ak.add(key)
    cons.contacts = (frozenset(dk), frozenset(ak))


def _graft_frozen(mol, conf_ids, frozen, ref):
    """Restore the `frozen` atoms to their EXACT input geometry `ref`, Kabsch-fitted onto the embedded pose.

    Distance geometry only *approximates* a rigid core (~0.2-0.4 Å internal RMSD — fine for a normal
    molecule, wrong for a TS whose partial bonds must be preserved), so we restore the core EXACTLY and
    orient it to best match the embedded periphery. The frozen atoms are then held by `AddFixedPoint`.
    """
    frozen = list(frozen)
    core = np.asarray(ref, float)  # input core, input frame
    if len(frozen) <= 1:  # a point has no shape to restore
        return
    if len(frozen) == _BOND_ATOMS:  # a bond: no orientation, but fix the LENGTH exactly
        dt = float(np.linalg.norm(core[0] - core[1]))  # (keep one atom, slide the other to the template
        for c in conf_ids:  #  distance along the embedded axis)
            conf = mol.GetConformer(c)
            pa = np.array(conf.GetAtomPosition(frozen[0]))
            pb = np.array(conf.GetAtomPosition(frozen[1]))
            v = pb - pa
            n = np.linalg.norm(v)
            v = v / n if n > _EPS else np.array([1.0, 0.0, 0.0])
            conf.SetAtomPosition(frozen[1], (pa + dt * v).tolist())
        return
    ca = core.mean(0)
    core0 = core - ca
    for c in conf_ids:
        conf = mol.GetConformer(c)
        emb = np.array([list(conf.GetAtomPosition(i)) for i in frozen])  # embedded core
        cb = emb.mean(0)
        h = core0.T @ (emb - cb)
        u, _s, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T  # rotate input core onto the embedded one
        fitted = core0 @ r.T + cb
        for i, p in zip(frozen, fitted, strict=False):
            conf.SetAtomPosition(i, p.tolist())


def _embed_isomer(iso, *, coordinate, contacts, fix, constrain, n, seed, knowledge, keep_input=False):
    """Embed a metal `Isomer`, optionally binding a substrate; yield one `Ensemble` per binding candidate.

    Usually one, but several when ``coordinate=`` is a SMARTS matching several donor atoms (one candidate
    per donor, so the user can `select` the binding they want or keep them all). Each carries a ``.tag``;
    `keep_input` adds the Mol's input conformer to the ensemble (the retain-input-arrangement path).
    """
    base = copy.deepcopy(iso.cons)
    graft_ref: dict = {}  # explicit/own coords the resolver wants Kabsch-grafted (a substrate fix on the metal)
    if contacts is not None or fix or constrain:  # a substrate bound via fix / constrain / NCI contacts
        has_geom = iso.mol.GetNumConformers() > 0
        sub, graft_ref = resolve_core(iso.mol, fix=fix, constrain=constrain, has_geometry=has_geom)
        _add_soft(sub, *_nci_windows(contacts))
        sphere_d, sphere_a = set(base.distances), set(base.angles)  # the coordination sphere already in base —
        soft_d, soft_a = sub.contacts  # never release it, so provenance records ONLY genuinely-new
        base.distances.update(sub.distances)  # substrate contacts: a spec landing ON a sphere hold is a user
        base.angles.update(sub.angles)  # override of that hold, not a releasable grip. mc(explore=) then frees
        base.frozen |= sub.frozen  # only the substrate; the sphere + any coordinate= dative bond stay held.
        base.contacts = (
            frozenset(k for k in soft_d if k not in sphere_d),
            frozenset(k for k in soft_a if k not in sphere_a),
        )

    nvac = iso.vertices.count(_metal.VACANT)
    if coordinate is None:
        choices = [None]
    elif coordinate == "auto":
        atoms = _metal.lone_pair_donors(iso.mol, iso.metal, exclude=iso.donors)
        if len(atoms) > nvac:
            logger.info(
                "coordinate=auto: %d candidate donor(s) for %d vacant site(s); using %d "
                "(name them explicitly to choose)",
                len(atoms),
                nvac,
                nvac,
            )
        choices = [atoms[:nvac]]
    elif isinstance(coordinate, (list, tuple)):  # explicit: one spec per vacancy
        choices = [[resolve_atom(iso.mol, s) for s in coordinate]]
    elif isinstance(coordinate, str):  # a SMARTS — may match several atoms
        ms = [m[0] for m in iso.mol.GetSubstructMatches(Chem.MolFromSmarts(coordinate))]
        if not ms:
            raise ValueError(f"coordinate={coordinate!r} matched no atoms")
        if len(ms) == 1:
            choices = [[ms[0]]]
        elif nvac == 1:  # ambiguous + 1 pocket -> one mode per donor
            logger.info(
                "coordinate=%r matched %d atoms; embedding %d coordination modes (one per donor) — "
                ".select(coordinate=<idx>) one or keep all",
                coordinate,
                len(ms),
                len(ms),
            )
            choices = [[a] for a in ms]
        else:  # ambiguous + several pockets -> can't map
            raise ValueError(
                f"coordinate={coordinate!r} matched {len(ms)} atoms for {nvac} vacant site(s); "
                f"name them explicitly as a list, e.g. ['[OX1]', '[OX2]']"
            )
    else:  # an int index
        choices = [[resolve_atom(iso.mol, coordinate)]]

    for atoms in choices:
        mol = Chem.Mol(iso.mol)  # own copy so candidates don't share conformers
        input_conf = Chem.Conformer(mol.GetConformer()) if (keep_input and mol.GetNumConformers()) else None
        cons = copy.deepcopy(base)
        # identity is geometric: arrangement (slot map) + metal chirality. `label` (cis/trans/mer/fac) is
        # kept only as a coarse, sometimes-wrong convenience tag — never the thing you select on.
        tag = {
            "geometry": iso.geometry,
            "arrangement": _metal.arrangement(iso),
            "chirality": iso.chirality,
            "label": iso.label,
        }
        if atoms is not None:
            extra = _metal.coordinate(iso, atoms)
            for (i, j), (lo, hi) in extra.distances.items():
                add_distance(cons.distances, i, j, lo, hi)
            cons.angles.update(extra.angles)
            if len(choices) > 1:
                tag["coordinate"] = atoms[0]
        # tether any free fragment (a co-crystallised solvent / an un-bonded ligand the SMILES wrote as a
        # separate `.` fragment) at vdW contact, so it embeds as a vdW complex rather than drifting to
        # infinity — the same encounter-bounds the non-metal multi-fragment path applies.
        cons.distances.update(_float_encounter_bounds(mol, cons))
        frozen = sorted(cons.frozen)  # capture the input TS core BEFORE the embed
        if frozen and mol.GetNumConformers():  # graft each frozen atom to its explicit-fix coord, else own coord
            own = mol.GetConformer().GetPositions()
            ref_core = np.array([graft_ref.get(i, own[i]) for i in frozen])
        elif graft_ref:  # SMILES metal + explicit-coords fix: no conformer, graft only the named atoms
            frozen = sorted(graft_ref)
            ref_core = np.array([graft_ref[i] for i in frozen])
        else:
            ref_core = None
        try:
            ids = _embed.embed(mol, cons, n or _embed.n_confs(mol, constrained=True), seed=seed, knowledge=knowledge)
        except RuntimeError as e:  # triangle smoothing -> infeasible bounds
            raise ValueError(
                f"could not embed {iso.geometry} {_metal.arrangement(iso)}"
                + (f" with atom(s) {atoms} coordinated" if atoms else "")
                + f": the coordination + substrate constraints are geometrically infeasible "
                f"(e.g. a substrate that can't chelate the requested vertices). [{e}]"
            ) from e
        ids = list(ids)
        if ref_core is not None:
            _graft_frozen(mol, ids, frozen, ref_core)  # restore the exact frozen TS core
        if input_conf is not None:  # ETKDG cleared confs; re-add the input as a seed
            ids = [mol.AddConformer(input_conf, assignId=True), *ids]
        logger.info(
            "embed[%s: %s%s%s]: %d seeds%s",  # name-agnostic identity: arrangement (+ chirality), not cis/trans
            iso.geometry,
            _metal.arrangement(iso),
            f" {iso.chirality}" if iso.chirality else "",
            f" coord@{atoms}" if atoms else "",
            len(ids),
            " (incl. input geometry)" if input_conf else "",
        )
        ens = Ensemble(mol, ids, cons, _MetalCtx(mol, iso.metal, iso.real_z, iso.donors, iso.geometry, extra=iso.extra))
        ens.tag = tag
        yield ens


def _auto_contacts_embed(source, *, metal, fix, constrain, coordinate, charge, n, seed, knowledge):
    """Resolve ``contacts='auto'``: discover the inter-fragment NCI binding modes and conf-search each.

    One `Ensemble` if there is a single mode, else an `EnsembleSet` (`.tag['nci']` = the mode).
    A mode that cannot be embedded (geometrically infeasible clamp) is skipped with a warning, not fatal.
    """
    disc = source.mol if isinstance(source, _metal.Isomer) else _normalize(source, charge)[0]
    try:
        modes = _nci.auto_binding_modes(disc, seed=seed)
    except Exception as err:
        raise ValueError(
            f"contacts='auto' could not discover binding modes ({type(err).__name__}: {err}); "
            "pass a geometry to discover from (a multi-fragment SMILES or an .xyz), or give "
            "contacts= explicitly"
        ) from err
    common = {
        "metal": metal,
        "fix": fix,
        "constrain": constrain,
        "coordinate": coordinate,
        "charge": charge,
        "n": n,
        "seed": seed,
        "knowledge": knowledge,
    }
    if not modes:
        logger.info("contacts='auto': no inter-fragment NCI binding mode detected -> plain embed")
        return _embed_dispatch(source, contacts=None, **common)
    logger.info("contacts='auto': %d binding mode(s) -> %s", len(modes), list(modes))
    out = EnsembleSet()
    for label, contact in modes.items():
        try:
            res = _embed_dispatch(source, contacts=contact, **common)
        except (ValueError, RuntimeError) as err:  # infeasible bounds (triangle smoothing) too
            logger.warning("contacts='auto': mode '%s' skipped (could not embed): %s", label, err)
            continue
        for ens in res if isinstance(res, EnsembleSet) else [res]:
            if not ens.ids:  # a 0-conformer embed (e.g. an orphaned anchor)
                logger.warning("contacts='auto': mode '%s' produced no conformers; skipped", label)
                continue
            ens.tag = {**(ens.tag or {}), "nci": label}
            out.append(ens)
    if not out:
        raise ValueError("contacts='auto': none of the discovered binding modes could be embedded")
    return out[0] if len(out) == 1 else out


def _attach_stereo(result, source, charge, stereo):
    """Tag the embedded ensemble(s) with the chirality filter ``(spec, reference signature)`` for `minimize`.

    ``'auto'`` engages ``'preserve'`` ONLY when the input geometry carries chirality
    the embed can't keep (planar/axial/helical); a SMILES or a stereo-less input is left untouched.
    """
    if stereo == "free":
        return
    ref = source.stereo_ref if isinstance(source, _metal.Isomer) else None
    if ref is None and not isinstance(source, _metal.Isomer):
        try:
            mol, has_geom = _normalize(source, charge)
            ref = _stereo.signature(mol, charge=charge) if has_geom else None
        except Exception:
            ref = None
    if not ref:
        return
    if stereo == "auto":  # auto-preserve ONLY a metallocene's PLANAR
        #   chirality — the handedness the embed genuinely cannot keep. xyzgraph's AXIAL/HELICAL perception is
        #   geometry-dependent and over-fires on labile, freely-rotating bonds (an aryl-N or P=N-C reads a
        #   different Rₐ/Sₐ every rotamer, not a real atropisomer), so auto-preserving it rejects perfectly
        #   valid conformers (the BImP TS embed kept 0/4). Point R/S is the embed's own job (ETKDG enforces the
        #   3D-assigned tags). A genuine organic atropisomer: opt in with stereo={'axial': 'preserve'}.
        spec = {"planar": "preserve", "default": "free"} if "planar" in set(ref) else None
    else:
        spec = stereo
    if spec is None:
        return
    for ens in result if isinstance(result, EnsembleSet) else [result]:
        if isinstance(ens, Ensemble):
            ens._stereo = (spec, ref)


def _reference_positions(reference, charge=0):
    """Resolve a template reference to an ``(N, 3)`` positions array (.xyz path / Mol / Ensemble / array)."""
    if isinstance(reference, Ensemble):
        cid = reference.ids[0] if reference.ids else reference.mol.GetConformers()[0].GetId()
        return reference.mol.GetConformer(cid).GetPositions()
    if isinstance(reference, Chem.Mol):
        if reference.GetNumConformers() == 0:
            raise ValueError("template reference Mol needs a conformer (a 3D geometry)")
        return reference.GetConformer().GetPositions()
    if isinstance(reference, os.PathLike):
        reference = os.fspath(reference)
    if isinstance(reference, str) and reference.lower().endswith(".xyz"):
        return _xyz_to_mol(reference, charge).GetConformer().GetPositions()
    arr = np.asarray(reference, float)
    if arr.ndim == _TEMPLATE_LEN and arr.shape[1] == _XYZ_DIM:
        return arr
    raise ValueError("template reference must be an .xyz path, a Mol with a conformer, an Ensemble, or an (N,3) array")


def _resolve_template_spec(template, charge):
    """Turn a ``template=(reference, {target_i: ref_i})`` kwarg into ``(positions, mapping)`` for `resolve_core`."""
    if not (isinstance(template, (tuple, list)) and len(template) == _TEMPLATE_LEN and isinstance(template[1], dict)):
        raise ValueError(
            "template= must be (reference, {target_index: reference_index}) — an explicit atom map "
            "(order-proof, no hidden SMARTS matching). E.g. template=('ts.xyz', {0: 5, 4: 1, 5: 6})."
        )
    reference, mapping = template
    return _reference_positions(reference, charge), mapping


def _float_encounter_bounds(mol, cons):
    """Bound fragments no constraint pins to vdW contact — so a free / stray fragment can't drift off.

    Generalises the old two cases into one: a fully unconstrained multi-fragment SMILES (every fragment
    floats -> all pairs bounded) and a fixed/templated core that leaves a spectator fragment (a counter-ion)
    untethered. A pair already linked by a fix / constrain / contact is left to that constraint.
    """
    frags = Chem.GetMolFrags(mol)
    if len(frags) < _MIN_FRAGS:
        return {}
    frag_of = {a: fi for fi, f in enumerate(frags) for a in f}
    touched = {frag_of[a] for a in cons.constrained_atoms()}
    if len(touched) >= len(frags):  # every fragment already pinned/linked
        return {}
    return {
        k: v for k, v in _encounter_bounds(mol).items() if frag_of[k[0]] not in touched or frag_of[k[1]] not in touched
    }


def _embed_dispatch(
    source,
    *,
    metal=None,
    fix=None,
    constrain=None,
    template=None,
    contacts=None,
    coordinate=None,
    charge=0,
    n=None,
    seed=0xF00D,
    knowledge=True,
):
    """Dispatch the embed by input type / spec — see the public `embed` for documentation."""
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path everywhere downstream
    if coordinate is not None and metal is None and not isinstance(source, _metal.Isomer):
        raise ValueError(
            "coordinate= only applies to a metal (pass metal=<geometry> or a metal Isomer "
            f"from rx.metal(...)); got {type(source).__name__} — use contacts=/constrain= otherwise"
        )
    if contacts == "auto":  # discover binding modes -> conf-search each
        return _auto_contacts_embed(
            source,
            metal=metal,
            fix=fix,
            constrain=constrain,
            coordinate=coordinate,
            charge=charge,
            n=n,
            seed=seed,
            knowledge=knowledge,
        )
    if metal is not None or isinstance(source, _metal.Isomer):  # the metal enumerate / isomer paths
        if template is not None:
            raise ValueError("template= pins a core by graft; it does not compose with metal=/an Isomer source")
        _iso_kw = {
            "coordinate": coordinate,
            "contacts": contacts,
            "fix": fix,
            "constrain": constrain,
            "n": n,
            "seed": seed,
            "knowledge": knowledge,
        }
        if metal is not None:
            if isinstance(source, _metal.Isomer):
                raise ValueError("pass either a metal Isomer source OR metal=<geometry>, not both")
            if isinstance(source, str) and source.lower().endswith(".xyz"):
                source = _normalize(source, charge)[0]  # xyz -> perceived Mol (enumerate wants a Mol/SMILES)
            out = EnsembleSet(
                e for iso in _metal.enumerate_isomers(source, metal) for e in _embed_isomer(iso, **_iso_kw)
            )
            logger.info(
                "metal: ENUMERATING isomers of %s -> %d distinct candidate(s) (each a separate "
                "ensemble; .summary() / .select() one to conf-search)",
                metal,
                len(out),
            )
            return out
        results = list(_embed_isomer(source, **_iso_kw))  # a single chosen isomer
        return results[0] if len(results) == 1 else EnsembleSet(results)

    mol, has_geom = _normalize(source, charge)
    if (
        has_geom
        and _metal.metal_index(mol) is not None
        and not (fix or constrain or contacts or coordinate or template)
    ):  # retain the input arrangement
        iso = _metal.from_geometry(mol)
        logger.info(
            "metal: source carries a geometry and no metal= -> RETAINING the input ligand "
            "arrangement (%s: %s); pass metal=<geometry> to enumerate isomers instead",
            iso.geometry,
            _metal.arrangement(iso),
        )
        return next(
            _embed_isomer(
                iso,
                coordinate=None,
                contacts=None,
                fix=None,
                constrain=None,
                n=n,
                seed=seed,
                knowledge=knowledge,
                keep_input=True,
            )
        )
    metal_ctx = None
    metal_core = set()
    metals_donors = {}
    hydrides = []
    if _metal.metal_index(mol) is not None and (fix or constrain or contacts or template):
        if has_geom:  # capture each metal's donors BEFORE the bonds
            metals_donors = {
                mi: [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()] for mi in _metal.metal_indices(mol)
            }
        else:  # no geometry to hold the sphere from -> at least
            hydrides = [
                (mi, mol.GetAtomWithIdx(mi).GetAtomicNum(), nb.GetIdx())  # keep a terminal HYDRIDE bonded
                for mi in _metal.metal_indices(mol)  # so an MH NCI donor mode (M-H
                for nb in mol.GetAtomWithIdx(mi).GetNeighbors()  # to an acceptor) survives the
                if nb.GetAtomicNum() == 1 and nb.GetDegree() == 1
            ]  # surrogate's bond strip
        mol, metals, core_donors = _metal.prepare_all(mol)  # surrogate EVERY metal (bimetallic-safe)
        (m, real_z), extra = metals[0], metals[1:]
        metal_ctx = _MetalCtx(mol, m, real_z, extra=extra)
        _metal_core_donors = core_donors  # deferred: only pin the sphere if a core is actually grafted (below)
        logger.debug("metal complex: %d metal(s) swapped to carbon surrogate for the FF", len(metals))
    else:
        _metal_core_donors = []

    tmpl = _resolve_template_spec(template, charge) if template is not None else None
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, template=tmpl, has_geometry=has_geom)
    _add_soft(cons, *_nci_windows(contacts))  # NCI grips: soft, releasable by mc(explore=)
    #   fix numbers + the frozen-core SHAPE + the metal-sphere / M-H / encounter holds are all structural and
    #   stay OUT of `contacts` (resolve_core recorded only constrain=; _add_soft only the NCI grips), so
    #   mc(explore=) releases exactly the soft grips and never the structure.
    user_graft = dict(ref)  # atoms to Kabsch-graft onto their exact coords (own / explicit / template)
    if user_graft and _metal.metal_index(mol) is not None:  # a grafted core -> pin the metal coordination
        metal_core = set(_metal.metal_indices(mol)) | set(_metal_core_donors)  # cores so they can't drift
    cons.frozen |= metal_core
    for mi, dons in metals_donors.items():  # a spectator metal is held intact-but-achiral by
        _metal.hold_shape(mol, [mi, *dons], cons)  # hold_shape (NOT grafted — its handedness must
        #                                          stay free for the stereo filter), then pinned at the embed
    for mi, rz, h in hydrides:  # covalent M-H window (no input geometry to read)
        d = _PT.GetRcovalent(rz) + _PT.GetRcovalent(1)
        add_distance(cons.distances, mi, h, d - 0.1, d + 0.15)
    cons.distances.update(_float_encounter_bounds(mol, cons))  # keep free/stray fragments from drifting off

    n = n or _embed.n_confs(mol, constrained=cons.is_constrained)
    graft_atoms = sorted(user_graft)
    ref_core = np.array([user_graft[i] for i in graft_atoms]) if graft_atoms else None
    ids = _embed.embed(mol, cons, n, seed=seed, knowledge=knowledge)
    if ref_core is not None:
        _graft_frozen(mol, ids, graft_atoms, ref_core)  # restore the exact fixed core (partial bonds preserved)
    n_frag = len(Chem.GetMolFrags(mol))
    logger.info(
        "embed: %d seeds (%d atoms, %d fragment%s, %d constraints)",
        len(ids),
        mol.GetNumAtoms(),
        n_frag,
        "s" if n_frag != 1 else "",
        len(cons.distances) + len(cons.angles),
    )
    return Ensemble(mol, list(ids), cons, metal_ctx)
