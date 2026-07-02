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
from rxembed.constraints import add_distance, from_spec, from_template, resolve_atom
from rxembed.constraints import metal as _metal
from rxembed.constraints import nci as _nci
from rxembed.log import logger
from rxembed.pipeline import Ensemble, EnsembleSet

from . import bounds as _embed

_PT = Chem.GetPeriodicTable()
_EPS = 1e-6  # near-zero norm floor for the graft axis
_AROMATIC_BO_TOL = 0.25  # |bond_order - 1.5| within this reads as aromatic
_BOND_ATOMS = 2  # a two-atom frozen core is a bond: fix its length, not an orientation


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
        mol = Chem.MolFromSmiles(source)
        if mol is None:
            raise ValueError(f"bad SMILES: {source!r}")
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


def _merge_contacts(distances, angles, contacts):
    """Fold ``Contact`` orientation into the distance/angle spec.

    `contacts` may be a single Contact, a list of them, a ``{label: Contact}`` dict (uses all), or a raw
    ``{(i,j):(lo,hi)}`` distance dict.
    """
    distances, angles = dict(distances or {}), dict(angles or {})
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


def _embed_isomer(iso, *, coordinate, contacts, distances, angles, n, seed, knowledge, keep_input=False):
    """Embed a metal `Isomer`, optionally binding a substrate; yield one `Ensemble` per binding candidate.

    Usually one, but several when ``coordinate=`` is a SMARTS matching several donor atoms (one candidate
    per donor, so the user can `select` the binding they want or keep them all). Each carries a ``.tag``;
    `keep_input` adds the Mol's input conformer to the ensemble (the retain-input-arrangement path).
    """
    from rxembed.constraints.builders import _window

    base = copy.deepcopy(iso.cons)
    if contacts is not None or distances or angles:  # substrate via NCI contacts
        d, a = _merge_contacts(distances, angles, contacts)
        sphere_d, sphere_a = set(base.distances), set(base.angles)  # the coordination sphere already in base —
        ckeys, akeys = set(), set()  # never release it, so provenance records
        for (i, j), val in d.items():  # ONLY genuinely new substrate contacts (a
            ri, rj = resolve_atom(iso.mol, i), resolve_atom(iso.mol, j)  # spec that lands ON a sphere hold is a
            key = (min(ri, rj), max(ri, rj))  # user override of that hold, not a
            add_distance(base.distances, ri, rj, *_window(val, 0.05))  # releasable NCI). mc(explore=) then frees
            if key not in sphere_d:  # only the substrate grip; the sphere and
                ckeys.add(key)  # any coordinate= dative bond stay held.
        for key, val in a.items():
            rk = tuple(resolve_atom(iso.mol, x) for x in key)
            base.angles[rk] = _window(val, 3.0)
            if rk not in sphere_a:
                akeys.add(rk)
        base.contacts = (frozenset(ckeys), frozenset(akeys))

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
        tag = {"geometry": iso.geometry, "label": iso.label, "ligands": _metal.arrangement(iso)}
        if atoms is not None:
            extra = _metal.coordinate(iso, atoms)
            for (i, j), (lo, hi) in extra.distances.items():
                add_distance(cons.distances, i, j, lo, hi)
            cons.angles.update(extra.angles)
            if len(choices) > 1:
                tag["coordinate"] = atoms[0]
        frozen = sorted(cons.frozen)  # capture the input TS core BEFORE the embed
        ref_core = (
            mol.GetConformer().GetPositions()[frozen]  # clears the conformer — restored exactly
            if frozen and mol.GetNumConformers()
            else None
        )  # onto each embedded pose below
        try:
            ids = _embed.embed(mol, cons, n or _embed.n_confs(mol, constrained=True), seed=seed, knowledge=knowledge)
        except RuntimeError as e:  # triangle smoothing -> infeasible bounds
            raise ValueError(
                f"could not embed {iso.geometry} {iso.label}"
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
            "embed[%s %s%s]: %d seeds%s",
            iso.geometry,
            iso.label,
            f" coord@{atoms}" if atoms else "",
            len(ids),
            " (incl. input geometry)" if input_conf else "",
        )
        ens = Ensemble(mol, ids, cons, _MetalCtx(mol, iso.metal, iso.real_z, iso.donors, iso.geometry, extra=iso.extra))
        ens.tag = tag
        yield ens


def _auto_contacts_embed(source, *, metal, freeze, distances, angles, planes, coordinate, charge, n, seed, knowledge):
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
        "freeze": freeze,
        "distances": distances,
        "angles": angles,
        "planes": planes,
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


def _resolve_template(template, charge):
    """Resolve a ``template=`` argument to a Mol carrying the TS geometry.

    Accepts an .xyz path, a Mol with a conformer, or an `Ensemble` (its first conformer is used).
    """
    if isinstance(template, Ensemble):
        m = Chem.Mol(template.mol)
        ids = list(template.ids) or [c.GetId() for c in m.GetConformers()]
        keep = ids[0]
        for c in [c.GetId() for c in m.GetConformers() if c.GetId() != keep]:
            m.RemoveConformer(c)
        return m
    if isinstance(template, Chem.Mol):
        if template.GetNumConformers() == 0:
            raise ValueError("template= Mol needs a conformer (a 3D geometry)")
        return template
    if isinstance(template, os.PathLike):
        template = os.fspath(template)
    if isinstance(template, str) and template.lower().endswith(".xyz"):
        return _xyz_to_mol(template, charge)
    raise ValueError("template= must be an .xyz path, an RDKit Mol with a conformer, or an Ensemble")


def _template_embed(source, *, template, match, anchor, charge, n, seed, knowledge):
    """Embed `source` with its reacting core pinned onto a `template` TS geometry (SMARTS-matched).

    The matched core is shape-biased in the embed, rigid-grafted onto the exact template coords, then held by a
    fixed-point relax while the rest conformer-searches — same machinery as a frozen TS core.
    """
    if not match:
        raise ValueError("template= needs match=<SMARTS> identifying the reacting core in the molecule")
    mol, has_geom = _normalize(source, charge)
    template_mol = _resolve_template(template, charge)
    cons, ref = from_template(mol, template_mol, match, anchor=anchor)  # match on the REAL molecule
    frag_of = {a: fi for fi, f in enumerate(Chem.GetMolFrags(mol)) for a in f}
    core_frags = {frag_of[a] for a in cons.frozen}  # fragments the matched core lives in
    if len(set(frag_of.values())) > len(core_frags):  # a STRAY fragment (e.g. a counter-ion) not in
        for (i, j), w in _encounter_bounds(mol).items():  # the core would float off — give it a sensible
            if frag_of[i] not in core_frags or frag_of[j] not in core_frags:  # vdW contact instead (the core
                add_distance(cons.distances, i, j, *w)  # pairs are already pinned by the template)
        logger.info(
            "template: %d stray fragment(s) placed at vdW contact to the core",
            len(set(frag_of.values())) - len(core_frags),
        )
    metal_ctx = None
    if _metal.metal_index(mol) is not None:  # a metal substrate: surrogate it for the FF (the
        donors = (
            {
                mi: [a.GetIdx() for a in mol.GetAtomWithIdx(mi).GetNeighbors()]  # template/graft pins the
                for mi in _metal.metal_indices(mol)
            }
            if has_geom
            else {}
        )  # matched core; the rest of
        mol, metals, _ = _metal.prepare_all(mol)  # the sphere is held by hold_shape if we have a
        metal_ctx = _MetalCtx(mol, metals[0][0], metals[0][1], extra=metals[1:])  # geometry to read it from)
        for mi, dons in donors.items():
            _metal.hold_shape(mol, [mi, *dons], cons)
    user_frozen = sorted(cons.frozen)
    ref_core = np.array([ref[i] for i in user_frozen])
    n = n or _embed.n_confs(mol, constrained=True)
    ids = list(_embed.embed(mol, cons, n, seed=seed, knowledge=knowledge))
    _graft_frozen(mol, ids, user_frozen, ref_core)  # restore the exact template core on each pose
    logger.info("template embed: %d seeds, %d-atom core pinned to the template", len(ids), len(user_frozen))
    return Ensemble(mol, ids, cons, metal_ctx)


def _embed_dispatch(
    source,
    *,
    metal=None,
    freeze=None,
    distances=None,
    angles=None,
    planes=None,
    contacts=None,
    coordinate=None,
    template=None,
    match=None,
    anchor=None,
    charge=0,
    n=None,
    seed=0xF00D,
    knowledge=True,
):
    """Dispatch the embed by input type / spec — see the public `embed` for documentation."""
    if isinstance(source, os.PathLike):
        source = os.fspath(source)  # accept pathlib.Path everywhere downstream
    if template is not None:
        clash = [
            k
            for k, v in {
                "metal": metal,
                "freeze": freeze,
                "contacts": contacts,
                "coordinate": coordinate,
                "distances": distances,
                "angles": angles,
                "planes": planes,
            }.items()
            if v
        ]
        if clash:  # don't silently drop them
            raise ValueError(
                f"template= pins the matched core and conf-searches the rest; it can't be "
                f"combined with {', '.join(clash)}="
            )
        return _template_embed(
            source, template=template, match=match, anchor=anchor, charge=charge, n=n, seed=seed, knowledge=knowledge
        )
    if match is not None or anchor is not None:
        raise ValueError("match=/anchor= only apply with template=<a TS geometry>")
    if coordinate is not None and metal is None and not isinstance(source, _metal.Isomer):
        raise ValueError(
            "coordinate= only applies to a metal (pass metal=<geometry> or a metal Isomer "
            f"from rx.metal(...)); got {type(source).__name__} — use contacts=/distances= otherwise"
        )
    if contacts == "auto":  # discover binding modes -> conf-search each
        return _auto_contacts_embed(
            source,
            metal=metal,
            freeze=freeze,
            distances=distances,
            angles=angles,
            planes=planes,
            coordinate=coordinate,
            charge=charge,
            n=n,
            seed=seed,
            knowledge=knowledge,
        )
    _iso_kw = {
        "coordinate": coordinate,
        "contacts": contacts,
        "distances": distances,
        "angles": angles,
        "n": n,
        "seed": seed,
        "knowledge": knowledge,
    }
    if metal is not None:  # enumerate isomers -> a selectable set
        if isinstance(source, _metal.Isomer):
            raise ValueError("pass either a metal Isomer source OR metal=<geometry>, not both")
        if isinstance(source, str) and source.lower().endswith(".xyz"):
            source = _normalize(source, charge)[0]  # xyz -> perceived Mol (enumerate wants a Mol/SMILES)
        out = EnsembleSet(e for iso in _metal.enumerate_isomers(source, metal) for e in _embed_isomer(iso, **_iso_kw))
        logger.info(
            "metal: ENUMERATING isomers of %s -> %d distinct candidate(s) (each a separate "
            "ensemble; .summary() / .select() one to conf-search)",
            metal,
            len(out),
        )
        return out
    if isinstance(source, _metal.Isomer):  # a single chosen isomer
        results = list(_embed_isomer(source, **_iso_kw))
        return results[0] if len(results) == 1 else EnsembleSet(results)

    mol, has_geom = _normalize(source, charge)
    if (
        metal is None
        and has_geom
        and _metal.metal_index(mol) is not None
        and not (freeze or distances or angles or contacts or coordinate)
    ):  # retain the input arrangement
        iso = _metal.from_geometry(mol)
        logger.info(
            "metal: source carries a geometry and no metal= -> RETAINING the input ligand "
            "arrangement (%s, %s); pass metal=<geometry> to enumerate isomers instead",
            iso.geometry,
            iso.label,
        )
        return next(
            _embed_isomer(
                iso,
                coordinate=None,
                contacts=None,
                distances=None,
                angles=None,
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
    if _metal.metal_index(mol) is not None and (freeze or distances or angles or contacts):
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
        if freeze:  # hold each metal + its donors so the bond-
            metal_core = {mi for mi, _ in metals} | set(core_donors)  # stripped coordination cores can't drift
        logger.debug("metal complex: %d metal(s) swapped to carbon surrogate for the FF", len(metals))

    distances, angles = _merge_contacts(distances, angles, contacts)
    cons = from_spec(mol, freeze=freeze, distances=distances, angles=angles, planes=planes, has_geometry=has_geom)
    cons.contacts = (
        frozenset(
            k
            for k in cons.distances  # SEEDED contacts = the non-frozen-pair
            if not (k[0] in cons.frozen and k[1] in cons.frozen)
        ),  # distances (a frozen-core SHAPE
        frozenset(cons.angles),
    )  # is a frozen-frozen pair) + all angles;
    #   the metal-sphere / M-H / encounter holds added below are structural and stay out of `contacts` (captured
    #   here, before them), so mc(explore=) releases only the real NCI/user contacts, never the structure.
    user_frozen = sorted(cons.frozen)  # the user's reacting TS core — grafted EXACTLY
    cons.frozen |= metal_core  # pin the metal coordination cores (freeze path);
    for mi, dons in metals_donors.items():  # a spectator metal is held intact-but-achiral by
        _metal.hold_shape(mol, [mi, *dons], cons)  # hold_shape (NOT grafted — its handedness must
        #                                          stay free for the stereo filter), then pinned at the embed
    for mi, rz, h in hydrides:  # covalent M-H window (no input geometry to read)
        d = _PT.GetRcovalent(rz) + _PT.GetRcovalent(1)
        add_distance(cons.distances, mi, h, d - 0.1, d + 0.15)
    if not cons.is_constrained and len(Chem.GetMolFrags(mol)) > 1:
        cons.distances.update(_encounter_bounds(mol))
        logger.debug("multi-fragment input: added vdW-aware encounter bounds")

    n = n or _embed.n_confs(mol, constrained=cons.is_constrained)
    ref_core = mol.GetConformer().GetPositions()[user_frozen] if user_frozen and has_geom else None
    ids = _embed.embed(mol, cons, n, seed=seed, knowledge=knowledge)
    if ref_core is not None:
        _graft_frozen(mol, ids, user_frozen, ref_core)  # restore the exact frozen TS core (partial bonds)
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
