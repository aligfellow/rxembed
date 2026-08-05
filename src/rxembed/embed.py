"""The core front door: ``embed(spec, fix=, constrain=) -> Conformers``.

`spec` is an RDKit `Mol` or a metal `Isomer`; parsing and NCI discovery are the caller's job. The verbs are
`fix` (rigid) and `constrain` (soft, releasable), both defined in the `constraints` module docstring; a
coordination `Isomer` contributes its polyhedron, onto which the substrate spec is folded. The result is a
`Conformers`: one Mol, its conformer ids, and the `Constraints` that shaped them, with `minimize()` relaxing
them under exactly those constraints.

The physics is `bounds.embed` (the edited ETKDG bounds matrix) + `relax.restrained_uff`, unchanged; this
module is the seam that stacks them: encounter bounds -> donor-chirality hold -> embed -> Kabsch graft, then
`minimize`'s stiffness ladder and the metal-centre handedness gate that closes it (`_hold_metal_hand`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from . import metal_core as _metal
from . import metal_polyhedron as _poly
from .bounds import DEFAULT_SEED, n_confs, probe_conformer
from .bounds import embed as _dg_embed
from .constraints import Constraints, compose, resolve_atom, resolve_core, template_to_fix
from .metal_isomers import Isomer, from_geometry
from .relax import MAX_ITERS, bonding_ok, restrained_uff

logger = logging.getLogger("rxembed")  # the package logger itself, as the front door has always used:
#   a pinned name (`set_verbose` configures it), independent of where this module sits.

_EPS = 1e-6  # near-zero norm floor for the graft axis
_BOND_ATOMS = 2  # a two-atom frozen core is a bond: fix its length, not an orientation
_MIN_FRAGS = 2  # below this there is no inter-fragment separation to enforce
# Restraint stiffness is dimensionless here and every constant it multiplies lives in `mechanisms`, on the one
# scale stated there. 1.0 is shipped. Half-order steps (not 10x jumps) so escalation lands on the MINIMAL
# stiffness that holds the sphere.
BASE_STIFFNESS = 1.0
FC_ESCALATION = (1.0, 3.0, 10.0, 30.0, 100.0)
# a ~1.5x-stretched bond in a coplanar coordination is a surrogate/tight-bite artifact xtb recovers, so the
# metal path keeps it (the coplanarity gate is the real plane check). Non-metal stays 1.3.
METAL_BOND_TOL = 1.5
BOND_TOL = 1.3
# A fresh seed re-rolls the metal hand, so each rescue round is a coin flip per conformer: bound the rounds
# (the pipeline's `_reembed_until_clean` bound) and over-embed a couple per round to cover that flip.
_MAX_HAND_ROUNDS = 5
_HAND_BUFFER = 2


def encounter_bounds(mol, slack=1.5, seed=DEFAULT_SEED):
    """vdW-aware inter-fragment bounds so a multi-fragment molecule does not embed on top of itself.

    For every pair of fragments, bound their closest heavy-atom pair to roughly van-der-Waals contact
    (sum of vdW radii .. + slack), giving the embedder a separated but touching encounter geometry. The
    probe conformer that decides which pair is closest comes from `bounds.probe_conformer`; see there
    for why its seed is fixed.
    """
    tmp = probe_conformer(mol, seed)
    if tmp is None:
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


def float_encounter_bounds(mol, cons):
    """Bound fragments no constraint pins to vdW contact, so a free or stray fragment can't drift off.

    One rule for two cases: a fully unconstrained multi-fragment molecule (every fragment floats -> all pairs
    bounded) and a fixed/templated core that leaves a spectator fragment (a counter-ion) untethered. A pair
    already linked by a fix / constrain / contact is left to that constraint.
    """
    frags = Chem.GetMolFrags(mol)
    if len(frags) < _MIN_FRAGS:
        return {}
    frag_of = {a: fi for fi, f in enumerate(frags) for a in f}
    touched = {frag_of[a] for a in cons.constrained_atoms()}
    if len(touched) >= len(frags):  # every fragment already pinned/linked
        return {}
    return {
        k: v for k, v in encounter_bounds(mol).items() if frag_of[k[0]] not in touched or frag_of[k[1]] not in touched
    }


def graft_frozen(mol, conf_ids, frozen, ref):
    """Restore the `frozen` atoms to their exact input geometry `ref`, Kabsch-fitted onto the embedded pose.

    Distance geometry only approximates a rigid core, at ~0.2-0.4 Å internal RMSD: fine for a normal
    molecule, wrong for a TS whose partial bonds must be preserved. So the core is restored exactly, then
    oriented to best match the embedded periphery, and held there by `AddFixedPoint`.
    """
    frozen = list(frozen)
    core = np.asarray(ref, float)  # input core, input frame
    if len(frozen) <= 1:  # a point has no shape to restore
        return
    if len(frozen) == _BOND_ATOMS:  # a bond: keep one atom, slide the other to the template distance along the
        dt = float(np.linalg.norm(core[0] - core[1]))  # embedded axis: no orientation to restore, just length
        for c in conf_ids:
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


def _frozen_core_ref(mol, frozen_atoms, graft_ref):
    """Return ``(frozen, ref_core)``: atoms to Kabsch-graft and their exact target coords, ``None`` if none."""
    frozen = sorted(frozen_atoms)  # capture the input core BEFORE the embed
    if frozen and mol.GetNumConformers():  # graft each frozen atom to its explicit-fix coord, else its own coord
        own = mol.GetConformer().GetPositions()
        return frozen, np.array([graft_ref.get(i, own[i]) for i in frozen])
    if graft_ref:  # SMILES metal + explicit-coords fix: no conformer, graft only the named atoms
        frozen = sorted(graft_ref)
        return frozen, np.array([graft_ref[i] for i in frozen])
    return frozen, None


def _determined_by_graft(base, graft_ref):
    """Return the grafted coordination-sphere atoms when the graft pins two or more of them (else ``[]``).

    Two sphere atoms grafted from one reference fix their mutual placement, which for sphere atoms is the
    arrangement, so the enumerated isomer would come back wearing the reference's geometry under its own
    label. The sphere is read off the stated distances, not `base.angles`, which at CN<=6 tabulates only a
    minimal vertex-pair subset and would miss exactly the pairs left free.
    """
    if not graft_ref:
        return []
    sphere = {a for k in base.distances for a in k}
    pinned = sorted(sphere & set(graft_ref))
    return pinned if len(pinned) > 1 else []


def fold_substrate(base, sub, graft_ref):
    """Fold a substrate's resolved constraints `sub` onto a coordination `base`.

    Provenance is subtractive, not the union `compose` would give: a user grip landing on a sphere hold stays
    STRUCTURAL (never releasable), or `mc(explore=)` could dissociate the coordination sphere. `compose` itself
    is field-driven so no spec field is silently dropped (a hand-listed merge lost a `constrain=` pi-stack in
    `sub.planes`); last-wins, so a spec landing on a sphere hold overrides it rather than being demoted to soft.

    `graft_ref` is the substrate's coordinate graft (`resolve_core`'s ``ref``): a graft that pins two or more
    coordination-sphere atoms is REFUSED, because the post-embed Kabsch restore would silently hand back the
    reference's arrangement wearing the enumerated isomer's label.
    """
    pinned = _determined_by_graft(base, graft_ref)
    if pinned:
        raise ValueError(
            f"the coordinate graft (fix={{i: (x,y,z)}} / template=) pins coordination-sphere atoms {pinned}: "
            f"the graft restores them to their exact reference coordinates after the embed, so it, not the "
            f"coordination geometry, would decide how they sit, and you would get the reference's arrangement "
            f"under this isomer's label. Graft at most one sphere atom (a core over ligand backbone atoms "
            f"composes fine), or embed the reference geometry itself as the source."
        )
    sphere_d, sphere_a = set(base.distances), set(base.angles)
    soft_d, soft_a = sub.contacts
    return compose(base, sub).copy(
        contacts=(
            frozenset(k for k in soft_d if k not in sphere_d),
            frozenset(k for k in soft_a if k not in sphere_a),
        )
    )


def seed_conformers(mol, cons, iso, n, *, seed=DEFAULT_SEED, knowledge=True, prune_rms=0.1, threads=0, graft_ref=None):
    """Embed `n` conformers under `cons` and graft the frozen core; return ``(mol, ids)``.

    The shared seam: free fragments tethered, a labile metal donor's hand capped for the DG, the bounds-matrix
    embed itself, then the exact core grafted back. `mol` is returned because the chirality cap rebuilds it.

    Everything in ``cons.frozen`` is grafted: ``frozen`` means "graft and hold", one set, for every caller.
    """
    if int(seed) < 0:  # every embed route validates here, not just the front door: RDKit's -1 draws from the
        raise ValueError(  # global RNG, so the result silently depends on prior consumption (measured: two
            f"seed={seed}: a negative seed draws from RDKit's global RNG and is not reproducible"
        )  # identical rx.embed('CCO', seed=-1) calls gave different coordinates)
    if n is not None and int(n) <= 0:  # `n or n_confs(...)` below would silently treat 0 as "use the default"
        raise ValueError(f"n={n}: give a positive conformer count, or None for the flexibility-scaled default")
    cons.distances.update(float_encounter_bounds(mol, cons))  # keep free/stray fragments from drifting off
    frozen, ref_core = _frozen_core_ref(mol, cons.frozen, graft_ref or {})
    held = []
    if iso is not None:  # hold a carbanion/amine donor's hand: a degree-3 centre with no M-C bond would
        mol, held = _metal._hold_donor_chirality(mol, iso.metal, iso.donors, cons)  # else invert freely
    ids = list(
        _dg_embed(
            mol,
            cons,
            n or n_confs(mol, constrained=cons.is_constrained),
            seed=seed,
            prune_rms=prune_rms,
            knowledge=knowledge,
            threads=threads,
        )
    )
    if iso is not None:
        mol = _metal._release_donor_chirality(mol, held, cons)  # drop the dummy D's + cons keys, restore charges
    if ref_core is not None:
        graft_frozen(mol, ids, frozen, ref_core)  # restore the exact frozen core
    return mol, ids


def _mirror_is_free(mol):
    """Return True when reflecting a conformer of `mol` would invert nothing but its metal centre.

    A reflection inverts every atomic and axial (allene, atropisomer) stereocentre in one go, so it is a free
    fix for the metal only where there are none to invert. The exception is what makes it worth asking: E/Z is
    reflection-INVARIANT, since mirroring a cis double bond leaves it cis, so a stated alkene geometry costs
    nothing here -- which is most of a coordination corpus.

    Chiral tags are read as well as perceived stereo, because the surrogate strips a donor's M-L bond and with
    it RDKit's view of a carbanion/amine stereocentre: only the tag `_hold_donor_chirality` enforces survives.
    """
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        return False
    return not any(e.type != Chem.StereoType.Bond_Double for e in Chem.FindPotentialStereo(mol))


def _reflect(mol, cid):
    """Negate x on conformer `cid` in place: the exact enantiomer, same graph, same atom order, same energy.

    Every term that shaped this geometry is mirror-symmetric -- UFF, and constraint rows that are distances,
    angles and a coplanarity window symmetric about 180 -- so the reflected conformer is not an approximation
    of a relaxed structure, it IS the converged relax of the reflected seed. Measured on a relaxed
    cis-[Co(en)2Cl2] batch: max |dE| under the reflection is 0.0 kcal/mol, not merely small.
    """
    conf = mol.GetConformer(int(cid))
    pos = conf.GetPositions()
    pos[:, 0] *= -1.0
    for a, xyz in enumerate(pos):
        conf.SetAtomPosition(a, xyz.tolist())


def _realised_hand(mol, iso, cid):
    """Return the metal-centre hand conformer `cid` of `mol` realises, ``''`` when it cannot be decided.

    `mol` is the connected, real-element graph (`Conformers.mol`): `from_geometry` re-derives the seating from
    the conformer, so its `chirality` is what the geometry HAS, against `Isomer.chirality`, which is what the
    caller ASKED for. The reading only means that when it names the same centre on the same polyhedron: a
    distorted sphere gets classified as some other shape and carries a tag over different vertices, and a
    spectator metal would be read in place of this one. Both come back '' rather than a confident wrong answer.
    """
    got = from_geometry(Chem.Mol(mol, False, int(cid)))  # a single-conformer copy: `from_geometry` takes a Mol
    return got.chirality if got.metal == iso.metal and got.geometry == iso.geometry else ""


@dataclass
class Conformers:
    """An embed result: one Mol carrying many conformers, plus the `Constraints` that shaped them.

    `ids` are the conformer ids in play and `.mol` is a real RDKit Mol, so drop to raw RDKit whenever you like.
    For a metal `Isomer` the working molecule keeps the bond-less surrogate every DG/FF stage needs, and `.mol`
    finalizes the user-facing graph on access (real element + oxidation state + M-donor DATIVE bonds) on a copy.

    `minimize` relaxes in place and chains; indexing (`confs[0]`, `confs[:3]`) returns a new `Conformers`
    sharing the same Mol.
    """

    _mol: Chem.Mol
    ids: list
    cons: Constraints = field(default_factory=Constraints)
    iso: Isomer | None = None
    energies: dict = field(default_factory=dict)  # {conf id: energy}, restrained-UFF here, filled by
    # `minimize`; a subclass that scores with a real calculator reuses this dict and tracks the kind itself
    unrelaxed: list = field(default_factory=list, kw_only=True)
    # ids handed back at their EMBED SEED: `_rescue_torn` found no
    # stiffness that relaxed them without tearing, so these coordinates never completed a constrained relax. They
    # are still returned (a seed beats a torn geometry) and still scored, so this list is the only thing that says
    # which -- their energy gives them away by 4-5 orders, but only a caller that reads energies sees it, and an
    # RMSD-ranking caller does not. Measured on benchmark/corpus: 17 of 269 conformers over 8 of 45 structures.
    seed: int | None = field(default=None, kw_only=True)
    # the ETKDG seed these came from, or None when they were not embedded here
    # (`minimize(spec)` relaxes what it is given). `_hold_metal_hand` re-seeds off it, so its absence is also
    # what keeps the search-free verb from searching.
    wrong_hand: list = field(default_factory=list, kw_only=True)
    # ids whose metal centre came back the MIRROR of the isomer
    # the caller named, and that neither the reflection nor a fresh seed could fix. Same contract as
    # `unrelaxed`: they are still returned, and this list is the only thing that says the arrangement you
    # selected is not the one this conformer realises.

    @property
    def mol(self):
        """The molecule, with a metal restored to its real element, charge and M-donor bonds.

        Always a fresh copy here, so an edit never reaches the working molecule the next `minimize` uses;
        bind it once (``m = confs.mol``) to edit. The guarantee is this class's own: a subclass that finalizes
        lazily may narrow it to a live handle.
        """
        mol = Chem.Mol(self._mol)  # never mutate the working surrogate
        if self.iso is None:
            return mol
        self.iso.restore(mol)  # real element AND oxidation state, on our copy
        return _metal.connect_metal(mol, self.iso.donor_bonds) if self.iso.donor_bonds else mol

    @property
    def _bond_tol(self):
        """The break threshold the accept gate uses; looser for a metal (see `METAL_BOND_TOL`)."""
        return METAL_BOND_TOL if self.iso is not None else BOND_TOL

    def _intact(self, cid):
        """Return True if conformer `cid` still has every bond the graph says it has."""
        return bonding_ok(
            self._mol, cid, bond_tol=self._bond_tol, exclude=self.cons.frozen, constrained=self.cons.distances
        )

    def _coordination_ok(self, cid, iso=None):
        """Return False if a declared planar polyhedron came out puckered: an infeasible arrangement.

        A planar record is coplanar by definition, so the escalating relax can only force an impossible
        arrangement (a chelate at trans vertices) through as an out-of-plane pucker, which `bonding_ok` does
        not catch. No-op for a non-planar record or a metal with no declared polyhedron.

        `iso` overrides `self.iso` for a caller that has already consumed it.
        """
        iso = self.iso if iso is None else iso
        if iso is None or not _poly.is_planar(iso.geometry) or not iso.donors:
            return True
        pos = self._mol.GetConformer(cid).GetPositions()
        # a haptic face is one vertex
        return _metal.coplanar(pos, iso.metal, iso.donors, haptic=self.cons.haptic)

    def _relax_constrained(self, stiffness, max_iters=MAX_ITERS, conf_ids=None):
        """Restrained-UFF relax; if the soft relax tears *every* bond, stiffen the walls and retry.

        The surrogate's carbon vdW crowds donors out to ~2.1 Å, so a base-stiffness constraint lets an unusual
        ligand tear; escalating (3x → 100x) lets the coordination windows dominate so the bonds survive.
        Escalation fires only when the softer relax leaves zero intact conformers, so
        a genuinely infeasible arrangement (an en forced *trans*) still tears at every stiffness and is dropped
        with no phantom resurrected. Returns the accepted relax's energies, or ``None`` if UFF can't type it.

        `conf_ids` restricts the relax to a subset (default: every conformer on the Mol, including any the
        caller has already set aside: the pipeline keeps its discarded conformers there and reads the
        energies back positionally).
        """
        # Hold a labile (carbanion/amine) donor's hand through the relax: the surrogate's bare degree-3 centre
        # inverts under UFF. Cap it with a dummy D, release when done.
        iso = self.iso
        held = _metal._hold_donor_chirality(self._mol, iso.metal, iso.donors, self.cons) if iso else (self._mol, [])
        self._mol, hold = held
        e, fc = None, stiffness
        try:
            all_ids = [c.GetId() for c in self._mol.GetConformers()]
            relaxing = all_ids if conf_ids is None else [int(c) for c in conf_ids]
            embed_pos = {
                c: [list(self._mol.GetConformer(c).GetAtomPosition(a)) for a in range(self._mol.GetNumAtoms())]
                for c in relaxing
            }
            for step, mult in enumerate(FC_ESCALATION):  # half-order steps: find the MINIMAL sufficient stiffness
                if step:  # restore the embed geometry before a stiffer retry (a big jump over-stiffens it)
                    for cid, pos in embed_pos.items():
                        conf = self._mol.GetConformer(cid)
                        for a, xyz in enumerate(pos):
                            conf.SetAtomPosition(a, xyz)
                fc = stiffness * mult
                try:
                    e = restrained_uff(self._mol, self.cons, stiffness=fc, max_iters=max_iters, conf_ids=conf_ids)
                except RuntimeError as err:  # UFF can't build a force field for this graph, so keep the embed
                    logger.warning(
                        "minimize: UFF could not relax this system (%s); keeping the embedded geometry",
                        err,
                    )
                    return None
                # accept a stiffness only if a conformer is both bonded and, for a planar polyhedron, coplanar,
                # so escalation never forces a phantom through as an out-of-plane pucker.
                kept = [i for i in self.ids if self._intact(i) and self._coordination_ok(i)]
                if kept or step == len(FC_ESCALATION) - 1:  # some survived (accept), or out of steps (caller drops)
                    if step and kept:
                        logger.info("minimize: relax tore the sphere; escalated restraint stiffness to %gx", fc)
                    elif step:  # exhausted: what comes back is the %gx relax, not the stiffness that was asked
                        logger.warning(  # for, and nothing else says so. Fail loud.
                            "minimize: no stiffness up to %gx satisfied the accept gate; returning the ungated relax",
                            fc,
                        )
                    break
        finally:  # release the hold on every exit path, including the early UFF-failure return; the dummy D
            if hold:  # is scaffolding for this relax alone
                self._mol = _metal._release_donor_chirality(self._mol, hold, self.cons)
        return e

    def _rescue_torn(self, seed_pos, stiffness):
        """Re-relax each conformer the relax tore at its own minimal sufficient stiffness; else keep its seed.

        `_relax_constrained` escalates GLOBALLY and stops at the first rung leaving any conformer intact, so a
        seed needing more stiffness than its siblings stays torn. Each is retried from its seed up the same
        ladder, and one that survives no rung keeps its seed coordinates, so the output is never worse than
        was embedded, and never a torn geometry wearing a plausible energy. Returns the number tried.

        NB the rescue runs outside `_hold_donor_chirality`, so a labile donor could invert here; the
        pipeline's `minimize` culls that (`_donor_hand`), and a caller on the raw core gets what the
        constraints state.
        """

        def place(cid, pos):
            conf = self._mol.GetConformer(cid)
            for a, xyz in enumerate(pos):
                conf.SetAtomPosition(a, xyz.tolist())

        torn = [c for c in self.ids if not self._intact(c)]
        rescued = 0
        for cid in torn:
            for mult in FC_ESCALATION[1:]:  # rung 0 is the pass that already tore it
                place(cid, seed_pos[cid])
                try:
                    restrained_uff(self._mol, self.cons, stiffness=stiffness * mult, conf_ids=[int(cid)])
                except RuntimeError:  # UFF cannot build for this graph, so the seed is the best available
                    break
                if self._intact(cid):
                    rescued += 1
                    break
            else:
                place(cid, seed_pos[cid])
                self.unrelaxed.append(cid)  # never relaxed: say so, or nothing downstream can tell
                continue
            if not self._intact(cid):
                place(cid, seed_pos[cid])
                self.unrelaxed.append(cid)
        if torn:
            logger.warning(  # a geometry that never completed a relax is not what `minimize` promises: say so
                "relax: tore %d of %d conformer(s); %d rescued by escalating their stiffness, "
                "%d kept their unrelaxed geometry",
                len(torn),
                len(self.ids),
                rescued,
                len(torn) - rescued,
            )
        return len(torn)

    def _rescore_restrained(self, stiffness):
        """Record one comparable restrained-UFF single point for every tracked conformer."""
        try:
            e = restrained_uff(self._mol, self.cons, stiffness=stiffness, max_iters=0, conf_ids=self.ids)
        except RuntimeError as err:
            logger.warning("minimize: UFF could not score the relaxed conformers (%s)", err)
            self.energies = {}
            return
        self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=True)}

    def _metal_hands(self):
        """Return ``{conformer id: realised metal-centre hand}`` over the tracked conformers.

        The perception logger is held at WARNING for the read: `from_geometry` re-classifies the polyhedron
        once per conformer and announces each, which is `n` near-identical lines describing a diagnostic
        rather than the caller's embed. Anything actionable still comes through.
        """
        mol = self.mol  # bind once: the finalize copies the whole molecule
        perception = logging.getLogger("rxembed.metal")
        was = perception.level
        perception.setLevel(max(was, logging.WARNING))
        try:
            return {c: _realised_hand(mol, self.iso, c) for c in self.ids}
        finally:
            perception.setLevel(was)

    def _reseed_hand(self, wrong, stiffness, max_iters):
        """Swap each id in `wrong` for a freshly embedded conformer of the right hand; return the ids left over.

        The remedy when the mirror is not free. Nothing biases the hand, so a fresh seed simply re-rolls it and
        a round is a coin flip per conformer: hence the bounded rounds and the over-embedded batch. The swap is
        positional (same graph, same atom order), so conformer ids and any slice a caller holds survive it, and
        the replacement brings its own energy rather than inheriting the geometry's it displaced.

        Each round works on a COPY of the constraints, because both `seed_conformers` and the batch relax edit
        the set they are handed (encounter bounds, and `_shift_phantoms` moving the reserved haptic-centroid
        block up past the labile-donor caps). `self.cons` is the record of what shaped the conformers this
        result already holds, and a retry must not rewrite it.
        """
        seed, iso = self.seed, self.iso
        if seed is None or iso is None:  # `_hold_metal_hand` gates both; a direct caller gets a no-op, not a
            return list(wrong)  # crash, and "fixed none of them" is the honest answer without a seed to re-roll
        left = list(wrong)
        for attempt in range(1, _MAX_HAND_ROUNDS + 1):
            if not left:
                break
            cons = self.cons.copy()
            mol, ids = seed_conformers(Chem.Mol(self._mol), cons, iso, len(left) + _HAND_BUFFER, seed=seed + attempt)
            if not ids:
                continue
            batch = Conformers(mol, ids, cons, iso).minimize(stiffness, max_iters, _retry=False)
            hands = batch._metal_hands()  # `_retry=False` above: this batch is read here, never re-seeded again
            spare = [c for c in batch.ids if hands[c] == iso.chirality and c not in batch.unrelaxed]
            for cid, src in zip(list(left), spare, strict=False):
                pos = batch._mol.GetConformer(int(src)).GetPositions()
                conf = self._mol.GetConformer(int(cid))
                for a, xyz in enumerate(pos):
                    conf.SetAtomPosition(a, xyz.tolist())
                self.energies.pop(cid, None)
                if src in batch.energies:
                    self.energies[cid] = batch.energies[src]
                if cid in self.unrelaxed:  # the geometry that flag described is gone: the replacement relaxed
                    self.unrelaxed.remove(cid)
                left.remove(cid)
        return left

    def _hold_metal_hand(self, stiffness, max_iters):
        """Make every conformer realise the hand `iso.chirality` names, or record which one does not.

        Nothing else in this engine can choose a metal-centre hand: the coordination constraints are
        distances, angles, pulls, floors and a coplanarity window symmetric about 180, every one of them
        mirror-invariant, and the surrogate metal is bond-less so ETKDG has no stereocentre to enforce. The
        hand is therefore whatever the seed fell on, and a caller who named an isomer got its mirror about
        half the time (measured on benchmark/corpus: 41.6% of conformers came back the input's enantiomer).
        Naming an isomer has to mean something, so this is where it is made to.

        Read after the relax, not at the seed. A raw DG seed's sphere is not yet the polyhedron that was
        stated -- 8 of 8 cis-[Co(en)2Cl2] seeds classify as trigonal_prismatic -- so its hand is not readable,
        and the reading that can be forced there disagrees with the relaxed one on 7 of 8. Post-relax also
        makes the reflection exact rather than approximate (see `_reflect`).

        A caller who named no hand is untouched: `iso.chirality` is '' for an achiral or undecidable centre,
        and an organic embed has no `iso` at all, so neither reaches the per-conformer read.
        """
        iso = self.iso
        if iso is None or not iso.chirality or not self.ids:
            return
        wrong = [c for c, hand in self._metal_hands().items() if hand and hand != iso.chirality]
        if not wrong:
            return
        reflected, reseeded = 0, 0
        # The mirror is free only where the metal centre is the one thing it inverts: no other stereocentre
        # (`_mirror_is_free`), no grafted core (its contract is the caller's EXACT geometry, and a chiral
        # core's mirror is a different core -- re-seeding re-grafts it instead), and no haptic face, which can
        # be planar-chiral in a way this tier cannot perceive (`pipeline.select_stereo` owns that).
        if _mirror_is_free(self._mol) and not self.cons.frozen and not iso.haptic:
            for cid in wrong:
                _reflect(self._mol, cid)
            reflected, wrong = len(wrong), []
        elif self.seed is not None:
            reseeded = len(wrong)
            wrong = self._reseed_hand(wrong, stiffness, max_iters)
            reseeded -= len(wrong)
        self.wrong_hand = wrong
        logger.info(
            "minimize: %d conformer(s) came back mirrored at the metal; %d reflected, %d re-seeded",
            reflected + reseeded + len(wrong),
            reflected,
            reseeded,
        )
        if wrong:  # the caller asked for one hand and is getting the other: only this list says so
            logger.warning(
                "minimize: %d of %d conformer(s) are not the %s centre that was asked for (see .wrong_hand)",
                len(wrong),
                len(self.ids),
                iso.chirality,
            )

    def minimize(self, stiffness=BASE_STIFFNESS, max_iters=MAX_ITERS, _retry=True):
        """Relax every conformer under the carried constraints (restrained UFF); in place, chainable.

        `stiffness` scales every flat-bottomed WALL (1.0 is shipped); raise it (10) to pull a tight `fix`
        harder. It does not touch the biases, which carry their own constants: see `mechanisms`, where the
        tiers and the physical scale they are stated on both live. Records the FF energies in `.energies`;
        they are a surrogate, not comparable across species.

        A constrained relax runs the stiffness ladder, not one pass: the restrained UFF can satisfy a
        window by pulling a bond apart, and nothing in an energy says so. So the force constant escalates
        while nothing is intact, each still-torn conformer is retried at its own stiffness, and one that
        survives no rung falls back to its embedded seed. Nothing is ever dropped: this verb hands back the
        `n` conformers it was given, and *deciding* is the caller's (measured on the metal corpus: without the
        ladder 7 of 7 BEGLUU conformers came back torn, and the first of them scored 1.161 Å against 0.518 Å
        with it).

        That promise is THIS verb's, not the class's. `pipeline.Ensemble` REPLACES it rather than extending it
        (there is no `super()` call in that module) and does drop, on five gates, because an unconverged seed
        reaching `prune` / `score` / `best` carries a plausible energy with nothing to say it is broken. A
        conformer handed back here at its seed is named in `.unrelaxed` and warned about, so a caller that
        wants the raw geometry -- to hand to a real optimiser, say -- can have it and know what it is.

        The relaxed sphere is also the first place the metal-centre hand can be read, so `_hold_metal_hand`
        runs here; `_retry=False` is how that gate relaxes its own replacement batch without recursing.
        """
        if not self.ids:
            return self
        if not self.cons.is_constrained:  # no window to tear against, so no ladder to climb
            e = restrained_uff(self._mol, self.cons, stiffness=stiffness, max_iters=max_iters, conf_ids=self.ids)
            self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=False)}
            return self
        self.unrelaxed = []  # a re-minimize re-decides it; a stale list would outlive the geometry it described
        self.wrong_hand = []
        seed_pos = {c: self._mol.GetConformer(c).GetPositions() for c in self.ids}
        e = self._relax_constrained(stiffness, max_iters, conf_ids=self.ids)
        self._rescue_torn(seed_pos, stiffness)
        if e is not None:
            self.energies = {i: float(v) for i, v in zip(self.ids, e, strict=False)}
        if _retry:
            self._hold_metal_hand(stiffness, max_iters)
        if e is not None or self.energies:
            # Rescue and hand re-seeding may relax different conformers at different stiffnesses. Rank them
            # only after one single-point pass on the caller's stated objective.
            self._rescore_restrained(stiffness)
        return self

    def measure(self, atoms):
        """Mean and range of a geometric measurement over every tracked conformer.

        The companion to a numbers-`fix`: that is a tight-window UFF pull rather than a snap, so a stated
        distance or angle is read back, not assumed. `atoms`, by index or SMARTS, gives a distance (2 atoms),
        an angle in degrees (3), or a dihedral in degrees (4). Returns ``{'mean', 'min', 'max', 'n'}``::

            confs = embed(mol, fix={(0, 4): 3.0})
            confs.measure((0, 4))  # {'mean': 3.02, 'min': 3.00, 'max': 3.04, 'n': 8}
        """
        if not self.ids:
            raise ValueError(
                "measure() on an empty result: embed/minimize may have produced no conformers (check the log)"
            )
        idx = [resolve_atom(self._mol, a) for a in atoms]
        fns = {2: rdMolTransforms.GetBondLength, 3: rdMolTransforms.GetAngleDeg, 4: rdMolTransforms.GetDihedralDeg}
        if len(idx) not in fns:
            raise ValueError("measure() takes 2 (distance), 3 (angle) or 4 (dihedral) atoms")
        f = fns[len(idx)]
        v = [f(self._mol.GetConformer(c), *idx) for c in self.ids]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}

    def xyz(self, conf_id=None):
        """Return conformer `conf_id` as an xyz block, or every tracked conformer as one multi-frame block.

        `conf_id` is an RDKit conformer id (what `.ids` holds), not a position, and must be one this result
        still tracks; otherwise a slice would emit a conformer it no longer owns.
        """
        if conf_id is not None and int(conf_id) not in [int(c) for c in self.ids]:
            raise ValueError(f"conformer id {conf_id} is not one of this result's ids {list(self.ids)}")
        mol = self.mol
        return "".join(Chem.MolToXYZBlock(mol, confId=int(c)) for c in (self.ids if conf_id is None else [conf_id]))

    def dump(self, path):
        """Write the tracked conformers to `path` as a multi-frame .xyz; return the path."""
        if not self.ids:  # a 0-byte file that reads as a successful write is the worst possible outcome
            raise ValueError("nothing to dump: this result has no conformers (the embed produced none)")
        with open(path, "w") as f:
            f.write(self.xyz())
        return path

    def __getitem__(self, key):
        """Pick conformer(s) by position as a new `Conformers`: ``confs[0]``, ``confs[:3]``."""
        sel = self.ids[key]
        sel = sel if isinstance(sel, list) else [sel]
        return Conformers(
            self._mol,
            sel,
            self.cons,
            self.iso,
            {i: self.energies[i] for i in sel if i in self.energies},
            unrelaxed=[i for i in self.unrelaxed if i in sel],  # a slice must not silently lose these flags
            seed=self.seed,
            wrong_hand=[i for i in self.wrong_hand if i in sel],
        )

    def __len__(self):
        """Return the number of conformers in play."""
        return len(self.ids)

    def __repr__(self):
        """Summarise the result: conformer count and the isomer identity when there is one."""
        who = f", {self.iso.summary()}" if self.iso is not None else ""
        return f"<Conformers: {len(self.ids)} conformer{'s' if len(self.ids) != 1 else ''}{who}>"


def _check_bare_mol(mol):
    """Refuse a plain `Mol` this engine would embed WRONG: implicit Hs, or an un-surrogated metal centre.

    Both produce a plausible-looking result rather than an error: heavy-atom-only geometries, or a metal with
    no coordination model at all (every metal FF term dropped, no excluded-volume sphere in the DG, no M-L
    length and no polyhedron). A metal complex has its own door, which owns the surrogate.

    The refusal is keyed on being a coordination centre (`metal_core.COORDINATION_METALS`), not on UFF
    typing, because UFF typing is not a property of the element: RDKit builds its label from element +
    coordination + oxidation state, so `Cl[Hg]Cl` types and `[Fe](Cl)(Cl)Cl` does not (measured over all 68
    centres). An element set keyed on the label would be a list of accidents, and the sphere is missing
    either way.
    """
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        raise ValueError("embed() needs a Mol with explicit hydrogens; pass Chem.AddHs(mol)")
    metals = _metal.metal_indices(mol)
    if metals:
        raise ValueError(
            f"atom(s) {metals} are metal centres: a metal is embedded through a SURROGATE, because neither "
            f"the bounds matrix nor UFF describes one directly. Route the complex through "
            f"Isomer(mol, geometry, sites) or enumerate_isomers(mol, geometry), which own that surrogate and "
            f"the coordination model with it"
        )


def embed(
    spec,
    *,
    fix=None,
    constrain=None,
    template=None,
    n=None,
    seed=DEFAULT_SEED,
    prune_rms=0.1,
    threads=0,
    knowledge=True,
):
    """Embed conformers of `spec` under ``fix``/``constrain``; return a `Conformers`.

    Keys are 0-based atom indices in `spec`'s own order; the resolver never SMARTS-matches for you. The full
    vocabulary and how the two verbs compose is the `constraints` module docstring.

    Parameters
    ----------
    spec : Mol | Isomer
        An RDKit `Mol` with explicit Hs (parsing and Hs are the caller's job), or a metal `Isomer`, whose
        polyhedron is composed with the spec so a substrate binds a named coordination sphere.
    fix : list | dict, optional
        Rigid, and the kinds may mix: a list of indices grafts them at `spec`'s own coordinates (needs a
        conformer), ``{i: (x, y, z)}`` at coordinates you supply, ``{(i, j): d, (i, j, k): angle}`` at stated
        numbers the relax is pulled toward -- a pull, not a snap, so read it back with `Conformers.measure`::

            embed(ts_mol, fix=[3, 7, 11, 12])   # graft a reacting core at its own coords, 0.000 Å
            embed(mol, fix={(3, 11): 2.05})     # state the forming bond; the rest is free
    constrain : dict, optional
        Soft: a window on the seed a real energy may overrule. ``{(i, j): (lo, hi)}`` distance,
        ``{(i, j, k): (lo, hi)}`` angle, ``{(ring_a, ring_b): separation}`` π-stack.
    n : int, optional
        Conformer count. ``None`` means a flexibility-scaled count (`bounds.n_confs`), not RDKit's 10.
    seed, prune_rms, threads, knowledge
        Passed to ETKDG. Two defaults are not RDKit's: `seed` is always set (RDKit's -1 draws from the global
        RNG, so every result depends on prior consumption) and `prune_rms` is 0.1, not off.
    """
    if not isinstance(spec, (Chem.Mol, Isomer)):
        raise TypeError(
            f"embed() takes an RDKit Mol or an Isomer, got {type(spec).__name__}; parse a SMILES / .xyz "
            f"yourself (perception is the caller's job) and add Hs"
        )
    if template is not None:  # sugar: a reference core is a coordinate fix. Dissolved before anything routes.
        src = spec.mol if isinstance(spec, Isomer) else spec
        own = src.GetConformer().GetPositions() if src.GetNumConformers() else None
        fix = template_to_fix(template, fix, own)
    iso = spec if isinstance(spec, Isomer) else None  # `seed`/`n` are validated at the seam (`seed_conformers`)
    if iso is None:
        _check_bare_mol(spec)
    mol = Chem.Mol(iso.mol if iso is not None else spec)  # our own copy: conformers are never shared with the caller
    cons, graft_ref = Constraints(), {}
    if iso is None or fix or constrain:
        cons, graft_ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=mol.GetNumConformers() > 0)
    if iso is not None:
        cons = (
            fold_substrate(iso.coordination().copy(), cons, graft_ref)
            if (fix or constrain)
            else iso.coordination().copy()
        )
    mol, ids = seed_conformers(
        mol, cons, iso, n, seed=seed, knowledge=knowledge, prune_rms=prune_rms, threads=threads, graft_ref=graft_ref
    )
    n_frag = len(Chem.GetMolFrags(mol))
    if not ids:  # ETKDG met no constraint set it could realise; silence reads as "embedded, then pruned"
        logger.warning(
            "embed: no conformer under %d constraint(s); the spec may not be realisable",
            len(cons.distances) + len(cons.angles),
        )
    logger.info(
        "embed[%s]: %d seeds (%d atoms, %d fragment%s, %d constraints)",
        iso.summary() if iso is not None else "molecule",
        len(ids),
        mol.GetNumAtoms(),
        n_frag,
        "s" if n_frag != 1 else "",
        len(cons.distances) + len(cons.angles),
    )
    return Conformers(mol, ids, cons, iso, seed=int(seed))  # `minimize`'s handedness gate re-seeds off this


def prepare_relax(spec, *, fix=None, constrain=None):
    """Set an existing geometry up for a search-free relax; return ``(mol, ids, cons, iso)``.

    The assembly `minimize` needs, exposed because a caller wanting its own result type reuses it (the
    pipeline's `Ensemble` does). `spec` is the Mol-or-`Isomer` `embed` takes, and is never touched: the
    working mol is a copy. A coordinate-`fix` core is grafted.

    `pipeline/dispatch.py` builds the same six-step surrogate assembly a second time and this function cannot
    yet be reused there, for reasons written up at that site. Two are structural rather than stylistic: this
    one hardcodes `has_geometry=True` because it relaxes a geometry that already exists, and it CONSUMES the
    graft reference where the embed path needs that reference to survive as far as `seed_conformers`, which
    grafts fresh seeds instead. Unifying them is a change HERE -- take `has_geometry`, hand `ref` back -- not
    a change there.

    The coordination sphere is held either way, from whichever record states it:

    * a plain `Mol`: the metal is surrogated here and its sphere held at the input geometry (`hold_shape`),
      because a perceived complex names no polyhedron to aim at;
    * an `Isomer`: its own `coordination()` is composed, exactly as `embed(spec)` composes it, so one spec
      means one set of constraints in both verbs. Which targets those are is the `Isomer`'s choice, not this
      verb's: `Isomer(mol, geometry, sites)` and `from_geometry` both read the M-donor distances off the input
      conformer when there is one, so "hold the sphere I gave you" is expressed by the isomer you build.
    """
    from .metal_distance import ff_terms
    from .metal_isomers import from_surrogate

    iso = spec if isinstance(spec, Isomer) else None
    mol = Chem.Mol(iso.mol if iso is not None else spec)  # our own copy: never the caller's conformers
    spheres, metals = {}, []
    if iso is None and _metal.metal_index(mol) is not None:  # as `embed`: a bond-less surrogate on a zero-vdW FF,
        spheres = {  # because handing UFF a real metal makes the force field depend on which metal you have
            mi: [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()] for mi in _metal.metal_indices(mol)
        }
        donor_bonds = [(d, mi) for mi, dons in spheres.items() for d in dons]  # re-added DATIVE on the output
        mol, metals = _metal.surrogate_all_metals(mol)
        iso = from_surrogate(mol, metals, donor_bonds, donors=spheres.get(metals[0][0], ()))
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=True)
    if spheres:  # perceived from a plain Mol: hold what the input realises
        rz = {mi: z for mi, z, _q in metals}
        for mi, dons in spheres.items():
            _metal.hold_shape(mol, [mi, *dons], cons)
        ff_terms(mol, cons, {mi: (rz[mi], dons) for mi, dons in spheres.items()})
    elif iso is not None:  # an Isomer arrives with its polyhedron built: the same fold `embed` does
        base = iso.coordination().copy()  # copy: the relax edits cons in place, the Isomer keeps its record
        cons = fold_substrate(base, cons, ref) if (fix or constrain) else base
    ids = [c.GetId() for c in mol.GetConformers()]
    if ref:
        graft = sorted(ref)
        graft_frozen(mol, ids, graft, np.array([ref[i] for i in graft]))
    return mol, ids, cons, iso


def minimize(spec, *, fix=None, constrain=None, template=None, stiffness=None):
    """Relax an existing geometry toward ``fix``/``constrain``: the search-free companion to `embed`.

    Same vocabulary as `embed`, no conformer search: the input geometry is kept and pulled toward the targets.
    Needs a `Mol` carrying at least one conformer, or an `Isomer` whose mol does; a bare graph has nothing to
    relax, and perception is the caller's job either way.

    An `Isomer` keeps its coordination: its polyhedron is composed with `fix`/`constrain` exactly as in
    `embed(iso)`, so the relax cannot ignore the arrangement it was handed (see `prepare_relax`).

        confs = minimize(mol, fix={(i, j): 2.0, (i, j, k): 178})   # pull toward a linear 3-centre core
    """
    if not isinstance(spec, (Chem.Mol, Isomer)):
        raise TypeError(f"minimize() takes an RDKit Mol or an Isomer, got {type(spec).__name__}")
    if template is not None:
        src = spec.mol if isinstance(spec, Isomer) else spec
        own = src.GetConformer().GetPositions() if src.GetNumConformers() else None
        fix = template_to_fix(template, fix, own)
    if not (spec.mol if isinstance(spec, Isomer) else spec).GetNumConformers():
        raise ValueError(
            "minimize() relaxes an existing geometry; give a Mol with a conformer (embed() first if you have "
            "only a graph)"
        )
    mol, ids, cons, iso = prepare_relax(spec, fix=fix, constrain=constrain)
    confs = Conformers(mol, ids, cons, iso)
    return confs.minimize() if stiffness is None else confs.minimize(stiffness=stiffness)
