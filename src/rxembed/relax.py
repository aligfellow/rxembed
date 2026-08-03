"""Force-field energies, constraint-enforcing restrained UFF, and the arbiter that accepts its output.

The embed only biases constraints, ETKDG torsion terms overriding soft bounds,
so a restrained minimisation is what actually enforces them: distance/angle
windows, and frozen atoms pinned exactly (zero DOF).

`bonding_ok` lives here because it is what every relax stage accepts on: a restrained UFF can pull a bond
apart to satisfy a window, and no energy tells you that happened.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import GetPeriodicTable, rdForceFieldHelpers

from . import mechanisms as _mech
from .metal_core import _METAL_Z, materialise_phantoms
from .utils import remove_bond

FF_SURROGATE = (
    3  # lithium: a bond-less UFF-typeable surrogate carrying only a vdW term, a soft sphere holding non-donors
)
# off the metal while the donors are held explicitly. Must stay bond-less: bonded, UFF types Li linear and
# its 1/(4 sin²θ₀) angle term is singular, injecting ~1e9 kcal/mol into a CN>=3 sphere.
UFF_GHOST = 54  # xenon: the FF type for a haptic centroid dummy, which sits ~0.8 Å inside its own ring where a real
# element's r⁻¹² vdW is astronomical. It must carry zero terms, reached the only way RDKit allows: an
# element the UFF typer cannot type. Only metal-side phantoms are ghosted; a D-cap keeps its terms.

logger = logging.getLogger("rxembed.relax")  # the name `set_verbose` configures, spelled out rather than
#   __name__ so a module rename cannot move a log line out from under it.

MAX_ITERS = 500  # the one restrained-UFF iteration cap: every relax entry point defaults from this name
_PT = GetPeriodicTable()


def bonding_ok(mol, conf_id, bond_tol=1.3, clash_tol=0.7, exclude=frozenset(), constrained=()):
    """Return True if geometry-perceived connectivity matches the graph (heavy atoms).

    Three things are skipped so *valid* geometries aren't rejected:

    * pairs inside a frozen or reacting core (``exclude``): a partial forming or breaking bond is held to the
      reference, not a ground-state bond;
    * any pair involving a metal, whose dative/coordinate distances covalent radii don't describe;
    * pairs in ``constrained`` (pass ``Constraints.distances``): a pair whose separation the constraint
      system *states* has no chemistry left for a radius rule to judge. ``rx.embed('CCCl', fix={(1, 2): 2.4})``
      asks for a dissociating C-Cl; calling the result broken rejects exactly the requested geometry. Whether
      the stated window was met is a different question, answered by ``check_constraints`` / ``.measure()``.

    The exemption is per pair, not per atom: a bond that tore elsewhere in a molecule that also carries
    constraints is still caught, and so is a genuinely broken free-periphery bond (the ensemble may
    legitimately empty).
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    metals = {i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() in _METAL_Z}
    exclude = set(exclude)
    stated = {frozenset(p) for p in constrained}
    rcov = {i: _PT.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in heavy}
    bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    for n, i in enumerate(heavy):
        for j in heavy[n + 1 :]:
            if (i in exclude and j in exclude) or i in metals or j in metals:
                continue
            if frozenset((i, j)) in stated:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            cut = rcov[i] + rcov[j]
            if frozenset((i, j)) in bonded:
                if d > bond_tol * cut or d < clash_tol * cut:  # bonded pair stretched/broken OR crushed
                    return False
            elif d < clash_tol * cut:  # non-bonded pair fused/clashing
                return False
    return True


@contextlib.contextmanager
def _quiet_uff(active):
    """Silence RDKit's UFF-typer log iff `active`: the haptic centroid ghost is untypeable by design.

    Being untypeable is how the ghost carries zero energy terms, so its warning is noise. Gated on a phantom
    actually being present, so a real typing failure in a normal system is still reported.
    """
    if active:
        with rdBase.BlockLogs():
            yield
    else:
        yield


def _bond_pruned(mol, frozen):
    """Copy `mol` with bonds *between two frozen atoms* removed; return None if there are none.

    Frozen atoms are held exactly by ``AddFixedPoint`` (zero DOF), so a bond purely among them carries no
    force on any relaxing atom. But a TS reacting core's partial or hypervalent bonds (a sub-Å H...H being
    cleaved, a bridging hydride, an over-coordinated centre) make UFF's line search diverge ("bad direction
    in linearSearch"). Dropping only frozen-frozen bonds removes those pathological terms; every
    frozen-FREE bond (which anchors a relaxing atom to the held core) is kept, and no atom moves.
    """
    ff_bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in frozen and b.GetEndAtomIdx() in frozen
    ]
    if not ff_bonds:
        return None
    rw = Chem.RWMol(mol)  # copies the conformers (atom indices unchanged)
    for a in frozen:
        rw.GetAtomWithIdx(a).SetNoImplicit(True)  # dropping a bond must not sprout an implicit H
    for i, j in ff_bonds:
        remove_bond(rw, i, j)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    return out


def ff_energies(mol, minimize=True):
    """FF energies (MMFF94s where typeable, else UFF); optimise in place first when ``minimize``."""
    use_mmff = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
    if minimize:
        res = (
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant="MMFF94s")
            if use_mmff
            else rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        )
        return np.array([e for _conv, e in res])
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s") if use_mmff else None

    def ff(c):
        return (
            rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=c)
            if use_mmff
            else rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=c)
        )

    return np.array([ff(c.GetId()).CalcEnergy() for c in mol.GetConformers()])


def _ff_surrogate(mol, metals, phantoms=()):
    """Return a copy of ``mol`` with metals re-typed to the zero-vdW FF surrogate and phantoms Xe-ghosted.

    Nothing is added or removed, so every atom index is unchanged on both sides and no restraint can be silently
    dropped or mis-keyed. Each metal -> ``FF_SURROGATE`` (Li, a soft zero-vdW sphere); each ``phantom`` (a haptic
    centroid dummy) -> ``UFF_GHOST`` (Xe, untypeable, so UFF omits every term: it sits inside its own ring
    where a real vdW would explode). ``mol`` itself when there is nothing to re-type.
    """
    if not metals and not phantoms:
        return mol  # an organic system: the identical object, so this whole path is a strict no-op
    rw = Chem.RWMol(mol)  # copies the conformers
    for m in metals:
        a = rw.GetAtomWithIdx(int(m))
        a.SetAtomicNum(FF_SURROGATE)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
    for p in phantoms:  # a haptic centroid dummy: kept for its restraints, but zero energy terms (see UFF_GHOST)
        a = rw.GetAtomWithIdx(int(p))
        a.SetAtomicNum(UFF_GHOST)
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    out.UpdatePropertyCache(strict=False)
    return out


def restrained_uff(mol, cons, distance_fc=500.0, max_iters=MAX_ITERS, conf_ids=None):
    """Minimise each conformer enforcing distance/angle windows + frozen atoms (pinned exactly, zero DOF).

    A metal brings more fields, all empty for an organic system (so this is bit-identical there): ``cons.metals``
    re-types the metal to the zero-vdW surrogate, ``cons.pulls`` adds the soft harmonic inside the flat-bottomed
    distance walls, ``cons.floors`` the anti-overbond guard, ``cons.coplanar`` the coplanarity cap. They ship
    together; each dropped alone regresses (see `FF_SURROGATE` / `distance.ff_terms`).

    `conf_ids` restricts the settle to a subset of conformers (default: all), used to settle different seeds to
    different in-window targets.
    """
    frozen = set(cons.frozen)
    work = materialise_phantoms(mol, cons.haptic)  # transient centroid dummies for a haptic face; `mol` else
    work = _ff_surrogate(work, cons.metals, cons.phantoms)  # `mol` itself when there is no metal/phantom

    def build(target, conf_id):
        """Build a restrained UFF for one conformer by walking the mechanism registry.

        A flat additive loop: unlike the DG build there is no phase order here, because FF terms are
        independent. `mechanisms.MECHANISM_ORDER` still fixes the sequence so the two writers stay in step and a
        field's two halves are read together.
        """
        with _quiet_uff(bool(cons.phantoms)):  # the Xe ghost is untypeable by design, so hush that warning
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(target, confId=conf_id, ignoreInterfragInteractions=False)
        conf = target.GetConformer(conf_id)
        for m in _mech.MECHANISM_ORDER:
            m.ff_terms(ff, cons, conf, distance_fc)
        ff.Initialize()
        return ff

    def bring_home(src_mol, conf):  # copy the relaxed coords back off a working copy (same indices, always)
        if src_mol is mol:
            return
        src = src_mol.GetConformer(conf.GetId())
        for a in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(a, src.GetAtomPosition(a))

    pruned = None  # a bond-pruned copy, built lazily only if a conformer's minimise diverges
    energies = []
    confs = mol.GetConformers() if conf_ids is None else [mol.GetConformer(int(i)) for i in conf_ids]
    for conf in confs:
        cid = conf.GetId()
        ff = build(work, cid)  # a BUILD failure (UFF can't type the graph) propagates -> caller keeps the embed
        try:
            ff.Minimize(maxIts=max_iters)
            energies.append(ff.CalcEnergy())
            bring_home(work, conf)
        except RuntimeError:  # the UFF line search diverged on this conformer: a pathological frozen-core
            if pruned is None:  # partial/hypervalent bond, or an RDKit BFGS "bad direction" on a hard metal core.
                pruned = _bond_pruned(work, frozen)  # never fatal: retry pruned, else keep this conformer unrelaxed.
            relaxed = False
            if pruned is not None:  # retry on a copy with the frozen-frozen partial bonds dropped
                try:
                    pff = build(pruned, cid)
                    pff.Minimize(maxIts=max_iters)
                    src = pruned.GetConformer(cid)  # copy the relaxed FREE-atom positions back (frozen unmoved)
                    for a in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(a, src.GetAtomPosition(a))
                    energies.append(pff.CalcEnergy())
                    relaxed = True
                except RuntimeError:
                    pass
            if not relaxed:  # keep this conformer's geometry (frozen core still pinned); bonding_ok is the arbiter
                logger.debug("restrained_uff: conformer %d could not relax (BFGS diverged); kept unrelaxed", cid)
                try:
                    energies.append(float(ff.CalcEnergy()))
                except RuntimeError:
                    energies.append(float("nan"))  # can't even score it; a placeholder kept aligned to confs
    return np.array(energies)
