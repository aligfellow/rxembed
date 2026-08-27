"""Chainable pipeline ensembles for search, selection and scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import rxembed.metal_core as _metal
import rxembed.metal_polyhedron as _poly
from rxembed.constraints import _graft_owns, constraint_value, match, resolve_atom, within_window
from rxembed.embed import BASE_STIFFNESS as _BASE_STIFFNESS
from rxembed.embed import Conformers
from rxembed.relax import MAX_ITERS as _MAX_ITERS
from rxembed.relax import _error_summary, ff_energies, restrained_uff
from rxembed.stereo import matches_stereo

from . import geom_check as _geometry
from . import metrics as _metrics
from . import nci as _nci
from . import search as _mc
from . import select as _dedup
from . import stereo_check as _stereo

logger = logging.getLogger("rxembed")

_MIN_OVERLAY_ATOMS = 3  # need >=3 atoms to define an alignment frame
_HARTREE_KCAL = 627.5094740631  # Eh -> kcal/mol
# Relax failures lie 1e3-1e11 kcal/mol above the minimum; 250 stays beyond any physical rotamer.
_RELAX_ENERGY_WINDOW = 250.0
_REPLACEMENT_SEED = 0xF00D  # fallback for a search-free metal minimize with no originating embed seed
_SEEDED, _RELAXED, _MINIMIZED = "seeded", "relaxed", "minimized"


def _last_line(err):
    """Return the last non-empty line of an exception (xtb dumps a stderr tail) for a one-line warning."""
    s = str(err).strip()
    return s.splitlines()[-1] if s else "no output"


class EnsembleSet(list):
    """Hold distinct candidate ensembles, each identified by its `.tag`."""

    def __repr__(self):
        """Summarise the candidates and their tags on one line."""
        labels = [e.tag.get("label") or e.tag.get("stereo") or e.tag.get("arrangement") or e.tag for e in self]
        return f"<EnsembleSet: {len(self)} candidate{'s' if len(self) != 1 else ''} {labels}>"

    def select(self, **tag):
        """Return the single candidate matching `tag`, as a chainable `Ensemble`.

        Raises if the tags match zero or several; narrow them (e.g. add `geometry=`), or use `filter` /
        iterate to keep several.
        """
        hits = self.filter(**tag)
        if len(hits) != 1:
            raise ValueError(
                f"select({tag}) matched {len(hits)} candidate(s): "
                f"{'narrow the tags' if hits else 'no match'}; have {[e.tag for e in self]}"
            )
        return hits[0]

    def filter(self, by=None, **tag):
        """Filter candidates by tag, or map a conformer filter when `by` is given."""
        if by is not None:
            return self._map("filter", by, **tag)
        # a geometry is nameable by its 3-letter code everywhere else (`rx.metal`, `IsomerSet.filter`), so it
        # must be here too, or filter(geometry='SPL') silently returns nothing
        tag = {k: (_poly.resolve_geometry(v) if k == "geometry" else v) for k, v in tag.items()}

        def matches(ensemble, key, value):
            stored = ensemble.tag.get(key)
            return stored == value or (key == "stereo" and matches_stereo(stored or "", value))

        return EnsembleSet(e for e in self if all(matches(e, k, v) for k, v in tag.items()))

    def summary(self):
        """Print each candidate (index, identity, #seeds) so you can pick one. Returns self (chainable).

        A metal candidate shows geometry / per-vertex arrangement / chirality; other candidates show their raw tag.
        """
        for k, e in enumerate(self):
            t = e.tag or {}
            if "arrangement" in t:
                chir = t.get("chirality") or "(achiral)"
                ident = f"{t.get('geometry', ''):16s} {t['arrangement']:26s} {chir}"
            else:
                ident = str(t)
            print(f"  [{k}] {ident}  ({e.n} seeds)")
        return self

    #: verbs whose per-candidate output is comparable across candidates only via a real calculator, not the FF
    _REAL_ENERGY_VERBS = frozenset({"score", "optimize"})

    def _map(self, method, *args, **kw):
        """Apply one verb within each candidate; never pool or cross-prune distinct species."""
        logger.info(
            "%s: %d candidates, each on its own constraints%s",
            method.lstrip("_"),
            len(self),
            "; real energies are comparable" if method in self._REAL_ENERGY_VERBS else "",
        )
        out = EnsembleSet()
        for e in self:
            res = getattr(e, method)(*args, **kw)
            res.tag = res.tag or dict(e.tag)  # a returned-new ensemble (score/optimize/lowest) carries the tag
            out.append(res)
        return out

    def mc(self, *args, **kw):
        """Monte-Carlo conformer search on each candidate (see `Ensemble.mc`)."""
        return self._map("mc", *args, **kw)

    def minimize(self, *args, **kw):
        """FF-relax each candidate (see `Ensemble.minimize`)."""
        return self._map("minimize", *args, **kw)

    def prune(self, *args, **kw):
        """Dedup each candidate's conformers (see `Ensemble.prune`)."""
        return self._map("prune", *args, **kw)

    def score(self, *args, **kw):
        """Real-energy score each candidate (see `Ensemble.score`); energies stay per-species."""
        return self._map("score", *args, **kw)

    def optimize(self, *args, **kw):
        """Geometry-optimise each candidate (see `Ensemble.optimize`)."""
        return self._map("optimize", *args, **kw)

    def lowest(self, *args, **kw):
        """Keep the lowest-energy conformer(s) within each candidate (see `Ensemble.lowest`)."""
        return self._map("lowest", *args, **kw)

    def representatives(self, *args, **kw):
        """Distinct representatives within each candidate (see `Ensemble.representatives`)."""
        return self._map("representatives", *args, **kw)

    def best(self, n=1):
        """Rank candidates by real energy; return one Ensemble or the best `n` as an EnsembleSet."""
        if any(e.energy_kind != "real" for e in self):
            raise ValueError(
                "best() needs real energies for distinct species; run .score('gxtb') or .optimize('gxtb') first"
            )

        def emin(e):
            return min(e.energies[i] for i in e.ids if i in e.energies)

        ranked = sorted(self, key=emin)
        lo = emin(ranked[0])
        order = ", ".join(f"{self._slug(e.tag or {}, k)}(+{emin(e) - lo:.1f})" for k, e in enumerate(ranked))
        logger.info("best: ranked %d by ΔE (kcal/mol): %s -> kept %d", len(self), order, min(n, len(ranked)))
        kept = EnsembleSet(ranked[:n])
        return kept[0] if n == 1 else kept

    def __getattr__(self, name):
        """Explain why a scalar Ensemble attribute has no answer for several candidates."""
        # annotations too: `Conformers.ids` / `.energies` are bare class-level annotations, not attributes
        known = hasattr(Ensemble, name) or any(name in getattr(k, "__annotations__", {}) for k in Ensemble.__mro__)
        if name.startswith("_") or not known:
            raise AttributeError(name)
        labels = [e.tag.get("label") or e.tag.get("stereo") or e.tag.get("nci") or e.tag for e in self]
        raise AttributeError(
            f"{name!r} is an Ensemble verb, but this is an EnsembleSet of {len(self)} distinct candidate(s) "
            f"{labels}, which are never pooled. Pick one (`[0]`, `.select(label=…)`), map a chainable verb "
            f"(mc/minimize/prune/score/optimize/lowest/representatives/dump), or pass stereo='free' to "
            f"rx.embed if an undefined stereocentre split this and you did not want it."
        )

    @staticmethod
    def _slug(tag, k):
        """Filename-safe identifier for a candidate from its tag (geometry / stereo / label / nci …), else its index."""
        # `geometry` leads: two candidates of different shapes otherwise slug identically and dump as c0/c1
        keys = ("geometry", "arrangement", "stereo", "label", "nci", "chirality", "coordinate")
        s = "_".join(str(tag[key]) for key in keys if tag.get(key)) or f"c{k}"
        return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in s)

    def dump(self, path, align=True):
        """Write one tagged multi-frame XYZ per candidate; return their paths."""
        from pathlib import Path

        base = Path(path)
        paths = []
        for k, e in enumerate(self):
            p = base.with_name(f"{base.stem}_{self._slug(e.tag or {}, k)}{base.suffix or '.xyz'}")
            e.dump(str(p), align=align)
            paths.append(str(p))
        return paths


@dataclass
class Ensemble(Conformers):
    """Add search, selection and scoring verbs to a core Conformers result.

    Mutating verbs return `self`; derived selections and real-energy calculations return a new Ensemble.
    `_mol` stays suitable for DG/FF work, while `.mol` exposes the restored connected metal graph.
    """

    _stage: str = _SEEDED
    discarded: list = field(default_factory=list)  # dropped ids retained privately for duplicates()
    tag: dict = field(default_factory=dict)  # what distinguishes this candidate in an EnsembleSet
    _stereo: tuple | None = None  # (spec, reference signature) for the minimize() chirality filter
    _donor_hand: dict = field(default_factory=dict)  # labile donor -> signed-volume hand
    energy_kind: str = ""  # "" | "ff" (surrogate, not cross-species) | "real" (xtb); best() ranks only "real"
    reacted: dict = field(default_factory=dict)  # {conf id: (formed, broken)} where a stage changed the graph
    sphere: dict = field(default_factory=dict)  # metal -> donors, durable across restored connected graphs

    def mc(
        self,
        *,
        preset="ensemble",
        seed=None,
        max_out=None,
        low_mode=None,
        config=None,
        replace=None,
        explore=False,
        **openconf_kw,
    ):
        """Openconf Monte-Carlo torsional search, in place.

        `preset` sets the effort ('rapid'|'ensemble'|'spectroscopic'|'docking'|'analogue'|'macrocycle'|
        'transition_metal'), though every preset also auto-adds a metal move budget. `seed`, `max_out`,
        `low_mode` (off under constraints) and `config` override single knobs; any other keyword is passed
        through as a ``ConformerConfig`` field, and kwargs win over `config=`.

        Unconstrained, openconf replaces the ETKDG seeds. Constrained or multi-fragment, it searches around the
        pose-frozen seeds and its output is added instead; `replace=` overrides.

        `explore=True` on a seeded NCI complex fires a second search with the contacts released and the
        structural holds kept, pools both, and swaps cons to the relaxed set so a later stage cannot yank the
        contacts back; energy then decides. On the metal path only substrate contacts release.
        """
        if not _mc.available():
            raise ImportError("mc needs openconf; pip install 'rxembed[search]'")
        if preset == "rapid" and self.cons.is_constrained:  # constrained pose-mode is rotor-only, so rapid
            logger.warning("mc: preset='rapid' under-samples a constrained system; use the default 'ensemble'")
        if self.cons.is_constrained:
            logger.info(
                "mc: %d constrained atom(s); pose-frozen rotor search",
                len(self.cons.constrained_atoms()),
            )

        # Search may mutate existing coordinates before it fails, and its new geometries have no score yet.
        # Invalidate before either can happen so stale energies never describe a changed conformer.
        self.energies = {}
        self.unrelaxed = []
        self.reacted = {}
        self.trajectory = None
        self.energy_kind = ""
        self._stage = _SEEDED
        if self.iso is not None:  # restore the selected surrogate graph before any stage moves atoms again
            current = self._mol
            self._mol = Chem.Mol(self.iso.mol)
            self._mol.RemoveAllConformers()
            for conf in current.GetConformers():
                self._mol.AddConformer(Chem.Conformer(conf), assignId=False)
        self._settle_seeds()  # spread the seeds across their windows before openconf pose-freezes them

        def _search(cons, label):
            try:
                return _mc.search(
                    self._mol,
                    cons,
                    preset=preset,
                    seed=seed,
                    max_out=max_out,
                    low_mode=low_mode,
                    config=config,
                    **openconf_kw,
                )
            except Exception as e:  # openconf can't handle every system (e.g. a TS hypervalent core)
                logger.warning(
                    "mc%s: openconf could not search this system (%s: %s); keeping the %d conformer(s)",
                    label,
                    type(e).__name__,
                    e,
                    len(self.ids),
                )
                return None

        added = _search(self.cons, "")
        if added is None:
            return self
        if replace is None:
            replace = not self.cons.is_constrained  # unconstrained: openconf supersedes ETKDG seeds
        if replace:
            self.ids = added
            logger.info("mc: %d conformers (openconf '%s' drove generation; replaced ETKDG seeds)", len(added), preset)
        else:
            self.ids += added
            logger.info(
                "mc: +%d (openconf '%s', pose-constrained around seeds; total %d)", len(added), preset, len(self.ids)
            )
        self._stage = _SEEDED  # openconf's geometries are its own FF's, so minimize must genuinely relax

        if explore and any(self.cons.contacts):  # second pass: contacts released, structure kept
            from rxembed.embed import encounter_bounds

            relaxed = self.cons.relaxed()
            # releasing the grip frees the fragments it linked, so explore re-bounds every inter-fragment pair
            # (setdefault never overrides a surviving structural hold), unlike `float_encounter_bounds`.
            if len(Chem.GetMolFrags(self._mol)) > 1:  # keep the fragments together once contacts are freed
                for k, v in encounter_bounds(self._mol).items():
                    relaxed.distances.setdefault(k, v)
            more = _search(relaxed, " explore")
            if more:
                self.ids += more
                self.cons = relaxed  # downstream relax/score/opt no longer yank contacts
                logger.info(
                    "mc explore: +%d conformers; contacts released, %d holds kept, %d total",
                    len(more),
                    len(relaxed.distances),
                    len(self.ids),
                )
        return self

    def _settle_seeds(self, bins=5):
        """Spread seeds across non-frozen constraint windows before an MC search."""
        cons = self.cons
        seeded_d = [(k, v) for k, v in cons.distances.items() if not (k[0] in cons.frozen and k[1] in cons.frozen)]
        shape_d = {k: v for k, v in cons.distances.items() if k[0] in cons.frozen and k[1] in cons.frozen}
        if not (seeded_d or cons.angles or cons.dihedrals) or not self.ids:
            return
        for b, group in enumerate(g for g in np.array_split(list(self.ids), min(len(self.ids), bins)) if len(g)):
            frac = (b + 0.5) / min(len(self.ids), bins)  # this bin's fraction across every window
            # A partial rebuild: coordinate windows are re-derived per bin, everything else is carried.
            tgt = cons.copy(
                distances={},
                angles={},
                dihedrals={},
                contacts=(frozenset(), frozenset()),
                dg_floors={},
            )
            tgt.distances.update(shape_d)  # keep the frozen-core shape exact
            for k, (lo, hi) in seeded_d:
                m = lo + frac * (hi - lo)
                tgt.distances[k] = (max(lo, m - 0.03), min(hi, m + 0.03))  # clamp inside the user window
            for k, (lo, hi) in cons.angles.items():
                m = lo + frac * (hi - lo)
                tgt.angles[k] = (max(lo, m - 2.0), min(hi, m + 2.0))
            for k, (lo, hi) in cons.dihedrals.items():
                m = lo + frac * (hi - lo)
                tgt.dihedrals[k] = (max(lo, m - 2.0), min(hi, m + 2.0))
            try:  # settling is best-effort pre-conditioning; an RDKit BFGS divergence must not sink mc()
                restrained_uff(self._mol, tgt, stiffness=_BASE_STIFFNESS, conf_ids=[int(i) for i in group])
            except RuntimeError:  # keep the raw ETKDG seeds for this group (openconf still searches around them)
                logger.debug("mc: seed-settle relax diverged on a group; keeping the raw seeds")

    def _workflow_failure(self, owner, cid):
        """Return the first pipeline-only publication failure for one conformer."""
        mol = owner.mol
        try:
            changed = self._scan_connectivity(mol, [cid])
        except ImportError:  # connectivity re-perception is a workflow-extra gate; core bonding still runs
            changed = {}
        if changed:
            return "changed connectivity"
        if self._stereo:
            spec, ref = self._stereo
            if not _stereo.satisfies_spec(_stereo.signature(mol, cid), ref, spec):
                return "wrong requested stereo"
        for donor, hand in self._donor_hand.items():
            if hand is not None and _metal.donor_chirality_sign(owner._mol, cid, donor) != hand:
                return f"wrong donor-{donor} hand"
        return None

    def _required_failure(self, failures):
        """Extend the core count contract with failed pipeline-only identity requests."""
        required = [super()._required_failure(failures)]
        if "wrong requested stereo" in failures:
            required.append("requested stereo")
        if any(reason.startswith("wrong donor-") for reason in failures):
            required.append("requested donor hand")
        return " and ".join(filter(None, required))

    def _relax_into_windows(self, trajectory=False):
        """Relax constrained seeds into their windows without finalizing the result.

        The metal context remains live for later acceptance and retry gates. Embed publishes geometry, not an
        energy, and unconstrained seeds receive no unrequested force-field pass.
        """
        if trajectory and not self.cons.is_constrained:
            raise ValueError("trajectory records restrained-UFF cleanup; this embed has no constraints to clean up")
        self.trajectory = None
        if not self.ids or not self.cons.is_constrained:
            return self
        before = list(self.ids)
        template = Chem.Mol(self._mol) if self.iso is not None else None
        frames = [] if trajectory else None
        e = self._relax_constrained(_BASE_STIFFNESS, operation="embed", _frames=frames)
        self._accept_relaxed(
            _BASE_STIFFNESS,
            _MAX_ITERS,
            operation="embed",
            validator=self._workflow_failure,
            template=template,
            seed=self.seed,
        )
        self.discarded += [cid for cid in before if cid not in set(self.ids)]
        self._store_trajectory(frames)
        self.energies = {}  # embed publishes geometry; minimize owns the FF score
        self.unrelaxed = [cid for cid in self.unrelaxed if cid in self.ids]
        if self.ids and e is not None and not self.unrelaxed:
            self._stage = _RELAXED
        return self

    def minimize(self, stiffness=_BASE_STIFFNESS, max_iters=_MAX_ITERS):
        """Relax, validate and finalize the ensemble in place.

        Every geometry uses the shared UFF relaxation; populated `Constraints` add their restraint terms.
        Rejected ids remain available to `duplicates()` and are listed in `discarded`. Structural failures use
        the core fresh-seed replacement. The call is idempotent until a search changes the geometries.
        """
        if self._stage == _MINIMIZED:
            return self
        if not self.ids:  # nothing embedded (e.g. ETKDG could not place this graph)
            self._stage = _MINIMIZED
            return self
        iso = self.iso  # capture before restore: the coordination-planarity gate needs metal + donors
        before = list(self.ids)
        template = Chem.Mol(self._mol) if iso is not None else None
        if self.cons.is_constrained and self._stage == _RELAXED:
            try:
                e = restrained_uff(self._mol, self.cons, stiffness=stiffness, max_iters=0)
            except RuntimeError as err:
                e = None
                logger.warning(
                    "minimize: UFF could not score the embedded geometry (%s)",
                    _error_summary(err),
                )
            if e is not None:
                self.energies = {c.GetId(): float(e[k]) for k, c in enumerate(self._mol.GetConformers())}
        else:
            self.trajectory = None
            self._relax_once(stiffness, max_iters)
        self.energy_kind = "ff" if self.energies else ""
        self._accept_relaxed(
            stiffness,
            max_iters,
            validator=self._workflow_failure,
            template=template,
            seed=self.seed if self.seed is not None else (_REPLACEMENT_SEED if iso is not None else None),
        )
        scored = {cid: self.energies[cid] for cid in self.ids if cid not in self.unrelaxed and cid in self.energies}
        if scored:
            emin = min(scored.values())
            high = [cid for cid, energy in scored.items() if energy > emin + _RELAX_ENERGY_WINDOW]
            self._remove(high)
            if high:
                logger.info("minimize: dropped %d non-physical high-energy conformer(s)", len(high))
        self._stereo = None
        if iso is not None:
            spheres = {}
            for donor, metal in iso.donor_bonds:
                spheres.setdefault(metal, []).append(donor)
            for metal, donors in spheres.items():
                self.sphere.setdefault(metal, donors)
            iso.restore(self._mol)
        if not self.ids:
            logger.warning(
                "minimize: 0 of %d conformer(s) survived acceptance; the arrangement may be infeasible",
                len(before),
            )
        if self.cons.is_constrained and self.energies:
            # Re-embedded batches may have needed different restraint stiffnesses. The final public energies
            # must share one objective before prune/lowest compares them.
            self._rescore_restrained(stiffness)
            self.energy_kind = "ff" if self.energies else ""
        self.discarded += [i for i in before if i not in set(self.ids)]
        self._validate()
        if iso is not None and iso.donor_bonds:  # connectivity finalize, last: geometry/element/charge settled
            # surrogate-stripped M-donor bonds as dative. Every gate above saw the bond-less surrogate.
            self._mol = _metal.connect_metal(self._mol, iso.donor_bonds)
        if self.trajectory is not None:
            frames = [conf.GetPositions().copy() for conf in self.trajectory.GetConformers()]
            self._store_trajectory(frames)  # clear it if a later acceptance gate replaced the endpoint
        self.unrelaxed = [i for i in self.unrelaxed if i in self.ids]
        self._stage = _MINIMIZED
        return self

    def _validate(self, dist_slack=0.15, ang_slack=5.0):
        """Warn when a relaxable constraint misses its window and tolerance.

        Frozen holds and approximate donor-metal-donor angles are debug diagnostics, not failures.
        """
        if not self.ids:
            return
        if self.iso is not None:
            donors = sorted({int(d) for ds in self.sphere.values() for d in ds} | {int(d) for d in self.iso.donors})
            failed = sum(
                not _geometry.check(self._mol, cid, frozen=self.cons.frozen, donors=donors or None).ok()
                for cid in self.ids
            )
            if failed:
                logger.debug("minimize: broad geometry QA flagged %d/%d conformer(s)", failed, len(self.ids))
        frozen = self.cons.frozen
        metals = {a.GetIdx() for a in self._mol.GetAtoms() if a.GetAtomicNum() in _metal.COORDINATION_METALS}
        terms = (
            ("distance", self.cons.distances, dist_slack),
            ("angle", self.cons.angles, ang_slack),
            ("dihedral", self.cons.dihedrals, ang_slack),
        )
        for name, windows, slack in terms:
            for atoms, window in windows.items():
                values = [
                    constraint_value(self._mol.GetConformer(cid).GetPositions(), atoms, self.cons.haptic, window)
                    for cid in self.ids
                ]
                if any(value is None or not np.isfinite(value) for value in values):
                    logger.warning(
                        "constraint not measurable: %s%s; check atom indices and haptic metadata", name, atoms
                    )
                values = [value for value in values if value is not None and np.isfinite(value)]
                missed = [value for value in values if not within_window(value, window, slack)]
                if not missed:
                    continue
                lo, hi = window
                value = max(missed, key=lambda v: max(lo - v, v - hi))
                approximate = _graft_owns(atoms, frozen, self.cons.haptic) or (name == "angle" and atoms[1] in metals)
                log = logger.debug if approximate else logger.warning
                log("constraint not held: %s%s = %.2f, target %.2f-%.2f", name, atoms, value, lo, hi)

    def score(self, refine="gxtb", solvent=None, charge=None):
        """Re-rank by a real single-point energy (g-xTB by default), returning a new Ensemble.

        ``refine`` accepts ``'gxtb'``, ``'gfn2'``, ``'ff'`` or a calculator. Energies are kcal/mol; the geometry
        is unchanged after the implicit ``minimize()``. ``solvent`` applies only to GFN2.
        """
        self.minimize()  # settle the periphery (a constrained core stays pinned)
        if not self.ids:
            return self._derive([])
        from .calculators import resolve

        q = self._calc_charge(charge)
        calc = resolve(refine, solvent, q)
        if calc is None:  # 'ff'/None -> single-point force field, geometry kept
            e = ff_energies(self._mol, minimize=False)
            by = {c.GetId(): float(e[k]) for k, c in enumerate(self._mol.GetConformers())}
            out = self._derive(self.ids)
            out.energies = {i: by[i] for i in self.ids if i in by}
            out._stage = _MINIMIZED
            out.energy_kind = "ff"  # a force-field single point is not a real energy, so best() still refuses it
            return out
        energies, kept = {}, []
        for i in self.ids:
            try:
                energies[i] = float(calc.energy(self._mol, i)) * _HARTREE_KCAL  # -> kcal/mol, unit-consistent
                kept.append(i)
            except Exception as err:  # a single conformer xtb failure shouldn't sink the run
                logger.warning("score: %s failed on conformer %d (%s); dropping it", refine, i, _last_line(err))
        if not kept:  # total failure raises: never hand back FF energies the caller thinks are xTB
            raise RuntimeError(
                f"score: {refine} produced no energies. Is the xtb binary on PATH "
                f"($XTB_EXE, or ~/bin/xtb)? not falling back to the force field silently"
            )
        logger.info("score: %s single point on %d conformer(s) (charge %d)", refine, len(kept), q)
        out = self._derive(kept)
        out.energies = energies
        out._stage = _MINIMIZED
        out.discarded += [i for i in self.ids if i not in set(kept)]
        out.energy_kind = "real"  # xtb/g-xTB, comparable across species, so EnsembleSet.best() accepts it
        return out

    def _calc_charge(self, charge):
        """Return the explicit charge or the molecule's formal charge, warning for perceived metal charges."""
        q = Chem.GetFormalCharge(self._mol) if charge is None else charge
        if charge is None and q != 0 and _metal.metal_index(self._mol) is not None:
            logger.warning(
                "perceived formal charge %d is likely a perception artefact; pass charge=<total>",
                q,
            )
        return q

    def optimize(self, refine="gxtb", level="normal", solvent=None, charge=None):
        """Geometry-optimise each conformer with xtb at `level`, returning a new ensemble.

        ``level`` accepts ``'loose'``, ``'normal'``, ``'tight'`` or ``'vtight'``. Frozen atoms stay fixed;
        numeric fixes are validated afterward. Optimised coordinates and kcal/mol energies are stored on a
        copy. Use GFN2 for solvent.
        """
        self.minimize()
        if not self.ids:
            return self._derive([])
        from .calculators import resolve

        q = self._calc_charge(charge)
        calc = resolve(refine, solvent, q)
        if calc is None:
            raise ValueError(
                "optimize needs a real calculator (refine='gxtb' or 'gfn2'); the force-field relaxation is minimize()"
            )
        fix = sorted(self.cons.frozen)  # the frozen TS core (reacting atoms)
        new_mol = Chem.Mol(self._mol)  # opt moves atoms, so work on a copy and never clobber self
        energies, kept = {}, []
        for i in self.ids:
            try:
                coords, e = calc.optimize(self._mol, i, level, fix)
                new_mol.GetConformer(i).SetPositions(np.asarray(coords, float))
                energies[i] = e * _HARTREE_KCAL
                kept.append(i)
            except (RuntimeError, OSError) as err:  # an xtb run failure drops this conformer; a config error
                # (bad level/solvent) is a ValueError and propagates instead
                logger.warning(
                    "optimize: %s --opt failed on conformer %d (%s); dropping it", refine, i, _last_line(err)
                )
        if not kept:
            raise RuntimeError(
                f"optimize: {refine} --opt produced nothing. Is the xtb binary on PATH ($XTB_EXE, or ~/bin/xtb)?"
            )
        out = self._derive(kept, new_mol)
        out.energies = energies
        out._remove(out._report_missed_fixes(out.ids, f"optimize[{refine}]"))
        kept = out.ids
        if not kept:
            raise RuntimeError(f"optimize: {refine} moved every numeric fix outside its accepted tolerance")
        logger.info(
            "optimize: %s --opt %s on %d conformer(s); %d core atom(s) held fixed", refine, level, len(kept), len(fix)
        )
        # The opt moved every free atom, so it may have returned a different molecule; check here since the
        # minimized stage makes every downstream minimize() (including prune's) a no-op.
        changed = self._flag_connectivity(new_mol, kept, f"optimize[{refine}]", charge)
        out._stage = _MINIMIZED
        out.discarded += [i for i in self.ids if i not in set(kept)]
        out.energy_kind = "real"  # geometry-optimised xtb/g-xTB energies, comparable across species
        out.reacted = changed  # flagged, not dropped: .filter('connectivity') drops them
        return out

    def relative(self, unit="kcal"):
        """Return ``{conformer_id: E - E_min}`` in kcal/mol or Hartree."""
        have = {i: self.energies[i] for i in self.ids if i in self.energies}
        if not have:
            return {}
        emin = min(have.values())
        scale = 1.0 if unit == "kcal" else 1.0 / _HARTREE_KCAL
        return {i: (e - emin) * scale for i, e in have.items()}

    def check(self, **kwargs):
        """Run the geometry gate on every tracked conformer; return ``{conformer_id: report}``.

        The stated metal donors are supplied unless ``donors=`` is passed.
        """
        if "donors" not in kwargs:
            donors = {int(d) for ds in self.sphere.values() for d in ds}
            if self.iso is not None:
                donors.update(int(d) for d in self.iso.donors)
            if donors:
                kwargs["donors"] = sorted(donors)
        mol = self.mol
        return {cid: _geometry.check(mol, cid, **kwargs) for cid in self.ids}

    def _scan_connectivity(self, mol=None, ids=None, charge=None):
        """Return formed and broken bonds for conformers whose graph changed.

        Metal dative pairs are checked as a coordination sphere rather than by covalent radii.
        """
        mol = self._mol if mol is None else mol
        ids = self.ids if ids is None else ids
        iso = self.iso
        metals, elements, spheres = frozenset(), None, dict(self.sphere)
        if iso is not None:  # pre-minimize: the mol still carries the surrogate, so name the real elements
            metals = {iso.metal, *(mi for mi, _rz, _rq in iso.spectator_metals)}
            elements = {iso.metal: iso.real_z, **{mi: rz for mi, rz, _rq in iso.spectator_metals}}
            if iso.donors:
                spheres.setdefault(iso.metal, list(iso.donors))
        else:  # post-minimize: the real elements are back, and `sphere` is what remembers the coordination
            metals = _metrics._metal_indices(mol)
        # Not _calc_charge: it warns about a metal's perceived charge and this runs on every stage. Perception
        # here is connectivity-only (no bond orders), which the total charge barely moves.
        q = Chem.GetFormalCharge(self._mol) if charge is None else charge
        intended = {frozenset(pair) for pair in self.cons.distances}
        out = {}
        for i in ids:
            formed, broken = _metrics.connectivity(
                mol, i, exclude=self.cons.frozen, metals=metals, charge=q, elements=elements
            )
            formed = [pair for pair in formed if frozenset(pair) not in intended]
            broken = [pair for pair in broken if frozenset(pair) not in intended]
            for m, donors in spheres.items():  # the metal's own check: a dative bond has no covalent yardstick,
                if not donors:  # so the coordination sphere is compared as a set instead
                    continue
                left, joined = _metrics.coordination_changed(
                    mol, i, m, donors, elements=elements, exclude=self.cons.frozen
                )
                formed = formed + [(m, a) for a in joined]
                broken = broken + [(m, d) for d in left]
            if formed or broken:
                out[i] = (formed, broken)
        return out

    def _flag_connectivity(self, mol, ids, stage, charge=None):
        """Warn, in chemistry, for every conformer a stage just turned into a different molecule."""
        changed = self._scan_connectivity(mol, ids, charge)
        for i, (formed, broken) in changed.items():
            logger.warning(
                "%s: conformer %d changed connectivity (%s); drop with .filter('connectivity')",
                stage,
                i,
                _metrics.describe(mol, formed, broken),
            )
        return changed

    def filter(self, by="connectivity", *, charge=None, **gate):
        """Drop conformers that fail the geometry or connectivity gate, in place.

        Geometry options are passed to `check()`, with the carried frozen core used by default. Carried
        distance constraints remain biases unless explicitly passed as ``constraints=``. Dropping every
        conformer raises.
        """
        if by == "geometry":
            if charge is not None:
                raise TypeError("filter('geometry') does not take charge=")
            unknown = set(gate) - {"frozen", "reference", "constraints", "donors"}
            if unknown:
                raise TypeError(f"filter('geometry') got unexpected option(s): {', '.join(sorted(unknown))}")
            gate.setdefault("frozen", self.cons.frozen)
            reports = self.check(**gate)
            if not reports:
                return self
            kept = [i for i, report in reports.items() if report.ok()]
            if not kept:
                raise RuntimeError(f"filter[geometry]: all {len(self.ids)} conformer(s) failed; inspect .check()")
            dropped = [i for i in self.ids if i not in set(kept)]
            logger.info("filter[geometry]: %d -> %d (dropped %d)", len(self.ids), len(kept), len(dropped))
            self.discarded += dropped
            self._remove(dropped)
            return self
        if by != "connectivity":
            raise ValueError(f"filter(by={by!r}): use 'geometry' or 'connectivity' (geometric dedup is prune())")
        if gate:
            raise TypeError(f"filter('connectivity') got geometry option(s): {', '.join(gate)}")
        changed = self._scan_connectivity(charge=charge)
        if not changed:
            logger.info("filter[connectivity]: %d conformer(s), all intact", len(self.ids))
            return self
        kept = [i for i in self.ids if i not in changed]
        if not kept:
            raise RuntimeError(
                f"filter[connectivity]: all {len(self.ids)} conformer(s) changed connectivity, so every one is "
                "a different species than the input. That is a result, not a filter failure: the geometry or "
                "the level of theory is reacting your molecule."
            )
        for i, (formed, broken) in changed.items():
            logger.info("filter[connectivity]: dropping #%d, %s", i, _metrics.describe(self._mol, formed, broken))
        logger.info("filter[connectivity]: %d -> %d (dropped %d reacted)", len(self.ids), len(kept), len(changed))
        self.discarded += list(changed)
        self._remove(changed)
        self.reacted = {**self.reacted, **changed}
        return self

    def prune(
        self,
        by="auto",
        *,
        energy_window=12.0,
        max_dist=0.75,
        moi_dev=0.01,
        max_rmsd=0.5,
        energy_tol=0.05,
    ):
        """Deduplicate by geometry, in place.

        The implicit ``minimize()`` settles raw seeds first. ``by`` accepts ``'rmsd'`` (the ``'auto'`` default),
        ``'moi'``, ``'descriptor'``, ``'energy'`` or a sequence. MOI can merge distinct multi-fragment poses.
        Discarded conformers remain available to ``duplicates()``.
        """
        self.minimize()
        if not self.ids:  # a geometrically-infeasible isomer minimised to empty
            return self
        from .select import _metal_present

        # a metal complex looks multi-fragment only because the surrogate stripped its coordinate bonds, and
        # those "fragments" are one molecule, so moi is a fine choice there
        n_frag = 1 if _metal_present(self._mol) else len(Chem.GetMolFrags(self._mol))
        if by == "auto":
            by = "rmsd"  # rigorous everywhere; moi/cascade are opt-in for speed
        methods = [by] if isinstance(by, str) else list(by)
        if n_frag > 1 and "moi" in methods:
            logger.warning(
                "prune[moi] on %d fragments: may over-merge distinct poses; prefer 'rmsd'",
                n_frag,
            )
        for method in methods:
            if method == "connectivity":  # a validity filter rather than a dedup, but it composes in a cascade
                self.filter("connectivity")  # prune(by=['connectivity', 'rmsd']): drop reacted, then dedup
                continue
            before = list(self.ids)
            energies = [self.energies.get(i, float("inf")) for i in self.ids]  # no energy -> outside any window,
            # never merged into a phantom 0.0-kcal band nor kept over a real-energy duplicate
            kept, _ = _dedup.apply(
                self._mol,
                self.ids,
                energies,
                method=method,
                energy_window=energy_window,
                max_dist=max_dist,
                moi_dev=moi_dev,
                max_rmsd=max_rmsd,
                energy_tol=energy_tol,
            )
            dropped = [i for i in before if i not in set(kept)]
            self.discarded += dropped
            self._remove(dropped)
            logger.info(
                "prune[%s]: %d -> %d  (merged %d near-duplicates; see .duplicates())",
                method,
                len(before),
                len(self.ids),
                len(dropped),
            )
            if dropped and logger.isEnabledFor(logging.DEBUG):  # spell out which absorbed which
                for k, lst in self._group_duplicates(self.ids, dropped).items():
                    logger.debug("  #%d absorbed %s", k, ", ".join(f"#{d} ({r:.2f} A)" for d, r in lst))
        return self

    def _group_duplicates(self, kept, dropped):
        groups = {}
        for d, (k, r) in _dedup.nearest_kept(self._mol, kept, dropped).items():
            groups.setdefault(k, []).append((d, r))
        return {k: sorted(v) for k, v in sorted(groups.items())}

    def duplicates(self):
        """Group discarded conformers by the nearest kept conformer and RMSD."""
        return self._group_duplicates(self.ids, self.discarded) if self.discarded else {}

    def binding_modes(self):
        """Count the inter-fragment NCI binding-mode signatures sampled across the conformers.

        An empty signature means the pose has no inter-fragment contact.
        """
        from collections import Counter

        from rxembed.metal_core import _frag_map

        from .select import _interfragment_contacts

        an = _nci.analyzer(self._mol)
        fmap = _frag_map(self._mol)

        def sig(c):
            contacts = _interfragment_contacts(an, self._mol.GetConformer(c).GetPositions(), fmap)
            return tuple(sorted({t for t, _a, _p in contacts}))

        return Counter(sig(c) for c in self.ids)

    def cluster(self, *, min_cluster=3, reduce=None, nci=True):
        """Return each conformer's binding-mode cluster label; -1 marks noise."""
        self.minimize()
        if not self.ids:
            return np.array([], dtype=int)
        return _dedup.cluster_labels(self._mol, self.ids, min_cluster=min_cluster, reduce=reduce, nci=nci)

    def landscape(self, method="pca", color="cluster", *, reduce=None, min_cluster=3, nci=True):
        """2D ensemble map (method='pca'|'tsne'), coloured by 'cluster' or 'energy'."""
        from . import viz

        self.minimize()  # ensure a latent (and, for color='energy', real energies) exist; mirrors cluster()
        return viz.landscape(self, color=color, method=method, reduce=reduce, min_cluster=min_cluster, nci=nci)

    # -- deriving a smaller / aligned ensemble (returns a new Ensemble) --------

    def _derive(self, ids, mol=None):
        """Return an independent Ensemble over a subset while preserving its state and metal records."""
        ids = list(ids)
        selected = set(ids)
        same_coordinates = mol is None
        mol = mol if mol is not None else Chem.Mol(self._mol)
        return Ensemble(
            mol,
            ids,
            self.cons,
            self.iso,
            {i: self.energies[i] for i in ids if i in self.energies},
            unrelaxed=[i for i in self.unrelaxed if i in selected],
            seed=self.seed,
            _stage=self._stage,
            discarded=list(self.discarded),
            tag=dict(self.tag),
            _stereo=self._stereo,
            _donor_hand=dict(self._donor_hand),
            energy_kind=self.energy_kind,
            reacted={i: v for i, v in self.reacted.items() if i in selected},
            sphere=dict(self.sphere),
            trajectory=(
                Chem.Mol(self.trajectory)
                if same_coordinates and self.trajectory is not None and ids == self.ids
                else None
            ),
        )

    def select_stereo(self, like, spec="preserve"):
        """Keep only conformers whose chirality matches `spec` against a reference `like`.

        ``like`` carries the reference geometry. ``spec`` accepts ``'preserve'``, ``'free'``, ``'invert'`` or
        a mapping from chirality kind to mode. This handles planar, axial and helical chirality not encoded in
        the RDKit graph.
        """
        if self._stage != _MINIMIZED:
            self._stereo = None  # an explicit select_stereo overrides the embed default, so
        self.minimize()  # select_stereo(.., 'free') still sees every pose
        ref = _stereo.signature(like) if isinstance(like, Chem.Mol) else _stereo.signature(*like)
        keep = [i for i in self.ids if _stereo.satisfies_spec(_stereo.signature(self._mol, i), ref, spec)]
        logger.info("select_stereo[%s]: %d -> %d (chirality-matched)", spec, len(self.ids), len(keep))
        return self._derive(keep)

    def lowest(self, n=1, refine=None, solvent=None):
        """Return the `n` lowest-energy conformers as a new Ensemble (n=1 -> the single best geometry).

        Pass ``refine='gxtb'`` or ``'gfn2'`` to rank by xTB instead of the force field.
        """
        src = self.score(refine, solvent) if refine else self
        src.minimize()
        # +inf, not 0.0: a conformer whose relax was skipped (untypable core) has no energy and must sort
        # last. Ranking it 0.0 against a real -250 kcal/mol would pick the phantom as "best".
        return src._derive(sorted(src.ids, key=lambda i: src.energies.get(i, float("inf")))[:n])

    def representatives(self, *, min_cluster=3, nci=True, recover_noise="auto", noise_window=10.0):
        """Return the lowest-energy conformer from each sampled mode.

        ``recover_noise='auto'`` keeps noise only when it represents a new mode within ``noise_window``
        kcal/mol. ``True`` keeps all noise and ``False`` drops it. The source ensemble is unchanged.
        """
        self.minimize()
        if not self.ids:  # infeasible isomer / nothing embedded -> empty summary
            return self._derive([])
        kind = _dedup.mode_kind(self._mol, self.ids, nci=nci)
        labels = _dedup.cluster_labels(self._mol, self.ids, min_cluster=min_cluster, nci=nci).tolist()

        def by_e(i):
            return self.energies.get(i, float("inf"))  # no energy (untypable core) -> sorts last, never "lowest"

        e_min = min(by_e(i) for i in self.ids)
        mode_reps = [
            min((self.ids[k] for k in range(len(labels)) if labels[k] == lab), key=by_e)
            for lab in sorted({lab for lab in labels if lab != -1})
        ]
        noise_k = [k for k in range(len(labels)) if labels[k] == -1]
        recovered, skipped_e = [], 0
        if noise_k and recover_noise is not False:
            sigs = None if recover_noise is True else _dedup.mode_signature(self._mol, self.ids, nci=nci)
            if recover_noise is True:
                recovered = [self.ids[k] for k in noise_k]
            elif sigs is not None:  # one rep per noise signature not already clustered
                shown = {sigs[k] for k in range(len(labels)) if labels[k] != -1}  # all members, not just reps
                fresh = {}
                for k in noise_k:
                    s, i = sigs[k], self.ids[k]
                    if s in shown:
                        continue
                    if by_e(i) - e_min > noise_window:  # ignore high-energy scatter
                        skipped_e += 1
                        continue
                    if s not in fresh or by_e(i) < by_e(fresh[s]):
                        fresh[s] = i
                recovered = list(fresh.values())
        folded = len(noise_k) - len(recovered) - skipped_e
        note = f" + {len(recovered)} rare binding mode(s) recovered" if recovered else ""
        if skipped_e:
            note += f"; {skipped_e} skipped (> {noise_window:g} kcal/mol)"
        if folded > 0:
            note += f"; {folded} scatter conformer(s) folded (still in ens.ids)"
        logger.info("representatives: %d %s mode(s)%s", len(mode_reps), kind, note)
        chosen = mode_reps + recovered
        return self._derive(sorted(chosen, key=by_e) or [min(self.ids, key=by_e)])

    def _align_atoms(self, on):
        if on is None:  # default core: constrained atoms > metal coordination > heavy
            core = sorted(self.cons.constrained_atoms())
            if len(core) >= _MIN_OVERLAY_ATOMS:
                return core
            from .select import _metal_donors, _metal_present

            if _metal_present(self._mol):  # a metal's natural overlay core is M + its donors, even when cons
                m, donors = _metal_donors(self._mol, self.ids)  # is empty (e.g. a wrapped metal mol)
                if donors:
                    return [m, *donors]
            return [a.GetIdx() for a in self._mol.GetAtoms() if a.GetAtomicNum() > 1]
        if isinstance(on, str):
            return list(match(self._mol, on))
        return [resolve_atom(self._mol, a) for a in on]

    def align(self, on=None):
        """Return a new Ensemble with every conformer Kabsch-superposed for a readable overlay.

        Align on the constrained core by default, or pass atom indices or SMARTS with ``on=``.
        """
        m = Chem.Mol(self._mol)
        if len(self.ids) > 1:
            rdMolAlign.AlignMolConformers(m, atomIds=self._align_atoms(on), confIds=list(self.ids))
        return self._derive(self.ids, mol=m)

    # -- looking (returns an artifact, never mutates) -------------------------
    # 3D rendering is notebook-level (align()/dump() plus a few lines of py3Dmol/xyzrender). Only `landscape`
    # lives here, because the dim-reduction is real reusable work.

    @property
    def n(self):
        """Return the number of tracked conformers."""
        return len(self.ids)

    def __getitem__(self, key):
        """Return selected conformers as an independent Ensemble."""
        sel = self.ids[key]
        return self._derive(sel if isinstance(sel, list) else [sel])

    def dump(self, path, align=True):
        """Write tracked conformers as a multi-frame XYZ and return its path.

        Real metal elements are restored. Frames are core-aligned by default; pass ``align=False`` for raw
        coordinates. The live ensemble is unchanged.
        """
        if not self.ids:  # `Conformers.dump`'s guard: a 0-byte file that reads as a successful write is the
            raise ValueError(  # worst possible outcome, and minimize() can drop every conformer
                "nothing to dump: this ensemble has no conformers (the embed produced none, or a stage "
                "dropped them all); check the log for what was discarded"
            )
        mol = Chem.Mol(self._mol)  # the ensemble mol is already real (no haptic centroid dummy); work on a copy
        if self.iso is not None:  # show the real metal(s) with their formal charge, not the C surrogate
            self.iso.restore(mol)
        if align and len(self.ids) > 1:  # overlay frames on the rigid core
            try:
                aln = self._align_atoms(None)
                rdMolAlign.AlignMolConformers(mol, atomIds=aln, confIds=list(self.ids))
            except Exception as e:
                logger.debug("dump: alignment skipped (%s)", e)
        with open(path, "w") as f:
            for i in self.ids:
                f.write(Chem.MolToXYZBlock(mol, confId=i))
        return path

    def __repr__(self):
        """Summarise the ensemble: conformer count, state, and energy spread."""
        n = len(self.ids)
        es = [self.energies[i] for i in self.ids if i in self.energies]
        de = f", ΔE 0.00-{max(es) - min(es):.2f} kcal/mol" if len(es) > 1 else ""
        state = self._stage
        tag = f", {self.tag}" if self.tag else ""
        return f"<Ensemble: {n} conformer{'s' if n != 1 else ''}, {state}{de}{tag}>"
