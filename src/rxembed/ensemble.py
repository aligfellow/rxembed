"""Hold what the verbs return: conformer ensembles and their sets, `rx.isomers`' species and co-embedded pairs."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import cache, cached_property, partial
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from rxembed.engine.acceptance import Failure, Gate, describe_groups, group_failures, shape_report
from rxembed.engine.coembed import pair_score
from rxembed.engine.conformers import BASE_STIFFNESS, Conformers
from rxembed.geometry.bounds import bounds_matrix, fragment_contacts
from rxembed.geometry.constraints import SoftKeys, is_index, match, resolve_atom
from rxembed.geometry.relax import MAX_ITERS, Coupling, UFFRecord, restrained_uff
from rxembed.graph import metal_index
from rxembed.identity.reading import named, named_species, requested_geometry
from rxembed.identity.smiles_io import mol_from_species
from rxembed.identity.species import SELECTORS, Species, matches, site_details
from rxembed.metal.centres import canonical_metal_graph
from rxembed.perception.changes import bond_changes, mismatch, signature, specified_like
from rxembed.perception.contacts import observed_contacts
from rxembed.stereo import defined_stereo_label, without_stale_point_tags
from rxembed.utils import require_mol, with_conformers
from rxembed.workflow import geom_check, search, viz
from rxembed.workflow.calculators import resolve
from rxembed.workflow.select import (
    apply,
    cluster_on,
    compare,
    features,
    metal_donors,
    mode_signature,
    nearest_kept,
)

logger = logging.getLogger("rxembed")

_MIN_OVERLAY_ATOMS = 3  # the fewest atoms that define an alignment frame
_HARTREE_KCAL = 627.5094740631  # Eh -> kcal/mol
_NOISE_WINDOW = 10.0  # kcal/mol
_SETTLE_BINS = 5  # seed groups `mc` settles at their own point across the windows
SEEDED, RELAXED, MINIMIZED, FINAL = "seeded", "relaxed", "minimized", "final"
# The energy kinds a force field writes (`minimize`, `score("ff")`); they do not compare distinct species.
_FORCE_FIELDS = frozenset(("UFF", "restrained UFF", "surrogate UFF", "restrained surrogate UFF"))
_SLUG_CHARS = 60  # characters of a member's name a dump file name keeps
_MAPPED = ("mc", "minimize", "prune", "score", "optimize", "lowest", "representatives")  # EnsembleSet maps these


def _calculated(stage, calc, compute, ids):
    """Return ``{id: compute(id)}`` over `ids`, warning of and dropping each conformer `calc` fails on.

    A RuntimeError or a non-finite energy fails one conformer; any other error raises. When every conformer fails, a
    RuntimeError names the calculator.
    """
    found = {}
    for cid in ids:
        try:
            result = compute(cid)
            energy = result if stage == "score" else result[1]
            if np.isfinite(energy):
                found[cid] = result
                continue
            message = "calculator returned a non-finite energy"
        except NotImplementedError:
            raise
        except RuntimeError as err:
            message = (str(err).strip().splitlines()[-1:] or ["no output"])[0]
        logger.warning("%s: %s failed on conformer %d (%s); dropping it", stage, calc.name, cid, message)
    if not found:
        raise RuntimeError(f"{stage}: {calc.name} failed on all {len(ids)} conformer(s); the warnings name each error")
    return found


def _physical_defect(report):
    """Return the `physical_geometry` `Failure` a `geom_check` report gives: its first defect, else None."""
    return Failure("physical_geometry", f"physical geometry: {report.defects[0]}") if report.defects else None


def _settled(windows, frac, pad):
    """Return each ``(key, (lo, hi))`` window narrowed to ``pad`` about its ``frac`` point, clamped inside it."""
    return {k: (max(lo, lo + frac * (hi - lo) - pad), min(hi, lo + frac * (hi - lo) + pad)) for k, (lo, hi) in windows}


def species_name(named):
    """Return the name a set gives a species: its centres' row, else its ligand stereo, else its CX name."""
    return str(named.species) or str(named.species.stereo) or named.name


def _label(ens):
    """Return the name an `EnsembleSet` shows for `ens`: its species as names read it, then its ``nci`` mode."""
    return " ".join(filter(None, (species_name(ens._named), ens.tag.get("nci"))))


def _picked(members, record, index, keys, names=SELECTORS):
    """Return the `members` that `index` and every name in `keys` pick.

    `index` is a position, None for every member. Each other key is a `species.SELECTORS` name matched on member k's
    species (`record(k)`), with ``geometry`` a polyhedron name or code. An unknown key, an index out of range or an
    unknown polyhedron raises before any member is read; `names` lists the keys the set takes.
    """
    if unknown := keys.keys() - SELECTORS:
        raise TypeError(f"unknown selector(s) {sorted(unknown)}; select by {sorted(names)} or index")
    if index is not None and not is_index(index):
        raise TypeError(f"index selects one member by its int position; got {index!r}")
    if index is not None and not -len(members) <= index < len(members):
        raise IndexError(f"index {index} is out of range for {len(members)} member(s)")
    if keys.get("geometry") is not None:
        geometry = requested_geometry(keys["geometry"])
        if len(geometry) != 1:
            raise TypeError(f"geometry selects by one polyhedron name or code; got {keys['geometry']!r}")
        keys = {**keys, "geometry": geometry[0]}
    return [
        member
        for k, member in enumerate(members)
        if index in (None, k, k - len(members)) and (not keys or matches(record(k).species, keys))
    ]


class EnsembleSet(list):
    """Hold distinct candidate ensembles, each named by its species and, per contact mode, its ``nci`` tag.

    `mc`, `minimize`, `prune`, `score`, `optimize`, `lowest`, `representatives` and `filter(by=)` run within each
    candidate and return an `EnsembleSet`, so energies stay per species. `errors` holds the `EmbeddingError` of each
    candidate `rx.embed` could not build.
    """

    def __init__(self, members=(), errors=()):
        super().__init__(members)
        self.errors = tuple(errors)

    def __getitem__(self, key):
        """Return one candidate, or an `EnsembleSet` keeping `errors` for a slice."""
        got = super().__getitem__(key)
        return EnsembleSet(got, self.errors) if isinstance(key, slice) else got

    def __add__(self, other):
        """Return this set's candidates, then those of `other`, an `EnsembleSet`, keeping both sets' `errors`."""
        if type(other) is not EnsembleSet:
            return NotImplemented
        return EnsembleSet((*self, *other), (*self.errors, *other.errors))

    def __repr__(self):
        """Summarise the candidates and their names (`_label`) on one line."""
        return f"<EnsembleSet: {len(self)} candidate{'s' if len(self) != 1 else ''} {[_label(e) for e in self]}>"

    def select(self, **keys):
        """Return the one candidate matching every key; raise naming the choices when none or several match."""
        hits = self.filter(**keys)
        if len(hits) != 1:
            raise ValueError(f"select({keys}) matched {len(hits)} candidates; have:\n{self.summary()}")
        return hits[0]

    def filter(self, by=None, **keys):
        """Return the candidates matching every key, or with `by` filter each candidate's conformers.

        ``nci`` matches a candidate's contact mode, ``index`` a position, and every other key a species name, so
        ``center=`` names one metal of several; an unknown key raises. The result keeps `errors`.
        """
        if by is not None:
            return self._map("filter", by, **keys)
        index, nci = keys.pop("index", None), keys.pop("nci", None)
        picked = _picked(self, lambda k: self[k]._named, index, keys, SELECTORS | {"nci"})
        return EnsembleSet((e for e in picked if nci in (None, e.tag.get("nci"))), self.errors)

    def summary(self):
        """Return one line per member (index, name as `_label` gives it, conformer count) to pick from."""
        return "\n".join(f"  [{k}] {_label(e)}  ({len(e.ids)} conformers)" for k, e in enumerate(self))

    def _map(self, method, *args, **kw):
        """Apply one verb within each candidate; never pool or cross-prune distinct species."""
        return EnsembleSet((getattr(member, method)(*args, **kw) for member in self), errors=self.errors)

    def best(self):
        """Return the candidate with the lowest energy under one model.

        Energies must be given (`wrap`) or a calculator's: a force field cannot compare distinct species, and nor can
        two models. A candidate a stage emptied ranks last.
        """
        kinds = {e.energy_kind for e in self if e.ids}
        if len(kinds) != 1 or not kinds.isdisjoint({"", *_FORCE_FIELDS}):
            raise ValueError(
                f"best() compares real energies from one model, not {sorted(kinds) or 'none'}; run .score('gxtb') first"
            )

        def emin(e):
            return min(map(e._rank, e.ids), default=math.inf)

        ranked = sorted(self, key=emin)
        lo = emin(ranked[0])
        order = ", ".join(f"{_label(e)} (+{emin(e) - lo:.1f})" for e in ranked)
        logger.info("best: ranked %d by ΔE (kcal/mol): %s", len(self), order)
        return ranked[0]

    def __getattr__(self, name):
        """Map a verb in `_MAPPED` over the candidates, or explain why another Ensemble attribute has no answer."""
        if name in _MAPPED:
            return partial(self._map, name)
        # annotations too: `Conformers.ids` / `.energies` are bare class-level annotations, not attributes
        known = hasattr(Ensemble, name) or any(name in getattr(k, "__annotations__", {}) for k in Ensemble.__mro__)
        if name.startswith("_") or not known:
            raise AttributeError(name)
        raise AttributeError(
            f"{name!r} needs one Ensemble; this set holds {len(self)}: {[_label(e) for e in self]}; pick one "
            "with [k] or .select()"
        )

    def dump(self, path, align=True):
        """Write one multi-frame XYZ per candidate, named by its index and `_label`; return their paths."""
        base = Path(path)
        paths = []
        for k, e in enumerate(self):
            # an ASCII name cut to a length any file system takes; the index keeps two cut names apart
            slug = re.sub(r"[^A-Za-z0-9.-]+", "_", _label(e))[:_SLUG_CHARS].strip("_-")
            p = base.with_name("_".join(filter(None, (base.stem, str(k), slug))) + (base.suffix or ".xyz"))
            e.dump(str(p), align=align)
            paths.append(str(p))
        return paths


class SpeciesSet(tuple):
    """Hold the species Mols a source can be, in enumeration order; index it, select one, or filter several.

    Select by the names `species.matches` reads (an organic has only ``stereo``), or by ``index``.
    """

    @classmethod
    def of_isomers(cls, isomers):
        """Return the species Mols of internal `Species` candidates."""
        return cls(map(mol_from_species, isomers))

    @property
    def records(self):
        """Read each member's current species as names read it (`reading.named`), in member order."""
        return tuple(named(with_conformers(mol, mol, ())) for mol in self)

    def __getitem__(self, key):
        """Return one species, or a `SpeciesSet` for a slice."""
        got = super().__getitem__(key)
        return SpeciesSet(got) if isinstance(key, slice) else got

    def __add__(self, other):
        """Return this set's species, then those of `other`, a `SpeciesSet`, as one `SpeciesSet`; else a TypeError."""
        if type(other) is not SpeciesSet:
            return NotImplemented
        return SpeciesSet((*self, *other))

    def filter(self, *, index=None, **keys):
        """Return the species at `index` matching every name in `keys` (`_picked`), as a `SpeciesSet`."""
        records = cache(lambda: self.records)
        return SpeciesSet(_picked(self, lambda k: records()[k], index, keys))

    def select(self, **keys):
        """Return the one species matching every key; raise naming the choices when none or several match."""
        hits = self.filter(**keys)
        if len(hits) != 1:
            raise ValueError(f"select({keys}) matched {len(hits)} species; have:\n{self.summary()}")
        return hits[0]

    def summary(self, details=False):
        """Return one line per species, its index and identity; `details` adds each metal's site relations."""
        lines = []
        for k, record in enumerate(self.records):
            lines.append(f"[{k:>2}] {species_name(record)}")
            lines.extend(site_details(record.species) if details else ())
        return "\n".join(lines)


@dataclass
class Ensemble(Conformers):
    """Add search, selection and scoring verbs to a `Conformers` result.

    Mutating verbs return `self`; selections and real-energy calculations return a new Ensemble. `.mol` returns a copy.
    """

    _stage: str = SEEDED
    discarded: list = field(default_factory=list)  # ids dropped by a stage; viz draws them faded
    tag: dict = field(default_factory=dict)  # {"nci": label} for a candidate of one contact mode, else empty
    _stereo_filter: tuple | None = None  # (StereoRequest, reference signature) for the minimize() chirality filter
    # the model that made `energies`: "given", a force field (`_FORCE_FIELDS`) or a calculator; "" for none
    energy_kind: str = ""

    def _coordinates_changed(self):
        """Forget what was read on the old coordinates, the stage's energies included."""
        super()._coordinates_changed()
        self._record(SEEDED, {}, "")

    def _record(self, stage, energies, kind, dropped=()):
        """Record a stage's result: the stage reached, its energies, the model that made them, the ids it dropped.

        The one writer of all four, so energies never sit under another model's name and every dropped conformer is
        listed in `discarded`; `minimize()` reads the stage to skip a relax already done or a final geometry.
        """
        self._remove(dropped)
        self._stage, self.energies, self.energy_kind = stage, energies, kind if energies else ""
        self.discarded = [*self.discarded, *dropped]

    def _rank(self, cid):
        """Return conformer `cid`'s energy as a ranking key; one without an energy (an untypable core) sorts last."""
        return self.energies.get(cid, math.inf)

    @cached_property
    def _named(self):
        """Return the known species' canonical name and seating once per Ensemble (`reading.named_species`)."""
        return named_species(self.species)

    def mc(self, *, preset="ensemble", seed=None, explore=False):
        """Run an openconf Monte-Carlo torsional search in place.

        `preset` sets the effort: 'rapid', 'ensemble', 'spectroscopic', 'docking', 'analogue', 'macrocycle' or
        'transition_metal', each with a metal move budget. `seed` reproduces a search; a preset leaves openconf's unset.

        Unconstrained, it replaces the ETKDG seeds; constrained, it searches from them with the held atoms frozen and
        adds what it finds. ``explore=True`` on a seeded NCI complex runs a second pass with the contacts released and
        the structural holds kept, pools both, and keeps the released `cons` so a later stage cannot re-tighten the
        contacts; on a metal only substrate contacts release. A pass openconf cannot search warns and adds nothing.
        """
        if not search.available():
            raise ImportError("mc needs openconf; pip install 'rxembed[search]'")
        cfg = search.config(preset, seed)  # before anything moves: an unknown preset is the caller's error
        if preset == "rapid" and self.cons.is_constrained:  # a pose-frozen search moves rotors only
            logger.warning("mc: preset='rapid' under-samples a constrained system; use the default 'ensemble'")
        if self.cons.is_constrained:
            logger.info(
                "mc: %d constrained atom(s); pose-frozen rotor search",
                len(self.cons.constrained_atoms()),
            )

        work = Chem.Mol(self._mol)  # settled and searched apart from the ensemble
        record = self._settle_seeds(work)  # spread the seeds across their windows before openconf pose-freezes them
        added = self._openconf_search(work, self.cons, "", cfg)
        if added is None:
            return self
        self._mol, self.ids = work, [*self.ids, *added] if self.cons.is_constrained else added
        self._coordinates_changed()  # the settled seeds and every conformer openconf copied from one
        self._uff = record  # the settle's relax, read on these coordinates
        if not self.cons.is_constrained:
            logger.info("mc: %d conformers (openconf '%s' drove generation; replaced ETKDG seeds)", len(added), preset)
        else:
            logger.info(
                "mc: +%d (openconf '%s', pose-constrained around seeds; total %d)", len(added), preset, len(self.ids)
            )

        if explore and self.cons.contacts:
            relaxed = self.cons.relaxed()
            # openconf never sees the bounds that hold fragments together, so bound every inter-fragment pair here
            if len(Chem.GetMolFrags(self._mol)) > 1:
                bounds = fragment_contacts(self._mol, relaxed, bounds_matrix(self._mol)).items()
                kept = {k: v for k, v in bounds if k not in relaxed.distances}
                relaxed = replace(relaxed, distances={**relaxed.distances, **kept})
            more = self._openconf_search(self._mol, relaxed, " explore", cfg)
            if more:
                self.ids += more
                self.cons = relaxed
                logger.info(
                    "mc explore: +%d conformers; contacts released, %d holds kept, %d total",
                    len(more),
                    len(relaxed.distances),
                    len(self.ids),
                )
        return self

    def _openconf_search(self, mol, cons, label, cfg):
        """Return the ids one openconf search under `cons` adds to `mol`; warn and return ``None`` when it cannot."""
        try:
            return search.search(mol, cons, cfg)
        except Exception as e:  # openconf can't handle every system (e.g. a TS hypervalent core)
            logger.warning(
                "mc%s: openconf could not search this system (%s: %s); the %d conformer(s) are left as they were",
                label,
                type(e).__name__,
                e,
                len(self.ids),
            )
            return None

    def _settle_seeds(self, mol):
        """Spread the seeds of `mol` across the non-frozen constraint windows before a search; return the record.

        The seeds split into up to `_SETTLE_BINS` groups, each settled at its own fraction across every window.
        """
        cons, record = self.cons, UFFRecord()
        seeded_d = [(k, v) for k, v in cons.distances.items() if not (k[0] in cons.frozen and k[1] in cons.frozen)]
        shape_d = {k: v for k, v in cons.distances.items() if k[0] in cons.frozen and k[1] in cons.frozen}
        if not (seeded_d or cons.angles or cons.dihedrals) or not self.ids:
            return record
        bins = min(len(self.ids), _SETTLE_BINS)
        for b, group in enumerate(g for g in np.array_split(list(self.ids), bins) if len(g)):
            frac = (b + 0.5) / bins  # this bin's fraction across every window
            tgt = replace(
                cons,
                distances={**shape_d, **_settled(seeded_d, frac, 0.03)},  # keep the frozen-core shape exact
                angles=_settled(cons.angles.items(), frac, 2.0),
                dihedrals=_settled(cons.dihedrals.items(), frac, 2.0),
                contacts=SoftKeys(),
                dg_floors={},
            )
            try:  # settling is best-effort pre-conditioning; an RDKit BFGS divergence must not sink mc()
                restrained_uff(mol, tgt, stiffness=BASE_STIFFNESS, conf_ids=[int(i) for i in group], record=record)
            except RuntimeError:
                logger.debug("mc: seed-settle relax diverged on a group; keeping the raw seeds")
        return record

    def _connectivity_failure(self, cid):
        """Return the bonds conformer `cid` forms or breaks against its graph."""
        mol = self._restored_mol(cid)
        found = bond_changes(mol, cid, self.cons)
        return None if found.unchanged else Failure("connectivity", f"connectivity changes ({found.describe(mol)})")

    def _requested_stereo_failure(self, cid):
        """Return the requested stereo `stereo_filter` says conformer `cid` lost."""
        if not self._stereo_filter:
            return None
        spec, ref = self._stereo_filter
        detail = mismatch(signature(self._restored_mol(cid), cid), ref, spec)
        return Failure("requested_stereo", f"requested stereo lost ({detail})") if detail else None

    def _physical_failure(self, cid):
        """Return the first defect `check()` reads on conformer `cid` of a species with centres, else None.

        An organic is judged by the bonding row; `filter('geometry')` reads the full report on one when asked.
        """
        if not self.species.centres:
            return None
        mol = with_conformers(self.specification.mol, self._mol, [cid])  # `check()`'s default reading of one conformer
        report = geom_check.report(mol, cid, constraints=self.cons, donors=None, check_windows=False, specified=None)
        return _physical_defect(report)

    workflow_gates = (
        Gate("connectivity", _connectivity_failure, ladder=False, calculator="flag"),
        Gate(
            "requested_stereo",
            _requested_stereo_failure,
            ladder=False,
            required=("requested_stereo",),
            calculator="drop",
        ),
        Gate("physical_geometry", _physical_failure, ladder=False),
    )
    gates = (*Conformers.gates, *workflow_gates)

    def relax_into_windows(self, trajectory=False, max_iters=MAX_ITERS):
        """Relax constrained seeds into their windows and accept them, without an energy; unconstrained seeds stay."""
        if trajectory and not self.cons.is_constrained:
            raise ValueError("trajectory records restrained-UFF cleanup; this embed has no constraints to clean up")
        self.trajectory = None
        if not self.ids or not self.cons.is_constrained:
            return self
        before = list(self.ids)
        frames = [] if trajectory else None
        typed = self._relax(BASE_STIFFNESS, max_iters, "embed", frames)
        self._accept(BASE_STIFFNESS, max_iters, "embed", typed=typed, raise_if_empty=True)
        self._store_trajectory(frames)
        stage = RELAXED if self.ids and typed and not self.unrelaxed else SEEDED
        self._record(stage, {}, "", [cid for cid in before if cid not in self.ids])
        return self

    def minimize(self, stiffness=None, max_iters=MAX_ITERS):
        """Relax, accept and score the ensemble under UFF, in place.

        Constraints add their restraint terms, which the energies keep: `energy_kind` is "UFF" or "restrained UFF", with
        "surrogate" where UFF typed a stand-in. A structural failure takes a fresh-seed replacement, and rejected ids
        are listed in `discarded`. Omitted `stiffness` keeps the current strength; retries relax only conformers that
        did not converge. Supplied, externally optimized and coembedded final geometries stay unchanged.
        """
        if self._stage == FINAL:
            return self
        stiffness = self._stiffness if stiffness is None else stiffness
        same_objective = self._stage == MINIMIZED and stiffness == self._stiffness
        if same_objective and not self.unrelaxed:
            return self
        if not self.ids:  # nothing embedded (e.g. ETKDG could not place this graph)
            self._record(MINIMIZED, {}, "")
            return self
        before = list(self.ids)
        typed = True
        if self.cons.is_constrained and self._stage == RELAXED and stiffness == self._stiffness:
            self._accept(stiffness, max_iters)  # the embed already relaxed into its windows: accept and score
            self._rescore_restrained(stiffness)
        else:
            self._stage, self.trajectory = SEEDED, None
            ids = self.unrelaxed if same_objective else None  # the same objective retries only what did not converge
            typed = self._relax(stiffness, max_iters, ids=ids)
            self._accept(stiffness, max_iters, typed=typed)
            if typed and (self.cons.is_constrained or ids is not None):  # pieces relaxed apart: score them once
                self._rescore_restrained(stiffness, ids=None if self.cons.is_constrained else self.ids)
        if not self.ids:
            logger.warning(
                "minimize: 0 of %d conformer(s) survived acceptance; the arrangement may be infeasible",
                len(before),
            )
        if self.trajectory is not None:
            frames = [conf.GetPositions().copy() for conf in self.trajectory.GetConformers()]
            self._store_trajectory(frames)  # clear it if a later acceptance gate replaced the endpoint
        # accepted minimization; unconverged ids stay retryable and a typing refusal stays uncompleted
        dropped = [cid for cid in before if cid not in self.ids]
        self._record(
            MINIMIZED if typed else SEEDED, self.energies, self._uff_name if typed else self.energy_kind, dropped
        )
        return self

    @property
    def _uff_name(self):
        """Name the restrained-UFF objective and any private typing approximation."""
        held = "restrained " if self.cons.is_constrained else ""
        return f"{held}{'surrogate ' if self._uff.surrogates or self._uff.retyped else ''}UFF"

    def score(self, refine="gxtb", solvent=None):
        """Return a new Ensemble scored by a single-point energy in kcal/mol, after an implicit ``minimize()``.

        ``refine`` is 'gxtb' (the default), 'gfn2', 'gfnff', 'ff' or a calculator, whose name `energy_kind` takes; a
        conformer it fails on is dropped. Only an xtb spec takes an ALPB ``solvent``. 'ff' scores the minimization
        objective, restraints included, at its stiffness.
        """
        calc = resolve(refine, solvent)
        self.minimize()  # settle the periphery (a constrained core stays pinned)
        if not self.ids:
            return self._derive([])
        if calc is None:  # 'ff'/None -> single-point force field, geometry kept
            out = self._derive(self.ids)
            out._rescore_restrained(self._stiffness, ids=self.ids)
            out._record(self._stage, out.energies, out._uff_name)
            return out
        mol = self.mol
        found = _calculated("score", calc, partial(calc.energy, mol), self.ids)
        energies = {cid: float(e) * _HARTREE_KCAL for cid, e in found.items()}
        logger.info(
            "score: %s single point on %d conformer(s) (charge %d)", calc.name, len(energies), Chem.GetFormalCharge(mol)
        )
        out = self._derive(self.ids)
        out._record(self._stage, energies, calc.name, [i for i in self.ids if i not in energies])
        return out

    def optimize(self, refine="gxtb", level="normal"):
        """Return a new Ensemble of each conformer optimised by the calculator `refine` names, at `level`.

        ``level`` is the calculator's (xtb's ``--opt`` level, 'normal' by default). Frozen atoms stay fixed. Conformers
        that leave their species are dropped, and it raises when none survives. Energies are kcal/mol. Physical checks
        are opt-in with `check()` or `filter("geometry")`.
        """
        calc = resolve(refine)
        if calc is None:
            raise ValueError(
                "optimize needs a real calculator (refine='gxtb', 'gfn2' or 'gfnff'); the force-field relaxation is "
                "minimize()"
            )
        self.minimize()
        if not self.ids:
            return self._derive([])
        mol = self.mol
        fix = sorted(self.cons.frozen)  # the frozen TS core (reacting atoms)
        found = _calculated("optimize", calc, lambda cid: calc.optimize(mol, cid, level, fix), self.ids)
        new_mol = Chem.Mol(self._mol)
        for cid, (coords, _) in found.items():
            new_mol.GetConformer(cid).SetPositions(np.asarray(coords, float))
        energies = {cid: e * _HARTREE_KCAL for cid, (_, e) in found.items()}
        out = replace(self, _mol=new_mol, ids=list(energies), tag=dict(self.tag))
        out._coordinates_changed()
        failed = out._calculator_failures(f"optimize[{calc.name}]")
        if len(failed) == len(out.ids):
            raise RuntimeError(
                f"optimize: {calc.name} moved every conformer off the specified species or its request "
                f"({describe_groups(group_failures(failed.values()))}); embed the species instead"
            )
        logger.info(
            "optimize: %s at level %s on %d conformer(s); %d core atom(s) held fixed",
            calc.name,
            level,
            len(energies),
            len(fix),
        )
        lost = [cid for cid in self.ids if cid not in energies or cid in failed]
        out._record(FINAL, {cid: energies[cid] for cid in out.ids if cid not in failed}, calc.name, lost)
        return out

    def _calculator_failures(self, label):
        """Return ``{id: Failure}`` for the conformers a ``calculator="drop"`` gate fails on.

        A ``"flag"`` gate is read first: a conformer it fails is another molecule, warned and kept for `filter` to drop,
        and not judged against the species it left.
        """
        rows = {
            policy: [
                row._replace(check=row.calculator_check or row.check) for row in self.gates if row.calculator == policy
            ]
            for policy in ("drop", "flag")
        }
        flagged = {cid: reason for cid in self.ids if (reason := self._first_failure(cid, rows["flag"]))}
        failed = {
            cid: reason for cid in self.ids if cid not in flagged and (reason := self._first_failure(cid, rows["drop"]))
        }
        if failed:
            groups = describe_groups(group_failures(failed.values()))
            logger.warning("%s: dropped %d conformer(s): %s", label, len(failed), groups)
        if flagged:
            groups = describe_groups(group_failures(flagged.values()))
            logger.warning("%s: kept %d flagged conformer(s): %s; .filter() drops them", label, len(flagged), groups)
        return failed

    def relative(self):
        """Return ``{conformer_id: E - E_min}`` in kcal/mol."""
        have = {i: self.energies[i] for i in self.ids if i in self.energies}
        emin = min(have.values(), default=0.0)
        return {i: e - emin for i, e in have.items()}

    def check(self, **options):
        """Return ``{conformer_id: GeometryReport}`` for every tracked conformer, with its `shape_report`.

        By default a report treats this ensemble's constraints as ownership, without checking their windows, and holds
        its frozen core, so it fails where the `physical_geometry` gate does. ``constraints=`` an `rx.Constraints`
        checks that record's windows and core instead, and None drops the constraints but keeps the core. ``frozen=``
        names another core; ``reference=`` and ``donors=`` are as `geom_check.check` takes them.
        """
        if unknown := set(options) - {"frozen", "reference", "constraints", "donors", "check_windows"}:
            raise TypeError(f"unexpected geometry option(s): {', '.join(sorted(unknown))}")
        carried, given = "constraints" not in options, options.get("constraints")
        cons = self.cons if carried else geom_check.held(given)
        core = cons.frozen if given is not None else self.cons.frozen
        cons = replace(cons, frozen=frozenset(options.get("frozen", core) or ()))
        mol = with_conformers(self.specification.mol, self._mol, self.ids)
        reference = options.get("reference")
        specified = None if reference is None else specified_like(mol, reference)
        options = {"check_windows": not carried, "donors": None, **options}
        return {
            cid: replace(
                geom_check.report(
                    mol,
                    cid,
                    constraints=cons,
                    donors=options["donors"],
                    check_windows=options["check_windows"],
                    specified=specified,
                ),
                shape=shape_report(self._mol, cid, self.specification),
            )
            for cid in self.ids
        }

    def bond_changes(self):
        """Return ``{conformer id: Changes}``: the bonds each geometry forms and breaks against the graph."""
        mol = self.mol
        return {cid: bond_changes(mol, cid, self.cons) for cid in self.ids}

    def filter(self, by="connectivity", **options):
        """Drop, in place, the conformers that fail the `connectivity` gate, or whose `check()` report fails.

        'geometry' reads `check()` under its options on any species. Dropping every conformer leaves an empty Ensemble
        and logs a warning.
        """
        reports = {"geometry": "check()", "connectivity": "bond_changes()"}
        if by not in ("geometry", "connectivity"):
            names = " or ".join(map(repr, reports))
            raise ValueError(f"filter(by={by!r}): use {names} (geometric dedup is prune())")
        if by == "geometry":
            failed = {
                cid: reason for cid, report in self.check(**options).items() if (reason := _physical_defect(report))
            }
        elif options:
            raise TypeError(f"filter('connectivity') takes no options; got {', '.join(options)}")
        else:
            row = next(gate for gate in self.gates if gate.name == "connectivity")
            failed = {cid: reason for cid in self.ids if (reason := self._first_failure(cid, (row,)))}
        label = f"filter[{by}]"
        for cid, reason in failed.items():
            logger.info("%s: dropping #%d, %s", label, cid, reason)
        if failed and len(failed) == len(self.ids):
            logger.warning("%s: all %d conformer(s) failed; inspect .%s", label, len(failed), reports[by])
        logger.info("%s: %d -> %d (dropped %d)", label, len(self.ids), len(self.ids) - len(failed), len(failed))
        self._record(self._stage, self.energies, self.energy_kind, list(failed))
        return self

    def compare(self, reference, candidate, *, max_rmsd, hydrogens=False):
        """Compare two selected conformers without moving or minimizing them.

        Return None or a Comparison with RMSD, maximum displacement in angstrom and the candidate's atom order matching
        the reference, under prune's geometric gates without its energy window.
        """
        if reference not in self.ids or candidate not in self.ids:
            raise ValueError(f"choose reference and candidate conformer ids from {self.ids}")
        if not np.isfinite(max_rmsd) or max_rmsd <= 0:
            raise ValueError("max_rmsd must be finite and positive, in angstrom")
        return compare(self.mol, reference, candidate, max_rmsd=max_rmsd, hydrogens=hydrogens)

    def prune(self, by="rmsd", *, max_rmsd=0.5, hydrogens=False):
        """Deduplicate by geometry, in place, after an implicit ``minimize()``.

        ``by`` is 'rmsd' (the default), 'energy', 'connectivity' or a sequence run in turn; each keeps the lowest-energy
        copy. RMSD merges conformers within ``max_rmsd``, every atom within twice it, under graph symmetry and proper
        rotation. It leaves out ordinary hydrogens unless ``hydrogens=True``, which
        keeps hydroxyl rotamers apart, but always compares hydrides, and never moves the stored atoms. Dropped ids go to
        `discarded`.
        """
        self.minimize()
        if not self.ids:  # a geometrically infeasible species minimised to empty
            return self
        mol = self.mol
        methods = [by] if isinstance(by, str) else list(by)
        for method in methods:
            if method == "connectivity":
                self.filter("connectivity")
                continue
            before = list(self.ids)
            kept = apply(
                mol, self.ids, [self._rank(i) for i in self.ids], method=method, max_rmsd=max_rmsd, hydrogens=hydrogens
            )
            dropped = [i for i in before if i not in set(kept)]
            self._record(self._stage, self.energies, self.energy_kind, dropped)
            logger.info(
                "prune[%s]: %d -> %d  (merged %d near-duplicates)",
                method,
                len(before),
                len(self.ids),
                len(dropped),
            )
            if dropped and logger.isEnabledFor(logging.DEBUG):  # each dropped conformer's nearest kept one
                absorbed = defaultdict(list)
                for d, (k, r) in nearest_kept(mol, self.ids, dropped, hydrogens=hydrogens).items():
                    absorbed[k].append((d, r))
                for k, lst in sorted(absorbed.items()):
                    logger.debug("  #%d absorbed %s", k, ", ".join(f"#{d} ({r:.2f} A)" for d, r in sorted(lst)))
        return self

    def binding_modes(self):
        """Count the observed inter-fragment contact-kind sets across the conformers.

        An empty signature means no contact. Read full sites and references with `rx.contacts(observed=True)`.
        """
        return Counter(
            tuple(sorted({c.kind for c in found}))
            for found in observed_contacts(self.mol, self.ids, interfragment=True)
        )

    def cluster(self):
        """Return each conformer's binding-mode cluster label; -1 marks noise."""
        self.minimize()
        if not self.ids:
            return np.array([], dtype=int)
        return cluster_on(features(self.mol, self.ids).matrix)

    def landscape(self, method="pca", color="cluster"):
        """Plot the 2D ensemble map (method='pca'|'tsne'), coloured by 'cluster' or 'energy'."""
        self.minimize()  # energies for color='energy', on the geometry cluster() reads
        mol = with_conformers(self.specification.mol, self._mol, [conf.GetId() for conf in self._mol.GetConformers()])
        return viz.landscape(self, mol, color=color, method=method)  # it draws the dropped conformers too

    def _derive(self, ids, mol=None):
        """Copy selected conformers with their workflow state and independent tags."""
        out = super()._derive(ids, mol)
        out.tag = dict(self.tag)
        out._record(self._stage, out.energies, self.energy_kind)
        return out

    def lowest(self, n=1):
        """Return the `n` lowest-energy conformers as a new Ensemble, after the implicit ``minimize()``."""
        self.minimize()
        return self._derive(sorted(self.ids, key=self._rank)[:n])

    def representatives(self):
        """Return a new Ensemble of the lowest-energy conformer of each sampled mode.

        A noise conformer is kept only when it is a binding mode no cluster holds, within `_NOISE_WINDOW` kcal/mol of
        the lowest.
        """
        self.minimize()
        if not self.ids:  # an infeasible species, or nothing embedded
            return self._derive([])
        mol = self.mol
        snapshot = features(mol, self.ids)
        labels = cluster_on(snapshot.matrix).tolist()
        e_min = min(map(self._rank, self.ids))
        mode_reps = [
            min((self.ids[k] for k in range(len(labels)) if labels[k] == lab), key=self._rank)
            for lab in sorted({lab for lab in labels if lab != -1})
        ]
        noise_k = [k for k in range(len(labels)) if labels[k] == -1]
        recovered, skipped_e = [], 0
        sigs = mode_signature(mol, self.ids, snapshot) if noise_k else None  # plain organic noise is scatter
        if sigs is not None:  # one rep per noise signature not already clustered
            shown = {sigs[k] for k in range(len(labels)) if labels[k] != -1}  # all members, not just reps
            fresh = {}
            for k in noise_k:
                s, i = sigs[k], self.ids[k]
                if s in shown:
                    continue
                if self._rank(i) - e_min > _NOISE_WINDOW:
                    skipped_e += 1
                    continue
                if s not in fresh or self._rank(i) < self._rank(fresh[s]):
                    fresh[s] = i
            recovered = list(fresh.values())
        folded = len(noise_k) - len(recovered) - skipped_e
        note = f" + {len(recovered)} rare binding mode(s) recovered" if recovered else ""
        if skipped_e:
            note += f"; {skipped_e} skipped (> {_NOISE_WINDOW:g} kcal/mol)"
        if folded > 0:
            note += f"; {folded} scatter conformer(s) folded (still in ens.ids)"
        logger.info("representatives: %d %s mode(s)%s", len(mode_reps), snapshot.kind, note)
        chosen = mode_reps + recovered
        return self._derive(sorted(chosen, key=self._rank) or [min(self.ids, key=self._rank)])

    def _align_atoms(self, on):
        mol = self.mol
        if on is None:  # default core: constrained atoms > metal coordination > heavy
            core = sorted(self.cons.constrained_atoms())
            if len(core) >= _MIN_OVERLAY_ATOMS:
                return core
            # a metal's natural overlay core is M + its donors, even when cons is empty (e.g. a wrapped metal mol)
            if donors := metal_donors(mol, self.ids):
                return [metal_index(mol), *donors]
            return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        if isinstance(on, str):
            return list(match(mol, on))
        return [resolve_atom(mol, a) for a in on]

    def align(self, on=None):
        """Return a new Ensemble with every conformer Kabsch-superposed for a readable overlay.

        Align on the constrained core by default, or pass atom indices or SMARTS with ``on=``.
        """
        m = Chem.Mol(self._mol)
        if len(self.ids) > 1:
            rdMolAlign.AlignMolConformers(m, atomIds=self._align_atoms(on), confIds=list(self.ids))
        return self._derive(self.ids, mol=m)

    @property
    def n(self):
        """Return the number of tracked conformers."""
        return len(self.ids)

    def dump(self, path, align=True):
        """Write tracked conformers as a multi-frame XYZ, real metals restored, and return its path.

        Frames are core-aligned by default; ``align=False`` writes raw coordinates.
        """
        if not self.ids:  # a stage, not only the embed, can drop every conformer
            raise ValueError("nothing to dump: this ensemble has no conformers; the log says which stage dropped them")
        return Conformers.dump(self.align() if align else self, path)

    def __repr__(self):
        """Summarise the ensemble: conformer count, state, energy spread and name (`_label`)."""
        n = len(self.ids)
        de = f", ΔE 0.00-{max(spread.values()):.2f} kcal/mol" if len(spread := self.relative()) > 1 else ""
        return f"<Ensemble: {n} conformer{'s' if n != 1 else ''}, {self._stage}{de}, {_label(self)}>"


@dataclass(frozen=True, slots=True, kw_only=True)
class Coembedded:
    """Hold accepted reaction endpoint poses on one atom order and one set of ids, in one frame.

    Atom i is the same atom in `reactant` and `product`, and conformer k of one pairs with conformer k of the other.
    `coupling` holds the atoms the relax coupled and `source_seeds` the seed of each pair, None for a fixed supplied
    conformer. Indexing, slicing and `lowest` return a `Coembedded` of the pairs they pick, and iterating yields
    one-pair results; selecting one endpoint's conformers alone would break the pairing, which `ids` refuses. Scores and
    RMSDs read the current coordinates. Accepted poses stay final under implicit minimization.
    """

    reactant: Ensemble
    product: Ensemble
    coupling: Coupling
    joint_results: dict = field(default_factory=dict)
    optimizer_attempts: list = field(default_factory=list)

    @property
    def ids(self):
        """Return the pair ids, refusing endpoints whose ids no longer agree."""
        if self.reactant.ids != self.product.ids:
            raise ValueError("coembedded endpoint ids differ; select both endpoints together or regenerate the pair")
        return list(self.reactant.ids)

    @property
    def source_seeds(self):
        """Return each pair's seed, None for a fixed supplied conformer."""
        return {cid: record["seed"] for cid, record in self.joint_results.items()}

    @property
    def react_atoms(self):
        """Return the reacting atoms."""
        return tuple(self.coupling.atoms[self.coupling.spectators :].tolist())

    @property
    def alignment_scores(self):
        """Return each pair's spectator RMSD in Å."""
        return {
            cid: self.coupling.alignment_rmsd(
                self.reactant._mol.GetConformer(cid).GetPositions(),
                self.product._mol.GetConformer(cid).GetPositions(),
            )
            for cid in self.ids
        }

    @property
    def react_rmsds(self):
        """Return each pair's RMSD over its reacting atoms in Å, 0 without any."""
        atoms = self.coupling.atoms[self.coupling.spectators :]
        out = {}
        for cid in self.ids:
            left = self.reactant._mol.GetConformer(cid).GetPositions()[atoms]
            right = self.product._mol.GetConformer(cid).GetPositions()[atoms]
            out[cid] = float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1)))) if len(atoms) else 0.0
        return out

    @property
    def scores(self):
        """Return each pair's score: both endpoints' restrained-UFF energies plus the coupling's penalty."""
        return {cid: pair_score((self.reactant, self.product), self.coupling, cid) for cid in self.ids}

    def lowest(self, n=1):
        """Return the `n` pairs of lowest `scores` as a `Coembedded`, best first."""
        scores = self.scores
        return self._narrowed(sorted(self.ids, key=scores.__getitem__)[:n])

    def _narrowed(self, ids):
        """Return this result over the pair `ids` only, both endpoints narrowed alike (`Ensemble._derive`)."""
        return replace(
            self,
            reactant=self.reactant._derive(ids),
            product=self.product._derive(ids),
            joint_results={cid: dict(self.joint_results[cid]) for cid in ids},
            optimizer_attempts=[dict(record) for record in self.optimizer_attempts],
        )

    def __len__(self):
        """Return the number of pairs."""
        return len(self.ids)

    def __getitem__(self, key):
        """Return the pair at position `key` of `ids`, or the pairs a slice picks, as a `Coembedded`."""
        picked = self.ids[key]
        return self._narrowed(picked if isinstance(picked, list) else [picked])


class CoembeddedSet(list):
    """Hold one `Coembedded` per binding mode, as ``rx.coembed(contacts='auto')`` gives them, and the failed modes.

    Each member's endpoints carry their mode as their ``nci`` tag. Slicing and `filter` (``index``, ``nci``) keep
    `errors`, the `EmbeddingError` of each failed mode.
    """

    def __init__(self, members=(), errors=()):
        super().__init__(members)
        self.errors = tuple(errors)

    def __getitem__(self, key):
        """Return one member, or a `CoembeddedSet` keeping `errors` for a slice."""
        got = super().__getitem__(key)
        return CoembeddedSet(got, self.errors) if isinstance(key, slice) else got

    def filter(self, *, index=None, nci=None):
        """Return the members at `index` (`_picked`) whose mode is `nci`, as a `CoembeddedSet` keeping `errors`."""
        picked = _picked(self, None, index, {})
        return CoembeddedSet((pair for pair in picked if nci in (None, pair.reactant.tag.get("nci"))), self.errors)


def wrap(mol, ids=None, *, energies=None, minimized=False):
    """Wrap a copy of an RDKit Mol with conformers as an Ensemble, to give it rxembed's methods.

    Only the graph is read, as the readers normalise it: donor bonds dative, a neutral M-H a hydride. A graph that is no
    species (open bond orders, a neutral bridging donor) is wrapped as it stands for the verbs that read bonds alone.
    The species has no centres, so `rx.minimize`, not this `minimize()`, relaxes a metal under its coordination.
    Geometries are taken as unrelaxed, so `prune()`, `representatives()` and `lowest()` first run one UFF `minimize()`.
    Give already optimised conformers (CREST, xtb or DFT) their ``energies=`` in kcal/mol, a list in `ids` order or a
    dict by id, or pass ``minimized=True`` to score them with ``score('ff')``; they then stay as they are. `ids` are
    conformer ids.
    """
    mol = Chem.Mol(without_stale_point_tags(require_mol(mol, "rx.wrap", geometry=True)))
    try:
        mol = canonical_metal_graph(mol)
    except ValueError as refused:
        logger.debug("wrap: %s; the graph is wrapped as it stands", refused)
    ids = list(ids) if ids is not None else [c.GetId() for c in mol.GetConformers()]
    available = {conf.GetId() for conf in mol.GetConformers()}
    listed = energies is not None and not isinstance(energies, dict)
    invalid = (
        any(not is_index(cid) or cid not in available for cid in ids)
        or len(set(ids)) != len(ids)
        or (
            energies is not None
            and ((listed and len(energies) != len(ids)) or (not listed and set(energies) - set(ids)))
        )
    )
    values = {}
    if energies is not None and not invalid:
        given = dict(zip(ids, energies, strict=False)) if listed else energies
        values = {int(i): float(e) for i, e in given.items()}
        invalid = not all(np.isfinite(value) for value in values.values())
    if invalid:
        raise ValueError(
            f"wrap(ids=) takes distinct conformer ids from {sorted(available)}; energies= takes {len(ids)} "
            "finite values or a dict"
        )
    ids = [int(cid) for cid in ids]
    graph = with_conformers(mol, mol, ())
    species = Species(mol=graph, stereo=defined_stereo_label(graph))  # its own graph is the engine's
    ens = Ensemble(mol, ids, species=species)
    if energies is not None:
        ens._record(FINAL, values, "given")
    elif minimized:
        ens._record(FINAL, {}, "")
        ens = ens.score("ff")
    return ens
