"""The user-facing pipeline: one `embed()` entry returning a chainable `Ensemble`.

    import rxembed as rx
    ens = rx.embed("CCO").mc().prune()                              # free, vdW-aware, sanity internal
    ens = rx.embed("OC(=O)CCc1ccccc1", constrain={(1, 9): (2.6, 3.0)})  # soft distance window (index-driven)
    ens = rx.embed("ts.xyz", fix=[11, 14, 15]).mc().prune()        # hold a TS core at its geometry (0.000 A graft)
    for iso in rx.metal("CCCN[Pd](Cl)(Cl)NCCC", "square_planar"):
        ens = rx.embed(iso).mc().prune()                           # metal surrogate handled internally

Sanity (bond-perception vs the graph), multi-fragment vdW separation, the metal carbon-surrogate
swap/restore, and constraint validation all happen inside — the user does not manage them. Every
stage logs (`rxembed.set_verbose()`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

from . import dedup as _dedup
from . import metrics as _metrics
from . import refine as _refine
from . import stereo as _stereo
from .constraints import Constraints, match, resolve_atom
from .constraints import metal as _metal
from .constraints import nci as _nci
from .embed import mc as _mc
from .log import logger

if TYPE_CHECKING:
    from .embed.dispatch import _MetalCtx

# The embed-dispatch machinery (source routing, metal surrogate, Kabsch graft, isomer/template/auto-NCI)
# lives in `rxembed.embed.dispatch`; it constructs the `Ensemble` / `EnsembleSet` defined here, so this
# module imports it lazily (inside `embed()` and `Ensemble.mc`) to keep the import cycle one-directional.
_MIN_OVERLAY_ATOMS = 3  # need >=3 atoms to define an alignment frame
_HARTREE_KCAL = 627.5094740631  # Eh -> kcal/mol, so a calculator's energies match the FF's unit


def _last_line(err):
    """Return the last non-empty line of an exception (xtb dumps a stderr tail) for a one-line warning."""
    s = str(err).strip()
    return s.splitlines()[-1] if s else "no output"


class EnsembleSet(list):
    """Several candidate `Ensemble`s to choose from (metal isomers, or ambiguous ``coordinate=`` donors).

    Each is ``.tag``ged by what makes it distinct. The same *enumerate-candidates -> select* pattern as NCI
    binding modes:

        cands = rx.embed('CCCN[Pd](Cl)Cl', metal='square_planar'); cands.summary()
        ens   = cands.select(label='trans').mc().prune()        # conf-search just the one you want

    `select(**tag)` returns the single match (or a smaller `EnsembleSet` if several match); iterate or
    index to keep them all.
    """

    def __repr__(self):
        """Summarise the candidates and their tags on one line."""
        labels = [e.tag.get("label", e.tag) for e in self]
        return f"<EnsembleSet: {len(self)} candidate{'s' if len(self) != 1 else ''} — {labels}>"

    def select(self, **tag):
        """Return the single candidate matching `tag`, as a chainable `Ensemble`.

        Raises if the tags match zero or several -- narrow them (e.g. add `geometry=`), or use
        `filter(...)` / iterate to keep several.
        """
        hits = self.filter(**tag)
        if len(hits) != 1:
            raise ValueError(
                f"select({tag}) matched {len(hits)} candidate(s) — "
                f"{'narrow the tags' if hits else 'no match'}; have {[e.tag for e in self]}"
            )
        return hits[0]

    def filter(self, **tag):
        """Return the subset of candidates matching `tag`, as an `EnsembleSet` (keep several)."""
        return EnsembleSet(e for e in self if all(e.tag.get(k) == v for k, v in tag.items()))

    def summary(self):
        """Print each candidate (index, tag, #seeds) so you can pick one. Returns self (chainable)."""
        for k, e in enumerate(self):
            print(f"  [{k}] {e.tag}  ({e.n} seeds)")
        return self


def embed(
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
    stereo="auto",
    **kw,
):
    """Embed conformers (optionally constrained), returning an `Ensemble` or an `EnsembleSet` of candidates.

    An `EnsembleSet` (of candidates to `.select` from) when the input is inherently several poses -- metal
    coordination isomers, discovered NCI modes, or an ambiguous ``coordinate=``.

    **Three constraint verbs** (all index-driven — 0-based atom indices in xyz/graph order; resolve any
    SMARTS yourself first):

    - **``fix``** *(rigid — the atoms WILL have this geometry)*:
      ``fix=[i, j, k]`` holds them at the source's own coords (Kabsch graft; needs a geometry);
      ``fix={i: (x, y, z)}`` at explicit coords; ``fix={(i, j): d, (i, j, k): θ}`` at exact numbers
      (tight-window UFF pull — verify with ``.measure()``). A ``fix`` dict may mix coords and numbers.
    - **``constrain``** *(soft — bias the seed, a real energy may win)*: ``constrain={(i, j): (lo, hi)}``
      distance/angle windows, plus π-stacks ``constrain={(ring_a, ring_b): separation}``.
    - **``template``** *(reference sugar for a coords-fix)*: ``template=(reference, {target_i: ref_i})`` —
      ``reference`` is an .xyz path / Mol / Ensemble / (N,3) array, mapped explicitly (order-proof).

    The one surface for every case:

    - free / flexible:     ``rx.embed('CCO')``
    - a distance / angle:  ``rx.embed(smi, constrain={(i, j): (2.6, 3.0)})``
    - NCI / vdW complex:   ``rx.embed('A.B', contacts='auto')``  (or a specific ``rx.nci_modes(mol)['HB:…']``)
    - frozen TS core:      ``rx.embed('ts.xyz', fix=[14, 15])`` — held to 0.000 Å
    - a TS from SMILES:    ``rx.embed(smi, fix={(f, c): 2.02, (c, cl): 2.28, (f, c, cl): 178})``
    - a known TS onto a molecule: ``rx.embed(smi, template=('ts.xyz', {0: 5, 4: 1, 5: 6}))``
    - **metal**:           ``rx.embed('…[Pd]…', metal='square_planar')`` → `EnsembleSet` of isomers
    - metal + substrate:   ``rx.embed('…[Pd]….O1CCCC1', metal='square_planar', coordinate='[OX2]')``

    Keyword detail lives with the machinery each drives: ``contacts=`` / ``contacts='auto'`` → `nci_modes`;
    ``metal=`` / ``coordinate=`` → `rx.metal` (they reuse ``fix``/``constrain`` for their held cores);
    ``stereo=`` (chirality the embed can't keep — planar/axial/helical) → `Ensemble.select_stereo`.
    """
    n_alias = kw.pop("n_confs", None)
    n_alias = kw.pop("num_confs", n_alias)
    if n is None:
        n = n_alias  # accept n_confs= / num_confs= as friendly aliases of n=
    if kw:
        raise TypeError(
            f"embed() got unexpected keyword(s) {sorted(kw)} — the constraint verbs are "
            f"fix / constrain / template (+ contacts / metal / coordinate / n_confs)"
        )
    from .embed.dispatch import _attach_stereo, _embed_dispatch  # lazy: breaks the dispatch<->pipeline cycle

    result = _embed_dispatch(
        source,
        metal=metal,
        fix=fix,
        constrain=constrain,
        template=template,
        contacts=contacts,
        coordinate=coordinate,
        charge=charge,
        n=n,
        seed=seed,
        knowledge=knowledge,
    )
    _attach_stereo(result, source, charge, stereo)
    return result


def minimize(source, *, fix=None, constrain=None, charge=0, distance_fc=1e4):
    """Relax an existing structure **toward** ``fix``/``constrain`` targets — the search-free companion to `embed`.

    Same vocabulary and resolver as `embed`, but it does not conf-search: it wraps the input geometry, grafts
    any coordinate-``fix`` core, and runs the restrained UFF pull toward the targets (a numbers-``fix`` is
    pulled exact, a ``constrain`` window is respected). Use it to nudge a geometry into a TS-like core or a
    contact without re-sampling the periphery. Needs an input geometry (an .xyz / a Mol with a conformer).

        rx.minimize('mol.xyz', fix={(i, j): 2.0, (i, j, k): 178})   # pull toward a linear 3-centre core
    """
    from .constraints import resolve_core
    from .embed.dispatch import _graft_frozen, _normalize

    mol, has_geom = _normalize(source, charge)
    if not has_geom:
        raise ValueError(
            "minimize() relaxes an existing geometry — give an .xyz or a Mol with a conformer, not a SMILES"
        )
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=has_geom)
    ids = [c.GetId() for c in mol.GetConformers()]
    if ref:
        graft = sorted(ref)
        _graft_frozen(mol, ids, graft, np.array([ref[i] for i in graft]))
    return Ensemble(mol, ids, cons).minimize(distance_fc=distance_fc)


def wrap(mol, ids=None, *, energies=None, minimized=False):
    """Wrap an existing RDKit Mol (with conformers) as an Ensemble to give it rxembed's methods.

    Any conformers -- from racerts, RDKit's own EmbedMultipleConfs, an xtb optimisation, or a loaded
    multi-frame file -- then get `align()`, `representatives()`, `cluster()`, `prune()`, `measure()`, `dump()`.

    By default the wrapped geometries are treated as un-relaxed: `prune()`/`representatives()`/`lowest()`
    will run one FF `minimize()` first (which **moves atoms** onto rxembed's force field). If your
    conformers are **already optimised** and must not be disturbed (racerts/xtb output), pass
    ``minimized=True`` — the relax is skipped and energies are taken from ``energies=`` (a list aligned
    to `ids`, or a dict) or computed as **single points** (no geometry change) if omitted.
    """
    ids = ids if ids is not None else [c.GetId() for c in mol.GetConformers()]
    ens = Ensemble(mol, list(ids))
    if energies is not None:
        if not isinstance(energies, dict) and len(energies) != len(ids):
            raise ValueError(f"wrap(energies=) has {len(energies)} values but there are {len(ids)} conformers")
        if isinstance(energies, dict):
            ens.energies = dict(energies)
        else:
            ens.energies = {i: float(e) for i, e in zip(ids, energies, strict=False)}
    if minimized:
        if not ens.energies:
            e = _refine.ff_energies(mol, minimize=False)  # single-point, geometry untouched
            by_id = {c.GetId(): float(e[k]) for k, c in enumerate(mol.GetConformers())}
            ens.energies = {i: by_id[i] for i in ids if i in by_id}
        ens._minimized = True
    return ens


@dataclass
class Ensemble:
    """A conformer ensemble: one RDKit Mol (the graph) holding many conformers (each a 3-D geometry).

    `ens.mol` is that shared molecule; `ens.ids` are the conformer ids in play; `ens.energies` maps
    id -> energy.

    **Mutation contract** (so nothing surprises you):
    - The pipeline steps that *build* the ensemble — `mc`, `minimize`, `prune` — change it in place
      and return it, so they chain: `embed(...).mc().prune()`.
    - Everything that *derives* a smaller/aligned set — `lowest`, `representatives`, `align` — returns
      a **new** Ensemble and leaves this one untouched.
    - **Looking never mutates.** `view`, `compare`, `landscape`, `cluster`, `binding_modes` change
      nothing (they relax once via `minimize` if energies are needed, which is idempotent).
    """

    mol: Chem.Mol
    ids: list
    cons: Constraints = field(default_factory=Constraints)
    _metal: _MetalCtx | None = None
    energies: dict = field(default_factory=dict)
    _minimized: bool = False
    discarded: list = field(default_factory=list)  # conformer ids prune merged away (still in `mol`)
    tag: dict = field(default_factory=dict)  # what distinguishes this candidate in an EnsembleSet
    # (e.g. {'geometry','label'} for a metal isomer)
    _stereo: tuple | None = None  # (spec, reference signature) for the chirality filter
    # applied in minimize() — set by embed(stereo=…)

    def mc(
        self, *, preset="ensemble", seed=None, max_out=None, low_mode=None, config=None, replace=None, explore=False
    ):
        """Openconf Monte-Carlo torsional search (rowansci), in place.

        `preset` effort ('rapid'|'ensemble'|'spectroscopic'|'docking'|'analogue'|'macrocycle'); `seed`
        reproducibility; `max_out` conformer cap; `low_mode` Hessian low-mode following (off under
        constraints); `config` a raw openconf ConformerConfig.

        Unconstrained: openconf drives generation and *replaces* the ETKDG seeds. Constrained / multi-
        fragment: it searches *around* the bounds-biased seeds with the held atoms pose-frozen, so its
        output is *added* (the seeds are first settled into their windows — see `_settle_seeds`).
        `replace=` overrides.

        `explore=True` (a seeded NCI complex): fire a second search with the NCI contacts *released*
        (structural holds — frozen core, π planes, encounter bounds — kept), pool both, and swap cons to the
        relaxed set so downstream minimize/score/optimize don't yank the released contacts back. Energy then
        decides — best with an NCI-aware calculator (`score('gfnff')`). On the metal path only the substrate
        contacts release; the coordination sphere stays held.
        """
        if not _mc.available():
            logger.warning("mc: openconf not installed; skipped")
            return self
        if preset == "rapid" and self.cons.is_constrained:  # constrained pose-mode is rotor-only -> rapid
            logger.warning(
                "mc: preset='rapid' on a constrained system under-samples (pose-mode is "  # under-
                "rotor-only); 'ensemble' (the default) is recommended for a constrained run"
            )  # samples

        self._settle_seeds()  # settle seeds INTO their windows (spread across them) before openconf pose-freezes

        def _search(cons, label):
            try:
                return _mc.search(
                    self.mol, cons, preset=preset, seed=seed, max_out=max_out, low_mode=low_mode, config=config
                )
            except Exception as e:  # openconf can't handle every system (e.g. a TS
                logger.warning(
                    "mc%s: openconf could not search this system (%s: %s) — keeping the %d conformer(s)",
                    label,
                    type(e).__name__,
                    e,
                    len(self.ids),
                )  # hypervalent core
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

        if explore and any(self.cons.contacts):  # second pass: contacts released, structure kept
            from .embed.dispatch import _encounter_bounds  # lazy: breaks the dispatch<->pipeline cycle

            relaxed = self.cons.relaxed()
            # NB distinct from dispatch's `_float_encounter_bounds`: releasing the grip frees the fragments it
            # linked, so explore re-bounds ALL inter-fragment pairs (setdefault: never override a surviving
            # structural hold) — not just the "no constraint touches them" subset that path tethers at embed.
            if len(Chem.GetMolFrags(self.mol)) > 1:  # keep the fragments together once contacts are freed
                for k, v in _encounter_bounds(self.mol).items():
                    relaxed.distances.setdefault(k, v)
            more = _search(relaxed, " explore")
            if more:
                self.ids += more
                self.cons = relaxed  # downstream relax/score/opt no longer yank contacts
                logger.info(
                    "mc explore: +%d relaxed conformer(s) (NCI contacts released; %d structural/"
                    "encounter hold(s) kept; total %d)",
                    len(more),
                    len(relaxed.distances),
                    len(self.ids),
                )
        return self

    def _settle_seeds(self, bins=5):
        """Pull the seeded (non-frozen) distance/angle constraints inside their windows before an MC search.

        Spread across the window over the conformers (each bin targets a different fraction of every window)
        rather than collapsed to one value, so the search keeps its breadth and does not over-dictate the
        contact. Frozen atoms and the frozen-core shape are held exactly and untouched. A narrow (point/TS)
        target barely spreads (stays effectively exact); a wide window keeps its full range. Fixes the raw
        ETKDG seed sitting *outside* a tight window.
        """
        cons = self.cons
        seeded_d = [(k, v) for k, v in cons.distances.items() if not (k[0] in cons.frozen and k[1] in cons.frozen)]
        shape_d = {k: v for k, v in cons.distances.items() if k[0] in cons.frozen and k[1] in cons.frozen}
        if not (seeded_d or cons.angles) or not self.ids:
            return
        for b, group in enumerate(g for g in np.array_split(list(self.ids), min(len(self.ids), bins)) if len(g)):
            frac = (b + 0.5) / min(len(self.ids), bins)  # this bin's fraction across every window
            tgt = Constraints(frozen=set(cons.frozen), planes=list(cons.planes))
            tgt.distances.update(shape_d)  # keep the frozen-core shape exact
            for k, (lo, hi) in seeded_d:
                m = lo + frac * (hi - lo)
                tgt.distances[k] = (max(lo, m - 0.03), min(hi, m + 0.03))  # clamp inside the user window
            for k, (lo, hi) in cons.angles.items():
                m = lo + frac * (hi - lo)
                tgt.angles[k] = (max(lo, m - 2.0), min(hi, m + 2.0))
            _refine.restrained_uff(self.mol, tgt, distance_fc=1e4, conf_ids=[int(i) for i in group])

    def minimize(self, distance_fc=1e4):
        """FF relax (stiff restrained UFF if constrained, else MMFF), restore any metal, drop clashes, validate."""
        if self._minimized:
            return self
        if not self.ids:  # nothing embedded (e.g. ETKDG could not place this graph)
            self._minimized = True
            return self
        e = None
        if self.cons.is_constrained:
            try:
                e = _refine.restrained_uff(self.mol, self.cons, distance_fc=distance_fc)
                if self._metal:
                    self._metal.fix_donor_protons(self.cons, self.ids, distance_fc)  # stage-2 donor-H relax
            except RuntimeError as err:  # UFF can't build a force field for this graph —
                logger.warning(
                    "minimize: UFF could not relax this system (%s) — an untypable TS/hypervalent "
                    "reacting core; keeping the embedded geometry (constraints biased, not "
                    "stiffened)",
                    err,
                )  # keep the embed; its constraints were only biased
        else:
            e = _refine.ff_energies(self.mol, minimize=True)
        if e is not None:
            self.energies = {c.GetId(): float(e[k]) for k, c in enumerate(self.mol.GetConformers())}
        if self._metal:
            self._metal.restore()
            self._metal = None
        before = len(self.ids)
        self.ids = [i for i in self.ids if _metrics.bonding_ok(self.mol, i, exclude=self.cons.frozen)]  # arbiter:
        if len(self.ids) < before:  # drop torn/clashed geometries
            logger.info(
                "minimize: dropped %d conformer(s) with a broken/clashing bond -> %d kept",
                before - len(self.ids),
                len(self.ids),
            )
        if not self.ids:  # all conformers broke a bond — the arrangement itself
            logger.warning(
                "minimize: every embedded conformer breaks a ligand bond — this arrangement is "
                "geometrically infeasible (e.g. a chelate forced to span a bite it can't reach); "
                "the ensemble is empty"
            )  # is impossible (an unreachable chelate isomer)
        if self._stereo and self.ids:  # keep only the requested handedness on chirality the
            spec, ref = self._stereo  # embed can't (metallocene planar / axial / helical)
            kept = [i for i in self.ids if _stereo.passes(_stereo.signature(self.mol, i), ref, spec)]
            if len(kept) < len(self.ids):
                logger.info(
                    "minimize: stereo=%r kept %d/%d (matched the input handedness)", spec, len(kept), len(self.ids)
                )
            self.ids = kept
            self._stereo = None
        self._validate()
        self._minimized = True
        return self

    def _validate(self, dist_slack=0.15, ang_slack=5.0):
        """Warn if a constraint the relax *can enforce* is not realised within its window (+ a small slack).

        Only constraints with at least one non-frozen atom are warned — the seeded contacts / user windows
        the restrained relax is responsible for. A constraint **between two frozen atoms** is a structural
        hold (a frozen-core shape, a spectator metal sphere) held by the exact graft / ``AddFixedPoint`` at
        the embedded geometry, not by the distance window — the relax cannot move a pinned atom to satisfy
        it, so it is logged at DEBUG. (An off-window value there is the embed's approximation of a
        *spectator* shape; the reacting core stays exact.)
        """
        if not self.ids:
            return
        frozen = self.cons.frozen

        def realised(fn, *atoms):
            return float(np.mean([fn(self.mol.GetConformer(c), *atoms) for c in self.ids]))

        for (i, j), (lo, hi) in self.cons.distances.items():
            d = realised(rdMolTransforms.GetBondLength, i, j)
            if lo - dist_slack <= d <= hi + dist_slack:
                logger.debug("held d(%d,%d) = %.2f (target %.2f-%.2f)", i, j, d, lo, hi)
            elif i in frozen and j in frozen:  # structural hold among pinned atoms — not the relax's to enforce
                logger.debug("frozen-shape d(%d,%d) = %.2f (embed approx of %.2f-%.2f)", i, j, d, lo, hi)
            else:
                logger.warning("constraint NOT held: d(%d,%d) = %.2f, target %.2f-%.2f", i, j, d, lo, hi)
        for (i, j, k), (lo, hi) in self.cons.angles.items():
            a = realised(rdMolTransforms.GetAngleDeg, i, j, k)
            if lo - ang_slack <= a <= hi + ang_slack:
                continue
            if i in frozen and j in frozen and k in frozen:
                logger.debug("frozen-shape angle(%d,%d,%d) = %.1f (target %.1f-%.1f)", i, j, k, a, lo, hi)
            else:
                logger.warning("constraint NOT held: angle(%d,%d,%d) = %.1f, target %.1f-%.1f", i, j, k, a, lo, hi)

    def score(self, refine="gxtb", solvent=None, charge=None):
        """Re-rank by a real single-point energy (g-xTB by default), returning a new Ensemble.

        The FF/surrogate-UFF energy is meaningless for a metal or charged TS. No geometry change of its own:
        it scores the geometry `minimize()` produced (a constrained core held exactly); energies are in
        kcal/mol (see `relative` for the unit contract). `charge` defaults to the formal charge; `refine` is
        'gxtb'/'gfn2'/'ff'/a Calculator; `solvent` adds a GFN2-ALPB correction. xTB is ~seconds/conformer --
        call `representatives()` / `lowest(k)` first on a big ensemble.
        """
        self.minimize()  # settle the periphery (a constrained core stays pinned)
        if not self.ids:
            return self._derive([])
        from .refine.calculator import resolve

        q = self._calc_charge(charge)
        calc = resolve(refine, solvent, q)
        if calc is None:  # 'ff'/None -> single-point force field, geometry kept
            e = _refine.ff_energies(self.mol, minimize=False)
            by = {c.GetId(): float(e[k]) for k, c in enumerate(self.mol.GetConformers())}
            return Ensemble(
                Chem.Mol(self.mol),
                list(self.ids),
                self.cons,
                None,
                {i: by[i] for i in self.ids if i in by},
                True,
                tag=dict(self.tag),
            )
        energies, kept = {}, []
        for i in self.ids:
            try:
                energies[i] = float(calc.energy(self.mol, i)) * _HARTREE_KCAL  # -> kcal/mol, unit-consistent
                kept.append(i)  # with the FF energies (windows
            except Exception as err:  # a single conformer xtb failure shouldn't sink the run
                logger.warning("score: %s failed on conformer %d (%s) — dropping it", refine, i, _last_line(err))
        if not kept:  # total failure -> RAISE (don't silently hand back FF
            raise RuntimeError(
                f"score: {refine} produced no energies — is the xtb binary on PATH "  # energies the
                f"($XTB_EXE, or ~/bin/xtb)? not falling back to the force field silently"
            )  # caller
            #                                                  thinks are xTB; honest energies were the whole point
        logger.info("score: %s single point on %d conformer(s) (charge %d)", refine, len(kept), q)
        return Ensemble(Chem.Mol(self.mol), kept, self.cons, None, energies, True, tag=dict(self.tag))

    def _calc_charge(self, charge):
        """Return the total charge for a calculator: the override, else the molecule's formal charge.

        Warns for a metal, whose perceived formal charge is a likely bond-perception artefact.
        """
        q = Chem.GetFormalCharge(self.mol) if charge is None else charge
        if charge is None and q != 0 and _metal.metal_index(self.mol) is not None:
            logger.warning(
                "a metal complex's PERCEIVED formal charge is %d — likely a bond-perception artefact; "
                "pass charge=<true total charge> for honest energies (relative ranking of same-charge "
                "isomers is still meaningful)",
                q,
            )
        return q

    def optimize(self, refine="gxtb", level="normal", solvent=None, charge=None):
        """Geometry-optimise each conformer with xtb at `level`, returning a new ensemble.

        `level` is 'loose'/'normal'/'tight'/'vtight'; `cons.frozen` is held fixed while everything else
        relaxes. The optimised geometries + energies (kcal/mol) are on a copy (this settles `self` once via
        `minimize`, then moves atoms on the copy -- never the fixed core).

        `cons.frozen` is the reacting core only: a `rx.metal(center=, fix=)` TS holds its coordination
        sphere by *soft* shape constraints (so it relaxes here), while `rx.embed(ts.xyz, fix=[...])` has
        the metal+donors *in* `cons.frozen` (held). Pass `charge=` for a metal; `refine='gfn2'` for a
        solvated opt (g-xTB has no ALPB).
        """
        self.minimize()
        if not self.ids:
            return self._derive([])
        from .refine.calculator import resolve

        q = self._calc_charge(charge)
        calc = resolve(refine, solvent, q)
        if calc is None:
            raise ValueError(
                "optimize needs a real calculator (refine='gxtb' or 'gfn2'); the force-field relaxation is minimize()"
            )
        fix = sorted(self.cons.frozen)  # the frozen TS core (reacting atoms)
        new_mol = Chem.Mol(self.mol)  # opt MOVES atoms -> work on a copy, never clobber self
        energies, kept = {}, []
        for i in self.ids:
            try:
                coords, e = calc.optimize(self.mol, i, level, fix)
                conf = new_mol.GetConformer(i)
                for a, xyz in enumerate(coords):
                    conf.SetAtomPosition(a, [float(v) for v in xyz])
                energies[i] = e * _HARTREE_KCAL
                kept.append(i)
            except (RuntimeError, OSError) as err:  # an xtb RUN failure -> drop this conformer; a
                logger.warning(
                    "optimize: %s --opt failed on conformer %d (%s) — dropping it", refine, i, _last_line(err)
                )
                #                                              config error (bad level/solvent) is a ValueError
                #                                              and propagates (it's not a per-conformer failure)
        if not kept:
            raise RuntimeError(
                f"optimize: {refine} --opt produced nothing — is the xtb binary on PATH ($XTB_EXE, or ~/bin/xtb)?"
            )
        logger.info(
            "optimize: %s --opt %s on %d conformer(s); %d core atom(s) held fixed", refine, level, len(kept), len(fix)
        )
        return Ensemble(new_mol, kept, self.cons, None, energies, True, tag=dict(self.tag))

    def relative(self, unit="kcal"):
        """Relative energies ``{conformer id -> E - E_min}`` -- the only physically meaningful read.

        An absolute single-point/FF total is not meaningful. Energies are stored in **kcal/mol** (FF or
        `score` alike), so ``unit='kcal'`` is identity; ``'hartree'`` divides back. Lowest conformer is 0.0.
        """
        have = {i: self.energies[i] for i in self.ids if i in self.energies}
        if not have:
            return {}
        emin = min(have.values())
        scale = 1.0 if unit == "kcal" else 1.0 / _HARTREE_KCAL
        return {i: (e - emin) * scale for i, e in have.items()}

    def prune(self, by="auto", **kw):
        """Deduplicate by geometry, in place.

        `by` (alias `method`) is **'auto'** (default) -- the rigorous, rotation-invariant **'rmsd'**, safe
        for any system -- or pick explicitly: 'rmsd' | 'moi' | 'descriptor' | 'energy', or a cheap-first
        cascade list, e.g. ['moi', 'rmsd'] (moi is a fast coarse pre-filter worth it only for *large
        single-molecule* ensembles; it can over-merge, so it is not the default). Avoid **'moi' on a
        multi-fragment system**: principal moments of inertia are nearly blind to a light or symmetric
        fragment's *relative pose*, so moi would silently merge distinct encounter geometries -- rxembed
        warns if you do. For a binding-mode *summary* use `representatives()`.

        Tuning knobs (keyword): `max_rmsd` (rmsd cutoff Å), `moi_dev`, `max_dist` (descriptor), `energy_tol`
        (energy), `energy_window` (kcal/mol gate). Nothing is lost: discarded conformers stay in `ens.mol`;
        `ens.discarded` lists their ids and `ens.duplicates()` says which kept conformer each duplicates.
        """
        by = kw.pop("method", by)  # 'method=' is the intuitive name; accept it as an alias of by=
        allowed = {"energy_window", "max_dist", "moi_dev", "max_rmsd", "energy_tol"}
        if set(kw) - allowed:
            raise TypeError(
                f"prune() got unexpected keyword(s) {sorted(set(kw) - allowed)}; the method is "
                f"by=/method= (e.g. by='rmsd'), tuning knobs are {sorted(allowed)}"
            )
        self.minimize()
        if not self.ids:  # a geometrically-infeasible isomer minimised to empty
            return self
        from .dedup.features import _metal_present

        # a metal complex looks multi-fragment only because the surrogate stripped its coordinate bonds;
        # those "fragments" are one molecule, so moi is a fine choice there — don't count them.
        n_frag = 1 if _metal_present(self.mol) else len(Chem.GetMolFrags(self.mol))
        if by == "auto":
            by = "rmsd"  # rigorous everywhere; moi/cascade are opt-in for speed
        methods = [by] if isinstance(by, str) else list(by)
        if n_frag > 1 and "moi" in methods:
            logger.warning(
                "prune[moi] on %d fragments: moment-of-inertia dedup is insensitive to a "
                "light/symmetric fragment's relative pose and may over-merge distinct "
                "encounter geometries — prefer 'rmsd' here (or representatives()).",
                n_frag,
            )
        for method in methods:
            before = list(self.ids)
            energies = [self.energies.get(i, 0.0) for i in self.ids]  # 0.0 if relax was skipped (untypable core)
            self.ids, _ = _dedup.apply(self.mol, self.ids, energies, method=method, **kw)
            kept = set(self.ids)
            dropped = [i for i in before if i not in kept]
            self.discarded += dropped
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
        for d, (k, r) in _dedup.nearest_kept(self.mol, kept, dropped).items():
            groups.setdefault(k, []).append((d, r))
        return {k: sorted(v) for k, v in sorted(groups.items())}

    def duplicates(self):
        """Explain what prune discarded: ``{kept_id: [(discarded_id, RMSD_Angstrom), ...]}``.

        Each discarded conformer is grouped under the kept one it most resembles, with their heavy-atom
        RMSD. The discarded geometries still live in `ens.mol`, so dump any with
        `rxembed.wrap(ens.mol, [discarded_id]).dump('dropped.xyz')`. (At `set_verbose('DEBUG')` prune
        logs this as it runs.)
        """
        return self._group_duplicates(self.ids, self.discarded) if self.discarded else {}

    def binding_modes(self):
        """Count the inter-fragment NCI binding-mode signatures sampled across the conformers.

        Uses the same inter-fragment scope as the latent / ``representatives`` -- intramolecular NCIs are
        conformational detail, not binding modes -- so the two views agree. Empty signature ``()`` means no
        inter-fragment contact in that pose.
        """
        from collections import Counter

        from .dedup.features import _frag_map, _interfragment_contacts

        an = _nci.analyzer(self.mol)
        fmap = _frag_map(self.mol)

        def sig(c):
            contacts = _interfragment_contacts(an, self.mol.GetConformer(c).GetPositions(), fmap)
            return tuple(sorted({t for t, _a, _p in contacts}))

        return Counter(sig(c) for c in self.ids)

    def cluster(self, *, min_cluster=3, reduce=None, nci=True):
        """Binding-mode cluster label per conformer (HDBSCAN on the shared latent; -1 = rare/noise).

        On the dihedral [+NCI] [+metal] latent that ``landscape()`` also projects.
        """
        self.minimize()
        if not self.ids:
            return np.array([], dtype=int)
        return _dedup.cluster_labels(self.mol, self.ids, min_cluster=min_cluster, reduce=reduce, nci=nci)

    def landscape(self, method="pca", color="cluster", *, reduce=None, min_cluster=3, nci=True):
        """2D ensemble map (method='pca'|'tsne'), coloured by 'cluster' or 'energy'."""
        from . import viz

        return viz.landscape(self, color=color, method=method, reduce=reduce, min_cluster=min_cluster, nci=nci)

    # -- deriving a smaller / aligned ensemble (returns a NEW Ensemble) --------

    def _derive(self, ids, mol=None):
        mol = mol if mol is not None else Chem.Mol(self.mol)  # own the Mol: a derived ensemble must not
        return Ensemble(
            mol,
            list(ids),
            self.cons,
            None,  # alias the parent (mc/prune on it would
            {i: self.energies[i] for i in ids if i in self.energies},
            self._minimized,
            tag=dict(self.tag),
        )  # else mutate the parent). Keep provenance.

    def select_stereo(self, like, spec="preserve"):
        """Keep only conformers whose chirality matches `spec` against a reference `like`.

        `like` is a Mol carrying the wanted handedness (usually the input geometry). This is for the
        chirality the **embed cannot keep** -- a metallocene's **planar** chirality, **axial**/**helical**
        atropisomerism -- sampled at random by a distance-geometry embed and selected here, *general* over
        all element kinds via xyzgraph.

        `spec`: ``'preserve'`` (default — keep the non-graph chirality, i.e. planar/axial/helical; leave
        **point R/S and E/Z free**, since those are the embed's own job or a labile centre you would not
        want to lock — e.g. a protic amine); ``'free'`` (sample every handedness); ``'invert'`` (the
        enantiomeric series); or a dict ``{kind|'default': mode}`` to **keep some and scramble others**
        (e.g. ``{'planar':'preserve','default':'free'}`` — hold a ferrocene, explore the metal centre).
        Returns a new Ensemble (a no-op if no managed chirality element is present).
        """
        if not self._minimized:
            self._stereo = None  # an EXPLICIT select_stereo overrides the embed default,
        self.minimize()  # so e.g. select_stereo(..,'free') still sees every pose
        ref = _stereo.signature(like) if isinstance(like, Chem.Mol) else _stereo.signature(*like)
        keep = [i for i in self.ids if _stereo.passes(_stereo.signature(self.mol, i), ref, spec)]
        logger.info("select_stereo[%s]: %d -> %d (chirality-matched)", spec, len(self.ids), len(keep))
        return self._derive(keep)

    def lowest(self, n=1, refine=None, solvent=None):
        """Return the `n` lowest-energy conformers as a new Ensemble (n=1 -> the single best geometry).

        Ranks on the force-field energy by default; pass ``refine='gxtb'`` (or ``'gfn2'``) to rank on a real
        **xTB single point** instead (geometry untouched -- see `score`).
        """
        src = self.score(refine, solvent) if refine else self
        src.minimize()
        return src._derive(sorted(src.ids, key=lambda i: src.energies.get(i, 0.0))[:n])

    def representatives(self, *, min_cluster=3, nci=True, recover_noise="auto", noise_window=10.0):
        """Return one lowest-energy conformer per **mode** as a new Ensemble (the distinct-shapes summary).

        A "mode" is whatever the latent distinguishes (`dedup.active_blocks`): a conformer family (organic),
        a contact pattern (NCI), or a ligand arrangement (metal). Sorted by energy; folded conformers stay
        in `ens.ids`.

        `recover_noise` handles HDBSCAN noise (label -1): ``"auto"`` (default) recovers a noise conformer
        only if its discrete `mode_signature` is a genuinely new binding mode (all noise folded for a plain
        organic); ``True`` keeps every noise conformer; ``False`` drops them. A recovered mode must sit
        within `noise_window` kcal/mol of the minimum.
        """
        self.minimize()
        if not self.ids:  # infeasible isomer / nothing embedded -> empty summary
            return self._derive([])
        kind = _dedup.mode_kind(self.mol, self.ids, nci=nci)
        labels = _dedup.cluster_labels(self.mol, self.ids, min_cluster=min_cluster, nci=nci).tolist()

        def by_e(i):
            return self.energies.get(i, 0.0)  # 0.0 when the relax was skipped (untypable core)

        e_min = min(by_e(i) for i in self.ids)
        mode_reps = [
            min((self.ids[k] for k in range(len(labels)) if labels[k] == lab), key=by_e)
            for lab in sorted({lab for lab in labels if lab != -1})
        ]
        noise_k = [k for k in range(len(labels)) if labels[k] == -1]
        recovered, skipped_e = [], 0
        if noise_k and recover_noise is not False:
            sigs = None if recover_noise is True else _dedup.mode_signature(self.mol, self.ids, nci=nci)
            if recover_noise is True:
                recovered = [self.ids[k] for k in noise_k]
            elif sigs is not None:  # one rep per noise signature not already clustered
                shown = {sigs[k] for k in range(len(labels)) if labels[k] != -1}  # ALL members, not just reps
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

    def measure(self, atoms):
        """Mean and range of a geometric measurement over the whole ensemble.

        The one-liner behind "is the constraint held?" / "how much does this vary?". `atoms` (by index or
        SMARTS) gives a **distance** (2 atoms), **angle** in degrees (3), or **dihedral** in degrees (4).
        Returns ``{'mean', 'min', 'max', 'n'}``.

            ens.measure((11, 14))           # d(11,14): {'mean': 2.60, 'min': 2.50, 'max': 2.65, 'n': 213}
            ens.measure(('[OX2H]', 'n'))    # a constrained contact, by SMARTS
        """
        if not self.ids:
            raise ValueError(
                "measure() on an empty ensemble — no conformers to measure (embed/minimize "
                "may have produced none; check the log)"
            )
        idx = [resolve_atom(self.mol, a) for a in atoms]
        fns = {2: rdMolTransforms.GetBondLength, 3: rdMolTransforms.GetAngleDeg, 4: rdMolTransforms.GetDihedralDeg}
        if len(idx) not in fns:
            raise ValueError("measure() takes 2 (distance), 3 (angle) or 4 (dihedral) atoms")
        f = fns[len(idx)]
        v = [f(self.mol.GetConformer(c), *idx) for c in self.ids]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}

    def _align_atoms(self, on):
        if on is None:  # default core: constrained atoms > metal coordination > heavy
            core = sorted(self.cons.constrained_atoms())
            if len(core) >= _MIN_OVERLAY_ATOMS:
                return core
            from .dedup.features import _metal_donors, _metal_present

            if _metal_present(self.mol):  # a metal's natural overlay core is M + its donors (even when
                m, donors = _metal_donors(self.mol, self.ids)  # cons is empty, e.g. a wrapped metal mol)
                if donors:
                    return [m, *donors]
            return [a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1]
        if isinstance(on, str):
            return list(match(self.mol, on))
        return [resolve_atom(self.mol, a) for a in on]

    def align(self, on=None):
        """Return a new Ensemble with every conformer Kabsch-superposed for a readable overlay.

        The original is untouched (looking never mutates your geometries). Aligns on the constrained core by
        default (a TS's frozen atoms sit still, the rest shows its variation); pass `on=` atom
        indices/SMARTS to align on something else.
        """
        m = Chem.Mol(self.mol)
        if len(self.ids) > 1:
            rdMolAlign.AlignMolConformers(m, atomIds=self._align_atoms(on), confIds=list(self.ids))
        return self._derive(self.ids, mol=m)

    # -- looking (returns an artifact, never mutates) -------------------------
    # 3D structure rendering is notebook-level: align()/dump() give you the geometry, then a couple of
    # lines of py3Dmol (interactive) or xyzrender (publication SVG) in the notebook. Only `landscape` (the
    # ensemble map — diversity, kept/dropped) lives here, because the dim-reduction is real reusable work.

    @property
    def n(self):
        """The number of conformers in play."""
        return len(self.ids)

    def dump(self, path, align=True):
        """Write the current conformers as a multi-frame **.xyz** (one frame per tracked id).

        Dump at any pipeline stage (``embed(...).dump('seeds.xyz')``, ``.mc().dump('mc.xyz')``,
        ``.prune().dump(...)``) or from any derived / filtered ensemble (``representatives()``, ``lowest()``,
        ``align()``, or ``wrap(ens.mol, ens.discarded)`` for the pruned-away ones). Real element symbols are
        written even before ``minimize()`` restores a metal surrogate.

        By default the frames are **aligned** (Kabsch-superposed on the rigid core — the constrained atoms,
        else a metal's coordination sphere, else all heavy atoms — via the same selector as ``align()``) so
        they overlay for viewing; pass ``align=False`` for raw embed-frame coordinates. Works on a **copy**,
        so the live ensemble's geometry is never moved. Returns the path.
        """
        mc = self._metal
        real = [(mc.metal, mc.real_z), *mc.extra] if mc is not None else []
        mol = Chem.Mol(self.mol)  # a copy — dump never moves the live ensemble
        for mi, rz in real:  # show the real metal(s), not the C surrogate
            mol.GetAtomWithIdx(mi).SetAtomicNum(rz)
        if align and len(self.ids) > 1:  # overlay frames on the rigid core
            try:
                rdMolAlign.AlignMolConformers(mol, atomIds=self._align_atoms(None), confIds=list(self.ids))
            except Exception as e:
                logger.debug("dump: alignment skipped (%s)", e)
        with open(path, "w") as f:
            for i in self.ids:
                f.write(Chem.MolToXYZBlock(mol, confId=i))
        return path

    write = dump  # alias

    def __len__(self):
        """Return the number of conformers in play."""
        return len(self.ids)

    def __repr__(self):
        """Summarise the ensemble: conformer count, state, and energy spread."""
        n = len(self.ids)
        es = [self.energies[i] for i in self.ids if i in self.energies]
        de = f", ΔE 0.00-{max(es) - min(es):.2f} kcal/mol" if len(es) > 1 else ""
        state = "minimized" if self._minimized else "embedded"
        tag = f", {self.tag}" if self.tag else ""
        return f"<Ensemble: {n} conformer{'s' if n != 1 else ''}, {state}{de}{tag}>"
