"""The user-facing pipeline: one `embed()` entry returning a chainable `Ensemble`.

    import rxembed as rx
    ens = rx.embed("CCO").mc().prune()
    ens = rx.embed("ts.xyz", fix=[11, 14, 15]).mc().prune()   # hold a TS core (0.000 A graft)

Sanity, vdW separation, the metal surrogate swap/restore and constraint validation all happen
inside. Every stage logs (`rxembed.set_verbose()`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

from rxembed import bounds as _bounds
from rxembed import metal_core as _metal
from rxembed import metal_polyhedron as _poly
from rxembed.constraints import match, resolve_atom
from rxembed.embed import BOND_TOL as _BOND_TOL
from rxembed.embed import DISTANCE_FC as _DISTANCE_FC
from rxembed.embed import METAL_BOND_TOL as _METAL_BOND_TOL
from rxembed.embed import Conformers
from rxembed.relax import MAX_ITERS as _MAX_ITERS

from . import calculators as _refine
from . import geom_check as _geometry
from . import metrics as _metrics
from . import nci as _nci
from . import search as _mc
from . import select as _dedup
from . import stereo_check as _stereo

logger = logging.getLogger("rxembed")

_MIN_OVERLAY_ATOMS = 3  # need >=3 atoms to define an alignment frame
_HARTREE_KCAL = 627.5094740631  # Eh -> kcal/mol
# Å slack over a rigid body's pairwise windows before it counts as torn; on top of hold_shape's own 0.1 pad,
# so a body must be a clear 0.2 Å out of shape to be rejected.
_SHAPE_TEAR_TOL = 0.10
# kcal/mol above the ensemble min past which a relax never converged (real is 1e3-1e11, orders past any real
# rotamer): it passed the loose bond/coplanarity gate but is geometrically broken. Drop it.
_RELAX_ENERGY_WINDOW = 250.0
# a torn ETKDG seed is a failure to RE-EMBED (a fresh seed), not to re-relax (same seed tears the same way),
# and shows only after the FF relax. So minimize re-embeds a metal up to _MAX_MIN_ROUNDS until N are
# geom.check-clean; the bonding-ok fallback is kept only if clean is unreachable.
_MAX_MIN_ROUNDS = 5
_EMBED_SEED = 0xF00D  # initial-embed seed; retry rounds step off it for a distinct seed each
_EMBED_BUFFER = 2  # over-embed a couple extra per round to cover that round's own tear rate


def _last_line(err):
    """Return the last non-empty line of an exception (xtb dumps a stderr tail) for a one-line warning."""
    s = str(err).strip()
    return s.splitlines()[-1] if s else "no output"


class EnsembleSet(list):
    """Several candidate `Ensemble`s to choose from (metal isomers, or ambiguous ``coordinate=`` donors).

    Each is ``.tag``ged by what makes it distinct.

        cands = rx.embed('CCCN[Pd](Cl)Cl', metal='square_planar'); cands.summary()
        ens   = cands.select(label='trans').mc().prune()

    `select(**tag)` returns the single match; iterate or index to keep them all.
    """

    def __repr__(self):
        """Summarise the candidates and their tags on one line."""
        labels = [e.tag.get("label", e.tag) for e in self]
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
        """Subset the candidates by `tag`, or with `by=` drop reacted conformers within each candidate.

        Told apart by how it is called:

            set.filter(geometry='square_planar')   # keep the candidates matching this tag -> EnsembleSet
            set.filter('connectivity')             # drop conformers whose graph changed, in every candidate
        """
        if by is not None:
            return self._map("filter", by, **tag)
        # a geometry is nameable by its 3-letter code everywhere else (`rx.metal`, `IsomerSet.filter`), so it
        # must be here too, or filter(geometry='SPL') silently returns nothing
        tag = {k: (_poly.resolve_geometry(v) if k == "geometry" else v) for k, v in tag.items()}
        return EnsembleSet(e for e in self if all(e.tag.get(k) == v for k, v in tag.items()))

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
        """Apply an `Ensemble` verb to every candidate, returning a new `EnsembleSet` (tags carried over).

        Each candidate is searched/pruned/scored on its own constraints; distinct species are never pooled or
        cross-pruned. To rank them, use `score`/`optimize` (a real calculator) then `best`.
        """
        tags = ", ".join(self._slug(e.tag or {}, k) for k, e in enumerate(self)) or "?"
        logger.info(
            "%s: %d candidate(s) [%s], each on its own constraints%s",
            method.lstrip("_"),
            len(self),
            tags,
            "; compare across them only by these real energies" if method in self._REAL_ENERGY_VERBS else "",
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
        """Rank the candidates by their lowest energy and keep the best `n`.

        The one place distinct species are ranked against each other, so it demands real energies (raises
        unless every candidate was `score`d / `optimize`d). Returns the winning `Ensemble` (``n==1``) or an
        `EnsembleSet` of the best `n`.
        """
        not_real = [self._slug(e.tag or {}, k) for k, e in enumerate(self) if e.energy_kind != "real"]
        if not_real:
            raise ValueError(
                f"best() ranks distinct species by real energy, but {not_real} have none (FF surrogate energies "
                f"are not comparable across species); run .score('gxtb') or .optimize('gxtb') on the set first"
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
        """Refuse a scalar `Ensemble` verb with the fix named, instead of a bare `AttributeError`.

        A set holds distinct species (stereoisomers, metal isomers, NCI grips), so `.mol` / `.ids` /
        `.relative()` have no single answer. The verbs that do map (`mc`, `prune`, `score`, …) are defined
        above; anything else lands here, most often because an undefined stereocentre split a plain
        `rx.embed(smiles)`.
        """
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
        keys = ("geometry", "stereo", "label", "nci", "chirality")
        s = "_".join(str(tag[key]) for key in keys if tag.get(key)) or f"c{k}"
        return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in s)

    def dump(self, path, align=True):
        """Write each candidate to its own multi-frame .xyz, its tag folded into the filename; return the paths.

        ``rx.embed('CC(N)C(=O)O').dump('amac.xyz')`` -> ``amac_1S.xyz`` + ``amac_1R.xyz``. Same `align`
        semantics as `Ensemble.dump`.
        """
        from pathlib import Path

        base = Path(path)
        paths = []
        for k, e in enumerate(self):
            p = base.with_name(f"{base.stem}_{self._slug(e.tag or {}, k)}{base.suffix or '.xyz'}")
            e.dump(str(p), align=align)
            paths.append(str(p))
        return paths

    write = dump  # alias, matching Ensemble


@dataclass
class Ensemble(Conformers):
    """A conformer ensemble: the core `Conformers` result plus the pipeline verbs.

    Inherits `Conformers`' fields and `.xyz` / `len()`, narrows `.mol` / `.minimize` / `.dump` /
    `.__getitem__` (each documented at its override), and adds the search / dedup / real-energy stages.
    `ens.energies` maps id -> energy of the kind `energy_kind` names, `ff` or `real`.

    The `_mol` / `.mol` split matters for metals: `_mol` is the bond-less surrogate every DG/FF stage needs,
    because UFF cannot type a bonded transition metal, while `.mol` finalizes the connected graph on access
    (real element, oxidation state, dative M-L bonds). Internal consumers read `_mol`.

    `iso` is the live metal context and does not outlive `minimize()`; `sphere` / `metal_bonds` / `tag` are
    the durable record after that.

    Which verbs change this object, and which hand back a new one:

    - change it and return it, so they chain: `mc`, `minimize`, `prune`, `filter`
    - return a new Ensemble: `lowest`, `representatives`, `align`, `score`, `optimize`, `select_stereo`
    - change nothing: `measure`, `landscape`, `cluster`, `binding_modes`
    """

    _minimized: bool = False
    _seeds_relaxed: bool = False  # embed already relaxed these into their windows, so minimize single-points
    discarded: list = field(default_factory=list)  # conformer ids a stage dropped (still in `mol`)
    tag: dict = field(default_factory=dict)  # what distinguishes this candidate in an EnsembleSet
    _stereo: tuple | None = None  # (spec, reference signature) for the minimize() chirality filter
    _donor_hand: dict = field(default_factory=dict)  # {labile donor: target signed-volume hand}; minimize()
    # culls any conformer whose metal-bound C/N donor inverted
    energy_kind: str = ""  # "" | "ff" (surrogate, not cross-species) | "real" (xtb); best() ranks only "real"
    reacted: dict = field(default_factory=dict)  # {conf id: (formed, broken)} where a stage changed the graph
    sphere: dict = field(default_factory=dict)  # {metal: [donors]}, durable past `iso`: how
    # `coordination_changed` tells a dissociated ligand from a healthy one
    metal_bonds: list = field(default_factory=list)  # stripped M-donor bonds, re-added dative by minimize();
    # durable past `iso`, so a re-minimize after mc re-connects too

    @property
    def mol(self):
        """The user-facing molecule: always a proper connected graph, finalized on access from `_mol`.

        Returns the connected graph the input had (real element + oxidation state + dative M-donor bonds) at
        any stage, via `restore_metal` + `connect_metal` on a copy of a pre-minimize metal complex. Organic
        and post-minimize inputs are the live `_mol`. This narrows `Conformers.mol`, which always copies: a
        wrapped or minimized ensemble hands back the mol its caller already holds, so editing it does reach
        the ensemble. Copy it yourself (``Chem.Mol(ens.mol)``) before mutating.

        Uncached: every access re-runs the finalize for a pre-minimize metal complex, so a hot loop should
        bind it once (`m = ens.mol`).
        """
        mol, iso = self._mol, self.iso
        bonds = self.metal_bonds or (list(iso.donor_bonds) if iso is not None else [])
        if iso is not None:  # pre-minimize: `_mol` still carries the carbon surrogate and no M-L bonds. Restore
            mol = iso.restore(Chem.Mol(mol))  # real element(s)/charge on a copy, never the working surrogate,
            # then connect below; `minimize` finalizes in that same order
        return _metal.connect_metal(mol, bonds) if bonds else mol  # connect_metal is idempotent (no-op if bonded)

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
            logger.warning("mc: openconf not installed; skipped")
            return self
        if preset == "rapid" and self.cons.is_constrained:  # constrained pose-mode is rotor-only, so rapid
            logger.warning("mc: preset='rapid' under-samples a constrained system; use the default 'ensemble'")
        if self.cons.is_constrained:
            logger.info(
                "mc: %d atom(s) constrained -> pose-frozen search (rotor-only; low-mode/ring/global moves off)",
                len(self.cons.constrained_atoms()),
            )

        if self.metal_bonds:  # a prior minimize() left the output connected; UFF can't type a bonded metal, and
            self._mol = _metal.disconnect_metal(self._mol)  # minimize re-connects afterwards
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
        self._minimized = self._seeds_relaxed = False  # openconf's geometries are its own FF's, so a follow-up
        # minimize() must genuinely re-relax them

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
                    "mc explore: +%d relaxed conformer(s) (NCI contacts released; %d structural/"
                    "encounter hold(s) kept; total %d)",
                    len(more),
                    len(relaxed.distances),
                    len(self.ids),
                )
        return self

    def _settle_seeds(self, bins=5):
        """Pull the seeded (non-frozen) distance/angle constraints inside their windows before an MC search.

        Spread across the window over the conformers (each bin targets a different fraction) rather than
        collapsed to one value, so the search keeps its breadth. Frozen atoms and the frozen-core shape are
        held exactly. Fixes the raw ETKDG seed sitting outside a tight window.
        """
        cons = self.cons
        seeded_d = [(k, v) for k, v in cons.distances.items() if not (k[0] in cons.frozen and k[1] in cons.frozen)]
        shape_d = {k: v for k, v in cons.distances.items() if k[0] in cons.frozen and k[1] in cons.frozen}
        if not (seeded_d or cons.angles) or not self.ids:
            return
        for b, group in enumerate(g for g in np.array_split(list(self.ids), min(len(self.ids), bins)) if len(g)):
            frac = (b + 0.5) / min(len(self.ids), bins)  # this bin's fraction across every window
            # A partial rebuild: `distances`/`angles` are re-derived per bin, everything else carried. Critically
            # `coplanar`, without which the stiff relax drives the metal out of the donor plane.
            tgt = cons.copy(distances={}, angles={}, contacts=(frozenset(), frozenset()), dg_floors={})
            tgt.distances.update(shape_d)  # keep the frozen-core shape exact
            for k, (lo, hi) in seeded_d:
                m = lo + frac * (hi - lo)
                tgt.distances[k] = (max(lo, m - 0.03), min(hi, m + 0.03))  # clamp inside the user window
            for k, (lo, hi) in cons.angles.items():
                m = lo + frac * (hi - lo)
                tgt.angles[k] = (max(lo, m - 2.0), min(hi, m + 2.0))
            try:  # settling is best-effort pre-conditioning; an RDKit BFGS divergence must not sink mc()
                _refine.restrained_uff(self._mol, tgt, distance_fc=_DISTANCE_FC, conf_ids=[int(i) for i in group])
            except RuntimeError:  # keep the raw ETKDG seeds for this group (openconf still searches around them)
                logger.debug("mc: seed-settle relax diverged on a group; keeping the raw seeds")

    def _shape_intact(self, cid, tol=_SHAPE_TEAR_TOL):
        """Return False if a rigid body (``cons.shapes``) came out torn.

        A `hold_shape` body (a retained or spectator coordination sphere) is pinned by all C(n,2) of its
        pairwise input distances: one object whose internal geometry is the input, not a preference. Nothing
        else in the accept gate checks those windows (`bonding_ok` skips metal pairs; `check_constraints`
        only logs), so a torn body would otherwise survive.
        """
        if not self.cons.shapes:
            return True
        pos = self._mol.GetConformer(cid).GetPositions()
        for body in self.cons.shapes:
            for (i, j), (lo, hi) in self.cons.distances.items():
                if i in body and j in body:
                    d = float(np.linalg.norm(pos[i] - pos[j]))
                    if d < lo - tol or d > hi + tol:
                        logger.debug("minimize: conformer %d rejected, rigid shape torn at (%d,%d)", cid, i, j)
                        return False
        return True

    def _relax_into_windows(self):
        """Relax the raw ETKDG seeds into their own constraint windows, in place; what `embed` returns through.

        A raw seed does not satisfy its constraints: measured, angle windows missed by 9.5° mean / 42.8° max,
        an organic window by 0.21-0.43 Å, and 17/20 square-planar crystals coming out tetrahedral. This relax
        enforces them, moving toward the crystal on every local axis (M-donor MAE 0.063 -> 0.006 Å) and
        neutral on global RMSD.

        It composes the core's stiffness ladder and torn-seed rescue (`Conformers.minimize`) by hand rather
        than calling it, for two reasons. The metal must not be restored here: doing it at embed time strips
        the rest of the chain of the coplanarity gate, donor-hand hold and re-embed (measured on the N-bound Ni,
        11/11 clean falls to 4/5). And `embed` publishes no energy, since `minimize` owns `energy_kind`. So
        `iso` stays live, `_minimized` stays unset, and `.energies` stays empty.

        Unconstrained embeds are left alone: no window, and no unbidden FF pass.
        """
        if not self.ids or not self.cons.is_constrained:
            return self
        seed_pos = {c: self._mol.GetConformer(c).GetPositions() for c in self.ids}
        self._relax_constrained(_DISTANCE_FC)
        # The relax can TEAR a seed; `embed` must not spend the caller's `n` re-embedding (that is `minimize`'s
        # job), so a torn conformer keeps its seed coordinates: output is never worse than the seed.
        self._rescue_torn(seed_pos, _DISTANCE_FC)
        self._seeds_relaxed = True
        return self

    def minimize(self, distance_fc=_DISTANCE_FC, max_iters=_MAX_ITERS, _retry=True):
        """FF relax (stiff restrained UFF if constrained, else MMFF), restore any metal, drop clashes, validate.

        `distance_fc` / `max_iters` are `Conformers.minimize`'s, and this is that relax plus the pipeline's
        gates: every conformer one rejects (torn bond, puckered sphere, non-physical relax energy, inverted
        donor hand, wrong stereo) is appended to ``discarded`` and, like prune's, stays in ``mol``.

        Two narrowings of the base verb: it is idempotent (a second call is a no-op until `mc` clears the
        flag), and it consumes `iso`, restoring the metal for good so the identity lives on in `sphere` /
        `metal_bonds` / `tag`.
        """
        if self._minimized:
            return self
        if not self.ids:  # nothing embedded (e.g. ETKDG could not place this graph)
            self._minimized = True
            return self
        iso = self.iso  # capture before restore: the coordination-planarity gate needs metal + donors
        target = len(self.ids)  # embed(n=N) means N good geometries; the retry re-embeds if the relax tears some
        # a pristine surrogate to re-embed on
        template = Chem.Mol(self._mol) if (_retry and iso is not None) else None
        self._relax_and_record(distance_fc, max_iters)
        if iso is not None:
            if iso.donors:  # remember the sphere: `iso` goes, but who coordinates whom is durable
                self.sphere.setdefault(iso.metal, list(iso.donors))
            iso.restore(self._mol)
            self.iso = None
        before = list(self.ids)  # ids, not a count: what minimize drops is recorded in `discarded` like prune's
        drops = self._drop_bad_geometries(iso)  # torn bond / puckered sphere / torn rigid body
        self._drop_unconverged(drops)  # non-physical relax energy
        if len(self.ids) < len(before):
            logger.info(  # name the gate(s) that fired, with counts
                "minimize: dropped %d conformer(s) (%s) -> %d kept",
                len(before) - len(self.ids),
                ", ".join(f"{n}x {why}" for why, n in drops.items() if n),
                len(self.ids),
            )
        if template is not None:  # a metal seed the relax tore is a bad embed, not a bad relax: re-embed fresh
            self._reembed_until_clean(template, iso, target, distance_fc, max_iters)  # until N are geom.check-clean
        self._cull_inverted_donors()  # a labile donor that inverted vs its enumerated hand
        if not self.ids:
            logger.warning(  # `embed` relaxes, so this fires at embed time: state the gates, not a guess
                "minimize: 0 of %d conformer(s) survived the relax (%s); the arrangement may be infeasible",
                len(before),
                ", ".join(f"{n}x {why}" for why, n in drops.items() if n) or "no gate recorded",
            )
        self._cull_wrong_stereo()  # embed-uncapturable handedness (metallocene planar / axial / helical)
        self.discarded += [i for i in before if i not in set(self.ids)]
        self._validate()
        if iso is not None and not self.metal_bonds:  # remember the stripped M-L bonds durably (past `iso`):
            self.metal_bonds = list(iso.donor_bonds)  # a re-minimize after mc has no `iso` but must re-connect
        if self.metal_bonds:  # connectivity finalize, last: geometry/element/charge now settled, so re-add the
            # surrogate-stripped M-donor bonds as dative. Every gate above saw the bond-less surrogate.
            self._mol = _metal.connect_metal(self._mol, self.metal_bonds)
        self._minimized = True
        return self

    def _relax_and_record(self, distance_fc, max_iters):
        """FF-relax the conformers (restrained UFF if constrained, else MMFF) and store the surrogate energies."""
        e = None
        if self.cons.is_constrained:
            if self._seeds_relaxed:  # embed already relaxed these into their windows; relaxing again only rides
                # the flat-bottomed walls further (measured: primary-phosphine M-P-H median 129.47 -> 130.00,
                # the splay cap's exact bound). Take a single point on the same FF; the gates below are owed.
                try:
                    e = _refine.restrained_uff(self._mol, self.cons, distance_fc=distance_fc, max_iters=0)
                except RuntimeError as err:  # mirror _relax_constrained's guard so both relax entry points
                    # degrade identically: an untypable/hypervalent core keeps its embedded geometry
                    logger.warning(
                        "minimize: UFF could not relax this system (%s); keeping the embedded geometry",
                        err,
                    )
            else:
                e = self._relax_constrained(distance_fc, max_iters)  # escalates if the soft relax tears every bond
        else:
            e = _refine.ff_energies(self._mol, minimize=True)
        if e is not None:
            self.energies = {c.GetId(): float(e[k]) for k, c in enumerate(self._mol.GetConformers())}
            self.energy_kind = "ff"  # surrogate FF, not comparable across species (EnsembleSet.best refuses it)

    def _drop_bad_geometries(self, iso):
        """Drop each conformer that broke a bond, puckered a planar sphere, or tore a rigid body; return the counts."""
        # A metal keeps the looser bond tol: a slightly-stretched bond in a coplanar coordination is a
        # surrogate artifact xtb recovers, and the coplanarity gate already rejects the phantom ones.
        bt = _METAL_BOND_TOL if iso else _BOND_TOL
        # name the polyhedron and the number in the reason: "out-of-plane" alone says neither which shape
        # declared itself planar nor what it was measured against
        oop = "out-of-plane coordination sphere"
        if iso is not None and _poly.is_planar(iso.geometry):
            oop = f"{oop} ({_poly.describe(iso.geometry)} is declared planar; RMS > {_metal.COPLANAR_TOL} A)"
        kept, drops = [], {"broken bond": 0, oop: 0, "torn rigid body": 0}
        for i in self.ids:  # one pass; record which gate rejected each so the log can name it
            if not _metrics.bonding_ok(
                self._mol, i, bond_tol=bt, exclude=self.cons.frozen, constrained=self.cons.distances
            ):
                drops["broken bond"] += 1
            elif not self._coordination_ok(i, iso):
                drops[oop] += 1
            elif not self._shape_intact(i):
                drops["torn rigid body"] += 1
            else:
                kept.append(i)
        self.ids = kept
        self._warn_shape_flattened(iso)
        return drops

    def _warn_shape_flattened(self, iso):
        """Warn when a kept conformer of a non-planar polyhedron relaxed flat, so will re-perceive as another shape.

        Not a drop, unlike its mirror above. A planar declaration is a feasibility claim, so violating it means
        the arrangement was impossible; flattening a pyramid is not, the surrogate FF having no lone pair and no
        d-electron preference, and the ±8° window being flat-bottomed so the relax rides its upper wall. Dropping
        those would return an empty ensemble for a shape the user asked for.

        `mechanisms.Umbrella` holds the pyramid, so this is now the check that the hold worked. It stays because
        the hold is FF-only: `score`/`optimize` hand xtb nothing but `cons.frozen`, and a real energy flattens 3
        of 4 flagship cases. It is also the only signal on the paths `Umbrella` skips -- a κ3 base of one ligand,
        a vacant vertex, a haptic face, a `fix=` core.

        The defect it prevents is silence: the `Isomer` printing the requested name while a re-perception of the
        dumped .xyz gives another.
        """
        if iso is None or not iso.donors or _poly.is_planar(iso.geometry):
            return
        flat = [
            i
            for i in self.ids
            if _metal.coplanar(self._mol.GetConformer(i).GetPositions(), iso.metal, iso.donors, haptic=self.cons.haptic)
        ]
        if flat:
            logger.warning(
                "minimize: %d of %d conformer(s) of %s relaxed flat (metal <%.2f A RMS from its donor plane)",
                len(flat),
                len(self.ids),
                _poly.describe(iso.geometry),
                _metal.COPLANAR_TOL,
            )

    def _drop_unconverged(self, drops):
        """Drop conformers whose relax energy sits above the window: a non-physical, un-converged geometry."""
        if not (self.energies and self.ids):
            return
        emin = min(self.energies[i] for i in self.ids if i in self.energies)  # geometry, not a real rotamer
        rel = sorted(self.energies[i] - emin for i in self.ids if i in self.energies)
        over = [round(r, 1) for r in rel if r > _RELAX_ENERGY_WINDOW]
        logger.info(
            "minimize: relax ΔE spread (kcal/mol above min): %s | window=%.0f%s",
            [round(r, 1) for r in rel],
            _RELAX_ENERGY_WINDOW,
            f" -> dropped {len(over)} un-converged: {over}" if over else "",
        )
        drops["non-physical relax energy"] = len(over)
        self.ids = [i for i in self.ids if self.energies.get(i, emin) <= emin + _RELAX_ENERGY_WINDOW]

    def _cull_inverted_donors(self):
        """Cull any conformer whose labile metal-donor hand inverted vs the enumerated (uniform-embed) hand."""
        for d, target_sign in self._donor_hand.items():
            if target_sign is None:  # a rare relax/re-embed/mc stray that escaped the dummy hold
                continue
            keep = [i for i in self.ids if _metal.donor_chirality_sign(self._mol, i, d) == target_sign]
            if 0 < len(keep) < len(self.ids):
                logger.info(
                    "minimize: culled %d conformer(s) whose donor-%d hand inverted", len(self.ids) - len(keep), d
                )
                self.ids = keep

    def _cull_wrong_stereo(self):
        """Keep only the requested handedness on chirality the embed can't (metallocene planar / axial / helical)."""
        if not (self._stereo and self.ids):
            return
        spec, ref = self._stereo
        kept = [i for i in self.ids if _stereo.satisfies_spec(_stereo.signature(self._mol, i), ref, spec)]
        if len(kept) < len(self.ids):
            logger.info("minimize: stereo=%r kept %d/%d (matched the input handedness)", spec, len(kept), len(self.ids))
        self.ids = kept
        self._stereo = None

    def _reembed_until_clean(self, template, iso, target, distance_fc, max_iters):
        """Re-embed fresh seeds until `target` conformers pass acceptance, merging the good ones in.

        The raw embed is clean and the FF relax tears a bad seed, so the fix is a fresh seed, never re-relaxing
        the torn one. Acceptance is geom.check-clean and, if ``stereo=`` is set, the requested handedness, so
        the retry count survives the later stereo cull. Good geometries are preferred and placed first; the
        bonding-ok fallback is kept only where good is unreachable. Metal-only.

        The intended donors (from `sphere` plus this run's `iso`) are handed to the gate, because both
        in-sphere checks are circular without them: a collapsed ligand is perceived as coordinating and so
        exempts itself.
        """
        donors = sorted({int(d) for ds in self.sphere.values() for d in ds} | {int(d) for d in iso.donors or ()})

        def is_good(mol, cid):  # geom.check-clean, and the requested handedness when a stereo spec is active
            if not _geometry.check(mol, cid, donors=donors or None).ok():  # structural gates only; a metal-donor
                return False  # distance is an approximate surrogate target, not a hard accept bar (see _validate)
            if self._stereo:
                spec, ref = self._stereo
                return _stereo.satisfies_spec(_stereo.signature(mol, cid), ref, spec)
            return True

        good = lambda: [i for i in self.ids if is_good(self._mol, i)]  # noqa: E731
        seed = _EMBED_SEED
        for _ in range(_MAX_MIN_ROUNDS):
            have = good()
            if len(have) >= target:
                break
            seed += 1  # a distinct seed each round: the point is to regenerate the embed, not repeat it
            tmpl = Chem.Mol(template)
            tmpl.RemoveAllConformers()
            # re-apply the labile-donor chirality hold: a fresh ETKDG seed is random-handed, so without it a
            # carbanion/amine donor's enumerated hand would silently invert. The batch relax below re-holds it.
            tmpl, held = _metal._hold_donor_chirality(tmpl, iso.metal, iso.donors, self.cons) if iso else (tmpl, [])
            # the one embed the pipeline hand-rolls rather than calling the core seam: `seed_conformers` would
            # also update `self.cons` with encounter bounds (shared, reused by every later stage) and graft the
            # frozen core onto these retry seeds, both behaviour changes
            new_ids = _bounds.embed(tmpl, self.cons, target - len(have) + _EMBED_BUFFER, seed=seed)
            if not new_ids:
                continue
            tmpl = _metal._release_donor_chirality(tmpl, held, self.cons)  # drop dummy + cons key; batch relax re-holds
            batch = Ensemble(tmpl, new_ids, self.cons, iso).minimize(distance_fc, max_iters, _retry=False)
            for cid in batch.ids:  # merge only good ones; the fallback already holds the bonding-ok geometries
                if not is_good(batch._mol, cid):
                    continue
                nid = self._mol.AddConformer(Chem.Conformer(batch._mol.GetConformer(cid)), assignId=True)
                self.ids.append(nid)
                if cid in batch.energies:  # never fabricate a 0.0: an absent energy stays absent
                    self.energies[nid] = batch.energies[cid]
        kept = good()
        if kept:  # good first, then the bonding-ok fallback, capped at target (embed(n=N) hands back N)
            self.ids = (kept + [i for i in self.ids if i not in kept])[:target]
        logger.info("minimize: re-embedded to %d/%d clean geometries", len(kept), target)

    def _validate(self, dist_slack=0.15, ang_slack=5.0):
        """Warn if a constraint the relax can enforce is not realised within its window, plus a small slack.

        Only constraints with a non-frozen atom are warned. A constraint between two frozen atoms is a
        structural hold (frozen-core shape, spectator sphere) held by the graft or ``AddFixedPoint``, not the
        relax, so an off-window value there is logged at DEBUG.
        """
        if not self.ids:
            return
        frozen = self.cons.frozen
        # a metal-donor distance or L-M-L angle is an approximate bias (a covalent-radius guess; the real value
        # is the calculator's, and the surrogate vdW legitimately pushes donors out), so off-window is DEBUG
        metals = {a.GetIdx() for a in self._mol.GetAtoms() if a.GetAtomicNum() in _metal.TRANSITION_METALS}
        n = self._mol.GetNumAtoms()  # a haptic centroid dummy is transient (index >= n); its real counterpart

        def realised(fn, *atoms):
            return float(np.mean([fn(self._mol.GetConformer(c), *atoms) for c in self.ids]))

        for (i, j), (lo, hi) in self.cons.distances.items():
            if i >= n or j >= n:  # (M -> each ring atom) is a real-indexed distance validated in this same loop
                continue
            d = realised(rdMolTransforms.GetBondLength, i, j)
            if lo - dist_slack <= d <= hi + dist_slack:
                logger.debug("held d(%d,%d) = %.2f (target %.2f-%.2f)", i, j, d, lo, hi)
            elif i in frozen and j in frozen:  # structural hold among pinned atoms, not the relax's to enforce
                logger.debug("frozen-shape d(%d,%d) = %.2f (embed approx of %.2f-%.2f)", i, j, d, lo, hi)
            elif i in metals or j in metals:  # an approximate coordination distance; the real value is the calc's
                logger.debug("metal-donor d(%d,%d) = %.2f (embed bias; target %.2f-%.2f)", i, j, d, lo, hi)
            else:
                logger.warning("constraint not held: d(%d,%d) = %.2f, target %.2f-%.2f", i, j, d, lo, hi)
        for (i, j, k), (lo, hi) in self.cons.angles.items():
            if i >= n or j >= n or k >= n:  # a constraint on a transient centroid dummy (see the distance loop)
                continue
            a = realised(rdMolTransforms.GetAngleDeg, i, j, k)
            if lo - ang_slack <= a <= hi + ang_slack:
                continue
            if (i in frozen and j in frozen and k in frozen) or j in metals:  # frozen shape / approx L-M-L angle
                logger.debug("metal/frozen angle(%d,%d,%d) = %.1f (target %.1f-%.1f)", i, j, k, a, lo, hi)
            else:
                logger.warning("constraint not held: angle(%d,%d,%d) = %.1f, target %.1f-%.1f", i, j, k, a, lo, hi)

    def score(self, refine="gxtb", solvent=None, charge=None):
        """Re-rank by a real single-point energy (g-xTB by default), returning a new Ensemble.

        The FF/surrogate-UFF energy is meaningless for a metal or charged TS. No geometry change of its own:
        it scores the geometry `minimize()` produced, with a constrained core held exactly. Energies are in
        kcal/mol (see `relative` for the unit contract). `charge` defaults to the formal charge; `refine` is
        'gxtb'/'gfn2'/'ff'/a Calculator; `solvent` adds a GFN2-ALPB correction. xTB is seconds per conformer,
        so call `representatives()` / `lowest(k)` first on a big ensemble.
        """
        self.minimize()  # settle the periphery (a constrained core stays pinned)
        if not self.ids:
            return self._derive([])
        from .calculators import resolve

        q = self._calc_charge(charge)
        calc = resolve(refine, solvent, q)
        if calc is None:  # 'ff'/None -> single-point force field, geometry kept
            e = _refine.ff_energies(self._mol, minimize=False)
            by = {c.GetId(): float(e[k]) for k, c in enumerate(self._mol.GetConformers())}
            out = Ensemble(
                Chem.Mol(self._mol),
                list(self.ids),
                self.cons,
                None,
                {i: by[i] for i in self.ids if i in by},
                True,
                tag=dict(self.tag),
                metal_bonds=list(self.metal_bonds),
            )
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
        out = Ensemble(
            Chem.Mol(self._mol),
            kept,
            self.cons,
            None,
            energies,
            True,
            tag=dict(self.tag),
            metal_bonds=list(self.metal_bonds),
        )
        out.energy_kind = "real"  # xtb/g-xTB, comparable across species, so EnsembleSet.best() accepts it
        return out

    def _calc_charge(self, charge):
        """Return the total charge for a calculator: the override, else the molecule's formal charge.

        Warns for a metal, whose perceived formal charge is a likely bond-perception artefact.
        """
        q = Chem.GetFormalCharge(self._mol) if charge is None else charge
        if charge is None and q != 0 and _metal.metal_index(self._mol) is not None:
            logger.warning(
                "perceived formal charge %d is likely a perception artefact; pass charge=<total>",
                q,
            )
        return q

    def optimize(self, refine="gxtb", level="normal", solvent=None, charge=None):
        """Geometry-optimise each conformer with xtb at `level`, returning a new ensemble.

        `level` is 'loose'/'normal'/'tight'/'vtight'; `cons.frozen` is held fixed while everything else
        relaxes. The optimised geometries and energies (kcal/mol) land on a copy: this settles `self` once via
        `minimize`, then moves atoms on the copy.

        `cons.frozen` is the reacting core only. A `rx.metal(center=, fix=)` TS holds its coordination sphere
        by soft shape constraints, so that relaxes here, whereas `rx.embed(ts.xyz, fix=[...])` puts the metal
        and donors in `cons.frozen` and they are held. Pass `charge=` for a metal; `refine='gfn2'` for a
        solvated opt, since g-xTB has no ALPB.
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
                conf = new_mol.GetConformer(i)
                for a, xyz in enumerate(coords):
                    conf.SetAtomPosition(a, [float(v) for v in xyz])
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
        logger.info(
            "optimize: %s --opt %s on %d conformer(s); %d core atom(s) held fixed", refine, level, len(kept), len(fix)
        )
        # The opt moved every free atom, so it may have returned a different molecule; check here since the
        # output's _minimized=True makes every downstream minimize() (incl. prune's) a no-op.
        changed = self._flag_connectivity(new_mol, kept, f"optimize[{refine}]", charge)
        out = Ensemble(
            new_mol,
            kept,
            self.cons,
            None,
            energies,
            True,
            tag=dict(self.tag),
            sphere=dict(self.sphere),
            metal_bonds=list(self.metal_bonds),
        )
        out.energy_kind = "real"  # geometry-optimised xtb/g-xTB energies, comparable across species
        out.reacted = changed  # flagged, not dropped: .filter('connectivity') drops them
        return out

    def relative(self, unit="kcal"):
        """Relative energies ``{conformer id -> E - E_min}``, the only physically meaningful read.

        An absolute single-point or FF total is not meaningful. Energies are stored in kcal/mol (FF or
        `score` alike), so ``unit='kcal'`` is identity and ``'hartree'`` divides back. Lowest conformer is 0.0.
        """
        have = {i: self.energies[i] for i in self.ids if i in self.energies}
        if not have:
            return {}
        emin = min(have.values())
        scale = 1.0 if unit == "kcal" else 1.0 / _HARTREE_KCAL
        return {i: (e - emin) * scale for i, e in have.items()}

    def _scan_connectivity(self, mol=None, ids=None, charge=None):
        """``{conf id: (formed, broken)}`` for every conformer whose graph no longer matches the intended one.

        The metal is handed to the perceiver as its real element (the pipeline may still be carrying the
        carbon surrogate), and its dative pairs are judged by coordination, not covalent radii.
        """
        mol = self._mol if mol is None else mol
        ids = self.ids if ids is None else ids
        iso = self.iso
        metals, elements, spheres = frozenset(), None, dict(self.sphere)
        if iso is not None:  # pre-minimize: the mol still carries the surrogate, so name the real elements
            metals = {iso.metal, *(mi for mi, _rz, _rq in iso.extra)}
            elements = {iso.metal: iso.real_z, **{mi: rz for mi, rz, _rq in iso.extra}}
            if iso.donors:
                spheres.setdefault(iso.metal, list(iso.donors))
        else:  # post-minimize: the real elements are back, and `sphere` is what remembers the coordination
            metals = _metrics._metal_indices(mol)
        # Not _calc_charge: it warns about a metal's perceived charge and this runs on every stage. Perception
        # here is connectivity-only (no bond orders), which the total charge barely moves.
        q = Chem.GetFormalCharge(self._mol) if charge is None else charge
        out = {}
        for i in ids:
            formed, broken = _metrics.connectivity(
                mol, i, exclude=self.cons.frozen, metals=metals, charge=q, elements=elements
            )
            for m, donors in spheres.items():  # the metal's own check: a dative bond has no covalent yardstick,
                if not donors:  # so the coordination sphere is compared as a set instead
                    continue
                left, joined = _metrics.coordination_changed(mol, i, m, donors, elements=elements)
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

    def filter(self, by="connectivity", *, charge=None):
        """Drop conformers that are no longer the molecule you asked for, in place.

        ``by='connectivity'`` re-perceives each graph and drops any whose bonds differ from the intended ones:
        a transferred proton, a formed/broken bond, a ligand that left the metal. This is the gate that drops
        what ``optimize()`` only flags. A frozen TS core is exempt, its partial bonds held to the reference.
        Dropping every conformer raises rather than returning a silent empty ensemble.
        """
        if by != "connectivity":
            raise ValueError(f"filter(by={by!r}): the only filter is 'connectivity' (geometric dedup is prune())")
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
        self.ids = kept
        self.reacted = {**self.reacted, **changed}
        return self

    def prune(self, by="auto", **kw):
        """Deduplicate by geometry, in place.

        Relaxes once via `minimize()` first, which moves atoms and is idempotent if already minimized, then
        dedups, so a raw embed is FF-settled before comparison.

        `by` (alias `method`) is 'auto' (the default, meaning rotation-invariant 'rmsd', safe anywhere), or
        one of 'rmsd' | 'moi' | 'descriptor' | 'energy', or a cheap-first cascade like ['moi', 'rmsd']. 'moi'
        on a multi-fragment system merges distinct encounter geometries, being nearly blind to a
        light/symmetric fragment's relative pose; rxembed warns. For a binding-mode summary use
        `representatives()`.

        Tuning knobs (keyword): `max_rmsd` (Å), `moi_dev`, `max_dist` (descriptor), `energy_tol`,
        `energy_window` (kcal/mol gate). Nothing is lost: discarded conformers stay in `ens.mol`, listed by
        `ens.discarded` / `ens.duplicates()`.
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
            self.ids, _ = _dedup.apply(self._mol, self.ids, energies, method=method, **kw)
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
        for d, (k, r) in _dedup.nearest_kept(self._mol, kept, dropped).items():
            groups.setdefault(k, []).append((d, r))
        return {k: sorted(v) for k, v in sorted(groups.items())}

    def duplicates(self):
        """Explain what was discarded: ``{kept_id: [(discarded_id, RMSD_Angstrom), ...]}``.

        Each discarded conformer is grouped under the kept one it most resembles. The geometries still live in
        `ens.mol` (dump any via `rxembed.wrap(ens.mol, [discarded_id]).dump(...)`).
        """
        return self._group_duplicates(self.ids, self.discarded) if self.discarded else {}

    def binding_modes(self):
        """Count the inter-fragment NCI binding-mode signatures sampled across the conformers.

        Same inter-fragment scope as ``representatives``, intramolecular NCIs being conformational detail, so
        the two views agree. An empty signature ``()`` means no inter-fragment contact in that pose.
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
        """Binding-mode cluster label per conformer (HDBSCAN on the shared latent; -1 = rare/noise).

        On the dihedral [+NCI] [+metal] latent that ``landscape()`` also projects. Relaxes once via
        `minimize()` first, which moves atoms and is idempotent if already minimized.
        """
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
        """Build a new Ensemble over a subset of `ids`, carrying every record the parent holds.

        The Mol is owned outright so a derived ensemble never mutates the parent. `iso` is carried rather
        than dropped, so `dump` on a pre-minimize derived ensemble (align/lowest) still restores the real
        element; it is read-only here, and a later `minimize()` relaxes this copy. Carrying `energy_kind`
        matters: these are the same energies, and without it a `best()` downstream of `lowest()` or
        `representatives()` would refuse energies that really are real. `sphere` and `metal_bonds` keep the
        derived ensemble the same complex, connected the same way.
        """
        mol = mol if mol is not None else Chem.Mol(self._mol)
        return Ensemble(
            mol,
            list(ids),
            self.cons,
            self.iso,
            {i: self.energies[i] for i in ids if i in self.energies},
            self._minimized,
            tag=dict(self.tag),
            energy_kind=self.energy_kind,
            sphere=dict(self.sphere),
            metal_bonds=list(self.metal_bonds),
        )

    def select_stereo(self, like, spec="preserve"):
        """Keep only conformers whose chirality matches `spec` against a reference `like`.

        `like` is a Mol carrying the wanted handedness, usually the input geometry. This covers the chirality
        the embed cannot keep (a metallocene's planar chirality, axial/helical atropisomerism), which the
        embed samples at random and this selects from, general over all elements via xyzgraph.

        `spec` is ``'preserve'`` (the default: keep non-graph chirality, leave point R/S and E/Z free),
        ``'free'`` (sample every handedness), ``'invert'`` (the enantiomeric series), or a dict
        ``{kind|'default': mode}`` to keep some and scramble others, e.g.
        ``{'planar':'preserve','default':'free'}``. Returns a new Ensemble, a no-op if no managed chirality
        element is present.
        """
        if not self._minimized:
            self._stereo = None  # an explicit select_stereo overrides the embed default, so
        self.minimize()  # select_stereo(.., 'free') still sees every pose
        ref = _stereo.signature(like) if isinstance(like, Chem.Mol) else _stereo.signature(*like)
        keep = [i for i in self.ids if _stereo.satisfies_spec(_stereo.signature(self._mol, i), ref, spec)]
        logger.info("select_stereo[%s]: %d -> %d (chirality-matched)", spec, len(self.ids), len(keep))
        return self._derive(keep)

    def lowest(self, n=1, refine=None, solvent=None):
        """Return the `n` lowest-energy conformers as a new Ensemble (n=1 -> the single best geometry).

        Ranks on the force-field energy by default; pass ``refine='gxtb'`` (or ``'gfn2'``) to rank on a real
        xTB single point instead, geometry untouched (see `score`).
        """
        src = self.score(refine, solvent) if refine else self
        src.minimize()
        # +inf, not 0.0: a conformer whose relax was skipped (untypable core) has no energy and must sort
        # last. Ranking it 0.0 against a real -250 kcal/mol would pick the phantom as "best".
        return src._derive(sorted(src.ids, key=lambda i: src.energies.get(i, float("inf")))[:n])

    def representatives(self, *, min_cluster=3, nci=True, recover_noise="auto", noise_window=10.0):
        """Return one lowest-energy conformer per mode as a new Ensemble: the distinct-shapes summary.

        A mode is whatever the latent distinguishes (`select.active_feature_kinds`): a conformer family, a
        contact pattern, or a ligand arrangement. Sorted by energy; folded conformers stay in `ens.ids`.
        Relaxes once via `minimize()` first, which moves atoms.

        `recover_noise` handles HDBSCAN noise (label -1): ``"auto"`` (default) recovers a noise conformer only
        if its `mode_signature` is a genuinely new binding mode, ``True`` keeps every noise conformer,
        ``False`` drops them. A recovered mode must sit within `noise_window` kcal/mol of the minimum.
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

        The original is untouched. Aligns on the constrained core by default, so a TS's frozen atoms sit still
        and the rest shows its variation; pass `on=` atom indices or SMARTS to align on something else.
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
        """The number of conformers in play; the notebook-facing alias for ``len(ens)``."""
        return len(self.ids)

    def __getitem__(self, key):
        """Pick conformer(s) by position as a new Ensemble: ``reps[1]`` the 2nd, ``ens[:3]`` the first three.

        Isolates one representative or conformer for a view or dump, in tracked
        (``representatives``/``lowest``) order. Returns a new Ensemble; never mutates this one.
        """
        sel = self.ids[key]
        return self._derive(sel if isinstance(sel, list) else [sel])

    def dump(self, path, align=True):
        """Write the current conformers as a multi-frame .xyz, one frame per tracked id.

        Dump at any pipeline stage or from any derived ensemble. Real element symbols are written even before
        ``minimize()`` restores a metal surrogate.

        Frames are Kabsch-superposed on the rigid core by default, the same selector as ``align()``; pass
        ``align=False`` for raw embed-frame coordinates, which is what the inherited ``xyz()`` always emits,
        so the two agree only with ``align=False``. Works on a copy, so the live geometry is never moved.
        Returns the path.
        """
        if not self.ids:  # `Conformers.dump`'s guard: a 0-byte file that reads as a successful write is the
            raise ValueError(  # worst possible outcome, and minimize() can drop every conformer
                "nothing to dump: this ensemble has no conformers (the embed produced none, or a stage "
                "dropped them all); check the log for what was discarded"
            )
        mol = Chem.Mol(self._mol)  # the ensemble mol is already real (no haptic centroid dummy); work on a copy
        if self.iso is not None:  # show the real metal(s) with their oxidation state, not the C surrogate
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

    write = dump  # alias

    def __repr__(self):
        """Summarise the ensemble: conformer count, state, and energy spread."""
        n = len(self.ids)
        es = [self.energies[i] for i in self.ids if i in self.energies]
        de = f", ΔE 0.00-{max(es) - min(es):.2f} kcal/mol" if len(es) > 1 else ""
        state = "minimized" if self._minimized else "embedded"
        tag = f", {self.tag}" if self.tag else ""
        return f"<Ensemble: {n} conformer{'s' if n != 1 else ''}, {state}{de}{tag}>"
