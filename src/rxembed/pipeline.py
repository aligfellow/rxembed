"""The user-facing pipeline: one `embed()` entry returning a chainable `Ensemble`.

    import rxembed as rx
    ens = rx.embed("CCO").mc().prune()
    ens = rx.embed("ts.xyz", fix=[11, 14, 15]).mc().prune()   # hold a TS core (0.000 A graft)

Sanity, vdW separation, the metal surrogate swap/restore and constraint validation all happen
inside. Every stage logs (`rxembed.set_verbose()`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

from rxembed.rdkit_embed.constraints import distance as _distance
from rxembed.rdkit_embed.constraints import metal as _metal
from rxembed.rdkit_embed.embed import bounds as _bounds
from rxembed.rdkit_embed.log import logger

from . import dedup as _dedup
from . import geometry as _geometry
from . import metrics as _metrics
from . import refine as _refine
from . import stereo as _stereo
from .constraints import Constraints, match, resolve_atom
from .constraints import nci as _nci
from .embed import mc as _mc

if TYPE_CHECKING:
    from .embed.dispatch import _MetalCtx

# dispatch constructs the Ensemble/EnsembleSet defined here, so it is imported lazily (in `embed()` and
# `Ensemble.mc`) to keep the import cycle one-directional.
_MIN_OVERLAY_ATOMS = 3  # need >=3 atoms to define an alignment frame
_HARTREE_KCAL = 627.5094740631  # Eh -> kcal/mol
_DISTANCE_FC = 1e4  # restrained-UFF distance force constant every relax entry point starts from
# half-order steps (not 10x jumps) so escalation lands on the MINIMAL stiffness that holds the sphere: 1e4..1e6
_FC_ESCALATION = (1.0, 3.0, 10.0, 30.0, 100.0)
# a ~1.5x-stretched bond in a coplanar coordination is a surrogate/tight-bite artifact xtb recovers, so the
# metal path keeps it (the coplanarity gate is the real plane check). Non-metal stays 1.3.
_METAL_BOND_TOL = 1.5
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

    def filter(self, by=None, **tag):
        """Subset the candidates by `tag`, or — with `by=` — drop reacted conformers within each candidate.

        Told apart by how it is called:

            set.filter(geometry='square_planar')   # keep the candidates matching this tag -> EnsembleSet
            set.filter('connectivity')             # drop conformers whose graph changed, in every candidate
        """
        if by is not None:
            return self._map("filter", by, **tag)
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

    #: verbs whose per-candidate output is comparable ACROSS candidates only via a REAL calculator, not the FF
    _REAL_ENERGY_VERBS = frozenset({"score", "optimize"})

    def _map(self, method, *args, **kw):
        """Apply an `Ensemble` verb to every candidate, returning a new `EnsembleSet` (tags carried over).

        Each candidate is searched/pruned/scored **on its own, with its own constraints** — distinct species
        are NEVER pooled or cross-pruned. To rank them, use `score`/`optimize` (a real calculator) then `best`.
        """
        tags = ", ".join(self._slug(e.tag or {}, k) for k, e in enumerate(self)) or "?"
        logger.info(
            "EnsembleSet.%s: applied to %d candidate(s) [%s] independently — WITHIN each (own constraints); "
            "distinct species are never pooled/cross-pruned%s",
            method,
            len(self),
            tags,
            " — compare across them only via these REAL energies" if method in self._REAL_ENERGY_VERBS else "",
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
        """Real-energy score each candidate (see `Ensemble.score`) — energies are per-species, not cross-comparable."""
        return self._map("score", *args, **kw)

    def optimize(self, *args, **kw):
        """Geometry-optimise each candidate (see `Ensemble.optimize`)."""
        return self._map("optimize", *args, **kw)

    def lowest(self, *args, **kw):
        """Keep the lowest-energy conformer(s) WITHIN each candidate (see `Ensemble.lowest`)."""
        return self._map("lowest", *args, **kw)

    def representatives(self, *args, **kw):
        """Distinct representatives WITHIN each candidate (see `Ensemble.representatives`)."""
        return self._map("representatives", *args, **kw)

    def best(self, n=1):
        """Rank the candidates by their LOWEST energy and keep the best `n` — the deliberate CROSS-species compare.

        The ONE place species are ranked against each other, so it demands **real** energies (raises unless
        every candidate was `score`d / `optimize`d; FF surrogate energies are refused). Returns the single
        winning `Ensemble` (``n==1``) or an `EnsembleSet` of the best `n`.
        """
        not_real = [self._slug(e.tag or {}, k) for k, e in enumerate(self) if e.energy_kind != "real"]
        if not_real:
            raise ValueError(
                f"best() ranks distinct species by REAL energy, but {not_real} have none (FF surrogate energies are "
                f"NOT comparable across species) — run .score('gxtb') or .optimize('gxtb') on the set first"
            )

        def emin(e):
            return min(e.energies[i] for i in e.ids if i in e.energies)

        ranked = sorted(self, key=emin)
        lo = emin(ranked[0])
        order = ", ".join(f"{self._slug(e.tag or {}, k)}(+{emin(e) - lo:.1f})" for k, e in enumerate(ranked))
        logger.info(
            "EnsembleSet.best: ranked %d by REAL ΔE (kcal/mol): %s -> kept %d", len(self), order, min(n, len(ranked))
        )
        kept = EnsembleSet(ranked[:n])
        return kept[0] if n == 1 else kept

    @staticmethod
    def _slug(tag, k):
        """Filename-safe identifier for a candidate from its tag (stereo / label / nci …), else its index."""
        s = "_".join(str(tag[key]) for key in ("stereo", "label", "nci", "chirality") if tag.get(key)) or f"c{k}"
        return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in s)

    def dump(self, path, align=True):
        """Write EACH candidate to its own multi-frame .xyz, its tag folded into the filename; return the paths.

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
    stereo="racemic",
    **kw,
):
    """Embed conformers (optionally constrained), returning an `Ensemble`, an `EnsembleSet`, or a `list`.

    Return shape follows the input: an `Ensemble` for one molecule; an `EnsembleSet` (to `.select` from) when
    the input is inherently several poses (metal isomers, NCI modes, ambiguous ``coordinate=``, a racemate);
    or a bare ``list[EnsembleSet]`` for ``stereo='separate'`` (one per configuration; iterate/index, it does
    not chain).

    A constrained embed comes back **relaxed into its windows** (`_relax_into_windows`); only the geometry
    moves, so the rest of the chain is unchanged.

    **Three constraint verbs**, all index-driven (0-based, xyz/graph order; resolve any SMARTS yourself):

    - **``fix``** *(rigid)*: ``fix=[i, j, k]`` at the source's own coords (Kabsch graft; needs a geometry);
      ``fix={i: (x, y, z)}`` at explicit coords; ``fix={(i, j): d, (i, j, k): θ}`` at exact numbers
      (tight-window UFF pull — verify with ``.measure()``). May mix coords and numbers.
    - **``constrain``** *(soft)*: ``{(i, j): (lo, hi)}`` distance/angle windows, plus π-stacks
      ``{(ring_a, ring_b): separation}``.
    - **``template``** *(coords-fix sugar)*: ``(reference, {target_i: ref_i})`` — ``reference`` is an .xyz
      path / Mol / Ensemble / (N,3) array, mapped explicitly.

    **``stereo=``** governs undefined chirality of a coordinate-free input (`stereo.enumerate_unassigned`):
    point R/S + double-bond E/Z (defined centres held, meso dropped, chiral-at-P included, metal never
    enumerated).

    - ``'racemic'`` *(default; alias ``'auto'``)*: embed every stereoisomer with equal effort, folded into one
      `EnsembleSet` (each tagged ``stereo=<config>``); never energy-pruned against each other. A fully-defined
      input is untouched (one `Ensemble`).
    - ``'separate'`` *(alias ``'enumerate'``)*: keep them apart — a ``list[EnsembleSet]``, one per configuration.
    - ``'free'``: opt out — one embed, stereocentres left to the ETKDG seed.

    For a *geometry* input stereo is 3D-defined, so ``stereo=`` instead tunes the preservation filter for
    chirality the embed can't keep (planar/axial/helical) — see `_attach_stereo`. ``contacts=`` → `nci_modes`;
    ``metal=`` / ``coordinate=`` → `rx.metal`.
    """
    n_alias = kw.pop("n_confs", None)
    n_alias = kw.pop("num_confs", n_alias)
    if n is None:
        n = n_alias  # accept n_confs= / num_confs= as friendly aliases of n=
    stereo = {"auto": "racemic", "enumerate": "separate"}.get(stereo, stereo)  # accept the older mode names
    if kw:
        raise TypeError(
            f"embed() got unexpected keyword(s) {sorted(kw)} — the constraint verbs are "
            f"fix / constrain / template (+ contacts / metal / coordinate / n_confs)"
        )
    # lazy imports: break the dispatch<->pipeline cycle
    from .embed.dispatch import _attach_stereo, _embed_dispatch, _stereo_enumerated_embed, _stereo_expand

    dispatch_kw = {
        "metal": metal,
        "fix": fix,
        "constrain": constrain,
        "template": template,
        "contacts": contacts,
        "coordinate": coordinate,
        "charge": charge,
        "n": n,
        "seed": seed,
        "knowledge": knowledge,
        "stereo": stereo,  # the metal load-in (enumerate_isomers) reads it; the organic path uses _stereo_expand
    }
    expanded = _stereo_expand(source, charge, stereo)  # undefined-stereocentre racemate load-in (or None)
    if expanded is not None:
        result = _stereo_enumerated_embed(expanded, stereo, dispatch_kw)
    else:
        result = _embed_dispatch(source, **dispatch_kw)
        _attach_stereo(result, source, charge, stereo)
    return _relax_embedded(result)


def _relax_embedded(result):
    """Relax every embedded candidate into its windows — the one seam `embed` returns through.

    `EnsembleSet` subclasses `list`, so the plain-list branch must come LAST (an earlier ``isinstance(…, list)``
    would downgrade an EnsembleSet to a bare list).
    """
    if isinstance(result, EnsembleSet):
        return result._map("_relax_into_windows")
    if isinstance(result, Ensemble):
        return result._relax_into_windows()
    return [r._map("_relax_into_windows") for r in result]  # stereo='separate' -> a plain list of EnsembleSet


def minimize(source, *, fix=None, constrain=None, charge=0, distance_fc=_DISTANCE_FC):
    """Relax an existing structure **toward** ``fix``/``constrain`` targets — the search-free companion to `embed`.

    Same vocabulary as `embed` but no conf-search: wrap the input geometry, graft any coordinate-``fix`` core,
    run the restrained UFF pull toward the targets. Needs an input geometry (an .xyz / a Mol with a conformer).

        rx.minimize('mol.xyz', fix={(i, j): 2.0, (i, j, k): 178})   # pull toward a linear 3-centre core
    """
    from .constraints import resolve_core
    from .embed.dispatch import _graft_frozen, _normalize

    mol, has_geom = _normalize(source, charge)
    if not has_geom:
        raise ValueError(
            "minimize() relaxes an existing geometry — give an .xyz or a Mol with a conformer, not a SMILES"
        )
    from .embed.dispatch import _MetalCtx

    metal_ctx = None
    if _metal.metal_index(mol) is not None:  # a metal gets the same treatment as in embed(): surrogated,
        spheres = {  # its sphere held from the input geometry, and the same zero-vdW force field. Handing UFF a
            mi: [n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors()]  # real metal is the worst option of
            for mi in _metal.metal_indices(mol)  # all: it types Ti and not Ir, so the force field would depend
        }  # on which metal you have.
        donor_bonds = [(d, mi) for mi, dons in spheres.items() for d in dons]  # re-added DATIVE on the output mol
        mol, metals, _ = _metal.surrogate_all_metals(mol)
        (m, real_z, real_q), extra = metals[0], metals[1:]  # extra: [(idx, real_z, real_q), ...]
        metal_ctx = _MetalCtx(mol, m, real_z, real_q, donors=spheres.get(m), extra=extra, donor_bonds=donor_bonds)
    cons, ref = resolve_core(mol, fix=fix, constrain=constrain, has_geometry=has_geom)
    if metal_ctx is not None:
        rz = {mi: z for mi, z, _q in metals}
        for mi, dons in spheres.items():
            _metal.hold_shape(mol, [mi, *dons], cons)  # hold the sphere at the input geometry
        _distance.ff_terms(mol, cons, {mi: (rz[mi], dons) for mi, dons in spheres.items()})
    ids = [c.GetId() for c in mol.GetConformers()]
    if ref:
        graft = sorted(ref)
        _graft_frozen(mol, ids, graft, np.array([ref[i] for i in graft]))
    return Ensemble(mol, ids, cons, metal_ctx).minimize(distance_fc=distance_fc)


def wrap(mol, ids=None, *, energies=None, minimized=False):
    """Wrap an existing RDKit Mol (with conformers) as an Ensemble to give it rxembed's methods.

    By default the wrapped geometries are treated as un-relaxed: `prune()`/`representatives()`/`lowest()` run
    one FF `minimize()` first (which **moves atoms**). If your conformers are **already optimised** and must
    not be disturbed (racerts/xtb output), pass ``minimized=True`` — the relax is skipped and energies come
    from ``energies=`` (a list aligned to `ids`, or a dict) or as **single points** if omitted.
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

    **The `_mol` / `.mol` split (metal complexes).** The stored `_mol` is the bond-less carbon/Li **surrogate**
    every DG/FF stage relies on (UFF cannot type a bonded transition metal). `.mol` is a **property** that
    finalizes the user-facing connected graph on access (real element + oxidation state + M-L DATIVE bonds,
    via `restore_metal` + `connect_metal`). Internal consumers read `_mol`; organic inputs pass through.

    **Mutation contract:**
    - Build/narrow-in-place-and-return (chainable): `mc`, `minimize`, `prune`, `filter`.
    - Derive a NEW Ensemble (this one untouched): `lowest`, `representatives`, `align`, `score`, `optimize`,
      `select_stereo`.
    - Looking never mutates: `view`, `compare`, `landscape`, `cluster`, `binding_modes` (they relax once via
      `minimize` if energies are needed, which is idempotent).
    """

    _mol: Chem.Mol  # the internal working surrogate; `.mol` finalizes the connected user-facing graph from it
    ids: list
    cons: Constraints = field(default_factory=Constraints)
    _metal: _MetalCtx | None = None
    energies: dict = field(default_factory=dict)
    _minimized: bool = False
    _seeds_relaxed: bool = False  # embed already relaxed these seeds into their windows, so minimize takes a
    # single point; cleared by mc() (its openconf geometries need a fresh relax)
    discarded: list = field(default_factory=list)  # conformer ids a stage dropped (still in `mol`)
    tag: dict = field(default_factory=dict)  # what distinguishes this candidate in an EnsembleSet
    _stereo: tuple | None = None  # (spec, reference signature) for the minimize() chirality filter (embed sets it)
    _donor_hand: dict = field(default_factory=dict)  # {labile-donor idx: target signed-volume hand} from the
    # uniform initial embed — minimize() culls any conformer whose metal-bound C/N donor inverted
    energy_kind: str = ""  # "" none | "ff" (surrogate — not cross-species) | "real" (xtb/g-xTB — is); best()
    # ranks only "real". Last so the positional Ensemble(...) constructors are unaffected.
    reacted: dict = field(default_factory=dict)  # {conf id: (formed, broken)} — graph changed under a stage that
    # moved atoms (an xtb opt can return a clean low-energy geometry of a different species). Flagged, not dropped.
    sphere: dict = field(default_factory=dict)  # {metal idx: [donor idx]} — the intended sphere, durable past
    # `_metal`'s teardown; the only way `coordination_changed` tells a dissociated ligand from a healthy one
    metal_bonds: list = field(default_factory=list)  # stripped M-donor bonds as (donor, metal) pairs; minimize()
    # re-adds them DATIVE on the output. Durable past `_metal`, so a re-`minimize` after `mc` re-connects too.

    @property
    def mol(self):
        """The user-facing molecule — ALWAYS a proper connected graph, finalized on access from `_mol`.

        Returns the connected graph the input had (real element + oxidation state + M-donor **DATIVE** bonds)
        at any stage, via `restore_metal` + `connect_metal` on a COPY (the working `_mol` is never mutated).
        Organic / post-minimize inputs pass through unchanged.

        **Uncached** — every access re-runs the finalize for a pre-minimize metal complex, so a hot loop should
        bind it once (`m = ens.mol`).
        """
        mol, metal_ctx = self._mol, self._metal
        bonds = self.metal_bonds or (list(metal_ctx.donor_bonds) if metal_ctx is not None else [])
        if metal_ctx is not None:  # pre-minimize: `_mol` still carries the carbon surrogate + no M-L bonds. Restore
            mol = Chem.Mol(mol)  # real element(s)/charge on a COPY (never touch the working surrogate), then
            # connect below — the same order
            for mi, rz, rq in [(metal_ctx.metal, metal_ctx.real_z, metal_ctx.real_q), *metal_ctx.extra]:
                _metal.restore_metal(mol, mi, rz, rq)  # `minimize` finalizes in (restore element+charge, then bond)
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
        """Openconf Monte-Carlo torsional search (rowansci), in place.

        `preset` effort ('rapid'|'ensemble'|'spectroscopic'|'docking'|'analogue'|'macrocycle'|
        'transition_metal'); use **'transition_metal'** for a coordination complex (every preset also
        auto-adds a metal move budget). `seed` reproducibility; `max_out` conformer cap; `low_mode` Hessian
        low-mode following (off under constraints); `config` a raw openconf ConformerConfig. **Any other
        keyword** is passed through as a ``ConformerConfig`` field override (unknown field raises; kwargs win
        over `config=`).

        Unconstrained: openconf *replaces* the ETKDG seeds. Constrained / multi-fragment: it searches *around*
        the pose-frozen seeds (settled into their windows first — see `_settle_seeds`), so its output is
        *added*. `replace=` overrides.

        `explore=True` (a seeded NCI complex): fire a second search with the NCI contacts *released* (structural
        holds kept), pool both, and swap cons to the relaxed set so downstream stages don't yank the contacts
        back — energy then decides (best with `score('gfnff')`). On the metal path only substrate contacts
        release; the sphere stays held.
        """
        if not _mc.available():
            logger.warning("mc: openconf not installed; skipped")
            return self
        if preset == "rapid" and self.cons.is_constrained:  # constrained pose-mode is rotor-only -> rapid
            logger.warning(
                "mc: preset='rapid' on a constrained system under-samples (pose-mode is "  # under-
                "rotor-only); 'ensemble' (the default) is recommended for a constrained run"
            )  # samples
        if self.cons.is_constrained:  # be explicit that we hand openconf a pose-constraint (limits its move set)
            logger.info(
                "mc: %d atom(s) constrained -> openconf runs POSE-FROZEN (rotor-only; low-mode/ring/global "
                "moves off) so the held contact is never broken",
                len(self.cons.constrained_atoms()),
            )

        if self.metal_bonds:  # a prior minimize() left the output CONNECTED; the search/relax needs the bare mol
            self._mol = _metal.disconnect_metal(self._mol)  # (UFF can't type a bonded metal) — minimize re-connects
        self._settle_seeds()  # settle seeds INTO their windows (spread across them) before openconf pose-freezes

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
        self._minimized = self._seeds_relaxed = False  # openconf's geometries are its own FF's; a follow-up
        # minimize() must genuinely re-relax them. Kept explicit (not auto here) so the pipeline stays legible.

        if explore and any(self.cons.contacts):  # second pass: contacts released, structure kept
            from .embed.dispatch import _encounter_bounds  # lazy: breaks the dispatch<->pipeline cycle

            relaxed = self.cons.relaxed()
            # releasing the grip frees the fragments it linked, so explore re-bounds ALL inter-fragment pairs
            # (setdefault never overrides a surviving structural hold), unlike dispatch's _float_encounter_bounds.
            if len(Chem.GetMolFrags(self._mol)) > 1:  # keep the fragments together once contacts are freed
                for k, v in _encounter_bounds(self._mol).items():
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
        held exactly. Fixes the raw ETKDG seed sitting *outside* a tight window.
        """
        cons = self.cons
        seeded_d = [(k, v) for k, v in cons.distances.items() if not (k[0] in cons.frozen and k[1] in cons.frozen)]
        shape_d = {k: v for k, v in cons.distances.items() if k[0] in cons.frozen and k[1] in cons.frozen}
        if not (seeded_d or cons.angles) or not self.ids:
            return
        for b, group in enumerate(g for g in np.array_split(list(self.ids), min(len(self.ids), bins)) if len(g)):
            frac = (b + 0.5) / min(len(self.ids), bins)  # this bin's fraction across every window
            # A PARTIAL rebuild: `distances`/`angles` are re-derived below per bin, so they start empty; but
            # everything else is carried — critically `coplanar`, without which the stiff relax loses the
            # sp2-donor coplanarity torsion and drives the metal out of plane. `contacts`/`dg_floors` are
            # omitted (provenance / bounds-writer only — this path relaxes, never embeds).
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

        A `hold_shape` body — a retained or spectator coordination sphere — is pinned by all C(n,2) of its
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
                        logger.debug("minimize: conformer %d rejected — rigid shape torn at (%d,%d)", cid, i, j)
                        return False
        return True

    def _coordination_ok(self, cid, metal_ctx):
        """Return False if a declared *planar* metal polyhedron came out puckered (a phantom the relax forced).

        A square-planar / T-shape / trigonal-planar centre **is** coplanar by definition; the escalating relax
        can force an impossible arrangement (a chelate at trans vertices) through as an out-of-plane pucker
        that `bonding_ok` doesn't catch. This rejects exactly that. No-op for non-planar geometries (octahedral
        etc.) or a metal with no declared polyhedron (a fixed TS core).
        """
        if metal_ctx is None or not _metal.is_planar(metal_ctx.geometry) or not metal_ctx.donors:
            return True
        pos = self._mol.GetConformer(cid).GetPositions()
        # a haptic face is one vertex
        return _metal.coplanar(pos, metal_ctx.metal, metal_ctx.donors, haptic=self.cons.haptic)

    def _relax_constrained(self, distance_fc):
        """Restrained-UFF relax; if the soft relax tears *every* bond, stiffen the distances and retry.

        The surrogate's carbon vdW crowds donors out to ~2.1 Å, so a soft (``1e4``) constraint lets an unusual
        ligand tear; escalating the force constant (``1e5`` → ``1e6``) lets the coordination distances dominate
        so the bonds survive. Escalation fires ONLY when the softer relax leaves **zero** intact conformers, so
        a genuinely infeasible arrangement (an en forced *trans*) still tears at every stiffness and is dropped
        — no phantom resurrected. Returns the accepted relax's energies (or ``None`` if UFF can't type the graph).
        """
        # Hold a labile (carbanion/amine) donor's hand through the relax: the surrogate's bare degree-3 centre
        # inverts under UFF. Cap it with a dummy D, release when done, keep the ctx's mol in sync.
        metal_ctx = self._metal
        held = (
            _metal._hold_donor_chirality(self._mol, metal_ctx.metal, metal_ctx.donors, self.cons)
            if metal_ctx
            else (self._mol, [])
        )
        self._mol, hold = held
        if metal_ctx and hold:
            metal_ctx.mol = self._mol
        e, fc = None, distance_fc
        try:
            embed_pos = {
                c.GetId(): [list(c.GetAtomPosition(a)) for a in range(self._mol.GetNumAtoms())]
                for c in self._mol.GetConformers()
            }
            for step, mult in enumerate(_FC_ESCALATION):  # half-order steps: find the MINIMAL sufficient stiffness
                if step:  # restore the embed geometry before a stiffer retry (a big jump over-stiffens it)
                    for cid, pos in embed_pos.items():
                        conf = self._mol.GetConformer(cid)
                        for a, xyz in enumerate(pos):
                            conf.SetAtomPosition(a, xyz)
                fc = distance_fc * mult
                try:
                    e = _refine.restrained_uff(self._mol, self.cons, distance_fc=fc)
                except RuntimeError as err:  # UFF can't build a force field for this graph — keep the embed
                    logger.warning(
                        "minimize: UFF could not relax this system (%s) — an untypable TS/hypervalent "
                        "reacting core; keeping the embedded geometry (constraints biased, not stiffened)",
                        err,
                    )
                    return None
                # accept a stiffness only if a conformer is BOTH bonded AND (for a planar polyhedron) coplanar,
                # so escalation never forces a phantom through as an out-of-plane pucker. Metal keeps a looser tol.
                bt = _METAL_BOND_TOL if self._metal else 1.3
                kept = [
                    i
                    for i in self.ids
                    if _metrics.bonding_ok(
                        self._mol, i, bond_tol=bt, exclude=self.cons.frozen, constrained=self.cons.distances
                    )
                    and self._coordination_ok(i, self._metal)
                ]
                if kept or step == len(_FC_ESCALATION) - 1:  # some survived (accept), or out of steps (caller drops)
                    if step and kept:
                        logger.info("minimize: relax tore the sphere; escalated distance_fc to %.0e", fc)
                    break
        finally:  # release the hold on every exit path, including the early UFF-failure return — the dummy D
            if metal_ctx and hold:  # is scaffolding for this relax alone
                self._mol = _metal._release_donor_chirality(self._mol, hold, self.cons)
                metal_ctx.mol = self._mol
        return e

    def _relax_into_windows(self):
        """Relax the raw ETKDG seeds into their own constraint windows, in place — what `embed` returns through.

        A raw seed does **not** satisfy its constraints (measured: angle windows missed by 9.5° mean / 42.8°
        max, an organic window by 0.21-0.43 Å, 17/20 square-planar crystals come out *tetrahedral*); this relax
        enforces them, moving toward the crystal on every local axis (M-donor MAE 0.063 -> 0.006 Å), neutral on
        global RMSD — `docs/findings/seed-vs-relax.md`.

        Uses `_relax_constrained`, NOT `minimize`: it carries the stiffness ladder, donor-hand hold and
        bonding/coordination acceptance WITHOUT restoring the metal / dropping the surrogate — doing that at
        embed time would strip the rest of the chain of the coplanarity gate, donor-hand hold and re-embed
        (measured on henry Ni: 11/11 clean falling to 4/5). So `_metal` stays live and `_minimized` unset.

        Unconstrained embeds are left alone (no window, and no unbidden FF pass).
        """
        if not self.ids or not self.cons.is_constrained:
            return self
        seed_pos = {c: self._mol.GetConformer(c).GetPositions() for c in self.ids}
        self._relax_constrained(_DISTANCE_FC)
        self._mol = self._metal.mol if self._metal else self._mol  # _relax_constrained may have re-bound it
        # The relax can TEAR a seed; `embed` must not spend the caller's `n` re-embedding (that is `minimize`'s
        # job), so a torn conformer keeps its seed coordinates — output is never worse than the seed.
        bond_tol = _METAL_BOND_TOL if self._metal else 1.3
        self._rescue_torn(seed_pos, bond_tol)
        self._seeds_relaxed = True
        return self

    def _rescue_torn(self, seed_pos, bond_tol):
        """Re-relax each conformer the window relax tore at its own minimal sufficient stiffness; else keep its seed.

        `_relax_constrained` escalates GLOBALLY and stops at the first rung leaving any conformer intact, so a
        seed needing more stiffness than its siblings stays torn. Each is retried from its seed up the same
        ladder (embed must not spend the caller's ``n`` re-embedding); one that survives no rung keeps its seed.

        NB the rescue runs outside `_hold_donor_chirality`, so a labile donor could invert here; `minimize`'s
        `_donor_hand` cull catches that.
        """

        def place(cid, pos):
            conf = self._mol.GetConformer(cid)
            for a, xyz in enumerate(pos):
                conf.SetAtomPosition(a, xyz.tolist())

        def intact(cid):
            return _metrics.bonding_ok(
                self._mol, cid, bond_tol=bond_tol, exclude=self.cons.frozen, constrained=self.cons.distances
            )

        torn = [c for c in self.ids if not intact(c)]
        rescued = 0
        for cid in torn:
            for mult in _FC_ESCALATION[1:]:  # rung 0 is the pass that already tore it
                place(cid, seed_pos[cid])
                try:
                    _refine.restrained_uff(self._mol, self.cons, distance_fc=_DISTANCE_FC * mult, conf_ids=[int(cid)])
                except RuntimeError:  # UFF cannot build for this graph — the seed is the best we have
                    break
                if intact(cid):
                    rescued += 1
                    break
            else:
                place(cid, seed_pos[cid])
                continue
            if not intact(cid):
                place(cid, seed_pos[cid])
        if torn:
            logger.info(
                "embed: the window relax tore %d of %d seed(s) — %d rescued by escalating their stiffness, "
                "%d kept their unrelaxed geometry (minimize() re-embeds those)",
                len(torn),
                len(self.ids),
                rescued,
                len(torn) - rescued,
            )

    def minimize(self, distance_fc=_DISTANCE_FC, _retry=True):
        """FF relax (stiff restrained UFF if constrained, else MMFF), restore any metal, drop clashes, validate.

        Every conformer a gate here rejects (torn bond, puckered sphere, non-physical relax energy, inverted
        donor hand, wrong stereo) is appended to ``discarded`` — like prune's, it stays in ``mol``.
        """
        if self._minimized:
            return self
        if not self.ids:  # nothing embedded (e.g. ETKDG could not place this graph)
            self._minimized = True
            return self
        metal_ctx = self._metal  # capture before restore — the coordination-planarity gate needs metal + donors
        target = len(self.ids)  # embed(n=N) -> N GOOD geometries; the retry re-embeds if the relax tears some
        # a pristine surrogate to re-embed on
        template = Chem.Mol(self._mol) if (_retry and metal_ctx is not None) else None
        self._relax_and_record(distance_fc, metal_ctx)
        if self._metal:
            if self._metal.donors:  # remember the sphere: `_metal` goes, but who coordinates whom is durable
                self.sphere.setdefault(self._metal.metal, list(self._metal.donors))
            self._metal.restore()
            self._metal = None
        before = list(self.ids)  # ids, not a count: what minimize drops is recorded in `discarded` like prune's
        drops = self._drop_bad_geometries(metal_ctx)  # torn bond / puckered sphere / torn rigid body
        self._drop_unconverged(drops)  # non-physical relax energy
        if len(self.ids) < len(before):  # drop torn/clashed/puckered geometries + non-physical relaxes
            logger.info(  # name the gate(s) that fired, with counts
                "minimize: dropped %d conformer(s) (%s) -> %d kept",
                len(before) - len(self.ids),
                ", ".join(f"{n}x {why}" for why, n in drops.items() if n),
                len(self.ids),
            )
        if template is not None:  # a metal seed the relax tore is a bad EMBED, not a bad relax — re-embed fresh
            self._reembed_until_clean(template, metal_ctx, target, distance_fc)  # seeds until N are geom.check-clean
        self._cull_inverted_donors()  # a labile donor that inverted vs its enumerated hand
        if not self.ids:  # every conformer failed a gate — usually the arrangement itself is impossible.
            logger.warning(  # `embed` relaxes, so this now fires at EMBED time: state the gates, not a guess
                "minimize: 0 of %d conformer(s) survived the relax (%s) — the ensemble is EMPTY and nothing "
                "downstream can run. Usually the arrangement is geometrically infeasible (a chelate forced to "
                "span a bite it can't reach); check the isomer/coordination requested, or supply a geometry "
                "that already satisfies it.",
                len(before),
                ", ".join(f"{n}x {why}" for why, n in drops.items() if n) or "no gate recorded",
            )
        self._cull_wrong_stereo()  # embed-uncapturable handedness (metallocene planar / axial / helical)
        self.discarded += [i for i in before if i not in set(self.ids)]
        self._validate()
        if metal_ctx is not None and not self.metal_bonds:  # remember the stripped M-L bonds durably (past `_metal`):
            self.metal_bonds = list(metal_ctx.donor_bonds)  # re-minimize after mc has no `_metal` but must re-connect
        if self.metal_bonds:  # connectivity finalize, LAST: geometry/element/charge now settled, re-add the
            # surrogate-stripped M-donor bonds DATIVE so the output is connected (every gate/relax above saw the
            # bond-less surrogate unchanged; coords/charge untouched).
            self._mol = _metal.connect_metal(self._mol, self.metal_bonds)
        self._minimized = True
        return self

    def _relax_and_record(self, distance_fc, metal_ctx):
        """FF-relax the conformers (restrained UFF if constrained, else MMFF) and store the surrogate energies."""
        e = None
        if self.cons.is_constrained:
            if self._seeds_relaxed:  # embed already relaxed these into their windows; relaxing again only rides
                # the flat-bottomed walls further (measured: primary-phosphine M-P-H median 129.47 -> 130.00, the
                # splay cap's exact bound). Take a single point on the same FF; the gates below are what's owed.
                try:
                    e = _refine.restrained_uff(self._mol, self.cons, distance_fc=distance_fc, max_iters=0)
                except RuntimeError as err:  # the two relax entry points must degrade identically: mirror
                    #  _relax_constrained's guard so an untypable/hypervalent core keeps its embedded geometry
                    logger.warning(
                        "minimize: UFF could not relax this system (%s) — an untypable TS/hypervalent "
                        "reacting core; keeping the embedded geometry (constraints biased, not stiffened)",
                        err,
                    )
            else:
                e = self._relax_constrained(distance_fc)  # escalates stiffness if the soft relax tears every bond
                # _relax_constrained may have re-bound self._mol via the ctx
                self._mol = metal_ctx.mol if metal_ctx else self._mol
        else:
            e = _refine.ff_energies(self._mol, minimize=True)
        if e is not None:
            self.energies = {c.GetId(): float(e[k]) for k, c in enumerate(self._mol.GetConformers())}
            self.energy_kind = "ff"  # surrogate FF — NOT comparable across species (EnsembleSet.best refuses it)

    def _drop_bad_geometries(self, metal_ctx):
        """Drop each conformer that broke a bond, puckered a planar sphere, or tore a rigid body; return the counts."""
        # the arbiter: drop a torn/clashed geometry, and a declared-planar polyhedron that came out puckered.
        # A metal keeps the looser bond tol — a slightly-stretched bond in a *coplanar* coordination is a
        # surrogate artifact xtb recovers, and the coplanarity gate already rejects the impossible (phantom) ones.
        bt = _METAL_BOND_TOL if metal_ctx else 1.3
        kept, drops = [], {"broken bond": 0, "out-of-plane coordination sphere": 0, "torn rigid body": 0}
        for i in self.ids:  # one pass; record which gate rejected each so the log can name it
            if not _metrics.bonding_ok(
                self._mol, i, bond_tol=bt, exclude=self.cons.frozen, constrained=self.cons.distances
            ):
                drops["broken bond"] += 1
            elif not self._coordination_ok(i, metal_ctx):
                drops["out-of-plane coordination sphere"] += 1
            elif not self._shape_intact(i):
                drops["torn rigid body"] += 1
            else:
                kept.append(i)
        self.ids = kept
        return drops

    def _drop_unconverged(self, drops):
        """Drop conformers whose relax energy sits above the window — a non-physical, un-converged geometry."""
        if not (self.energies and self.ids):
            return
        emin = min(self.energies[i] for i in self.ids if i in self.energies)  # geometry, not a real rotamer
        rel = sorted(self.energies[i] - emin for i in self.ids if i in self.energies)
        over = [round(r, 1) for r in rel if r > _RELAX_ENERGY_WINDOW]
        logger.info(  # show the FACTS — the relax-energy spread (kcal/mol above the min) and the generous window
            "minimize: relax ΔE spread (kcal/mol above min): %s | window=%.0f%s",
            [round(r, 1) for r in rel],
            _RELAX_ENERGY_WINDOW,
            f" -> dropping {len(over)} un-converged: {over}" if over else " -> all within window",
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

    def _reembed_until_clean(self, template, metal_ctx, target, distance_fc):
        """Re-embed fresh seeds until `target` conformers pass acceptance, merging the good ones in.

        The raw embed is clean and the FF relax tears a bad seed, so the fix is a *fresh seed*, never
        re-relaxing the torn one. Acceptance is geom.check-clean AND (if ``stereo=`` set) the requested
        handedness, so the retry count survives the later stereo cull. Good geometries are preferred and placed
        first; the bonding-ok fallback is kept only where good is unreachable. Metal-only.

        The intended donors (from `sphere` + this run's `_MetalCtx`) are handed to the gate — both in-sphere
        checks are circular without them (a collapsed ligand is perceived as coordinating and exempts itself).
        """
        donors = sorted({int(d) for ds in self.sphere.values() for d in ds} | {int(d) for d in metal_ctx.donors or ()})

        def is_good(mol, cid):  # geom.check-clean, and the requested handedness when a stereo spec is active
            if not _geometry.check(mol, cid, donors=donors or None).ok():  # structural gates only — a metal-donor
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
            seed += 1  # a distinct seed each round — the point is to regenerate the embed, not repeat it
            tmpl = Chem.Mol(template)
            tmpl.RemoveAllConformers()
            # re-apply the labile-donor chirality hold — a fresh ETKDG seed is random-handed, so without it a
            # carbanion/amine donor's enumerated hand would silently invert; the batch relax below re-holds it.
            tmpl, held = (
                _metal._hold_donor_chirality(tmpl, metal_ctx.metal, metal_ctx.donors, self.cons)
                if metal_ctx
                else (tmpl, [])
            )
            new_ids = _bounds.embed(tmpl, self.cons, target - len(have) + _EMBED_BUFFER, seed=seed)
            if not new_ids:
                continue
            tmpl = _metal._release_donor_chirality(tmpl, held, self.cons)  # drop dummy + cons key; batch relax re-holds
            batch = Ensemble(tmpl, new_ids, self.cons, replace(metal_ctx, mol=tmpl)).minimize(distance_fc, _retry=False)
            for cid in batch.ids:  # merge only good ones — the fallback already holds the bonding-ok geometries
                if not is_good(batch._mol, cid):
                    continue
                nid = self._mol.AddConformer(Chem.Conformer(batch._mol.GetConformer(cid)), assignId=True)
                self.ids.append(nid)
                if cid in batch.energies:  # never fabricate a 0.0 — an absent energy (untypable graph) stays absent
                    self.energies[nid] = batch.energies[cid]
        kept = good()
        if kept:  # good first, then the bonding-ok fallback, capped at target (embed(n=N) hands back N, not N+buffer)
            self.ids = (kept + [i for i in self.ids if i not in kept])[:target]
        logger.info("minimize: re-embedded to %d/%d clean geometries", len(kept), target)

    def _validate(self, dist_slack=0.15, ang_slack=5.0):
        """Warn if a constraint the relax *can enforce* is not realised within its window (+ a small slack).

        Only constraints with a non-frozen atom are warned. A constraint between two **frozen** atoms is a
        structural hold (frozen-core shape, spectator sphere) held by the graft / ``AddFixedPoint``, not the
        relax, so an off-window value there is logged at DEBUG.
        """
        if not self.ids:
            return
        frozen = self.cons.frozen
        # a metal-donor distance / L-M-L angle is an APPROXIMATE bias (covalent-radius guess; the real value is
        # the calculator's, and the surrogate vdW legitimately pushes donors out) — off-window there is DEBUG.
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
            elif i in frozen and j in frozen:  # structural hold among pinned atoms — not the relax's to enforce
                logger.debug("frozen-shape d(%d,%d) = %.2f (embed approx of %.2f-%.2f)", i, j, d, lo, hi)
            elif i in metals or j in metals:  # an approximate coordination distance — real value is the calc's
                logger.debug("metal-donor d(%d,%d) = %.2f (embed bias; target %.2f-%.2f)", i, j, d, lo, hi)
            else:
                logger.warning("constraint NOT held: d(%d,%d) = %.2f, target %.2f-%.2f", i, j, d, lo, hi)
        for (i, j, k), (lo, hi) in self.cons.angles.items():
            if i >= n or j >= n or k >= n:  # a constraint on a transient centroid dummy — skip (see the distance loop)
                continue
            a = realised(rdMolTransforms.GetAngleDeg, i, j, k)
            if lo - ang_slack <= a <= hi + ang_slack:
                continue
            if (i in frozen and j in frozen and k in frozen) or j in metals:  # frozen shape / approx L-M-L angle
                logger.debug("metal/frozen angle(%d,%d,%d) = %.1f (target %.1f-%.1f)", i, j, k, a, lo, hi)
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
            out.energy_kind = "ff"  # a force-field single point is NOT a real energy — best() still refuses it
            return out
        energies, kept = {}, []
        for i in self.ids:
            try:
                energies[i] = float(calc.energy(self._mol, i)) * _HARTREE_KCAL  # -> kcal/mol, unit-consistent
                kept.append(i)
            except Exception as err:  # a single conformer xtb failure shouldn't sink the run
                logger.warning("score: %s failed on conformer %d (%s) — dropping it", refine, i, _last_line(err))
        if not kept:  # total failure -> RAISE, never silently hand back FF energies the caller thinks are xTB
            raise RuntimeError(
                f"score: {refine} produced no energies — is the xtb binary on PATH "
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
        out.energy_kind = "real"  # xtb/g-xTB — comparable across species, so EnsembleSet.best() accepts it
        return out

    def _calc_charge(self, charge):
        """Return the total charge for a calculator: the override, else the molecule's formal charge.

        Warns for a metal, whose perceived formal charge is a likely bond-perception artefact.
        """
        q = Chem.GetFormalCharge(self._mol) if charge is None else charge
        if charge is None and q != 0 and _metal.metal_index(self._mol) is not None:
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
        new_mol = Chem.Mol(self._mol)  # opt MOVES atoms -> work on a copy, never clobber self
        energies, kept = {}, []
        for i in self.ids:
            try:
                coords, e = calc.optimize(self._mol, i, level, fix)
                conf = new_mol.GetConformer(i)
                for a, xyz in enumerate(coords):
                    conf.SetAtomPosition(a, [float(v) for v in xyz])
                energies[i] = e * _HARTREE_KCAL
                kept.append(i)
            except (RuntimeError, OSError) as err:  # an xtb RUN failure -> drop this conformer (a config error
                # — bad level/solvent — is a ValueError and propagates, not a per-conformer failure)
                logger.warning(
                    "optimize: %s --opt failed on conformer %d (%s) — dropping it", refine, i, _last_line(err)
                )
        if not kept:
            raise RuntimeError(
                f"optimize: {refine} --opt produced nothing — is the xtb binary on PATH ($XTB_EXE, or ~/bin/xtb)?"
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
        out.energy_kind = "real"  # geometry-optimised xtb/g-xTB energies — comparable across species
        out.reacted = changed  # flagged, not dropped: .filter('connectivity') drops them
        return out

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

    def _scan_connectivity(self, mol=None, ids=None, charge=None):
        """``{conf id: (formed, broken)}`` for every conformer whose graph no longer matches the intended one.

        The metal is handed to the perceiver as its real element (the pipeline may still be carrying the
        carbon surrogate), and its dative pairs are judged by coordination, not covalent radii.
        """
        mol = self._mol if mol is None else mol
        ids = self.ids if ids is None else ids
        metal_ctx = self._metal
        metals, elements, spheres = frozenset(), None, dict(self.sphere)
        if metal_ctx is not None:  # pre-minimize: the mol still carries the surrogate, so name the real elements
            metals = {metal_ctx.metal, *(mi for mi, _rz, _rq in metal_ctx.extra)}
            elements = {metal_ctx.metal: metal_ctx.real_z, **{mi: rz for mi, rz, _rq in metal_ctx.extra}}
            if metal_ctx.donors:
                spheres.setdefault(metal_ctx.metal, list(metal_ctx.donors))
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
                "%s: conformer %d CHANGED CONNECTIVITY — %s. The geometry (and any energy) is for a "
                "DIFFERENT species than the one you asked for; drop these with .filter('connectivity').",
                stage,
                i,
                _metrics.describe(mol, formed, broken),
            )
        return changed

    def filter(self, by="connectivity", *, charge=None):
        """Drop conformers that are no longer the molecule you asked for, in place.

        ``by='connectivity'`` re-perceives each graph and drops any whose bonds differ from the intended ones
        (a transferred proton, a formed/broken bond, a ligand that left the metal) — the gate that drops what
        ``optimize()`` only flags. A frozen TS core is exempt (partial bonds held to the reference). Dropping
        every conformer raises rather than returning a silent empty ensemble.
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
                f"filter[connectivity]: all {len(self.ids)} conformer(s) changed connectivity — every one is a "
                "different species than the input. That is a result, not a filter failure: the geometry or the "
                "level of theory is reacting your molecule."
            )
        for i, (formed, broken) in changed.items():
            logger.info("filter[connectivity]: dropping #%d — %s", i, _metrics.describe(self._mol, formed, broken))
        logger.info("filter[connectivity]: %d -> %d (dropped %d reacted)", len(self.ids), len(kept), len(changed))
        self.discarded += list(changed)
        self.ids = kept
        self.reacted = {**self.reacted, **changed}
        return self

    def prune(self, by="auto", **kw):
        """Deduplicate by geometry, in place.

        Relaxes once via `minimize()` first (idempotent if already minimized) — which **moves atoms** — then
        dedups, so a raw embed is FF-settled before comparison.

        `by` (alias `method`) is **'auto'** (default → rotation-invariant **'rmsd'**, safe anywhere), or pick:
        'rmsd' | 'moi' | 'descriptor' | 'energy', or a cheap-first cascade e.g. ['moi', 'rmsd']. **'moi' on a
        multi-fragment system** silently merges distinct encounter geometries (nearly blind to a light/symmetric
        fragment's relative pose) — rxembed warns. For a binding-mode summary use `representatives()`.

        Tuning knobs (keyword): `max_rmsd` (Å), `moi_dev`, `max_dist` (descriptor), `energy_tol`,
        `energy_window` (kcal/mol gate). Nothing is lost: discarded conformers stay in `ens.mol`; see
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
        from .dedup.features import _metal_present

        # a metal complex looks multi-fragment only because the surrogate stripped its coordinate bonds;
        # those "fragments" are one molecule, so moi is a fine choice there — don't count them.
        n_frag = 1 if _metal_present(self._mol) else len(Chem.GetMolFrags(self._mol))
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
            if method == "connectivity":  # a validity filter, not a dedup — but it composes in the cascade
                self.filter("connectivity")  # (e.g. prune(by=['connectivity', 'rmsd']): drop reacted, then dedup)
                continue
            before = list(self.ids)
            energies = [self.energies.get(i, float("inf")) for i in self.ids]  # no energy -> outside any window,
            # never merged into a phantom 0.0-kcal band nor kept over a real-energy duplicate (untypable core)
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

        Same inter-fragment scope as ``representatives`` (intramolecular NCIs are conformational detail), so
        the two views agree. Empty signature ``()`` = no inter-fragment contact in that pose.
        """
        from collections import Counter

        from .dedup.features import _frag_map, _interfragment_contacts

        an = _nci.analyzer(self._mol)
        fmap = _frag_map(self._mol)

        def sig(c):
            contacts = _interfragment_contacts(an, self._mol.GetConformer(c).GetPositions(), fmap)
            return tuple(sorted({t for t, _a, _p in contacts}))

        return Counter(sig(c) for c in self.ids)

    def cluster(self, *, min_cluster=3, reduce=None, nci=True):
        """Binding-mode cluster label per conformer (HDBSCAN on the shared latent; -1 = rare/noise).

        On the dihedral [+NCI] [+metal] latent that ``landscape()`` also projects. Relaxes once via
        `minimize()` first (idempotent if already minimized) — which **moves atoms** — before clustering.
        """
        self.minimize()
        if not self.ids:
            return np.array([], dtype=int)
        return _dedup.cluster_labels(self._mol, self.ids, min_cluster=min_cluster, reduce=reduce, nci=nci)

    def landscape(self, method="pca", color="cluster", *, reduce=None, min_cluster=3, nci=True):
        """2D ensemble map (method='pca'|'tsne'), coloured by 'cluster' or 'energy'."""
        from . import viz

        self.minimize()  # ensure a latent (and, for color='energy', real energies) exist — mirrors cluster()
        return viz.landscape(self, color=color, method=method, reduce=reduce, min_cluster=min_cluster, nci=nci)

    # -- deriving a smaller / aligned ensemble (returns a NEW Ensemble) --------

    def _derive(self, ids, mol=None):
        mol = mol if mol is not None else Chem.Mol(self._mol)  # own the Mol: a derived ensemble must not
        metal = replace(self._metal, mol=mol) if self._metal is not None else None  # else mutate the parent.
        return Ensemble(
            mol,
            list(ids),
            self.cons,
            metal,  # rebound to THIS mol (not None): dump restores the real element on a pre-minimize derived
            {i: self.energies[i] for i in ids if i in self.energies},  # ensemble (align/lowest); a later
            self._minimized,  # minimize() then relaxes this copy, never the parent's shared context.
            tag=dict(self.tag),
            sphere=dict(self.sphere),  # a derived ensemble is still the same complex — keep its coordination
            metal_bonds=list(self.metal_bonds),  # and its M-L connectivity record (so it stays connected too)
        )  # Keep provenance.

    def select_stereo(self, like, spec="preserve"):
        """Keep only conformers whose chirality matches `spec` against a reference `like`.

        `like` is a Mol carrying the wanted handedness (usually the input geometry). For the chirality the
        **embed cannot keep** — a metallocene's **planar** chirality, **axial**/**helical** atropisomerism —
        sampled at random by the embed and selected here (general over all elements via xyzgraph).

        `spec`: ``'preserve'`` (default — keep non-graph chirality, leave point R/S and E/Z free);
        ``'free'`` (sample every handedness); ``'invert'`` (the enantiomeric series); or a dict
        ``{kind|'default': mode}`` to keep some and scramble others (e.g. ``{'planar':'preserve','default':'free'}``).
        Returns a new Ensemble (no-op if no managed chirality element is present).
        """
        if not self._minimized:
            self._stereo = None  # an EXPLICIT select_stereo overrides the embed default,
        self.minimize()  # so e.g. select_stereo(..,'free') still sees every pose
        ref = _stereo.signature(like) if isinstance(like, Chem.Mol) else _stereo.signature(*like)
        keep = [i for i in self.ids if _stereo.satisfies_spec(_stereo.signature(self._mol, i), ref, spec)]
        logger.info("select_stereo[%s]: %d -> %d (chirality-matched)", spec, len(self.ids), len(keep))
        return self._derive(keep)

    def lowest(self, n=1, refine=None, solvent=None):
        """Return the `n` lowest-energy conformers as a new Ensemble (n=1 -> the single best geometry).

        Ranks on the force-field energy by default; pass ``refine='gxtb'`` (or ``'gfn2'``) to rank on a real
        **xTB single point** instead (geometry untouched -- see `score`).
        """
        src = self.score(refine, solvent) if refine else self
        src.minimize()
        # +inf, not 0.0: a conformer whose relax was skipped (untypable core) has NO energy — it must sort
        # LAST, never rank as artificially lowest (0.0 < a real -250 kcal/mol would pick a phantom as "best").
        return src._derive(sorted(src.ids, key=lambda i: src.energies.get(i, float("inf")))[:n])

    def representatives(self, *, min_cluster=3, nci=True, recover_noise="auto", noise_window=10.0):
        """Return one lowest-energy conformer per **mode** as a new Ensemble (the distinct-shapes summary).

        A "mode" is whatever the latent distinguishes (`dedup.active_feature_kinds`): a conformer family, a
        contact pattern, or a ligand arrangement. Sorted by energy; folded conformers stay in `ens.ids`.
        Relaxes once via `minimize()` first (**moves atoms**).

        `recover_noise` handles HDBSCAN noise (label -1): ``"auto"`` (default) recovers a noise conformer only
        if its `mode_signature` is a genuinely new binding mode; ``True`` keeps every noise conformer; ``False``
        drops them. A recovered mode must sit within `noise_window` kcal/mol of the minimum.
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
        idx = [resolve_atom(self._mol, a) for a in atoms]
        fns = {2: rdMolTransforms.GetBondLength, 3: rdMolTransforms.GetAngleDeg, 4: rdMolTransforms.GetDihedralDeg}
        if len(idx) not in fns:
            raise ValueError("measure() takes 2 (distance), 3 (angle) or 4 (dihedral) atoms")
        f = fns[len(idx)]
        v = [f(self._mol.GetConformer(c), *idx) for c in self.ids]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}

    def _align_atoms(self, on):
        if on is None:  # default core: constrained atoms > metal coordination > heavy
            core = sorted(self.cons.constrained_atoms())
            if len(core) >= _MIN_OVERLAY_ATOMS:
                return core
            from .dedup.features import _metal_donors, _metal_present

            if _metal_present(self._mol):  # a metal's natural overlay core is M + its donors (even when
                m, donors = _metal_donors(self._mol, self.ids)  # cons is empty, e.g. a wrapped metal mol)
                if donors:
                    return [m, *donors]
            return [a.GetIdx() for a in self._mol.GetAtoms() if a.GetAtomicNum() > 1]
        if isinstance(on, str):
            return list(match(self._mol, on))
        return [resolve_atom(self._mol, a) for a in on]

    def align(self, on=None):
        """Return a new Ensemble with every conformer Kabsch-superposed for a readable overlay.

        The original is untouched (looking never mutates your geometries). Aligns on the constrained core by
        default (a TS's frozen atoms sit still, the rest shows its variation); pass `on=` atom
        indices/SMARTS to align on something else.
        """
        m = Chem.Mol(self._mol)
        if len(self.ids) > 1:
            rdMolAlign.AlignMolConformers(m, atomIds=self._align_atoms(on), confIds=list(self.ids))
        return self._derive(self.ids, mol=m)

    # -- looking (returns an artifact, never mutates) -------------------------
    # 3D rendering is notebook-level (align()/dump() + a few lines of py3Dmol/xyzrender). Only `landscape`
    # lives here, because the dim-reduction is real reusable work.

    @property
    def n(self):
        """The number of conformers in play."""
        return len(self.ids)

    def __getitem__(self, key):
        """Pick conformer(s) by position as a new Ensemble: ``reps[1]`` the 2nd, ``ens[:3]`` the first three.

        Isolates one representative / conformer for a view or dump, ranked as tracked
        (``representatives``/``lowest`` order). Returns a new Ensemble (never mutates this one).
        """
        sel = self.ids[key]
        return self._derive(sel if isinstance(sel, list) else [sel])

    def dump(self, path, align=True):
        """Write the current conformers as a multi-frame **.xyz** (one frame per tracked id).

        Dump at any pipeline stage or from any derived ensemble. Real element symbols are written even before
        ``minimize()`` restores a metal surrogate.

        By default the frames are **aligned** (Kabsch-superposed on the rigid core, same selector as
        ``align()``); pass ``align=False`` for raw embed-frame coordinates. Works on a **copy**, so the live
        geometry is never moved. Returns the path.
        """
        metal_ctx = self._metal
        real = []
        if metal_ctx is not None:
            real = [(metal_ctx.metal, metal_ctx.real_z, metal_ctx.real_q), *metal_ctx.extra]
        mol = Chem.Mol(self._mol)  # the ensemble mol is already real (no haptic centroid dummy); work on a copy
        for mi, rz, rq in real:  # show the real metal(s) with their oxidation state, not the C surrogate
            mol.GetAtomWithIdx(mi).SetAtomicNum(rz)
            mol.GetAtomWithIdx(mi).SetFormalCharge(rq)
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
