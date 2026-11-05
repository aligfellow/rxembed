"""Adapt xyzgraph candidates and observations, choose construction modes and compile their geometry restraints."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from rdkit import Chem

from rxembed.geometry.bounds import probe_conformer
from rxembed.geometry.constraints import canonical_key
from rxembed.graph import frag_map
from rxembed.utils import atom_label, flat_graph, require_mol

if TYPE_CHECKING:
    from xyzgraph.nci import NCIAnalyzer

logger = logging.getLogger("rxembed")


def analyzer(mol: Chem.Mol) -> NCIAnalyzer:
    """Build an xyzgraph ``NCIAnalyzer`` from ``mol``, with bonds from its graph and positions from its conformer."""
    from xyzgraph.data_loader import DATA
    from xyzgraph.nci import NCIAnalyzer

    conf = mol.GetConformer()
    g = nx.Graph()
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        g.add_node(
            a.GetIdx(),
            symbol=a.GetSymbol(),
            position=(p.x, p.y, p.z),
            formal_charge=a.GetFormalCharge(),
        )
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        attributes = {
            "bond_order": 1.5 if b.GetIsAromatic() else b.GetBondTypeAsDouble(),
            "metal_coord": g.nodes[i]["symbol"] in DATA.metals or g.nodes[j]["symbol"] in DATA.metals,
        }
        if b.GetBondType() == Chem.BondType.DATIVE:
            attributes["dative_donor"] = i
        g.add_edge(i, j, **attributes)
    for i in g.nodes:
        g.nodes[i]["valence"] = sum(
            g[i][j]["bond_order"] for j in g.neighbors(i) if g.nodes[j]["symbol"] not in DATA.metals
        )
    # xyzgraph takes this list as the pi rings, so it holds aromatic rings only; SymmSSSR rewrites RingInfo, so a copy.
    rings = Chem.Mol(mol)
    g.graph["aromatic_rings"] = [
        tuple(map(int, r)) for r in Chem.GetSymmSSSR(rings) if all(rings.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]
    return NCIAnalyzer(g)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContactKind:
    """Define one contact kind: the xyzgraph detector types it reads and how an embed holds it.

    An atom row holds the anchor-acceptor distance `window` in Å, a ring row the anchor's `height` in Å over the ring
    centroid, and a row with neither has no hold. `via` names an H anchor, an R apex opposite a sigma hole, or none, and
    `orient` the preferred donor-H...acceptor or R-X...acceptor angle in degrees. These are seed biases, not detection
    criteria.
    """

    name: str
    reads: tuple[str, ...] = ()
    window: tuple[float, float] | None = None
    height: float | None = None
    orient: tuple[float, float] | None = None
    via: str | None = None


_CHB = ContactKind(
    name="ChB",
    reads=("chalcogen_bond",),
    window=(3.0, 3.6),
    orient=(155.0, 180.0),
    via="apex",
)

# one row per kind; each window or height is a seed bias, not a fit, and a pi stack is unholdable
KINDS = {
    k.name: k
    for k in [
        ContactKind(
            name="HB",
            reads=("hbond", "hbond_bifurcated", "salt_bridge"),
            window=(1.6, 2.2),
            orient=(140.0, 180.0),
            via="anchor",
        ),
        replace(_CHB, name="XB", reads=("halogen_bond",), window=(2.5, 3.1), orient=(160.0, 180.0)),
        _CHB,
        replace(_CHB, name="PnB", reads=("pnictogen_bond",)),
        ContactKind(name="IONIC", reads=("ionic",), window=(2.6, 3.8)),
        ContactKind(name="CATLP", reads=("cation_lp",), window=(2.6, 3.4)),
        ContactKind(name="CHPI", reads=("ch_pi",), height=3.0, via="anchor"),
        ContactKind(name="HBPI", reads=("hb_pi",), height=3.0, via="anchor"),
        ContactKind(name="CATPI", reads=("cation_pi",), height=3.5),
        ContactKind(name="ANPI", reads=("anion_pi",), height=3.5),
        ContactKind(name="HALPI", reads=("halogen_pi",), height=3.5),
        ContactKind(name="PIPI", reads=("pi_pi_parallel",)),
        ContactKind(name="PIPIT", reads=("pi_pi_t_shaped",)),
        ContactKind(name="PIPIRD", reads=("pi_pi_ring_domain",)),
        ContactKind(name="PIPIDD", reads=("pi_pi_domain_domain",)),
    ]
}
_ROW_OF = {read: kind.name for kind in KINDS.values() for read in kind.reads}  # detector type -> KINDS row


@dataclass(frozen=True, slots=True, kw_only=True)
class Contact:
    """Name one candidate, request or observed contact in the Mol's atom order.

    `kind` selects a `KINDS` row; `site_a` and `site_b` keep the native donor and acceptor roles. `via` names an H or R
    reference, None for none or an ordinary H still to be allocated. Measured geometry stays on xyzgraph's NCIData.
    """

    kind: str
    site_a: tuple[int, ...]
    site_b: tuple[int, ...]
    via: int | None = None


def _anchor(contact):
    """Return the contact's H when chosen, otherwise its donor atom."""
    return contact.via if KINDS[contact.kind].via == "anchor" and contact.via is not None else contact.site_a[0]


def _donor_key(contact):
    """Distinguish deferred donor-H capacity from concrete atom resources, including shared HB/HBPI hydrogens."""
    return (
        ("hydrogen", contact.site_a[0])
        if KINDS[contact.kind].via == "anchor" and contact.via is None
        else ("atom", _anchor(contact))
    )


def _label(mol, contact):
    """Name a contact by its kind, sites and optional reference: ``'HB:O3H9->N5'`` or ``'XB:Cl1[C0]->N2'``.

    References appended beyond the input Mol are H placed for observation.
    """
    a, b = (_site_label(mol, site) for site in (contact.site_a, contact.site_b))
    reference = ""
    if contact.via is not None:
        reference = atom_label(mol, contact.via) if contact.via < mol.GetNumAtoms() else f"H{contact.via}"
        if KINDS[contact.kind].via != "anchor":
            reference = f"[{reference}]"
    return f"{contact.kind}:{a}{reference}->{b}"


def _site_label(mol, site):
    """Name a site as every contact label does: its atom, ``'N7'``, or a ring's sorted atoms, ``'pi(5,6,7,8,9,10)'``."""
    return atom_label(mol, site[0]) if len(site) == 1 else f"pi({','.join(map(str, sorted(site)))})"


def _ordinary_hydrogens(mol, donor):
    """Return the neutral terminal hydrogens on a non-H nonmetal donor, in neighbour order.

    Metal, molecular, charged and bridging hydrogens are left out, and a reference a caller or detector named is never
    reassigned.
    """
    from xyzgraph.data_loader import DATA

    atom = mol.GetAtomWithIdx(donor)
    if atom.GetAtomicNum() == 1 or atom.GetSymbol() in DATA.metals:
        return ()
    return tuple(
        h.GetIdx()
        for h in atom.GetNeighbors()
        if h.GetAtomicNum() == 1 and h.GetDegree() == 1 and h.GetFormalCharge() == 0
    )


def _opposite_reference(contact, positions):
    """Rank native eligible references by how nearly opposite the acceptor they lie for a seed."""
    donor, acceptor = contact.site_a[0], contact.site_b[0]
    vr = positions[contact.via] - positions[donor]
    va = positions[acceptor] - positions[donor]
    return -float(vr @ va / (np.linalg.norm(vr) * np.linalg.norm(va) + 1e-9))


def _atom_over_ring(pos, atom, ring, d_centroid, max_centroid):
    """Return the distance windows to each ring atom that hold `atom` ``d_centroid`` above the ring centroid.

    The height becomes sqrt(d^2 + r^2), r the ring atom's distance from the centroid, with no centroid dummy.
    """
    c = pos[list(ring)].mean(0)
    d_centroid = min(d_centroid, max_centroid)
    out = {}
    for k in ring:
        radius_sq = float(np.linalg.norm(pos[k] - c)) ** 2
        dk = (d_centroid**2 + radius_sq) ** 0.5
        upper = min(round(dk + 0.3, 2), (max_centroid**2 + radius_sq) ** 0.5)
        out[canonical_key((atom, k))] = (round(dk - 0.3, 2), upper)
    return out


# One probe seed for every reader, never the embed's: `rx.contacts` lists the modes `contacts='auto'` embeds.
_PROBE_SEED = 0xC0FFEE


def _rough(mol):
    """Return `mol` with the one conformer contact discovery reads: its first, else a probe at `_PROBE_SEED`.

    The geometry decides the sigma-hole apex, the rings xyzgraph reads and the ring radii.
    """
    if mol.GetNumConformers():
        return Chem.Mol(mol, False, mol.GetConformer().GetId())
    work = probe_conformer(mol, _PROBE_SEED)
    if work is None:
        raise ValueError("could not embed a probe conformer to read contacts on; pass a Mol with a conformer")
    return work


def candidate_contacts(work, kinds):
    """Return native candidates in requested kind order, retaining every site and reference."""
    found = [
        Contact(kind=c.kind, site_a=c.site_a, site_b=c.site_b, via=c.aux_atoms[0] if c.aux_atoms else None)
        for c in analyzer(work).candidates()
    ]
    return tuple(c for name in kinds for c in found if c.kind == name)


def _construction_contacts(work, kinds):
    """Choose the inter-fragment holds, hydrogen budget and seed references from eligible candidates.

    Automatic requests take the first native H-bond reference, with ordinary hydrogens allocated later and protected
    ones kept, and the most opposite sigma-hole reference. The strongest style wins on each site pair, ties keeping the
    requested kind order.
    """
    pos, frag = work.GetConformer().GetPositions(), frag_map(work)
    found = {}
    for candidate in candidate_contacts(work, kinds):
        contact = candidate
        row = KINDS[contact.kind]
        if not (row.window or row.height) or len({frag[i] for i in (*contact.site_a, *contact.site_b)}) < 2:  # noqa: PLR2004
            continue
        ordinary = _ordinary_hydrogens(work, contact.site_a[0]) if row.via == "anchor" else ()
        if contact.via in ordinary:
            contact = replace(contact, via=None)
        donor = contact.site_a if contact.kind == "HB" else _donor_key(contact)
        key = (contact.kind, donor, contact.site_b)
        if key not in found or (
            row.via == "apex" and _opposite_reference(contact, pos) > _opposite_reference(found[key], pos)
        ):
            found[key] = contact
    pair = {c: frozenset((frozenset(c.site_a), frozenset(c.site_b))) for c in found.values()}  # in either direction
    preferred = {}
    for contact in sorted(found.values(), key=lambda c: -_STRENGTH.get(c.kind, 0)):
        preferred.setdefault(pair[contact], contact.kind)
    return tuple(c for c in found.values() if c.kind == preferred[pair[c]])


def observed_contacts(mol, conf_ids, *, interfragment=False):
    """Yield the native contacts each conformer makes, optionally only between fragments.

    xyzgraph reads each conformer against topology built once from the first, and nothing moves. Implicit hydrogens are
    placed on a copy. A dative bond joins its atoms into one fragment.
    """
    conf_ids = tuple(conf_ids)
    if not conf_ids:
        return
    mol = Chem.AddHs(mol, addCoords=True)
    frag = frag_map(mol)
    if interfragment and len(set(frag.values())) < 2:  # noqa: PLR2004 - two fragments make an inter-fragment contact
        yield from (() for _ in conf_ids)
        return
    an = analyzer(mol)
    for cid in conf_ids:
        found = {}
        for n in an.detect(mol.GetConformer(int(cid)).GetPositions()):
            contact = Contact(
                kind=_ROW_OF[n.type],
                site_a=n.site_a,
                site_b=n.site_b,
                via=int(n.aux_atoms[0]) if n.aux_atoms else None,
            )
            if not interfragment or len({frag[a] for a in (*contact.site_a, *contact.site_b)}) > 1:
                found[contact] = contact
        yield tuple(found.values())


# modes read atom kinds by default: ring contacts are soft, and a rough conformer shows spurious ones
_AUTO_KINDS = ("HB", "XB", "ChB", "PnB", "IONIC", "CATLP")
_ANCHOR_KINDS = {"HB", "IONIC", "CATLP"}  # strong enough to make a mode alone; the rest ride along one
# a directional H-bond defines a structure; a bare ion pair (R4N+...Cl-) is strong but undirected
_STRENGTH = {"CATLP": 4, "HB": 4, "IONIC": 3, "XB": 3, "ChB": 2, "PnB": 2}
# Only bent contacts share an acceptor (a bifurcated clamp); several sigma holes cannot all aim at one.
_BIFURCATING = {"HB", "IONIC", "CATLP"}
_ACCEPTOR_CAP = 2  # bent contacts one acceptor takes in a binding mode
_MAX_MODES = 8  # binding modes kept, best first


def _resolve_reciprocal(sk):
    """Return `sk` once per choice of one direction of each reciprocal pair it grips.

    A-H...B with B-H...A cannot both hold in one pose, so each direction is its own binding mode; a larger H-bond ring
    passes unchanged.
    """
    reciprocal = [(c, o) for i, c in enumerate(sk) for o in sk[i + 1 :]]
    pairs = [(c, o) for c, o in reciprocal if (c.site_a, c.site_b) == (o.site_b, o.site_a)]
    if not pairs:
        return [sk]
    variants = [list(sk)]
    for c, o in pairs:
        nxt = []
        for v in variants:
            if c in v and o in v:
                nxt.append([x for x in v if x != o])
                nxt.append([x for x in v if x != c])
            else:
                nxt.append(v)
        variants = nxt
    return variants


def _acc_key(c):
    return ("atom", c.site_b[0]) if len(c.site_b) == 1 else ("ring", *c.site_b)


def _accepts(c, count):
    """Return whether the acceptor of `c` can take it: bent contacts up to `_ACCEPTOR_CAP`, a sigma hole only alone.

    `count[acc]` is ``[#bent, #linear]`` already seated there. A sigma hole aims along a lone pair the grip already
    uses, so it rides along only on an acceptor no other contact takes.
    """
    bent, lin = count[_acc_key(c)]
    return bent < _ACCEPTOR_CAP if c.kind in _BIFURCATING else bent == lin == 0


def _seat(c, count, delta):
    cls = 0 if c.kind in _BIFURCATING else 1
    count[_acc_key(c)][cls] += delta


_MAX_ASSIGNMENTS = 256  # a count bound on the enumeration, not a score bound: hitting it warns


def _can_saturate(contact, count, remaining):
    """Return whether remaining donor groups could fill `contact`'s acceptor, counting each group once."""
    if not _accepts(contact, count):
        return True
    key = _acc_key(contact)
    bent = contact.kind in _BIFURCATING
    possible = {key: list(count[key])}
    for _donor, options in remaining:
        if any(_acc_key(c) == key and (c.kind in _BIFURCATING) == bent for c in options):
            _seat(contact, possible, +1)
    return not _accepts(contact, possible)


def _maximal_assignments(by_near):
    """Return every maximal donor-to-acceptor assignment of the anchor contacts, up to `_MAX_ASSIGNMENTS`.

    `by_near` is ``[(anchor, [contacts]), ...]``, each donor's mutually exclusive contacts. A donor is skipped only when
    every option is blocked, and a branch is pruned when the remaining donors cannot fill an acceptor.
    """
    out = []
    truncated = False

    def bt(i, chosen, count, skipped):
        nonlocal truncated
        if len(out) >= _MAX_ASSIGNMENTS:
            truncated = True
            return
        if any(not _can_saturate(c, count, by_near[i:]) for c in skipped):
            return
        if i == len(by_near):
            out.append(list(chosen))
            return
        placeable = [c for c in by_near[i][1] if _accepts(c, count)]
        if not placeable:
            bt(i + 1, chosen, count, skipped)
            return
        for c in placeable:
            _seat(c, count, +1)
            chosen.append(c)
            bt(i + 1, chosen, count, skipped)
            chosen.pop()
            _seat(c, count, -1)
        bt(i + 1, chosen, count, (*skipped, *placeable))

    bt(0, [], defaultdict(lambda: [0, 0]), ())
    if truncated:  # depth-first, so the kept assignments favour the first donors' first acceptors
        logger.warning(
            "contacts: stopped at %d assignments and may miss the best grip; see rx.contacts(mol, modes=False)",
            _MAX_ASSIGNMENTS,
        )
    return out


def _augment_with_aux(combo, aux, mol):
    """Add auxiliary contacts to an anchor grip where their donors and acceptors have capacity; return `combo`.

    A weak contact only strengthens a grip an anchor makes.
    """
    used = defaultdict(int)
    count = defaultdict(lambda: [0, 0])
    for c in combo:
        used[_donor_key(c)] += 1
        _seat(c, count, +1)
    for c in aux:
        donor = _donor_key(c)
        capacity = len(_ordinary_hydrogens(mol, donor[1])) if donor[0] == "hydrogen" else 1
        if used[donor] < capacity and _accepts(c, count):
            combo.append(c)
            used[donor] += 1
            _seat(c, count, +1)
    return combo


def _mode_score(mol, contacts):
    """Return a binding mode's sort key: contact count, then strength, acceptor basicity and bifurcated clamps.

    An anionic acceptor outranks a neutral one. xyzgraph offers no N whose lone pair a pi system spends (amide,
    aniline, pyrrole), so among neutral acceptors an N keeps its pair and outranks O, S and halides.
    """
    accs = defaultdict(int)
    for c in contacts:
        accs[_acc_key(c)] += 1
    return (
        len(contacts),  # most contacts (best grip)
        sum(_STRENGTH.get(c.kind, 0) for c in contacts),  # then strongest contact types
        sum(len(c.site_b) == 1 and mol.GetAtomWithIdx(c.site_b[0]).GetFormalCharge() < 0 for c in contacts),
        sum(len(c.site_b) == 1 and mol.GetAtomWithIdx(c.site_b[0]).GetAtomicNum() == 7 for c in contacts),  # noqa: PLR2004  nitrogen
        sum(1 for v in accs.values() if v > 1),  # then bifurcated clamps
    )


def _mode_label(mol, contacts):
    """Name a binding mode by its contacts, grouped by acceptor, as ``kinds:anchors->acceptor`` parts."""
    groups = defaultdict(list)
    for c in contacts:
        groups[_acc_key(c)].append(c)
    parts = []
    for ak, gs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ks = "/".join(sorted({c.kind for c in gs}))
        donors = ",".join(
            atom_label(mol, c.site_a[0]) + (atom_label(mol, c.via) if c.via is not None and _anchor(c) == c.via else "")
            for c in gs
        )  # the donor atom, then its H once chosen, as one contact's label names it
        parts.append(f"{ks}:{donors}->{_site_label(mol, ak[1:])}")
    return " + ".join(parts)


def _symmetry_key(mol, contacts):
    """Return the canonical SMILES of `mol`'s `flat_graph` with each contact drawn in as dummy atoms.

    Modes a graph symmetry maps onto each other share it, the drawn Lewis form aside: an H-bond to either O of a
    carboxylate is one mode.
    """
    rw = Chem.RWMol(flat_graph(mol))
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(0)
    for c in contacts:
        role = 3 * list(KINDS).index(c.kind) + 1  # an isotope per kind and end: donor, acceptor, reference
        ends = []
        for offset, atoms in enumerate((c.site_a, c.site_b, () if c.via is None else (c.via,))):
            dummy = rw.AddAtom(Chem.Atom(0))
            rw.GetAtomWithIdx(dummy).SetIsotope(role + offset)
            for atom in atoms:
                rw.AddBond(dummy, atom, Chem.BondType.SINGLE)
            ends.append(dummy)
        rw.AddBond(ends[0], ends[1], Chem.BondType.SINGLE)
        rw.AddBond(ends[0], ends[2], Chem.BondType.SINGLE)
    rw.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(rw)
    return Chem.MolToSmiles(rw)


def auto_binding_modes(work, kinds=_AUTO_KINDS):
    """Return up to `_MAX_MODES` binding modes on `work`, best first, as ``{label: (Contact, ...)}``.

    Anchors are matched to acceptors first, up to `_ACCEPTOR_CAP` each, and reciprocal directions split; auxiliary
    contacts then join a grip, and with no anchor each auxiliary is its own mode. Modes rank by `_mode_score`.
    ``embed(contacts='auto')`` embeds each, and ``embed(contacts=mode)`` one.
    """
    cands = _construction_contacts(work, kinds)
    anchors = [c for c in cands if c.kind in _ANCHOR_KINDS]
    aux = sorted((c for c in cands if c.kind not in _ANCHOR_KINDS), key=lambda c: -_STRENGTH.get(c.kind, 0))
    by_near = {}
    for c in anchors:
        by_near.setdefault(_donor_key(c), []).append(c)
    skeletons = _maximal_assignments(list(by_near.items())) if anchors else [[c] for c in aux]
    unique: dict[str, list[Contact]] = {}
    for skeleton in skeletons:
        for choice in _resolve_reciprocal(skeleton):
            if choice:
                unique.setdefault(_symmetry_key(work, choice), choice)
    ranked = sorted(unique.values(), key=partial(_mode_score, work), reverse=True)

    modes = {}
    for sk in ranked[:_MAX_MODES]:
        mode = tuple(_augment_with_aux(list(sk), aux, work) if anchors else sk)
        modes[_mode_label(work, mode)] = mode
    return modes


def contacts(mol, *, modes=True, kinds=None, observed=False, conf_id=-1):
    """Return labelled tuples of non-covalent contacts of `mol`, to pass to ``embed(contacts=)``.

    ``observed=True`` reads conformer `conf_id` without moving it, and ``modes=False`` lists the candidates; both give
    one-contact tuples. Otherwise the result is the inter-fragment binding modes, read on the first conformer or a
    probe. `kinds` selects `KINDS` rows: directional and ionic kinds for modes by default, every row otherwise.
    Hydrogens are added on a copy, heavy-atom indices kept. Pass the whole result to embed every mode, or one value.
    """
    mol = require_mol(mol, "rx.contacts", geometry=observed)
    chosen = tuple(KINDS) if kinds is None else tuple(kinds)
    if unknown := [name for name in chosen if name not in KINDS]:
        raise ValueError(f"unknown contact kind(s) {unknown}; choose from {list(KINDS)}")
    if observed:
        return {_label(mol, c): (c,) for c in next(observed_contacts(mol, [conf_id])) if c.kind in chosen}
    mol = Chem.AddHs(mol, addCoords=bool(mol.GetNumConformers()))
    work = _rough(mol)
    if modes:
        return auto_binding_modes(work, kinds=_AUTO_KINDS if kinds is None else chosen)
    return {_label(work, contact): (contact,) for contact in candidate_contacts(work, chosen)}


def parse_contacts(contacts):
    """Return a ``contacts=`` request parsed once: None, 'auto', 'input', one mode, or ``{label: mode}`` modes.

    A mode is one `Contact` or a sequence of them, held together; each must be of a kind with a hold. The labelled
    modes `rx.contacts` returns, or any subset of them, give one result per mode, as 'auto' does.
    """
    if contacts is None or (isinstance(contacts, str) and contacts in ("auto", "input")):
        return contacts
    if isinstance(contacts, dict):
        if not all(isinstance(label, str) for label in contacts):
            raise TypeError(
                "contacts= takes modes labelled as rx.contacts labels them; a {(i, j): (lo, hi)} window goes to "
                "constrain="
            )
        return {label: _held_mode(mode) for label, mode in contacts.items()}
    return _held_mode(contacts) or None


def _held_mode(mode):
    """Return one ``contacts=`` mode as a tuple of Contacts, refusing anything else or a kind with no hold."""
    held = (mode,) if isinstance(mode, (Contact, dict, str)) else tuple(mode)
    if not all(isinstance(c, Contact) for c in held):
        raise TypeError("contacts= takes 'auto', 'input', or rx.contacts' modes or one of them")
    rows = [(c, KINDS.get(c.kind)) for c in held]
    if unheld := sorted(
        {
            c.kind
            for c, row in rows
            if not (
                row
                and (row.window or row.height)
                and ((c.via is None) == (row.via is None) or row.via == "anchor" or row.name == "HALPI")
            )
        }
    ):
        raise ValueError(
            f"contacts= cannot hold this {'/'.join(unheld)} contact; it holds a Contact as rx.contacts gives it, of "
            f"a kind in {[k for k, v in KINDS.items() if v.window or v.height]}"
        )
    return held


def contact_modes(disc, contacts):
    """Return ``{mode label: terms}`` for `disc` under a parsed ``contacts=``.

    'input' gives the one mode of the directional and ionic contacts (`_AUTO_KINDS`) between fragments that the first
    conformer makes, and names the pi contacts it leaves; 'auto' gives every `auto_binding_modes` mode, given modes
    each under its label, one given mode under the label its contacts write, and none one mode labelled None.
    """
    if contacts == "input":
        if not disc.GetNumConformers():
            raise ValueError("contacts='input' holds the contacts an input geometry makes; pass an .xyz or use 'auto'")
        observed = next(observed_contacts(disc, [-1], interfragment=True))
        mode = tuple(c for c in observed if c.kind in _AUTO_KINDS)
        if left := sorted({_label(disc, c) for c in observed if c.kind not in _AUTO_KINDS}):
            hint = "pass one from rx.contacts(observed=True) to hold it"
            logger.info("contacts='input' leaves %s; %s", ", ".join(left), hint)
        if not mode:
            logger.info("contacts='input': the input geometry makes no NCI contact -> plain embed")
        return {_mode_label(disc, mode): contact_terms(mode, disc)} if mode else {None: None}
    if isinstance(contacts, dict):
        return {label: contact_terms(mode, disc) for label, mode in contacts.items()} or {None: None}
    if isinstance(contacts, tuple):
        return {_mode_label(disc, contacts): contact_terms(contacts, disc)}
    if contacts is None:
        return {None: None}
    try:
        work = _rough(disc)
        modes = auto_binding_modes(work)
    except (RuntimeError, ValueError, nx.NetworkXException) as err:  # RDKit's probe embed and the contact reading
        raise ValueError(
            f"contacts='auto' found no binding modes ({type(err).__name__}: {err}); give a geometry or contacts= "
            "explicitly"
        ) from err
    if not modes:
        logger.info("contacts='auto': no inter-fragment NCI binding mode detected -> plain embed")
        return {None: None}
    logger.info("contacts='auto': %d binding modes", len(modes))
    return {label: contact_terms(mode, work) for label, mode in modes.items()}


def contact_terms(contacts, mol):
    """Return the windows that hold `contacts` on `mol` as one ``{key: window}`` dict, or None.

    Explicit H references are reserved first; each other request takes its donor's first free ordinary H, identical
    requests spending one. An atom contact is held as the anchor-acceptor `window` plus, for a directional row, the
    `orient` angle; a ring contact as one distance per ring atom that holds the anchor at its `height` over the
    centroid, with radii from the conformer or a probe. Each height, and each window but HB's, ends at xyzgraph's cutoff
    for that candidate, past which it is never detected (an HB window holds H...A, and xyzgraph screens D...A);
    relaxation can still move a contact out of detection.
    """
    from xyzgraph.nci import NCICandidate

    contacts = tuple(dict.fromkeys(contacts))
    reserved = {c.via for c in contacts if KINDS[c.kind].via == "anchor" and c.via is not None}
    work = _rough(mol)
    nci, pos = analyzer(work), work.GetConformer().GetPositions()
    terms = {}
    for contact in contacts:
        kind, anchor = KINDS[contact.kind], _anchor(contact)
        if kind.via == "anchor" and contact.via is None:
            ordinary = _ordinary_hydrogens(mol, contact.site_a[0])
            anchor = next((h for h in ordinary if h not in reserved), None)
            if anchor is None or anchor in reserved:
                raise ValueError(
                    f"contacts= has no free ordinary H on {atom_label(mol, contact.site_a[0])}; "
                    "request fewer contacts or name an explicit via H to share it"
                )
            reserved.add(anchor)
        cutoff = nci.cutoff(NCICandidate(kind=contact.kind, site_a=contact.site_a, site_b=contact.site_b))
        if kind.window is None:  # a ring row
            terms.update(_atom_over_ring(pos, anchor, contact.site_b, kind.height, cutoff))
            continue
        (lo, hi), acceptor = kind.window, contact.site_b[0]
        terms[canonical_key((anchor, acceptor))] = (lo, hi if kind.name == "HB" else min(hi, cutoff))
        if kind.orient:  # hold it linear
            terms[(contact.via if kind.via == "apex" else contact.site_a[0], anchor, acceptor)] = kind.orient
    return terms or None
