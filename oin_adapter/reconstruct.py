"""Build ONE fixed rxembed Isomer from an OIN slot map, embed it, cull the wrong hand via OIN's gate.

The embed ENGINE is entirely rxembed's (DG bounds + restrained-UFF, through `rx.embed(iso).minimize()`).
The chirality / losslessness gate is entirely OIN's (`_accept`, mirror-try + byte-exact re-encode) — we
do NOT invent a geometric hand check. This module is the marshalling seam between the two.
"""

from __future__ import annotations

from oinsmiles.generation.rdkit_embed import _accept
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed import isomers as _isomers
from rxembed.rdkit_embed.constraints import coordination_builders as _cbuild
from rxembed.rdkit_embed.constraints import metal as _metal

from .parse import Perceived, slot_to_vertex

# OIN's confirmed restrained-UFF force constants (rdkit_embed.py): a stiff distance wall + soft pull
# + firm angle wall. rxembed's default minimize is softer (1e4) with auto-escalation to 1e6; we pin
# OIN's wall stiffness from the start so a tight sphere seats the same way OIN's does.
OIN_DISTANCE_FC = 1e5
OIN_ANGLE_FC = 2e4
OIN_MAX_ITS = 400


def _build_isomer(p: Perceived) -> tuple[_metal.Isomer, list[int], list[int], Chem.Mol]:
    """Construct the single fixed Isomer honouring the OIN slot map (no permutation enumeration).

    Returns (isomer, padded, donors, dative_ref) where `dative_ref` is the AddHs'd, dative-bonded,
    real-metal Mol whose atom order matches the embedded ensemble — the template for the OIN gate.
    """
    geometry = p.rx_geometry
    dative_ref = Chem.AddHs(p.mol, addCoords=False)  # keep the real metal + DATIVE bonds for the gate

    base, m, donors, real_z, real_q = _metal.surrogate_metal(dative_ref)  # surrogate + strip M-donor bonds
    base, donors, haptic = _metal._collapse_haptic(base, donors)  # no-op here (parse rejects haptic)
    sites = _metal.n_sites(geometry)
    padded = list(donors) + [_metal.VACANT] * (sites - len(donors))

    # order[vertex] = index-into-padded of the donor seated at that vertex.
    pos_in_padded = {d: k for k, d in enumerate(padded) if d != _metal.VACANT}
    order: list[int | None] = [None] * sites
    for d, oin_slot in p.slot_of.items():
        v = slot_to_vertex(geometry, oin_slot)
        if not (0 <= v < sites) or order[v] is not None:
            from .parse import Fallback

            raise Fallback(f"slot {oin_slot}->vertex {v} out of range / collision for {geometry}")
        order[v] = pos_in_padded[d]
    vacant_k = [k for k, d in enumerate(padded) if d == _metal.VACANT]
    for v in range(sites):
        if order[v] is None:
            order[v] = vacant_k.pop()
    assert None not in order, order  # every vertex seated
    assert sorted(order) == list(range(sites)), order  # a valid permutation of padded indices

    cons = _cbuild.coordination(base, m, padded, geometry, order, real_z, haptic=haptic)
    od = [padded[k] for k in order]
    real_donors = [d for d in donors if d not in haptic] + sorted({a for ring in haptic.values() for a in ring})
    iso = _metal.Isomer(
        _metal.strip_phantoms(Chem.Mol(base), set(haptic)),
        cons,
        m,
        real_donors,
        real_z,
        real_q,
        _isomers._order_label(base, padded, geometry, order),
        geometry,
        od,
        chirality=_metal.chirality_of(base, donors, geometry, od),  # the TARGET metal Λ/Δ
        haptic=dict(haptic),
    )
    return iso, padded, donors, dative_ref


def _cand_for(dative_ref: Chem.Mol, ens_mol: Chem.Mol, cid: int) -> Chem.Mol:
    """Materialise the chemically-complete metal-dative Mol for one conformer (OIN gate input).

    `dative_ref` shares atom order with `ens_mol` (both descend from the same AddHs'd mol), so the
    conformer's coordinates transfer index-for-index onto the real-metal, dative-bonded reference —
    exactly the shape OIN's `_accept` re-perceives from geometry.
    """
    cand = Chem.Mol(dative_ref)
    src = ens_mol.GetConformer(cid)
    conf = Chem.Conformer(cand.GetNumAtoms())
    for i in range(cand.GetNumAtoms()):
        p = src.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x, p.y, p.z))
    cand.RemoveAllConformers()
    cand.AddConformer(conf, assignId=True)
    return cand


def reconstruct(p: Perceived, cx: str, n: int = 24) -> Chem.Mol:
    """Embed the ONE encoded isomer with rxembed, keep the first conformer OIN's gate accepts.

    `cx` MUST be the FULL cxSMILES (including the ``|atomProp:...|`` suffix) — that is the string
    `_accept` byte-compares against. Returns a real-metal dative Mol with a 3D conformer; raises when
    no embedded conformer re-encodes to `cx`.
    """
    iso, _padded, _donors, dative_ref = _build_isomer(p)

    # rxembed engine: single candidate (stereo='free'), OIN wall stiffness, auto-escalation still on.
    ens = rx.embed(iso, stereo="free", n=n).minimize(distance_fc=OIN_DISTANCE_FC)

    for cid in ens.ids:
        cand = _cand_for(dative_ref, ens.mol, cid)
        if _accept(cand, cx, Chem.GetFormalCharge(cand)):  # mirror-tried byte-exact re-encode; leaves cand handed
            return cand
    raise ValueError(f"no embedded conformer re-encodes to the target isomer ({len(ens.ids)} tried)")
