"""Droppable adapter: rxembed's embedding ENGINE behind OIN's perception + losslessness shell.

Drop this package into an environment where both ``oinsmiles`` and ``rxembed`` import. It swaps the
embed engine — the DG bounds + restrained-UFF physics is entirely rxembed's — while PERCEPTION
(``oinsmiles.utils.cxsmiles.cxsmiles_to_mol``) and the CHIRALITY / byte-losslessness gate
(``oinsmiles.generation.rdkit_embed._accept``) stay OIN's, reused verbatim, never rebuilt.

Public API (mirrors OIN's own signatures):
    embed_cxsmiles(cxsmiles, n=24) -> Mol         # the ONE encoded isomer, hand-culled by OIN's gate
    enumerate_cxsmiles(cxsmiles, n=8) -> list[Mol] # ALL coordination isomers, embedded + validated
    validate_geometry(mol, metal=None, donors=None) -> dict  # index-free internal-chemistry verdict

OIN symbols imported:
    perception     — oinsmiles.utils.cxsmiles.cxsmiles_to_mol / strip_to_dative_smiles / geo_of
    chirality gate — oinsmiles.generation.rdkit_embed._accept (mirror-try + byte-exact re-encode)
"""

from __future__ import annotations

import logging

from rdkit import Chem

import rxembed as rx

from .parse import Fallback, parse_cxsmiles
from .reconstruct import reconstruct
from .validate import validate_geometry

__all__ = ["Fallback", "embed_cxsmiles", "enumerate_cxsmiles", "validate_geometry"]

logger = logging.getLogger(__name__)


def embed_cxsmiles(cxsmiles: str, n: int = 24) -> Chem.Mol:
    """Reconstruct the ONE isomer an OIN cxSMILES encodes, correctly handed (OIN's gate accepts it).

    Perceives via OIN, builds the fixed rxembed Isomer from the encoded slots, embeds + UFF-refines
    with rxembed's engine, and keeps the first conformer OIN's ``_accept`` byte-matches to `cxsmiles`.
    Falls back to the enumerate path (best validated isomer) for a multi-metal / haptic / unclassified
    complex the fixed-isomer builder cannot honour.
    """
    try:
        p = parse_cxsmiles(cxsmiles)
    except Fallback as e:
        logger.info("embed_cxsmiles: %s -> enumerate fallback", e)
        cands = enumerate_cxsmiles(cxsmiles, n=max(4, n // 3))
        if not cands:
            raise ValueError(f"no valid isomer embedded on the enumerate fallback ({e})") from e
        return cands[0]
    return reconstruct(p, cxsmiles, n=n)


def enumerate_cxsmiles(cxsmiles: str, n: int = 8) -> list[Chem.Mol]:
    """Embed EVERY coordination isomer of the (annotation-stripped) dative SMILES; return the sane ones.

    Strips the ``|atomProp:...|`` block, enumerates isomers with ``rx.metal``, embeds each with
    ``rx.embed(iso).minimize()``, restores the real metal, and keeps those that pass
    ``validate_geometry``. Order-agnostic to the encoded slots — this is the Part-1 breadth path.
    """
    from oinsmiles.utils.cxsmiles import strip_to_dative_smiles

    from .reconstruct import OIN_DISTANCE_FC

    plain = strip_to_dative_smiles(cxsmiles)
    out: list[Chem.Mol] = []
    for iso in rx.metal(plain):
        try:  # OIN's wall stiffness, as the fixed path uses — rxembed's softer default drops otherwise-sane
            # geometries outright (measured: XAMPUY 0/1 conformers survive by default vs 1/1 at 1e5)
            ens = rx.embed(iso, stereo="free", n=n).minimize(distance_fc=OIN_DISTANCE_FC)
        except Exception as e:  # a single infeasible isomer must not sink the set
            logger.info("enumerate_cxsmiles: isomer %s failed to embed (%s)", iso.label, e)
            continue
        if not ens.ids:
            continue
        iso.restore()  # hand the real element + oxidation state back before validating
        for cid in ens.ids:
            verdict = validate_geometry(ens.mol, metal=iso.metal, donors=iso.donors, cid=cid, real_z=iso.real_z)
            if verdict["ok"]:
                m = Chem.Mol(ens.mol)
                m.RemoveAllConformers()
                m.AddConformer(ens.mol.GetConformer(cid), assignId=True)
                out.append(m)
                break
    return out
