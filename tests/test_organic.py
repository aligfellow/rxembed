"""Organic reactivity — NCI binding-mode discovery through the new API.

The complement to ``test_embed_core`` / ``test_frozen``: the inter-fragment NCI path (``contacts='auto'`` →
one candidate per grip; a specific ``rx.nci_modes(...)`` contact), verifying both that a mode is
discovered *and* that the seeded grip actually forms (donor-acceptor at an H-bond distance). All xtb-free
(discovery is RDKit + xyzgraph; enforcement is restrained UFF). Reaction TS cores from SMILES numbers (W3)
are covered in ``test_embed_core``.
"""

import pytest

from rxembed import geometry as geom


@pytest.mark.parametrize(
    "complex_smiles",
    [
        "OC(=O)c1ccccc1.n1ccccc1",  # carboxylic acid + pyridine -> O-H···N
        "CC(=O)[O-].C[NH3+]",  # acetate + methylammonium -> salt-bridge H-bond
    ],
)
def test_nci_binding_modes_discovered_and_clean(complex_smiles):
    import rxembed as rx

    es = rx.embed(complex_smiles, contacts="auto", n=6)  # -> Ensemble or EnsembleSet, one per grip
    ensembles = list(es) if isinstance(es, rx.EnsembleSet) else [es]
    assert ensembles
    for ens in ensembles:
        assert ens.n >= 1
        grip = ens.cons.contacts[0]  # the seeded inter-fragment contact pair(s)
        assert grip, "a discovered binding mode must seed a releasable contact"
        settled = ens.minimize()
        for cid in settled.ids:
            geom.check(settled.mol, cid).assert_ok()
        for pair in grip:  # the grip actually formed: the contact sits at an H-bond distance, not arbitrary
            assert settled.measure(pair)["mean"] < 2.6  # H-bond / salt-bridge donor-acceptor distance


def test_specific_nci_mode_by_contact():
    import rxembed as rx

    modes = rx.nci_modes(_pyridine_acid_mol())
    assert modes  # at least the O-H···N grip
    label = next(iter(modes))
    ens = rx.embed("OC(=O)c1ccccc1.n1ccccc1", contacts=modes[label], n=6).minimize()
    assert ens.n >= 1
    for cid in ens.ids:
        geom.check(ens.mol, cid).assert_ok()


def _pyridine_acid_mol():
    from rxembed.embed.dispatch import _normalize

    return _normalize("OC(=O)c1ccccc1.n1ccccc1")[0]
