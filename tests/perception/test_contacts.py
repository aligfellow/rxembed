"""`perception/contacts.py`: the `KINDS` registry, the contacts a structure offers, binding modes and their holds."""

from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

import rxembed as rx
from rxembed.geometry.bounds import probe_conformer
from rxembed.perception import contacts
from rxembed.perception.contacts import Contact
from tests.geometry_fixtures import placed

_ALL_PROPS = Chem.PropertyPickleOptions.AllProps
_X, _Y, _Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _hexagon(centre, u, v, radius, phase=0.0):
    angles = phase + np.arange(6) * np.pi / 3
    return [np.add(centre, radius * (np.cos(t) * np.array(u) + np.sin(t) * np.array(v))) for t in angles]


def _benzenes(centre, u, v, phase=0.0):
    """Return two benzenes: one flat at the origin, one centred at `centre` in the plane of `u` and `v`."""
    rings = [((0, 0, 0), _X, _Y, 0.0), (centre, u, v, phase)]
    carbons = [p for c, a, b, t in rings for p in _hexagon(c, a, b, 1.39, t)]
    return placed(_mol("c1ccccc1.c1ccccc1"), carbons + [p for c, a, b, t in rings for p in _hexagon(c, a, b, 2.48, t)])


def _butadiene(z):
    """Return s-trans butadiene's carbons and hydrogens in the plane at height `z`, centred on the z axis."""
    carbons = [(-1.40, 1.16, z), (-0.73, 0.0, z), (0.73, 0.0, z), (1.40, -1.16, z)]
    hydrogens = [(-2.48, 1.16, z), (-0.86, 2.10, z), (-1.27, -0.94, z), (1.27, 0.94, z), (2.48, -1.16, z)]
    return carbons, [*hydrogens, (0.86, -2.10, z)]


def _over_butadiene(smiles):
    """Return a benzene or a butadiene, flat at the origin, under a butadiene 3.6 A above it."""
    top_c, top_h = _butadiene(3.6)
    if smiles.startswith("c1"):
        ring = _hexagon((0, 0, 0), _X, _Y, 1.39)
        return placed(_mol(smiles), [*ring, *top_c, *_hexagon((0, 0, 0), _X, _Y, 2.48), *top_h])
    low_c, low_h = _butadiene(0.0)
    return placed(_mol(smiles), [*low_c, *top_c, *low_h, *top_h])


# Per kind with candidates, a pair it fires on and a near miss it must not fire on (None where none is pinned); a kind
# only a geometry makes has its geometry in OBSERVED. A new kind adds its row to one of them.
PARTNERS = {
    "HB": ("CO.n1ccccc1", None),
    "XB": ("CI.n1ccccc1", None),
    "ChB": ("CSC.n1ccccc1", "COC.n1ccccc1"),  # an ether O has no usable sigma hole; its S congener does
    "PnB": ("C[As](C)C.n1ccccc1", None),
    "IONIC": ("[Na+].[Cl-]", None),
    "CATLP": ("[Na+].O", None),
    "CHPI": ("c1ccccc1.C", None),
    "HBPI": ("c1ccccc1.O", None),
    "CATPI": ("c1ccccc1.[NH4+]", "C1CCCCC1.[NH4+]"),  # a saturated ring has no pi face
    "ANPI": ("c1ccccc1.[Cl-]", None),
    "HALPI": ("c1ccccc1.CI", None),
}
OBSERVED = {
    "PIPI": lambda: _benzenes((0, 0, 3.6), _X, _Y),
    "PIPIT": lambda: _benzenes((0, 0, 4.9), _X, _Z, phase=np.pi / 6),  # an edge H points at the lower face
    "PIPIRD": lambda: _over_butadiene("c1ccccc1.C=CC=C"),
    "PIPIDD": lambda: _over_butadiene("C=CC=C.C=CC=C"),
}


def _windows(terms, arity):
    return {key: window for key, window in terms.items() if len(key) == arity}


def test_each_contact_kind_is_found_and_held_by_its_row():
    """An atom contact holds its row's window, its top clipped to xyzgraph's cutoff but for HB, and its orientation; a
    ring contact holds its height over the centroid, within the cutoff; a kind with no hold is refused naming the kinds
    that can be held. A silent partner forms none."""
    from xyzgraph.nci import NCICandidate

    read, expected = {}, {}
    for kind in contacts.KINDS.values():
        audible, silent = PARTNERS.get(kind.name, (None, None))
        if audible:
            mol = probe_conformer(Chem.AddHs(rx.parse_smiles(audible)), 0xC0FFEE)
            found = rx.contacts(mol, modes=False, kinds=(kind.name,))
        else:
            mol = OBSERVED[kind.name]()
            found = rx.contacts(mol, observed=True, kinds=(kind.name,))
        held, holds = [], []
        for (contact,) in found.values():
            if not (kind.window or kind.height):
                with pytest.raises(ValueError, match="cannot hold"):
                    rx.embed(mol, contacts=contact, n=1)
                continue
            cutoff = contacts.analyzer(mol).cutoff(
                NCICandidate(kind=kind.name, site_a=contact.site_a, site_b=contact.site_b)
            )
            terms = contacts.contact_terms([contact], mol)
            spans = _windows(terms, 2)
            if kind.height is not None:
                pos = mol.GetConformer().GetPositions()
                centre = pos[list(contact.site_b)].mean(0)
                radius = {
                    key: np.linalg.norm(pos[next(a for a in key if a in contact.site_b)] - centre) for key in spans
                }
                height = min(kind.height, cutoff)
                held.append(
                    (
                        all(0 < lo < hi for lo, hi in spans.values()),
                        all((lo + hi) / 2 >= height for lo, hi in spans.values()),
                        all(hi <= np.hypot(cutoff, radius[key]) + 1e-9 for key, (_lo, hi) in spans.items()),
                    )
                )
                holds.append((True, True, True))
            elif kind.window is not None:
                held.append((set(spans.values()), set(_windows(terms, 3).values())))
                lo, hi = kind.window
                top = hi if kind.name == "HB" else min(hi, cutoff)
                holds.append(({(lo, top)}, {kind.orient} if kind.orient else set()))
        labels = {label.split(":")[0] for label in found}
        heard = silent is not None and bool(rx.contacts(_mol(silent), modes=False, kinds=(kind.name,)))
        read[kind.name] = (labels, heard, held)
        expected[kind.name] = ({kind.name}, False, holds)
    assert read == expected


def test_each_detector_type_is_read_by_exactly_one_kind():
    from xyzgraph.nci import NCI_TYPES

    assert Counter(read for kind in contacts.KINDS.values() for read in kind.reads) == Counter(NCI_TYPES)


_WATER_DIMER = [(0, 0, 0), (2.8, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0), (3.45, 0.72, 0), (3.45, -0.72, 0)]
_AMMONIUM_CHLORIDE = [(0, 0, 0), (2.9, 0, 0), (1.03, 0, 0), (-0.34, 0.97, 0), (-0.34, -0.49, 0.84)]
_AMMONIUM_CHLORIDE += [(-0.34, -0.49, -0.84)]  # H2 points at the chloride
_PD_AMMINE_WATER = "[Cl-]->[Pd+2](<-[NH3])(<-[NH3])<-[NH3].O"  # Cl0 Pd1 N2 N3 N4, then water O5


def _water_dimer():
    """Return the water dimer O0-H2...O1, its second conformer pulled 10 A apart."""
    mol = placed(_mol("O.O"), _WATER_DIMER)
    separated = Chem.Conformer(mol.GetConformer())
    separated.SetPositions(separated.GetPositions() + [(10 * (i in (1, 4, 5)), 0, 0) for i in range(6)])
    mol.AddConformer(separated, assignId=True)
    return mol


def _sulfide_ammonia():
    """Return dimethyl sulfide's S1 3.6 A from ammonia's N3, its C2 straight opposite N3 and C0 bent off, off-origin."""
    mol = Chem.MolFromSmiles("CSC.N")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetPositions(np.add([(-1.41, 0.51, 0), (0, 0, 0), (-1.5, 0, 0), (2.9, 0, 0)], (1.0, -2.0, 3.0)))
    mol.AddConformer(conformer)
    return Chem.AddHs(mol, addCoords=True)


def _benzene_iodomethane():
    """Return iodomethane's I7 4 A over a benzene centroid with its C6 behind it, an X-pi contact."""
    ring_h = _hexagon((0, 0, 0), _X, _Y, 2.48)
    return placed(
        _mol("c1ccccc1.CI"),
        [*_hexagon((0, 0, 0), _X, _Y, 1.4), (0, 0, 4), (0, 0, 3), *ring_h, (1, 0, 4), (-0.5, 0.8, 4), (-0.5, -0.8, 4)],
    )


def _propanediol():
    """Return propane-1,3-diol with O0-H5 aimed at O4 of the same molecule."""
    points = [(0, 0, 0), (-1, 1, 0), (-1, 2, 1), (2, 1, 1), (2.8, 0, 0), (0.96, 0, 0)]
    return placed(_mol("OCCCO"), [*points, *((i, 5, 0) for i in range(6)), (3.5, 0.7, 0)])


def _platinum_hydride():
    """Return ammonia 2.9 A from a platinum hydride's H1, its H3 aimed at H1."""
    mol = Chem.AddHs(rx.parse_smiles("N.[H-]->[Pt+2]", remove_hs=False))
    return placed(mol, [(0, 0, 0), (2.9, 0, 0), (4.4, 0, 0), (1.02, 0, 0), (-0.34, 0.96, 0), (-0.34, -0.48, 0.84)])


def _protected_h_contacts(fragment):
    """Return `fragment`, water and benzene placed so the fragment's H1 (deuterium, map 7) can reach both."""
    mol = Chem.AddHs(rx.parse_smiles(fragment + ".O.c1ccccc1", remove_hs=False))
    ring = mol.GetRingInfo().AtomRings()[0]
    oxygen = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    points = np.array([(-1, 0.2 * i, 1) for i in range(mol.GetNumAtoms())], float)
    points[[0, 1, oxygen]] = [(0, 0, 0), (1, 0, 0), (2.6, 0, 0)]
    for i, atom in enumerate(mol.GetAtomWithIdx(oxygen).GetNeighbors()):
        points[atom.GetIdx()] = (2.9, (-1) ** i * 0.9, 0)
    points[list(ring)] = _hexagon((0, 0, -3), _X, _Y, 1.39)
    mol = placed(mol, points)
    mol.GetAtomWithIdx(1).SetIsotope(2)
    mol.GetAtomWithIdx(1).SetAtomMapNum(7)
    return mol


def _implicit_ammonia_water():
    """Return ammonia and water with implicit H, the water O 2.8 A out along the N-H that observation places."""
    mol = Chem.MolFromSmiles("N.O")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    conformer.SetAtomPosition(1, (4, 0, 0))
    mol.AddConformer(conformer)
    h = Chem.AddHs(mol, addCoords=True).GetConformer().GetPositions()[2]
    mol.GetConformer().SetAtomPosition(1, tuple(2.8 * h / np.linalg.norm(h)))
    return mol


def _reads_unchanged(mol, **options):
    """Return the labels `rx.contacts` gives for `mol`, checking that it returns Contacts and leaves `mol` as it was."""
    before = mol.ToBinary(_ALL_PROPS)
    found = rx.contacts(mol, **options)
    assert all(isinstance(group, tuple) and all(isinstance(c, Contact) for c in group) for group in found.values())
    assert mol.ToBinary(_ALL_PROPS) == before
    return found


def test_contacts_observed_reads_a_conformer_as_xyzgraph_does():
    """Each contact names its references: an H anchor, a sigma-hole apex in brackets, an H placed on a copy by index."""
    rows = (
        ("water dimer", _water_dimer(), {"conf_id": 0}, {"HB:O0H2->O1"}),
        ("dimer apart", _water_dimer(), {"conf_id": 1}, set()),
        ("implicit H", _implicit_ammonia_water(), {}, {"HB:N0H2->O1"}),
        ("propanediol", _propanediol(), {"kinds": ("HB",)}, {"HB:O0H5->O4"}),
        ("Me2S ammonia", _sulfide_ammonia(), {"kinds": ("ChB",)}, {"ChB:S1[C0]->N3"}),
        ("T benzenes", OBSERVED["PIPIT"](), {"kinds": ("PIPIT",)}, {"PIPIT:pi(6,7,8,9,10,11)[H22]->pi(0,1,2,3,4,5)"}),
    )
    read = {name: set(_reads_unchanged(mol, observed=True, **options)) for name, mol, options, _labels in rows}
    assert read == {name: labels for name, *_, labels in rows}


def test_contact_candidates_keep_native_sites_in_kind_order():
    """A metal-bound chloride or hydride still accepts an H-bond, and a chloride bound only to Pd has no sigma hole."""
    pd_ammine = {f"HB:N{n}H{h}->{a}" for n in (2, 3, 4) for h in range(3 * n, 3 * n + 3) for a in ("Cl0", "O5")}
    thiol_hb, thiol_chb = (
        ["HB:S1H6->N2", "HB:N2H7->S1", "HB:N2H8->S1", "HB:N2H9->S1"],
        ["ChB:S1[C0]->N2", "ChB:S1[H6]->N2"],
    )
    rows = (
        ("water dimer", _water_dimer(), ("HB",), {"HB:O0H2->O1", "HB:O0H3->O1", "HB:O1H4->O0", "HB:O1H5->O0"}),
        ("Me2S ammonia", _sulfide_ammonia(), ("ChB",), {"ChB:S1[C0]->N3", "ChB:S1[C2]->N3"}),
        ("MeSH ammonia", _mol("CS.N"), ("HB", "ChB"), thiol_hb + thiol_chb),
        ("sigma first", _mol("CS.N"), ("ChB", "HB"), thiol_chb + thiol_hb),
        ("NH3 on Pt-H", _platinum_hydride(), ("HB",), {"HB:N0H3->H1", "HB:N0H4->H1", "HB:N0H5->H1"}),
        ("water on Pd ammines", _mol(_PD_AMMINE_WATER), ("HB",), pd_ammine | {"HB:O5H15->Cl0", "HB:O5H16->Cl0"}),
        ("Pd chloride", _mol(_PD_AMMINE_WATER), ("XB",), set()),
    )
    read = {name: type(labels)(_reads_unchanged(mol, modes=False, kinds=kinds)) for name, mol, kinds, labels in rows}
    assert read == {name: labels for name, *_, labels in rows}


def test_contact_modes_rank_the_inter_fragment_grips_offered():
    """Each mode holds one direction of a reciprocal pair, the stronger style on a site pair, up to two bent contacts on
    one acceptor and the most opposite sigma-hole reference; modes rank by count, strength, then acceptor charge and
    basicity. An ordinary H is allocated later (None); a molecular or bridging H keeps its index."""
    ammonium_chloride = placed(_mol("[NH4+].[Cl-]"), _AMMONIUM_CHLORIDE)
    for name, mol, kinds, modes in (
        ("water dimer, one mode by symmetry", _water_dimer(), ("HB",), [("HB:O0->O1", (None,))]),
        ("NH4Cl", ammonium_chloride, ("IONIC",), [("IONIC:N0->Cl1", (None,))]),
        ("NH4Cl prefers HB", ammonium_chloride, ("IONIC", "HB"), [("HB:N0->Cl1", (None,))]),
        ("Me2S ammonia", _sulfide_ammonia(), ("ChB",), [("ChB:S1->N3", (2,))]),
        ("MeSH ammonia", _mol("CS.N"), ("ChB", "HB"), [("HB:S1->N2", (None,)), ("HB:N2->S1", (None,))]),
        ("imine N first", _mol("CO.CC(=NC)CCC(C)=O"), None, [("HB:O1->N4", (None,)), ("HB:O1->O10", (None,))]),
        ("pyrrole-type N spent", _mol("CO.Cn1ccnc1"), None, [("HB:O1->N6", (None,))]),
        ("anionic O first", _mol("CO.C[O-].CC(C)=O"), None, [("HB:O1->O3", (None,)), ("HB:O1->O7", (None,))]),
        ("water over benzene", _mol("c1ccccc1.O"), ("HBPI",), [("HBPI:O6->pi(0,1,2,3,4,5)", (None,))]),
        (
            "three NH donors",
            _mol("C[NH3+].C[NH2+]C.C[NH+](C)C.[Cl-]"),
            ("HB",),
            [("HB:N3,N1->Cl9", (None, None)), ("HB:N3,N6->Cl9", (None, None)), ("HB:N1,N6->Cl9", (None, None))],
        ),
        (
            "acid amine",
            rx.parse_smiles("CC(=O)O.CN"),
            None,
            [("HB:N5->O2 + HB:O3->N5", (None, None)), ("HB:O3->N5", (None,)), ("HB:N5->O3", (None,))],
        ),
        ("thiourea leaves acetone's O to its N-H", _mol("NC(=S)N.CC(C)=O"), None, [("HB:N0,N3->O7", (None, None))]),
        ("either acetate O alike", _mol("CCC[NH3+].CC(=O)[O-]"), ("HB",), [("HB:N3->O7", (None,))]),
        ("H2", _protected_h_contacts("[H][H]"), ("HB",), [("HB:H0H1,H1H0->O2", (1, 0))]),
        ("bridging B-H", _protected_h_contacts("[BH3-][H]->[Fe+]"), ("HB",), [("HB:B0H1->O3", (1,))]),
    ):
        found = _reads_unchanged(mol, kinds=kinds)
        assert [(label, tuple(c.via for c in mode)) for label, mode in found.items()] == modes, name


def test_contacts_refuses_a_request_it_cannot_read():
    for options, error in (
        ({"observed": True}, "has no conformer"),
        ({"kinds": ("HX",)}, r"unknown contact kind\(s\) \['HX'\]"),
    ):
        with pytest.raises(ValueError, match=error):
            rx.contacts(_mol("O.O"), **options)


def test_observed_contacts_yield_one_tuple_per_conformer():
    """A dative bond joins a diammine palladium dichloride into one fragment, which has no inter-fragment contact."""
    diammine = placed(
        rx.parse_smiles("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"), [(2, 0, 0), (0, 0, 0), (-2.3, 0, 0), (0, 2.3, 0), (0, -2, 0)]
    )
    water = (Contact(kind="HB", site_a=(0,), site_b=(1,), via=2),)
    rows = (
        ("water dimer", _water_dimer(), [0, 1], [water, ()]),
        ("Pd diammine", diammine, [0, 0], [(), ()]),
        ("no conformer", _water_dimer(), [], []),
    )
    read = {name: list(contacts.observed_contacts(mol, ids, interfragment=True)) for name, mol, ids, _found in rows}
    assert read == {name: found for name, *_, found in rows}


def test_contact_modes_spend_each_donor_h_once_across_kinds():
    """An ammonium's H-bond and cation-pi use separate capacity; each water's ring contacts use its remaining H."""
    mol = _mol("[NH4+].O.O." + ".".join(["c1ccccc1"] * 4))

    modes = rx.contacts(mol, kinds=("HB", "CATPI", "HBPI"))

    assert modes
    for mode in modes.values():
        ammonium = [contact for contact in mode if contact.site_a == (0,)]
        assert Counter(contact.kind for contact in ammonium) == {"HB": 1, "CATPI": 1}
        assert all(contact.via is None for contact in ammonium)
        waters = [[c for c in mode if c.site_a == (donor,) and c.kind in ("HB", "HBPI")] for donor in (1, 2)]
        assert all(len(found) <= 2 for found in waters)
        assert any(len({c.site_b for c in found if c.kind == "HBPI"}) == 2 for found in waters)
        assert contacts.contact_terms(mode, mol)


def test_auto_or_chosen_modes_embed_one_member_each():
    """'auto' embeds every mode `rx.contacts` lists; a chosen subset of them gives a set of just those."""
    species = rx.isomers("[H-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-].N", geometry="SPL")[0]
    modes = rx.contacts(species)
    label, mode = next(iter(modes.items()))

    embedded = rx.embed(species, contacts="auto", n=1, seed=1)
    chosen = rx.embed(species, contacts={label: mode}, n=1, seed=1)

    assert type(chosen) is rx.EnsembleSet
    assert ([ens.tag["nci"] for ens in embedded], [ens.tag["nci"] for ens in chosen]) == (list(modes), [label])


_WATER_CONTACT = Contact(kind="HB", site_a=(0,), site_b=(1,), via=2)
_HALOGEN_PI = Contact(kind="HALPI", site_a=(7,), site_b=(0, 1, 2, 3, 4, 5), via=6)
_HALOGEN_PI_UNREFERENCED = Contact(kind="HALPI", site_a=(7,), site_b=(0, 1, 2, 3, 4, 5))


def test_parse_contacts_reads_each_accepted_form_once():
    """One Contact, or a sequence such as a binding mode, is held together, and labelled modes each apart; halogen-pi
    may drop its reference."""
    rows = (
        (None, None),
        ("auto", "auto"),
        ("input", "input"),
        ({"HB:O0H2->O1": [_WATER_CONTACT]}, {"HB:O0H2->O1": (_WATER_CONTACT,)}),
        (_WATER_CONTACT, (_WATER_CONTACT,)),
        ([_WATER_CONTACT], (_WATER_CONTACT,)),
        ((_HALOGEN_PI, _HALOGEN_PI_UNREFERENCED), (_HALOGEN_PI, _HALOGEN_PI_UNREFERENCED)),
    )
    assert [contacts.parse_contacts(given) for given, _parsed in rows] == [parsed for _given, parsed in rows]


def test_parse_contacts_refuses_a_form_naming_what_it_takes():
    for given, error, message in (
        ({(0, 1): (2.5, 3.0)}, TypeError, "goes to constrain="),
        ([1, 2], TypeError, "rx.contacts' modes"),
        ("Auto", TypeError, "rx.contacts' modes"),
        (Contact(kind="hbond", site_a=(0,), site_b=(1,), via=2), ValueError, "cannot hold"),  # a detector type
        (Contact(kind="IONIC", site_a=(0,), site_b=(1,), via=2), ValueError, "cannot hold"),  # an ion has no H
        (Contact(kind="PIPI", site_a=(0, 1, 2, 3, 4, 5), site_b=(6, 7, 8, 9, 10, 11)), ValueError, "cannot hold"),
    ):
        with pytest.raises(error, match=message):
            contacts.parse_contacts(given)


def test_contact_modes_give_the_modes_each_request_form_holds():
    """'input' holds the inter-fragment contacts the first conformer makes, every H the detector names; given modes
    keep their labels, given Contacts are one mode named by its contacts, and no modes or 'auto' with no inter-fragment
    contact none."""
    held = {(1, 2): (1.6, 2.2), (0, 2, 1): (140.0, 180.0)}
    bifurcated = placed(
        _mol("O.O"), [(0, 0, 0), (2.8, 0, 0), (0.8, 0.4, 0), (0.8, -0.4, 0), (3.7, 0.4, 0), (3.7, -0.4, 0)]
    )
    rows = (
        ("dimer input", _water_dimer(), "input", {"HB:O0H2->O1": held}),
        (
            "bifurcated input",
            bifurcated,
            "input",
            {"HB:O0H2,O0H3->O1": held | {(1, 3): (1.6, 2.2), (0, 3, 1): (140.0, 180.0)}},
        ),
        ("propanediol input", _propanediol(), "input", {None: None}),
        ("modes", _water_dimer(), {"HB:O0H2->O1": (_WATER_CONTACT,)}, {"HB:O0H2->O1": held}),
        ("no modes", _water_dimer(), {}, {None: None}),
        ("contacts", _water_dimer(), (_WATER_CONTACT,), {"HB:O0H2->O1": held}),
        ("methanes auto", _mol("C.C"), "auto", {None: None}),
    )
    read = {name: contacts.contact_modes(mol, request) for name, mol, request, _modes in rows}
    assert read == {name: modes for name, *_, modes in rows}


def test_contacts_input_refuses_a_structure_with_no_geometry():
    with pytest.raises(ValueError, match="contacts='input' holds the contacts an input geometry makes"):
        contacts.contact_modes(_mol("O.O"), "input")


def _deuterated_water_on_ammonias():
    mol = _mol("O.N.N.N")
    mol.GetAtomWithIdx(4).SetIsotope(2)
    mol.GetAtomWithIdx(4).SetAtomMapNum(7)
    return mol


def test_contact_terms_hold_each_contact_spending_one_h_each():
    """An explicit H is reserved before a deferred request takes its donor's first free ordinary H; an explicit H may
    bifurcate, and a sigma hole keeps its chosen reference with no geometry."""
    deferred = Contact(kind="HB", site_a=(0,), site_b=(1,))
    over = np.hypot(3.31, 1.4)  # I held at xyzgraph's 3.31 A cutoff over the centroid, below the row's 3.5 A
    halogen_pi = {(i, 7): (pytest.approx(over - 0.3, abs=0.01), pytest.approx(over, abs=0.01)) for i in range(6)}
    for name, mol, held, terms in (
        (
            "NH4Cl ion pair",
            placed(_mol("[NH4+].[Cl-]"), _AMMONIUM_CHLORIDE),
            (Contact(kind="IONIC", site_a=(0,), site_b=(1,)),),
            {(0, 1): (2.6, pytest.approx(3.40, abs=0.01))},  # xyzgraph never reports this pair past 3.40 A
        ),
        (
            "Me2S pyridine",
            _mol("CSC.n1ccccc1"),
            (Contact(kind="ChB", site_a=(1,), site_b=(3,), via=2),),
            {(1, 3): (3.0, pytest.approx(3.48, abs=0.01)), (2, 1, 3): (155.0, 180.0)},  # S...N cut at 3.48 A
        ),
        (
            "D2O on ammonias",
            _deuterated_water_on_ammonias(),
            (deferred, *(Contact(kind="HB", site_a=(0,), site_b=(a,), via=4) for a in (2, 3)), deferred),
            {(1, 5): (1.6, 2.2), (0, 5, 1): (140.0, 180.0)}
            | {(a, 4): (1.6, 2.2) for a in (2, 3)}
            | {(0, 4, a): (140.0, 180.0) for a in (2, 3)},
        ),
        (
            "H2 named H",
            _protected_h_contacts("[H][H]"),
            (Contact(kind="HB", site_a=(0,), site_b=(2,), via=1),),
            {(1, 2): (1.6, 2.2), (0, 1, 2): (140.0, 180.0)},
        ),
        (
            "borane terminal H",
            _protected_h_contacts("[BH3-][H]->[Fe+]"),
            (Contact(kind="HB", site_a=(0,), site_b=(3,)),),
            {(3, 10): (1.6, 2.2), (0, 10, 3): (140.0, 180.0)},
        ),
        ("MeI over benzene", _benzene_iodomethane(), (_HALOGEN_PI, _HALOGEN_PI_UNREFERENCED), halogen_pi),
    ):
        before = mol.ToBinary(_ALL_PROPS)
        assert contacts.contact_terms(held, mol) == terms, name
        assert mol.ToBinary(_ALL_PROPS) == before, name


def test_contact_terms_refuse_a_donor_with_no_free_ordinary_h():
    reserved = tuple(Contact(kind="HB", site_a=(0,), site_b=(3,), via=h) for h in (10, 11, 12))
    for mol, held, message in (
        (
            _mol("CO.N.N"),
            tuple(Contact(kind="HB", site_a=(1,), site_b=(a,)) for a in (2, 3)),
            r"no free ordinary H on O1.*fewer contacts",
        ),
        (_protected_h_contacts("[H][H]"), (Contact(kind="HB", site_a=(0,), site_b=(2,)),), "explicit via H"),
        (
            _protected_h_contacts("[BH3-][H]->[Fe+]"),
            (*reserved, Contact(kind="HB", site_a=(0,), site_b=(3,))),
            r"no free ordinary H on B0.*explicit via H",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            contacts.contact_terms(held, mol)


def test_cation_pi_windows_end_at_the_detector_distance():
    """An ammonium over a pyridine stretched off its regular hexagon."""
    mol = Chem.MolFromSmiles("n1ccccc1.[NH4+]")
    ring = np.array(_hexagon((0, 0, 0), _X, _Y, 1.39))
    ring[0] *= 1.08
    centroid = ring.mean(0)
    mol = placed(mol, [*ring, np.add(centroid, (0.2, 0.1, 3.0))])

    terms = contacts.contact_terms((Contact(kind="CATPI", site_a=(6,), site_b=tuple(range(6))),), mol)

    upper_sq = np.array([terms[(i, 6)][1] ** 2 for i in range(6)])
    # sum |x - r_i|^2 = n |x - centroid|^2 + sum |r_i - centroid|^2, so this height meets the mean upper bound
    height = float(np.sqrt(np.mean(upper_sq - np.sum((ring - centroid) ** 2, axis=1))))
    conformer = mol.GetConformer()
    conformer.SetAtomPosition(6, tuple(np.add(centroid, (0, 0, height - 1e-8))))
    ((detected,),) = rx.contacts(mol, observed=True, kinds=("CATPI",)).values()
    assert (detected.kind, detected.site_a, set(detected.site_b)) == ("CATPI", (6,), set(range(6)))
    assert all(
        lo < hi and np.linalg.norm(conformer.GetPositions()[6] - ring[i]) <= hi for (i, _), (lo, hi) in terms.items()
    )
    conformer.SetAtomPosition(6, tuple(np.add(centroid, (0, 0, height + 1e-4))))
    assert not rx.contacts(mol, observed=True, kinds=("CATPI",))
