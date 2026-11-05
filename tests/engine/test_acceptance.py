"""`engine/acceptance.py`: the core gates, the reason each names for a refused conformer, and the remedy it offers."""

import logging
import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import rxembed as rx
from rxembed.engine.acceptance import (
    CORE_GATES,
    EmbeddingError,
    Gate,
    conformer_misses,
    coordination_state_failure,
    fix_failure,
    remedy,
    shape_report,
)
from rxembed.engine.conformers import Conformers
from rxembed.engine.seeding import prepare
from rxembed.geometry.constraints import Constraints, CoplanarRow, SoftKeys, resolve_core
from rxembed.identity.reading import Specification, read_species
from rxembed.identity.smiles_io import mol_from_species
from rxembed.identity.species import Species
from rxembed.metal.centres import Centre, Site, engine_graph
from rxembed.metal.polyhedra import POLYHEDRA
from rxembed.stereo import Miss
from tests.geometry_fixtures import embedded, mirrored, placed
from tests.metal_fixtures import on_its_vertices, seated, seeds, zinc_copies

PLATINUM = "[Pt](F)(Cl)(Br)I"  # Pt0 F1 Cl2 Br3 I4
DIAMMINE_PD = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"  # N0 Pd1 Cl2 Cl3 N4
NICKEL = "[Cl-]->[Ni+2](<-[Cl-])(<-[Cl-])<-[Cl-]"  # Cl0 Ni1 Cl2 Cl3 Cl4
CO_ALL_CIS = "N->[Co+3](<-N)(<-O)(<-O)(<-[Cl-])<-[Cl-]"  # N0 Co1 N2 O3 O4 Cl5 Cl6
GATES = {gate.name: gate for gate in CORE_GATES}


class OneGate(Conformers):
    """Hold conformers judged only by the gate rows an instance is given."""

    gates: tuple[Gate, ...] = ()


def _folded(mol, degrees):
    """Return square-planar species Mol `mol` on its vertices, each donor lifted `degrees` out of the plane."""
    dirs = np.asarray(POLYHEDRA["square_planar"].vertex_dirs, float)
    lift = np.radians(degrees)
    return on_its_vertices(mol, np.cos(lift) * dirs + np.sin(lift) * np.array([0.0, 0.0, 1.0]))


@pytest.fixture(scope="module")
def refused():
    """Return, by gate name, conformer 0 of a minimal structure that gate refuses, its kind, atoms and site label."""
    straight = placed("CCCC", [(0, 0, 0), (1.5, 0, 0), (3, 0, 0), (3, 1.5, 0)])
    torn = placed("CCCC", [(0, 0, 0), (10, 0, 0), (11, 1, 0), (12.5, 1, 0)])
    platinum = rx.isomers(PLATINUM, geometry="TET")[0]
    diammine = rx.isomers(DIAMMINE_PD, geometry="SPL").select(label="cis")
    palladium = read_species(diammine)
    stretched = on_its_vertices(diammine)
    pos = stretched.GetConformer().GetPositions()
    pos[2] *= 3.5 / np.linalg.norm(pos[2])  # Pd1 at the origin, Cl2 out to 3.5 A
    stretched.GetConformer().SetPositions(pos)
    nickel = rx.isomers(NICKEL, geometry="SPL")[0]
    square = read_species(nickel)
    butanol = embedded("C[C@@H](O)CC")
    helium = placed("[He].[He].[He].[He].[He]", [(0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, -1, 0)])
    planes = Constraints(  # a flat torsion held, then He2 out of a plane its three collinear neighbours cannot span
        coplanar=[
            CoplanarRow(kind="donor_torsion", atoms=(0, 1, 2, 3), cap=10.0),
            CoplanarRow(kind="donor_plane", atoms=(2, 0, 1, 4), cap=10.0, anchor=0.0),
        ]
    )
    vinyl_fluoride = rx.isomers("[Cl-]->[Pt+2]1(<-[Cl-])(<-[Cl-])<-[CH2]=[CH]->1F", geometry="SPL")[0]
    rows = {
        "fixed": (straight, resolve_core(straight, fix={(0, 1, 2, 3): 60.0}).cons, Species(mol=straight)),
        "bonding": (torn, Constraints(), Species(mol=torn)),
        "coordination_state": (
            on_its_vertices(platinum, POLYHEDRA["seesaw"].vertex_dirs),
            Constraints(),
            read_species(platinum),
        ),
        "metal_hand": (zinc_copies([-1]), Constraints(), read_species(zinc_copies([1]))),
        "ligand_stereo": (mirrored(butanol), Constraints(), read_species(butanol)),
        "ml_windows": (stretched, prepare(palladium, resolve_core(palladium.mol)).cons, palladium),
        "coplanarity": (helium, planes, Species(mol=helium)),
        "umbrella": (_folded(nickel, 14.0), prepare(square, resolve_core(square.mol)).cons, square),
        "metal_face": (mirrored(on_its_vertices(vinyl_fluoride)), Constraints(), read_species(vinyl_fluoride)),
    }
    expected = {
        "fixed": ("numeric_fix", (), "C0-C1-C2-C3"),
        "bonding": ("bonding", (0, 1), "C0-C1"),
        "coordination_state": ("coordination_shape", (0,), "Pt0"),
        "metal_hand": ("metal_hand", (1,), "Zn1"),
        "ligand_stereo": ("ligand_stereo", (1,), "C1"),
        "ml_windows": ("ml_distance", (1, 2), "Pd1-Cl2"),
        "coplanarity": ("structural_constraint", (2, 0, 1, 4), "He2-He0-He1-He4"),
        "umbrella": ("coordination_shape", (0, 1, 2, 3), "Cl0-Ni1-Cl2-Cl3"),
        "metal_face": ("metal_face", (1,), "Pt1"),
    }
    return {
        name: (Conformers(mol, [0], cons, species=species), expected[name])
        for name, (mol, cons, species) in rows.items()
    }


def test_core_gates_name_their_refusals_and_act_on_flags(refused, caplog):
    """Each gate reads its failure at its site; first in a relax, a ladder row restores the seed and a required kind
    raises with its headline, any other kind dropping the conformer with a warning.
    """
    read, expected = {}, {}
    for gate in CORE_GATES:
        conformers, (kind, atoms, site) = refused[gate.name]
        failure = gate.check(conformers, 0)
        assert failure is not None, gate.name
        chloroethane = embedded("CCCl")
        held = OneGate(
            chloroethane, [0], resolve_core(chloroethane, fix={(1, 2): 2.4}).cons, species=Species(mol=chloroethane)
        )
        held.gates = (Gate(gate.name, lambda _conformers, _cid, failure=failure: failure, gate.ladder, gate.required),)
        headline = re.escape(f"{failure.detail} in 1/1 rejected seeds; try ")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="rxembed"):
            try:
                held.minimize(max_iters=0)
                outcome = "warned" if re.search(headline, caplog.text) else caplog.text
            except EmbeddingError as error:
                outcome = "raised" if re.search(headline, str(error)) else str(error)
        read[gate.name] = (failure.kind, failure.atoms, site in failure.detail, outcome)
        read[gate.name] += (held.ids, held.unrelaxed, 0 in held.relax_failures)
        expected[gate.name] = (kind, atoms, True, "raised" if kind in gate.required else "warned", [], [], gate.ladder)

    assert read == expected


def test_remedy_offers_only_what_the_request_can_change():
    """Another isomer needs centres, a looser fix a fix and a looser constrain a contact; none left is a new seed."""
    zinc, ethane = read_species(zinc_copies([1])), Species(mol=Chem.MolFromSmiles("CC"))
    contacts = Constraints(contacts=SoftKeys(distances={(0, 1)}))
    rows = (
        ("coordination_shape", zinc, Constraints(), "try geometry= or another isomer"),
        ("bonding", ethane, Constraints(frozen={0}), "try a looser fix="),
        ("bonding", ethane, Constraints(), "try another seed="),
        ("seeding", zinc, contacts, "try a looser constrain= or another isomer"),
        ("an_unlisted_kind", zinc, Constraints(), "try another isomer"),
        ("uff", ethane, Constraints(), "try a different relaxation backend"),
    )
    assert [remedy(kind, species, cons) for kind, species, cons, _offered in rows] == [row[3] for row in rows]


def _two_centres(primary, secondary):
    """Build two ideal, separated coordination spheres on one conformer, and their species."""
    rw = Chem.RWMol()
    states = []
    for geometry in (primary, secondary):
        metal = rw.AddAtom(Chem.Atom(46))
        rw.GetAtomWithIdx(metal).SetFormalCharge(2)
        donors = tuple(rw.AddAtom(Chem.Atom(7)) for _ in POLYHEDRA[geometry].vertex_dirs)
        for donor in donors:
            rw.AddBond(donor, metal, Chem.BondType.DATIVE)
        states.append(Centre(metal, 46, 2, geometry, tuple(Site(atoms=(donor,)) for donor in donors)))
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for origin, state in zip((np.zeros(3), np.array([8.0, 0.0, 0.0])), states, strict=True):
        conf.SetAtomPosition(state.atom, tuple(origin))
        for site, direction in zip(state.vertices, POLYHEDRA[state.geometry].vertex_dirs, strict=True):
            unit = np.asarray(direction, dtype=float)
            conf.SetAtomPosition(site.atoms[0], tuple(origin + 2.0 * unit / np.linalg.norm(unit)))
    mol.AddConformer(conf)
    return engine_graph(mol), Species(mol=mol, centres=tuple(states))


def test_coordination_state_failure_names_the_first_misfit():
    """The first centre off its polyhedron or its seating is named. A calculator's geometry needs only the species, so
    a distorted linear copper passes it and fails a build."""
    platinum = read_species(rx.isomers(PLATINUM, geometry="TET")[0])
    copper = rx.isomers("[NH3]->[Cu+]<-[NH3]")[0]  # N0 Cu1 N2
    nickel = seated(rx.parse_smiles(NICKEL), "square_pyramidal", {1: 0, 2: 2, 3: 3, 4: 4})
    nickel_mol = mol_from_species(nickel)
    nickel_mol.RemoveAllConformers()
    apex, base = POLYHEDRA["square_pyramidal"].vertex_dirs[0], POLYHEDRA["square_pyramidal"].vertex_dirs[1:]
    undefined = on_its_vertices(mol_from_species(platinum))
    undefined.GetConformer().SetAtomPosition(1, (float("nan"), 0.0, 0.0))
    two_mol, two = _two_centres("tetrahedral", "square_planar")
    lifted = two.centres[1].vertices[0].atoms[0]
    two_mol.GetConformer().SetAtomPosition(lifted, tuple(two_mol.GetConformer().GetPositions()[lifted] + [0, 0, 2.0]))
    cobalt = rx.isomers(CO_ALL_CIS, geometry="OCT").select(hand="delta")  # N0 O3 N2 Cl5 O4 Cl6 on slots 0-5
    swapped = np.asarray(POLYHEDRA["octahedral"].vertex_dirs, float)[[3, 1, 2, 0, 4, 5]]  # N0 and Cl5 trade slots
    seesaw = on_its_vertices(mol_from_species(platinum), POLYHEDRA["seesaw"].vertex_dirs)
    bent = on_its_vertices(copper, [(1, 0, 0), (0, 1, 0)])
    flat = on_its_vertices(nickel_mol, [apex, *POLYHEDRA["square_planar"].vertex_dirs])
    for name, mol, species, require_fit, reason in (
        ("Pt seesaw", seesaw, platinum, True, r"Pt0 relaxes from TET \(0\.433\) to SEE \(0\.000\)"),
        ("Pt seesaw calc", seesaw, platinum, False, r"Pt0 relaxes from TET .* to SEE .*"),
        ("Cu bent", bent, read_species(copper), True, r"Cu1 distorts far from LIN \(0\.77\)"),
        ("Cu bent calc", bent, read_species(copper), False, None),
        ("Pt undefined donor", undefined, platinum, True, "Pt0 has an undefined donor direction"),
        ("Ni on SPY base", on_its_vertices(nickel_mol, [apex, *base]), nickel, True, None),
        ("Ni flat", flat, nickel, True, r"Ni1 relaxes from SPY .* to SPL .*"),
        ("Pd5 lifted", two_mol, two, True, r"Pd5 relaxes from SPL .*"),
        (
            "Co swapped",
            on_its_vertices(cobalt, swapped),
            read_species(cobalt),
            True,
            r"Co1 donors swap sites \(another isomer\)",
        ),
    ):
        failure = coordination_state_failure(mol, 0, Specification(species=species), require_fit=require_fit)
        assert (failure is None) if reason is None else re.fullmatch(reason, str(failure)), (name, failure)


def test_shape_report_reads_each_centre_beside_its_input():
    platinum = rx.isomers(PLATINUM, geometry="TET")[0]
    two_mol, two = _two_centres("tetrahedral", "square_planar")
    ethanol = Chem.MolFromSmiles("CCO")
    rows = {"ethanol": (ethanol, Species(mol=ethanol)), "Pt": (on_its_vertices(platinum), read_species(platinum))}
    rows["Pd2"] = (two_mol, two)

    assert {name: shape_report(mol, 0, Specification(species=species)) for name, (mol, species) in rows.items()} == {
        "ethanol": None,
        "Pt": "Pt0 TET 0.000 (next SEE 0.433)",
        "Pd2": "Pd0 TET 0.000 (next SEE 0.433); input TET 0.000 (next SEE 0.433) | "
        "Pd5 SPL 0.000 (next SEE 0.366); input SPL 0.000 (next SEE 0.366)",
    }


def test_fix_failure_names_the_numeric_fix_missed_furthest():
    """A dihedral with no defined value misses by more than any measured one."""
    bent = [(0, 0, 0), (1.5, 0, 0), (2.5, 1.0, 0), (4.0, 1.0, 0)]
    straight = [(0, 0, 0), (1.5, 0, 0), (3, 0, 0), (3, 1.5, 0)]
    rows = (
        ("bond held", bent, {(0, 1): 1.5}, None),
        ("angle in range", bent, {(0, 1, 2): (120.0, 140.0)}, None),
        ("span missed", bent, {(1, 2): 2.4}, "C1-C2"),
        ("further miss", bent, {(0, 1): 1.6, (1, 2): 2.4}, "C1-C2"),
        ("straight dihedral", straight, {(0, 1): 1.0, (0, 1, 2, 3): 60.0}, "C0-C1-C2-C3"),
    )
    missed = {}
    for name, points, fix, _named in rows:
        mol = placed("CCCC", points)
        failure = fix_failure(Conformers(mol, [0], resolve_core(mol, fix=fix).cons, species=Species(mol=mol)), 0)
        missed[name] = None if failure is None else str(failure).partition(" not held")[0]
    assert missed == {name: named and f"fix= {named}" for name, *_row, named in rows}


def test_ligand_stereo_gate_names_each_element_read_otherwise():
    """A centre it no longer reads, an axis or a double bond turned, and a coordination-locked donor turned."""
    pushed = embedded("C[C@@H](O)CC")
    pos = pushed.GetConformer().GetPositions()
    pos[1] += 3.0 * (pos[1] - pos[0])
    butanol = read_species(Chem.Mol(pushed))
    pushed.GetConformer().SetPositions(pos)
    biaryl = embedded("CC1=CC=CC(I)=C1N1C(C)=CC=C1Br |wU:7.7|")
    butene = embedded("C/C=C/C")
    cis = Chem.Mol(butene)
    rdMolTransforms.SetDihedralDeg(cis.GetConformer(), 0, 1, 2, 3, 0.0)
    amine = seeds(
        mol_from_species(seated(Chem.AddHs(rx.parse_smiles("[Pd](Cl)(Cl)(Cl)([N@H](C)O)")), "SPL", [1, 2, 3, 4])),
        1,
        seed=2,
    )
    rows = {
        "flat butanol": (pushed, butanol),
        "mirrored biaryl": (mirrored(biaryl), read_species(biaryl)),
        "cis butene": (cis, read_species(butene)),
        "mirrored Pd amine": (mirrored(amine.mol), amine.species),
    }
    failures = {
        name: GATES["ligand_stereo"].check(Conformers(mol, [0], species=species), 0)
        for name, (mol, species) in rows.items()
    }

    assert {name: failure and (failure.kind, failure.atoms) for name, failure in failures.items()} == {
        "flat butanol": ("ligand_stereo_unassigned", (1,)),
        "mirrored biaryl": ("ligand_stereo", (7, 8)),
        "cis butene": ("ligand_stereo", (1, 2)),
        "mirrored Pd amine": ("locked_stereo", (4,)),
    }


def _at_origin(species):
    """Return the engine graph of `species` with one conformer that puts every atom at the origin."""
    mol = engine_graph(species.mol)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    return mol


def test_collapsed_sphere_misses_only_a_specified_hand():
    """Every atom at the origin: an achiral cis-PdCl2(NH3)2 misses nothing, a chiral Co reads no hand."""
    achiral = rx.isomers(DIAMMINE_PD, geometry="SPL").select(label="cis")
    chiral = read_species(rx.isomers("O->[Co+3](<-[Cl-])(<-[CH3-])(<-N)(<-[F-])<-P", geometry="OCT")[0])
    (cobalt,) = chiral.centres
    collapsed = Miss(kind="hand", atoms=(cobalt.atom,), specified=cobalt.hand, observed="")
    species = (read_species(achiral), chiral)
    assert [conformer_misses(Specification(species=one), _at_origin(one), 0) for one in species] == [(), (collapsed,)]
