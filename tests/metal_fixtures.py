"""Build chemistry fixtures shared by the engine tests."""

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import rxembed as rx
from rxembed.engine.conformers import Conformers
from rxembed.engine.seeding import prepare, seed_conformers, seeded
from rxembed.geometry.bounds import EmbedParams, resolve_sampling
from rxembed.geometry.constraints import resolve_core
from rxembed.graph import metal_indices
from rxembed.identity.arrangements import enumerate_isomers
from rxembed.identity.reading import IsomerRequest, observed_species, read_species, requested_centre, requested_geometry
from rxembed.identity.smiles_io import mol_from_species
from rxembed.identity.species import Species, centre_from_slots, observe_stereo
from rxembed.identity.stereoisomers import StereoRequest
from rxembed.metal.centres import canonical_metal_graph, chemical_graph
from rxembed.metal.polyhedra import POLYHEDRA, VACANT, resolve_geometry
from rxembed.metal.sphere import LigandFacts
from tests.geometry_fixtures import placed

# Each amine N carries one acid and one carboxylate arm, and only one carboxylate binds Pt(II). Cutting the metal
# leaves a C2-symmetric ligand; the complex has no such symmetry, so its four bound-N hand pairs are distinct.
ONE_ARM_BOUND_PT = "OC(=O)CN1(CC(=O)[O-]->[Pt+2]12<-[Cl-])CCN->2(CC(=O)O)CC(=O)[O-]"
# (Buta-1,3-diene)Fe(CO)3 and its isoprene analogue: an eta4 diene whose central bond rotates when free, so each
# bound s-cis or s-trans form is an isomer. Isoprene's methyl makes its two s-cis faces distinct.
BUTADIENE_FE_CO3 = "[O+]#[C-]->[Fe]123(<-[C-]#[O+])(<-[C-]#[O+])<-[CH2]=[CH]->1[CH]->2=[CH2]->3"
ISOPRENE_FE_CO3 = "[O+]#[C-]->[Fe]123(<-[C-]#[O+])(<-[C-]#[O+])<-[CH2]=[C](C)->1[CH]->2=[CH2]->3"
# Planar chloride-bridged dimers: two graph-equivalent metals, each seating two terminal donors and both bridges.
PD2CL6 = "[Cl-]->[Pd+2]1(<-[Cl-])<-[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]->1"
PT2CL4_AMMINES = "[Cl-]->[Pt+2]1(<-[NH3])<-[Cl-]->[Pt+2](<-[Cl-])(<-[NH3])<-[Cl-]->1"
# Fe(II) ferrocenes with one or two 1-fluoro-2-bromo Cp faces, each face planar-chiral
FERROCENE_ONE_FACE = "F[c]12->[Fe+2]3456789(<-[cH]%10[cH]->3[cH]->4[cH-]->5[cH]->6%10)<-[cH]([cH]->71)[cH]->8[c-]->92Br"
FERROCENE_TWO_FACES = (
    "F[c]12->[Fe+2]3456789(<-[cH]([cH]->31)[cH]->4[c-]->52Br)<-[cH]1[cH]->6[c]->7(F)[c-]->8(Br)[cH]->91"
)
# (BINAP)PdCl2: the P,P chelate locks the binaphthyl axis, unspecified here.
BINAP_PDCL2 = (
    "[Pd+2]%90(<-[Cl-])(<-[Cl-])(<-P(c1ccccc1)(c2ccccc2)c3ccc4ccccc4c3-c3c(P(c4ccccc4)(c5ccccc5)->%90)ccc4ccccc34)"
)


def enumerated(mol, geometry=None, *, stereo=None, center=None, screen=True, observed_only=False):
    """Return the species `enumerate_isomers` lists for `mol` under these choices, read as `rx.isomers` reads them."""
    mol = canonical_metal_graph(mol)
    request = IsomerRequest(
        stereo=StereoRequest.parse(stereo).on(mol),
        center=requested_centre(mol, center),
        pins=resolve_core(mol),
        geometry=requested_geometry(geometry),
        observed=observed_only,
        screen=screen,
    )
    return enumerate_isomers(mol, request)


def bound_donors(mol):
    """Return the atoms of a species Mol that bond a metal, sorted."""
    return sorted({n.GetIdx() for m in metal_indices(mol) for n in mol.GetAtomWithIdx(m).GetNeighbors()})


def seated(mol, geometry, slots):
    """Return the species of one-metal `mol` whose donors sit at the given polyhedron slots (`centre_from_slots`).

    `geometry` is a name or code, and `slots` a ``{slot: atom}`` dict or a slot-ordered list.
    """
    if not isinstance(slots, dict):
        slots = {slot: atom for slot, atom in enumerate(slots) if atom is not VACANT}
    facts = LigandFacts(chemical_graph(canonical_metal_graph(mol)))
    metal = min(metal_indices(facts.graph))
    centre = centre_from_slots(facts, metal, resolve_geometry(geometry), slots)
    return Species(mol=facts.graph, centres=(centre,))


def conformer_species(mol):
    """Return the species the first conformer of `mol` shows at every metal (`reading.observed_species`)."""
    graph = canonical_metal_graph(mol)
    return observed_species(observe_stereo(graph, graph.GetConformer()))


def orientation_parity_of(mol, cid, iso):
    """Fit every observed vertex to the ideal shape and return proper (+1) or mirrored (-1)."""
    pos = mol.GetConformer(int(cid)).GetPositions()
    ((centre,), (sites,)) = iso.centres, iso.sites.values()

    def point(atom):
        ring = sites.haptic.get(atom)
        return np.mean(pos[list(ring)], axis=0) if ring else pos[atom]

    observed = np.asarray([point(atom) - pos[centre.atom] for atom in sites.vertices])
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    ideal = np.asarray(POLYHEDRA[centre.geometry].vertex_dirs, float)
    ideal /= np.linalg.norm(ideal, axis=1, keepdims=True)
    u, _s, vt = np.linalg.svd(observed.T @ ideal)
    return 1 if np.linalg.det(u @ vt) >= 0 else -1


def on_its_vertices(species, rays=None):
    """Return one-metal species Mol `species` with its metal at the origin and each donor 2.1 A out on its vertex.

    `rays` replaces the polyhedron's vertex directions; a haptic face straddles its vertex, and every other atom is
    parked on the x axis far from the sphere.
    """
    read = read_species(species)
    ((metal, sites),) = read.sites.items()
    dirs = np.asarray(POLYHEDRA[read.centres[0].geometry].vertex_dirs if rays is None else rays, float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = {atom: (100.0 + atom, 0.0, 0.0) for atom in range(species.GetNumAtoms())} | {metal: (0.0, 0.0, 0.0)}
    for vertex, site in enumerate(sites.vertices):
        if site is VACANT:
            continue
        face = sites.haptic.get(site, (site,))
        side = np.cross(dirs[vertex], [0.3, 0.5, 0.8])
        side = 0.7 * side / np.linalg.norm(side) if len(face) > 1 else 0 * side
        for atom, sign in zip(face, (1, -1, 0), strict=False):
            points[atom] = tuple(2.1 * dirs[vertex] + sign * side)
    return placed(Chem.Mol(species), [points[atom] for atom in range(species.GetNumAtoms())])


def specified_mol(species):
    """Return the species Mol of internal record `species` without its conformers: what the record specifies."""
    mol = mol_from_species(species)
    mol.RemoveAllConformers()
    return mol


def record_seeds(species, n, sampling):
    """Return ``(mol, ids, cons)``: DG seeds of an internal `Species` record that no species Mol specifies.

    A free haptic winding or an implicit-H haptic record has no public route, so the core stages run directly.
    """
    prepared = prepare(species, resolve_core(species.mol), params=sampling.params)
    seeds = seed_conformers(prepared.mol, prepared.cons, species, n, sampling, graft_ref=prepared.graft)
    return seeds.mol, seeds.ids, prepared.cons


def seeds(mol, n, *, fix=None, constrain=None, seed=None, threads=None, **params):
    """Return `n` raw DG seeds of explicit-H species Mol `mol` as `Conformers`, before any relax or gate reads them.

    These are the stages `rx.embed` runs before it relaxes; `params` are `EmbedParams` fields.
    """
    pins = resolve_core(mol, fix=fix, constrain=constrain)
    species, sampling = read_species(mol, pins=pins), resolve_sampling(EmbedParams(**params), seed, threads)
    return seeded(species, prepare(species, pins, params=sampling.params), n, sampling, Conformers)


def ferrocene():
    """Build two Cp anions datively bound to Fe(II) with an eta5 sandwich geometry."""
    rw = Chem.RWMol()
    iron = rw.AddAtom(Chem.Atom(26))
    rw.GetAtomWithIdx(iron).SetFormalCharge(2)
    rings = []
    for _ in range(2):
        ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
        for k, bond in enumerate([Chem.BondType.SINGLE, Chem.BondType.DOUBLE] * 2 + [Chem.BondType.SINGLE]):
            rw.AddBond(ring[k], ring[(k + 1) % 5], bond)
        rw.GetAtomWithIdx(ring[0]).SetFormalCharge(-1)
        for carbon in ring:
            rw.AddBond(iron, carbon, Chem.BondType.DATIVE)
        rings.append(ring)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(iron, Point3D(0, 0, 0))
    for face, ring in enumerate(rings):
        z = 1.66 if face == 0 else -1.66
        for k, carbon in enumerate(ring):
            angle = 2 * np.pi * k / 5 + (0.2 if face else 0.0)
            conf.SetAtomPosition(carbon, Point3D(1.21 * np.cos(angle), 1.21 * np.sin(angle), z))
    mol.AddConformer(conf)
    return mol


def one_arm_bound_pt():
    """Embed `ONE_ARM_BOUND_PT` once: a geometry input whose two bound amine N are coordination-locked."""
    return rx.embed(rx.isomers(ONE_ARM_BOUND_PT, geometry="SPL")[0], n=1, seed=7).mol


def pd2cl6():
    """Return [Pd2Cl6]2-: two PdCl4 squares (Pd1, Pd4) sharing chlorides 3 and 7, planar as in the crystal."""
    mol = rx.parse_smiles("[Cl-]->[Pd+2]1(<-[Cl-])<-[Cl-]->[Pd+2](<-[Cl-])(<-[Cl-])<-[Cl-]->1")
    xy = ((-3.35, 1.6), (-1.75, 0.0), (-3.35, -1.6), (0.0, 1.65), (1.75, 0.0), (3.35, 1.6), (3.35, -1.6), (0.0, -1.65))
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetPositions(np.array([(x, y, 0.0) for x, y in xy]))  # Pd...Pd 3.5 A
    mol.AddConformer(conf)
    return mol


def ideal_sphere(dirs, r):
    """A bare Mol whose atom 0 is a metal and 1..N its vertices at radius `r` along `dirs`."""
    rw = Chem.RWMol()
    for _ in range(len(dirs) + 1):
        rw.AddAtom(Chem.Atom(6))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    for i, d in enumerate(dirs):
        u = np.array(d, float)
        conf.SetAtomPosition(i + 1, Point3D(*(u / np.linalg.norm(u) * r)))
    mol.AddConformer(conf, assignId=True)
    return mol


# A six-site sphere no polyhedron fits, and a CN5 one almost equidistant between square pyramid and bipyramid.
SQUASHED_SIX = [(np.cos(t) * 0.5, np.sin(t) * 0.5, 0.87) for t in np.radians([0, 60, 120, 180, 240, 300])]
SPY_TBP_TIE = 0.662 * np.array(POLYHEDRA["square_pyramidal"].vertex_dirs, float) + 0.338 * np.array(
    POLYHEDRA["trigonal_bipyramidal"].vertex_dirs, float
)


def berry_intermediate(t):
    """Return 5 unit vectors on the SPY->TBP Berry pseudorotation path at parameter `t` (0 = SPY, 1 = TBP).

    Spherical-linear interpolation, slot by slot, between the two idealized templates' own `vertex_dirs`.
    Not the literature pivot/turnstile pairing (this codebase's own fit is seating-invariant, so any
    continuous path between the two templates crosses their residual tie the same way); only the residual
    gap it lands at matters here.
    """
    spy = np.array(POLYHEDRA["square_pyramidal"].vertex_dirs, float)
    tbp = np.array(POLYHEDRA["trigonal_bipyramidal"].vertex_dirs, float)
    spy /= np.linalg.norm(spy, axis=1, keepdims=True)
    tbp /= np.linalg.norm(tbp, axis=1, keepdims=True)

    def slerp(a, b):
        theta = np.arccos(np.clip(a @ b, -1.0, 1.0))
        if theta < 1e-9:  # SPY and TBP already share this slot's direction (both templates' axial/apex)
            return a
        return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)

    pts = np.array([slerp(spy[i], tbp[i]) for i in range(5)])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def zinc_copies(signs):
    """Return disconnected tetrahedral Zn(NH3)(PH3)ClBr, 10 A apart, one per sign of `signs`; -1 is the mirror image."""
    mol = rx.parse_smiles(".".join(["[NH3]->[Zn+2](<-[PH3])(<-[Cl-])<-[Br-]"] * len(signs)))
    tetrahedron = np.array([(1.3, 1.3, 1.3), (-1.3, -1.3, 1.3), (-1.3, 1.3, -1.3), (1.3, -1.3, -1.3)])
    points = []
    for copy, sign in enumerate(signs):
        donors = tetrahedron * [sign, 1, 1] + [10.0 * copy, 0, 0]
        points += [donors[0], (10.0 * copy, 0, 0), *donors[1:]]
    return placed(mol, points)


def bridged_dimer(smiles, right):
    """Return a planar M2(mu-X)2 dimer of `smiles` (terminal, M, terminal, bridge, M, `right` pair, bridge) placed."""
    mol = rx.parse_smiles(smiles)
    left = [(-3.35, 1.6, 0.0), (-1.75, 0.0, 0.0), (-3.35, -1.6, 0.0), (0.0, 1.65, 0.0), (1.75, 0.0, 0.0)]
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, xyz in enumerate([*left, *right, (0.0, -1.65, 0.0)]):
        conf.SetAtomPosition(atom, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol
