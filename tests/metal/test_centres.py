"""`metal/centres.py`: one canonical graph per metal species, its chemical and engine projections, and its ligands."""

from __future__ import annotations

import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom

import rxembed as rx
from rxembed import graph, stereo
from rxembed.metal import centres
from rxembed.perception.readers import read_xyz
from tests.conftest import EXAMPLES_DIR
from tests.metal_fixtures import BUTADIENE_FE_CO3

_SPL = "[Pd+2](<-[Cl-])(<-[Cl-])<-N"
_ANTI_DIMER = "[Cl-]->[Pd+2]1(<-[NH3])<-[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])"
# id -> drawings of one species: a covalent or ionic metal bond, or a delocalised -1 drawn on another seat
DRAWINGS = {
    "PdCl(NMe2)": ("CN(C)[Pd]Cl", "C[N-](C)->[Pd+2]<-[Cl-]"),
    "FeCl halo-Cp": (
        "F[c-]12->[Fe+2]345(<-[Cl-])<-[cH]([cH]->3[c]->41Br)[cH]->52",
        "F[c]12->[Fe+2]345(<-[Cl-])<-[cH]([cH]->3[c]->41Br)[cH-]->52",
    ),
    "kappa1 acetate": ("CC(=O)[O-]->" + _SPL, "CC([O-])=O->" + _SPL),
    "carbonate": ("[O-]C1=O->[Pd+2](<-N)(<-N)<-[O-]1", "O=C1[O-]->[Pd+2](<-N)(<-N)<-[O-]1"),
    "N-thiocyanate": ("[S-]C#N->" + _SPL, "S=C=[N-]->" + _SPL),
    "amidinate": ("CC1=N(C)->[Pd+2](<-[Cl-])(<-N)<-[N-]1CC", "CC1[N-](C)->[Pd+2](<-[Cl-])(<-N)<-N=1CC"),
    "tfacac": (
        "CC1=CC(C(F)(F)F)=O->[Pd+2](<-[Cl-])(<-N)<-[O-]1",
        "CC1C=C(C(F)(F)F)[O-]->[Pd+2](<-[Cl-])(<-N)<-O=1",
        "CC1[CH-]C(C(F)(F)F)=O->[Pd+2](<-[Cl-])(<-N)<-O=1",
    ),
    "dienolate": (  # slashes beside a single bond state nothing, and a move must not read them as an E/Z
        "[CH2-]C=C(/Cl)C(/F)=CC=O->" + _SPL,
        "[CH2-]C=C(Cl)C(F)=CC=O->" + _SPL,
        "C=CC(Cl)=C(F)C=C[O-]->" + _SPL,
    ),
    # the charge on the O bound to one metal or the other
    "acetate-bridged Pd2": (_ANTI_DIMER + "<-[O-]C(C)=O->1", _ANTI_DIMER + "<-O=C(C)[O-]->1"),
}


def _fields(mol):
    """Return the SMILES and every atom and bond field `canonical_metal_graph` may rewrite, in index order."""
    atoms = [
        (a.GetFormalCharge(), a.GetNumRadicalElectrons(), a.GetNumExplicitHs(), a.GetNoImplicit(), a.GetChiralTag())
        for a in mol.GetAtoms()
    ]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()]
    return Chem.MolToSmiles(mol), atoms, bonds


def test_canonical_metal_graph_draws_each_species_one_way():
    """A covalent bond becomes a dative one with the charges it implies, and a delocalised -1 moves to one seat, in
    any atom order; drawing the result again changes nothing."""
    drawn, once, twice = {}, {}, {}
    for name, drawings in DRAWINGS.items():
        for smiles in drawings:
            mol = rx.parse_smiles(smiles)
            for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
                canonical = centres.canonical_metal_graph(Chem.RenumberAtoms(mol, order))
                once[smiles, order[0]] = _fields(canonical)
                twice[smiles, order[0]] = _fields(centres.canonical_metal_graph(canonical))
                drawn.setdefault(name, set()).add(Chem.MolToSmiles(canonical))

    assert {name: len(seen) for name, seen in drawn.items()} == dict.fromkeys(DRAWINGS, 1)
    assert twice == once
    acac, mapped = (
        Chem.AddHs(rx.parse_smiles(s)) for s in ("CC1=[O]->[Pd+2]<-[O-]1", "[CH3:1][C:2]1=[O:4]->[Pd+2:5]<-[O-:3]1")
    )
    charges = [[a.GetFormalCharge() for a in centres.canonical_metal_graph(mol).GetAtoms()] for mol in (acac, mapped)]
    assert charges[0] == charges[1]  # an atom map is a label, so it moves no charge


def test_charge_never_moves_across_an_unshown_ez():
    """A specified amidinate C=N keeps its -1 on the drawn N; a geometry that shows the E/Z frees it to move."""
    unspecified = centres.canonical_metal_graph(rx.parse_smiles(f"CN(->{_SPL})=C(C)[N-]C"))
    specified = [centres.canonical_metal_graph(rx.parse_smiles(f"C/N(->{_SPL})=C({side}C)[N-]C")) for side in "/\\"]
    assert unspecified.GetAtomWithIdx(1).GetFormalCharge() == -1  # the bound N takes it
    assert [mol.GetAtomWithIdx(8).GetFormalCharge() for mol in specified] == [-1, -1]
    assert len({Chem.MolToSmiles(mol) for mol in specified}) == 2

    embedded = Chem.AddHs(rx.parse_smiles(f"C/N=C(/C)[N-](C)->{_SPL}"))
    assert rdDistGeom.EmbedMolecule(embedded, randomSeed=2) == 0
    redrawn = Chem.RWMol(embedded)  # the -1 on the free N, the E/Z read from the geometry as a Mol file read tags it
    Chem.RemoveStereochemistry(redrawn)
    redrawn.GetBondBetweenAtoms(4, 2).SetBondType(Chem.BondType.DOUBLE)
    redrawn.GetBondBetweenAtoms(1, 2).SetBondType(Chem.BondType.SINGLE)
    redrawn.GetAtomWithIdx(4).SetFormalCharge(0)
    redrawn.GetAtomWithIdx(1).SetFormalCharge(-1)
    redrawn = redrawn.GetMol()
    Chem.SanitizeMol(redrawn)
    Chem.AssignStereochemistryFrom3D(redrawn)
    assert rx.cxsmiles(redrawn) == rx.cxsmiles(embedded)


def test_open_orders_cages_and_unbalanced_bridges_are_refused():
    """A Cp drawn with single bonds only leaves its pi bonds and charge open, so it is not one species."""
    open_cp = Chem.MolFromSmiles("[CH]12[CH]3[CH]4[CH]5[CH]1[Fe]2345Cl", sanitize=False)
    with pytest.raises(ValueError, match=r"bond orders or charges open at C0, C1, C2, C3 and 1 more.*rx\.read_graph"):
        rx.identity(open_cp)
    open_cp.UpdatePropertyCache(strict=False)
    assert rx.identity(rx.read_graph(Chem.AddHs(open_cp), 0)).centres[0].charge == 2  # Cp- Fe(II) Cl-
    with pytest.raises(ValueError, match=r"multi-centre cage vertex atom\(s\) \[0\]"):
        rx.identity(Chem.MolFromSmiles("B(B)(B)(B)(B)B", sanitize=False))
    with pytest.raises(
        ValueError, match=r"neutral bridging donor N1 cannot say which of metals \[3, 9\].*give the charges"
    ):
        rx.identity(Chem.MolFromSmiles("C[N]1(C)[Pd](Cl)(Cl)[N](C)(C)[Pd]1(Cl)Cl", sanitize=False))
    assert rx.isomers("[CH2]1[CH2][Pt-]1(Cl)(Cl)Cl")  # a covalent sigma pair settles as ionic
    assert rx.parse_smiles("[Zn](n1ccccc1)n1ccccc1.[O][O]")  # a bracket diradical states its electrons


def test_native_metal_stereo_and_donor_hands_survive():
    """An RDKit SP tag counts its metal's bonds in drawn order, so the metal redraws every bond or none, leaving its
    input unchanged; an implicit-H amine donor keeps its hand through CX once its metal bond is cut."""
    read = {}
    for tag in ("SP1", "SP2"):
        source = rx.parse_smiles(f"[Cl-]->[Pt@{tag}+2](<-[Cl-])(<-[NH3])<-[NH3]")
        before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
        (selected,) = rx.isomers(source)
        read[tag] = (rx.identity(selected).labels, source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before)
    assert read == {"SP1": (("cis",), True), "SP2": (("trans",), True)}
    text = (
        "CC[N@@H](C)->[Pt+2](<-[Cl-])(<-[Br-])<-[N@@H](C)CC "
        "|atomProp:2.atomNote.s0:4.atomNote.SPL:5.atomNote.s2:6.atomNote.s1:7.atomNote.s3|"
    )
    assert rx.cxsmiles(rx.parse_smiles(text)) == text


def _ligand(smiles, mirror):
    """Return the SMILES of the largest ligand rx.ligands cuts from `smiles` placed by RDKit, mirrored in x by -1."""
    placed = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(placed, randomSeed=1) == 0
    placed.GetConformer().SetPositions(placed.GetConformer().GetPositions() * [mirror, 1, 1])
    return max((Chem.MolToSmiles(Chem.RemoveHs(lig.mol, sanitize=False)) for lig in rx.ligands(placed)), key=len)


def test_ligands_split_per_metal_with_their_own_stereo():
    """The Mn/Fe backbone of mn-h2 bridges kappa5 to Mn and kappa3 to Fe; each ligand carries its own geometry.

    A P-stereogenic phosphine keeps its specified hand on a mirrored geometry, and an imine whose E/Z was drawn
    against Pt has it re-read from the conformer.
    """
    mol = read_xyz(str(EXAMPLES_DIR / "mn-h2.xyz"), metal_charges={0: 2, 1: 1})
    ligands = rx.ligands(mol)
    assert all(lig.mol.GetNumConformers() and lig.mol.GetNumAtoms() == len(lig.atoms) for lig in ligands)
    assert [sorted(len(d) for d in lig.donors.values()) for lig in ligands if len(lig.donors) > 1] == [[3, 5]]
    seen = sorted(lig.atoms[d] for lig in ligands for donors in lig.donors.values() for d in donors)
    assert seen == sorted(n.GetIdx() for m in graph.metal_indices(mol) for n in mol.GetAtomWithIdx(m).GetNeighbors())
    with pytest.raises(ValueError, match="no metal centre found"):
        rx.ligands(Chem.MolFromSmiles("CCO"))
    with pytest.raises(ValueError, match=r"rx\.ligands got no atoms; pass a species"):
        rx.ligands(Chem.Mol())

    assert _ligand("[P@@](C)(CC)(->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-])c1ccccc1", -1) == "CC[P@@](C)c1ccccc1"
    assert "/" in _ligand("C[C@H](F)/C=N(/[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-])C", 1)


def test_bridging_amine_keeps_its_hand_through_the_strip():
    """Stripping both metal bonds leaves the N two neighbours; seeding carries them back to read its hand."""
    mol = Chem.AddHs(rx.parse_smiles("C[N@H](->[Pd](Cl)(Cl)Cl)->[Pt](Br)(Br)Br"))
    with rdBase.BlockLogs():
        assert rdDistGeom.EmbedMolecule(mol, randomSeed=2) == 0
    (species,) = rx.isomers(mol)
    specified = stereo.stereo_from_3d(mol, exclude=graph.metal_indices(mol))

    ens = rx.embed(species, n=1, seed=2)

    assert specified.points == {1: "R"}
    assert ens.n == 1
    assert {
        stereo.stereo_from_3d(Chem.Mol(ens.mol, False, int(cid)), exclude=graph.metal_indices(ens.mol))
        for cid in ens.ids
    } == {specified}


def test_engine_graph_keeps_a_cyclic_boranes_native_bounds():
    """N1CCN->[BH2-]1 alone, beside a sigma-bound copper or a side-on ethylene platinum, keeps RDKit's N...N window and
    its input; its ring through boron is perceived only where a haptic partner makes dative rings count."""
    native = Chem.AddHs(rx.parse_smiles("N1CCN->[BH2-]1"))
    read, expected = {}, {}
    for partner, ring in (("", False), ("N->[Cu+]<-N.", False), ("C1=C->[Pt+2]<-1(<-[Cl-])<-[Cl-].", True)):
        mol = centres.canonical_metal_graph(Chem.AddHs(rx.parse_smiles(partner + "N1CCN->[BH2-]1")))
        Chem.GetDistanceMatrix(mol)
        before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        engine = centres.engine_graph(mol)
        boron = next(atom.GetIdx() for atom in engine.GetAtoms() if atom.GetAtomicNum() == 5)
        fragment = next(atoms for atoms in Chem.GetMolFrags(engine) if boron in atoms)
        cycle = {atom for atom in fragment if engine.GetAtomWithIdx(atom).GetAtomicNum() != 1}
        left, right = sorted(a.GetIdx() for a in engine.GetAtomWithIdx(boron).GetNeighbors() if a.GetAtomicNum() == 7)
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(engine, doTriangleSmoothing=False)
        reference = Chem.Mol(native)
        Chem.GetSymmSSSR(reference, includeDativeBonds=ring)
        first, _boron, last = reference.GetSubstructMatch(Chem.MolFromSmarts("[#7]~[#5]~[#7]"))
        first, last = sorted((first, last))
        window = rdDistGeom.GetMoleculeBoundsMatrix(reference, doTriangleSmoothing=False)
        read[partner] = (
            mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before,
            any(set(atoms) == cycle for atoms in engine.GetRingInfo().AtomRings()),
            (bounds[right, left], bounds[left, right]),
        )
        expected[partner] = (True, ring, pytest.approx((window[last, first], window[first, last])))

    assert read == expected


def test_chemical_graph_keeps_a_bound_dienes_conjugation():
    graph = centres.chemical_graph(centres.canonical_metal_graph(Chem.AddHs(rx.parse_smiles(BUTADIENE_FE_CO3))))
    conjugated = [graph.GetBondBetweenAtoms(left, right).GetIsConjugated() for left, right in ((7, 8), (8, 9), (9, 10))]
    assert conjugated == [True, True, True]
