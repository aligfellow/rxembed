"""The organic corpus and its selection rule.

SELECTION RULE (deterministic, no RNG, reproducible by inspection):

  Take EVERY metal-free `rx.embed(...)` call that appears verbatim in the repo's own demonstration and
  test surface -- `examples/*.ipynb` code cells and `tests/test_*.py` -- keeping the call's own source,
  constraint verb and indices. Deduplicate by (source, constraint spec). Then:

    * EXCLUDE any call whose input contains a transition metal (that is the metal study's corpus).
    * EXCLUDE borane / carborane cages (UFF lacks B_5/B_6 -- a known parameterisation failure, maintainer
      instruction). No case in the organic surface contains one; the rule is stated for completeness.
    * EXCLUDE `sn2.xyz` -- a documented ETKDG failure (perceived valence-5 C), see memory note
      `xyz-hypervalent-core-limit`. It is not an embed the package claims to support.
    * KEEP unconstrained calls, marked `constrained=False`: `_relax_into_windows` returns early for them,
      so they measure the claim "the relax is a no-op on a free embed" rather than a seed/relax delta.

  Each entry is `(id, family, builder)` where `builder(seed) -> (kwargs for rx.embed, reference)`.
  `reference` is an .xyz path when a real geometry exists for that input, else None.

FAMILIES
  ts        -- frozen reacting-core grafts from a real geometry (`fix=` list, .xyz source)
  ts_smiles -- a TS built from SMILES by exact numbers (`fix=` dict)
  soft      -- soft distance / angle / pi-stack windows (`constrain=`)
  nci       -- NCI contact-seeded complexes (`contacts=`)
  template  -- a reference core transferred onto an analogue (`template=`)
  free      -- unconstrained conformer search (control: the relax must not fire)
"""

from __future__ import annotations

from rdkit import Chem

S = "examples/structures"

# --- the isothiourea / thiourea organocatalysis motifs the concern names -------------------------
SCHREINER = "FC(F)(F)c1cc(cc(c1)C(F)(F)F)NC(=S)Nc1cc(cc(c1)C(F)(F)F)C(F)(F)F.CC(C)=O"  # Schreiner thiourea + acetone
BIMP_SMI = "CC(CN=P(C)(C)C)NC(=S)Nc1ccccc1.NC(=O)C(O)c1ccccc1"  # the BIMP bifunctional catalyst + substrate
TAKEMOTO = "CN(C)[C@@H]1CCCC[C@H]1NC(=S)Nc1cc(cc(c1)C(F)(F)F)C(F)(F)F.CC(=O)C"  # Takemoto thiourea + acetone
CATALYSTS = {
    "tetramisole": "C1CSC2=NC(CN12)c1ccccc1",
    "BTM": "C1CN2C(=NC1c1ccccc1)Sc1ccccc12",
    "HyperBTM": "CC(C)[C@]1CN2C(=N[C@]1c1ccccc1)Sc1ccccc12",
}
ACID = "O=C(O)CCCCc1ccccc1"


def _smarts_idx(smi, smarts, k=0):
    return Chem.MolFromSmiles(smi).GetSubstructMatch(Chem.MolFromSmarts(smarts))[k]


def _nci(smi, prefix):
    """First `rx.nci_candidates` contact whose key starts with `prefix` (candidates are ordered, so this is stable)."""
    import rxembed as rx

    c = rx.nci_candidates(Chem.AddHs(Chem.MolFromSmiles(smi)))
    return c[next(k for k in c if k.startswith(prefix))]


def _sn2():
    smi = "[F-].CCCCCl"
    f = _smarts_idx(smi, "[F-]")
    c = _smarts_idx(smi, "[CH2][Cl]")
    cl = _smarts_idx(smi, "[Cl-,Cl]")
    return smi, {(f, c): 2.02, (c, cl): 2.28, (f, c, cl): 178}


def _amide_ref(seed=7):
    from rdkit.Chem import rdDistGeom

    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"))
    assert rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0
    return m.GetConformer().GetPositions()


def _pi_rings():
    smi = "c1ccccc1CCCCc1ccccc1"
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    ra, rb = (tuple(m) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1")))
    return smi, {(ra, rb): 3.7}


def corpus(n=4):
    """Return the corpus as a list of dicts. `n` is the conformer budget applied uniformly."""
    smi_sn2, fix_sn2 = _sn2()
    smi_pi, con_pi = _pi_rings()
    amide = _amide_ref()
    tm_core = [0, 1, 2, 3]
    oh = _smarts_idx("CC(=O)O.c1ccncc1", "[OX2H]")
    npy = _smarts_idx("CC(=O)O.c1ccncc1", "n")

    C = []

    def add(cid, family, ref=None, **kw):
        kw.setdefault("n", n)
        C.append({"id": cid, "family": family, "ref": ref, "kw": kw})

    # --- ts: frozen reacting-core graft from a REAL geometry (a true reference exists) ------------
    add("bimp", "ts", ref=f"{S}/bimp.xyz", source=f"{S}/bimp.xyz", fix=[10, 11, 12, 14])
    # cores are the DOCUMENTED graphrc ones -- bimp from tests/test_frozen.py, the rest from
    # examples/10_retarget_ts.ipynb ("graphrc reacting bonds"). No index here is invented.
    add("thia-ma", "ts", ref=f"{S}/thia-ma.xyz", source=f"{S}/thia-ma.xyz", fix=[35, 47])
    add("spiro-ts1", "ts", ref=f"{S}/spiro-ts1.xyz", source=f"{S}/spiro-ts1.xyz", fix=[14, 15])
    add("cpa", "ts", ref=f"{S}/cpa.xyz", source=f"{S}/cpa.xyz", fix=[23, 35, 70, 74, 75])
    add("jacob_ts4", "ts", ref=f"{S}/jacob_ts4.xyz", source=f"{S}/jacob_ts4.xyz", fix=[3, 32, 11, 27])
    # bimp_small.xyz is EXCLUDED: no documented reacting core exists for it anywhere in the repo, and
    # inventing indices would make the case unreproducible.

    # --- ts_smiles: exact numbers, no reference geometry ------------------------------------------
    add("sn2-smiles", "ts_smiles", source=smi_sn2, fix=fix_sn2)
    add("nh-stretch", "ts_smiles", source=Chem.AddHs(Chem.MolFromSmiles("CN")), fix=None)  # patched below

    # --- soft: constrain= windows -----------------------------------------------------------------
    add("acid-dist", "soft", source=ACID, constrain={(1, 9): (2.6, 3.0)})
    add("acid-dist-tight", "soft", source=ACID, constrain={(1, 9): (2.70, 2.75)})
    add("acid-angle", "soft", source=ACID, constrain={(2, 1, 0): 120})
    add("acetic-pyridine", "soft", source="CC(=O)O.c1ccncc1", constrain={(oh, npy): (2.6, 3.0)})
    add("pi-stack", "soft", source=smi_pi, constrain=con_pi)

    # --- nci: contact-seeded complexes -------------------------------------------------------------
    add("schreiner-acetone", "nci", source=SCHREINER, contacts=("nci_modes", SCHREINER))
    add("bimp-smiles-auto", "nci", source=BIMP_SMI, contacts="auto")
    add("takemoto-acetone", "nci", source=TAKEMOTO, contacts=("nci_modes", TAKEMOTO))
    add("xb-i-n", "nci", source="FC(F)(F)C(F)(F)I.c1ccncc1", contacts=("cand", "FC(F)(F)C(F)(F)I.c1ccncc1", "XB:"))
    add("catpi", "nci", source="C[N+](C)(C)C.c1ccccc1", contacts=("cand", "C[N+](C)(C)C.c1ccccc1", "CATPI"))
    add(
        "chb-tetramisole",
        "nci",
        source=f"{CATALYSTS['tetramisole']}.CC(=O)OC(C)=O",
        contacts=("cand", f"{CATALYSTS['tetramisole']}.CC(=O)OC(C)=O", "ChB"),
    )
    add("salt-bridge", "nci", source="CC(=O)[O-].CCC[NH3+]", contacts=("cand", "CC(=O)[O-].CCC[NH3+]", "HB:"))
    add(
        "chalcogen-benzothiazole",
        "nci",
        source="c1nc2ccccc2s1.O=C(C)C",
        contacts=("cand", "c1nc2ccccc2s1.O=C(C)C", "ChB"),
    )

    # --- template: reference core onto an analogue -------------------------------------------------
    add("tmpl-anilide", "template", source="CC(=O)Nc1ccccc1", template=(amide, {i: i for i in tm_core}))
    add("tmpl-pMe", "template", source="CC(=O)Nc1ccc(C)cc1", template=(amide, {i: i for i in tm_core}))
    add("tmpl-ptBu", "template", source="CC(=O)Nc1ccc(C(C)(C)C)cc1", template=(amide, {i: i for i in tm_core}))

    # --- free: the control. `_relax_into_windows` must NOT fire ------------------------------------
    for nm, smi in CATALYSTS.items():
        add(f"free-{nm}", "free", source=smi)
    add("free-schreiner", "free", source=SCHREINER.split(".")[0])

    return C


def resolve(entry):
    """Materialise the lazy contact specs into the actual `rx.embed` kwargs."""
    import rxembed as rx

    kw = dict(entry["kw"])
    c = kw.get("contacts")
    if isinstance(c, tuple):
        if c[0] == "nci_modes":
            modes = rx.nci_modes(Chem.AddHs(Chem.MolFromSmiles(c[1])))
            kw["contacts"] = next(iter(modes.values()))
        else:
            kw["contacts"] = _nci(c[1], c[2])
    if entry["id"] == "nh-stretch":
        mol = kw["source"]
        nidx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
        h = next(a.GetIdx() for a in mol.GetAtomWithIdx(nidx).GetNeighbors() if a.GetAtomicNum() == 1)
        kw["fix"] = {(nidx, h): 1.20}
    return kw
