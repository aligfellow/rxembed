"""Round-trip metal structures through rxembed's public API. Commands and columns are in README.md."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import re
import sys
import time
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolAlign
from rdkit.SimDivFilters import rdSimDivPickers

import rxembed as rx
from rxembed.metal_core import COORDINATION_METALS

HERE = Path(__file__).resolve().parent
_PT = Chem.GetPeriodicTable()
_SEEDS = (42, 7, 1234)  # tried in order per structure; stop at the first that round-trips
_MAX_MATCHES = 500  # cap on symmetry-equivalent substructure matches searched for core_rmsd
_CARBORANE_B = 5  # boron count at which a B/C row reads as a carborane cage, out of the tmQMg sample
FIELDS = "id status stage detail seed seconds core_rmsd metal stereo kappa eta".split()


def _fixtures(only):
    """Return (id, xyz path, charge) for the shipped fixtures, or just `only`."""
    with (HERE / "fixtures.csv").open(newline="") as fh:
        charges = {row["id"]: int(row["charge"]) for row in csv.DictReader(fh)}
    ids = only or list(charges)
    if missing := [i for i in ids if i not in charges]:
        raise SystemExit(f"unknown ids: {' '.join(missing)}")
    return [(i, HERE / "fixtures" / f"{i}.xyz", charges[i]) for i in ids]


def _in_scope(row):
    """Exclude rows `_fingerprint` cannot describe: no/unparsable SMILES, or a carborane-like cage."""
    smiles = row.get("smiles", "")
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    counts = Counter(a.GetAtomicNum() for a in mol.GetAtoms())
    return not (counts[5] >= _CARBORANE_B and counts[6])


def _donor_groups(mol, donor_idxs):
    """Group donor atom indices into sites: donors bonded to each other in `mol` are one site (BFS over
    its bonds). A site of size 1 is a plain (kappa) donor; size >= 2 is a haptic face (eta = its size).
    """  # noqa: D205
    remaining, groups = set(donor_idxs), []
    while remaining:
        group, frontier = set(), [remaining.pop()]
        while frontier:
            a = frontier.pop()
            group.add(a)
            for n in mol.GetAtomWithIdx(a).GetNeighbors():
                if n.GetIdx() in remaining:
                    remaining.discard(n.GetIdx())
                    frontier.append(n.GetIdx())
        groups.append(group)
    return groups


def _fingerprint(row):
    """Build a metal/donor/ligand-graph fingerprint for MaxMin diversity picking.

    Only called on `_in_scope` rows: a parseable SMILES with exactly one `metal_center` atom.
    """
    tokens = [f"M:{row['metal_center']}", f"q:{row['charge']}"]
    mol = Chem.MolFromSmiles(row["smiles"], sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    z = _PT.GetAtomicNumber(row["metal_center"])
    (metal,) = (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == z)
    donors = [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()]
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(metal)
    ligand = rw.GetMol()
    ligand.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(ligand)
    donors = {a - (a > metal) for a in donors}
    elements = Counter(ligand.GetAtomWithIdx(a).GetSymbol() for a in donors)
    denticity, hapticity = Counter(), Counter()
    for fragment in Chem.GetMolFrags(ligand):
        sites = donors & set(fragment)
        if not sites:
            continue
        denticity[len(sites)] += 1
        for group in _donor_groups(ligand, sites):
            if len(group) > 1:
                hapticity[len(group)] += 1
    tokens.append(f"CN:{len(donors)}")
    tokens += [f"D:{e}:{n}" for e, c in elements.items() for n in range(1, c + 1)]
    tokens += [f"dent:{v}:{n}" for v, c in denticity.items() for n in range(1, c + 1)]
    tokens += [f"eta:{v}:{n}" for v, c in hapticity.items() for n in range(1, c + 1)]
    fp = DataStructs.ExplicitBitVect(128)
    for bit in Chem.RDKFingerprint(ligand, minPath=1, maxPath=2, fpSize=32, nBitsPerHash=1).GetOnBits():
        fp.SetBit(bit)
    for token in tokens:
        fp.SetBit(32 + zlib.crc32(token.encode()) % 96)
    return fp, tuple(sorted(tokens))


_COVERAGE_PANELS = ("metal", "CN", "donor element", "denticity", "hapticity")


def _categories(tokens):
    """Read one row's coverage categories from its `_fingerprint` tokens, each a set of string labels.

    `denticity` (kappa-n) and `hapticity` (eta-n) both come from `_donor_groups`: a fragment's donor
    atoms bonded to each other are one site, so a multi-atom site is a haptic face rather than several
    kappa teeth. A row with no haptic face at all reads as `none`, not an empty set.
    """
    return {
        "metal": {t.split(":", 1)[1] for t in tokens if t.startswith("M:")},
        "CN": {t.split(":", 1)[1] for t in tokens if t.startswith("CN:")},
        "donor element": {t.split(":")[1] for t in tokens if t.startswith("D:")},
        "denticity": {f"κ{t.split(':')[1]}" for t in tokens if t.startswith("dent:")},
        "hapticity": {f"η{t.split(':')[1]}" for t in tokens if t.startswith("eta:")} or {"none"},
    }


def _coverage_counts(rows, features, sample_ids):
    """Count in-scope tmQMg structures per coverage category, for the population; record each sampled
    structure's own categories as `panel:category` tokens, for `plot.py` to split its sample bars into
    pass/fail by joining these tokens against a results CSV's `status` by id.

    Every count is read from the `_fingerprint` tokens the MaxMin pick itself used (`features`, aligned
    with `rows`), so the coverage figure compares the sample to the same chemistry space it was picked
    from. A row missing a panel's tokens (e.g. no single-metal SMILES parse) is excluded from that
    panel's population total, not counted as a zero; a sampled row missing a panel's tokens carries no
    token for that panel either.
    """  # noqa: D205
    sample_ids = set(sample_ids)
    counts = {p: Counter() for p in _COVERAGE_PANELS}
    totals = dict.fromkeys(_COVERAGE_PANELS, 0)
    sample_tokens = {}
    for row, (_fp, tokens) in zip(rows, features, strict=True):
        cats = _categories(tokens)
        for panel, labels in cats.items():
            if not labels:
                continue
            counts[panel].update(labels)
            totals[panel] += 1
        if row["id"] in sample_ids:
            sample_tokens[row["id"]] = [f"{panel}:{label}" for panel, labels in cats.items() for label in labels]
    return {
        "population_n": len(rows),
        "sample_n": len(sample_ids),
        "panels": {p: {"population": dict(counts[p]), "population_total": totals[p]} for p in _COVERAGE_PANELS},
        "sample_tokens": sample_tokens,
    }


def _sample_tmqmg(rows, size):
    """Return (`size` tmQMg IDs, coverage counts) via a deterministic (seed=0) MaxMin pick seeded by one
    typical structure per metal.
    """  # noqa: D205
    features = [_fingerprint(row) for row in rows]
    first = []
    for metal in sorted({row["metal_center"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["metal_center"] == metal]
        counts = Counter(features[i][1] for i in indices)
        mode = max(counts, key=lambda value: (counts[value], value))
        typical = [i for i in indices if features[i][1] == mode]
        middle = sorted(int(rows[i]["n_atoms"]) for i in typical)[len(typical) // 2]
        first.append(min(typical, key=lambda i: (abs(int(rows[i]["n_atoms"]) - middle), rows[i]["id"])))
    started = time.monotonic()
    picks = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(
        [fp for fp, _tokens in features], len(rows), size, first, seed=0
    )
    picks = list(picks)[:size]  # never fewer than the one-per-metal seed picks, even for a smaller `size`
    print(f"tmqmg: picked {len(picks)} of {len(rows)} in-scope structures in {time.monotonic() - started:.1f} s")
    ids = [rows[i]["id"] for i in picks]
    return ids, _coverage_counts(rows, features, ids)


def _tmqmg(size, only):
    """Return ((id, xyz path, charge) for a MaxMin-diverse tmQMg sample, or just `only`; coverage counts,
    or None for an `--only` subset).
    """  # noqa: D205
    data = os.environ.get("RXEMBED_TMQMG_DIR", "")
    xyz_dir, table = Path(data, "xyz"), Path(data, "tmQMg_properties_and_targets.csv")
    if not data or not xyz_dir.is_dir():
        raise SystemExit("set RXEMBED_TMQMG_DIR to a tmQMg clone's data/ directory and unzip tmQMg_xyz.zip there")
    with table.open(newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if _in_scope(row)]
    charges = {row["id"]: int(row["charge"]) for row in rows}
    ids, coverage = (only, None) if only else _sample_tmqmg(rows, size)
    if missing := [i for i in ids if i not in charges]:
        raise SystemExit(f"unknown ids: {' '.join(missing)}")
    return [(i, xyz_dir / f"{i}.xyz", charges[i]) for i in ids], coverage


def _positions_mol(positions):
    """Build a bare Mol whose sole conformer holds the given Cartesian positions, for rdMolAlign."""
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(positions))
    for i, pos in enumerate(positions):
        mol.AddAtom(Chem.Atom(6))
        conf.SetAtomPosition(i, [float(v) for v in pos])
    mol = mol.GetMol()
    mol.AddConformer(conf)
    return mol


def _flattened(mol):
    """Copy `mol` with formal charges zeroed and every bond set to a plain single bond, so a resonance
    split or a dative/covalent difference cannot fracture a substructure match.
    """  # noqa: D205
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return mol


def core_rmsd(ref, mol):
    """Symmetry-aware RMSD of the metal(s) and their bonded donors, between `ref` (read from the XYZ) and
    `mol` (embedded fresh from `ref`'s own CX string, so its atom order does not match `ref`'s).

    Sites are matched between the two mols by element and graph position, never by index: a full-graph
    substructure search on both mols, flattened (see `_flattened`), finds every symmetry-consistent
    correspondence (e.g. which of two equivalent Cl legs maps to which); the correspondence giving the
    lowest RMSD over the metal(s) and their directly bonded donor atoms is kept.
    """  # noqa: D205
    metals = [a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() in COORDINATION_METALS]
    if not metals:
        return float("nan")
    sites = metals + sorted({n.GetIdx() for m in metals for n in ref.GetAtomWithIdx(m).GetNeighbors()})
    matches = _flattened(mol).GetSubstructMatches(_flattened(ref), uniquify=False, maxMatches=_MAX_MATCHES)
    if not matches:
        return float("nan")
    ref_pos, mol_pos = ref.GetConformer().GetPositions(), mol.GetConformer().GetPositions()
    target = _positions_mol(ref_pos[sites])
    identity = [[(i, i) for i in range(len(sites))]]
    return min(
        rdMolAlign.GetBestRMS(_positions_mol(mol_pos[[match[i] for i in sites]]), target, map=identity)
        for match in matches
    )


def _short(err):
    """Format an error (string or exception) as one line capped at 200 characters."""
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    return msg.replace("\n", " ")[:200]


def _metal_symbol(mol):
    """Return the first coordination-metal element symbol in `mol`, or '' if none."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in COORDINATION_METALS:
            return atom.GetSymbol()
    return ""


_METAL_SYMBOLS = {_PT.GetElementSymbol(z) for z in COORDINATION_METALS}


def _xyz_metal(path):
    """Return the first coordination-metal element symbol in `path`'s raw XYZ atom list, or ''.

    Used only when `rx.read_xyz` itself failed, so no perceived Mol exists to read `_metal_symbol` from;
    the raw element column still gives every by-metal chart a bucket for a read failure.
    """
    for line in Path(path).read_text().splitlines()[2:]:
        fields = line.split()
        if fields and fields[0] in _METAL_SYMBOLS:
            return fields[0]
    return ""


def _kappa_eta(mol):
    """Read denticity (kappa) and hapticity (eta) from `rx.ligands`, with no isomer enumeration.

    Per ligand, per metal it binds: group its donor atoms into sites by the ligand's own bonds, so donors
    bonded to each other (a Cp ring, an eta2 alkene) are one site, matching rxembed's adjacent-donor rule.
    `kappa` is each ligand's site count; `eta` is the sizes of sites with more than one atom, empty with
    no haptic site.
    """
    kappas, etas = set(), set()
    for lig in rx.ligands(mol):
        for donor_idxs in lig.donors.values():
            groups = _donor_groups(lig.mol, donor_idxs)
            kappas.add(len(groups))
            etas |= {len(g) for g in groups if len(g) > 1}
    return ";".join(str(k) for k in sorted(kappas)), ";".join(str(e) for e in sorted(etas))


def _stereo_kinds(ref):
    """Read the input's own stereo kinds through the public, non-enumerating `rx.metal(ref, lengths=
    "model", observed_only=True)[0]`: it reads the one arrangement `ref` already has, nothing more.

    Descriptor only, never a gate: if perception raises (an unsupported polyhedron, say), stereo comes
    back empty and pass/fail is untouched. A kind is a point (R/S/r/s -> 'point', CW/CCW -> 'donor
    hand', locked with no CIP), a bond ('E/Z'), or an axis ('diene' when both its atoms sit in one
    haptic face, else 'atrop'), plus 'winding' / 'planar' for the metal centre itself.
    """  # noqa: D205
    try:
        iso = rx.metal(ref, lengths="model", observed_only=True)[0]
    except Exception:
        return ""
    kinds = {k for k, v in (("winding", iso.chirality), ("planar", iso.haptic_configuration)) if v}
    faces = [set(atoms) for atoms in iso.haptic.values()]
    for part in filter(None, iso.stereo_label.split(",")):
        if "=" in part:
            kinds.add("E/Z")
        elif "-" in part:
            atoms = {int(n) for n in re.findall(r"\d+", part.split(":")[0])}
            kinds.add("diene" if any(atoms <= face for face in faces) else "atrop")
        else:
            kinds.add("donor hand" if part.rsplit(":", 1)[-1] in ("CW", "CCW") else "point")
    return ";".join(sorted(kinds))


def _attempt(q, path, charge, seed):
    """Read the structure, take its own CX, embed that CX string fresh at `seed`, and check the embedded
    CX still matches: the round trip goes through the CX string alone. Puts one row-dict on `q`.
    """  # noqa: D205
    try:
        ref = rx.read_xyz(str(path), charge=charge, connectivity="xyzgraph", bond_orders="xyz2mol")
    except Exception as exc:
        q.put({"status": "fail", "stage": "read", "detail": _short(exc), "metal": _xyz_metal(path)})
        return
    metal = _metal_symbol(ref)
    try:
        cx = rx.cxsmiles(ref)
    except Exception as exc:
        q.put({"status": "fail", "stage": "cx", "detail": _short(exc), "metal": metal})
        return
    stereo = _stereo_kinds(ref)
    try:
        ens = rx.embed(cx, n=1, seed=seed, threads=1)
        if isinstance(ens, rx.EnsembleSet):
            ens = ens[0]  # a labile stereocentre cxsmiles leaves free; the CX check does not depend on which
        mol = Chem.Mol(ens.mol, False, int(ens.ids[0]))
    except Exception as exc:
        q.put({"status": "fail", "stage": "embed", "detail": _short(exc), "metal": metal, "stereo": stereo})
        return
    rmsd = ""
    with contextlib.suppress(Exception):
        rmsd = f"{core_rmsd(ref, mol):.4f}"
    try:
        cx_embed = rx.cxsmiles(mol)
    except Exception as exc:
        q.put(
            {
                "status": "fail",
                "stage": "match",
                "detail": _short(exc),
                "core_rmsd": rmsd,
                "metal": metal,
                "stereo": stereo,
            }
        )
        return
    if cx_embed != cx:
        q.put(
            {
                "status": "fail",
                "stage": "match",
                "detail": "embedded CX != input CX",
                "core_rmsd": rmsd,
                "metal": metal,
                "stereo": stereo,
            }
        )
        return
    kappa, eta = _kappa_eta(ref)
    q.put(
        {
            "status": "pass",
            "stage": "",
            "stereo": stereo,
            "detail": "",
            "core_rmsd": rmsd,
            "metal": metal,
            "kappa": kappa,
            "eta": eta,
        }
    )


def _run_seed(path, charge, seed, deadline):
    """Run one seed's `_attempt` in a forked child killed at the deadline; RDKit's C++ cannot be
    interrupted from Python.
    """  # noqa: D205
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_attempt, args=(q, path, charge, seed))
    proc.start()
    proc.join(deadline)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"status": "timeout", "stage": "", "detail": f"exceeded {deadline} s", "metal": _xyz_metal(path)}
    if not q.empty():
        return q.get()
    return {"status": "fail", "stage": "worker", "detail": "worker exited without a result", "metal": _xyz_metal(path)}


def _run_one(path, charge, deadline):
    """Try `_SEEDS` in order, each its own forked attempt, stopping at the first that round-trips."""
    row = {}
    for seed in _SEEDS:
        row = _run_seed(path, charge, seed, deadline)
        if row["status"] == "pass":
            return row | {"seed": str(seed)}
    return row | {"seed": ""}


def _compare(rows, baseline_path):
    """Print every structure this run (or `--only` subset) lost or gained against baseline.csv's pass/
    fail verdict; return 1 if any lost.
    """  # noqa: D205
    with baseline_path.open(newline="") as fh:
        base = {row["id"]: row for row in csv.DictReader(line for line in fh if not line.startswith("#"))}
    lost = gained = 0
    for r in sorted(rows, key=lambda r: r["id"]):
        b_pass, n_pass = base.get(r["id"], {}).get("status") == "pass", r["status"] == "pass"
        if b_pass and not n_pass:
            lost += 1
            print(f"lost: {r['id']} ({r['stage'] or 'missing'}: {r['detail']})")
        elif n_pass and not b_pass:
            gained += 1
            print(f"gained: {r['id']}")
    print(f"{lost} lost, {gained} gained against {baseline_path.name}")
    return int(lost > 0)


def _summarize(rows):
    """Print the pass count, failures grouped by stage, and the ten slowest structures."""
    print(f"{sum(r['status'] == 'pass' for r in rows)}/{len(rows)} pass")
    for stage, n in Counter(r["stage"] for r in rows if r["status"] != "pass").most_common():
        print(f"{n} failed at {stage or '(none)'}")
    for r in sorted(rows, key=lambda r: float(r["seconds"]), reverse=True)[:10]:
        print(f"{r['seconds']}s {r['id']} {r['status']}")


def main(argv=None):
    """Run a cohort into one results CSV; a `fixtures` run also prints its comparison against baseline.csv."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("fixtures", "tmqmg"):
        argv = ["fixtures", *argv]  # `just bench [--only ...]` needs no subcommand

    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--only", nargs="+", metavar="ID", help="run just these IDs")
    common.add_argument("--timeout", type=int, default=10, help="seconds per seed attempt")
    common.add_argument("--out", type=Path, help="results CSV (default benchmark/results/<cohort>-<UTC>.csv)")
    sub = parser.add_subparsers(dest="cohort", required=True)
    sub.add_parser("fixtures", parents=[common], help="the 100 shipped fixtures, compared against baseline.csv")
    tmqmg = sub.add_parser("tmqmg", parents=[common], help="a MaxMin-diverse sample of a local tmQMg clone")
    tmqmg.add_argument("--size", type=int, default=100, help="sample size (ignored with --only)")
    args = parser.parse_args(argv)

    jobs, coverage = (_fixtures(args.only), None) if args.cohort == "fixtures" else _tmqmg(args.size, args.only)
    out = args.out or HERE / "results" / f"{args.cohort}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if coverage is not None:
        out.with_name(f"{out.stem}-coverage.json").write_text(json.dumps(coverage))
    header = {
        "rxembed_version": rx.__version__,
        "rdkit_version": rdkit.__version__,
        "seeds": _SEEDS,
        "timeout": args.timeout,
    }
    rows = []
    with out.open("x", newline="") as fh:
        fh.write("# " + json.dumps(header) + "\n")
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for structure_id, path, charge in jobs:
            started = time.monotonic()
            row = _run_one(path, charge, args.timeout)
            row |= {"id": structure_id, "seconds": f"{time.monotonic() - started:.2f}"}
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(structure_id, row["status"], row["stage"], row["seconds"])
    _summarize(rows)
    return _compare(rows, HERE / "baseline.csv") if args.cohort == "fixtures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
