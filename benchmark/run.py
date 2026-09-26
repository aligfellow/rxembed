"""Round-trip metal structures through rxembed's public API and independently re-check them.

Commands, output columns and the compare rule are in README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

import rxembed as rx

# The one private import: the redox re-read must judge both sides through the choke point `rx.metal` applies.
from rxembed.metal_core import canonical_metal_graph as _canonical_metal_graph

HERE = Path(__file__).resolve().parent
# Hashed at import, not at write time: an edit mid-run must not be recorded as the code that ran.
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Above the M-L distance model's worst case; a wrongly seated sphere moves donors much further.
MAX_CORE_RMSD = 0.75  # Angstrom
# The core's coordination-metal range, copied so `canonical_metal_graph` stays the only private import.
_METALS = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(57, 81)) | frozenset(range(89, 113))
_MIN_CORE_SITES = 2  # a metal and at least one site; fewer cannot define an alignment

FIELDS = ["id", "seed", "status", "stage", "detail", "seconds", "core_rmsd", "isomers", "embedded", "valid", "valid_cx"]


def _ids(name):
    """Return the IDs listed in benchmark/<name>, ignoring `#` comments."""
    return [i for line in (HERE / name).read_text().splitlines() for i in line.split("#", 1)[0].split()]


def _cohort(args):
    """Return (id, xyz path, charge) per structure; exit up front on a missing tmQMg clone or an unknown ID."""
    if args.cohort == "fixtures":
        xyz, table = HERE / "fixtures", HERE / "fixtures.csv"
    else:
        data = os.environ.get("RXEMBED_TMQMG_DIR", "")
        xyz, table = Path(data, "xyz"), Path(data, "tmQMg_properties_and_targets.csv")
        if not data or not xyz.is_dir():
            raise SystemExit("set RXEMBED_TMQMG_DIR to a tmQMg clone's data/ directory and unzip tmQMg_xyz.zip there")
    with table.open(newline="") as fh:
        charges = {row["id"]: row["charge"] for row in csv.DictReader(fh)}
    if args.cohort == "sample":
        listed = _ids("tmqmg.txt")[: args.size]
    elif args.cohort == "issues":
        listed = _ids("issues.txt")
    else:
        listed = list(charges)
    ids = args.only or listed
    missing = [i for i in ids if i not in charges]
    if missing:
        raise SystemExit(f"unknown ids: {' '.join(missing)}")
    return [(i, xyz / f"{i}.xyz", int(charges[i])) for i in ids]


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


def _site_centroids(mol, groups):
    """Return one point per site: its atoms' mean position."""
    positions = mol.GetConformer().GetPositions()
    return np.array([positions[group].mean(axis=0) for group in groups])


def _site_groups(ref, iso):
    """Return one atom-index group per coordination site: a lone donor, or a whole haptic face.

    A face scored per carbon would read a free spin about the metal axis as a large error.
    """
    metals = [atom.GetIdx() for atom in ref.GetAtoms() if atom.GetAtomicNum() in _METALS]
    if iso is not None:
        sites = [list(iso.haptic.get(v, [v])) for v in iso.vertices]
        return [[i] for i in metals] + sites
    donors = {n.GetIdx() for i in metals for n in ref.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() > 1}
    return [[i] for i in metals] + [[i] for i in sorted(donors)]


def core_rmsd(ref, cand, iso=None, max_matches=5000):
    """Return the RMSD of the metal and its site centroids, minimised over swaps of graph-equivalent sites.

    A fixed pairing would read two equivalent donors that changed places as an error. The search stops
    after `max_matches` swaps, which can only overstate the RMSD.
    """
    groups = _site_groups(ref, iso)
    n = min(ref.GetNumAtoms(), cand.GetNumAtoms())
    if len(groups) < _MIN_CORE_SITES or any(i >= n for g in groups for i in g):
        return float("nan")
    graph = Chem.Mol(ref)
    for bond in graph.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in graph.GetAtoms():
        atom.SetIsAromatic(False)
        # Formal charge is the same Lewis artefact as bond order: a resonance-delocalised donor pair
        # (dithiocarbamate S,S-; a Cp ring anion) carries its -1 on one arbitrarily chosen atom, which
        # would rank two equivalent donors apart and hide the swap that makes them interchangeable.
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
    ranks = list(Chem.CanonicalRankAtoms(graph, breakTies=False))
    classes = {}
    for i, group in enumerate(groups):
        classes.setdefault(tuple(sorted(ranks[a] for a in group)), []).append(i)
    orders = [{}]
    for members in classes.values():
        orders = [
            base | dict(zip(members, order, strict=True))
            for base in orders
            for order in itertools.permutations(members)
        ][:max_matches]
    n_sites = len(groups)
    target = _positions_mol(_site_centroids(ref, groups))
    probe = _positions_mol(_site_centroids(cand, groups))  # candidate sites in their own natural order
    return rdMolAlign.GetBestRMS(
        probe, target, map=[[(order.get(i, i), i) for i in range(n_sites)] for order in orders]
    )


def _edge_diff(a, b):
    """Return (formed, lost) sorted atom-index bond pairs between two graphs of the same atom order."""

    def edges(mol):
        return {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in mol.GetBonds()
            if bond.GetBondType() != Chem.BondType.ZERO
        }

    ea, eb = edges(a), edges(b)
    return sorted(eb - ea), sorted(ea - eb)


def _constitution(mol):
    """Return a stereo-free `rx.dative_smiles` key, whose canonical ionic form hides no graph change.

    Bond stereo is cleared too: `RemoveStereochemistry` keeps atropisomer stereo, on which `dative_smiles` raises.
    """
    graph = Chem.Mol(mol)
    Chem.RemoveStereochemistry(graph)
    for bond in graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    graph.RemoveAllConformers()
    return rx.dative_smiles(graph)


def _redox_key(mol):
    """Return a per-fragment key that ignores which side of a metal-ligand bond carries the charge.

    A non-innocent ligand (dithiolene, porphyrin) has closed-shell readings that only move charge between
    metal and ligand. The key keeps each fragment's element graph, H counts, total charge and radical count,
    re-parsed from SMILES so a stale ring cache on a rebuilt isomer cannot steer the canonical form.
    """
    keys = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        work = Chem.RWMol(frag)
        work.RemoveAllConformers()
        work.UpdatePropertyCache(strict=False)
        charge = sum(atom.GetFormalCharge() for atom in work.GetAtoms())
        radicals = sum(atom.GetNumRadicalElectrons() for atom in work.GetAtoms())
        for atom in work.GetAtoms():
            h = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(h)
            for prop in list(atom.GetPropNames()):
                atom.ClearProp(prop)
        for bond in work.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
        flat = Chem.MolFromSmiles(Chem.MolToSmiles(work), sanitize=False)
        flat.UpdatePropertyCache(strict=False)
        keys.append((Chem.MolToSmiles(flat), charge, radicals))
    return sorted(keys)


def _transplant(mol, fresh):
    """Return `mol`'s own graph (bond types, charges, radicals) placed on `fresh`'s coordinates."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    positions = fresh.GetConformer().GetPositions()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*positions[i]))
    return out


def _short(err):
    """Format an error (string or exception) as one line capped at 200 characters."""
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    return msg.replace("\n", " ")[:200]


def _requested_cx(mol, iso):
    """Return the CX of `mol` read in the isomer's requested polyhedron, not its argmin reading."""
    if len(iso.centres) != 1:
        return rx.cxsmiles(mol)
    return rx.cxsmiles(rx.metal(mol, geometry=iso.geometry, observed_only=True)[0])


def _validate(ref, mol, ens, cid, iso, expected_cx, charge, xyz_path):
    """Re-check one written candidate from a fresh `rx.read_xyz`; return (ok, stage, detail, fresh_cx).

    The first failed check wins, and an exception is labelled with the step that raised it. A wrong
    charge raises inside `rx.read_xyz` (validate:read). A constitution change that is only a Lewis
    re-read (same `_redox_key`) passes with a note, and the cx step then reads the candidate's own
    graph at the fresh coordinates.
    """
    step = "read"
    try:
        fresh = rx.read_xyz(str(xyz_path), charge=charge, connectivity="xyzgraph", bond_orders="xyz2mol")

        step = "fresh"
        heavy = [
            atom.GetIdx() for atom in ref.GetAtoms() if atom.GetAtomicNum() > 1 and atom.GetIdx() < mol.GetNumAtoms()
        ]
        target = _positions_mol(ref.GetConformer().GetPositions()[heavy])
        probe = _positions_mol(mol.GetConformer().GetPositions()[heavy])
        if rdMolAlign.AlignMol(probe, target) < 0.01:  # noqa: PLR2004
            return False, "validate:fresh", "input geometry returned", ""

        step = "connectivity"
        # `read_xyz` drops donorless bridgehead M-X bonds on every read, so both graphs had the same guard.
        formed, lost = _edge_diff(mol, fresh)
        if formed or lost:
            return False, "validate:connectivity", _short(f"formed {formed}, lost {lost}"), ""

        step = "constitution"
        note = ""
        if _constitution(mol) != _constitution(fresh):
            if _redox_key(_canonical_metal_graph(mol)) != _redox_key(_canonical_metal_graph(fresh)):
                return False, "validate:constitution", "graph changed across the XYZ round trip", ""
            note = "Lewis re-read: same redox key, different electron split"
            fresh = _transplant(mol, fresh)

        step = "cx"
        fresh_cx = rx.cxsmiles(fresh)
        if _requested_cx(fresh, iso) != expected_cx:
            return False, "validate:cx", "fresh CX != expected isomer CX", fresh_cx

        step = "geometry"
        report = ens.check()[cid]
        if not report:
            return False, "validate:geometry", _short(report.summary()), fresh_cx

        return True, "", note, fresh_cx
    except Exception as exc:
        return False, f"validate:{step}", _short(exc), ""


def _counts(candidates):
    """Return a row's isomer counts and the sorted 8-character hashes of its valid isomers' CX."""
    valid = [c for c in candidates if c["valid"]]
    return {
        "isomers": len(candidates),
        "embedded": sum(c["stage"] != "embed" for c in candidates),
        "valid": len(valid),
        "valid_cx": " ".join(sorted(hashlib.sha1(c["cx"].encode()).hexdigest()[:8] for c in valid)),
    }


def _verdict(reference_cx, candidates):
    """Return the structure's row: its reference isomer must be unique, valid and near the input."""
    row = _counts(candidates) | {"core_rmsd": ""}
    refs = [c for c in candidates if c["cx"] == reference_cx]
    if len(refs) != 1:
        detail = "reference CX matched twice" if refs else "reference CX absent from the enumeration"
        return row | {"status": "fail", "stage": "enumerate", "detail": detail}
    ref = refs[0]
    row["core_rmsd"] = ref["core_rmsd"]
    other = next((c for c in candidates if c is not ref and c["fresh_cx"] == reference_cx), None)
    if other is not None:
        detail = f"candidate {other['k']} also read back as the reference structure"
        return row | {"status": "fail", "stage": "validate:identity", "detail": detail}
    # An empty or nan core_rmsd could not be measured and never gates (nan compares false).
    if ref["valid"] and ref["core_rmsd"] and float(ref["core_rmsd"]) > MAX_CORE_RMSD:
        detail = f"reference core RMSD {float(ref['core_rmsd']):.2f} A exceeds {MAX_CORE_RMSD} A"
        return row | {"status": "fail", "stage": "validate:core-rmsd", "detail": detail}
    return row | {"status": "pass" if ref["valid"] else "fail", "stage": ref["stage"], "detail": ref["error"]}


def _worker(send, path, charge, seed, xyz_stem):
    """Round-trip one structure, sending each stage name and each finished isomer as it happens."""
    stage = "read"
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        send.send(("stage", stage))
        ref = rx.read_xyz(str(path), charge=charge, connectivity="xyzgraph", bond_orders="xyz2mol")

        stage = "enumerate"
        send.send(("stage", stage))
        isos = rx.metal(ref, lengths="model")
        reference_cx = rx.cxsmiles(ref)
        expected = [rx.cxsmiles(iso) for iso in isos]

        candidates = []
        for k, (iso, cx) in enumerate(zip(isos, expected, strict=True), start=1):
            stage = f"embed {k}/{len(isos)}"
            send.send(("stage", stage))
            c = {"k": k, "cx": cx, "valid": False, "stage": "embed", "error": "", "fresh_cx": "", "core_rmsd": ""}
            try:  # a failure here belongs to this isomer alone
                ens = rx.embed(iso, n=1, seed=seed, threads=1)
                c["stage"] = "write"
                cid = ens.ids[0]
                xyz_path = Path(f"{xyz_stem}-k{k}.xyz")
                Chem.MolToXYZFile(ens.mol, str(xyz_path), confId=cid)
                mol = Chem.Mol(ens.mol, False, cid)
                if cx == reference_cx:
                    with contextlib.suppress(Exception):
                        c["core_rmsd"] = f"{core_rmsd(ref, mol, iso):.4f}"
                c["valid"], c["stage"], c["error"], c["fresh_cx"] = _validate(
                    ref, mol, ens, cid, iso, cx, charge, xyz_path
                )
            except Exception as exc:
                c["error"] = _short(exc)
            candidates.append(c)
            send.send(("iso", c))

        send.send(("done", _verdict(reference_cx, candidates)))
    except Exception as exc:
        send.send(("done", {"status": "fail", "stage": stage, "detail": _short(exc)}))


def _run_one(path, charge, seed, xyz_stem, timeout):
    """Run `_worker` in a forked child killed at the deadline, so a hang or a crash can never pass.

    RDKit's C++ cannot be interrupted from Python. A timeout or a dead child keeps the last stage
    reached and the isomers that already finished.
    """
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(send, path, charge, seed, xyz_stem))
    proc.start()
    send.close()
    deadline = time.monotonic() + timeout
    stage, candidates = "start", []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not recv.poll(remaining):
            row = {"status": "timeout", "stage": stage, "detail": f"exceeded {timeout} s during {stage}"}
            break
        try:
            kind, value = recv.recv()
        except EOFError:
            row = {"status": "fail", "stage": "worker", "detail": f"worker exited without a result during {stage}"}
            break
        if kind == "stage":
            stage = value
        elif kind == "iso":
            candidates.append(value)
        else:
            row = value
            break
    proc.kill()
    proc.join()
    return _counts(candidates) | {"core_rmsd": ""} | row


def _load(path):
    """Return {(id, seed): row} from a results CSV, skipping its `#` provenance line."""
    rows = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            key = (row["id"], int(row["seed"]))
            if key in rows:
                raise SystemExit(f"{path}: duplicate row for {key}")
            rows[key] = row
    return rows


def compare(base_path, new_path):
    """Print what NEW lost against BASE by majority vote over their shared seeds; return 1 if anything was lost.

    A reference or a valid isomer (by CX hash) is lost when it held a majority of seeds in BASE and
    not in NEW. A missing row counts as a non-pass. Isomers of an ID with a timeout in NEW are skipped,
    since unreached isomers are not evidence either way.
    """
    base, new = _load(base_path), _load(new_path)
    seeds = sorted({s for _, s in base} & {s for _, s in new})
    need = len(seeds) // 2 + 1
    base_ids, new_ids = {i for i, _ in base}, {i for i, _ in new}
    ids = sorted(base_ids & new_ids)
    if not ids or not seeds:
        raise SystemExit(f"{base_path} and {new_path} share no ids or no seeds")
    print(f"{len(base_ids - new_ids)} ids only in BASE, {len(new_ids - base_ids)} only in NEW")
    tally = Counter()

    def vote(kind, what, b, n, why=""):
        change = "lost" if b >= need > n else "gained" if n >= need > b else "noise" if b != n else "same"
        tally[kind, change] += 1
        if change == "lost":
            print(f"lost {kind}: {what} {b}/{len(seeds)} -> {n}/{len(seeds)}{why}")

    for i in ids:
        olds = [base.get((i, s), {}) for s in seeds]
        news = [new.get((i, s), {}) for s in seeds]
        passes = [sum(r.get("status") == "pass" for r in rows) for rows in (olds, news)]
        failed = next((r for r in news if r.get("status") != "pass"), {})
        vote("reference", i, *passes, f" ({failed.get('stage', 'missing')}: {failed.get('detail', '')})")
        if any(r.get("status") == "timeout" for r in news):
            tally["isomer", "skipped"] += 1
            continue
        b, n = (Counter(h for r in rows for h in set(r.get("valid_cx", "").split())) for rows in (olds, news))
        for h in sorted(b | n):
            vote("isomer", f"{i} {h}", b[h], n[h])
    for kind in ("reference", "isomer"):
        print(f"{kind}s:", ", ".join(f"{tally[kind, c]} {c}" for c in ("lost", "gained", "noise", "skipped")))
    for name, rows in (("BASE", base), ("NEW", new)):
        shared = [rows[i, s] for i in ids for s in seeds if (i, s) in rows]
        passed = [sum(r["status"] == "pass" for r in shared if int(r["seed"]) == s) for s in seeds]
        seconds = [float(r["seconds"]) for r in shared]
        median, total = statistics.median(seconds), sum(seconds)
        print(f"{name}: {passed} of {len(ids)} pass per seed, {median:.1f} s median, {total:.0f} s total")
    return int(tally["reference", "lost"] + tally["isomer", "lost"] > 0)


def main(argv=None):
    """Run a cohort into one results CSV, or `compare BASE.csv NEW.csv`."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compare"]:
        if len(argv) != 3:  # noqa: PLR2004
            raise SystemExit("usage: run.py compare BASE.csv NEW.csv")
        return compare(Path(argv[1]), Path(argv[2]))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("fixtures", "sample", "issues"))
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these IDs")
    parser.add_argument("--size", type=int, default=100, help="tmQMg sample size (sample cohort only)")
    parser.add_argument("--seed", type=int, nargs="+", default=[42], help="rx.embed random seed(s)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per structure and seed")
    parser.add_argument("--out", type=Path, help="results CSV (default benchmark/results/<cohort>-<UTC>.csv)")
    parser.add_argument("--keep-xyz", action="store_true", help="keep candidate XYZ files in <out stem>-xyz/")
    args = parser.parse_args(argv)

    jobs = _cohort(args)
    out = args.out or HERE / "results" / f"{args.cohort}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pkg = Path(rx.__file__).resolve().parent
    source = hashlib.sha256()
    for path in sorted(pkg.rglob("*.py")):
        source.update(path.relative_to(pkg).as_posix().encode())
        source.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    header = {
        "cohort": args.cohort,
        "seeds": args.seed,
        "timeout": args.timeout,
        "argv": argv,
        "rxembed_version": rx.__version__,
        "rxembed_source": str(pkg),
        "rxembed_source_sha256": source.hexdigest(),
        "runner_sha256": RUNNER_SHA256,
        "rdkit_version": rdkit.__version__,
        "python_version": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "utc_start": datetime.now(UTC).isoformat(),
    }
    with out.open("x", newline="") as fh, tempfile.TemporaryDirectory(prefix="rxembed-bench-") as tmp:
        xyz_dir = out.with_name(f"{out.stem}-xyz") if args.keep_xyz else Path(tmp)
        xyz_dir.mkdir(exist_ok=True)
        fh.write("# " + json.dumps(header) + "\n")
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for structure_id, path, charge in jobs:
            for seed in args.seed:
                started = time.monotonic()
                row = _run_one(path, charge, seed, xyz_dir / f"{structure_id}-s{seed}", args.timeout)
                row |= {"id": structure_id, "seed": seed, "seconds": f"{time.monotonic() - started:.2f}"}
                writer.writerow(row)
                fh.flush()
                print(structure_id, seed, row["status"], row["stage"], row["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
