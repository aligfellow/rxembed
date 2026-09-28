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
import statistics
import sys
import tempfile
import time
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolAlign
from rdkit.SimDivFilters import rdSimDivPickers

import rxembed as rx

HERE = Path(__file__).resolve().parent
_MAX_SWAP_MATCHES = 5000  # cap on permutations searched across a structure's equivalent-site classes
_CARBORANE_BORONS = 5  # boron count at which a B/C row reads as a carborane cage, out of the tmQMg sample

FIELDS = ["id", "seed", "status", "stage", "detail", "seconds", "core_rmsd", "isomers", "embedded", "valid", "valid_cx"]


# --- fixture and tmQMg selection ---------------------------------------------


def _fixtures(only):
    """Return (id, xyz path, charge) for the shipped fixtures, or just `only`."""
    with (HERE / "fixtures.csv").open(newline="") as fh:
        charges = {row["id"]: int(row["charge"]) for row in csv.DictReader(fh)}
    ids = only or list(charges)
    missing = [i for i in ids if i not in charges]
    if missing:
        raise SystemExit(f"unknown ids: {' '.join(missing)}")
    return [(i, HERE / "fixtures" / f"{i}.xyz", charges[i]) for i in ids]


def _in_scope(row):
    """Exclude carborane-cage rows (>= 5 borons and any carbon) from the tmQMg sample."""
    mol = Chem.MolFromSmiles(row.get("smiles", ""), sanitize=False)
    if mol is None:
        return True
    counts = Counter(atom.GetAtomicNum() for atom in mol.GetAtoms())
    return not (counts[5] >= _CARBORANE_BORONS and counts[6])


def _fingerprint(row):
    """Build a metal/donor/ligand-graph fingerprint for MaxMin diversity picking."""
    tokens = [f"M:{row['metal_center']}", f"q:{row['charge']}"]
    ligand = None
    mol = Chem.MolFromSmiles(row["smiles"], sanitize=False) if row["smiles"] else None
    if mol is not None:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        z = Chem.GetPeriodicTable().GetAtomicNumber(row["metal_center"])
        metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == z]
        if len(metals) == 1:
            metal = metals[0]
            donors = [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()]
            rw = Chem.RWMol(mol)
            rw.RemoveAtom(metal)
            ligand = rw.GetMol()
            ligand.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(ligand)
            donors = {a - (a > metal) for a in donors}
            elements = Counter(ligand.GetAtomWithIdx(a).GetSymbol() for a in donors)
            denticity, carbon_sites = Counter(), Counter()
            for fragment in Chem.GetMolFrags(ligand):
                sites = donors & set(fragment)
                if sites:
                    denticity[len(sites)] += 1
                    carbon_sites[sum(ligand.GetAtomWithIdx(a).GetSymbol() == "C" for a in sites)] += 1
            tokens.append(f"CN:{len(donors)}")
            tokens += [f"D:{e}:{n}" for e, c in elements.items() for n in range(1, c + 1)]
            tokens += [f"dent:{v}:{n}" for v, c in denticity.items() for n in range(1, c + 1)]
            tokens += [f"Csites:{v}:{n}" for v, c in carbon_sites.items() for n in range(1, c + 1)]
    if ligand is None:
        tokens += ["missing-smiles", f"atoms:{int(row['n_atoms']) // 10}"]
    fp = DataStructs.ExplicitBitVect(128)
    if ligand is not None:
        ligand_fp = Chem.RDKFingerprint(ligand, minPath=1, maxPath=2, fpSize=32, nBitsPerHash=1)
        for bit in ligand_fp.GetOnBits():
            fp.SetBit(bit)
    for token in tokens:
        fp.SetBit(32 + zlib.crc32(token.encode()) % 96)
    return fp, tuple(sorted(tokens))


def _sample_tmqmg(rows, size):
    """Return `size` tmQMg IDs via a MaxMin diversity pick seeded by one typical structure per metal.

    Deterministic (seed=0): the same clone and size give the same IDs, and a smaller size is a prefix
    of a larger one, since each pick depends only on the picks made before it.
    """
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
    # LazyBitVectorPick never returns fewer than the one-per-metal seed picks, even for a smaller `size`.
    picks = list(picks)[:size]
    print(f"tmqmg: picked {len(picks)} of {len(rows)} in-scope structures in {time.monotonic() - started:.1f} s")
    return [rows[i]["id"] for i in picks]


def _tmqmg(size, only):
    """Return (id, xyz path, charge) for a MaxMin-diverse tmQMg sample, or just `only`."""
    data = os.environ.get("RXEMBED_TMQMG_DIR", "")
    xyz_dir = Path(data, "xyz")
    table = Path(data, "tmQMg_properties_and_targets.csv")
    if not data or not xyz_dir.is_dir():
        raise SystemExit("set RXEMBED_TMQMG_DIR to a tmQMg clone's data/ directory and unzip tmQMg_xyz.zip there")
    with table.open(newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if _in_scope(row)]
    charges = {row["id"]: int(row["charge"]) for row in rows}
    ids = only or _sample_tmqmg(rows, size)
    missing = [i for i in ids if i not in charges]
    if missing:
        raise SystemExit(f"unknown ids: {' '.join(missing)}")
    return [(i, xyz_dir / f"{i}.xyz", charges[i]) for i in ids]


# --- core RMSD (reported, never a gate) --------------------------------------


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


def _site_groups(iso):
    """One atom-index group per coordination site: the metal(s), then a lone donor or a whole haptic face."""
    return [[m] for m, _, _ in iso.metals] + [list(iso.haptic.get(v, [v])) for v in iso.vertices]


def core_rmsd(ref, mol, iso):
    """RMSD of the metal and site centroids, via RDKit's own automorphism search over equivalent sites.

    A haptic face scored per atom would read a free spin about the metal axis as error, so it is scored at
    its centroid (`Isomer.haptic`/`vertices`, both public). `CanonicalRankAtoms` finds which sites are
    graph-equivalent (e.g. two identical Cl); `GetBestRMS` then searches those permutations for the true
    minimum through an explicit atom map, which a fixed pairing cannot.
    """
    groups = _site_groups(iso)
    n = min(ref.GetNumAtoms(), mol.GetNumAtoms())
    if len(groups) < 2 or any(i >= n for g in groups for i in g):  # noqa: PLR2004
        return float("nan")
    graph = Chem.Mol(ref)
    for atom in graph.GetAtoms():
        atom.SetFormalCharge(0)  # a resonance-delocalised donor pair must not split into separate classes
        atom.SetNoImplicit(True)
    for bond in graph.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
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
        ][:_MAX_SWAP_MATCHES]

    def centroids(m):
        pos = m.GetConformer().GetPositions()
        return np.array([pos[g].mean(axis=0) for g in groups])

    target, probe = _positions_mol(centroids(ref)), _positions_mol(centroids(mol))
    k = len(groups)
    return rdMolAlign.GetBestRMS(probe, target, map=[[(order.get(i, i), i) for i in range(k)] for order in orders])


# --- the round trip -----------------------------------------------------------


def _heavy_bonds(mol):
    """Return each heavy-atom bond as an unordered index pair, ignoring bond order and zero-order NCI legs."""
    return {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
        if b.GetBondType() != Chem.BondType.ZERO
        and b.GetBeginAtom().GetAtomicNum() > 1
        and b.GetEndAtom().GetAtomicNum() > 1
    }


def _short(err):
    """Format an error (string or exception) as one line capped at 200 characters."""
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    return msg.replace("\n", " ")[:200]


def _validate(mol, ens, cid, expected_cx, charge, xyz_path):
    """Re-check one written candidate: same heavy-atom bonds on re-read, same requested CX, clean geometry.

    Return (ok, stage, detail, own_cx); an exception is labelled with the step that raised it.
    """
    step = "read"
    try:
        fresh = rx.read_xyz(str(xyz_path), charge=charge, connectivity="xyzgraph", bond_orders="xyz2mol")

        step = "connectivity"
        if _heavy_bonds(mol) != _heavy_bonds(fresh):
            return False, "validate:connectivity", "heavy-atom bonds changed across the XYZ round trip", ""

        step = "cx"
        own_cx = rx.cxsmiles(mol)
        if own_cx != expected_cx:
            return False, "validate:cx", "embedded CX != requested isomer CX", own_cx

        step = "geometry"
        report = ens.check()[cid]
        if not report:
            return False, "validate:geometry", _short(report.summary()), own_cx

        return True, "", "", own_cx
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
    """Return the structure's row: its reference isomer must be unique in the enumeration and pass every check."""
    row = _counts(candidates) | {"core_rmsd": ""}
    refs = [c for c in candidates if c["cx"] == reference_cx]
    if len(refs) != 1:
        detail = "reference CX matched twice" if refs else "reference CX absent from the enumeration"
        return row | {"status": "fail", "stage": "enumerate", "detail": detail}
    ref = refs[0]
    row["core_rmsd"] = ref["core_rmsd"]
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

        candidates = []
        for k, iso in enumerate(isos, start=1):
            stage = f"embed {k}/{len(isos)}"
            send.send(("stage", stage))
            cx = rx.cxsmiles(iso)
            c = {"k": k, "cx": cx, "valid": False, "stage": "embed", "error": "", "core_rmsd": ""}
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
                c["valid"], c["stage"], c["error"], _own_cx = _validate(mol, ens, cid, cx, charge, xyz_path)
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


# --- compare -------------------------------------------------------------------


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


# --- summary and CLI ------------------------------------------------------------


def _summarize(rows, seeds):
    """Print pass counts per seed, failures grouped by stage, and the ten slowest structure-seed runs."""
    for seed in seeds:
        at_seed = [r for r in rows if r["seed"] == seed]
        passed = sum(r["status"] == "pass" for r in at_seed)
        print(f"seed {seed}: {passed}/{len(at_seed)} pass")
    stages = Counter(r["stage"] for r in rows if r["status"] != "pass")
    for stage, n in stages.most_common():
        print(f"{n} failed at {stage or '(none)'}")
    for r in sorted(rows, key=lambda r: float(r["seconds"]), reverse=True)[:10]:
        print(f"{r['seconds']}s {r['id']} seed {r['seed']} {r['status']}")


def main(argv=None):
    """Run a cohort into one results CSV, or `compare BASE.csv NEW.csv`."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compare"]:
        if len(argv) != 3:  # noqa: PLR2004
            raise SystemExit("usage: run.py compare BASE.csv NEW.csv")
        return compare(Path(argv[1]), Path(argv[2]))

    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--only", nargs="+", metavar="ID", help="run just these IDs")
    common.add_argument("--seed", type=int, nargs="+", default=[42], help="rx.embed random seed(s)")
    common.add_argument("--timeout", type=int, default=600, help="seconds per structure and seed")
    common.add_argument("--out", type=Path, help="results CSV (default benchmark/results/<cohort>-<UTC>.csv)")
    common.add_argument("--keep-xyz", action="store_true", help="keep candidate XYZ files in <out stem>-xyz/")
    sub = parser.add_subparsers(dest="cohort", required=True)
    sub.add_parser("fixtures", parents=[common], help="the 100 shipped fixtures")
    tmqmg = sub.add_parser("tmqmg", parents=[common], help="a MaxMin-diverse sample of a local tmQMg clone")
    tmqmg.add_argument("--size", type=int, default=100, help="sample size (ignored with --only)")
    args = parser.parse_args(argv)

    jobs = _fixtures(args.only) if args.cohort == "fixtures" else _tmqmg(args.size, args.only)
    out = args.out or HERE / "results" / f"{args.cohort}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "rxembed_version": rx.__version__,
        "rdkit_version": rdkit.__version__,
        "seeds": args.seed,
        "timeout": args.timeout,
    }
    rows = []
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
                rows.append(row)
                print(structure_id, seed, row["status"], row["stage"], row["seconds"])
    _summarize(rows, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
