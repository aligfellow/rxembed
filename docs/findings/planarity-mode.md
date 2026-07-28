# The `planarity` regression the embed relax surfaces — diagnosed

**Companion to `seed-vs-relax-organic.md` §9**, which measured `planarity` violations rising **15 → 281**
after `rx.embed()`'s relax (driven by `cpa` 17/20 → 0/20 and `chb-tetramisole` 40/40 → 0/40, both with
conjugation flat at 0°) but explicitly did **not** diagnose the cause. This is that diagnosis. Measurement
only — no `src/` was changed. Date 2026-07-21, branch `rdkit-embed-kernel` at `4dee01b`. Scripts in
`playground/t3d_planarity/` (`t3d_diag.py`, `t3d_ff_compare.py`, `t3d_context.py`).

## Verdict in one line

**REAL geometric defect on both cases, not a gate false positive.** The dominant mode (≈256 of the 281) is
a **UFF force-field defect**: bare UFF pyramidalises the **thiourea / isothiourea (S–C(=N)–N) sp2 carbon**,
which MMFF94s, GFN-FF, GFN2 *and* the DFT reference all keep planar. `cpa` is a separate, marginal mode —
a **frozen-core-adjacency** distortion that grazes the 0.15 Å gate line. Both are the relax genuinely
distorting a centre the truth says is flat; the gate threshold is correctly calibrated (every reference
clears it). The root FF weakness is the *same one* the C=S conjugation mode has, seen from a different
angle — a soft coplanarity cap on the conjugated thio-carbon would address both.

---

## 1. What the `planarity` gate checks (quoted)

`src/rxembed/geometry.py::planarity` (lines 273–301), wired into `check()` at line 624 as
`v += planarity(mol, pos, exclude=exclude | coord_c)`:

```python
def planarity(mol, pos, oop: float = 0.15, ring_rms: float = 0.10, exclude=frozenset()) -> list[Violation]:
    """sp2 carbons stay planar and aromatic rings stay flat (broken-conjugation detector)."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != _CARBON_Z or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        if atom.GetIdx() in exclude:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) != _SP2_DEGREE:                     # exactly three neighbours
            continue
        off = _plane_offset(pos[atom.GetIdx()], pos[nbrs])   # centre's distance from its 3-neighbour plane
        if off > oop:                                    # oop = 0.15 Å
            out.append(Violation("planarity", (atom.GetIdx(), *nbrs), off, oop, "sp2 out of plane"))
    for ring in mol.GetRingInfo().AtomRings():
        if any(i in exclude for i in ring) or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        p = pos[list(ring)]
        rms = float(np.sqrt((np.linalg.svd(p - p.mean(0))[1][2] ** 2) / len(ring)))
        if rms > ring_rms:                               # ring_rms = 0.10 Å
            out.append(Violation("planarity", tuple(ring), rms, ring_rms, "aromatic ring puckered"))
    return out
```

Two sub-checks: (a) **per sp2 carbon** — a carbon (element 6) RDKit types `SP2` with exactly three
neighbours must sit within **0.15 Å** of the best-fit plane of those neighbours (`_plane_offset`, line 161;
0.15 Å ≈ a 6° improper for ~1.4 Å bonds); (b) **per aromatic ring** — the smallest-singular-value RMS out
of the ring's best-fit plane must stay under **0.10 Å**. Nitrogen is deliberately excluded (an amine N is
physically pyramidal); the twisted-amide signal is `conjugation()`'s job. **Every violation in these two
cases is kind (a), "sp2 out of plane"** — the actual `Violation` objects were printed, not inferred
(`t3d_diag.py`).

## 2. Which atoms trip it — named, seed vs relax vs threshold

`t3d_diag.py` snapshots the raw ETKDG seed (monkeypatched immediately before the real
`Ensemble._relax_into_windows`) and scores the *same* mol driven to seed then relaxed coords. Each case
reduces to **one** sp2 carbon:

| case | atom | sp2 neighbours | in ring? | seed off (Å) | relax off (Å) | thr |
|---|---|---|---|---|---|---|
| `chb-tetramisole` | **C3** — the isothiourea/amidine carbon | S2, N4, N7 | yes | **0.000** | **0.19–0.21** | 0.15 |
| `cpa` | **C73** — a vinyl sp2 carbon | C72, C74(=), C76 | no | 0.05–0.16 | **0.15–0.17** | 0.15 |

- `chb-tetramisole` C3 is `S–C(=N)–N` — the Lewis-basic **isothiourea carbon** of the tetramisole
  organocatalyst (the package's target motif). Seed **perfectly planar (0.000)**; the relax pyramidalises
  it to ~0.20 Å (≈8°), a clean, well-over-threshold excursion.
- `cpa` C73 sits **right on the 0.15 line** (0.151–0.168), a ≈6° tilt. Its neighbour **C74 is a frozen-core
  atom** (`fix=[23,35,70,74,75]`; C73 dangles off the pinned core — see §4).

Seed-independent: both flag **100% of conformers on all five seeds** (1, 2, 3, 7, 0xF00D), in a tight band
(`t3d_context.py` §3). So 15 → 281 is not a few-seed artefact.

## 3. Real defect or gate false positive — the decisive ablation

`t3d_ff_compare.py` takes the planar seed and relaxes it several ways, measuring the same sp2 off-plane. If
bare UFF pyramidalises but a proper FF / real energy stays flat, the flat geometry is the true minimum and
the relax is wrong (gate right); if a real energy *also* pyramidalises, the gate is a false positive.

**`chb-tetramisole` C3 (S,N,N):**

| method | off-plane (Å) | pyr (°) | |
|---|---|---|---|
| seed | 0.000 | 0.0 | |
| embed relax (shipped) | 0.202 | 8.0 | **FLAGGED** |
| **uff_plain** (no constraints) | **0.174** | 6.9 | **FLAGGED — bare UFF alone puckers it** |
| mmff94s (no constraints) | 0.008 | 0.3 | flat |
| restrained_uff (real constraints) | 0.202 | 8.0 | FLAGGED |
| **gfnff_opt** (real energy) | 0.030 | 1.2 | flat |
| **gfn2_opt** (real energy) | 0.008 | 0.3 | flat |

Bare UFF pyramidalises the isothiourea carbon on its own; MMFF94s, GFN-FF and GFN2 all keep it planar. The
constraints add only ~0.03 Å more. → **UFF force-field defect. REAL. The gate is correct.**

**`cpa` C73 (C,C,C — vinyl):**

| method | off-plane (Å) | pyr (°) | |
|---|---|---|---|
| **DFT REFERENCE** (cpa.xyz) | **0.010** | 0.4 | **flat — ground truth** |
| seed | 0.051 | 2.0 | |
| embed relax (shipped) | 0.151 | 6.0 | FLAGGED |
| **uff_plain** (no constraints) | **0.018** | 0.7 | **flat — bare UFF keeps it planar** |
| mmff94s | 0.012 | 0.5 | flat |
| restrained_uff (real constraints) | 0.151 | 6.0 | FLAGGED |
| gfnff_opt | 0.021 | 0.8 | flat |
| gfn2_opt | 0.018 | 0.7 | flat |

Here bare UFF, MMFF, both xtb methods **and the DFT reference (0.010)** all keep C73 planar. Only the
**constrained** relax (0.151) puckers it, grazing the gate line. → **REAL but marginal, and a different
mechanism** (constraints, not bare UFF).

Neither is a gate false positive: on both, at least two independent real energies — and for `cpa` the DFT
reference itself — say the centre is planar, while the relaxed geometry is not. The threshold (0.15 Å) is
sound: every reference clears it; it is the UFF relax that fails.

## 4. Mechanism

**Mode A — the bulk (≈256 of 281): a UFF sp2-inversion defect on the thio-carbon.** The restrained relax
is plain **UFF** (`refine/ff.py::restrained_uff` → `rdForceFieldHelpers.UFFGetMoleculeForceField`), whose
out-of-plane (inversion) term on an sp2 carbon bearing an **S substituent in a C=N / C–N π system** — the
thiourea `S=C(–N)(–N)` and isothiourea/amidine `N–C(=N)–S` carbon — is too weak to hold it flat against the
surrounding angle/torsion terms. The bulk contributor to the 281 is **`bimp-smiles-auto`** (the study's
0 → 216): its top flagged sp2 class is **`NNS` = 45** (the thiourea carbon), dwarfing all others (CCH 4,
CCN 3, CNO 3, CCC 1, one puckered ring). Ablation on one `bimp` thiourea carbon (atom 9): seed 0.003 →
uff_plain **0.095** → restrained **0.377**, vs mmff94s 0.008 and gfnff 0.005 (`t3d_context.py` §2). So bare
UFF puckers it and the constrained relax amplifies it far past threshold, while a proper FF / real energy
keeps it flat — the same signature as `chb-tetramisole`. **`chb-tetramisole` (isothiourea) +
`bimp-smiles-auto` (thiourea) ≈ 256 of the 281 violations, one mechanism.**

This is the **same underlying UFF weakness as the C=S conjugation mode** (`seed-vs-relax-organic.md` §6.1):
UFF has no proper thiocarbonyl restraint, so the thio-carbon centre is under-restrained. The conjugation
mode sees it as the `C–N` *torsion* twisting; this planarity mode sees it as the *carbon itself*
pyramidalising. Same defect, two projections.

**Mode B — `cpa` (≈20 of 281): a frozen-core-adjacency distortion.** `cpa`'s constraints are the 5-atom
frozen core `[23,35,70,74,75]` plus its 10 core-internal distance windows (C(5,2)=10; none touches atom 73).
C73's neighbour **C74 is frozen** (pinned to the DFT reference). When the free periphery relaxes to UFF's
minimum, that minimum cannot match the DFT core, and the mismatch transmitted through the pinned C73–C74
bond tilts C73's local plane to ~6°, landing right on the 0.15 line. Bare UFF (no pins) keeps it planar
(0.018); the DFT reference is planar (0.010). It is a real distortion the *constrained* relax introduces at
the free/frozen junction — small, threshold-grazing, and unrelated to the thio-carbon defect.

## 5. Where a fix would go, and how big

- **Mode A (primary, real work).** Add a soft sp2 out-of-plane / improper restraint on the perceived
  conjugated thio-carbon (or on conjugated sp2 carbons generally) to the restrained-UFF build. It belongs
  in the mechanism registry `restrained_uff` walks — a new `ff_terms` writer in
  `src/rxembed/constraints/mechanisms.py` fed by a `Constraints` field (built via a perception helper in
  `constraints/builders.py`), directly analogous to the existing `constraints/metal._coplanar_donor` soft
  dihedral cap. **Best unified with the pending C=S coplanarity cap** the organic study proposed — both are
  the same thio-carbon FF weakness, and one "hold this conjugated sp2 centre and its substituents coplanar"
  cap covers the torsion (conjugation) *and* the improper (planarity). Size: a real piece of work — a new
  FF term + perception + a `Constraints` field + a regression test, comparable to the C=S cap. **Not a
  one-liner.**
- **Broader alternative (not recommended first).** `restrained_uff` hardcodes UFF; MMFF94s keeps every one
  of these planar (0.008). Switching the organic relax to MMFF-where-typeable would fix it, but it touches
  the golden bit-identity harness and the metal surrogate path (which needs the UFF surrogate). Larger blast
  radius.
- **Mode B (`cpa`, optional, marginal).** Not a gate false positive (DFT ref 0.010 vs relax 0.151, a real
  6° distortion). If a cheap mitigation is wanted, `geometry.py::check` could widen the window for — or add
  to `exclude` — sp2 carbons *directly bonded to a frozen-core atom* (they inherit the core's geometry
  mismatch): a small change in `check`/`planarity`. But it masks a real distortion, so it is a judgment call,
  not a clear fix; the Mode-A cap would also pull C73 partway back.

## 6. What could not be established / caveats

- The **UFF internal cause** (which inversion force constant / atom type) was not inspected in RDKit's UFF
  source; the defect is established by *ablation* (UFF vs MMFF vs GFN, constraints held out), not by reading
  the UFF term. That is sufficient to attribute it to UFF, not to identify the exact parameter.
- The 281 breakdown (216 `bimp` + ~40 `chb` + ~20 `cpa` + a scatter of CCH/CCN/CNO) is measured at seed 1
  for `bimp-smiles-auto`; the two named cases are confirmed across five seeds. The minor classes (CCH etc.)
  were not individually ablated — they are a small tail.
- **Whether the unified C=S/planarity cap actually fixes both** is a proposal from the shared mechanism, not
  a tested result. It should be implemented and regressed before "done".
