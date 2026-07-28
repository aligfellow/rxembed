# Coplanar-cap architecture — the structural rule that replaces the crowding count

Follow-up to `coplanar-cap-softening.md` (whose crowding-conditional FC the maintainer **rejected** as a
heuristic branch) and `karoline-metal-embed.md` (cases 1–4). The maintainer's steer: *"conditional is not a
clean embedding rule; if that's interfering there's probably a better setup of constraints."* This finding
derives that setup — a structural predicate, not a count — measures it, and shows it **beats the rejected
conditional** while leaving every control bit-identical.

> **Bottom line — the principle.** The cap restores exactly one thing: the improper the stripped M–D bond took
> away, which held the metal in the donor's sp2 plane. But **a plane is pinned to the metal by TWO coplanar
> contacts, never one** — two M–D distances plus the L–M–L angle between them, both of which the polyhedron
> *already* imposes. So the improper is only real information when the metal meets the donor's plane at a
> **single point**. When a **second donor of the same metal lies in that same conjugated sp2 plane** (a
> conjugated bidentate — pyridylimine, picolinate, acac), the pair already pins the metal in the plane and the
> cap merely restates a constraint the distances + bite angle make. So:
>
> **Cap an sp2 donor's plane only when the metal contacts that plane at one point. Skip it when a co-donor of
> the same metal lies in the same conjugated sp2 plane — i.e. is reachable through a path of only sp2 atoms.**
>
> That is general chemical reasoning about the geometry set we already apply (two-point plane pinning + our own
> L–M–L angles), not a crowding count and not an element list. Measured (FF-only, golden-safe): **case2
> 59%→22%, case3 65%→40%** (the rejected conditional got 40%/52%), controls HENRY/KETONE **bit-identical**,
> PICO still held at ~2° by its backbone, the RDP FC-pin test 6/6. Spikes in `playground/cap_architecture/`;
> `src/` untouched (`git diff src/` empty).

---

## 1. The constraint stack on one donor of each type (`caparch_map.py`)

Every capped donor is `sp2` in-plane (`geometry.inplane_sp2_donor`). Four things can constrain its **in-plane
roll about the M–D axis** — the DOF the cap targets:

| constraint | what it fixes | DOF |
|---|---|---|
| **coplanarity cap** (`_coplanar_donor` → `cons.coplanar`; DG `Coplanar.dg_post` + FF `Coplanar.ff_terms`) | metal in donor's own sp2 plane (dihedral M–D–C–X) | **the roll** — directly |
| **polyhedron angles** (`metal.ANGLES`, square_planar 0-2-180 / 1-3-180 / 0-1-90) | donor **positions** / coordination plane (L–M–L angles) | donor placement, **not** the roll |
| **fold-wall / `_orient_donor`** (M–D–X angle wall) | donor's substituents splayed off M (in-plane/axial) | a **different** angle, not the roll |
| **backbone bonds + UFF's own ligand torsions** | the ligand's internal planarity/conjugation | fixes the roll **only if** a rigid path ties the donor to a co-donor bonded to M |

The separator between the controls is purely backbone topology: **KETONE**'s capped donors (ketone O, pyridine
N) reach **no** co-donor through the backbone (monodentate, two separate ligands); **HENRY/PICO/cases 1–4**
all chelate. So "chelate vs monodentate" alone cannot separate control from problem (HENRY & PICO chelate and
are controls). The real separator is whether the chelate path is **rigid** (below).

## 2. The located conflict — cap-vs-backbone (`caparch_redundancy.py`)

Per-donor **metal-out-of-donor-plane** deviation (median/max, deg) post-`minimize`, seeds `1,7,13,21,0xF00D`,
n=8, isomer 0, three regimes: **BASE** (all caps) / **SELF-off** (only this donor's cap removed) / **ALL-off**.
`Δself = SELF-off − BASE` median. Removal patches **both** halves of the cap (DG + FF). (Guard: BASE reproduces
the established `capsweep` baseline exactly — HENRY O 0.6°, N 6.1° — under the same seeds; the metric is
bimodal, mass near 0° with a tail to the census p95 ~40°, so medians need the 5-seed set.)

| complex | donor | BASE med/max | SELF-off med | Δself | verdict |
|---|---|---|---|---|---|
| **PICO** (rigid picolinate) | O2 | 1.4 / 19.6 | 2.3 | +0.9 | **REDUNDANT** |
| | N8 | 0.7 / 15.4 | 0.6 | −0.0 | **REDUNDANT** |
| **KETONE** (monodentate) | O3 | 22.0 / 35.3 | 46.9 | **+24.9** | LOAD-BEARING |
| | N7 | 0.3 / 42.3 | 50.0 | **+49.7** | LOAD-BEARING |
| **HENRY** (sp3-Cα chelate) | O13 | 7.2 / 39.5 | 23.7 | **+16.5** | LOAD-BEARING |
| | N23 | 8.7 / 42.3 | 16.2 | +7.5 | partial |
| **case2** | N15 (pyridyl) | 14.3 / 41.0 | 21.7 | +7.4 | partial→redundant |
| | N34 (imine) | 6.3 / 25.2 | 4.7 | −1.6 | **REDUNDANT** |
| | O17 (carboxylate) | 4.7 / 43.7 | 18.5 | **+13.9** | LOAD-BEARING |
| | N27 (anilide) | 15.2 / 42.3 | 26.8 | +11.7 | partial |
| **case3** | N15 (pyridyl) | 15.6 / 39.6 | 25.0 | +9.4 | partial |
| | N34 (imine) | 8.2 / 40.6 | 10.0 | +1.8 | **REDUNDANT** |
| | O17 (carboxylate) | 24.1 / 42.7 | 18.1 | −6.0 | **REDUNDANT** |
| **case4** | O4 (ester) | 3.5 / 26.1 | 2.2 | −1.3 | **REDUNDANT** |
| | N23 (amidate) | 2.4 / 23.2 | 1.9 | −0.5 | **REDUNDANT** |
| | O6 (amidate) | 18.2 / 35.5 | 11.2 | −7.0 | **REDUNDANT** |
| **case1** | S6 (thione) | 9.3 / 43.8 | 10.6 | +1.3 | REDUNDANT |
| | O8 | 1.1 / 38.7 | 25.6 | **+24.5** | LOAD-BEARING |
| | N18 | 6.6 / 42.3 | 10.9 | +4.3 | REDUNDANT |

**It is cap-vs-backbone, not cap-vs-cap or cap-vs-polyhedron.** For a donor rigidly locked in a conjugated
chelate, removing its **own** cap barely moves its plane (PICO 1–2°, case4 all three, case2/3 imine N34) — and
several *improve* (case4 N23 max 30→8, O6 −7.0°, case3 O17 −6.0°): the backbone + polyhedron + the ligand's own
UFF torsions already hold it. At full cap stiffness these redundant caps each pull the metal toward their own
slightly-tilted local plane; on a crowded chelate those planes cannot all contain the metal, so the FF twists
the conjugated backbone to compromise — exactly the `planarity`/`conjugation` flags. The polyhedron angles are
NOT the conflict: they constrain donor *positions* (the coordination plane), a different DOF from the roll.

## 3. Redundancy on backbone-locked donors — **YES, decisively**

A donor **rigidly locked in a conjugated chelate** keeps its plane WITHOUT its cap (PICO, case4, case2/3
imine): SELF-off ≈ BASE, and ALL-off stays low too (PICO 1–2°). A **monodentate** donor (KETONE, 0.3°→50°) or
a donor chelating only through a **flexible sp3 hinge** (HENRY carboxylate 7°→24°; case2 amino-carboxylate O17
5°→19°) drifts hard when its cap is removed — the cap is genuinely load-bearing there. The `±45°/1.0°-median`
prior claim that "the cap keeps henry in plane" is confirmed as a **per-donor** truth (HENRY O Δself +16.5),
*not* a blanket one.

## 4. The derived rule — from the geometry we already apply (NOT a count)

The reasoning is entirely about the constraint set we already build, plus one geometric fact:

1. **What the cap is.** The surrogate strips the M–D bond and with it the UFF improper that held the metal in
   the donor's sp2 plane. The cap (`cons.coplanar`) restores *only* that improper — nothing more.
2. **A plane needs two contacts.** A single M–D contact leaves one rotational DOF: the metal can swing around
   the M–D axis, in or out of the donor's plane (the roll). Fixing that needs a **second** point of the plane
   tied to the metal. We already impose exactly such a pair for every donor pair: **two M–D distances** and the
   **L–M–L angle** (the chelate bite for one ligand, `metal.ANGLES`/±8° for two). Two coplanar contacts + their
   subtended angle pin the metal into the plane.
3. **So the cap is the one-contact special case.** It is real information only when the donor's plane meets the
   metal at a single point. When a **second donor of the same metal lies in that same plane**, the pair already
   pins it and the improper is a redundant restatement.

> **Cap an sp2 donor only when the metal contacts its plane at one point. Skip it when a co-donor of the same
> metal lies in the same conjugated sp2 plane — operationally, a co-donor reachable through a path of only
> sp2 atoms.**

"Same conjugated sp2 plane" is the general chemical description of a **conjugated bidentate** (pyridylimine,
picolinate, α-diimine, acac): its two donors are coplanar by conjugation, so the ligand presents the metal a
ready-made plane and the bite angle we already apply pins the metal in it. A **monodentate** donor (KETONE) or
a chelate whose second donor is out of the first's plane across an **sp3 hinge** (HENRY / case2 amino-carboxylate:
the path hits an sp3 Cα) presents only one coplanar contact — the cap is the sole information and is kept.

**The partially-conjugated chelate `M–sp2–sp2–sp3–…–M` — the cap stays, correctly.** The predicate keys on the
**second donor's coplanarity, not on the donor's own local sp2-ness.** A donor is sp2 with an sp2 neighbour
(its own π-plane is real and rigid), but if the path to the *other* donor crosses an sp3 hinge, that second
donor is NOT in this donor's plane, so it gives no second in-plane contact — one contact, cap kept. This is not
hypothetical: **HENRY's carboxylate is exactly this case** — the path O(sp2)–C(sp2)–Cα(**sp3**)–N(sp2), so the
predicate keeps its cap (skip set `[]`), and §2 measured that cap **load-bearing** (Δself +16.5°). The metal is
held in the carboxylate's sp2 plane by the kept cap, precisely as it should be. Requiring the **whole** path to
be sp2 before skipping is what makes this fall out correctly — the local sp2 unit near the donor is never enough
to skip; only a genuinely coplanar second donor is.

The detection is one predicate, element-agnostic, reading only hybridisation (no element list, no π-flag —
RDKit mis-marks isolated C=X, so we ask "are all path atoms sp2?" directly). It extends the existing
`inplane_sp2_donor` gate with one more clause of the same shape. **The atom-based form ("all path atoms sp2")
and the equivalent bond-based form ("every bond aromatic/double/sp2–sp2-single") give the identical skip set
on all 8 complexes — 0 mismatches** (`caparch_spike.plane_locked` used the bond form; the atom form is the
cleaner statement). Verified skip set (matches §2 ground truth):

| complex | capped | **skipped (plane-locked)** | kept | matches redundancy? |
|---|---|---|---|---|
| KETONE | 3,7 | — (monodentate) | 3,7 | ✓ both load-bearing |
| HENRY | 13,23 | — (sp3-Cα hinge) | 13,23 | ✓ load-bearing/partial |
| PICO | 2,8 | **2,8** (rigid picolinate) | — | ✓ both redundant |
| acac_ni (golden) | 5,9 | **5,9** (rigid) | — | rigid conjugated chelate |
| case2 | 15,34,17,27 | **15,34** (pyridylimine) | 17,27 | ✓ imine redundant, O17 load-bearing |
| case3 | 15,34,17 | **15,34** | 17 | ✓ |
| case4 | 4,23,6 | **4,23** | 6 | ✓ (O6 also redundant → kept is just conservative) |
| case1 | 6,8,18 | — | 6,8,18 | conservative; O8 is load-bearing |

**Key safety property (proven by §2 + §5):** the predicate **never strips a load-bearing cap.** Every donor it
skips is redundant/partial; every load-bearing donor (KETONE O/N, HENRY O, case2 O17, case1 O8) is kept. It is
conservative in the safe direction — it keeps a few redundant caps (case4 O6, case3 O17, case1 S6) with no
harm.

## 5. The spike, measured (`caparch_spike.py`, seeds `1,7,13,21` pooled, n=8)

Two placements: **ff** = skip only the FF torsion for plane-locked donors (touches `Coplanar.ff_terms` only →
**golden unchanged**); **both** = skip the whole cap at the builder (`_coplanar_donor` → no DG bound, no FF term
→ **moves golden**). Null guard: skip counter fired (ff: 2736 skipped / 3664 kept across the run).

| case | capped→kept | **baseline** | **FF-only rule** | **both rule** | rejected conditional |
|---|---|---|---|---|---|
| **case2** | 4→2 | 59.0% (pl43,cj28) | **21.9%** (pl16,cj12) | 21.9% | 40% |
| **case3** | 3→1 | 64.8% (cj34,pl42) | **39.6%** (cj22,pl10) | 39.6% | 52% |
| **case4** flag | 3→1 | 0% | 0% | 0% | — |
| **case4** ester collapse (iso3, 40 seeds) | | 0/40 | 0/40 | 0/40 | 0/40 |
| **case1** (thione, cosmetic) | 3→3 | 0% | 0% | 0% | — |
| **HENRY** hold (O/N med, deg) | kept | 0.6 / 6.1 | **0.6 / 6.1** (identical) | 0.6 / 6.1 | — |
| **KETONE** hold | kept | 25.2 / 0.3 | **25.2 / 0.3** (identical) | 25.2 / 0.3 | — |
| **PICO** hold | removed | 1.3 / 0.7 | **2.4 / 1.1** (still held) | 3.1 / 1.1 | — |
| **RDP** FC-pin test | kept (skip=0) | 6/6 clean | **6/6** | 6/6 | 6/6 |

Per-seed (seed-insensitive): FF-only case2 = 25/25/19/19%, case3 = 36/33/40/47% (baseline case2 57/62/53/62%,
case3 62/64/62/71%).

Reading it:
- **Beats the rejected conditional** on both cases (case2 22% vs 40%, case3 40% vs 52%) with a clean predicate.
- **Controls bit-identical.** HENRY & KETONE keep every cap (load-bearing) → hold unchanged. PICO's redundant
  caps are removed yet it stays at ~2° — its rigid backbone holds it, as §2 predicted.
- **RDP FC-pin test untouched** — the predicate never fires on it (all caps kept), so the exact control a
  *global* FC-drop breaks (4/6) is left at 6/6. This is why the structural rule succeeds where the global lever
  failed.
- **case4 needs no fix now** — the ester collapse is already 0/40 (session mitigations + the silent-fusion gate
  fix `8338de1`); the rule keeps it at 0/40 and drops case4's redundant amidate caps for free.
- **FF ≡ both on every flag rate** → the DG seed bound is *also* redundant on plane-locked donors (removing it
  changes nothing). No case2/case3 DG tension appears (unlike the prior *global* crowded-DG-off), because the
  structural predicate removes the DG bound only where it is redundant.

## Recommendation

**Skip the cap on a donor whose plane a co-donor already shares** — a deletion, not an addition: fewer FF
torsions, one clause on the existing `inplane_sp2_donor` gate, grounded in the geometry set we already apply
(two coplanar contacts + our own L–M–L angle pin a plane; the improper only adds information at one contact).
Not a count, not an element list. The predicate:

```python
# in geometry, beside inplane_sp2_donor: a co-donor of the same metal lies in this donor's sp2 plane
def codonor_in_plane(mol, d, donors):
    return any(dd != d and (p := Chem.GetShortestPath(mol, d, dd)) and len(p) >= 2
               and all(hyb[a] == SP2 for a in p) for dd in donors)
```

**Architecture: full embedding in the DG, then the FF after — never embed-separate-and-graft.** The coplanarity
is not a graft. The DG **keeps its coplanarity bound for every sp2 donor** (`Coplanar.dg_post`), and it
**composes in the one bounds matrix**: for a conjugated bidentate the co-donor is bonded to M and tied to this
donor's reference atom through the rigid backbone, so triangle-smoothing already fixes that 1,4 distance — the
cap's bound intersects it and is subsumed (the existing INTERSECT discipline, `mechanisms.Angle`).
**Measured** (instrumenting `dg_post`, case2): the pre-existing matrix window on the cap's M···w pair is
**0.6 Å for the conjugated-bidentate donors N15/N34** (the backbone has already pinned it — cap subsumed) but
**effectively unbounded, 3.2 → 1000 Å, for the one-contact carboxylate O17** (the matrix says nothing — the cap
is the sole DG information, i.e. genuinely load-bearing). The only thing that does NOT compose is the **FF
torsion**, because independent per-donor torsions *add* rather than intersect, so on a crowded chelate they
fight (§2). So the fix lives on the FF side, after the full DG embed:

1. **Recommended: FF-only.** In `Coplanar.ff_terms`, skip the torsion for a donor whose plane a co-donor
   already shares. The DG still embeds the metal coplanar for *all* donors (composed, one matrix); the FF
   refines only where the improper is the sole information (one-contact donors). Touches no DG bound →
   `tests/golden` unchanged; full case2 22% / case3 40%. **Measured: keeping vs dropping the DG bound for these
   donors changes nothing (FF ≡ both)** — direct evidence the DG half already composed and the FF was the
   fighting term.
2. **Optional: also gate the DG at the builder `_coplanar_donor`** (never create the tuple). Purely a
   tidy-up — removes a bound the matrix already subsumes. **Moves golden** for `acac_ni` and any
   conjugated-bidentate fixture; `henry_ni` does not move. Not needed for the fix; only take it if the
   redundant DG bound should be gone on principle.

The chemistry is one rule at both stages: **a conjugated bidentate hands the metal a plane the bite angle
already pins, so it needs no per-donor improper; a monodentate or sp3-hinged donor hands over one contact and
keeps it.** The graph answers "does a co-donor share this plane?" for every element and coordination number at
once — no count, no graft, one embedding.

### Provenance

All numbers are monkeypatch measurements on the installed env; `src/` untouched (`git diff src/` empty).
Scripts + raw outputs in `playground/cap_architecture/`:
`caparch_map.py` (§1), `caparch_redundancy.py` + `rr_*.txt` (§2 — the 5-seed clean run; `out_*.txt` are the
superseded 2-seed noisy first pass), `caparch_rigidity.py` (§4 predicate derivation; the atom-based ≡ bond-based
equivalence is checked inline in the same session), `caparch_dgwindow.py` (the DG-subsumption evidence for the
architecture recommendation), `caparch_spike.py` + `spike_{base,ff,both}.txt` (§5), `caparch_epoxide.py`
(case4 collapse).
Absolute flag rates track `coplanar-cap-softening.md`'s baseline (59%/65% here vs its 55%/65%).
```
uv run --no-sync python playground/cap_architecture/caparch_redundancy.py case2 case4
uv run --no-sync python playground/cap_architecture/caparch_spike.py ff
```
