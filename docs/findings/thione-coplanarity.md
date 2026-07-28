# Thione S=C coplanarity with the coordination plane (case1) — is it ensured?

**Question (maintainer).** Does the current constraint setup ensure the thiourea/thiosemicarbazone
**S donor's S=C is coplanar with the S–M bond and the other M–D bonds** (lies in the coordination plane)?
Recent work made `Coplanar.ff_terms` skip a donor's coplanarity cap when
`geometry.codonor_in_plane(mol, d, donors, hyb)` is True — a co-donor of the same metal reachable through an
all-sp2 backbone path (a conjugated bidentate the polyhedron's two contacts + bite already pin). Is the thione
in that skip set, and is its S=C actually held in the coordination plane?

**Molecule (case1).** `C[N]1(C)NC(N)=[S]->[Ni+2]<-12<-[O-]C(=O)C(c1ccccc1)[N-]->2c1ccccc1` — a square-planar
Ni(II) complex. Two ligands: an **N,N-dimethyl thiosemicarbazide / aminothiourea** (chelating N,S) and an
**amidate·carboxylate** (chelating N,O).

## Verdict — (b): genuine one-contact donor, coplanarity NOT ensured, and that is physically expected

The thione S is a **KEEP** case (`codonor_in_plane → False`), and that is **correct**, not a rule gap. Its only
backbone co-donor is a **genuine sp3** nitrogen (the coordinating dimethylamino N), so the S chelates through a
real sp3 hinge — like HENRY's sp3-Cα carboxylate. The S=C therefore rides the ±45° census window and **scatters**:
measured out-of-plane **median ~8–9°, max ~44°** (matches the ~17–44° spread the maintainer saw). The question
"is the S=C ensured coplanar?" is **NO** — correctly. There is no missed co-donor, so it is **not** verdict (c).

## 1. The donor set and the S's co-donor paths (the crux)

`rx.metal(SMI, "square_planar")[0]`, surrogate metal in place, `geometry._stripped_hybridisation`
(`playground/thione_diag/thione_paths.py`):

```
metal = Ni (idx 7)      donors = [1, 6, 8, 18]
  D=  1  N  hyb=SP3  deg=3   <- coordinating dimethylamino/hydrazino N (thiosemicarbazide arm)
  D=  6  S  hyb=SP2  deg=1   <- the thione C=S sulfur
  D=  8  O  hyb=SP2  deg=1   <- carboxylate O (other ligand)
  D= 18  N  hyb=SP2  deg=2   <- amidate N (other ligand)
capped donors = [6, 8, 18]   (D1 sp3 → no cap; not inplane_sp2)
```

Shortest paths from S6 to each co-donor (metal stripped, per-atom stripped hyb):

```
S6 -> D1 (N):  all-sp2 = False   path: 6S[SP2] - 4C[SP2] - 3N[SP2] - 1N[SP3]   <- crosses the sp3 amino N
S6 -> D8 (O):  NO backbone path  (separate ligand — connects only through the metal)
S6 -> D18(N):  NO backbone path  (separate ligand — connects only through the metal)
```

`codonor_in_plane(mol, 6, donors) → False` ⇒ **cap KEPT** on the thione.

**Why the path is correctly not-all-sp2.** The coordinating nitrogen N1 is a **tertiary amine**: three single
bonds (two methyl C, one to N3), no double bond, formal charge 0, no H (`thione_paths` + bond dump). Both
estimators agree it is sp3 (`_stripped_hybridisation` emits a class only on agreement), so this is **not a
mis-perception**. The thioamide unit S6=C4(–N3)(–N5) is planar/conjugated, but N1 hangs off N3 by an N–N single
bond and is genuinely pyramidal — it is **not** in the thione's π-plane. So the chelate presents the metal only
**one** coplanar contact (the S itself); the cap is the sole in-plane information and is correctly kept.

**Not a thiosemicarbazone azomethine.** A classic thiosemicarbazone `R2C=N–NH–C(=S)–NR2` chelates through the
**azomethine** N (=N–, sp2, conjugated) — that path *S=C–N–N=C* is all-sp2 and *would* correctly skip. This
SMILES has **no C=N**; it binds through the amino/hydrazino terminus, which is sp3. So the task's (c) scenario
(a hydrazine N mis-perceived sp3 where an imine co-donor really lies in the plane) **does not apply here** — there
is no imine co-donor, and N1 is genuinely sp3.

## 2. Measured S=C-vs-coordination-plane angle (`playground/thione_diag/thione_angle.py`)

Isomer 0, n=8 conformers/seed, seeds `1,7,13,21,0xF00D`, `rx.embed(iso).minimize()`. Three equivalent measures
(pure Cartesian, element-independent):
- **oop_S** — metal out of the thione's own sp2 plane (S6/C4/N3), the cap's DOF;
- **dih_dev** — |Ni–S6–C4–N3 dihedral − nearest well (0/180)|, the cap target;
- **sc_coord** — elevation of the S=C4 bond out of the best-fit plane through {Ni, N1, S6, O8, N18} (the
  maintainer's exact question). 0° = coplanar.

```
                oop_S            dih_dev          sc_coord   (deg, per-seed median/max)
  seed      1:  11.9/43.8        11.9/44.0        15.7/44.4
  seed      7:   2.0/28.2         2.0/28.3         5.8/41.1
  seed     13:  10.0/23.5        10.0/23.6         8.6/25.8
  seed     21:   6.5/24.3         6.6/24.3         4.0/27.3
  seed  0xF00D:  9.3/26.5         9.4/26.6         7.9/27.1

  pooled (n=40):  oop_S  median 9.3  mean 11.0  max 43.8
                  sc_coord median 8.1  mean 12.6  max 44.4
```

**Seed sensitivity:** per-seed medians `sc_coord = [15.7, 5.8, 8.6, 4.0, 7.9]` (spread 11.7°), `oop_S =
[11.9, 2.0, 10.0, 6.5, 9.3]` (spread 9.9°). The median is seed-sensitive (single-digit to mid-teens) but every
seed shows the same shape — a median in the single-digit-to-teens and a tail to ~25–44°. So coplanarity is **not
ensured**; the S=C scatters across the full ±45° window.

## 3. The cap is nearly redundant for the thione anyway (`caparch_redundancy.py case1`, reproduced)

Removing S6's *own* cap (both DG + FF halves) barely moves its plane — the thioamide backbone + polyhedron
already hold it roughly, and nothing pins it tightly:

```
   donor      BASE(cap on)   SELF-off       ALL-off       verdict
   D=6 (S)     9.3/43.8       10.6/38.9      10.8/43.1     Δself +1.3  REDUNDANT
   D=8 (O)     1.1/38.7       25.6/35.9      24.3/39.3     Δself+24.5  load-bearing
   D=18(N)     6.6/42.3       10.9/50.7       8.9/51.2     Δself +4.3  REDUNDANT
```

Two consequences:
1. The thione's ~9°/44° scatter is **intrinsic to the ligand**, not a cap failure — cap-on and cap-off give the
   same distribution (9.3 vs 10.6 median). Nothing the cap can do would tighten it, because there is no rigid
   all-sp2 backbone tying the S plane to a co-donor.
2. The KEEP is *conservative-safe*: the cap is redundant here (co-donor N1 out of plane), so keeping it costs
   nothing and the skip-rule (which keeps it) is right by the safe-direction property. Contrast **D8** (the
   carboxylate O): its cap is genuinely load-bearing (Δself +24.5°, the HENRY-carboxylate analogue) and is also
   correctly kept.

## 4. Is the census ±45° width defensible for a thione?

`CENSUS_OOP_P95 = 40°`, `_COPLANAR_CAP = 45°` — a **global** p95 of the metal-out-of-conjugated-donor-plane
angle over the tmQM/Kulik crystals (the thione is in that pool), **not** a thione-specific number. For **this**
ligand it is defensible: the chelate closes through a genuine sp3 amino hinge (N1) and an N–N single bond, so the
S=C orientation relative to the coordination plane is a real conformational DOF, not a conjugation-locked one.
The maintainer's implicit expectation of PICO-tight coplarity (~2°) has **no chemical grounds here**: PICO is a
rigid, fully-conjugated picolinate (all-sp2 O–C···N backbone, a true conjugated bidentate that hands the metal a
ready-made plane the bite angle pins). Case1's thiosemicarbazide has no such backbone — its two donors are an sp2
thione and an **sp3** amine, so neither the cap nor the backbone can (or should) pin the S flatter than the ~9°
median / ~44° tail the census window allows.

## Bottom line

The thione S is a **KEEP** case and correctly so: its only backbone co-donor is a genuine sp3 amine, so
`codonor_in_plane` returns False by design (not a missed co-donor). The S=C is held only by the ±45° census
window and scatters — median ~8–9°, max ~44° out of the coordination plane, seed-sensitive in the median but
uniformly wide. **Coplanarity is NOT ensured, and that is physically expected** for a thione whose sole chelate
partner is an sp3-hinged amine. No fix indicated: the rule is behaving correctly, and the cap is in fact nearly
redundant on this donor (Δself +1.3°).

### Provenance
All numbers are read-only measurements on the installed env; `src/` untouched. Scripts in
`playground/thione_diag/`: `thione_paths.py` (§1 donor set + paths + hyb), `thione_angle.py` (§2 angles).
§3 reproduces `playground/cap_architecture/caparch_redundancy.py case1` verbatim.
```
uv run --no-sync python playground/thione_diag/thione_paths.py
uv run --no-sync python playground/thione_diag/thione_angle.py
uv run --no-sync python playground/cap_architecture/caparch_redundancy.py case1
```
```
