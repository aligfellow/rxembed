# Donor-set generality: EMPIRICAL corpus experiment

READ-ONLY experiment. No src/test edits, no commit. This is the empirical follow-up to
`gen-element-sets.md` (a pure predicate-equivalence audit that leaned KEEP but explicitly noted
"Se/As/Te are real donors chemically but **were never validated here**"). This run validates them
against the crystal corpus and **reverses the `_LONE_PAIR_Z` verdict to BROADEN**.

## Method

Corpus = `../OIN-SMILES/tests/integration/tmQM/*.xyz` (103) + `../OIN-SMILES/tests/fixtures/*.xyz`
(41) = 144 plain-XYZ crystal geometries. Coordination perceived with **`xyzgraph.build_graph(...,
quick=True)`** — the project's own transition-metal-aware bond perceiver, the same one rxembed uses in
`metrics.connectivity`. Every edge carries a `metal_coord` flag; the donor end of each metal
coordination bond (metal ⊕ non-metal, including xyzgraph's `metal_coord=True` datives) is counted.
144/144 files parsed, 0 failures. Metals identified by d-block/f-block/main-group-metal atomic number
(24 distinct metals across tmQM Ti…Au…Hg; 13 across fixtures).

Script: `scratchpad/census.py` (+ spot-check + screen-behaviour demo below).

## 1. Corpus donor-element census

Coordinating-atom counts (donor element → number of atoms of that element bonded to a metal):

| Element | tmQM | fixtures | screen status |
|---|---|---|---|
| C | 225 | 142 | (organometallic / haptic) |
| N | 146 | 30 | **SCREENED** (7) |
| O | 52 | 8 | **SCREENED** (8) |
| P | 48 | 20 | **SCREENED** (15) |
| Cl | 46 | 35 | halogen (not screened) |
| S | 30 | 0 | **SCREENED** (16) |
| Br | 13 | 2 | halogen |
| **Se** | **9** | 0 | **MISSED by current screen** (34) |
| **As** | **2** | 0 | **MISSED by current screen** (33) |
| Si, B, F, H, I | few | few | (organometallic / halide) |

### Heavy p-block (the elements `{7,8,15,16}` misses) — the load-bearing number

| Donor | present in corpus | **COORDINATING a metal** | present but NOT coordinating (false-positive risk) |
|---|---|---|---|
| **Se** (34) | 9 | **9** | **0** |
| **As** (33) | 2 | **2** | **0** |
| **Sb** (51) | 0 | 0 | 0 |
| **Te** (52) | 0 | 0 | 0 |

**11 real, verified metal-donor contacts (9 Se + 2 As) that the current `{7,8,15,16}` screen cannot
propose — and ZERO Se/As/Sb/Te atoms anywhere in the corpus that are present without coordinating.**
In these crystals the heavy p-block heteroatom is *always* the donor ligand; the empirical
false-positive rate of broadening is 0/11 present atoms.

### Spot-check — all 11 are genuine coordination, not perception noise

| CSD | metal | contact | d(M–donor) | ligand type |
|---|---|---|---|---|
| WELROW | La | 6× La–Se | 2.93–3.10 Å | hexakis(selenophosphinate) Se₆ (Se=P) |
| ZAHZUD | Ti | 2× Ti–Se | 2.76 Å | bis(selenophosphinoyl) |
| WUVJAB | Zn | 1× Zn–Se | 2.80 Å | selenolate/selenone (C–Se–Zn) |
| ROLWAR | Pd | 2× Pd–As | 2.35 Å | bis(triarylarsine), AsPh₃ |

All flagged `metal_coord=True` by xyzgraph; distances are textbook M–Se / M–As coordination.

## 2. The exact general predicate

RDKit's periodic table exposes **no group or block API** — only `GetRow` (period) and
`GetNOuterElecs` (NOE). NOE **conflates** group 15/16 with the group-5/6 transition metals: it returns
5 for {N,P,As,Sb,Bi **and** V,Nb,Ta} and 6 for {O,S,Se,Te,Po **and** Cr,Mo,W}. So "NOE ∈ {5,6}" alone
is exactly the trap the prior finding warned about (naively admits V/Cr/Nb/Mo/Ta/W).

The correct, TM-safe general predicate keys on the **true group** (chalcogen = noble-gas Z − 2,
pnictogen = noble-gas Z − 3), which RDKit lets you derive from the period boundaries:

```python
_NOBLE = (2, 10, 18, 36, 54, 86, 118)
def _pnictogen_or_chalcogen(z: int) -> bool:
    """True for a p-block group-15/16 main-group atom (N,O,P,S,As,Se,Sb,Te,Bi,Po …)."""
    end = next(g for g in _NOBLE if g >= z)   # noble gas ending z's period
    return (end - z) in (2, 3)                # 2 -> group 16 (chalcogen), 3 -> group 15 (pnictogen)
```

Verified over Z = 1…118: yields **exactly** `{7,8,15,16,33,34,51,52,83,84,115,116}` =
N,O,P,S,As,Se,Sb,Te,Bi,Po,Mc,Lv. It returns **False** for every transition metal (V,Cr,Nb,Mo,Ta,W →
False) and for the halogens/noble gases (F,Ne,Cl,Br,I → False). It **subsumes and generalises** the
old period-≤-3 cap: the cap was only ever a crude proxy for "exclude the d-block", and the group test
excludes the d-block directly, so admitting Se/As/Sb/Te no longer risks pulling in a TM.

Equivalently, and matching the codebase's explicit-`Z`-set house style, the enumeration of that
predicate through Te is:

```python
_LONE_PAIR_Z = {7, 8, 15, 16, 33, 34, 51, 52}  # N O P S | As Se Sb Te — pnictogen/chalcogen p-block donors
```

(add `83` Bi for completeness if desired; Po/Mc/Lv are radioactive and irrelevant to any substrate).
Both forms exclude the TMs; pick the explicit set for house-style consistency or the helper if a true
general rule is wanted — they enumerate the same donors over the chemically relevant range.

## 3. Impact of broadening on `coordinate="auto"`

`lone_pair_donors` (metal.py:1035) is the sole consumer of `_LONE_PAIR_Z`; its sole caller is
`embed/dispatch.py:234` for `coordinate="auto"`. It is a **candidate-donor screen** on *substrate*
fragments (atoms not in the metal's fragment / not already-known donors) — a seed **bias**, not truth,
and it already logs-and-truncates when candidates outnumber vacancies.

Direct behavioural demo (runtime only; substrate = thioether + selenoether + arsine + ethanol as
separate fragments around a vacant Pd):

```
substrate heteroatoms:  S(thioether)  Se(selenoether)  As(arsine)  O(ethanol)
CURRENT  {7,8,15,16}              proposes:  S, O                 # Se, As silently dropped
BROADEN  {7,8,15,16,33,34,51,52}  proposes:  S, Se, As, O         # real donors caught
```

Broadening catches exactly the real Se/As donors and adds no spurious proposal (the census shows 0
Se/As present-not-coordinating in 144 crystals). Because the screen only *biases the seed* and the
real energy decides, an over-proposed candidate is cheap; a *missed* real donor is a silent
`coordinate="auto"` failure. Net: broadening is a strict improvement here.

## 4. `_CONJUGATING_LP` verdict

`_CONJUGATING_LP = {7,8}` (donor_orient.py:94) is a **different axis** and must not be conflated with
the donor screen. It lives in `_pi_hybridisation`: a **zero-π** lone-pair atom is re-typed sp3→sp2
*only* when its lone pair conjugates into an adjacent π (amide N, carboxylate O). That is a **2p–2p
overlap** phenomenon — strong for period-2 N/O, which planarise; period-3+ P/S/As/Se **pyramidalise**
(PPh₃, AsPh₃ are pyramidal, high inversion barrier). No corpus or textbook evidence exists that a
period-3+ donor planarises into an adjacent π the way an amide does; the heavier the congener, the
*worse* the p–π overlap and the higher the inversion barrier.

Critically, broadening `_CONJUGATING_LP` would type **every triarylphosphine/arsine donor sp2**,
voiding the fold/coplanarity gate on exactly the P/As donors these catalysts are built from — the code
comment already states this. And the two verdicts are **mutually consistent**: broadening
`_LONE_PAIR_Z` sends Se/As donors through `_stripped_hybridisation`, where keeping `_CONJUGATING_LP =
{7,8}` correctly leaves a selenoether Se / arsine As **sp3** (a lone-pair σ-donor, like a thioether S),
which is the right class. (The prior finding also correctly noted the maintainer's alt phrasing
"period-2 p-block with a lone pair" = `{7,8,9,10}` wrongly admits F/Ne.)

**KEEP `_CONJUGATING_LP = {7,8}`.** Period-2-only conjugation is real physics, not a placeholder.

---

## VERDICTS

- **`_LONE_PAIR_Z` — BROADEN (reverses the prior lean-KEEP, on new corpus evidence).**
  Adopt the p-block group-15/16 predicate. Exact enumeration through Te:
  `{7, 8, 15, 16, 33, 34, 51, 52}` (N O P S As Se Sb Te; +83 Bi optional). Or the equivalent computed
  helper `_pnictogen_or_chalcogen` (chalcogen = noble − 2, pnictogen = noble − 3), which excludes the
  transition metals **directly** and so replaces the old period-≤-3 cap without re-introducing the
  V/Cr/Nb/Mo/Ta/W trap. Evidence: **11 real Se/As donors (9 Se + 2 As) coordinating across 3 tmQM
  crystals that the current screen misses; 0 non-coordinating Se/As/Sb/Te in 144 crystals** (0
  empirical false positives), and the screen is a seed bias where a miss is a silent failure.

- **`_CONJUGATING_LP` — KEEP `{7,8}`.** Conjugation into an adjacent π is a 2p–2p phenomenon (period-2
  N/O planarise; period-3+ P/S/As/Se pyramidalise). Broadening would mis-type every phosphine/arsine
  donor sp2 and void the fold gate on them. No corpus/textbook support for period-3+ conjugation.

**Se/As/Sb/Te coordinating in the corpus: Se = 9, As = 2, Sb = 0, Te = 0 (total 11), all genuine,
zero present-not-coordinating.**
