# Two hardcoded element sets: general-predicate audit

READ-ONLY investigation. No src/test edits. Verdicts below; RDKit membership checks reproduced inline.

## Summary table

| Set | Elements | Maintainer's proposed rule | Exact general equivalent? | Verdict |
|---|---|---|---|---|
| `_CONJUGATING_LP` | `{7,8}` N,O | "p-block period-2 with a lone pair" | **NO** — that rule = `{7,8,9,10}`, admits **F, Ne** | **KEEP the allow-list** |
| `_LONE_PAIR_Z` | `{7,8,15,16}` N,O,P,S | "group 15/16 and period ≤ 3" | **YES** — exactly `{7,8,15,16}`, excludes Se/As | KEEP (exact alt exists but re-spells the list; period cap is load-bearing) |

---

## Set 1 — `_CONJUGATING_LP = frozenset({7, 8})`

**Location:** `src/rxembed/rdkit_embed/constraints/donor_orient.py:94`, used once at line 127 inside
`_pi_hybridisation`. That function runs on **every atom** of the metal-stripped graph (called in the loop of
`_stripped_hybridisation`, line 164-165). No other consumers.

**Intent (the encoded chemistry).** In `_pi_hybridisation`, an atom with **zero** π bonds is `sp3` *unless* it
is N or O with an aromatic/π neighbour — then its lone pair conjugates into the adjacent π system and it is
re-typed `sp2` (amide N, carboxylate O). The comment states the discriminant explicitly: a **period-2** p-orbital
lone pair conjugates; a **period-3+** one does not — "PPh₃ is pyramidal. Letting P/S conjugate would type every
triarylphosphine sp2, voiding the gate on exactly the phosphine donors these catalysts are made of." So the
load-bearing axis is **period 2 vs period 3+**, and the atoms that ever reach this branch as real donors are the
lone-pair heteroatoms N/O/P/S — of which only the period-2 pair is kept.

**Is the maintainer's exact predicate equivalent? NO.**
"p-block period-2 with a lone pair" = period-2 (Z 3-10) with ≥1 lone pair (`GetNOuterElecs ≥ 5`):

```
period2 & NOE>=5  ->  [7, 8, 9, 10]  =  N, O, F, Ne
```

It **wrongly admits F (9) and Ne (10)**. So the proposal is *not* exactly `{7,8}`. (Chemically F's lone pair
*does* show +M/resonance donation into a ring, so "F conjugates" is not absurd — but F is monovalent: the sp2/sp3
flip is about pyramidalising an atom's *substituents*, and F/Ne have none to flatten. It would be an untested,
meaningless widening the fold/coplanarity census never validated.)

**Does an exact equivalent exist at all?** Technically yes — `NOE ∈ {5,6} ∧ period == 2` gives exactly `{7,8}`
(group-15/16, period-2 = N, O). But it is a strictly worse encoding:
- It is a longer re-spelling of "N and O" with **no added information** over the existing comment.
- The load-bearing discriminant is *period == 2*; framing it as "group 15/16" actively **implies the heavier
  congeners P/S belong** — the precise confusion the comment exists to refute (P/S must NOT conjugate).

**VERDICT: (b) KEEP the explicit allow-list.** No clean general rule yields exactly `{7,8}` and reads more
truthfully than the two-element list already does. The maintainer's specific candidate silently admits F. The
"why" (period-2 conjugates, period-3 does not, PPh₃) is already in the comment — that is the right place for it.

---

## Set 2 — `_LONE_PAIR_Z = {7, 8, 15, 16}`

**Location:** `src/rxembed/rdkit_embed/constraints/metal.py:1032`, used once at line 1046 in `lone_pair_donors`,
whose single caller is `embed/dispatch.py:234` for `coordinate="auto"` — pick substrate atoms that *could* grab a
vacant metal site. It is a **candidate-donor screen**, not a hard physics gate (it even logs and truncates when
candidates outnumber vacancies).

**Intent.** N, O, P, S — groups 15+16, periods 2+3 — the common substrate lone-pair donors.

**Is "group 15 or 16 and period ≤ 3" exactly equivalent? YES.**

```
NOE in {5,6}  &  period<=3 (Z<=18)  ->  [7, 8, 15, 16]  =  N, O, P, S
```

It **excludes Se (34) and As (33)** — both are period 4, so the period cap drops them; it does **not** wrongly
admit Se. The equivalence is exact and self-documenting ("pnictogens + chalcogens, periods 2-3").

**Caveat — the period cap is load-bearing, and a bare group predicate is dangerous.** Without the `period ≤ 3`
(equivalently `Z ≤ 18`) bound, `NOE ∈ {5,6}` admits not only the genuine heavier donors Se/As/Sb/Te but also
transition metals **V, Cr, Nb, Mo** (RDKit reports NOE 5/6 for them):

```
NOE in {5,6}, no cap  ->  N O P S | V Cr | As Se | Nb Mo | Sb Te
```

So any "generalisation" that relaxes the cap would silently pull in untested elements — the exact failure the
maintainer worries about, in the opposite direction. The `{7,8,15,16}` list makes the N/O/P/S boundary a
**deliberate, visible** choice; Se/As/Te are real donors chemically but were never validated here.

**VERDICT: (a/b, lean KEEP).** Unlike Set 1, an *exactly-equivalent* general predicate genuinely exists
(`GetNOuterElecs(z) in (5,6) and _PT... period(z) <= 3`, or `Z <= 18`) and is self-documenting. If the maintainer
prefers group/period phrasing, it is safe to adopt **provided the `period ≤ 3` cap is retained verbatim** — that
cap, not the group membership, is the validated boundary. But the four-element list with its `# N O P S` comment
already states the same thing at equal clarity, and a validated 4-element allow-list is not worse than a predicate
that invites a future "the period cap looks arbitrary, drop it" edit. Net: adoption is *permissible and exact*
here, but yields no correctness gain — keep unless a group/period house style is being applied consistently.

---

## Bottom line

- **`_CONJUGATING_LP {7,8}`** — no exactly-equivalent *and* clearer general predicate; the maintainer's specific
  proposal admits F/Ne. **Keep the allow-list.**
- **`_LONE_PAIR_Z {7,8,15,16}`** — an exact equivalent exists ("group 15/16 ∧ period ≤ 3", excludes Se/As), and
  is self-documenting, but only re-spells the list; the period cap is the load-bearing, validated part. **Keep
  (adopt only as a house-style choice, cap retained).**
