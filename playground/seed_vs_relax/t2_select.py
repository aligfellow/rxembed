"""T2 selection rule (deterministic, no RNG).

Round-robin over CN buckets in ascending CN; within a bucket, round-robin over metal elements in
alphabetical order, each turn taking the alphabetically-first not-yet-taken refcode of that metal.
Emit until 30. This spans coordination number first, metal element second, and is a pure function
of the corpus listing.
"""

import json, os, itertools
from collections import defaultdict

HERE = os.path.dirname(__file__)
rows = json.load(open(os.path.join(HERE, "t2_survey.json")))
by_cn = defaultdict(lambda: defaultdict(list))
for r in sorted(rows, key=lambda r: r["name"]):
    by_cn[r["cn"]][r["metal"]].append(r["name"])

cns = sorted(by_cn)


# per-CN generator that round-robins its metals
def cn_stream(cn):
    metals = sorted(by_cn[cn])
    pools = {m: list(by_cn[cn][m]) for m in metals}
    while any(pools.values()):
        for m in metals:
            if pools[m]:
                yield pools[m].pop(0)


streams = {cn: cn_stream(cn) for cn in cns}
picked = []
while len(picked) < 30:
    progressed = False
    for cn in cns:
        if len(picked) >= 30:
            break
        nxt = next(streams[cn], None)
        if nxt is not None:
            picked.append(nxt)
            progressed = True
    if not progressed:
        break

info = {r["name"]: r for r in rows}
picked = sorted(picked)
json.dump([info[p] for p in picked], open(os.path.join(HERE, "t2_selection.json"), "w"), indent=1)
print(f"{len(picked)} selected")
for p in picked:
    r = info[p]
    print(f"  {r['name']:8s} {r['metal']:2s} CN={r['cn']:2d} q={r['q']:+d} n={r['n']:3d} donors={r['donors']}")
from collections import Counter

print("CN spread:", sorted(Counter(info[p]["cn"] for p in picked).items()))
print("metals:", sorted(Counter(info[p]["metal"] for p in picked).items()))
print(
    "donor elements:", sorted({c for p in picked for c in __import__("re").findall(r"[A-Z][a-z]?", info[p]["donors"])})
)
