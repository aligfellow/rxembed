"""Plot a results CSV into PNGs: round-trip performance, and, for a tmQMg run, sample coverage.

just bench-plot                                  # benchmark/baseline.csv -> benchmark/docs/
uv run python benchmark/plot.py PATH.csv OUT_DIR  # any other results CSV and output directory
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("benchmark/plot.py needs matplotlib: uv sync --extra workflow") from exc

HERE = Path(__file__).resolve().parent
TEAL, MAROON, GREY = "#2a9d8f", "#800000", "#c9c9c9"
_ROTATE_LABELS_OVER = 10  # metal-axis labels get rotated 90 degrees past this count, to stay legible
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold"})


def _rows(path):
    """Return a results CSV's data rows, skipping its `#` provenance line."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def _median(values):
    """Return the median of a non-empty sequence."""
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _passed(row):
    """Return whether a row's round trip passed; every other status (a failure at any stage, or a
    timeout) is fail.
    """  # noqa: D205
    return row["status"] == "pass"


def plot_time(rows, out):
    """Fig: seconds per structure, log-x, stacked teal (pass) under maroon (fail)."""
    all_secs = [float(r["seconds"]) for r in rows]
    passed = [float(r["seconds"]) for r in rows if _passed(r)]
    other = [float(r["seconds"]) for r in rows if not _passed(r)]
    lo, hi = min(all_secs), max(all_secs)
    bins = [lo * (hi / lo) ** (i / 30) for i in range(31)] if lo > 0 else 30
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        [passed, other],
        bins=bins,
        stacked=True,
        color=[TEAL, MAROON],
        label=[f"pass ({len(passed)})", f"fail ({len(other)})"],
    )
    ax.set_xscale("log")
    ax.axvline(_median(all_secs), color="black", linestyle="--")
    ax.set_xlabel("seconds per structure")
    ax.set_ylabel("structures")
    ax.set_title(f"time per structure (median {_median(all_secs):.1f} s, n={len(rows)})")
    ax.legend(frameon=False)
    _save(fig, out)


def plot_core_rmsd(rows, out):
    """Fig: core RMSD histogram, stacked teal (pass) under maroon (fail; a match-stage failure still has
    one), median marked.
    """  # noqa: D205
    passed = [float(r["core_rmsd"]) for r in rows if r["core_rmsd"] and _passed(r)]
    other = [float(r["core_rmsd"]) for r in rows if r["core_rmsd"] and not _passed(r)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(
        [passed, other],
        bins=20,
        stacked=True,
        color=[TEAL, MAROON],
        label=[f"pass ({len(passed)})", f"fail ({len(other)})"],
    )
    ax.axvline(_median(passed + other), color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Å")
    ax.set_ylabel("structures")
    ax.set_title(f"core RMSD (median {_median(passed + other):.3f} Å, n={len(passed) + len(other)})")
    if other:
        ax.legend(frameon=False)
    _save(fig, out)


def plot_by_metal(rows, out):
    """Fig: per metal, pass count (teal) with every other status stacked on top (maroon)."""
    metals = [m for m, _ in Counter(r["metal"] for r in rows if r["metal"]).most_common()]
    passed = [sum(_passed(r) for r in rows if r["metal"] == m) for m in metals]
    total = [sum(r["metal"] == m for r in rows) for m in metals]
    other = [t - p for t, p in zip(total, passed, strict=True)]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(metals)), 6))
    ax.bar(metals, passed, color=TEAL, label="pass")
    ax.bar(metals, other, bottom=passed, color=MAROON, label="fail")
    ax.set_ylabel("structures")
    ax.set_title(f"pass rate by metal ({len(metals)} metals)")
    ax.tick_params(axis="x", labelrotation=90 if len(metals) > _ROTATE_LABELS_OVER else 0)
    ax.legend(frameon=False)
    _save(fig, out)


def plot_by_class(rows, out, set_label):
    """Fig: pass rate per class, grouped under stereo kind / denticity (kappa) / hapticity (eta)."""
    stereo_kinds = sorted({k for r in rows for k in r["stereo"].split(";") if k})
    kappas = sorted({int(k) for r in rows for k in r["kappa"].split(";") if k})
    etas = sorted({int(k) for r in rows for k in r["eta"].split(";") if k})
    groups = [
        ("stereo", [(k, lambda r, k=k: k in r["stereo"].split(";")) for k in stereo_kinds]),
        ("denticity", [(f"κ{k}", lambda r, k=k: str(k) in r["kappa"].split(";")) for k in kappas]),
        ("hapticity", [(f"η{k}", lambda r, k=k: str(k) in r["eta"].split(";")) for k in etas]),
    ]
    labels, ns, pct, xpos, spans = [], [], [], [], []
    x = 0
    for gname, members in groups:
        start = x
        for label, test in members:
            members_rows = [r for r in rows if test(r)]
            if not members_rows:
                continue
            labels.append(label)
            ns.append(len(members_rows))
            pct.append(100 * sum(_passed(r) for r in members_rows) / len(members_rows))
            xpos.append(x)
            x += 1
        spans.append((start, x - 1, gname))
        x += 1

    tick_labels = [f"{lab} (n={n})" for lab, n in zip(labels, ns, strict=True)]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(xpos, pct, color=TEAL, width=0.7, label="pass")
    ax.bar(xpos, [100 - p for p in pct], bottom=pct, color=MAROON, width=0.7, label="fail")
    ax.set_xticks(xpos)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right")
    # A 40-degree, right-anchored tick label sweeps left and down roughly with its length, so the
    # heading clears it only if pushed down (and the figure given room) by the longest label present.
    drop = 55 + 4.3 * max(len(lab) for lab in tick_labels)
    for start, end, gname in spans:
        ax.annotate(
            gname,
            xy=((start + end) / 2, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -drop),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass rate (%)")
    ax.set_title(f"pass rate by class: {set_label} (n={len(rows)})")
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=min(0.55, 0.20 + 0.003 * drop))
    _save(fig, out)


_COVERAGE_ORDER = {
    "metal": "count",
    "CN": "natural",
    "donor element": "count",
    "denticity": "kappa",
    "hapticity": "eta",
}
# tmQMg's own SMILES draws a bond from every haptic-ring atom to the metal, so CN counts M-X bonds and
# hapticity's face sizes come from that same convention; the panel titles say so.
_COVERAGE_TITLES = {"CN": "M-X bonds (tmQMg SMILES)", "hapticity": "haptic face size (tmQMg SMILES)"}


def _coverage_categories(panel, pop, samp):
    """Order one coverage panel's categories: by population count, numerically, or kappa-n/eta-n by n
    ("none" first for hapticity).
    """  # noqa: D205
    cats = set(pop) | set(samp)
    order = _COVERAGE_ORDER[panel]
    if order == "natural":
        return sorted(cats, key=float)
    if order in ("kappa", "eta"):
        return sorted(cats, key=lambda c: -1 if c == "none" else int(c[1:]))
    return sorted(cats, key=lambda c: -pop.get(c, 0))


def _sample_pass_fail(coverage, id_status):
    """Split each sampled id's coverage tokens by round-trip status, from the sidecar's `sample_tokens`
    (`{id: ["panel:category", ...]}`) and a results CSV's `status` by id.

    Returns `{panel: Counter}` for pass and for fail, plus `{panel: n}` sample totals: the count of
    sampled ids that carried at least one token for that panel, matching the population total's own
    per-row (not per-category) counting.
    """  # noqa: D205
    panels = coverage["panels"]
    pass_counts = {p: Counter() for p in panels}
    fail_counts = {p: Counter() for p in panels}
    totals = dict.fromkeys(panels, 0)
    for sample_id, tokens in coverage["sample_tokens"].items():
        counts = pass_counts if id_status.get(sample_id) == "pass" else fail_counts
        seen = set()
        for token in tokens:
            panel, category = token.split(":", 1)
            counts[panel][category] += 1
            seen.add(panel)
        for panel in seen:
            totals[panel] += 1
    return pass_counts, fail_counts, totals


def _coverage_panel(ax, panel, pop, pop_total, samp_pass, samp_fail, samp_total):
    """Draw one coverage small multiple: population fraction (grey) next to the sample fraction, the
    sample stacked pass (teal) under fail (maroon).
    """  # noqa: D205
    cats = _coverage_categories(panel, pop, samp_pass + samp_fail)
    pop_pct = {c: 100 * pop.get(c, 0) / pop_total if pop_total else 0 for c in cats}
    pass_pct = {c: 100 * samp_pass.get(c, 0) / samp_total if samp_total else 0 for c in cats}
    fail_pct = {c: 100 * samp_fail.get(c, 0) / samp_total if samp_total else 0 for c in cats}
    samp_pct = {c: pass_pct[c] + fail_pct[c] for c in cats}
    cats = [c for c in cats if max(pop_pct[c], samp_pct[c]) >= 0.5]  # noqa: PLR2004 - <0.5% in both is noise
    x = range(len(cats))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [pop_pct[c] for c in cats], w, color=GREY, label="population")
    ax.bar([i + w / 2 for i in x], [pass_pct[c] for c in cats], w, color=TEAL, label="pass")
    ax.bar(
        [i + w / 2 for i in x],
        [fail_pct[c] for c in cats],
        w,
        bottom=[pass_pct[c] for c in cats],
        color=MAROON,
        label="fail",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=90 if len(cats) > _ROTATE_LABELS_OVER else 0)
    ax.set_ylabel("% of structures")
    ax.set_title(_COVERAGE_TITLES.get(panel, panel))


def plot_coverage(coverage, rows, out):
    """Coverage fig: in-scope tmQMg population vs the MaxMin-diverse sample, by `_fingerprint` category.

    The population bar is counts only (grey); the sample bar next to it joins the sidecar's per-id
    tokens against `rows`' `status` by id, so it stacks pass (teal) under fail (maroon).

    The metal panel (most categories, ordered by population count) gets the full-width top row; the
    other four panels (CN, donor element, denticity, hapticity) sit two by two below it.
    """
    panels = coverage["panels"]
    id_status = {r["id"]: r["status"] for r in rows}
    pass_counts, fail_counts, samp_totals = _sample_pass_fail(coverage, id_status)

    def draw(ax, panel):
        p = panels[panel]
        _coverage_panel(
            ax,
            panel,
            p["population"],
            p["population_total"],
            pass_counts[panel],
            fail_counts[panel],
            samp_totals[panel],
        )

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.25)
    draw(fig.add_subplot(gs[0, :]), "metal")
    for i, panel in enumerate(("CN", "donor element", "denticity", "hapticity")):
        draw(fig.add_subplot(gs[1 + i // 2, i % 2]), panel)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"tmQMg sample coverage (population n={coverage['population_n']}, sample n={coverage['sample_n']})",
        fontsize=17,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.08)
    _save(fig, out)


FIGURES = [
    ("time.png", plot_time),
    ("by_metal.png", plot_by_metal),
    ("core_rmsd.png", plot_core_rmsd),
]


def main(argv=None):
    """Plot `argv[0]` (default benchmark/baseline.csv) into `argv[1]` (default benchmark/docs/), with
    every filename prefixed by its cohort ("fixtures" or "tmqmg", read from the results path's stem).

    A `tmqmg` results CSV has a `<stem>-coverage.json` sidecar (written by `run.py`); when present, it
    also draws `<prefix>_coverage.png`.
    """  # noqa: D205
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else HERE / "baseline.csv"
    out_dir = Path(argv[1]) if len(argv) > 1 else HERE / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = "tmqmg" if path.stem.startswith("tmqmg") else "fixtures"
    set_label = "tmQMg sample" if cohort == "tmqmg" else "fixtures"
    rows = _rows(path)
    for name, fn in FIGURES:
        fn(rows, out_dir / f"{cohort}_{name}")
    plot_by_class(rows, out_dir / f"{cohort}_by_class.png", set_label)
    coverage_path = path.with_name(f"{path.stem}-coverage.json")
    if coverage_path.exists():
        plot_coverage(json.loads(coverage_path.read_text()), rows, out_dir / f"{cohort}_coverage.png")


if __name__ == "__main__":
    main()
