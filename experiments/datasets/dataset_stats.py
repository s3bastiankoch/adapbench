# Stats for our datasets: Java (tcp8) and our own Go recordings (goduet).
#
# Everything works on the measurement-phase duet diff d = (s1-s2)/(s1+s2):
# unitless, 0 = same speed, positive = version 1 slower. Labelling lives in
# adapbench.datasets (samples are autocorrelated, so it block-bootstraps a 99%
# CI of the mean from ~tau-sized blocks and compares it to the +/-3% band);
# load() returns a magnitude per arm (0.0 = confirmed no-change, None =
# undecided). Pool stats below use the exact arm universe the experiments
# run on (common.load_pools: labelled arms with >=60 finite samples).
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from adapbench.datasets import load, project_of  # noqa: E402

SUITE_LABEL = {"tcp8": "Java (tcp8)", "goduet": "Go (goduet)"}
SUITE_COLOR = {"tcp8": "#2a78d6", "goduet": "#eb6834"}

meas_per_project = {}  # suite -> {project: total measurements}
effects_per_project = {}  # suite -> {project: [change magnitudes, fraction]}

for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
    pools = common.load_pools(suite)
    _tps, mags = load(suite)

    # Full label universe, before the pool filter
    labels_all = [m for arms in mags.values() for m in arms.values()]
    decided = [m for m in labels_all if m is not None]
    changed = [m for m in decided if m != 0.0]
    print(f"\n{suite}: {len(labels_all)} labelled arms, {len(decided)} decided, "
          f"{len(changed)} real changes (>=3%)")

    # Print benchmark distributions (how many benchmarks + how many measurements per benchmark)
    traces = sorted({name.split("/", 1)[0] for name in pools})
    depths = [len(d) for d in pools.values()]
    print(f"  pool: {len(traces)} traces, {len(pools)} benchmarks, "
          f"{sum(depths):,} measurements")
    print(f"  measurements per benchmark: median {np.median(depths):.0f}  "
          f"(min {min(depths)}, max {max(depths)})")

    # Print trace durations + overall dataset duration (recorded wall-clock).
    # tcp8 measured the two versions in separate 1s JMH iterations, so a sample
    # pair costs 2x iter_duration_s; goduet's iter_duration_s already holds the
    # per-pair wall cost (max of the sides for concurrent duet/corim arms, sum
    # for sequential rmit; see stage_rmitduet.py).
    pair_cost = 2.0 if suite == "tcp8" else 1.0
    arms_of = {}
    for name in pools:
        trace, fn = name.split("/", 1)
        arms_of.setdefault(trace, []).append(fn)
    hours, span_h = [], 0.0
    for tp in _tps:
        if tp.name not in arms_of:
            continue
        data = np.load(tp / "trace.data.npz")
        secs = sum(float(data[f"{fn}/iter_duration_s"][data[f"{fn}/phase"] == 1].sum())
                   for fn in arms_of[tp.name] if f"{fn}/iter_duration_s" in data)
        hours.append(pair_cost * secs / 3600.0)
        span_h += common.load_shared(tp).meta.get("run_span_s", 0.0) / 3600.0
    if sum(hours):
        print(f"  trace duration: median {np.median(hours):.1f} h  "
              f"(min {min(hours):.1f}, max {max(hours):.1f})  --  "
              f"dataset total {sum(hours):,.0f} h = {sum(hours)/24:.1f} days")
        if span_h:
            # recorder wall span (first->last pass start): adds compile/setup
            # and inter-pass overhead on top of the measured benchmark time
            print(f"  recording span: {span_h:,.0f} h = {span_h/24:.1f} days "
                  f"({span_h / sum(hours):.1f}x the measured time)")
    else:
        print("  trace duration: no timing recorded for this suite")

    # Print label distribution + effect sizes per project
    print(f"  {'project':<20s}{'benchmarks':>11s}{'change':>8s}{'no-change':>11s}"
          f"{'undecided':>11s}{'med effect':>12s}")
    for proj in sorted({project_of(suite, t) for t in traces}):
        labels = [mags[trace][fn] for name in pools
                  for trace, fn in [name.split("/", 1)]
                  if project_of(suite, trace) == proj]
        chg = [m for m in labels if m not in (None, 0.0)]
        n_nc = sum(1 for m in labels if m == 0.0)
        med = f"{100 * np.median(chg):.1f}%" if chg else "-"
        print(f"  {proj:<20s}{len(labels):>11d}{len(chg):>8d}{n_nc:>11d}"
              f"{len(labels) - len(chg) - n_nc:>11d}{med:>12s}")
        if chg:
            effects_per_project.setdefault(suite, {})[proj] = chg

    counts = {}
    for name, d in pools.items():
        proj = project_of(suite, name.split("/", 1)[0])
        counts[proj] = counts.get(proj, 0) + len(d)
    meas_per_project[suite] = counts


# Figures (both suites in one diagram each), written to experiments/figures/
FIGDIR = Path(__file__).resolve().parents[1] / "figures"
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)

# Measurement count per project, grouped by suite, largest first within a suite
rows = [(s, p, n) for s in meas_per_project
        for p, n in sorted(meas_per_project[s].items(), key=lambda kv: -kv[1])]
fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.0))
fig.patch.set_facecolor("#fcfcfb")
style_axis(ax)
ys = range(len(rows))
ax.set_xscale("log")
for y, (s, _, n) in zip(ys, rows):
    ax.axhline(y, color=GRID, linewidth=0.8, zorder=1)
    ax.scatter(n, y, s=55, color=SUITE_COLOR[s], zorder=3)
    ax.annotate(f"{n / 1e3:,.0f}k", (n, y), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK)
ax.set_yticks(ys, [p for _, p, _ in rows])
ax.invert_yaxis()
ax.margins(x=0.15)
ax.set_xticks([50e3, 100e3, 200e3, 500e3, 1e6, 2e6])
ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6
                             else f"{x / 1e3:g}k")
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel("measurements (pooled arms, log scale)", color=INK, fontsize=9)
ax.set_title("Measurements per project", color=INK, fontsize=11, loc="left")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SUITE_COLOR[s],
                                 label=SUITE_LABEL[s]) for s in meas_per_project],
          frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(FIGDIR / "dataset_stats_measurements.png", dpi=200)
print(f"\nwrote {FIGDIR / 'dataset_stats_measurements.png'}")

# Change magnitudes (effect sizes) per project as boxplots, one diagram.
# Projects with zero confirmed changes (e.g. etcd) have no box.
rows = [(s, p, np.asarray(v) * 100) for s in effects_per_project
        for p, v in sorted(effects_per_project[s].items(),
                           key=lambda kv: -np.median(kv[1]))]
fig, ax = plt.subplots(figsize=(0.75 * len(rows) + 1.5, 4.2))
fig.patch.set_facecolor("#fcfcfb")
style_axis(ax)
rng = np.random.default_rng(7)
for i, (s, _, v) in enumerate(rows):
    ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(v)), v, s=12,
               color=SUITE_COLOR[s], alpha=0.4, linewidths=0)
    ax.hlines(np.median(v), i + 1 - 0.3, i + 1 + 0.3,
              color=SUITE_COLOR[s], linewidth=2.5)
ax.axhline(3.0, color=MUTED, linewidth=1, linestyle="--")
ax.annotate("3% relevance threshold", (0.99, 3.0), xycoords=("axes fraction", "data"),
            xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
ax.set_xticks(range(1, len(rows) + 1),
              [f"{p} ({len(v)})" for _, p, v in rows], rotation=30, ha="right")
ax.set_yscale("log")
ax.set_yticks([3, 5, 10, 20, 50, 100, 200, 500, 1000])
ax.yaxis.set_major_formatter(lambda y, _: f"{y:g}%")
ax.yaxis.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_ylabel("change magnitude", color=INK, fontsize=9)
ax.set_title("Confirmed change magnitudes per project", color=INK,
             fontsize=11, loc="left")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SUITE_COLOR[s],
                                 label=SUITE_LABEL[s]) for s in effects_per_project],
          frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(FIGDIR / "dataset_stats_effects.png", dpi=200)
print(f"wrote {FIGDIR / 'dataset_stats_effects.png'}")
