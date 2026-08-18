#!/usr/bin/env python3
"""A/A test of the deployed FastScheduler. Center every recorded marginal, park
it on the +-3% band edge, run the real online loop, and count false "change"
claims split by route (CS vs pool bootstrap). The worse edge has to hold
alpha = 0.01. The same runs are also aggregated per project (each project's
worse edge, cs + boot decomposition), which backs the paper's worst-project
sentence (eclipsecollections 0.0087 = cs 0.0046 + boot 0.0041) and the A/A
column of tab:perproject.
Run: .venv/bin/python experiments/aa_scheduler.py [tcp8 goduet ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adabench.datasets import EPS_D, project_of  # noqa: E402
from adabench.scheduler import FastScheduler, ReplayRunner  # noqa: E402

ALPHA = 0.01
SEEDS = common.SEEDS  # ADABENCH_SEEDS=5,6,7,8,9 -> held-out A/A, tagged output file
EDGES = ((+1.0, "pos"), (-1.0, "neg"))
# The deployed splits, so this tests exactly what scheduler_savings_pertrace ships
# (the one-sided table under ADABENCH_SIDE=up). Suites without an entry fall back
# to one shared alpha.
SPLIT = common.SPLIT_1S if common.SIDE == "up" else common.SPLIT


def _one_run(task):
    """One (edge, seed) pass of the real scheduler over the band-edge null."""
    sys.path.insert(0, str(HERE.parent / "src"))
    suite, sign, tag, seed = task
    band = EPS_D
    pools = common.load_pools(suite)
    # Least-favorable null: recenter each marginal, then shift it onto the band edge.
    null = {a: (d - d.mean()) + sign * band for a, d in pools.items()}
    budget = {a: len(null[a]) for a in null}  # deployed config caps at pool length
    ac, ap = SPLIT.get(suite, (ALPHA, ALPHA))
    block = common.block_map(suite, null)
    # suites without a calibrated split ship CS-only (scheduler_savings_pertrace
    # does the same), so the A/A must not turn the pool route on for them
    sched = FastScheduler(null, band=band, alpha_claim=ac, alpha_pool=ap, block=block,
                          budget_per_arm=budget, seed=seed, pool_decide=suite in SPLIT,
                          side=("up" if common.SIDE == "up" else None))
    dec, _spent = sched.run(ReplayRunner(null, mode="rmit", seed=seed))
    proj = defaultdict(lambda: [0, 0, 0])  # project -> [arms, cs claims, boot claims]
    for a, arm in sched.arms.items():
        p = proj[project_of(suite, a.split("/")[0])]
        p[0] += 1
        if dec[a] != "change":
            continue
        if arm.spent < budget[a]:  # stopped before budget, so the CS claimed it
            p[1] += 1
        else:  # ran to budget, so the pool bootstrap claimed it
            p[2] += 1
    return tag, dict(proj)


def _wilson(k, n, z=1.96):
    """95% binomial CI on k/n; the A/A rate is a proportion, so print it as one."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (center - half) / denom, (center + half) / denom


def _report(suite, by_edge, out_dir):
    rows = {}
    for tag in ("pos", "neg"):
        n = sum(v[0] for v in by_edge[tag].values())
        cs = sum(v[1] for v in by_edge[tag].values())
        boot = sum(v[2] for v in by_edge[tag].values())
        rows[tag] = dict(n=n, k=cs + boot, cs=cs / max(n, 1), boot=boot / max(n, 1),
                         total=(cs + boot) / max(n, 1))
    worse = max(rows, key=lambda t: rows[t]["total"])
    w = rows[worse]
    tag = "  <- A/A > alpha!" if w["total"] > ALPHA else "  (<= alpha, valid)"
    print(f"\n{suite}  (deployed FastScheduler band-edge A/A, {len(SEEDS)} seeds, alpha={ALPHA})")
    print(f"  {'edge':>6s}{'A/A cs':>10s}{'A/A boot':>10s}{'A/A total':>11s}{'arms':>9s}")
    for t in ("pos", "neg"):
        r = rows[t]
        mark = "  <- worse" if t == worse else ""
        print(f"  {('+band' if t=='pos' else '-band'):>6s}{r['cs']:>10.4f}{r['boot']:>10.4f}"
              f"{r['total']:>11.4f}{r['n']:>9d}{mark}")
    print(f"  worse-edge A/A = {w['total']:.4f}{tag}   (cs {w['cs']:.4f} + bootstrap {w['boot']:.4f})")
    lo, hi = _wilson(w["k"], w["n"])
    print(f"  worse-edge 95% CI [{lo:.4f}, {hi:.4f}]   ({w['k']}/{w['n']} claims)")
    # the same runs per project, each at its own worse edge (tab:perproject A/A column)
    projects = {}
    print(f"  {'project':>20s}{'edge':>7s}{'A/A':>8s}{'cs':>8s}{'boot':>8s}{'claims/arms':>14s}")
    for pj in sorted(by_edge["pos"]):
        per = {}
        for t in ("pos", "neg"):
            n, cs, boot = by_edge[t][pj]
            per[t] = dict(n=n, k=cs + boot, cs=cs / max(n, 1), boot=boot / max(n, 1),
                          total=(cs + boot) / max(n, 1))
        wt = max(per, key=lambda t: per[t]["total"])
        wp = per[wt]
        projects[pj] = {"worse_edge": wt, **per}
        flag = "  <- > alpha!" if wp["total"] > ALPHA else ""
        kn = f"{wp['k']}/{wp['n']}"
        print(f"  {pj:>20s}{('+band' if wt=='pos' else '-band'):>7s}{wp['total']:>8.4f}"
              f"{wp['cs']:>8.4f}{wp['boot']:>8.4f}{kn:>14s}{flag}")
    (out_dir / f"aa_scheduler_{suite}{common.SIDE_TAG}{common.SEEDS_TAG}.json").write_text(
        json.dumps({"suite": suite, "alpha": ALPHA, "side": common.SIDE,
                    "seeds": list(SEEDS), "worse_edge": worse, "rows": rows,
                    "projects": projects}))


def main() -> int:
    suites = sys.argv[1:] or ["tcp8", "goduet"]
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    for suite in suites:
        tasks = [(suite, sign, tag, s) for sign, tag in EDGES for s in SEEDS]
        # per (edge, project): [arms, cs, boot] summed over seeds
        agg = {"pos": defaultdict(lambda: [0, 0, 0]), "neg": defaultdict(lambda: [0, 0, 0])}
        with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as ex:
            for tag, proj in ex.map(_one_run, tasks):
                for pj, (n, cs, boot) in proj.items():
                    a = agg[tag][pj]
                    a[0] += n
                    a[1] += cs
                    a[2] += boot
        _report(suite, agg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
