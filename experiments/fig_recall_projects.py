#!/usr/bin/env python3
"""fig_recall_projects data: recall development per project, ours vs the
validity-matched uniform baseline.

Headline protocol (per-trace FastScheduler, shipped config, 5 seeds). Per
project and recall level t, the budget to reach t:

  ours     the ceil(t*npos)-th smallest within-trace detection fraction of
           the project's relevant changes (traces run concurrently at equal
           fractional rate), averaged over seeds
  uniform  the validity-matched fixed-depth baseline's budget fraction at
           recall t on the project's arms (interpolated on its depth sweep),
           averaged over seeds

Curves stop where any seed can no longer reach t on either side (recall
ceilings). Projects with fewer than 5 relevant changes in any seed are
dropped (their recall curve would be a couple of steps).

Output: results/recall_projects.json, rendered by the paper's figure
pipeline as fig_recall_projects.

Run: .venv/bin/python experiments/fig_recall_projects.py [tcp8 goduet ...]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adapbench.datasets import EPS_D, project_of  # noqa: E402
from adapbench.scheduler import FastScheduler, ReplayRunner  # noqa: E402


def _uniform_clears(names, X, pos, grid, band, uni_alpha):
    """Per-arm uniform verdicts: clears[i, j] = arm i claims at depth grid[j];
    depth[i, j] = samples actually spent. Iteration order and rng usage are
    identical to common.uniform_frontier, so aggregations reproduce it."""
    rng = np.random.default_rng(0)
    ref_at_full = uni_alpha == common.ALPHA
    clears = np.zeros((len(names), len(grid)), bool)
    depth = np.zeros((len(names), len(grid)), int)
    for i, a in enumerate(names):
        x = X[a]; n = len(x)
        for j, d0 in enumerate(grid):
            d = min(d0, n)
            depth[i, j] = d
            if d0 >= n and ref_at_full:
                c = bool(pos[i])
            elif common.SIDE == "up":
                bm = x[:d][rng.integers(0, d, size=(common.BOOT, d))].mean(axis=1)
                c = bool(np.percentile(bm, 100 * uni_alpha) > band)
            else:
                bm = x[:d][rng.integers(0, d, size=(common.BOOT, d))].mean(axis=1)
                lo, hi = np.percentile(bm, [100 * uni_alpha / 2,
                                            100 * (1 - uni_alpha / 2)])
                c = bool(lo > band or hi < -band)
            clears[i, j] = c
    return clears, depth


def _worker(task):
    sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
    suite, seed = task
    pools = common.load_pools(suite)
    names = list(pools)
    block = common.block_map(suite, names)
    mode = common.replay_mode(suite)
    ref = {a: common.ref_verdict(a, common.ref_stream(a, pools[a], len(pools[a]), seed, mode),
                                 seed, EPS_D)
           for a in names}
    pos = np.array([ref[a] == "change" for a in names])
    posset = {a for a, p in zip(names, pos) if p}
    ac, ap, pool = common.shipped_config(suite)

    nd_proj = defaultdict(list)
    for trace, arms in common.by_trace(names).items():
        sub = {a: pools[a] for a in arms}
        blk = {a: block[a] for a in arms}
        sched = FastScheduler(sub, band=EPS_D, alpha_claim=ac, alpha_pool=ap,
                              block=blk,
                              budget_per_arm={a: len(sub[a]) for a in arms},
                              seed=seed, pool_decide=pool,
                              side=("up" if common.SIDE == "up" else None),
                              pool_checkpoints=common.CKPT)
        sched.run(ReplayRunner(sub, mode=mode, seed=seed))
        tt = sum(len(pools[a]) for a in arms)
        nd_proj[project_of(suite, trace)] += [
            sched.detected_at[a] / tt for a in arms
            if a in posset and a in sched.detected_at]

    X = {a: common.ref_stream(a, pools[a], len(pools[a]), seed, mode) for a in names}
    grid = common.depth_grid(max(len(pools[a]) for a in names))
    clears, depth = _uniform_clears(names, X, pos, grid, EPS_D, common.uni_alpha(suite))
    proj_of = np.array([project_of(suite, a.split("/")[0]) for a in names])
    projects = {}
    for pj in sorted(set(proj_of)):
        m = proj_of == pj
        npos = max(int(pos[m].sum()), 1)
        tot = sum(len(pools[a]) for a, mm in zip(names, m) if mm)
        projects[pj] = {
            "npos": int(pos[m].sum()),
            "nd": sorted(nd_proj.get(pj, [])),
            "ub": (depth[m].sum(axis=0) / tot).tolist(),
            "ur": ((clears[m] & pos[m, None]).sum(axis=0) / npos).tolist(),
        }
    return projects


def main() -> int:
    out = {}
    for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(common.SEEDS)) as ex:
            rows = list(ex.map(_worker, [(suite, s) for s in common.SEEDS]))
        for pj in rows[0]:
            pr = [r[pj] for r in rows]
            if min(x["npos"] for x in pr) < 5:
                print(f"    {pj}: dropped (<5 relevant changes)")
                continue
            t_max = min([len(x["nd"]) / x["npos"] for x in pr]
                        + [max(x["ur"]) for x in pr])
            ts = np.arange(0.05, t_max + 1e-9, 0.01)
            # fp guard: t at a seed's exact ceiling must not round past the list
            ours = np.mean([[x["nd"][min(int(np.ceil(t * x["npos"] - 1e-9)),
                                         len(x["nd"])) - 1]
                             for t in ts] for x in pr], axis=0)
            uni = np.mean([[common.uniform_budget_at(t, np.array(x["ub"]),
                                                     np.array(x["ur"]))
                            for t in ts] for x in pr], axis=0)
            out[pj] = {"suite": suite, "npos": int(np.mean([x["npos"] for x in pr])),
                       "t": ts.tolist(), "ours": ours.tolist(),
                       "uniform": uni.tolist(), "t_max": float(t_max)}
            print(f"    {pj:>20s}: npos ~{out[pj]['npos']}, curves to "
                  f"recall {t_max:.2f}")
        print(f"  [{suite}: {time.perf_counter() - t0:.1f}s]")
        # written after every suite, so a late failure never loses a long run
        path = common.RESULTS / f"recall_projects{common.SIDE_TAG}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
