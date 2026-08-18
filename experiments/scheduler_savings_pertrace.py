#!/usr/bin/env python3
"""Headline evaluation: recall-vs-budget frontier of the real online scheduler
(adabench.FastScheduler) run independently per version-pair trace, the CI-run
deployment unit, so no trace subsidizes another's budget.

Each trace gets its own scheduler over its own budget; a detection costs
within-trace spent over that trace's total. Traces run concurrently at equal
fractional rate, which makes the recall-t budget the t-quantile of these
fractions. Recall/FP are scored as agreement with the full-recording verdict,
and matched-recall budgets are compared against a validity-matched uniform
baseline (which doesn't care about allocation scope).

The same scheduler runs also produce the per-project table (tab:perproject):
detections, spend, and the uniform frontier re-scoped to each project's arms
alone, so that matched-recall comparison is fully within-project. Projects
with fewer than 5 relevant changes in any seed are dropped (this removes
etcd, which has zero confirmed changes).

Output: stdout tables + results/perproject_savings.json

Run: .venv/bin/python experiments/scheduler_savings_pertrace.py [tcp8 goduet ...]
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
from adabench.datasets import EPS_D, project_of  # noqa: E402
from adabench.scheduler import FastScheduler, ReplayRunner  # noqa: E402

# shipped blocks + calibrated alpha split live in common.py, next to the
# validity-matched uniform levels (UNI, from uniform_validity.py).
ALPHA, SEEDS, TARGETS, DET = common.ALPHA, common.SEEDS, common.TARGETS, common.DET
RECALLS = tuple(sorted(set(TARGETS) | set(DET)))
PJ_TARGETS = (0.70, 0.80, 0.85, 0.90)  # per-project matched-recall levels


def _one_trace(pools, arms, block, ac, ap, pool, seed, mode):
    """The real scheduler over one trace's arms, budgeted at that trace's pools."""
    sub = {a: pools[a] for a in arms}
    blk = {a: block[a] for a in arms}
    sched = FastScheduler(sub, band=EPS_D, alpha_claim=ac, alpha_pool=ap, block=blk,
                          budget_per_arm={a: len(sub[a]) for a in arms}, seed=seed,
                          pool_decide=pool, side=("up" if common.SIDE == "up" else None),
                          pool_checkpoints=common.CKPT)  # opt-in confirm ladder via ADABENCH_CKPT
    dec, sp = sched.run(ReplayRunner(sub, mode=mode, seed=seed))
    return dec, sp, dict(sched.detected_at)


def _frontier_budget(norm_det, npos, targets):
    """Budget to reach recall t: the ceil(t*npos)-th smallest per-arm detection fraction."""
    s = np.sort(np.asarray(norm_det, float))
    out = {}
    for t in targets:
        kk = int(np.ceil(t * npos))
        out[t] = float(s[kk - 1]) if kk <= len(s) else float("nan")
    return out


def _worker(task):
    sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
    suite, seed = task
    pools = common.load_pools(suite)  # reloading in the worker beats pickling the pools over
    names = list(pools)
    block = common.block_map(suite, names)
    mode = common.replay_mode(suite)
    ref = {a: common.ref_verdict(a, common.ref_stream(a, pools[a], len(pools[a]), seed, mode),
                                 seed, EPS_D)
           for a in names}
    pos = np.array([ref[a] == "change" for a in names])
    neg = np.array([ref[a] == "no_change" for a in names])
    posset = {a for a, p in zip(names, pos) if p}
    npos, nneg = max(int(pos.sum()), 1), max(int(neg.sum()), 1)
    total = sum(len(pools[a]) for a in names)
    ac, ap, pool = common.shipped_config(suite)

    # Per-trace scheduling: within-trace detection fractions + pooled decisions/spend,
    # accumulated suite-wide (headline) and per project (tab:perproject).
    nd_trace, spent, decisions = [], 0, {}
    nd_proj, spent_proj = defaultdict(list), defaultdict(int)
    for trace, arms in common.by_trace(names).items():
        dec, sp, det = _one_trace(pools, arms, block, ac, ap, pool, seed, mode)
        tt = sum(len(pools[a]) for a in arms)
        nd = [det[a] / tt for a in arms if a in posset and a in det]
        pj = project_of(suite, trace)
        nd_trace += nd
        nd_proj[pj] += nd
        spent += sp
        spent_proj[pj] += sp
        decisions.update(dec)
    flagged = np.array([decisions[a] == "change" for a in names])
    st = (100 * (1 - spent / total),
          (flagged & pos).sum() / npos, (flagged & neg).sum() / nneg)

    # The uniform validity-matched baseline (allocation-scope-free by construction).
    X = {a: common.ref_stream(a, pools[a], len(pools[a]), seed, mode) for a in names}
    grid = common.depth_grid(max(len(pools[a]) for a in names))
    ua = common.uni_alpha(suite)
    ub, ur, _uf = common.uniform_frontier(names, X, pos, neg, EPS_D, grid, total, ua)

    bo = _frontier_budget(nd_trace, npos, RECALLS)
    bu = {t: common.uniform_budget_at(t, ub, ur) for t in RECALLS}

    # Per project: the same runs and machinery, scoped to the project's arms alone.
    proj_of = np.array([project_of(suite, a.split("/")[0]) for a in names])
    projects = {}
    for pj in sorted(set(proj_of)):
        m = proj_of == pj
        pj_names = [a for a, mm in zip(names, m) if mm]
        pj_pos, pj_neg = max(int(pos[m].sum()), 1), max(int(neg[m].sum()), 1)
        tot = sum(len(pools[a]) for a in pj_names)
        pj_ub, pj_ur, _pf = common.uniform_frontier(pj_names, X, pos[m], neg[m], EPS_D,
                                                    grid, tot, ua)
        projects[pj] = {
            "n_arms": int(m.sum()), "npos": int(pos[m].sum()),
            "bo": _frontier_budget(nd_proj.get(pj, []), pj_pos, PJ_TARGETS),
            "bu": {t: common.uniform_budget_at(t, pj_ub, pj_ur) for t in PJ_TARGETS},
            "st_saved": 100 * (1 - spent_proj[pj] / tot),
            "st_recall": (flagged[m] & pos[m]).sum() / pj_pos,
            "st_fp": (flagged[m] & neg[m]).sum() / pj_neg,
        }
    return {"n_arms": len(names), "n_pos": int(pos.sum()), "n_traces": len(common.by_trace(names)),
            "bo": bo, "bu": bu, "st": st, "projects": projects}


def main() -> int:
    out = {}  # per-project rows, accumulated across suites
    for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(SEEDS)) as ex:
            rows = list(ex.map(_worker, [(suite, s) for s in SEEDS]))
        dt = time.perf_counter() - t0
        r0 = rows[0]
        nanmean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")  # noqa: E731
        pct = lambda v: f"{100*v:.1f}%" if v == v else "  -"  # noqa: E731
        cols = lambda t: ([r["bu"][t] for r in rows], [r["bo"][t] for r in rows])  # noqa: E731
        st = [r["st"] for r in rows]
        ua = common.uni_alpha(suite)
        tgt = "slowdown >3% (one-sided)" if common.SIDE == "up" else "|effect| >3% (two-sided)"
        print(f"\n{suite}: {r0['n_arms']} arms / {r0['n_traces']} traces, {r0['n_pos']} {tgt}   "
              f"[{len(SEEDS)} seeds, {dt:.1f}s, replay={common.replay_mode(suite)}, per-trace scheduling, "
              f"uniform alpha={ua:g}{' (validity-matched)' if ua < ALPHA else ''}]")
        print(f"  self-terminating: saved {nanmean([x[0] for x in st]):.1f}%   "
              f"recall {nanmean([x[1] for x in st]):.3f}   FP {nanmean([x[2] for x in st]):.4f}")

        # Compute ratios per seed, then average; a ratio of seed-averaged budgets can flip sign.
        print(f"  {'matched recall':>14s}{'uniform budget':>16s}{'our budget':>13s}{'saved':>9s}")
        for t in TARGETS:
            bus, bos = cols(t)
            saved = [100 * (1 - o / u) for o, u in zip(bos, bus) if o == o and u == u and u > 0]
            s = f"{np.mean(saved):.1f}%" if saved else "  -"
            print(f"  {t*100:>13.1f}%{pct(nanmean(bus)):>16s}{pct(nanmean(bos)):>13s}{s:>9s}")

        print(f"  {'detected':>14s}{'ours (budget)':>16s}{'uniform':>13s}{'speedup':>9s}")
        for t in DET:
            bus, bos = cols(t)
            spd = [u / o for o, u in zip(bos, bus) if o == o and u == u and o > 0]
            s = f"{np.mean(spd):.1f}x" if spd else "  -"
            print(f"  {t*100:>13.0f}%{pct(nanmean(bos)):>16s}{pct(nanmean(bus)):>13s}{s:>9s}")

        # tab:perproject: the same runs, matched fully within-project.
        hdr = "".join(f"{f'saved@{t:.2f}':>13s}" for t in PJ_TARGETS)
        print(f"  {'project':>18s}{'arms':>6s}{'npos':>6s}{hdr}{'uni@0.90':>10s}"
              f"{'st saved':>10s}{'recall':>8s}{'FP':>8s}")
        for pj in rows[0]["projects"]:
            pr = [r["projects"][pj] for r in rows]
            if min(x["npos"] for x in pr) < 5:
                print(f"  {pj:>18s}: dropped (<5 relevant changes)")
                continue
            # Ratios per seed, then averaged; a ratio of seed-averaged budgets can flip sign.
            saved, uni = {}, {}
            for t in PJ_TARGETS:
                s = [100 * (1 - x["bo"][t] / x["bu"][t]) for x in pr
                     if x["bo"][t] == x["bo"][t] and x["bu"][t] == x["bu"][t]
                     and x["bu"][t] > 0]
                saved[t] = float(np.mean(s)) if len(s) == len(pr) else float("nan")
                u = [100 * x["bu"][t] for x in pr if x["bu"][t] == x["bu"][t]]
                uni[t] = float(np.mean(u)) if len(u) == len(pr) else float("nan")
            mean = lambda k: float(np.mean([x[k] for x in pr]))  # noqa: E731
            out[pj] = {"suite": suite, "n_arms": pr[0]["n_arms"],
                       "npos": int(np.mean([x["npos"] for x in pr])),
                       "saved": {f"{t:.2f}": saved[t] for t in PJ_TARGETS},
                       "uni_budget": {f"{t:.2f}": uni[t] for t in PJ_TARGETS},
                       "st_saved": mean("st_saved"), "st_recall": mean("st_recall"),
                       "st_fp": mean("st_fp")}
            cells = "".join(f"{saved[t]:>12.1f}%" if saved[t] == saved[t]
                            else f"{'-':>13s}" for t in PJ_TARGETS)
            u90 = f"{uni[0.90]:>9.1f}%" if uni[0.90] == uni[0.90] else f"{'-':>10s}"
            print(f"  {pj:>18s}{pr[0]['n_arms']:>6d}{out[pj]['npos']:>6d}{cells}{u90}"
                  f"{mean('st_saved'):>9.1f}%{mean('st_recall'):>8.3f}{mean('st_fp'):>8.4f}")
        # written after every suite, so a late failure never loses a long run
        path = common.RESULTS / f"perproject_savings{common.SIDE_TAG}{common.SEEDS_TAG}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
