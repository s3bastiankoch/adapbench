#!/usr/bin/env python3
"""Detection latency of the deployed allocation vs round-robin: at what budget
fraction has each policy surfaced a given share of the true changes, by count
and weighted by |effect|? Both sides run the identical FastScheduler and
stopping rules; only the allocation order differs (round-robin = ef 1.0).
Per-trace scheduling as in scheduler_savings_pertrace.py, so a detection costs
within-trace spent over that trace's total. Decisions and total spend are
allocation-invariant; only detection times move.
Run: .venv/bin/python experiments/detection_latency.py [tcp8 goduet ...]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adapbench.datasets import EPS_D, project_of  # noqa: E402
from adapbench.scheduler import FastScheduler, ReplayRunner  # noqa: E402

STRATS = (("ours", 0.15), ("round-robin", 1.00))  # (name, explore fraction)
PROJ_DET = (0.50, 0.75)  # per-project sooner tally milestones


def _milestones(nd, w, npos, totw, targets):
    """Budget fraction at which the detections (nd) cover each target share of
    all npos true changes: by count and by |effect| weight. nan when the policy
    never gets there (undetected changes count against both denominators)."""
    order = np.argsort(nd)
    nd_s = np.asarray(nd, float)[order]
    cum = np.cumsum(np.asarray(w, float)[order])
    cnt, mag = {}, {}
    for t in targets:
        k = int(np.ceil(t * npos))
        cnt[t] = float(nd_s[k - 1]) if 0 < k <= len(nd_s) else float("nan")
        j = np.where(cum >= t * totw)[0]
        mag[t] = float(nd_s[j[0]]) if len(j) else float("nan")
    return cnt, mag


def _worker(task):
    sys.path.insert(0, str(HERE.parent / "src"))
    sys.path.insert(0, str(HERE))
    suite, seed = task
    pools = common.load_pools(suite)
    names = list(pools)
    block = common.block_map(suite, names)
    mode = common.replay_mode(suite)
    X = {a: common.ref_stream(a, pools[a], len(pools[a]), seed, mode) for a in names}
    ref = {a: common.ref_verdict(a, X[a], seed, EPS_D) for a in names}
    posset = {a for a in names if ref[a] == "change"}
    npos = max(len(posset), 1)
    wts = {a: abs(float(np.mean(X[a]))) for a in posset}
    totw = sum(wts.values()) or 1.0
    ac, ap, pool = common.shipped_config(suite)
    groups = common.by_trace(names)

    out, spents = {}, set()
    for pname, ef in STRATS:
        nd, spent = {}, 0
        for arms in groups.values():
            sub = {a: pools[a] for a in arms}
            blk = {a: block[a] for a in arms}
            sched = FastScheduler(sub, band=EPS_D, alpha_claim=ac, alpha_pool=ap, block=blk,
                                  budget_per_arm={a: len(sub[a]) for a in arms}, seed=seed,
                                  pool_decide=pool, ef=ef)
            _dec, sp = sched.run(ReplayRunner(sub, mode=mode, seed=seed))
            spent += sp
            tt = sum(len(sub[a]) for a in arms)
            for a in arms:
                if a in posset and a in sched.detected_at:
                    nd[a] = sched.detected_at[a] / tt
        spents.add(spent)
        det = list(nd)
        cnt, mag = _milestones([nd[a] for a in det], [wts[a] for a in det],
                               npos, totw, common.DET)
        per_proj = {}
        for p in sorted({project_of(suite, a.split("/")[0]) for a in posset}):
            arms_p = [a for a in posset if project_of(suite, a.split("/")[0]) == p]
            det_p = [a for a in arms_p if a in nd]
            c_p, _ = _milestones([nd[a] for a in det_p], [wts[a] for a in det_p],
                                 len(arms_p), sum(wts[a] for a in arms_p) or 1.0, PROJ_DET)
            per_proj[p] = c_p
        out[pname] = {"count": cnt, "mag": mag, "per_project": per_proj}
    return {"strats": out, "n_pos": len(posset), "n_arms": len(names),
            "n_traces": len(groups), "spent_invariant": len(spents) == 1}


nanmean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")  # noqa: E731


def main():
    suites = sys.argv[1:] or ["tcp8"]
    for suite in suites:
        t0 = time.time()
        tasks = [(suite, s) for s in common.SEEDS]
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            runs = list(ex.map(_worker, tasks))

        agg = {"suite": suite, "replay": common.replay_mode(suite), "side": common.SIDE or "two-sided",
               "seeds": list(common.SEEDS), "n_arms": runs[0]["n_arms"],
               "n_traces": runs[0]["n_traces"],
               "n_pos": nanmean([r["n_pos"] for r in runs]),
               "spent_invariant": all(r["spent_invariant"] for r in runs), "strats": {}}
        for pname, _ef in STRATS:
            s = {"count": {t: nanmean([r["strats"][pname]["count"][t] for r in runs])
                           for t in common.DET},
                 "mag": {t: nanmean([r["strats"][pname]["mag"][t] for r in runs])
                         for t in common.DET},
                 "per_project": {}}
            for p in runs[0]["strats"][pname]["per_project"]:
                s["per_project"][p] = {t: nanmean([r["strats"][pname]["per_project"][p][t]
                                                   for r in runs]) for t in PROJ_DET}
            agg["strats"][pname] = s

        ours, rr = agg["strats"]["ours"], agg["strats"]["round-robin"]
        projs = list(ours["per_project"])
        sooner = {t: sum(ours["per_project"][p][t] < rr["per_project"][p][t] for p in projs)
                  for t in PROJ_DET}
        agg["sooner_projects"] = {t: f"{sooner[t]}/{len(projs)}" for t in PROJ_DET}

        print(f"\n{suite}: {agg['n_arms']} arms / {agg['n_traces']} traces, "
              f"{agg['n_pos']:.0f} changes ({agg['side']})   "
              f"[{len(common.SEEDS)} seeds, {time.time() - t0:.1f}s, "
              f"replay={common.replay_mode(suite)}, per-trace scheduling, "
              f"spend-invariant={agg['spent_invariant']}]")
        print(f"  {'':>10}  {'changes detected':^28}  {'magnitude captured':^28}")
        print(f"  {'share':>10}  {'ours':>8} {'r-robin':>8} {'sooner':>8}  "
              f"{'ours':>8} {'r-robin':>8} {'sooner':>8}")
        for t in common.DET:
            print(f"  {100 * t:>9.0f}%  {100 * ours['count'][t]:>7.1f}% "
                  f"{100 * rr['count'][t]:>7.1f}% "
                  f"{rr['count'][t] / ours['count'][t]:>7.1f}x  "
                  f"{100 * ours['mag'][t]:>7.1f}% {100 * rr['mag'][t]:>7.1f}% "
                  f"{rr['mag'][t] / ours['mag'][t]:>7.1f}x")
        print("  ours sooner (count-based): " +
              ", ".join(f"{agg['sooner_projects'][t]} projects at {100 * t:.0f}%"
                        for t in PROJ_DET))

        out = common.RESULTS / f"detection_latency_{suite}{common.SEEDS_TAG}{common.SIDE_TAG}.json"
        out.write_text(json.dumps(agg, indent=1, default=str))
        print(f"  -> {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
