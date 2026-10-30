#!/usr/bin/env python3
"""At matched budget AND recall, what false-positive rate must uniform pay?

The headline comparison holds validity fixed (both sides calibrated to alpha)
and compares budgets. This flips the third axis: take OUR budget at recall
0.90 (per-trace scheduler, headline protocol), give the uniform baseline the
same budget (the smallest grid depth of at least that cost — generous), and
loosen its alpha until it reaches the same recall. Report the alpha it needs
and the false-positive rate it then pays on the real no-change arms, next to
ours at the same operating point.

Run: .venv/bin/python experiments/fp_at_matched.py [tcp8 goduet ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adapbench.datasets import EPS_D  # noqa: E402
from adapbench.scheduler import FastScheduler, ReplayRunner  # noqa: E402

T = float(os.environ.get("ADAPBENCH_T", "0.90"))   # the matched recall anchor
AGRID = (0.007, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50)


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
    npos, nneg = max(int(pos.sum()), 1), max(int((~pos).sum()), 1)
    total = sum(len(pools[a]) for a in names)
    ac, ap, pool = common.shipped_config(suite)

    # ours: within-trace flag times for EVERY flagged arm, true or false
    nd = {}
    for trace, arms in common.by_trace(names).items():
        sub = {a: pools[a] for a in arms}
        sched = FastScheduler(sub, band=EPS_D, alpha_claim=ac, alpha_pool=ap,
                              block={a: block[a] for a in arms},
                              budget_per_arm={a: len(sub[a]) for a in arms},
                              seed=seed, pool_decide=pool,
                              side=("up" if common.SIDE == "up" else None),
                              pool_checkpoints=common.CKPT)
        sched.run(ReplayRunner(sub, mode=mode, seed=seed))
        tt = sum(len(sub[a]) for a in arms)
        nd.update({a: sched.detected_at[a] / tt for a in arms
                   if a in sched.detected_at})
    pos_f = sorted(f for a, f in nd.items() if ref[a] == "change")
    k = int(np.ceil(T * npos))
    if k > len(pos_f):
        return None                                   # recall 0.90 unreachable
    b = pos_f[k - 1]                                  # our budget at recall T
    fp_ours = sum(1 for a, f in nd.items()
                  if ref[a] == "no_change" and f <= b) / nneg

    # uniform at >= the same budget: smallest grid depth costing at least b
    grid = common.depth_grid(max(len(pools[a]) for a in names))
    ub = [sum(min(d, len(pools[a])) for a in names) / total for d in grid]
    d_star = next((d for d, u in zip(grid, ub) if u >= b), grid[-1])
    b_uni = sum(min(d_star, len(pools[a])) for a in names) / total

    rng = np.random.default_rng(0)
    rec = np.zeros(len(AGRID)); fp = np.zeros(len(AGRID))
    for i, a in enumerate(names):
        x = common.ref_stream(a, pools[a], len(pools[a]), seed, mode)
        d = min(d_star, len(x))
        bm = x[:d][rng.integers(0, d, size=(common.BOOT, d))].mean(axis=1)
        if common.SIDE == "up":
            clears = np.quantile(bm, list(AGRID)) > EPS_D
        else:
            los = np.quantile(bm, [al / 2 for al in AGRID])
            his = np.quantile(bm, [1 - al / 2 for al in AGRID])
            clears = (los > EPS_D) | (his < -EPS_D)
        if pos[i]:
            rec += clears / npos
        else:
            fp += clears / nneg
    return {"b": b, "fp_ours": fp_ours, "b_uni": b_uni, "d_star": int(d_star),
            "rec_uni": rec.tolist(), "fp_uni": fp.tolist()}


def main() -> int:
    for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(common.SEEDS)) as ex:
            rows = [r for r in ex.map(_worker, [(suite, s) for s in common.SEEDS])
                    if r is not None]
        # per seed: smallest alpha reaching recall T, and the FP it pays there
        need_a, need_fp, reach = [], [], 0
        for r in rows:
            j = next((j for j, rc in enumerate(r["rec_uni"]) if rc >= T), None)
            if j is None:
                continue
            reach += 1
            need_a.append(AGRID[j])
            need_fp.append(r["fp_uni"][j])
        print(f"\n{suite}  [{time.perf_counter() - t0:.0f}s, {len(rows)} seeds, "
              f"recall anchor {T}]")
        print(f"  ours:    budget {100 * np.mean([r['b'] for r in rows]):.1f}%   "
              f"FP {np.mean([r['fp_ours'] for r in rows]):.4f}")
        if reach:
            print(f"  uniform at the same budget "
                  f"(depth {rows[0]['d_star']}, {100 * np.mean([r['b_uni'] for r in rows]):.1f}%): "
                  f"needs alpha {np.mean(need_a):.3f} "
                  f"(per seed: {sorted(set(need_a))}) -> FP {np.mean(need_fp):.4f} "
                  f"[{reach}/{len(rows)} seeds reach recall {T}]")
        else:
            print(f"  uniform at the same budget (depth {rows[0]['d_star']}): "
                  f"recall {T} UNREACHABLE even at alpha {max(AGRID)}")
        path = common.RESULTS / f"fp_at_matched_{suite}_t{int(round(100 * T))}{common.SIDE_TAG}.json"
        path.write_text(json.dumps({"T": T, "agrid": AGRID, "rows": rows}, indent=1))
        print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
