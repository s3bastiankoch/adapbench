#!/usr/bin/env python3
"""Bake-off of within-trace allocation orders: which one reaches each matched
recall on the least budget? Decisions and total spend are allocation-invariant
here, so the only thing that can move is when each change gets detected. tcp8 only.
Run: .venv/bin/python experiments/allocation_bakeoff.py
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adabench.datasets import EPS_D  # noqa: E402
from adabench.scheduler import FastScheduler, ReplayRunner  # noqa: E402


# Every strategy just overrides _score (the exploit key); ef sets the explore mix.
def _cold(arm):
    return -(abs(arm.sm / arm.nb) if arm.nb else 0.0)


class Magnitude(FastScheduler):  # chase the biggest apparent |effect|
    def _score(self, arm):
        return _cold(arm) if arm.nb < self.minn else -abs(arm.sm / arm.nb)


class ZToClaim(FastScheduler):  # most standard errors past the band
    def _score(self, arm):
        if arm.nb < self.minn:
            return _cold(arm)
        m = arm.sm / arm.nb
        var = max(arm.sq / arm.nb - m * m, 1e-9)
        return -(abs(m) - self.band) / math.sqrt(var / arm.nb)


class LowVar(FastScheduler):  # cheapest to resolve first
    def _score(self, arm):
        if arm.nb < self.minn:
            return _cold(arm)
        m = arm.sm / arm.nb
        return max(arm.sq / arm.nb - m * m, 1e-9)


# (scheduler class, explore fraction) per strategy
STRATS = [
    ("dc.15 [current]", FastScheduler, 0.15),
    ("dc.05",           FastScheduler, 0.05),
    ("dc.00",           FastScheduler, 0.00),
    ("round-robin",     FastScheduler, 1.00),
    ("magnitude",       Magnitude,     0.00),
    ("z-to-claim",      ZToClaim,      0.00),
    ("low-variance",    LowVar,        0.00),
]


def _frontier_budget(norm_det, npos, targets):
    s = np.sort(np.asarray(norm_det, float))
    return {t: (float(s[int(np.ceil(t * npos)) - 1]) if int(np.ceil(t * npos)) <= len(s)
                else float("nan")) for t in targets}


def _pertrace(pools, groups, block, ac, ap, pool, seed, cls, ef, posset, npos, targets,
              mode):
    nd, spent = [], 0
    for arms in groups.values():
        sub = {a: pools[a] for a in arms}
        blk = {a: block[a] for a in arms} if isinstance(block, dict) else block
        sched = cls(sub, band=EPS_D, alpha_claim=ac, alpha_pool=ap, block=blk,
                    budget_per_arm={a: len(sub[a]) for a in arms}, seed=seed,
                    pool_decide=pool, ef=ef)
        _dec, sp = sched.run(ReplayRunner(sub, mode=mode, seed=seed))
        tt = sum(len(sub[a]) for a in arms)
        nd += [sched.detected_at[a] / tt for a in arms if a in posset and a in sched.detected_at]
        spent += sp
    return _frontier_budget(nd, npos, targets), spent


def _worker(task):
    sys.path.insert(0, str(HERE.parent / "src"))
    sys.path.insert(0, str(HERE))
    suite, seed = task
    pools = common.load_pools(suite)
    names = list(pools)
    block = common.block_map(suite, names)
    mode = common.replay_mode(suite)
    ref = {a: common.ref_verdict(a, common.ref_stream(a, pools[a], len(pools[a]), seed, mode),
                                 seed, EPS_D)
           for a in names}
    pos = np.array([ref[a] == "change" for a in names])
    neg = np.array([ref[a] == "no_change" for a in names])
    posset = {a for a, p in zip(names, pos) if p}
    npos = max(int(pos.sum()), 1)
    total = sum(len(pools[a]) for a in names)
    ac, ap, pool = common.shipped_config(suite)
    groups = common.by_trace(names)

    # The uniform frontier doesn't depend on the strategy, so build it once.
    X = {a: common.ref_stream(a, pools[a], len(pools[a]), seed, mode) for a in names}
    grid = common.depth_grid(max(len(pools[a]) for a in names))
    ub, ur, _ = common.uniform_frontier(names, X, pos, neg, EPS_D, grid, total,
                                        common.uni_alpha(suite))
    bu = {t: common.uniform_budget_at(t, ub, ur) for t in common.TARGETS}

    out, spents = {}, set()
    for name, cls, ef in STRATS:
        bo, sp = _pertrace(pools, groups, block, ac, ap, pool, seed, cls, ef, posset, npos,
                           common.TARGETS, mode)
        spents.add(sp)
        out[name] = {t: (100 * (1 - bo[t] / bu[t]) if bo[t] == bo[t] and bu[t] and bu[t] == bu[t]
                         else float("nan")) for t in common.TARGETS}
    return {"saved": out, "n_pos": int(pos.sum()), "spent_invariant": len(spents) == 1}


def main():
    tgts = [0.70, 0.80, 0.85, 0.90, 0.95]  # what we print; frontiers cover all of common.TARGETS
    with ProcessPoolExecutor(max_workers=len(common.SEEDS)) as ex:
        rows = list(ex.map(_worker, [("tcp8", s) for s in common.SEEDS]))
    inv = all(r["spent_invariant"] for r in rows)
    print(f"\nPER-TRACE allocation bake-off (tcp8, {len(common.SEEDS)} seeds, {rows[0]['n_pos']} changes)"
          f"   saved % vs validity-matched uniform   [spent-invariant across strategies: {inv}]")
    hdr = "".join(f"{int(t*100):>7d}%" for t in tgts)
    print(f"  {'strategy':<18s}{hdr}{'mean':>8s}")
    scored = []
    for name, _c, _e in STRATS:
        vals = {t: float(np.nanmean([r["saved"][name][t] for r in rows])) for t in tgts}
        mean = float(np.nanmean(list(vals.values())))
        scored.append((mean, name, vals))
        cells = "".join(f"{vals[t]:>7.1f}%" for t in tgts)
        print(f"  {name:<18s}{cells}{mean:>7.1f}%")
    best = max(scored, key=lambda x: x[0])
    print(f"\n  BEST (mean saved): {best[1]}  at {best[0]:.1f}%")
    ranked = sorted(scored, key=lambda x: -x[0])
    print("  ranking: " + " > ".join(f"{n} ({m:.1f})" for m, n, _ in ranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
