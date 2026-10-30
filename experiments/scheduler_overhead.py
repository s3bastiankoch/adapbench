#!/usr/bin/env python3
"""What does the scheduler itself cost per measurement?

Replays each recorded suite through the shipped FastScheduler exactly as in the
savings experiments, except every measurement value is pre-drawn before the clock
starts and the runner just hands them back. The timed run therefore contains only
scheduler work (benchmark selection + anytime-valid update); wall time divided by
the number of measurements is the per-measurement scheduling overhead.

Run: .venv/bin/python experiments/scheduler_overhead.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adapbench.datasets import EPS_D  # noqa: E402
from adapbench.scheduler import FastScheduler  # noqa: E402


class PredrawnRunner:
    """Returns the next pre-drawn value for an arm at ~zero cost, so a timed run is all scheduler."""

    def __init__(self, streams):
        self._streams = streams
        self._next = {arm: 0 for arm in streams}

    def measure(self, arm):
        i = self._next[arm]
        self._next[arm] = i + 1
        return float(self._streams[arm][i])


def time_suite(suite):
    """Run the full suite under the shipped config; return the amortized us per measurement."""
    pools = common.load_pools(suite)
    alpha_claim, alpha_pool = common.SPLIT[suite]
    block = common.block_map(suite, list(pools))
    # per-arm cap is the recorded pool length, same as the savings experiments
    budget = {arm: len(pool) for arm, pool in pools.items()}
    # pre-draw with replacement, exactly what ReplayRunner would feed the scheduler
    rng = np.random.default_rng(0)
    streams = {arm: pool[rng.integers(0, len(pool), size=budget[arm])]
               for arm, pool in pools.items()}
    sched = FastScheduler(pools, band=EPS_D, alpha_claim=alpha_claim, alpha_pool=alpha_pool,
                          block=block, budget_per_arm=budget, seed=0, pool_decide=True)
    t0 = time.perf_counter()
    _dec, spent = sched.run(PredrawnRunner(streams))
    elapsed = time.perf_counter() - t0
    return {"n_arms": len(pools), "spent": int(spent),
            "us": elapsed / max(spent, 1) * 1e6, "total_s": elapsed}


def main() -> int:
    print("per-measurement scheduler overhead on the recorded pools (shipped config):")
    real = {}
    for suite in sorted(common.SPLIT):
        r = time_suite(suite)
        real[suite] = r
        print(f"  {suite:>6s}: {r['us']:.2f} us/measurement"
              f"  ({r['n_arms']} benchmarks, {r['spent']:,} measurements,"
              f" {r['total_s']:.1f} s total scheduling)")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "scheduler_overhead.json").write_text(json.dumps({"real": real}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
