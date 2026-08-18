#!/usr/bin/env python3
"""fig_savings_projects data: per-project savings, attributed to stop rules.

The shipped FastScheduler run per version-pair trace to self-termination
(identical protocol/config to scheduler_savings_pertrace.py), 5 seeds
averaged. Every SAVED measurement (pool minus spent) is attributed to the
rule that retired its arm, recorded at the moment the scheduler fires it:

  claim      _finish("change")      early CS claim
  futility   _finish("no_change")   futility collapse inside the band
  certified  _finish("undecided")   certified budget stop
  exhausted  no _finish             ran its whole recording (saves nothing)

Projects from every suite land in one dict (tcp8 Java projects + the Go
corpora from goduet), each tagged with its suite for the renderer.

Output: results/stop_rules.json, rendered by the paper's figure
pipeline as fig_savings_projects.

Run: .venv/bin/python experiments/fig_stop_rules.py [tcp8 goduet ...]
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

RULES = ("claim", "futility", "certified")
_REASON = {"change": "claim", "no_change": "futility", "undecided": "certified"}


class _Recording(FastScheduler):
    """FastScheduler that remembers which rule first retired each arm."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reason: dict = {}

    def _finish(self, arm, verdict):
        self.reason.setdefault(arm.name, _REASON[verdict])
        FastScheduler._finish(arm, verdict)


def _worker(task):
    sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
    suite, seed = task
    pools = common.load_pools(suite)
    names = list(pools)
    block = common.block_map(suite, names)
    mode = common.replay_mode(suite)
    ac, ap, pool = common.shipped_config(suite)

    # per project: total pool, saved samples per rule, arm counts per rule
    saved = defaultdict(lambda: np.zeros(len(RULES)))
    n_rule = defaultdict(lambda: np.zeros(len(RULES)))
    budget = defaultdict(int)
    n_arms = defaultdict(int)
    n_exhausted = defaultdict(int)
    for trace, arms in common.by_trace(names).items():
        proj = project_of(suite, trace)
        sub = {a: pools[a] for a in arms}
        blk = {a: block[a] for a in arms}
        sched = _Recording(sub, band=EPS_D, alpha_claim=ac, alpha_pool=ap,
                           block=blk,
                           budget_per_arm={a: len(sub[a]) for a in arms},
                           seed=seed, pool_decide=pool,
                           side=("up" if common.SIDE == "up" else None),
                           pool_checkpoints=common.CKPT)
        sched.run(ReplayRunner(sub, mode=mode, seed=seed))
        for a in arms:
            budget[proj] += len(sub[a])
            n_arms[proj] += 1
            r = sched.reason.get(a)
            if r is None:
                n_exhausted[proj] += 1  # ran the full recording, saves nothing
                continue
            j = RULES.index(r)
            saved[proj][j] += len(sub[a]) - sched.arms[a].spent
            n_rule[proj][j] += 1
    return {p: {"budget": budget[p], "arms": n_arms[p],
                "saved": saved[p].tolist(), "n_rule": n_rule[p].tolist(),
                "n_exhausted": n_exhausted[p]}
            for p in budget}


def main() -> int:
    out = {}
    for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(common.SEEDS)) as ex:
            rows = list(ex.map(_worker, [(suite, s) for s in common.SEEDS]))
        for p in rows[0]:
            sv = np.mean([r[p]["saved"] for r in rows], axis=0)
            nr = np.mean([r[p]["n_rule"] for r in rows], axis=0)
            tot = sv.sum()
            out[p] = {"suite": suite,
                      "arms": rows[0][p]["arms"], "budget": rows[0][p]["budget"],
                      "saved_pct": float(100 * tot / rows[0][p]["budget"]),
                      **{f"share_{k}": float(100 * sv[j] / tot) if tot else 0.0
                         for j, k in enumerate(RULES)},
                      **{f"n_{k}": float(nr[j]) for j, k in enumerate(RULES)},
                      "n_exhausted": float(np.mean([r[p]["n_exhausted"]
                                                    for r in rows]))}
            print(f"  {p:>20s} ({suite}): saved {out[p]['saved_pct']:5.1f}%  "
                  f"claim/futility/certified = "
                  f"{out[p]['share_claim']:.0f}/{out[p]['share_futility']:.0f}"
                  f"/{out[p]['share_certified']:.0f}%")
        print(f"  [{suite}: {time.perf_counter() - t0:.1f}s, "
              f"{len(common.SEEDS)} seeds, replay={common.replay_mode(suite)}]")
    path = common.RESULTS / f"stop_rules{common.SIDE_TAG}{common.SEEDS_TAG}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
