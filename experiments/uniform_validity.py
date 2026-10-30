#!/usr/bin/env python3
"""Calibrate the uniform baseline: the loosest alpha_uni whose worst-case
band-edge A/A (over depths, worse of the two edges) still holds alpha. The
percentile bootstrap is anti-conservative at shallow depth (BCa and bootstrap-t
were no better), so widening the level is the fix. common.UNI carries the result
to scheduler_savings_pertrace.py. Also reported at the 0.8*alpha bar.
Run: .venv/bin/python experiments/uniform_validity.py [tcp8 goduet ...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from adapbench.datasets import EPS_D  # noqa: E402
import common  # noqa: E402

ALPHA = common.ALPHA
BOOT = common.BOOT
SIDE = common.SIDE  # ADAPBENCH_SIDE=up -> slowdown-only baseline, _1s outputs
DEPTHS = (80, 200, 500, 1000, 1490)  # stress every depth, keep the worst
AU_GRID = (0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010)  # candidate levels


def _extract(task):
    """One (edge, seed) pass. Bootstraps once per (arm, depth); the alpha_uni
    sweep just re-percentiles the same draws."""
    sys.path.insert(0, str(HERE.parent / "src"))
    suite, sign, seed = task
    band = EPS_D
    pools = common.load_pools(suite)
    rng = np.random.default_rng(1)
    hit = np.zeros((len(DEPTHS), len(AU_GRID)))
    n = 0
    for a in pools:
        d = pools[a]
        nlen = len(d)
        # Band-edge null draw. A/A is a rate, so it is always measured under i.i.d.
        # resampling (the aa_scheduler.py / split_calibration.py convention), even
        # for suites whose recall/savings replay the recorded order.
        st = common.ref_stream(a, (d - d.mean() + sign * band), nlen, seed, "rmit")
        n += 1
        for i, dep in enumerate(DEPTHS):
            k = min(dep, nlen)
            bm = st[:k][rng.integers(0, k, size=(BOOT, k))].mean(1)
            if SIDE == "up":  # boot_change one-sided: lower bound at alpha clears +band
                hit[i] += np.quantile(bm, list(AU_GRID)) > band
                continue
            los = np.quantile(bm, [au / 2 for au in AU_GRID])
            his = np.quantile(bm, [1 - au / 2 for au in AU_GRID])
            hit[i] += (los > band) | (his < -band)
    return sign, hit, n


def _report(suite, agg, out_dir):
    print(f"\n{suite}{'  [one-sided: slowdowns only]' if SIDE == 'up' else ''}"
          f"  (uniform baseline band-edge A/A vs its confidence level; worst over depths"
          f" {list(DEPTHS)}, worse of +-band, {len(common.SEEDS)} seeds)")
    print(f"  {'alpha_uni':>10s}{'worst A/A':>11s}")
    chosen_a = chosen_8 = None
    rows = {}
    for j, au in enumerate(AU_GRID):
        rp = agg[+1.0][0][:, j] / max(agg[+1.0][1], 1)
        rn = agg[-1.0][0][:, j] / max(agg[-1.0][1], 1)
        worst = float(max(rp.max(), rn.max()))
        rows[au] = worst
        tag = ""
        if worst <= ALPHA:
            chosen_a = au
            tag = "  <= alpha"
        if worst <= 0.8 * ALPHA:
            chosen_8 = au
            tag = "  <= 0.8*alpha"
        print(f"  {au:>10.3f}{worst:>11.4f}{tag}")
    print(f"  -> validity-matched alpha_uni = {chosen_a}  (<= alpha);  {chosen_8}  (<= 0.8*alpha)")
    (out_dir / f"uniform_validity_{suite}{common.SIDE_TAG}.json").write_text(json.dumps(
        {"suite": suite, "alpha": ALPHA, "side": SIDE, "worst_by_alpha_uni": rows,
         "chosen_alpha": chosen_a, "chosen_margin": chosen_8}))


def main() -> int:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
        tasks = [(suite, s, seed) for s in (+1.0, -1.0) for seed in common.SEEDS]
        agg = {+1.0: [np.zeros((len(DEPTHS), len(AU_GRID))), 0],
               -1.0: [np.zeros((len(DEPTHS), len(AU_GRID))), 0]}
        with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as ex:
            for sign, hit, n in ex.map(_extract, tasks):
                agg[sign][0] += hit
                agg[sign][1] += n
        _report(suite, agg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
