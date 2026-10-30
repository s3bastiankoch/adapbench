#!/usr/bin/env python3
"""Figure-1 data: two arms' evidence accumulation over the measurements.

One chart: x = measurements; per measurement one dot per
benchmark (the measured change, left axis) and the accumulated betting wealth
of the shipped claims certifier (right axis) with the 200:1 claim threshold.
One arm certifies quickly, the other never does.

Both arms are the SAME JMH benchmark, netty's
ByteBufAllocatorConcurrentBenchmark.allocateRelease (size=00256), across two
different version pairs:
  claim:      4.1.26 -> 4.1.32, +7.9%, wealth crosses 200:1 at n = 108
  undecided:  4.1.13 -> 4.1.19, +3.2% (band edge), max log e 4.23 < 5.30,
              never certifies over all 1,490 recorded measurements

The streams and the certifier are bit-identical to what the scheduler runs:
common.ref_stream replay (seed 0), block-4 means into cs.claims_cs,
alpha_claim from the calibrated tcp8 split.

Output: results/evidence_traj_<suite>.json, rendered by the paper's figure
pipeline as fig_evidence.

Run: .venv/bin/python experiments/fig_evidence_traj.py [suite]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adapbench.cs import claims_cs  # noqa: E402
from adapbench.datasets import EPS_D, _to_rel  # noqa: E402

_BENCH = ("io.netty.microbench.buffer."
          "ByteBufAllocatorConcurrentBenchmark.allocateRelease|size=00256")
ARMS = (("claim", f"netty-04_01_26-04_01_32-t1/{_BENCH}"),
        ("undecided", f"netty-04_01_13-04_01_19-t1/{_BENCH}"))
SEED = 0


def _traj(arm, pool, block, ac, thr, mode):
    """Replay one arm: per-measurement change (rel %) + per-block log e-value."""
    x = common.ref_stream(arm, pool, len(pool), SEED, mode)
    cs = claims_cs(ac, eps=EPS_D)
    t, log_e = [], []
    for i in range(len(x) // block):
        cs.update(float(np.mean(x[i * block:(i + 1) * block])))
        t.append((i + 1) * block)
        log_e.append(cs.band_log_e(EPS_D))
    log_e = np.array(log_e)
    hit = np.where(log_e >= thr)[0]
    return {"arm": arm, "n_pool": len(pool),
            # exact duet-diff -> relative map (r-1 = 2d/(1-d)); +EPS_D is exactly +3%
            "rel_pct": float(100.0 * _to_rel(float(np.mean(pool)))),
            "stop_t": int(t[hit[0]]) if len(hit) else None,
            "t": t, "log_e": [float(v) for v in log_e],
            "d_pct": [float(100.0 * _to_rel(float(v))) for v in np.asarray(x, float)]}


def main() -> int:
    suite = sys.argv[1] if len(sys.argv) > 1 else "tcp8"
    pools = common.load_pools(suite)
    ac = common.SPLIT.get(suite, (common.ALPHA,))[0]
    thr = math.log(1.0 / ac)

    arms = []
    for role, arm in ARMS:
        block = common.block_map(suite, [arm])[arm]
        r = {"role": role, **_traj(arm, pools[arm], block, ac, thr,
                                   common.replay_mode(suite))}
        arms.append(r)
        s = (f"claims at n = {r['stop_t']} of {r['n_pool']}" if r["stop_t"]
             else f"never claims (max log e {max(r['log_e']):.2f} < {thr:.2f})")
        print(f"  {role:>9s}: {arm.split('/')[0]}  effect {r['rel_pct']:+.2f}%  {s}")

    out = {"suite": suite, "seed": SEED, "replay": common.replay_mode(suite),
           "block": common.block_map(suite, [a for _, a in ARMS]),
           "alpha_claim": ac, "thr": thr, "band_pct": 3.0, "arms": arms}
    path = common.RESULTS / f"evidence_traj_{suite}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
