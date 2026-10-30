"""Shared plumbing for the experiment scripts: the suite loader, the per-arm RNG
(which must match Scheduler._pool_decide bit for bit), the full-recording
bootstrap verdict, and the shipped per-suite scheduler config.

Import this before adapbench; it puts src/ on sys.path.
"""
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "src"))

from adapbench.datasets import load, project_of  # noqa: E402
from adapbench import load_shared                # noqa: E402

RESULTS = HERE / "results"

ALPHA = 0.01
BOOT = 800
# Default seeds 0-4 are ALSO split_calibration's selection seeds, so numbers on them
# are in-sample w.r.t. the alpha split and UNI. ADAPBENCH_SEEDS=5,6,7,8,9 evaluates
# the exact shipped config on the held-out set; tagged outputs keep the runs apart.
_ENV_SEEDS = os.environ.get("ADAPBENCH_SEEDS")
SEEDS = tuple(int(s) for s in _ENV_SEEDS.split(",")) if _ENV_SEEDS else (0, 1, 2, 3, 4)
SEEDS_TAG = f"_seeds{'-'.join(str(s) for s in SEEDS)}" if _ENV_SEEDS else ""
TARGETS = (0.70, 0.80, 0.85, 0.90, 0.95, 0.955, 0.957)
DET = (0.50, 0.75, 0.90)  # detection-latency milestones

SIDE = os.environ.get("ADAPBENCH_SIDE")  # 'up' = slowdown-only, anything else = two-sided
SIDE_TAG = "_1s" if SIDE == "up" else ""  # suffix for one-sided result files
# 'rmit' resamples i.i.d. from the pool; 'ordered' walks the recording once, in
# order. The reference stream and the runner switch together. Per-suite default:
# tcp8 was recorded sequentially, so i.i.d. resampling is the RMIT approximation;
# goduet keeps only RMIT-collected arms (datasets._GODUET_MODES) and replays the
# recording in its RMIT order. ADAPBENCH_REPLAY overrides both (stress-test knob).
_ENV_REPLAY = os.environ.get("ADAPBENCH_REPLAY")
REPLAY_DEFAULT = {"tcp8": "rmit", "goduet": "ordered"}
REPLAY = _ENV_REPLAY or "rmit"  # suite-agnostic fallback (arm-level helpers)


def replay_mode(suite):
    """The replay mode a suite's experiments run under."""
    return _ENV_REPLAY or REPLAY_DEFAULT.get(suite, "rmit")
_ENV_CKPT = os.environ.get("ADAPBENCH_CKPT")  # confirm-only checkpoints; level 0.0 never claims
CKPT = tuple((float(f), 0.0) for f in _ENV_CKPT.split(",")) if _ENV_CKPT else None

# The fixed-n bootstrap over-claims at shallow depth (tcp8 band-edge A/A 0.0118), so
# the uniform baseline gets a widened CI: alpha tightened until its worst-case
# band-edge A/A is back under our alpha (uniform_validity.py). Both sides then pay
# the same false-claim rate.
UNI = {"tcp8": 0.007, "goduet": 0.004}  # goduet: 0.005 fails on the RMIT-only universe (edge A/A 0.0102 > alpha, uniform_validity_goduet.json)

# shipped per-suite scheduler config: A/A-gated tail block (jctools needs block 6
# for its CS route to hold alpha) and the calibrated alpha_claim/alpha_pool split
# from split_calibration.py
BLOCK = {"tcp8": 4, "goduet": 2}
BLOCK_PROJ = {"tcp8": {"jctools": 6}}
# goduet = our own GCP rmitduet go corpora (datasets/stage_rmitduet.py): etcd is
# excluded at the loader (datasets._GODUET_EXCLUDED, reference contamination),
# only the RMIT-collected arms load (datasets._GODUET_MODES), and they replay
# in recorded order (replay_mode). Calibrated on exactly that universe:
# split_calibration_goduet.json sel A/A 0.0079 recall 1.000, held-out 0.0039/0.990.
SPLIT = {"tcp8": (0.005, 0.006), "goduet": (0.005, 0.002)}

# One-sided (ADAPBENCH_SIDE=up, slowdown-only) calibrations
# (split_calibration_tcp8_1s.json: sel A/A 0.0076, held-out 0.0075, recall 0.926;
# uniform_validity_tcp8_1s.json: 0.004 fails by a hair at 0.0100, chosen 0.003).
SPLIT_1S = {"tcp8": (0.005, 0.004)}
UNI_1S = {"tcp8": 0.003}


def shipped_config(suite):
    """(alpha_claim, alpha_pool, pool_decide) under the shipped config: the one-sided
    split when ADAPBENCH_SIDE=up has one, else the calibrated two-sided split, else
    CS-only at plain alpha (the pool route stays off, which keeps uncalibrated
    suites valid without a calibration run)."""
    if SIDE == "up" and suite in SPLIT_1S:
        ac, ap = SPLIT_1S[suite]
        return ac, ap, True
    if suite in SPLIT:
        ac, ap = SPLIT[suite]
        return ac, ap, True
    return ALPHA, ALPHA, False


def uni_alpha(suite):
    """The uniform baseline's validity-matched confidence level (uniform_validity.py)."""
    return (UNI_1S if SIDE == "up" else UNI).get(suite, ALPHA)


def by_trace(names):
    """Arms grouped by version-pair trace, the id before the '/'. Per-trace
    scheduling runs one scheduler per group."""
    g = defaultdict(list)
    for a in names:
        g[a.split("/")[0]].append(a)
    return g


def load_pools(suite, min_len=60):
    """{'trace/arm': finite duet diffs} for every labelled arm with enough samples."""
    tps, mags = load(suite)
    pools = {}
    for tp in tps:
        sh = load_shared(tp)
        for f in sh.functions:
            if f not in mags[tp.name]:
                continue
            d = np.asarray(sh.pool_d[f], float)
            d = d[np.isfinite(d)]
            if len(d) >= min_len:
                pools[f"{tp.name}/{f}"] = d
    return pools


def block_map(suite, names):
    """Per-arm tail block under the shipped config."""
    proj_blk = BLOCK_PROJ.get(suite, {})
    return {a: proj_blk.get(project_of(suite, a.split("/")[0]), BLOCK.get(suite, 1))
            for a in names}


def arm_rng(name, seed):
    """The exact per-arm rng Scheduler._pool_decide draws from. Experiments lean on
    this to rebuild the scheduler's reference stream, so keep it bit-identical."""
    h = hashlib.blake2b(f"{seed}|{name}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def boot_means(x, rng, n_boot=800):
    """n_boot bootstrap means of x, one row of draws per resample."""
    x = np.asarray(x, float)
    return x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)


def boot_change(bm, band, alpha, side=None):
    """True when the bootstrap CI clears the band: the full-recording 'change' verdict."""
    if side == "up":
        return bool(np.percentile(bm, 100 * alpha) > band)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return bool(lo > band or hi < -band)


def ref_stream(name, pool, n, seed, mode=None):
    """Full-budget stream for one arm, identical to what the runner feeds the
    scheduler. Callers pass mode=replay_mode(suite); None falls back to REPLAY."""
    mode = mode or REPLAY
    if mode == "shuffle":
        return np.asarray(pool, float)[arm_rng(name, seed).permutation(len(pool))[:n]]
    if mode == "window":
        from adapbench.scheduler import window_perm
        return np.asarray(pool, float)[window_perm(arm_rng(name, seed), len(pool))[:n]]
    if mode == "hier":
        from adapbench.scheduler import HIER_SIZES, hier_perm
        return np.asarray(pool, float)[hier_perm(arm_rng(name, seed), len(pool), HIER_SIZES)[:n]]
    if mode == "ordered":
        return np.asarray(pool[:n], float)
    return pool[arm_rng(name, seed).integers(0, len(pool), size=n)]


def ref_verdict(name, stream, seed, band):
    """Full-run reference verdict. Shares the scheduler's per-arm bootstrap seed, so only
    early-stopped arms can disagree; recall/FP then measure the cost of stopping early."""
    bm = boot_means(stream, arm_rng(name, seed), BOOT)
    return "change" if boot_change(bm, band, ALPHA, SIDE) else "no_change"


def depth_grid(maxlen):
    """Uniform-baseline depth sweep; denser when pools are shallow, always ending at
    the full pool so running the whole recording is sweepable."""
    grid = list(range(20, maxlen + 1, 60 if maxlen > 400 else 20))
    if grid[-1] != maxlen:
        grid.append(maxlen)
    return grid


def uniform_frontier(names, X, pos, neg, band, grid, total, uni_alpha):
    """Every arm to depth min(d, pool) over the grid; per-depth budget fraction, recall, FP.
    Note the validity-matched uni_alpha < ALPHA is stricter than the reference, which is
    why the baseline's recall ceiling sits below 1.0."""
    npos, nneg = max(int(pos.sum()), 1), max(int(neg.sum()), 1)
    rng = np.random.default_rng(0)
    ref_at_full = uni_alpha == ALPHA  # only then does full depth equal the reference
    ub = np.zeros(len(grid)); ur = np.zeros(len(grid)); uf = np.zeros(len(grid))
    for i, a in enumerate(names):
        x = X[a]; n = len(x)
        for j, d0 in enumerate(grid):
            d = min(d0, n)  # can't measure past the recording
            ub[j] += d
            if d0 >= n and ref_at_full:  # ran the whole recording, so this IS the reference
                clears = bool(pos[i])
            elif SIDE == "up":
                bm = x[:d][rng.integers(0, d, size=(BOOT, d))].mean(axis=1)
                clears = bool(np.percentile(bm, 100 * uni_alpha) > band)
            else:
                bm = x[:d][rng.integers(0, d, size=(BOOT, d))].mean(axis=1)
                lo, hi = np.percentile(bm, [100 * uni_alpha / 2, 100 * (1 - uni_alpha / 2)])
                clears = bool(lo > band or hi < -band)
            if clears:
                ur[j] += pos[i] / npos
                uf[j] += neg[i] / nneg
    return ub / total, ur, uf


def uniform_budget_at(t, ub, ur):
    """Interpolated uniform budget fraction at recall t (nan if the frontier never gets
    there). Interpolation only removes an upward bias in the baseline's budget."""
    j = np.where(ur >= t)[0]
    if not len(j):
        return float("nan")
    k = j[0]
    if k == 0 or ur[k] <= ur[k - 1]:
        return float(ub[k])
    return float(ub[k - 1] + (t - ur[k - 1]) / (ur[k] - ur[k - 1]) * (ub[k] - ub[k - 1]))
