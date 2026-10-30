"""Load benchmark suites and compute the ground-truth labels.

Single source of the "truth" the scheduler is scored against. Benchmark
samples are autocorrelated, so a naive CI is overconfident; the labeller:

  1. replays each arm's stream in recorded order,
  2. estimates the correlation length tau,
  3. averages blocks of ~tau samples into roughly independent block means,
  4. bootstraps a CI from the block means,
  5. compares it to the +/-3% band.

CI outside the band = real change; inside = confirmed no-change; straddling
an edge = label None (run by the scheduler, never scored).
"""
from __future__ import annotations

import json
import math
import os
import zlib
from pathlib import Path

import numpy as np

from .trace import SharedTrace

# "Big enough to matter" = +/-3% relative change...
BIG_REL = 0.03
# ...which in duet-diff coordinates d=(s1-s2)/(s1+s2) is this half-width.
EPS_D = BIG_REL / (2 + BIG_REL)

# Labelling knobs.
LABEL_ALPHA = 0.01        # two-sided CI level (99% CI)
LABEL_BLOCK_MULT = 1.0    # block size = ceil(block_mult * tau)
LABEL_MIN_N = 30          # fewer samples: "too_short"
LABEL_MIN_BLOCKS = 8      # fewer blocks: "low_neff"
LABEL_N_BOOT = 4000       # bootstrap resamples per CI
LABEL_SEED = 0            # base seed for the per-arm RNG
# Bump on any labelling-logic change; other tags recompute.
LABEL_METHOD = "ordered-replay non-overlapping block-bootstrap CI v2 (full-suite run set; undecided=None)"


def _label_params() -> dict:
    """All knobs that affect a label; stored in the cache and compared on
    load, so any change invalidates stale caches automatically."""
    return {"alpha": LABEL_ALPHA, "block_mult": LABEL_BLOCK_MULT,
            "min_n": LABEL_MIN_N, "min_blocks": LABEL_MIN_BLOCKS,
            "n_boot": LABEL_N_BOOT, "big_rel": BIG_REL, "eps_d": EPS_D,
            "seed": LABEL_SEED}

# Statuses with a known truth; everything else is run but not scored.
DECIDED = ("regression", "improvement", "no_change")

DATA_ROOT = Path(os.environ.get(
    "ADAPBENCH_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))

# The canonical TCP suite: 8 open-source Java projects.
_TCP_PROJECTS = ("bytebuddy", "eclipsecollections", "jctools", "jenetics",
                 "netty", "okio", "xodus", "zipkin")

# goduet corpora whose recordings fail the reference-contamination gate
# (rmitduet_replication.py): etcd's same-version arm rmit/aa/BenchmarkStorePutUpdate
# drifted +7.5% mid-run, so no verdict derived from that recording can be trusted.
# The staged data stays on disk; it just never reaches an experiment.
_GODUET_EXCLUDED = ("etcd",)

# Only RMIT-collected arms reach the experiments: randomized trial order is a
# property of the collection there, so the recorded stream can be replayed as-is.
# The duet/corim arms (concurrent paired execution, no order randomization) stay
# staged on disk but out of the evaluation universe.
_GODUET_MODES = ("rmit",)


def integrated_act(x: np.ndarray) -> float:
    """Integrated autocorrelation time tau: how many consecutive samples
    equal one independent sample (1.0 = uncorrelated). Sokal's windowing
    rule stops the noisy sum once the lag outruns 5*tau."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return 1.0
    x = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n].real
    if acov[0] <= 0.0:
        return 1.0
    rho = acov / acov[0]
    tau = 1.0
    for k in range(1, n):
        tau += 2.0 * rho[k]
        if k >= 5.0 * tau:
            break
    return float(min(max(tau, 1.0), n / 2.0))


def arm_rng(trace: str, fn: str, seed: int = LABEL_SEED) -> np.random.Generator:
    """One RNG per (trace, arm), so labels don't depend on processing order.
    NUL-joined key so distinct pairs can't concatenate to the same string."""
    key = zlib.crc32(f"{trace}\x00{fn}".encode())
    return np.random.default_rng((np.uint32(seed) ^ np.uint32(key)).item())


def block_ci(d: np.ndarray, *, rng: np.random.Generator) -> dict | None:
    """Correlation-respecting CI on the mean: block-average (~tau) first,
    then bootstrap the block means. None = too short; status 'low_neff' =
    too few blocks to trust."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < LABEL_MIN_N:
        return None
    b = max(1, math.ceil(LABEL_BLOCK_MULT * integrated_act(d)))  # block size
    m = n // b                                                   # whole blocks
    # Earlier labeller versions drew a naive iid CI here; the draw is kept
    # (its output unused) so previously computed labels stay bit-identical.
    rng.integers(0, n, size=(LABEL_N_BOOT, n))
    if m < LABEL_MIN_BLOCKS:
        return {"status": "low_neff"}
    block_means = d[:m * b].reshape(m, b).mean(axis=1)
    boot = block_means[rng.integers(0, m, size=(LABEL_N_BOOT, m))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * LABEL_ALPHA / 2,
                                  100 * (1 - LABEL_ALPHA / 2)])
    return {"lo": float(lo), "hi": float(hi)}


def _to_rel(x: float) -> float:
    """Duet diff d back to a relative change (s1/s2 - 1)."""
    x = min(max(x, -0.999999), 0.999999)
    return 2.0 * x / (1.0 - x)


def classify(lo: float, hi: float,
             eps: float = EPS_D) -> tuple[str, bool, float]:
    """CI [lo, hi] -> (status, is-real-change, magnitude)."""
    if lo > eps:                       # CI above +eps: confirmed slowdown
        return "regression", True, abs(_to_rel(lo))
    if hi < -eps:                      # CI below -eps: confirmed speedup
        return "improvement", True, abs(_to_rel(hi))
    if lo > -eps and hi < eps:         # CI inside the band: no change
        return "no_change", False, 0.0
    return "undecided", False, 0.0     # straddles an edge: unknown


def project_of(suite: str, trace_name: str) -> str:
    """Which project a trace belongs to."""
    return trace_name.split("-")[0]


def _trace_dirs(suite: str) -> list[Path]:
    root = DATA_ROOT / "traces" / suite
    if not root.is_dir():
        raise FileNotFoundError(
            f"trace data not found at {root}. Set ADAPBENCH_DATA_ROOT or run "
            "scripts/fetch_data.sh to download the corpus.")
    return sorted(p for p in root.iterdir()
                  if (p / "trace.meta.json").is_file())


def _corpus_signature(tps: list[Path]) -> str:
    """Fingerprint of the corpus (name + data-file size + mtime per trace).
    In the cache key, so a re-staged corpus invalidates stale labels."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for tp in sorted(tps, key=lambda p: p.name):
        try:
            st = (tp / "trace.data.npz").stat()
            sig = f"{tp.name}\x00{st.st_size}\x00{st.st_mtime_ns}"
        except OSError:
            sig = f"{tp.name}\x00missing"
        h.update(sig.encode() + b"\x00")
    return h.hexdigest()


def _compute_labels(tps: list[Path]) -> dict:
    """Label every arm of every trace: decided arms get a magnitude (0.0 =
    confirmed no-change), ambiguous arms None, too-short arms are dropped."""
    mags = {}
    for tp in tps:
        trace = SharedTrace(tp)
        entry = {}
        for fn in trace.functions:
            ci = block_ci(trace.pool_d[fn], rng=arm_rng(tp.name, fn))
            if ci is None:
                continue
            if "lo" in ci:
                status, _truth, mag = classify(ci["lo"], ci["hi"])
            else:
                status = ci["status"]                # low_neff
            entry[fn] = mag if status in DECIDED else None
        if entry:
            mags[tp.name] = entry
    return mags


def load(name: str):
    """Load a suite -> ``(trace_paths, magnitudes)``.

    ``magnitudes[trace][arm]`` is a number where the truth is known
    (0.0 = confirmed no-change), None where it isn't (run, not scored).
    """
    tps = _trace_dirs(name)
    if name == "tcp8":
        tps = [t for t in tps if project_of("tcp8", t.name) in _TCP_PROJECTS]
    if name == "goduet":
        tps = [t for t in tps
               if project_of("goduet", t.name) not in _GODUET_EXCLUDED]

    # Labels are deterministic but slow, so cached under data/labels/, keyed
    # by method + knobs + corpus fingerprint. ADAPBENCH_RELABEL forces a
    # recompute.
    cache = DATA_ROOT / "labels" / f"{name}.json"
    sig = _corpus_signature(tps)
    doc = None
    if cache.is_file() and not os.environ.get("ADAPBENCH_RELABEL"):
        doc = json.loads(cache.read_text())
        stale = (doc.get("method") != LABEL_METHOD
                 or doc.get("params") != _label_params()
                 or doc.get("corpus") != sig)
        if stale:
            doc = None
    if doc is not None:
        mags = doc["mags"]
    else:
        mags = _compute_labels(tps)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"method": LABEL_METHOD, "params": _label_params(), "corpus": sig,
             "mags": mags}))

    if name == "goduet":
        # Mode filter applied after the cache: labels stay whole-corpus, the
        # experiment universe is RMIT-collected arms only (_GODUET_MODES).
        mags = {t: {fn: m for fn, m in arms.items()
                    if fn.split("/")[0] in _GODUET_MODES}
                for t, arms in mags.items()}
        mags = {t: arms for t, arms in mags.items() if arms}

    tps = [t for t in tps if t.name in mags]
    return tps, mags
