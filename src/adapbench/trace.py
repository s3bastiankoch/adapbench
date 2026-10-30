"""Load recorded benchmark traces from disk.

A trace is one recorded session comparing two versions of a project:
``trace.meta.json`` names the benchmarks, ``trace.data.npz`` holds paired
times (sut1_ns, sut2_ns) and a warmup/measure phase flag.

All downstream code works on the duet diff d = (s1-s2)/(s1+s2): unitless,
0 = same speed, positive = version 1 slower.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE_MEASURE = 1

_SHELL_CACHE: dict = {}   # trace dir -> SharedTrace, one load per process


def load_shared(trace_dir: str | Path) -> "SharedTrace":
    """Load a trace, reading its files at most once per process."""
    key = str(trace_dir)
    trace = _SHELL_CACHE.get(key)
    if trace is None:
        trace = SharedTrace(trace_dir)
        if len(_SHELL_CACHE) >= 512:   # crude bound on cache growth
            _SHELL_CACHE.clear()
        _SHELL_CACHE[key] = trace
    return trace


class SharedTrace:
    """One loaded trace. Per benchmark name in ``functions``:

    * ``pool_d[fn]`` - measurement-phase duet diffs. NaNs KEPT so indices
      match the recording; consumers filter as they see fit.
    """

    def __init__(self, trace_dir: str | Path):
        trace_dir = Path(trace_dir)
        self.trace_id = trace_dir.name
        self.meta = json.loads((trace_dir / "trace.meta.json").read_text())
        data = np.load(trace_dir / "trace.data.npz", allow_pickle=False)
        self._init_from(data, self.meta["functions"])

    def _init_from(self, data, functions) -> None:
        self.functions: list[str] = []
        self.pool_d: dict[str, np.ndarray] = {}

        for fn in functions:
            if f"{fn}/phase" not in data.files:
                continue   # in metadata but never recorded
            phase = data[f"{fn}/phase"]
            s1 = data[f"{fn}/sut1_ns"].astype(np.float64)
            s2 = data[f"{fn}/sut2_ns"].astype(np.float64)

            # Duet diff; zero denominator or missing time -> NaN.
            denom = s1 + s2
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.where(denom != 0, (s1 - s2) / denom, np.nan)
            d = np.where(np.isnan(s1) | np.isnan(s2), np.nan, d)

            self.functions.append(fn)
            self.pool_d[fn] = d[np.where(phase == PHASE_MEASURE)[0]]
