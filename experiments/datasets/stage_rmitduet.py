#!/usr/bin/env python3
"""Stage our own GCP rmitduet go corpora (the tab:govalid recordings) as the
'goduet' suite, one SharedTrace per corpus. /aa/ arms are same-version
duplicates, i.e. genuine nulls.

Two staging paths:

* **raw** (preferred): read the recorder's parquets from the rmitduet run
  directory (../../datasets/rmitduet; the recorder harness and its raw runs
  are not part of this repo) and keep the real per-version times, per-side
  iteration durations (b.N x ns/op) and per-pass wall timestamps. The duet
  diff recomputed from the raw sides must be bit-identical to the committed
  ``results/rmitduet_<tag>-replication_streams.npz`` arrays or staging aborts,
  so pool_d (and therefore every label and result) cannot move.
* **synthesized** (fallback, when the raw runs are absent): the committed
  stream files carry only d, so per-version times are faked as
  s1 = 1e9*(1+d), s2 = 1e9*(1-d): (s1-s2)/(s1+s2) reconstructs d exactly.
  In that case do not use goduet sut1_ns/sut2_ns for timing analyses.

Wall-clock caveat (from the recorder README): 'duet'/'corim' arms run the two
versions concurrently, 'rmit' sequentially, so per-sample wall cost is
max(dur1,dur2) for the former and dur1+dur2 for the latter; that rule fills
``iter_duration_s``. Cross-mode wall-span comparisons are confounded; per-side
durations are stored raw so consumers can pick their own accounting.

Run: .venv/bin/python experiments/datasets/stage_rmitduet.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402,F401
from adabench.datasets import DATA_ROOT  # noqa: E402

CORPORA = ("prometheus_common", "compress_flate", "gotoml", "zap", "etcd")
RMITDUET = HERE.parent.parent.parent / "datasets" / "rmitduet"
CONCURRENT_MODES = {"duet", "corim"}  # both suts run simultaneously

# Extraction runs in the rmitduet venv when pyarrow is missing here. Same
# grouping/ordering as rmitduet_replication.py so d comes out bit-identical.
_EXTRACT = r'''
import pyarrow.parquet as pq, glob, collections, json, sys
import numpy as np
run_dir, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(run_dir + "/samples/pass_*.parquet")):
    t = pq.read_table(f).to_pydict()
    rows.extend({k: t[k][i] for k in t} for i in range(len(t["ts_ns"])))
g = collections.defaultdict(lambda: collections.defaultdict(lambda: {1: [], 2: []}))
pass_ts = collections.defaultdict(list)
for r in rows:
    if not r["mode"]:
        continue
    pass_ts[r["pass_idx"]].append(r["ts_ns"])
    dur_s = json.loads(r["extra"])["n"] * r["value_ns"] / 1e9
    g[(r["mode"], r["pairing"], r["bench"])][r["pass_idx"]][r["sut"]].append(
        (r["unit_seq"], r["iter"], r["value_ns"], dur_s))
arrs = {}
for key, passes in g.items():
    s1, s2, d1, d2, pidx = [], [], [], [], []
    for p in sorted(passes):
        a = sorted(passes[p][1])
        b = sorted(passes[p][2])
        for (_, _, v1, u1), (_, _, v2, u2) in zip(a, b):
            if v1 > 0 and v2 > 0:
                s1.append(v1); s2.append(v2); d1.append(u1); d2.append(u2)
                pidx.append(p)
    if s1 and len(s1) >= 100:
        arm = f"{key[0]}/{key[1]}/{key[2]}"
        arrs[arm + "\ts1"] = np.asarray(s1)
        arrs[arm + "\ts2"] = np.asarray(s2)
        arrs[arm + "\tdur1_s"] = np.asarray(d1)
        arrs[arm + "\tdur2_s"] = np.asarray(d2)
        arrs[arm + "\tpass_idx"] = np.asarray(pidx, np.int32)
arrs["\tpass_ids"] = np.asarray(sorted(pass_ts), np.int32)
arrs["\tpass_start_ts_ns"] = np.asarray([min(pass_ts[p]) for p in sorted(pass_ts)])
np.savez(out, **arrs)
'''


def _extract_raw(run_dir: Path, out: Path) -> None:
    try:
        import pyarrow  # noqa: F401
        exe = sys.executable
    except ImportError:
        exe = str(RMITDUET / ".venv" / "bin" / "python")
    subprocess.run([exe, "-c", _EXTRACT, str(run_dir), str(out)], check=True)


def stage_raw(tag: str, committed: Path, out_root: Path) -> None:
    cache = HERE.parent / "results" / f"rmitduet_{tag}-replication_full.npz"
    if not cache.is_file():
        _extract_raw(RMITDUET / "data" / "runs" / f"{tag}-seed0", cache)
    raw = np.load(cache)
    arms = sorted({k.split("\t")[0] for k in raw.files if "\ts1" in k})

    ref = np.load(committed)
    if sorted(ref.files) != arms:
        raise SystemExit(f"{tag}: arm set differs from committed streams")

    arrs, dur_tot = {}, 0.0
    for fn in arms:
        s1 = np.asarray(raw[fn + "\ts1"], float)
        s2 = np.asarray(raw[fn + "\ts2"], float)
        d = (s1 - s2) / (s1 + s2)
        if not np.array_equal(d, np.asarray(ref[fn], float)):
            raise SystemExit(f"{tag}/{fn}: raw-derived d != committed stream")
        d1 = np.asarray(raw[fn + "\tdur1_s"], float)
        d2 = np.asarray(raw[fn + "\tdur2_s"], float)
        wall = (np.maximum(d1, d2) if fn.split("/")[0] in CONCURRENT_MODES
                else d1 + d2)
        arrs[f"{fn}/phase"] = np.ones(len(s1), np.int8)  # raw is all-measurement
        arrs[f"{fn}/sut1_ns"] = s1
        arrs[f"{fn}/sut2_ns"] = s2
        arrs[f"{fn}/sut1_dur_s"] = d1
        arrs[f"{fn}/sut2_dur_s"] = d2
        arrs[f"{fn}/iter_duration_s"] = wall
        arrs[f"{fn}/pass_idx"] = raw[fn + "\tpass_idx"]
        arrs[f"{fn}/noise_combined"] = np.zeros(len(s1))
        dur_tot += float(d1.sum() + d2.sum())

    ts = np.asarray(raw["\tpass_start_ts_ns"], np.int64)
    meta = {
        "functions": arms,
        "source": f"raw parquets {tag}-seed0 (papers/datasets/rmitduet), "
                  f"d verified bit-identical to {committed.name}",
        "note": "sut ns are real recorded times; iter_duration_s = per-sample "
                "wall cost (max of sides for duet/corim, sum for rmit); "
                "cross-mode wall comparisons are confounded, see recorder README",
        "pass_start_ts_ns": ts.tolist(),
        "run_span_s": float(ts.max() - ts.min()) / 1e9,  # first->last pass start
        "measured_cpu_s": dur_tot,
    }
    _write(out_root / tag, meta, arrs)
    print(f"{tag}: raw-staged {len(arms)} arms x {len(s1)} samples, "
          f"span {meta['run_span_s'] / 3600:.1f}h, measured {dur_tot / 3600:.1f}h")


def stage_synth(tag: str, committed: Path, out_root: Path) -> None:
    z = np.load(committed)
    arrs = {}
    for fn in z.files:
        d = np.asarray(z[fn], float)
        arrs[f"{fn}/phase"] = np.ones(len(d), np.int8)
        arrs[f"{fn}/sut1_ns"] = 1e9 * (1.0 + d)
        arrs[f"{fn}/sut2_ns"] = 1e9 * (1.0 - d)
        arrs[f"{fn}/noise_combined"] = np.zeros(len(d))
    meta = {"functions": sorted(z.files), "source": committed.name,
            "note": "s1/s2 synthesized from d; see stage_rmitduet.py"}
    _write(out_root / tag, meta, arrs)
    print(f"{tag}: synthesized {len(z.files)} arms x {len(d)} samples (no raw runs)")


def _write(tdir: Path, meta: dict, arrs: dict) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "trace.meta.json").write_text(json.dumps(meta))
    np.savez_compressed(tdir / "trace.data.npz", **arrs)


def main() -> int:
    out_root = DATA_ROOT / "traces" / "goduet"
    for tag in CORPORA:
        committed = HERE.parent / "results" / f"rmitduet_{tag}-replication_streams.npz"
        if (RMITDUET / "data" / "runs" / f"{tag}-seed0" / "samples").is_dir():
            stage_raw(tag, committed, out_root)
        else:
            stage_synth(tag, committed, out_root)
    print("staged; load('goduet') will relabel on first use (pool_d unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
