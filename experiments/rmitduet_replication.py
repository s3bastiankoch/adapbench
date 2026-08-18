#!/usr/bin/env python3
"""Regenerates every number in the paper's Go-corpus validity table (tab:govalid)
from the raw recorded parquets. All draws are seeded, so the output has to land
exactly on the frozen table values; any drift makes the script exit non-zero.
Extracted streams are cached under results/ and committed, so the check runs
without the raw recorder output; delete them to force re-extraction.

Recall/FP/saved come from the single deterministic recorded-order run; the A/A
column is a rate, so it is measured under band-edge i.i.d. resampling (10 seeds,
sel 0-4 / held-out 5-9), matching aa_scheduler.py. The i.i.d.-replay contingency
of that guarantee is the same one disclosed for the headline suites.

Run: .venv/bin/python experiments/rmitduet_replication.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adabench.scheduler import FastScheduler, ReplayRunner  # noqa: E402
from adabench.datasets import EPS_D, _GODUET_MODES  # noqa: E402

RMITDUET = HERE.parent.parent / "datasets" / "rmitduet"
BOOT, ALPHA = 800, 0.01
CFG = dict(block=2, alpha_claim=0.005, pool_decide=False)  # the paper config, ladder off

RUNS = {
    "prometheus_common": RMITDUET / "data/runs/prometheus_common-seed0",
    "compress_flate":    RMITDUET / "data/runs/compress_flate-seed0",
    "gotoml":            RMITDUET / "data/runs/gotoml-seed0",
    "zap":               RMITDUET / "data/runs/zap-seed0",
}
# First three rows are the values frozen into tab:govalid; zap is guarded here too
# but stays out of the paper table. verify() must reproduce all of these. The
# universe is RMIT-collected arms only (datasets._GODUET_MODES), matching the
# headline experiments. aa = worse of sel/ho under band-edge i.i.d.
# resampling (prometheus_common: 1 claim in 340 sel decisions).
FROZEN = {
    "prometheus_common": dict(arms=68, phys=3, aa=0.0029, recall=1.000, fp=0.0, saved=87.7),
    "compress_flate":    dict(arms=36, phys=4, aa=0.0, recall=0.750, fp=0.0, saved=83.4),
    "gotoml":            dict(arms=70, phys=8, aa=0.0, recall=1.000, fp=0.0, saved=86.5),
    "zap":               dict(arms=80, phys=6, aa=0.0, recall=0.833, fp=0.0, saved=83.2),
}
# etcd (the 5th goduet corpus) is deliberately NOT guarded here: its recording fails
# verify()'s reference-contamination gate: the same-version arm rmit/aa/
# BenchmarkStorePutUpdate carries a +7.5% mean duet diff (99% CI +1.8..+13.8%), i.e.
# the recording itself drifted mid-run. Its staged-data integrity is still anchored
# by rmitduet_etcd-replication_streams.npz via stage_rmitduet.py's bit-identity check.
EXCLUDED = {"etcd": "reference contaminated: rmit/aa/BenchmarkStorePutUpdate at +7.5%"}

_EXTRACT = r'''
import pyarrow.parquet as pq, glob, collections, numpy as np, sys
run_dir, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(run_dir + "/samples/pass_*.parquet")):
    t = pq.read_table(f).to_pydict()
    rows.extend({k: t[k][i] for k in t} for i in range(len(t["ts_ns"])))
g = collections.defaultdict(lambda: collections.defaultdict(lambda: {1: [], 2: []}))
for r in rows:
    if not r["mode"]:
        continue
    g[(r["mode"], r["pairing"], r["bench"])][r["pass_idx"]][r["sut"]].append(
        (r["unit_seq"], r["iter"], r["value_ns"]))
streams = {}
for key, passes in g.items():
    d = []
    for p in sorted(passes):
        s1 = [v for _, _, v in sorted(passes[p][1])]
        s2 = [v for _, _, v in sorted(passes[p][2])]
        d += [(v1 - v2) / (v1 + v2) for v1, v2 in zip(s1, s2) if v1 > 0 and v2 > 0]
    if d and len(d) >= 100:
        streams[f"{key[0]}/{key[1]}/{key[2]}"] = np.asarray(d)
np.savez(out, **streams)
print(f"  extracted {len(streams)} arms")
'''


def extract(run_dir: Path, npz: Path) -> None:
    try:
        import pyarrow  # noqa: F401
        exe = sys.executable
    except ImportError:
        exe = str(RMITDUET / ".venv" / "bin" / "python")
    subprocess.run([exe, "-c", _EXTRACT, str(run_dir), str(npz)], check=True)


def verify(tag: str, npz: Path) -> dict:
    z = np.load(npz)
    # Same universe as the headline experiments: RMIT-collected arms only
    # (datasets._GODUET_MODES); the extracted npz caches keep every mode.
    pools = {k: np.asarray(z[k], float) for k in z.files
             if k.split("/")[0] in _GODUET_MODES}
    total = sum(len(p) for p in pools.values())
    rng = np.random.default_rng(0)
    ref = {}
    # full-recording bootstrap reference: this is the ground truth everything is scored against
    for a, p in pools.items():
        ref[a] = common.boot_change(common.boot_means(p, rng, BOOT), EPS_D, ALPHA)
    pos = [a for a, r in ref.items() if r]
    assert not any("/aa/" in a for a in pos), "A/A arm flagged positive: reference contaminated"
    phys = {a.split("/", 2)[2] for a in pos}  # dedup by benchmark name

    # Band-edge A/A: pin the POOL (population) mean at the edge, then draw fresh
    # i.i.d. samples per seed (mode="rmit", the aa_scheduler.py methodology). Ordered
    # replay is useless here: it is seed-invariant, so it yields one deterministic
    # realization however many seeds loop over it.
    aa = {}
    for label, grp in (("sel", range(5)), ("ho", range(5, 10))):
        worst = 0.0
        for sgn in (+1.0, -1.0):  # take the worse of the two band edges
            null = {a: (p - p.mean()) + sgn * EPS_D for a, p in pools.items()}
            c = 0
            for seed in grp:
                s = FastScheduler(null, band=EPS_D, budget_per_arm={a: len(p) for a, p in null.items()},
                                  seed=seed, **CFG)
                dec, _ = s.run(ReplayRunner(null, mode="rmit", seed=seed))
                c += sum(1 for d in dec.values() if d == "change")
            worst = max(worst, c / (len(pools) * len(grp)))
        aa[label] = worst

    # Recorded-order run: with pool_decide off every per-arm decision (and spend) is a
    # deterministic function of that arm's recorded stream, so this is ONE realization
    # of the recording, with no seed loop pretending otherwise.
    s = FastScheduler(pools, band=EPS_D, budget_per_arm={a: len(p) for a, p in pools.items()},
                      seed=0, **CFG)
    dec, spent = s.run(ReplayRunner(pools, mode="ordered", seed=0))
    rec = sum(1 for a in pos if dec[a] == "change") / max(len(pos), 1)
    fp = sum(1 for a in pools if not ref[a] and dec[a] == "change") / max(len(pools) - len(pos), 1)
    return dict(arms=len(pools), phys=len(phys), aa=max(aa.values()),
                recall=float(rec), fp=float(fp),
                saved=100 * (1 - spent / total), aa_sel=aa["sel"], aa_ho=aa["ho"])


def main() -> int:
    ok = True
    print("suite               arms phys  A/A(sel) A/A(ho)  recall     FP   saved")
    for tag, run_dir in RUNS.items():
        npz = HERE / "results" / f"rmitduet_{tag}-replication_streams.npz"
        if not npz.exists():
            print(f"{tag}: extracting from {run_dir} ...")
            extract(run_dir, npz)
        got = verify(tag, npz)
        exp = FROZEN[tag]
        row_ok = (got["arms"] == exp["arms"] and got["phys"] == exp["phys"]
                  and abs(got["aa"] - exp["aa"]) < 5e-4  # compare to frozen, not just alpha
                  and abs(got["recall"] - exp["recall"]) < 5e-4
                  and got["fp"] == exp["fp"] and abs(got["saved"] - exp["saved"]) < 0.05)
        ok &= row_ok
        print(f"{tag:<19} {got['arms']:>4} {got['phys']:>4}   {got['aa_sel']:.4f}  {got['aa_ho']:.4f}"
              f"   {got['recall']:.3f}  {got['fp']:.4f}  {got['saved']:5.1f}%"
              f"   {'== frozen' if row_ok else '!! DRIFT vs frozen ' + str(exp)}")
    for tag, why in EXCLUDED.items():
        print(f"{tag:<19} SKIPPED ({why})")
    print("\nREPLICATION " + ("PASSED: every tab:govalid number regenerated from raw parquets."
                              if ok else "FAILED - investigate before citing the table."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
