# AdapBench

Adaptive online scheduling for microbenchmark suites. Instead of running every
benchmark for a fixed number of iterations, AdapBench allocates samples to the
benchmarks nearest a decision and stops each one as soon as an anytime-valid
confidence sequence resolves it, matching an exhaustive run at a fraction of
the cost under a measured type-I error guarantee. The method, results, and
threats to validity are described in the accompanying paper; this repository
is the implementation and evaluation harness.

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -e .
python -m adabench.scheduler                          # smoke test: no data needed
scripts/fetch_data.sh                                 # stage the traces (see data/README.md)
.venv/bin/python experiments/scheduler_savings_pertrace.py tcp8 goduet
```

The traces (tcp8: sequentially recorded Java microbenchmark traces; goduet:
our RMIT-collected Go corpora) are released separately. `scripts/fetch_data.sh` stages them, or
point `ADABENCH_DATA_ROOT` at a local copy; layout, sizes, and provenance are
in [`data/README.md`](data/README.md). Runs are deterministic (all seeds
fixed); tcp8 takes minutes, goduet seconds. The scheduler smoke test runs on
synthetic data and needs no traces.

## Experiments

- `experiments/scheduler_savings_pertrace.py`: the headline evaluation — the
  online scheduler run independently per version-pair trace against the
  validity-matched uniform baseline; prints the self-terminating and
  matched-recall savings tables and writes the per-project scoring
  (`results/perproject_savings.json`).
- `experiments/aa_scheduler.py`: the validity experiment — band-edge A/A
  false-claim rate of the end-to-end scheduler, checked held-out on fresh
  seeds; `split_calibration.py` and `uniform_validity.py` calibrate the α
  split and the baseline's validity-matched levels.
- `experiments/detection_latency.py`: detection latency of the deployed
  allocation vs round-robin — at what budget fraction each policy surfaces a
  given share of the true changes.
- `experiments/scheduler_overhead.py`: per-measurement scheduler cost, timed
  with pre-drawn values so the run contains only scheduler work.

## Deploying the scheduler

The online scheduler is a self-contained component with a pluggable
measurement engine. Implement `Runner.measure(arm)` for your harness and it
plugs in unchanged; no historical data is required:

```python
from adabench.scheduler import Scheduler, ReplayRunner

sched = Scheduler(pools.keys(), block={a: gated_block[a] for a in pools},
                  budget_per_arm=1490)
decisions, spent = sched.run(ReplayRunner(pools))   # or your own Runner
# decisions[arm] in {"change", "no_change", "undecided"}
```

`ReplayRunner` drives it from recorded pools; a live deployment supplies a
`Runner` that executes the actual benchmark. State is O(1) per benchmark.

## Repo layout

- `src/adabench/`: the library. `scheduler.py` is the deployable online loop
  shown above; `cs.py` the betting confidence sequence; `datasets.py`/
  `trace.py` data loading and labelling.
- `experiments/`: the evaluation scripts above.
- `tests/test_core.py`: unit tests.
