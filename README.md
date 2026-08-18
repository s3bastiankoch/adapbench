# AdaBench

Adaptive online scheduling for microbenchmark suites. Instead of running every
benchmark for a fixed number of iterations, AdaBench spends the measurement
budget adaptively: it allocates samples to the benchmarks nearest a decision
and stops each one as soon as an anytime-valid confidence sequence resolves it.
The result matches an exhaustive run at a fraction of the cost, under a
measured type-I error guarantee.

This repository is the artifact for the i.i.d. / RMIT (independent-collection)
evaluation.

## Method

Each benchmark is an *arm*. On the duet difference `d = (s1 - s2)/(s1 + s2)`
between two versions, a benchmark counts as a relevant change when `|mean(d)|`
exceeds a ±3% band. The policy has three parts:

- **Stopping.** An anytime-valid betting confidence sequence runs
  sample-by-sample. It can claim early (effect beyond the band), cut early
  (confidently inside the band, via futility or a certified budget stop), or,
  if neither fires, run the benchmark to its full recording.
- **Deciding.** Benchmarks that reach full depth are decided by a
  distribution-free 99% bootstrap CI (with replacement) on the mean; a
  benchmark is flagged iff the CI clears the ±3% band.
- **Allocating.** `coldmag` ranks arms by their estimated samples-to-claim,
  with a magnitude cold start, so detections are front-loaded and flat arms get
  only a light probe.

Recall and false positives are scored as agreement with the full run (every
benchmark measured to its full pool). Confidence is 99% (`alpha = 0.01`).

## Result

The scheduler runs independently per version-pair trace, the CI-run deployment
unit, so no trace subsidizes another's budget. At matched recall against a
validity-matched uniform baseline (fixed depth per recall level, its α
tightened until its band-edge A/A holds), it returns a large fraction of the
measurement budget with false positives no worse
(`experiments/results/scheduler_savings_pertrace.out`):

| suite | saved @ 85% / 90% / 95% recall | self-terminating | A/A false-claim | detection vs uniform |
|---|---|---|---|---|
| tcp8 (Java, deep pools) | 40% / 42% / 35% | saved 30%, recall 0.959 | 0.0063 ≤ α | 1.1 to 1.7× faster |
| goduet (Go, RMIT) | 19% / 83% / 47% | saved 85%, recall 1.000 | 0.0079 ≤ α | 1.6 to 5.9× faster |

Validity is measured rather than assumed: a band-edge A/A test (shift each
arm's mean onto ±band, the tightest null, and run the end-to-end scheduler)
confirms the false-claim rate stays ≤ α on both suites (worse edge;
`aa_scheduler.py`), and the calibrated α split is checked held-out on fresh
seeds.

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -e .
python -m adabench.scheduler                          # smoke test: no data needed
scripts/fetch_data.sh                                 # stage the traces (see data/README.md)
.venv/bin/python experiments/scheduler_savings_pertrace.py tcp8 goduet
```

The traces are released separately. `scripts/fetch_data.sh` stages them, or
point `ADABENCH_DATA_ROOT` at a local copy; see
[`data/README.md`](data/README.md). The scheduler smoke test runs on synthetic
data and needs no traces.

`scheduler_savings_pertrace.py` runs the real online scheduler independently
per version-pair trace (the CI-run deployment unit) and prints the
self-terminating savings/recall/FP, the matched-recall savings table against
the validity-matched uniform baseline, the detection-latency table, and the
same runs scored within each project (`results/perproject_savings.json`). Runs
are deterministic (all seeds fixed). TCP takes minutes, Go seconds.
`aa_scheduler.py` measures the band-edge A/A false-claim rate of the same
end-to-end scheduler run.

## Datasets

- **tcp8**: Java microbenchmarks, 118 recorded duet traces, ~8k arms, pool
  median ~1500 samples/arm. Low-SNR; carries the statistics.
- **goduet**: our own GCP RMIT-collected Go duet corpora (prometheus_common /
  compress_flate / gotoml / zap; a fifth recorded corpus, etcd, is excluded
  because its reference recording is contaminated — see
  `experiments/rmitduet_replication.py`), 254 RMIT arms evaluated. High-SNR.

The suites are unlabeled recordings; ground-truth labels are derived from the
full-pool bootstrap verdict over the entire recording. Layout, sizes, and how
to obtain each suite are in [`data/README.md`](data/README.md).

## Deploying the scheduler

The online scheduler is a self-contained, runnable component with a pluggable
measurement engine. Implement `Runner.measure(arm)` for your harness and it
plugs in unchanged; no historical data is required:

```python
from adabench.scheduler import Scheduler, ReplayRunner

sched = Scheduler(pools.keys(), block={a: gated_block[a] for a in pools},
                  budget_per_arm=1490)
decisions, spent = sched.run(ReplayRunner(pools))   # or your own Runner
# decisions[arm] in {"change", "no_change", "undecided"}
```

Each step allocates to the benchmark nearest a decision, measures it once,
folds the sample (block-averaged) into that benchmark's confidence sequence,
and retires it on a claim, futility cut, or certified budget stop. State is
O(1) per benchmark. `ReplayRunner` drives it from recorded pools (the RMIT
scenario used in the paper); a live deployment supplies a `Runner` that
executes the actual benchmark.

## Repo layout

- `experiments/scheduler_savings_pertrace.py`: the headline evaluation above,
  the online scheduler run per version-pair trace against the validity-matched
  uniform baseline; the same runs also produce the per-project savings table.
- `experiments/aa_scheduler.py`: the validity experiment. The band-edge A/A
  false-claim rate of the end-to-end scheduler must stay ≤ α, checked held-out
  on fresh seeds and decomposed per claim route and per project;
  `split_calibration.py` and `uniform_validity.py` calibrate the α split and
  the baseline's validity-matched levels.
- `experiments/scheduler_overhead.py`: per-measurement cost of the online
  scheduler (arm selection plus anytime-valid update), timed by replaying the
  recorded suites with pre-drawn values so the run contains only scheduler
  work (~9 µs per measurement).
- `src/adabench/`: the library. `scheduler.py` is the deployable prototype
  (the online loop shown above); `cs.py` the betting confidence sequence;
  `datasets.py`/`trace.py` data loading and labelling.
- `tests/test_core.py`: unit tests.

## Threats to validity

- **Independent collection.** Results assume RMIT / randomized measurement
  order. The i.i.d. resampling of each arm's recorded marginal further assumes
  a stationary marginal over the window, which is the main threat to the
  simulation.
- **Replay, not deployment.** These are high-fidelity replays of recorded
  traces, not a live CI integration measuring billed VM-hours.
- **Between-invocation variance** is omitted (the resample draws from one
  recording), which is optimistic for benchmarks with fork-to-fork variance.
