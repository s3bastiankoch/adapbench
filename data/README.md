# Data: recorded duet-benchmark traces

The experiments replay recorded microbenchmark measurements. The traces are released
**separately from the code** (they are large and carry their own upstream licenses), so
this directory is empty in a fresh clone. Nothing here is required for a smoke test:
`python -m adabench.scheduler` runs the online scheduler on synthetic data with no
traces.

## Expected layout

The loader reads from `$ADABENCH_DATA_ROOT/traces` (default `./data/traces`):

```
data/traces/tcp8/<trace>/{trace.meta.json, trace.data.npz}     # 118 Java traces
data/traces/goduet/<corpus>/{trace.meta.json, trace.data.npz}  # 5 Go duet corpora
data/labels/{tcp8,goduet}.json                                 # label cache (auto-built)
```

Each `trace.data.npz` holds per-iteration `sut1_ns` / `sut2_ns` (the two versions'
timings) for every benchmark; the loader forms the duet difference `d=(s1-s2)/(s1+s2)`.
Labels under `data/labels/` are a deterministic cache and are rebuilt automatically the
first time a suite is loaded.

## Getting the traces

Pick one:

1. **Fetch the release archive.** Point `DATA_URL` at the published trace archive:
   ```bash
   DATA_URL=https://<release-host>/adabench-traces.tar.zst scripts/fetch_data.sh
   ```
2. **Use a local copy.** If you already have the traces somewhere, set
   `ADABENCH_DATA_ROOT` to the directory that contains `traces/`:
   ```bash
   export ADABENCH_DATA_ROOT=/path/to/adabench-data
   ```

## Suites and sources

| suite | traces | size | source |
|---|---|---|---|
| **goduet** | 5 (prometheus_common / compress_flate / gotoml / zap / etcd) | ~10 MB | our own GCP RMIT-collected duet corpora (`experiments/datasets/stage_rmitduet.py`) |
| **tcp8** | 118 (8 Java projects) | ~378 MB | derived from the `microbench-tcp` replication package (per-arm pool ≤ 1500) |

The archive ships all 5 goduet corpora, but the loader evaluates 4: etcd is
excluded because its reference recording is contaminated (see
`experiments/rmitduet_replication.py`).

`tcp8` is the deep, low-SNR suite that carries the statistics; `goduet` is the small,
high-SNR suite and is fast enough for a quick end-to-end check
(`python experiments/scheduler_savings_pertrace.py goduet`).

## Publishing the archive (maintainers)

`scripts/pack_data.sh` bundles the `tcp8` and `goduet` suite directories into the
`.tar.zst` archive that `fetch_data.sh` expects; upload it and set `DATA_URL` to its
URL.
