# BMSSP Replication

Rust implementations of SSSP [single-source shortest path] algorithms, benchmarked against Castro et al. (2025).

## Prerequisites

- **Rust** — `rustup` (stable toolchain)
- **Python** — [`uv`](https://github.com/astral-sh/uv) (manages virtualenvs and deps automatically)

## Running benchmarks

All benchmark runs are driven by `bench/run.py`. Results are written to `bench/results/` as timestamped JSON files.

**Replicate Castro et al.:**

```bash
uv run --project bench bench/run.py --config bench/castro_config.toml
```

DIMACS road graphs (USA-road-t.*) must be fetched first:

```bash
bash scripts/fetch_dimacs_roads.sh
```

**Run the default benchmark matrix:**

```bash
uv run --project bench bench/run.py
```

**Useful flags:**

| Flag | Effect |
|---|---|
| `--algorithms dijkstra,bmssp_o_6` | Restrict to specific algorithms |
| `--max-size 65536` | Cap graph size |
| `--graph-instances 1` | Override instance count |
| `--dry-run` | Print what would run, skip execution |
| `--fresh` | Ignore cached Criterion results |

## Generating figures

Figures are written to `../msc-thesis/figures/` by default.

```bash
uv run --project eval eval/thesis_plots.py bench/results/run_<timestamp>.json
```

Pass multiple result files to overlay runs:

```bash
uv run --project eval eval/thesis_plots.py bench/results/run_A.json bench/results/run_B.json
```

Use `--out-dir` to write figures elsewhere:

```bash
uv run --project eval eval/thesis_plots.py bench/results/run_<timestamp>.json --out-dir /tmp/figs
```
