# deep_hedging

Canonical dataset generation and plotting tools for simulated hedging observations.

## Dataset Format

All generated runs follow:

`data/datasets/v1.0/<sim>/<run_id>/`

Where:

- `<sim>` is `bs`, `heston`, or `nga`
- `<run_id>` is timestamped by default and includes seed, path count, and step count
  (or explicitly set with `--run_id`)

Each run contains:

- `metadata.json`
- `contracts.parquet`
- `observations/train/part-00000.parquet`
- `observations/val/part-00000.parquet`
- `observations/test/part-00000.parquet`

Split membership is encoded by subfolders (`train`, `val`, `test`). It is not inferred from any partition column.

## Observations Schema (v1.0)

Every observations parquet file contains exactly these columns:

- `path_id` (`int64`)
- `t_idx` (`int32`)
- `t_years` (`float32`)
- `S` (`float32`)

No latent simulator state (`v`, variance factors, or other hidden fields) is written to observations data.

## Environment Setup

`requirements/environment.yml` lists core dependencies. It should include:

- `python`
- `numpy`
- `pandas`
- `pyarrow`
- `pytest`
- `matplotlib`
- formatting/lint tools (`ruff`, `black`)

If you are using a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install dependencies as needed for your workflow.

## Generate Datasets

Run from project root.

Black-Scholes:

```bash
python -m src.generate_dataset --sim bs
```

Heston:

```bash
python -m src.generate_dataset --sim heston
```

NGA:

```bash
python -m src.generate_dataset --sim nga
```

Example with explicit options:

```bash
python -m src.generate_dataset \
  --sim heston \
  --out_root data/datasets/v1.0 \
  --run_id demo-heston \
  --n_paths 1000 \
  --n_steps 30 \
  --seed 7 \
  --s0 100 \
  --m 0.0 \
  --v0 0.04 \
  --kappa 1.5 \
  --theta 0.04 \
  --xi 0.3 \
  --rho -0.7
```

NGA example with explicit bounds:

```bash
python -m src.generate_dataset \
  --sim nga \
  --out_root data/datasets/v1.0 \
  --run_id demo-nga \
  --n_paths 1000 \
  --n_steps 30 \
  --seed 7 \
  --x0 100 \
  --a0_min 0.3 \
  --a0_max 0.7 \
  --a1_min 0.04 \
  --a1_max 0.06 \
  --b0_min -0.2 \
  --b0_max 0.2 \
  --b1_min -0.1 \
  --b1_max 0.1 \
  --gamma_min 0.5 \
  --gamma_max 1.5
```

## Plot Paths

```bash
python -m src.plot_observations \
  --run_dir data/datasets/v1.0/bs/<run_id> \
  --split test \
  --n_paths_plot 12
```

The PNG is written under the selected split folder:

`data/datasets/v1.0/<sim>/<run_id>/observations/<split>/`

Generate a PNG for the most recently generated dataset run:

```bash
RUN_DIR="$(ls -dt data/datasets/v1.0/*/* | head -n 1)"
python -m src.plot_observations --run_dir "$RUN_DIR" --split test --n_paths_plot 12
```

## Run Tests

```bash
pytest -q
```

Tests cover:

- strict schema enforcement in parquet writing
- generated folder layout and split counts
- required columns and dtypes
- plotting script execution and PNG output (when `matplotlib` is available)

## CLI Parameters

Common parameters:

- `--sim`: `bs`, `heston`, or `nga`
- `--out_root`: output root (default `data/datasets/v1.0`)
- `--run_id`: optional deterministic run id (default: auto timestamped id)
- `--n_paths`: number of simulated paths
- `--n_steps`: number of time steps (records use `n_steps + 1` time points including maturity)
- `--maturity_years`: horizon in years
- `--seed`: RNG seed
- `--s0`: initial spot
- `--m`: drift in the spot SDE

Black-Scholes parameter:

- `--sigma`: volatility

Heston parameters:

- `--v0`: initial variance
- `--kappa`: mean-reversion speed
- `--theta`: long-run variance level
- `--xi`: volatility-of-variance
- `--rho`: correlation between spot and variance Brownian shocks

NGA parameters:

- `--x0`: initial state
- `--a0_min`, `--a0_max`
- `--a1_min`, `--a1_max`
- `--b0_min`, `--b0_max`
- `--b1_min`, `--b1_max`
- `--gamma_min`, `--gamma_max`

Contract parameters:

- `--contract_type`: `call` or `put`
- `--strike`: strike price

Split parameters:

- `--split_train`
- `--split_val`
- `--split_test`
