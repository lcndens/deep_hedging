# Deep Hedging — Comparative Study in Incomplete Markets

**Thesis:** A Comparative Study of Deep Hedging Methods in Incomplete Markets  
**Author:** Lucian Densmore | UMass Amherst | Spring 2026  
**License:** MIT

---

## Overview

This repository implements and compares multiple deep hedging frameworks for pricing and hedging European call options under three market simulators: Black-Scholes (BS), Heston stochastic volatility, and Normal-Gamma (NGA) jump diffusion. The project is structured around three aims:

- **Aim 1:** Data pipeline and simulator infrastructure
- **Aim 2:** Baseline deep hedging implementation (Buehler et al., 2019)
- **Aim 3:** Extended frameworks including No-Transaction Band, Robust Deep Hedging, and Adversarial Training

All models are trained on the UMass Unity Cluster using PyTorch with CVaR at α=0.95 as the risk objective.

---

## Installation

This project runs on the UMass Unity Cluster with a micromamba environment.

```bash
# Clone the repository
git clone https://github.com/lcndens/deep_hedging
cd deep_hedging

# Create environment
micromamba create -n deep_hedging python=3.12
micromamba activate deep_hedging

# Install dependencies
pip install torch numpy pandas pyarrow scipy matplotlib pytest
```

---

## Directory Structure

```
deep_hedging/
├── src/
│   ├── simulators/          # Market simulators (BS, Heston, NGA)
│   ├── schema/              # Parquet schema definitions
│   ├── io/                  # Dataset generation and loading
│   ├── utils/               # Train/val/test splitting
│   ├── state/               # Feature engineering (state construction)
│   ├── derivatives/         # Option payoff functions
│   ├── frictions/           # Transaction cost models
│   ├── pnl/                 # PnL computation
│   ├── policy/              # Neural network hedging policy
│   ├── objective_functions/ # CVaR and other risk measures
│   ├── train/               # Training loop and checkpointing
│   ├── evaluation/          # Evaluation suite and BS delta check
│   ├── generate_dataset.py  # CLI: generate and save a dataset
│   └── plot_observations.py # CLI: visualise simulated paths
│
├── tests/                   # Pytest test suite (288 tests)
├── data/                    # Simulated datasets (git-ignored)
│   └── datasets/v1.0/<sim>/<run_id>/
├── results/                 # Training runs and evaluation (git-ignored)
│   ├── runs/<sim>/<run_name>/
│   └── evaluation/<eval_name>/
└── logs/                    # SLURM job logs (git-ignored)
```

---

## Quickstart

### 1. Generate a dataset

```bash
python -m src.generate_dataset \
    --sim bs \
    --n_paths 100000 \
    --n_steps 30 \
    --seed 42 \
    --strike 100.0 \
    --maturity_years 0.25
```

Replace `--sim bs` with `heston` or `nga` for other simulators. Datasets are saved to `data/datasets/v1.0/<sim>/<run_id>/`.

### 2. Train a model

```bash
python -m src.train.trainer \
    --dataset_dir data/datasets/v1.0/bs/<run_id> \
    --run_name    bs_baseline_frictionless \
    --sim         bs \
    --epsilon     0.0 \
    --n_epochs    1500 \
    --device      cuda
```

Training outputs are saved to `results/runs/<sim>/<run_name>/`:

```
results/runs/bs/bs_baseline_frictionless/
    checkpoints/
        best_model.pt      ← best validation loss checkpoint
        final_model.pt     ← final epoch checkpoint
    logs/
        train_log.csv      ← epoch-by-epoch train/val loss
        train.log          ← full training log
    config.json            ← all hyperparameters
```

### 3. Run the BS delta sanity check (BS only)

After training a Black-Scholes model, verify that the network has converged to the analytical BS delta:

```bash
python -m src.evaluation.bs_delta_check \
    --checkpoint  results/runs/bs/<run_name>/checkpoints/best_model.pt \
    --dataset_dir data/datasets/v1.0/bs/<run_id> \
    --sigma       0.2 \
    --out_dir     results/evaluation/<run_name>
```

A passed check (MAE < 0.02) confirms the pipeline is correct before proceeding to Heston and NGA.

### 4. Evaluate and generate results

Pass one or more completed run directories to produce Table 1 and all charts:

```bash
python -m src.evaluation.evaluate \
    results/runs/bs/bs_baseline_frictionless \
    results/runs/heston/heston_baseline_frictionless \
    results/runs/nga/nga_baseline_frictionless \
    --out_dir results/evaluation/baseline_frictionless
```

Outputs:

```
results/evaluation/baseline_frictionless/
    table1_results.csv               ← primary results table
    chart1_pnl_histogram_bs.png      ← PnL distribution per simulator
    chart1_pnl_histogram_heston.png
    chart1_pnl_histogram_nga.png
    chart2_loss_curves.png           ← training convergence
    chart4_per_timestep_bs.png       ← per-timestep delta analysis
    chart4_per_timestep_heston.png
    chart4_per_timestep_nga.png
```

---

## Training on the Unity Cluster (SLURM)

Full training runs are submitted via SLURM. The workflow for each simulator is:

```bash
# 1. Generate dataset (interactive node or login node — fast)
python -m src.generate_dataset --sim heston --n_paths 100000 ...

# 2. Update DATASET_RUN_ID in the training script
#    scripts/train_heston_baseline.sh

# 3. Submit the job
sbatch scripts/train_heston_baseline.sh

# 4. Monitor
squeue -u ldensmore_umass_edu
tail -f logs/slurm/<job_id>_heston_baseline.out
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Single module
pytest tests/test_trainer.py -v
```

288 tests across all pipeline stages. All should pass before running training.

---

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `n_steps` | 30 | Daily rebalancing over T = 30/365 yr |
| `n_paths` | 100,000 | Training paths |
| `batch_size` | 10,000 | Mini-batch size |
| `n_epochs` | 1,500 | Max training epochs |
| `early_stop_patience` | 100 | Epochs without val improvement before stopping |
| `lr` | 1e-3 | Adam learning rate |
| `alpha` | 0.95 | CVaR confidence level |
| `hidden` | 64 | Hidden units per layer (2 layers) |
| `epsilon` | 0.0 | Proportional transaction cost rate |

---

## References

- Buehler et al. (2019). *Deep Hedging.*
- He et al. (2025). *Empirical Deep Hedging.*
- Carbonneau & Godin (2021). *Equal Risk Pricing of Derivatives with Deep Hedging.*
- Lütkebohmert et al. (2022). *Robust Deep Hedging.*
- Imaki et al. (2021). *No-Transaction Band Network.*
