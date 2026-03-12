# Deep Hedging — Pipeline Technical Reference

**Thesis:** A Comparative Study of Deep Hedging Methods in Incomplete Markets  
**Author:** Lucian Densmore | UMass Amherst | Spring 2026

---

## Table of Contents

1. [Design Decisions](#1-design-decisions)
2. [Directory Structure](#2-directory-structure)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Data Flow and Tensor Shapes](#4-data-flow-and-tensor-shapes)
5. [Module Reference](#5-module-reference)
   - [Simulators](#51-simulators)
   - [Dataset Generation](#52-dataset-generation)
   - [Data Loading](#53-data-loading)
   - [State Construction](#54-state-construction)
   - [Derivatives](#55-derivatives)
   - [Frictions](#56-frictions)
   - [PnL Computation](#57-pnl-computation)
   - [Policy Network](#58-policy-network)
   - [CVaR Objective](#59-cvar-objective)
   - [Training Loop](#510-training-loop)
   - [Evaluation](#511-evaluation)
6. [Extensibility Map](#6-extensibility-map)

---

## 1. Design Decisions

The following decisions are fixed across all thesis experiments. They are grounded in consensus across the literature and chosen to allow fair cross-framework comparison.

| Decision | Choice | Rationale |
|---|---|---|
| Heston variance `v` | Observable; stored in separate latent parquet | Standard in all papers; clean separation of observable vs latent state |
| `v` for BS / NGA | Zero placeholder | One network architecture across all three simulators — no branching |
| Network sharing | One shared network across all timesteps | Markov formulation (Buehler Remark 4.6); standard practice |
| `p0` during training | Fixed at `0.0` | Cash-invariance of CVaR; He et al. (2025) explicit standard |
| `n_steps` | 30 | Daily rebalancing over T = 30/365 yr; matches Buehler et al. (2019) |
| Dataset size | 100,000 training paths | Consistent with He et al. and Carbonneau & Godin |
| Data loading | Full CPU RAM load; mini-batch sample to GPU | Avoids disk I/O bottleneck during training |
| Optimizer | Adam | Universal across all papers |
| Risk measure | CVaR via OCE (`alpha=0.95`) | Differentiable; standard in Buehler, He, Carbonneau |
| Network architecture | 2-layer feedforward, H=64, ReLU | Baseline for all Aim 2/3 experiments |
| Contract | European call, K=100, T=0.25 yr | Simplicity; standard in literature |

---

## 2. Directory Structure

```
deep_hedging/
│
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py        ← CLI entrypoint: simulate and save a dataset
│   ├── plot_observations.py       ← CLI: visualise simulated price paths
│   │
│   ├── simulators/                ← market price simulators
│   │   ├── __init__.py
│   │   ├── bs.py                  ← Black-Scholes GBM
│   │   ├── heston.py              ← Heston stochastic volatility
│   │   └── nga.py                 ← Normal-Gamma jump diffusion
│   │
│   ├── schema/                    ← parquet column schema definitions
│   │   ├── __init__.py
│   │   └── v1_0.py
│   │
│   ├── io/                        ← disk I/O
│   │   ├── __init__.py
│   │   ├── parquet_writer.py      ← write simulated paths to parquet
│   │   └── dataset_loader.py      ← load parquet → DatasetBatch tensors
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── split.py               ← train/val/test path splitting
│   │
│   ├── state/
│   │   ├── __init__.py
│   │   └── builder.py             ← feature engineering → (N, T, 3) tensor
│   │
│   ├── derivatives/
│   │   ├── __init__.py
│   │   └── european.py            ← call/put payoff functions
│   │
│   ├── frictions/
│   │   ├── __init__.py
│   │   └── proportional.py        ← proportional transaction cost
│   │
│   ├── pnl/
│   │   ├── __init__.py
│   │   └── compute.py             ← terminal PnL from deltas and paths
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   └── baseline_feedforward_network.py  ← hedging policy network
│   │
│   ├── objective_functions/
│   │   ├── __init__.py
│   │   └── cvar.py                ← CVaR loss via OCE formulation
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   └── trainer.py             ← training loop, checkpointing, CLI
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── bs_delta_check.py      ← sanity check: network vs BS analytical delta
│       └── evaluate.py            ← full evaluation suite: Table 1, Charts 1/2/4
│
├── tests/                         ← pytest test suite (288 tests)
│   ├── conftest.py                ← shared fixtures
│   ├── test_dataset_loader.py
│   ├── test_builder.py
│   ├── test_stage3.py
│   ├── test_baseline_feedforward_network.py
│   ├── test_cvar.py
│   ├── test_trainer.py
│   └── test_bs_delta_check.py
│
├── data/                          ← git-ignored
│   └── datasets/v1.0/<sim>/<run_id>/
│       ├── metadata.json
│       ├── contracts.parquet
│       ├── observations/
│       │   ├── train/part-00000.parquet
│       │   ├── val/part-00000.parquet
│       │   └── test/part-00000.parquet
│       └── latent_state/
│           ├── train/part-00000.parquet
│           ├── val/part-00000.parquet
│           └── test/part-00000.parquet
│
├── results/                       ← git-ignored
│   ├── runs/<sim>/<run_name>/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   └── final_model.pt
│   │   ├── logs/
│   │   │   ├── train_log.csv
│   │   │   └── train.log
│   │   └── config.json
│   └── evaluation/<eval_name>/
│       ├── table1_results.csv
│       ├── chart1_pnl_histogram_<sim>.png
│       ├── chart2_loss_curves.png
│       └── chart4_per_timestep_<sim>.png
│
├── logs/                          ← git-ignored
│   └── slurm/
│
├── README.md
├── PIPELINE.md
├── LICENSE
└── .gitignore
```

---

## 3. Pipeline Overview

The pipeline has two phases: **data generation** (run once per simulator) and **training** (run per experiment). Both phases are fully deterministic given a fixed seed.

```
┌─────────────────────────────────────────────────────────┐
│                    DATA GENERATION                      │
│                                                         │
│  Simulator → parquet_writer → disk (data/datasets/)    │
│  (BS / Heston / NGA)                                    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                      TRAINING                           │
│                                                         │
│  dataset_loader → build_features → policy network      │
│       ↓                                  ↓             │
│  paths_S (N,T+1)               deltas (N,T)            │
│       ↓                                  ↓             │
│  call_payoff ──────────────► compute_pnl → CVaR loss   │
│  proportional_cost ──────────────────────────►         │
│                                          ↓             │
│                               loss.backward()          │
│                               Adam step                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     EVALUATION                          │
│                                                         │
│  best_model.pt + test split → metrics + charts         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow and Tensor Shapes

Every tensor flowing through the pipeline uses a consistent shape convention. `N` is the number of paths, `T` is the number of timesteps (30).

| Tensor | Shape | Dtype | Description |
|---|---|---|---|
| `paths_S` | `(N, T+1)` | float32 | Spot price at each timestep including maturity |
| `paths_v` | `(N, T+1)` | float32 | Variance process (Heston); zero for BS/NGA |
| `paths_t` | `(N, T+1)` | float32 | Calendar time at each step (0.0 → T_mat) |
| `features` | `(N, T, 3)` | float32 | Network input: [S/K, tau, v] per step |
| `deltas` | `(N, T)` | float32 | Hedge ratios output by policy network |
| `payoff` | `(N,)` | float32 | Call option payoff at maturity |
| `total_cost` | `(N,)` | float32 | Total proportional transaction costs |
| `pnl` | `(N,)` | float32 | Terminal PnL per path |
| `loss` | `scalar` | float32 | CVaR estimate — backprop entry point |

The three features fed to the network at each timestep are:

1. **Moneyness** `S_t / K` — normalised spot price
2. **Time to maturity** `tau_t = T_mat - t` — remaining option life
3. **Variance** `v_t` — Heston variance; zero placeholder for BS and NGA

---

## 5. Module Reference

### 5.1 Simulators

**`src/simulators/bs.py`**  
Black-Scholes geometric Brownian motion. Parameters: `S0=100`, `mu=0.0`, `sigma=0.2`, `r=0.0`. Writes `v=0.0` placeholder to latent state parquet.

**`src/simulators/heston.py`**  
Heston stochastic volatility model. Euler-Maruyama discretisation. Parameters: `S0=100`, `v0=0.04`, `kappa=2.0`, `theta=0.04`, `xi=0.3`, `rho=-0.7`. Writes true variance `v_t` to latent state parquet — the network observes this as an input feature.

**`src/simulators/nga.py`**  
Normal-Gamma jump diffusion (variance-gamma process). Writes `v=0.0` placeholder. Captures skewness and excess kurtosis absent from GBM.

---

### 5.2 Dataset Generation

**`src/generate_dataset.py`** — CLI entrypoint.

Simulates `n_paths` price trajectories and writes them to disk in long-format parquet. Applies the `utils/split.py` train/val/test split (70/15/15 by default). Each dataset is identified by a `run_id` generated from the simulation parameters.

```bash
python -m src.generate_dataset \
    --sim            bs \
    --n_paths        100000 \
    --n_steps        30 \
    --seed           42 \
    --strike         100.0 \
    --maturity_years 0.25
```

**Output layout:**

```
data/datasets/v1.0/bs/<run_id>/
    metadata.json              ← sim params, n_paths, seed, schema version
    contracts.parquet          ← K, T_mat per path
    observations/
        train/part-00000.parquet   ← columns: path_id, step, S, t
        val/part-00000.parquet
        test/part-00000.parquet
    latent_state/
        train/part-00000.parquet   ← columns: path_id, step, v
        val/part-00000.parquet
        test/part-00000.parquet
```

---

### 5.3 Data Loading

**`src/io/dataset_loader.py`**

Loads a parquet dataset from disk and pivots it from long format (one row per path × step) to wide format tensors. Returns a `DatasetBatch` dataclass. The full dataset is loaded into CPU RAM once at training startup — no repeated disk reads during training.

```python
@dataclass
class DatasetBatch:
    paths_S:  torch.Tensor   # (N, T+1)  spot prices
    paths_v:  torch.Tensor   # (N, T+1)  variance (zero for BS/NGA)
    paths_t:  torch.Tensor   # (N, T+1)  calendar time
    K:        float           # strike price
    T_mat:    float           # maturity in years
    n_paths:  int
    n_steps:  int
```

Usage:
```python
from src.io.dataset_loader import load_dataset
batch = load_dataset(Path("data/datasets/v1.0/bs/<run_id>"), split="train")
```

---

### 5.4 State Construction

**`src/state/builder.py`**

Constructs the `(N, T, 3)` feature tensor from a `DatasetBatch`. The three features per timestep are moneyness, time-to-maturity, and variance.

```python
# features[:, t, :] = [S_t / K,  T_mat - t,  v_t]
features = build_features(batch)   # (N, T, 3)
```

This is the only place where raw paths are transformed into network inputs. All three simulators produce the same feature shape — BS and NGA simply have zero in the variance slot.

---

### 5.5 Derivatives

**`src/derivatives/european.py`**

Computes option payoffs at maturity. No parameters.

```python
payoff = call_payoff(S_T, K)   # max(S_T - K, 0),  shape (N,)
payoff = put_payoff(S_T, K)    # max(K - S_T, 0),   shape (N,)
```

---

### 5.6 Frictions

**`src/frictions/proportional.py`**

Computes total proportional transaction costs over the full trajectory. Cost is charged on the absolute change in position at each rebalancing step.

```
C_T = epsilon * sum_t [ S_t * |delta_t - delta_{t-1}| ]
```

```python
total_cost = proportional_cost(paths_S[:, :-1], deltas, epsilon=0.01)  # (N,)
```

When `epsilon=0.0` (frictionless), returns a zero tensor.

---

### 5.7 PnL Computation

**`src/pnl/compute.py`**

Computes terminal PnL for each path using the Buehler et al. (2019) formula:

```
PnL_i = p0 + sum_t [ delta_t * (S_{t+1} - S_t) ] - C_T - payoff(S_T)
```

```python
pnl = compute_pnl(paths_S, deltas, payoff, total_cost, p0=0.0)  # (N,)
```

`p0=0.0` during training by cash-invariance of CVaR. Positive PnL means the hedge was profitable. The goal is to minimise the CVaR of the negative tail of this distribution.

---

### 5.8 Policy Network

**`src/policy/baseline_feedforward_network.py`**

The primary baseline network for all Aim 2 and Aim 3 experiments. A 2-layer feedforward network with semi-recurrent delta feedback.

**Architecture:**

```
input (4) → Linear(4→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → delta_t
```

Input dimension is 4: the 3 features from state construction plus `delta_{t-1}` (previous hedge ratio). This makes the strategy path-dependent without requiring an RNN — the delta feedback encodes the current position, which is necessary for computing transaction costs correctly.

At `t=0`, `delta_{-1} = 0` (no initial position).

**Key properties:**
- ~8,513 trainable parameters at default `H=64`
- Shared weights across all T=30 timesteps (Markov assumption)
- No final activation — delta can take any real value (unbounded)
- Gradient flows back through all 30 unrolled timesteps via autograd

```python
net = BaselineFeedforwardNetwork(hidden=64)
deltas = net.forward_trajectory(features)   # features: (N,T,3) → deltas: (N,T)
```

---

### 5.9 CVaR Objective

**`src/objective_functions/cvar.py`**

CVaR at confidence level α=0.95, implemented via the Optimised Certainty Equivalent (OCE) formulation (Rockafellar & Uryasev, 2002):

```
L(θ, ω) = ω + E[ max(-PnL - ω, 0) ] / (1 - α)
```

`ω` is a trainable scalar parameter that converges to `VaR_α` at the optimum. It is jointly optimised with the network weights in the same Adam step. The `clamp` operation focuses the loss only on paths where the hedge failed — where the loss exceeded `ω`.

```python
cvar = CVaRLoss(alpha=0.95)
loss = cvar(pnl)        # scalar, differentiable
loss.backward()

# Diagnostic accessors (no grad)
cvar.cvar_estimate(pnl)   # float
cvar.var_estimate()        # float — current ω value
```

---

### 5.10 Training Loop

**`src/train/trainer.py`**

Orchestrates the full training pipeline. Configuration is passed as a `TrainConfig` dataclass or via CLI arguments.

**Key design choices:**
- Full dataset loaded to CPU RAM once; mini-batches sampled and moved to GPU each step
- Per epoch: dataset is shuffled, then iterated in mini-batches of 10,000 — every path is seen once per epoch
- Validation is run on the full val split in one forward pass with `torch.no_grad()`
- Early stopping on val CVaR loss with patience=100
- `best_model.pt` is saved whenever val loss improves; `final_model.pt` is saved at the end regardless

**CLI:**

```bash
python -m src.train.trainer \
    --dataset_dir  data/datasets/v1.0/bs/<run_id> \
    --run_name     bs_baseline_frictionless \
    --sim          bs \
    --epsilon      0.0 \
    --n_epochs     1500 \
    --lr           1e-3 \
    --batch_size   10000 \
    --alpha        0.95 \
    --hidden       64 \
    --early_stop_patience 100 \
    --seed         42 \
    --out_root     results/runs \
    --device       cuda
```

**Output structure:**

```
results/runs/<sim>/<run_name>/
    checkpoints/
        best_model.pt      ← state_dict: model, cvar, optimizer, epoch
        final_model.pt
    logs/
        train_log.csv      ← epoch, train_loss, val_loss, omega
        train.log
    config.json
```

**Loading a checkpoint:**

```python
import torch
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork

net  = BaselineFeedforwardNetwork(hidden=64)
ckpt = torch.load("results/runs/bs/bs_baseline_frictionless/checkpoints/best_model.pt",
                  map_location="cpu", weights_only=True)
net.load_state_dict(ckpt["model_state_dict"])
net.eval()
```

---

### 5.11 Evaluation

#### BS Delta Sanity Check

**`src/evaluation/bs_delta_check.py`**

Verifies that a model trained on Black-Scholes data has converged to the analytical BS delta. This is the primary correctness check for the pipeline — if the network has learned the right strategy under the simplest possible model, the architecture and training procedure are correct.

Passes if MAE < 0.02 between network delta and `N(d1)` across all paths and timesteps.

```bash
python -m src.evaluation.bs_delta_check \
    --checkpoint  results/runs/bs/<run_name>/checkpoints/best_model.pt \
    --dataset_dir data/datasets/v1.0/bs/<run_id> \
    --sigma       0.2 \
    --out_dir     results/evaluation/<run_name>
```

#### Full Evaluation Suite

**`src/evaluation/evaluate.py`**

Runs the complete evaluation across one or more trained runs. Produces:

- **Table 1** (`table1_results.csv`): CVaR₀.₉₅, VaR₀.₉₅, mean PnL, std PnL, P10 PnL per run
- **Chart 1** (`chart1_pnl_histogram_<sim>.png`): PnL distribution histogram. For BS: overlays analytical BS delta benchmark. For Heston/NGA: deep hedge only.
- **Chart 2** (`chart2_loss_curves.png`): Train and val CVaR loss over epochs for all runs on one figure
- **Chart 4** (`chart4_per_timestep_<sim>.png`): Per-timestep delta standard deviation; for BS also overlays MAE vs analytical delta

```bash
python -m src.evaluation.evaluate \
    results/runs/bs/bs_baseline_frictionless \
    results/runs/heston/heston_baseline_frictionless \
    results/runs/nga/nga_baseline_frictionless \
    --out_dir results/evaluation/baseline_frictionless
```

---

## 6. Extensibility Map

Each Aim 3 framework modifies at most one or two modules. All other modules are shared unchanged.

| Framework | Module(s) Changed | Change Description |
|---|---|---|
| No-Transaction Band (Imaki 2021) | `policy/` | Network outputs `(b_lower, b_upper)` bands; delta = clamp(delta_{t-1}, bl, bu) |
| Mean-Variance (Cao 2021) | `objective_functions/` | New `mean_variance.py`; loss = −E[PnL] + λ·Var[PnL] |
| Robust Deep Hedging (Lütkebohmert 2022) | `train/` | Sample fresh NGA parameters per path per step during training |
| Adversarial Training (He 2025) | `train/` | Add FGSM/PGD perturbation to features before each forward pass |
| Equal Risk Pricing (Carbonneau 2021) | `policy/` + `train/` | Two networks (long + short); joint CVaR minimisation |
| BS Delta Benchmark | `evaluation/` | Analytical N(d1); no training needed; plugs into shared eval metrics |
