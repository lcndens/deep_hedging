# Deep Hedging Framework — Pipeline Architecture

**Thesis:** A Comparative Study of Deep Hedging Methods in Incomplete Markets
**Author:** Lucian Densmore | UMass Amherst | Spring 2026
**Baseline:** Buehler et al. (2019) | PyTorch | `n_steps=30` | `N=100,000 paths`

---

## Table of Contents

1. [Resolved Design Decisions](#1-resolved-design-decisions)
2. [Directory Structure](#2-directory-structure)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage-by-Stage Specification](#4-stage-by-stage-specification)
   - [Stage 0 — Data on Disk](#stage-0--data-on-disk)
   - [Stage 1 — Data Loading](#stage-1--data-loading)
   - [Stage 2 — State Construction](#stage-2--state-construction)
   - [Stage 3 — Derivatives](#stage-3--derivatives)
   - [Stage 4 — Frictions](#stage-4--frictions)
   - [Stage 5 — Policy Network](#stage-5--policy-network)
   - [Stage 6 — PnL Computation](#stage-6--pnl-computation)
   - [Stage 7 — Objective Function](#stage-7--objective-function)
   - [Stage 8 — Training Loop](#stage-8--training-loop)
5. [Complete Shape Flow](#5-complete-shape-flow)
6. [Module Interaction Contract](#6-module-interaction-contract)
7. [Extensibility Map](#7-extensibility-map-aim-3-frameworks)
8. [Implementation Order](#8-implementation-order)

---

## 1. Resolved Design Decisions

The following decisions are pinned for all thesis experiments, grounded in consensus across the literature and optimized for fair cross-framework comparison in Aim 3.

| Decision | Choice | Rationale |
|---|---|---|
| `v` treatment (Heston) | Observable; stored in separate latent parquet | Standard in all papers; Option B preserves clean obs schema |
| `v` for BS / NGA | Zero placeholder; F stays = 3 | One network architecture across all simulators — no branching |
| Network sharing | One shared network across all timesteps | Markov formulation (Buehler Remark 4.6); standard practice |
| `p0` during training | Fixed at `0.0` | Cash-invariance of CVaR; He et al. (2025) explicit standard |
| `n_steps` | `30` (daily rebalancing, T = 30/365 yr) | Matches Buehler et al. (2019) exactly |
| Dataset size | 100,000 training paths | Consistent with He et al. and Carbonneau & Godin |
| Data loading | Full CPU RAM load; batch-sample to GPU | Avoids disk I/O during training; fits Unity memory |
| Optimizer | Adam | Universal across all papers in the literature |
| Risk measure | CVaR via OCE formulation (`alpha=0.95`) | Differentiable; standard in Buehler, He, Carbonneau |

---

## 2. Directory Structure

```
deep_hedging/
├── data/
│   └── datasets/v1.0/<sim>/<run_id>/
│       ├── metadata.json
│       ├── contracts.parquet
│       ├── observations/
│       │   ├── train/part-00000.parquet
│       │   ├── val/part-00000.parquet
│       │   └── test/part-00000.parquet
│       └── latent_state/                  ← NEW (Aim 2)
│           ├── train/part-00000.parquet
│           ├── val/part-00000.parquet
│           └── test/part-00000.parquet
├── src/
│   ├── schema/
│   │   └── v1_0.py                        ← add LATENT_STATE_SCHEMA
│   ├── io/
│   │   ├── parquet_writer.py              ← existing
│   │   └── dataset_loader.py              ← NEW
│   ├── simulators/
│   │   ├── bs.py                          ← update: write v=0.0 to latent state
│   │   ├── heston.py                      ← update: expose v array
│   │   └── nga.py                         ← update: write v=0.0 to latent state
│   ├── state/
│   │   └── builder.py                     ← NEW
│   ├── derivatives/
│   │   └── european.py                    ← NEW
│   ├── frictions/
│   │   └── proportional.py                ← NEW
│   ├── pnl/
│   │   └── compute.py                     ← NEW
│   ├── policy/
│   │   └── network.py                     ← NEW
│   ├── objective_functions/
│   │   └── cvar.py                        ← NEW
│   ├── train/
│   │   └── trainer.py                     ← NEW
│   └── baselines/
│       └── black_scholes_delta.py         ← NEW (benchmark)
└── experiments/
    └── <run_name>/
        ├── checkpoints/best_model.pt
        ├── logs/train_loss.csv
        └── config.json
```

---

## 3. Pipeline Overview

```
DISK (long-format parquet)
         │
         ▼  Stage 1: dataset_loader.py
         │  Pivot long→wide. Load to CPU RAM once before training.
         │
         ├── paths_S   (N, T+1)   spot prices
         ├── paths_v   (N, T+1)   variance  [0.0 for BS/NGA]
         ├── paths_t   (N, T+1)   time in years
         └── K, T_mat  scalars
         │
         ▼  Stage 2: state/builder.py
         │  Feature engineering. Vectorized over N and T.
         │
         └── features  (N, T, 3)
                       [:,t,0] = log(S_t / K)
                       [:,t,1] = T_mat - t_years_t
                       [:,t,2] = v_t  or  0.0
         │
         ▼  Stage 5: policy/network.py
         │  Semi-recurrent forward pass. T=30 steps.
         │  Input per step: concat(features[:,t,:], δ_{t-1}) → (N, 4)
         │
         └── deltas    (N, T)     hedge ratios δ_t for t=0..T-1
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼  Stage 4: frictions/proportional.py      ▼  Stage 3: derivatives/european.py
         │  ε · S_t · |δ_t − δ_{t-1}|.sum          │  max(S_T − K, 0)
         └── total_cost  (N,)                       └── payoff  (N,)
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼  Stage 6: pnl/compute.py
                        │  PnL = p0 + gains − costs − payoff
                        │
                        └── pnl   (N,)
                        │
                        ▼  Stage 7: objective_functions/cvar.py
                        │  L(θ,ω) = ω + mean(max(−PnL−ω, 0)) / (1−α)
                        │
                        └── loss  scalar
                        │
                        ▼  Stage 8: train/trainer.py
                           loss.backward()
                           optimizer.step()  ← updates network weights + ω
```

---

## 4. Stage-by-Stage Specification

---

### Stage 0 — Data on Disk
**Files:** `src/schema/v1_0.py` | `src/generate_dataset.py`

**Purpose:** Persistent storage of all simulated market data. The single source of truth. Nothing downstream ever modifies these files.

#### File Schemas

| File | Columns | Notes |
|---|---|---|
| `observations` | `path_id · t_idx · t_years · S` | Long format, one row per path × timestep |
| `latent_state` | `path_id · t_idx · v` | Actual `v` for Heston; `0.0` for BS/NGA — **NEW** |
| `contracts` | `contract_id · type · strike · maturity_years` | Single-row parquet |
| `metadata.json` | `simulator · seed · n_paths · n_steps · splits` | JSON config for reproducibility |

#### Required Aim 1 Changes

- `generate_dataset.py` — add latent state writer alongside existing obs writer
- `src/schema/v1_0.py` — add `LATENT_STATE_SCHEMA` constant
- `src/simulators/heston.py` — expose `v` array from `simulate_observations()`
- `src/simulators/bs.py`, `nga.py` — write zeros for `v` to keep loader uniform
- `tests/` — add latent state schema and layout checks

---

### Stage 1 — Data Loading
**File:** `src/io/dataset_loader.py`

**Purpose:** Read long-format parquet files and pivot into dense CPU tensors. Executed once at training startup — the only stage that touches disk during a training run.

#### Inputs

| Input | Type | Description |
|---|---|---|
| `run_dir` | `str` | Path to dataset run folder |
| `split` | `str` | `'train'`, `'val'`, or `'test'` |

#### Outputs

| Output | Shape / Type | Notes |
|---|---|---|
| `paths_S` | `(N, T+1)  float32` | Spot prices at all timesteps e.g. `(100000, 31)` |
| `paths_v` | `(N, T+1)  float32` | Variance at all timesteps — `0.0` for BS/NGA |
| `paths_t` | `(N, T+1)  float32` | Time in years at all timesteps |
| `K` | `scalar float` | Strike price from `contracts.parquet` |
| `T_mat` | `scalar float` | Maturity in years from `contracts.parquet` |
| `metadata` | `dict` | Run config for logging and validation |

#### Notes

- All tensors stay on **CPU** after loading. GPU transfer happens per batch inside the training loop only.
- With 100,000 paths × 31 steps, `paths_S` is approximately 12MB — trivially fits in RAM.
- Loader falls back to a zero tensor if `latent_state/` folder is absent.
- Validates `paths_t[:,0] == 0` and `paths_t[:,-1] ≈ T_mat` as a sanity check.

---

### Stage 2 — State Construction
**File:** `src/state/builder.py`

**Purpose:** Transform raw simulator output into the information process $I_t$ that the network sees at each timestep. The financial feature engineering layer. Isolated so features can be changed per experiment without touching any other module.

#### Inputs

| Input | Shape / Type | Notes |
|---|---|---|
| `paths_S` | `(N, T+1)  float32` | Spot prices |
| `paths_v` | `(N, T+1)  float32` | Variance (or zeros) |
| `paths_t` | `(N, T+1)  float32` | Time in years |
| `K` | `scalar` | Strike price |
| `T_mat` | `scalar` | Maturity in years |
| `include_v` | `bool` | `True` = Heston (real v), `False` = BS/NGA (zero slot) |

#### Feature Vector — F = 3 always

| Slot | Formula | Meaning |
|---|---|---|
| `feature[0]` | `log(S_t / K)` | Log-moneyness |
| `feature[1]` | `T_mat - t_years_t` | Time to maturity τ |
| `feature[2]` | `v_t` or `0.0` | Variance slot (zero when `include_v=False`) |

#### Output

| Output | Shape | Notes |
|---|---|---|
| `features` | `(N, T, 3)  float32` | e.g. `(100000, 30, 3)` — no maturity step included |

#### Notes

- Shape is `(N, T, 3)` not `(N, T+1, 3)`. Maturity step T has no action — used only in PnL.
- F=3 held constant across all simulators. The zero in slot `[2]` for BS/NGA is neutral — network learns to ignore it.
- All operations are fully vectorized over N and T. No Python loops.
- Future features (implied vol, realized vol) are added here and **only** here.

---

### Stage 3 — Derivatives
**File:** `src/derivatives/european.py`

**Purpose:** Define the option payoff at maturity. Isolated so payoff functions can be swapped cleanly (call, put, lookback) without affecting any other module.

#### Inputs & Output

| Name | Shape / Type | Notes |
|---|---|---|
| `S_T` (in) | `(N,)  float32` | Spot price at maturity `[= paths_S[:, -1]]` |
| `K` (in) | `scalar` | Strike price |
| `payoff` (out) | `(N,)  float32` | Option payoff per path |

#### Functions

```python
call_payoff(S_T, K) = max(S_T - K, 0)    # European call
put_payoff(S_T, K)  = max(K - S_T, 0)    # European put
```

#### Notes

- Terminal quantity — computed once per forward pass, not per timestep.
- No trainable parameters. No gradients flow through payoff (fixed target).

---

### Stage 4 — Frictions
**File:** `src/frictions/proportional.py`

**Purpose:** Compute total transaction cost for a hedging trajectory. Isolated so cost structures can be swapped per experiment: frictionless (`epsilon=0`), proportional, fixed, or complex.

#### Inputs

| Input | Shape / Type | Notes |
|---|---|---|
| `S_obs` | `(N, T)  float32` | Spot prices at rebalancing steps `[= paths_S[:, :-1]]` |
| `deltas` | `(N, T)  float32` | Hedge ratios from policy network |
| `epsilon` | `scalar float` | Proportional cost rate e.g. `0.0`, `0.001`, `0.01` |

#### Output

| Output | Shape | Notes |
|---|---|---|
| `total_cost` | `(N,)  float32` | Total transaction cost per path |

#### Formula

```python
delta_prev = concat([zeros(N,1), deltas[:,:-1]], dim=1)   # δ_{-1} = 0
cost_t     = epsilon * S_t * |delta_t - delta_prev_t|     # elementwise (N, T)
total_cost = cost_t.sum(dim=1)                            # (N,)
```

#### Notes

- `delta_{-1} = 0` — agent starts with no position before `t=0`.
- With `epsilon=0.0` this returns a zero tensor (frictionless baseline for Aim 2).
- Costs are symmetric — incurred on both buys and sells.

---

### Stage 5 — Policy Network
**File:** `src/policy/network.py`

**Purpose:** Approximate the optimal hedging strategy $\delta_t = f_\theta(I_t, \delta_{t-1})$. A single shared feedforward network applied identically at every timestep. Recurrence comes through feeding the previous hedge ratio back as an input — not through hidden state.

#### Architecture

| Component | Value | Notes |
|---|---|---|
| Input dim | `F + 1 = 4` | `log-moneyness, τ, v_or_zero, δ_{t-1}` |
| Hidden layers | `2 × Linear(64) + ReLU` | Default H=64; configurable |
| Output dim | `1` (no final activation) | Unconstrained hedge ratio `δ_t` |
| Trainable params | ~8,500 | Small and fast; fits in GPU L2 cache |
| Shared weights | Yes — same θ at all t | Markov formulation; one network for all 30 steps |

#### Forward Pass — Full Trajectory

```python
delta_prev = zeros(N)                                      # δ_{-1} = 0
deltas = []

for t in range(T):                                         # T = 30 iterations
    x_t     = cat([features[:,t,:], delta_prev.unsqueeze(-1)], dim=-1)  # (N, 4)
    delta_t = network(x_t).squeeze(-1)                     # (N,)
    deltas.append(delta_t)
    delta_prev = delta_t

deltas = stack(deltas, dim=1)                              # (N, T)
```

#### Output

| Output | Shape | Notes |
|---|---|---|
| `deltas` | `(N, T)  float32` | Hedge ratio at each of T=30 rebalancing steps |

#### Notes

- No final activation — delta is unconstrained. Network learns its own range.
- Gradient flows back through all T=30 unrolled steps via the recurrent `δ_{t-1}` input. No vanishing gradient concern at this depth.
- Shared weights across time is valid for European options on Markov price processes (Buehler Remark 4.6).

---

### Stage 6 — PnL Computation
**File:** `src/pnl/compute.py`

**Purpose:** Compute terminal profit and loss for each simulated path. Pure tensor arithmetic — no parameters, no neural network. This is the direct financial objective being optimized.

#### Formula (Buehler et al. eq. 2.1)

```
PnL_i  =  p0
        +  Σ_{t=0}^{T-1}  δ_t · (S_{t+1} − S_t)     trading gains
        −  C_T(δ)                                      transaction costs
        −  payoff(S_T)                                 liability at maturity
```

#### Inputs

| Input | Shape / Type | Notes |
|---|---|---|
| `paths_S` | `(N, T+1)  float32` | Full spot path including maturity step |
| `deltas` | `(N, T)  float32` | Hedge ratios from policy network |
| `payoff` | `(N,)  float32` | Option payoff at maturity from derivatives module |
| `total_cost` | `(N,)  float32` | Total transaction costs from frictions module |
| `p0` | `scalar float` | Initial option price — fixed at `0.0` during training |

#### Output

| Output | Shape | Notes |
|---|---|---|
| `pnl` | `(N,)  float32` | Terminal PnL per path — the quantity being risk-minimized |

#### Implementation

```python
gains      = (deltas * (paths_S[:,1:] - paths_S[:,:-1])).sum(dim=1)   # (N,)
pnl        = p0 + gains - total_cost - payoff                          # (N,)
```

#### Shape Verification

```
paths_S[:,1:]    prices at t=1..T      (N, T)
paths_S[:,:-1]   prices at t=0..T-1    (N, T)
difference       S_{t+1} - S_t         (N, T)   ✓ aligns with deltas (N, T)
```

#### Notes

- Positive PnL = profit. Negative PnL = loss. Goal is to minimize CVaR of PnL — not maximize mean.
- `p0=0.0` is correct during training by cash-invariance of CVaR; it shifts all PnL by a constant that does not affect optimization.

---

### Stage 7 — Objective Function
**File:** `src/objective_functions/cvar.py`

**Purpose:** Compute CVaR (Conditional Value at Risk) of the PnL distribution. The differentiable scalar loss driving all gradient updates. Uses the OCE formulation which makes CVaR tractable for SGD.

#### CVaR Definition

```
CVaR_α(Z)  =  expected loss in the worst (1−α) fraction of outcomes

α = 0.95   →   penalizes worst 5% of paths  (thesis default)
α = 0.0    →   reduces to mean(−PnL)         (risk-neutral)
α → 1.0    →   approaches worst-case loss
```

#### OCE Form — Deep Hedging Loss (He et al. eq. 2.5)

```
L(θ, ω)  =  ω  +  mean( max(−PnL − ω, 0) ) / (1 − α)
```

#### Implementation

```python
class CVaRLoss(nn.Module):
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.omega = nn.Parameter(torch.tensor(0.0))   # trainable scalar

    def forward(self, pnl):                            # pnl: (N,)
        excess = torch.clamp(-pnl - self.omega, min=0.0)
        loss   = self.omega + excess.mean() / (1.0 - self.alpha)
        return loss                                    # scalar
```

#### Outputs

| Output | Type | Notes |
|---|---|---|
| `loss` | `scalar float32` | Differentiable CVaR estimate — backprop entry point |
| `omega` | trainable scalar | Converges to VaR_α at optimum; joint Adam update |

#### Notes

- `omega` (ω) is a trainable `nn.Parameter` optimized alongside network weights in the same Adam step.
- `clamp(min=0)` focuses loss only on paths where the hedge failed — where loss exceeded ω.
- Full computation graph from `loss` → `pnl` → `deltas` → network weights is differentiable end-to-end via autograd.

---

### Stage 8 — Training Loop
**File:** `src/train/trainer.py`

**Purpose:** Orchestrate all stages. Manage epochs, mini-batches, GPU transfer, gradient updates, validation, logging, and checkpointing.

#### Hyperparameter Defaults

| Hyperparameter | Default | Notes |
|---|---|---|
| `batch_size` | `10,000 paths` | Consistent with He et al. and Distributional paper |
| `n_epochs` | `500` with early stopping | Stop if val loss flat for 50 consecutive epochs |
| `lr` | `1e-3` | Adam default; reduce to `1e-4` if unstable |
| `hidden` | `64` | Network hidden layer dimension; configurable |
| `alpha` | `0.95` | CVaR confidence level |
| `epsilon` | `0.0` | Transaction cost rate; swept in Aim 3 to `0.001` and `0.01` |
| `p0` | `0.0` | Fixed; cash-invariance makes this correct during training |
| `n_steps` | `30` | Pinned for all thesis experiments |

#### Loop Structure

```python
# Setup (once before training)
paths_S, paths_v, paths_t, K, T_mat = loader.load(run_dir, split='train')
features_train = builder.build(paths_S, paths_v, paths_t, K, T_mat)   # (N, T, 3)
features_val   = builder.build(...)

network   = HedgingNetwork(input_dim=4, hidden=64)
cvar      = CVaRLoss(alpha=0.95)
optimizer = Adam([*network.parameters(), cvar.omega], lr=1e-3)

# Training loop
for epoch in range(500):
    idx = torch.randperm(N)
    for batch_start in range(0, N, batch_size):
        batch_idx = idx[batch_start : batch_start + batch_size]

        feat_b = features_train[batch_idx].to(device)    # (B, T, 3)
        S_b    = paths_S_train[batch_idx].to(device)     # (B, T+1)

        deltas     = network.forward_trajectory(feat_b)  # (B, T)
        payoff     = call_payoff(S_b[:, -1], K)          # (B,)
        costs      = proportional_cost(S_b[:,:-1], deltas, epsilon)  # (B,)
        pnl        = compute_pnl(S_b, deltas, payoff, costs)         # (B,)
        loss       = cvar(pnl)                            # scalar

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = evaluate(features_val, paths_S_val, network, cvar)
    # checkpoint if val_loss improved
```

#### Sanity Check — First Convergence Test

> Train on BS with `epsilon=0.0`. Verify that the learned `δ_t` converges to the Black-Scholes analytical delta `Δ_t`. This is the canonical validation from Buehler et al. Section 5.2 and confirms end-to-end correctness before any further experiments.

#### Saved Outputs

| File | Location | Contents |
|---|---|---|
| `best_model.pt` | `experiments/<run>/checkpoints/` | Network weights + ω at best validation CVaR loss |
| `train_loss.csv` | `experiments/<run>/logs/` | Epoch-level train and validation CVaR loss |
| `config.json` | `experiments/<run>/` | Full hyperparameter record for reproducibility |

---

## 5. Complete Shape Flow

Every tensor crossing a module boundary. These shapes must be respected by all implementations.

| From | To | Tensor | Shape | Notes |
|---|---|---|---|---|
| Loader | State Builder | `paths_S` | `(N, T+1)` | Full path including maturity |
| Loader | State Builder | `paths_v` | `(N, T+1)` | `0.0` for BS/NGA |
| Loader | State Builder | `paths_t` | `(N, T+1)` | Time in years |
| Loader | PnL | `paths_S` | `(N, T+1)` | Also used directly in PnL formula |
| State Builder | Policy | `features` | `(N, T, 3)` | No maturity step included |
| Policy | Frictions | `deltas` | `(N, T)` | Hedge ratios `t=0..T-1` |
| Policy | PnL | `deltas` | `(N, T)` | Same tensor, two consumers |
| Loader slice | Frictions | `paths_S[:,:-1]` | `(N, T)` | Steps `0..T-1` only |
| Frictions | PnL | `total_cost` | `(N,)` | Summed over all T steps |
| Derivatives | PnL | `payoff` | `(N,)` | Computed from `paths_S[:,-1]` |
| PnL | CVaR | `pnl` | `(N,)` | Core quantity to risk-minimize |
| CVaR | Trainer | `loss` | `scalar` | Backprop entry point |

---

## 6. Module Interaction Contract

```
DISK (long-format parquet)
         │
         ▼ dataset_loader.py  ── pivot long→wide, load to CPU RAM once
         │
    paths_S  (N, T+1)
    paths_v  (N, T+1)   ──────────────────────────────────────────────┐
    paths_t  (N, T+1)                                                  │
    K, T_mat  scalars                                                  │
         │                                                             │
         ▼ state/builder.py  ── feature engineering                    │
         │                                                             │
    features  (N, T, 3)                                                │
         │                                                             │
         ▼ policy/network.py  ── concat δ_{t-1}, forward T=30 steps   │
         │  input per step: (N, 4)                                     │
         │                                                             │
    deltas  (N, T)  ──────────────────────────────────┐               │
         │                                            │               │
         ▼ frictions/proportional.py                  ▼ pnl/compute.py│
    total_cost  (N,)  ──────────────────────────────► pnl  (N,)  ◄────┘
                                                       ▲
    derivatives/european.py ─── payoff  (N,)  ─────────┘
         │
         ▼ objective_functions/cvar.py  ── CVaR via OCE
         │
    loss  scalar  ──► loss.backward()  ──► network weights + ω
```

---

## 7. Extensibility Map (Aim 3 Frameworks)

Each future framework touches at most one or two modules. PnL computation, data loading, and state construction are **fully shared** across all frameworks.

| Framework | Module Changed | Change Description |
|---|---|---|
| No-Transaction Band (Imaki 2021) | `policy/network.py` | Output `(b_lower, b_upper)` bands; `delta = clamp(delta_{t-1}, bl, bu)` |
| Mean-Variance (Cao 2021) | `objective_functions/` | New `mean_variance.py`; `loss = -E[PnL] + λ·Var[PnL]` |
| Robust Deep Hedging (Lütkebohmert 2022) | `train/trainer.py` | Sample fresh NGA params per path per step during training |
| Adversarial Training (He 2025) | `train/trainer.py` | Add FGSM/PGD perturbation loop on `features` before each forward pass |
| Equal Risk Pricing (Carbonneau 2021) | `policy/` + `train/` | Two networks (long + short); joint CVaR minimization for pricing |
| Black-Scholes Delta Benchmark | `src/baselines/` | Analytical `δ = N(d1)`; no training needed; shared evaluation metrics |

---

## 8. Implementation Order

Ordered so each module depends only on already-completed modules. Each week ends with passing tests.

| Week | Target | Deliverable |
|---|---|---|
| 1 | Schema + Latent State | Add `LATENT_STATE_SCHEMA` · update `generate_dataset.py` · update simulators · tests |
| 2 | Data Loading + State | `dataset_loader.py` · `state/builder.py` · unit tests for shapes and feature values |
| 3 | Derivatives + Frictions + PnL | `european.py` · `proportional.py` · `pnl/compute.py` · unit tests with analytic answers |
| 4 | Policy + Objective | `policy/network.py` · `objective_functions/cvar.py` · unit tests for forward pass shapes |
| 5 | Training Loop — Smoke Test | `train/trainer.py` · train BS frictionless · verify `δ_t` converges to BS analytical delta |
| 6 | Full Training + Evaluation | Train on BS and Heston · loss curves · evaluation plots · compare to BS delta benchmark |

---

*Deep Hedging Framework | Lucian Densmore | UMass Amherst | Spring 2026*
