"""Training loop for deep hedging — Stage 5.

Trains a BaselineFeedforwardNetwork to minimise CVaR via the OCE objective.
Supports all three simulators (BS, Heston, NGA) with optional proportional
transaction costs.

Hyperparameters (matching Buehler et al. 2019 / He et al. 2025):
    batch_size  = 10,000
    n_epochs    = 500
    lr          = 1e-3
    early_stop  = 50 epochs patience on validation CVaR
    epsilon     = 0.0  (no transaction costs by default)
    alpha       = 0.95 (CVaR confidence level)

Output directory layout:
    results/runs/<sim>/<run_id>/
        checkpoints/
            best_model.pt       ← best validation checkpoint
            final_model.pt      ← last epoch checkpoint
        logs/
            train_log.csv       ← epoch, train_loss, val_loss, omega
        config.json             ← full experiment configuration

Usage (CLI):
    python -m src.train.trainer \\
        --dataset_dir  data/datasets/v1.0/bs/<run_id> \\
        --sim          bs \\
        --epsilon      0.0 \\
        --n_epochs     500 \\
        --lr           1e-3 \\
        --batch_size   10000 \\
        --alpha        0.95 \\
        --seed         42

Usage (Python API):
    from src.train.trainer import train
    results = train(cfg)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.io.dataset_loader import DatasetBatch, load_dataset
from src.state.builder import build_features
from src.derivatives.european import call_payoff
from src.derivatives.barrier import compute_barrier_payoff
from src.derivatives.asian import asian_call_payoff
from src.derivatives.lookback import lookback_call_payoff
from src.derivatives.binary import binary_call_payoff
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.objective_functions.cvar import CVaRLoss
from src.objective_functions.mean_variance import MeanVarianceLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Full experiment configuration.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the dataset run directory (output of generate_dataset.py).
    sim : str
        Simulator name: "bs", "heston", or "nga".
    epsilon : float
        Proportional transaction cost rate. 0.0 = frictionless.
    n_epochs : int
        Maximum number of training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size. Must be <= n_train_paths.
    alpha : float
        CVaR confidence level in [0, 1). Only used when objective="cvar".
    lam : float
        Risk-aversion coefficient λ for mean-variance objective. Only used
        when objective="mean_variance". Default: 1.0.
    objective : str
        Risk objective to minimise. One of "cvar" or "mean_variance".
        Default: "cvar".
    hidden : int
        Hidden layer size for BaselineFeedforwardNetwork.
    early_stop_patience : int
        Stop training if val loss does not improve for this many epochs.
    seed : int
        Random seed for reproducibility.
    out_root : str or Path
        Root directory for experiment outputs. Default: "results/runs".
    run_id : str or None
        Explicit run ID. If None, auto-generated from timestamp + config.
    device : str
        "cpu" or "cuda". Auto-detected if not specified.
    """
    dataset_dir:          str
    sim:                  str       = "bs"
    epsilon:              float     = 0.0
    n_epochs:             int       = 500
    lr:                   float     = 1e-3
    batch_size:           int       = 10_000
    alpha:                float     = 0.95
    lam:                  float     = 1.0
    objective:            str       = "cvar"
    hidden:               int       = 64
    early_stop_patience:  int       = 50
    seed:                 int       = 42
    out_root:             str       = "results/runs"
    run_id:               Optional[str] = None
    device:               str       = "auto"
    instrument:           str       = "single"
    payoff:               str       = "european"
    barrier:              Optional[float] = None
    cash:                 float     = 1.0


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Summary of a completed training run."""
    run_dir:          Path
    best_val_loss:    float
    best_epoch:       int
    final_train_loss: float
    final_val_loss:   float
    n_epochs_trained: int
    stopped_early:    bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> TrainResult:
    """Run the full training loop.

    Parameters
    ----------
    cfg : TrainConfig

    Returns
    -------
    TrainResult
    """
    # --- Validate instrument / simulator compatibility ---
    if cfg.instrument == "multi" and cfg.sim != "heston":
        raise ValueError(
            f"--instrument multi requires --sim heston, got --sim {cfg.sim}. "
            "The variance swap price path (paths_S2) is only available for "
            "Heston datasets."
        )

    n_instruments = 2 if cfg.instrument == "multi" else 1

    if cfg.payoff == "barrier" and cfg.barrier is None:
        raise ValueError("--payoff barrier requires --barrier B to be specified.")

    # --- Setup ---
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    run_dir = _make_run_dir(cfg)

    _setup_logging(run_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    # --- Load data ---
    logger.info("Loading datasets...")
    train_batch = load_dataset(Path(cfg.dataset_dir), split="train")
    val_batch   = load_dataset(Path(cfg.dataset_dir), split="val")

    logger.info(
        f"Train paths: {train_batch.n_paths} | "
        f"Val paths: {val_batch.n_paths} | "
        f"Steps: {train_batch.n_steps}"
    )

    # --- Pre-compute features (CPU RAM) ---
    logger.info("Building features...")
    train_features = build_features(
        train_batch,
        n_instruments=n_instruments,
        payoff=cfg.payoff,
        barrier_level=cfg.barrier,
    )
    val_features = build_features(
        val_batch,
        n_instruments=n_instruments,
        payoff=cfg.payoff,
        barrier_level=cfg.barrier,
    )

    # --- Move full datasets to device ---
    train_features = train_features.to(device)
    val_features   = val_features.to(device)

    # Stack instrument price paths: (N, T+1) → (N, T+1, I) for multi.
    if n_instruments == 2:
        train_S = torch.stack(
            [train_batch.paths_S, train_batch.paths_S2], dim=-1
        ).to(device)   # (N, T+1, 2)
        val_S   = torch.stack(
            [val_batch.paths_S, val_batch.paths_S2], dim=-1
        ).to(device)
    else:
        train_S = train_batch.paths_S.to(device)   # (N, T+1)
        val_S   = val_batch.paths_S.to(device)

    K     = train_batch.K
    T_mat = train_batch.T_mat

    # --- Build payoff function ---
    # payoff_fn takes the full paths_S batch ((N, T+1) or (N, T+1, I)) and
    # returns per-path payoff (N,). Path-dependent payoffs operate on the
    # spot sub-path; terminal payoffs use only the last column.
    _K    = K
    _cash = cfg.cash
    _B    = cfg.barrier

    if cfg.payoff == "barrier":
        def payoff_fn(S_full: torch.Tensor) -> torch.Tensor:
            spot = S_full[:, :, 0] if n_instruments == 2 else S_full
            return compute_barrier_payoff(spot, _K, _B)
    elif cfg.payoff == "asian":
        def payoff_fn(S_full: torch.Tensor) -> torch.Tensor:
            spot = S_full[:, :, 0] if n_instruments == 2 else S_full
            return asian_call_payoff(spot, _K)
    elif cfg.payoff == "lookback":
        def payoff_fn(S_full: torch.Tensor) -> torch.Tensor:
            spot = S_full[:, :, 0] if n_instruments == 2 else S_full
            return lookback_call_payoff(spot, _K)
    elif cfg.payoff == "binary":
        def payoff_fn(S_full: torch.Tensor) -> torch.Tensor:
            S_T = S_full[:, -1, 0] if n_instruments == 2 else S_full[:, -1]
            return binary_call_payoff(S_T, _K, cash=_cash)
    else:  # european
        def payoff_fn(S_full: torch.Tensor) -> torch.Tensor:
            S_T = S_full[:, -1, 0] if n_instruments == 2 else S_full[:, -1]
            return call_payoff(S_T, _K)

    # --- Build model and objective ---
    n_features = train_features.shape[2]
    net = BaselineFeedforwardNetwork(
        hidden=cfg.hidden, n_instruments=n_instruments, feature_dim=n_features
    ).to(device)
    objective = _build_objective(cfg).to(device)

    # Include objective.omega in optimizer only if it is a trainable Parameter
    # (CVaR has a trainable omega; mean-variance does not).
    opt_params = list(net.parameters())
    if isinstance(objective.omega, nn.Parameter):
        opt_params.append(objective.omega)

    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr)

    logger.info(f"Network parameters: {net.n_parameters():,}")

    # --- Save config ---
    _save_config(cfg, run_dir)

    # --- Training loop ---
    log_rows   = []
    best_val   = float("inf")
    best_epoch = 0
    patience   = 0

    N_train = train_batch.n_paths

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()

        # -- Train --
        net.train()
        train_loss = _run_epoch(
            net, objective, optimizer,
            train_features, train_S,
            cfg.epsilon, cfg.batch_size,
            payoff_fn,
            n_instruments=n_instruments,
            training=True,
        )

        # -- Validate --
        net.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                net, objective, optimizer,
                val_features, val_S,
                cfg.epsilon, val_batch.n_paths,   # full val set
                payoff_fn,
                n_instruments=n_instruments,
                training=False,
            )

        elapsed = time.time() - t0
        omega   = objective.omega.item() if isinstance(objective.omega, nn.Parameter) else float("nan")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "omega":      omega,
            "elapsed_s":  elapsed,
        })

        if epoch % 10 == 0 or epoch == 1:
            omega_str = f"ω={omega:.4f} | " if not isinstance(omega, float) or not (omega != omega) else ""
            logger.info(
                f"Epoch {epoch:4d}/{cfg.n_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"{omega_str}{elapsed:.1f}s"
            )

        # -- Checkpoint best model --
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            patience   = 0
            _save_checkpoint(net, objective, optimizer, epoch, run_dir, name="best_model")
        else:
            patience += 1

        # -- Early stopping --
        if patience >= cfg.early_stop_patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.early_stop_patience} epochs)"
            )
            _save_checkpoint(net, objective, optimizer, epoch, run_dir, name="final_model")
            _save_log(log_rows, run_dir)
            return TrainResult(
                run_dir          = run_dir,
                best_val_loss    = best_val,
                best_epoch       = best_epoch,
                final_train_loss = train_loss,
                final_val_loss   = val_loss,
                n_epochs_trained = epoch,
                stopped_early    = True,
            )

    # -- Save final model if not early stopped --
    _save_checkpoint(net, objective, optimizer, cfg.n_epochs, run_dir, name="final_model")
    _save_log(log_rows, run_dir)

    logger.info(
        f"Training complete. Best val={best_val:.4f} at epoch {best_epoch}."
    )

    return TrainResult(
        run_dir          = run_dir,
        best_val_loss    = best_val,
        best_epoch       = best_epoch,
        final_train_loss = log_rows[-1]["train_loss"],
        final_val_loss   = log_rows[-1]["val_loss"],
        n_epochs_trained = cfg.n_epochs,
        stopped_early    = False,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_objective(cfg: TrainConfig) -> nn.Module:
    """Instantiate the risk objective from config."""
    if cfg.objective == "cvar":
        return CVaRLoss(alpha=cfg.alpha)
    elif cfg.objective == "mean_variance":
        return MeanVarianceLoss(lam=cfg.lam)
    else:
        raise ValueError(
            f"Unknown objective '{cfg.objective}'. "
            "Choose 'cvar' or 'mean_variance'."
        )


def _run_epoch(
    net:           BaselineFeedforwardNetwork,
    objective:     nn.Module,
    optimizer:     torch.optim.Optimizer,
    features:      torch.Tensor,
    paths_S:       torch.Tensor,
    epsilon:       float,
    batch_size:    int,
    payoff_fn,
    n_instruments: int = 1,
    training:      bool = True,
) -> float:
    """Run one full pass over the data (train or eval).

    For training, shuffles the dataset and iterates through all mini-batches,
    performing one gradient step per batch. This ensures every path is seen
    once per epoch.

    For eval, runs the full dataset in one forward pass with no_grad.

    Parameters
    ----------
    payoff_fn : callable
        Takes the full paths_S batch ``(N, T+1)`` or ``(N, T+1, I)`` and
        returns per-path payoff ``(N,)``. Encapsulates both the payoff type
        (European / barrier) and the strike / barrier levels.

    Returns
    -------
    float
        Mean loss across all batches for this epoch.
    """
    N = features.shape[0]

    if not training:
        # Full dataset in one forward pass
        deltas     = net.forward_trajectory(features)
        payoff     = payoff_fn(paths_S)
        total_cost = proportional_cost(paths_S[:, :-1], deltas, epsilon=epsilon)
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)
        return objective(pnl).item()

    # Training: shuffle and iterate through all mini-batches
    perm       = torch.randperm(N, device=features.device)
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, N, batch_size):
        idx    = perm[start:start + batch_size]
        feat_b = features[idx]
        S_b    = paths_S[idx]

        optimizer.zero_grad()
        deltas     = net.forward_trajectory(feat_b)
        payoff     = payoff_fn(S_b)
        total_cost = proportional_cost(S_b[:, :-1], deltas, epsilon=epsilon)
        pnl        = compute_pnl(S_b, deltas, payoff, total_cost)
        loss       = objective(pnl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_run_dir(cfg: TrainConfig) -> Path:
    if cfg.run_id is None:
        ts     = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{cfg.sim}_eps-{cfg.epsilon}_seed-{cfg.seed}"
    else:
        run_id = cfg.run_id

    run_dir = Path(cfg.out_root) / cfg.sim / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "logs" / "train.log"
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def _save_config(cfg: TrainConfig, run_dir: Path) -> None:
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def _save_checkpoint(
    net:       BaselineFeedforwardNetwork,
    objective: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    run_dir:   Path,
    name:      str,
) -> None:
    path = run_dir / "checkpoints" / f"{name}.pt"
    torch.save({
        "epoch":            epoch,
        "model_state_dict": net.state_dict(),
        "cvar_state_dict":  objective.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
    }, path)


def _save_log(rows: list[dict], run_dir: Path) -> None:
    import csv
    path = run_dir / "logs" / "train_log.csv"
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a deep hedging network.")
    p.add_argument("--dataset_dir",         required=True)
    p.add_argument("--run_name",            required=True,
                   help="Experiment name. Outputs saved to <out_root>/<sim>/<run_name>/")
    p.add_argument("--sim",                 default="bs")
    p.add_argument("--epsilon",             type=float, default=0.0)
    p.add_argument("--n_epochs",            type=int,   default=500)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--batch_size",          type=int,   default=10_000)
    p.add_argument("--alpha",               type=float, default=0.95)
    p.add_argument("--hidden",              type=int,   default=64)
    p.add_argument("--early_stop_patience", type=int,   default=50)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--objective",            default="cvar",
                   choices=["cvar", "mean_variance"],
                   help="Risk objective: 'cvar' or 'mean_variance'")
    p.add_argument("--lam",                  type=float, default=1.0,
                   help="Risk-aversion λ for mean_variance objective")
    p.add_argument("--out_root",            default="results/runs")
    p.add_argument("--device",              default="auto")
    p.add_argument("--instrument",          default="single",
                   choices=["single", "multi"],
                   help="Hedging instrument mode. 'multi' requires --sim heston.")
    p.add_argument("--payoff",              default="european",
                   choices=["european", "barrier", "asian", "lookback", "binary"],
                   help="Option payoff type.")
    p.add_argument("--barrier",             type=float, default=None,
                   help="Barrier level B for up-and-out call. Required when --payoff barrier.")
    p.add_argument("--cash",                type=float, default=1.0,
                   help="Fixed cash payment for binary call. Used when --payoff binary.")

    args = p.parse_args()
    d = vars(args)
    d["run_id"] = d.pop("run_name")
    return TrainConfig(**d)


if __name__ == "__main__":
    cfg    = _parse_args()
    result = train(cfg)
    print(f"\nDone. Best val CVaR = {result.best_val_loss:.4f} at epoch {result.best_epoch}")
    print(f"Run directory: {result.run_dir}")
