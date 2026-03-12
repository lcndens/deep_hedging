"""Training loop for deep-hedging policy optimization.

This module trains the baseline feedforward hedge policy against the CVaR OCE
objective, evaluates on a validation split each epoch, and persists checkpoints
plus logs for downstream evaluation.
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
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.objective_functions.cvar import CVaRLoss

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
        CVaR confidence level in [0, 1).
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
    hidden:               int       = 64
    early_stop_patience:  int       = 50
    seed:                 int       = 42
    out_root:             str       = "results/runs"
    run_id:               Optional[str] = None
    device:               str       = "auto"


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Summary statistics and output location for a training run.

    Parameters
    ----------
    run_dir : Path
        Directory containing checkpoints, logs, and configuration.
    best_val_loss : float
        Lowest validation loss observed during training.
    best_epoch : int
        Epoch index where ``best_val_loss`` was achieved.
    final_train_loss : float
        Training loss from the final processed epoch.
    final_val_loss : float
        Validation loss from the final processed epoch.
    n_epochs_trained : int
        Number of epochs actually executed.
    stopped_early : bool
        Whether early stopping terminated the run before ``cfg.n_epochs``.
    """
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
        Full training configuration.

    Returns
    -------
    TrainResult
        Aggregated run outputs and final metrics.
    """
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
    train_features = build_features(train_batch)   # (N_train, T, 3)
    val_features   = build_features(val_batch)     # (N_val,   T, 3)

    # --- Move full datasets to device ---
    train_features = train_features.to(device)
    val_features   = val_features.to(device)

    train_S = train_batch.paths_S.to(device)
    val_S   = val_batch.paths_S.to(device)

    K     = train_batch.K
    T_mat = train_batch.T_mat

    # --- Build model and objective ---
    net  = BaselineFeedforwardNetwork(hidden=cfg.hidden).to(device)
    cvar = CVaRLoss(alpha=cfg.alpha).to(device)

    optimizer = torch.optim.Adam(
        [*net.parameters(), cvar.omega],
        lr=cfg.lr,
    )

    logger.info(f"Network parameters: {net.n_parameters():,}")

    # --- Save config ---
    _save_config(cfg, run_dir)

    # --- Training loop ---
    log_rows   = []
    best_val   = float("inf")
    best_epoch = 0
    patience   = 0

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()

        # -- Train --
        net.train()
        train_loss = _run_epoch(
            net, cvar, optimizer,
            train_features, train_S,
            K, cfg.epsilon, cfg.batch_size,
            training=True,
        )

        # -- Validate --
        net.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                net, cvar, optimizer,
                val_features, val_S,
                K, cfg.epsilon, val_batch.n_paths,   # full val set
                training=False,
            )

        elapsed = time.time() - t0
        omega   = cvar.omega.item()

        log_rows.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "omega":      omega,
            "elapsed_s":  elapsed,
        })

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{cfg.n_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"ω={omega:.4f} | {elapsed:.1f}s"
            )

        # -- Checkpoint best model --
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            patience   = 0
            _save_checkpoint(net, cvar, optimizer, epoch, run_dir, name="best_model")
        else:
            patience += 1

        # -- Early stopping --
        if patience >= cfg.early_stop_patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.early_stop_patience} epochs)"
            )
            _save_checkpoint(net, cvar, optimizer, epoch, run_dir, name="final_model")
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
    _save_checkpoint(net, cvar, optimizer, cfg.n_epochs, run_dir, name="final_model")
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

def _run_epoch(
    net:        BaselineFeedforwardNetwork,
    cvar:       CVaRLoss,
    optimizer:  torch.optim.Optimizer,
    features:   torch.Tensor,
    paths_S:    torch.Tensor,
    K:          float,
    epsilon:    float,
    batch_size: int,
    training:   bool,
) -> float:
    """Run one full pass over the data (train or eval).

    For training, shuffles the dataset and iterates through all mini-batches,
    performing one gradient step per batch. This ensures every path is seen
    once per epoch.

    For eval, runs the full dataset in one forward pass with no_grad.

    Parameters
    ----------
    net : BaselineFeedforwardNetwork
        Hedge policy network.
    cvar : CVaRLoss
        CVaR objective module containing trainable ``omega``.
    optimizer : torch.optim.Optimizer
        Optimizer used during training mode.
    features : torch.Tensor
        Feature tensor with shape ``(N, T, 3)``.
    paths_S : torch.Tensor
        Spot paths with shape ``(N, T+1)``.
    K : float
        Strike price.
    epsilon : float
        Proportional transaction-cost rate.
    batch_size : int
        Minibatch size used when ``training=True``.
    training : bool
        If ``True``, perform gradient updates. Otherwise run evaluation only.

    Returns
    -------
    float
        Mean CVaR loss across all batches for this epoch.
    """
    N = features.shape[0]

    if not training:
        # Full dataset in one forward pass
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(paths_S[:, -1], K)
        total_cost = proportional_cost(paths_S[:, :-1], deltas, epsilon=epsilon)
        pnl        = compute_pnl(paths_S, deltas, payoff, total_cost)
        return cvar(pnl).item()

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
        payoff     = call_payoff(S_b[:, -1], K)
        total_cost = proportional_cost(S_b[:, :-1], deltas, epsilon=epsilon)
        pnl        = compute_pnl(S_b, deltas, payoff, total_cost)
        loss       = cvar(pnl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible torch operations.

    Parameters
    ----------
    seed : int
        Seed value applied to CPU and CUDA generators.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    """Resolve target torch device from configuration.

    Parameters
    ----------
    device : str
        Device string. ``"auto"`` selects CUDA when available.

    Returns
    -------
    torch.device
        Resolved device instance.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_run_dir(cfg: TrainConfig) -> Path:
    """Create and return the output directory for a training run.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration containing simulator, run id, and output root.

    Returns
    -------
    Path
        Run directory path with ``checkpoints/`` and ``logs/`` created.
    """
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
    """Configure logging handlers for console and file output.

    Parameters
    ----------
    run_dir : Path
        Run directory containing the ``logs`` folder.
    """
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
    """Write the resolved training configuration to ``config.json``.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration to persist.
    run_dir : Path
        Output run directory.
    """
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def _save_checkpoint(
    net:       BaselineFeedforwardNetwork,
    cvar:      CVaRLoss,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    run_dir:   Path,
    name:      str,
) -> None:
    """Persist model, objective, and optimizer state to disk.

    Parameters
    ----------
    net : BaselineFeedforwardNetwork
        Trained policy network.
    cvar : CVaRLoss
        Objective module containing ``omega`` state.
    optimizer : torch.optim.Optimizer
        Optimizer state to checkpoint.
    epoch : int
        Epoch number at save time.
    run_dir : Path
        Output run directory.
    name : str
        Checkpoint stem name (for example ``"best_model"``).
    """
    path = run_dir / "checkpoints" / f"{name}.pt"
    torch.save({
        "epoch":            epoch,
        "model_state_dict": net.state_dict(),
        "cvar_state_dict":  cvar.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
    }, path)


def _save_log(rows: list[dict], run_dir: Path) -> None:
    """Write epoch-level training metrics to CSV.

    Parameters
    ----------
    rows : list[dict]
        Sequence of per-epoch metric dictionaries.
    run_dir : Path
        Output run directory.
    """
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
    """Parse CLI arguments into a ``TrainConfig`` object.

    Returns
    -------
    TrainConfig
        Configuration populated from command-line arguments.
    """
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
    p.add_argument("--out_root",            default="results/runs")
    p.add_argument("--device",              default="auto")

    args = p.parse_args()
    d = vars(args)
    d["run_id"] = d.pop("run_name")
    return TrainConfig(**d)


if __name__ == "__main__":
    cfg    = _parse_args()
    result = train(cfg)
    print(f"\nDone. Best val CVaR = {result.best_val_loss:.4f} at epoch {result.best_epoch}")
    print(f"Run directory: {result.run_dir}")
