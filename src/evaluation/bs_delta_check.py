"""Black-Scholes delta convergence check for trained hedge policies.

This module compares network hedge ratios against analytical Black-Scholes
call deltas to validate frictionless BS training behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.io.dataset_loader import DatasetBatch, load_dataset
from src.state.builder import build_features
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.objective_functions.cvar import CVaRLoss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BSCheckConfig:
    """Configuration for the BS delta convergence check.

    Parameters
    ----------
    checkpoint : str or Path
        Path to a .pt checkpoint file (output of trainer.py).
    dataset_dir : str or Path
        Path to the BS dataset run directory.
    hidden : int
        Hidden layer size matching the trained network. Default: 64.
    sigma : float
        Volatility used to generate the dataset. Default: 0.2.
    r : float
        Risk-free rate. Default: 0.0 (consistent with simulator).
    split : str
        Dataset split to evaluate on. Default: "test".
    out_dir : str or Path or None
        If provided, save CSV and plot here.
    plot : bool
        If True, display scatter plot interactively (requires matplotlib).
    """
    checkpoint:  str
    dataset_dir: str
    hidden:      int   = 64
    sigma:       float = 0.2
    r:           float = 0.0
    split:       str   = "test"
    out_dir:     Optional[str] = None
    plot:        bool  = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BSCheckResult:
    """Results of the BS delta convergence check.

    Parameters
    ----------
    mae : float
        Mean absolute error between network delta and BS delta,
        averaged over all paths and timesteps.
    r2 : float
        R² between network delta and BS delta.
        Should be > 0.99 for a well-trained frictionless network.
    per_step_mae : list[float]
        MAE at each individual timestep t=0,...,T-1.
    n_paths : int
        Number of evaluation paths.
    n_steps : int
        Number of timesteps.
    passed : bool
        True if mae < 0.02 (practical threshold for convergence).
    """
    mae:          float
    r2:           float
    per_step_mae: list[float]
    n_paths:      int
    n_steps:      int
    passed:       bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_bs_delta_check(cfg: BSCheckConfig) -> BSCheckResult:
    """Run the BS delta convergence check.

    Parameters
    ----------
    cfg : BSCheckConfig

    Returns
    -------
    BSCheckResult
    """
    # --- Load dataset ---
    batch = load_dataset(Path(cfg.dataset_dir), split=cfg.split)

    # --- Build features ---
    features = build_features(batch)   # (N, T, 3)

    # --- Load trained network ---
    net = _load_network(cfg.checkpoint, cfg.hidden)
    net.eval()

    # --- Compute network deltas ---
    with torch.no_grad():
        net_deltas = net.forward_trajectory(features)   # (N, T)

    # --- Compute BS analytical deltas ---
    bs_deltas = _bs_delta_trajectory(batch, cfg.sigma, cfg.r)   # (N, T)

    # --- Compute metrics ---
    diff         = net_deltas - bs_deltas                        # (N, T)
    mae          = diff.abs().mean().item()
    per_step_mae = diff.abs().mean(dim=0).tolist()               # (T,)
    r2           = _r2_score(net_deltas, bs_deltas)

    passed = mae < 0.02

    result = BSCheckResult(
        mae          = mae,
        r2           = r2,
        per_step_mae = per_step_mae,
        n_paths      = batch.n_paths,
        n_steps      = batch.n_steps,
        passed       = passed,
    )

    # --- Print summary ---
    _print_summary(result)

    # --- Save outputs ---
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(result, out_dir)
        _save_scatter(net_deltas, bs_deltas, out_dir, show=cfg.plot)
    elif cfg.plot:
        _save_scatter(net_deltas, bs_deltas, out_dir=None, show=True)

    return result


# ---------------------------------------------------------------------------
# BS analytical delta
# ---------------------------------------------------------------------------

def bs_call_delta(
    S:     torch.Tensor,
    K:     float,
    tau:   torch.Tensor,
    sigma: float,
    r:     float = 0.0,
) -> torch.Tensor:
    """Black-Scholes call delta N(d1).

    Parameters
    ----------
    S : torch.Tensor
        Spot price. Any shape.
    K : float
        Strike price.
    tau : torch.Tensor
        Time to maturity (years). Same shape as S.
    sigma : float
        Volatility (annualised).
    r : float
        Risk-free rate (annualised). Default: 0.0.

    Returns
    -------
    delta : torch.Tensor
        BS call delta in [0, 1]. Same shape as S.
    """
    # Clamp tau to avoid division by zero at maturity
    tau_safe = tau.clamp(min=1e-8)

    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau_safe) / (
        sigma * torch.sqrt(tau_safe)
    )

    return _normal_cdf(d1)


def _bs_delta_trajectory(
    batch: DatasetBatch,
    sigma: float,
    r:     float,
) -> torch.Tensor:
    """Compute BS delta at every (path, timestep) in the batch.

    Uses paths_S[:, :-1] (spot at rebalancing times, excluding maturity)
    and paths_t[:, :-1] (time grid, excluding maturity).

    Returns
    -------
    deltas : torch.Tensor, shape (N, T)
    """
    S_t   = batch.paths_S[:, :-1]   # (N, T)
    t_t   = batch.paths_t[:, :-1]   # (N, T)
    tau_t = batch.T_mat - t_t       # (N, T)

    return bs_call_delta(S_t, batch.K, tau_t, sigma, r)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute coefficient of determination for tensor predictions.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Reference target values with the same shape as ``pred``.

    Returns
    -------
    float
        R-squared statistic ``1 - SS_res / SS_tot``.
    """
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    if ss_tot.item() == 0.0:
        return 1.0
    return (1.0 - ss_res / ss_tot).item()


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------

def _load_network(checkpoint_path: str, hidden: int) -> BaselineFeedforwardNetwork:
    """Load baseline network weights from a trainer checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint ``.pt`` file.
    hidden : int
        Hidden layer width used when constructing the model.

    Returns
    -------
    BaselineFeedforwardNetwork
        Network with loaded state dictionary.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    ValueError
        If expected state-dict keys are missing.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint at {path} does not contain 'model_state_dict'. "
            "Was it saved by trainer.py?"
        )

    net = BaselineFeedforwardNetwork(hidden=hidden)
    net.load_state_dict(ckpt["model_state_dict"])
    return net


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(result: BSCheckResult) -> None:
    """Print terminal summary metrics for the BS delta check.

    Parameters
    ----------
    result : BSCheckResult
        Evaluation result object to display.
    """
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    print(f"\n{'='*55}")
    print(f"  BS Delta Convergence Check  —  {status}")
    print(f"{'='*55}")
    print(f"  Paths evaluated : {result.n_paths:,}")
    print(f"  Timesteps       : {result.n_steps}")
    print(f"  MAE             : {result.mae:.6f}  (threshold < 0.02)")
    print(f"  R²              : {result.r2:.6f}  (target > 0.99)")
    print(f"{'='*55}\n")

    if not result.passed:
        print(
            "  WARNING: MAE exceeds 0.02. The network has not converged to\n"
            "  BS delta. Check: (1) sufficient training epochs, (2) correct\n"
            "  sigma value passed to this script, (3) epsilon=0.0 during training.\n"
        )


def _save_csv(result: BSCheckResult, out_dir: Path) -> None:
    """Write per-timestep MAE values to CSV.

    Parameters
    ----------
    result : BSCheckResult
        Evaluation result containing per-step MAE values.
    out_dir : Path
        Destination directory for ``bs_delta_check.csv``.
    """
    path = out_dir / "bs_delta_check.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "mae"])
        for t, mae in enumerate(result.per_step_mae):
            writer.writerow([t, mae])
    print(f"  Per-step MAE saved to: {path}")


def _save_scatter(
    net_deltas: torch.Tensor,
    bs_deltas:  torch.Tensor,
    out_dir:    Optional[Path],
    show:       bool,
) -> None:
    """Create and optionally display a network-vs-analytical delta scatter plot.

    Parameters
    ----------
    net_deltas : torch.Tensor
        Network deltas with shape ``(N, T)``.
    bs_deltas : torch.Tensor
        Analytical BS deltas with shape ``(N, T)``.
    out_dir : Optional[Path]
        Output directory for saved image. If ``None``, no file is written.
    show : bool
        If ``True``, display the plot interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot.")
        return

    # Subsample for readability (max 5000 points)
    N, T    = net_deltas.shape
    n_total = N * T
    if n_total > 5000:
        idx = torch.randperm(n_total)[:5000]
    else:
        idx = torch.arange(n_total)

    net_flat = net_deltas.flatten()[idx].detach().numpy()
    bs_flat  = bs_deltas.flatten()[idx].detach().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(bs_flat, net_flat, alpha=0.3, s=6, c="steelblue", label="samples")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="perfect alignment")
    ax.set_xlabel("BS Analytical Delta")
    ax.set_ylabel("Network Delta")
    ax.set_title("Network Delta vs BS Analytical Delta")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    if out_dir is not None:
        path = out_dir / "bs_delta_scatter.png"
        fig.savefig(path, dpi=150)
        print(f"  Scatter plot saved to: {path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> BSCheckConfig:
    """Parse command-line arguments for the BS delta check CLI.

    Returns
    -------
    BSCheckConfig
        Configuration populated from CLI arguments.
    """
    p = argparse.ArgumentParser(
        description="BS delta convergence sanity check for trained deep hedging networks."
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to best_model.pt checkpoint.")
    p.add_argument("--dataset_dir", required=True,
                   help="Path to BS dataset run directory.")
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--sigma",       type=float, default=0.2,
                   help="Volatility used when generating the dataset.")
    p.add_argument("--r",           type=float, default=0.0,
                   help="Risk-free rate (default 0.0).")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--out_dir",     default=None,
                   help="Directory to save CSV and scatter plot.")
    p.add_argument("--plot",        action="store_true",
                   help="Display scatter plot interactively.")

    args = p.parse_args()
    return BSCheckConfig(**vars(args))


if __name__ == "__main__":
    cfg    = _parse_args()
    result = run_bs_delta_check(cfg)
    exit(0 if result.passed else 1)
