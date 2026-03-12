"""Evaluation utilities for trained deep-hedging runs.

The module loads trained run directories, recomputes test-set PnL metrics, and
produces summary tables and diagnostic charts used in thesis reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from src.io.dataset_loader import load_dataset, DatasetBatch
from src.state.builder import build_features
from src.derivatives.european import call_payoff
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.objective_functions.cvar import CVaRLoss
from src.evaluation.bs_delta_check import bs_call_delta


# ---------------------------------------------------------------------------
# Run descriptor
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """Container for artifacts loaded from one training run.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.
    run_name : str
        Run identifier (directory name).
    sim : str
        Simulator label such as ``bs``, ``heston``, or ``nga``.
    config : dict
        Parsed run configuration.
    pnl : torch.Tensor
        Deep-hedge terminal PnL tensor with shape ``(N,)``.
    bs_pnl : Optional[torch.Tensor]
        Analytical BS-delta benchmark PnL with shape ``(N,)`` when available.
    log : list[dict]
        Parsed training log rows.
    """
    run_dir:    Path
    run_name:   str
    sim:        str
    config:     dict
    pnl:        torch.Tensor    # (N_test,)  — deep hedge PnL
    bs_pnl:     Optional[torch.Tensor]  # (N_test,) — BS delta PnL (BS only)
    log:        list[dict]      # train_log.csv rows


@dataclass
class Metrics:
    """Scalar evaluation metrics for one run.

    Parameters
    ----------
    run_name : str
        Run identifier.
    sim : str
        Simulator label.
    cvar_95 : float
        Empirical CVaR at confidence level ``alpha=0.95`` from loss view.
    var_95 : float
        Empirical VaR at confidence level ``alpha=0.95`` from loss view.
    mean_pnl : float
        Mean terminal PnL.
    std_pnl : float
        Standard deviation of terminal PnL.
    p10_pnl : float
        10th percentile of terminal PnL.
    best_epoch : int
        Epoch with lowest validation loss.
    n_epochs : int
        Number of logged epochs.
    """
    run_name:   str
    sim:        str
    cvar_95:    float
    var_95:     float
    mean_pnl:   float
    std_pnl:    float
    p10_pnl:    float   # 10th percentile of PnL
    best_epoch: int
    n_epochs:   int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(run_dirs: list[Path], out_dir: Path) -> None:
    """Full evaluation pipeline for a list of run directories.

    Parameters
    ----------
    run_dirs : list[Path]
        Paths to trained run directories (e.g. results/runs/bs/run_name).
    out_dir : Path
        Directory to save all outputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {len(run_dirs)} run(s) → {out_dir}\n")

    # Load all runs
    runs = [_load_run(d) for d in run_dirs]

    # Table 1
    metrics = [_compute_metrics(r) for r in runs]
    _save_table1(metrics, out_dir)

    # Chart 2 — loss curves (all runs on one figure)
    _plot_loss_curves(runs, out_dir)

    # Chart 1 + Chart 4 — per run
    for run in runs:
        _plot_pnl_histogram(run, out_dir)
        _plot_per_timestep_error(run, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir}\n")


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> RunInfo:
    """Load configuration, model outputs, and logs for one run.

    Parameters
    ----------
    run_dir : Path
        Path to a trained run directory.

    Returns
    -------
    RunInfo
        Structured run artifacts for downstream evaluation.

    Raises
    ------
    FileNotFoundError
        If the run directory or checkpoint is missing.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_name = run_dir.name

    # Config
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    sim         = config["sim"]
    dataset_dir = Path(config["dataset_dir"])
    hidden      = config.get("hidden", 64)
    epsilon     = config.get("epsilon", 0.0)
    alpha       = config.get("alpha", 0.95)

    print(f"Loading run: {run_name}  (sim={sim})")

    # Dataset — test split
    batch = load_dataset(dataset_dir, split="test")

    # Features
    features = build_features(batch)

    # Load network
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    net = BaselineFeedforwardNetwork(hidden=hidden)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Deep hedge PnL
    with torch.no_grad():
        deltas     = net.forward_trajectory(features)
        payoff     = call_payoff(batch.paths_S[:, -1], batch.K)
        total_cost = proportional_cost(
            batch.paths_S[:, :-1], deltas, epsilon=epsilon
        )
        pnl = compute_pnl(batch.paths_S, deltas, payoff, total_cost)

    # BS analytical delta PnL (BS frictionless only)
    bs_pnl = None
    if sim.lower() == "bs" and epsilon == 0.0:
        sigma = config.get("simulator_params", {}).get("sigma", 0.2)
        bs_pnl = _compute_bs_delta_pnl(batch, sigma=sigma)

    # Training log
    log = _load_log(run_dir)

    return RunInfo(
        run_dir  = run_dir,
        run_name = run_name,
        sim      = sim,
        config   = config,
        pnl      = pnl,
        bs_pnl   = bs_pnl,
        log      = log,
    )


def _load_log(run_dir: Path) -> list[dict]:
    """Load ``train_log.csv`` rows for a run directory.

    Parameters
    ----------
    run_dir : Path
        Run directory containing the ``logs`` subfolder.

    Returns
    -------
    list[dict]
        Parsed CSV rows. Returns an empty list if the log file is absent.
    """
    log_path = run_dir / "logs" / "train_log.csv"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# BS analytical delta PnL
# ---------------------------------------------------------------------------

def _compute_bs_delta_pnl(
    batch: DatasetBatch,
    sigma: float = 0.2,
    r:     float = 0.0,
) -> torch.Tensor:
    """Compute benchmark PnL from analytical BS deltas.

    Parameters
    ----------
    batch : DatasetBatch
        Loaded dataset split with tensors of shape ``(N, T+1)``.
    sigma : float, default=0.2
        Black-Scholes volatility parameter.
    r : float, default=0.0
        Risk-free rate.

    Returns
    -------
    torch.Tensor
        Terminal PnL tensor with shape ``(N,)`` under the analytical strategy.
    """
    S_t   = batch.paths_S[:, :-1]   # (N, T)
    t_t   = batch.paths_t[:, :-1]   # (N, T)
    tau_t = batch.T_mat - t_t

    deltas     = bs_call_delta(S_t, batch.K, tau_t, sigma, r)
    payoff     = call_payoff(batch.paths_S[:, -1], batch.K)
    total_cost = proportional_cost(batch.paths_S[:, :-1], deltas, epsilon=0.0)
    return compute_pnl(batch.paths_S, deltas, payoff, total_cost)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(run: RunInfo) -> Metrics:
    """Compute scalar summary metrics from run outputs.

    Parameters
    ----------
    run : RunInfo
        Run data including PnL samples and optional training logs.

    Returns
    -------
    Metrics
        Aggregated metrics for tabular reporting.
    """
    pnl  = run.pnl.numpy()
    N    = len(pnl)

    # CVaR_0.95: mean of worst 5%
    alpha    = run.config.get("alpha", 0.95)
    k        = max(1, int(math.ceil((1 - alpha) * N)))
    sorted_pnl = np.sort(pnl)          # ascending — worst first
    cvar_95  = float(-sorted_pnl[:k].mean())   # negate: loss perspective
    var_95   = float(-sorted_pnl[k - 1])

    # Best epoch from log
    best_epoch = 0
    n_epochs   = 0
    if run.log:
        n_epochs   = len(run.log)
        val_losses = [float(r["val_loss"]) for r in run.log]
        best_epoch = int(np.argmin(val_losses)) + 1

    return Metrics(
        run_name   = run.run_name,
        sim        = run.sim,
        cvar_95    = cvar_95,
        var_95     = var_95,
        mean_pnl   = float(pnl.mean()),
        std_pnl    = float(pnl.std()),
        p10_pnl    = float(np.percentile(pnl, 10)),
        best_epoch = best_epoch,
        n_epochs   = n_epochs,
    )


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------

def _save_table1(metrics: list[Metrics], out_dir: Path) -> None:
    """Write Table 1 metrics to CSV.

    Parameters
    ----------
    metrics : list[Metrics]
        Per-run scalar metric records.
    out_dir : Path
        Output directory for ``table1_results.csv``.
    """
    # CSV
    csv_path = out_dir / "table1_results.csv"
    fields   = [
        "run_name", "sim", "cvar_95", "var_95",
        "mean_pnl", "std_pnl", "p10_pnl", "best_epoch", "n_epochs"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "run_name":   m.run_name,
                "sim":        m.sim,
                "cvar_95":    f"{m.cvar_95:.4f}",
                "var_95":     f"{m.var_95:.4f}",
                "mean_pnl":   f"{m.mean_pnl:.4f}",
                "std_pnl":    f"{m.std_pnl:.4f}",
                "p10_pnl":    f"{m.p10_pnl:.4f}",
                "best_epoch": m.best_epoch,
                "n_epochs":   m.n_epochs,
            })
    print(f"  Table 1 (CSV)   → {csv_path}")


# ---------------------------------------------------------------------------
# Chart 1 — PnL distribution histogram
# ---------------------------------------------------------------------------

def _plot_pnl_histogram(run: RunInfo, out_dir: Path) -> None:
    """Plot and save PnL histogram for a run.

    Parameters
    ----------
    run : RunInfo
        Run data containing deep-hedge PnL and optional benchmark PnL.
    out_dir : Path
        Output directory for chart image files.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available — skipping Chart 1")
        return

    sim      = run.sim.lower()
    pnl_np   = run.pnl.numpy()
    fig, ax  = plt.subplots(figsize=(8, 5))

    # Shared bin range
    all_pnl = [pnl_np]
    if run.bs_pnl is not None:
        all_pnl.append(run.bs_pnl.numpy())
    vmin = min(p.min() for p in all_pnl)
    vmax = max(p.max() for p in all_pnl)
    bins = np.linspace(vmin, vmax, 80)

    ax.hist(pnl_np, bins=bins, alpha=0.6, color="steelblue",
            label="Deep Hedge", density=True)

    if run.bs_pnl is not None:
        ax.hist(run.bs_pnl.numpy(), bins=bins, alpha=0.5,
                color="tomato", label="BS $\\Delta$ Benchmark", density=True)

    # CVaR marker
    alpha   = run.config.get("alpha", 0.95)
    k       = max(1, int(math.ceil((1 - alpha) * len(pnl_np))))
    var_val = -float(np.sort(pnl_np)[k - 1])
    ax.axvline(-var_val, color="steelblue", linestyle="--",
               linewidth=1.5, label=f"VaR$_{{0.95}}$ = {var_val:.2f}")

    sim_label = {"bs": "Black-Scholes", "heston": "Heston",
                 "nga": "NGA"}.get(sim, sim.upper())
    ax.set_title(f"PnL Distribution — {sim_label} ({run.run_name})", fontsize=13)
    ax.set_xlabel("Terminal PnL", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()

    path = out_dir / f"chart1_pnl_histogram_{sim}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 1 ({sim})    → {path}")


# ---------------------------------------------------------------------------
# Chart 2 — Loss curves
# ---------------------------------------------------------------------------

def _plot_loss_curves(runs: list[RunInfo], out_dir: Path) -> None:
    """Plot training and validation loss curves across runs.

    Parameters
    ----------
    runs : list[RunInfo]
        Collection of runs, each optionally containing logged losses.
    out_dir : Path
        Output directory for ``chart2_loss_curves.png``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping Chart 2")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]
    sim_labels = {"bs": "Black-Scholes", "heston": "Heston", "nga": "NGA"}

    for i, run in enumerate(runs):
        if not run.log:
            continue
        epochs     = [int(r["epoch"])      for r in run.log]
        train_loss = [float(r["train_loss"]) for r in run.log]
        val_loss   = [float(r["val_loss"])   for r in run.log]

        sim_label = sim_labels.get(run.sim.lower(), run.sim.upper())
        c         = colors[i % len(colors)]

        ax.plot(epochs, train_loss, color=c, linewidth=1.5, alpha=0.7,
                label=f"{sim_label} train")
        ax.plot(epochs, val_loss, color=c, linewidth=1.5, linestyle="--",
                label=f"{sim_label} val")

    ax.set_title("Training Loss Curves (CVaR$_{0.95}$)", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("CVaR Loss", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = out_dir / "chart2_loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 2           → {path}")


# ---------------------------------------------------------------------------
# Chart 4 — Per-timestep hedging error
# ---------------------------------------------------------------------------

def _plot_per_timestep_error(run: RunInfo, out_dir: Path) -> None:
    """Plot per-timestep delta diagnostics for one run.

    Parameters
    ----------
    run : RunInfo
        Run descriptor with configuration and checkpoint paths.
    out_dir : Path
        Output directory for saved chart files.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping Chart 4")
        return

    sim         = run.sim.lower()
    dataset_dir = Path(run.config["dataset_dir"])
    hidden      = run.config.get("hidden", 64)
    # Reload test batch and compute per-step delta error
    batch    = load_dataset(dataset_dir, split="test")
    features = build_features(batch)

    ckpt = torch.load(
        run.run_dir / "checkpoints" / "best_model.pt",
        map_location="cpu", weights_only=True
    )
    net = BaselineFeedforwardNetwork(hidden=hidden)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    with torch.no_grad():
        net_deltas = net.forward_trajectory(features)   # (N, T)

    timesteps = np.arange(batch.n_steps)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Deep hedge: per-step std of delta (proxy for hedging uncertainty)
    delta_std = net_deltas.std(dim=0).numpy()
    ax.plot(timesteps, delta_std, color="steelblue", linewidth=2,
            label="Deep Hedge $\\delta$ std")

    # For BS sim: overlay per-step MAE vs analytical delta
    if sim == "bs":
        sigma = run.config.get("simulator_params", {}).get("sigma", 0.2)
        S_t   = batch.paths_S[:, :-1]
        tau_t = batch.T_mat - batch.paths_t[:, :-1]
        bs_d  = bs_call_delta(S_t, batch.K, tau_t, sigma=sigma, r=0.0)
        mae_per_step = (net_deltas - bs_d).abs().mean(dim=0).numpy()
        ax.plot(timesteps, mae_per_step, color="tomato", linewidth=2,
                linestyle="--", label="MAE vs BS $\\Delta$")

    sim_label = {"bs": "Black-Scholes", "heston": "Heston",
                 "nga": "NGA"}.get(sim, sim.upper())
    ax.set_title(f"Per-Timestep Delta Analysis — {sim_label} ({run.run_name})",
                 fontsize=13)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = out_dir / f"chart4_per_timestep_{sim}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 4 ({sim})    → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation entrypoint.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Deep hedging evaluation suite — Table 1, Charts 1/2/4."
    )
    p.add_argument(
        "run_dirs", nargs="+",
        help="One or more run directories, e.g. results/runs/bs/bs_baseline_frictionless"
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory. Default: results/evaluation/<first_run_name>"
    )
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]

    if args.out_dir is None:
        out_dir = Path("results/evaluation") / run_dirs[0].name
    else:
        out_dir = Path(args.out_dir)

    run_evaluation(run_dirs, out_dir)
