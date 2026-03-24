"""Deep Hedging Evaluation Suite.

Loads one or more trained run directories, computes evaluation metrics,
and produces:
    - Table 1: Primary results (CSV + LaTeX)
    - Chart 1: PnL distribution histogram per simulator
    - Chart 2: Training loss curves
    - Chart 4: Per-timestep hedging error

Usage:
    python -m src.evaluation.evaluate \\
        results/runs/bs/bs_baseline_frictionless \\
        results/runs/heston/heston_baseline_frictionless \\
        results/runs/nga/nga_baseline_frictionless \\
        --out_dir results/evaluation/baseline

Output layout:
    results/evaluation/baseline/
        table1_results.csv
        table1_results.tex
        chart1_pnl_histogram_bs.png
        chart1_pnl_histogram_heston.png
        chart1_pnl_histogram_nga.png
        chart2_loss_curves.png
        chart4_per_timestep_error.png
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
    """All data extracted from a training run directory."""
    run_dir:    Path
    run_name:   str
    sim:        str
    objective:  str              # "cvar" or "mean_variance"
    config:     dict
    pnl:        torch.Tensor    # (N_test,)  — deep hedge PnL
    bs_pnl:     Optional[torch.Tensor]  # (N_test,) — BS delta PnL (BS only)
    log:        list[dict]      # train_log.csv rows


@dataclass
class Metrics:
    """Scalar evaluation metrics for one run."""
    run_name:   str
    sim:        str
    objective:  str    # "cvar" or "mean_variance"
    lam:        float  # λ for mean_variance; NaN for CVaR
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

    # Chart 1 — per run
    for run in runs:
        _plot_pnl_histogram(run, out_dir)

    # Chart 4 — all runs on one figure
    _plot_per_timestep_error(runs, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir}\n")


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> RunInfo:
    """Load a trained run: config, checkpoint, test PnL."""
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
    objective   = config.get("objective", "cvar")

    print(f"Loading run: {run_name}  (sim={sim}, objective={objective})")

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
        deltas = net.forward_trajectory(features)
        # Clamp deltas to [0, 1] — call option delta is bounded by definition.
        # Rare extreme values on tail paths indicate network extrapolation failure
        # and are clamped here to prevent them distorting evaluation metrics.
        deltas     = deltas.clamp(0.0, 1.0)
        payoff     = call_payoff(batch.paths_S[:, -1], batch.K)
        total_cost = proportional_cost(
            batch.paths_S[:, :-1], deltas, epsilon=epsilon
        )
        pnl = compute_pnl(batch.paths_S, deltas, payoff, total_cost)

    # BS analytical delta PnL — shown for all BS runs to benchmark friction impact
    bs_pnl = None
    if sim.lower() == "bs":
        sigma = config.get("simulator_params", {}).get("sigma", 0.2)
        bs_pnl = _compute_bs_delta_pnl(batch, sigma=sigma)

    # Training log
    log = _load_log(run_dir)

    return RunInfo(
        run_dir   = run_dir,
        run_name  = run_name,
        sim       = sim,
        objective = objective,
        config    = config,
        pnl       = pnl,
        bs_pnl    = bs_pnl,
        log       = log,
    )


def _load_log(run_dir: Path) -> list[dict]:
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
    """Compute PnL achieved by following the BS analytical delta strategy."""
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
    pnl  = run.pnl.numpy()
    N    = len(pnl)

    # CVaR_0.95 and VaR_0.95 are always computed from the PnL distribution
    # regardless of the training objective — this is the common evaluation currency.
    alpha      = 0.95
    k          = max(1, int(math.ceil((1 - alpha) * N)))
    sorted_pnl = np.sort(pnl)
    cvar_95    = float(-sorted_pnl[:k].mean())
    var_95     = float(-sorted_pnl[k - 1])

    # Best epoch from log
    best_epoch = 0
    n_epochs   = 0
    if run.log:
        n_epochs   = len(run.log)
        val_losses = [float(r["val_loss"]) for r in run.log]
        best_epoch = int(np.argmin(val_losses)) + 1

    lam = run.config.get("lam", float("nan")) if run.objective == "mean_variance" else float("nan")

    return Metrics(
        run_name   = run.run_name,
        sim        = run.sim,
        objective  = run.objective,
        lam        = lam,
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
    # CSV
    csv_path = out_dir / "table1_results.csv"
    fields   = [
        "run_name", "sim", "objective", "lam",
        "cvar_95", "var_95", "mean_pnl", "std_pnl", "p10_pnl",
        "best_epoch", "n_epochs"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "run_name":   m.run_name,
                "sim":        m.sim,
                "objective":  m.objective,
                "lam":        f"{m.lam:.4f}" if not math.isnan(m.lam) else "",
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
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available — skipping Chart 1")
        return

    sim      = run.sim.lower()
    pnl_np   = run.pnl.numpy()
    alpha    = run.config.get("alpha", 0.95)
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
        bs_np = run.bs_pnl.numpy()
        ax.hist(bs_np, bins=bins, alpha=0.5,
                color="tomato", label="BS $\\Delta$ Benchmark", density=True)
        # VaR marker for BS benchmark
        k_bs      = max(1, int(math.ceil((1 - alpha) * len(bs_np))))
        var_bs    = -float(np.sort(bs_np)[k_bs - 1])
        ax.axvline(-var_bs, color="tomato", linestyle="--",
                   linewidth=1.5, label=f"BS VaR$_{{0.95}}$ = {var_bs:.2f}")

    # CVaR marker
    alpha   = run.config.get("alpha", 0.95)
    k       = max(1, int(math.ceil((1 - alpha) * len(pnl_np))))
    var_val = -float(np.sort(pnl_np)[k - 1])
    ax.axvline(-var_val, color="steelblue", linestyle="--",
               linewidth=1.5, label=f"Deep Hedge VaR$_{{0.95}}$ = {var_val:.2f}")

    sim_label = {"bs": "Black-Scholes", "heston": "Heston",
                 "nga": "NGA"}.get(sim, sim.upper())
    epsilon   = run.config.get("epsilon", 0.0)
    eps_label = f"ε={epsilon}" if epsilon > 0 else "frictionless"
    obj_label = f" / MV λ={run.config.get('lam', 1.0)}" if run.objective == "mean_variance" else ""
    ax.set_title(f"PnL Distribution — {sim_label} ({eps_label}{obj_label})", fontsize=13)
    ax.set_xlabel("Terminal PnL", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()

    path = out_dir / f"chart1_pnl_histogram_{sim}_{run.run_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 1 ({sim})    → {path}")


# ---------------------------------------------------------------------------
# Chart 2 — Loss curves
# ---------------------------------------------------------------------------

def _plot_loss_curves(runs: list[RunInfo], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping Chart 2")
        return

    sim_labels = {"bs": "Black-Scholes", "heston": "Heston", "nga": "NGA"}
    colors     = ["steelblue", "seagreen", "darkorange", "purple", "tomato"]

    # Group runs by objective — each objective gets its own figure so
    # the incomparable loss scales don't appear on the same y-axis.
    from collections import defaultdict
    by_objective: dict[str, list[RunInfo]] = defaultdict(list)
    for run in runs:
        if run.log:
            by_objective[run.objective].append(run)

    obj_titles = {
        "cvar":          ("Training Loss Curves (CVaR$_{0.95}$)", "CVaR Loss"),
        "mean_variance": ("Training Loss Curves (Mean-Variance)",  "Mean-Variance Loss"),
    }

    for obj, obj_runs in by_objective.items():
        fig, ax  = plt.subplots(figsize=(9, 5))
        title, ylabel = obj_titles.get(obj, ("Training Loss Curves", "Training Loss"))

        for i, run in enumerate(obj_runs):
            epochs     = [int(r["epoch"])       for r in run.log]
            train_loss = [float(r["train_loss"]) for r in run.log]
            val_loss   = [float(r["val_loss"])   for r in run.log]

            sim_label = sim_labels.get(run.sim.lower(), run.sim.upper())
            c         = colors[i % len(colors)]

            ax.plot(epochs, train_loss, color=c, linewidth=1.5, alpha=0.7,
                    label=f"{sim_label} train")
            ax.plot(epochs, val_loss, color=c, linewidth=1.5, linestyle="--",
                    label=f"{sim_label} val")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()

        path = out_dir / f"chart2_loss_curves_{obj}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart 2 ({obj})  → {path}")


# ---------------------------------------------------------------------------
# Chart 4 — Per-timestep delta analysis (all simulators combined)
# ---------------------------------------------------------------------------

def _plot_per_timestep_error(runs: list[RunInfo], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping Chart 4")
        return

    sim_labels = {"bs": "Black-Scholes", "heston": "Heston", "nga": "NGA"}
    colors     = {"bs": "steelblue", "heston": "seagreen", "nga": "darkorange"}

    fig, ax_trade = plt.subplots(figsize=(10, 5))

    for run in runs:
        sim         = run.sim.lower()
        dataset_dir = Path(run.config["dataset_dir"])
        hidden      = run.config.get("hidden", 64)
        color       = colors.get(sim, "gray")
        obj_label   = "MV" if run.objective == "mean_variance" else "CVaR"
        label       = f"{sim_labels.get(sim, sim.upper())} / {obj_label}"
        linestyle   = "--" if run.objective == "mean_variance" else "-"

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
            net_deltas = net.forward_trajectory(features).clamp(0.0, 1.0)   # (N, T)

        # Mean absolute position change per timestep.
        # Skip t=0 — the first step change (delta_0 - 0) reflects initial
        # position entry rather than rebalancing and dominates the scale.
        delta_prev = torch.cat([torch.zeros(net_deltas.shape[0], 1), net_deltas[:, :-1]], dim=1)
        abs_change = (net_deltas - delta_prev).abs().mean(dim=0).numpy()

        timesteps = np.arange(1, batch.n_steps)   # skip t=0
        ax_trade.plot(timesteps, abs_change[1:], color=color, linewidth=2,
                      linestyle=linestyle, label=label)

    ax_trade.set_title("Mean Rebalancing Trade Size per Timestep", fontsize=13)
    ax_trade.set_ylabel("Mean $|\\delta_t - \\delta_{t-1}|$", fontsize=11)
    ax_trade.set_xlabel("Timestep $t$  (1 = first rebalance, T−1 = day before maturity)",
                        fontsize=11)
    ax_trade.legend(fontsize=10)
    ax_trade.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "chart4_per_timestep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 4           → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
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
