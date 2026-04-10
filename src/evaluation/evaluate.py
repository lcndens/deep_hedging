"""Deep Hedging Evaluation Suite — v2.

Loads trained run directories and produces focused tables and charts
organized by comparison axis.

Outputs
-------
tables/
    table1_payoff_difficulty.csv / .tex
    table2_objective_comparison.csv / .tex
    table3_friction_impact.csv / .tex
    table4_greek_gap.csv / .tex
charts/
    chart1_pnl_kde_{payoff}.png          — one per payoff
    chart2_convergence_{payoff}.png      — one per payoff
    chart3_cvar_heatmap.png
    chart4_delta_holdings_{payoff}.png   — one per payoff
    chart5_greek_gap.png

Usage
-----
    python -m src.evaluation.evaluate \\
        results/runs/bs/AA_bs_european_single_cvar_frictionless \\
        results/runs/heston/AA_heston_european_single_cvar_frictionless \\
        ... \\
        --out_dir results/evaluation/full_grid

    # Filter to a single payoff
    python -m src.evaluation.evaluate ... --out_dir ... --payoff_filter barrier

    # Filter to a single simulator
    python -m src.evaluation.evaluate ... --out_dir ... --sim_filter heston
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.io.dataset_loader import load_dataset, DatasetBatch
from src.state.builder import build_features
from src.derivatives.european import call_payoff
from src.derivatives.barrier import compute_barrier_payoff
from src.derivatives.asian import asian_call_payoff
from src.derivatives.lookback import lookback_call_payoff
from src.derivatives.binary import binary_call_payoff
from src.frictions.proportional import proportional_cost
from src.pnl.compute import compute_pnl
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.evaluation.bs_delta_check import bs_call_delta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYOFF_ORDER   = ["european", "asian", "lookback", "barrier", "binary"]
PAYOFF_LABELS  = {
    "european": "European",
    "asian":    "Asian",
    "lookback": "Lookback",
    "barrier":  "Barrier",
    "binary":   "Binary",
}
SIM_LABELS = {"bs": "Black-Scholes", "heston": "Heston"}
OBJ_LABELS = {"cvar": "CVaR", "mean_variance": "MV"}

CLAMP_PAYOFFS = {"european", "asian", "binary"}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """All data extracted from a single trained run directory."""
    run_dir:    Path
    run_name:   str
    sim:        str           # "bs" or "heston"
    payoff:     str           # "european" | "asian" | "lookback" | "barrier" | "binary"
    instrument: str           # "single" | "multi"
    objective:  str           # "cvar" | "mean_variance"
    lam:        float         # λ for mean_variance; nan for cvar
    epsilon:    float         # 0.0 = frictionless, 0.01 = proportional
    config:     dict
    pnl:        torch.Tensor  # (N_test,)
    bs_pnl:     Optional[torch.Tensor]  # (N_test,) BS delta benchmark, BS runs only
    deltas:     torch.Tensor  # (N_test, T) single or (N_test, T, 2) multi
    log:        list[dict]    # rows from train_log.csv


@dataclass
class Metrics:
    """Scalar evaluation metrics for one run."""
    run_name:   str
    sim:        str
    payoff:     str
    instrument: str
    objective:  str
    lam:        float
    epsilon:    float
    cvar_95:    float
    var_95:     float
    mean_pnl:   float
    std_pnl:    float
    best_epoch: int
    n_epochs:   int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    run_dirs:      list[Path],
    out_dir:       Path,
    payoff_filter: Optional[str] = None,
    sim_filter:    Optional[str] = None,
) -> None:
    """Full evaluation pipeline.

    Parameters
    ----------
    run_dirs : list[Path]
        Paths to trained run directories.
    out_dir : Path
        Root output directory. Sub-directories tables/ and charts/ are created.
    payoff_filter : str, optional
        If given, restrict analysis to runs with this payoff type.
    sim_filter : str, optional
        If given, restrict analysis to runs with this simulator.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "charts").mkdir(exist_ok=True)

    print(f"\nEvaluating {len(run_dirs)} run(s) → {out_dir}\n")

    # Load all runs, skipping failures gracefully
    runs: list[RunInfo] = []
    for d in run_dirs:
        try:
            r = _load_run(d)
            runs.append(r)
        except Exception as e:
            print(f"  WARNING: skipping {d} — {e}")

    if not runs:
        print("No runs loaded successfully. Exiting.")
        return

    # Apply filters
    if payoff_filter:
        runs = [r for r in runs if r.payoff == payoff_filter]
        print(f"  Filtered to payoff={payoff_filter}: {len(runs)} run(s)")
    if sim_filter:
        runs = [r for r in runs if r.sim == sim_filter]
        print(f"  Filtered to sim={sim_filter}: {len(runs)} run(s)")

    metrics = [_compute_metrics(r) for r in runs]

    # Tables
    _save_table1(metrics, runs, out_dir / "tables")
    _save_table2(metrics, runs, out_dir / "tables")
    _save_table3(metrics, runs, out_dir / "tables")
    _save_table4(metrics, runs, out_dir / "tables")

    # Charts
    _plot_pnl_kde(runs, out_dir / "charts")
    _plot_convergence(runs, out_dir / "charts")
    _plot_cvar_heatmap(metrics, out_dir / "charts")
    _plot_delta_holdings(runs, out_dir / "charts")
    _plot_greek_gap(metrics, out_dir / "charts")

    print(f"\nDone. All outputs saved to: {out_dir}\n")


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> RunInfo:
    """Load a trained run from disk and compute test PnL and deltas."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    sim           = config["sim"]
    payoff_type   = config.get("payoff", "european")
    instrument    = config.get("instrument", "single")
    objective     = config.get("objective", "cvar")
    lam           = float(config.get("lam", float("nan"))) if objective == "mean_variance" else float("nan")
    epsilon       = float(config.get("epsilon", 0.0))
    hidden        = int(config.get("hidden", 64))
    barrier_level = config.get("barrier", None)
    cash          = float(config.get("cash", 1.0))
    n_instruments = 2 if instrument == "multi" else 1
    dataset_dir   = Path(config["dataset_dir"])

    print(f"  Loading: {run_dir.name}  (sim={sim}, payoff={payoff_type}, "
          f"instrument={instrument}, obj={objective}, ε={epsilon})")

    # Dataset
    batch = load_dataset(dataset_dir, split="test")

    # Features
    features = build_features(
        batch,
        n_instruments=n_instruments,
        payoff=payoff_type,
        barrier_level=barrier_level,
    )

    # Network
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    net = BaselineFeedforwardNetwork(
        hidden=hidden,
        n_instruments=n_instruments,
        feature_dim=features.shape[2],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    with torch.no_grad():
        deltas = net.forward_trajectory(features)   # (N,T) or (N,T,2)

        # Clamp spot delta for payoffs where delta is theoretically in [0,1]
        if payoff_type in CLAMP_PAYOFFS:
            if n_instruments == 1:
                deltas = deltas.clamp(0.0, 1.0)
            else:
                # Clamp only the spot instrument (index 0); VS delta is unbounded
                d_spot = deltas[:, :, :1].clamp(0.0, 1.0)
                deltas = torch.cat([d_spot, deltas[:, :, 1:]], dim=-1)

        # Payoff
        payoff_tensor = _compute_payoff(
            batch, payoff_type, barrier_level, cash, n_instruments
        )

        # PnL
        if n_instruments == 2:
            paths_prices = torch.stack([batch.paths_S, batch.paths_S2], dim=-1)
            total_cost   = proportional_cost(paths_prices[:, :-1], deltas, epsilon=epsilon)
            pnl          = compute_pnl(paths_prices, deltas, payoff_tensor, total_cost)
        else:
            total_cost = proportional_cost(batch.paths_S[:, :-1], deltas, epsilon=epsilon)
            pnl        = compute_pnl(batch.paths_S, deltas, payoff_tensor, total_cost)

    # BS delta benchmark (BS runs only, always uses European call payoff)
    bs_pnl = None
    if sim.lower() == "bs":
        sigma = config.get("simulator_params", {}).get("sigma", 0.2)
        bs_pnl = _compute_bs_delta_pnl(batch, sigma=sigma)

    log = _load_log(run_dir)

    return RunInfo(
        run_dir    = run_dir,
        run_name   = run_dir.name,
        sim        = sim,
        payoff     = payoff_type,
        instrument = instrument,
        objective  = objective,
        lam        = lam,
        epsilon    = epsilon,
        config     = config,
        pnl        = pnl,
        bs_pnl     = bs_pnl,
        deltas     = deltas,
        log        = log,
    )


def _compute_payoff(
    batch:         DatasetBatch,
    payoff_type:   str,
    barrier_level: Optional[float],
    cash:          float,
    n_instruments: int,
) -> torch.Tensor:
    """Dispatch to the correct payoff function."""
    S_full = batch.paths_S   # (N, T+1)
    K      = batch.K

    if payoff_type == "barrier":
        return compute_barrier_payoff(S_full, K, barrier_level)
    elif payoff_type == "asian":
        return asian_call_payoff(S_full, K)
    elif payoff_type == "lookback":
        return lookback_call_payoff(S_full, K)
    elif payoff_type == "binary":
        return binary_call_payoff(S_full[:, -1], K, cash=cash)
    else:  # european
        return call_payoff(S_full[:, -1], K)


def _compute_bs_delta_pnl(
    batch: DatasetBatch,
    sigma: float = 0.2,
    r:     float = 0.0,
) -> torch.Tensor:
    """BS analytical delta benchmark PnL using European call payoff."""
    S_t    = batch.paths_S[:, :-1]
    t_t    = batch.paths_t[:, :-1]
    tau_t  = batch.T_mat - t_t
    deltas = bs_call_delta(S_t, batch.K, tau_t, sigma, r)
    payoff = call_payoff(batch.paths_S[:, -1], batch.K)
    cost   = proportional_cost(batch.paths_S[:, :-1], deltas, epsilon=0.0)
    return compute_pnl(batch.paths_S, deltas, payoff, cost)


def _load_log(run_dir: Path) -> list[dict]:
    log_path = run_dir / "logs" / "train_log.csv"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(run: RunInfo) -> Metrics:
    pnl  = run.pnl.numpy()
    N    = len(pnl)
    alpha = 0.95
    k     = max(1, int(math.ceil((1 - alpha) * N)))
    sorted_pnl = np.sort(pnl)
    cvar_95    = float(-sorted_pnl[:k].mean())
    var_95     = float(-sorted_pnl[k - 1])

    best_epoch = 0
    n_epochs   = 0
    if run.log:
        n_epochs   = len(run.log)
        val_losses = [float(r["val_loss"]) for r in run.log]
        best_epoch = int(np.argmin(val_losses)) + 1

    return Metrics(
        run_name   = run.run_name,
        sim        = run.sim,
        payoff     = run.payoff,
        instrument = run.instrument,
        objective  = run.objective,
        lam        = run.lam,
        epsilon    = run.epsilon,
        cvar_95    = cvar_95,
        var_95     = var_95,
        mean_pnl   = float(pnl.mean()),
        std_pnl    = float(pnl.std()),
        best_epoch = best_epoch,
        n_epochs   = n_epochs,
    )


def _lookup(
    metrics:    list[Metrics],
    sim:        str,
    instrument: str,
    objective:  str,
    payoff:     str,
    epsilon:    float,
    lam:        Optional[float] = None,
) -> Optional[Metrics]:
    """Find a single matching metric entry."""
    for m in metrics:
        if (m.sim == sim and m.instrument == instrument
                and m.objective == objective and m.payoff == payoff
                and abs(m.epsilon - epsilon) < 1e-6):
            if lam is not None:
                if math.isnan(m.lam) or abs(m.lam - lam) > 1e-6:
                    continue
            return m
    return None


def _fmt(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _write_latex(path: Path, headers: list[str], rows: list[list[str]],
                 caption: str = "", label: str = "") -> None:
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines))


def _write_csv_table(path: Path, headers: list[str],
                     rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Table 1 — Payoff difficulty ranking
# Filter: frictionless, CVaR objective
# Rows: payoffs | Cols: BS-single, Heston-single, Heston-multi
# ---------------------------------------------------------------------------

def _save_table1(metrics: list[Metrics], runs: list[RunInfo],
                 out_dir: Path) -> None:
    headers = ["Payoff", "BS Single CVaR95",
               "Heston Single CVaR95", "Heston Multi CVaR95"]
    rows = []
    for payoff in PAYOFF_ORDER:
        bs_m   = _lookup(metrics, "bs",     "single", "cvar", payoff, 0.0)
        hs_m   = _lookup(metrics, "heston", "single", "cvar", payoff, 0.0)
        hm_m   = _lookup(metrics, "heston", "multi",  "cvar", payoff, 0.0)
        rows.append([
            PAYOFF_LABELS.get(payoff, payoff),
            _fmt(bs_m.cvar_95  if bs_m  else None),
            _fmt(hs_m.cvar_95  if hs_m  else None),
            _fmt(hm_m.cvar_95  if hm_m  else None),
        ])

    _write_csv_table(out_dir / "table1_payoff_difficulty.csv", headers, rows)
    _write_latex(
        out_dir / "table1_payoff_difficulty.tex", headers, rows,
        caption="Payoff difficulty ranking (frictionless, CVaR objective, CVaR$_{95}$)",
        label="tab:payoff_difficulty",
    )
    print(f"  Table 1 → {out_dir / 'table1_payoff_difficulty.csv'}")


# ---------------------------------------------------------------------------
# Table 2 — Objective comparison
# Filter: Heston, frictionless, single instrument
# Rows: objectives | Cols: payoff × (CVaR95, VaR95)
# ---------------------------------------------------------------------------

def _save_table2(metrics: list[Metrics], runs: list[RunInfo],
                 out_dir: Path) -> None:
    obj_configs = [
        ("cvar",          float("nan"), "CVaR"),
        ("mean_variance", 1.0,          "MV λ=1"),
        ("mean_variance", 0.0,          "MV λ=0 (PnL)"),
    ]
    # Build headers: Payoff | (PayoffA CVaR95, PayoffA VaR95, ...) per payoff
    sub_headers = []
    for p in PAYOFF_ORDER:
        lbl = PAYOFF_LABELS.get(p, p)
        sub_headers += [f"{lbl} CVaR95", f"{lbl} VaR95"]
    headers = ["Objective"] + sub_headers

    rows = []
    for obj, lam, obj_label in obj_configs:
        row = [obj_label]
        for payoff in PAYOFF_ORDER:
            lam_arg = lam if obj == "mean_variance" else None
            m = _lookup(metrics, "heston", "single", obj, payoff, 0.0,
                        lam=lam_arg)
            row.append(_fmt(m.cvar_95 if m else None))
            row.append(_fmt(m.var_95  if m else None))
        rows.append(row)

    _write_csv_table(out_dir / "table2_objective_comparison.csv", headers, rows)
    _write_latex(
        out_dir / "table2_objective_comparison.tex", headers, rows,
        caption="Objective comparison — Heston, frictionless, single instrument",
        label="tab:objective_comparison",
    )
    print(f"  Table 2 → {out_dir / 'table2_objective_comparison.csv'}")


# ---------------------------------------------------------------------------
# Table 3 — Friction impact
# Filter: Heston, CVaR, single instrument
# Rows: payoffs | Cols: frictionless CVaR95, proportional CVaR95, delta
# ---------------------------------------------------------------------------

def _save_table3(metrics: list[Metrics], runs: list[RunInfo],
                 out_dir: Path) -> None:
    headers = ["Payoff", "Frictionless CVaR95", "Proportional CVaR95",
               "Δ (friction cost)"]
    rows = []
    for payoff in PAYOFF_ORDER:
        mf = _lookup(metrics, "heston", "single", "cvar", payoff, 0.0)
        mp = _lookup(metrics, "heston", "single", "cvar", payoff, 0.01)
        if mf is not None and mp is not None:
            delta = mp.cvar_95 - mf.cvar_95
            rows.append([
                PAYOFF_LABELS.get(payoff, payoff),
                _fmt(mf.cvar_95),
                _fmt(mp.cvar_95),
                _fmt(delta),
            ])
        else:
            rows.append([
                PAYOFF_LABELS.get(payoff, payoff),
                _fmt(mf.cvar_95 if mf else None),
                _fmt(mp.cvar_95 if mp else None),
                "—",
            ])

    _write_csv_table(out_dir / "table3_friction_impact.csv", headers, rows)
    _write_latex(
        out_dir / "table3_friction_impact.tex", headers, rows,
        caption="Friction impact — Heston, CVaR objective, single instrument",
        label="tab:friction_impact",
    )
    print(f"  Table 3 → {out_dir / 'table3_friction_impact.csv'}")


# ---------------------------------------------------------------------------
# Table 4 — Greek exposure gap
# Filter: Heston, frictionless, CVaR
# Rows: payoffs sorted by gap desc | Cols: single, multi, gap
# ---------------------------------------------------------------------------

def _save_table4(metrics: list[Metrics], runs: list[RunInfo],
                 out_dir: Path) -> None:
    headers = ["Payoff", "Single CVaR95", "Multi CVaR95", "Gap (↓ = improvement)"]

    gap_rows = []
    for payoff in PAYOFF_ORDER:
        ms = _lookup(metrics, "heston", "single", "cvar", payoff, 0.0)
        mm = _lookup(metrics, "heston", "multi",  "cvar", payoff, 0.0)
        gap = (ms.cvar_95 - mm.cvar_95) if (ms and mm) else None
        gap_rows.append((payoff, ms, mm, gap))

    # Sort by gap descending (largest gap first); None gaps go last
    gap_rows.sort(key=lambda x: (x[3] is None, -(x[3] or 0)))

    rows = []
    for payoff, ms, mm, gap in gap_rows:
        rows.append([
            PAYOFF_LABELS.get(payoff, payoff),
            _fmt(ms.cvar_95 if ms else None),
            _fmt(mm.cvar_95 if mm else None),
            _fmt(gap),
        ])

    _write_csv_table(out_dir / "table4_greek_gap.csv", headers, rows)
    _write_latex(
        out_dir / "table4_greek_gap.tex", headers, rows,
        caption="Greek exposure gap — Heston, frictionless, CVaR objective",
        label="tab:greek_gap",
    )
    print(f"  Table 4 → {out_dir / 'table4_greek_gap.csv'}")


# ---------------------------------------------------------------------------
# Chart 1 — PnL distribution KDE (one figure per payoff)
# ---------------------------------------------------------------------------

def _plot_pnl_kde(runs: list[RunInfo], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError:
        print("  WARNING: matplotlib/scipy not available — skipping Chart 1")
        return

    for payoff in PAYOFF_ORDER:
        bs_run = _find_run(runs, "bs",     "single", "cvar", payoff, 0.0)
        hs_run = _find_run(runs, "heston", "single", "cvar", payoff, 0.0)
        hm_run = _find_run(runs, "heston", "multi",  "cvar", payoff, 0.0)

        available = [(bs_run, "steelblue",  "BS Single"),
                     (hs_run, "seagreen",   "Heston Single"),
                     (hm_run, "darkorange", "Heston Multi")]
        available = [(r, c, l) for r, c, l in available if r is not None]

        if not available:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        all_pnl = np.concatenate([r.pnl.numpy() for r, _, _ in available])
        x_min, x_max = np.percentile(all_pnl, 0.5), np.percentile(all_pnl, 99.5)
        x_grid = np.linspace(x_min, x_max, 400)

        for run, color, label in available:
            pnl_np = run.pnl.numpy()
            try:
                kde = gaussian_kde(pnl_np, bw_method="scott")
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)
            except Exception:
                pass

            # CVaR95 marker
            k = max(1, int(math.ceil(0.05 * len(pnl_np))))
            var95 = float(-np.sort(pnl_np)[k - 1])
            ax.axvline(-var95, color=color, linestyle="--", linewidth=1.2,
                       alpha=0.8, label=f"{label} VaR$_{{95}}$={var95:.3f}")

        ax.set_title(f"PnL Distribution — {PAYOFF_LABELS.get(payoff, payoff)}",
                     fontsize=13)
        ax.set_xlabel("Terminal PnL", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        path = out_dir / f"chart1_pnl_kde_{payoff}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart 1 ({payoff}) → {path}")


# ---------------------------------------------------------------------------
# Chart 2 — Training convergence (one figure per payoff, 3 subplots)
# ---------------------------------------------------------------------------

def _plot_convergence(runs: list[RunInfo], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping Chart 2")
        return

    obj_configs = [
        ("cvar",          float("nan"), "CVaR"),
        ("mean_variance", 1.0,          "MV λ=1"),
        ("mean_variance", 0.0,          "MV λ=0"),
    ]
    sim_styles = [
        ("bs",     "single", "steelblue", "BS"),
        ("heston", "single", "seagreen",  "Heston"),
    ]

    for payoff in PAYOFF_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        fig.suptitle(
            f"Convergence — {PAYOFF_LABELS.get(payoff, payoff)}", fontsize=13
        )

        for ax, (obj, lam, obj_label) in zip(axes, obj_configs):
            ax.set_title(obj_label, fontsize=11)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Val Loss", fontsize=10)

            for sim, instr, color, sim_label in sim_styles:
                run = _find_run(runs, sim, instr, obj, payoff, 0.0, lam=lam)
                if run is None or not run.log:
                    continue
                epochs    = [int(r["epoch"])       for r in run.log]
                val_loss  = [float(r["val_loss"])   for r in run.log]
                ax.plot(epochs, val_loss, color=color, linewidth=1.5,
                        label=sim_label)

            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        path = out_dir / f"chart2_convergence_{payoff}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart 2 ({payoff}) → {path}")


# ---------------------------------------------------------------------------
# Chart 3 — CVaR95 heatmap
# Filter: frictionless, CVaR objective
# ---------------------------------------------------------------------------

def _plot_cvar_heatmap(metrics: list[Metrics], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  WARNING: matplotlib not available — skipping Chart 3")
        return

    cols = [
        ("bs",     "single", "BS\nSingle"),
        ("heston", "single", "Heston\nSingle"),
        ("heston", "multi",  "Heston\nMulti"),
    ]

    data   = np.full((len(PAYOFF_ORDER), len(cols)), np.nan)
    labels = np.full((len(PAYOFF_ORDER), len(cols)), "", dtype=object)

    for i, payoff in enumerate(PAYOFF_ORDER):
        for j, (sim, instr, _) in enumerate(cols):
            m = _lookup(metrics, sim, instr, "cvar", payoff, 0.0)
            if m is not None:
                data[i, j]   = m.cvar_95
                labels[i, j] = f"{m.cvar_95:.3f}"

    fig, ax = plt.subplots(figsize=(7, 5))
    valid = data[~np.isnan(data)]
    vmin  = float(valid.min()) if len(valid) else 0
    vmax  = float(valid.max()) if len(valid) else 1

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], fontsize=10)
    ax.set_yticks(range(len(PAYOFF_ORDER)))
    ax.set_yticklabels(
        [PAYOFF_LABELS.get(p, p) for p in PAYOFF_ORDER], fontsize=10
    )

    for i in range(len(PAYOFF_ORDER)):
        for j in range(len(cols)):
            txt = labels[i, j]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="black" if data[i, j] < (vmin + vmax) / 2 else "white")

    plt.colorbar(im, ax=ax, label="CVaR$_{95}$")
    ax.set_title("CVaR$_{95}$ Heatmap — Frictionless, CVaR Objective", fontsize=12)
    plt.tight_layout()

    path = out_dir / "chart3_cvar_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 3           → {path}")


# ---------------------------------------------------------------------------
# Chart 4 — Delta holdings over time (one figure per payoff)
# ---------------------------------------------------------------------------

def _plot_delta_holdings(runs: list[RunInfo], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping Chart 4")
        return

    series_cfg = [
        ("bs",     "single", "cvar", 0.0, "steelblue",  "BS Single",         False),
        ("heston", "single", "cvar", 0.0, "seagreen",   "Heston Single",      False),
        ("heston", "multi",  "cvar", 0.0, "darkorange",  "Heston Multi δ¹",   False),
        ("heston", "multi",  "cvar", 0.0, "darkorange",  "Heston Multi δ² (VS)", True),
    ]

    for payoff in PAYOFF_ORDER:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False

        for sim, instr, obj, eps, color, label, is_vs in series_cfg:
            run = _find_run(runs, sim, instr, obj, payoff, eps)
            if run is None:
                continue

            deltas_np = run.deltas.numpy()   # (N, T) or (N, T, 2)
            n_instr   = 2 if deltas_np.ndim == 3 else 1

            if is_vs:
                if n_instr < 2:
                    continue
                d = deltas_np[:, :, 1]   # variance swap delta
            elif n_instr == 2:
                d = deltas_np[:, :, 0]   # spot delta from multi run
            else:
                d = deltas_np            # (N, T) single instrument

            T         = d.shape[1]
            timesteps = np.arange(T)
            median    = np.median(d, axis=0)
            p10       = np.percentile(d, 10, axis=0)
            p90       = np.percentile(d, 90, axis=0)
            linestyle = "--" if is_vs else "-"

            ax.plot(timesteps, median, color=color, linewidth=2,
                    linestyle=linestyle, label=label)
            ax.fill_between(timesteps, p10, p90, color=color, alpha=0.12)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_title(
            f"Delta Holdings — {PAYOFF_LABELS.get(payoff, payoff)}", fontsize=13
        )
        ax.set_xlabel("Timestep $t$", fontsize=11)
        ax.set_ylabel("Hedge ratio $\\delta_t$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        path = out_dir / f"chart4_delta_holdings_{payoff}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart 4 ({payoff}) → {path}")


# ---------------------------------------------------------------------------
# Chart 5 — Greek exposure gap bar chart
# Filter: Heston, frictionless, CVaR
# ---------------------------------------------------------------------------

def _plot_greek_gap(metrics: list[Metrics], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping Chart 5")
        return

    gap_data = []
    for payoff in PAYOFF_ORDER:
        ms = _lookup(metrics, "heston", "single", "cvar", payoff, 0.0)
        mm = _lookup(metrics, "heston", "multi",  "cvar", payoff, 0.0)
        if ms is None and mm is None:
            continue
        gap = (ms.cvar_95 - mm.cvar_95) if (ms and mm) else None
        gap_data.append((payoff, ms, mm, gap))

    if not gap_data:
        print("  Chart 5: no matching runs — skipping")
        return

    # Sort by gap descending
    gap_data.sort(key=lambda x: (x[3] is None, -(x[3] or 0)))

    labels = [PAYOFF_LABELS.get(p, p) for p, _, _, _ in gap_data]
    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    single_vals = [ms.cvar_95 if ms else 0 for _, ms, _, _ in gap_data]
    multi_vals  = [mm.cvar_95 if mm else 0 for _, _, mm, _ in gap_data]

    bars_s = ax.bar(x - width / 2, single_vals, width, label="Single instrument",
                    color="steelblue", alpha=0.85)
    bars_m = ax.bar(x + width / 2, multi_vals,  width, label="Multi instrument",
                    color="darkorange", alpha=0.85)

    # Value labels on top of bars
    for bar in bars_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_m:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("CVaR$_{95}$", fontsize=11)
    ax.set_title("Greek Exposure Gap — Heston, Frictionless, CVaR", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()

    path = out_dir / "chart5_greek_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart 5           → {path}")


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _find_run(
    runs:       list[RunInfo],
    sim:        str,
    instrument: str,
    objective:  str,
    payoff:     str,
    epsilon:    float,
    lam:        Optional[float] = None,
) -> Optional[RunInfo]:
    """Find a single matching RunInfo."""
    for r in runs:
        if (r.sim == sim and r.instrument == instrument
                and r.objective == objective and r.payoff == payoff
                and abs(r.epsilon - epsilon) < 1e-6):
            if lam is not None:
                if math.isnan(r.lam) or abs(r.lam - lam) > 1e-6:
                    continue
            return r
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep hedging evaluation suite — Tables 1-4, Charts 1-5."
    )
    p.add_argument(
        "run_dirs", nargs="+",
        help="One or more trained run directories."
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Root output directory. Sub-dirs tables/ and charts/ are created."
    )
    p.add_argument(
        "--payoff_filter", default=None,
        choices=["european", "asian", "lookback", "barrier", "binary"],
        help="Restrict evaluation to runs with this payoff type."
    )
    p.add_argument(
        "--sim_filter", default=None,
        choices=["bs", "heston"],
        help="Restrict evaluation to runs with this simulator."
    )
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    out_dir  = Path(args.out_dir)

    run_evaluation(
        run_dirs      = run_dirs,
        out_dir       = out_dir,
        payoff_filter = args.payoff_filter,
        sim_filter    = args.sim_filter,
    )
