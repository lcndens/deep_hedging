"""CLI for plotting sample paths from generated observations data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_metadata(run_dir: Path) -> dict:
    """Load metadata.json for a dataset run."""
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_split_df(run_dir: Path, split: str) -> pd.DataFrame:
    """Load observation records for one split from parquet files."""
    split_dir = run_dir / "observations" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    parquet_files = sorted(split_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")

    # Supports one or many part files
    table = pq.read_table([str(p) for p in parquet_files])
    df = table.to_pandas()

    # Minimal required columns for plotting
    for col in ("path_id", "t_years", "S"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {split_dir}")

    return df


def _stable_seed_from_run_id(run_id: str) -> int:
    """Convert run_id to a deterministic 32-bit seed."""
    h = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for plotting."""
    p = argparse.ArgumentParser(
        description="Plot sample simulated paths from observations/<split>/ and save the plot in that split folder."
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Dataset run directory (e.g., data/datasets/v1.0/bs/<run_id> or .../heston/<run_id>)",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n_paths_plot", type=int, default=12, help="Number of distinct path IDs to plot")
    p.add_argument("--out_name", type=str, default=None, help="Output PNG filename (default auto)")
    p.add_argument("--random", action="store_true", help="Randomly sample paths instead of taking the first N")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Load one split, plot selected paths, and save a PNG under that split folder."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in your environment first."
        ) from exc

    args = build_parser().parse_args(argv)

    run_dir = Path(args.run_dir)
    meta = load_metadata(run_dir)
    df = load_split_df(run_dir, args.split)

    sim = meta.get("simulator", "unknown")
    run_id = meta.get("run_id", run_dir.name)

    # Pick paths
    unique_ids = df["path_id"].drop_duplicates().to_numpy()
    if len(unique_ids) == 0:
        raise ValueError("No paths found to plot.")

    n = min(args.n_paths_plot, len(unique_ids))
    if args.random:
        # Deterministic randomness based on run_id hash for reproducibility
        rng = np.random.default_rng(_stable_seed_from_run_id(run_id))
        ids = rng.choice(unique_ids, size=n, replace=False)
    else:
        ids = unique_ids[:n]

    # Plot
    plt.figure()
    for pid in ids:
        sub = df[df["path_id"] == pid].sort_values("t_years")
        plt.plot(sub["t_years"], sub["S"])

    plt.xlabel("t (years)")
    plt.ylabel("S")
    plt.title(f"{sim} sample paths ({args.split}) — n={n}")

    out_name = args.out_name or f"sample_paths_{sim.lower()}_{args.split}.png"
    out_path = run_dir / "observations" / args.split / out_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
