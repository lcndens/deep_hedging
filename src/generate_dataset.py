"""CLI for generating canonical simulated hedging datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schema.v1_0 import (
    SCHEMA_VERSION,
    OBS_SCHEMA,
    LATENT_STATE_SCHEMA,
    CONTRACTS_SCHEMA,
)
from src.io.parquet_writer import write_parquet_part
from src.utils.split import SplitConfig, split_path_ids

from src.simulators.bs import BSParams, simulate_observations as simulate_bs
from src.simulators.heston import HestonParams, simulate_observations as simulate_heston
from src.simulators.nga import NGAParams, simulate_observations as simulate_nga




def make_run_id(sim: str, seed: int, n_paths: int, n_steps: int) -> str:
    """Create a run identifier used as the dataset folder name."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{sim}_seed-{seed}_n-{n_paths}_steps-{n_steps}"


def write_contracts(out_path: Path, contract_type: str, strike: float, maturity_years: float) -> None:
    """Write the one-row contracts Parquet file for a dataset run."""
    df = pd.DataFrame([{
        "contract_id": 0,
        "type": contract_type,
        "strike": float(strike),
        "maturity_years": float(maturity_years),
    }])
    table = pa.Table.from_pandas(df, schema=CONTRACTS_SCHEMA, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset generation."""
    p = argparse.ArgumentParser(description="Generate canonical Parquet datasets (schema v1.0).")

    p.add_argument("--sim", type=str, default="bs", choices=["bs", "heston", "nga"])
    p.add_argument("--out_root", type=str, default="data/datasets/v1.0")
    p.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional deterministic run id (defaults to timestamped id).",
    )

    p.add_argument("--n_paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--maturity_years", type=float, default=1.0)
    p.add_argument("--n_steps", type=int, default=30)

    # Common spot args
    p.add_argument("--s0", type=float, default=100.0)
    p.add_argument("--m", type=float, default=0.0)

    # BS args
    p.add_argument("--sigma", type=float, default=0.2)

    # Heston args
    p.add_argument("--v0", type=float, default=0.04)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--theta", type=float, default=0.04)
    p.add_argument("--xi", type=float, default=0.3)
    p.add_argument("--rho", type=float, default=-0.7)

    # NGA args
    p.add_argument("--x0", type=float, default=100.0)
    p.add_argument("--a0_min", type=float, default=0.3)
    p.add_argument("--a0_max", type=float, default=0.7)
    p.add_argument("--a1_min", type=float, default=0.04)
    p.add_argument("--a1_max", type=float, default=0.06)
    p.add_argument("--b0_min", type=float, default=-0.2)
    p.add_argument("--b0_max", type=float, default=0.2)
    p.add_argument("--b1_min", type=float, default=-0.1)
    p.add_argument("--b1_max", type=float, default=0.1)
    p.add_argument("--gamma_min", type=float, default=0.5)
    p.add_argument("--gamma_max", type=float, default=1.5)

    # Contract (one contract per dataset)
    p.add_argument("--contract_type", type=str, default="call", choices=["call", "put"])
    p.add_argument("--strike", type=float, default=100.0)

    # Splits
    p.add_argument("--split_train", type=float, default=0.70)
    p.add_argument("--split_val", type=float, default=0.15)
    p.add_argument("--split_test", type=float, default=0.15)
    return p


def main(argv: Sequence[str] | None = None) -> Path:
    """Generate one dataset run and return the created run directory path."""
    args = build_parser().parse_args(argv)

    split_cfg = SplitConfig(train=args.split_train, val=args.split_val, test=args.split_test)

    run_id = args.run_id or make_run_id(args.sim, args.seed, args.n_paths, args.n_steps)
    run_dir = Path(args.out_root) / args.sim / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Simulate
    if args.sim == "bs":
        sim_cfg = BSParams(
            s0=args.s0,
            sigma=args.sigma,
            m=args.m,
            maturity_years=args.maturity_years,
            n_steps=args.n_steps,
            n_paths=args.n_paths,
            seed=args.seed,
        )
        obs, latent_state = simulate_bs(sim_cfg)
        simulator_name = "BS"
        simulator_params = asdict(sim_cfg)

    elif args.sim == "heston":
        sim_cfg = HestonParams(
            s0=args.s0,
            m=args.m,
            v0=args.v0,
            kappa=args.kappa,
            theta=args.theta,
            xi=args.xi,
            rho=args.rho,
            maturity_years=args.maturity_years,
            n_steps=args.n_steps,
            n_paths=args.n_paths,
            seed=args.seed,
        )
        obs, latent_state = simulate_heston(sim_cfg)
        simulator_name = "Heston"
        simulator_params = asdict(sim_cfg)

    elif args.sim == "nga":
        sim_cfg = NGAParams(
            x0=args.x0,
            a0_min=args.a0_min, a0_max=args.a0_max,
            a1_min=args.a1_min, a1_max=args.a1_max,
            b0_min=args.b0_min, b0_max=args.b0_max,
            b1_min=args.b1_min, b1_max=args.b1_max,
            gamma_min=args.gamma_min, gamma_max=args.gamma_max,
            maturity_years=args.maturity_years,
            n_steps=args.n_steps,
            n_paths=args.n_paths,
            seed=args.seed,
        )
        obs, latent_state = simulate_nga(sim_cfg)
        simulator_name = "NGA"
        simulator_params = asdict(sim_cfg)

    else:
        raise ValueError(f"Unsupported sim: {args.sim}")

    # Enforce schema columns exactly (no extras)
    obs_cols = [f.name for f in OBS_SCHEMA]
    if set(obs.columns) != set(obs_cols):
        raise ValueError(f"Observations columns mismatch. expected={obs_cols}, got={list(obs.columns)}")
    obs = obs.loc[:, obs_cols].copy()

    latent_cols = [f.name for f in LATENT_STATE_SCHEMA]
    if set(latent_state.columns) != set(latent_cols):
        raise ValueError(
            f"Latent-state columns mismatch. expected={latent_cols}, got={list(latent_state.columns)}"
        )
    latent_state = latent_state.loc[:, latent_cols].copy()

    # Split path ids and write splits to non-hive folder names
    splits = split_path_ids(n_paths=args.n_paths, seed=args.seed, cfg=split_cfg)

    for split_name, ids in splits.items():
        split_obs_df = obs[obs["path_id"].isin(ids)].loc[:, obs_cols].copy()
        obs_out_dir = run_dir / "observations" / split_name  # train/val/test
        write_parquet_part(split_obs_df, out_dir=obs_out_dir, schema=OBS_SCHEMA, compression="zstd")

        split_latent_df = latent_state[latent_state["path_id"].isin(ids)].loc[:, latent_cols].copy()
        latent_out_dir = run_dir / "latent_state" / split_name  # train/val/test
        write_parquet_part(
            split_latent_df,
            out_dir=latent_out_dir,
            schema=LATENT_STATE_SCHEMA,
            compression="zstd",
        )

    # Contracts + metadata
    write_contracts(run_dir / "contracts.parquet", args.contract_type, args.strike, args.maturity_years)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "simulated",
        "simulator": simulator_name,
        "run_id": run_id,
        "paths": {"n_paths": args.n_paths},
        "time_grid": {
            "maturity_years": args.maturity_years,
            "n_steps": args.n_steps,
            "dt": args.maturity_years / args.n_steps,
            "includes_maturity": True,
        },
        "splits": {"train": args.split_train, "val": args.split_val, "test": args.split_test},
        "simulator_params": simulator_params,
        "contract": {
            "contract_id": 0,
            "type": args.contract_type,
            "strike": args.strike,
            "maturity_years": args.maturity_years,
        },
        "latent_state": {
            "schema_version": "v1.0",
            "columns": ["path_id", "t_idx", "v"],
            "note": "v is actual variance for heston; 0.0 placeholder for bs and nga",
        },
        "notes": "Schema v1.0 observations: path_id, t_idx, t_years, S. Splits are folders train/val/test.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote dataset to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
