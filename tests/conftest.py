"""Shared fixtures and constants for the deep-hedging test suite.

All session-scoped dataset fixtures generate small datasets into pytest's
temporary directory once per session and are reused across test modules.

Dataset parameters:
    n_paths=200, n_steps=5, seed=0, maturity_years=0.25, strike=100.0
"""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_dataset import main as generate_dataset
from src.io.dataset_loader import load_dataset

# ---------------------------------------------------------------------------
# Shared constants — imported directly by test modules.
# ---------------------------------------------------------------------------

N_PATHS = 200
N_STEPS = 5
T1      = N_STEPS + 1
SEED    = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen(tmp_path: Path, sim: str) -> Path:
    return generate_dataset([
        "--sim",            sim,
        "--n_paths",        str(N_PATHS),
        "--n_steps",        str(N_STEPS),
        "--seed",           str(SEED),
        "--out_root",       str(tmp_path),
        "--run_id",         f"test_{sim}",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])


# ---------------------------------------------------------------------------
# Session-scoped run-directory fixtures (return Path to dataset run dir).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bs_run(tmp_path_factory):
    return _gen(tmp_path_factory.mktemp("datasets"), "bs")


@pytest.fixture(scope="session")
def heston_run(tmp_path_factory):
    return _gen(tmp_path_factory.mktemp("datasets"), "heston")


@pytest.fixture(scope="session")
def nga_run(tmp_path_factory):
    return _gen(tmp_path_factory.mktemp("datasets"), "nga")


# ---------------------------------------------------------------------------
# Session-scoped batch fixtures (return DatasetBatch for the train split).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bs_batch(bs_run):
    return load_dataset(bs_run, split="train")


@pytest.fixture(scope="session")
def heston_batch(heston_run):
    return load_dataset(heston_run, split="train")


@pytest.fixture(scope="session")
def nga_batch(nga_run):
    return load_dataset(nga_run, split="train")
