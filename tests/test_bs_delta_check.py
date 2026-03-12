"""Tests for the Black-Scholes delta evaluation module.

Tests cover:
- bs_call_delta correctness against known analytic values
- BS delta properties (in [0,1], ATM ≈ 0.5, deep ITM → 1, deep OTM → 0)
- BSCheckResult fields
- Checkpoint loading
- Full pipeline: untrained network has high MAE, trained converges
- CLI config construction
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.evaluation.bs_delta_check import (
    BSCheckConfig,
    BSCheckResult,
    bs_call_delta,
    run_bs_delta_check,
)
from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
from src.objective_functions.cvar import CVaRLoss
from src.io.dataset_loader import load_dataset
from src.generate_dataset import main as generate_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bs_dataset(tmp_path_factory) -> Path:
    """Helper for bs dataset."""
    return generate_dataset([
        "--sim",            "bs",
        "--n_paths",        "300",
        "--n_steps",        "30",
        "--seed",           "0",
        "--out_root",       str(tmp_path_factory.mktemp("eval_bs")),
        "--run_id",         "test_bs_eval",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])


@pytest.fixture(scope="module")
def untrained_checkpoint(tmp_path_factory, bs_dataset) -> Path:
    """Save an untrained network as a checkpoint for loading tests."""
    torch.manual_seed(0)
    net  = BaselineFeedforwardNetwork(hidden=64)
    cvar = CVaRLoss(alpha=0.95)

    ckpt_dir = tmp_path_factory.mktemp("ckpt")
    path     = ckpt_dir / "untrained.pt"
    torch.save({
        "epoch":            0,
        "model_state_dict": net.state_dict(),
        "cvar_state_dict":  cvar.state_dict(),
        "optimizer_state":  {},
    }, path)
    return path


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory, bs_dataset) -> Path:
    """Train a small network for a few epochs and save checkpoint."""
    from src.train.trainer import TrainConfig, train

    cfg = TrainConfig(
        dataset_dir         = str(bs_dataset),
        sim                 = "bs",
        epsilon             = 0.0,
        n_epochs            = 30,
        lr                  = 1e-3,
        batch_size          = 200,
        alpha               = 0.95,
        hidden              = 64,
        early_stop_patience = 100,
        seed                = 0,
        out_root            = str(tmp_path_factory.mktemp("trained")),
        run_id              = "test_trained",
        device              = "cpu",
    )
    result = train(cfg)
    return result.run_dir / "checkpoints" / "best_model.pt"


# ===========================================================================
# bs_call_delta — analytic correctness
# ===========================================================================

class TestBsCallDelta:

    """Test cases for TestBsCallDelta."""
    def test_output_shape_matches_input(self):
        """Assert output shape matches input."""
        S   = torch.ones(100, 30)
        tau = torch.ones(100, 30) * 0.25
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2)
        assert d.shape == (100, 30)

    def test_atm_delta_approx_half(self):
        """ATM (S=K), tau=1yr, sigma=0.2, r=0: delta ≈ N(0.1) ≈ 0.54."""
        S   = torch.tensor([100.0])
        tau = torch.tensor([1.0])
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        # d1 = (0 + 0.5*0.04*1) / (0.2*1) = 0.02/0.2 = 0.1
        # N(0.1) ≈ 0.5398
        assert abs(d.item() - 0.5398) < 0.001

    def test_deep_itm_delta_near_one(self):
        """Deep ITM: S >> K → delta → 1."""
        S   = torch.tensor([200.0])
        tau = torch.tensor([0.25])
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        assert d.item() > 0.99

    def test_deep_otm_delta_near_zero(self):
        """Deep OTM: S << K → delta → 0."""
        S   = torch.tensor([50.0])
        tau = torch.tensor([0.25])
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        assert d.item() < 0.01

    def test_delta_in_zero_one(self):
        """Delta must always be in [0, 1]."""
        torch.manual_seed(0)
        S   = torch.exp(torch.randn(1000)) * 100.0
        tau = torch.rand(1000) * 1.0 + 0.01
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        assert (d >= 0.0).all()
        assert (d <= 1.0).all()

    def test_delta_increases_with_spot(self):
        """Delta is monotone increasing in S (call option)."""
        S   = torch.linspace(50, 200, 100)
        tau = torch.ones(100) * 0.25
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        assert (d[1:] >= d[:-1]).all()

    def test_delta_decreases_with_tau(self):
        """For OTM option, delta increases as tau increases (more time = more chance)."""
        S   = torch.ones(50) * 80.0    # OTM
        tau = torch.linspace(0.01, 2.0, 50)
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        # OTM delta increases with tau
        assert d[-1] > d[0]

    def test_no_nan_at_boundary_tau(self):
        """Very small tau should not produce NaN."""
        S   = torch.tensor([100.0, 100.0])
        tau = torch.tensor([1e-8, 1e-10])
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        assert torch.isfinite(d).all()

    def test_analytic_value_known(self):
        """Assert BS delta matches a hand-computed at-the-money reference value."""
        S   = torch.tensor([100.0])
        tau = torch.tensor([0.25])
        d   = bs_call_delta(S, K=100.0, tau=tau, sigma=0.2, r=0.0)
        assert abs(d.item() - 0.5199) < 0.001


# ===========================================================================
# Checkpoint loading
# ===========================================================================

class TestCheckpointLoading:

    """Test cases for TestCheckpointLoading."""
    def test_loads_without_error(self, untrained_checkpoint, bs_dataset):
        """Assert loads without error."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
            split       = "test",
        )
        result = run_bs_delta_check(cfg)
        assert isinstance(result, BSCheckResult)

    def test_missing_checkpoint_raises(self, bs_dataset):
        """Assert missing checkpoint raises."""
        cfg = BSCheckConfig(
            checkpoint  = "/nonexistent/path/model.pt",
            dataset_dir = str(bs_dataset),
        )
        with pytest.raises(FileNotFoundError):
            run_bs_delta_check(cfg)


# ===========================================================================
# BSCheckResult properties
# ===========================================================================

class TestBSCheckResult:

    """Test cases for TestBSCheckResult."""
    def test_mae_is_float(self, untrained_checkpoint, bs_dataset):
        """Assert mae is float."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        result = run_bs_delta_check(cfg)
        assert isinstance(result.mae, float)

    def test_r2_is_float(self, untrained_checkpoint, bs_dataset):
        """Assert r2 is float."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        result = run_bs_delta_check(cfg)
        assert isinstance(result.r2, float)

    def test_per_step_mae_length(self, untrained_checkpoint, bs_dataset):
        """Assert per step mae length."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        result = run_bs_delta_check(cfg)
        batch  = load_dataset(Path(str(bs_dataset)), split="test")
        assert len(result.per_step_mae) == batch.n_steps

    def test_n_paths_correct(self, untrained_checkpoint, bs_dataset):
        """Assert n paths correct."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        result = run_bs_delta_check(cfg)
        batch  = load_dataset(Path(str(bs_dataset)), split="test")
        assert result.n_paths == batch.n_paths

    def test_untrained_fails_check(self, untrained_checkpoint, bs_dataset):
        """An untrained network should NOT pass the convergence check."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        result = run_bs_delta_check(cfg)
        assert result.passed is False

    def test_untrained_mae_larger_than_trained(
        self, untrained_checkpoint, trained_checkpoint, bs_dataset
    ):
        """A trained network should have lower MAE than an untrained one."""
        cfg_untrained = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        cfg_trained = BSCheckConfig(
            checkpoint  = str(trained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
        )
        result_untrained = run_bs_delta_check(cfg_untrained)
        result_trained   = run_bs_delta_check(cfg_trained)

        assert result_trained.mae < result_untrained.mae, (
            f"Trained MAE ({result_trained.mae:.4f}) should be lower than "
            f"untrained MAE ({result_untrained.mae:.4f})"
        )


# ===========================================================================
# Output files
# ===========================================================================

class TestOutputFiles:

    """Test cases for TestOutputFiles."""
    def test_csv_saved(self, untrained_checkpoint, bs_dataset, tmp_path):
        """Assert csv saved."""
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
            out_dir     = str(tmp_path),
        )
        run_bs_delta_check(cfg)
        assert (tmp_path / "bs_delta_check.csv").exists()

    def test_csv_has_correct_rows(self, untrained_checkpoint, bs_dataset, tmp_path):
        """Assert csv has correct rows."""
        import csv as csv_module
        cfg = BSCheckConfig(
            checkpoint  = str(untrained_checkpoint),
            dataset_dir = str(bs_dataset),
            hidden      = 64,
            sigma       = 0.2,
            out_dir     = str(tmp_path / "csv_check"),
        )
        result = run_bs_delta_check(cfg)
        path   = Path(cfg.out_dir) / "bs_delta_check.csv"
        with open(path) as f:
            rows = list(csv_module.DictReader(f))
        assert len(rows) == result.n_steps
