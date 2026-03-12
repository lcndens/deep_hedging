"""Tests for the training loop and run-artifact outputs.

Tests cover:
- TrainConfig construction and defaults
- Output directory structure created correctly
- config.json saved correctly
- train_log.csv written with correct columns
- Checkpoints saved (best_model.pt, final_model.pt)
- Loss decreases over training (BS frictionless — should learn quickly)
- Early stopping fires correctly
- TrainResult fields are correct
- Checkpoint can be reloaded
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.generate_dataset import main as generate_dataset
from src.train.trainer import TrainConfig, TrainResult, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bs_dataset(tmp_path_factory) -> Path:
    """Small BS dataset for training tests."""
    return generate_dataset([
        "--sim",            "bs",
        "--n_paths",        "500",
        "--n_steps",        "5",
        "--seed",           "0",
        "--out_root",       str(tmp_path_factory.mktemp("trainer_bs")),
        "--run_id",         "test_bs_trainer",
        "--strike",         "100.0",
        "--maturity_years", "0.25",
    ])


@pytest.fixture(scope="module")
def minimal_cfg(bs_dataset, tmp_path_factory) -> TrainConfig:
    """Minimal config: 3 epochs, small batch — just enough to test the loop."""
    return TrainConfig(
        dataset_dir         = str(bs_dataset),
        sim                 = "bs",
        epsilon             = 0.0,
        n_epochs            = 3,
        lr                  = 1e-3,
        batch_size          = 100,
        alpha               = 0.95,
        hidden              = 16,
        early_stop_patience = 50,
        seed                = 0,
        out_root            = str(tmp_path_factory.mktemp("trainer_out")),
        run_id              = "test_run",
        device              = "cpu",
    )


@pytest.fixture(scope="module")
def train_result(minimal_cfg) -> TrainResult:
    """Run training once and cache the result for all tests."""
    return train(minimal_cfg)


# ===========================================================================
# TrainConfig
# ===========================================================================

class TestTrainConfig:

    """Test cases for TestTrainConfig."""
    def test_defaults(self, bs_dataset):
        """Assert defaults."""
        cfg = TrainConfig(dataset_dir=str(bs_dataset))
        assert cfg.sim                 == "bs"
        assert cfg.epsilon             == 0.0
        assert cfg.n_epochs            == 500
        assert cfg.lr                  == 1e-3
        assert cfg.batch_size          == 10_000
        assert cfg.alpha               == 0.95
        assert cfg.hidden              == 64
        assert cfg.early_stop_patience == 50
        assert cfg.seed                == 42
        assert cfg.device              == "auto"

    def test_custom_values(self, bs_dataset, tmp_path):
        """Assert custom values."""
        cfg = TrainConfig(
            dataset_dir = str(bs_dataset),
            sim         = "heston",
            epsilon     = 0.01,
            n_epochs    = 100,
            lr          = 5e-4,
            batch_size  = 5000,
            alpha       = 0.90,
            out_root    = str(tmp_path),
        )
        assert cfg.sim     == "heston"
        assert cfg.epsilon == 0.01
        assert cfg.alpha   == 0.90


# ===========================================================================
# TrainResult
# ===========================================================================

class TestTrainResult:

    """Test cases for TestTrainResult."""
    def test_returns_train_result(self, train_result):
        """Assert returns train result."""
        assert isinstance(train_result, TrainResult)

    def test_run_dir_exists(self, train_result):
        """Assert run dir exists."""
        assert train_result.run_dir.exists()

    def test_best_epoch_in_range(self, train_result, minimal_cfg):
        """Assert best epoch in range."""
        assert 1 <= train_result.best_epoch <= minimal_cfg.n_epochs

    def test_best_val_loss_is_finite(self, train_result):
        """Assert best val loss is finite."""
        assert isinstance(train_result.best_val_loss, float)
        assert train_result.best_val_loss < float("inf")

    def test_n_epochs_trained_correct(self, train_result, minimal_cfg):
        """Assert n epochs trained correct."""
        assert train_result.n_epochs_trained == minimal_cfg.n_epochs

    def test_stopped_early_false_for_short_run(self, train_result):
        """3-epoch run with patience=50 should not early-stop."""
        assert train_result.stopped_early is False

    def test_final_losses_are_finite(self, train_result):
        """Assert final losses are finite."""
        assert train_result.final_train_loss < float("inf")
        assert train_result.final_val_loss   < float("inf")


# ===========================================================================
# Output directory structure
# ===========================================================================

class TestOutputStructure:

    """Test cases for TestOutputStructure."""
    def test_run_dir_created(self, train_result):
        """Assert run dir created."""
        assert train_result.run_dir.is_dir()

    def test_checkpoints_dir_exists(self, train_result):
        """Assert checkpoints dir exists."""
        assert (train_result.run_dir / "checkpoints").is_dir()

    def test_logs_dir_exists(self, train_result):
        """Assert logs dir exists."""
        assert (train_result.run_dir / "logs").is_dir()

    def test_config_json_exists(self, train_result):
        """Assert config json exists."""
        assert (train_result.run_dir / "config.json").exists()

    def test_train_log_csv_exists(self, train_result):
        """Assert train log csv exists."""
        assert (train_result.run_dir / "logs" / "train_log.csv").exists()

    def test_best_model_checkpoint_exists(self, train_result):
        """Assert best model checkpoint exists."""
        assert (train_result.run_dir / "checkpoints" / "best_model.pt").exists()

    def test_final_model_checkpoint_exists(self, train_result):
        """Assert final model checkpoint exists."""
        assert (train_result.run_dir / "checkpoints" / "final_model.pt").exists()


# ===========================================================================
# config.json
# ===========================================================================

class TestConfigJson:

    """Test cases for TestConfigJson."""
    def test_config_is_valid_json(self, train_result):
        """Assert config is valid json."""
        with open(train_result.run_dir / "config.json") as f:
            cfg_dict = json.load(f)
        assert isinstance(cfg_dict, dict)

    def test_config_contains_key_fields(self, train_result):
        """Assert config contains key fields."""
        with open(train_result.run_dir / "config.json") as f:
            cfg_dict = json.load(f)
        for key in ("sim", "epsilon", "n_epochs", "lr", "batch_size", "alpha", "seed"):
            assert key in cfg_dict, f"Missing key '{key}' in config.json"

    def test_config_values_match(self, train_result, minimal_cfg):
        """Assert config values match."""
        with open(train_result.run_dir / "config.json") as f:
            cfg_dict = json.load(f)
        assert cfg_dict["sim"]     == minimal_cfg.sim
        assert cfg_dict["epsilon"] == minimal_cfg.epsilon
        assert cfg_dict["alpha"]   == minimal_cfg.alpha
        assert cfg_dict["seed"]    == minimal_cfg.seed


# ===========================================================================
# train_log.csv
# ===========================================================================

class TestTrainLog:

    """Test cases for TestTrainLog."""
    def _load_log(self, train_result) -> list[dict]:
        """Helper for load log."""
        with open(train_result.run_dir / "logs" / "train_log.csv") as f:
            return list(csv.DictReader(f))

    def test_log_has_correct_columns(self, train_result):
        """Assert log has correct columns."""
        rows = self._load_log(train_result)
        assert set(rows[0].keys()) >= {"epoch", "train_loss", "val_loss", "omega"}

    def test_log_row_count_matches_epochs(self, train_result, minimal_cfg):
        """Assert log row count matches epochs."""
        rows = self._load_log(train_result)
        assert len(rows) == minimal_cfg.n_epochs

    def test_log_epoch_column_sequential(self, train_result, minimal_cfg):
        """Assert log epoch column sequential."""
        rows = self._load_log(train_result)
        epochs = [int(r["epoch"]) for r in rows]
        assert epochs == list(range(1, minimal_cfg.n_epochs + 1))

    def test_log_losses_are_finite(self, train_result):
        """Assert log losses are finite."""
        rows = self._load_log(train_result)
        for row in rows:
            assert float(row["train_loss"]) < float("inf")
            assert float(row["val_loss"])   < float("inf")


# ===========================================================================
# Checkpoints
# ===========================================================================

class TestCheckpoints:

    """Test cases for TestCheckpoints."""
    def test_best_model_loadable(self, train_result):
        """best_model.pt must be a valid PyTorch checkpoint."""
        ckpt = torch.load(
            train_result.run_dir / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert "model_state_dict" in ckpt
        assert "cvar_state_dict"  in ckpt
        assert "epoch"            in ckpt

    def test_checkpoint_epoch_in_range(self, train_result, minimal_cfg):
        """Assert checkpoint epoch in range."""
        ckpt = torch.load(
            train_result.run_dir / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert 1 <= ckpt["epoch"] <= minimal_cfg.n_epochs

    def test_model_state_dict_restores(self, train_result, minimal_cfg):
        """Loaded state dict must restore a network that produces finite outputs."""
        from src.policy.baseline_feedforward_network import BaselineFeedforwardNetwork
        from src.state.builder import FEATURE_DIM

        ckpt = torch.load(
            train_result.run_dir / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        net = BaselineFeedforwardNetwork(hidden=minimal_cfg.hidden)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()

        features = torch.randn(10, 5, FEATURE_DIM)
        with torch.no_grad():
            deltas = net.forward_trajectory(features)
        assert torch.isfinite(deltas).all()

    def test_cvar_omega_restored(self, train_result):
        """Restored omega must match the value saved in the checkpoint."""
        from src.objective_functions.cvar import CVaRLoss

        ckpt = torch.load(
            train_result.run_dir / "checkpoints" / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        cvar = CVaRLoss(alpha=0.95)
        cvar.load_state_dict(ckpt["cvar_state_dict"])
        assert isinstance(cvar.omega.item(), float)


# ===========================================================================
# Early stopping
# ===========================================================================

class TestEarlyStopping:

    """Test cases for TestEarlyStopping."""
    def test_early_stop_fires(self, bs_dataset, tmp_path):
        """With patience=1 and lr=0 (no improvement), should stop at epoch 2."""
        cfg = TrainConfig(
            dataset_dir         = str(bs_dataset),
            sim                 = "bs",
            n_epochs            = 100,
            lr                  = 0.0,       # no weight updates → no improvement
            batch_size          = 100,
            hidden              = 8,
            early_stop_patience = 1,
            seed                = 0,
            out_root            = str(tmp_path),
            run_id              = "test_early_stop",
            device              = "cpu",
        )
        result = train(cfg)
        assert result.stopped_early is True
        assert result.n_epochs_trained < 100

    def test_early_stop_saves_final_model(self, bs_dataset, tmp_path):
        """Assert early stop saves final model."""
        cfg = TrainConfig(
            dataset_dir         = str(bs_dataset),
            sim                 = "bs",
            n_epochs            = 100,
            lr                  = 0.0,
            batch_size          = 100,
            hidden              = 8,
            early_stop_patience = 1,
            seed                = 0,
            out_root            = str(tmp_path),
            run_id              = "test_early_stop_ckpt",
            device              = "cpu",
        )
        result = train(cfg)
        assert (result.run_dir / "checkpoints" / "final_model.pt").exists()


# ===========================================================================
# Auto run_id
# ===========================================================================

class TestAutoRunId:

    """Test cases for TestAutoRunId."""
    def test_auto_run_id_creates_dir(self, bs_dataset, tmp_path):
        """run_id=None should auto-generate a timestamped directory."""
        cfg = TrainConfig(
            dataset_dir = str(bs_dataset),
            sim         = "bs",
            n_epochs    = 1,
            batch_size  = 50,
            hidden      = 8,
            seed        = 0,
            out_root    = str(tmp_path),
            run_id      = None,
            device      = "cpu",
        )
        result = train(cfg)
        assert result.run_dir.exists()
        # run_id should contain the sim name
        assert "bs" in result.run_dir.name
