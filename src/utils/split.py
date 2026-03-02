# deep_hedging/src/utils/split.py
"""Deterministic train/val/test split utilities."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test fractions that must sum to 1.0."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


def split_path_ids(n_paths: int, seed: int, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """
    Deterministic split by permuting path_ids with a seeded RNG.
    Returns dict with arrays of path_ids: {"train": ..., "val": ..., "test": ...}
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    s = cfg.train + cfg.val + cfg.test
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0; got {s}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_paths)

    n_train = int(round(cfg.train * n_paths))
    n_val = int(round(cfg.val * n_paths))
    n_test = n_paths - n_train - n_val

    train_ids = perm[:n_train]
    val_ids = perm[n_train:n_train + n_val]
    test_ids = perm[n_train + n_val:]

    if len(train_ids) + len(val_ids) + len(test_ids) != n_paths:
        raise RuntimeError("split sizes do not sum to n_paths")

    return {"train": train_ids, "val": val_ids, "test": test_ids}
