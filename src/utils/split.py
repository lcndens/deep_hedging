"""Deterministic path-index splitting for dataset generation.

This module partitions simulated path identifiers into train/validation/test
subsets using a seeded permutation so splits are reproducible across runs.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    """Fractions for train/validation/test partitions.

    Parameters
    ----------
    train : float, default=0.70
        Fraction assigned to the training split.
    val : float, default=0.15
        Fraction assigned to the validation split.
    test : float, default=0.15
        Fraction assigned to the test split.
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


def split_path_ids(n_paths: int, seed: int, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """Split path identifiers into deterministic train/val/test sets.

    Parameters
    ----------
    n_paths : int
        Total number of paths ``N``.
    seed : int
        Random seed used for the permutation.
    cfg : SplitConfig
        Split fractions that must sum to 1.0.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``{"train", "val", "test"}`` to arrays of integer path IDs.

    Raises
    ------
    ValueError
        If ``n_paths`` is not positive or split fractions do not sum to 1.0.
    RuntimeError
        If computed split sizes do not sum back to ``n_paths``.
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
