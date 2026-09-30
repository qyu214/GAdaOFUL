"""Tests for deterministic simulation behavior."""

import numpy as np

from src.algorithms import run_greedy
from src.config import ExperimentConfig


def test_greedy_is_reproducible_with_fixed_seed():
    """The same random seed should reproduce the same trajectory."""
    config = ExperimentConfig(
        dim=2,
        sigma=1.0,
        corruption=1,
        T=4,
        repeat=1,
        actions=3,
        norm=1.0,
    )

    np.random.seed(123)
    first = run_greedy(config)

    np.random.seed(123)
    second = run_greedy(config)

    np.testing.assert_allclose(first, second)

    assert first.shape == (config.T,)
    assert np.all(np.isfinite(first))
    assert np.all(np.diff(first) >= -1e-12)
