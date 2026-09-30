"""Experiment configuration for the GAdaOFUL simulations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the simulation experiments."""

    dim: int = 10
    sigma: float = 1.0
    corruption: int = 50
    T: int = 1000
    repeat: int = 5
    actions: int = 20
    norm: float = 1.0

    @property
    def bmu(self) -> np.ndarray:
        """Return the true parameter vector used in the experiment."""
        return np.ones(self.dim) / np.sqrt(self.dim)


DEFAULT_CONFIG = ExperimentConfig()
