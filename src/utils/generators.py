"""
Centralized random number generators for the project.

All stochastic operations should import from this file to ensure
reproducibility and consistent seeding across the codebase.
"""

import random

import numpy as np

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set the random seed for both Python's random and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def generate_uniform(low: float = 0.0, high: float = 1.0, size=None):
    """Generate random floats in the interval [low, high)."""
    return np.random.uniform(low, high, size)


def generate_random():
    """Return a single random float in [0.0, 1.0)."""
    return np.random.random()


def generate_bernoulli(prob: float, size=None):
    """Generate Bernoulli trials with success probability *prob*."""
    if size is None:
        return 1 if np.random.random() < prob else 0
    return (np.random.random(size) < prob).astype(int)


def generate_choice(a, size=None, replace=True, p=None):
    """Randomly sample from *a* (NumPy-style choice)."""
    return np.random.choice(a, size=size, replace=replace, p=p)
