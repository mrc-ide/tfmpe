"""Shared fixtures for preprocessing tests."""

import jax.numpy as jnp
import pytest

from tfmpe.preprocessing.utils import Labeller

@pytest.fixture
def simple_pytree():
    """Simple 2-level hierarchical structure."""
    return {
        'mu': jnp.array([[1.0]]),  # (1, 1)
        'theta': jnp.array([[2.0], [3.0], [4.0]]),  # (3, 1)
        'obs': jnp.array([[0.1], [0.2], [0.3]])  # (3, 1)
    }

@pytest.fixture
def simple_labeller():
    """Labeller for simple hierarchical structure."""
    return Labeller(label_map={'mu': 0, 'theta': 1, 'obs': 2})
