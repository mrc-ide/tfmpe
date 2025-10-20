"""Shared fixtures for preprocessing tests."""

import jax.numpy as jnp
import pytest


@pytest.fixture
def simple_pytree():
    """Simple 2-level hierarchical structure."""
    return {
        'mu': jnp.array([[1.0]]),  # (1, 1)
        'theta': jnp.array([[2.0], [3.0], [4.0]]),  # (3, 1)
        'obs': jnp.array([[0.1], [0.2], [0.3]])  # (3, 1)
    }


@pytest.fixture
def simple_independence():
    """Independence spec for simple hierarchical structure."""
    return {
        'local': ['obs', 'theta'],
        'cross': [('mu', 'obs'), ('obs', 'mu')],
        'cross_local': [('theta', 'obs', None)]
    }
