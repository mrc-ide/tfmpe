"""Shared fixtures for estimator tests."""

import diffrax
import jax.numpy as jnp
import pytest


@pytest.fixture
def doubling_vf():
    """Vector field for doubling flow: f(θ, t) = log(2) · θ."""
    def vf(x, t):
        return jnp.log(2.0) * x
    return vf


@pytest.fixture
def solver():
    """Diffrax ODE solver instance."""
    return diffrax.Dopri5()
