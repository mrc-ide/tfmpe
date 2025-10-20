"""Tests for functional input processing utilities."""

import jax.numpy as jnp
import pytest
from tfmpe.preprocessing.functional_inputs import (
    flatten_functional_inputs,
    FUNCTIONAL_INPUT_PAD_VALUE
)

@pytest.fixture
def simple_tokens_slices():
    """Simple slices dict matching token structure."""
    return {
        'mu': {
            'offset': 0,
            'event_shape': (1,),
            'batch_shape': (1,)
        },
        'theta': {
            'offset': 1,
            'event_shape': (5,),
            'batch_shape': (1,)
        },
        'obs': {
            'offset': 6,
            'event_shape': (5, 3),
            'batch_shape': (1,)
        }
    }


@pytest.fixture
def matching_functional_inputs():
    """Functional inputs with matching shapes to token structure."""
    return {
        'mu': jnp.zeros((1, 1)),  # (event=1, batch=1)
        'theta': jnp.zeros((5, 1)),  # (event=5, batch=1)
        'obs': jnp.ones((5, 3, 1))  # (event1=5, event2=3, batch=1)
    }


@pytest.fixture
def padded_functional_inputs():
    """Functional inputs requiring padding to max batch."""
    return {
        'mu': jnp.zeros((1, 1)),  # (event=1, batch=1) - needs padding to 2
        'theta': jnp.zeros((5, 1)),  # (event=5, batch=1) - needs padding to 2
        'obs': jnp.ones((5, 3, 2))  # (event1=5, event2=3, batch=2) - max
    }


def test_flatten_functional_inputs_matching_shapes(
    simple_tokens_slices,
    matching_functional_inputs
):
    """Test flattening functional inputs with matching shapes."""
    result = flatten_functional_inputs(
        matching_functional_inputs,
        simple_tokens_slices,
        sample_ndims=0
    )

    # Check shape: (total_tokens=21, max_batch=1)
    assert result is not None
    assert result.shape == (21, 1)

    # Check mu values (offset 0, size 1)
    assert jnp.allclose(result[0, 0], 0.0)

    # Check theta values (offset 1, size 5)
    assert jnp.allclose(result[1:6, 0], 0.0)

    # Check obs values (offset 6, size 15)
    assert jnp.allclose(result[6:21, 0], 1.0)


def test_flatten_functional_inputs_with_padding(
    simple_tokens_slices,
    padded_functional_inputs
):
    """Test flattening functional inputs requiring padding."""
    result = flatten_functional_inputs(
        padded_functional_inputs,
        simple_tokens_slices,
        sample_ndims=0
    )

    # Check shape: (total_tokens=21, max_batch=2)
    assert result is not None
    assert result.shape == (21, 2)

    # Check mu padding (offset 0, size 1)
    # mu has batch=1, so index 1 should be padded
    assert jnp.allclose(result[0, 0], 0.0)
    assert jnp.allclose(result[0, 1], FUNCTIONAL_INPUT_PAD_VALUE)

    # Check theta padding (offset 1, size 5)
    # theta has batch=1, so index 1 should be padded
    assert jnp.allclose(result[1:6, 0], 0.0)
    assert jnp.allclose(result[1:6, 1], FUNCTIONAL_INPUT_PAD_VALUE)

    # Check obs no padding needed (offset 6, size 15)
    # obs has batch=2, so both indices should be valid
    assert jnp.allclose(result[6:21, 0], 1.0)
    assert jnp.allclose(result[6:21, 1], 1.0)


def test_flatten_functional_inputs_none_returns_none():
    """Test that None functional inputs return None."""
    result = flatten_functional_inputs(
        None,
        {},
        sample_ndims=0
    )

    assert result is None


def test_flatten_functional_inputs_alignment_with_token_slices(
    simple_tokens_slices
):
    """Test that functional inputs align correctly with token offsets."""
    # Create functional inputs with distinct values per key
    functional_inputs = {
        'mu': jnp.full((1, 1), 10.0),  # (event=1, batch=1)
        'theta': jnp.full((5, 1), 20.0),  # (event=5, batch=1)
        'obs': jnp.full((5, 3, 1), 30.0)  # (event1=5, event2=3, batch=1)
    }

    result = flatten_functional_inputs(
        functional_inputs,
        simple_tokens_slices,
        sample_ndims=0
    )

    # Verify each key's functional inputs appear at correct offsets
    assert result is not None
    mu_slice = simple_tokens_slices['mu']
    theta_slice = simple_tokens_slices['theta']
    obs_slice = simple_tokens_slices['obs']

    mu_offset = mu_slice['offset']
    mu_size = 1  # prod(event_shape)
    assert jnp.allclose(
        result[mu_offset:mu_offset + mu_size, 0],
        10.0
    )

    theta_offset = theta_slice['offset']
    theta_size = 5  # prod(event_shape)
    assert jnp.allclose(
        result[theta_offset:theta_offset + theta_size, 0],
        20.0
    )

    obs_offset = obs_slice['offset']
    obs_size = 15  # prod(event_shape) = 5*3
    assert jnp.allclose(
        result[obs_offset:obs_offset + obs_size, 0],
        30.0
    )


def test_flatten_functional_inputs_subset_of_keys(
    simple_tokens_slices
):
    """Test flattening when functional inputs are a subset of keys."""
    # Only provide functional inputs for 'obs', not 'mu' or 'theta'
    functional_inputs = {
        'obs': jnp.full((5, 3, 1), 30.0)
    }

    result = flatten_functional_inputs(
        functional_inputs,
        simple_tokens_slices,
        sample_ndims=0
    )

    # Check shape: (total_tokens=21, batch=1)
    assert result is not None
    assert result.shape == (21, 1)

    # Check mu is padded (offset 0, size 1)
    assert jnp.allclose(result[0, 0], FUNCTIONAL_INPUT_PAD_VALUE)

    # Check theta is padded (offset 1, size 5)
    assert jnp.allclose(result[1:6, 0], FUNCTIONAL_INPUT_PAD_VALUE)

    # Check obs has values (offset 6, size 15)
    assert jnp.allclose(result[6:21, 0], 30.0)


@pytest.mark.parametrize("sample_ndims", [0, 1, 2])
def test_flatten_functional_inputs_with_sample_dims(
    simple_tokens_slices,
    sample_ndims
):
    """Test flattening with different sample dimensions."""
    # Add sample dimensions to functional inputs
    sample_shape = (2,) * sample_ndims if sample_ndims > 0 else ()

    functional_inputs = {
        'mu': jnp.zeros(sample_shape + (1, 1)),
        'theta': jnp.zeros(sample_shape + (5, 1)),
        'obs': jnp.ones(sample_shape + (5, 3, 1))
    }

    result = flatten_functional_inputs(
        functional_inputs,
        simple_tokens_slices,
        sample_ndims=sample_ndims
    )

    # Check shape: (*sample_shape, total_tokens=21, batch=1)
    assert result is not None
    expected_shape = sample_shape + (21, 1)
    assert result.shape == expected_shape
