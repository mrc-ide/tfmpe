"""Tests for mask generation utilities.

Tests verify attention mask and padding mask generation from independence
specifications for hierarchical parameter structures.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tfmpe.preprocessing.masks import (
    build_self_attention_mask,
    build_cross_attention_mask,
    build_padding_mask
)


@pytest.fixture
def hierarchical_gaussian_independence():
    """Independence spec from hierarchical_gaussian.py."""
    return {
        'local': ['obs', 'theta'],
        'cross': [('mu', 'obs'), ('obs', 'mu')],
        'cross_local': [('theta', 'obs', (0, 0))]
    }


@pytest.fixture
def simple_slices():
    """Simple slice metadata for testing."""
    return {
        'mu': {
            'offset': 0,
            'event_shape': (1,),
            'batch_shape': (1,)
        },
        'theta': {
            'offset': 1,
            'event_shape': (2,),
            'batch_shape': (1,)
        },
        'obs': {
            'offset': 3,
            'event_shape': (2,),
            'batch_shape': (1,)
        }
    }


def test_self_attention_mask_local_independence(simple_slices):
    """Test self-attention mask with local independence."""
    independence = {
        'local': ['theta', 'obs']
    }

    mask = build_self_attention_mask(simple_slices, independence)

    # Total size: 1 (mu) + 2 (theta) + 2 (obs) = 5
    # mu: [0], theta: [1:3], obs: [3:5]
    expected = np.array([
        # mu theta    obs
        [1., 1., 1., 1., 1.],  # mu
        [1., 0., 0., 1., 1.],  # theta[0]
        [1., 0., 0., 1., 1.],  # theta[1]
        [1., 1., 1., 0., 0.],  # obs[0]
        [1., 1., 1., 0., 0.],  # obs[1]
    ], dtype=np.float32)

    assert mask.shape == (5, 5)
    assert jnp.allclose(mask, expected)

def test_self_attention_mask_cross_independence(simple_slices):
    """Test self-attention mask with cross independence."""
    independence = {
        'cross': [('mu', 'obs'), ('obs', 'mu')]
    }

    mask = build_self_attention_mask(simple_slices, independence)

    # Total size: 1 (mu) + 2 (theta) + 2 (obs) = 5
    # mu: [0], theta: [1:3], obs: [3:5]
    expected = np.array([
        # mu theta    obs
        [1., 1., 1., 0., 0.],  # mu
        [1., 1., 1., 1., 1.],  # theta[0]
        [1., 1., 1., 1., 1.],  # theta[1]
        [0., 1., 1., 1., 1.],  # obs[0]
        [0., 1., 1., 1., 1.],  # obs[1]
    ], dtype=np.float32)

    assert mask.shape == (5, 5)
    assert jnp.allclose(mask, expected)

def test_self_attention_mask_cross_local_with_functional_inputs(
    simple_slices
):
    """Test cross-local mask with functional input mapping."""
    # theta[i] connects to obs[i] via dimension (0, 0)
    independence = {
        'cross_local': [('theta', 'obs', (0, 0))]
    }

    mask = build_self_attention_mask(simple_slices, independence)

    # Total size: 1 (mu) + 2 (theta) + 2 (obs) = 5
    # mu: [0], theta: [1:3], obs: [3:5]
    expected = np.array([
        # mu theta    obs
        [1., 1., 1., 1., 1.],  # mu
        [1., 1., 1., 1., 0.],  # theta[0]
        [1., 1., 1., 0., 1.],  # theta[1]
        [1., 1., 0., 1., 1.],  # obs[0]
        [1., 0., 1., 1., 1.],  # obs[1]
    ], dtype=np.float32)

    assert mask.shape == (5, 5)
    assert jnp.allclose(mask, expected)

def test_self_attention_mask_cross_local_diagonal(simple_slices):
    """Test cross-local mask with diagonal (None idx_map)."""
    # When idx_map is None, it's diagonal only
    independence = {
        'cross_local': [('theta', 'obs', None)]
    }

    mask = build_self_attention_mask(simple_slices, independence)

    # Total size: 1 (mu) + 2 (theta) + 2 (obs) = 5
    # mu: [0], theta: [1:3], obs: [3:5]
    expected = np.array([
        # mu theta    obs
        [1., 1., 1., 1., 1.],  # mu
        [1., 1., 1., 1., 0.],  # theta[0]
        [1., 1., 1., 0., 1.],  # theta[1]
        [1., 1., 0., 1., 1.],  # obs[0]
        [1., 0., 1., 1., 1.],  # obs[1]
    ], dtype=np.float32)

    assert mask.shape == (5, 5)
    assert jnp.allclose(mask, expected)

def test_self_attention_mask_combined_rules(
    simple_slices,
    hierarchical_gaussian_independence
):
    """Test combining multiple independence rules."""
    mask = build_self_attention_mask(
        simple_slices,
        hierarchical_gaussian_independence
    )

    # Total size: 1 (mu) + 2 (theta) + 2 (obs) = 5
    # mu: [0], theta: [1:3], obs: [3:5]
    # Rules: local=['obs', 'theta'], cross=[('mu', 'obs'), ('obs', 'mu')],
    #        cross_local=[('theta', 'obs', (0, 0))]
    expected = np.array([
        # mu theta    obs
        [1., 1., 1., 0., 0.],  # mu (cross blocks obs)
        [1., 0., 0., 1., 0.],  # theta[0] (local, cross_local with obs[0])
        [1., 0., 0., 0., 1.],  # theta[1] (local, cross_local with obs[1])
        [0., 1., 0., 0., 0.],  # obs[0] (cross blocks mu, local, cross_local)
        [0., 0., 1., 0., 0.],  # obs[1] (cross blocks mu, local, cross_local)
    ], dtype=np.float32)

    assert mask.shape == (5, 5)
    assert jnp.allclose(mask, expected)

def test_cross_local_multidim_event_shapes():
    """Test cross_local with multi-dimensional event shapes."""
    # theta has shape (3, 2) = 6 tokens flattened
    # obs has shape (3, 4) = 12 tokens flattened
    # Connect along dimension 0 (size 3 in both)
    slices = {
        'theta': {
            'offset': 0,
            'event_shape': (3, 2),
            'batch_shape': (1,)
        },
        'obs': {
            'offset': 6,
            'event_shape': (3, 4),
            'batch_shape': (1,)
        }
    }

    # Connect theta[i, :] to obs[i, :] via dimension (0, 0)
    independence = {
        'cross_local': [('theta', 'obs', (0, 0))]
    }

    mask = build_self_attention_mask(slices, independence)

    # Total: 6 + 12 = 18 tokens
    assert mask.shape == (18, 18)

    # theta-obs block: [0:6, 6:18]
    theta_obs = mask[0:6, 6:18]

    # theta is flattened row-major: [0,0], [0,1], [1,0], [1,1], [2,0], [2,1]
    # obs is flattened row-major: [0,0], [0,1], [0,2], [0,3], [1,0], ...
    # theta[0,:] (indices 0,1) should connect to obs[0,:] (indices 0,1,2,3)
    # theta[1,:] (indices 2,3) should connect to obs[1,:] (indices 4,5,6,7)
    # theta[2,:] (indices 4,5) should connect to obs[2,:] (indices 8,9,10,11)

    expected = np.zeros((6, 12), dtype=np.float32)
    # theta[0,:] -> obs[0,:]
    expected[0:2, 0:4] = 1.0
    # theta[1,:] -> obs[1,:]
    expected[2:4, 4:8] = 1.0
    # theta[2,:] -> obs[2,:]
    expected[4:6, 8:12] = 1.0

    assert jnp.allclose(theta_obs, expected)


def test_cross_local_different_dimensions():
    """Test cross_local connecting different dimensions."""
    # theta: (5, 3) - 5 sites, 3 features
    # obs: (2, 5) - 2 timepoints, 5 sites
    # Connect theta site i to obs site i via (0, 1)
    slices = {
        'theta': {
            'offset': 0,
            'event_shape': (5, 3),
            'batch_shape': (1,)
        },
        'obs': {
            'offset': 15,
            'event_shape': (2, 5),
            'batch_shape': (1,)
        }
    }

    independence = {
        'cross_local': [('theta', 'obs', (0, 1))]
    }

    mask = build_self_attention_mask(slices, independence)

    assert mask.shape == (25, 25)

    # theta-obs block
    theta_obs = mask[0:15, 15:25]

    # Flattening (row-major):
    # theta (5, 3): [site0_feat0, site0_feat1, site0_feat2,
    #                site1_feat0, site1_feat1, site1_feat2, ...]
    #   site 0 -> indices 0,1,2
    #   site 1 -> indices 3,4,5
    #   site 2 -> indices 6,7,8
    #   site 3 -> indices 9,10,11
    #   site 4 -> indices 12,13,14
    #
    # obs (2, 5): [time0_site0, time0_site1, ..., time0_site4,
    #              time1_site0, time1_site1, ..., time1_site4]
    #   site 0 -> indices 0 (time 0), 5 (time 1)
    #   site 1 -> indices 1 (time 0), 6 (time 1)
    #   site 2 -> indices 2 (time 0), 7 (time 1)
    #   site 3 -> indices 3 (time 0), 8 (time 1)
    #   site 4 -> indices 4 (time 0), 9 (time 1)

    expected = np.zeros((15, 10), dtype=np.float32)

    # Site 0: theta[0:3] -> obs[0, 5]
    expected[0:3, 0] = 1.0
    expected[0:3, 5] = 1.0

    # Site 1: theta[3:6] -> obs[1, 6]
    expected[3:6, 1] = 1.0
    expected[3:6, 6] = 1.0

    # Site 2: theta[6:9] -> obs[2, 7]
    expected[6:9, 2] = 1.0
    expected[6:9, 7] = 1.0

    # Site 3: theta[9:12] -> obs[3, 8]
    expected[9:12, 3] = 1.0
    expected[9:12, 8] = 1.0

    # Site 4: theta[12:15] -> obs[4, 9]
    expected[12:15, 4] = 1.0
    expected[12:15, 9] = 1.0
    assert jnp.allclose(theta_obs, expected)


def test_cross_attention_mask_basic():
    """Test cross-attention mask between query and key sets."""
    query_slices = {
        'theta': {
            'offset': 0,
            'event_shape': (2,),
            'batch_shape': (1,)
        }
    }

    key_slices = {
        'mu': {
            'offset': 0,
            'event_shape': (1,),
            'batch_shape': (1,)
        },
        'obs': {
            'offset': 1,
            'event_shape': (2,),
            'batch_shape': (1,)
        }
    }

    independence = {
        'cross': [('theta', 'mu')]
    }

    mask = build_cross_attention_mask(
        query_slices,
        key_slices,
        independence
    )

    # Query: 2 (theta), Key: 1 (mu) + 2 (obs) = 3
    # theta queries: [0:2], keys: mu [0], obs [1:3]
    expected = np.array([
        # mu obs
        [0., 1., 1.],  # theta[0]
        [0., 1., 1.],  # theta[1]
    ], dtype=np.float32)

    assert mask.shape == (2, 3)
    assert jnp.allclose(mask, expected)

def test_cross_attention_mask_with_cross_local():
    """Test cross-attention mask with cross_local independence."""
    query_slices = {
        'theta': {
            'offset': 0,
            'event_shape': (2,),
            'batch_shape': (1,)
        }
    }

    key_slices = {
        'obs': {
            'offset': 0,
            'event_shape': (2,),
            'batch_shape': (1,)
        }
    }

    independence = {
        'cross_local': [('theta', 'obs', (0, 0))]
    }

    mask = build_cross_attention_mask(
        query_slices,
        key_slices,
        independence
    )

    # Query: 2 (theta), Key: 2 (obs)
    # theta[i] can only attend to obs[i]
    expected = np.array([
        # obs
        [1., 0.],  # theta[0]
        [0., 1.],  # theta[1]
    ], dtype=np.float32)

    assert mask.shape == (2, 2)

    assert jnp.allclose(mask, expected)


def test_padding_mask_uniform_shapes():
    """Test padding mask with uniform event shapes."""
    slices = {
        'a': {
            'offset': 0,
            'event_shape': (3,),
            'batch_shape': (1,)
        },
        'b': {
            'offset': 3,
            'event_shape': (2,),
            'batch_shape': (1,)
        }
    }

    # All actual shapes match padded shapes
    event_shapes = {
        'a': jnp.array([3]),
        'b': jnp.array([2])
    }

    mask = build_padding_mask(slices, event_shapes)

    # Expected shape: (total_tokens,) = (5,)
    assert mask.shape == (5,)

    # All should be valid (1.0)
    assert jnp.allclose(mask, 1.0)


def test_padding_mask_variable_shapes():
    """Test padding mask with variable event shapes."""
    slices = {
        'a': {
            'offset': 0,
            'event_shape': (5,),  # padded to 5
            'batch_shape': (1,)
        },
        'b': {
            'offset': 5,
            'event_shape': (3,),  # padded to 3
            'batch_shape': (1,)
        }
    }

    # Actual shapes are smaller
    event_shapes = {
        'a': jnp.array([3]),  # actual: 3, padded: 5
        'b': jnp.array([2])   # actual: 2, padded: 3
    }

    mask = build_padding_mask(slices, event_shapes)

    assert mask.shape == (8,)

    # 'a': first 3 valid, last 2 padding
    assert jnp.allclose(mask[0:3], 1.0)
    assert jnp.allclose(mask[3:5], 0.0)

    # 'b': first 2 valid, last 1 padding
    assert jnp.allclose(mask[5:7], 1.0)
    assert jnp.allclose(mask[7:8], 0.0)


def test_padding_mask_with_sample_dims():
    """Test padding mask with sample dimensions."""
    slices = {
        'x': {
            'offset': 0,
            'event_shape': (3,),
            'batch_shape': (1,)
        }
    }

    # Different actual shapes per sample
    event_shapes = {
        'x': jnp.array([
            [2],  # sample 0: actual size 2
            [3],  # sample 1: actual size 3
            [1]   # sample 2: actual size 1
        ])
    }

    mask = build_padding_mask(slices, event_shapes)

    # Expected shape: (n_samples, total_tokens) = (3, 3)
    assert mask.shape == (3, 3)

    # Sample 0: first 2 valid
    assert jnp.allclose(mask[0, 0:2], 1.0)
    assert jnp.allclose(mask[0, 2:3], 0.0)

    # Sample 1: all 3 valid
    assert jnp.allclose(mask[1, :], 1.0)

    # Sample 2: only first valid
    assert jnp.allclose(mask[2, 0:1], 1.0)
    assert jnp.allclose(mask[2, 1:3], 0.0)


