"""Tests for PyTree reconstruction utilities.

Tests verify reconstruction of hierarchical parameter structures from flat
arrays using slice metadata, with support for selective key reconstruction.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array

from tfmpe.preprocessing import decode_pytree, decode_pytree_keys


def test_decode_pytree_full():
    """Test full PyTree reconstruction."""
    # Flat array: [1.0, 2.0, 0.5] representing mu=[1.0, 2.0], sigma=[0.5]
    flat_array = jnp.array([[1.0, 2.0, 0.5]]).T  # (3, 1)

    slices_dict = {
        'mu': {
            'offset': 0,
            'event_shape': (2,),
            'batch_shape': (1,)
        },
        'sigma': {
            'offset': 2,
            'event_shape': (1,),
            'batch_shape': (1,)
        }
    }

    # Reconstruct
    reconstructed = decode_pytree(
        flat_array,
        slices_dict,
        sample_shape=()
    )

    # Verify all keys present
    assert set(reconstructed.keys()) == {'mu', 'sigma'}

    # Verify shapes and values match
    assert reconstructed['mu'].shape == (2, 1)
    assert jnp.allclose(reconstructed['mu'], jnp.array([[1.0], [2.0]]))
    assert reconstructed['sigma'].shape == (1, 1)
    assert jnp.allclose(reconstructed['sigma'], jnp.array([[0.5]]))


def test_decode_pytree_keys_selective():
    """Test selective key reconstruction with decode_pytree_keys()."""
    # Flat array: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Representing: mu=[1.0], sigma=[2.0, 3.0], obs=[4.0, 5.0, 6.0]
    flat_array = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).T  # (6, 1)

    slices_dict = {
        'mu': {
            'offset': 0,
            'event_shape': (1,),
            'batch_shape': (1,)
        },
        'sigma': {
            'offset': 1,
            'event_shape': (2,),
            'batch_shape': (1,)
        },
        'obs': {
            'offset': 3,
            'event_shape': (3,),
            'batch_shape': (1,)
        }
    }

    # Reconstruct only mu and obs
    selected_keys = ['mu', 'obs']
    reconstructed = decode_pytree_keys(
        flat_array,
        slices_dict,
        sample_shape=(),
        keys=selected_keys
    )

    # Verify only selected keys present
    assert set(reconstructed.keys()) == set(selected_keys)

    # Verify values match
    assert jnp.allclose(reconstructed['mu'], jnp.array([[1.0]]))
    assert jnp.allclose(reconstructed['obs'], jnp.array([[4.0], [5.0], [6.0]]))

    # Verify sigma not present
    assert 'sigma' not in reconstructed


def test_decode_pytree_key_order_preserved():
    """Test that key order is consistent with slices_dict."""
    # Flat array with 3 keys in specific order
    flat_array = jnp.array([[1.0, 2.0, 3.0, 4.0]]).T  # (4, 1)

    # Note: dict insertion order is preserved in Python 3.7+
    slices_dict = {
        'z': {
            'offset': 0,
            'event_shape': (1,),
            'batch_shape': (1,)
        },
        'a': {
            'offset': 1,
            'event_shape': (2,),
            'batch_shape': (1,)
        },
        'x': {
            'offset': 3,
            'event_shape': (1,),
            'batch_shape': (1,)
        }
    }

    # Reconstruct
    reconstructed = decode_pytree(
        flat_array,
        slices_dict,
        sample_shape=()
    )

    # Verify key order matches slices_dict order
    assert list(reconstructed.keys()) == list(slices_dict.keys())
    assert list(reconstructed.keys()) == ['z', 'a', 'x']


@pytest.mark.parametrize(
    "sample_ndims,sample_shape,flat_array,expected_shapes",
    [
        # No sample dims
        (
            0,
            (),
            jnp.array([[1.0, 2.0, 3.0]]).T,  # (3, 1)
            {'a': (2, 1), 'b': (1, 1)}
        ),
        # 1 sample dim
        (
            1,
            (2,),
            jnp.array([
                [[1.0], [2.0], [3.0]],  # sample 0
                [[4.0], [5.0], [6.0]]   # sample 1
            ]),  # (2, 3, 1)
            {'a': (2, 2, 1), 'b': (2, 1, 1)}
        ),
        # 2 sample dims
        (
            2,
            (2, 2),
            jnp.ones((2, 2, 3, 1)),  # (2, 2, 3, 1)
            {'a': (2, 2, 2, 1), 'b': (2, 2, 1, 1)}
        ),
    ]
)
def test_decode_pytree_with_different_sample_dims(
    sample_ndims,
    sample_shape,
    flat_array,
    expected_shapes
):
    """Test shape preservation with different batch dims."""
    slices_dict = {
        'a': {
            'offset': 0,
            'event_shape': (2,),
            'batch_shape': (1,)
        },
        'b': {
            'offset': 2,
            'event_shape': (1,),
            'batch_shape': (1,)
        }
    }

    # Reconstruct
    reconstructed = decode_pytree(
        flat_array,
        slices_dict,
        sample_shape=sample_shape
    )

    # Verify shapes preserved
    for key in ['a', 'b']:
        assert reconstructed[key].shape == expected_shapes[key]


def test_decode_pytree_with_padding():
    """Test reconstruction correctly handles padded batch dimensions."""
    # Flat array with padding:
    # a=[1.0, 2.0, 3.0] (batch size 3), b=[4.0, -999, -999] (batch size 1)
    flat_array = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, -999.0, -999.0]
    ])  # (2, 3)

    slices_dict = {
        'a': {
            'offset': 0,
            'event_shape': (1,),
            'batch_shape': (3,)  # Actual batch size 3
        },
        'b': {
            'offset': 1,
            'event_shape': (1,),
            'batch_shape': (1,)  # Actual batch size 1 (padded to 3)
        }
    }

    # Reconstruct
    reconstructed = decode_pytree(
        flat_array,
        slices_dict,
        sample_shape=()
    )

    # Verify shapes match (padding removed)
    assert reconstructed['a'].shape == (1, 3)
    assert reconstructed['b'].shape == (1, 1)

    # Verify values match (no padding in output)
    assert jnp.allclose(reconstructed['a'], jnp.array([[1.0, 2.0, 3.0]]))
    assert jnp.allclose(reconstructed['b'], jnp.array([[4.0]]))


def test_decode_pytree_error_on_invalid_slice():
    """Verify error on invalid slice metadata."""
    # Create flat array
    flat_array = jnp.array([[1.0, 2.0, 3.0]]).T  # (3, 1)

    # Invalid slices: offset + event_size exceeds array size
    invalid_slices = {
        'a': {
            'offset': 0,
            'event_shape': (5,),  # Too large!
            'batch_shape': (1,)
        }
    }

    # Should raise error (IndexError or ValueError)
    with pytest.raises((IndexError, ValueError)):
        decode_pytree(
            flat_array,
            invalid_slices,
            sample_shape=()
        )


def test_decode_pytree_roundtrip_with_multidim_events():
    """Test round-trip with multi-dimensional event shapes."""
    # Flat array: 2x2 matrix (flattened) + 3-vector (flattened)
    # [1, 2, 3, 4, 5, 6, 7]
    flat_array = jnp.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ]).T  # (7, 1)

    slices_dict = {
        'matrix': {
            'offset': 0,
            'event_shape': (2, 2),  # 2x2 matrix
            'batch_shape': (1,)
        },
        'vector': {
            'offset': 4,
            'event_shape': (3, 1),  # 3x1 vector
            'batch_shape': (1,)
        }
    }

    # Reconstruct
    reconstructed = decode_pytree(
        flat_array,
        slices_dict,
        sample_shape=()
    )

    # Verify shapes
    assert reconstructed['matrix'].shape == (2, 2, 1)
    assert reconstructed['vector'].shape == (3, 1, 1)

    # Verify values
    expected_matrix = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    expected_vector = jnp.array([[[5.0]], [[6.0]], [[7.0]]])
    assert jnp.allclose(reconstructed['matrix'], expected_matrix)
    assert jnp.allclose(reconstructed['vector'], expected_vector)
