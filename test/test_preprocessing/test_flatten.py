"""Tests for PyTree flattening utilities.

Tests verify flattening of hierarchical parameter structures into flat
arrays, with padding, slice tracking, and value override capabilities.
"""

import jax.numpy as jnp
import pytest
from tfmpe.preprocessing.flatten import (
    flatten_leaf,
    flatten_pytree,
    update_flat_array,
)
from tfmpe.preprocessing.utils import SliceInfo

@pytest.mark.parametrize(
    "sample_ndims,event_shape,batch_shape,max_batch_size",
    [
        # No sample dims, various event shapes, batch_ndims=1
        (0, (3,), (1,), 1),  # 1D event, no padding
        (0, (3,), (1,), 5),  # 1D event, with padding
        (0, (2, 3), (1,), 1),  # 2D event, no padding
        (0, (2, 3), (2,), 4),  # 2D event, with padding
        (0, (2, 3, 4), (1,), 3),  # 3D event, with padding
        # With 1 sample dim
        (1, (3,), (1,), 1),  # 1D event, no padding
        (1, (2, 3), (1,), 3),  # 2D event, with padding
        (1, (2, 3, 4), (2,), 2),  # 3D event, no padding
        # With 2 sample dims
        (2, (3,), (1,), 1),  # 1D event, no padding
        (2, (2, 3), (1,), 2),  # 2D event, with padding
        # batch_ndims=2 cases
        (0, (3,), (2, 2), 4),  # 1D event, batch=(2,2), no padding
        (0, (2, 3), (2, 3), 10),  # 2D event, batch=(2,3), with padding
        (1, (2, 3), (2, 2), 6),  # 2D event, batch=(2,2), with padding
    ]
)
def test_flatten_leaf_shapes_and_padding(
    sample_ndims, event_shape, batch_shape, max_batch_size
):
    """Test flatten_leaf with various sample, event, batch dimensions."""
    batch_ndims = len(batch_shape)
    batch_size = int(jnp.prod(jnp.array(batch_shape)))

    # Construct sample shape
    sample_shape = tuple(2 for _ in range(sample_ndims))

    # Create input array with shape (*sample_shape, *event_shape, *batch_shape)
    full_shape = sample_shape + event_shape + batch_shape
    leaf = jnp.arange(
        1.0, float(jnp.prod(jnp.array(full_shape)) + 1)
    ).reshape(full_shape)

    # Flatten
    pad_value = -999.0
    result = flatten_leaf(
        leaf, sample_ndims, batch_ndims, pad_value, max_batch_size
    )

    # Verify output shape
    event_flat = int(jnp.prod(jnp.array(event_shape)) if event_shape else 1)
    expected_shape = sample_shape + (event_flat, max_batch_size)
    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {result.shape}"
    )

    # Verify all non-padded values are preserved
    assert jnp.allclose(
        result[..., :event_flat, :batch_size].flatten(),
        leaf.flatten()
    )

    # Check padding if applicable
    if max_batch_size > batch_size:
        assert jnp.allclose(
            result[..., batch_size:], pad_value
        )

def test_flatten_pytree_simple():
    """Test flatten_pytree with simple 2-key PyTree."""
    # Simple PyTree with 2 parameters
    # Convention: (*event, *batch)
    pytree = {
        'mu': jnp.array([[1.0], [2.0]]),  # shape: (2, 1) - 2 events, 1 batch
        'sigma': jnp.array([[0.5]])  # shape: (1, 1) - 1 event, 1 batch
    }

    flat_array, slices_dict = flatten_pytree(
        pytree, sample_ndims=0, batch_ndims={'mu': 1, 'sigma': 1}
    )

    # Check flat array shape: (2+1, 1) = (3, 1)
    assert flat_array.shape == (3, 1)
    # Check concatenation: [1.0, 2.0, 0.5]
    assert jnp.allclose(flat_array[:, 0], jnp.array([1.0, 2.0, 0.5]))

    # Check slices metadata
    assert 'mu' in slices_dict
    assert 'sigma' in slices_dict
    assert slices_dict['mu'].offset == 0
    assert slices_dict['sigma'].offset == 2
    assert slices_dict['mu'].event_shape == (2,)
    assert slices_dict['sigma'].event_shape == (1,)


@pytest.mark.parametrize(
    "batch_dims,expected_batch_size",
    [
        ({'mu': 1, 'sigma': 1}, 3),  # max(3, 1) = 3
        ({'mu': 1, 'sigma': 1}, 5),  # max(5, 1) = 5
    ]
)
def test_flatten_pytree_padding(batch_dims, expected_batch_size):
    """Test padding with different batch dimensions."""
    # Convention: (*event, *batch)
    # PyTree with different batch sizes
    pytree = {
        'mu': jnp.ones((2, expected_batch_size)),  # 2 events, batch size
        'sigma': jnp.ones((1, 1))  # 1 event, batch size 1
    }

    flat_array, slices_dict = flatten_pytree(
        pytree,
        sample_ndims=0,
        batch_ndims=batch_dims,
        pad_value=-999.0
    )

    # Check shape: (2+1 events, max_batch)
    assert flat_array.shape == (3, expected_batch_size)

    # Check padding applied to sigma
    # sigma should be padded from batch_size=1 to expected_batch_size
    sigma_offset = slices_dict['sigma'].offset
    sigma_data = flat_array[sigma_offset, :]

    # First element should be valid (1.0)
    assert jnp.allclose(sigma_data[:1], jnp.ones(1))
    # Remaining should be padding (-999.0)
    if expected_batch_size > 1:
        assert jnp.allclose(
            sigma_data[1:],
            jnp.full(expected_batch_size - 1, -999.0)
        )


@pytest.mark.parametrize(
    "structure",
    [
        {
            'mu': jnp.array([[1.0]]),  # (1 event, 1 batch)
            'theta': jnp.array([[2.0], [3.0]]),  # (2 events, 1 batch)
            'obs': jnp.array([[4.0]])  # (1 event, 1 batch)
        },
        {
            'hyperprior': jnp.array([[1.0]]),  # (1, 1)
            'prior_params': jnp.array([[2.0], [3.0]]),  # (2, 1)
            'theta': jnp.array([[4.0], [5.0]]),  # (2, 1)
            'obs': jnp.array([[6.0], [7.0], [8.0]])  # (3, 1)
        },
    ]
)
def test_flatten_reconstruct_roundtrip(structure):
    """Test round-trip: flatten → reconstruct using slices."""
    batch_ndims = {key: 1 for key in structure.keys()}

    # Flatten
    flat_array, slices_dict = flatten_pytree(
        structure, sample_ndims=0, batch_ndims=batch_ndims
    )

    # Reconstruct using slices
    reconstructed = {}
    for key, slice_info in slices_dict.items():
        offset = slice_info.offset
        event_shape = slice_info.event_shape
        batch_shape = slice_info.batch_shape

        # Calculate end offset
        event_size = int(jnp.prod(jnp.array(event_shape)))
        end_offset = offset + event_size

        # Extract slice and reshape
        slice_data = flat_array[offset:end_offset, :batch_shape[0]]
        reconstructed[key] = slice_data.reshape(
            event_shape + (batch_shape[0],)
        )

    # Verify round-trip
    for key in structure.keys():
        assert jnp.allclose(reconstructed[key], structure[key])


def test_update_flat_array():
    """Test update_flat_array updates correct offsets."""
    # Hardcoded flat array: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Representing: a=[1.0, 2.0], b=[3.0], c=[4.0, 5.0, 6.0]
    flat_array = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).T

    slices_dict = {
        'a': SliceInfo(
            offset=0,
            event_shape=(2,),
            batch_shape=(1,)
        ),
        'b': SliceInfo(
            offset=2,
            event_shape=(1,),
            batch_shape=(1,)
        ),
        'c': SliceInfo(
            offset=3,
            event_shape=(3,),
            batch_shape=(1,)
        )
    }

    # Update 'b' with new values
    # Shape should be (*event_shape, *batch_shape) = (1, 1)
    new_b_values = jnp.array([[99.0]])  # (1, 1)
    updated_array = update_flat_array(
        flat_array, slices_dict, 'b', new_b_values
    )

    # Check that only 'b' was updated
    assert jnp.allclose(updated_array[0:2, 0], jnp.array([1.0, 2.0]))  # a
    assert jnp.allclose(updated_array[2, 0], 99.0)  # b (updated)
    assert jnp.allclose(
        updated_array[3:6, 0], jnp.array([4.0, 5.0, 6.0])
    )  # c

    # Verify original array unchanged
    assert jnp.allclose(flat_array[2, 0], 3.0)


def test_update_flat_array_multiple_keys():
    """Test updating multiple keys in sequence."""
    # Hardcoded flat array: [1.0, 2.0, 3.0]
    # Representing: x=[1.0], y=[2.0, 3.0]
    flat_array = jnp.array([[1.0, 2.0, 3.0]]).T

    slices_dict = {
        'x': SliceInfo(
            offset=0,
            event_shape=(1,),
            batch_shape=(1,)
        ),
        'y': SliceInfo(
            offset=1,
            event_shape=(2,),
            batch_shape=(1,)
        )
    }

    # Update both keys
    # Shapes: (*event, *batch)
    updated = update_flat_array(
        flat_array, slices_dict, 'x', jnp.array([[10.0]])  # (1, 1)
    )
    updated = update_flat_array(
        updated, slices_dict, 'y', jnp.array([[20.0], [30.0]])  # (2, 1)
    )

    # Check both updates applied
    assert jnp.allclose(updated[0, 0], 10.0)
    assert jnp.allclose(updated[1:3, 0], jnp.array([20.0, 30.0]))


def test_update_flat_array_multi_dim_event():
    """Test update_flat_array with multi-dimensional event shape."""
    # Flat array with a 2x2 event shape flattened
    # Key 'matrix' has event_shape=(2, 2), batch_shape=(1,)
    flat_array = jnp.array([[1.0, 2.0, 3.0, 4.0]]).T

    slices_dict = {
        'matrix': SliceInfo(
            offset=0,
            event_shape=(2, 2),
            batch_shape=(1,)
        )
    }

    # New values: shape (*event_shape, *batch_shape) = (2, 2, 1)
    new_matrix = jnp.array([[[10.0], [20.0]], [[30.0], [40.0]]])

    updated = update_flat_array(
        flat_array, slices_dict, 'matrix', new_matrix
    )

    # Should be flattened as [10, 20, 30, 40]
    assert jnp.allclose(
        updated[:, 0], jnp.array([10.0, 20.0, 30.0, 40.0])
    )


def test_update_flat_array_with_sample_dims():
    """Test update_flat_array with sample dimensions."""
    # Flat array with sample dimension
    # Shape: (2 samples, 3 events, 1 batch)
    flat_array = jnp.array([
        [[1.0], [2.0], [3.0]],  # sample 0
        [[4.0], [5.0], [6.0]]   # sample 1
    ])

    slices_dict = {
        'a': SliceInfo(
            offset=0,
            event_shape=(2,),
            batch_shape=(1,)
        ),
        'b': SliceInfo(
            offset=2,
            event_shape=(1,),
            batch_shape=(1,)
        )
    }

    # Update 'a' across all samples
    # Shape: (2 samples, 2 events, 1 batch)
    new_a = jnp.array([
        [[10.0], [20.0]],  # sample 0
        [[30.0], [40.0]]   # sample 1
    ])

    updated = update_flat_array(flat_array, slices_dict, 'a', new_a)

    # Check sample 0
    assert jnp.allclose(updated[0, 0:2, 0], jnp.array([10.0, 20.0]))
    assert jnp.allclose(updated[0, 2, 0], 3.0)  # 'b' unchanged

    # Check sample 1
    assert jnp.allclose(updated[1, 0:2, 0], jnp.array([30.0, 40.0]))
    assert jnp.allclose(updated[1, 2, 0], 6.0)  # 'b' unchanged


def test_flatten_pytree_with_sample_dims():
    """Test flattening with sample_ndims > 0."""
    # Convention: (*sample, *event, *batch)
    # PyTree with sample dimension
    pytree = {
        'a': jnp.array([
            [[1.0], [2.0]],  # sample 0: 2 events, 1 batch
            [[3.0], [4.0]]   # sample 1: 2 events, 1 batch
        ]),  # shape: (2 samples, 2 events, 1 batch)
        'b': jnp.array([
            [[5.0]],  # sample 0: 1 event, 1 batch
            [[6.0]]   # sample 1: 1 event, 1 batch
        ])  # shape: (2 samples, 1 event, 1 batch)
    }

    flat_array, slices_dict = flatten_pytree(
        pytree,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    # Expected shape: (n_samples, total_event, batch)
    # = (2, 2+1, 1) = (2, 3, 1)
    assert flat_array.shape == (2, 3, 1)

    # Check sample 0
    assert jnp.allclose(
        flat_array[0, :, 0], jnp.array([1.0, 2.0, 5.0])
    )

    # Check sample 1
    assert jnp.allclose(
        flat_array[1, :, 0], jnp.array([3.0, 4.0, 6.0])
    )

    # Check slices are same for all samples
    assert slices_dict['a'].offset == 0
    assert slices_dict['b'].offset == 2


def test_flatten_pytree_with_multiple_sample_dims():
    """Test flattening with sample_ndims=2."""
    # PyTree with 2 sample dimensions
    pytree = {
        'param': jnp.array([
            [  # sample group 0
                [[1.0]],  # sample 0
                [[2.0]]   # sample 1
            ],
            [  # sample group 1
                [[3.0]],  # sample 0
                [[4.0]]   # sample 1
            ]
        ])  # shape: (2, 2, 1, 1)
    }

    flat_array, slices_dict = flatten_pytree(
        pytree,
        sample_ndims=2,
        batch_ndims={'param': 1}
    )

    # Expected shape: (2, 2, 1, 1)
    assert flat_array.shape == (2, 2, 1, 1)

    # Check values preserved across sample dims
    assert jnp.allclose(flat_array[0, 0, 0, 0], 1.0)
    assert jnp.allclose(flat_array[0, 1, 0, 0], 2.0)
    assert jnp.allclose(flat_array[1, 0, 0, 0], 3.0)
    assert jnp.allclose(flat_array[1, 1, 0, 0], 4.0)
