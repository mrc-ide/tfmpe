"""PyTree flattening utilities for parameter processing."""

from typing import Dict, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from tfmpe.preprocessing.utils import SliceInfo

def flatten_leaf(
    leaf: Array,
    sample_ndims: int,
    batch_ndims: int,
    pad_value: float,
    max_batch_size: int,
) -> Array:
    """
    Flatten a single leaf array.

    Flattens event dimensions and pads batch dimensions to
    max_batch_size.

    Parameters
    ----------
    leaf : Array
        Array with shape (*sample_dims, *event_dims, *batch_dims)
    sample_ndims : int
        Number of leading sample dimensions
    batch_ndims : int
        Number of trailing batch dimensions
    pad_value : float
        Value to use for padding
    max_batch_size : int
        Maximum batch size to pad to

    Returns
    -------
    Array
        Flattened array with shape
        (*sample_dims, event_flat, batch_flat)
    """
    # Extract sample shape
    sample_shape = leaf.shape[:sample_ndims]

    # Compute flattened sizes
    event_axes = tuple(
        range(sample_ndims, len(leaf.shape) - batch_ndims)
    )
    batch_axes = tuple(
        range(len(leaf.shape) - batch_ndims, len(leaf.shape))
    )
    flatten_evt_size = jnp.size(leaf, axis=event_axes)
    batch_size = jnp.size(leaf, axis=batch_axes)

    # Reshape to (*sample_dims, flatten_evt_size, batch_size)
    leaf_reshaped = leaf.reshape(
        sample_shape + (flatten_evt_size,) + (batch_size,)
    )

    # Pad batch dim to max_batch_size
    pad_width = (
        [(0, 0)] * (sample_ndims + 1) +
        [(0, max_batch_size - batch_size)]
    )
    leaf_reshaped = jnp.pad(
        leaf_reshaped,
        pad_width,
        constant_values=pad_value
    )
    return leaf_reshaped


def flatten_pytree(
    pytree: Dict[str, Array],
    sample_ndims: int,
    batch_ndims: Dict[str, int],
    pad_value: float = 0.0
) -> Tuple[Array, Dict[str, SliceInfo]]:
    """
    Flatten a PyTree into a single array with slice metadata.

    Converts a dictionary of parameter arrays into a single flattened
    array by concatenating along the event dimension. Each array's
    event dimensions are flattened, and batch dimensions are padded
    to the maximum batch size across all keys.

    Parameters
    ----------
    pytree : Dict[str, Array]
        Dictionary of parameter arrays. Each array should have shape
        (*sample_dims, *event_dims, *batch_dims), where event_dims
        are in the middle and batch_dims are trailing.
    sample_ndims : int
        Number of leading sample dimensions (preserved in output).
    batch_ndims : Dict[str, int]
        Number of trailing batch dimensions for each key.
    pad_value : float, optional
        Value to use for padding smaller batches. Default is 0.0.

    Returns
    -------
    flat_array : Array
        Flattened array with shape
        (*sample_dims, total_event, max_batch)
        where total_event is the sum of flattened event dimensions
        and max_batch is the maximum batch size across all keys.
    slices_dict : Dict[str, SliceInfo]
        Metadata for reconstructing original structure

    Examples
    --------
    >>> pytree = {
    ...     'mu': jnp.array([[1.0, 2.0]]),  # (1, 2)
    ...     'sigma': jnp.array([[0.5]])  # (1, 1)
    ... }
    >>> flat, slices = flatten_pytree(
    ...     pytree, sample_ndims=0, batch_ndims={'mu': 1, 'sigma': 1}
    ... )
    >>> flat.shape
    (3, 1)
    """
    # Determine max batch size
    max_batch_size = 1
    for key, leaf in pytree.items():
        batch_ndim = batch_ndims.get(key, 1)
        batch_axes = tuple(
            range(len(leaf.shape) - batch_ndim, len(leaf.shape))
        )
        batch_size = jnp.size(leaf, axis=batch_axes)
        max_batch_size = max(max_batch_size, batch_size)

    # Flatten each leaf and collect metadata
    flattened_leaves = []
    slices = {}
    current_offset = 0

    for key, leaf in pytree.items():
        batch_ndim = batch_ndims.get(key, 1)

        # Flatten leaf
        leaf_flat = flatten_leaf(
            leaf,
            sample_ndims,
            batch_ndim,
            pad_value,
            max_batch_size
        )

        # leaf_flat shape: (*sample_dims, flatten_evt_size, max_batch)
        block_size = leaf_flat.shape[sample_ndims]

        # Store metadata
        event_shape = leaf.shape[sample_ndims:-batch_ndim]
        batch_shape = leaf.shape[-batch_ndim:]
        slices[key] = SliceInfo(
            offset=current_offset,
            event_shape=event_shape,
            batch_shape=batch_shape
        )
        current_offset += block_size
        flattened_leaves.append(leaf_flat)

    # Concatenate along event dimension (axis=sample_ndims)
    flat_data = jnp.concatenate(flattened_leaves, axis=sample_ndims)
    return flat_data, slices
