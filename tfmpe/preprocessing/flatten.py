"""PyTree flattening utilities for parameter processing."""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from tfmpe.preprocessing.utils import size_along_axes


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
    flatten_evt_size = size_along_axes(leaf, event_axes)
    batch_size = size_along_axes(leaf, batch_axes)

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
) -> Tuple[Array, Dict[str, Dict[str, Any]]]:
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
    slices_dict : Dict[str, Dict[str, Any]]
        Metadata for reconstructing original structure. Each key maps
        to:
        - 'offset': Starting index in flattened event dimension
        - 'event_shape': Original event dimensions
        - 'batch_shape': Original batch dimensions

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
        batch_size = size_along_axes(leaf, batch_axes)
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
        slices[key] = {
            "offset": current_offset,
            "event_shape": event_shape,
            "batch_shape": batch_shape
        }
        current_offset += block_size
        flattened_leaves.append(leaf_flat)

    # Concatenate along event dimension (axis=sample_ndims)
    flat_data = jnp.concatenate(flattened_leaves, axis=sample_ndims)
    return flat_data, slices


def update_flat_array(
    flat_array: Array,
    slices_dict: Dict[str, Dict[str, Any]],
    key: str,
    new_values: Array
) -> Array:
    """
    Update values for a specific key in the flat array.

    Creates a new flat array with the specified key's values replaced.
    The new values are flattened and inserted at the correct offset.

    Parameters
    ----------
    flat_array : Array
        The flat array to update, with shape
        (*sample_dims, total_event, max_batch)
    slices_dict : Dict[str, Dict[str, Any]]
        Slice metadata from flatten_pytree
    key : str
        Key to update
    new_values : Array
        New values for the key, with shape matching the original
        (batch_shape, *event_shape)

    Returns
    -------
    Array
        Updated flat array with the same shape as flat_array

    Examples
    --------
    >>> flat = jnp.array([[1.0, 2.0, 3.0]]).T
    >>> slices = {
    ...     'a': {'offset': 0, 'event_shape': (2,),
    ...           'batch_shape': (1,)},
    ...     'b': {'offset': 2, 'event_shape': (1,),
    ...           'batch_shape': (1,)}
    ... }
    >>> updated = update_flat_array(
    ...     flat, slices, 'b', jnp.array([[99.0]])
    ... )
    >>> updated[2, 0]
    Array(99., dtype=float32)
    """
    slice_info = slices_dict[key]
    offset = slice_info['offset']
    event_shape = slice_info['event_shape']
    batch_shape = slice_info['batch_shape']

    # Flatten new values
    # new_values has shape (*batch_shape, *event_shape)
    batch_axes = tuple(range(len(batch_shape)))
    event_axes = tuple(range(len(batch_shape), len(new_values.shape)))
    batch_size = size_along_axes(new_values, batch_axes)
    event_size = size_along_axes(new_values, event_axes)

    # Reshape to (event_size, batch_size)
    new_values_flat = new_values.reshape(
        (batch_size,) + event_shape
    ).reshape((batch_size, event_size)).T

    # Update the flat array
    # Handle both with and without sample dimensions
    if flat_array.ndim == 2:
        # Shape: (total_event, max_batch)
        updated = flat_array.at[
            offset:offset + event_size,
            :batch_size
        ].set(new_values_flat)
    else:
        # Shape: (*sample_dims, total_event, max_batch)
        # For now, assume updating all samples identically
        updated = flat_array.at[
            ...,
            offset:offset + event_size,
            :batch_size
        ].set(new_values_flat)

    return updated
