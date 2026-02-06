"""PyTree reconstruction utilities for parameter processing."""

import math
from typing import Dict, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from .utils import SliceInfo


def decode_pytree(
    flat_array: Array,
    slices_dict: Dict[str, SliceInfo],
    sample_shape: Tuple[int, ...],
    is_subset: bool = False
) -> Dict[str, Array]:
    """
    Reconstruct PyTree from flat array using slice metadata.

    Converts a flattened array back into a dictionary of parameter arrays
    using the slice metadata from flatten_pytree. Removes padding and
    restores original event and batch shapes.

    Parameters
    ----------
    flat_array : Array
        Flattened array with shape (*sample_shape, total_event, max_batch)
    slices_dict : Dict[str, SliceInfo]
        Slice metadata from flatten_pytree
    sample_shape : Tuple[int, ...]
        Shape of sample dimensions (empty tuple if no sample dims)
    is_subset : bool, optional
        If True, only validate that slices don't exceed bounds.
        If False, validate that slices exactly match flat array size.
        Default is False.

    Returns
    -------
    Dict[str, Array]
        Dictionary of reconstructed arrays, each with shape
        (*sample_shape, *event_shape, *batch_shape)

    Examples
    --------
    >>> flat = jnp.array([[1.0, 2.0, 0.5]]).T
    >>> slices = {
    ...     'mu': SliceInfo(offset=0, event_shape=(2,),
    ...                     batch_shape=(1,)),
    ...     'sigma': SliceInfo(offset=2, event_shape=(1,),
    ...                        batch_shape=(1,))
    ... }
    >>> result = decode_pytree(flat, slices, sample_shape=())
    >>> result['mu'].shape
    (2, 1)
    """
    # Get total size of flat array along event dimension
    event_dim_idx = len(sample_shape)
    total_event_size = flat_array.shape[event_dim_idx]

    # Validate slice metadata
    slice_event_sizes = [
        math.prod(info.event_shape) if info.event_shape else 1
        for info in slices_dict.values()
    ]
    expected_total = sum(slice_event_sizes)

    if is_subset:
        # For subset decoding, just check we don't exceed bounds
        if expected_total > total_event_size:
            raise ValueError(
                f"Invalid slices. Slice event sizes {slice_event_sizes} "
                f"sum to {expected_total} exceeds flat event size "
                f"{total_event_size}"
            )
    else:
        # For full decoding, require exact match
        if expected_total != total_event_size:
            raise ValueError(
                f"Invalid slices. Slice event sizes {slice_event_sizes} "
                f"sum to {expected_total} but flat event size is "
                f"{total_event_size}"
            )

    reconstructed = {}

    for key, slice_info in slices_dict.items():
        offset = slice_info.offset
        event_shape = slice_info.event_shape
        batch_shape = slice_info.batch_shape

        # Calculate event and batch sizes
        event_size = math.prod(event_shape) if event_shape else 1
        batch_size = math.prod(batch_shape) if batch_shape else 1

        # Extract slice from flat array
        # Shape: (*sample_shape, event_size, max_batch)
        block_slice = flat_array[..., offset:offset + event_size, :]

        # Extract only the valid batch elements (remove padding)
        # Shape: (*sample_shape, event_size, batch_size)
        block_slice = block_slice[..., :batch_size]

        # Reshape to final shape
        new_shape = sample_shape + event_shape + batch_shape
        reconstructed[key] = jnp.reshape(block_slice, new_shape)

    return reconstructed


def decode_pytree_keys(
    flat_array: Array,
    slices_dict: Dict[str, SliceInfo],
    sample_shape: Tuple[int, ...],
    keys: list[str]
) -> Dict[str, Array]:
    """
    Reconstruct only specified keys from flat array.

    Similar to decode_pytree, but only reconstructs a subset of keys.
    Useful when only a portion of the flattened data is needed.

    Parameters
    ----------
    flat_array : Array
        Flattened array with shape (*sample_shape, total_event, max_batch)
    slices_dict : Dict[str, SliceInfo]
        Slice metadata from flatten_pytree
    sample_shape : Tuple[int, ...]
        Shape of sample dimensions
    keys : list[str]
        List of keys to reconstruct

    Returns
    -------
    Dict[str, Array]
        Dictionary containing only the specified keys, each with shape
        (*sample_shape, *event_shape, *batch_shape)

    Examples
    --------
    >>> flat = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).T
    >>> slices = {
    ...     'mu': SliceInfo(offset=0, event_shape=(1,),
    ...                     batch_shape=(1,)),
    ...     'sigma': SliceInfo(offset=1, event_shape=(2,),
    ...                        batch_shape=(1,)),
    ...     'obs': SliceInfo(offset=3, event_shape=(3,),
    ...                      batch_shape=(1,))
    ... }
    >>> result = decode_pytree_keys(
    ...     flat, slices, sample_shape=(), keys=['mu', 'obs']
    ... )
    >>> set(result.keys())
    {'mu', 'obs'}
    """
    # Create filtered slices dict with only selected keys
    filtered_slices = {k: slices_dict[k] for k in keys}

    # Use decode_pytree with filtered slices, marking as subset
    return decode_pytree(
        flat_array, filtered_slices, sample_shape, is_subset=True
    )
