"""Utility functions for parameter processing."""

import math
from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array


def size_along_axes(arr: Array, axes: Tuple[int, ...]) -> int:
    """
    Compute the product of array dimensions along specified axes.

    Parameters
    ----------
    arr : Array
        Input array
    axes : Tuple[int, ...]
        Axes to compute size over

    Returns
    -------
    int
        Product of dimensions along specified axes

    Notes
    -----
    Uses math.prod to avoid tracer issues in JIT contexts.
    """
    if not axes:
        return 1
    shape_subset = tuple(arr.shape[ax] for ax in axes)
    return math.prod(shape_subset)


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
