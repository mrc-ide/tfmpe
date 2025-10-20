"""Utility functions for parameter processing."""

import math
from typing import NamedTuple, Tuple
from jaxtyping import Array


class SliceInfo(NamedTuple):
    """
    Metadata for a slice in a flattened array.

    Attributes
    ----------
    offset : int
        Starting index in the flattened event dimension
    event_shape : Tuple[int, ...]
        Original event dimensions
    batch_shape : Tuple[int, ...]
        Original batch dimensions
    """
    offset: int
    event_shape: Tuple[int, ...]
    batch_shape: Tuple[int, ...]

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
