"""Utility functions for parameter processing."""

import math
from typing import Tuple

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
