"""Utility functions for parameter processing."""

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple
from jaxtyping import Array


@dataclass
class Independence:
    """
    Independence specification for attention masking.

    Defines which parameters should have restricted attention patterns
    in the transformer architecture.

    Attributes
    ----------
    local : List[str]
        Keys with local independence (no self-attention within the key).
        Example: ['obs', 'theta'] means obs tokens cannot attend to other
        obs tokens, and theta tokens cannot attend to other theta tokens.
    cross : List[Tuple[str, str]]
        Pairs of keys with cross independence (no attention between them).
        Example: [('mu', 'obs')] means mu tokens cannot attend to obs
        tokens and vice versa.
    cross_local : List[Tuple[str, str, Optional[Tuple[int, int]]]]
        Pairs of keys with cross-local independence (structured attention).
        Each tuple is (keyA, keyB, idx_map) where:
        - If idx_map is None: diagonal attention only (sizes must match)
        - If idx_map is (dimA, dimB): tokens can attend across matching
          indices along the specified dimensions
        Example: [('theta', 'obs', (0, 0))] means theta[i] can attend
        to obs[i] but not obs[j] for j != i.
    """
    local: List[str] = field(default_factory=list)
    cross: List[Tuple[str, str]] = field(default_factory=list)
    cross_local: List[Tuple[str, str, Optional[Tuple[int, int]]]] = (
        field(default_factory=list)
    )

    def __bool__(self) -> bool:
        """
        Return True if any independence rules are defined, False otherwise.

        Allows Independence to be used as a circuit breaker:
        >>> if independence:
        ...     apply_masking()
        """
        return bool(self.local or self.cross or self.cross_local)


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
