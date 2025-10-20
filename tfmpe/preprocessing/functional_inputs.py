"""Functional input processing utilities for parameter tokens."""

import math
from typing import Dict, Any, Optional

import jax.numpy as jnp
from jaxtyping import Array

from .flatten import flatten_leaf

# Padding value used for non-functional inputs
FUNCTIONAL_INPUT_PAD_VALUE = -1e8

def flatten_functional_inputs(
    functional_inputs: Optional[Dict[str, Array]],
    slices: Dict[str, Dict[str, Any]],
    sample_ndims: int
) -> Optional[Array]:
    """
    Flatten functional inputs aligned with token slices.

    Functional inputs (e.g., observation times, spatial coordinates)
    are auxiliary data that vary with the parameter structure. This
    function flattens them to match the flattened token structure,
    with padding for keys that have smaller batch dimensions or
    missing functional inputs.

    The legacy code assumes functional inputs always have batch_ndims=1
    (the last dimension is the batch dimension).

    Parameters
    ----------
    functional_inputs : Optional[Dict[str, Array]]
        Dictionary of functional input arrays. Each array should have
        shape (*sample_dims, *event_dims, batch_dim). The last
        dimension is treated as the batch dimension. If None,
        returns None.
    slices : Dict[str, Dict[str, Any]]
        Slice metadata from flatten_pytree, containing 'offset',
        'event_shape', and 'batch_shape' for each key.
    sample_ndims : int
        Number of leading sample dimensions to preserve.

    Returns
    -------
    Optional[Array]
        Flattened functional inputs with shape
        (*sample_dims, total_event, max_batch), or None if
        functional_inputs is None.

    Examples
    --------
    >>> functional_inputs = {
    ...     'obs': jnp.ones((5, 3, 1))  # (event1, event2, batch)
    ... }
    >>> slices = {
    ...     'obs': {
    ...         'offset': 0,
    ...         'event_shape': (5, 3),
    ...         'batch_shape': (1,)
    ...     }
    ... }
    >>> result = flatten_functional_inputs(
    ...     functional_inputs, slices, sample_ndims=0
    ... )
    >>> result.shape
    (15, 1)
    """
    if functional_inputs is None:
        return None

    # Determine max batch size - functional inputs always have
    # batch_ndims=1 (last dim)
    max_batch_size = 1
    for key, f_in in functional_inputs.items():
        if key in slices:
            batch_size = f_in.shape[-1]
            max_batch_size = max(max_batch_size, batch_size)

    # Compute total event size from slices
    total_event_size = sum(
        math.prod(info['event_shape']) for info in slices.values()
    )

    # Get sample shape from first functional input
    first_key = next(iter(functional_inputs.keys()))
    sample_shape = functional_inputs[first_key].shape[:sample_ndims]

    # Initialize output array filled with pad_value
    output_shape = sample_shape + (total_event_size, max_batch_size)
    output = jnp.full(output_shape, FUNCTIONAL_INPUT_PAD_VALUE)

    # Process each key in slices order
    for key, slice_info in slices.items():
        offset = slice_info['offset']
        event_shape = slice_info['event_shape']
        event_size = math.prod(event_shape)

        if key not in functional_inputs:
            # Key has no functional inputs, leave as pad_value
            continue

        f_in = functional_inputs[key]

        # Flatten with batch_ndims=1 (legacy behavior)
        f_in_flat = flatten_leaf(
            f_in,
            sample_ndims,
            batch_ndims=1,
            pad_value=FUNCTIONAL_INPUT_PAD_VALUE,
            max_batch_size=max_batch_size
        )

        # Insert into output at correct offset
        if sample_ndims == 0:
            output = output.at[offset:offset + event_size, :].set(
                f_in_flat
            )
        else:
            output = output.at[
                ..., offset:offset + event_size, :
            ].set(f_in_flat)

    return output
