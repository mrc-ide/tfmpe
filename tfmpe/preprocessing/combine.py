"""Combine multiple Tokens objects for dataset accumulation."""

from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array

from .functional_inputs import FUNCTIONAL_INPUT_PAD_VALUE
from .masks import build_self_attention_mask
from .tokens import Tokens
from .utils import Independence, SliceInfo


def combine_tokens(tokens1: Tokens, tokens2: Tokens) -> Tokens:
    """
    Combine two Tokens objects by concatenating samples and padding tokens.

    Parameters
    ----------
    tokens1 : Tokens
        First Tokens object
    tokens2 : Tokens
        Second Tokens object

    Returns
    -------
    Tokens
        Combined Tokens with concatenated samples and padded tokens

    Raises
    ------
    ValueError
        If tokens have different key_order or independence specs
    ValueError
        If one has functional_inputs and the other doesn't

    Notes
    -----
    - Sample dimension (axis 0) is concatenated
    - Token dimension (axis 1) is padded to max across both inputs
    - Slices are updated to reflect max event_shape per key
    - Labels are extended to match padded token dimension
    - Self-attention masks are padded with zeros
    - Padding masks track which tokens are real vs padded
    """
    # Validate compatibility
    if tokens1.key_order != tokens2.key_order:
        raise ValueError(
            f"Cannot combine tokens with different key_order: "
            f"{tokens1.key_order} vs {tokens2.key_order}"
        )

    if tokens1.independence != tokens2.independence:
        raise ValueError(
            "Cannot combine tokens with different independence specs"
        )

    # Check functional_inputs consistency
    has_func1 = tokens1.functional_inputs is not None
    has_func2 = tokens2.functional_inputs is not None
    if has_func1 != has_func2:
        raise ValueError(
            "Cannot combine tokens: one has functional_inputs, "
            "the other does not"
        )

    # Compute max event_shape per key
    max_event_shapes: Dict[str, tuple] = {}
    for key in tokens1.key_order:
        shape1 = tokens1.slices[key].event_shape
        shape2 = tokens2.slices[key].event_shape
        # Element-wise max
        max_shape = tuple(
            max(s1, s2) for s1, s2 in zip(shape1, shape2)
        )
        max_event_shapes[key] = max_shape

    # Build new slices with max event shapes
    new_slices: Dict[str, SliceInfo] = {}
    offset = 0
    for key in tokens1.key_order:
        event_shape = max_event_shapes[key]
        n_tokens = 1
        for dim in event_shape:
            n_tokens *= dim

        # Use batch_shape from tokens1 (should be same for both)
        batch_shape = tokens1.slices[key].batch_shape

        new_slices[key] = SliceInfo(
            offset=offset,
            event_shape=event_shape,
            batch_shape=batch_shape
        )
        offset += n_tokens

    max_n_tokens = offset

    # Get sample shapes
    sample_ndims = len(tokens1.sample_shape)
    n_samples1 = tokens1.data.shape[0]
    n_samples2 = tokens2.data.shape[0]

    # Pad data arrays to max_n_tokens
    data1_padded = _pad_data_to_max_tokens(
        tokens1.data, max_n_tokens, sample_ndims
    )
    data2_padded = _pad_data_to_max_tokens(
        tokens2.data, max_n_tokens, sample_ndims
    )

    # Concatenate data along sample dimension (axis 0)
    combined_data = jnp.concatenate([data1_padded, data2_padded], axis=0)

    # Build combined labels by extending per key
    combined_labels = _build_combined_labels(
        tokens1, tokens2, new_slices, n_samples1, n_samples2,
        max_n_tokens
    )

    # Build combined self-attention mask using new slices
    independence = (
        tokens1.independence if tokens1.independence is not None
        else Independence()
    )
    combined_sa_mask = build_self_attention_mask(new_slices, independence)

    # Build combined padding mask
    combined_padding_mask = _build_combined_padding_mask(
        tokens1, tokens2, new_slices, n_samples1, n_samples2,
        max_n_tokens
    )

    # Combine functional inputs if present
    combined_func_inputs = None
    if has_func1 and tokens1.functional_inputs is not None and tokens2.functional_inputs is not None:
        func1_padded = _pad_data_to_max_tokens(
            tokens1.functional_inputs,
            max_n_tokens,
            sample_ndims,
            pad_value=FUNCTIONAL_INPUT_PAD_VALUE
        )
        func2_padded = _pad_data_to_max_tokens(
            tokens2.functional_inputs,
            max_n_tokens,
            sample_ndims,
            pad_value=FUNCTIONAL_INPUT_PAD_VALUE
        )
        combined_func_inputs = jnp.concatenate(
            [func1_padded, func2_padded], axis=0
        )

    return Tokens(
        data=combined_data,
        labels=combined_labels,
        self_attention_mask=combined_sa_mask,
        padding_mask=combined_padding_mask,
        functional_inputs=combined_func_inputs,
        slices=new_slices,
        label_map=tokens1.label_map,
        key_order=tokens1.key_order,
        independence=tokens1.independence
    )


def _pad_data_to_max_tokens(
    data: Array,
    max_n_tokens: int,
    sample_ndims: int,
    pad_value: float = 0.0
) -> Array:
    """
    Pad data array to max_n_tokens on the token dimension.

    Parameters
    ----------
    data : Array
        Data array with shape (*sample_shape, n_tokens, max_batch_size)
    max_n_tokens : int
        Target number of tokens
    sample_ndims : int
        Number of sample dimensions
    pad_value : float
        Value to use for padding

    Returns
    -------
    Array
        Padded array with shape (*sample_shape, max_n_tokens,
        max_batch_size)
    """
    current_n_tokens = data.shape[sample_ndims]

    if current_n_tokens >= max_n_tokens:
        return data

    # Calculate padding: [(0, 0), ..., (0, pad_amount), (0, 0)]
    pad_width = [(0, 0)] * len(data.shape)
    # Pad on token dimension (axis=sample_ndims)
    pad_width[sample_ndims] = (0, max_n_tokens - current_n_tokens)

    return jnp.pad(
        data, pad_width, mode='constant', constant_values=pad_value
    )


def _build_combined_labels(
    tokens1: Tokens,
    tokens2: Tokens,
    new_slices: Dict,
    n_samples1: int,
    n_samples2: int,
    max_n_tokens: int
) -> Array:
    """
    Build combined labels array with extended labels per key.

    Parameters
    ----------
    tokens1 : Tokens
        First Tokens object
    tokens2 : Tokens
        Second Tokens object
    new_slices : Dict
        Updated slices with max event shapes
    n_samples1 : int
        Number of samples in tokens1
    n_samples2 : int
        Number of samples in tokens2
    max_n_tokens : int
        Maximum number of tokens

    Returns
    -------
    Array
        Combined labels array
    """
    # Build new labels based on new_slices
    labels_list = []
    for key in tokens1.key_order:
        event_shape = new_slices[key].event_shape
        n_tokens = 1
        for dim in event_shape:
            n_tokens *= dim
        key_labels = jnp.full(
            n_tokens, tokens1.label_map[key], dtype=jnp.int32
        )
        labels_list.append(key_labels)

    labels_1d = jnp.concatenate(labels_list)

    # Broadcast to combined sample shape
    sample_ndims = len(tokens1.sample_shape)
    if sample_ndims > 0:
        for _ in range(sample_ndims):
            labels_1d = jnp.expand_dims(labels_1d, 0)
        n_samples_combined = n_samples1 + n_samples2
        broadcast_shape = (n_samples_combined,) + (max_n_tokens,)
        labels = jnp.broadcast_to(labels_1d, broadcast_shape)
    else:
        labels = labels_1d

    return labels


def _build_combined_padding_mask(
    tokens1: Tokens,
    tokens2: Tokens,
    new_slices: Dict,
    n_samples1: int,
    n_samples2: int,
    max_n_tokens: int
) -> Array:
    """
    Build combined padding mask tracking real vs padded tokens.

    Parameters
    ----------
    tokens1 : Tokens
        First Tokens object
    tokens2 : Tokens
        Second Tokens object
    new_slices : Dict
        Updated slices with max event shapes
    n_samples1 : int
        Number of samples in tokens1
    n_samples2 : int
        Number of samples in tokens2
    max_n_tokens : int
        Maximum number of tokens

    Returns
    -------
    Array
        Combined padding mask with shape (n_samples_combined,
        max_n_tokens)
    """
    # Build padding masks per sample based on original event shapes
    def _build_padding_for_tokens(tokens: Tokens, n_samples: int) -> Array:
        mask = jnp.zeros((n_samples, max_n_tokens), dtype=jnp.int32)

        for key in tokens.key_order:
            orig_event_shape = tokens.slices[key].event_shape

            # Number of real tokens for this key
            n_real_tokens = 1
            for dim in orig_event_shape:
                n_real_tokens *= dim

            # Offset in the new (combined) token space
            offset = new_slices[key].offset

            # Set mask to 1 for real tokens
            mask = mask.at[:, offset:offset + n_real_tokens].set(1)

        return mask

    mask1 = _build_padding_for_tokens(tokens1, n_samples1)
    mask2 = _build_padding_for_tokens(tokens2, n_samples2)

    return jnp.concatenate([mask1, mask2], axis=0)
