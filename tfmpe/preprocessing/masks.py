"""Mask generation utilities for attention and padding.

Functions for creating self-attention, cross-attention, and padding masks
from independence specifications and slice metadata.
"""

from math import prod
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .utils import SliceInfo


def _event_indices(
    shape: Tuple[int, ...],
    dim: int
) -> List[np.ndarray]:
    """
    Generate flat indices for slicing along a specific dimension.

    For a flattened array from shape `shape`, returns indices that
    correspond to fixing dimension `dim` to each possible value.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Original event shape before flattening
    dim : int
        Dimension to slice along

    Returns
    -------
    List[np.ndarray]
        List of length shape[dim], where each entry contains flat
        indices corresponding to that slice

    Examples
    --------
    >>> shape = (3, 2)  # 3 rows, 2 cols
    >>> indices = _event_indices(shape, 0)
    >>> # indices[0] = [0, 1] (first row)
    >>> # indices[1] = [2, 3] (second row)
    >>> # indices[2] = [4, 5] (third row)
    """
    total_elems = int(np.prod(shape))
    flat_idx = np.arange(total_elems, dtype=int).reshape(shape)
    return [
        flat_idx.take(i, axis=dim).ravel()
        for i in range(shape[dim])
    ]


def build_self_attention_mask(
    block_slices: Dict[str, SliceInfo],
    independence: Dict
) -> Array:
    """
    Build self-attention mask from independence specification.

    Creates a binary mask where 1 indicates allowed attention and 0
    indicates blocked attention, based on independence rules.

    Independence Types
    ------------------

    1. 'local': List of keys (e.g. ['keyA']) - Blocks self-attention within a key

       key = keyA, event_shape = (2,)
       key = keyB, event_shape = (2,)

       keyA tokens: [0, 1], keyB tokens: [2, 3]

              keyA     keyB
               0  1    2  3
          0 [0  0  | 1  1]  keyA
          1 [0  0  | 1  1]
            ------|------
          2 [1  1  | 1  1]  keyB
          3 [1  1  | 1  1]

    2. 'cross': List of (keyA, keyB) - Blocks all attention between
       keyA and keyB

       keyA, event_shape = (3,)
       keyB, event_shape = (3,)

       keyA tokens: [0, 1, 2], keyB tokens: [3, 4, 5]

              keyA        keyB
               0  1  2   3  4  5
          0 [1  1  1 | 0  0  0]  keyA
          1 [1  1  1 | 0  0  0]
          2 [1  1  1 | 0  0  0]
            --------|--------
          3 [0  0  0 | 1  1  1]  keyB
          4 [0  0  0 | 1  1  1]
          5 [0  0  0 | 1  1  1]

    3. 'cross_local': List of (keyA, keyB, idx_map)

       a) idx_map = None: Diagonal attention only (sizes must match)

          keyA, event_shape = (3,)
          keyB, event_shape = (3,)

              keyA        keyB
               0  1  2   3  4  5
          0 [1  1  1 | 1  0  0]  keyA
          1 [1  1  1 | 0  1  0]
          2 [1  1  1 | 0  0  1]
            --------|--------
          3 [1  0  0 | 1  1  1]  keyB
          4 [0  1  0 | 1  1  1]
          5 [0  0  1 | 1  1  1]

       b) idx_map = (dimA, dimB): Attention along shared dimension

          keyA, event_shape = (2, 3)  # 2 groups, 3 dims
          keyB, event_shape = (3,)    # 3 dims
          idx_map = (1, 0)  # keyA's dim 1 matches keyB's dim 0

          keyA tokens: [0,1,2, 3,4,5], keyB tokens: [6,7,8]
                        group0  group1

              keyA                 keyB
               0  1  2  3  4  5   6  7  8
          0 [1  1  1  1  1  1 | 1  0  0]  keyA group 0, dim 0
          1 [1  1  1  1  1  1 | 0  1  0]  keyA group 0, dim 1
          2 [1  1  1  1  1  1 | 0  0  1]  keyA group 0, dim 2
          3 [1  1  1  1  1  1 | 1  0  0]  keyA group 1, dim 0
          4 [1  1  1  1  1  1 | 0  1  0]  keyA group 1, dim 1
          5 [1  1  1  1  1  1 | 0  0  1]  keyA group 1, dim 2
            -----------------|---------
          6 [1  0  0  1  0  0 | 1  1  1]  keyB dim 0
          7 [0  1  0  0  1  0 | 1  1  1]  keyB dim 1
          8 [0  0  1  0  0  1 | 1  1  1]  keyB dim 2

    Parameters
    ----------
    block_slices : Dict[str, SliceInfo]
        Slice metadata for each block
    independence : Dict
        Independence specification with keys:
        - 'local': List of keys with local independence (no
          self-attention)
        - 'cross': List of (keyA, keyB) tuples with cross independence
        - 'cross_local': List of (keyA, keyB, idx_map) tuples where
          idx_map is either None (diagonal) or (dimA, dimB) tuple

    Returns
    -------
    Array
        Binary mask of shape (total_tokens, total_tokens) with dtype
        float32

    Examples
    --------
    Cross-local with diagonal attention:

    >>> slices = {
    ...     'theta': {'offset': 0, 'event_shape': (3,)},
    ...     'obs': {'offset': 3, 'event_shape': (3,)}
    ... }
    >>> independence = {'cross_local': [('theta', 'obs', None)]}
    >>> mask = build_self_attention_mask(slices, independence)
    >>> mask
    Array([[1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 0., 1., 0.],
           [1., 1., 1., 0., 0., 1.],
           [1., 0., 0., 1., 1., 1.],
           [0., 1., 0., 1., 1., 1.],
           [0., 0., 1., 1., 1., 1.]], dtype=float32)

    Cross independence blocks all attention:

    >>> independence = {'cross': [('theta', 'obs')]}
    >>> mask = build_self_attention_mask(slices, independence)
    >>> mask
    Array([[1., 1., 1., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.]], dtype=float32)

    Local independence blocks self-attention:

    >>> slices = {'theta': {'offset': 0, 'event_shape': (3,)}}
    >>> independence = {'local': ['theta']}
    >>> mask = build_self_attention_mask(slices, independence)
    >>> mask
    Array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    """
    total_size = sum(
        prod(s.event_shape) for s in block_slices.values()
    )

    # Build default mask (all ones)
    mask_np = np.ones((total_size, total_size), dtype=np.int8)

    # Apply cross independence (zero out entire sub-blocks)
    for (blockA, blockB) in independence.get("cross", []):
        if blockA not in block_slices or blockB not in block_slices:
            continue
        offA = block_slices[blockA].offset
        sizeA = prod(block_slices[blockA].event_shape)
        offB = block_slices[blockB].offset
        sizeB = prod(block_slices[blockB].event_shape)

        # Zero out sub-block
        mask_np[offA:offA+sizeA, offB:offB+sizeB] = 0

    # Apply local independence (zero out self-blocks)
    for key in independence.get("local", []):
        if key in block_slices:
            off = block_slices[key].offset
            sz = prod(block_slices[key].event_shape)
            mask_np[off:off+sz, off:off+sz] = 0

    # Apply cross_local independence
    for spec in independence.get("cross_local", []):
        blockA, blockB, idx_map = spec
        if blockA not in block_slices or blockB not in block_slices:
            continue

        offA = block_slices[blockA].offset
        sizeA = prod(block_slices[blockA].event_shape)
        shapeA = block_slices[blockA].event_shape

        offB = block_slices[blockB].offset
        sizeB = prod(block_slices[blockB].event_shape)
        shapeB = block_slices[blockB].event_shape

        # Zero out entire sub-block first
        mask_np[offA:offA+sizeA, offB:offB+sizeB] = 0
        mask_np[offB:offB+sizeB, offA:offA+sizeA] = 0

        if idx_map is None:
            # Diagonal only (sizes must match)
            if sizeA != sizeB:
                raise ValueError(
                    f"Cannot do diagonal cross_local if block sizes "
                    f"differ: {blockA}={sizeA}, {blockB}={sizeB}"
                )
            for i in range(sizeA):
                mask_np[offA+i, offB+i] = 1
                mask_np[offB+i, offA+i] = 1
        else:
            # Use index map to connect matching indices
            dim_a, dim_b = idx_map
            if dim_a >= len(shapeA) or dim_b >= len(shapeB):
                raise ValueError(
                    f"Index map dimensions invalid: "
                    f"dim_a={dim_a} >= {len(shapeA)} or "
                    f"dim_b={dim_b} >= {len(shapeB)}"
                )
            if shapeA[dim_a] != shapeB[dim_b]:
                raise ValueError(
                    f"Cannot do cross_local if event shapes do not "
                    f"match along specified dimensions: "
                    f"{shapeA[dim_a]} != {shapeB[dim_b]}"
                )

            # Get flat indices for each slice along the dimension
            a_idx = _event_indices(shapeA, dim_a)
            b_idx = _event_indices(shapeB, dim_b)

            # Enable attention between matching slices
            for (a_indices, b_indices) in zip(a_idx, b_idx):
                for a in a_indices:
                    for b in b_indices:
                        mask_np[offA+a, offB+b] = 1
                        mask_np[offB+b, offA+a] = 1

    return jnp.array(mask_np, dtype=jnp.float32)


def build_cross_attention_mask(
    query_slices: Dict[str, SliceInfo],
    key_slices: Dict[str, SliceInfo],
    independence: Dict
) -> Array:
    """
    Build cross-attention mask between query and key tokens.

    Creates a binary mask for cross-attention where rows correspond to
    query tokens and columns to key tokens.

    Parameters
    ----------
    query_slices : Dict[str, SliceInfo]
        Slice metadata for query blocks
    key_slices : Dict[str, SliceInfo]
        Slice metadata for key blocks
    independence : Dict
        Independence specification (same format as
        build_self_attention_mask)

    Returns
    -------
    Array
        Binary mask of shape (total_query_tokens, total_key_tokens)
        with dtype float32

    Examples
    --------
    Cross-local with diagonal attention:

    >>> query_slices = {'theta': {'offset': 0, 'event_shape': (3,)}}
    >>> key_slices = {'obs': {'offset': 0, 'event_shape': (3,)}}
    >>> independence = {'cross_local': [('theta', 'obs', None)]}
    >>> mask = build_cross_attention_mask(
    ...     query_slices, key_slices, independence
    ... )
    >>> mask
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

    Cross independence blocks all attention:

    >>> independence = {'cross': [('theta', 'obs')]}
    >>> mask = build_cross_attention_mask(
    ...     query_slices, key_slices, independence
    ... )
    >>> mask
    Array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

    Multiple query and key blocks (default all ones):

    >>> query_slices = {
    ...     'theta': {'offset': 0, 'event_shape': (2,)},
    ...     'phi': {'offset': 2, 'event_shape': (2,)}
    ... }
    >>> key_slices = {'obs': {'offset': 0, 'event_shape': (3,)}}
    >>> independence = {}
    >>> mask = build_cross_attention_mask(
    ...     query_slices, key_slices, independence
    ... )
    >>> mask
    Array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    """
    Q = sum(prod(b.event_shape) for b in query_slices.values())
    K = sum(prod(b.event_shape) for b in key_slices.values())
    mask_np = np.ones((Q, K), dtype=np.int8)

    # Apply cross independence
    for (blockA, blockB) in independence.get("cross", []):
        if blockA in query_slices and blockB in key_slices:
            q_off = query_slices[blockA].offset
            q_size = prod(query_slices[blockA].event_shape)
            k_off = key_slices[blockB].offset
            k_size = prod(key_slices[blockB].event_shape)
            mask_np[q_off:q_off+q_size, k_off:k_off+k_size] = 0

    # Apply cross_local independence
    for spec in independence.get("cross_local", []):
        blockA, blockB, idx_map = spec
        if blockA in query_slices and blockB in key_slices:
            q_off = query_slices[blockA].offset
            q_size = prod(query_slices[blockA].event_shape)
            q_shape = query_slices[blockA].event_shape

            k_off = key_slices[blockB].offset
            k_size = prod(key_slices[blockB].event_shape)
            k_shape = key_slices[blockB].event_shape

            # Zero out entire sub-block
            mask_np[q_off:q_off+q_size, k_off:k_off+k_size] = 0

            if idx_map is None:
                # Diagonal only
                if q_size != k_size:
                    raise ValueError(
                        "Cannot do cross_local diagonal if sizes differ"
                    )
                for i in range(q_size):
                    mask_np[q_off + i, k_off + i] = 1
            else:
                # Use index map
                dim_a, dim_b = idx_map
                if dim_a >= len(q_shape) or dim_b >= len(k_shape):
                    raise ValueError(
                        "Index map has invalid event shape dimensions"
                    )
                if q_shape[dim_a] != k_shape[dim_b]:
                    raise ValueError(
                        "Cannot do cross_local if event shapes do not "
                        "match"
                    )

                a_idx = _event_indices(q_shape, dim_a)
                b_idx = _event_indices(k_shape, dim_b)

                for (a_indices, b_indices) in zip(a_idx, b_idx):
                    for a in a_indices:
                        for b in b_indices:
                            mask_np[q_off+a, k_off+b] = 1

    return jnp.array(mask_np, dtype=jnp.float32)


def build_padding_mask(
    block_slices: Dict[str, SliceInfo],
    event_shapes: Dict[str, Array]
) -> Array:
    """
    Build padding mask for variable-length sequences.

    Creates a binary mask where 1 indicates valid tokens and 0 indicates
    padding.

    Parameters
    ----------
    block_slices : Dict[str, SliceInfo]
        Slice metadata for each block (with padded event_shape)
    event_shapes : Dict[str, Array]
        Actual (unpadded) event shapes. Arrays should have shape
        (*sample_dims, n_event_dims) where the last dimension contains
        the actual sizes along each event dimension

    Returns
    -------
    Array
        Binary mask with shape (*sample_dims, total_tokens) where
        sample_dims matches the leading dimensions of event_shapes
        values

    Examples
    --------
    >>> slices = {'x': {'offset': 0, 'event_shape': (5,), ...}}
    >>> event_shapes = {'x': jnp.array([3])}  # actual size is 3
    >>> mask = build_padding_mask(slices, event_shapes)
    >>> mask.shape
    (5,)
    >>> mask[:3]  # first 3 valid
    Array([1., 1., 1.], dtype=float32)
    >>> mask[3:]  # last 2 padding
    Array([0., 0.], dtype=float32)
    """
    def _build_block_mask(key: str, info: SliceInfo) -> Array:
        """Build mask for a single block."""
        block_size = prod(info.event_shape)
        actual_event_shape = event_shapes[key]

        # Build coordinate grid for padded shape
        ranges = [jnp.arange(r) for r in info.event_shape]
        coords = jnp.meshgrid(*ranges, indexing="ij")

        # Check validity along each dimension
        n_event_dims = len(info.event_shape)
        valid_in_dimension = [
            coord < jnp.expand_dims(
                actual_event_shape[..., i],
                tuple(range(-n_event_dims, 0))
            )
            for i, coord in enumerate(coords)
        ]

        # Token is valid if valid in all dimensions
        is_valid = jnp.all(jnp.stack(valid_in_dimension), axis=0)

        # Reshape to (*sample_dims, block_size)
        sample_shape = actual_event_shape.shape[:-1]
        is_valid_flat = is_valid.reshape(*sample_shape, block_size)

        return is_valid_flat.astype(jnp.float32)

    # Build mask for each block and concatenate
    block_masks = [
        _build_block_mask(key, info)
        for key, info in block_slices.items()
    ]

    return jnp.concatenate(block_masks, axis=-1)
