"""Unified parameter data interface.

Provides the Tokens class for consolidating structured parameter handling,
masking, functional inputs, and metadata into a single coherent interface.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from .flatten import flatten_pytree
from .functional_inputs import flatten_functional_inputs
from .masks import build_self_attention_mask
from .reconstruct import decode_pytree, decode_pytree_keys

FUNCTIONAL_INPUT_PAD_VALUE = -1e8


@dataclass
class Tokens:
    """
    Unified container for structured token data.

    Stores all parameters and observations in a single flattened array,
    enabling dynamic slicing into subsets via key selection.

    Attributes
    ----------
    data : Array
        All flattened tokens, shape (*sample_shape, n_total_tokens,
        max_batch_size)
    labels : Array
        Integer labels per token, shape (*sample_shape, n_total_tokens)
    self_attention_mask : Array
        Self-attention mask for all tokens, shape
        (n_total_tokens, n_total_tokens)
    padding_mask : Array
        Padding mask, shape (*sample_shape, n_total_tokens)
    functional_inputs : Optional[Array]
        Functional inputs for tokens, shape (*sample_shape,
        n_total_tokens, max_batch_size)
    slices : Dict[str, Dict]
        Metadata per key: {name: {offset, event_shape, batch_shape}}
    label_map : Dict[str, int]
        Mapping from key names to label integers
    key_order : List[str]
        Ordered list of keys (matches slice order)
    """

    data: Array
    labels: Array
    self_attention_mask: Array
    padding_mask: Optional[Array]
    functional_inputs: Optional[Array]
    slices: Dict[str, Dict[str, Any]]
    label_map: Dict[str, int]
    key_order: List[str]

    @property
    def sample_shape(self) -> Tuple[int, ...]:
        """
        Get sample shape from data array.

        Returns
        -------
        Tuple[int, ...]
            Shape of sample dimensions
        """
        sample_ndims = len(self.data.shape) - 2  # Remove event and batch
        return self.data.shape[:sample_ndims]

    @classmethod
    def from_pytree(
        cls,
        data: Dict[str, Array],
        independence: Dict,
        functional_inputs: Optional[Dict[str, Array]] = None,
        sample_ndims: int = 0,
        batch_ndims: Optional[Dict[str, int]] = None,
    ) -> 'Tokens':
        """
        Create Tokens from structured PyTree.

        All keys in data are flattened into a single token array.

        Parameters
        ----------
        data : Dict[str, Array]
            Dictionary of parameter arrays. Each array should have shape
            (*sample_dims, *event_dims, *batch_dims).
        independence : Dict
            Independence specification with keys:
            - 'local': List of keys with local independence
            - 'cross': List of (keyA, keyB) tuples with cross
              independence
            - 'cross_local': List of (keyA, keyB, idx_map) tuples
        functional_inputs : Optional[Dict[str, Array]], optional
            Dictionary of functional input arrays matching data structure
        sample_ndims : int, optional
            Number of leading sample dimensions. Default is 0.
        batch_ndims : Optional[Dict[str, int]], optional
            Number of trailing batch dimensions for each key.
            If None, defaults to 1 for all keys.

        Returns
        -------
        Tokens
            Unified token object with all data and metadata
        """
        # Default batch_ndims to 1 for all keys
        if batch_ndims is None:
            batch_ndims = {key: 1 for key in data.keys()}

        # Flatten the PyTree
        flat_data, slices = flatten_pytree(
            data,
            sample_ndims=sample_ndims,
            batch_ndims=batch_ndims
        )

        # Create key order from slices (sorted by offset)
        key_order = sorted(slices.keys(), key=lambda k: slices[k]['offset'])

        # Create label map and labels array
        label_map = {key: i for i, key in enumerate(key_order)}

        # Build labels array
        total_tokens = flat_data.shape[sample_ndims]
        sample_shape = flat_data.shape[:sample_ndims]

        # Create labels for each token
        labels_list = []
        for key in key_order:
            event_shape = slices[key]['event_shape']
            n_tokens = 1
            for dim in event_shape:
                n_tokens *= dim
            key_labels = jnp.full(n_tokens, label_map[key], dtype=jnp.int32)
            labels_list.append(key_labels)

        labels_1d = jnp.concatenate(labels_list)

        # Broadcast to sample shape if needed
        if sample_ndims > 0:
            # Expand dims and broadcast
            for _ in range(sample_ndims):
                labels_1d = jnp.expand_dims(labels_1d, 0)
            broadcast_shape = sample_shape + (total_tokens,)
            labels = jnp.broadcast_to(labels_1d, broadcast_shape)
        else:
            labels = labels_1d

        # Build self-attention mask
        self_attention_mask = build_self_attention_mask(
            slices,
            independence
        )

        # Flatten functional inputs if provided
        func_inputs_flat = None
        if functional_inputs is not None:
            func_inputs_flat = flatten_functional_inputs(
                functional_inputs,
                slices,
                sample_ndims=sample_ndims,
                pad_value=FUNCTIONAL_INPUT_PAD_VALUE
            )

        return cls(
            data=flat_data,
            labels=labels,
            self_attention_mask=self_attention_mask,
            padding_mask=None,
            functional_inputs=func_inputs_flat,
            slices=slices,
            label_map=label_map,
            key_order=key_order
        )

    def decode(self, flat_array: Optional[Array] = None) -> Dict[str, Array]:
        """
        Reconstruct PyTree from flat array.

        If flat_array is None, uses self.data.

        Parameters
        ----------
        flat_array : Optional[Array], optional
            Flat array to decode. If None, uses self.data.

        Returns
        -------
        Dict[str, Array]
            Dictionary of reconstructed arrays
        """
        if flat_array is None:
            flat_array = self.data

        return decode_pytree(
            flat_array,
            self.slices,
            self.sample_shape,
            is_subset=False
        )

    def decode_keys(
        self,
        flat_array: Array,
        keys: List[str]
    ) -> Dict[str, Array]:
        """
        Reconstruct only specified keys from flat array.

        Useful when target_keys is a subset of all keys.

        Parameters
        ----------
        flat_array : Array
            Flat array to decode
        keys : List[str]
            List of keys to reconstruct

        Returns
        -------
        Dict[str, Array]
            Dictionary containing only the specified keys
        """
        return decode_pytree_keys(
            flat_array,
            self.slices,
            self.sample_shape,
            keys
        )
