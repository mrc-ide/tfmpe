"""Unified parameter data interface.

Provides the Tokens class for consolidating structured parameter handling,
masking, functional inputs, and metadata into a single coherent interface.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from .flatten import flatten_pytree, update_flat_array
from .utils import Independence, SliceInfo
from .functional_inputs import flatten_functional_inputs
from .masks import build_cross_attention_mask, build_self_attention_mask
from .reconstruct import decode_pytree, decode_pytree_keys
from .token_view import TokenView


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
    padding_mask : Optional[Array]
        Padding mask, shape (*sample_shape, n_total_tokens)
    functional_inputs : Optional[Array]
        Functional inputs for tokens, shape (*sample_shape,
        n_total_tokens, max_batch_size)
    slices : Dict[str, SliceInfo]
        Metadata per key mapping name to SliceInfo
    label_map : Dict[str, int]
        Mapping from key names to label integers
    key_order : List[str]
        Ordered list of keys (matches slice order)
    independence : Independence
        Independence specification controlling attention patterns between
        tokens. See Independence class for details.
    """

    data: Array
    labels: Array
    self_attention_mask: Array
    padding_mask: Optional[Array]
    functional_inputs: Optional[Array]
    slices: Dict[str, SliceInfo]
    label_map: Dict[str, int]
    key_order: List[str]
    independence: Optional[Independence] = None

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
        independence: Independence,
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
        independence : Independence
            Independence specification controlling attention patterns between
            tokens. See Independence class for details.
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
        key_order = sorted(slices.keys(), key=lambda k: slices[k].offset)

        # Create label map and labels array
        label_map = {key: i for i, key in enumerate(key_order)}

        # Build labels array
        total_tokens = flat_data.shape[sample_ndims]
        sample_shape = flat_data.shape[:sample_ndims]

        # Create labels for each token
        labels_list = []
        for key in key_order:
            event_shape = slices[key].event_shape
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
                sample_ndims=sample_ndims
            )

        return cls(
            data=flat_data,
            labels=labels,
            self_attention_mask=self_attention_mask,
            padding_mask=None,
            functional_inputs=func_inputs_flat,
            slices=slices,
            label_map=label_map,
            key_order=key_order,
            independence=independence
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

    def select_tokens(self, keys: List[str]) -> TokenView:
        """
        Create view selecting specified keys.

        Returns a TokenView that provides slices, labels, and masks
        for only the selected keys without copying data.

        Parameters
        ----------
        keys : List[str]
            Keys to include in the view

        Returns
        -------
        TokenView
            View into this Tokens object with only selected keys

        Raises
        ------
        KeyError
            If any key in keys is not present in this Tokens object
        """
        return TokenView(parent=self, selected_keys=keys)

    def cross_attention_mask(
        self,
        query_view: TokenView,
        key_view: TokenView
    ) -> Array:
        """
        Generate cross-attention mask between query and key tokens.

        Uses independence specification to zero out prohibited
        connections.

        Parameters
        ----------
        query_view : TokenView
            TokenView for query tokens
        key_view : TokenView
            TokenView for key/value tokens

        Returns
        -------
        Array
            Cross-attention mask with shape
            (n_query_tokens, n_key_tokens)
        """
        # Use re-indexed slices from TokenViews
        independence = self.independence or Independence()
        return build_cross_attention_mask(
            query_view.slices,
            key_view.slices,
            independence
        )

    def with_values(self, **key_values: Array) -> 'Tokens':
        """
        Create new Tokens with specified keys replaced by new values.

        The new values are re-flattened and inserted at the correct
        offsets in the unified array.

        Parameters
        ----------
        **key_values : Array
            Keyword arguments mapping key names to new value arrays

        Returns
        -------
        Tokens
            New Tokens object with updated values

        Raises
        ------
        KeyError
            If any key is not present in this Tokens object
        ValueError
            If any new value has incompatible shape

        Examples
        --------
        >>> new_tokens = tokens.with_values(mu=new_mu_samples)
        >>> new_tokens = tokens.with_values(
        ...     mu=new_mu,
        ...     sigma=new_sigma
        ... )
        """
        # Validate all keys exist
        for key in key_values:
            if key not in self.key_order:
                raise KeyError(
                    f"Key '{key}' not found. "
                    f"Available keys: {self.key_order}"
                )

        # Start with current data
        new_data = self.data

        # Update each key
        for key, new_value in key_values.items():
            # Validate shape matches expected
            slice_info = self.slices[key]
            expected_shape = (
                self.sample_shape +
                slice_info.event_shape +
                slice_info.batch_shape
            )

            if new_value.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for key '{key}': "
                    f"expected {expected_shape}, got {new_value.shape}"
                )

            # Update the flat array
            new_data = update_flat_array(
                new_data,
                self.slices,
                key,
                new_value
            )

        # Create new Tokens with updated data
        return Tokens(
            data=new_data,
            labels=self.labels,
            self_attention_mask=self.self_attention_mask,
            padding_mask=self.padding_mask,
            functional_inputs=self.functional_inputs,
            slices=self.slices,
            label_map=self.label_map,
            key_order=self.key_order,
            independence=self.independence
        )
