"""TokenView class for efficient subset views into Tokens."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional

import jax.numpy as jnp
from jaxtyping import Array

from .reconstruct import decode_pytree_keys
from .masks import build_cross_attention_mask
from .utils import SliceInfo

if TYPE_CHECKING:
    from .tokens import Tokens


@dataclass
class TokenView:
    """
    Efficient view into Tokens selecting a subset of keys.

    Does not copy data, only computes indices for selected keys.
    Properties are lazily evaluated and cached using functools.

    Attributes
    ----------
    parent : 'Tokens'
        Parent Tokens object
    selected_keys : List[str]
        Keys to include in this view
    """

    parent: Tokens
    selected_keys: List[str]

    def __post_init__(self):
        """Validate that selected keys exist in parent."""
        for key in self.selected_keys:
            if key not in self.parent.key_order:
                raise KeyError(
                    f"Key '{key}' not found. "
                    f"Available keys: {self.parent.key_order}"
                )

    @cached_property
    def _token_indices(self) -> Array:
        """
        Compute flat indices for selected keys' tokens.

        Returns
        -------
        Array
            1D array of token indices to extract from parent
        """
        # Get ordered keys (respecting parent's key_order)
        ordered_selected = [
            k for k in self.parent.key_order if k in self.selected_keys
        ]

        # Collect indices for each selected key
        indices_list = []
        for key in ordered_selected:
            slice_info = self.parent.slices[key]
            offset = slice_info.offset
            event_shape = slice_info.event_shape

            # Compute number of tokens for this key
            n_tokens = 1
            for dim in event_shape:
                n_tokens *= dim

            # Add indices for this key's tokens
            key_indices = jnp.arange(offset, offset + n_tokens)
            indices_list.append(key_indices)

        return jnp.concatenate(indices_list)

    @cached_property
    def data(self) -> Array:
        """
        Get data for selected keys.

        Returns
        -------
        Array
            Data array containing only selected keys' tokens
        """
        indices = self._token_indices
        sample_ndims = len(self.parent.sample_shape)

        if sample_ndims > 0:
            # Data has shape (*sample, n_tokens, max_batch)
            return self.parent.data[..., indices, :]
        else:
            return self.parent.data[indices, :]

    @cached_property
    def labels(self) -> Array:
        """
        Get labels for selected keys.

        Returns
        -------
        Array
            Label array containing only selected keys' labels
        """
        indices = self._token_indices
        sample_ndims = len(self.parent.sample_shape)

        if sample_ndims > 0:
            return self.parent.labels[..., indices]
        else:
            return self.parent.labels[indices]

    @cached_property
    def self_attention_mask(self) -> Array:
        """
        Get self-attention mask for selected keys.

        Returns
        -------
        Array
            Self-attention mask for selected tokens only
        """
        # Extract submatrix from parent's mask
        indices = self._token_indices
        full_mask = self.parent.self_attention_mask

        # Use jnp.ix_ to extract submatrix
        return full_mask[jnp.ix_(indices, indices)]

    @cached_property
    def slices(self) -> Dict[str, SliceInfo]:
        """
        Get slice metadata for selected keys (re-indexed to 0).

        Returns
        -------
        Dict[str, SliceInfo]
            Slice metadata with offsets re-indexed to start at 0
        """
        reindexed = {}
        current_offset = 0
        for key in self.selected_keys:
            slice_info = self.parent.slices[key]
            n_tokens = 1
            for dim in slice_info.event_shape:
                n_tokens *= dim
            reindexed[key] = slice_info._replace(offset=current_offset)
            current_offset += n_tokens
        return reindexed

    @property
    def padding_mask(self) -> Optional[Array]:
        """
        Get padding mask for selected keys.

        Returns
        -------
        Optional[Array]
            Padding mask for selected tokens, or None if not present
        """
        if self.parent.padding_mask is None:
            return None

        indices = self._token_indices
        sample_ndims = len(self.parent.sample_shape)

        if sample_ndims > 0:
            return self.parent.padding_mask[..., indices]
        else:
            return self.parent.padding_mask[indices]

    @property
    def functional_inputs(self) -> Optional[Array]:
        """
        Get functional inputs for selected keys.

        Returns
        -------
        Optional[Array]
            Functional inputs for selected tokens, or None
        """
        if self.parent.functional_inputs is None:
            return None

        indices = self._token_indices
        sample_ndims = len(self.parent.sample_shape)

        if sample_ndims > 0:
            return self.parent.functional_inputs[..., indices, :]
        else:
            return self.parent.functional_inputs[indices, :]

    @property
    def sample_shape(self) -> tuple:
        """
        Get sample shape from parent Tokens.

        Returns
        -------
        tuple
            Sample shape tuple from parent
        """
        return self.parent.sample_shape

    def cross_attention_mask(
        self,
        key_view: 'TokenView',
    ) -> Array:
        """
        Get cross-attention mask from this view (query) to key view.

        Parameters
        ----------
        key_view : TokenView
            Key view for cross-attention

        Returns
        -------
        Array
            Cross-attention mask for this query view attending to
            key view
        """
        independence = (
            self.parent.independence
            if self.parent.independence is not None
            else {}
        )
        return build_cross_attention_mask(
            self.slices,
            key_view.slices,
            independence,
        )

    def decode(self) -> Dict[str, Array]:
        """
        Decode selected keys back to PyTree.

        Returns
        -------
        Dict[str, Array]
            Dictionary containing only selected keys
        """
        return decode_pytree_keys(
            self.parent.data,
            self.parent.slices,
            self.parent.sample_shape,
            self.selected_keys
        )
