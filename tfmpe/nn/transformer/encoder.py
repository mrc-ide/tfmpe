"""Encoder and decoder blocks for transformer architecture."""

from typing import Callable, Optional

from jaxtyping import Array
from flax import nnx

from .config import TransformerConfig


class FFLayer(nnx.Module):
    """Single feedforward layer with linear, dropout, and activation.

    Attributes
    ----------
    linear : nnx.Linear
        Linear transformation layer
    dropout : nnx.Dropout
        Dropout regularization
    activation : Callable
        Activation function (e.g., nnx.relu)
    """

    activation: Callable

    def __init__(
        self,
        dim: int,
        dropout: float,
        activation: Callable,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize feedforward layer.

        Parameters
        ----------
        dim : int
            Feature dimension (input and output)
        dropout : float
            Dropout rate
        activation : Callable
            Activation function
        rngs : nnx.Rngs
            Random number generator state
        """
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        """Apply feedforward transformation.

        Parameters
        ----------
        x : Array
            Input array of shape (..., dim)

        Returns
        -------
        Array
            Output array of shape (..., dim)
        """
        x = self.linear(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class MLP(nnx.Module):
    """Multi-layer feedforward network.

    Applies sequential feedforward layers with dropout and activation
    functions. Maintains input shape through the network.

    Attributes
    ----------
    n_layers : int
        Number of feedforward layers
    layers : nnx.Module
        Vmapped FFLayer modules
    """

    n_layers: int

    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize MLP network.

        Parameters
        ----------
        config : TransformerConfig
            Configuration containing latent_dim, n_ff, dropout, activation
        rngs : nnx.Rngs
            Random number generator state
        """
        n_ff = config.n_ff
        dim = config.latent_dim
        dropout = config.dropout
        activation = config.activation

        @nnx.split_rngs(splits=n_ff)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs) -> FFLayer:
            """Create a single feedforward layer."""
            return FFLayer(dim, dropout, activation, rngs=rngs)

        self.layers = create_layer(rngs)
        self.n_layers = n_ff

    def __call__(self, x: Array) -> Array:
        """Apply MLP transformation.

        Sequentially applies each feedforward layer, preserving input
        shape.

        Parameters
        ----------
        x : Array
            Input array of shape (..., latent_dim)

        Returns
        -------
        Array
            Output array of shape (..., latent_dim)
        """

        @nnx.split_rngs(splits=self.n_layers)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(
            x: Array,
            model: FFLayer,
        ) -> Array:
            """Apply a single layer and return updated state."""
            x = model(x)
            return x

        return forward(x, self.layers)
