"""Main transformer model for TFMPE."""

from typing import Optional, Union

import jax.numpy as jnp
from jaxtyping import Array
from flax import nnx

from ...preprocessing import Tokens
from ...preprocessing.token_view import TokenView
from .config import TransformerConfig
from .embedding import Embedding
from .encoder import EncoderBlock, DecoderBlock


class Transformer(nnx.Module):
    """Encoder-decoder transformer for TFMPE.

    Processes context and parameter tokens through separate encoder
    blocks with shared attention, then decoder blocks with
    cross-attention.

    Attributes
    ----------
    config : TransformerConfig
        Configuration for transformer architecture
    embedding : Embedding
        Embedding layer for token data
    encoder_blocks : nnx.Module
        Vmapped encoder blocks
    decoder_blocks : nnx.Module
        Vmapped decoder blocks
    output_linear : nnx.Linear
        Linear layer projecting from latent_dim to value_dim
    """

    config: TransformerConfig
    value_dim: int

    def __init__(
        self,
        config: TransformerConfig,
        tokens: Tokens,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize transformer.

        Deduces value_dim, n_labels, and functional_inputs_dim from
        tokens.

        Parameters
        ----------
        config : TransformerConfig
            Configuration containing latent_dim, n_encoder, n_decoder,
            n_heads, n_ff, label_dim, index_out_dim, dropout,
            activation
        tokens : Tokens
            Full Tokens object containing all data
        rngs : nnx.Rngs
            JAX random number generators for parameter
            initialization
        """
        self.config = config
        self.value_dim = tokens.data.shape[-1]

        # Deduce n_labels from tokens' key_order length
        n_labels = len(tokens.key_order)
        functional_inputs_dim = (
            tokens.functional_inputs.shape[-1]
            if tokens.functional_inputs is not None
            else 0
        )

        # Create embedding layer
        self.embedding = Embedding(
            value_dim=self.value_dim,
            n_labels=n_labels,
            label_dim=config.label_dim,
            latent_dim=config.latent_dim,
            rngs=rngs,
            functional_inputs_dim=functional_inputs_dim,
        )

        # Create encoder blocks via vmap
        @nnx.split_rngs(splits=config.n_encoder)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_encoder_block(rngs: nnx.Rngs) -> EncoderBlock:
            """Create a single encoder block."""
            return EncoderBlock(config=config, rngs=rngs)

        self.encoder_blocks = create_encoder_block(rngs)

        # Create decoder blocks via vmap
        @nnx.split_rngs(splits=config.n_decoder)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_decoder_block(rngs: nnx.Rngs) -> DecoderBlock:
            """Create a single decoder block."""
            return DecoderBlock(config=config, rngs=rngs)

        self.decoder_blocks = create_decoder_block(rngs)

        # Create output linear layer
        self.output_linear = nnx.Linear(
            config.latent_dim,
            self.value_dim,
            rngs=rngs,
        )

    def encode(
        self,
        tokens: Union[Tokens, TokenView],
        time: Array,
        deterministic: bool = False,
    ) -> Array:
        """Encode tokens through encoder blocks.

        Parameters
        ----------
        tokens : TokenView
            Input token view to encode
        time : Array
            Time values, shape (*sample_shape,) or (*sample_shape, 1)
        deterministic : bool, optional
            If True, disable dropout. Default is False.

        Returns
        -------
        Array
            Encoded tokens, shape (*sample_shape, n_tokens,
            latent_dim)
        """
        # Embed tokens
        x = self.embedding(tokens)

        # Apply encoder blocks sequentially via scan
        @nnx.split_rngs(splits=self.config.n_encoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(
            x: Array,
            encoder_block: EncoderBlock,
        ) -> Array:
            """Apply a single encoder block and return updated state."""
            x = encoder_block(
                x,
                mask=tokens.self_attention_mask,
                deterministic=deterministic,
            )
            return x

        x = forward(x, self.encoder_blocks)

        return x

    def decode(
        self,
        tokens: Union[Tokens, TokenView],
        encoded_context: Array,
        time: Array,
        cross_attention_mask: Optional[Array] = None,
        deterministic: bool = False,
    ) -> Array:
        """Decode tokens with cross-attention to context.

        Parameters
        ----------
        tokens : Union[Tokens, TokenView]
            Parameter token view to decode
        encoded_context : Array
            Encoded context, shape (*sample_shape, n_context,
            latent_dim)
        time : Array
            Time values, shape (*sample_shape,) or (*sample_shape, 1)
        cross_attention_mask : Optional[Array]
            Cross-attention mask between param and context tokens,
            shape (*sample_shape, n_param, n_context). Default is None.
        deterministic : bool, optional
            If True, disable dropout. Default is False.

        Returns
        -------
        Array
            Output vectors, shape (*sample_shape, n_tokens, value_dim)
        """
        # Embed tokens
        x = self.embedding(tokens)

        # Apply encoder blocks to parameter tokens
        @nnx.split_rngs(splits=self.config.n_encoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def encoder_forward(
            x: Array,
            encoder_block: EncoderBlock,
        ) -> Array:
            """Apply encoder block with self-attention."""
            x = encoder_block(
                x,
                mask=tokens.self_attention_mask,
                deterministic=deterministic,
            )
            return x

        x = encoder_forward(x, self.encoder_blocks)

        # Apply decoder blocks with cross-attention to context
        @nnx.split_rngs(splits=self.config.n_decoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def decoder_forward(
            x: Array,
            decoder_block: DecoderBlock,
        ) -> Array:
            """Apply decoder block with cross-attention."""
            x = decoder_block(
                x,
                context=encoded_context,
                mask=cross_attention_mask,
                deterministic=deterministic,
            )
            return x

        x = decoder_forward(x, self.decoder_blocks)

        # Project to output dimension
        output = self.output_linear(x)

        return output

    def __call__(
        self,
        context: TokenView,
        params: TokenView,
        time: Array,
        deterministic: bool = False,
    ) -> Array:
        """Forward pass through transformer.

        Parameters
        ----------
        context : TokenView
            Context token view to encode
        params : TokenView
            Parameter token view to decode
        time : Array
            Time values, shape (*sample_shape,) or (*sample_shape, 1)
        deterministic : bool, optional
            If True, disable dropout. Default is False.

        Returns
        -------
        Array
            Output vector field, shape matching params.data

        Notes
        -----
        Supports two use cases:
        1. Same sample_shape: context and params have identical
           sample_shape, processed with shared batch dimensions.
        2. Broadcast from (1,): context.sample_shape == (1,) while
           params.sample_shape is arbitrary. Encoded context is
           broadcast to match params sample shape.
        """
        context_shape = context.sample_shape
        param_shape = params.sample_shape

        # Check for valid use cases
        if context_shape != param_shape:
            # Allow broadcasting when context has sample_shape of (1,)
            if context_shape != (1,):
                raise ValueError(
                    f"context and params must have same sample_shape "
                    f"or context must have sample_shape (1,). "
                    f"Got context: {context_shape}, "
                    f"param: {param_shape}"
                )

        # Encode context
        encoded_context = self.encode(
            tokens=context,
            time=time,
            deterministic=deterministic,
        )

        # Broadcast encoded_context if needed
        if context_shape == (1,) and context_shape != param_shape:
            # Remove the (1,) dimension and broadcast
            encoded_context = jnp.squeeze(encoded_context, axis=0)
            # Broadcast to param_shape
            broadcast_shape = param_shape + encoded_context.shape[-2:]
            encoded_context = jnp.broadcast_to(
                encoded_context,
                broadcast_shape,
            )

        # Compute cross-attention mask using TokenView method
        cross_mask = params.cross_attention_mask(context)

        # Decode parameters with cross-attention to context
        output = self.decode(
            tokens=params,
            encoded_context=encoded_context,
            time=time,
            cross_attention_mask=cross_mask,
            deterministic=deterministic,
        )

        return output
