"""Tests for transformer encoder components (MLP, EncoderBlock, DecoderBlock)."""

import pytest
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

from tfmpe.nn.transformer.config import TransformerConfig
from tfmpe.nn.transformer.encoder import MLP, EncoderBlock, DecoderBlock


class TestMLP:
    """Tests for MLP feedforward network."""

    @pytest.mark.parametrize("n_ff", [1, 2, 4, 8])
    def test_output_shape_preservation(self, n_ff: int) -> None:
        """Test that MLP preserves input shape.

        Parameters
        ----------
        n_ff : int
            Number of feedforward layers
        """
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=128,
            n_encoder=4,
            n_decoder=4,
            n_heads=8,
            n_ff=n_ff,
            label_dim=32,
            index_out_dim=64,
            dropout=0.1,
        )

        mlp = MLP(config=config, rngs=rngs)

        batch_size, n_tokens = 4, 10
        inputs = jnp.ones((batch_size, n_tokens, config.latent_dim))
        output = mlp(inputs)

        expected_shape = (batch_size, n_tokens, config.latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """Test MLP with different batch sizes.

        Parameters
        ----------
        batch_size : int
            Number of samples in batch
        """
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_decoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        mlp = MLP(config=config, rngs=rngs)

        n_tokens = 20
        inputs = jnp.ones((batch_size, n_tokens, config.latent_dim))
        output = mlp(inputs)

        expected_shape = (batch_size, n_tokens, config.latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_train_eval_dropout_difference(self) -> None:
        """Test that dropout behaves differently in train vs eval mode."""
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=128,
            n_encoder=4,
            n_decoder=4,
            n_heads=8,
            n_ff=2,
            label_dim=32,
            index_out_dim=64,
            dropout=0.5,
        )

        mlp = MLP(config=config, rngs=rngs)

        batch_size, n_tokens = 4, 10
        inputs = jnp.ones((batch_size, n_tokens, config.latent_dim))

        # Set to eval mode
        mlp.eval()
        eval_output = mlp(inputs)

        # In eval mode, output should be deterministic
        eval_output_2 = mlp(inputs)
        assert jnp.allclose(eval_output, eval_output_2), (
            "Eval mode should produce deterministic outputs"
        )

    def test_with_multidimensional_batch(self) -> None:
        """Test MLP with multiple sample dimensions."""
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_decoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        mlp = MLP(config=config, rngs=rngs)

        # Shape: (samples, batch, n_tokens, latent_dim)
        inputs = jnp.ones((2, 3, 10, config.latent_dim))
        output = mlp(inputs)

        expected_shape = (2, 3, 10, config.latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )


class TestEncoderBlock:
    """Tests for EncoderBlock self-attention transformer block."""

    @pytest.mark.parametrize(
        "sample_shape",
        [
            (4, 10),  # (batch, n_tokens)
            (2, 3, 10),  # (samples, batch, n_tokens)
        ],
    )
    def test_output_shape_with_mask(
        self, sample_shape: tuple[int, ...]
    ) -> None:
        """Test EncoderBlock output shape and mask application.

        Verifies that EncoderBlock preserves input shape through
        self-attention and feedforward layers, and that mask parameter
        is properly applied.

        Parameters
        ----------
        sample_shape : tuple
            Shape of input excluding latent_dim dimension
        """
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_decoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        encoder_block = EncoderBlock(config=config, rngs=rngs)

        inputs = jnp.ones((*sample_shape, config.latent_dim))

        # Get n_tokens from sample_shape
        n_tokens = sample_shape[-1]

        # Create a causal mask (lower triangular)
        mask = jnp.tril(jnp.ones((n_tokens, n_tokens)))

        # Test with mask
        output_masked = encoder_block(inputs, mask=mask)

        # Test without mask
        encoder_block2 = EncoderBlock(config=config, rngs=rngs)
        output_unmasked = encoder_block2(inputs, mask=None)

        # Both should have correct shape
        expected_shape = (*sample_shape, config.latent_dim)
        assert output_masked.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output_masked.shape}"
        )
        assert output_unmasked.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output_unmasked.shape}"
        )


class TestDecoderBlock:
    """Tests for DecoderBlock cross-attention transformer block."""

    @pytest.mark.parametrize(
        "sample_shape,context_shape",
        [
            ((4, 10), (4, 20)),  # (batch, n_q) vs (batch, n_context)
            ((2, 3, 10), (2, 3, 20)),  # (samples, batch, n_q) vs context
        ],
    )
    def test_output_shape_with_cross_attention(
        self,
        sample_shape: tuple[int, ...],
        context_shape: tuple[int, ...],
    ) -> None:
        """Test DecoderBlock output shape with cross-attention.

        Verifies that DecoderBlock preserves query shape through
        cross-attention and feedforward layers, even when context
        has different sequence length.

        Parameters
        ----------
        sample_shape : tuple
            Shape of query input excluding latent_dim dimension
        context_shape : tuple
            Shape of context input excluding latent_dim dimension
        """
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_decoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        decoder_block = DecoderBlock(config=config, rngs=rngs)

        # Query and context inputs
        query = jnp.ones((*sample_shape, config.latent_dim))
        context = jnp.ones((*context_shape, config.latent_dim))

        # Get n_tokens from shapes
        n_q = sample_shape[-1]
        n_context = context_shape[-1]

        # Create a causal mask for queries (lower triangular)
        mask = jnp.tril(jnp.ones((n_q, n_context)))

        # Test with mask
        output_masked = decoder_block(
            query, context=context, mask=mask
        )

        # Test without mask
        decoder_block2 = DecoderBlock(config=config, rngs=rngs)
        output_unmasked = decoder_block2(
            query, context=context, mask=None
        )

        # Both should have shape matching query shape
        expected_shape = (*sample_shape, config.latent_dim)
        assert output_masked.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output_masked.shape}"
        )
        assert output_unmasked.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output_unmasked.shape}"
        )
