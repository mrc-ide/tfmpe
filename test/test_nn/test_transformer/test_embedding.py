"""Tests for transformer embedding components."""

import pytest
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

from tfmpe.nn.transformer.embedding import (
    GaussianFourierEmbedding,
    Embedding,
)
from tfmpe.preprocessing.tokens import Tokens


class TestGaussianFourierEmbedding:
    """Tests for GaussianFourierEmbedding layer."""

    @pytest.mark.parametrize("in_dim,out_dim", [
        (1, 2),
        (1, 64),
        (2, 4),
        (3, 64),
        (10, 128),
    ])
    def test_output_shape(self, in_dim: int, out_dim: int) -> None:
        """Test that output shape matches expected dimensions.

        Parameters
        ----------
        in_dim : int
            Input feature dimension
        out_dim : int
            Output feature dimension
        """
        rngs = random.PRNGKey(0)
        embedding = GaussianFourierEmbedding(in_dim, out_dim, rngs)

        # Test with simple 1D input
        inputs = jnp.ones((in_dim,))
        output = embedding(inputs)
        assert output.shape == (out_dim,), (
            f"Expected shape {(out_dim,)}, got {output.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_batch_shape(self, batch_size: int) -> None:
        """Test output shape with batch dimension.

        Parameters
        ----------
        batch_size : int
            Number of samples in batch
        """
        in_dim, out_dim = 3, 64
        rngs = random.PRNGKey(0)
        embedding = GaussianFourierEmbedding(in_dim, out_dim, rngs)

        inputs = jnp.ones((batch_size, in_dim))
        output = embedding(inputs)
        assert output.shape == (batch_size, out_dim), (
            f"Expected shape {(batch_size, out_dim)}, "
            f"got {output.shape}"
        )

    @pytest.mark.parametrize(
        "sample_dims,n_tokens,in_dim,out_dim",
        [
            ((2, 3), 10, 2, 64),
            ((4,), 20, 3, 128),
            ((2, 3, 4), 5, 1, 32),
        ]
    )
    def test_multidimensional_batch(
        self,
        sample_dims: tuple,
        n_tokens: int,
        in_dim: int,
        out_dim: int,
    ) -> None:
        """Test output shape with multiple sample dimensions.

        Parameters
        ----------
        sample_dims : tuple
            Sample dimensions before tokens
        n_tokens : int
            Number of tokens
        in_dim : int
            Input feature dimension
        out_dim : int
            Output feature dimension
        """
        rngs = random.PRNGKey(0)
        embedding = GaussianFourierEmbedding(in_dim, out_dim, rngs)

        shape = sample_dims + (n_tokens, in_dim)
        inputs = jnp.ones(shape)
        output = embedding(inputs)

        expected_shape = sample_dims + (n_tokens, out_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )


class TestEmbedding:
    """Tests for Embedding layer."""

    @pytest.fixture
    def tokens_without_functional_inputs(self) -> Tokens:
        """Create tokens without functional inputs."""
        batch_size, n_tokens, value_dim = 4, 10, 2
        data = jnp.ones((batch_size, n_tokens, value_dim))
        labels = jnp.zeros((batch_size, n_tokens), dtype=jnp.int32)

        return Tokens(
            data=data,
            labels=labels,
            self_attention_mask=jnp.eye(n_tokens),
            padding_mask=None,
            functional_inputs=None,
            slices={},
            label_map={},
            key_order=[],
        )

    @pytest.fixture
    def tokens_with_functional_inputs(self) -> Tokens:
        """Create tokens with functional inputs."""
        batch_size, n_tokens, value_dim = 4, 10, 2
        functional_inputs_dim = 2
        data = jnp.ones((batch_size, n_tokens, value_dim))
        labels = jnp.zeros((batch_size, n_tokens), dtype=jnp.int32)
        functional_inputs = jnp.ones(
            (batch_size, n_tokens, functional_inputs_dim)
        )

        return Tokens(
            data=data,
            labels=labels,
            self_attention_mask=jnp.eye(n_tokens),
            padding_mask=None,
            functional_inputs=functional_inputs,
            slices={},
            label_map={},
            key_order=[],
        )

    def test_output_shape_without_functional_inputs(
        self,
        tokens_without_functional_inputs: Tokens,
    ) -> None:
        """Test output shape without functional inputs.

        Parameters
        ----------
        tokens_without_functional_inputs : Tokens
            Tokens fixture without functional inputs
        """
        rngs = nnx.Rngs(0)
        latent_dim = 128
        value_dim = 2
        n_labels = 5
        label_dim = 32

        embedding = Embedding(
            value_dim=value_dim,
            n_labels=n_labels,
            label_dim=label_dim,
            latent_dim=latent_dim,
            rngs=rngs,
            functional_inputs_dim=0,
        )

        batch_size, n_tokens = 4, 10
        output = embedding(tokens_without_functional_inputs)

        expected_shape = (batch_size, n_tokens, latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_output_shape_with_functional_inputs(
        self,
        tokens_with_functional_inputs: Tokens,
    ) -> None:
        """Test output shape with functional inputs.

        Parameters
        ----------
        tokens_with_functional_inputs : Tokens
            Tokens fixture with functional inputs
        """
        rngs = nnx.Rngs(0)
        latent_dim = 128
        value_dim = 2
        n_labels = 5
        label_dim = 32
        functional_inputs_dim = 2

        embedding = Embedding(
            value_dim=value_dim,
            n_labels=n_labels,
            label_dim=label_dim,
            latent_dim=latent_dim,
            rngs=rngs,
            functional_inputs_dim=functional_inputs_dim,
        )

        batch_size, n_tokens = 4, 10
        output = embedding(tokens_with_functional_inputs)

        expected_shape = (batch_size, n_tokens, latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    @pytest.mark.parametrize("sample_shape", [(2,), (2, 3), (4,)])
    def test_sample_dimensions(
        self,
        tokens_without_functional_inputs: Tokens,
        sample_shape: tuple,
    ) -> None:
        """Test with sample dimensions.

        Parameters
        ----------
        tokens_without_functional_inputs : Tokens
            Base tokens fixture
        sample_shape : tuple
            Sample dimensions
        """
        rngs = nnx.Rngs(0)
        latent_dim = 128
        value_dim = 2
        n_labels = 5
        label_dim = 32

        embedding = Embedding(
            value_dim=value_dim,
            n_labels=n_labels,
            label_dim=label_dim,
            latent_dim=latent_dim,
            rngs=rngs,
            functional_inputs_dim=0,
        )

        # Broadcast tokens to desired sample shape
        def broadcast_to_shape(x: jnp.ndarray) -> jnp.ndarray:
            """Broadcast array to sample shape."""
            # Slice first element and broadcast to sample shape
            first = x[0]
            target_shape = sample_shape + first.shape
            return jnp.broadcast_to(first, target_shape)

        tokens = jax.tree.map(
            broadcast_to_shape,
            tokens_without_functional_inputs,
        )

        n_tokens = tokens_without_functional_inputs.data.shape[1]
        output = embedding(tokens)

        expected_shape = sample_shape + (n_tokens, latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
