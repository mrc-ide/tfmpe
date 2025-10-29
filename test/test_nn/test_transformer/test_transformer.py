"""Tests for Transformer model end-to-end functionality."""

import pytest
import jax.numpy as jnp
from flax import nnx

from tfmpe.preprocessing import Tokens
from tfmpe.preprocessing.token_view import TokenView
from tfmpe.nn.transformer.config import TransformerConfig


@pytest.fixture
def simple_pytree():
    """Simple 2-level hierarchical structure."""
    return {
        'mu': jnp.array([[1.0]]),  # (1, 1)
        'theta': jnp.array([[2.0], [3.0]]),  # (2, 1)
        'obs': jnp.array([[0.1], [0.2]]),  # (2, 1)
    }


@pytest.fixture
def simple_independence():
    """Independence spec for simple hierarchical structure."""
    return {
        'local': ['obs', 'theta'],
        'cross': [('mu', 'obs'), ('obs', 'mu')],
        'cross_local': [('theta', 'obs', None)],
    }


@pytest.fixture
def tokens_obj(
    simple_pytree: dict,
    simple_independence: dict,
) -> Tokens:
    """Create tokens from simple structure."""
    return Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1},
    )


@pytest.fixture
def context_view(tokens_obj: Tokens) -> TokenView:
    """Create context token view."""
    return tokens_obj.select_tokens(keys=['mu', 'obs'])


@pytest.fixture
def param_view(tokens_obj: Tokens) -> TokenView:
    """Create parameter token view."""
    return tokens_obj.select_tokens(keys=['theta'])


class TestTransformerInit:
    """Tests for Transformer initialization."""

    def test_init_deduces_params_from_tokens(
        self,
        tokens_obj: Tokens,
    ) -> None:
        """Test that Transformer.__init__ deduces all params from
        Tokens."""
        from tfmpe.nn.transformer import Transformer

        config = TransformerConfig(
            latent_dim=128,
            n_encoder=2,
            n_decoder=2,
            n_heads=8,
            n_ff=2,
            label_dim=32,
            index_out_dim=64,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)

        # Initialize transformer - deduce value_dim, n_labels,
        # functional_inputs_dim from tokens
        transformer = Transformer(
            config=config,
            tokens=tokens_obj,
            rngs=rngs,
        )

        assert transformer is not None

    def test_init_accepts_config_dataclass(
        self,
        tokens_obj: Tokens,
    ) -> None:
        """Test that Transformer.__init__ accepts config dataclass."""
        from tfmpe.nn.transformer import Transformer

        config = TransformerConfig(
            latent_dim=128,
            n_encoder=2,
            n_decoder=2,
            n_heads=8,
            n_ff=2,
            label_dim=32,
            index_out_dim=64,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)

        # Should accept config as primary parameter
        transformer = Transformer(
            config=config,
            tokens=tokens_obj,
            rngs=rngs,
        )

        assert transformer is not None


class TestTransformerForwardPass:
    """Tests for Transformer forward pass."""

    def test_forward_pass_output_shape(
        self,
        tokens_obj: Tokens,
        context_view: TokenView,
        param_view: TokenView,
    ) -> None:
        """Test forward pass output shape matches input param
        shape."""
        from tfmpe.nn.transformer import Transformer

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

        rngs = nnx.Rngs(0)

        transformer = Transformer(
            config=config,
            tokens=tokens_obj,
            rngs=rngs,
        )

        # Time is a scalar for now
        time = jnp.array(0.5)

        # Forward pass using __call__
        output = transformer(
            context=context_view,
            params=param_view,
            time=time,
        )

        # Expected shape from simple_pytree: theta is (2, 1)
        expected_shape = (2, 1)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    @pytest.mark.parametrize(
        "n_param_tokens,latent_dim",
        [
            (2, 64),
            (5, 128),
        ],
    )
    def test_forward_pass_various_sizes(
        self,
        n_param_tokens: int,
        latent_dim: int,
    ) -> None:
        """Test forward pass with various token and latent
        dimensions."""
        from tfmpe.nn.transformer import Transformer

        # Create simple pytree for testing
        pytree = {
            'context': jnp.ones((2, 1)),
            'param': jnp.ones((n_param_tokens, 1)),
        }
        independence = {
            'local': ['param'],
            'cross': [('context', 'param'), ('param', 'context')],
            'cross_local': [],
        }

        tokens = Tokens.from_pytree(
            pytree,
            independence=independence,
            sample_ndims=0,
            batch_ndims={'context': 1, 'param': 1},
        )

        config = TransformerConfig(
            latent_dim=latent_dim,
            n_encoder=2,
            n_decoder=2,
            n_heads=min(4, latent_dim // 16),
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config=config,
            tokens=tokens,
            rngs=rngs,
        )

        time = jnp.array(0.5)
        context_view = tokens.select_tokens(keys=['context'])
        param_view = tokens.select_tokens(keys=['param'])
        output = transformer(
            context=context_view,
            params=param_view,
            time=time,
        )

        # Expected shape: param is (n_param_tokens, 1)
        expected_shape = (n_param_tokens, 1)
        assert output.shape == expected_shape


class TestTransformerEncode:
    """Tests for Transformer.encode method."""

    def test_encode_method_output_shape(
        self,
        tokens_obj: Tokens,
        context_view: TokenView,
    ) -> None:
        """Test encode method returns correct shape."""
        from tfmpe.nn.transformer import Transformer

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

        rngs = nnx.Rngs(0)

        transformer = Transformer(
            config=config,
            tokens=tokens_obj,
            rngs=rngs,
        )

        time = jnp.array(0.5)

        # Encode context tokens
        encoded = transformer.encode(
            tokens=context_view,
            time=time,
        )

        # Expected shape: (3 tokens total for mu + obs, latent_dim)
        expected_shape = (3, config.latent_dim)
        assert encoded.shape == expected_shape


class TestTransformerDecode:
    """Tests for Transformer.decode method."""

    def test_decode_method_output_shape(
        self,
        tokens_obj: Tokens,
        context_view: TokenView,
        param_view: TokenView,
    ) -> None:
        """Test decode method output shape."""
        from tfmpe.nn.transformer import Transformer

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

        rngs = nnx.Rngs(0)

        transformer = Transformer(
            config=config,
            tokens=tokens_obj,
            rngs=rngs,
        )

        time = jnp.array(0.5)

        # Encode context
        encoded_context = transformer.encode(
            tokens=context_view,
            time=time,
        )

        # Decode parameters
        cross_mask = param_view.cross_attention_mask(context_view)
        output = transformer.decode(
            tokens=param_view,
            encoded_context=encoded_context,
            time=time,
            cross_attention_mask=cross_mask,
        )

        # Expected shape: theta is (2, 1)
        expected_shape = (2, 1)
        assert output.shape == expected_shape


class TestTransformerSampleDimensions:
    """Tests for Transformer with multiple sample dimensions."""

    def test_sample_same_sample_dimensions(self) -> None:
        """Test __call__ with same sample dimensions for both
        tokens."""
        from tfmpe.nn.transformer import Transformer

        pytree = {
            'context': jnp.ones((2, 4, 3, 1)),
            'param': jnp.ones((2, 4, 5, 1)),
        }
        independence = {
            'local': ['param'],
            'cross': [('context', 'param'), ('param', 'context')],
            'cross_local': [],
        }

        tokens = Tokens.from_pytree(
            pytree,
            independence=independence,
            sample_ndims=2,
            batch_ndims={'context': 1, 'param': 1},
        )

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

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config=config,
            tokens=tokens,
            rngs=rngs,
        )

        time = jnp.ones((2, 4))
        context_view = tokens.select_tokens(keys=['context'])
        param_view = tokens.select_tokens(keys=['param'])

        output = transformer(
            context=context_view,
            params=param_view,
            time=time,
        )

        # Expected shape: param is (2, 4, 5, 1)
        expected_shape = (2, 4, 5, 1)
        assert output.shape == expected_shape

    def test_single_context_sample_with_tokens(self) -> None:
        """Test __call__ with broadcast from (1,) sample_shape."""
        from tfmpe.nn.transformer import Transformer

        # Create context tokens with sample_shape=(1,)
        context_pytree = {
            'context': jnp.ones((1, 3, 1)),
        }
        context_independence = {
            'local': [],
            'cross': [],
            'cross_local': [],
        }
        context_tokens = Tokens.from_pytree(
            context_pytree,
            independence=context_independence,
            sample_ndims=1,
            batch_ndims={'context': 1},
        )

        # Create param tokens with sample_shape=(4,)
        param_pytree = {
            'param': jnp.ones((4, 5, 1)),
        }
        param_independence = {
            'local': [],
            'cross': [],
            'cross_local': [],
        }
        param_tokens = Tokens.from_pytree(
            param_pytree,
            independence=param_independence,
            sample_ndims=1,
            batch_ndims={'param': 1},
        )

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

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config=config,
            tokens=param_tokens,
            rngs=rngs,
        )

        time = jnp.ones((4,))
        context_view = context_tokens.select_tokens(
            keys=['context']
        )
        param_view = param_tokens.select_tokens(keys=['param'])

        output = transformer(
            context=context_view,
            params=param_view,
            time=time,
        )

        # Expected shape: param is (4, 5, 1)
        expected_shape = (4, 5, 1)
        assert output.shape == expected_shape

    def test_single_context_sample_with_tokenviews(self) -> None:
        """Test __call__ with TokenView objects and broadcast."""
        from tfmpe.nn.transformer import Transformer

        # Create a full tokens object with mixed sample shapes
        # We'll create separate tokens and use views
        context_pytree = {
            'context': jnp.ones((1, 3, 1)),
        }
        context_independence = {
            'local': [],
            'cross': [],
            'cross_local': [],
        }
        context_tokens = Tokens.from_pytree(
            context_pytree,
            independence=context_independence,
            sample_ndims=1,
            batch_ndims={'context': 1},
        )

        param_pytree = {
            'param': jnp.ones((4, 5, 1)),
        }
        param_independence = {
            'local': [],
            'cross': [],
            'cross_local': [],
        }
        param_tokens = Tokens.from_pytree(
            param_pytree,
            independence=param_independence,
            sample_ndims=1,
            batch_ndims={'param': 1},
        )

        # Create views from the tokens
        context_view = context_tokens.select_tokens(
            keys=['context']
        )
        param_view = param_tokens.select_tokens(keys=['param'])

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

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config=config,
            tokens=param_tokens,
            rngs=rngs,
        )

        time = jnp.ones((4,))

        output = transformer(
            context=context_view,
            params=param_view,
            time=time,
        )

        # Expected shape: param is (4, 5, 1)
        expected_shape = (4, 5, 1)
        assert output.shape == expected_shape
