"""Tests for data generation and batching utilities.

Tests verify TokenGenerator yields correct batches, PyTree registration
enables proper batching with tree.map, and aux_data fields remain unchanged.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

from tfmpe.preprocessing import Tokens
from tfmpe.preprocessing import TokenGenerator


@pytest.fixture
def mock_prior_fn():
    """Mock prior function that generates parameter samples."""
    def prior(key: Array, n_samples: int) -> dict[str, Array]:
        k1, k2 = jr.split(key)
        return {
            'mu': jr.normal(k1, (n_samples, 1, 1)),
            'theta': jr.normal(k2, (n_samples, 3, 1))
        }
    return prior


@pytest.fixture
def mock_simulator_fn():
    """Mock simulator function that generates observations."""
    def simulator(
        key: Array,
        params: dict[str, Array]
    ) -> dict[str, Array]:
        n_samples = params['theta'].shape[0]
        return {
            'obs': jr.normal(key, (n_samples, 3, 1)) + params['theta']
        }
    return simulator


@pytest.fixture
def simple_independence():
    """Simple independence specification."""
    return {
        'local': ['obs', 'theta'],
        'cross': [('mu', 'obs'), ('obs', 'mu')],
        'cross_local': [('theta', 'obs', None)]
    }


class TestTokenGenerator:
    """Test TokenGenerator class."""

    def test_generator_yields_correct_number_of_batches(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        simple_independence
    ):
        """Test that generator yields expected number of batches."""
        n_samples = 100
        batch_size = 10
        expected_batches = n_samples // batch_size

        gen = TokenGenerator(
            prior_fn=mock_prior_fn,
            simulator_fn=mock_simulator_fn,
            functional_input_fn=None,
            independence=simple_independence,
            n_samples=n_samples,
            batch_size=batch_size,
            seed=42
        )

        assert len(gen) == expected_batches

        batch_count = 0
        for batch in gen:
            assert isinstance(batch, Tokens)
            batch_count += 1

        assert batch_count == expected_batches

    def test_generator_batch_shapes_consistent(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        simple_independence
    ):
        """Test that all batches have consistent shapes."""
        batch_size = 10
        gen = TokenGenerator(
            prior_fn=mock_prior_fn,
            simulator_fn=mock_simulator_fn,
            functional_input_fn=None,
            independence=simple_independence,
            n_samples=30,
            batch_size=batch_size,
            seed=42
        )

        batches = list(gen)
        first_batch = batches[0]

        for batch in batches:
            assert batch.data.shape == first_batch.data.shape
            assert batch.labels.shape == first_batch.labels.shape
            assert (
                batch.self_attention_mask.shape ==
                first_batch.self_attention_mask.shape
            )

        assert first_batch.data.shape[0] == batch_size

    def test_generator_with_uneven_batch_sizes(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        simple_independence
    ):
        """Test that generator drops incomplete final batch."""
        n_samples = 105
        batch_size = 10
        expected_batches = 10

        gen = TokenGenerator(
            prior_fn=mock_prior_fn,
            simulator_fn=mock_simulator_fn,
            functional_input_fn=None,
            independence=simple_independence,
            n_samples=n_samples,
            batch_size=batch_size,
            seed=42
        )

        assert len(gen) == expected_batches

        batch_count = 0
        for batch in gen:
            assert batch.data.shape[0] == batch_size
            batch_count += 1

        assert batch_count == expected_batches


class TestPyTreeRegistration:
    """Test PyTree registration for Tokens."""

    def test_aux_data_not_transformed_by_tree_map(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        simple_independence
    ):
        """Test that aux_data fields are not transformed by tree.map."""
        key = jr.key(42)
        k1, k2 = jr.split(key)
        params = mock_prior_fn(k1, 10)
        obs = mock_simulator_fn(k2, params)
        data = params | obs

        tokens = Tokens.from_pytree(
            data,
            simple_independence,
            sample_ndims=1
        )

        batched = jax.tree.map(
            lambda x: x.reshape((2, 5) + x.shape[1:]),
            tokens
        )

        assert (
            jnp.array_equal(
                batched.self_attention_mask,
                tokens.self_attention_mask
            )
        )
        assert batched.slices == tokens.slices
        assert batched.label_map == tokens.label_map
        assert batched.key_order == tokens.key_order

        assert batched.data.shape[0] == 2
        assert batched.data.shape[1] == 5
        assert batched.labels.shape[0] == 2
        assert batched.labels.shape[1] == 5

    def test_batching_with_reshape_pattern(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        simple_independence
    ):
        """Test batching with reshape pattern for training loops."""
        n = 100
        batch_size = 10
        n_batches = n // batch_size

        key = jr.key(42)
        k1, k2 = jr.split(key)
        params = mock_prior_fn(k1, n)
        obs = mock_simulator_fn(k2, params)
        data = params | obs

        tokens = Tokens.from_pytree(
            data,
            simple_independence,
            sample_ndims=1
        )

        idx = jr.permutation(jr.key(0), n)
        batches = jax.tree.map(
            lambda x: x[idx].reshape((n_batches, batch_size) + x.shape[1:]),
            tokens
        )

        assert batches.data.shape[0] == n_batches
        assert batches.data.shape[1] == batch_size
        assert batches.labels.shape[0] == n_batches
        assert batches.labels.shape[1] == batch_size

        assert (
            jnp.array_equal(
                batches.self_attention_mask,
                tokens.self_attention_mask
            )
        )


class TestBatchingWithFunctionalInputs:
    """Test batching with functional inputs present."""

    @pytest.fixture
    def mock_functional_input_fn(self):
        """Mock functional input function."""
        def func_input(params: dict[str, Array]) -> dict[str, Array]:
            return {
                'obs': jnp.ones_like(params['theta']) * 0.5
            }
        return func_input

    def test_functional_inputs_batched_correctly(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        mock_functional_input_fn,
        simple_independence
    ):
        """Test that functional inputs are batched as children."""
        key = jr.key(42)
        k1, k2 = jr.split(key)
        params = mock_prior_fn(k1, 10)
        obs = mock_simulator_fn(k2, params)
        data = params | obs
        func_inputs = mock_functional_input_fn(params)

        tokens = Tokens.from_pytree(
            data,
            simple_independence,
            functional_inputs=func_inputs,
            sample_ndims=1
        )

        batched = jax.tree.map(
            lambda x: x.reshape((2, 5) + x.shape[1:]),
            tokens
        )

        assert batched.functional_inputs is not None
        assert batched.functional_inputs.shape[0] == 2
        assert batched.functional_inputs.shape[1] == 5

    def test_none_functional_inputs_preserved(
        self,
        mock_prior_fn,
        mock_simulator_fn,
        simple_independence
    ):
        """Test that None functional inputs remain None after batching."""
        key = jr.key(42)
        k1, k2 = jr.split(key)
        params = mock_prior_fn(k1, 10)
        obs = mock_simulator_fn(k2, params)
        data = params | obs

        tokens = Tokens.from_pytree(data, simple_independence, sample_ndims=1)

        batched = jax.tree.map(
            lambda x: x.reshape((2, 5) + x.shape[1:]),
            tokens
        )

        assert batched.functional_inputs is None
