"""Tests for combine_tokens functionality.

Tests verify combining multiple Tokens objects for dataset accumulation.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array

from tfmpe.preprocessing import Tokens, combine_tokens
from tfmpe.preprocessing.utils import Independence


@pytest.fixture
def simple_pytree_sample2():
    """Simple pytree with sample_shape=(2,), same n_tokens."""
    return {
        'mu': jnp.array([[[1.0]], [[2.0]]]),  # (2, 1, 1)
        'theta': jnp.array([[[3.0]], [[4.0]]])  # (2, 1, 1)
    }


@pytest.fixture
def simple_pytree_sample3():
    """Simple pytree with sample_shape=(3,), same n_tokens."""
    return {
        'mu': jnp.array([[[5.0]], [[6.0]], [[7.0]]]),  # (3, 1, 1)
        'theta': jnp.array([[[8.0]], [[9.0]], [[10.0]]])  # (3, 1, 1)
    }


@pytest.fixture
def simple_independence():
    """Independence spec for simple structure."""
    return Independence(
        local=['theta'],
        cross=[('mu', 'theta'), ('theta', 'mu')]
    )


@pytest.fixture
def different_n_tokens_pytree1():
    """Pytree with 3 tokens total (2 + 1)."""
    return {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1) -> 2 tokens
        'b': jnp.array([[[3.0]]])  # (1, 1, 1) -> 1 token
    }


@pytest.fixture
def different_n_tokens_pytree2():
    """Pytree with 5 tokens total (3 + 2)."""
    return {
        'a': jnp.array([[[4.0], [5.0], [6.0]]]),  # (1, 3, 1) -> 3 tokens
        'b': jnp.array([[[7.0], [8.0]]])  # (1, 2, 1) -> 2 tokens
    }


@pytest.fixture
def different_n_tokens_independence():
    """Independence spec for different n_tokens case."""
    return Independence(
        local=['a', 'b']
    )


@pytest.fixture
def mixed_event_sizes_pytree1():
    """Pytree where key 'a' is larger, key 'b' is smaller."""
    return {
        'a': jnp.array([[[1.0], [2.0], [3.0]]]),  # (1, 3, 1)
        'b': jnp.array([[[4.0]]])  # (1, 1, 1)
    }


@pytest.fixture
def mixed_event_sizes_pytree2():
    """Pytree where key 'a' is smaller, key 'b' is larger."""
    return {
        'a': jnp.array([[[5.0], [6.0]]]),  # (1, 2, 1)
        'b': jnp.array([[[7.0], [8.0], [9.0]]])  # (1, 3, 1)
    }


def test_combine_same_n_tokens(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test combining Tokens with same n_tokens (no padding)."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check sample dimension is concatenated: 2 + 3 = 5
    assert combined.data.shape[0] == 5

    # Check n_tokens dimension is preserved (both have 2 tokens)
    assert combined.data.shape[1] == tokens1.data.shape[1]
    assert combined.data.shape[1] == tokens2.data.shape[1]

    # Check batch dimension preserved
    assert combined.data.shape[2] == tokens1.data.shape[2]


def test_combine_different_n_tokens(
    different_n_tokens_pytree1,
    different_n_tokens_pytree2,
    different_n_tokens_independence
):
    """Test combining Tokens with different n_tokens (padding)."""
    tokens1 = Tokens.from_pytree(
        different_n_tokens_pytree1,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    tokens2 = Tokens.from_pytree(
        different_n_tokens_pytree2,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check n_tokens padded to max
    max_n = max(tokens1.data.shape[1], tokens2.data.shape[1])
    assert combined.data.shape[1] == max_n

    # Check sample dimension concatenated: 1 + 1 = 2
    assert combined.data.shape[0] == 2


def test_combine_labels_same_n_tokens(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test labels when n_tokens is same."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Labels should be identical since same structure
    # First samples (2) should match tokens1
    assert jnp.array_equal(
        combined.labels[0:2],
        tokens1.labels
    )

    # Next samples (3) should match tokens2
    assert jnp.array_equal(
        combined.labels[2:5],
        tokens2.labels
    )


def test_combine_labels_extended_by_max_event_shape(
    mixed_event_sizes_pytree1,
    mixed_event_sizes_pytree2,
    different_n_tokens_independence
):
    """Test labels extended per key's max event_shape."""
    tokens1 = Tokens.from_pytree(
        mixed_event_sizes_pytree1,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    tokens2 = Tokens.from_pytree(
        mixed_event_sizes_pytree2,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Combined has max event shapes: a=(3,), b=(3,) -> 6 tokens total
    assert combined.data.shape[1] == 6

    # Get label IDs
    label_a = combined.label_map['a']
    label_b = combined.label_map['b']

    # Sample 0 (from tokens1): a has 3 tokens, b has 1 token
    # After padding: a has 3, b has 3 (extended from 1)
    # Labels should be: [a, a, a, b, b, b]
    expected_labels_0 = jnp.array(
        [label_a, label_a, label_a, label_b, label_b, label_b]
    )
    assert jnp.array_equal(combined.labels[0], expected_labels_0)

    # Sample 1 (from tokens2): a has 2 tokens, b has 3 tokens
    # After padding: a has 3 (extended from 2), b has 3
    # Labels should be: [a, a, a, b, b, b]
    expected_labels_1 = jnp.array(
        [label_a, label_a, label_a, label_b, label_b, label_b]
    )
    assert jnp.array_equal(combined.labels[1], expected_labels_1)


def test_combine_self_attention_mask_same_n_tokens(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test SA mask when n_tokens is same."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # SA mask is shared (no sample dimension)
    assert combined.self_attention_mask.shape == (2, 2)

    # SA mask should match tokens1/tokens2 (same structure)
    assert jnp.array_equal(
        combined.self_attention_mask,
        tokens1.self_attention_mask
    )
    assert jnp.array_equal(
        combined.self_attention_mask,
        tokens2.self_attention_mask
    )


def test_combine_self_attention_mask_different_n_tokens(
    different_n_tokens_pytree1,
    different_n_tokens_pytree2,
    different_n_tokens_independence
):
    """Test SA mask with different n_tokens per sample."""
    tokens1 = Tokens.from_pytree(
        different_n_tokens_pytree1,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    tokens2 = Tokens.from_pytree(
        different_n_tokens_pytree2,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # SA mask is shared (no sample dimension), padded to max_n_tokens
    assert combined.self_attention_mask.shape == (5, 5)

    # Top-left of combined mask should preserve structure
    # The mask is rebuilt with new slices, preserving independence
    # For 'local': ['a', 'b'], the diagonal blocks should be 0
    # Key 'a' goes from offset 0 to 3 (3 tokens), key 'b' from 3 to 5 (2 tokens)
    # So diagonal blocks are [0:3, 0:3] and [3:5, 3:5], both should be 0
    assert jnp.all(combined.self_attention_mask[0:3, 0:3] == 0)
    assert jnp.all(combined.self_attention_mask[3:5, 3:5] == 0)
    # Cross blocks between 'a' and 'b' should be 1
    assert jnp.all(combined.self_attention_mask[0:3, 3:5] == 1)
    assert jnp.all(combined.self_attention_mask[3:5, 0:3] == 1)


def test_combine_functional_inputs_both_present(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test functional_inputs combined when both present."""
    func_inputs1 = {
        'mu': jnp.array([[[0.0]], [[0.0]]]),
        'theta': jnp.array([[[1.0]], [[1.0]]])
    }

    func_inputs2 = {
        'mu': jnp.array([[[0.0]], [[0.0]], [[0.0]]]),
        'theta': jnp.array([[[2.0]], [[2.0]], [[2.0]]])
    }

    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        functional_inputs=func_inputs2,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check functional_inputs is not None
    assert combined.functional_inputs is not None

    # Check shape matches combined data
    assert combined.functional_inputs.shape == combined.data.shape


def test_combine_functional_inputs_error_one_absent(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test error when only one Tokens has functional_inputs."""
    func_inputs1 = {
        'mu': jnp.array([[[0.0]], [[0.0]]]),
        'theta': jnp.array([[[1.0]], [[1.0]]])
    }

    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    # Should raise error for mismatched functional_inputs
    with pytest.raises(ValueError, match="functional_inputs"):
        combine_tokens(tokens1, tokens2)


def test_combine_functional_inputs_both_absent(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test functional_inputs is None when both absent."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    assert combined.functional_inputs is None


def test_combine_functional_inputs_with_padding():
    """Test functional_inputs padded with -1e8 for smaller tokens."""
    pytree1 = {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1)
    }
    pytree2 = {
        'a': jnp.array([[[3.0], [4.0], [5.0]]]),  # (1, 3, 1)
    }
    func_inputs1 = {
        'a': jnp.array([[[10.0], [20.0]]]),  # (1, 2, 1)
    }
    func_inputs2 = {
        'a': jnp.array([[[30.0], [40.0], [50.0]]]),  # (1, 3, 1)
    }

    tokens1 = Tokens.from_pytree(
        pytree1,
        independence=Independence(local=[]),
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'a': 1}
    )

    tokens2 = Tokens.from_pytree(
        pytree2,
        independence=Independence(local=[]),
        functional_inputs=func_inputs2,
        sample_ndims=1,
        batch_ndims={'a': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check functional_inputs is not None
    assert combined.functional_inputs is not None

    # Sample 0 (from tokens1): first 2 match, last is -1e8
    assert jnp.allclose(
        combined.functional_inputs[0, 0:2, :],
        jnp.array([[10.0], [20.0]])
    )
    assert jnp.allclose(combined.functional_inputs[0, 2, :], -1e8)

    # Sample 1 (from tokens2): all positions match
    assert jnp.allclose(
        combined.functional_inputs[1, :, :],
        jnp.array([[30.0], [40.0], [50.0]])
    )


def test_combine_slices_match_larger_tokens(
    different_n_tokens_pytree1,
    different_n_tokens_pytree2,
    different_n_tokens_independence
):
    """Test slices match the tokens with larger n_tokens."""
    tokens1 = Tokens.from_pytree(
        different_n_tokens_pytree1,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    tokens2 = Tokens.from_pytree(
        different_n_tokens_pytree2,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # tokens2 has larger n_tokens, so slices should match tokens2
    assert combined.slices == tokens2.slices


def test_combine_slices_max_per_key(
    mixed_event_sizes_pytree1,
    mixed_event_sizes_pytree2,
    different_n_tokens_independence
):
    """Test slices use max event_shape per key."""
    tokens1 = Tokens.from_pytree(
        mixed_event_sizes_pytree1,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    tokens2 = Tokens.from_pytree(
        mixed_event_sizes_pytree2,
        independence=different_n_tokens_independence,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Key 'a': tokens1 has (3,), tokens2 has (2,) -> max is (3,)
    assert combined.slices['a'].event_shape == (3,)

    # Key 'b': tokens1 has (1,), tokens2 has (3,) -> max is (3,)
    assert combined.slices['b'].event_shape == (3,)

    # Total tokens should be 3 + 3 = 6
    assert combined.data.shape[1] == 6


def test_combine_multiple_via_repeated_calls(
    simple_pytree_sample2,
    simple_independence
):
    """Test combining >2 Tokens via repeated calls."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens3 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    # Combine three tokens
    combined12 = combine_tokens(tokens1, tokens2)
    combined123 = combine_tokens(combined12, tokens3)

    # Check sample dimension: 2 + 2 + 2 = 6
    assert combined123.data.shape[0] == 6


def test_combine_error_different_key_order():
    """Test error when tokens have different key_order."""
    pytree1 = {
        'mu': jnp.array([[[1.0]]]),
        'theta': jnp.array([[[2.0]]])
    }

    pytree2 = {
        'theta': jnp.array([[[3.0]]]),
        'sigma': jnp.array([[[4.0]]])
    }

    tokens1 = Tokens.from_pytree(
        pytree1,
        independence=Independence(local=[]),
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        pytree2,
        independence=Independence(local=[]),
        sample_ndims=1,
        batch_ndims={'theta': 1, 'sigma': 1}
    )

    with pytest.raises(ValueError, match="key_order"):
        combine_tokens(tokens1, tokens2)


def test_combine_error_different_independence():
    """Test error when tokens have different independence specs."""
    pytree = {
        'mu': jnp.array([[[1.0]]]),
        'theta': jnp.array([[[2.0]]])
    }

    independence1 = Independence(local=['theta'])
    independence2 = Independence(local=['mu', 'theta'])

    tokens1 = Tokens.from_pytree(
        pytree,
        independence=independence1,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        pytree,
        independence=independence2,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    with pytest.raises(ValueError, match="independence"):
        combine_tokens(tokens1, tokens2)


def test_combine_data_values_preserved(
    simple_pytree_sample2,
    simple_pytree_sample3,
    simple_independence
):
    """Test that actual data values are correctly concatenated."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        independence=simple_independence,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Decode and check values
    combined_decoded = combined.decode()

    # First samples should match tokens1
    assert jnp.array_equal(
        combined_decoded['mu'][0:2],
        simple_pytree_sample2['mu']
    )
    assert jnp.array_equal(
        combined_decoded['theta'][0:2],
        simple_pytree_sample2['theta']
    )

    # Next samples should match tokens2
    assert jnp.array_equal(
        combined_decoded['mu'][2:5],
        simple_pytree_sample3['mu']
    )
    assert jnp.array_equal(
        combined_decoded['theta'][2:5],
        simple_pytree_sample3['theta']
    )


def test_combine_padding_mask_set():
    """Test padding_mask is correctly set on combined Tokens."""
    pytree1 = {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1) -> 2 tokens
    }
    pytree2 = {
        'a': jnp.array([[[3.0], [4.0], [5.0]]]),  # (1, 3, 1) -> 3 tokens
    }

    tokens1 = Tokens.from_pytree(
        pytree1,
        independence=Independence(local=[]),
        sample_ndims=1,
        batch_ndims={'a': 1}
    )

    tokens2 = Tokens.from_pytree(
        pytree2,
        independence=Independence(local=[]),
        sample_ndims=1,
        batch_ndims={'a': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Padding mask should exist
    assert combined.padding_mask is not None

    # Shape should match sample and token dimensions
    assert combined.padding_mask.shape == (2, 3)

    # Sample 0 (from tokens1): first 2 tokens are 1, last is 0
    assert jnp.array_equal(combined.padding_mask[0], jnp.array([1, 1, 0]))

    # Sample 1 (from tokens2): all tokens are 1
    assert jnp.array_equal(combined.padding_mask[1], jnp.array([1, 1, 1]))
