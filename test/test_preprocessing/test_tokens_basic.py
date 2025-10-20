"""Tests for Tokens class basic functionality.

Tests verify creation of Tokens from PyTree and decoding back to PyTree.
"""

import jax.numpy as jnp
import pytest

from tfmpe.preprocessing import Tokens


@pytest.fixture
def three_level_pytree():
    """3-level hierarchical structure."""
    return {
        'global_mu': jnp.array([[1.0]]),  # (1, 1)
        'group_mu': jnp.array([[2.0], [3.0]]),  # (2, 1)
        'local_theta': jnp.array(
            [[4.0], [5.0], [6.0], [7.0]]
        ),  # (4, 1)
        'obs': jnp.array([[0.1], [0.2], [0.3], [0.4]])  # (4, 1)
    }


@pytest.fixture
def three_level_independence():
    """Independence spec for 3-level structure."""
    return {
        'local': ['local_theta', 'obs'],
        'cross': [
            ('global_mu', 'obs'),
            ('obs', 'global_mu')
        ],
        'cross_local': [
            ('group_mu', 'local_theta', None),
            ('local_theta', 'obs', None)
        ]
    }


def test_from_pytree_data_shape(simple_pytree, simple_independence):
    """Test that data array has correct shape."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Total tokens: 1 (mu) + 3 (theta) + 3 (obs) = 7
    # Max batch size: 1
    assert tokens.data.shape == (7, 1)


def test_from_pytree_labels_shape(simple_pytree, simple_independence):
    """Test that labels have correct shape and values."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Labels should have shape (7,) for 7 total tokens
    assert tokens.labels.shape == (7,)

    # Check that each block has consistent labels
    mu_label = tokens.labels[0]
    theta_labels = tokens.labels[1:4]
    obs_labels = tokens.labels[4:7]

    # All tokens from same key should have same label
    assert jnp.all(theta_labels == theta_labels[0])
    assert jnp.all(obs_labels == obs_labels[0])

    # Different keys should have different labels
    assert mu_label != theta_labels[0]
    assert mu_label != obs_labels[0]
    assert theta_labels[0] != obs_labels[0]


def test_from_pytree_labels_with_sample_dims():
    """Test labels with sample dimensions."""
    pytree = {
        'a': jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]]),  # (2, 2, 1)
        'b': jnp.array([[[5.0]], [[6.0]]])  # (2, 1, 1)
    }

    tokens = Tokens.from_pytree(
        pytree,
        independence={'local': []},
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    # Labels should have shape (2, 3) for 2 samples, 3 total tokens
    assert tokens.labels.shape == (2, 3)

    # All samples should have same label structure
    assert jnp.array_equal(tokens.labels[0], tokens.labels[1])

    # Check label values
    a_labels = tokens.labels[0, 0:2]
    b_labels = tokens.labels[0, 2:3]

    assert jnp.all(a_labels == a_labels[0])
    assert jnp.all(b_labels == b_labels[0])
    assert a_labels[0] != b_labels[0]


def test_from_pytree_mask_shapes(simple_pytree, simple_independence):
    """Test that masks have correct shapes."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Self-attention mask: (7, 7)
    assert tokens.self_attention_mask.shape == (7, 7)

    # Padding mask: None for basic case
    assert tokens.padding_mask is None


def test_decode_round_trip(simple_pytree, simple_independence):
    """Test that decode() recovers original PyTree."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    reconstructed = tokens.decode()

    # Check keys match
    assert set(reconstructed.keys()) == set(simple_pytree.keys())

    # Check shapes match
    for key in simple_pytree:
        assert reconstructed[key].shape == simple_pytree[key].shape

    # Check values match
    for key in simple_pytree:
        assert jnp.allclose(reconstructed[key], simple_pytree[key])


def test_decode_after_modification(simple_pytree, simple_independence):
    """Test decode() after modifying different keys."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Modify different keys with different coefficients
    modified_data = tokens.data.copy()
    # mu: offset 0, size 1 -> multiply by 10
    modified_data = modified_data.at[0:1, :].set(
        modified_data[0:1, :] * 10.0
    )
    # theta: offset 1, size 3 -> multiply by 2
    modified_data = modified_data.at[1:4, :].set(
        modified_data[1:4, :] * 2.0
    )
    # obs: offset 4, size 3 -> multiply by 0.5
    modified_data = modified_data.at[4:7, :].set(
        modified_data[4:7, :] * 0.5
    )

    reconstructed = tokens.decode(modified_data)

    # Check that values have correct coefficients applied
    assert jnp.allclose(reconstructed['mu'], simple_pytree['mu'] * 10.0)
    assert jnp.allclose(reconstructed['theta'], simple_pytree['theta'] * 2.0)
    assert jnp.allclose(reconstructed['obs'], simple_pytree['obs'] * 0.5)


def test_decode_keys_subset(simple_pytree, simple_independence):
    """Test decode_keys() with subset of keys."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Decode only mu and obs
    subset = tokens.decode_keys(tokens.data, ['mu', 'obs'])

    # Check only requested keys present
    assert set(subset.keys()) == {'mu', 'obs'}

    # Check values match original
    assert jnp.allclose(subset['mu'], simple_pytree['mu'])
    assert jnp.allclose(subset['obs'], simple_pytree['obs'])


def test_decode_keys_single_key(simple_pytree, simple_independence):
    """Test decode_keys() with single key."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    subset = tokens.decode_keys(tokens.data, ['theta'])

    assert set(subset.keys()) == {'theta'}
    assert jnp.allclose(subset['theta'], simple_pytree['theta'])


def test_from_pytree_with_functional_inputs(
    simple_pytree,
    simple_independence
):
    """Test from_pytree with functional inputs."""
    # Create functional inputs matching pytree structure
    functional_inputs = {
        'mu': jnp.array([[0.0]]),
        'theta': jnp.array([[1.0], [1.0], [1.0]]),
        'obs': jnp.array([[2.0], [2.1], [2.2]])
    }

    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        functional_inputs=functional_inputs,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Check functional_inputs is not None
    assert tokens.functional_inputs is not None

    # Check shape matches data
    assert tokens.functional_inputs.shape == tokens.data.shape


def test_from_pytree_functional_inputs_with_sample_dims():
    """Test functional inputs with sample dimensions."""
    pytree = {
        'a': jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]]),  # (2, 2, 1)
        'b': jnp.array([[[5.0]], [[6.0]]])  # (2, 1, 1)
    }

    functional_inputs = {
        'a': jnp.array([[[0.0], [0.1]], [[0.0], [0.1]]]),  # (2, 2, 1)
        'b': jnp.array([[[1.0]], [[1.0]]])  # (2, 1, 1)
    }

    tokens = Tokens.from_pytree(
        pytree,
        independence={'local': []},
        functional_inputs=functional_inputs,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    # Check functional_inputs shape: (2, 3, 1)
    assert tokens.functional_inputs is not None
    assert tokens.functional_inputs.shape == (2, 3, 1)
    assert tokens.functional_inputs.shape == tokens.data.shape


def test_label_map_consistent(simple_pytree, simple_independence):
    """Test that label_map is consistent with labels array."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Check all keys have labels
    assert set(tokens.label_map.keys()) == set(simple_pytree.keys())

    # Expected labels array
    mu_id = tokens.label_map['mu']
    theta_id = tokens.label_map['theta']
    obs_id = tokens.label_map['obs']

    expected_labels = jnp.array([
        mu_id,  # mu (1 token)
        theta_id, theta_id, theta_id,  # theta (3 tokens)
        obs_id, obs_id, obs_id  # obs (3 tokens)
    ])

    assert jnp.array_equal(tokens.labels, expected_labels)


def test_key_order_matches_slices(simple_pytree, simple_independence):
    """Test that key_order matches the order of slices."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Check all keys present
    assert set(tokens.key_order) == set(simple_pytree.keys())

    # Check offsets are in order
    offsets = [tokens.slices[k].offset for k in tokens.key_order]
    assert offsets == sorted(offsets)
