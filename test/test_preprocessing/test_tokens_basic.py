"""Tests for Tokens class basic functionality.

Tests verify creation of Tokens from PyTree and decoding back to PyTree.
"""

import jax.numpy as jnp
from tfmpe.preprocessing import Tokens, Labeller

def test_from_pytree_data_shape(simple_pytree):
    """Test that data array has correct shape."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        condition=['obs'],
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Total tokens: 1 (mu) + 3 (theta) + 3 (obs) = 7
    # Max batch size: 1
    assert tokens.data.shape == (7, 1)


def test_from_pytree_labels_shape(simple_pytree):
    """Test that labels have correct shape and values."""
    tokens = Tokens.from_pytree(
        simple_pytree,
        condition=['obs'],
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )

    # Labels should have shape (7,) for 7 total tokens
    assert tokens.labels.shape == (7,)

    # Check that each block has consistent labels
    # condition is moved to first
    obs_labels = tokens.labels[1:3]
    mu_label = tokens.labels[3]
    theta_labels = tokens.labels[4:8]

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
        condition=[],
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

def test_decode_round_trip(simple_pytree):
    """Test that decoder recovers original PyTree."""
    tokens, decoder = Tokens.from_pytree_with_decoder(
        simple_pytree,
        condition=['obs'],
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1},
    )

    reconstructed = decoder(tokens)

    # Check keys match
    assert set(reconstructed.keys()) == set(simple_pytree.keys())

    # Check shapes match
    for key in simple_pytree:
        assert reconstructed[key].shape == simple_pytree[key].shape

    # Check values match
    for key in simple_pytree:
        assert jnp.allclose(reconstructed[key], simple_pytree[key])


def test_decode_after_modification(simple_pytree):
    """Test decoder after modifying different keys."""
    labeller = Labeller.for_keys(['mu', 'theta', 'obs'])
    _, decoder = Tokens.from_pytree_with_decoder(
        simple_pytree,
        condition=['obs'],
        sample_ndims=0,
        labeller=labeller,
    )

    new_pytree = {
        'mu': simple_pytree['mu'] * 10,
        'theta': simple_pytree['theta'] * 2.0,
        'obs': simple_pytree['obs'] * 0.5,
    }

    modified_tokens = Tokens.from_pytree(
        new_pytree,
        condition=['obs'],
        sample_ndims=0,
        labeller=labeller,
    )

    reconstructed = decoder(modified_tokens)

    # Check that values have correct coefficients applied
    assert jnp.allclose(reconstructed['mu'], simple_pytree['mu'] * 10.0)
    assert jnp.allclose(reconstructed['theta'], simple_pytree['theta'] * 2.0)
    assert jnp.allclose(reconstructed['obs'], simple_pytree['obs'] * 0.5)

def test_from_pytree_with_functional_inputs(simple_pytree):
    """Test from_pytree with functional inputs."""
    # Create functional inputs matching pytree structure
    functional_inputs = {
        'mu': jnp.array([[0.0]]),
        'theta': jnp.array([[1.0], [1.0], [1.0]]),
        'obs': jnp.array([[2.0], [2.1], [2.2]])
    }

    tokens = Tokens.from_pytree(
        simple_pytree,
        condition=['obs'],
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
        condition=[],
        functional_inputs=functional_inputs,
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    # Check functional_inputs shape: (2, 3, 1)
    assert tokens.functional_inputs is not None
    assert tokens.functional_inputs.shape == (2, 3, 1)
    assert tokens.functional_inputs.shape == tokens.data.shape
