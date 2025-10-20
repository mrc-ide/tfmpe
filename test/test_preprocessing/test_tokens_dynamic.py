"""Tests for Tokens class dynamic functionality.

Tests verify dynamic slicing, value override, and cross-attention mask
generation.
"""

import jax.numpy as jnp
import pytest

from tfmpe.preprocessing import Tokens


@pytest.fixture
def tokens_simple(simple_pytree, simple_independence):
    """Create Tokens object from simple pytree."""
    return Tokens.from_pytree(
        simple_pytree,
        independence=simple_independence,
        sample_ndims=0,
        batch_ndims={'mu': 1, 'theta': 1, 'obs': 1}
    )


def test_select_tokens_returns_view(tokens_simple):
    """Test that select_tokens returns a TokenView object."""
    view = tokens_simple.select_tokens(['theta', 'obs'])

    # Check that we got a view (not Tokens)
    assert type(view).__name__ == 'TokenView'
    assert hasattr(view, 'parent')
    assert view.parent is tokens_simple


def test_select_tokens_subset_data(simple_pytree, tokens_simple):
    """Test that select_tokens extracts correct data subset."""
    # Select only theta and obs (skip mu)
    view = tokens_simple.select_tokens(['theta', 'obs'])

    # Data should have 6 tokens (3 theta + 3 obs), not 7
    assert view.data.shape[0] == 6

    # Verify values match
    expected_data = jnp.concatenate([
        simple_pytree['theta'],
        simple_pytree['obs']
    ], axis=0)
    assert jnp.allclose(view.data, expected_data)


def test_select_tokens_subset_labels(tokens_simple):
    """Test that select_tokens extracts correct labels."""
    view = tokens_simple.select_tokens(['theta', 'obs'])

    # Labels should have 6 tokens
    assert view.labels.shape == (6,)

    # First 3 should be theta label, next 3 should be obs label
    assert jnp.all(view.labels[:3] == view.labels[0])
    assert jnp.all(view.labels[3:] == view.labels[3])
    assert view.labels[0] != view.labels[3]


def test_tokenview_decode_consistent_with_tokens(tokens_simple):
    """Test that TokenView.decode matches Tokens.decode for selected
    keys."""
    view = tokens_simple.select_tokens(['mu', 'theta'])

    # Decode from view
    view_decoded = view.decode()

    # Decode from original tokens
    tokens_decoded = tokens_simple.decode()

    # Should match for selected keys
    assert jnp.allclose(view_decoded['mu'], tokens_decoded['mu'])
    assert jnp.allclose(view_decoded['theta'], tokens_decoded['theta'])

    # View should only contain selected keys
    assert set(view_decoded.keys()) == {'mu', 'theta'}


def test_select_tokens_self_attention_mask_independent(tokens_simple):
    """Test self-attention mask for locally independent keys."""
    # theta and obs both have local independence
    view = tokens_simple.select_tokens(['theta', 'obs'])
    mask = view.self_attention_mask

    # Mask should be (6, 6) for 6 tokens
    assert mask.shape == (6, 6)

    # theta block (indices 0-2) - local independence zeros entire block
    theta_block = mask[:3, :3]
    assert jnp.allclose(theta_block, 0.0)

    # obs block (indices 3-5) - local independence zeros entire block
    obs_block = mask[3:, 3:]
    assert jnp.allclose(obs_block, 0.0)

    # Cross blocks (theta-obs and obs-theta) - cross_local means diagonal
    theta_obs_block = mask[:3, 3:]
    obs_theta_block = mask[3:, :3]
    assert jnp.allclose(theta_obs_block, jnp.eye(3))
    assert jnp.allclose(obs_theta_block, jnp.eye(3))


def test_select_tokens_self_attention_mask_dependent(tokens_simple):
    """Test self-attention mask for dependent keys."""
    # mu has no local independence, theta does
    view = tokens_simple.select_tokens(['mu', 'theta'])
    mask = view.self_attention_mask

    # Mask should be (4, 4) for 4 tokens (1 mu + 3 theta)
    assert mask.shape == (4, 4)

    # mu block (just 1 token) - no local independence, so all 1s
    mu_block = mask[0:1, 0:1]
    assert jnp.allclose(mu_block, jnp.ones((1, 1)))

    # theta block (indices 1-3) - local independence zeros entire block
    theta_block = mask[1:, 1:]
    assert jnp.allclose(theta_block, 0.0)

    # mu-theta cross block - no cross independence specified, so all 1s
    mu_theta_block = mask[0:1, 1:]
    theta_mu_block = mask[1:, 0:1]
    assert jnp.allclose(mu_theta_block, jnp.ones((1, 3)))
    assert jnp.allclose(theta_mu_block, jnp.ones((3, 1)))


def test_cross_attention_mask_independent_pair(tokens_simple):
    """Test cross_attention_mask for cross-independent pair."""
    # mu and obs have cross independence
    query_view = tokens_simple.select_tokens(['mu'])
    key_view = tokens_simple.select_tokens(['obs'])
    mask = tokens_simple.cross_attention_mask(query_view, key_view)

    # Should be (1, 3) - 1 mu token, 3 obs tokens
    assert mask.shape == (1, 3)

    # mu and obs are cross-independent, so all zeros
    assert jnp.all(mask == 0)


def test_cross_attention_mask_dependent_pair(tokens_simple):
    """Test cross_attention_mask for dependent pair."""
    # mu and theta have no cross independence
    query_view = tokens_simple.select_tokens(['mu'])
    key_view = tokens_simple.select_tokens(['theta'])
    mask = tokens_simple.cross_attention_mask(query_view, key_view)

    # Should be (1, 3) - 1 mu token, 3 theta tokens
    assert mask.shape == (1, 3)

    # No cross independence specified, so all ones
    assert jnp.allclose(mask, jnp.ones((1, 3)))


def test_cross_attention_mask_cross_local(tokens_simple):
    """Test cross_attention_mask with cross_local independence."""
    # theta and obs have cross_local independence (diagonal only)
    query_view = tokens_simple.select_tokens(['theta'])
    key_view = tokens_simple.select_tokens(['obs'])
    mask = tokens_simple.cross_attention_mask(query_view, key_view)

    # Should be (3, 3) - both theta and obs have 3 tokens
    assert mask.shape == (3, 3)

    # With cross_local diagonal, only diagonal should be 1
    assert jnp.allclose(mask, jnp.eye(3))


def test_with_values_single_key(tokens_simple):
    """Test with_values updates a single key correctly."""
    # Create new values for theta
    new_theta = jnp.array([[10.0], [11.0], [12.0]])

    new_tokens = tokens_simple.with_values(theta=new_theta)

    # Should return a new Tokens object
    assert isinstance(new_tokens, Tokens)
    assert new_tokens is not tokens_simple

    # Decode and verify theta was updated
    decoded = new_tokens.decode()
    assert jnp.allclose(decoded['theta'], new_theta)

    # Other keys should be unchanged
    original_decoded = tokens_simple.decode()
    assert jnp.allclose(decoded['mu'], original_decoded['mu'])
    assert jnp.allclose(decoded['obs'], original_decoded['obs'])


def test_with_values_multiple_keys(tokens_simple):
    """Test with_values updates multiple keys correctly."""
    new_mu = jnp.array([[5.0]])
    new_obs = jnp.array([[0.5], [0.6], [0.7]])

    new_tokens = tokens_simple.with_values(mu=new_mu, obs=new_obs)

    # Decode and verify both were updated
    decoded = new_tokens.decode()
    assert jnp.allclose(decoded['mu'], new_mu)
    assert jnp.allclose(decoded['obs'], new_obs)

    # theta should be unchanged
    original_decoded = tokens_simple.decode()
    assert jnp.allclose(decoded['theta'], original_decoded['theta'])


def test_with_values_preserves_metadata(tokens_simple):
    """Test that with_values preserves slices and other metadata."""
    new_mu = jnp.array([[100.0]])
    new_tokens = tokens_simple.with_values(mu=new_mu)

    # Metadata should be identical
    assert new_tokens.slices == tokens_simple.slices
    assert new_tokens.label_map == tokens_simple.label_map
    assert new_tokens.key_order == tokens_simple.key_order

    # Masks should be identical
    assert jnp.array_equal(
        new_tokens.self_attention_mask,
        tokens_simple.self_attention_mask
    )


def test_with_values_invalid_key_raises(tokens_simple):
    """Test that with_values raises error for invalid key."""
    with pytest.raises(KeyError, match="sigma"):
        tokens_simple.with_values(sigma=jnp.array([[1.0]]))


def test_with_values_wrong_shape_raises(tokens_simple):
    """Test that with_values raises error for wrong shape."""
    # theta should be (3, 1), not (2, 1)
    wrong_theta = jnp.array([[1.0], [2.0]])

    with pytest.raises((ValueError, AssertionError)):
        tokens_simple.with_values(theta=wrong_theta)


def test_select_tokens_single_key(tokens_simple):
    """Test selecting a single key works correctly."""
    view = tokens_simple.select_tokens(['mu'])

    # Should have 1 token
    assert view.data.shape == (1, 1)
    assert view.labels.shape == (1,)
    assert view.self_attention_mask.shape == (1, 1)


def test_select_tokens_invalid_key_raises(tokens_simple):
    """Test that select_tokens raises error for invalid key."""
    with pytest.raises(KeyError, match="sigma"):
        tokens_simple.select_tokens(['sigma'])
