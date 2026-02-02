"""Tests for Labeller class functionality.

Tests verify Labeller initialization, label array generation,
and error handling.
"""

import jax.numpy as jnp
import pytest

from tfmpe.preprocessing.utils import Labeller, SliceInfo


class TestLabellerInitialization:
    """Tests for Labeller initialization."""

    def test_labeller_init_with_label_map(self):
        """Test Labeller initialization with label_map."""
        label_map = {'mu': 0, 'theta': 1, 'obs': 2}
        labeller = Labeller(label_map=label_map)
        assert labeller.label_map == label_map

    def test_labeller_init_empty_raises(self):
        """Test Labeller with empty label_map raises ValueError."""
        with pytest.raises(ValueError, match="at least one key"):
            labeller = Labeller(label_map={})
            # Call label to trigger the error
            labeller.label({})

    def test_labeller_single_key(self):
        """Test Labeller initialization with single key."""
        label_map = {'obs': 0}
        labeller = Labeller(label_map=label_map)
        assert labeller.label_map == label_map

    def test_labeller_init_colliding_labels_raises(self):
        """Test Labeller with colliding label indices raises ValueError."""
        label_map = {'mu': 0, 'theta': 0, 'obs': 2}
        with pytest.raises(ValueError, match="collision"):
            Labeller(label_map=label_map)

    def test_labeller_n_labels(self):
        """Test Labeller.n_labels property."""
        label_map = {'mu': 0, 'theta': 1, 'obs': 2}
        labeller = Labeller(label_map=label_map)
        assert labeller.n_labels == 3

    def test_labeller_n_labels_single_key(self):
        """Test Labeller.n_labels with single key."""
        label_map = {'obs': 0}
        labeller = Labeller(label_map=label_map)
        assert labeller.n_labels == 1

    def test_labeller_n_labels_sparse_indices(self):
        """Test Labeller.n_labels with sparse label indices."""
        label_map = {'a': 0, 'b': 5, 'c': 10}
        labeller = Labeller(label_map=label_map)
        # n_labels should be the count of entries in label_map
        assert labeller.n_labels == 3


class TestLabellerForKeys:
    """Tests for Labeller.for_keys() classmethod."""

    def test_labeller_for_keys_single_key(self):
        """Test for_keys with single key."""
        labeller = Labeller.for_keys(['obs'])
        assert labeller.label_map == {'obs': 0}
        assert labeller.n_labels == 1

    def test_labeller_for_keys_multiple_keys(self):
        """Test for_keys with multiple keys."""
        keys = ['mu', 'theta', 'obs']
        labeller = Labeller.for_keys(keys)
        assert labeller.label_map == {'mu': 0, 'theta': 1, 'obs': 2}
        assert labeller.n_labels == 3

    def test_labeller_for_keys_preserves_order(self):
        """Test for_keys assigns sequential indices in key order."""
        keys = ['c', 'a', 'b']
        labeller = Labeller.for_keys(keys)
        # Indices should be assigned in the order provided
        assert labeller.label_map == {'c': 0, 'a': 1, 'b': 2}


class TestLabellerLabel:
    """Tests for Labeller.label() method."""

    def test_labeller_label_single_key(self):
        """Test label generation with single key."""
        label_map = {'obs': 0}
        labeller = Labeller(label_map=label_map)

        slices = {
            'obs': SliceInfo(offset=0, event_shape=(3,), batch_shape=(1,))
        }

        labels = labeller.label(slices)

        assert labels.shape == (3,)
        assert jnp.all(labels == 0)
        assert labels.dtype == jnp.int32

    def test_labeller_label_multiple_keys(self):
        """Test label generation with multiple keys."""
        label_map = {'mu': 0, 'theta': 1, 'obs': 2}
        labeller = Labeller(label_map=label_map)

        slices = {
            'mu': SliceInfo(offset=0, event_shape=(1,), batch_shape=(1,)),
            'theta': SliceInfo(offset=1, event_shape=(3,), batch_shape=(1,)),
            'obs': SliceInfo(offset=4, event_shape=(3,), batch_shape=(1,))
        }

        labels = labeller.label(slices)

        # Total tokens: 1 + 3 + 3 = 7
        assert labels.shape == (7,)

        # Check labels are correct (in slice dict iteration order)
        # Since we're iterating slices.keys(), order depends on dict
        # ordering
        assert labels.dtype == jnp.int32
        assert jnp.all(
            jnp.isin(labels, jnp.array([0, 1, 2], dtype=jnp.int32))
        )

    def test_labeller_label_different_event_shapes(self):
        """Test label generation with various event shapes."""
        label_map = {'a': 0, 'b': 1, 'c': 2}
        labeller = Labeller(label_map=label_map)

        slices = {
            'a': SliceInfo(offset=0, event_shape=(2, 3), batch_shape=(1,)),
            'b': SliceInfo(offset=6, event_shape=(4,), batch_shape=(1,)),
            'c': SliceInfo(offset=10, event_shape=(2, 2), batch_shape=(1,))
        }

        labels = labeller.label(slices)

        # Total tokens: (2*3) + 4 + (2*2) = 6 + 4 + 4 = 14
        assert labels.shape == (14,)
        # Each key should have consistent labels
        assert jnp.all(
            jnp.isin(labels, jnp.array([0, 1, 2], dtype=jnp.int32))
        )

    def test_labeller_label_correct_shapes(self):
        """Test output shape matches total tokens."""
        label_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        labeller = Labeller(label_map=label_map)

        slices = {
            'a': SliceInfo(offset=0, event_shape=(5,), batch_shape=(1,)),
            'b': SliceInfo(offset=5, event_shape=(3, 2), batch_shape=(1,)),
            'c': SliceInfo(offset=11, event_shape=(1,), batch_shape=(1,)),
            'd': SliceInfo(offset=12, event_shape=(4,), batch_shape=(1,))
        }

        labels = labeller.label(slices)

        # Total: 5 + (3*2) + 1 + 4 = 5 + 6 + 1 + 4 = 16
        assert labels.shape == (16,)

    def test_labeller_label_slice_order(self):
        """Test that labels are generated in slices.keys() order."""
        label_map = {'c': 0, 'a': 1, 'b': 2}
        labeller = Labeller(label_map=label_map)

        # Create slices dict with specific key order
        slices = {
            'c': SliceInfo(offset=0, event_shape=(2,), batch_shape=(1,)),
            'a': SliceInfo(offset=2, event_shape=(1,), batch_shape=(1,)),
            'b': SliceInfo(offset=3, event_shape=(2,), batch_shape=(1,))
        }

        labels = labeller.label(slices)

        # Labels should follow slices.keys() order: c, a, b
        expected = jnp.array([0, 0, 1, 2, 2], dtype=jnp.int32)
        assert jnp.all(labels == expected)

    def test_labeller_label_dtype(self):
        """Test that output is int32."""
        label_map = {'a': 0}
        labeller = Labeller(label_map=label_map)

        slices = {
            'a': SliceInfo(offset=0, event_shape=(5,), batch_shape=(1,))
        }

        labels = labeller.label(slices)
        assert labels.dtype == jnp.int32

    def test_labeller_label_missing_key_raises(self):
        """Test label generation with key not in label_map raises
        KeyError."""
        label_map = {'mu': 0, 'theta': 1}
        labeller = Labeller(label_map=label_map)

        # Include 'obs' in slices but not in label_map
        slices = {
            'mu': SliceInfo(offset=0, event_shape=(1,), batch_shape=(1,)),
            'obs': SliceInfo(offset=1, event_shape=(3,), batch_shape=(1,))
        }

        with pytest.raises(KeyError, match="not found in label_map"):
            labeller.label(slices)
