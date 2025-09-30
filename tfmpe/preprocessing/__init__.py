"""Pipelines for processing datasets for use with the estimators."""

from tfmpe.preprocessing.flatten import flatten_pytree, update_flat_array
from tfmpe.preprocessing.functional_inputs import flatten_functional_inputs
from tfmpe.preprocessing.reconstruct import (
    decode_pytree,
    decode_pytree_keys,
)

__all__ = [
    "flatten_pytree",
    "update_flat_array",
    "flatten_functional_inputs",
    "decode_pytree",
    "decode_pytree_keys",
]