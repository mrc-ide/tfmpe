"""Memory-efficient token data generator.

Provides TokenGenerator for generating Tokens batches on-demand without
pre-allocating full datasets.
"""

from typing import Callable, Iterator, Optional

import jax.random as jr
from jaxtyping import Array

from .tokens import Tokens
from .utils import Independence


class TokenGenerator:
    """
    Generate Tokens batches on-demand.

    Yields batches of Tokens without pre-allocating the full dataset,
    enabling memory-efficient training on large datasets.

    Parameters
    ----------
    prior_fn : Callable[[Array, int], dict[str, Array]]
        Function that takes (key, n_samples) and returns parameter samples
    simulator_fn : Callable[[Array, dict[str, Array]], dict[str, Array]]
        Function that takes (key, params) and returns simulated observations
    functional_input_fn : Optional[Callable]
        Function that takes params and returns functional inputs
    independence : Independence
        Independence specification for mask generation
    n_samples : int
        Total number of samples to generate
    batch_size : int
        Number of samples per batch
    seed : int
        Random seed for reproducibility

    Examples
    --------
    >>> gen = TokenGenerator(
    ...     prior_fn=my_prior,
    ...     simulator_fn=my_simulator,
    ...     functional_input_fn=None,
    ...     independence={'local': ['obs']},
    ...     n_samples=1000,
    ...     batch_size=100,
    ...     seed=42
    ... )
    >>> for batch in gen:
    ...     loss = train_step(model, batch)
    """

    def __init__(
        self,
        prior_fn: Callable[[Array, int], dict[str, Array]],
        simulator_fn: Callable[[Array, dict[str, Array]], dict[str, Array]],
        functional_input_fn: Optional[
            Callable[[dict[str, Array]], dict[str, Array]]
        ],
        independence: Independence,
        n_samples: int,
        batch_size: int,
        seed: int,
    ):
        """Initialize generator with sampling functions."""
        self.prior_fn = prior_fn
        self.simulator_fn = simulator_fn
        self.functional_input_fn = functional_input_fn
        self.independence = independence
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed
        self.n_batches = n_samples // batch_size

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches

    def __iter__(self) -> Iterator[Tokens]:
        """Yield batches of Tokens."""
        key = jr.key(self.seed)

        for _ in range(self.n_batches):
            k1, k2, key = jr.split(key, 3)

            params = self.prior_fn(k1, self.batch_size)
            obs = self.simulator_fn(k2, params)
            data = params | obs

            func_inputs = None
            if self.functional_input_fn is not None:
                func_inputs = self.functional_input_fn(params)

            tokens = Tokens.from_pytree(
                data,
                self.independence,
                functional_inputs=func_inputs,
                sample_ndims=1
            )

            yield tokens
