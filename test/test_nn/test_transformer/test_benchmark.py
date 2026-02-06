"""Benchmark tests for Transformer memory and speed."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array

from tfmpe.nn.transformer import Transformer
from tfmpe.nn.transformer.config import TransformerConfig
from tfmpe.preprocessing import Labeller, Tokens


def create_benchmark_tokens(batch_size: int, seq_len: int) -> Tokens:
    """Create tokens for benchmarking.

    Parameters
    ----------
    batch_size : int
        Batch size
    seq_len : int
        Number of parameter tokens

    Returns
    -------
    Tokens
        Tokens object for benchmarking
    """
    labeller = Labeller(label_map={'context': 0, 'param': 1})

    data = {
        'context': jnp.ones((batch_size, 2, 1)),
        'param': jnp.ones((batch_size, seq_len, 1)),
    }

    return Tokens.from_pytree(
        data,
        labeller=labeller,
        condition=['context'],
        sample_ndims=1,
    )


class TestTransformerBenchmark:
    """Memory and speed benchmarks for Transformer forward/backward pass."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", [10, 50, 100, 200, 500])
    def test_forward_and_backward_benchmark(self, seq_len: int) -> None:
        """Benchmark forward and backward pass for varying sequence lengths.

        Parameters
        ----------
        seq_len : int
            Number of parameter tokens
        """
        batch_size = 32
        n_warmup = 3
        n_runs = 10

        config = TransformerConfig(
            latent_dim=128,
            n_encoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=32,
        )

        tokens = create_benchmark_tokens(batch_size, seq_len)
        time_input = jnp.ones((batch_size,))

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config=config,
            tokens=tokens,
            rngs=rngs,
        )

        @nnx.jit
        def forward_fn(model: Transformer, tokens: Tokens, t: Array) -> Array:
            return model(tokens=tokens, time=t, deterministic=True)

        def loss_fn(model: Transformer) -> Array:
            output = model(tokens=tokens, time=time_input, deterministic=True)
            return jnp.mean(output ** 2)

        grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))

        # Warmup forward pass
        for _ in range(n_warmup):
            result = forward_fn(transformer, tokens, time_input)
            result.block_until_ready()

        # Get memory stats if available
        device = jax.local_devices()[0]
        has_memory_stats = hasattr(device, 'memory_stats')

        # Benchmark forward pass
        if has_memory_stats:
            _ = device.memory_stats()

        forward_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = forward_fn(transformer, tokens, time_input)
            result.block_until_ready()
            forward_times.append(time.perf_counter() - start)

        forward_peak_memory = None
        if has_memory_stats:
            stats = device.memory_stats()
            if stats and 'peak_bytes_in_use' in stats:
                forward_peak_memory = stats['peak_bytes_in_use'] / (1024 * 1024)

        # Warmup backward pass
        for _ in range(n_warmup):
            loss, grads = grad_fn(transformer)
            loss.block_until_ready()

        # Reset memory stats before backward benchmark
        if has_memory_stats:
            _ = device.memory_stats()

        backward_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            loss, grads = grad_fn(transformer)
            loss.block_until_ready()
            backward_times.append(time.perf_counter() - start)

        backward_peak_memory = None
        if has_memory_stats:
            stats = device.memory_stats()
            if stats and 'peak_bytes_in_use' in stats:
                backward_peak_memory = stats['peak_bytes_in_use'] / (1024 * 1024)

        # Calculate statistics
        forward_mean = np.mean(forward_times) * 1000
        forward_std = np.std(forward_times) * 1000
        backward_mean = np.mean(backward_times) * 1000
        backward_std = np.std(backward_times) * 1000

        time_ratio = backward_mean / forward_mean if forward_mean > 0 else 0

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Benchmark: seq_len={seq_len}, batch_size={batch_size}")
        print(f"{'=' * 60}")
        print(f"Forward pass:  {forward_mean:.2f} +/- {forward_std:.2f} ms")
        if forward_peak_memory is not None:
            print(f"               Peak memory: {forward_peak_memory:.2f} MB")
        print(f"Backward pass: {backward_mean:.2f} +/- {backward_std:.2f} ms")
        if backward_peak_memory is not None:
            print(f"               Peak memory: {backward_peak_memory:.2f} MB")
        print(f"Time ratio (backward/forward): {time_ratio:.2f}x")
        if forward_peak_memory is not None and backward_peak_memory is not None:
            memory_ratio = backward_peak_memory / forward_peak_memory
            print(f"Memory ratio (backward/forward): {memory_ratio:.2f}x")
        print(f"{'=' * 60}")

        # Basic sanity checks
        assert forward_mean > 0, "Forward pass should take positive time"
        assert backward_mean > 0, "Backward pass should take positive time"
