"""Unit tests for ODE solving helpers."""

import numpy as np
import jax
import jax.numpy as jnp

from tfmpe.estimators.ode import (
    solve_forward_ode,
    solve_backward_ode,
    solve_augmented_ode,
    batch_solve_forward_ode,
    batch_solve_augmented_ode,
)


# ============================================================================
# Doubling Flow and Forward/Backward ODE
# ============================================================================


class TestSolveForwardODE:
    """Test forward ODE solver with doubling flow."""

    def test_doubling_flow_transformation(self, doubling_vf, solver):
        """Test that doubling flow transforms N(0,1) to N(0,4).

        The exponential scaling vector field f(θ) = log(2)·θ causes
        the variance to double: σ² → 4σ²

        Expected analytical result:
        - Initial: θ₀ ~ N(0, 1)
        - Final: θ₁ ~ N(0, 4)
        - Verify: mean ≈ 0, std ≈ 2 with rtol=0.1
        """
        seed = jax.random.PRNGKey(42)
        n_samples = 1000

        # Sample initial states from N(0, 1)
        x0_batch = jax.random.normal(seed, (n_samples,))

        # Solve forward ODE for all samples
        x1_batch = jax.vmap(
            lambda x0: solve_forward_ode(
                doubling_vf,
                x0,
                solver,
                rtol=1e-5,
                atol=1e-5,
            )
        )(x0_batch)

        # Compute statistics
        mean = jnp.mean(x1_batch)
        std = jnp.std(x1_batch)

        # Verify statistics within tolerance
        np.testing.assert_allclose(mean, 0.0, atol=0.1)
        np.testing.assert_allclose(std, 2.0, rtol=0.1)

    def test_single_sample_shapes(self, doubling_vf, solver):
        """Test that solve_forward_ode preserves input shape."""
        x0 = jnp.array([1.0, 2.0, 3.0])

        x1 = solve_forward_ode(
            doubling_vf,
            x0,
            solver,
        )

        assert x1.shape == x0.shape

    def test_multidim_initial_state(self, doubling_vf, solver):
        """Test forward ODE with multidimensional initial state."""
        # 2x3 matrix initial state
        x0 = jnp.ones((2, 3))

        x1 = solve_forward_ode(
            doubling_vf,
            x0,
            solver,
        )

        assert x1.shape == (2, 3)


class TestSolveBackwardODE:
    """Test backward ODE solver with trajectory reversal."""

    def test_backward_reverses_forward(self, doubling_vf, solver):
        """Test that backward(forward(x)) ≈ x.

        Verify that solving the backward ODE recovers the original state
        within numerical tolerance.
        """
        x0 = jnp.array([1.0, 2.0, 3.0])

        # Forward: x0 → x1
        x1 = solve_forward_ode(
            doubling_vf,
            x0,
            solver,
            rtol=1e-5,
            atol=1e-5,
        )

        # Backward: x1 → x0_recovered (using same vector field)
        x0_recovered = solve_backward_ode(
            doubling_vf,
            x1,
            solver,
            rtol=1e-5,
            atol=1e-5,
        )

        # Verify recovery within tolerance
        np.testing.assert_allclose(x0, x0_recovered, rtol=1e-3, atol=1e-4)

    def test_backward_shape_preservation(self, doubling_vf, solver):
        """Test that solve_backward_ode preserves input shape."""
        x1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        x0 = solve_backward_ode(
            doubling_vf,
            x1,
            solver,
        )

        assert x0.shape == x1.shape


# ============================================================================
# Augmented ODE and Batch Operations
# ============================================================================


class TestSolveAugmentedODE:
    """Test augmented ODE with trace-based log determinant."""

    def test_linear_vf_trace_2x2(self, solver):
        """Test trace computation for linear vf with 2x2 matrix.

        For linear f(x) = A·x, analytical log_det_jacobian = log|det(A)|
        Test with det(A) = 2.0 → log_det ≈ log(2)
        """
        # Create 2x2 matrix A with det = 2.0
        A = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        det_A = 2.0
        expected_log_det = jnp.log(det_A)

        def linear_vf(x, t):
            return A @ x

        x0 = jnp.array([1.0, 1.0])

        x_final, log_det_jacobian = solve_augmented_ode(
            linear_vf,
            x0,
            solver,
            rtol=1e-5,
            atol=1e-5,
        )

        # Verify log determinant within tolerance
        np.testing.assert_allclose(
            log_det_jacobian,
            expected_log_det,
            rtol=0.1,
            atol=0.01,
        )

    def test_linear_vf_trace_5x5(self, solver):
        """Test trace computation for 5x5 matrix.

        For f(x) = A·x with det(A) = 2.5
        """
        # Create 5x5 diagonal matrix with det = 2.5
        A = jnp.eye(5) * jnp.array([2.0, 1.0, 1.0, 1.0, 1.25])
        det_A = jnp.prod(jnp.diag(A))  # 2 * 1 * 1 * 1 * 1.25 = 2.5
        expected_log_det = jnp.log(det_A)

        def linear_vf(x, t):
            return A @ x

        x0 = jnp.ones(5)

        x_final, log_det_jacobian = solve_augmented_ode(
            linear_vf,
            x0,
            solver,
            rtol=1e-5,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            log_det_jacobian,
            expected_log_det,
            rtol=0.1,
            atol=0.01,
        )

    def test_augmented_state_shape(self, solver):
        """Test that augmented ODE returns correct output shape."""
        def vf(x, t):
            return -x  # Simple decay

        x0 = jnp.array([1.0, 2.0, 3.0])

        x_final, log_det = solve_augmented_ode(
            vf,
            x0,
            solver,
        )

        assert x_final.shape == x0.shape
        assert log_det.shape == ()  # Scalar


class TestBatchSolveForwardODE:
    """Test batch forward ODE solving with vmap."""

    def test_batch_vs_loop(self, doubling_vf, solver):
        """Test that batch_solve gives same results as loop.

        Verify vmap correctness by comparing vectorized vs looped
        application of solve_forward_ode.
        """
        seed = jax.random.PRNGKey(123)
        batch_size = 10
        state_dim = 3

        x0_batch = jax.random.normal(seed, (batch_size, state_dim))

        # Batch solve
        x1_batch = batch_solve_forward_ode(
            doubling_vf,
            x0_batch,
            solver,
            rtol=1e-5,
            atol=1e-5,
        )

        # Loop solve for comparison
        x1_loop = jnp.array([
            solve_forward_ode(
                doubling_vf,
                x0,
                solver,
                rtol=1e-5,
                atol=1e-5,
            )
            for x0 in x0_batch
        ])

        # Verify shapes match
        assert x1_batch.shape == (batch_size, state_dim)
        np.testing.assert_allclose(x1_batch, x1_loop, rtol=1e-4, atol=1e-5)

    def test_batch_shape_correctness(self, doubling_vf, solver):
        """Test batch ODE preserves batch shape.

        Input: (batch, d)
        Output: (batch, d)
        """
        x0_batch = jnp.ones((5, 2))

        x1_batch = batch_solve_forward_ode(
            doubling_vf,
            x0_batch,
            solver,
        )

        assert x1_batch.shape == (5, 2)

    def test_batch_multidim_state(self, doubling_vf, solver):
        """Test batch with multidimensional state.

        Input: (batch, d1, d2)
        Output: (batch, d1, d2)
        """
        x0_batch = jnp.ones((8, 2, 3))

        x1_batch = batch_solve_forward_ode(
            doubling_vf,
            x0_batch,
            solver,
        )

        assert x1_batch.shape == (8, 2, 3)

    def test_batch_vectorization_consistency(self, doubling_vf, solver):
        """Test that batch operations are truly vectorized.

        All samples should be solved independently with same vf.
        """
        seed = jax.random.PRNGKey(456)
        x0_batch = jax.random.normal(seed, (100, 5))

        x1_batch = batch_solve_forward_ode(
            doubling_vf,
            x0_batch,
            solver,
        )

        # For doubling flow, x1 = x0 * 2
        # (analytical solution: e^(log(2) * t) * x0 = 2^t * x0)
        expected = x0_batch * 2.0

        np.testing.assert_allclose(x1_batch, expected, rtol=0.01)


class TestBatchSolveAugmentedODE:
    """Test batch augmented ODE solving with vmap."""

    def test_batch_augmented_vs_loop(self, solver):
        """Test batch_solve_augmented gives same results as loop."""
        def linear_vf(x, t):
            A = jnp.array([[2.0, 0.0], [0.0, 1.0]])
            return A @ x

        seed = jax.random.PRNGKey(789)
        batch_size = 8
        state_dim = 2

        x0_batch = jax.random.normal(seed, (batch_size, state_dim))

        # Batch augmented solve
        x_final_batch, log_det_batch = batch_solve_augmented_ode(
            linear_vf,
            x0_batch,
            solver,
            rtol=1e-5,
            atol=1e-5,
        )

        # Loop augmented solve for comparison
        x_final_loop = []
        log_det_loop = []
        for x0 in x0_batch:
            x_f, ld = solve_augmented_ode(
                linear_vf,
                x0,
                solver,
                rtol=1e-5,
                atol=1e-5,
            )
            x_final_loop.append(x_f)
            log_det_loop.append(ld)

        x_final_loop = jnp.array(x_final_loop)
        log_det_loop = jnp.array(log_det_loop)

        # Verify shapes and values
        assert x_final_batch.shape == (batch_size, state_dim)
        assert log_det_batch.shape == (batch_size,)
        np.testing.assert_allclose(
            x_final_batch,
            x_final_loop,
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            log_det_batch,
            log_det_loop,
            rtol=1e-3,
            atol=1e-4,
        )

    def test_batch_augmented_output_shapes(self, solver):
        """Test batch augmented ODE output shapes."""
        def vf(x, t):
            return -x

        x0_batch = jnp.ones((10, 4))

        x_final, log_det = batch_solve_augmented_ode(
            vf,
            x0_batch,
            solver,
        )

        assert x_final.shape == (10, 4)
        assert log_det.shape == (10,)

    def test_batch_augmented_multidim_state(self, solver):
        """Test batch augmented ODE with multidimensional state.

        Input: (batch, d1, d2)
        Output: (batch, d1, d2) and (batch,)
        """
        def vf(x, t):
            return 0.1 * x

        x0_batch = jnp.ones((6, 3, 2))

        x_final, log_det = batch_solve_augmented_ode(
            vf,
            x0_batch,
            solver,
        )

        assert x_final.shape == (6, 3, 2)
        assert log_det.shape == (6,)
