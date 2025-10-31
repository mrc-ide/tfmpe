"""ODE solving helpers for flow matching estimators."""

from typing import Callable, Tuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar, PRNGKeyArray


def solve_forward_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x0: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Array:
    """Solve forward ODE from t=0 to t=1 for sampling.

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(x, t) -> Array
    x0 : Array
        Initial state, shape (*state_shape)
    solver : diffrax solver
        Solver instance (e.g., Diffrax Dopri5)
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver

    Returns
    -------
    Array
        Final state at t=1, shape (*x0.shape)
    """
    def ode_func(t: Scalar, y: Array, args) -> Array:
        return vf_fn(y, t)

    step_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_func),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=x0,
        stepsize_controller=step_controller,
        saveat=diffrax.SaveAt(t1=True),
    )

    return solution.ys[0]


def solve_backward_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x1: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Array:
    """Solve backward ODE from t=1 to t=0 for log probability.

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(x, t) -> Array
    x1 : Array
        Final state at t=1, shape (*state_shape)
    solver : diffrax solver
        Solver instance
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver

    Returns
    -------
    Array
        Initial state at t=0, shape (*x1.shape)
    """
    def ode_func(t: Scalar, y: Array, args) -> Array:
        # Negate for backward direction (t: 1 -> 0)
        return -vf_fn(y, 1.0 - t)

    step_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_func),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=x1,
        stepsize_controller=step_controller,
        saveat=diffrax.SaveAt(t1=True),
    )

    return solution.ys[0]


def solve_augmented_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x0: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    rng: PRNGKeyArray = None,
    n_epsilon: int = 1,
) -> Tuple[Array, Scalar]:
    """Solve augmented ODE with trace-based log determinant.

    Augmented state: [x, log_det_jacobian]

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(x, t) -> Array
    x0 : Array
        Initial state, shape (*state_shape)
    solver : diffrax solver
        Solver instance
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver
    rng : PRNGKeyArray, optional
        PRNG key for sampling epsilon. If None, uses PRNGKey(0)
    n_epsilon : int
        Number of trace samples for Hutchinson estimator

    Returns
    -------
    Tuple[Array, Scalar]
        (final_x, final_log_det_jacobian)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Pre-sample epsilon array for the ODE
    epsilon_array = jax.random.normal(
        rng, (n_epsilon,) + x0.shape
    )

    def augmented_ode_func(
        t: Scalar,
        aug_state: Tuple[Array, Scalar],
        args,
    ) -> Tuple[Array, Scalar]:
        x, log_det = aug_state

        # Compute vector field
        def vf_wrapper(x_inner):
            return vf_fn(x_inner, t)

        # Compute trace via stochastic VJP
        # tr(∂f/∂x) ≈ mean_i[eps_i^T @ (∂f/∂x)^T @ eps_i]
        _, vjp_fn = jax.vjp(vf_wrapper, x)

        def compute_trace_single(eps):
            g = vjp_fn(eps)[0]
            return jnp.sum(g * eps)

        # Average over all epsilon samples
        trace_estimates = jax.vmap(
            compute_trace_single, in_axes=0
        )(epsilon_array)
        trace_estimate = jnp.mean(trace_estimates)

        dx_dt = vf_wrapper(x)
        dlog_det_dt = trace_estimate

        return (dx_dt, dlog_det_dt)

    # Initial augmented state
    aug_y0 = (x0, jnp.array(0.0))

    step_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(augmented_ode_func),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=aug_y0,
        stepsize_controller=step_controller,
        saveat=diffrax.SaveAt(t1=True),
    )

    # solution.ys is a tuple (x_trajectory, log_det_trajectory)
    # Each has shape (num_saves, *state_shape) or (num_saves,)
    # With saveat=SaveAt(t1=True), num_saves=1
    final_x = solution.ys[0][0]
    final_log_det = solution.ys[1][0]
    return final_x, final_log_det


def batch_solve_forward_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x0_batch: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Array:
    """Solve forward ODE for batch of samples using vmap.

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(x, t) -> Array
    x0_batch : Array
        Batch of initial states, shape (batch, *state_shape)
    solver : diffrax solver
        Solver instance
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver

    Returns
    -------
    Array
        Batch of final states, shape (batch, *state_shape)
    """
    return jax.vmap(
        lambda x0: solve_forward_ode(
            vf_fn, x0, solver, rtol=rtol, atol=atol
        )
    )(x0_batch)


def batch_solve_augmented_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x0_batch: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    rng: PRNGKeyArray = None,
    n_epsilon: int = 1,
) -> Tuple[Array, Array]:
    """Solve augmented ODE for batch of samples using vmap.

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(x, t) -> Array
    x0_batch : Array
        Batch of initial states, shape (batch, *state_shape)
    solver : diffrax solver
        Solver instance
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver
    rng : PRNGKeyArray, optional
        PRNG key for sampling epsilon
    n_epsilon : int
        Number of trace samples for Hutchinson estimator

    Returns
    -------
    Tuple[Array, Array]
        (final_x_batch, final_log_det_jacobian)
        - final_x_batch: shape (batch, *state_shape)
        - final_log_det_jacobian: shape (batch,)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Split RNG for each batch element
    rngs = jax.random.split(rng, x0_batch.shape[0])

    x_batch, log_det_batch = jax.vmap(
        lambda x0, rng_: solve_augmented_ode(
            vf_fn,
            x0,
            solver,
            rtol=rtol,
            atol=atol,
            rng=rng_,
            n_epsilon=n_epsilon,
        )
    )(x0_batch, rngs)
    return x_batch, log_det_batch
