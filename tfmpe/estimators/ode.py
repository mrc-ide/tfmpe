"""ODE solving helpers for flow matching estimators."""

from typing import Callable, Tuple

from jaxtyping import Array, Scalar


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
    raise NotImplementedError()


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
    raise NotImplementedError()


def solve_augmented_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x0: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
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

    Returns
    -------
    Tuple[Array, Scalar]
        (final_x, final_log_det_jacobian)
    """
    raise NotImplementedError()


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
    raise NotImplementedError()


def batch_solve_augmented_ode(
    vf_fn: Callable[[Array, Scalar], Array],
    x0_batch: Array,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
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

    Returns
    -------
    Tuple[Array, Array]
        (final_x_batch, final_log_det_jacobian)
        - final_x_batch: shape (batch, *state_shape)
        - final_log_det_jacobian: shape (batch,)
    """
    raise NotImplementedError()
