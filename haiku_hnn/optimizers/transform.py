"""Gradient transformations in Riemannian space, adaptation of optax.transform"""

from typing import Any, NamedTuple, Optional

import chex
import jax
from jax import numpy as jnp
import optax
from optax._src.utils import canonicalize_dtype, cast_tree

from haiku_hnn.core import conformal_factor, parallel_transport, norm
from haiku_hnn.optimizers.update import apply_riemannian_updates, label_riemannian_fn


def mixed_optimizer(
    non_hyperbolic_optimizer: optax.GradientTransformation,
    hyperbolic_optimizer: optax.GradientTransformation,
):
    return optax.multi_transform(
        {"riemannian": hyperbolic_optimizer, "euclidian": non_hyperbolic_optimizer},
        label_riemannian_fn,
    )


def riemannian_scale() -> optax.GradientTransformation:
    """Rescale gradients to riemannian space

    References:
        [Bécigneul and Ganea, 2019](http://arxiv.org/abs/1810.00760)
    """

    def init_fn(params):
        del params
        return optax.ScaleState()

    def update_fn(updates, state, params):
        updates = jax.tree_util.tree_map(
            lambda g, p: g / conformal_factor(p**2), updates, params
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def riemannian_trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    """Compute a trace of past updates.

    References:
        [Bonnabel, 2013](https://arxiv.org/abs/1111.5280)

    Args:
        decay: Decay rate for the trace of past updates.
        nesterov: Whether to use Nesterov momentum.
        accumulator_dtype: Optional `dtype` to be used for the accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
    Returns:
        A `GradientTransformation` object.
    """

    accumulator_dtype = canonicalize_dtype(accumulator_dtype)

    def init_fn(params):
        return optax.TraceState(
            trace=jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params
            )
        )

    def update_fn(updates, state: optax.TraceState, params):
        """Params are expected to have been riemannian scaled before"""
        f = lambda g, t: g + decay * t
        new_trace = jax.tree_util.tree_map(f, updates, state.trace)
        updates = (
            jax.tree_util.tree_map(f, updates, new_trace) if nesterov else new_trace
        )
        # calculate new params for updating purpose only
        new_params = apply_riemannian_updates(params, updates)
        new_trace = jax.tree_util.tree_map(
            lambda p, new_p, new_t: parallel_transport(p, new_p, new_t),
            params,
            new_params,
            new_trace,
        )

        return updates, optax.TraceState(trace=new_trace)

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByRAdamState(NamedTuple):
    """State for the Riemannian Adam algorithm"""

    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates
    tau: optax.Updates


def riemannian_scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the Riemannian Adam algorithm.

    References:
        [Bécigneul and Ganea, 2019](http://arxiv.org/abs/1810.00760)

    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
        mu_dtype: optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype is inferred from `params` and `updates`.

    Returns:
        A `GradientTransformation` object.
    """

    mu_dtype = canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        tau = jax.tree_util.tree_map(jnp.zeros_like, params)  # Translated first moment
        return ScaleByRAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, tau=tau)

    def update_fn(updates, state: ScaleByRAdamState, params):
        mu = optax.update_moment(updates, state.tau, b1, 1)
        square_norm_updates = jax.tree_util.tree_map(
            lambda g, p: norm(p, g) ** 2, updates, params
        )
        nu = optax.update_moment(square_norm_updates, state.nu, b2, 2)
        count_inc = optax.safe_int32_increment(state.count)
        mu_hat = optax.bias_correction(mu, b1, count_inc)
        nu_hat = optax.bias_correction(nu, b2, count_inc)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )
        mu = cast_tree(mu, mu_dtype)

        new_params = apply_riemannian_updates(params, updates)
        tau = jax.tree_util.tree_map(
            lambda p, new_p, m: parallel_transport(p, new_p, m), params, new_params, mu
        )

        return updates, ScaleByRAdamState(count=count_inc, mu=mu, nu=nu, tau=tau)

    return optax.GradientTransformation(init_fn, update_fn)


def riemannian_scale_by_rss(
    initial_accumulator_value: float = 0.1, eps: float = 1e-7
) -> optax.GradientTransformation:
    """Rescale updates by the root of the sum of all squared gradients norms to date.

    References:
        [Bécigneul and Ganea, 2019](http://arxiv.org/abs/1810.00760)

    Args:
        initial_accumulator_value: Starting value for accumulators, must be >= 0.
        eps: A small floating point value to avoid zero denominator.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        sum_of_squares = jax.tree_util.tree_map(
            lambda t: jnp.full_like(t, initial_accumulator_value), params
        )
        return optax.ScaleByRssState(sum_of_squares=sum_of_squares)

    def update_fn(updates, state: optax.ScaleByRssState, params):
        sum_of_squares = jax.tree_util.tree_map(
            lambda g, t, p: norm(p, g) + t, updates, state.sum_of_squares, params
        )
        inv_sqrt_g_square = jax.tree_util.tree_map(
            lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares
        )
        updates = jax.tree_util.tree_map(
            lambda scale, g: scale * g, inv_sqrt_g_square, updates
        )
        return updates, optax.ScaleByRssState(sum_of_squares=sum_of_squares)

    return optax.GradientTransformation(init_fn, update_fn)
