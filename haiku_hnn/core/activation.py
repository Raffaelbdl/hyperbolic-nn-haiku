from typing import Callable

from jax import nn
from jax import numpy as jnp

from haiku_hnn.core.stereographic import expmap0, logmap0


def k_fn(k: float, fn: Callable) -> Callable:
    """Wraps a function to make it compatible in the K-stereographic model

    The given function should have:
        a first argument which is the jnp.ndarray on which it is applied
        kwargs that are specific arguments (eg. axis)

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)

    Args:
        k (float): the curvature of the manifold
        fn (Callable): the function to wrap
    """
    if k == 0:
        return fn

    def wrapper(x: jnp.ndarray, **kwargs):
        return expmap0(fn(logmap0(x, k), **kwargs), k)

    return wrapper


k_relu = lambda x, k, **kwargs: k_fn(k, nn.relu)(x, **kwargs)
k_softmax = lambda x, k, **kwargs: k_fn(k, nn.softmax)(x, **kwargs)
k_tanh = lambda x, k, **kwargs: k_fn(k, nn.tanh)(x, **kwargs)
