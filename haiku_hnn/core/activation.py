from typing import Callable

from jax import nn
from jax import numpy as jnp

from haiku_hnn.core.manifolds.base import Manifold
from haiku_hnn.core.manifolds.stereographic import Stereographic


def k_fn(manifold: Manifold, fn: Callable) -> Callable:
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
    if manifold.k == 0:
        return fn

    if isinstance(manifold, Stereographic):
        eps = 4e-3
    else:
        eps = 0.0

    def wrapper(x: jnp.ndarray, **kwargs):
        return manifold.proj(manifold.expmap0(fn(manifold.logmap0(x), **kwargs)), eps)

    return wrapper


k_relu = lambda x, manifold, **kwargs: k_fn(manifold, nn.relu)(x, **kwargs)
k_softmax = lambda x, manifold, **kwargs: k_fn(manifold, nn.softmax)(x, **kwargs)
k_tanh = lambda x, manifold, **kwargs: k_fn(manifold, nn.tanh)(x, **kwargs)
