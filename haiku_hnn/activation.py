from typing import Callable, Optional, Tuple, Union

from jax import nn
from jax import numpy as jnp
from haiku_hnn.core import expmap_zero, logmap_zero


def with_logexpmap(f, c: float):
    """Decorator to map functions in hyperbolic space"""

    def mapper(*args, **kwargs):
        v = logmap_zero(args[0], c)
        v = f(v, **kwargs)
        v = expmap_zero(v, c)
        return v

    return mapper


def map_activation(
    x: jnp.ndarray,
    c: float,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    **kwargs,
) -> jnp.ndarray:
    return with_logexpmap(activation, c)(x, **kwargs)


def map_relu(x: jnp.ndarray, c: float):
    return map_activation(x, c, nn.relu)


def map_softmax(
    x: jnp.ndarray, c: float, axis: Optional[Union[int, Tuple[int, ...]]] = -1
):
    return map_activation(x, c, nn.softmax, axis=axis)
