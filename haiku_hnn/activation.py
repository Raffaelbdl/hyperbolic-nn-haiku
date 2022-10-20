from typing import Callable, Optional, Tuple, Union

from jax import nn
from jax import numpy as jnp
from haiku_hnn.core import expmap_zero, logmap_zero


def map_activation(
    x: jnp.ndarray,
    c: float,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    **kwargs,
) -> jnp.ndarray:
    v = logmap_zero(x, c)
    v = activation(v, **kwargs)
    return expmap_zero(v, c)


def map_relu(x: jnp.ndarray, c: float):
    return map_activation(x, c, nn.relu)


def map_softmax(
    x: jnp.ndarray, c: float, axis: Optional[Union[int, Tuple[int, ...]]] = -1
):
    return map_activation(x, c, nn.softmax, axis=axis)
