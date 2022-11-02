from typing import Callable, Optional, Tuple, Union

from jax import nn
from jax import numpy as jnp

from haiku_hnn.core.stereographic import expmap, expmap0, logmap, logmap0


def m_relu(x: jnp.ndarray, k: float) -> jnp.ndarray:
    return expmap0(nn.relu(logmap0(x, k)), k)


def m_softmax(
    x: jnp.ndarray, k: float, axis: Optional[Union[int, Tuple[int, ...]]] = -1
) -> jnp.ndarray:
    return expmap0(nn.softmax(logmap0(x, k), axis=axis), k)


def m_tanh(x: jnp.ndarray, k: float) -> jnp.ndarray:
    return expmap0(nn.tanh(logmap0(x, k)), k)
