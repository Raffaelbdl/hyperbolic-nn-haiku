from typing import Callable, Optional, Tuple, Union

from jax import nn
from jax import numpy as jnp

from haiku_hnn.src.core import exponential_mapping_in_zero, logarithmic_mapping_in_zero


def remapping_activation(
    x: jnp.ndarray,
    c: float,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    **kwargs,
) -> jnp.ndarray:
    v = logarithmic_mapping_in_zero(x, c)
    v = activation(v, **kwargs)
    return exponential_mapping_in_zero(v, c)


def remapping_relu(x: jnp.ndarray, c: float):
    return remapping_activation(x, c, nn.relu)


def remapping_softmax(
    x: jnp.ndarray, c: float, axis: Optional[Union[int, Tuple[int, ...]]] = -1
):
    return remapping_activation(x, c, nn.softmax, axis=axis)
