from typing import Optional

from jax import lax
from jax import numpy as jnp


def conformal_factor(x: jnp.ndarray) -> float:
    """Computes the conformal factor"""
    return 2 / (1 - jnp.sum(jnp.square(x), axis=-1))


def exponential_mapping_in_zero(v: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes the exponential mapping in zero
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (13)
    """
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.tanh(jnp.sqrt(c) * v_norm) * v / jnp.sqrt(c) / v_norm


def logarithmic_mapping_in_zero(y: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes the logarithmic mapping in zero
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (13)
    """
    y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    return jnp.arctanh(jnp.sqrt(c) * y_norm) * y / jnp.sqrt(c) / y_norm


def mobius_matrix_vector_multiplication(
    w: jnp.ndarray, x: jnp.ndarray, c: float, precision: Optional[lax.Precision] = None
) -> jnp.ndarray:
    """Computes the mobius matrix vector multiplication
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (27)
    """
    x_dot_w = jnp.dot(x, w, precision=precision)
    x_dot_w_norm = jnp.linalg.norm(x_dot_w, axis=-1, keepdims=True)
    x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)

    res = 1 / jnp.sqrt(c)
    res *= jnp.tanh(x_dot_w_norm / x_norm * jnp.arctanh(jnp.sqrt(c) * x_norm))
    res *= x_dot_w / x_dot_w_norm
    return res


def mobius_bias_translation(b: jnp.ndarray, x: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes the mobius bias translation
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (28)
    """
    return exponential_mapping_in_zero(
        2 * logarithmic_mapping_in_zero(b, c=c) / conformal_factor(x), c=c
    )
