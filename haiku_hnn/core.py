from typing import Optional

from jax import lax
from jax import numpy as jnp
from optax._src.numerics import safe_norm


def m_add(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Computes the Möbius Addition"""
    dot = jnp.dot(a, b)
    norm_square_a = jnp.sum(jnp.square(a))
    norm_square_b = jnp.sum(jnp.square(b))

    num = (1 + 2 * dot + norm_square_b) * a + (1 - norm_square_a) * b
    denom = 1 + 2 * dot + norm_square_a * norm_square_b

    return num / denom


def conformal_factor(x: jnp.ndarray) -> float:
    """Computes the conformal factor"""
    return 2 / (1 - safe_norm(x, 0) ** 2)


def expmap(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    conf_factor = conformal_factor(x)
    norm_v = jnp.linalg.norm(v)

    return m_add(x, jnp.tanh(conf_factor * norm_v / 2) * v / norm_v)


def expmap_zero(v: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes the exponential mapping in zero
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (13)
    """
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.tanh(jnp.sqrt(c) * v_norm) * v / jnp.sqrt(c) / v_norm


def logmap_zero(y: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes the logarithmic mapping in zero
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (13)
    """
    y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    return jnp.arctanh(jnp.sqrt(c) * y_norm) * y / jnp.sqrt(c) / y_norm


def m_matrix_vector_multiplication(
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


def m_bias_translation(b: jnp.ndarray, x: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes the mobius bias translation
    as defined in Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112) in equation (28)
    """
    return expmap_zero(2 * logmap_zero(b, c=c) / conformal_factor(x), c=c)


def expmap_firstorder(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    return m_add(x, v)


def parallel_transport(x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    return gyration(y, -x, v) * conformal_factor(x) / conformal_factor(y)


def gyration(u, v, w):
    norm_u_square = jnp.sum(jnp.square(u))
    norm_v_square = jnp.sum(jnp.square(v))
    dot_uv = jnp.dot(u, v)
    dot_uw = jnp.dot(u, w)
    dot_vw = jnp.dot(v, w)

    a = -dot_uw * norm_v_square - dot_vw + 2 * dot_uv * dot_vw
    b = -dot_vw * norm_u_square + dot_uw
    d = 1 - 2 * dot_uv + norm_u_square * norm_v_square
    return w + 2 * (a * u + b * v) / (d + 1e-15)


def norm(x, u):
    return conformal_factor(x) * jnp.sqrt(jnp.dot(u, u))
