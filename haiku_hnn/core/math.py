from jax import numpy as jnp


def safe_tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(jnp.clip(x, -15, 15))


def safe_arctanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctanh(jnp.clip(x, -1 + 1e-7, 1 - 1e-7))


def tan_k(x: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the tan_k function

    References:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the input array
        k (float): the curvature of the manifold
    """
    if k > 0:
        return jnp.power(k, -0.5) * jnp.tan(x)
    elif k < 0:
        return jnp.power(-k, -0.5) * safe_tanh(x)
    else:
        return jnp.tan(x)


def arctan_k(y: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the arctan_k function

    References:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the input array
        k (float): the curvature of the manifold
    """
    if k > 0:
        return jnp.arctan(jnp.sqrt(k) * y)
    elif k < 0:
        return safe_arctanh(jnp.sqrt(-k) * y)
    else:
        return jnp.arctan(y)
