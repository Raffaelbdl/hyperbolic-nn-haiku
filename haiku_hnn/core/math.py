from jax import numpy as jnp


def tanh(x):
    return jnp.tanh(jnp.clip(x, -15, 15))


def arctanh(x):
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
        return jnp.power(jnp.abs(k), -0.5) * jnp.tan(x)
    elif k < 0:
        return jnp.power(jnp.abs(k), -0.5) * tanh(x)
    else:
        raise NotImplementedError("K = 0 is not implemented")


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
        return jnp.arctan(jnp.sqrt(jnp.abs(k)) * y)
    elif k < 0:
        return arctanh(jnp.sqrt(jnp.abs(k)) * y)
    else:
        raise NotImplementedError("K = 0 is not implemented")
