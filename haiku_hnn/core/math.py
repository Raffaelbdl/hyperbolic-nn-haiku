from jax import numpy as jnp


def tan_k(x: jnp.ndarray, k: float) -> jnp.ndarray:
    return jnp.power(jnp.abs(k), -0.5) * jnp.tan(x)


def arctan_k(y: jnp.ndarray, k: float) -> jnp.ndarray:
    return jnp.arctan(jnp.sqrt(jnp.abs(k)) * y)
