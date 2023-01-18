import jax.numpy as jnp


class Manifold:
    def __init__(self, k: int = -1) -> None:
        self.k = k

    def inner(x: jnp.ndarray, u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        raise NotImplementedError()

    def inner0(u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        raise NotImplementedError()

    def expmap(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def expmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def logmap(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def logmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def proj(self, x: jnp.ndarray, eps: float) -> jnp.ndarray:
        raise NotImplementedError()

    def get_origin_like(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def conformal_factor(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def parallel_transport(
        self, x: jnp.ndarray, y: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError()

    def norm(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()
