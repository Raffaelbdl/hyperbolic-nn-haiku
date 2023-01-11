import jax.numpy as jnp

from haiku_hnn.core.manifolds.base import Manifold


class Lorentz(Manifold):
    def inner(self, x: jnp.ndarray, u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        xu = x * u
        time = -jnp.sum(xu[:, :1], axis=-1, keepdims=keepdims)  # B, 1
        space = jnp.sum(xu[:, 1:], axis=-1, keepdims=keepdims)  # B, n
        return time + space

    def dist(self, x: jnp.ndarray, y: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        d = -self.inner(x, y, True)
        return jnp.sqrt(-self.k) * jnp.arccosh(d / self.k)

    def inner0(self, u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        time = -jnp.sum(u[:, :1], axis=-1, keepdims=keepdims)
        return time

    def expmap(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        alpha = jnp.sqrt(-self.k) * jnp.sqrt(self.inner(u, u, True))
        return jnp.cosh(alpha) * x + jnp.sinh(alpha) * u / alpha

    def expmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        x = self.get_origin_like(u)
        return self.expmap(x, u)

    def logmap(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        beta = self.inner(x, u, True)
        d = jnp.arccosh(-beta)
        nomin = u + 1.0 / self.k * beta * x
        # denom = jnp.sqrt(jnp.square(beta) - 1.0) # said as irrelevant in code, despite being written in paper
        denom = jnp.sqrt(self.inner(nomin, nomin, True))
        return d * nomin / denom

    def logmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        x = self.get_origin_like(u)
        return self.logmap(x, u)

    def proj(self, u: jnp.ndarray, eps: float) -> jnp.ndarray:
        space = u[:, 1:]
        time = jnp.sqrt(self.k + jnp.sum(jnp.square(space), axis=-1, keepdims=True))
        return jnp.concatenate([time, space], axis=-1)

    def get_origin_like(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [jnp.ones_like((u[:, :1]), u.dtype), jnp.zeros_like((u[:, 1:]), u.dtype)],
            axis=-1,
        )

    def parallel_transport(
        self, x: jnp.ndarray, y: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        lmap = self.logmap(x, y)
        rmap = self.logmap(y, x)

        nomin = self.inner(x, u, True)
        denom = jnp.square(self.dist(x, y))

        res = u - nomin / denom * (lmap + rmap)
        return res
