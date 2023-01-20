import jax.numpy as jnp
from colorama import Fore

from haiku_hnn.core.manifolds.base import Manifold


class Lorentz(Manifold):
    def __init__(self, k: int = -1) -> None:
        super().__init__(k)
        raise NotImplementedError(
            Fore.RED
            + "\nThe Lorentz Manifold is not properly implemented due to\n"
            + "lack of functional theory. Indeed, a lot of nan appear when\n"
            + "sticking to the theory.\n"
            + "A prototype is nonetheless available as `_Lorentz` class."
            + Fore.RESET
        )


class _Lorentz(Manifold):
    def inner(self, x: jnp.ndarray, u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        xu = x * u
        time = -jnp.sum(xu[..., :1], axis=-1, keepdims=keepdims)  # B, 1
        space = jnp.sum(xu[..., 1:], axis=-1, keepdims=keepdims)  # B, n
        return time + space

    def dist(self, x: jnp.ndarray, y: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        d = -self.inner(x, y, True)
        return jnp.sqrt(-self.k) * jnp.arccosh(d / self.k)

    def inner0(self, u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        time = -jnp.sum(u[..., :1], axis=-1, keepdims=keepdims)
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
        denom = jnp.sqrt(
            jnp.square(beta) - 1.0
        )  # said as irrelevant in code, despite being written in paper

        # denom = jnp.sqrt(self.inner(nomin, nomin, True)) # written in code, despite not working

        return d * nomin / denom

    def logmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        x = self.get_origin_like(u)
        return self.logmap(x, u)

    def proj(self, u: jnp.ndarray, eps: float) -> jnp.ndarray:
        space = u[..., 1:]
        time = jnp.sqrt(self.k + jnp.sum(jnp.square(space), axis=-1, keepdims=True))
        return jnp.concatenate([time, space], axis=-1)

    def get_origin_like(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [
                jnp.sqrt(-1 / self.k) * jnp.ones_like((u[..., :1]), u.dtype),
                jnp.zeros_like((u[..., 1:]), u.dtype),
            ],
            axis=-1,
        )

    def parallel_transport(
        self, x: jnp.ndarray, y: jnp.ndarray, u: jnp.ndarray
    ) -> jnp.ndarray:
        lmap = self.logmap(x, y)
        rmap = self.logmap(y, x)

        nomin = self.inner(x, u, True)
        denom = jnp.square(self.dist(x, y, True))

        res = u - nomin / denom * (lmap + rmap)
        return res

    def _mobius_add(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        u2 = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        v2 = jnp.sum(jnp.square(v), axis=-1, keepdims=True)
        uv = jnp.sum(u * v, axis=-1, keepdims=True)

        nomin = (1.0 - 2.0 * self.k * uv - self.k - v2) * u
        nomin += (1.0 + self.k * u2) * v

        denom = 1.0 - 2.0 * self.k * uv + jnp.square(self.k) * u2 * v2

        return nomin / (denom + 1e-15)

    def _mobius_dot(
        self, u: jnp.ndarray, w: jnp.ndarray, precision=None
    ) -> jnp.ndarray:
        return self.expmap0(jnp.dot(self.logmap0(u), w, precision=precision))

    def conformal_factor(self, u: jnp.ndarray) -> jnp.ndarray:
        if u.ndim > 0:
            return 2.0 / (1.0 + self.k * jnp.sum(jnp.square(u), axis=-1, keepdims=True))
        return 2.0 / (1.0 + self.k * jnp.sum(jnp.square(u), keepdims=True))

    def norm(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        if y.ndim > 1:
            return jnp.sqrt(self.inner(y, y, True)) * self.conformal_factor(x)
        return jnp.sqrt(y * y) * self.conformal_factor(x)
