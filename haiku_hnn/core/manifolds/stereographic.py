import jax.numpy as jnp

from haiku_hnn.core.manifolds.base import Manifold


def safe_tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(jnp.clip(x, -15, 15))


def safe_arctanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctanh(jnp.clip(x, -1 + 1e-7, 1 - 1e-7))


class Stereographic(Manifold):
    def inner(x: jnp.ndarray, u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        raise NotImplementedError()

    def inner0(u: jnp.ndarray, keepdims: bool) -> jnp.ndarray:
        raise NotImplementedError()

    def expmap(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        norm_u = jnp.sqrt(jnp.sum(jnp.square(u), axis=-1, keepdims=True)) + 1e-15
        conf_factor = self.conformal_factor(x)

        transf_u = self._tan_k(jnp.sqrt(jnp.abs(self.k)) * conf_factor * norm_u / 2.0)
        transf_u *= u / norm_u

        return self._mobius_add(x, transf_u)

    def expmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        u += 1e-15
        norm_u = jnp.sqrt(jnp.sum(jnp.square(u), axis=-1, keepdims=True))
        return self._tan_k(norm_u) * (u / norm_u)

    def logmap(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        mx_add_u = self._mobius_add(-x, u)
        norm_mx_add_u = jnp.sqrt(jnp.sum(jnp.square(mx_add_u), axis=-1, keepdims=True))
        conf_factor = self._conformal_factor(x)

        res = 2 * jnp.power(jnp.abs(self.k), -0.5) / conf_factor
        res *= self._arctan_k(norm_mx_add_u)
        res *= mx_add_u / norm_mx_add_u

        return res

    def logmap0(self, u: jnp.ndarray) -> jnp.ndarray:
        u += 1e-15
        norm_u = jnp.sqrt(jnp.sum(jnp.square(u), axis=-1, keepdims=True))
        return self._arctan_k(norm_u) * (u / norm_u)

    def proj(self, u: jnp.ndarray, eps: float) -> jnp.ndarray:
        max_norm = (1 - eps) / jnp.power(jnp.abs(self.k), 0.5)
        norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=-1, keepdims=True))

        cond = norm > max_norm
        return jnp.where(cond, 1.0 / norm * max_norm, 1.0) * u

    def get_origin_like(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.proj(self.expmap0(jnp.zeros_like(u)), 4e-3)

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

    def _mobius_bias(self, u: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return self.expmap(u, 2.0 / self.conformal_factor(u) * self.logmap0(b))

    def _mobius_scale(self, u: jnp.ndarray, s: float) -> jnp.ndarray:
        norm_u = jnp.sqrt(jnp.sum(jnp.square(u), axis=-1, keepdims=True))
        return self._tan_k(s * self._arctan_k(norm_u)) * u / norm_u

    def conformal_factor(self, u: jnp.ndarray) -> jnp.ndarray:
        u = self.proj(u, 4e-3)
        return 2.0 / (1.0 + self.k * jnp.sum(jnp.square(u), axis=-1, keepdims=True))

    def _tan_k(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.k > 0:
            return jnp.power(self.k, -0.5) * jnp.tan(x)
        elif self.k < 0:
            return jnp.power(-self.k, -0.5) * safe_tanh(x)
        else:
            return jnp.tan(x)

    def _arctan_k(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.k > 0:
            return jnp.arctan(jnp.sqrt(self.k) * x)
        elif self.k < 0:
            return safe_arctanh(jnp.sqrt(-self.k) * x)
        else:
            return jnp.arctan(x)

    def norm(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Computes the norm in the K-stereographic model

        Reference:
            Constant Curvature Graph Convolutional Networks
                (https://arxiv.org/pdf/1911.05076v1.pdf)

        Args:
            x (jnp.ndarray): the point where the norm is computed
            k (float): the curvature of the manifold

            axis (int): the axis along which the sum is performed
            keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
        """
        return self.conformal_factor(x) * jnp.sqrt(
            jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        )

    def _gyration(self, u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """Computes the gyration vector of w by [u, v]

        Reference:
            Constant Curvature Graph Convolutional Networks
                (https://arxiv.org/pdf/1911.05076v1.pdf)

        Args:
            u, v, w (jnp.ndarray): the points where gyr[u, v]w is computed
            k (float): the curvature of the manifold

            axis (int): the axis along which the sum is performed
            keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
        """
        u2 = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        v2 = jnp.sum(jnp.square(v), axis=-1, keepdims=True)
        uv = jnp.sum(u * v, axis=-1, keepdims=True)
        uw = jnp.sum(u * w, axis=-1, keepdims=True)
        vw = jnp.sum(v * w, axis=-1, keepdims=True)
        k2 = jnp.square(self.k)

        a = -k2 * uw * v2 - self.k * vw + 2 * k2 * uv * vw
        b = -k2 * vw * u2 + self.k * uw
        d = 1 - 2 * self.k * uv + k2 * u2 * v2
        return w + 2 * (a * u + b * v) / jnp.maximum(d, 1e-15)

    def parallel_transport(
        self, x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the parallel transport between two tangent spaces

        Reference:
            Constant Curvature Graph Convolutional Networks
                (https://arxiv.org/pdf/1911.05076v1.pdf)

        Args:
            x (jnp.ndarray): the first tangent space
            y (jnp.ndarray): the second tangent space
            v (jnp.ndarray): the point to transport
            k (float): the curvature of the manifold

        Returns:
            The point v after parallel transport from tangent space of x to y
        """
        return (
            self._gyration(y, -x, v)
            * self.conformal_factor(x)
            / self.conformal_factor(y)
        )
