from typing import Optional

import haiku as hk
from jax import lax
from jax import numpy as jnp
import numpy as np

from haiku_hnn.core.stereographic import m_bias, m_dot, m_add


class StereographicLinear(hk.Linear):
    """Linear Module in K-stereographic model

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    # TODO Change Attribute section
    Non-inherited attributes:
        k (float): the curvature of the manifold
    """

    def __init__(
        self,
        output_size: int,
        k: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(output_size, with_bias, w_init, b_init, name)
        self.k = k

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Computes a hyperbolic linear transform of the input"""
        if not inputs.shape:
            raise ValueError("Input must not be a scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter(
            "riemannian_w", [input_size, output_size], dtype, init=w_init
        )

        out = m_dot(inputs, w, self.k, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("riemannian_b", [output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = m_bias(b, out, self.k)

        return out


class StereographicConcatLinear(hk.Linear):
    def __init__(
        self,
        output_size: int,
        k: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(output_size, with_bias, w_init, b_init, name)
        self.k = k

    def __call__(
        self,
        inputs1: jnp.ndarray,
        inputs2: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Computes a hyperbolic linear transform of the input"""
        if not inputs1.shape or not inputs2.shape:
            raise ValueError("Input must not be a scalar.")

        input_size = self.input_size = inputs1.shape[-1] + inputs2.shape[-1]
        output_size = self.output_size
        dtype = inputs1.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter(
            "riemannian_w", [input_size, output_size], dtype, init=w_init
        )
        w_1, w_2 = jnp.split(w, [inputs1.shape[-1]], axis=0)
        out = m_add(
            m_dot(inputs1, w_1, self.k, precision=precision),
            m_dot(inputs2, w_2, self.k, precision=precision),
            self.k,
        )

        if self.with_bias:
            b = hk.get_parameter("riemannian_b", [output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = m_bias(b, out, self.k)

        return out
