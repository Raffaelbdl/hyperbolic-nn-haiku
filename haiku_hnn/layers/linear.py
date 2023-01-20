from typing import Optional

import haiku as hk
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np

from haiku_hnn.core.manifolds.stereographic import Stereographic
from haiku_hnn.core.manifolds.lorentz import Lorentz
from haiku_hnn.initializers import HyperbolicInitializer


def get_scalar(
    name: str, value: float, is_learnable: bool, dtype: jnp.dtype
) -> jnp.ndarray:
    """Returns a scalar inside hk.transform

    If the scalar is not supposed to be learned, it will return its default value.
    Otherwise the scalar will be added to the parameters to be optimized
    """
    if is_learnable:
        return hk.get_parameter(
            f"riemannian_{name}", [], dtype, lambda *args: jnp.array(value, dtype)
        )
    return value


class StereographicLinear(hk.Linear):
    """Linear Module in K-stereographic model

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)
    """

    def __init__(
        self,
        output_size: int,
        k: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        learnable: bool = False,
    ):
        super().__init__(output_size, with_bias, w_init, b_init, name)
        self.manifold = Stereographic(k)
        self.learnable = learnable

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

        k = self.manifold.k = get_scalar("k", self.manifold.k, self.learnable, dtype)

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = HyperbolicInitializer(
                hk.initializers.TruncatedNormal(stddev=stddev), self.manifold
            )
        w = hk.get_parameter(
            "riemannian_w", [input_size, output_size], dtype, init=w_init
        )

        out = self.manifold._mobius_dot(inputs, w, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("riemannian_b", [output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = self.manifold._mobius_bias(out, b)

        return self.manifold.proj(out, 4e-3)


class LorentzLinear(hk.Linear):
    """Lorentz Module in the Lorentz model

    References:
        Fully Hyperbolic Neural Networks (https://arxiv.org/abs/2105.14686)
    """

    def __init__(
        self,
        output_size: int,
        k: float,
        scale: float,
        eps: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        learnable_k: bool = False,
        learnable_scale: bool = False,
    ):
        super().__init__(output_size, with_bias, w_init, b_init, name)
        self.manifold = Lorentz(k)
        self.scale = scale
        self.eps = eps

        self.learnable_k = learnable_k
        self.learnable_scale = learnable_scale

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Computes a hyperbolic linear transform of the input"""
        if not inputs.shape:
            raise ValueError("Input must not be a scalar")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype
        eps = self.eps

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = HyperbolicInitializer(
                hk.initializers.TruncatedNormal(stddev=stddev), self.manifold
            )
        w = hk.get_parameter(
            "riemannian_w", [input_size, output_size], dtype, init=w_init
        )
        wx = jnp.dot(inputs, w, precision=precision)

        scale = get_scalar("scale", self.scale, self.learnable_scale, dtype)

        time = (
            scale * jax.nn.sigmoid(wx[..., :1]) + eps
        )  # should be x @ v + b inside sigmoid

        if self.with_bias:
            b = hk.get_parameter("riemannian_b", [output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, wx.shape)
            wx = wx + b
        k = self.manifold.k = get_scalar("k", self.manifold.k, self.learnable_k, dtype)

        space = wx[..., 1:]
        space = (
            jnp.sqrt(
                (jnp.square(time) + 1 / k)
                / jnp.sum(jnp.square(space), axis=-1, keepdims=True)
            )
            * space
        )

        return self.manifold.proj(jnp.concatenate([time, space], axis=-1), 4e-3)
