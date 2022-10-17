from typing import Optional

import haiku as hk
from jax import lax
from jax import numpy as jnp
import numpy as np

from haiku_hnn.src.core import (
    mobius_bias_translation,
    mobius_matrix_vector_multiplication,
)


class RemappingLinear(hk.Linear):
    """Hyperbolic Linear module with remapping"""

    def __init__(
        self,
        output_size: int,
        c: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(output_size, with_bias, w_init, b_init, name)
        self.c = c

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
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        out = mobius_matrix_vector_multiplication(
            w, inputs, self.c, precision=precision
        )

        if self.with_bias:
            b = hk.get_parameter("hyperbolic-b", [output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = mobius_bias_translation(b, out, self.c)

        return out
