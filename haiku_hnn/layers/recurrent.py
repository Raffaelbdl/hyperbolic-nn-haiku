from typing import Optional

import haiku as hk
from jax import nn
from jax import numpy as jnp

from haiku_hnn.activation import map_activation, map_relu
from haiku_hnn.core import m_add, logmap_zero, m_matrix_vector_multiplication
from haiku_hnn.layers.linear import RemappingLinear


class MappingVanillaRNN(hk.VanillaRNN):
    """Hyperbolic VanillaRNN module with remapping

    based on Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)
    """

    def __init__(
        self,
        hidden_size: int,
        c: float,
        double_bias: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(hidden_size, double_bias, name)
        self.c = c

    def __call__(self, inputs, prev_state):
        input_to_hidden = RemappingLinear(self.hidden_size, self.c)
        hidden_to_hidden = RemappingLinear(
            self.hidden_size, self.c, with_bias=self.double_bias
        )
        # arbitrary order for Möbius addition used here
        pre_nonlinearity = m_add(hidden_to_hidden(prev_state), input_to_hidden(inputs))
        out = map_relu(pre_nonlinearity, self.c)
        return out, out


class MappingGRU(hk.GRU):
    """Hyperbolic GRU module with remapping

    based on Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)
    """

    def __init__(
        self,
        hidden_size: int,
        c: float,
        w_i_init: Optional[hk.initializers.Initializer] = None,
        w_h_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(hidden_size, w_i_init, w_h_init, b_init, name)
        self.c = c

    def __call__(self, inputs, state):
        # naive implementation from the paper
        # TODO copy haiku native implementation

        if inputs.ndim not in (1, 2):
            raise ValueError("GRU inputs must be rank-1 or rank-2.")

        r_input_to_hidden = RemappingLinear(self.hidden_size, self.c)
        r_hidden_to_hidden = RemappingLinear(self.hidden_size, self.c, with_bias=False)

        z_input_to_hidden = RemappingLinear(self.hidden_size, self.c)
        z_hidden_to_hidden = RemappingLinear(self.hidden_size, self.c, with_bias=False)

        r = m_add(r_hidden_to_hidden(state), r_input_to_hidden(inputs))
        r = nn.sigmoid(logmap_zero(r, self.c))

        z = m_add(z_hidden_to_hidden(state), z_input_to_hidden(inputs))
        z = nn.sigmoid(logmap_zero(z, self.c))

        h_tilt_input_to_hidden = RemappingLinear(self.hidden_size, self.c)
        # TODO add args in get_parameter
        h_tilt_hidden_to_hidden = hk.get_parameter(
            "riemannian_u", [self.hidden_size, self.hidden_size]
        )
        h_tilt_1 = m_matrix_vector_multiplication(
            jnp.dot(h_tilt_hidden_to_hidden, jnp.diag(r)), state
        )
        h_tilt_2 = h_tilt_input_to_hidden(inputs)
        h_tilt = map_activation(m_add(h_tilt_1, h_tilt_2), self.c, nn.tanh)

        h = m_add(
            state, m_matrix_vector_multiplication(jnp.diag(z)), m_add(-state, h_tilt)
        )

        return h, h
