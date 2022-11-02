from typing import Optional

import haiku as hk
from jax import nn
from jax import numpy as jnp

from haiku_hnn.activation import m_relu, m_tanh
from haiku_hnn.core.stereographic import logmap0, m_add, m_dot
from haiku_hnn.layers.linear import StereographicLinear


class StereographicVanillaRNN(hk.VanillaRNN):
    """VanillaRNN Module in K-stereographic model

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)

    # TODO Change Attribute section
    Non-inherited attributes:
        k (float): the curvature of the manifold
    """

    def __init__(
        self,
        hidden_size: int,
        k: float,
        double_bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(hidden_size, double_bias, name)
        self.k = k

    def __call__(self, inputs, prev_state):
        """Computes a hyperbolic rnn transform of the input and the previous state"""
        input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=self.double_bias
        )

        # arbitrary order for Möbius addition used here
        out = m_add(hidden_to_hidden(prev_state), input_to_hidden(inputs))
        out = m_relu(out, self.k)
        return out, out


class StereographicGRU(hk.GRU):
    """GRU Module in K-stereographic model

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)

    # TODO Change Attribute section
    Non-inherited attributes:
        k (float): the curvature of the manifold
    """

    def __init__(
        self,
        hidden_size: int,
        k: float,
        w_i_init: Optional[hk.initializers.Initializer] = None,
        w_h_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(hidden_size, w_i_init, w_h_init, b_init, name)
        self.k = k

    def __call__(self, inputs, state):
        """Computes a hyperbolic gru transform of the input and the previous state"""
        # Naive implementation from the paper
        # TODO Copy Haiku native implementation

        if inputs.ndim not in (1, 2):
            raise ValueError("GRU inputs must be rank-1 or rank-2.")

        r_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        r_hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=False
        )

        z_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        z_hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=False
        )

        r = m_add(r_hidden_to_hidden(state), r_input_to_hidden(inputs))
        r = nn.sigmoid(logmap0(r, self.k))

        z = m_add(z_hidden_to_hidden(state), z_input_to_hidden(inputs))
        z = nn.sigmoid(logmap0(z, self.k))

        h_tilt_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        # TOTO Add args in get_parameter
        h_tilt_hidden_to_hidden = hk.get_parameter(
            "riemannian_u", [self.hidden_size, self.hidden_size]
        )

        # first term of the addition
        h_tilt_1 = jnp.dot(h_tilt_hidden_to_hidden, jnp.diag(r))
        h_tilt_1 = m_tanh(h_tilt_1, self.k)
        h_tilt_1 = m_dot(state, h_tilt_1)
        # second term of the addition
        h_tilt_2 = h_tilt_input_to_hidden(inputs)
        h_tilt = m_add(h_tilt_1, h_tilt_2)

        state = m_add(state, m_dot(m_add(-state, h_tilt), jnp.diag(z)))

        return state, state
