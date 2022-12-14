from typing import Optional

import haiku as hk
import jax
from jax import nn
from jax import numpy as jnp

from haiku_hnn.core.activation import k_relu, k_tanh, k_fn
from haiku_hnn.core.stereographic import expmap0, m_add, m_dot, project
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
        out = m_add(hidden_to_hidden(prev_state), input_to_hidden(inputs), self.k)
        out = project(k_relu(out, self.k), self.k)
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
        if inputs.ndim not in (1, 2):
            raise ValueError("GRU inputs must be rank-1 or rank-2.")

        inputs = expmap0(inputs, self.k, use_project=True)
        state = project(state, self.k)

        r_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        r_hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=False
        )

        z_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        z_hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=False
        )
        r = m_add(r_hidden_to_hidden(state), r_input_to_hidden(inputs), self.k)
        r = k_fn(self.k, nn.sigmoid)(r)

        z = m_add(z_hidden_to_hidden(state), z_input_to_hidden(inputs), self.k)
        z = k_fn(self.k, nn.sigmoid)(z)

        h_tilt_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        # TOTO Add args in get_parameter
        h_tilt_hidden_to_hidden = hk.get_parameter(
            "riemannian_u",
            [self.hidden_size, self.hidden_size],
            init=hk.initializers.VarianceScaling(),
        )

        # first term of the addition
        def to_diag(x):
            return jnp.diag(x)

        # h_tilt_1 = jnp.matmul(h_tilt_hidden_to_hidden, jax.vmap(to_diag)(r))
        h_tilt_1 = jnp.matmul(h_tilt_hidden_to_hidden, to_diag(r))
        h_tilt_1 = k_tanh(h_tilt_1, self.k)
        h_tilt_1 = m_dot(state, h_tilt_1, self.k)
        # second term of the addition

        h_tilt_2 = h_tilt_input_to_hidden(inputs)

        h_tilt = m_add(h_tilt_1, h_tilt_2, self.k)
        # state = m_add(
        #     state,
        #     m_dot(m_add(-state, h_tilt, self.k), jax.vmap(to_diag)(z), self.k),
        #     self.k,
        # )

        state = m_add(
            state,
            m_dot(m_add(-state, h_tilt, self.k), to_diag(z), self.k),
            self.k,
        )
        state = project(state, self.k)

        return state, state
