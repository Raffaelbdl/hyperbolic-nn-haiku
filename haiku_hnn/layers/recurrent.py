from typing import Optional

import haiku as hk
from jax import nn
from jax import numpy as jnp

from haiku_hnn.core.manifolds.stereographic import Stereographic
from haiku_hnn.core.activation import r_relu, r_tanh, r_fn
from haiku_hnn.layers.linear import StereographicLinear
from haiku_hnn.initializers import HyperbolicInitializer


class StereographicVanillaRNN(hk.VanillaRNN):
    """VanillaRNN Module in K-stereographic model

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)
    """

    def __init__(
        self,
        hidden_size: int,
        k: float,
        double_bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(hidden_size, double_bias, name)
        self.manifold = Stereographic(k)
        self.k = self.manifold.k

    def __call__(self, inputs, prev_state):
        """Computes a hyperbolic rnn transform of the input and the previous state"""
        input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=self.double_bias
        )

        # arbitrary order for Möbius addition used here
        out = self.manifold._mobius_add(
            hidden_to_hidden(prev_state), input_to_hidden(inputs)
        )
        out = r_relu(out, self.manifold)
        return out, out


class StereographicGRU(hk.GRU):
    """GRU Module in K-stereographic model

    References:
        Hyperbolic Neural Networks (http://arxiv.org/abs/1805.09112)
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
        self.manifold = Stereographic(k)
        self.k = self.manifold.k

    def __call__(self, inputs, state):
        """Computes a hyperbolic gru transform of the input and the previous state"""
        if inputs.ndim not in (1, 2):
            raise ValueError("GRU inputs must be rank-1 or rank-2.")

        inputs = self.manifold.proj(self.manifold.expmap0(inputs), 4e-3)
        state = self.manifold.proj(state, 4e-3)

        r_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        r_hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=False
        )

        z_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        z_hidden_to_hidden = StereographicLinear(
            self.hidden_size, self.k, with_bias=False
        )
        r = self.manifold._mobius_add(
            r_hidden_to_hidden(state), r_input_to_hidden(inputs)
        )
        r = r_fn(self.manifold, nn.sigmoid)(r)

        z = self.manifold._mobius_add(
            z_hidden_to_hidden(state), z_input_to_hidden(inputs)
        )
        z = r_fn(self.manifold, nn.sigmoid)(z)

        h_tilt_input_to_hidden = StereographicLinear(self.hidden_size, self.k)
        # TOTO Add args in get_parameter
        h_tilt_hidden_to_hidden = hk.get_parameter(
            "riemannian_u",
            [self.hidden_size, self.hidden_size],
            init=HyperbolicInitializer(
                hk.initializers.VarianceScaling(), self.manifold
            ),
        )

        # first term of the addition
        def to_diag(x):
            return jnp.diag(x)

        # h_tilt_1 = jnp.matmul(h_tilt_hidden_to_hidden, jax.vmap(to_diag)(r))
        h_tilt_1 = jnp.matmul(h_tilt_hidden_to_hidden, to_diag(r))
        h_tilt_1 = r_tanh(h_tilt_1, self.manifold)
        h_tilt_1 = self.manifold._mobius_dot(state, h_tilt_1)
        # second term of the addition

        h_tilt_2 = h_tilt_input_to_hidden(inputs)

        h_tilt = self.manifold._mobius_add(h_tilt_1, h_tilt_2)
        # state = m_add(
        #     state,
        #     m_dot(m_add(-state, h_tilt, self.k), jax.vmap(to_diag)(z), self.k),
        #     self.k,
        # )

        state = self.manifold._mobius_add(
            state,
            self.manifold._mobius_dot(
                self.manifold._mobius_add(-state, h_tilt), to_diag(z)
            ),
        )
        state = self.manifold.proj(state, 4e-3)

        return state, state
