from typing import Optional

import haiku as hk

from haiku_hnn.src.activation import remapping_relu

from haiku_hnn.src.linear import RemappingLinear


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
        out = remapping_relu(
            input_to_hidden(inputs) + hidden_to_hidden(prev_state), self.c
        )
        return out, out
