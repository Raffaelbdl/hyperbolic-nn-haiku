from typing import Any

import haiku as hk

from haiku_hnn.core.manifolds.base import Manifold


class HyperbolicInitializer(hk.initializers.Initializer):
    def __init__(
        self, initializer: hk.initializers.Initializer, manifold: Manifold, **args
    ):
        self.initializer = initializer
        self.manifold = manifold

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        output = self.initializer(*args, **kwds)
        return self.manifold.expmap0(output)
