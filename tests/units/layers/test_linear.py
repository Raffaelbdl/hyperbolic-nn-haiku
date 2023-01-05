import pytest
import pytest_check as check
import pytest_mock as mock

import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

import haiku_hnn as hknn
from haiku_hnn.core.stereographic import project


def test_linear_k_in_params():
    @hk.transform
    def fwd_fn(x):
        return hknn.StereographicLinear(10, -1, learnable=True, name="test_layer")(x)

    key = jax.random.PRNGKey(0)

    x = project(hknn.expmap0(jnp.zeros((100)), -1), -1)

    params = fwd_fn.init(key, x)
    check.is_in("riemannian_k", params["test_layer"].keys())
