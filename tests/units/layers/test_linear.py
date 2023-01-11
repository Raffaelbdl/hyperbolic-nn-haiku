import pytest
import pytest_check as check
import pytest_mock as mock

import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

import haiku_hnn as hknn


def test_stereographiclinear_learnables_in_params():
    @hk.transform
    def fwd_fn(x):
        return hknn.StereographicLinear(10, -1, learnable=True, name="test_layer")(x)

    key = jax.random.PRNGKey(0)

    manifold = hknn.Stereographic(-1)
    x = manifold.proj(manifold.expmap0(jnp.zeros((1, 100))), 4e-3)

    params = fwd_fn.init(key, x)
    check.is_in("riemannian_k", params["test_layer"].keys())


def test_lorentzlinear_learnables_in_params():
    @hk.transform
    def fwd_fn(x):
        return hknn.LorentzLinear(
            10, -1, 1.0, 1.1, learnable_k=True, learnable_scale=True, name="test_layer"
        )(x)

    key = jax.random.PRNGKey(0)

    manifold = hknn.Stereographic(-1)
    x = manifold.proj(manifold.expmap0(jnp.zeros((1, 100))), 4e-3)

    params = fwd_fn.init(key, x)
    check.is_in("riemannian_k", params["test_layer"].keys())
    check.is_in("riemannian_scale", params["test_layer"].keys())
