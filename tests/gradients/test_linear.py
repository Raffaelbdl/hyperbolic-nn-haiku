import pytest
import pytest_check as check
import pytest_mock as mock

import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

import haiku_hnn as hknn
from tests.gradients.utils import is_nan_in_pytree


def test_nan_stereographic_linear():
    @hk.transform
    def fwd_fn(x):
        return hknn.StereographicLinear(10, -1, learnable=True)(x)

    key = jax.random.PRNGKey(0)

    manifold = hknn.Stereographic(-1)
    x = manifold.proj(manifold.expmap0(jnp.zeros((1, 100))), 4e-3)

    params = fwd_fn.init(key, x)

    def loss_fn(params, x, y):
        pred = fwd_fn.apply(params, key, x)
        return jnp.mean(jnp.sum(jnp.square(pred - y)))

    y = jnp.ones((10))
    l, g = jax.value_and_grad(loss_fn)(params, x, y)

    check.is_false(is_nan_in_pytree(g))


def test_nan_lorentz_linear():
    @hk.transform
    def fwd_fn(x):
        return hknn.LorentzLinear(
            10, -1, 1.0, 1.1, learnable_k=True, learnable_scale=True
        )(x)

    key = jax.random.PRNGKey(0)

    manifold = hknn.Lorentz(-1)
    x = manifold.get_origin_like(jnp.zeros((1, 100)))

    params = fwd_fn.init(key, x)

    def loss_fn(params, x, y):
        pred = fwd_fn.apply(params, key, x)
        return jnp.mean(jnp.sum(jnp.square(pred - y)))

    y = jnp.ones((10))
    l, g = jax.value_and_grad(loss_fn)(params, x, y)

    check.is_false(is_nan_in_pytree(g))
