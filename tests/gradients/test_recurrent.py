import pytest
import pytest_check as check
import pytest_mock as mock

import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

import haiku_hnn as hknn
from tests.gradients.utils import is_nan_in_pytree


def test_nan_stereographic_vanilla_rnn():
    @hk.transform
    def fwd_fn(x, h):
        return hknn.StereographicVanillaRNN(10, -1)(x, h)

    key = jax.random.PRNGKey(0)

    manifold = hknn.Stereographic(-1)
    x = manifold.proj(manifold.expmap0(jnp.zeros((1, 100))), 4e-3)
    h = manifold.proj(manifold.expmap0(jnp.zeros((10))), 4e-3)

    params = fwd_fn.init(key, x, h)

    def loss_fn(params, x, h, y):
        pred, h = fwd_fn.apply(params, key, x, h)
        return jnp.mean(jnp.sum(jnp.square(pred - y)))

    y = jnp.ones((1, 10))
    l, g = jax.value_and_grad(loss_fn)(params, x, h, y)

    check.is_false(is_nan_in_pytree(g))


def test_nan_stereographic_gru():
    @hk.transform
    def fwd_fn(x, h):
        return hknn.StereographicGRU(10, -1)(x, h)

    key = jax.random.PRNGKey(0)

    manifold = hknn.Stereographic(-1)
    x = manifold.proj(manifold.expmap0(jnp.zeros((1, 100))), 4e-3)[0]
    h = manifold.proj(manifold.expmap0(jnp.zeros((10))), 4e-3)

    params = fwd_fn.init(key, x, h)

    def loss_fn(params, x, h, y):
        pred, h = fwd_fn.apply(params, key, x, h)
        return jnp.mean(jnp.sum(jnp.square(pred - y)))

    y = jnp.ones((10))
    l, g = jax.value_and_grad(loss_fn)(params, x, h, y)

    check.is_false(is_nan_in_pytree(g))
