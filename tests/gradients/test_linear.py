import pytest
import pytest_check as check
import pytest_mock as mock

import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

import haiku_hnn as hknn
from haiku_hnn.core.stereographic import project
from tests.gradients.utils import is_nan_in_pytree


def test_nan_linear():
    @hk.transform
    def fwd_fn(x):
        return hknn.StereographicLinear(10, -1)(x)

    key = jax.random.PRNGKey(0)

    x = project(hknn.expmap0(jnp.zeros((100)), -1), -1)

    params = fwd_fn.init(key, x)

    def loss_fn(params, x, y):
        pred = fwd_fn.apply(params, key, x)
        return jnp.mean(jnp.sum(jnp.square(pred - y)))

    y = jnp.ones((10))
    l, g = jax.value_and_grad(loss_fn)(params, x, y)

    check.is_false(is_nan_in_pytree(g))


def test_nan_concat_linear():
    @hk.transform
    def fwd_fn(x, u):
        return hknn.StereographicConcatLinear(10, -1)(x, u)

    key = jax.random.PRNGKey(0)

    x = project(hknn.expmap0(jnp.zeros((100)), -1), -1)
    u = project(hknn.expmap0(jnp.zeros((100)), -1), -1)

    params = fwd_fn.init(key, x, u)

    def loss_fn(params, x, u, y):
        pred = fwd_fn.apply(params, key, x, u)
        return jnp.mean(jnp.sum(jnp.square(pred - y)))

    y = jnp.ones((10))
    l, g = jax.value_and_grad(loss_fn)(params, x, u, y)

    check.is_false(is_nan_in_pytree(g))
