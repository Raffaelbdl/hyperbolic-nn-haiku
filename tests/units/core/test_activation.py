import pytest
import pytest_check as check
import pytest_mock as mock

import jax.numpy as jnp
import jax.nn as nn

from haiku_hnn.core.manifolds.base import Manifold
from haiku_hnn.core.activation import k_fn


def test_k_fn_with_k_equals_zero():
    dummy_x = jnp.array([-1, 0, 1])

    m = Manifold(0)

    check.is_true(jnp.array_equal(k_fn(m, nn.relu)(dummy_x), nn.relu(dummy_x)))
    check.is_true(jnp.array_equal(k_fn(m, nn.softmax)(dummy_x), nn.softmax(dummy_x)))
    check.is_true(jnp.array_equal(k_fn(m, nn.tanh)(dummy_x), nn.tanh(dummy_x)))
