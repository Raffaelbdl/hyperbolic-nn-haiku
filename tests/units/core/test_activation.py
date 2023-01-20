import pytest
import pytest_check as check
import pytest_mock as mock

import jax.numpy as jnp
import jax.nn as nn

from haiku_hnn.core.manifolds.base import Manifold
from haiku_hnn.core.activation import r_fn


def test_r_fn_with_r_equals_zero():
    dummy_x = jnp.array([-1, 0, 1])

    m = Manifold(0)

    check.is_true(jnp.array_equal(r_fn(m, nn.relu)(dummy_x), nn.relu(dummy_x)))
    check.is_true(jnp.array_equal(r_fn(m, nn.softmax)(dummy_x), nn.softmax(dummy_x)))
    check.is_true(jnp.array_equal(r_fn(m, nn.tanh)(dummy_x), nn.tanh(dummy_x)))
