import pytest
import pytest_check as check
import pytest_mock as mock

import jax.numpy as jnp
import jax.nn as nn

from haiku_hnn.core.math import safe_tanh, safe_arctanh
from haiku_hnn.core.math import tan_k, arctan_k


def test_safe_tanh_clip():
    onbound_x = 15
    outbound_x = 16
    check.equal(safe_tanh(onbound_x), safe_tanh(outbound_x))


def test_safe_arctanh_clip():
    onbound_x = 1 - 1e-7
    outbound_x = 1
    check.equal(safe_arctanh(onbound_x), safe_arctanh(outbound_x))
