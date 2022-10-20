"""Aliases for popular Riemannian optimizers, adaptation of optax.alias"""
from typing import Any, Optional, Union

import optax

from haiku_hnn.optimizers import transform

ScalarOrSchedule = Union[float, optax.Schedule]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)


def rsgd(
    learning_rate: ScalarOrSchedule,
) -> optax.GradientTransformation:
    """Simplest Riemannian Stochastic Gradient Descent optimizer

    References:
        [Bonnabe, 2013](https://arxiv.org/abs/1111.5280)

    Args:
        learning_rate (ScalarOrSchedule): A fixed global scaling factor

    Returns:
        A GradientTransformation.
    """

    return optax.chain(
        transform.riemannian_scale(),
        _scale_by_learning_rate(learning_rate),
    )


def riemannian_adagrad(
    learning_rate: ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
) -> optax.GradientTransformation:
    """The Riemmanian Adagrad optimizer.

    References:
        [Bécigneul and Ganea, 2019](http://arxiv.org/abs/1810.00760)

    Args:
        learning_rate: A fixed global scaling factor.
        initial_accumulator_value: Initial value for the accumulator.
        eps: A small constant applied to denominator inside of the square root
            (as in RMSProp) to avoid dividing by zero when rescaling.

    Returns:
        The corresponding `GradientTransformation`.

    """
    return optax.chain(
        transform.riemannian_scale(),
        transform.riemannian_scale_by_rss(initial_accumulator_value, eps),
        _scale_by_learning_rate(learning_rate),
    )


def riemannian_adam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    """The Riemannian Adam optimizer.

    References:
        [Bécigneul and Ganea, 2019](http://arxiv.org/abs/1810.00760)

    Args:
        learning_rate: A fixed global scaling factor.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
            in RMSProp), to avoid dividing by zero when rescaling. This is needed for
            example when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        the corresponding `GradientTransformation`.
    """
    return optax.chain(
        transform.riemannian_scale(),
        transform.riemannian_scale_by_adam(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype
        ),
        _scale_by_learning_rate(learning_rate),
    )
