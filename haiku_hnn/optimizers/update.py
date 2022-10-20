import jax
import jax.numpy as jnp
import optax

from haiku_hnn.core import exp_mapping


def apply_riemannian_updates(
    params: optax.Params, updates: optax.Updates
) -> optax.Params:
    return jax.tree_util.tree_map(
        lambda p, u: jnp.asarray(exp_mapping(p, u).astype(jnp.asarray(p).dtype)),
        params,
        updates,
    )
