import jax
import jax.numpy as jnp


def is_nan_in_pytree(pytree) -> bool:
    nan_tree = jax.tree_util.tree_map(lambda leaf: jnp.any(jnp.isnan(leaf)), pytree)

    for leaf in jax.tree_util.tree_leaves(nan_tree):
        if leaf:
            return True
    return False
