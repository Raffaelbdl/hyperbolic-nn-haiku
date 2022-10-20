from pydoc import doc
import jax
import jax.numpy as jnp
import optax

from haiku_hnn.core import expmap


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict"""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


label_riemannian_fn = map_nested_fn(
    lambda k, _: "riemannian" if "riemannian" in k else "euclidian"
)


def apply_riemannian_updates(
    params: optax.Params, updates: optax.Updates
) -> optax.Params:
    return jax.tree_util.tree_map(
        lambda p, u: jnp.asarray(expmap(p, u).astype(jnp.asarray(p).dtype)),
        params,
        updates,
    )


def apply_mixed_updates(params: optax.Params, updates: optax.Updates) -> optax.Params:
    def update_fn(p, u, l):
        if l == "riemannian":
            return jnp.asarray(expmap(p, u).astype(jnp.asarray(p).dtype))
        return jnp.asarray(p + u).astype(jnp.asarray(p).dtype)

    return jax.tree_util.tree_map(
        update_fn, params, updates, label_riemannian_fn(params)
    )
