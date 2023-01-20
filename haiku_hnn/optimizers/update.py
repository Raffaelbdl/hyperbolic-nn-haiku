import jax
import jax.numpy as jnp
import optax

from haiku_hnn.core.manifolds.base import Manifold


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
    params: optax.Params, updates: optax.Updates, manifold: Manifold
) -> optax.Params:
    """Applies an riemannian update to the corresponding parameters

    This will apply a riemannian update indiscriminately of whether the parameters
    are euclidian of riemannian.

    Args:
        params (Params): a tree of parameters
        updates (Updates): a tree of updates, the tree structure and the shape of the leaf
        nodes must match that of `params`
        k (float): the curvature of the manifold
    """
    return jax.tree_util.tree_map(
        lambda p, u: jnp.asarray(manifold.expmap(p, u).astype(jnp.asarray(p).dtype)),
        params,
        updates,
    )


def apply_mixed_updates(
    params: optax.Params, updates: optax.Updates, manifold: Manifold
) -> optax.Params:
    """Applies an update to the corresponding riemannian & euclidian parameters

    This assumes the haiku_hnn syntax where riemannian parameters have 'riemannian'
    in their name (eg. 'riemannian_w').

    Args:
        params (Params): a tree of parameters
        updates (Updates): a tree of updates, the tree structure and the shape of the leaf
        nodes must match that of `params`
        k (float): the curvature of the manifold
    """

    def update_fn(p, u, l):
        if "riemannian" == l:
            return manifold.proj(
                jnp.asarray(manifold.expmap(p, u).astype(jnp.asarray(p).dtype)), 4e-3
            )
        return jnp.asarray(p + u).astype(jnp.asarray(p).dtype)

    return jax.tree_util.tree_map(
        update_fn, params, updates, label_riemannian_fn(params)
    )
