from jax import numpy as jnp

from haiku_hnn.core.math import arctan_k, tan_k


def conformal_factor(x: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the conformal factor of the K-stereographic model at a given point

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point where the conformal factor is computed
        k (float): the curvature of the manifold
    """
    return 4 / (1 + k * jnp.sum(jnp.square(x), axis=-1, keepdims=True) + 1e-15)


def m_add(a: jnp.ndarray, b: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the Möbius addition in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        a, b (jnp.ndarray): the first and second arguments for Möbius addition
        k (float): the curvature of the manifold

    Returns:
        The Möbius addition of a and b in the K-stereographic model
    """
    norm_a2 = jnp.sum(jnp.square(a), axis=-1, keepdims=True)
    norm_b2 = jnp.sum(jnp.square(b), axis=-1, keepdims=True)
    ab = jnp.sum(a * b, axis=-1, keepdims=True)

    numerator = (1 - 2 * k * ab - k * norm_b2) * a
    numerator += (1 + k * norm_a2) * b

    denominator = 1 - 2 * k * ab + k**2 * norm_a2 * norm_b2

    return project(numerator / (denominator + 1e-15), k)


def m_scale(a: jnp.ndarray, s: float, k: float) -> jnp.ndarray:
    """Compute the Möbius scaling in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        a (jnp.ndarray): the starting point of the Möbius scaling
        s (jnp.ndarray): the scale of the Möbius scaling
        k (float): the curvature of the manifold

    Returns:
        The Möbius scaling of a per s in the K-stereographic model
    """
    norm_a = jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-15
    return project(tan_k(s / arctan_k(norm_a), k) * (a / norm_a), k)


def expmap(x: jnp.ndarray, v: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the exponential mapping in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point from which is defined the starting tangent plane
        v (jnp.ndarray): the point from the tangent plane we want to project on the
            hyperboloid surface
        k (float): the curvature of the manifold

    Returns:
        The projection of v from the tangent plane of x onto the hyperboloid surface
    """
    norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-15

    transformed_v = tan_k(jnp.sqrt(jnp.abs(k)) * conformal_factor(x, k) * norm_v / 2, k)
    transformed_v *= v / norm_v
    return project(m_add(x, transformed_v, k), k)


def expmap0(v: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the exponential mapping in the K-stereographic model at the origin

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        v (jnp.ndarray): the point from the tangent plane we want to project on the
            hyperboloid surface
        k (float): the curvature of the manifold

    Returns:
        The projection of v from the tangent plane of the origin onto the hyperboloid surface
    """
    norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-15

    transformed_v = tan_k(jnp.sqrt(jnp.abs(k)) * 2 * norm_v, k)
    transformed_v *= v / norm_v
    return project(transformed_v, k)


def logmap(x: jnp.ndarray, y: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the logarithmic mapping in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point from which is defined the starting tangent plane
        y (jnp.ndarray): the point from the hyperboloid surface we want to project on the
            tangent plane
        k (float): the curvature of the manifold

    Returns:
        The projection of y from the hyperboloid surface onto the tangent plane of x
    """
    mx_madd_y = m_add(-x, y, k)
    norm_mx_madd_y = jnp.linalg.norm(mx_madd_y, axis=-1, keepdims=True) + 1e-15

    res = 2 * jnp.power(jnp.abs(k), -0.5) / conformal_factor(x, k)
    res *= arctan_k(norm_mx_madd_y, k)
    res *= mx_madd_y / norm_mx_madd_y
    return res


def logmap0(y: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the logarithmic mapping in the K-stereographic model at the origin

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        y (jnp.ndarray): the point from the hyperboloid surface we want to project on the
            tangent plane
        k (float): the curvature of the manifold

    Returns:
        The projection of y from the hyperboloid surface onto the tangent plane of the origin
    """
    norm_y = jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-15

    return 0.5 * jnp.power(jnp.abs(k), -0.5) * arctan_k(norm_y, k) * (y / norm_y)


def dist(x: jnp.ndarray, y: jnp.ndarray, k: float) -> float:
    """Computes the distance between x and y in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x, y (jnp.ndarray): the points between which we want to calculate the distance
        k (float): the curvature of the manifold
    """
    dist_euclid = jnp.linalg.norm(x - y, axis=-1, keepdims=True)
    dot_xy = jnp.sum(x * y, axis=-1, keepdims=True)

    return 2 * dist_euclid * (1 - k * dist_euclid * (dist_euclid / 3 + dot_xy))


def m_dot(x: jnp.ndarray, w: jnp.ndarray, k: float, precision=None) -> jnp.ndarray:
    """Computes the dot product between x and w in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the right side of the dot product, usually a vector
        w (jnp.ndarray): the left side of the dot product, usually a weight matrix
        k (float): the curvature of the manifold
        precision: the precision of the dot product

    Returns:
        The Möbius dot product between x and w in the K-stereographic model
    """
    return project(expmap0(jnp.dot(logmap0(x, k), w, precision=precision), k), k)


def m_bias(x: jnp.ndarray, b: jnp.ndarray, k: float) -> float:
    """Computes the bias translation of x by b in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the starting point of the translation, usually a vector
        b (jnp.ndarray): the quantity by which we translate, usually a bias vector
        k (float): the curvature of the manifold

    Returns:
        The Möbius bias translation of x by b in the K-stereographic model
    """
    return project(expmap(x, 4 / conformal_factor(x, k) * logmap0(b, k), k), k)


def parallel_transport(
    x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray, k: float
) -> jnp.ndarray:
    """Computes the parallel transport between two tangent spaces

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the first tangent space
        y (jnp.ndarray): the second tangent space
        v (jnp.ndarray): the point to transport
        k (float): the curvature of the manifold

    Returns:
        The point v after parallel transport from tangent space of x to y
    """
    return gyration(y, -x, v) * conformal_factor(x, k) / conformal_factor(y, k)


def gyration(u, v, w):
    """Computes the gyration vector"""
    norm_u2 = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
    norm_v2 = jnp.sum(jnp.square(v), axis=-1, keepdims=True)
    dot_uv = jnp.sum(u * v, axis=-1, keepdims=True)
    dot_uw = jnp.sum(u * w, axis=-1, keepdims=True)
    dot_vw = jnp.sum(v * w, axis=-1, keepdims=True)

    a = -dot_uw * norm_v2 - dot_vw + 2 * dot_uv * dot_vw
    b = -dot_vw * norm_u2 + dot_uw
    d = 1 - 2 * dot_uv + norm_u2 * norm_v2
    return w + 2 * (a * u + b * v) / (d + 1e-15)


def norm(x: jnp.ndarray, u: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the norm in the K-stereographic model"""
    return conformal_factor(x, k) * jnp.linalg.norm(u, axis=-1, keepdims=True)


def project(x: jnp.ndarray, k: float, eps: float = 4e-3) -> jnp.ndarray:
    """Projects on the manifold to ensure numerical stability

    References:
        Geoopt: Riemannian Optimization in PyTorch
            (https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/stereographic/math.py)

    Args:
        x (jnp.ndarray): the point to project
        k (float): the curvature of the manifold
        eps (float): distance to the edge of the manifold
    """
    max_norm = (1 - eps) * jnp.power(jnp.abs(k), -0.5)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-15
    cond = norm > max_norm
    projected = x / norm * max_norm
    return jnp.where(cond, projected, x)
