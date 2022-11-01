from typing import Optional

from jax import numpy as jnp

from haiku_hnn.core.math import arctan_k, tan_k


def conformal_factor(x: jnp.ndarray, k: float):
    """Computes the conformal factor of the K-stereographic model at a given point

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point where the conformal factor is computed
        k (float): the curvature of the manifold
    """
    return 4 / (1 + k * jnp.linalg.norm(x) ** 2)


# TODO beware of when denominator is null
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
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    a_inner_b = jnp.inner(a, b)

    numerator = (1 - 2 * k * a_inner_b - k * norm_b**2) * a
    numerator += (1 + k * norm_a**2) * b

    denominator = 1 - 2 * k * a_inner_b + k**2 * norm_a**2 * norm_b**2

    return numerator / denominator


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
    norm_a = jnp.linalg.norm(a)

    return tan_k(s / arctan_k(norm_a)) * (a / norm_a)


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
    norm_v = jnp.linalg.norm(v)

    transformed_v = tan_k(jnp.sqrt(jnp.abs(k)) * conformal_factor(x, k) * norm_v / 2)
    transformed_v *= v / norm_v
    return m_add(x, transformed_v, k)


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
    norm_v = jnp.linalg.norm(v)

    transformed_v = tan_k(jnp.sqrt(jnp.abs(k)) * 2 * norm_v)
    transformed_v *= v / norm_v
    return transformed_v


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
    norm_mx_madd_y = jnp.linalg.norm(mx_madd_y)

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
    norm_y = jnp.linalg.norm(y)

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
    dist_euclid = jnp.linalg.norm(x - y)
    x_inner_y = jnp.inner(x, y)

    return 2 * dist_euclid * (1 - k * dist_euclid * (dist_euclid / 3 + x_inner_y))


def m_dot(x: jnp.ndarray, w: jnp.ndarray, k: float) -> jnp.ndarray:
    """Computes the dot product between x and w in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the right side of the dot product, usually a vector
        w (jnp.ndarray): the left side of the dot product, usually a weight matrix
        k (float): the curvature of the manifold

    Returns:
        The Möbius dot product between x and w in the K-stereographic model
    """
    return expmap0(jnp.dot(logmap0(x, k), w), k)


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
    return expmap(x, 4 / conformal_factor(x, k) * logmap0(b, k), k)
