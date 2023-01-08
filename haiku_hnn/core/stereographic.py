from jax import numpy as jnp
import jax

from haiku_hnn.core.math import arctan_k, tan_k


def conformal_factor(
    x: jnp.ndarray, k: float, axis: int = -1, keepdims: bool = True
) -> jnp.ndarray:
    """Computes the conformal factor of the K-stereographic model at a given point

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point where the conformal factor is computed
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed
        keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
    """
    return 2 / (1 + k * jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + 1e-15)


def norm(
    x: jnp.ndarray, u: jnp.ndarray, k: float, axis: int = -1, keepdims: bool = True
) -> jnp.ndarray:
    """Computes the norm in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point where the norm is computed
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed
        keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
    """
    return conformal_factor(x, k, axis, keepdims) * jnp.linalg.norm(
        u, axis=axis, keepdims=keepdims
    )


def gyration(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    k: int,
    axis: int = -1,
    keepdims: bool = True,
) -> jnp.ndarray:
    """Computes the gyration vector of w by [u, v]

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        u, v, w (jnp.ndarray): the points where gyr[u, v]w is computed
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed
        keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
    """
    norm_u2 = jnp.sum(jnp.square(u), axis=-1, keepdims=keepdims)
    norm_v2 = jnp.sum(jnp.square(v), axis=-1, keepdims=keepdims)
    dot_uv = jnp.sum(u * v, axis=axis, keepdims=keepdims)
    dot_uw = jnp.sum(u * w, axis=axis, keepdims=keepdims)
    dot_vw = jnp.sum(v * w, axis=axis, keepdims=keepdims)
    k2 = jnp.square(k)

    a = -k2 * dot_uw * norm_v2 - k * dot_vw + 2 * k2 * dot_uv * dot_vw
    b = -k2 * dot_vw * norm_u2 + k * dot_uw
    d = 1 - 2 * k * dot_uv + k2 * norm_u2 * norm_v2
    return w + 2 * (a * u + b * v) / (d + 1e-15)


def dist(
    x: jnp.ndarray, y: jnp.ndarray, k: float, axis: int = -1, keepdims: bool = False
) -> float:
    """Computes the distance between x and y in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x, y (jnp.ndarray): the points between which we want to calculate the distance
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed
        keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
    """
    return 2 * arctan_k(m_add(-x, y, k, axis, keepdims), k)


def dist0(x: jnp.ndarray, k: float, axis: int = -1, keepdims: bool = False) -> float:
    """Computes the distance between x and the manifold origin

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        x (jnp.ndarray): the point at which we want to calculate the distance to the origin
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed
        keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one
    """
    return 2 * arctan_k(jnp.linalg.norm(x, axis=axis, keepdims=keepdims), k)


def m_add(
    a: jnp.ndarray,
    b: jnp.ndarray,
    k: float,
    axis: int = -1,
    keepdims: bool = True,
    use_project: bool = False,
) -> jnp.ndarray:
    """Computes the Möbius addition in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        a, b (jnp.ndarray): the first and second arguments for Möbius addition
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed
        keepdims (bool): if True, the axes which are reduced are left in the result as dimensions with size one

    Returns:
        The Möbius addition of a and b in the K-stereographic model
    """
    norm_a2 = jnp.sum(jnp.square(a), axis=axis, keepdims=keepdims)
    norm_b2 = jnp.sum(jnp.square(b), axis=axis, keepdims=keepdims)
    ab = jnp.sum(a * b, axis=axis, keepdims=keepdims)

    numerator = (1 - 2 * k * ab - k * norm_b2) * a
    numerator += (1 + k * norm_a2) * b

    denominator = 1 - 2 * k * ab + k**2 * norm_a2 * norm_b2

    if use_project:
        return project(numerator / (denominator + 1e-15), k)
    return numerator / (denominator + 1e-15)


def m_scale(
    a: jnp.ndarray,
    s: float,
    k: float,
    axis: int = -1,
    use_project: bool = False,
) -> jnp.ndarray:
    """Compute the Möbius scaling in the K-stereographic model

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)

    Args:
        a (jnp.ndarray): the starting point of the Möbius scaling
        s (jnp.ndarray): the scale of the Möbius scaling
        k (float): the curvature of the manifold

        axis (int): the axis along which the sum is performed

    Returns:
        The Möbius scaling of a per s in the K-stereographic model
    """
    norm_a = jnp.linalg.norm(a, axis=axis, keepdims=True) + 1e-15
    if use_project:
        return project(tan_k(s * arctan_k(norm_a), k) * (a / norm_a), k)
    return tan_k(s * arctan_k(norm_a), k) * (a / norm_a)


def expmap(
    x: jnp.ndarray,
    v: jnp.ndarray,
    k: float,
    axis: int = -1,
    use_project: bool = False,
) -> jnp.ndarray:
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
    norm_v = jnp.linalg.norm(v, axis=axis, keepdims=True) + 1e-15
    conf_factor = conformal_factor(x, k, axis=axis, keepdims=True)

    transformed_v = tan_k(jnp.sqrt(jnp.abs(k)) * conf_factor * norm_v / 2, k)
    transformed_v *= v / norm_v

    if use_project:
        project(m_add(x, transformed_v, k, axis=axis, keepdims=True), k)
    return m_add(x, transformed_v, k, axis=axis, keepdims=True)


def expmap0(
    v: jnp.ndarray,
    k: float,
    axis: int = -1,
    use_project: bool = False,
) -> jnp.ndarray:
    """Computes the exponential mapping in the K-stereographic model at the origin

    Reference:
        Constant Curvature Graph Convolutional Networks
            (https://arxiv.org/pdf/1911.05076v1.pdf)
        Hyperbolic Neural Networks
            (http://arxiv.org/abs/1805.09112)

    Args:
        v (jnp.ndarray): the point from the tangent plane we want to project on the
            hyperboloid surface
        k (float): the curvature of the manifold

    Returns:
        The projection of v from the tangent plane of the origin onto the hyperboloid surface
    """
    # norm_v = safe_norm(v, axis=axis, keepdims=True) # introduce nans in gradients
    v += 1e-15
    norm_v = jnp.sqrt(jnp.sum(jnp.square(v), axis=axis, keepdims=True))

    if use_project:
        return project(tan_k(norm_v, k) * (v / norm_v), k)
    return tan_k(norm_v, k) * (v / norm_v)


def logmap(x: jnp.ndarray, y: jnp.ndarray, k: float, axis: int = -1) -> jnp.ndarray:
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
    mx_madd_y = m_add(-x, y, k, axis=axis, keepdims=True)
    norm_mx_madd_y = jnp.linalg.norm(mx_madd_y, axis=axis, keepdims=True) + 1e-15
    conf_factor = conformal_factor(x, k, axis=axis, keepdims=True)

    res = 2 * jnp.power(jnp.abs(k), -0.5) / conf_factor
    res *= arctan_k(norm_mx_madd_y, k)
    res *= mx_madd_y / norm_mx_madd_y
    return res


def logmap0(y: jnp.ndarray, k: float, axis: int = -1) -> jnp.ndarray:
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
    norm_y = jnp.linalg.norm(y, axis=axis, keepdims=True) + 1e-15
    return arctan_k(norm_y, k) * (y / norm_y)


def m_dot(
    x: jnp.ndarray,
    w: jnp.ndarray,
    k: float,
    precision=None,
    use_project: bool = False,
) -> jnp.ndarray:
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
    if use_project:
        return project(expmap0(jnp.dot(logmap0(x, k), w, precision=precision), k), k)
    return expmap0(jnp.dot(logmap0(x, k), w, precision=precision), k)


def m_bias(
    x: jnp.ndarray,
    b: jnp.ndarray,
    k: float,
    use_project: bool = False,
) -> float:
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
    if use_project:
        return project(expmap(x, 2 / conformal_factor(x, k) * logmap0(b, k), k), k)
    return expmap(x, 2 / conformal_factor(x, k) * logmap0(b, k), k)


def parallel_transport(
    x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray, k: float, axis: int = -1
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
    return gyration(y, -x, v, k, axis) * conformal_factor(x, k) / conformal_factor(y, k)


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
    max_norm = (1 - eps) / jnp.power(jnp.abs(k), 0.5)
    # norm = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-15
    norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True) + 1e-15

    cond = norm > max_norm
    return jnp.where(cond, 1 / norm * max_norm, 1.0) * x
