import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from haiku_hnn.core.manifolds.stereographic import Stereographic


def visualize_stereographic_manifold(
    k: int = -1, plot: bool = True, save: bool = False
):
    manifold = Stereographic(k)

    point_array = []
    for i in np.linspace(-10, 10, 100):
        for j in np.linspace(-10, 10, 100):
            point_array.append([i, j])
    point_array = np.array(point_array)
    hyperbolic_array = manifold.proj(manifold.expmap0(point_array), 0)

    fig, ax = plt.subplots()

    ax.scatter(hyperbolic_array[..., 0], hyperbolic_array[..., 1], s=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Representation of [-10, 10]² in hyperbolic space")
    if save:
        plt.savefig(f"./tmp/{k}.jpg")

    if plot:
        plt.show()

    plt.close()


def make_stereographic_curvature_gif():
    filenames = []
    for k in np.linspace(-1, 1, 100):
        visualize_stereographic_manifold(k, False, True)
        filenames.append(f"./tmp/{k}.jpg")

    with imageio.get_writer("curvature.gif", mode="I") as writer:
        for f in filenames:
            img = imageio.imread(f)
            writer.append_data(img)

    for f in filenames:
        os.remove(f)


def visualize_stereographic_transformation(k: int = -1):
    manifold = Stereographic(k)

    x0 = manifold.expmap0(np.array([1, 1], dtype=np.float32))

    b = manifold.expmap0(np.array([-10, 0], dtype=np.float32))
    xb = manifold._mobius_bias(x0, b)

    fig, ax = plt.subplots()

    ax.scatter([x0[0], xb[0]], [x0[1], xb[1]])

    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Representation of [-10, 10]² in hyperbolic space")

    plt.show()


def plot_and_save(a, b, path: str, title: str):
    fig, ax = plt.subplots()
    ax.scatter([a[0], b[0]], [a[1], b[1]])
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(title)
    plt.savefig(path)
    plt.close()


def make_stereographic_bias_gif(k: int = -1):
    manifold = Stereographic(k)

    x0 = manifold.proj(manifold.expmap0(np.array([0.5, 0.5], dtype=np.float32)), 4e-3)

    b = manifold.proj(manifold.expmap0(np.array([1, 0], dtype=np.float32)), 4e-3)
    xb = manifold.proj(manifold._mobius_bias(x0, b), 4e-3)

    filenames = [f"./tmp/{0}.jpg"]
    plot_and_save(
        x0,
        xb,
        f"./tmp/{0}.jpg",
        f"Repeated translation by [1, 0] in hyperbolic space ({0})",
    )
    for i in range(1, 100):
        xb = manifold.proj(manifold._mobius_bias(xb, b), 4e-3)
        plot_and_save(
            x0,
            xb,
            f"./tmp/{i}.jpg",
            f"Repeated translation by [1, 0] in hyperbolic space ({i})",
        )
        filenames.append(f"./tmp/{i}.jpg")

    with imageio.get_writer("translation.gif", mode="I") as writer:
        for f in filenames:
            img = imageio.imread(f)
            writer.append_data(img)

    for f in filenames:
        os.remove(f)


def make_stereographic_scale_gif(k: int = -1):
    manifold = Stereographic(k)

    x0 = manifold.proj(manifold.expmap0(np.array([1, 1], dtype=np.float32)), 4e-3)

    s = 0.8
    xb = manifold.proj(manifold._mobius_scale(x0, s), 4e-3)

    filenames = [f"./tmp/{0}.jpg"]
    plot_and_save(
        x0,
        xb,
        f"./tmp/{0}.jpg",
        f"Repeated scaling by 0.8 in hyperbolic space ({0})",
    )
    for i in range(1, 100):
        xb = manifold.proj(manifold._mobius_scale(xb, s), 4e-3)
        plot_and_save(
            x0,
            xb,
            f"./tmp/{i}.jpg",
            f"Repeated scaling by 0.8 in hyperbolic space ({i})",
        )
        filenames.append(f"./tmp/{i}.jpg")

    with imageio.get_writer("scale.gif", mode="I") as writer:
        for f in filenames:
            img = imageio.imread(f)
            writer.append_data(img)

    for f in filenames:
        os.remove(f)


if __name__ == "__main__":
    pass
