from enum import Enum
from functools import partial
from typing import Callable, Tuple

from absl import app, flags
import haiku as hk
import haiku_hnn as hkhn
import jax
from jax import nn
from jax import numpy as jnp
import optax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tqdm import tqdm

from haiku_hnn.core.activation import k_relu
from haiku_hnn.core.manifolds.base import Manifold
from haiku_hnn.core.manifolds.stereographic import Stereographic
from haiku_hnn.core.manifolds.lorentz import Lorentz

jax.config.update("jax_enable_x64", True)


class OPTIMIZERS(Enum):
    ADAM = "adam"
    ADAGRAD = "adagrad"
    SGD = "sgd"


FLAGS = flags.FLAGS
flags.DEFINE_integer("SEED", 0, "Seed for reproductibility")
flags.DEFINE_integer("BATCH_SIZE", 64, "Batch size during training")
flags.DEFINE_integer("EPOCHS", 20, "Number of epochs during training")
flags.DEFINE_string(
    "ARCHITECTURE",
    "EUCLID",
    "Architecture of the network ('EUCLID', 'HYPERBOLIC', 'HYBRID')",
)
flags.DEFINE_float("LEARNING_RATE_E", 1e-3, "Learning rate for euclidian parameters")
flags.DEFINE_float("LEARNING_RATE_H", 1e-1, "Learning rate for riemannian parameters")
flags.DEFINE_float("K", -1, "The curvature of the manifold")
flags.DEFINE_string(
    "OPTIMIZER", "ADAM", "Name of the optimizer used ('ADAM', 'ADAGRAD', 'SGD'"
)
flags.DEFINE_bool(
    "PLOT_VAL", True, "Whether we plot the embedding of validation or the train"
)


def make_euclid_transform() -> hk.Transformed:
    def forward_fn(x):
        return hk.Sequential(
            [
                hk.Conv2D(32, 3, 2, padding="VALID"),
                nn.relu,
                hk.Conv2D(32, 3, 2, padding="VALID"),
                nn.relu,
                hk.Flatten(),
                hk.Linear(64),
                nn.relu,
                hk.Linear(32),
                nn.relu,
                hk.Linear(10),
            ]
        )(x)

    return hk.without_apply_rng(hk.transform(forward_fn))


def make_trunc_euclid_transform() -> hk.Transformed:
    def forward_fn(x):
        return hk.Sequential(
            [
                hk.Conv2D(32, 3, 2, padding="VALID"),
                nn.relu,
                hk.Conv2D(32, 3, 2, padding="VALID"),
                nn.relu,
                hk.Flatten(),
                hk.Linear(64),
                nn.relu,
                hk.Linear(32),
            ]
        )(x)

    return hk.without_apply_rng(hk.transform(forward_fn))


def make_stereo_transform() -> hk.Transformed:
    manifold = Stereographic(-1)

    def forward_fn(x, k):
        layers = [
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Flatten(),
            manifold.expmap0,
            hkhn.StereographicLinear(32, k),
            partial(k_relu, manifold=manifold),
            hkhn.StereographicLinear(32, k),
            partial(k_relu, manifold=manifold),
            hkhn.StereographicLinear(10, k),
        ]
        for i, l in enumerate(layers):
            x = l(x)
        return x

    return hk.without_apply_rng(hk.transform(forward_fn))


def make_trunc_stereo_transform() -> hk.Transformed:
    manifold = Stereographic(-1)

    def forward_fn(x, k):
        layers = [
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Flatten(),
            manifold.expmap0,
            hkhn.StereographicLinear(32, k),
            partial(k_relu, manifold=manifold),
            hkhn.StereographicLinear(32, k),
        ]
        for i, l in enumerate(layers):
            x = l(x)
        return x

    return hk.without_apply_rng(hk.transform(forward_fn))


def make_hyb_stereo_transform() -> hk.Transformed:
    manifold = Stereographic(-1)

    def forward_fn(x, k):
        layers = [
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Flatten(),
            manifold.expmap0,
            hkhn.StereographicLinear(32, k),
            partial(k_relu, manifold=manifold),
            hkhn.StereographicLinear(32, k),
            manifold.logmap0,
            nn.relu,
            hk.Linear(10),
        ]
        for i, l in enumerate(layers):
            x = l(x)
        return x

    return hk.without_apply_rng(hk.transform(forward_fn))


def make_trunc_hyb_stereo_transform() -> hk.Transformed:
    manifold = Stereographic(-1)

    def forward_fn(x, k):
        layers = [
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Conv2D(32, 3, 2, padding="VALID"),
            nn.relu,
            hk.Flatten(),
            manifold.expmap0,
            hkhn.StereographicLinear(32, k),
            partial(k_relu, manifold=manifold),
            hkhn.StereographicLinear(32, k),
        ]
        for i, l in enumerate(layers):
            x = l(x)
        return x

    return hk.without_apply_rng(hk.transform(forward_fn))


def preprocess_data(x, y):
    x = jnp.expand_dims((jnp.array(x) / 255.0 - 0.5) * 2, axis=-1)
    y = nn.one_hot(jnp.array(y), 10)
    return x, y


@partial(jax.jit, static_argnums=(1, 2))
def categorical_loss_forward(
    params: hk.Params, forward_fn: Callable, k: float, x: jnp.ndarray, y: jnp.ndarray
):
    y_pred = forward_fn(params, x, k)
    return jnp.mean(optax.softmax_cross_entropy(y_pred, y)), y_pred


@partial(jax.jit, static_argnums=(0, 4))
def apply_updates(
    optimizer: optax.GradientTransformation,
    params: hk.Params,
    opt_state: optax.OptState,
    grads,
    manifold: Manifold,
):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = hkhn.apply_mixed_updates(params, updates, manifold)
    return params, opt_state


def accuracy(y_pred, y):
    return jnp.mean(jnp.argmax(nn.softmax(y_pred), axis=-1) == jnp.argmax(y, axis=-1))


def get_transformed(architecture: str) -> hk.Transformed:
    return {
        "EUCLID": make_euclid_transform(),
        "HYPERBOLIC": make_stereo_transform(),
        "HYBRID": make_hyb_stereo_transform(),
    }[architecture]


def main(_):
    key = jax.random.PRNGKey(FLAGS.SEED)
    key, key1 = jax.random.split(key)

    manifold = Stereographic(FLAGS.K)

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

    x_train, y_train = preprocess_data(x_train, y_train)

    transformed = get_transformed(FLAGS.ARCHITECTURE)

    if FLAGS.ARCHITECTURE == "EUCLID":
        params = transformed.init(key1, x_train[: FLAGS.BATCH_SIZE])
        forward = lambda p, x, k: transformed.apply(p, x)
    else:
        params = transformed.init(key1, x_train[: FLAGS.BATCH_SIZE], FLAGS.K)
        forward = lambda p, x, k: transformed.apply(p, x, k)

    optimizers = {
        OPTIMIZERS.ADAM: {
            "EUCLID": optax.adam(FLAGS.LEARNING_RATE_E),
            "HYPERBOLIC": hkhn.mixed_optimizer(
                optax.adam(FLAGS.LEARNING_RATE_E),
                hkhn.riemannian_adam(
                    manifold, FLAGS.LEARNING_RATE_H, eps_root=1, eps=1
                ),
            ),
            "HYBRID": hkhn.mixed_optimizer(
                optax.adam(FLAGS.LEARNING_RATE_E),
                hkhn.riemannian_adam(
                    manifold, FLAGS.LEARNING_RATE_H, eps_root=1, eps=1
                ),
            ),
        },
        OPTIMIZERS.ADAGRAD: {
            "EUCLID": optax.adagrad(FLAGS.LEARNING_RATE_E),
            "HYPERBOLIC": hkhn.mixed_optimizer(
                optax.adagrad(FLAGS.LEARNING_RATE_E),
                hkhn.riemannian_adagrad(manifold, FLAGS.LEARNING_RATE_H),
            ),
            "HYBRID": hkhn.mixed_optimizer(
                optax.adagrad(FLAGS.LEARNING_RATE_E),
                hkhn.riemannian_adagrad(manifold, FLAGS.LEARNING_RATE_H),
            ),
        },
        OPTIMIZERS.SGD: {
            "EUCLID": optax.sgd(FLAGS.LEARNING_RATE_E),
            "HYPERBOLIC": hkhn.mixed_optimizer(
                optax.sgd(FLAGS.LEARNING_RATE_E),
                hkhn.rsgd(manifold, FLAGS.LEARNING_RATE_H),
            ),
            "HYBRID": hkhn.mixed_optimizer(
                optax.sgd(FLAGS.LEARNING_RATE_E),
                hkhn.rsgd(manifold, FLAGS.LEARNING_RATE_H),
            ),
        },
    }
    optimizer = optimizers[getattr(OPTIMIZERS, FLAGS.OPTIMIZER)][FLAGS.ARCHITECTURE]
    opt_state = optimizer.init(params)

    idx = jnp.arange(len(x_train))
    for e in range(FLAGS.EPOCHS):
        key, key1 = jax.random.split(key)
        idx = jax.random.shuffle(key1, idx)

        accs, losses = [], []
        for b in tqdm(range(len(idx) // FLAGS.BATCH_SIZE)):
            _idx = idx[b * FLAGS.BATCH_SIZE : (b + 1) * FLAGS.BATCH_SIZE]
            _x = x_train[_idx]
            _y = y_train[_idx]

            (loss, logits), grads = jax.value_and_grad(
                categorical_loss_forward, has_aux=True
            )(params, forward, FLAGS.K, _x, _y)
            params, opt_state = apply_updates(
                optimizer, params, opt_state, grads, manifold
            )

            accs.append(accuracy(logits, _y))
            losses.append(loss)
            if np.isnan(loss):
                print(f"nan at {b}")

        print(
            f"Epoch {e + 1} | Loss {sum(losses)/len(losses)} | Acc {sum(accs) / len(accs)}"
        )

    x_val, _ = preprocess_data(x_val, y_val)

    trunc_transformed = {
        "EUCLID": make_trunc_euclid_transform(),
        "HYPERBOLIC": make_trunc_stereo_transform(),
        "HYBRID": make_trunc_hyb_stereo_transform(),
    }[FLAGS.ARCHITECTURE]

    if FLAGS.ARCHITECTURE == "EUCLID":
        trunc_forward = lambda p, x, k: trunc_transformed.apply(p, x)
    else:
        trunc_forward = lambda p, x, k: trunc_transformed.apply(p, x, k)

    x, y = (x_val, y_val) if FLAGS.PLOT_VAL else (x_train, np.argmax(y_train, axis=-1))

    print("XY SHAPE", x.shape, y.shape)

    def _plot(_last_emb, _y):
        pca = PCA(n_components=2)
        pca.fit(_last_emb)
        X = pca.transform(_last_emb)
        colours = [
            "red",
            "orange",
            "yellow",
            "green",
            "cyan",
            "blue",
            "magenta",
            "pink",
            "grey",
            "black",
        ]

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(
            X[..., 0], X[..., 1], c=list(_y), cmap=mpl.colors.ListedColormap(colours)
        )

        cb = plt.colorbar()
        loc = np.arange(0, 10)
        cb.set_ticks(loc)
        cb.set_ticklabels(range(len(colours)))
        plt.show()

    x, y = x_val, y_val
    last_emb = trunc_forward(params, x, FLAGS.K)
    _plot(last_emb, y)

    x, y = x_train, np.argmax(y_train, axis=-1)
    last_emb = []
    for i in range(len(x) // 10000):
        last_emb.append(trunc_forward(params, x[i * 10000 : (i + 1) * 10000], FLAGS.K))
    last_emb = jnp.concatenate(last_emb, axis=0)
    _plot(last_emb, y)


if __name__ == "__main__":
    app.run(main)
