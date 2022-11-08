# Hyperbolic Neural Networks with `dm-haiku`

This is a work in progress ❗

Hyperbolic embedding has proved to be powerful in numerous applications like graph embedding. The goal of this project is to make some of the current advances in Hyperbolic Neural Networks avaialable in JAX.

The neural networks are implemented with [Haiku](https://github.com/deepmind/dm-haiku) and the optimizers are based on [Optax](https://github.com/deepmind/optax).

## Installation 

The repository is not pip-installable yet. Please clone it locally:

```bash
git clone git@github.com:Raffaelbdl/hyperbolic-nn-haiku.git
cd hyperbolic-nn-haiku
pip install -r requirements.txt
```

Make sure to install jax by following the [official guide](https://github.com/google/jax#installation).

## Content
The following content is currently implemented.

### Core
Contains most of the functions related to riemannian spaces (eg. Möbius operations, distance).
It also includes a wrapper for functions to make them applyable in the K-stereographic space.

### Layers
* `StereographicLinearLayer`: base linear layer in K-stereographic model
* `StereographicVanillaRNN`: base rnn layer in K-stereographic model
* `StereographicGRU`: gru cell in K-stereographic model

### Optimizers
* `rsgd`: base riemannian stochastic gradient descent
* `riemannian_adagrad`: riemannian version of the adagrad optimizer
* `riemannian_adam`: riemannian version of the adam optimizer

## References
This project is heavily inspired by [Geoopt](https://github.com/geoopt/geoopt) (Pytorch).

## Citing 

```bibtex
@software{hnn_haiku2022bolladilorenzo,
    title = {Hyperbolic Neural Networks with dm-haiku},
    author = {Raffael Bolla Di Lorenzo},
    url = {https://github.com/Raffaelbdl/hyperbolic-nn-haiku},
    version = {0.0.1},
    year = {2022}
}
```