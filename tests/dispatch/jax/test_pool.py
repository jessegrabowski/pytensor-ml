from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("jax")

from pytensor_ml.layers import AvgPool2D, MaxPool1D, MaxPool2D
from tests.dispatch.jax.test_basic import compare_jax_and_py

floatX = pytensor.config.floatX
assert_close = partial(np.testing.assert_allclose, atol=1e-5, rtol=1e-4)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "jax pool")))


@pytest.mark.parametrize("layer_cls", [MaxPool2D, AvgPool2D], ids=["max", "avg"])
def test_the_pool_op_matches_the_graph_it_replaces(layer_cls, rng):
    """Both reductions, with a rectangular window and a stride that overlaps it, so a backend that
    swapped the spatial axes or tiled where it should slide would show up."""
    X_np = rng.normal(size=(2, 9, 11, 3)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)

    out = layer_cls("pool", kernel_size=(2, 3), stride=(2, 1))(X)
    compare_jax_and_py([X], out, [X_np], assert_fn=assert_close)


@pytest.mark.parametrize("layer_cls", [MaxPool2D, AvgPool2D], ids=["max", "avg"])
def test_the_pool_gradient_dispatch_matches_the_graph(layer_cls, rng):
    """The backend differentiates its own pooling, so this checks two engines agree rather than that
    one of them runs."""
    X_np = rng.normal(size=(2, 8, 10, 3)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)

    gradient = pt.grad((layer_cls("pool", kernel_size=2)(X) ** 2).sum(), X)
    compare_jax_and_py([X], gradient, [X_np], assert_fn=assert_close)


def test_max_pooling_conserves_the_gradient_of_a_tied_window():
    """A tied window is routine after a rectifier, and the frameworks disagree about which subgradient
    to take -- jax and torch give the whole cotangent to one tap, mlx splits it. What every backend has
    to agree on is that a window returns exactly the gradient it received, which is the property
    pytensor's own `max` breaks by handing the full cotangent to each tied tap."""
    X_np = np.array([1.0, 1.0, 2.0, 2.0], dtype=floatX).reshape(1, 4, 1)
    X = pt.tensor("X", shape=X_np.shape)

    gradient = pt.grad(MaxPool1D("pool", kernel_size=2)(X).sum(), X)
    dispatched = pytensor.function([X], gradient, mode="JAX")(X_np)
    assert_close(np.asarray(dispatched).reshape(2, 2).sum(axis=1), [1.0, 1.0])
