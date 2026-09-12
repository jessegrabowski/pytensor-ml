import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers import (
    ConstantPad1D,
    ConstantPad2D,
    ReflectionPad1D,
    ReflectionPad2D,
    ReplicationPad1D,
    ReplicationPad2D,
    ZeroPad1D,
    ZeroPad2D,
)

floatX = pytensor.config.floatX


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "padding")))


@pytest.mark.parametrize(
    "layer, pad_width, numpy_mode, numpy_kwargs",
    [
        (
            ZeroPad2D(padding=((1, 2), (3, 4))),
            ((1, 2), (3, 4)),
            "constant",
            {"constant_values": 0.0},
        ),
        (
            ConstantPad2D(padding=((2, 1), (1, 3)), value=-1.5),
            ((2, 1), (1, 3)),
            "constant",
            {"constant_values": -1.5},
        ),
        (ReflectionPad2D(padding=((1, 2), (2, 1))), ((1, 2), (2, 1)), "reflect", {}),
        (ReplicationPad2D(padding=((2, 1), (1, 2))), ((2, 1), (1, 2)), "edge", {}),
    ],
    ids=["zero", "constant", "reflection", "replication"],
)
def test_each_mode_pads_the_spatial_axes_like_numpy(
    layer, pad_width, numpy_mode, numpy_kwargs, rng
):
    """numpy is the independent implementation here -- comparing against `pt.pad` would only restate
    what the layer already calls. The widths are spelled out rather than read back off the layer, so a
    resolver that swapped the axes or reversed an end would move both sides together and pass. Every
    case is asymmetric for the same reason."""
    X_np = rng.normal(size=(2, 6, 7, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 6, 7, 3), dtype=floatX)

    expected = np.pad(X_np, [(0, 0), *pad_width, (0, 0)], mode=numpy_mode, **numpy_kwargs)
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected)


@pytest.mark.parametrize(
    "layer, numpy_mode, numpy_kwargs",
    [
        (ZeroPad1D(padding=(2, 4)), "constant", {"constant_values": 0.0}),
        (ConstantPad1D(padding=(2, 4), value=3.5), "constant", {"constant_values": 3.5}),
        (ReflectionPad1D(padding=(2, 4)), "reflect", {}),
        (ReplicationPad1D(padding=(2, 4)), "edge", {}),
    ],
    ids=["zero", "constant", "reflection", "replication"],
)
def test_one_spatial_axis_reads_a_bare_pair_as_its_two_ends(layer, numpy_mode, numpy_kwargs, rng):
    """Over a single axis `(2, 4)` can only mean its two ends, and that is how torch reads it too --
    the per-axis reading would be a length mismatch."""
    X_np = rng.normal(size=(2, 9, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 9, 3), dtype=floatX)

    expected = np.pad(X_np, [(0, 0), (2, 4), (0, 0)], mode=numpy_mode, **numpy_kwargs)
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected)


def test_padding_wider_than_the_axis_keeps_reflecting(rng):
    """Torch refuses a reflection wider than what it has to mirror; numpy folds back and forth, and so
    do we. Pinned because it is a deliberate difference from the framework these layers are named
    after."""
    X_np = np.arange(4, dtype=floatX).reshape(1, 4, 1)
    X = pt.tensor("X", shape=(1, 4, 1), dtype=floatX)

    padded = pytensor.function([X], ReflectionPad1D(padding=5)(X))(X_np)
    np.testing.assert_allclose(padded, np.pad(X_np, [(0, 0), (5, 5), (0, 0)], mode="reflect"))


def test_the_batch_and_channel_axes_are_left_alone(rng):
    """Padding is spatial. A layer that padded every axis would still produce a plausible-looking
    array, so the untouched axes are asserted rather than assumed."""
    X_np = rng.normal(size=(2, 5, 6, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 5, 6, 3), dtype=floatX)

    padded = pytensor.function([X], ZeroPad2D(padding=2)(X))(X_np)
    assert padded.shape == (2, 9, 10, 3)
    np.testing.assert_allclose(padded[:, 2:-2, 2:-2, :], X_np)


def test_gradients_reach_the_input_through_the_padding(rng):
    """The padded positions are constants, so the gradient of a padded output has to arrive back at
    exactly the elements that came from the input and nowhere else."""
    X_np = rng.normal(size=(1, 4, 4, 1)).astype(floatX)
    X = pt.tensor("X", shape=(1, 4, 4, 1), dtype=floatX)

    padded = ZeroPad2D(padding=1)(X)
    gradient = pytensor.function([X], pt.grad(padded.sum(), X))(X_np)
    np.testing.assert_allclose(gradient, np.ones_like(X_np))


def test_an_input_of_the_wrong_rank_is_rejected():
    """The layers are channels-last over a fixed number of spatial axes, so a channels-first image or
    a batch of vectors is a mistake worth naming rather than padding the wrong axes."""
    with pytest.raises(ValueError, match="4-dimensional input"):
        ZeroPad2D(padding=1)(pt.tensor("X", shape=(2, 8, 3), dtype=floatX))


@pytest.mark.parametrize(
    "padding, message",
    [
        (-1, "cannot be negative"),
        (((1, 2), (3, -4)), "cannot be negative"),
        ((1, 2, 3), "one amount per axis"),
    ],
    ids=["negative_scalar", "negative_in_pair", "wrong_count"],
)
def test_padding_amounts_are_validated(padding, message):
    """Each of these describes something the layer cannot do, and each would otherwise surface as a
    confusing shape further downstream."""
    with pytest.raises(ValueError, match=message):
        ZeroPad2D(padding=padding)
