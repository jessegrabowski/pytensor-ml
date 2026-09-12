import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers import Conv2D, ConvTranspose1D, ConvTranspose2D, Input, Linear
from pytensor_ml.layers.combinators import Flatten
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import adam

floatX = pytensor.config.floatX
ATOL = 1e-4 if floatX == "float32" else 1e-8


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "conv transpose")))


def scatter_through_kernel(X, W, stride, padding, output_padding, dilation):
    """
    Scatter each input position through the kernel, which is the definition of the operation.

    Written out as loops rather than built from the library's own ops, so the tests below compare the
    layer against the arithmetic it claims to do rather than against another spelling of itself.
    ``padding`` is one amount per axis because a transposed convolution always crops the same number
    of elements from both ends.
    """
    batch, *spatial, _ = X.shape
    kernel, out_channels = W.shape[:-2], W.shape[-1]
    uncropped = [
        (length - 1) * step + spacing * (extent - 1) + 1 + extra
        for length, step, spacing, extent, extra in zip(
            spatial, stride, dilation, kernel, output_padding
        )
    ]
    out = np.zeros((batch, *uncropped, out_channels), dtype=X.dtype)
    for source in np.ndindex(*spatial):
        for tap in np.ndindex(*kernel):
            target = tuple(i * s + t * d for i, s, t, d in zip(source, stride, tap, dilation))
            out[(slice(None), *target)] += X[(slice(None), *source)] @ W[tap]
    cropped = tuple(slice(before, size - before) for before, size in zip(padding, uncropped))
    return out[(slice(None), *cropped, slice(None))]


@pytest.mark.parametrize(
    "stride, padding, output_padding, dilation",
    [
        (1, 0, 0, 1),
        (2, 0, 0, 1),
        (2, 1, 1, 1),
        (3, 2, 2, 1),
        (1, 1, 0, 2),
        ((2, 3), (1, 2), (1, 2), 1),
    ],
    ids=["plain", "strided", "strided_cropped", "wide_stride", "dilated", "per_axis"],
)
def test_conv_transpose_2d_scatters_its_input_through_the_kernel(
    stride, padding, output_padding, dilation, rng
):
    """Every argument changes where a contribution lands, so each is exercised against the scatter the
    layer stands for. The per-axis case is the one that catches an axis order swapped somewhere."""
    X_np = rng.normal(size=(2, 5, 6, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 5, 6, 3), dtype=floatX)
    layer = ConvTranspose2D(
        "transpose",
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        bias=False,
    )
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))

    expected = scatter_through_kernel(
        X_np,
        layer.W.get_value(),
        layer.stride,
        [before for before, _ in layer.padding],
        layer.output_padding,
        layer.dilation,
    )
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected, atol=ATOL)


@pytest.mark.parametrize(
    "stride, padding, output_padding",
    [(1, 0, 0), (2, 1, 1), (3, 1, 2)],
    ids=["plain", "strided_cropped", "wide_stride"],
)
def test_conv_transpose_1d_scatters_its_input_through_the_kernel(
    stride, padding, output_padding, rng
):
    """The rank-1 layer shares `_ConvTransposeNd`, so this checks the spatial count reaches every
    argument rather than repeating the rank-2 coverage."""
    X_np = rng.normal(size=(2, 7, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 7, 3), dtype=floatX)
    layer = ConvTranspose1D(
        "transpose",
        in_channels=3,
        out_channels=5,
        kernel_size=4,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        bias=False,
    )
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))

    expected = scatter_through_kernel(
        X_np,
        layer.W.get_value(),
        layer.stride,
        [before for before, _ in layer.padding],
        layer.output_padding,
        layer.dilation,
    )
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected, atol=ATOL)


def test_the_bias_is_added_once_per_output_channel(rng):
    """The bias is added after the scatter, so a layer that folded it in before would add it once per
    contributing tap instead of once per output position."""
    X_np = rng.normal(size=(2, 5, 5, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 5, 5, 3), dtype=floatX)
    layer = ConvTranspose2D("transpose", in_channels=3, out_channels=4, kernel_size=3, stride=2)
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))
    bias = rng.normal(size=(4,)).astype(floatX)
    layer.b.set_value(bias)

    with_bias = pytensor.function([X], layer(X))(X_np)
    layer.b.set_value(np.zeros_like(bias))
    without = pytensor.function([X], layer(X))(X_np)
    np.testing.assert_allclose(
        with_bias - without, np.broadcast_to(bias, with_bias.shape), atol=ATOL
    )


def test_conv_transpose_inverts_a_convolutions_shape():
    """The operation exists to undo what a convolution did to a shape, so a transpose built with the
    same geometry has to return the size the convolution consumed."""
    X = pt.tensor("X", shape=(None, 9, 9, 3), dtype=floatX)
    forward = Conv2D("down", in_channels=3, out_channels=4, kernel_size=3, stride=2)
    backward = ConvTranspose2D("up", in_channels=4, out_channels=3, kernel_size=3, stride=2)

    reduced = forward(X)
    assert reduced.type.shape[1:3] == (4, 4)
    assert backward(reduced).type.shape[1:3] == (9, 9)


@pytest.mark.parametrize("output_padding", [2, -1], ids=["equals_stride", "negative"])
def test_output_padding_must_stay_below_the_stride(output_padding):
    """A stride of `s` maps `s` input sizes onto one output size, so an `output_padding` of `s` names a
    size the stride never produced, and a negative one names nothing at all. Both are caught at
    construction rather than as a shape error later."""
    with pytest.raises(ValueError, match="less than that stride"):
        ConvTranspose2D(
            "t",
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            output_padding=output_padding,
        )


def test_padding_does_not_take_the_forward_layers_keywords():
    """`padding="same"` reads as a request the layer cannot honour: here padding removes output rather
    than adding input, so there is no size for "same" to preserve."""
    with pytest.raises(ValueError, match="removes elements from its output"):
        ConvTranspose2D("t", in_channels=1, out_channels=1, kernel_size=3, padding="same")


def test_an_encoder_decoder_trains_end_to_end(rng):
    """The step the layer exists for: something that downsamples, then upsamples back, learning through
    both. Gradients have to cross the transpose, which is the op run forward, so this is what says its
    pullback works inside a real graph rather than only under `verify_grad`."""
    X = Input("X", shape=(None, 8, 8, 2))
    encoded = Conv2D("down", in_channels=2, out_channels=4, kernel_size=3, stride=2)(X)
    decoded = ConvTranspose2D("up", in_channels=4, out_channels=2, kernel_size=3, stride=2)(encoded)
    y = Linear("head", n_in=7 * 7 * 2, n_out=1)(Flatten(decoded))

    model = Model(y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())

    X_np = rng.normal(size=(32, 8, 8, 2)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2, 3))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


def test_cropping_more_than_the_output_holds_is_rejected():
    """Padding here removes output rather than adding input, so enough of it leaves a zero-length axis.
    That reaches the op and comes back empty rather than failing, which is the shape of mistake worth
    catching where the input's length is known."""
    X = pt.tensor("X", shape=(2, 2, 2, 3), dtype=floatX)
    layer = ConvTranspose2D("t", in_channels=3, out_channels=4, kernel_size=3, padding=2)

    with pytest.raises(ValueError, match="leaving nothing"):
        layer(X)


def test_a_crop_that_only_a_short_input_cannot_afford_is_allowed():
    """The same crop that empties a length-1 input is right for a longer one, so the check has to read
    the input rather than the arguments alone -- rejecting this at construction would be wrong."""
    X = pt.tensor("X", shape=(2, 2, 2, 3), dtype=floatX)
    layer = ConvTranspose2D("t", in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=2)

    assert layer(X).type.shape[1:3] == (1, 1)


def test_a_symbolic_spatial_extent_skips_the_crop_check(rng):
    """Only a statically known length can be checked, so a symbolic one is let through rather than
    guessed at -- and then has to compute the same answer a declared length would."""
    symbolic = pt.tensor("symbolic", shape=(None, None, None, 3), dtype=floatX)
    declared = pt.tensor("declared", shape=(2, 5, 5, 3), dtype=floatX)
    layer = ConvTranspose2D(
        "t", in_channels=3, out_channels=4, kernel_size=3, padding=2, bias=False
    )
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))
    X_np = rng.normal(size=(2, 5, 5, 3)).astype(floatX)

    unchecked = pytensor.function([symbolic], layer(symbolic))(X_np)
    checked = pytensor.function([declared], layer(declared))(X_np)
    assert unchecked.shape == (2, 3, 3, 4)
    np.testing.assert_allclose(unchecked, checked, atol=ATOL)
