import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.gradient import verify_grad

from pytensor_ml.layers import Upsample1D, Upsample2D

floatX = pytensor.config.floatX
ATOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml upsample")))


@pytest.mark.parametrize("factor", [2, 3, (2, 3)], ids=["square", "square_odd", "non_square"])
def test_nearest_repeats_each_element(factor, rng):
    height_factor, width_factor = (factor, factor) if isinstance(factor, int) else factor
    X = pt.tensor("X", shape=(None, 3, 4, 2))
    out = Upsample2D(scale_factor=factor)(X)

    X_np = rng.normal(size=(2, 3, 4, 2)).astype(floatX)
    expected = np.repeat(np.repeat(X_np, height_factor, axis=1), width_factor, axis=2)

    np.testing.assert_allclose(out.eval({X: X_np}), expected, rtol=1e-6, atol=ATOL)


def test_nearest_indexes_by_floor_of_the_ratio():
    """Stretching 5 to 7 does not repeat evenly, and which elements get the extra copy is the whole
    convention: output j reads input floor(j * 5 / 7)."""
    X = pt.tensor("X", shape=(None, 5, 1))
    out = Upsample1D(size=7)(X)

    X_np = np.arange(5, dtype=floatX).reshape(1, 5, 1)

    np.testing.assert_allclose(out.eval({X: X_np}).ravel(), [0, 0, 1, 2, 2, 3, 4])


def test_bilinear_half_pixel_convention():
    """With align_corners=False each element covers a unit interval and output centers map to input
    centers, so a 2x2 doubled leaves the corners unchanged and the outermost samples land outside
    the input and clamp. This is torch's default and the one every diffusion decoder uses."""
    X = pt.tensor("X", shape=(1, 2, 2, 1))
    out = Upsample2D(scale_factor=2, mode="bilinear")(X)

    X_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=floatX).reshape(1, 2, 2, 1)
    expected = [
        [1.00, 1.25, 1.75, 2.00],
        [1.50, 1.75, 2.25, 2.50],
        [2.50, 2.75, 3.25, 3.50],
        [3.00, 3.25, 3.75, 4.00],
    ]

    np.testing.assert_allclose(out.eval({X: X_np})[0, :, :, 0], expected, rtol=1e-6, atol=ATOL)


def test_bilinear_corner_aligned_convention():
    """With align_corners=True the outermost outputs are pinned to the outermost inputs and the rest
    are spread evenly between them, which puts the samples a half pixel away from where the default
    convention puts them."""
    X = pt.tensor("X", shape=(1, 2, 2, 1))
    out = Upsample2D(scale_factor=2, mode="bilinear", align_corners=True)(X)

    X_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=floatX).reshape(1, 2, 2, 1)
    third = 1.0 / 3.0
    expected = [
        [1.0, 1 + third, 1 + 2 * third, 2.0],
        [1 + 2 * third, 2.0, 2 + third, 2 + 2 * third],
        [2 + third, 2 + 2 * third, 3.0, 3 + third],
        [3.0, 3 + third, 3 + 2 * third, 4.0],
    ]

    np.testing.assert_allclose(out.eval({X: X_np})[0, :, :, 0], expected, rtol=1e-6, atol=ATOL)


@pytest.mark.parametrize(
    "align_corners, expected",
    [(False, [1.0, 1.25, 1.75, 2.25, 2.75, 3.0]), (True, [1.0, 1.4, 1.8, 2.2, 2.6, 3.0])],
    ids=["half_pixel", "corner_aligned"],
)
def test_linear_interpolates_along_a_sequence(align_corners, expected):
    X = pt.tensor("X", shape=(1, 3, 1))
    out = Upsample1D(scale_factor=2, mode="linear", align_corners=align_corners)(X)

    X_np = np.array([1.0, 2.0, 3.0], dtype=floatX).reshape(1, 3, 1)

    np.testing.assert_allclose(out.eval({X: X_np}).ravel(), expected, rtol=1e-6, atol=ATOL)


@pytest.mark.parametrize("size", [(8, 6), (2, 9)], ids=["grow", "shrink_one_axis"])
def test_explicit_size_resamples_to_that_extent(size, rng):
    """A size that is not a uniform factor of the input, so nothing else covers these extents. The
    axes are given height-first, which a swap turns into the wrong shape rather than wrong values."""
    X = pt.tensor("X", shape=(None, 3, 4, 2))
    out = Upsample2D(size=size, mode="bilinear")(X)

    res = out.eval({X: rng.normal(size=(2, 3, 4, 2)).astype(floatX)})

    assert out.type.shape == (None, *size, 2)
    assert res.shape == (2, *size, 2)


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_explicit_size_agrees_with_the_equivalent_factor(mode, rng):
    """Asking a 3x4 input for size (6, 8) is the same resampling as scale_factor=2, reached through
    the other branch of the extent resolution."""
    X = pt.tensor("X", shape=(None, 3, 4, 2))
    X_np = rng.normal(size=(2, 3, 4, 2)).astype(floatX)

    np.testing.assert_allclose(
        Upsample2D(size=(6, 8), mode=mode)(X).eval({X: X_np}),
        Upsample2D(scale_factor=2, mode=mode)(X).eval({X: X_np}),
        rtol=1e-6,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "align_corners, expected", [(False, 6.5), (True, 1.0)], ids=["half_pixel", "corner_aligned"]
)
def test_a_single_element_output_is_well_defined(align_corners, expected):
    """Spreading over a closed interval divides by one less than the output extent, which is zero
    when the output is a single element."""
    X = pt.tensor("X", shape=(1, 3, 4, 1))
    out = Upsample2D(size=1, mode="bilinear", align_corners=align_corners)(X)

    X_np = np.arange(1, 13, dtype=floatX).reshape(1, 3, 4, 1)

    np.testing.assert_allclose(out.eval({X: X_np}).ravel(), [expected], rtol=1e-6, atol=ATOL)


def test_a_known_extent_survives_the_gather():
    """The resampled extent is an ordinary index gather, which reports a static extent only when the
    index vector has one. A downstream pool checks its window against these."""
    X = pt.tensor("X", shape=(None, 3, 4, 2))

    assert Upsample2D(scale_factor=2)(X).type.shape == (None, 6, 8, 2)


def test_an_unknown_extent_still_resamples(rng):
    """The extents go symbolic when the input's are unknown, which must change the graph and not the
    answer."""
    dynamic = pt.tensor("dynamic", shape=(None, None, None, 2))
    static = pt.tensor("static", shape=(None, 3, 4, 2))
    out = Upsample2D(scale_factor=2, mode="bilinear")(dynamic)

    assert out.type.shape == (None, None, None, 2)

    X_np = rng.normal(size=(1, 3, 4, 2)).astype(floatX)
    np.testing.assert_allclose(
        out.eval({dynamic: X_np}),
        Upsample2D(scale_factor=2, mode="bilinear")(static).eval({static: X_np}),
        rtol=1e-6,
        atol=ATOL,
    )


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_upsample_gradient(mode, rng):
    X_np = rng.normal(size=(2, 3, 4, 2)).astype(floatX)

    verify_grad(Upsample2D(scale_factor=2, mode=mode), [X_np], rng=np.random.default_rng(0))


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_resampling_keeps_the_input_dtype(dtype, mode):
    """The source coordinates divide integer extents, and an integer division is float64 whatever
    floatX is, so the upcast lands in a float32 network's activations."""
    with pytensor.config.change_flags(floatX=dtype):
        X = pt.tensor("X", shape=(None, 3, 4, 2), dtype=dtype)

        assert Upsample2D(scale_factor=2, mode=mode)(X).dtype == dtype


# Every way of asking for a resampling that means nothing. `mode` is spelled for its rank, so the
# name that works on one class is wrong on the other rather than a harmless alias.
REJECTED_CONFIGURATIONS = [
    (Upsample2D, {}, "exactly one of scale_factor and size"),
    (Upsample2D, {"scale_factor": 2, "size": 4}, "exactly one of scale_factor and size"),
    (Upsample2D, {"scale_factor": 2, "mode": "linear"}, "interpolates either 'nearest'"),
    (Upsample1D, {"scale_factor": 2, "mode": "bilinear"}, "interpolates either 'nearest'"),
    (Upsample2D, {"scale_factor": 2, "align_corners": True}, "no corners to align"),
    (Upsample2D, {"scale_factor": 0}, "needs a positive"),
    (Upsample2D, {"size": (4, -1)}, "needs a positive"),
]


@pytest.mark.parametrize(
    "layer, kwargs, message",
    REJECTED_CONFIGURATIONS,
    ids=[
        "neither",
        "both",
        "2d_linear",
        "1d_bilinear",
        "corners_without_interpolation",
        "zero_factor",
        "negative_size",
    ],
)
def test_a_meaningless_configuration_raises(layer, kwargs, message):
    with pytest.raises(ValueError, match=message):
        layer(**kwargs)


def test_wrong_rank_raises():
    with pytest.raises(ValueError, match="4-dimensional"):
        Upsample2D(scale_factor=2)(pt.tensor("X", shape=(None, 4, 2)))


def test_a_factor_in_the_name_slot_raises():
    with pytest.raises(TypeError, match=r"Upsample2D\(scale_factor=2\)"):
        Upsample2D(2)
