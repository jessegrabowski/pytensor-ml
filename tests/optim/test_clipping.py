import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import clip_by_global_norm, clip_by_value
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function


def test_clip_by_global_norm_rescales_oversized_step():
    p = trainable(np.zeros(3), name="w")
    step = pt.constant(np.array([3.0, 4.0, 0.0]))  # global norm 5
    out = clip_by_global_norm(1.0)({p: p + step}, [p])
    # clip = 1 / (5 + eps); new p = clip * step -> unit norm in the same direction
    np.testing.assert_allclose(function([], out[p])(), [0.6, 0.8, 0.0], rtol=1e-6)


def test_clip_by_global_norm_leaves_small_step():
    p = trainable(np.zeros(2), name="w")
    step = pt.constant(np.array([0.3, 0.4]))  # norm 0.5 < 1.0
    out = clip_by_global_norm(1.0)({p: p + step}, [p])
    np.testing.assert_allclose(function([], out[p])(), [0.3, 0.4], rtol=1e-6)


def test_clip_by_global_norm_uses_joint_norm_across_params():
    weight = trainable(np.zeros(2), name="w")
    bias = trainable(np.zeros(1), name="b")
    # Joint norm sqrt(3^2 + 4^2) = 5, so every step is scaled by 1/5; a per-parameter
    # norm would instead scale the two steps by 1/3 and 1/4.
    updates = {
        weight: weight + pt.constant(np.array([3.0, 0.0])),
        bias: bias + pt.constant(np.array([4.0])),
    }
    out = clip_by_global_norm(1.0)(updates, [weight, bias])
    new_weight, new_bias = function([], [out[weight], out[bias]])()
    np.testing.assert_allclose(new_weight, [0.6, 0.0], rtol=1e-6)
    np.testing.assert_allclose(new_bias, [0.8], rtol=1e-6)


def test_clip_by_value_clamps_elementwise():
    p = trainable(np.zeros(3), name="w")
    step = pt.constant(np.array([5.0, -5.0, 0.5]))
    out = clip_by_value(-1.0, 1.0)({p: p + step}, [p])
    np.testing.assert_allclose(function([], out[p])(), [1.0, -1.0, 0.5])


@pytest.mark.parametrize("max_norm", [0.0, -1.0], ids=["zero", "negative"])
def test_clip_by_global_norm_rejects_a_non_positive_norm(max_norm):
    with pytest.raises(ValueError, match="max_norm is a norm to clip to"):
        clip_by_global_norm(max_norm)


def test_clip_by_global_norm_takes_a_shape_derived_norm():
    """A bound written as arithmetic on a shape has no value until the function runs, so it has to reach
    the graph rather than be refused for failing a comparison that cannot be made yet."""
    X = pt.matrix("X")
    p = trainable(np.zeros(3), name="w")
    step = pt.constant(np.array([3.0, 4.0, 0.0]))  # global norm 5
    out = clip_by_global_norm(X.shape[0])({p: p + step}, [p])
    clipped_to = function([X], out[p])

    np.testing.assert_allclose(clipped_to(np.zeros((1, 3))), [0.6, 0.8, 0.0], rtol=1e-6)
    # A bound of 10 exceeds the step's norm of 5, so the step passes through untouched.
    np.testing.assert_allclose(clipped_to(np.zeros((10, 3))), [3.0, 4.0, 0.0], rtol=1e-6)


def test_a_shape_derived_norm_is_checked_when_the_function_runs():
    """The build-time check cannot fire on a bound with no value yet, so it travels with the graph. The
    jax backend drops assertions, which is why this is the best that can be offered rather than a
    guarantee."""
    X = pt.matrix("X")
    p = trainable(np.zeros(3), name="w")
    out = clip_by_global_norm(X.shape[0])({p: p + pt.constant(np.array([3.0, 4.0, 0.0]))}, [p])
    clipped_to = function([X], out[p])

    with pytest.raises(ValueError, match="max_norm is a norm to clip to"):
        clipped_to(np.zeros((0, 3)))


@pytest.mark.parametrize(
    ("min_value", "max_value", "complaint"),
    [
        (1.0, -1.0, "max_value is the upper bound"),
        (pt.vector("v"), 1.0, "min_value must be a single number"),
        (-1.0, pt.vector("v"), "max_value must be a single number"),
    ],
    ids=["inverted", "non-scalar-lower", "non-scalar-upper"],
)
def test_clip_by_value_rejects_a_malformed_range(min_value, max_value, complaint):
    """An inverted range reaches `pt.clip`, which returns the lower bound for every element rather than
    complaining, so nothing downstream would reveal the mistake."""
    with pytest.raises(ValueError, match=complaint):
        clip_by_value(min_value, max_value)


def test_clip_by_value_allows_a_degenerate_range():
    """Equal bounds clamp every coordinate to one value. That is odd but well defined, so it builds."""
    p = trainable(np.zeros(2), name="w")
    out = clip_by_value(0.5, 0.5)({p: p + pt.constant(np.array([-3.0, 7.0]))}, [p])
    np.testing.assert_allclose(function([], out[p])(), [0.5, 0.5], rtol=1e-6)


def test_clip_by_value_takes_shape_derived_bounds():
    X = pt.matrix("X")
    p = trainable(np.zeros(3), name="w")
    step = pt.constant(np.array([-9.0, 0.5, 9.0]))
    out = clip_by_value(-X.shape[0], X.shape[0])({p: p + step}, [p])
    clamped_to = function([X], out[p])

    np.testing.assert_allclose(clamped_to(np.zeros((1, 3))), [-1.0, 0.5, 1.0], rtol=1e-6)
    np.testing.assert_allclose(clamped_to(np.zeros((4, 3))), [-4.0, 0.5, 4.0], rtol=1e-6)


def test_a_shape_derived_range_is_checked_when_the_function_runs():
    """The build-time check cannot fire on bounds with no value yet, so it travels with the graph. The
    jax backend drops assertions, which is why this is the best that can be offered rather than a
    guarantee."""
    X = pt.matrix("X")
    p = trainable(np.zeros(2), name="w")
    # Inverted for any batch with rows: the lower bound climbs above the fixed upper bound of 1.
    out = clip_by_value(X.shape[0], 1.0)({p: p + pt.constant(np.array([0.0, 0.0]))}, [p])
    clamped_to = function([X], out[p])

    np.testing.assert_allclose(clamped_to(np.zeros((1, 2))), [1.0, 1.0], rtol=1e-6)
    with pytest.raises(ValueError, match="max_value is the upper bound"):
        clamped_to(np.zeros((5, 2)))
