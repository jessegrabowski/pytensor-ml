import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers.attention import scaled_dot_product_attention
from pytensor_ml.layers.positional import RotaryEmbedding, rotary_embedding
from tests.test_attention import sdpa_np

floatX = pytensor.config.floatX


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "Rotary Test")))


def rope_np(x, positions, *, base=10_000.0, pairing="half", scaling="none", scaling_factor=1.0):
    """
    Independent RoPE reference, built pair by pair from explicit 2x2 rotations.

    Deliberately written from the RoFormer definition rather than from the implementation under test:
    it imports no pytensor and no pytensor_ml, indexes scalars instead of slicing halves, and applies
    the rotation matrix literally. A shared mistake in the pairing, a sign, or the frequency ladder
    therefore cannot cancel out between the two.

    Parameters
    ----------
    x : ndarray
        Shape ``(..., seq, head_dim)``.
    positions : ndarray
        Shape ``(seq,)``, shared by every leading axis of ``x``.
    """
    x = np.asarray(x, dtype="float64")
    positions = np.asarray(positions)
    head_dim = x.shape[-1]
    half = head_dim // 2

    if scaling == "ntk":
        base = base * scaling_factor ** (head_dim / (head_dim - 2))

    theta = np.array([base ** (-2.0 * i / head_dim) for i in range(half)])
    if scaling == "linear":
        theta = theta / scaling_factor

    out = np.empty_like(x)
    for lead in np.ndindex(*x.shape[:-2]):
        for step, position in enumerate(positions):
            for i in range(half):
                first, second = (i, i + half) if pairing == "half" else (2 * i, 2 * i + 1)
                cos, sin = np.cos(position * theta[i]), np.sin(position * theta[i])
                x_first, x_second = x[(*lead, step, first)], x[(*lead, step, second)]
                out[(*lead, step, first)] = x_first * cos - x_second * sin
                out[(*lead, step, second)] = x_second * cos + x_first * sin

    return out


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
@pytest.mark.parametrize(
    "scaling, scaling_factor",
    [("none", 1.0), ("linear", 4.0), ("ntk", 4.0)],
    ids=["unscaled", "linear", "ntk"],
)
def test_rope_matches_independent_reference(pairing, scaling, scaling_factor, rng):
    x_np = rng.normal(size=(2, 3, 5, 8)).astype(floatX)
    positions_np = np.arange(5)

    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")
    out = rotary_embedding(
        x, positions, pairing=pairing, scaling=scaling, scaling_factor=scaling_factor
    )

    result = out.eval({x: x_np, positions: positions_np})
    expected = rope_np(
        x_np, positions_np, pairing=pairing, scaling=scaling, scaling_factor=scaling_factor
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
def test_rope_one_step_at_a_time_equals_whole_sequence(pairing, rng):
    x_np = rng.normal(size=(2, 3, 6, 4)).astype(floatX)
    positions_np = np.arange(6)

    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")
    whole = rotary_embedding(x, positions, pairing=pairing).eval({x: x_np, positions: positions_np})

    step_x = pt.tensor("step_x", shape=(2, 3, 1, 4))
    step_out = rotary_embedding(step_x, positions, pairing=pairing)
    for step, position in enumerate(positions_np):
        one = step_out.eval({step_x: x_np[:, :, step : step + 1], positions: np.array([position])})
        np.testing.assert_allclose(one[:, :, 0], whole[:, :, step], atol=1e-6)


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
def test_rope_scores_depend_only_on_relative_position(pairing, rng):
    q_np = rng.normal(size=(1, 1, 1, 8)).astype(floatX)
    k_np = rng.normal(size=(1, 1, 1, 8)).astype(floatX)

    q, k = pt.tensor("q", shape=q_np.shape), pt.tensor("k", shape=k_np.shape)
    positions = pt.lvector("positions")
    q_out = rotary_embedding(q, positions, pairing=pairing)
    k_out = rotary_embedding(k, positions, pairing=pairing)

    def score(query_position, key_position):
        rotated_q = q_out.eval({q: q_np, positions: np.array([query_position])})
        rotated_k = k_out.eval({k: k_np, positions: np.array([key_position])})
        return float((rotated_q * rotated_k).sum())

    np.testing.assert_allclose(score(3, 0), score(10, 7), rtol=1e-6)
    np.testing.assert_allclose(score(3, 0), score(103, 100), rtol=1e-6)
    assert not np.isclose(score(3, 0), score(4, 0))


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
def test_rope_is_an_orthogonal_rotation(pairing, rng):
    x_np = rng.normal(size=(2, 4, 6)).astype(floatX)
    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")
    out = rotary_embedding(x, positions, pairing=pairing)

    rotated = out.eval({x: x_np, positions: np.arange(4)})
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=-1), np.linalg.norm(x_np, axis=-1), rtol=1e-6
    )

    identity = out.eval({x: x_np, positions: np.zeros(4, dtype="int64")})
    np.testing.assert_allclose(identity, x_np, atol=1e-6)


def test_rope_positions_broadcast_shared_and_per_sequence(rng):
    x_np = rng.normal(size=(2, 3, 5, 8)).astype(floatX)
    x = pt.tensor("x", shape=x_np.shape)

    shared = pt.lvector("shared")
    per_sequence = pt.lmatrix("per_sequence")
    shared_out = rotary_embedding(x, shared)
    batched_out = rotary_embedding(x, per_sequence)

    positions_np = np.arange(5)
    shared_result = shared_out.eval({x: x_np, shared: positions_np})
    repeated = batched_out.eval({x: x_np, per_sequence: np.tile(positions_np, (2, 1))})
    np.testing.assert_allclose(shared_result, repeated, atol=1e-6)

    offsets = np.stack([positions_np, positions_np + 100])
    offset_result = batched_out.eval({x: x_np, per_sequence: offsets})
    np.testing.assert_allclose(offset_result[0], shared_result[0], atol=1e-6)
    np.testing.assert_allclose(offset_result[1], rope_np(x_np[1], positions_np + 100), atol=1e-6)


@pytest.mark.parametrize("scaling", ["linear", "ntk"])
def test_rope_scaling_factor_one_is_the_identity(scaling, rng):
    x_np = rng.normal(size=(3, 6)).astype(floatX)
    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")
    positions_np = np.arange(3)

    unscaled = rotary_embedding(x, positions).eval({x: x_np, positions: positions_np})
    scaled = rotary_embedding(x, positions, scaling=scaling, scaling_factor=1.0).eval(
        {x: x_np, positions: positions_np}
    )
    np.testing.assert_allclose(unscaled, scaled, atol=1e-6)


def test_linear_scaling_interpolates_positions(rng):
    x_np = rng.normal(size=(1, 6)).astype(floatX)
    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")

    interpolated = rotary_embedding(x, positions, scaling="linear", scaling_factor=4.0).eval(
        {x: x_np, positions: np.array([8])}
    )
    plain = rotary_embedding(x, positions).eval({x: x_np, positions: np.array([2])})
    np.testing.assert_allclose(interpolated, plain, atol=1e-6)


def test_rope_composes_with_attention(rng):
    q_np, k_np, v_np = (rng.normal(size=(2, 3, 5, 4)).astype(floatX) for _ in range(3))
    q = pt.tensor("q", shape=q_np.shape)
    k = pt.tensor("k", shape=k_np.shape)
    v = pt.tensor("v", shape=v_np.shape)
    positions = pt.lvector("positions")
    positions_np = np.arange(5)
    options = dict(base=500.0, pairing="adjacent", scaling="linear", scaling_factor=2.5)

    rope = RotaryEmbedding("rope", **options)
    output = scaled_dot_product_attention(rope(q, positions), rope(k, positions), v, is_causal=True)
    result = output.eval({q: q_np, k: k_np, v: v_np, positions: positions_np})
    expected = sdpa_np(
        rope_np(q_np, positions_np, **options),
        rope_np(k_np, positions_np, **options),
        v_np,
        is_causal=True,
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
def test_rope_input_gradient_matches_transpose_rotation(pairing, rng):
    """Backpropagation applies the inverse rotation to the output cotangent."""
    x_np = rng.normal(size=(2, 3, 4)).astype(floatX)
    weights = rng.normal(size=x_np.shape).astype(floatX)
    positions_np = np.array([1, 7, 19])
    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")
    out = rotary_embedding(x, positions, pairing=pairing)

    grad = pt.grad(((out * weights) ** 2).sum(), x).eval({x: x_np, positions: positions_np})
    cotangent = 2 * weights**2 * rope_np(x_np, positions_np, pairing=pairing)
    expected = rope_np(cotangent, -positions_np, pairing=pairing)
    np.testing.assert_allclose(grad, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "pairing",
    [
        pytest.param(
            "half",
            marks=pytest.mark.xfail(
                condition=pytensor.config.mode == "JAX",
                reason="PyTensor's JAX lowering requires static slice bounds for the half split",
                raises=IndexError,
                strict=True,
            ),
        ),
        "adjacent",
    ],
)
@pytest.mark.xfail(
    condition=pytensor.config.mode == "MLX",
    reason="PyTensor's MLX arange lowering requires a statically known head dimension",
    raises=NotImplementedError,
    strict=True,
)
def test_unknown_head_dimension_is_supported(pairing, rng):
    x_np = rng.normal(size=(3, 8)).astype(floatX)
    positions_np = np.arange(3)

    x = pt.tensor("x", shape=(3, None))
    positions = pt.lvector("positions")
    out = rotary_embedding(x, positions, pairing=pairing)

    np.testing.assert_allclose(
        out.eval({x: x_np, positions: positions_np}),
        rope_np(x_np, positions_np, pairing=pairing),
        atol=1e-6,
    )


def test_odd_head_dimension_raises():
    with pytest.raises(ValueError, match="head_dim must be even"):
        rotary_embedding(pt.tensor("x", shape=(4, 7)), pt.lvector("positions"))


def test_integer_input_raises():
    with pytest.raises(ValueError, match="floating-point input"):
        rotary_embedding(pt.tensor("x", shape=(4, 8), dtype="int64"), pt.lvector("positions"))


def test_positions_wider_than_input_raises():
    with pytest.raises(ValueError, match="more than the 1 non-feature dimensions"):
        rotary_embedding(pt.tensor("x", shape=(4, 8)), pt.lmatrix("positions"))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pairing": "interleaved"}, "pairing must be one of"),
        ({"scaling": "yarn"}, "scaling must be one of"),
        ({"scaling": "linear", "scaling_factor": 0.0}, "scaling_factor must be positive"),
    ],
    ids=["bad_pairing", "bad_scaling", "bad_factor"],
)
def test_invalid_options_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rotary_embedding(pt.tensor("x", shape=(4, 8)), pt.lvector("positions"), **kwargs)
    with pytest.raises(ValueError, match=match):
        RotaryEmbedding("rope", **kwargs)


def test_ntk_scaling_needs_more_than_two_channels():
    with pytest.raises(ValueError, match="undefined for head_dim <= 2"):
        rotary_embedding(
            pt.tensor("x", shape=(4, 2)), pt.lvector("positions"), scaling="ntk", scaling_factor=2.0
        )
