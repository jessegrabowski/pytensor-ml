import importlib
import inspect

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.mode import Mode
from pytensor.gradient import verify_grad
from pytensor.graph.replace import vectorize_graph

import pytensor_ml.layers

from pytensor_ml.activations import ReLU
from pytensor_ml.base import Layer
from pytensor_ml.layers import (
    BatchNorm,
    Conv2D,
    Dropout,
    Embedding,
    Flatten,
    GroupNorm,
    Input,
    LayerNorm,
    Linear,
    MaxPool2D,
    RMSNorm,
    Sequential,
)
from pytensor_ml.layers.norm import _standardize
from pytensor_ml.layers.recurrent import RecurrentCell
from pytensor_ml.optim import adam
from pytensor_ml.pytensorf import (
    collect_non_trainable_updates,
    collect_trainable_params,
    function,
    rewrite_for_prediction,
)
from pytensor_ml.state import NormalInitializer, initialize_params
from tests.conftest import constant

floatX = pytensor.config.floatX

# The numpy references below sum in a different order than the graph, so the gap tracks the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml layers")))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_layer(bias, rng):
    X = pt.tensor("X", shape=(None, 6))
    linear = Linear(name="Linear_1", n_in=6, n_out=3, bias=bias)
    out = linear(X)

    X_in, *weights = out.owner.inputs
    [X_out] = out.owner.outputs

    assert out.owner.op.name == "Linear_1[(?,6) -> (?,3)]"

    expected_names = ["Linear_1_W", "Linear_1_b"] if bias else ["Linear_1_W"]
    assert [w.name for w in weights] == expected_names

    assert X_out.name == "Linear_1_output"

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W_np = rng.normal(size=(6, 3)).astype(floatX)
    b_np = rng.normal(size=(3,)).astype(floatX)

    linear.W.set_value(W_np)
    if bias:
        linear.b.set_value(b_np)

    res = out.eval({X: X_np})
    expected = X_np @ W_np + b_np if bias else X_np @ W_np
    np.testing.assert_allclose(res, expected)


def test_sequential(rng):
    linear1 = Linear(name="Linear_1", n_in=6, n_out=3)
    linear2 = Linear(name="Linear_2", n_in=3, n_out=1)
    mlp = Sequential(linear1, linear2)

    X = pt.tensor("X", shape=(None, 6))
    out = mlp(X)
    assert out.type.shape == (None, 1)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W1_np = rng.normal(size=(6, 3)).astype(floatX)
    b1_np = rng.normal(size=(3,)).astype(floatX)
    W2_np = rng.normal(size=(3, 1)).astype(floatX)
    b2_np = rng.normal(size=(1,)).astype(floatX)

    linear1.W.set_value(W1_np)
    linear1.b.set_value(b1_np)
    linear2.W.set_value(W2_np)
    linear2.b.set_value(b2_np)

    f = pytensor.function([X], out)
    res = f(X_np)

    np.testing.assert_allclose(res, (X_np @ W1_np + b1_np) @ W2_np + b2_np)


def test_dropout(rng):
    X = pt.tensor("X", shape=(None, 6))
    dropout = Dropout(name="Dropout_1", p=1.0)
    out = dropout(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)

    res = out.eval({X: X_np})
    np.testing.assert_allclose(res, np.zeros_like(X_np))


def test_invalid_dropout_p_raises():
    with pytest.raises(
        ValueError, match=r"Dropout probability has to be between 0 and 1, but got -0\.1"
    ):
        Dropout(name=None, p=-0.1)

    with pytest.raises(
        ValueError, match=r"Dropout probability has to be between 0 and 1, but got 1\.1"
    ):
        Dropout(name=None, p=1.1)


def test_input_accepts_a_varying_dimension():
    """One compiled graph has to serve batches of different sizes, which is what an unknown dimension is
    for -- asserting the declared type alone would pass even if the batch axis were pinned."""
    X = Input("X", (None, 64))
    forward = pytensor.function([X], Linear("fc", n_in=64, n_out=8)(X))

    assert X.type.shape == (None, 64)
    assert forward(np.zeros((3, 64), dtype=floatX)).shape == (3, 8)
    assert forward(np.zeros((7, 64), dtype=floatX)).shape == (7, 8)


def test_embedding_forward(rng):
    n_embeddings, n_features = 10, 4
    embedding = Embedding("emb", n_embeddings=n_embeddings, n_features=n_features)
    W_np = rng.normal(size=(n_embeddings, n_features)).astype(floatX)
    embedding.W.set_value(W_np)

    ids = Input("ids", (2, 3), dtype="int64")  # a batch of index rows
    out = embedding(ids)
    assert out.name == "emb_output"

    ids_np = np.array([[1, 2, 3], [4, 5, 6]])
    res = out.eval({ids: ids_np})
    np.testing.assert_allclose(res, W_np[ids_np])
    assert res.shape == (2, 3, n_features)


def test_embedding_table_is_trainable(rng):
    # The OpFromGraph marker must pass the gradient through to the selected rows -- and only
    # those rows -- so the table trains; the integer indices are non-differentiable.
    embedding = Embedding("emb", n_embeddings=6, n_features=3)
    embedding.W.set_value(rng.normal(size=(6, 3)).astype(floatX))
    ids = pt.lvector("ids")
    grad_fn = pytensor.function([ids], pytensor.grad((embedding(ids) ** 2).sum(), embedding.W))

    grad = grad_fn(np.array([1, 1, 4]))
    selected = np.zeros(6, dtype=bool)
    selected[[1, 4]] = True
    assert np.any(grad[selected] != 0)
    assert np.all(grad[~selected] == 0)


@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_batch_norm_forward(n_in, rng):
    X = pt.tensor("X", shape=(None, 6))
    batch_norm = BatchNorm(name="BatchNorm_1", n_in=n_in)
    out = batch_norm(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX) ** 2
    loc_np = rng.normal(size=(6,)).astype(floatX) ** 2
    batch_norm.scale.set_value(scale_np)
    batch_norm.loc.set_value(loc_np)

    res = out.eval({X: X_np})
    mean_np = X_np.mean(axis=0)
    var_np = X_np.var(axis=0)
    expected = (X_np - mean_np) / np.sqrt(var_np + batch_norm.epsilon) * scale_np + loc_np

    np.testing.assert_allclose(res, expected, rtol=1e-5)


# Rank > 2 is the transformer case (batch, seq, d_model): the last axis is normalized and the affine
# parameters broadcast over every leading dimension.
@pytest.mark.parametrize("batch_shape", [(10,), (2, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_layer_norm_forward(n_in, batch_shape, rng):
    X = pt.tensor("X", shape=(*(None,) * len(batch_shape), 6))
    layer_norm = LayerNorm(name="LayerNorm_1", n_in=n_in)
    out = layer_norm(X)
    assert out.name == "LayerNorm_1_output"

    X_np = rng.normal(size=(*batch_shape, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX)
    loc_np = rng.normal(size=(6,)).astype(floatX)
    layer_norm.scale.set_value(scale_np)
    layer_norm.loc.set_value(loc_np)

    res = out.eval({X: X_np})
    mean_np = X_np.mean(axis=-1, keepdims=True)
    var_np = X_np.var(axis=-1, keepdims=True)
    expected = (X_np - mean_np) / np.sqrt(var_np + layer_norm.epsilon) * scale_np + loc_np
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_layer_norm_prediction_matches_training(rng):
    # LayerNorm normalizes over per-sample statistics, identical in train and eval, so unlike
    # BatchNorm it needs no prediction rewrite: rewrite_for_prediction leaves its output unchanged.
    X = pt.tensor("X", shape=(None, 6))
    layer_norm = LayerNorm("ln", n_in=6)
    out = layer_norm(X)
    layer_norm.scale.set_value(rng.normal(size=6).astype(floatX))
    layer_norm.loc.set_value(rng.normal(size=6).astype(floatX))

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    np.testing.assert_allclose(
        rewrite_for_prediction(out).eval({X: X_np}), out.eval({X: X_np}), rtol=1e-6
    )


def test_vectorize_graph_batches_independent_predictions(rng):
    # A model built for a single sample must vectorize over a batch through the OpFromGraph-based
    # layers (Linear, LayerNorm); the batched result must match looping the single-sample graph.
    x = pt.vector("x", shape=(4,))
    net = Sequential(
        Linear("fc1", n_in=4, n_out=8),
        ReLU(),
        LayerNorm("ln", n_in=8),
        Linear("fc2", n_in=8, n_out=3),
    )
    out = net(x)
    for parameter in collect_trainable_params(out):
        parameter.set_value(rng.normal(size=parameter.get_value().shape).astype(floatX))

    X = pt.matrix("X", shape=(None, 4))
    f_single = pytensor.function([x], out)
    f_batch = pytensor.function([X], vectorize_graph(out, {x: X}))

    X_np = rng.normal(size=(5, 4)).astype(floatX)
    np.testing.assert_allclose(f_batch(X_np), np.stack([f_single(row) for row in X_np]), rtol=1e-5)


def test_layer_norm_no_affine_standardizes_each_row(rng):
    X = pt.tensor("X", shape=(None, 8))
    out = LayerNorm(name="LayerNorm_1", n_in=8, affine=False)(X)

    X_np = rng.normal(loc=3.0, scale=2.0, size=(10, 8)).astype(floatX)
    res = out.eval({X: X_np})

    np.testing.assert_allclose(res.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(res.var(axis=-1), 1.0, rtol=1e-3)


def group_norm_reference(X_np, n_groups, epsilon):
    """Loop over (sample, group) blocks rather than reshaping the way the layer does, so the two
    share nothing but the definition and cannot agree about a wrong axis set."""
    channels_per_group = X_np.shape[-1] // n_groups
    expected = np.empty_like(X_np)
    for sample in range(X_np.shape[0]):
        for group in range(n_groups):
            channels = slice(group * channels_per_group, (group + 1) * channels_per_group)
            block = X_np[sample, ..., channels]
            expected[sample, ..., channels] = (block - block.mean()) / np.sqrt(
                block.var() + epsilon
            )
    return expected


@pytest.mark.parametrize("n_groups", [1, 2, 8], ids=["one_group", "two_groups", "per_channel"])
@pytest.mark.parametrize(
    "batch_shape", [(10,), (2, 5, 3)], ids=["no_spatial_axes", "two_spatial_axes"]
)
@pytest.mark.parametrize("n_in", [8, None], ids=["specified", "lazy"])
def test_group_norm_forward(n_in, batch_shape, n_groups, rng):
    X = pt.tensor("X", shape=(*(None,) * len(batch_shape), 8))
    group_norm = GroupNorm(name="GroupNorm_1", n_groups=n_groups, n_in=n_in)
    out = group_norm(X)
    assert out.name == "GroupNorm_1_output"

    X_np = rng.normal(size=(*batch_shape, 8)).astype(floatX)
    scale_np = rng.normal(size=(8,)).astype(floatX)
    loc_np = rng.normal(size=(8,)).astype(floatX)
    group_norm.scale.set_value(scale_np)
    group_norm.loc.set_value(loc_np)

    expected = group_norm_reference(X_np, n_groups, group_norm.epsilon) * scale_np + loc_np

    np.testing.assert_allclose(out.eval({X: X_np}), expected, rtol=1e-5, atol=ATOL)


def test_group_norm_preserves_the_static_shape():
    """The grouping reshape drops the static shape, and a downstream layer infers its n_in from
    what comes back."""
    X = pt.tensor("X", shape=(None, 5, 3, 8))
    out = GroupNorm("gn", n_groups=4)(X)

    assert out.type.shape == (None, 5, 3, 8)


@pytest.mark.parametrize(
    "n_groups, statistic_axes",
    [(1, (1, 2, 3)), (4, (1, 2))],
    ids=["one_group_covers_the_feature_map", "per_channel_is_instance_norm"],
)
def test_group_norm_extremes_standardize_what_they_claim_to(n_groups, statistic_axes, rng):
    X = pt.tensor("X", shape=(None, 5, 3, 4))
    out = GroupNorm("gn", n_groups=n_groups, n_in=4, affine=False)(X)

    res = out.eval({X: rng.normal(loc=3.0, scale=2.0, size=(6, 5, 3, 4)).astype(floatX)})

    np.testing.assert_allclose(res.mean(axis=statistic_axes), 0.0, atol=1e-5)
    np.testing.assert_allclose(res.var(axis=statistic_axes), 1.0, rtol=1e-3)


def test_group_norm_applies_epsilon_inside_the_square_root(rng):
    # At the default epsilon, dividing by sqrt(var + eps) and by sqrt(var) + eps agree to well
    # under the tolerance any other test runs at, so only a large one separates them.
    X = pt.tensor("X", shape=(None, 4, 6))
    group_norm = GroupNorm("gn", n_groups=2, n_in=6, epsilon=0.5, affine=False)
    out = group_norm(X)

    X_np = rng.normal(size=(3, 4, 6)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}),
        group_norm_reference(X_np, 2, group_norm.epsilon),
        rtol=1e-5,
        atol=ATOL,
    )


def test_group_norm_gradient(rng):
    X_np = rng.normal(size=(3, 4, 6)).astype(floatX)
    group_norm = GroupNorm("gn", n_groups=3, n_in=6)
    group_norm.scale.set_value(rng.normal(size=(6,)).astype(floatX))
    group_norm.loc.set_value(rng.normal(size=(6,)).astype(floatX))

    verify_grad(group_norm, [X_np], rng=np.random.default_rng(0))


@pytest.mark.parametrize("n_groups", [0, -1], ids=["zero", "negative"])
def test_group_norm_needs_a_positive_group_count(n_groups):
    with pytest.raises(ValueError, match=f"needs at least one group, but got n_groups={n_groups}"):
        GroupNorm("gn", n_groups=n_groups, n_in=8)


def test_group_norm_uneven_groups_raise():
    with pytest.raises(ValueError, match="3 groups do not divide 8 channels"):
        GroupNorm("gn", n_groups=3, n_in=8)


def test_group_norm_uneven_groups_raise_on_a_lazily_inferred_channel_count():
    with pytest.raises(ValueError, match="3 groups do not divide 8 channels"):
        GroupNorm("gn", n_groups=3)(pt.tensor("X", shape=(None, 8)))


def test_group_norm_without_a_channel_axis_raises():
    with pytest.raises(ValueError, match="got a 1-dimensional input"):
        GroupNorm("gn", n_groups=2, n_in=4)(pt.vector("x", shape=(4,)))


def test_batch_norm_learns_population_stats(rng):
    population_mean, population_std = 3.2, 6.2
    X = pt.tensor("X", shape=(None, 32))
    batch_norm = BatchNorm(name="BatchNorm_1", n_in=32, momentum=0.05, epsilon=1e-8)
    X_normalized = batch_norm(X)

    loss = pt.square(X_normalized - X).mean()
    d_loss = pt.grad(loss, [batch_norm.loc, batch_norm.scale])

    learning_rate = 1e-1
    updates = {
        batch_norm.loc: batch_norm.loc - learning_rate * d_loss[0],
        batch_norm.scale: batch_norm.scale - learning_rate * d_loss[1],
        batch_norm.running_mean: batch_norm.new_running_mean,
        batch_norm.running_var: batch_norm.new_running_var,
    }
    train = pytensor.function([X], X_normalized, updates=updates)

    def sample_batch():
        return rng.normal(loc=population_mean, scale=population_std, size=(100, 32)).astype(floatX)

    for _ in range(500):
        data = sample_batch()
        # Read before stepping: the affine parameters this batch is normalized with are the ones the
        # updates are about to overwrite.
        scale, loc = batch_norm.scale.get_value(), batch_norm.loc.get_value()

        np.testing.assert_allclose(
            train(data),
            (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0) + batch_norm.epsilon) * scale
            + loc,
            rtol=1e-4,
            atol=ATOL,
        )

    scale, loc = batch_norm.scale.get_value(), batch_norm.loc.get_value()
    running_mean = batch_norm.running_mean.get_value()
    running_var = batch_norm.running_var.get_value()

    np.testing.assert_allclose(loc, population_mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(scale, population_std, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(running_mean, population_mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(np.sqrt(running_var), population_std, rtol=1e-1, atol=1e-1)

    predict = pytensor.function([X], rewrite_for_prediction(X_normalized))
    data = sample_batch()

    np.testing.assert_allclose(
        predict(data),
        (data - running_mean) / np.sqrt(running_var + batch_norm.epsilon) * scale + loc,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "op_name, submodule",
    [
        ("LinearLayer", "linear"),
        ("EmbeddingLayer", "embedding"),
        ("DropoutLayer", "dropout"),
        ("BatchNormLayer", "norm"),
        ("NoRunningStatsBatchNormLayer", "norm"),
        ("PredictionBatchNormLayer", "norm"),
        ("LayerNormLayer", "norm"),
        ("GroupNormLayer", "norm"),
    ],
)
def test_marker_ops_stay_reachable_from_the_package(op_name, submodule):
    # deserialize_graph resolves an op's recorded import path with getattr on this package, so these
    # bindings are load-bearing rather than convenience re-exports.
    from_package = getattr(pytensor_ml.layers, op_name)
    from_submodule = getattr(importlib.import_module(f"pytensor_ml.layers.{submodule}"), op_name)

    assert from_package is from_submodule


def test_batch_norm_without_running_stats_normalizes_with_batch_statistics(rng):
    X = pt.tensor("X", shape=(None, 4))
    normalize = pytensor.function([X], BatchNorm("bn", n_in=4, track_running_stats=False)(X))

    out = normalize(rng.normal(loc=5.0, scale=3.0, size=(256, 4)).astype(floatX))

    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, rtol=1e-3)


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_batch_norm_running_stats_write_back_to_the_right_inputs(affine):
    # The update map indexes inputs positionally, and the affine parameters shift the running
    # statistics along by two when present.
    X = pt.tensor("X", shape=(None, 4))
    batch_norm = BatchNorm("bn", n_in=4, affine=affine)

    updates = collect_non_trainable_updates(batch_norm(X))

    assert updates == {
        batch_norm.running_mean: batch_norm.new_running_mean,
        batch_norm.running_var: batch_norm.new_running_var,
    }


def test_one_batch_norm_applied_twice_chains_its_statistics():
    # Weight sharing across two towers is the ordinary reason to apply one layer object twice, and both
    # applications have to reach the statistics. The chain is built while the graph is, which is where the
    # order of the two calls is known.
    a, b = pt.matrix("a"), pt.matrix("b")
    batch_norm = BatchNorm("bn", n_in=4)
    first, second = batch_norm(a), batch_norm(b)

    # Read through the op's own update map, so this says "the second application reads what the first
    # wrote" rather than pinning the positions the affine parameters happen to shift.
    for output_index, input_index in second.owner.op.update_map().items():
        assert second.owner.inputs[input_index] is first.owner.outputs[output_index]

    updates = collect_non_trainable_updates([first, second])

    assert updates == {
        batch_norm.running_mean: second.owner.outputs[1],
        batch_norm.running_var: second.owner.outputs[2],
    }


def test_one_batch_norm_applied_twice_counts_both_batches(rng):
    a, b = pt.matrix("a"), pt.matrix("b")
    batch_norm = BatchNorm("bn", n_in=4, momentum=0.1)
    first, second = batch_norm(a), batch_norm(b)
    loss = (first**2).sum() + (second**2).sum()
    step = function([a, b], loss, updates=adam(1e-3)(loss, collect_trainable_params(loss)))

    first_batch = rng.normal(loc=100.0, size=(8, 4)).astype(floatX)
    second_batch = rng.normal(loc=-100.0, size=(8, 4)).astype(floatX)
    step(first_batch, second_batch)

    # Each application reads what the one before it wrote, so the momentum applies twice -- the same value
    # torch accumulates from two forward calls on one module.
    momentum = batch_norm.momentum
    expected = (1 - momentum) * momentum * first_batch.mean(axis=0) + momentum * second_batch.mean(
        axis=0
    )

    np.testing.assert_allclose(batch_norm.running_mean.get_value(), expected, rtol=1e-5)


def test_batch_norm_variants_agree_on_output_arity():
    X = pt.tensor("X", shape=(None, 4))
    tracked = BatchNorm("tracked", n_in=4)(X)
    untracked = BatchNorm("untracked", n_in=4, track_running_stats=False)(X)

    # Matching arity is what lets BatchNorm use one code path; the untracked variant reports the
    # batch statistics but must not write them anywhere.
    assert len(tracked.owner.outputs) == len(untracked.owner.outputs)
    assert collect_non_trainable_updates(untracked) == {}


# Every parameter a layer owns, and the value a redraw must give it. A constant is part of the layer's
# definition -- a batch-norm scale drawn as a random factor would rescale a normalized activation and defeat
# the layer, and a drawn bias would undo the zero start these layers are documented to have. None means the
# declaration is a real draw rather than a constant, so the assertion is that the value varies.
FEATURES = pt.tensor("features", shape=(None, 4), dtype=floatX)
IDS = pt.tensor("ids", shape=(None, 4), dtype="int32")

DECLARED_BY_LAYERS = {
    "Linear": (lambda: Linear("fc", n_in=4, n_out=4), FEATURES, {"fc_W": None, "fc_b": 0.0}),
    "Embedding": (
        lambda: Embedding("emb", n_embeddings=6, n_features=4),
        IDS,
        {"emb_W": None},
    ),
    "BatchNorm": (lambda: BatchNorm("bn", n_in=4), FEATURES, {"bn_scale": 1.0, "bn_loc": 0.0}),
    "LayerNorm": (lambda: LayerNorm("ln", n_in=4), FEATURES, {"ln_scale": 1.0, "ln_loc": 0.0}),
    "GroupNorm": (
        lambda: GroupNorm("gn", n_groups=2, n_in=4),
        FEATURES,
        {"gn_scale": 1.0, "gn_loc": 0.0},
    ),
}


@pytest.mark.parametrize("layer_name", sorted(DECLARED_BY_LAYERS), ids=sorted(DECLARED_BY_LAYERS))
def test_a_layer_declares_an_initializer_for_every_parameter_it_builds(layer_name):
    """A redraw consults nothing but the declaration, so a parameter declaring none cannot be redrawn at all.
    Every parameter here has to come back with its layer's definition intact."""
    build, layer_input, expected = DECLARED_BY_LAYERS[layer_name]
    prediction = build()(layer_input)
    params = collect_trainable_params(prediction)

    values = initialize_params(params, rng=0)

    assert {p.name for p in params} == set(expected)
    for param, value in zip(params, values):
        if expected[param.name] is None:
            assert len(np.unique(value)) > 1, f"{param.name} redrew to one repeated value"
        else:
            np.testing.assert_allclose(value, expected[param.name], err_msg=param.name)


# One case per keyword: the layer to build with it, the parameter it must reach, and the parameters it must
# leave alone. A keyword that hits the wrong parameter, or that quietly strips a sibling's declaration, is
# the failure this is aimed at.
INITIALIZER_KEYWORDS = {
    "Linear.weight": (
        lambda init: Linear("fc", n_in=4, n_out=4, weight_initializer=init),
        FEATURES,
        "fc_W",
        {"fc_b": 0.0},
    ),
    "Linear.bias": (
        lambda init: Linear("fc", n_in=4, n_out=4, bias_initializer=init),
        FEATURES,
        "fc_b",
        {"fc_W": None},  # a bias initializer reaching the weight would show up as the sentinel here
    ),
    "Embedding.weight": (
        lambda init: Embedding("emb", n_embeddings=6, n_features=4, weight_initializer=init),
        IDS,
        "emb_W",
        {},
    ),
    "BatchNorm.scale": (
        lambda init: BatchNorm("bn", n_in=4, scale_initializer=init),
        FEATURES,
        "bn_scale",
        {"bn_loc": 0.0},
    ),
    "BatchNorm.loc": (
        lambda init: BatchNorm("bn", n_in=4, loc_initializer=init),
        FEATURES,
        "bn_loc",
        {"bn_scale": 1.0},
    ),
    "LayerNorm.scale": (
        lambda init: LayerNorm("ln", n_in=4, scale_initializer=init),
        FEATURES,
        "ln_scale",
        {"ln_loc": 0.0},
    ),
    "LayerNorm.loc": (
        lambda init: LayerNorm("ln", n_in=4, loc_initializer=init),
        FEATURES,
        "ln_loc",
        {"ln_scale": 1.0},
    ),
    "GroupNorm.scale": (
        lambda init: GroupNorm("gn", n_groups=2, n_in=4, scale_initializer=init),
        FEATURES,
        "gn_scale",
        {"gn_loc": 0.0},
    ),
    "GroupNorm.loc": (
        lambda init: GroupNorm("gn", n_groups=2, n_in=4, loc_initializer=init),
        FEATURES,
        "gn_loc",
        {"gn_scale": 1.0},
    ),
}


@pytest.mark.parametrize("case", sorted(INITIALIZER_KEYWORDS), ids=sorted(INITIALIZER_KEYWORDS))
def test_an_initializer_keyword_reaches_only_the_parameter_it_names(case):
    build, layer_input, target, siblings = INITIALIZER_KEYWORDS[case]
    sentinel = 7.0
    layer = build(constant(value=sentinel))
    prediction = layer(layer_input)

    params = collect_trainable_params(prediction)
    values = dict(zip((p.name for p in params), initialize_params(params, rng=0)))

    np.testing.assert_allclose(values[target], sentinel)
    for name, expected in siblings.items():
        if expected is None:
            assert len(np.unique(values[name])) > 1, name
        else:
            np.testing.assert_allclose(values[name], expected, err_msg=name)


def test_a_bias_initializer_replaces_the_zero_declaration_rather_than_fighting_it():
    """The keyword becomes the declaration, so a redraw uses it rather than the zero it replaced. torch draws
    biases from a fan-scaled uniform, and this is how you say that here."""
    layer = Linear("fc", n_in=4, n_out=4, bias_initializer=NormalInitializer(0.0, 1.0))
    prediction = layer(pt.tensor("features", shape=(None, 4), dtype=floatX))

    params = collect_trainable_params(prediction)
    values = dict(zip((p.name for p in params), initialize_params(params, rng=0)))

    assert not np.allclose(values["fc_b"], 0.0)


@pytest.mark.parametrize("layer_name", sorted(DECLARED_BY_LAYERS), ids=sorted(DECLARED_BY_LAYERS))
def test_a_layer_is_born_holding_a_draw_from_its_own_initializer(layer_name):
    """Creating a parameter and giving it a value are one event, as they are in torch, flax and keras. A
    weight left at zero passes no gradient below the last layer of a stack, so a network nobody remembered to
    initialize trains its output bias and nothing else."""
    build, layer_input, expected = DECLARED_BY_LAYERS[layer_name]
    prediction = build()(layer_input)

    for param in collect_trainable_params(prediction):
        declared = expected[param.name]
        value = param.get_value()
        if declared is None:
            assert len(np.unique(value)) > 1, (
                f"{param.name} was born holding one repeated value rather than a draw"
            )
        else:
            np.testing.assert_allclose(value, declared, err_msg=param.name)


def test_a_keyword_initializer_reaches_the_value_and_not_only_the_declaration():
    """The bias case: a keyword used to be recorded as a declaration while the value stayed zero, so it took
    effect only if someone happened to call initialize afterwards."""
    sentinel = 7.0
    layer = Linear(
        "fc",
        n_in=4,
        n_out=4,
        bias_initializer=constant(value=sentinel),
    )

    np.testing.assert_allclose(layer.b.get_value(), sentinel)


def test_construction_draws_do_not_leak_into_a_seeded_initialize():
    """Construction draws from fresh entropy, so two identically seeded runs must still agree: the seeded
    initialize has to overwrite every value rather than build on what construction happened to produce."""

    def seeded_values():
        X = pt.tensor("features", shape=(None, 4), dtype=floatX)
        prediction = Sequential(Linear("fc1", n_in=4, n_out=4), Linear("fc2", n_in=4, n_out=4))(X)
        params = collect_trainable_params(prediction)
        return dict(zip((p.name for p in params), initialize_params(params, rng=0)))

    first, second = seeded_values(), seeded_values()

    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((32, 4, 4, 3), (32, 48)),
        ((None, 6, 6, 16), (None, 576)),
        ((None, 5), (None, 5)),
        ((32, None, 4, 3), (32, None)),
        ((7,), (7, 1)),
    ],
    ids=["static", "dynamic_batch", "already_flat", "dynamic_feature", "rank_one"],
)
def test_flatten_keeps_the_feature_count_it_can_work_out(shape, expected):
    """A dense head is constructed from its input's feature count and has nothing but the static shape
    to read it from, while the batch axis is unknown in any graph built for variable batches. So the
    count has to survive an unknown batch, and only the axes that are themselves unknown may be lost."""
    assert Flatten(pt.tensor("X", shape=shape)).type.shape == expected


def test_flatten_collapses_every_axis_after_the_batch(rng):
    """Ravelling per row is the whole contract, and an implementation that flattened the wrong axes
    would still return a two-dimensional result of plausible size."""
    X_np = rng.normal(size=(3, 2, 5, 4)).astype(floatX)
    X = pt.tensor("X", shape=(None, 2, 5, 4))

    np.testing.assert_allclose(Flatten(X).eval({X: X_np}), X_np.reshape(3, -1))


def test_flatten_reaches_a_dense_head_from_a_convolution(rng):
    """The reason `Flatten` exists: `Linear` is constructed from the feature count, so a stack that
    loses it cannot be built at all."""
    X = pt.tensor("X", shape=(None, 8, 8, 3))
    convolved = Conv2D("conv", in_channels=3, out_channels=4, kernel_size=3)(X)
    pooled = MaxPool2D("pool", kernel_size=2)(convolved)
    flattened = Flatten(pooled)

    features = flattened.type.shape[-1]
    assert features == 3 * 3 * 4

    head = Linear("head", n_in=features, n_out=2)
    X_np = rng.normal(size=(5, 8, 8, 3)).astype(floatX)
    assert head(flattened).eval({X: X_np}).shape == (5, 2)


def test_batch_norm_on_a_flat_input_reduces_over_the_batch_alone(rng):
    """On `(batch, features)` there is nothing to reduce but the batch, so this pins the whole contract
    at that rank: the normalized values, the per-feature running statistics and the prediction path that
    reads them. Every other rank adds axes to the same reduction rather than changing it."""
    X = pt.tensor("X", shape=(None, 4))
    batch_norm = BatchNorm("bn", n_in=4, momentum=0.25, epsilon=1e-5)
    out = batch_norm(X)
    batch_norm.scale.set_value(np.array([1.5, 0.5, 2.0, 1.0], dtype=floatX))
    batch_norm.loc.set_value(np.array([0.25, -0.5, 0.0, 1.0], dtype=floatX))

    X_np = rng.normal(size=(6, 4)).astype(floatX)
    train = pytensor.function(
        [X],
        out,
        updates={
            batch_norm.running_mean: batch_norm.new_running_mean,
            batch_norm.running_var: batch_norm.new_running_var,
        },
    )

    scale = batch_norm.scale.get_value()
    loc = batch_norm.loc.get_value()
    expected = (X_np - X_np.mean(axis=0)) / np.sqrt(
        X_np.var(axis=0) + batch_norm.epsilon
    ) * scale + loc
    np.testing.assert_allclose(train(X_np), expected, rtol=1e-6, atol=ATOL)

    # The running statistics carry one entry per feature and move by `momentum` toward the batch.
    np.testing.assert_allclose(
        batch_norm.running_mean.get_value(), 0.25 * X_np.mean(axis=0), rtol=1e-6, atol=ATOL
    )
    np.testing.assert_allclose(
        batch_norm.running_var.get_value(),
        0.25 * X_np.var(axis=0) + 0.75,
        rtol=1e-6,
        atol=ATOL,
    )

    predict = pytensor.function([X], rewrite_for_prediction(out))
    running_mean = batch_norm.running_mean.get_value()
    running_var = batch_norm.running_var.get_value()
    np.testing.assert_allclose(
        predict(X_np),
        (X_np - running_mean) / np.sqrt(running_var + batch_norm.epsilon) * scale + loc,
        rtol=1e-6,
        atol=ATOL,
    )


def test_batch_norm_takes_per_channel_statistics_from_an_image(rng):
    """On a `(batch, height, width, channels)` activation a channel is one feature, however many
    positions it appears at. Reducing over the batch alone would give every pixel position its own mean
    and variance, which normalizes each position against the batch instead of normalizing the channel."""
    X = pt.tensor("X", shape=(None, 5, 3, 2))
    batch_norm = BatchNorm("bn", n_in=2, affine=False)
    out = batch_norm(X)

    X_np = rng.normal(size=(4, 5, 3, 2)).astype(floatX)
    spatial = (0, 1, 2)
    expected = (X_np - X_np.mean(axis=spatial)) / np.sqrt(
        X_np.var(axis=spatial) + batch_norm.epsilon
    )

    np.testing.assert_allclose(out.eval({X: X_np}), expected, rtol=1e-5, atol=ATOL)


def test_batch_norm_keeps_one_running_statistic_per_channel(rng):
    """The running statistics are allocated from `n_in` and written from the batch statistics, so a
    reduction over the wrong axes makes the update a shape error rather than a wrong number -- unless
    the image happens to be square, where it silently writes per-pixel values into a per-channel slot."""
    X = pt.tensor("X", shape=(None, 3, 3, 3))
    batch_norm = BatchNorm("bn", n_in=3, momentum=1.0, affine=False)
    out = batch_norm(X)

    X_np = rng.normal(size=(2, 3, 3, 3)).astype(floatX)
    train = pytensor.function(
        [X],
        out,
        updates={
            batch_norm.running_mean: batch_norm.new_running_mean,
            batch_norm.running_var: batch_norm.new_running_var,
        },
    )
    train(X_np)

    assert batch_norm.running_mean.get_value().shape == (3,)
    np.testing.assert_allclose(
        batch_norm.running_mean.get_value(), X_np.mean(axis=(0, 1, 2)), rtol=1e-5, atol=ATOL
    )


def test_batch_norm_predicts_an_image_from_its_running_statistics(rng):
    """Prediction broadcasts the per-channel statistics over every spatial position, which is only
    correct once those statistics are per-channel in the first place."""
    X = pt.tensor("X", shape=(None, 4, 2, 3))
    batch_norm = BatchNorm("bn", n_in=3)
    out = batch_norm(X)
    batch_norm.scale.set_value(rng.normal(size=3).astype(floatX))
    batch_norm.loc.set_value(rng.normal(size=3).astype(floatX))
    batch_norm.running_mean.set_value(rng.normal(size=3).astype(floatX))
    batch_norm.running_var.set_value((rng.normal(size=3) ** 2 + 0.5).astype(floatX))

    X_np = rng.normal(size=(6, 4, 2, 3)).astype(floatX)
    predict = pytensor.function([X], rewrite_for_prediction(out))
    running_mean = batch_norm.running_mean.get_value()
    running_var = batch_norm.running_var.get_value()

    np.testing.assert_allclose(
        predict(X_np),
        (X_np - running_mean)
        / np.sqrt(running_var + batch_norm.epsilon)
        * batch_norm.scale.get_value()
        + batch_norm.loc.get_value(),
        rtol=1e-5,
        atol=ATOL,
    )


def test_batch_norm_rejects_an_input_with_no_batch_axis():
    """Reducing over every axis but the last leaves nothing to reduce on a rank-1 input, which
    standardizes to zeros -- a plausible-looking result from a layer that cannot do its job without a
    batch to take statistics over."""
    with pytest.raises(ValueError, match="needs at least a batch axis and a feature axis"):
        BatchNorm("bn", n_in=4)(pt.tensor("X", shape=(4,)))


def test_batch_norm_reduces_over_time_as_well_as_the_batch(rng):
    """A `(batch, time, channels)` activation is what `Conv1D` produces, and a channel is one feature
    however many time steps it spans. Reducing over the batch alone would give each time step its own
    statistics, which is the same bug at one axis fewer."""
    X = pt.tensor("X", shape=(None, 7, 3))
    batch_norm = BatchNorm("bn", n_in=3, affine=False)
    out = batch_norm(X)

    X_np = rng.normal(size=(4, 7, 3)).astype(floatX)
    expected = (X_np - X_np.mean(axis=(0, 1))) / np.sqrt(X_np.var(axis=(0, 1)) + batch_norm.epsilon)

    np.testing.assert_allclose(out.eval({X: X_np}), expected, rtol=1e-5, atol=ATOL)


# The layers every hyperparameter of which has a default, so a stray value reaches the name slot
# rather than failing as a missing argument. Each is paired with the hyperparameter a torch or keras
# user would have put there positionally, a value for it, and what the layer stores for that value --
# a scalar fans out to one entry per spatial axis.
OPTIONAL_NAME_LAYERS = [
    ("Dropout", "p", 0.1, 0.1),
    ("BatchNorm", "n_in", 32, 32),
    ("LayerNorm", "n_in", 32, 32),
    ("MaxPool1D", "kernel_size", 3, (3,)),
    ("MaxPool2D", "kernel_size", 3, (3, 3)),
    ("AvgPool1D", "kernel_size", 3, (3,)),
    ("AvgPool2D", "kernel_size", 3, (3, 3)),
    ("ZeroPad1D", "padding", 1, ((1, 1),)),
    ("ZeroPad2D", "padding", 1, ((1, 1), (1, 1))),
    ("ConstantPad1D", "padding", 1, ((1, 1),)),
    ("ConstantPad2D", "padding", 1, ((1, 1), (1, 1))),
    ("ReflectionPad1D", "padding", 1, ((1, 1),)),
    ("ReflectionPad2D", "padding", 1, ((1, 1), (1, 1))),
    ("ReplicationPad1D", "padding", 1, ((1, 1),)),
    ("ReplicationPad2D", "padding", 1, ((1, 1), (1, 1))),
]
OPTIONAL_NAME_LAYER_IDS = [layer_name for layer_name, *_ in OPTIONAL_NAME_LAYERS]


@pytest.mark.parametrize(
    "layer_name, parameter, value, _stored", OPTIONAL_NAME_LAYERS, ids=OPTIONAL_NAME_LAYER_IDS
)
def test_hyperparameter_in_the_name_slot_raises(layer_name, parameter, value, _stored):
    """The torch and keras spelling puts the hyperparameter where ``name`` goes, so only a type
    check on ``name`` can catch it."""
    layer = getattr(pytensor_ml.layers, layer_name)
    with pytest.raises(TypeError, match=rf"{layer_name}\({parameter}="):
        layer(value)


@pytest.mark.parametrize(
    "layer_name, parameter, value, _stored", OPTIONAL_NAME_LAYERS, ids=OPTIONAL_NAME_LAYER_IDS
)
def test_hyperparameter_is_keyword_only(layer_name, parameter, value, _stored):
    """A name in the name slot is not enough on its own: keyword-only arguments are what stop a
    hyperparameter landing one place off and binding anyway."""
    layer = getattr(pytensor_ml.layers, layer_name)
    with pytest.raises(TypeError, match="positional argument"):
        layer("layer", value)


@pytest.mark.parametrize(
    "layer_name, parameter, value, stored", OPTIONAL_NAME_LAYERS, ids=OPTIONAL_NAME_LAYER_IDS
)
def test_name_and_hyperparameter_by_keyword(layer_name, parameter, value, stored):
    layer = getattr(pytensor_ml.layers, layer_name)
    built = layer("layer", **{parameter: value})
    assert built.name == "layer"
    assert getattr(built, parameter) == stored

    defaulted = layer(**{parameter: value})
    assert defaulted.name == layer_name
    assert getattr(defaulted, parameter) == stored


@pytest.mark.parametrize("layer_name", OPTIONAL_NAME_LAYER_IDS)
def test_name_defaults_to_the_class_name(layer_name):
    layer = getattr(pytensor_ml.layers, layer_name)
    assert layer().name == layer_name
    assert layer(None).name == layer_name


def test_no_layer_takes_a_hyperparameter_positionally():
    """The only positional argument a layer takes is its name, or the layers it wraps. Anything else
    reopens the slot torch and keras calls land in, so the rule is checked over the whole exported
    surface rather than a fixed list."""
    wrapped = {"Recurrent": ("cell",), "Bidirectional": ("forward", "backward")}
    offenders = {}
    for layer_name in pytensor_ml.layers.__all__:
        layer = getattr(pytensor_ml.layers, layer_name)
        if not inspect.isclass(layer) or not issubclass(layer, (Layer, RecurrentCell)):
            continue
        if layer.__init__ is object.__init__:  # a container taking its layers as *args
            continue
        parameters = list(inspect.signature(layer.__init__).parameters.values())[1:]
        positional = tuple(
            p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        )
        expected = wrapped.get(layer_name, ("name",))
        if positional != expected:
            offenders[layer_name] = positional
    assert not offenders, (
        f"layers taking more positionally than a name or wrapped layers: {offenders}"
    )


def test_a_layer_with_required_hyperparameters_can_go_unnamed():
    """A name is never required, so a stack can leave it out and take the class name."""
    assert Linear(n_in=4, n_out=4).name == "Linear"


def test_a_layer_with_required_hyperparameters_rejects_a_number_in_the_name_slot():
    """``Linear(4)`` fails as a missing argument rather than a bad name, so the check is only
    reachable once the hyperparameters it needs are supplied."""
    with pytest.raises(TypeError, match=r"Linear\(n_in="):
        Linear(4, n_in=4, n_out=4)


def test_a_layer_with_no_displaceable_hyperparameter_suggests_nothing():
    """``Recurrent`` takes its cell first and has no hyperparameter a name could displace, so the
    error reports the bad name without inventing a parameter to blame for it."""
    cell = pytensor_ml.layers.ElmanCell("cell", n_in=2, n_hidden=2)
    with pytest.raises(TypeError, match=r"`name` must be a string, but got int 5\.$"):
        pytensor_ml.layers.Recurrent(cell, name=5)


@pytest.mark.parametrize("layer_name", ["RNN", "LSTM", "GRU"])
def test_a_recurrent_layer_defaults_to_its_own_name_not_its_base(layer_name):
    """All three subclass ``Recurrent``, which names an unnamed layer after the class the caller
    built rather than after itself."""
    layer = getattr(pytensor_ml.layers, layer_name)
    assert layer(None, n_in=2, n_hidden=2).name == layer_name


def test_bidirectional_rejects_a_non_string_name():
    forward = pytensor_ml.layers.RNN("forward", n_in=2, n_hidden=2)
    backward = pytensor_ml.layers.RNN("backward", n_in=2, n_hidden=2)
    with pytest.raises(TypeError, match=r"Bidirectional's `name` must be a string"):
        pytensor_ml.layers.Bidirectional(forward, backward, name=5)


def test_standardizing_accumulates_wider_than_float16():
    """Squaring a float16 activation overflows past |x| of about 256, and a diffusion decoder's
    activations reach the thousands -- the variance goes to infinity and every element standardizes
    to NaN, which is a black image rather than a wrong one. Run on the python linker, since the
    default backend cannot execute float16 at all."""
    values = (np.random.default_rng(0).normal(size=(4, 8)) * 3000).astype("float16")
    assert np.abs(values.astype("float64")).max() ** 2 > np.finfo(np.float16).max

    X = pt.tensor("X", shape=values.shape, dtype="float16")
    standardized, mu, sigma_sq = _standardize(X, 1e-6, axis=0)
    computed = pytensor.function([X], standardized, mode=Mode(linker="py", optimizer=None))(values)

    assert not np.isnan(computed).any()
    assert (computed.dtype, mu.dtype, sigma_sq.dtype) == ("float16",) * 3
    np.testing.assert_allclose(computed.astype("float64").std(), 1.0, rtol=1e-3)


def rms_norm_reference(X_np, scale_np, epsilon):
    """Walk the rows one at a time rather than broadcasting the way the layer does, so the two share
    nothing but the definition and cannot agree about a wrong axis."""
    flat = X_np.reshape(-1, X_np.shape[-1])
    expected = np.empty_like(flat)
    for row in range(flat.shape[0]):
        features = flat[row].astype("float64")
        root_mean_square = np.sqrt(sum(value**2 for value in features) / len(features) + epsilon)
        expected[row] = features / root_mean_square * scale_np

    return expected.reshape(X_np.shape)


@pytest.mark.parametrize("batch_shape", [(10,), (2, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_rms_norm_forward(n_in, batch_shape, rng):
    X = pt.tensor("X", shape=(*(None,) * len(batch_shape), 6))
    rms_norm = RMSNorm(name="RMSNorm_1", n_in=n_in)
    out = rms_norm(X)
    assert out.name == "RMSNorm_1_output"

    X_np = rng.normal(size=(*batch_shape, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX)
    rms_norm.scale.set_value(scale_np)

    res = out.eval({X: X_np})

    np.testing.assert_allclose(res, rms_norm_reference(X_np, scale_np, rms_norm.epsilon), rtol=1e-5)


def test_rms_norm_leaves_the_mean_where_it_found_it(rng):
    """The one thing separating it from LayerNorm is that it rescales without centering. An
    implementation that subtracted the mean would pass every magnitude check and quietly discard
    whatever offset the previous layer encoded."""
    X = pt.tensor("X", shape=(None, 8))
    out = RMSNorm(name="RMSNorm_1", n_in=8, affine=False)(X)

    X_np = rng.normal(loc=3.0, scale=0.1, size=(10, 8)).astype(floatX)
    res = out.eval({X: X_np})

    # The rows are a tight cloud around 3, so dividing by their root mean square lands them near 1.
    # Centering first would land them at 0 instead.
    np.testing.assert_allclose(res.mean(axis=-1), 1.0, rtol=0.02)
    np.testing.assert_allclose((res**2).mean(axis=-1), 1.0, rtol=1e-3)


def test_rms_norm_applies_epsilon_inside_the_square_root(rng):
    # At the default epsilon, dividing by sqrt(mean_square + eps) and by sqrt(mean_square) + eps
    # agree to well under the tolerance any other test runs at, so only a large one separates them.
    X = pt.tensor("X", shape=(None, 6))
    rms_norm = RMSNorm("rms", n_in=6, epsilon=0.5, affine=False)
    out = rms_norm(X)

    X_np = rng.normal(size=(3, 6)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}),
        rms_norm_reference(X_np, np.ones(6, dtype=floatX), rms_norm.epsilon),
        rtol=1e-5,
    )


def test_rms_norm_accumulates_wider_than_float16():
    """Squaring is the whole operation here, so a float16 activation of a few thousand overflows
    before it is ever averaged. Run on the python linker, since the default backend cannot execute
    float16 at all."""
    values = (np.random.default_rng(0).normal(size=(4, 8)) * 3000).astype("float16")
    assert np.abs(values.astype("float64")).max() ** 2 > np.finfo(np.float16).max

    X = pt.tensor("X", shape=values.shape, dtype="float16")
    out = RMSNorm(name="RMSNorm_1", n_in=8, affine=False)(X)
    computed = pytensor.function([X], out, mode=Mode(linker="py", optimizer=None))(values)

    assert not np.isnan(computed).any()
    assert computed.dtype == np.dtype("float16")
    np.testing.assert_allclose((computed.astype("float64") ** 2).mean(axis=-1), 1.0, rtol=1e-2)
