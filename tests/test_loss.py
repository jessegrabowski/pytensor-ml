import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from scipy.special import softmax
from sklearn.metrics import log_loss

from pytensor_ml.layers import Input, Linear
from pytensor_ml.loss import CrossEntropy, Reductions, SquaredError, supervised_loss
from pytensor_ml.optim import adam, compile_train
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import initialize_params

floatX = pytensor.config.floatX


def generate_categorical_data(expect_logits: bool, seed: int = 0):
    # Seeded: sklearn is an exact oracle here, so extra random draws buy little, and an unreproducible
    # numerical failure costs a lot. Logits are drawn from a normal so the log-softmax path sees negative
    # values, which a uniform draw never produced.
    rng = np.random.default_rng(seed)
    n_classes = rng.integers(2, 10)
    y_true = rng.integers(0, n_classes, size=(100,))
    y_true_onehot = np.eye(n_classes)[y_true]
    y_pred = (
        rng.normal(size=(100, n_classes))
        if expect_logits
        else rng.dirichlet(np.ones(n_classes), size=(100,))
    )

    return y_true, y_true_onehot, y_pred


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("expect_logits", [True, False])
@pytest.mark.parametrize("expect_onehot_labels", [True, False])
def test_cross_entropy(reduction: Reductions, expect_logits, expect_onehot_labels):
    loss = CrossEntropy(
        reduction=reduction, expect_logits=expect_logits, expect_onehot_labels=expect_onehot_labels
    )

    y_true, y_true_onehot, y_pred = generate_categorical_data(expect_logits)

    if expect_onehot_labels:
        loss_value = loss(y_true_onehot, y_pred).eval()
    else:
        loss_value = loss(y_true, y_pred).eval()

    if expect_logits:
        y_pred = softmax(y_pred, axis=-1)

    sklearn_loss = log_loss(y_true, y_pred, normalize=reduction == "mean")
    np.testing.assert_allclose(loss_value, sklearn_loss)


@pytest.mark.parametrize(
    "reduction, expected", [("mean", 4.25 / 3), ("sum", 4.25), (lambda x: x, [0.25, 0.0, 4.0])]
)
def test_squared_error(reduction, expected):
    # Plain sequences, not tensors: SquaredError coerces its inputs like CrossEntropy does, so even a
    # non-reducing reduction returns a TensorVariable rather than a bare ndarray.
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 2.0, 1.0]

    loss = SquaredError(reduction=reduction)(y_true, y_pred)

    assert isinstance(loss, pt.TensorVariable)
    np.testing.assert_allclose(loss.eval(), expected)


def test_cross_entropy_accepts_a_callable_reduction():
    y_true, _, y_pred = generate_categorical_data(expect_logits=True)

    per_sample = CrossEntropy(expect_logits=True, reduction=lambda x: x)(y_true, y_pred).eval()
    pooled = CrossEntropy(expect_logits=True, reduction="mean")(y_true, y_pred).eval()

    assert per_sample.shape == y_true.shape
    np.testing.assert_allclose(per_sample.mean(), pooled)


@pytest.mark.parametrize(
    "loss_fn, expected_shape, expected_dtype",
    [
        (SquaredError(), (None, 3), floatX),
        (CrossEntropy(), (None,), "int64"),
        (CrossEntropy(expect_onehot_labels=True), (None, 3), floatX),
    ],
    ids=["squared_error", "cross_entropy_labels", "cross_entropy_onehot"],
)
def test_the_loss_decides_the_target_it_reads(loss_fn, expected_shape, expected_dtype):
    """The rank and dtype of the target follow from the prediction and the loss together, so neither is
    a question the caller is asked. Integer labels index the class axis rather than subtracting from it,
    which is a different rank and a different dtype from the prediction."""
    prediction = pt.tensor("prediction", shape=(None, 3), dtype=floatX)

    _, target = supervised_loss(prediction, loss_fn)

    assert target.type.shape == expected_shape
    assert target.type.dtype == expected_dtype


def test_a_regression_head_is_scored_row_against_row():
    """A (batch, 1) prediction against a (batch,) target broadcasts to (batch, batch), scoring every
    pair of rows. The target has to take the prediction's rank for the loss to mean what it says."""
    prediction = pt.tensor("prediction", shape=(None, 1), dtype=floatX)
    loss, target = supervised_loss(prediction, SquaredError())
    score = pytensor.function([prediction, target], loss)

    predicted = np.array([[1.0], [2.0], [3.0]], dtype=floatX)
    observed = np.array([[1.5], [2.5], [3.5]], dtype=floatX)

    np.testing.assert_allclose(score(predicted, observed), 0.25)


def test_cross_entropy_trains_from_integer_labels():
    """Integer labels are the ordinary classification target, and reach the loss only if the target
    placeholder is built with an integer dtype."""
    X = Input("X", shape=(None, 4))
    logits = Linear("logits", n_in=4, n_out=3)(X)
    loss, target = supervised_loss(logits, CrossEntropy(expect_logits=True))
    parameters = collect_trainable_params(loss)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)
    step = compile_train(loss, adam(1e-1), inputs=[X, target])

    rng = np.random.default_rng(0)
    features = rng.normal(size=(32, 4)).astype(floatX)
    labels = rng.integers(0, 3, size=32)

    losses = [float(step(features, labels)) for _ in range(50)]
    assert losses[-1] < losses[0]
