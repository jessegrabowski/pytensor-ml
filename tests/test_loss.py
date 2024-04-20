import pytest
from pytensor_ml.loss import CrossEntropy, Reductions
import numpy as np
from sklearn.metrics import log_loss
from scipy.special import softmax


def generate_categorical_data(expect_logits: bool):
    rng = np.random.default_rng()
    n_classes = rng.integers(2, 10)
    y_true = rng.integers(0, n_classes, size=(100,))
    y_true_onehot = np.eye(n_classes)[y_true]
    y_pred = rng.random((100, n_classes)) if expect_logits else rng.dirichlet(np.ones(n_classes), size=(100, ))

    return y_true, y_true_onehot, y_pred

@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('expect_logits', [True, False])
def test_cross_entropy_onehot_vs_labels(reduction: Reductions, expect_logits):
    y_true, y_true_onehot, y_pred = generate_categorical_data(expect_logits)

    loss = CrossEntropy(expect_logits=expect_logits, expect_onehot_labels=False, reduction=reduction)
    loss_onehot = CrossEntropy(expect_logits=expect_logits, expect_onehot_labels=True, reduction=reduction)

    loss_value = loss(y_true, y_pred).eval()
    loss_value_onehot = loss_onehot(y_true_onehot, y_pred).eval()

    np.testing.assert_allclose(loss_value, loss_value_onehot)


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('expect_logits', [True, False])
@pytest.mark.parametrize('expect_onehot_labels', [True, False])
def test_cross_entropy(reduction: Reductions, expect_logits, expect_onehot_labels):
    loss = CrossEntropy(reduction=reduction,
                        expect_logits=expect_logits,
                        expect_onehot_labels=expect_onehot_labels)

    y_true, y_true_onehot, y_pred = generate_categorical_data(expect_logits)

    if expect_onehot_labels:
        loss_value = loss(y_true_onehot, y_pred).eval()
    else:
        loss_value = loss(y_true, y_pred).eval()

    if expect_logits:
        y_pred = softmax(y_pred, axis=-1)

    sklearn_loss = log_loss(y_true, y_pred, normalize=reduction == 'mean')
    np.testing.assert_allclose(loss_value, sklearn_loss)
