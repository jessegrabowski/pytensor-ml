from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import pytensor.tensor as pt

from pytensor.tensor.basic import as_tensor_variable

Reductions = Literal["mean", "sum"]
ReductionFunction = Callable[[pt.TensorVariable], pt.TensorVariable]

# A callable covers the unreduced case (``reduction=lambda x: x``), for weighting individual losses.
ReductionLike = Reductions | ReductionFunction

_REDUCTIONS: dict[Reductions, ReductionFunction] = {"mean": pt.mean, "sum": pt.sum}


def _as_reduction(reduction: ReductionLike) -> ReductionFunction:
    return reduction if callable(reduction) else _REDUCTIONS[reduction]


class Loss(ABC):
    """
    Scalar objective a training step differentiates, called as ``loss(y_true, y_pred)``.

    Examples
    --------
    Subclass it by implementing :meth:`loss`; the base class makes the instance callable:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml.loss import Loss


        class MeanAbsoluteError(Loss):
            def loss(self, y_true, y_pred):
                return pt.abs(y_true - y_pred).mean()


        objective = MeanAbsoluteError()(pt.vector("y_true"), pt.vector("y_pred"))
    """

    @abstractmethod
    def loss(self, y_true, y_pred) -> pt.TensorVariable: ...

    def target_ndim(self, prediction_ndim: int) -> int:
        """
        Return the rank of the target this loss reads, given a prediction of rank ``prediction_ndim``.

        The default suits a loss that compares elementwise; override it for one that reads labels in a
        denser encoding.
        """
        return prediction_ndim

    def target_dtype(self, prediction_dtype: str) -> str:
        """
        Return the dtype of the target this loss reads, given a prediction of ``prediction_dtype``.

        The default suits a loss that compares against the prediction; override it for one that indexes
        with the target rather than subtracting it.
        """
        return prediction_dtype

    def __call__(self, y_true, y_pred) -> pt.TensorVariable:
        return self.loss(y_true, y_pred)


class SquaredError(Loss):
    """
    Mean or summed squared deviation between prediction and target, for regression.

    Examples
    --------
    Call the loss on a target and a prediction to get the scalar a training step differentiates:

    .. code-block:: python

        import numpy as np
        import pytensor
        import pytensor.tensor as pt

        from pytensor_ml.loss import SquaredError

        y_true = pt.matrix("y_true")
        y_pred = pt.matrix("y_pred")

        objective = SquaredError()(y_true, y_pred)
        value = pytensor.function([y_true, y_pred], objective)(np.ones((4, 1)), np.zeros((4, 1)))
    """

    def __init__(self, reduction: ReductionLike = "mean"):
        self.reduction = _as_reduction(reduction)

    def loss(self, y_true, y_pred) -> pt.TensorVariable:
        y_true = as_tensor_variable(y_true)
        y_pred = as_tensor_variable(y_pred)
        return self.reduction((y_true - y_pred) ** 2)


class CrossEntropy(Loss):
    """
    Negative log likelihood of the true class under the predicted distribution, for classification.

    Examples
    --------
    Integer labels against predicted probabilities, the default:

    .. code-block:: python

        import numpy as np
        import pytensor
        import pytensor.tensor as pt

        from pytensor_ml.loss import CrossEntropy

        labels = pt.vector("labels", dtype="int64")
        probabilities = pt.matrix("probabilities")

        objective = CrossEntropy()(labels, probabilities)
        value = pytensor.function([labels, probabilities], objective)(
            np.array([0, 2]), np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        )

    A classifier head emits logits, and one-hot labels are what an encoder produces, so both flags are
    usually set together. Passing logits keeps the loss on the numerically stable log-softmax path:

    .. code-block:: python

        import numpy as np
        import pytensor
        import pytensor.tensor as pt

        from pytensor_ml.loss import CrossEntropy

        onehot = pt.matrix("onehot")
        logits = pt.matrix("logits")

        objective = CrossEntropy(expect_logits=True, expect_onehot_labels=True)(onehot, logits)
        value = pytensor.function([onehot, logits], objective)(
            np.eye(3)[[0, 2]], np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        )
    """

    def __init__(
        self,
        reduction: ReductionLike = "mean",
        expect_logits: bool = False,
        expect_onehot_labels: bool = False,
    ):
        self.reduction = _as_reduction(reduction)
        self.expect_logits = expect_logits
        self.expect_onehot_labels = expect_onehot_labels

    def target_ndim(self, prediction_ndim: int) -> int:
        """One-hot labels are shaped like the prediction; integer labels drop its class axis."""
        return prediction_ndim if self.expect_onehot_labels else prediction_ndim - 1

    def target_dtype(self, prediction_dtype: str) -> str:
        """Integer labels index the class axis, so they cannot carry the prediction's float dtype."""
        return prediction_dtype if self.expect_onehot_labels else "int64"

    def loss(self, y_true: pt.TensorVariable, y_pred: pt.TensorVariable) -> pt.TensorVariable:
        """
        Parameters
        ----------
        y_true : TensorVariable
            Ground-truth class membership: a matrix of one-hot rows when ``expect_onehot_labels`` is set,
            otherwise a vector of integer class labels.
        y_pred : TensorVariable
            Predicted class membership, as unnormalized logits when ``expect_logits`` is set, otherwise as
            probabilities.

        Returns
        -------
        loss : TensorVariable
            The reduced loss -- a scalar under the named reductions, or whatever shape a callable
            ``reduction`` leaves.
        """
        y_true = as_tensor_variable(y_true)
        y_pred = as_tensor_variable(y_pred)
        if self.expect_logits:
            log_softmax = pt.special.log_softmax(y_pred, axis=-1)
        else:
            log_softmax = pt.log(y_pred)

        if not self.expect_onehot_labels:
            log_softmax = pt.take_along_axis(log_softmax, y_true[..., None], axis=-1)[..., 0]
            return -self.reduction(log_softmax)

        return -self.reduction((y_true * log_softmax).sum(axis=-1))


def supervised_loss(
    prediction: pt.TensorVariable,
    loss_fn: Loss,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """
    Build a training loss and its target placeholder from a model prediction.

    The target is a fresh input variable shaped like the labeled slice of ``prediction``, whose rank and
    dtype ``loss_fn`` decides through :meth:`Loss.target_ndim` and :meth:`Loss.target_dtype`.
    :class:`SquaredError` compares elementwise and takes a target shaped like the prediction;
    :class:`CrossEntropy` reading integer labels drops its class axis and takes integers.

    Parameters
    ----------
    prediction : TensorVariable
        Model output to compare against the target.
    loss_fn : Loss
        Callable ``(target, prediction) -> scalar loss``, which also states the target it reads.

    Returns
    -------
    loss : TensorVariable
        Scalar training loss.
    target : TensorVariable
        Input placeholder for the ground-truth labels, to be supplied at call time.

    Examples
    --------
    Point it at a model's output and it hands back the target placeholder to feed alongside each batch:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import CrossEntropy, supervised_loss
        from pytensor_ml.optim import adam, compile_train

        X = Input("X", shape=(None, 4))
        logits = Linear("logits", n_in=4, n_out=3)(X)

        loss_fn = CrossEntropy(expect_logits=True, expect_onehot_labels=True)
        objective, target = supervised_loss(logits, loss_fn)

        step = compile_train(objective, adam(1e-3), inputs=[X, target])
    """
    target_ndim = loss_fn.target_ndim(prediction.ndim)
    label_slice = (slice(None),) * target_ndim + (0,) * (prediction.ndim - target_ndim)
    target = pt.tensor(
        "target",
        shape=prediction[label_slice].type.shape,
        dtype=loss_fn.target_dtype(prediction.dtype),
    )
    return loss_fn(target, prediction), target
