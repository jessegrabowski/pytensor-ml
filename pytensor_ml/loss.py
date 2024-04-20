import pytensor.tensor as pt
from abc import ABC, abstractmethod
from typing import Literal


Reductions = Literal['mean', 'sum']
reduction_dict = {"mean": pt.mean, "sum": pt.sum}


class Loss(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred) -> pt.TensorVariable:
        ...

    def __call__(self, y_true, y_pred) -> pt.TensorVariable:
        return self.loss(y_true, y_pred)



class SquaredError(Loss):
    def __init__(self, reduction: Reductions = 'mean'):
        self.reduction = reduction_dict[reduction]

    def loss(self, y_true, y_pred) -> pt.TensorVariable:
            return self.reduction(((y_true - y_pred) ** 2))


class CrossEntropy(Loss):
    def __init__(self,
                 reduction: Reductions = 'mean',
                 expect_logits: bool = False,
                 expect_onehot_labels: bool = False,
                 epsilon: pt.TensorLike = 1e-8):

        self.reduction = reduction_dict[reduction]
        self.expect_logits = expect_logits
        self.expect_onehot_labels = expect_onehot_labels
        self.epsilon = epsilon

    def loss(self, y_true: pt.TensorVariable, y_pred: pt.TensorVariable) -> pt.TensorVariable:
        """

        Parameters
        ----------
        y_true: Tensor variable
            Vector of class labels
        y_pred: Tensor variable
            Matrix of unnormalized log probabilities of class membership

        Returns
        -------

        """

        if self.expect_logits:
            y_pred = pt.log(pt.special.softmax(y_pred, axis=-1) + self.epsilon)
        if not self.expect_onehot_labels:
            # TODO: This doesn't work -- needs a custom gradient to get us back to the right shape
            y_pred = pt.take_along_axis(y_pred,
                                        pt.broadcast_to(y_true, y_pred.shape).astype(int), axis=-1)

        return -self.reduction(y_pred * y_true)

