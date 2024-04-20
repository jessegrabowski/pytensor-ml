from abc import ABC, abstractmethod
from pytensor_ml.model import Model
from pytensor.gradient import grad
from pytensor.compile.function import function
from pytensor.compile.function.types import Function
from pytensor.tensor import TensorVariable, TensorLike
from typing import Callable, Sequence
import numpy as np


class Optimizer(ABC):

    def __init__(self, model: Model, loss_fn: Callable, ndim_out: int=1):
        self.model = model
        self.loss_fn = loss_fn
        self.ndim_out = ndim_out

        self.update_fn = self.build_update_fn()


    @abstractmethod
    def update_parameters(self, params: Sequence[TensorVariable], loss: TensorVariable):
        ...


    def build_update_fn(self) -> Function:
        """
        Compile a function to update model weights

        Compile a pytensor function that maps data (x), targets (y), and current parameter values to new parameter
        values. By default, the functions also returns the loss value. To return additional diagnostics, this method
        should be overloaded.

        Returns
        -------
        update_fn: Function
            A function that updates the model weights given a batch of data, targets, and current weights.
        """

        x, y_hat = self.model.X, self.model.y

        label_slice = (slice(None),) * self.ndim_out + (0, ) * (y_hat.ndim - self.ndim_out)
        y = y_hat[np.s_[label_slice]].type()

        params = self.model.weights

        loss = self.loss_fn(y, y_hat)
        new_parameters = self.update_parameters(params, loss)
        fn = function([x, y, *params], [*new_parameters, loss])
        return fn

    def step(self, x_values, y_values) -> np.ndarray:
        """
        This function updates the model weights in place given a new batch of x_values and y_values.

        Returns
        -------
        loss
        """
        *new_weights, loss_values = self.update_fn(x_values, y_values, *self.model.weight_values)
        self.model.weight_values = list(new_weights)
        self.update_fn.trust_input = True
        return loss_values


class SGD(Optimizer):
    def __init__(self, model, loss_fn, *, ndim_out:int=1, learning_rate: TensorLike = 0.01):
        self.learning_rate = learning_rate
        super().__init__(model, loss_fn, ndim_out=ndim_out)

    def update_parameters(self, params: Sequence[TensorVariable], loss: TensorVariable) -> list[TensorVariable]:
        grads = grad(loss, params)
        new_params = []
        for param, d_loss_d_param in zip(params, grads):
            new_params.append(param - self.learning_rate * d_loss_d_param)

        return new_params
