from abc import ABC, abstractmethod

from pytensor import config

from pytensor_ml.model import Model
from pytensor.gradient import grad
from pytensor.compile.function import function
from pytensor.compile.function.types import Function
from pytensor.tensor import TensorVariable, TensorLike, tensor, sqrt
from typing import Callable, Sequence
import numpy as np


class Optimizer(ABC):

    def __init__(
        self,
        model: Model,
        loss_fn: Callable,
        ndim_out: int = 1,
        optimizer_weights: list[TensorVariable] | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.ndim_out = ndim_out
        self.optimizer_weights = (
            optimizer_weights if optimizer_weights is not None else []
        )
        self._optimizer_weights_values = self._initialize_weights()

        self.update_fn = self.build_update_fn()

    @abstractmethod
    def update_parameters(
        self, params: Sequence[TensorVariable], loss: TensorVariable
    ): ...

    @property
    def optimizer_weights_values(self) -> list[np.ndarray]:
        return self._optimizer_weights_values

    @optimizer_weights_values.setter
    def optimizer_weights_values(self, values: list[np.ndarray]):
        for i, new_value in enumerate(values):
            self._optimizer_weights_values[i][:] = new_value

    def _initialize_weights(self) -> list[np.ndarray]:
        if self.optimizer_weights:
            return [np.zeros(param.type.shape) for param in self.optimizer_weights]
        return []

    def _split_weights(
        self, all_weights: list[TensorLike]
    ) -> tuple[list[TensorLike], list[TensorLike]]:
        n_params = len(self.model.weights)
        return all_weights[:n_params], all_weights[n_params:]

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

        label_slice = (slice(None),) * self.ndim_out + (0,) * (
            y_hat.ndim - self.ndim_out
        )
        y = y_hat[np.s_[label_slice]].type()

        weights = self.model.weights
        optimizer_weights = self.optimizer_weights

        loss = self.loss_fn(y, y_hat)

        all_weights = self.update_parameters(weights + optimizer_weights, loss)
        fn = function([x, y, *weights, *optimizer_weights], [*all_weights, loss])
        return fn

    def step(self, x_values, y_values) -> np.ndarray:
        """
        This function updates the model weights in place given a new batch of x_values and y_values.

        Returns
        -------
        loss
        """
        *new_weights, loss_values = self.update_fn(
            x_values,
            y_values,
            *self.model.weight_values,
            *self.optimizer_weights_values,
        )
        self.update_fn.trust_input = True

        self.model.weight_values, self.optimizer_weights_values = self._split_weights(
            new_weights
        )

        return loss_values


class SGD(Optimizer):
    def __init__(
        self, model, loss_fn, *, ndim_out: int = 1, learning_rate: TensorLike = 0.01
    ):
        self.learning_rate = learning_rate
        super().__init__(model, loss_fn, ndim_out=ndim_out)

    def update_parameters(
        self, params: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        grads = grad(loss, params)
        new_params = []
        for param, d_loss_d_param in zip(params, grads):
            new_params.append(param - self.learning_rate * d_loss_d_param)

        return new_params


class ADAGrad(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 0.01,
        epsilon: TensorLike = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        g2_weights = [param.type() for param in model.weights]

        super().__init__(
            model, loss_fn, ndim_out=ndim_out, optimizer_weights=g2_weights
        )

    def update_parameters(
        self, weights: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        weights, optimizer_weights = self._split_weights(weights)
        grads = grad(loss, weights)

        new_weights = []
        new_optimizer_weights = []

        for param, d_loss_d_param, g2 in zip(weights, grads, optimizer_weights):
            new_g2 = g2 + d_loss_d_param**2
            weight_update = d_loss_d_param / np.sqrt(new_g2 + self.epsilon)
            new_weights.append(param - self.learning_rate * weight_update)
            new_optimizer_weights.append(new_g2)

        return new_weights + new_optimizer_weights


class Adadelta(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 1.0,
        rho: TensorLike = 0.9,
        epsilon: TensorLike = 1e-8,
    ):

        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

        u_weights = [param.type() for param in model.weights]
        v_weights = [param.type() for param in model.weights]
        optimizer_weights = u_weights + v_weights
        super().__init__(
            model, loss_fn, ndim_out=ndim_out, optimizer_weights=optimizer_weights
        )

    def update_parameters(
        self, weights: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        weights, optimizer_weights = self._split_weights(weights)
        u_weights, v_weights = self._split_weights(optimizer_weights)

        grads = grad(loss, weights)

        new_weights = []
        new_u_weights = []
        new_v_weights = []

        for param, d_loss_d_param, u, v in zip(weights, grads, u_weights, v_weights):
            new_v = v * self.rho + d_loss_d_param**2 * (1 - self.rho)
            weight_update = (
                sqrt(u + self.epsilon) / sqrt(new_v + self.epsilon)
            ) * d_loss_d_param
            new_u = u * self.rho + weight_update**2 * (1 - self.rho)

            new_weights.append(param - self.learning_rate * weight_update)
            new_u_weights.append(new_u)
            new_v_weights.append(new_v)

        return new_weights + new_u_weights + new_v_weights


class Adam(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 0.01,
        beta1: TensorLike = 0.9,
        beta2: TensorLike = 0.999,
        epsilon: TensorLike = 1e-8,
    ):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        m_weights = [param.type() for param in model.weights]
        v_weights = [param.type() for param in model.weights]
        t = tensor("t", shape=(1,), dtype=config.floatX)

        optimizer_weights = m_weights + v_weights + [t]
        super().__init__(
            model, loss_fn, ndim_out=ndim_out, optimizer_weights=optimizer_weights
        )

    def update_parameters(
        self, weights: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        weights, optimizer_weights = self._split_weights(weights)
        t = optimizer_weights.pop(-1)
        m_weights, v_weights = self._split_weights(optimizer_weights)

        grads = grad(loss, weights)

        new_weights = []
        new_m_weights = []
        new_v_weights = []

        new_t = t + 1
        a_t = sqrt(1 - self.beta2**new_t) / (1 - self.beta1**new_t)

        for param, d_loss_d_param, m, v in zip(weights, grads, m_weights, v_weights):
            weight_update = a_t * m / (sqrt(v) + self.epsilon)
            new_weights.append(param - self.learning_rate * weight_update)

            new_m = self.beta1 * m + (1 - self.beta1) * d_loss_d_param
            new_v = self.beta2 * v + (1 - self.beta2) * d_loss_d_param**2
            new_m_weights.append(new_m)
            new_v_weights.append(new_v)

        return new_weights + new_m_weights + new_v_weights + [new_t]
