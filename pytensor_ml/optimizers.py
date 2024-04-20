from abc import ABC, abstractmethod
from pytensor_ml.model import Model
from pytensor.gradient import grad
from pytensor.compile.function import function
from pytensor.compile.function.types import Function
from pytensor.tensor import TensorVariable
from typing import Callable, Sequence
import numpy as np


class Optimizer(ABC):

    def __init__(self, model: Model, loss_fn: Callable):
        self.model = model
        self.loss_fn = loss_fn
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

        x, y_hat = self.model.build()
        y = y_hat.type()
        params = self.model.weights

        loss = self.loss_fn(y_hat, y)
        new_parameters = self.update_parameters(params, loss)
        fn = function([x, y, params], [new_parameters, loss])
        return fn

    def step(self, x_values, y_values) -> np.ndarray:
        """
        This function updates the model weights in place given a new batch of x_values and y_values.

        Returns
        -------
        loss
        """
        self.model.weight_values, loss_values = self.update_fn(x_values, y_values, self.model.weight_values)
        self.update_fn.trust_input = True
        return loss_values


class SGD(Optimizer):
    def __init__(self, model, loss_fn, *, learning_rate : float = 0.01):
        super().__init__(model, loss_fn)
        self.learning_rate = learning_rate

    def update_parameters(self, params: Sequence[TensorVariable], loss: TensorVariable) -> list[TensorVariable]:
        grads = grad(loss, params)
        new_params = []
        for param, d_loss_d_param in zip(params, grads):
            new_params.append(param - self.learning_rate * d_loss_d_param)

        return new_params


"""
    model = Model([Squeeze(), FullyConnected(64*64, 100), relu, FullyConnected(100, 10), softmax]
    optimizer = SGD(model, loss)
    
    for i, (xs, ys) in enumerate(data)
        diagnostics = optimizer.step(xs, ys)
        print(f"{i}: loss={diagnostics["loss"]}")
    
    # start actual training loop
    for i, batch in enumerate(data):
        step_diagnostics = f(data)
        
        optimizer.zero_grad()
        x_hat = model(batch)
        loss = cross_entropy(x_hat, batch)
        loss.backward()
        optimizer.step()
        
        # new_weights = f(current_weights, batch)
        
    
    
    
    

"""


