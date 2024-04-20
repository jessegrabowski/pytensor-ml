import pytensor.tensor as pt
from pytensor_ml.layers import Layer


class Activation(Layer):
    ...

class ReLU(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.maximum(0, x)
        out.name = f'ReLU'
        return out


class LeakyReLU(Activation):
    def __init__(self, negative_slope: pt.TensorLike = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.switch(x > 0, x, -self.negative_slope * x)
        out.name = 'LeakyReLU'
        return out

class Tanh(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out =  pt.tanh(x)
        out.name = 'TanH'
        return out

class Sigmoid(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.sigmoid(x)
        out.name = 'Sigmoid'
        return out



__all__ = [
    'ReLU',
    'LeakyReLU',
    'Sigmoid',
    'Tanh',
]