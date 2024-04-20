from pytensor.tensor import tensor, TensorLike, as_tensor, squeeze, concatenate
from typing import Callable
from abc import ABC

class Layer(ABC):
    def __call__(self, x: TensorLike) -> TensorLike:
        ...


class Linear(Layer):
    __props__ = ('name', 'n_out')

    def __init__(self, name: str | None, n_in: int, n_out: int):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out

        self.W = tensor(f'{self.name}_W', shape=(n_in, self.n_out))
        self.b = tensor(f'{self.name}_b', shape=(self.n_out,))

    def __call__(self, X: TensorLike) -> TensorLike:
        X = as_tensor(X)

        init_st_shape = [st_dim if st_dim is not None else "?" for st_dim in X.type.shape]
        X = X @ self.W + self.b
        final_st_shape = [st_dim if st_dim is not None else "?" for st_dim in X.type.shape]
        X.name = f"{self.__class__.__name__}[{init_st_shape} -> {final_st_shape}]"

        return X


def Input(name: str, shape: tuple[int]) -> TensorLike:
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError("All dimensions must be integers")

    return tensor(name=name, shape=shape)


def Sequential(*layers: Callable) -> Callable:
    def forward(x: TensorLike) -> TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


__all__ = [
    'Linear',
    'squeeze',
    'concatenate'
]