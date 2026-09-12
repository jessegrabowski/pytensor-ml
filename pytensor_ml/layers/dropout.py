from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytensor.tensor.random as ptr

from pytensor import config
from pytensor.tensor.random.variable import RandomGeneratorSharedVariable, shared_rng

from pytensor_ml.base import Layer, UnaryLayerOp, _resolve_layer_name


class DropoutLayer(UnaryLayerOp):
    __props__ = ("p",)

    def build_inner_graph(self, X, mask):
        return [pt.where(mask, ift=X / (1 - self.p), iff=0)]


class Dropout(Layer):
    """
    Zero a random fraction of its input, rescaling the rest so the expected sum is unchanged.

    Parameters
    ----------
    name : str, optional
        Name of the layer, used to name its output and its generators. Default "Dropout".
    p : float
        Probability of zeroing each element. Default 0.5.
    random_state : int, Generator, or other seed, optional
        Seed for the layer's masks. Draws are reproducible under a given seed, including when the layer is
        applied at several points in a network. Seeded from fresh entropy when omitted.

    Attributes
    ----------
    generators : list of RandomGeneratorSharedVariable
        One generator per application of the layer, in the order they were applied. Set a value on these to
        steer or restore the masks.

    Examples
    --------
    Zero a random share of activations during training, rescaling the survivors so the mean is unchanged.
    :meth:`~pytensor_ml.model.Model.predict` drops the layer entirely, so inference is deterministic:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import Dropout, Input, Linear, Sequential

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            ReLU(),
            Dropout(p=0.1, random_state=0),
        )

        activations = network(X)
    """

    def __init__(self, name: str | None = None, *, p: float = 0.5, random_state: Any | None = None):
        self.name = _resolve_layer_name(name, type(self).__name__, "p")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.generators: list[RandomGeneratorSharedVariable] = []
        self._generator_source = np.random.default_rng(random_state)

    def _own_generator(self) -> RandomGeneratorSharedVariable:
        """Return a generator for one application of this layer, spawned from its seed.

        Each application draws its own mask, so each needs a generator of its own: a generator read by two
        draws has no single next state, and nothing can advance it. Spawning keeps the whole layer
        reproducible under ``random_state`` however many times it is applied.
        """
        generator = shared_rng(
            self._generator_source.spawn(1)[0], name=f"{self.name}/rng_{len(self.generators)}"
        )
        self.generators.append(generator)
        return generator

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        p = pt.as_tensor(self.p, dtype=config.floatX)
        _, mask = ptr.bernoulli(
            p=1 - p, size=X.shape, rng=self._own_generator(), return_next_rng=True
        )
        mask = mask.astype(config.floatX)

        X_masked = DropoutLayer(
            name=f"{self.name}[p = {self.p}]",
            p=self.p,
        )(X, mask)
        X_masked.name = f"{self.name}_output"

        return X_masked
