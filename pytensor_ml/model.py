from typing import Sequence, cast, Literal, Generator

from pytensor.tensor import TensorVariable, TensorType
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.printing import debugprint
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant
from pytensor import config

import numpy as np
from pytensor_ml.layers import Layer


def required_graph_inputs(tensor: TensorVariable) -> Generator[TensorVariable, None, None]:
    return (cast(TensorVariable, var) for var in graph_inputs([tensor])
            if not isinstance(var, (Constant, SharedVariable)))


class Model:
    def __init__(self,
                 X: TensorVariable,
                 y: TensorVariable):

        self.X = X
        self.y = y

        self.weight_values: list[np.ndarray[float]] | None = None


    def initalize_weights(self) -> dict[TensorVariable, TensorType]:
        raise NotImplementedError

    @property
    def weights(self) -> list[TensorVariable]:
        return [var for var in required_graph_inputs(self.y) if var is not self.X]

    def __str__(self):
        return debugprint(self.y, file="str")


Schemes = Literal['zeros', 'xavier_uniform', 'xavier_normal']
def initialize_weights(model, scheme: Schemes, random_seed: int | str | np.random.Generator):
    if isinstance(random_seed, str):
        random_seed = sum(map(ord, random_seed))
    if isinstance(random_seed, int):
        random_seed = np.random.default_rng(random_seed)





